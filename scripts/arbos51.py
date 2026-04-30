"""
ArbOS 51 dynamic-pricing equations.

Single-dimensional 6-constraint geometric ladder. One base fee per block,
applied uniformly to every gas unit. ArbOS 51 has only one set i, so the
i-index is suppressed; only j (constraint within the ladder) varies.
Backlog evolves at Δt = 1 s ticks per the proposal spec ("every second,
add inflow, remove T"); per-block backlog is a lookup of the per-t time
series. Blocks within the same wall-clock t therefore share the same
backlog and the same price.

    backlog (Δt = 1 s tick, per j):
        B_j(t+1) = max(0, B_j(t) + g_total(t) − T_j · Δt)

    raw exponent (per t):
        E(t) = Σ_j B_j(t) / (A_j · T_j)

    base fee (per t):
        p(t) = p_min · taylor4_exp(max(0, E(t)))

    per-tx fee:
        fee_tx = p(block_of_tx) · g_total_tx

Public surface — `Arbos51GasPricing`. Method order mirrors the equation flow:

    1. Constants                (P_min, ladder)
    2. Time / aggregation       (block_seconds_utc, aggregate_per_second)
    3. Exp approximation        (taylor4_exp)
    4. Inflow per t             (compute_inflow_per_t)
    5. Backlog                  (backlog_per_second)
    6. Per-t price              (price_per_second)
    7. Per-tx pricing           (price_per_tx, fee_per_tx)
"""

from __future__ import annotations

import numpy as np


# ArbOS 51 Dia activated on Arbitrum One at 2026-01-08 17:00:00 UTC. Before
# this the chain ran a single-constraint pricing controller (T = 7 Mgas/s,
# A = 102 s, p_min = 0.01 gwei since ArbOS 20 Atlas, Mar 2024); after, the
# 6-rung ladder + 0.02 gwei min that this class's defaults represent.
ARBOS_DIA_ACTIVATION_UTC = np.datetime64("2026-01-08T17:00:00")
ARBOS_DIA_ACTIVATION_S = int(
    ARBOS_DIA_ACTIVATION_UTC.astype("datetime64[s]").astype(np.int64)
)


class Arbos51GasPricing:
    """ArbOS 51 single-dim geometric-ladder pricing engine."""

    # ── 1. Constants ────────────────────────────────────────────────────────
    # Minimum L2 base fee since ArbOS 51 Dia (gwei).
    P_MIN_GWEI: float = 0.02

    # On-chain Dia ladder. Each entry is (T_j Mgas/s, A_j seconds). Single-
    # dim: every constraint receives the full `total_l2_gas` per block.
    LADDER: list[tuple[float, int]] = [
        (10.0, 86_400),
        (14.0, 13_485),
        (20.0,  2_105),
        (29.0,    329),
        (41.0,     52),
        (60.0,      9),
    ]

    # Pre-Dia (ArbOS 20 Atlas → ArbOS 50) single-constraint config.
    PRE_DIA_LADDER: list[tuple[float, int]] = [(7.0, 102)]
    PRE_DIA_P_MIN_GWEI: float = 0.01

    def __init__(
        self,
        ladder: list[tuple[float, int]] | None = None,
        p_min_gwei: float | None = None,
    ):
        self.ladder     = ladder     if ladder     is not None else self.LADDER
        self.p_min_gwei = p_min_gwei if p_min_gwei is not None else self.P_MIN_GWEI

    @classmethod
    def pre_dia(cls) -> "Arbos51GasPricing":
        """Pre-Dia engine: single (T = 7 Mgas/s, A = 102 s) constraint and
        p_min = 0.01 gwei. Same backlog/exp machinery as Dia, just one
        constraint and a lower floor."""
        return cls(ladder=cls.PRE_DIA_LADDER, p_min_gwei=cls.PRE_DIA_P_MIN_GWEI)

    # ── 2. Time / aggregation helpers ───────────────────────────────────────
    @staticmethod
    def block_seconds_utc(block_times_utc: np.ndarray) -> np.ndarray:
        """Per-block integer-second UTC timestamps (= per-block t)."""
        return block_times_utc.astype("datetime64[s]").astype(np.int64)

    @staticmethod
    def aggregate_per_second(
        values_per_block: np.ndarray,
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum per-block values into per-t buckets over the contiguous
        t range covered by `block_t`.  Empty seconds (no blocks landed)
        contribute 0 to the inflow but still drain the backlog — required
        for the backlog recursion downstream to be correct.

        Assumes `block_t` is sorted ascending: `block_t[0]` is the window
        start and `block_t[-1]` the end.

        Returns (t_axis, sum_per_t).  t_axis is dense int64 from
        min..max(block_t), inclusive.
        """
        t_min  = int(block_t[0])                                # window start (UTC s)
        t_max  = int(block_t[-1])                               # window end   (UTC s)
        n_t    = t_max - t_min + 1                              # # of 1 s buckets
        t_axis = np.arange(t_min, t_max + 1, dtype=np.int64)    # dense t axis
        t_idx  = (block_t - t_min).astype(np.int64)             # bucket index per block
        summed = np.bincount(                                   # Σ values_per_block per bucket
            t_idx,
            weights=np.asarray(values_per_block, dtype=np.float64),
            minlength=n_t,                                      # pads empty seconds with 0
        )
        return t_axis, summed

    # ── 3. Exp approximation (matches Nitro `model.go`) ─────────────────────
    @staticmethod
    def taylor4_exp(x: np.ndarray) -> np.ndarray:
        """Degree-4 Taylor approximation of exp.
        Matches nitro's `ApproxExpBasisPoints(x, 4)` in `model.go`.
        https://github.com/OffchainLabs/nitro/blob/master/arbos/l2pricing/model.go"""
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        return 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0

    # ── 4. Inflow per t (single-set: just g_total aggregated) ──────────────
    def compute_inflow_per_t(
        self,
        total_gas: np.ndarray,
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-t inflow for ArbOS 51's single set: aggregate the per-block
        total L2 gas to per-t (1 s buckets).  Returns (t_axis, inflow_per_t).
        Counterpart to ArbOS 60's `compute_inflow_per_t`, simplified because
        there is no resource weighting (one set, single dimension).
        """
        return self.aggregate_per_second(total_gas, block_t)

    # ── 5. Backlog (1-s tick loop) ──────────────────────────────────────────
    @staticmethod
    def backlog_per_second(
        inflow_per_t: np.ndarray,   # = g_total(t), per t (single set)
        T_j: np.ndarray,            # T_j ladder for the constraints
    ) -> np.ndarray:
        """Backlog tick, vectorised over j:
            B_j(t+1) = max(0, B_j(t) + inflow(t) − T_j·Δt)

        `inflow_per_t` is supplied by compute_inflow_per_t.  Returns
        (n_j, n_t); out[j, t] is the start-of-t backlog (pre-update),
        which is what a block landing in t reads.
        """
        drain = T_j * 1e6                                         # Mgas/s → gas/s (one number per j)
        out   = np.empty((T_j.shape[0], inflow_per_t.shape[0]))   # output buffer (n_j, n_t)
        B_j   = np.zeros(T_j.shape[0])                            # B_j(0) = 0  for every j
        for t, inflow_t in enumerate(inflow_per_t):               # walk t = 0, 1, … n_t-1
            out[:, t] = B_j                                       # snapshot B_j(t) before update
            B_j = np.maximum(0.0, B_j + inflow_t - drain)         # advance to B_j(t+1)
        return out

    # ── 6. Per-t price ──────────────────────────────────────────────────────
    def price_per_second(
        self,
        total_gas: np.ndarray,
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-t ArbOS 51 base fee in gwei:

            E(t) = Σ_j  B_j(t) / (A_j · T_j)
            p(t) = p_min · taylor4_exp(E(t))

        Returns (t_axis, p_per_t).  Caller gathers to per-block via:
            t_idx       = (block_t - t_axis[0]).astype(np.int64)
            p_per_block = p_per_t[t_idx]
        """
        # ── Step 4 ─────────────────────────────────────────────────────────
        # Inflow on the dense t axis: just g_total aggregated to 1 s buckets
        # (single set, no resource weighting).
        t_axis, inflow_per_t = self.compute_inflow_per_t(total_gas, block_t)

        # Pull the j-indexed (T_j, A_j) ladder pairs.
        T_j = np.array([T for T, _ in self.ladder], dtype=np.float64)
        A_j = np.array([A for _, A in self.ladder], dtype=np.float64)

        # ── Step 5 ─────────────────────────────────────────────────────────
        # Backlog tick → B_j(t), shape (n_j, n_t).
        B_j = self.backlog_per_second(inflow_per_t, T_j)

        # ── Step 6 ─────────────────────────────────────────────────────────
        # E(t) = Σ_j B_j(t) / (A_j · T_j).  T_j*1e6 puts the denominator in
        # gas/s so it matches the gas-units backlog; broadcast over t, then
        # fold over j → shape (n_t,).
        E = (B_j / (A_j * T_j * 1e6)[:, None]).sum(axis=0)

        # p(t) = p_min · taylor4_exp(E(t)).  E(t) ≥ 0 by construction
        # (B, A, T all non-negative), so no max(0, …) clip is needed.
        # taylor4_exp(0) = 1, monotone on [0, ∞), so p ≥ p_min always.
        p_per_t = self.p_min_gwei * self.taylor4_exp(E)
        return t_axis, p_per_t

    @classmethod
    def historical_price_per_second(
        cls,
        total_gas: np.ndarray,
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-t price honoring the ArbOS 51 Dia activation boundary.

        Blocks with `block_t < ARBOS_DIA_ACTIVATION_S` are priced with the
        pre-Dia engine (single constraint, p_min = 0.01 gwei); blocks at or
        after activation use the Dia engine (6-rung ladder, p_min = 0.02
        gwei).  Backlogs warm up independently per regime — matches the
        on-chain reset at the upgrade.

        Returns (t_axis, p_per_t) covering the union of both regimes in
        chronological order.  Assumes `block_t` is sorted ascending.
        """
        n = block_t.shape[0]
        if n == 0:                                                # nothing to price
            empty = np.zeros(0, dtype=np.int64)
            return empty, np.zeros(0, dtype=np.float64)

        # Find the index where block_t crosses the Dia activation second.
        # `side="left"` puts ties at activation into the Dia regime — matches
        # the on-chain rule that activation block t belongs to Dia.
        split = int(np.searchsorted(block_t, ARBOS_DIA_ACTIVATION_S, side="left"))

        # All blocks already at/after activation → pure Dia run, no split needed.
        if split == 0:
            return cls().price_per_second(total_gas, block_t)
        # All blocks before activation → pure pre-Dia run.
        if split == n:
            return cls.pre_dia().price_per_second(total_gas, block_t)

        # Mixed window: run each regime on its own slice with its own backlog
        # state (B_j(0) = 0 reset at the upgrade boundary, matching the
        # on-chain reset).  Then concatenate the two output slices.
        t_pre, p_pre = cls.pre_dia().price_per_second(total_gas[:split], block_t[:split])
        t_dia, p_dia = cls()        .price_per_second(total_gas[split:], block_t[split:])
        return (
            np.concatenate([t_pre, t_dia]),     # t axis: pre-Dia then Dia (chronological)
            np.concatenate([p_pre, p_dia]),     # p(t): same order
        )

    # ── 7. Per-tx pricing ───────────────────────────────────────────────────
    @staticmethod
    def price_per_tx(
        p_per_block: np.ndarray,
        tx_block_idx: np.ndarray,
    ) -> np.ndarray:
        """Per-tx effective gas price (gwei). Under ArbOS 51 every tx in
        block N pays exactly `p_per_block[N]` — just an indexed lookup."""
        return p_per_block[tx_block_idx]

    @staticmethod
    def fee_per_tx(
        p_per_block: np.ndarray,
        tx_block_idx: np.ndarray,
        tx_total_gas: np.ndarray,
    ) -> np.ndarray:
        """Per-tx fee (gwei·gas):  fee_tx = p(block_of_tx) · g_total_tx."""
        return p_per_block[tx_block_idx] * tx_total_gas
