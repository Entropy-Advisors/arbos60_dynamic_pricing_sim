"""
ArbOS 51 dynamic-pricing equations.

Single-dimensional 6-constraint geometric ladder. One base fee per block,
applied uniformly to every gas unit. Backlog evolves at **1-second** ticks
per the proposal spec ("every second, add inflow, remove T"); per-block
backlog is a lookup of the per-second time series. Blocks within the same
wall-clock second therefore share the same backlog and the same price.

Public surface — the `Arbos51GasPricing` class. Method order mirrors the
equation flow:

    1. Constants                (P_min, ladder)
    2. Time / aggregation       (block_seconds_utc, aggregate_per_second)
    3. Pure exp approximation   (taylor4_exp)
    4. Backlog                  (backlog_per_second)
    5. Per-second price         (price_per_second)
    6. Per-tx pricing           (price_per_tx, fee_per_tx)
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
        """Per-block integer-second UTC timestamps."""
        return block_times_utc.astype("datetime64[s]").astype(np.int64)

    @staticmethod
    def aggregate_per_second(
        values_per_block: np.ndarray,
        block_seconds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum per-block values into per-second buckets over the contiguous
        second range covered by `block_seconds`. Empty seconds (no blocks
        landed) contribute 0 to the inflow but still drain the backlog —
        critical for the backlog loop downstream.

        Assumes `block_seconds` is sorted ascending: `block_seconds[0]` is
        used as the window start and `block_seconds[-1]` as the end.

        Returns (seconds_axis, sum_per_second). seconds_axis is dense int64
        from min..max of `block_seconds`, inclusive.
        """
        s_min     = int(block_seconds[0])
        s_max     = int(block_seconds[-1])
        n_seconds = s_max - s_min + 1
        seconds   = np.arange(s_min, s_max + 1, dtype=np.int64)
        idx       = (block_seconds - s_min).astype(np.int64)
        summed    = np.bincount(
            idx,
            weights=np.asarray(values_per_block, dtype=np.float64),
            minlength=n_seconds,
        )
        return seconds, summed

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

    # ── 4. Backlog (1-s tick loop) ──────────────────────────────────────────
    @staticmethod
    def backlog_per_second(
        inflow_per_second: np.ndarray,
        T_arr: np.ndarray,
    ) -> np.ndarray:
        """Start-of-each-second backlog seen by blocks landing in that
        second. `T_arr` is the 1-D ladder (Mgas/s); returns shape
        (n_T, n_sec).

            B(0) = 0;   B(s+1) = max(0, B(s) + inflow(s) - T*1).

        out[:, s] is recorded *before* second s's update — that's what a
        block in second s reads, before that second's inflow / drain.
        """
        drain = T_arr * 1e6                                       # Mgas/s -> gas/s
        out = np.empty((T_arr.shape[0], inflow_per_second.shape[0]))
        B   = np.zeros(T_arr.shape[0])
        for s, inflow_s in enumerate(inflow_per_second):
            out[:, s] = B                                         # snapshot before update
            B = np.maximum(0.0, B + inflow_s - drain)
        return out

    # ── 5. Per-second price ─────────────────────────────────────────────────
    def price_per_second(
        self,
        total_gas: np.ndarray,
        block_seconds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-second ArbOS 51 base fee in gwei.

            E(s) = Σ_j  B_j(s) / (A_j · T_j)
            p(s) = p_min · taylor4_exp(max(0, E(s)))

        Returns (seconds_axis, p_per_sec). To gather to per-block:
            sec_idx = (block_seconds - seconds_axis[0]).astype(np.int64)
            p_per_block = p_per_sec[sec_idx]
        """
        seconds_axis, inflow_ps = self.aggregate_per_second(total_gas, block_seconds)
        T_arr = np.array([T for T, _ in self.ladder], dtype=np.float64)
        A_arr = np.array([A for _, A in self.ladder], dtype=np.float64)
        B = self.backlog_per_second(inflow_ps, T_arr)             # (n_T, n_sec)
        E = (B / (A_arr * T_arr * 1e6)[:, None]).sum(axis=0)      # (n_sec,)
        p_per_sec = self.p_min_gwei * self.taylor4_exp(np.maximum(E, 0.0))
        return seconds_axis, p_per_sec

    @classmethod
    def historical_price_per_second(
        cls,
        total_gas: np.ndarray,
        block_seconds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-second price honoring the ArbOS 51 Dia activation boundary.

        Blocks with `block_seconds < ARBOS_DIA_ACTIVATION_S` are priced with
        the pre-Dia engine (single constraint, p_min = 0.01 gwei); blocks at
        or after activation use the Dia engine (6-rung ladder, p_min = 0.02
        gwei). Backlogs warm up independently per regime — matches the
        on-chain reset at the upgrade.

        Returns (seconds_axis, p_per_sec) covering the union of both regimes
        in chronological order. Assumes `block_seconds` is sorted ascending.
        """
        n = block_seconds.shape[0]
        if n == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, np.zeros(0, dtype=np.float64)
        split = int(np.searchsorted(block_seconds, ARBOS_DIA_ACTIVATION_S, side="left"))
        if split == 0:
            return cls().price_per_second(total_gas, block_seconds)
        if split == n:
            return cls.pre_dia().price_per_second(total_gas, block_seconds)
        s_pre, p_pre = cls.pre_dia().price_per_second(total_gas[:split], block_seconds[:split])
        s_dia, p_dia = cls().price_per_second(total_gas[split:], block_seconds[split:])
        return np.concatenate([s_pre, s_dia]), np.concatenate([p_pre, p_dia])

    # ── 6. Per-tx pricing ───────────────────────────────────────────────────
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
        """Per-tx fee in gwei·gas:  fee_tx = p_block(of tx) · g_tx_total"""
        return p_per_block[tx_block_idx] * tx_total_gas

