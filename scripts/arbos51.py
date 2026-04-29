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
    4. Backlog                  (backlog_per_second, backlog_per_block)
    5. Exponent contribution    (exponent_contribution)
    6. Per-block price          (price_per_block)
    7. Per-tx pricing           (price_per_tx, fee_per_tx)
    8. Diagnostics              (backlogs_all_constraints)
"""

from __future__ import annotations

import numpy as np


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

    def __init__(
        self,
        ladder: list[tuple[float, int]] | None = None,
        p_min_gwei: float | None = None,
    ):
        self.ladder     = ladder     if ladder     is not None else self.LADDER
        self.p_min_gwei = p_min_gwei if p_min_gwei is not None else self.P_MIN_GWEI

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
        critical for the backlog recursion downstream.

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

    # ── 4. Backlog (1-s tick recursion + per-block lookup) ─────────────────
    @staticmethod
    def backlog_per_second(
        inflow_per_second: np.ndarray,
        T_mgas_s,
    ) -> np.ndarray:
        """
        Floored backlog at the END of each 1-second tick (gas units).
        One step per second:
            B(s+1) = max(0, B(s) + inflow(s) - T·1)
        Vectorised via the cumsum/running-min identity.

        Polymorphic in T: scalar T → 1-D output (n_seconds,);
        array T (shape (n_T,)) → 2-D output (n_T, n_seconds), one row per T.
        """
        T = np.asarray(T_mgas_s, dtype=np.float64)
        scalar = (T.ndim == 0)
        T_vec = np.atleast_1d(T)                                  # (n_T,)
        # T is in Mgas/s; multiply by 1e6 to get the per-second drain in raw
        # gas units (the same units as `inflow_per_second`).
        drain = T_vec * 1e6
        n = inflow_per_second.shape[0]
        d = inflow_per_second[None, :] - drain[:, None]           # (n_T, n)
        C = np.empty((T_vec.shape[0], n + 1), dtype=np.float64)
        C[:, 0] = 0.0
        np.cumsum(d, axis=1, out=C[:, 1:])
        rmin = np.minimum.accumulate(C, axis=1)
        out = np.maximum(0.0, C - rmin)[:, :n]                    # (n_T, n)
        return out[0] if scalar else out

    @classmethod
    def backlog_per_block(
        cls,
        inflow_per_block: np.ndarray,
        block_seconds: np.ndarray,
        T_mgas_s,
    ) -> np.ndarray:
        """
        Per-block backlog as seen by each block's pricing step: B at the
        START of the block's wall-clock second (= B at end of the previous
        second). Initial condition: B = 0 at the start of the first second
        in the window. Blocks within the same second share the same B.

        Polymorphic in T: scalar T → 1-D output (n_blocks,);
        array T (n_T,) → 2-D (n_T, n_blocks).
        """
        seconds, inflow_per_sec = cls.aggregate_per_second(inflow_per_block, block_seconds)
        T = np.asarray(T_mgas_s, dtype=np.float64)
        scalar = (T.ndim == 0)
        B_end = cls.backlog_per_second(inflow_per_sec, T)
        # Right-shift end-of-second backlog by one tick to get start-of-second
        # values; B_start[0] = 0 is the initial condition.
        if scalar:
            B_start = np.empty_like(B_end)
            B_start[0]  = 0.0
            B_start[1:] = B_end[:-1]
        else:
            B_start = np.empty_like(B_end)
            B_start[:, 0]  = 0.0
            B_start[:, 1:] = B_end[:, :-1]
        block_sec_idx = (block_seconds - seconds[0]).astype(np.int64)
        return B_start[block_sec_idx] if scalar else B_start[:, block_sec_idx]

    @classmethod
    def _ladder_exponent_from_inflow(
        cls,
        inflow_per_block: np.ndarray,
        block_seconds: np.ndarray,
        ladder: list[tuple[float, int]],
    ) -> np.ndarray:
        """Σ_j B_j / (A_j · T_j) per block for one ladder, in a single
        batched pass: aggregate inflow once, run the backlog recursion over
        all (T_j) in one cumsum/min call, sum the normalized contributions.

        Output shape: (n_blocks,)."""
        T_arr = np.fromiter((T for T, _ in ladder), dtype=np.float64, count=len(ladder))
        A_arr = np.fromiter((A for _, A in ladder), dtype=np.float64, count=len(ladder))
        B_per_block = cls.backlog_per_block(inflow_per_block, block_seconds, T_arr)  # (n_T, n_blk)
        norm = (A_arr * T_arr * 1e6)[:, None]
        return (B_per_block / norm).sum(axis=0)

    # ── 5. Exponent contribution per constraint ─────────────────────────────
    def exponent_contribution(
        self,
        inflow_per_block: np.ndarray,
        block_seconds: np.ndarray,
        T_mgas_s: float,
        A_s: float,
    ) -> np.ndarray:
        """
        B / (A·T) per block for one constraint (uncapped). B is the
        start-of-second backlog from `backlog_per_block` (already ≥ 0).
        """
        B    = self.backlog_per_block(inflow_per_block, block_seconds, T_mgas_s)
        norm = A_s * T_mgas_s * 1e6
        return B / norm

    # ── 6. Per-block price ──────────────────────────────────────────────────
    def price_per_block(
        self,
        total_gas: np.ndarray,
        block_seconds: np.ndarray,
    ) -> np.ndarray:
        """
        Per-block ArbOS 51 base fee in gwei.

            e = Σ_j  B_j / (A_j · T_j)              (B_j is start-of-second backlog)
            p = p_min · taylor4_exp(e)              (no upper cap; e floored at 0)

        With e ≥ 0 and the Taylor-4 polynomial monotone increasing on [0, ∞),
        p ≥ p_min is preserved.

        Inputs:
          total_gas       per-block total L2 gas (gas units, not Mgas).
          block_seconds   per-block integer UTC seconds (see `block_seconds_utc`).
        """
        expo = self._ladder_exponent_from_inflow(total_gas, block_seconds, self.ladder)
        expo = np.maximum(expo, 0.0)
        return self.p_min_gwei * self.taylor4_exp(expo)

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
        """Per-tx fee in gwei·gas:  fee_tx = p_block(of tx) · g_tx_total"""
        return p_per_block[tx_block_idx] * tx_total_gas

    # ── 8. Diagnostics ──────────────────────────────────────────────────────
    def backlogs_all_constraints(
        self,
        total_gas: np.ndarray,
        block_seconds: np.ndarray,
    ) -> dict[tuple[float, int], np.ndarray]:
        """Per-block backlog (gas units) for each (T, A) in the ladder
        (1s-tick recursion)."""
        T_arr = np.fromiter((T for T, _ in self.ladder), dtype=np.float64, count=len(self.ladder))
        B = self.backlog_per_block(total_gas, block_seconds, T_arr)   # (n_T, n_blocks)
        return {(T, A): B[i] for i, (T, A) in enumerate(self.ladder)}
