"""
ArbOS 60 dynamic-pricing equations (per-resource pricing, BB-alternative form).

Four constraint sets, 19 constraints total. Each set i is a weighted
inequality with its own (T_{i,j}, A_{i,j}) ladder. Same backlog/exp
machinery as ArbOS 51, but wrapped with per-set inflows and a max-over-sets
projection that yields one price per resource. Backlog evolves at 1-second
ticks; per-block backlog is a lookup of the per-second time series.

    backlog (1-s tick, per (i, j)):
        B_{i,j}(s+1) = max(0, B_{i,j}(s) + Σ_k a_{i,k}·g_k(s) - T_{i,j}·1)

    set i raw exponent (per block, via start-of-second lookup):
        E_i = Σ_j B_{i,j} / (A_{i,j}·T_{i,j})

    per-resource price (BB alternative form):
        p_k = p_min · exp(max_i { a_{i,k} · E_i })

    per-tx fee / "price" (inner product):
        fee_tx = Σ_k g_tx,k · p_k(block_of_tx)

    fee aggregated over a window (e.g. an hour):
        p̄ = Σ_tx fee_tx / Σ_tx Σ_k g_tx,k

Public surface — the `Arbos60GasPricing` class. Method order mirrors the
equation flow:

    1. Constants                (P_min)
    2. Constraint sets          (SET_WEIGHTS, SET_LADDERS, PRICED_SYMBOLS)
    3. Time / aggregation       (block_seconds_utc, aggregate_per_second)
    4. Exp approximation        (taylor4_exp)
    5. Resource gas builders    (per_block_resource_gas, per_tx_resource_split)
    6. Backlog                  (backlog_per_second, backlog_per_block)
    7. Exponent + per-set raw   (exponent_contribution, compute_set_exponents)
    8. Per-resource prices      (price_per_resource)
    9. Per-tx pricing           (fee_per_tx, price_per_tx)
   10. Diagnostics              (backlogs_all_constraints)
"""

from __future__ import annotations

import numpy as np
import polars as pl


class Arbos60GasPricing:
    """ArbOS 60 per-resource max-over-sets pricing engine."""

    # ── 1. Constants ────────────────────────────────────────────────────────
    # Same minimum L2 base fee as ArbOS 51 (gwei).
    P_MIN_GWEI: float = 0.02

    # ── 2. Constraint sets — proposal definitions ──────────────────────────
    # Weights and ladders below come from the ArbOS 60 proposal; the same
    # tables are reproduced in docs/arbos51_vs_arbos60_equations.md.
    #
    # Each set is a weighted inequality Σ_k a_{i,k}·g_k ≤ T_{i,j}.
    # Symbols:
    #   c  = Computation (+ WasmComputation)
    #   sw = Storage Write (storageAccessWrite from per-tx multigas)
    #   sr = Storage Read  (storageAccessRead  from per-tx multigas)
    #   sg = Storage Growth
    #   hg = History Growth
    #   l2 = L2 Calldata
    SET_WEIGHTS: dict[str, dict[str, float]] = {
        "First Set — Storage/Compute mix 1":   {"c": 1.0,    "sw": 0.67, "sr": 0.14, "sg": 0.06},
        "Second Set — Storage/Compute mix 2":  {"c": 0.0625, "sw": 1.0,  "sr": 0.21, "sg": 0.09},
        "Third Set — History Growth":           {"hg": 1.0},
        "Fourth Set — Long-term Disk Growth":   {"sw": 0.8812, "sg": 0.2526, "hg": 0.301, "l2": 1.0},
    }

    # Per-set ladders: list of (T Mgas/s, A seconds). Sets 1-3 have 6
    # constraints, Set 4 has 1.
    SET_LADDERS: dict[str, list[tuple[float, int]]] = {
        "First Set — Storage/Compute mix 1": [
            (15.40, 10_000), (20.41, 2_861), (27.06,   819),
            (35.86,    234), (47.53,    67), (63.00,    19),
        ],
        "Second Set — Storage/Compute mix 2": [
            ( 3.13, 10_000), ( 4.16, 4_488), ( 5.53, 2_014),
            ( 7.35,    904), ( 9.77,   406), (12.99,   182),
        ],
        "Third Set — History Growth": [
            ( 67.30, 10_000), ( 81.27, 1_591), ( 98.14,   253),
            (118.50,     40), (143.10,     6), (172.80,     1),
        ],
        "Fourth Set — Long-term Disk Growth": [
            (2.30, 36_000),
        ],
    }

    # 6 priced resources. Order = dimension index k for p_k.
    PRICED_SYMBOLS: list[str] = ["c", "sw", "sr", "sg", "hg", "l2"]
    PRICED_SYMBOL_LABELS: dict[str, str] = {
        "c":  "Computation",
        "sw": "Storage Write",
        "sr": "Storage Read",
        "sg": "Storage Growth",
        "hg": "History Growth",
        "l2": "L2 Calldata",
    }

    def __init__(
        self,
        set_weights: dict[str, dict[str, float]] | None = None,
        set_ladders: dict[str, list[tuple[float, int]]] | None = None,
        p_min_gwei: float | None = None,
    ):
        self.set_weights = set_weights if set_weights is not None else self.SET_WEIGHTS
        self.set_ladders = set_ladders if set_ladders is not None else self.SET_LADDERS
        self.p_min_gwei  = p_min_gwei  if p_min_gwei  is not None else self.P_MIN_GWEI

    # ── 3. Time / aggregation helpers ───────────────────────────────────────
    @staticmethod
    def block_seconds_utc(block_times_utc: np.ndarray) -> np.ndarray:
        """Per-block integer-second UTC timestamps."""
        return block_times_utc.astype("datetime64[s]").astype(np.int64)

    @staticmethod
    def aggregate_per_second(
        values_per_block: np.ndarray,
        block_seconds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum per-block values into per-second buckets, dense from min..max
        of `block_seconds`. Empty seconds (no blocks landed) contribute 0
        to inflow but still drain backlog downstream — required for the
        backlog recursion to be correct.

        Assumes `block_seconds` is sorted ascending."""
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

    # ── 4. Exp approximation ────────────────────────────────────────────────
    @staticmethod
    def taylor4_exp(x: np.ndarray) -> np.ndarray:
        """Degree-4 Taylor approximation of exp. Identical to
        `Arbos51GasPricing.taylor4_exp` — see that docstring for the Nitro
        `model.go` https://github.com/OffchainLabs/nitro/blob/master/arbos/l2pricing/model.go."""
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        return 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0

    # ── 5. Resource gas builders ────────────────────────────────────────────
    def per_block_resource_gas(
        self,
        blocks: pl.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Per-block resource gas (raw gas units), keyed by the 6 priced symbols.
        Reads the per-tx multigas columns aggregated to per-block sums.

        Expects `blocks` to have these columns:
          computation, wasmComputation, storageAccessRead, storageAccessWrite,
          storageGrowth, historyGrowth, l2Calldata.
        """
        return {
            "c":  (blocks["computation"].to_numpy()
                   + blocks["wasmComputation"].to_numpy()).astype(np.float64),
            "sw": blocks["storageAccessWrite"].to_numpy().astype(np.float64),
            "sr": blocks["storageAccessRead"].to_numpy().astype(np.float64),
            "sg": blocks["storageGrowth"].to_numpy().astype(np.float64),
            "hg": blocks["historyGrowth"].to_numpy().astype(np.float64),
            "l2": blocks["l2Calldata"].to_numpy().astype(np.float64),
        }

    def per_tx_resource_split(
        self,
        computation: np.ndarray,
        wasm_computation: np.ndarray,
        storage_access_read: np.ndarray,
        storage_access_write: np.ndarray,
        storage_growth: np.ndarray,
        history_growth: np.ndarray,
        l2_calldata: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build the per-tx 6-resource gas dict expected by `fee_per_tx` /
        `price_per_tx` from raw multigas columns. l1Calldata is NOT included
        here — ArbOS 60 prices it separately via the L1 fee mechanism."""
        return {
            "c":  computation + wasm_computation,
            "sw": storage_access_write,
            "sr": storage_access_read,
            "sg": storage_growth,
            "hg": history_growth,
            "l2": l2_calldata,
        }

    # ── 6. Backlog (1-s tick recursion + per-block lookup) ─────────────────
    @staticmethod
    def backlog_per_second(
        inflow_per_second: np.ndarray,
        T_mgas_s,
    ) -> np.ndarray:
        """B at the END of each 1-s tick (gas units). One step per second:
            B(s+1) = max(0, B(s) + inflow(s) - T·1).

        Polymorphic in T: scalar T → 1-D output; array T (n_T,) → (n_T, n_seconds)."""
        T = np.asarray(T_mgas_s, dtype=np.float64)
        scalar = (T.ndim == 0)
        T_vec = np.atleast_1d(T)
        # T is in Mgas/s; multiply by 1e6 to get the per-second drain in
        # raw gas units (matching `inflow_per_second`).
        drain = T_vec * 1e6
        n = inflow_per_second.shape[0]
        d = inflow_per_second[None, :] - drain[:, None]
        C = np.empty((T_vec.shape[0], n + 1), dtype=np.float64)
        C[:, 0] = 0.0
        np.cumsum(d, axis=1, out=C[:, 1:])
        rmin = np.minimum.accumulate(C, axis=1)
        out = np.maximum(0.0, C - rmin)[:, :n]
        return out[0] if scalar else out

    @classmethod
    def backlog_per_block(
        cls,
        inflow_per_block: np.ndarray,
        block_seconds: np.ndarray,
        T_mgas_s,
    ) -> np.ndarray:
        """B as seen by each block's pricing step: B at the START of the
        block's wall-clock second (= B at end of the previous second).
        Initial condition: B = 0 at the start of the first second in the
        window. Blocks within the same second share the same B.

        Polymorphic in T: scalar T → (n_blocks,); array T (n_T,) → (n_T, n_blocks)."""
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
        batched pass over all (T_j) constraints. Output shape (n_blocks,)."""
        T_arr = np.fromiter((T for T, _ in ladder), dtype=np.float64, count=len(ladder))
        A_arr = np.fromiter((A for _, A in ladder), dtype=np.float64, count=len(ladder))
        B_per_block = cls.backlog_per_block(inflow_per_block, block_seconds, T_arr)  # (n_T, n_blk)
        norm = (A_arr * T_arr * 1e6)[:, None]
        return (B_per_block / norm).sum(axis=0)

    # ── 7. Exponent contribution + per-set raw exponents ────────────────────
    def exponent_contribution(
        self,
        inflow: np.ndarray,
        block_seconds: np.ndarray,
        T_mgas_s: float,
        A_s: float,
    ) -> np.ndarray:
        """B / (A·T) per block (uncapped). B is the start-of-second backlog
        from `backlog_per_block` (already ≥ 0)."""
        B    = self.backlog_per_block(inflow, block_seconds, T_mgas_s)
        norm = A_s * T_mgas_s * 1e6
        return B / norm

    def compute_set_exponents(
        self,
        parts: dict[str, np.ndarray],
        block_seconds: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Per-set raw exponent  E_i = Σ_j B_{i,j}/(A_{i,j}·T_{i,j})  over
        time. Inflow into set i is Σ_k a_{i,k}·g_k built from `parts`."""
        n = next(iter(parts.values())).shape[0]
        E_per_set: dict[str, np.ndarray] = {}
        for set_name, weights in self.set_weights.items():
            inflow = np.zeros(n, dtype=np.float64)
            for sym, w in weights.items():
                inflow = inflow + w * parts[sym]
            E_per_set[set_name] = self._ladder_exponent_from_inflow(
                inflow, block_seconds, self.set_ladders[set_name]
            )
        return E_per_set

    # ── 8. Per-resource prices (BB-alternative max-over-sets) ───────────────
    def price_per_resource(
        self,
        parts: dict[str, np.ndarray],
        block_seconds: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Per-block ArbOS 60 prices (one per priced resource symbol).

            e_k = max_i { a_{i,k} · E_i }            (only the binding set lifts p_k)
            p_k = p_min · taylor4_exp(e_k)            (no upper cap; e_k floored at 0)

        With e_k ≥ 0 and the Taylor-4 polynomial monotone increasing on
        [0, ∞), p_k ≥ p_min is preserved.

        Returns (prices, E_per_set):
          prices     {symbol → ndarray (n_blocks,)}
          E_per_set  {set_name → ndarray (n_blocks,)} — useful for diagnostics
        """
        n = next(iter(parts.values())).shape[0]
        E_per_set = self.compute_set_exponents(parts, block_seconds)

        prices: dict[str, np.ndarray] = {}
        for k in self.PRICED_SYMBOLS:
            # Walk the sets and take the max of a_{i,k}·E_i over those that
            # actually weight resource k (a_{i,k} > 0). Sets with a_{i,k}=0
            # contribute nothing to this resource's price by construction.
            e_k = np.zeros(n, dtype=np.float64)
            for set_name, weights in self.set_weights.items():
                a_ik = float(weights.get(k, 0.0))
                if a_ik == 0.0:
                    continue
                e_k = np.maximum(e_k, a_ik * E_per_set[set_name])
            e_k = np.maximum(e_k, 0.0)
            prices[k] = self.p_min_gwei * self.taylor4_exp(e_k)

        return prices, E_per_set

    # ── 9. Per-tx pricing (vectorised over txs) ─────────────────────────────
    def fee_per_tx(
        self,
        p_per_resource: dict[str, np.ndarray],
        tx_block_idx: np.ndarray,
        tx_g_per_resource: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Per-tx fee (gwei·gas) under ArbOS 60 per-resource pricing:
            fee_tx = Σ_k g_tx,k · p_k(block_of_tx)
        """
        n_tx = tx_block_idx.shape[0]
        fee = np.zeros(n_tx, dtype=np.float64)
        for k in self.PRICED_SYMBOLS:
            fee = fee + tx_g_per_resource[k] * p_per_resource[k][tx_block_idx]
        return fee

    def price_per_tx(
        self,
        p_per_resource: dict[str, np.ndarray],
        tx_block_idx: np.ndarray,
        tx_g_per_resource: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Per-tx price under ArbOS 60 — inner product:
            p_tx = Σ_k g_tx,k · p_k(block_of_tx)

        Dimensionally a fee (gwei·gas), not a per-gas price. Identical to
        `fee_per_tx`; kept as a separate name for callers that speak in
        "p_tx" terms.
        """
        return self.fee_per_tx(p_per_resource, tx_block_idx, tx_g_per_resource)

    # ── 10. Diagnostics ─────────────────────────────────────────────────────
    def backlogs_all_constraints(
        self,
        parts: dict[str, np.ndarray],
        block_seconds: np.ndarray,
    ) -> dict[str, dict[tuple[float, int], np.ndarray]]:
        """Per-block backlog (gas units) for every (set, constraint) pair,
        using the 1s-tick backlog recursion + start-of-second lookup.
        Returns {set_name: {(T, A): B_per_block}}."""
        out: dict[str, dict[tuple[float, int], np.ndarray]] = {}
        n = next(iter(parts.values())).shape[0]
        for set_name, weights in self.set_weights.items():
            inflow = np.zeros(n, dtype=np.float64)
            for sym, w in weights.items():
                inflow = inflow + w * parts[sym]
            ladder = self.set_ladders[set_name]
            T_arr = np.fromiter((T for T, _ in ladder), dtype=np.float64, count=len(ladder))
            B = self.backlog_per_block(inflow, block_seconds, T_arr)   # (n_T, n_blocks)
            out[set_name] = {(T, A): B[i] for i, (T, A) in enumerate(ladder)}
        return out
