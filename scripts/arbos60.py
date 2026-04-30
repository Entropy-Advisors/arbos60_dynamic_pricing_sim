"""
ArbOS 60 dynamic-pricing equations — per-resource max-over-sets pricing.

Four constraint sets, 19 constraints total. Each set i is a weighted
inequality with its own (T_{i,j}, A_{i,j}) ladder. Same backlog/exp
machinery as ArbOS 51, but wrapped with per-set inflows and a max-over-sets
projection that yields one price per resource. Backlog evolves at 1-second
ticks; per-block backlog is a lookup of the per-second time series.

    backlog (Δt = 1 s tick, per (i, j)):
        B_{i,j}(t+1) = max(0, B_{i,j}(t) + Σ_k a_{i,k}·g_k(t) - T_{i,j}·Δt)

    set i raw exponent (per block, via start-of-second lookup):
        E_i(t) = Σ_j B_{i,j}(t) / (A_{i,j}·T_{i,j})

    per-resource price:
        p_k = p_min · exp(max_i { a_{i,k} · E_i })

    per-tx fee / "price" (inner product):
        fee_tx = Σ_k g_tx,k · p_k(block_of_tx)

    fee aggregated over a window (e.g. an hour):
        p̄ = Σ_tx fee_tx / Σ_tx Σ_k g_tx,k

Public surface — the `Arbos60GasPricing` class. Method order mirrors the
equation flow:

    1.  Constants               (P_min)
    2.  Constraint sets         (SET_WEIGHTS_1/2, SET_LADDERS_1/2, GAS_RESOURCES)
    3.  Time / aggregation      (block_seconds_utc, aggregate_per_second)
    4.  Exp approximation       (taylor4_exp)
    5.  Resource gas builders   (per_block_resource_gas, per_tx_resource_split)
    6.  Per-set inflow per t    (compute_inflow_per_t)
    7.  Backlog                 (backlog_per_second)
    8.  Per-set raw exponent    (compute_set_exponents)
    9.  Per-resource prices     (price_per_resource)
    10. Per-tx pricing          (fee_per_tx)
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
    # tables are reproduced in docs/arbos51_vs_arbos60_equations.md
    #
    # Each set is a weighted inequality Σ_k a_{i,k}·g_k ≤ T_{i,j}.
    # Symbols:
    #   c  = Computation (+ WasmComputation)
    #   sw = Storage Write (storageAccessWrite from per-tx multigas)
    #   sr = Storage Read  (storageAccessRead  from per-tx multigas)
    #   sg = Storage Growth
    #   hg = History Growth
    #   l2 = L2 Calldata
    # ── Set 1 ──────────────────────────────────────────────────────────────
    SET_WEIGHTS_1: dict[str, dict[str, float]] = {
        "First Set — Storage/Compute mix 1":   {"c": 1.0,    "sw": 0.67, "sr": 0.14, "sg": 0.06},
        "Second Set — Storage/Compute mix 2":  {"c": 0.0625, "sw": 1.0,  "sr": 0.21, "sg": 0.09},
        "Third Set — History Growth":           {"hg": 1.0},
        "Fourth Set — Long-term Disk Growth":   {"sw": 0.8812, "sg": 0.2526, "hg": 0.301, "l2": 1.0},
    }
    SET_LADDERS_1: dict[str, list[tuple[float, int]]] = {
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

    # ── Set 2 ──────────────────────────────────────────────────────────────
    SET_WEIGHTS_2: dict[str, dict[str, float]] = {
        "First Set — Storage/Compute mix 1":   {"c": 1.0,    "sw": 0.6714, "sr": 0.141, "sg": 0.0604},
        "Second Set — Storage/Compute mix 2":  {"c": 0.0625, "sw": 1.0,    "sr": 0.21,  "sg": 0.09},
        "Third Set — History Growth":          {"hg": 1.0},
        "Fourth Set — Long-term Disk Growth":  {"sw": 0.8812, "sg": 0.2526, "hg": 0.301, "l2": 1.0},
    }
    SET_LADDERS_2: dict[str, list[tuple[float, int]]] = {
        "First Set — Storage/Compute mix 1":   [(15.40, 10_000), ( 55.12,    14)],
        "Second Set — Storage/Compute mix 2":  [( 3.13, 10_000), ( 10.09,   102)],
        "Third Set — History Growth":          [(67.30, 10_000), (166.04,     1)],
        "Fourth Set — Long-term Disk Growth":  [( 2.30, 36_000)],
    }

    # Map: set number → (weights, ladders) pair.
    _SETS: dict[int, tuple[dict, dict]] = {
        1: (SET_WEIGHTS_1, SET_LADDERS_1),
        2: (SET_WEIGHTS_2, SET_LADDERS_2),
    }

    # 6 priced resources. Order = dimension index k for p_k.
    GAS_RESOURCES: list[str] = ["c", "sw", "sr", "sg", "hg", "l2"]
    GAS_RESOURCE_LABELS: dict[str, str] = {
        "c":  "Computation",
        "sw": "Storage Write",
        "sr": "Storage Read",
        "sg": "Storage Growth",
        "hg": "History Growth",
        "l2": "L2 Calldata",
    }

    def __init__(
        self,
        version: int = 1,
        set_weights: dict[str, dict[str, float]] | None = None,
        set_ladders: dict[str, list[tuple[float, int]]] | None = None,
        p_min_gwei: float | None = None,
    ):
        """`version` selects which (weights, ladders) preset to use; see
        `_SETS`. Explicit `set_weights` / `set_ladders` override the preset."""
        if version not in self._SETS:
            raise ValueError(
                f"Unknown version {version}; available: {sorted(self._SETS)}"
            )
        default_weights, default_ladders = self._SETS[version]
        self.version     = version
        self.set_weights = set_weights if set_weights is not None else default_weights
        self.set_ladders = set_ladders if set_ladders is not None else default_ladders
        self.p_min_gwei  = p_min_gwei  if p_min_gwei  is not None else self.P_MIN_GWEI

    # ── 3. Time / aggregation helpers ───────────────────────────────────────
    @staticmethod
    def block_seconds_utc(block_times_utc: np.ndarray) -> np.ndarray:
        """Per-block integer-second UTC timestamps."""
        return block_times_utc.astype("datetime64[s]").astype(np.int64)

    @staticmethod
    def aggregate_per_second(
        values_per_block: np.ndarray,
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum per-block values into per-t (1 s) buckets, dense from
        min..max of `block_t`.  Empty seconds (no blocks landed) carry 0
        inflow but still drain backlog downstream — required for the
        backlog tick to be correct.

        Assumes `block_t` is sorted ascending."""
        t_min  = int(block_t[0])                                # window start (UTC s)
        t_max  = int(block_t[-1])                               # window end   (UTC s)
        n_t    = t_max - t_min + 1                              # # of 1 s buckets
        t_axis = np.arange(t_min, t_max + 1, dtype=np.int64)    # dense t axis
        t_idx  = (block_t - t_min).astype(np.int64)             # bucket index per block
        summed = np.bincount(                                   # Σ values_per_block in each bucket
            t_idx,
            weights=np.asarray(values_per_block, dtype=np.float64),
            minlength=n_t,                                       # pads empty seconds with 0
        )
        return t_axis, summed


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
        """Build the per-tx 6-resource gas dict expected by `fee_per_tx`
        from raw multigas columns. l1Calldata is not included."""
        return {
            "c":  computation + wasm_computation,
            "sw": storage_access_write,
            "sr": storage_access_read,
            "sg": storage_growth,
            "hg": history_growth,
            "l2": l2_calldata,
        }

    # ── 6. Per-set inflow (Σ_k a_{i,k} · g_k aggregated to t) ───────────────
    def compute_inflow_per_t(
        self,
        g_per_block: dict[str, np.ndarray],
        a_i: dict[str, float],
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-t weighted inflow for one set i:
            inflow_i(t) = Σ_k a_{i,k} · g_k(t)

        Each priced resource k contributes its per-block gas weighted by
        a_{i,k}; the weighted sum is aggregated from per-block to per-t
        (1 s buckets) so it can be fed into the backlog tick.

        Returns (t_axis, inflow_i_per_t).  Empty seconds in the window
        contribute 0 inflow but still drain B_{i,j} downstream, required
        for the backlog recursion to be correct.
        """
        # Pull n_blocks from any one of the resource arrays — they share length
        n_blocks = next(iter(g_per_block.values())).shape[0]
        # Running per-block accumulator for Σ_k a_{i,k} · g_k.
        inflow_i_per_block = np.zeros(n_blocks, dtype=np.float64)
        # Walk the (resource → weight) entries of set i; missing keys mean
        # a_{i,k} = 0, so they're naturally skipped by the dict iteration
        for k, a_ik in a_i.items():
            inflow_i_per_block += a_ik * g_per_block[k]              # add a_{i,k} · g_k
        # Per-block → per-t aggregation (1 s buckets, dense over window)
        return self.aggregate_per_second(inflow_i_per_block, block_t)

    # ── 7. Backlog (1-s tick loop) ──────────────────────────────────────────
    @staticmethod
    def backlog_per_second(
        inflow_i_per_t: np.ndarray,   # = Σ_k a_{i,k}·g_k(t), per t (one set i)
        T_i: np.ndarray,              # T_{i,j} ladder for the set i's constraints
    ) -> np.ndarray:
        """Backlog tick for one set i, vectorised over j:
            B_{i,j}(t+1) = max(0, B_{i,j}(t) + inflow_i(t) − T_{i,j}·Δt)

        `inflow_i_per_t` is supplied by compute_inflow_per_t.  Returns
        (n_j, n_t); out[j, t] is the start-of-t backlog (pre-update),
        which is what a block landing in t reads.
        """
        drain = T_i * 1e6                                       # Mgas/s -> gas/s
        out = np.empty((T_i.shape[0], inflow_i_per_t.shape[0]))
        B_ij = np.zeros(T_i.shape[0])                           # B_{i,j}(0) = 0
        for t, inflow_i_t in enumerate(inflow_i_per_t):
            out[:, t] = B_ij                                    # snapshot B_{i,j}(t)
            B_ij = np.maximum(0.0, B_ij + inflow_i_t - drain)   # → B_{i,j}(t+1)
        return out

    # ── 8. Per-set raw exponent ─────────────────────────────────────────────
    def compute_set_exponents(
        self,
        g_per_block: dict[str, np.ndarray],
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """E_i(t) = Σ_j B_{i,j}(t) / (A_{i,j}·T_{i,j})   for each set i.

        Indices:
            i = set of constraints
            j = constraint within set i  (rung of the ladder)
            k = resource  (6 priced resources in GAS_RESOURCES)

        `g_per_block` maps k → g_k(per block).  `block_t` is the per-block
        wall-clock t (UTC seconds since epoch).  Returns
        (t_axis, {set_name → E_i, shape (n_t,)}).
        """
        E_per_set: dict[str, np.ndarray] = {}
        t_axis = np.empty(0, dtype=np.int64)  # shared across sets (overwritten in-loop)
        for set_name, a_i in self.set_weights.items():  # iterate sets i
            # Step 6: build inflow_i(t) = Σ_k a_{i,k}·g_k(t) on the dense t axis
            t_axis, inflow_i_per_t = self.compute_inflow_per_t(
                g_per_block, a_i, block_t,
            )
            # Pull the j-indexed (T_{i,j}, A_{i,j}) pairs for this set.
            ladder = self.set_ladders[set_name]
            T_i = np.array([T for T, _ in ladder], dtype=np.float64)  # T_{i,j}, j = 0..n_j-1
            A_i = np.array([A for _, A in ladder], dtype=np.float64)  # A_{i,j}, j = 0..n_j-1
            # Step 7: backlog tick → B_{i,j}(t), shape (n_j, n_t).
            B_i = self.backlog_per_second(inflow_i_per_t, T_i)
            # E_i(t) = Σ_j B_{i,j}(t) / (A_{i,j} · T_{i,j}).  T_i*1e6 puts the
            # denominator in gas/s so it matches the gas-units backlog
            E_per_set[set_name] = (
                B_i / (A_i * T_i * 1e6)[:, None]  # broadcast over t
            ).sum(axis=0)   # fold over j → (n_t,)
        return t_axis, E_per_set

    # ── 9. Per-resource prices (max-over-sets) ─────────────────────────────
    def price_per_resource(
        self,
        g_per_block: dict[str, np.ndarray],
        block_t: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Per-t ArbOS 60 prices (one per priced resource k).

            e_k(t) = max_i { a_{i,k} · E_i(t) }
            p_k(t) = p_min · taylor4_exp(e_k(t))

        Indices: i = set, k = resource (∈ GAS_RESOURCES).
        Returns (t_axis, prices, E_per_set), all per-t.
        Caller gathers to per-block / per-tx via:
            t_idx            = (block_t - t_axis[0]).astype(np.int64)
            prices_per_block = {k: prices[k][t_idx] for k in GAS_RESOURCES}
        """
        # ── Step 1 ─────────────────────────────────────────────────────────
        # E_i(t) = Σ_j B_{i,j}(t) / (A_{i,j}·T_{i,j})  for every set i.
        t_axis, E_per_set = self.compute_set_exponents(g_per_block, block_t)
        n_t = len(t_axis)

        # ── Step 2 ─────────────────────────────────────────────────────────
        # For each resource k, the spec says
        #     e_k(t) = max_i { a_{i,k} · E_i(t) }                       (eq. 1)
        # i.e. take the *maximum over sets i* of the resource-weighted
        # exponent.  Implemented as a running np.maximum over the i loop:
        # `e_k` accumulates the largest a_{i,k}·E_i seen so far.
        # Sets with a_{i,k}=0 cannot lift the max (they contribute 0), so
        # we skip them as an optimisation — semantically equivalent.
        prices: dict[str, np.ndarray] = {}
        for k in self.GAS_RESOURCES:
            e_k = np.zeros(n_t, dtype=np.float64)               # max-over-i, init at 0
            for set_name, a_i in self.set_weights.items():      # iterate sets i
                a_ik = float(a_i.get(k, 0.0))                   # weight a_{i,k}
                if a_ik == 0.0:
                    continue                                    # a_{i,k}=0 -> can't be the max
                candidate = a_ik * E_per_set[set_name]          # a_{i,k} · E_i(t)
                e_k = np.maximum(e_k, candidate)                # max over sets i

            # ── Step 3 ─────────────────────────────────────────────────────
            # p_k(t) = p_min · taylor4_exp(e_k(t)).
            # No max(0, …) clip needed: e_k ≥ 0 by construction (B, A, T,
            # a are all non-negative; max is initialised at 0).
            prices[k] = self.p_min_gwei * self.taylor4_exp(e_k)

        return t_axis, prices, E_per_set

    # ── 10. Per-tx fee (vectorised over txs) ────────────────────────────────
    def fee_per_tx(
        self,
        p_per_resource: dict[str, np.ndarray],
        tx_block_idx: np.ndarray,
        tx_g_per_resource: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Per-tx fee (gwei·gas) under ArbOS 60 per-resource pricing:

            fee_tx = Σ_k g_tx,k · p_k(block_of_tx)"""
        n_tx = tx_block_idx.shape[0]
        fee = np.zeros(n_tx, dtype=np.float64)
        for k in self.GAS_RESOURCES:
            fee = fee + tx_g_per_resource[k] * p_per_resource[k][tx_block_idx]
        return fee

