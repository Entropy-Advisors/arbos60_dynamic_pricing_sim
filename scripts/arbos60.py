"""
ArbOS 60 dynamic-pricing equations (per-resource pricing, BB-alternative form).

Four constraint sets, 19 constraints total. Each set i is a weighted
inequality with its own (T_{i,j}, A_{i,j}) ladder. The same Lindley/exponent
machinery as ArbOS 51, but wrapped with per-set inflows and a max-over-sets
projection that yields one price per resource.

    backlog (Lindley, per (i, j)):
        B_{i,j}(t+1) = max(0, B_{i,j}(t) + Σ_k a_{i,k}·g_k(t) - T_{i,j}·Δt)

    set i raw exponent:
        E_i = Σ_j B_{i,j} / (A_{i,j}·T_{i,j})

    per-resource price (BB alternative — see Adjustment_Windows.pdf):
        p_k = p_min · exp( min(EXPONENT_CAP, max_i { a_{i,k} · E_i }) )

    per-tx price (inner product of resource shares with the price vector):
        p_tx = Σ_k w_{tx,k} · p_k    where w_{tx,k} = g_{tx,k} / Σ_k g_{tx,k}

    fee aggregated over a window (e.g. an hour):
        p̄ = Σ_tx fee_tx / Σ_tx Σ_k g_{tx,k}
          = Σ_block Σ_k p_k(block)·G_k(block) / Σ_block Σ_k G_k(block)

The `max_i` form (BB note in the PDF) was chosen over the canonical
`Σ_i a_{i,k}·E_i` form: only the binding set drives each resource's price,
which avoids double-counting when multiple sets share resources.

Storage access is split into read/write at module level (R_OVER_W_RATIO),
since this is the per-set inflow weighting that the proposal expects.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# ── Constants ───────────────────────────────────────────────────────────────
P_MIN_GWEI          = 0.02
EXPONENT_CAP        = 8.5
BACKLOG_TOLERANCE_S = 0.0

# Storage-access R:W split. The Jan 2026 source data didn't separate read/
# write, so we apply a fixed ratio (midpoint of two observed point estimates).
# The Oct 2025 dump has `storageAccessRead` / `storageAccessWrite` natively;
# downstream code should prefer those when available.
R_OVER_W_RATIO = 3.22
SA_WRITE_SHARE = 1.0             / (R_OVER_W_RATIO + 1.0)   # 0.237
SA_READ_SHARE  = R_OVER_W_RATIO  / (R_OVER_W_RATIO + 1.0)   # 0.763

# ── Constraint sets — proposal definitions ──────────────────────────────────
# Each set is a weighted inequality Σ_k a_{i,k}·g_k ≤ T_{i,j}.
# Symbols:
#   c  = Computation (+ WasmComputation)
#   sw = Storage Write (storageAccess × SA_WRITE_SHARE)
#   sr = Storage Read  (storageAccess × SA_READ_SHARE)
#   sg = Storage Growth
#   hg = History Growth
#   l2 = L2 Calldata
SET_WEIGHTS: dict[str, dict[str, float]] = {
    "First Set — Storage/Compute mix 1":   {"c": 1.0,    "sw": 0.67, "sr": 0.14, "sg": 0.06},
    "Second Set — Storage/Compute mix 2":  {"c": 0.0625, "sw": 1.0,  "sr": 0.21, "sg": 0.09},
    "Third Set — History Growth":           {"hg": 1.0},
    "Fourth Set — Long-term Disk Growth":   {"sw": 0.8812, "sg": 0.2526, "hg": 0.301, "l2": 1.0},
}

# Per-set ladders: list of (T Mgas/s, A seconds). Sets 1-3 have 6 constraints,
# Set 4 has 1.
SET_LADDERS: dict[str, list[tuple[float, int]]] = {
    "First Set — Storage/Compute mix 1": [
        (15.40, 10_000), (20.41, 4_236), (27.06, 1_795),
        (35.86,    760), (47.53,   322), (63.00,   136),
    ],
    "Second Set — Storage/Compute mix 2": [
        ( 3.13, 10_000), ( 4.16, 4_682), ( 5.53, 2_192),
        ( 7.35,  1_026), ( 9.77,   480), (12.99,   225),
    ],
    "Third Set — History Growth": [
        ( 67.30, 10_000), ( 81.27, 1_593), ( 98.14,   254),
        (118.50,     40), (143.10,     6), (172.80,     1),
    ],
    "Fourth Set — Long-term Disk Growth": [
        (2.30, 36_000),
    ],
}

# 6 priced resources. Order matters — must match the dimension index k for p_k.
PRICED_SYMBOLS: list[str] = ["c", "sw", "sr", "sg", "hg", "l2"]
PRICED_SYMBOL_LABELS: dict[str, str] = {
    "c":  "Computation",
    "sw": "Storage Write",
    "sr": "Storage Read",
    "sg": "Storage Growth",
    "hg": "History Growth",
    "l2": "L2 Calldata",
}


# ── Vectorised Lindley / exponent helpers (same form as ArbOS 51) ───────────
def taylor4_exp(x: np.ndarray) -> np.ndarray:
    """Degree-4 Taylor approximation of exp — matches nitro's
    ApproxExpBasisPoints(x, 4)."""
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    return 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0


def backlog_per_block(
    inflow_gas_per_block: np.ndarray,
    dt_s_per_block: np.ndarray,
    T_mgas_s: float,
) -> np.ndarray:
    """Lindley-floored backlog (gas units). See arbos51.backlog_per_block."""
    drain = T_mgas_s * 1e6 * dt_s_per_block
    d = inflow_gas_per_block - drain
    C = np.empty(len(d) + 1, dtype=np.float64)
    C[0] = 0.0
    np.cumsum(d, out=C[1:])
    rmin = np.minimum.accumulate(C)
    return np.maximum(0.0, C - rmin)[: len(d)]


def exponent_contribution(
    inflow: np.ndarray,
    dt_s: np.ndarray,
    T_mgas_s: float,
    A_s: float,
    tolerance_s: float = BACKLOG_TOLERANCE_S,
) -> np.ndarray:
    """B_effective / (A·T) per block for one constraint (uncapped)."""
    B = backlog_per_block(inflow, dt_s, T_mgas_s)
    forgivable = tolerance_s * T_mgas_s * 1e6
    B_eff = np.maximum(0.0, B - forgivable)
    norm = A_s * T_mgas_s * 1e6
    return B_eff / norm


def dt_seconds_per_block(block_times_utc: np.ndarray) -> np.ndarray:
    """Integer-second wall-clock delta between consecutive block headers."""
    t = block_times_utc.astype("datetime64[s]").astype(np.int64)
    dt = np.empty_like(t, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = np.diff(t).astype(np.float64)
    return dt


# ── Per-block resource gas split ────────────────────────────────────────────
def per_block_resource_gas(
    blocks: pl.DataFrame,
    sa_write_share: float = SA_WRITE_SHARE,
    sa_read_share:  float = SA_READ_SHARE,
) -> dict[str, np.ndarray]:
    """
    Per-block resource gas (in raw gas units), keyed by the 6 priced symbols.

    Expects `blocks` to have these columns:
      total_l2_gas, w_Computation, w_WasmComputation, w_StorageAccess,
      w_StorageGrowth, w_HistoryGrowth, w_L2Calldata.
    """
    total = blocks["total_l2_gas"].to_numpy()
    sa = blocks["w_StorageAccess"].to_numpy() * total
    return {
        "c":  (blocks["w_Computation"].to_numpy()
               + blocks["w_WasmComputation"].to_numpy()) * total,
        "sw": sa * sa_write_share,
        "sr": sa * sa_read_share,
        "sg": blocks["w_StorageGrowth"].to_numpy() * total,
        "hg": blocks["w_HistoryGrowth"].to_numpy() * total,
        "l2": blocks["w_L2Calldata"].to_numpy()    * total,
    }


# ── Per-set raw exponents and per-resource prices ───────────────────────────
def compute_set_exponents(
    parts: dict[str, np.ndarray],
    dt_s: np.ndarray,
    set_weights: dict[str, dict[str, float]] = SET_WEIGHTS,
    set_ladders: dict[str, list[tuple[float, int]]] = SET_LADDERS,
) -> dict[str, np.ndarray]:
    """
    Per-set raw exponent E_i = Σ_j B_{i,j}/(A_{i,j}·T_{i,j}) over time.
    Inflow into set i is Σ_k a_{i,k}·g_k built from `parts`.
    """
    n = next(iter(parts.values())).shape[0]
    E_per_set: dict[str, np.ndarray] = {}
    for set_name, weights in set_weights.items():
        inflow = np.zeros(n, dtype=np.float64)
        for sym, w in weights.items():
            inflow = inflow + w * parts[sym]
        E = np.zeros(n, dtype=np.float64)
        for T, A in set_ladders[set_name]:
            E = E + exponent_contribution(inflow, dt_s, T, A)
        E_per_set[set_name] = E
    return E_per_set


def price_per_resource(
    parts: dict[str, np.ndarray],
    dt_s: np.ndarray,
    p_min_gwei: float = P_MIN_GWEI,
    set_weights: dict[str, dict[str, float]] = SET_WEIGHTS,
    set_ladders: dict[str, list[tuple[float, int]]] = SET_LADDERS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Per-block ArbOS 60 prices (one per priced resource symbol) under the
    BB-alternative max-over-sets formulation.

        e_k = min(EXPONENT_CAP, max_i { a_{i,k} · E_i })
        p_k = p_min · taylor4_exp(e_k)

    Returns (prices, E_per_set) where prices is keyed by symbol and
    E_per_set is the raw uncapped per-set exponent (useful for diagnostics).
    """
    n = next(iter(parts.values())).shape[0]
    E_per_set = compute_set_exponents(parts, dt_s, set_weights, set_ladders)

    prices: dict[str, np.ndarray] = {}
    for k in PRICED_SYMBOLS:
        e_k = np.zeros(n, dtype=np.float64)
        for set_name, weights in set_weights.items():
            a_ik = float(weights.get(k, 0.0))
            if a_ik == 0.0:
                continue
            e_k = np.maximum(e_k, a_ik * E_per_set[set_name])
        e_k = np.clip(e_k, 0.0, EXPONENT_CAP)
        prices[k] = p_min_gwei * taylor4_exp(e_k)

    return prices, E_per_set


def block_fee_and_gas(
    parts: dict[str, np.ndarray],
    dt_s: np.ndarray,
    p_min_gwei: float = P_MIN_GWEI,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-block (fee, priced_gas) — the pieces needed to aggregate per-tx
    fee/gas to any time window.

        fee_block = Σ_k p_k(block) · G_k(block)
        gas_block = Σ_k G_k(block)             (priced symbols only)

    Within a block all txs share the same p_k vector, so this per-block
    aggregation is identical to summing per-tx fees first.
    """
    prices, _ = price_per_resource(parts, dt_s, p_min_gwei)
    fee = np.zeros(len(dt_s), dtype=np.float64)
    gas = np.zeros(len(dt_s), dtype=np.float64)
    for k in PRICED_SYMBOLS:
        fee = fee + prices[k] * parts[k]
        gas = gas + parts[k]
    return fee, gas


def backlogs_all_constraints(
    parts: dict[str, np.ndarray],
    dt_s: np.ndarray,
    set_weights: dict[str, dict[str, float]] = SET_WEIGHTS,
    set_ladders: dict[str, list[tuple[float, int]]] = SET_LADDERS,
) -> dict[str, dict[tuple[float, int], np.ndarray]]:
    """
    Per-block backlog (gas units) for every (set, constraint) pair.
    Returns {set_name: {(T, A): B_per_block}}.
    """
    out: dict[str, dict[tuple[float, int], np.ndarray]] = {}
    n = next(iter(parts.values())).shape[0]
    for set_name, weights in set_weights.items():
        inflow = np.zeros(n, dtype=np.float64)
        for sym, w in weights.items():
            inflow = inflow + w * parts[sym]
        out[set_name] = {
            (T, A): backlog_per_block(inflow, dt_s, T)
            for T, A in set_ladders[set_name]
        }
    return out
