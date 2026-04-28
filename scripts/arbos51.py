"""
ArbOS 51 dynamic-pricing equations.

Single-dimensional 6-constraint geometric ladder. One base fee per block,
applied uniformly to every gas unit. The full mechanism, in math:

    backlog (Lindley):
        B_j(t+1) = max(0, B_j(t) + g_total(t) - T_j · Δt)
        for j ∈ {0..5}, all six constraints fed the same total gas inflow.

    exponent contribution per constraint:
        e_j = max(0, B_j - tolerance·T_j) / (A_j · T_j)

    block price (same for all txs in a block):
        p_tx = p_min · exp( min(EXPONENT_CAP, Σ_j e_j) )

The Taylor-4 approximation of exp matches nitro's `ApproxExpBasisPoints(x, 4)`
inside `model.go` — without it, peak prices over-shoot real on-chain by 2-3×.

Calibration intent: the published Dia ladder values, BACKLOG_TOLERANCE_S = 0,
and Taylor-4 exp jointly reproduce on-chain hourly mean ArbOS 51 prices to
within ~1% over the Jan 2026 window.

The equations are fully vectorised over an array of consecutive blocks via
Lindley's running-max identity: with `d = inflow - drain`, cumulative sum
`C` and running min `rmin`, the backlog is `B = max(0, C - rmin)`.
"""

from __future__ import annotations

import numpy as np


# ── Constants ───────────────────────────────────────────────────────────────
# Minimum L2 base fee since ArbOS 51 Dia (gwei).
P_MIN_GWEI = 0.02

# Cap on the summed exponent — MaxPricingExponentBips = 85,000 bips = 8.5,
# producing a max multiplier of ≈ 4,915× over P_MIN_GWEI.
EXPONENT_CAP = 8.5

# Forgivable backlog before exponent lifts off floor (seconds × T gas).
# Nitro's compiled-in constant is 10; empirical calibration against on-chain
# prices for the multi-constraint ladder matches best with 0 per constraint.
BACKLOG_TOLERANCE_S = 0.0

# On-chain Dia ladder. Each entry is (T_j Mgas/s, A_j seconds). Single-dim:
# every constraint receives the full `total_l2_gas` per block.
ARBOS51_LADDER: list[tuple[float, int]] = [
    (10.0, 86_400),
    (14.0, 13_485),
    (20.0,  2_105),
    (29.0,    329),
    (41.0,     52),
    (60.0,      9),
]


# ── Vectorised helpers ──────────────────────────────────────────────────────
def taylor4_exp(x: np.ndarray) -> np.ndarray:
    """
    Degree-4 Taylor approximation of exp(x).
    Matches nitro's `ApproxExpBasisPoints(x, 4)` in `model.go`.
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    return 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0


def backlog_per_block(
    inflow_gas_per_block: np.ndarray,
    dt_s_per_block: np.ndarray,
    T_mgas_s: float,
) -> np.ndarray:
    """
    Lindley-floored backlog B_j per block, in gas units.

    Vectorised form:
        d = inflow - drain            (per block)
        C = cumsum(d), C[0] = 0
        rmin = running_min(C)
        B = max(0, C - rmin)          (Lindley identity)
    """
    drain = T_mgas_s * 1e6 * dt_s_per_block          # Mgas/s × s = gas
    d = inflow_gas_per_block - drain
    C = np.empty(len(d) + 1, dtype=np.float64)
    C[0] = 0.0
    np.cumsum(d, out=C[1:])
    rmin = np.minimum.accumulate(C)
    return np.maximum(0.0, C - rmin)[: len(d)]


def exponent_contribution(
    inflow_gas_per_block: np.ndarray,
    dt_s_per_block: np.ndarray,
    T_mgas_s: float,
    A_s: float,
    tolerance_s: float = BACKLOG_TOLERANCE_S,
) -> np.ndarray:
    """
    B_effective / (A·T) per block for one constraint (uncapped).

    The 8.5 cap is applied once to the SUM of all constraints' contributions
    by the caller (`price_per_block`), not per constraint.
    """
    B = backlog_per_block(inflow_gas_per_block, dt_s_per_block, T_mgas_s)
    forgivable = tolerance_s * T_mgas_s * 1e6
    B_eff = np.maximum(0.0, B - forgivable)
    norm = A_s * T_mgas_s * 1e6
    return B_eff / norm


def dt_seconds_per_block(block_times_utc: np.ndarray) -> np.ndarray:
    """
    Integer-second wall-clock delta between consecutive block headers.

    Arbitrum's pricing model drains backlog by `(header.Time - lastHeader.Time)
    × T` where the diff is an integer number of seconds. On ~250 ms blocks
    that means ~3 of 4 consecutive blocks drain zero and every 4th drains a
    full second's worth.

    `block_times_utc` must be an array of numpy datetime64 (any unit).
    """
    t = block_times_utc.astype("datetime64[s]").astype(np.int64)
    dt = np.empty_like(t, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = np.diff(t).astype(np.float64)
    return dt


# ── Per-block price ─────────────────────────────────────────────────────────
def price_per_block(
    total_gas: np.ndarray,
    dt_s: np.ndarray,
    p_min_gwei: float = P_MIN_GWEI,
    ladder: list[tuple[float, int]] | None = None,
) -> np.ndarray:
    """
    Per-block ArbOS 51 base fee in gwei.

        e = Σ_j  B_j / (A_j · T_j)
        p = p_min · taylor4_exp( min(EXPONENT_CAP, e) )

    Inputs:
      total_gas — per-block total L2 gas (gas units, not Mgas).
      dt_s      — per-block integer-second deltas (see dt_seconds_per_block).
      ladder    — defaults to ARBOS51_LADDER.
    """
    if ladder is None:
        ladder = ARBOS51_LADDER
    expo = np.zeros_like(total_gas, dtype=np.float64)
    for T, A in ladder:
        expo = expo + exponent_contribution(total_gas, dt_s, T, A)
    expo = np.clip(expo, 0.0, EXPONENT_CAP)
    return p_min_gwei * taylor4_exp(expo)


def backlogs_all_constraints(
    total_gas: np.ndarray,
    dt_s: np.ndarray,
    ladder: list[tuple[float, int]] | None = None,
) -> dict[tuple[float, int], np.ndarray]:
    """Per-block backlog (gas units) for each (T, A) in the ladder."""
    if ladder is None:
        ladder = ARBOS51_LADDER
    return {
        (T, A): backlog_per_block(total_gas, dt_s, T) for T, A in ladder
    }
