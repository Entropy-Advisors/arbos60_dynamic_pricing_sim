"""
Capacity headroom under ArbOS 60 vs ArbOS 51.

Ben Berger's framing (Apr 30 sync):
    "Capacity = the gas-per-second throughput at which the price first
     starts going up.  In ArbOS 51 it's a constant (10 Mgas/s post-DIA).
     In ArbOS 60 it depends on the workload mix — for the second
     constraint set, my back-of-envelope says ~15 Mgas/s, i.e. about a
     50 % capacity gain on average."

Per-second math:
    For every set i, rung j the spec says
        Σ_k a_{i,k}·g_k(t)  ≤  T_{i,j}
    Define the workload mix α_k(t) = g_k(t)/G(t).  Then plugging in:
        G(t) ≤ T_{i,j} / Σ_k a_{i,k}·α_k(t)
    Taking j = 0 (smallest T per set ⇒ binding for sustained throughput):
        capacity_60(t) = min_i  T_{i,0} / Σ_k a_{i,k}·α_k(t)
    ArbOS 51 just has G(t) ≤ T_51, i.e. capacity_51 = T_51 (regime-aware).

The metric we plot:
    headroom_pct(t) = ( capacity(t) − G(t) ) / capacity(t) × 100
                    = % more gas/sec the chain could absorb at this mix
                      before the price would move above p_min.
A "saturation" second has headroom_pct(t) < (1 − THRESHOLD) × 100,
i.e. realised G is at or above THRESHOLD of the capacity ceiling.

Top panel shows the ArbOS 60 simulated prices that come out of
`arbos60.Arbos60GasPricing.price_per_resource` — the same engine the
historical-sim chart uses — so you can sanity-check that the prices we
derive capacity from are the prices our code actually produces.
"""

from __future__ import annotations

import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import historical_sim as hs                                      # noqa: E402
import arbos60 as a60_mod                                        # noqa: E402

_ROOT     = _HERE.parent
OUT_HTML  = _ROOT / "figures" / "capacity_estimator.html"
PRICES_CACHE = _ROOT / "data" / "capacity_hourly_prices.parquet"

# ── Capacity constants ──────────────────────────────────────────────────────
T51_PRE_DIA   = 7.0       # Mgas/s, single rung pre-DIA
T51_POST_DIA  = 10.0      # Mgas/s, smallest rung of the Dia ladder
DIA_LAUNCH_TS = datetime(2026, 1, 8, 17, 0, 0)

# Both Set 1 and Set 2 share the same first-rung T per set i, so
# steady-state capacity is identical between configs (they differ only
# in higher rungs and a tiny weight perturbation).  We use Set 1.
SET_WEIGHTS = a60_mod.Arbos60GasPricing.SET_WEIGHTS_1
SET_LADDERS = a60_mod.Arbos60GasPricing.SET_LADDERS_1

RESOURCES = ["c", "sw", "sr", "sg", "hg", "l2"]
RESOURCE_LABEL = {
    "c":  "Computation",
    "sw": "Storage Write",
    "sr": "Storage Read",
    "sg": "Storage Growth",
    "hg": "History Growth",
    "l2": "L2 Calldata",
}
RESOURCE_COL = {
    "c":  ("computation", "wasmComputation"),
    "sw": ("storageAccessWrite",),
    "sr": ("storageAccessRead",),
    "sg": ("storageGrowth",),
    "hg": ("historyGrowth",),
    "l2": ("l2Calldata",),
}
RESOURCE_COLOR = {
    "c":  "#1f77b4", "sw": "#2ca02c", "sr": "#98df8a",
    "sg": "#d62728", "hg": "#ff7f0e", "l2": "#9467bd",
}

THRESHOLD = 0.80   # G ≥ THRESHOLD · capacity ⇒ "near saturation"
P_MIN_GWEI = a60_mod.Arbos60GasPricing.P_MIN_GWEI


# ── Data loading ───────────────────────────────────────────────────────────
def load_per_block_with_time() -> pl.DataFrame:
    print("Loading per-block resources (rebuild if missing)...")
    res = hs.build_per_block_resources()
    print(f"  per-block resources: {res.height:,} rows")
    blocks_pq = _ROOT / "data" / "onchain_blocks_transactions" / "per_block.parquet"
    blocks = (
        pl.scan_parquet(str(blocks_pq))
          .select(["block_number", "block_time"])
          .collect()
    )
    df = (
        res.rename({"block": "block_number"})
           .join(blocks, on="block_number", how="inner")
           .sort("block_number")
    )
    print(f"  joined: {df.height:,} blocks with both gas + time")
    return df


# ── Per-second aggregation (numpy bincount) ────────────────────────────────
def aggregate_per_second(df: pl.DataFrame):
    """Returns (sec_epoch [n_sec], G [n_sec], g_per_resource [n_sec, 6])."""
    secs = (
        df["block_time"].cast(pl.Datetime("us")).dt.timestamp("ms").to_numpy() // 1000
    ).astype(np.int64)
    sec_min = int(secs.min())
    bucket  = (secs - sec_min).astype(np.int64)
    n_sec   = int(bucket.max()) + 1

    g = np.zeros((n_sec, len(RESOURCES)), dtype=np.float64)
    for k_idx, k in enumerate(RESOURCES):
        cols = RESOURCE_COL[k]
        col_sum = df[cols[0]].cast(pl.Float64).to_numpy()
        for extra in cols[1:]:
            col_sum = col_sum + df[extra].cast(pl.Float64).to_numpy()
        g[:, k_idx] = np.bincount(bucket, weights=col_sum, minlength=n_sec)

    G = g.sum(axis=1)
    sec_epoch = np.arange(n_sec, dtype=np.int64) + sec_min
    return sec_epoch, G, g


# ── Capacity per second (Mgas/s) ───────────────────────────────────────────
def capacity_60_per_second(G: np.ndarray, g: np.ndarray) -> np.ndarray:
    """capacity_60(t) = min_i T_{i,0} / Σ_k a_{i,k}·α_k(t)
                     = min_i T_{i,0} · G(t) / Σ_k a_{i,k}·g_k(t)
    Both are unitless ratios times T_{i,0} (Mgas/s) ⇒ result in Mgas/s.
    Empty seconds return +inf so they can't pull down the per-set min."""
    cap_min = np.full_like(G, np.inf, dtype=np.float64)
    for set_name, a_i in SET_WEIGHTS.items():
        T_i0 = SET_LADDERS[set_name][0][0]      # Mgas/s
        denom = np.zeros_like(G)
        for k_idx, k in enumerate(RESOURCES):
            a_ik = float(a_i.get(k, 0.0))
            if a_ik == 0.0:
                continue
            denom = denom + a_ik * g[:, k_idx]
        nonzero = denom > 0
        cap_i = np.full_like(G, np.inf)
        cap_i[nonzero] = T_i0 * G[nonzero] / denom[nonzero]
        cap_min = np.minimum(cap_min, cap_i)
    return cap_min


def capacity_51_per_second(sec_epoch: np.ndarray) -> np.ndarray:
    dia_s = int(DIA_LAUNCH_TS.timestamp())
    return np.where(sec_epoch < dia_s, T51_PRE_DIA, T51_POST_DIA).astype(np.float64)


# ── Run arbos60 price simulation, cache hourly gas-weighted means ──────────
def compute_or_load_hourly_prices(df: pl.DataFrame) -> pl.DataFrame:
    """Hourly gas-weighted mean of p_k(t) (gwei) per resource.  Runs the
    full arbos60.price_per_resource simulation once; cached to parquet."""
    if PRICES_CACHE.exists():
        print(f"  loading cached hourly prices: {PRICES_CACHE}")
        return pl.read_parquet(PRICES_CACHE)

    print("  running arbos60.price_per_resource over the full window "
          "(this is slow the first time, ~few min)...")
    g_per_block = {}
    for k in RESOURCES:
        cols = RESOURCE_COL[k]
        v = df[cols[0]].cast(pl.Float64).to_numpy()
        for extra in cols[1:]:
            v = v + df[extra].cast(pl.Float64).to_numpy()
        g_per_block[k] = v
    # Per-block UTC seconds (same convention as the historical-sim engine).
    block_t = (
        df["block_time"].cast(pl.Datetime("us")).dt.timestamp("ms").to_numpy() // 1000
    ).astype(np.int64)

    engine = a60_mod.Arbos60GasPricing()
    t0 = time.time()
    t_axis, prices_per_t, _E = engine.price_per_resource(g_per_block, block_t)
    print(f"  sim done in {time.time()-t0:.1f}s, n_t = {len(t_axis):,}")

    # Aggregate to hourly via gas-weighted mean — same convention as the
    # observed-on-chain price line in the historical-sim chart.  We weight
    # each second by its total gas G(t) so quiet seconds don't pull the
    # mean toward the p_min floor.
    hour_idx = (t_axis // 3600).astype(np.int64)
    hour_min = int(hour_idx.min())
    hour_b   = hour_idx - hour_min
    n_hr     = int(hour_b.max()) + 1

    # Recompute G per second from g_per_block (cheap; arbos60 also has it
    # but doesn't return it).  Use second-level bucketing of blocks.
    sec_min = int(t_axis[0])
    bucket  = (t_axis - sec_min).astype(np.int64)
    # For weighting, use the gas total at each t after arbos60 aggregated
    # blocks into seconds — we don't have that array, so reconstruct it
    # by aggregating g_per_block to per-second the same way arbos60 does.
    secs_block = ((block_t - sec_min)).astype(np.int64)
    n_t = len(t_axis)
    G_per_t = np.zeros(n_t, dtype=np.float64)
    for k in RESOURCES:
        G_per_t += np.bincount(secs_block, weights=g_per_block[k], minlength=n_t)

    safe_G_hr = np.bincount(hour_b, weights=G_per_t, minlength=n_hr)
    safe_G_hr = np.where(safe_G_hr > 0, safe_G_hr, 1.0)

    out = {"hour": ((np.arange(n_hr, dtype=np.int64) + hour_min) * 3600 * 1000
                    ).astype("datetime64[ms]")}
    for k in RESOURCES:
        weighted = np.bincount(hour_b, weights=prices_per_t[k] * G_per_t,
                               minlength=n_hr)
        out[f"p_{k}"] = weighted / safe_G_hr

    PRICES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df_out = pl.DataFrame(out)
    df_out.write_parquet(PRICES_CACHE)
    print(f"  cached → {PRICES_CACHE}")
    return df_out


# ── Ben's "hourly-averaged mix" recipe ─────────────────────────────────────
def aggregate_capacity_hourly_mix(
    sec_epoch: np.ndarray, G: np.ndarray, g: np.ndarray,
    cap_51: np.ndarray, threshold: float,
) -> pl.DataFrame:
    """Apply Ben's recipe per hour: aggregate g_k over the hour, derive
    one mix α_k(hr) = Σ g_k / Σ G, plug it into the inequality once to
    get a single capacity ceiling for that hour, then compare with the
    hour's realised mean G/sec.  Saturation rate uses each second's G
    against the *hour's* capacity (so a noisy second doesn't get its
    own ceiling)."""
    hour_idx = (sec_epoch // 3600).astype(np.int64)
    hour_min = int(hour_idx.min())
    hour_b   = hour_idx - hour_min
    n_hr     = int(hour_b.max()) + 1

    # Σ g_k per hour and Σ G per hour.
    sum_g = np.zeros((n_hr, len(RESOURCES)), dtype=np.float64)
    for k_idx in range(len(RESOURCES)):
        sum_g[:, k_idx] = np.bincount(hour_b, weights=g[:, k_idx],
                                       minlength=n_hr)
    sum_G = np.bincount(hour_b, weights=G, minlength=n_hr)
    safe_G = np.where(sum_G > 0, sum_G, 1.0)

    # capacity_60 per hour from the hourly-averaged mix.
    cap_min = np.full(n_hr, np.inf, dtype=np.float64)
    for set_name, a_i in SET_WEIGHTS.items():
        T_i0 = SET_LADDERS[set_name][0][0]
        denom = np.zeros(n_hr, dtype=np.float64)
        for k_idx, k in enumerate(RESOURCES):
            a_ik = float(a_i.get(k, 0.0))
            if a_ik == 0.0:
                continue
            denom = denom + a_ik * sum_g[:, k_idx]
        nz = denom > 0
        cap_i = np.full(n_hr, np.inf)
        cap_i[nz] = T_i0 * sum_G[nz] / denom[nz]
        cap_min = np.minimum(cap_min, cap_i)

    mean_G_mgas = (sum_G / 1e6) / 3600.0       # average Mgas/s for the hour
    cap_51_hr   = np.where(np.bincount(hour_b, minlength=n_hr) > 0,
                            T51_POST_DIA, T51_PRE_DIA)
    # Use the hour's ArbOS 51 ceiling: pre/post DIA based on most seconds
    # in the hour (cheap proxy).
    pre_count = np.bincount(hour_b[cap_51 == T51_PRE_DIA], minlength=n_hr)
    post_count = np.bincount(hour_b[cap_51 == T51_POST_DIA], minlength=n_hr)
    cap_51_hr = np.where(pre_count > post_count, T51_PRE_DIA, T51_POST_DIA)

    # Headroom and gain.
    headroom_60 = np.zeros(n_hr)
    finite = np.isfinite(cap_min)
    headroom_60[finite] = np.clip(
        100.0 * (cap_min[finite] - mean_G_mgas[finite]) / cap_min[finite],
        0.0, 100.0,
    )
    headroom_51 = np.clip(100.0 * (cap_51_hr - mean_G_mgas) / cap_51_hr,
                          0.0, 100.0)
    gain = np.zeros(n_hr)
    gain[finite] = 100.0 * (cap_min[finite] - cap_51_hr[finite]) / cap_51_hr[finite]

    # Saturation rate per hour: % of seconds with G ≥ θ · hour-capacity.
    G_mgas = G / 1e6
    cap_per_sec = cap_min[hour_b]
    sat = (G > 0) & np.isfinite(cap_per_sec) \
          & (G_mgas >= threshold * cap_per_sec)
    sat_60 = np.bincount(hour_b[sat], minlength=n_hr) / 3600.0

    return pl.DataFrame({
        "hour":         ((np.arange(n_hr, dtype=np.int64) + hour_min)
                          * 3600 * 1000).astype("datetime64[ms]"),
        "mean_G":       mean_G_mgas,
        "cap_51":       cap_51_hr,
        "cap_60":       np.where(finite, cap_min, np.nan),
        "headroom_51":  headroom_51,
        "headroom_60":  headroom_60,
        "gain_60":      gain,
        "sat_rate_60":  sat_60,
    })


# ── Hourly aggregation of capacity / headroom / saturation ─────────────────
def aggregate_capacity_hourly(
    sec_epoch: np.ndarray, G: np.ndarray,
    cap_51: np.ndarray, cap_60: np.ndarray,
    threshold: float,
) -> pl.DataFrame:
    hour_idx = (sec_epoch // 3600).astype(np.int64)
    hour_min = int(hour_idx.min())
    hour_b   = hour_idx - hour_min
    n_hr     = int(hour_b.max()) + 1

    G_mgas = G / 1e6
    used = G > 0
    used_n = np.bincount(hour_b[used], minlength=n_hr).astype(np.float64)
    safe   = np.where(used_n > 0, used_n, 1.0)

    finite_60 = used & np.isfinite(cap_60)
    used60_n  = np.bincount(hour_b[finite_60], minlength=n_hr).astype(np.float64)
    safe60    = np.where(used60_n > 0, used60_n, 1.0)

    # Per-second headroom %: (cap − G) / cap × 100.  Clipped to [0, 100].
    headroom_60 = np.zeros_like(G)
    headroom_60[finite_60] = np.clip(
        100.0 * (cap_60[finite_60] - G_mgas[finite_60]) / cap_60[finite_60],
        0.0, 100.0,
    )
    headroom_51 = np.clip(100.0 * (cap_51 - G_mgas) / cap_51, 0.0, 100.0)

    sat_60 = used & np.isfinite(cap_60) & (G_mgas >= threshold * cap_60)
    sat_51 = used & (G_mgas >= threshold * cap_51)

    # Per-second capacity gain (%): (cap_60 − cap_51) / cap_51 × 100
    gain = np.zeros_like(G)
    gain[finite_60] = 100.0 * (cap_60[finite_60] - cap_51[finite_60]) \
                            / cap_51[finite_60]
    # Mean per hour from a streaming sum.
    gain_mean = np.bincount(hour_b[finite_60], weights=gain[finite_60],
                            minlength=n_hr) / safe60
    # Median per hour via polars groupby (no streaming median in numpy
    # without sorting per group).
    gain_df = pl.DataFrame({
        "hour": hour_b[finite_60].astype(np.int32),
        "gain": gain[finite_60],
    }).group_by("hour").agg(pl.col("gain").median().alias("gm")).sort("hour")
    gain_median = np.zeros(n_hr)
    gain_median[gain_df["hour"].to_numpy()] = gain_df["gm"].to_numpy()

    return pl.DataFrame({
        "hour":     ((np.arange(n_hr, dtype=np.int64) + hour_min) * 3600 * 1000
                     ).astype("datetime64[ms]"),
        "mean_G":   np.bincount(hour_b[used], weights=G_mgas[used],
                                minlength=n_hr) / safe,
        "cap_51":   np.bincount(hour_b[used], weights=cap_51[used],
                                minlength=n_hr) / safe,
        "cap_60":   np.bincount(hour_b[finite_60], weights=cap_60[finite_60],
                                minlength=n_hr) / safe60,
        "headroom_51": np.bincount(hour_b[used], weights=headroom_51[used],
                                   minlength=n_hr) / safe,
        "headroom_60": np.bincount(hour_b[finite_60],
                                   weights=headroom_60[finite_60],
                                   minlength=n_hr) / safe60,
        "sat_rate_51": np.bincount(hour_b[sat_51], minlength=n_hr) / 3600.0,
        "sat_rate_60": np.bincount(hour_b[sat_60], minlength=n_hr) / 3600.0,
        "gain_mean":   gain_mean,
        "gain_median": gain_median,
    })


# ── Plotly figure + Ben's framing ──────────────────────────────────────────
def build_figure(prices_hr: pl.DataFrame, cap_hr: pl.DataFrame,
                 cap_hr_mix: pl.DataFrame, threshold: float) -> go.Figure:
    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.18, 0.13, 0.13, 0.12, 0.15, 0.15, 0.14],
        subplot_titles=(
            "ArbOS 60 simulated prices p_k(t), gwei "
            "(hourly gas-weighted mean — output of arbos60.py)",
            "Capacity gain Δ%, per-second mix",
            "Headroom %, per-second mix",
            f"Saturation rate (G ≥ {int(threshold*100)} % cap), "
            f"per-second mix",
            "<b>Capacity gain Δ%, hourly-averaged mix</b> (Ben's recipe)",
            "Headroom %, hourly-averaged mix",
            f"Saturation rate, hourly-averaged mix "
            f"(G ≥ {int(threshold*100)} % cap)",
        ),
    )

    # ── Panel 1: per-resource simulated prices ────────────────────────────
    x = prices_hr["hour"].to_list()
    for k in RESOURCES:
        fig.add_trace(go.Scatter(
            x=x, y=prices_hr[f"p_{k}"].to_numpy(),
            name=RESOURCE_LABEL[k],
            line=dict(color=RESOURCE_COLOR[k], width=1.2),
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"p_{{{k}}} = " "%{y:.4f} gwei<extra></extra>"),
        ), row=1, col=1)
    fig.add_hline(y=P_MIN_GWEI, row=1, col=1,
                  line=dict(color="#444", width=0.8, dash="dot"),
                  annotation_text=f"p_min = {P_MIN_GWEI} gwei",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#444"))
    fig.update_yaxes(title_text="gwei", type="log", row=1, col=1)

    # ── Panel 2: capacity gain (median + mean) ────────────────────────────
    xh = cap_hr["hour"].to_list()
    fig.add_trace(go.Scatter(
        x=xh, y=cap_hr["gain_median"].to_numpy(),
        name="Gain median (per-second median, hourly)",
        line=dict(color="#08519c", width=1.6),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "median gain = %{y:+.1f}%<extra></extra>"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=xh, y=cap_hr["gain_mean"].to_numpy(),
        name="Gain mean",
        line=dict(color="#6baed6", width=1.2, dash="dot"),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "mean gain = %{y:+.1f}%<extra></extra>"),
    ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color="#444", width=0.8, dash="dot"),
                  annotation_text="parity (cap_60 = cap_51)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#666"))
    fig.update_yaxes(title_text="Δ% vs ArbOS 51", row=2, col=1)

    # ── Panel 3: capacity headroom % ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xh, y=cap_hr["headroom_51"].to_numpy(),
        name="ArbOS 51 headroom",
        line=dict(color="#d62728", width=1.4, dash="dash"),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "headroom (51) = %{y:.1f}%<extra></extra>"),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=xh, y=cap_hr["headroom_60"].to_numpy(),
        name="ArbOS 60 headroom",
        line=dict(color="#1f77b4", width=1.6),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "headroom (60) = %{y:.1f}%<extra></extra>"),
    ), row=3, col=1)
    sat_y = (1.0 - threshold) * 100.0
    fig.add_hline(y=sat_y, row=3, col=1,
                  line=dict(color="#888", width=0.8, dash="dot"),
                  annotation_text=f"saturation threshold "
                                  f"({sat_y:.0f}% headroom)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#666"))
    fig.update_yaxes(title_text="% headroom", row=3, col=1,
                     range=[0, 100])

    # ── Panel 4: saturation rate ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xh, y=cap_hr["sat_rate_51"].to_numpy() * 100.0,
        name="51 saturation",
        line=dict(color="#d62728", width=1.4, dash="dash"),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "sat (51) = %{y:.2f}%<extra></extra>"),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=xh, y=cap_hr["sat_rate_60"].to_numpy() * 100.0,
        name="60 saturation",
        line=dict(color="#1f77b4", width=1.6),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "sat (60) = %{y:.2f}%<extra></extra>"),
    ), row=4, col=1)
    fig.update_yaxes(title_text="% of seconds", row=4, col=1,
                     rangemode="tozero")

    # ── Panels 5–7: Ben's hourly-averaged-mix recipe ──────────────────────
    xm = cap_hr_mix["hour"].to_list()

    # Panel 5: gain (single line — one capacity per hour from one mix).
    fig.add_trace(go.Scatter(
        x=xm, y=cap_hr_mix["gain_60"].to_numpy(),
        name="Gain (hourly mix)",
        line=dict(color="#08519c", width=1.6),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "gain (mix) = %{y:+.1f}%<extra></extra>"),
    ), row=5, col=1)
    fig.add_hline(y=0, row=5, col=1,
                  line=dict(color="#444", width=0.8, dash="dot"),
                  annotation_text="parity",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#666"))
    fig.update_yaxes(title_text="Δ% vs ArbOS 51", row=5, col=1)

    # Panel 6: headroom under hourly mix.
    fig.add_trace(go.Scatter(
        x=xm, y=cap_hr_mix["headroom_51"].to_numpy(),
        name="ArbOS 51 (mix)",
        line=dict(color="#d62728", width=1.4, dash="dash"),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "headroom 51 = %{y:.1f}%<extra></extra>"),
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=xm, y=cap_hr_mix["headroom_60"].to_numpy(),
        name="ArbOS 60 (mix)",
        line=dict(color="#1f77b4", width=1.6),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "headroom 60 = %{y:.1f}%<extra></extra>"),
    ), row=6, col=1)
    sat_y = (1.0 - threshold) * 100.0
    fig.add_hline(y=sat_y, row=6, col=1,
                  line=dict(color="#888", width=0.8, dash="dot"),
                  annotation_text=f"saturation ({sat_y:.0f}%)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#666"))
    fig.update_yaxes(title_text="% headroom", row=6, col=1, range=[0, 100])

    # Panel 7: saturation rate (hourly capacity ceiling).
    fig.add_trace(go.Scatter(
        x=xm, y=cap_hr_mix["sat_rate_60"].to_numpy() * 100.0,
        name="60 saturation (mix)",
        line=dict(color="#1f77b4", width=1.6),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "sat (mix) = %{y:.2f}%<extra></extra>"),
    ), row=7, col=1)
    fig.update_yaxes(title_text="% of seconds", row=7, col=1,
                     rangemode="tozero")

    # DIA marker on every panel.
    dia_ms = int(DIA_LAUNCH_TS.timestamp() * 1000)
    for r in (1, 2, 3, 4, 5, 6, 7):
        fig.add_vline(x=dia_ms, row=r, col=1,
                      line=dict(color="#444", width=1.0, dash="dash"))

    fig.update_layout(
        template="plotly_white",
        height=1700,
        margin=dict(l=70, r=40, t=110, b=40),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="center", x=0.5, font=dict(size=11)),
    )
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    return fig


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>ArbOS 60 capacity headroom</title>

<script>
  window.MathJax = {
    tex: { inlineMath: [['\\(', '\\)']] },
    svg: { fontCache: 'global' },
  };
</script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 30px auto; max-width: 1400px; color: #222; line-height: 1.55; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  h2 { font-size: 16px; margin: 22px 0 8px; color: #333; font-weight: 600; }

  .ben-quote {
    background: rgba(31, 119, 180, 0.05);
    border-left: 3px solid #1f77b4;
    padding: 0.8em 1.2em;
    font-size: 13.5px; margin: 16px 0 8px; color: #333;
  }
  .ben-quote .who { font-weight: 600; color: #08519c;
                     font-size: 12.5px; margin-bottom: 0.3em; }
  .ben-quote em { color: #555; }

  .methodology {
    background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
    padding: 0.8em 1.4em; margin: 6px 0 24px;
    font-size: 13px;
  }
  .methodology ol { padding-left: 1.4em; margin: 0.4em 0 0; }
  .methodology li { margin: 0.5em 0; }
  .methodology .label {
    font-weight: 600; color: #333;
  }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
         font-size: 12.5px; }
</style>
</head><body>

<h1>ArbOS 60 capacity headroom</h1>

<div class="ben-quote">
  <div class="who">Ben Berger — definition (Apr 30 sync)</div>
  Capacity = the gas-per-second throughput at which the price first
  starts rising above <code>p_min</code>. In ArbOS 51 it's a constant
  (10&nbsp;Mgas/s post-DIA). In ArbOS 60 it depends on the workload
  mix.  <em>"On the second set the price will start going up when
  big G is about 15&nbsp;Mgas/s — about a 50&nbsp;% capacity gain on
  average."</em>
</div>

<h2>Methodology</h2>
<div class="methodology">
  <ol>
    <li>
      <span class="label">Per-second aggregation.</span>
      Bucket every block into 1&nbsp;s windows (≈ 4 blocks each at
      Arbitrum's 0.25&nbsp;s block time).  For each second
      \(t\) define per-resource gas
      \(g_k(t)\), total gas
      \(G(t) = \sum_{{k}} g_k(t)\),
      and the workload mix
      \(\alpha_k(t) = g_k(t)\,/\,G(t)\).
    </li>
    <li>
      <span class="label">ArbOS&nbsp;60 capacity (mix-dependent).</span>
      The spec says, for every set \(i\) and constraint \(j\):
      \[\sum_{k} a_{i,k}\,g_k(t) \;\le\; T_{i,j}\]
      Substituting \(g_k = \alpha_k G\) and taking
      \(j = 0\) (the smallest \(T\) per set, which binds for sustained
      throughput):
      \[\text{capacity}_{60}(t) \;=\;
        \min_{i} \frac{T_{i,0}}{\sum_{k} a_{i,k}\,\alpha_k(t)}\]
    </li>
    <li>
      <span class="label">ArbOS&nbsp;51 capacity (constant per regime).</span>
      ArbOS&nbsp;51 has the single inequality
      \(G(t) \le T_{51}\), so capacity is just \(T_{51}\):
      \[\text{capacity}_{51} = \begin{cases}
          7  \text{ Mgas/s} & \text{pre-DIA} \\
          10 \text{ Mgas/s} & \text{post-DIA (Dia ladder, } j = 0)
        \end{cases}\]
    </li>
    <li>
      <span class="label">Headroom %.</span>
      How much more gas/s the chain could absorb before \(p_k(t)\)
      moves above \(p_{\min}\):
      \[\text{headroom}(t) \;=\;
        \frac{\text{capacity}(t) - G(t)}{\text{capacity}(t)}
        \times 100 \%\]
      Plotted hourly as the mean of all active seconds.
    </li>
    <li>
      <span class="label">Saturation threshold.</span>
      A second is "near saturation" when realised \(G(t)\) reaches
      \(\theta\) of the capacity ceiling
      (default \(\theta = {{THRESHOLD}}\)):
      \[G(t) \;\ge\; \theta \cdot \text{capacity}(t)\]
      Panel 3 shows, per hour, the share of seconds that crossed this
      line.
    </li>
    <li>
      <span class="label">Top panel — simulated prices.</span>
      Per-resource prices come straight from
      <code>arbos60.Arbos60GasPricing.price_per_resource</code>
      (same engine as the historical-sim chart):
      \[e_k(t) = \max_i\{a_{i,k}\,E_i(t)\}, \qquad
        p_k(t) = p_{\min} \cdot \exp\bigl(e_k(t)\bigr)\]
      with set exponents
      \(E_i(t) = \sum_j B_{i,j}(t)\,/\,(A_{i,j}\,T_{i,j})\) and
      backlog tick
      \(B_{i,j}(t{+}1) = \max(0,\; B_{i,j}(t)
        + \sum_k a_{i,k} g_k(t) - T_{i,j}\,\Delta t)\).
      Plotted as the gas-weighted hourly mean.
    </li>
  </ol>
</div>

{FIG}

</body></html>
"""


# ── Driver ─────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_per_block_with_time()

    print("Aggregating gas per second...")
    sec_epoch, G_total, g = aggregate_per_second(df)
    print(f"  {len(sec_epoch):,} seconds")

    print("Computing capacity_60 per second...")
    cap_60 = capacity_60_per_second(G_total, g)
    cap_51 = capacity_51_per_second(sec_epoch)
    used = G_total > 0
    print(f"  median capacity_60 (active seconds) = "
          f"{np.median(cap_60[used & np.isfinite(cap_60)]):.2f} Mgas/s")

    print("Computing or loading hourly arbos60 prices...")
    prices_hr = compute_or_load_hourly_prices(df)

    print("Aggregating capacity to hourly (per-second mix)...")
    cap_hr = aggregate_capacity_hourly(sec_epoch, G_total, cap_51, cap_60,
                                        THRESHOLD)

    print("Aggregating capacity to hourly (hourly-averaged mix, Ben's recipe)...")
    cap_hr_mix = aggregate_capacity_hourly_mix(sec_epoch, G_total, g,
                                                cap_51, THRESHOLD)
    finite_gain = np.isfinite(cap_hr_mix["gain_60"].to_numpy()) \
                  & (cap_hr_mix["mean_G"].to_numpy() > 0)
    if finite_gain.any():
        gain_v = cap_hr_mix["gain_60"].to_numpy()[finite_gain]
        print(f"  hourly-mix gain: median = {np.median(gain_v):+.1f}%   "
              f"mean = {np.mean(gain_v):+.1f}%")

    print("Rendering...")
    fig = build_figure(prices_hr, cap_hr, cap_hr_mix, THRESHOLD)
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False,
                           config={"displaylogo": False, "responsive": True})
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    # Substitute via str.replace so the {} in LaTeX equations don't
    # collide with `.format()` placeholder syntax.
    out = (
        PAGE_TEMPLATE
        .replace("{{THRESHOLD}}", f"{THRESHOLD:.2f}")
        .replace("{FIG}",         fig_html)
    )
    OUT_HTML.write_text(out)
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
