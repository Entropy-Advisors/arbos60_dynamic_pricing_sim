"""
Revenue comparison dashboard — ArbOS 51 vs ArbOS 60, **STATIC DEMAND**.

Demand is held at historical gas-usage levels — i.e. the same per-tx
gas vectors are re-priced under each regime, with no adjustment for how
demand would actually respond to price changes.  See
docs/demand_elasticity_plan.md for the elastic-demand version.

Reuses the heavy compute from historical_sim.py:
  load_per_block → merge_blocks_with_resources → price_arbos51 / price_arbos60
  → aggregate_per_tx_hourly  (≈1 min of streaming over the per-tx parquets,
                              once per ArbOS 60 set)

Output: figures/revenue_no_elasticity.html
Sections:
  1. Headline hourly + daily ETH revenue time series, ArbOS 51 vs 60.
  2. Cumulative ETH revenue per period (Full / 90D / 30D / 7D), both
     lines starting at 0 ETH on the first day of the slice.
  3. Spike-window zooms (Jan 29-Feb 7; March windows skipped if outside
     the data range).
  4. Distribution boxplots (hourly + daily) for 51 vs 60.
  5. Boxplots — ArbOS 60 p_min sweep (0.02 / 0.03 / 0.04 / 0.05 gwei)
     vs ArbOS 51.
  6. Boxplots — ArbOS 60 set 1 vs set 2 (vs ArbOS 51).
  7. Summary stats table (Last 7D / 30D / 90D / Full window) with totals,
     mean and median for both hourly and daily series.

Run:
    python scripts/revenue_no_elasticity.py
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Reuse the already-built helpers in historical_sim.py.
import historical_sim as hs

OUT_HTML = _HERE.parent / "figures" / "revenue_no_elasticity.html"
PER_BLOCK_PARQUET = (
    _HERE.parent / "data" / "onchain_blocks_transactions" / "per_block.parquet"
)
CACHE_DIR = _HERE.parent / "data" / "revenue_comparison_cache"
HOURLY_CACHE = CACHE_DIR / "hourly.parquet"

# Spike windows to zoom on, in UTC.
SPIKE_WINDOWS = [
    ("Jan 29 – Feb 7",  datetime(2026, 1, 29),  datetime(2026, 2,  7)),
    ("Mar 5 ± 2 days",  datetime(2026, 3,  3),  datetime(2026, 3,  7)),
    ("Mar 23 ± 2 days", datetime(2026, 3, 21),  datetime(2026, 3, 25)),
]

GWEI_TO_ETH = 1e-9

# ArbOS 60 p_min sweep (gwei).  Default sim ran at p_min = 0.02; the price
# formula p_k(t) = p_min · taylor4_exp(e_k(t)) is LINEAR in p_min (the
# exponent depends only on inflow vs target, not on p_min), so we can derive
# every other p_min case by scaling the eth_60 series by p_min/0.02.  No
# re-streaming required.
PMIN_BASE   = 0.02
PMIN_SWEEP  = [0.02, 0.03, 0.04, 0.05]
# All ArbOS 60 variants in shades of blue, sequential (light → dark) with p_min.
PMIN_COLORS = {
    0.02: "#6baed6",    # light blue
    0.03: "#3182bd",    # medium blue
    0.04: "#08519c",    # dark blue
    0.05: "#08306b",    # darkest blue
}

# Color scheme — ArbOS 51 red, every ArbOS 60 variant a shade of blue.
COL51   = "#d62728"   # ArbOS 51 — red
COL60   = "#1f77b4"   # ArbOS 60 set 1 (default p_min) — base blue
COL60V2 = "#17becf"   # ArbOS 60 set 2 — teal/cyan blue (still in the blue family)
COLΔ    = "#888888"   # delta panels — grey

# Cumulative-period definitions.  Each is (label, days-or-None).  None = full
# window.  All cumulative-revenue charts re-zero at the first point of the
# slice so 51 and 60 start at 0 ETH together.
CUM_PERIODS = [
    ("Full window",  None),
    ("Last 90 days", 90),
    ("Last 30 days", 30),
    ("Last 7 days",   7),
]


# ── Hourly + daily revenue series ───────────────────────────────────────────
def _price_arbos60_set(blocks: pl.DataFrame, version: int) -> dict[str, np.ndarray]:
    """Per-block ArbOS 60 prices for a given preset (set 1 or set 2)."""
    from arbos60 import Arbos60GasPricing
    engine = Arbos60GasPricing(version=version)
    g_per_block = engine.per_block_resource_gas(blocks)
    block_t = hs._block_seconds(blocks)
    t_axis, prices_per_t, _ = engine.price_per_resource(g_per_block, block_t)
    t_idx = (block_t - t_axis[0]).astype(np.int64)
    return {k: prices_per_t[k][t_idx] for k in engine.GAS_RESOURCES}


def compute_hourly_revenue(use_cache: bool = True) -> pl.DataFrame:
    """Returns a polars DataFrame with columns
        hour | eth_real | eth_51 | eth_60 | eth_60_v2
    where each `eth_*` column is ETH revenue per hour under that pricing
    regime (eth_60 = ArbOS 60 set 1, eth_60_v2 = ArbOS 60 set 2; both
    at p_min = 0.02 gwei).  Cached at HOURLY_CACHE — pass `use_cache=False`
    to force a fresh streaming pass.  Cache invalidates automatically if
    `eth_60_v2` is missing.
    """
    if use_cache and HOURLY_CACHE.exists():
        cached = pl.read_parquet(HOURLY_CACHE)
        if "eth_60_v2" in cached.columns:
            print(f"loading hourly revenue cache: {HOURLY_CACHE}")
            return cached
        print(f"cache {HOURLY_CACHE} missing eth_60_v2 — re-streaming")

    print("loading blocks + resources, simulating per-block prices (sets 1+2)...")
    blocks = hs.load_per_block(str(PER_BLOCK_PARQUET))
    blocks = hs.merge_blocks_with_resources(
        blocks, hs.build_per_block_resources(),
    )
    p51_pb       = hs.price_arbos51_per_block(blocks)
    p60_v1       = _price_arbos60_set(blocks, version=1)
    p60_v2       = _price_arbos60_set(blocks, version=2)

    print("streaming per-tx parquet → hourly fees, set 1 ...")
    hourly_v1 = hs.aggregate_per_tx_hourly(blocks, p51_pb, p60_v1)
    print("streaming per-tx parquet → hourly fees, set 2 ...")
    hourly_v2 = hs.aggregate_per_tx_hourly(blocks, p51_pb, p60_v2)

    base = hourly_v1.select([
        "hour",
        (pl.col("_real_fee") * GWEI_TO_ETH).alias("eth_real"),
        (pl.col("_p51_fee")  * GWEI_TO_ETH).alias("eth_51"),
        (pl.col("_p60_fee")  * GWEI_TO_ETH).alias("eth_60"),
    ])
    v2 = hourly_v2.select([
        "hour",
        (pl.col("_p60_fee")  * GWEI_TO_ETH).alias("eth_60_v2"),
    ])
    hourly = base.join(v2, on="hour", how="left")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(HOURLY_CACHE)
    print(f"  cached → {HOURLY_CACHE}")
    return hourly


def add_pmin_sweep(hourly: pl.DataFrame) -> pl.DataFrame:
    """Add eth_60_pmin_<X> columns for every p_min in PMIN_SWEEP.
    p_k(t) = p_min · taylor4_exp(e_k(t)); exponent is independent of p_min,
    so scaling eth_60 by p_min / PMIN_BASE is exact."""
    cols = [pl.col(c) for c in hourly.columns]
    for pmin in PMIN_SWEEP:
        scale = pmin / PMIN_BASE
        col = (pl.col("eth_60") * scale).alias(f"eth_60_pmin_{pmin:g}")
        cols.append(col)
    return hourly.select(cols)


def hourly_to_daily(hourly: pl.DataFrame) -> pl.DataFrame:
    """Sum every ETH column per UTC day."""
    eth_cols = [c for c in hourly.columns if c.startswith("eth_")]
    return (
        hourly
        .with_columns(pl.col("hour").dt.truncate("1d").alias("day"))
        .group_by("day")
        .agg([pl.col(c).sum() for c in eth_cols])
        .sort("day")
    )


# ── Stats table ─────────────────────────────────────────────────────────────
def _window_slice(df: pl.DataFrame, time_col: str, days: int | None) -> pl.DataFrame:
    if days is None:
        return df
    end = df[time_col].max()
    cutoff = end - timedelta(days=days)
    return df.filter(pl.col(time_col) >= cutoff)


def stats_row(
    hourly: pl.DataFrame, daily: pl.DataFrame, days: int | None, label: str,
) -> dict:
    h = _window_slice(hourly, "hour", days)
    d = _window_slice(daily,  "day",  days)
    e51 = float(h["eth_51"].sum())
    e60 = float(h["eth_60"].sum())
    return {
        "window":             label,
        "n_hours":            h.height,
        "total_eth_51":       e51,
        "total_eth_60":       e60,
        "delta_eth":          e60 - e51,
        "delta_pct":          (e60 - e51) / e51 * 100.0 if e51 else float("nan"),
        "mean_hourly_51":     float(h["eth_51"].mean()) if h.height else 0.0,
        "mean_hourly_60":     float(h["eth_60"].mean()) if h.height else 0.0,
        "median_hourly_51":   float(h["eth_51"].median()) if h.height else 0.0,
        "median_hourly_60":   float(h["eth_60"].median()) if h.height else 0.0,
        "mean_daily_51":      float(d["eth_51"].mean()) if d.height else 0.0,
        "mean_daily_60":      float(d["eth_60"].mean()) if d.height else 0.0,
        "median_daily_51":    float(d["eth_51"].median()) if d.height else 0.0,
        "median_daily_60":    float(d["eth_60"].median()) if d.height else 0.0,
    }


def build_stats_table(hourly: pl.DataFrame, daily: pl.DataFrame) -> str:
    rows = [
        stats_row(hourly, daily, 7,    "Last 7D"),
        stats_row(hourly, daily, 30,   "Last 30D"),
        stats_row(hourly, daily, 90,   "Last 90D"),
        stats_row(hourly, daily, None, f"Full window ({daily.height} days)"),
    ]

    def _eth(x: float) -> str:
        if abs(x) >= 1000:  return f"{x:,.1f}"
        if abs(x) >= 10:    return f"{x:,.2f}"
        return f"{x:.4f}"

    th_style = (
        "padding:6px 10px; border-bottom:2px solid #333; "
        "background:#f4f4f4; text-align:right; font-size:13px;"
    )
    td_style = (
        "padding:5px 10px; border-bottom:1px solid #ddd; "
        "text-align:right; font-size:13px; font-family:monospace;"
    )
    label_style = td_style + " text-align:left; font-family:inherit;"

    head = (
        f"<tr><th style='{th_style}; text-align:left'>Window</th>"
        f"<th style='{th_style}'>Total 51 (ETH)</th>"
        f"<th style='{th_style}'>Total 60 (ETH)</th>"
        f"<th style='{th_style}'>Δ ETH</th>"
        f"<th style='{th_style}'>Δ %</th>"
        f"<th style='{th_style}'>Hourly mean 51 / 60</th>"
        f"<th style='{th_style}'>Hourly median 51 / 60</th>"
        f"<th style='{th_style}'>Daily mean 51 / 60</th>"
        f"<th style='{th_style}'>Daily median 51 / 60</th></tr>"
    )
    body = []
    for r in rows:
        body.append(
            f"<tr>"
            f"<td style='{label_style}'>{r['window']}</td>"
            f"<td style='{td_style}'>{_eth(r['total_eth_51'])}</td>"
            f"<td style='{td_style}'>{_eth(r['total_eth_60'])}</td>"
            f"<td style='{td_style}'>{_eth(r['delta_eth'])}</td>"
            f"<td style='{td_style}'>{r['delta_pct']:+.1f}%</td>"
            f"<td style='{td_style}'>{_eth(r['mean_hourly_51'])} / {_eth(r['mean_hourly_60'])}</td>"
            f"<td style='{td_style}'>{_eth(r['median_hourly_51'])} / {_eth(r['median_hourly_60'])}</td>"
            f"<td style='{td_style}'>{_eth(r['mean_daily_51'])} / {_eth(r['mean_daily_60'])}</td>"
            f"<td style='{td_style}'>{_eth(r['median_daily_51'])} / {_eth(r['median_daily_60'])}</td>"
            f"</tr>"
        )
    return (
        "<table style='border-collapse:collapse; margin:20px auto; "
        "max-width:1400px; box-shadow:0 1px 3px rgba(0,0,0,0.08);'>"
        + head + "".join(body) + "</table>"
    )


# ── Figure construction ─────────────────────────────────────────────────────
def _add_revenue_pair(fig, x, y51, y60, *, row_top, row_bot, col=1, show_legend=False):
    """Add 51/60 lines on the top row + filled-area delta on the bottom row."""
    fig.add_trace(
        go.Scatter(x=x, y=y51, mode="lines", name="ArbOS 51",
                   line=dict(color=COL51, width=1.4),
                   showlegend=show_legend),
        row=row_top, col=col,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y60, mode="lines", name="ArbOS 60",
                   line=dict(color=COL60, width=1.4),
                   showlegend=show_legend),
        row=row_top, col=col,
    )
    delta = np.asarray(y60) - np.asarray(y51)
    # Filled area chart for the delta — gives an immediate visual sense of
    # cumulative under/over-collection.  Positive = ArbOS 60 above 51.
    fig.add_trace(
        go.Scatter(x=x, y=delta, mode="lines", name="Δ (60 − 51)",
                   line=dict(color=COLΔ, width=0.8),
                   fill="tozeroy",
                   fillcolor="rgba(120, 120, 120, 0.35)",
                   showlegend=show_legend),
        row=row_bot, col=col,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)",
                  row=row_bot, col=col)


def build_figure(hourly: pl.DataFrame, daily: pl.DataFrame) -> go.Figure:
    """Sections, top to bottom:
       1. Headline hourly + daily revenue (4 rows: line + delta × 2)
       2. Cumulative ETH revenue per period — both lines start at 0 ETH on
          the first day of the slice
       3. Spike-window zooms (line + delta × n_spikes)
       4. Distribution boxplots — 51 vs 60 (default p_min)
       5. Distribution boxplots — 51 vs 60 across the p_min sweep
       6. Distribution boxplots — 51 vs 60 set 1 vs 60 set 2
    """
    # Drop spike windows entirely outside the data range — otherwise we'd
    # create empty rows.  Warn so the user knows which were skipped.
    data_min = hourly["hour"].min()
    data_max = hourly["hour"].max()
    spike_windows = []
    for label, t0, t1 in SPIKE_WINDOWS:
        if t1 < data_min or t0 > data_max:
            print(f"  [warn] dropping spike window '{label}' "
                  f"({t0.date()}..{t1.date()}) — outside data range "
                  f"({data_min.date()}..{data_max.date()})")
            continue
        spike_windows.append((label, t0, t1))
    n_spikes = len(spike_windows)
    # Cumulative now lives in its own figure (build_cumulative_grid).
    # The main figure handles the time-series + boxplots only.
    n_rows = (
        2                              # headline hourly: line + delta
        + 2                            # headline daily:  line + delta
        + 2 * n_spikes                 # spike zooms
        + 1                            # boxplot row 1: 51 vs 60 default
        + 1                            # boxplot row 2: 51 vs 60 p_min sweep
        + 1                            # boxplot row 3: 51 vs 60 set 1 vs set 2
    )
    titles = [
        "Hourly ETH revenue — ArbOS 51 vs 60 (full window)",
        "Δ hourly ETH revenue (60 − 51)",
        "Daily ETH revenue — ArbOS 51 vs 60 (full window)",
        "Δ daily ETH revenue (60 − 51)",
    ]
    for label, _, _ in spike_windows:
        titles.append(f"Hourly ETH revenue — {label}")
        titles.append(f"Δ hourly ETH revenue — {label}")
    titles += [
        "Hourly ETH revenue distribution (p_min = 0.02 gwei)",
        "Daily ETH revenue distribution (p_min = 0.02 gwei)",
        "Hourly ETH revenue — ArbOS 60 p_min sweep (vs 51)",
        "Daily ETH revenue — ArbOS 60 p_min sweep (vs 51)",
        "Hourly ETH revenue — ArbOS 60 set 1 vs set 2 (vs 51)",
        "Daily ETH revenue — ArbOS 60 set 1 vs set 2 (vs 51)",
    ]

    row_heights = (
        [0.10, 0.05]                       # hourly line + delta
        + [0.10, 0.05]                     # daily  line + delta
        + [0.08, 0.04] * n_spikes          # spike zooms
        + [0.14]                           # boxplot row 1
        + [0.16]                           # boxplot row 2 (sweep)
        + [0.14]                           # boxplot row 3 (set1 vs set2)
    )

    # Three boxplot rows at the end span 2 cols; all line rows above span 2.
    fig = make_subplots(
        rows=n_rows, cols=2,
        row_heights=row_heights,
        subplot_titles=titles,
        vertical_spacing=0.018,
        horizontal_spacing=0.08,
        specs=(
            [[{"colspan": 2}, None]] * (n_rows - 3)
            + [[{}, {}], [{}, {}], [{}, {}]]
        ),
    )

    # ── Section 1: headline hourly + daily ──────────────────────────────────
    h_x   = hourly["hour"].to_list()
    h_51  = hourly["eth_51"].to_numpy()
    h_60  = hourly["eth_60"].to_numpy()
    h_60v2 = hourly["eth_60_v2"].to_numpy()
    _add_revenue_pair(fig, h_x, h_51, h_60, row_top=1, row_bot=2, show_legend=True)
    fig.update_yaxes(title_text="ETH/hour", row=1, col=1)
    fig.update_yaxes(title_text="Δ ETH/hour", row=2, col=1)

    d_x   = daily["day"].to_list()
    d_51  = daily["eth_51"].to_numpy()
    d_60  = daily["eth_60"].to_numpy()
    d_60v2 = daily["eth_60_v2"].to_numpy()
    _add_revenue_pair(fig, d_x, d_51, d_60, row_top=3, row_bot=4)
    fig.update_yaxes(title_text="ETH/day", row=3, col=1)
    fig.update_yaxes(title_text="Δ ETH/day", row=4, col=1)

    # ── Section 2: spike-window zooms (hourly resolution) ──────────────────
    # (Cumulative section moved to its own figure — see build_cumulative_grid.)
    spike_base = 5
    for i, (label, t0, t1) in enumerate(spike_windows):
        sub = hourly.filter(
            (pl.col("hour") >= t0) & (pl.col("hour") <= t1)
        )
        if sub.is_empty():
            continue
        x   = sub["hour"].to_list()
        y51 = sub["eth_51"].to_numpy()
        y60 = sub["eth_60"].to_numpy()
        rt = spike_base + 2 * i
        rb = rt + 1
        _add_revenue_pair(fig, x, y51, y60, row_top=rt, row_bot=rb)
        fig.update_yaxes(title_text="ETH/hour", row=rt, col=1)
        fig.update_yaxes(title_text="Δ ETH/hour", row=rb, col=1)

    # ── Section 4: default-p_min boxplots (hourly + daily) ──────────────────
    box_row1 = n_rows - 2
    fig.add_trace(go.Box(y=h_51, name="ArbOS 51", marker_color=COL51,
                         boxpoints="outliers", showlegend=False),
                  row=box_row1, col=1)
    fig.add_trace(go.Box(y=h_60, name="ArbOS 60", marker_color=COL60,
                         boxpoints="outliers", showlegend=False),
                  row=box_row1, col=1)
    fig.update_yaxes(title_text="ETH/hour", row=box_row1, col=1, type="log")
    fig.add_trace(go.Box(y=d_51, name="ArbOS 51", marker_color=COL51,
                         boxpoints="outliers", showlegend=False),
                  row=box_row1, col=2)
    fig.add_trace(go.Box(y=d_60, name="ArbOS 60", marker_color=COL60,
                         boxpoints="outliers", showlegend=False),
                  row=box_row1, col=2)
    fig.update_yaxes(title_text="ETH/day", row=box_row1, col=2, type="log")

    # ── Section 5: p_min sweep boxplots (51 vs 60 at 0.02/0.03/0.04/0.05) ───
    box_row2 = n_rows - 1
    fig.add_trace(go.Box(y=h_51, name="ArbOS 51", marker_color=COL51,
                         boxpoints="outliers", showlegend=False),
                  row=box_row2, col=1)
    for pmin in PMIN_SWEEP:
        fig.add_trace(go.Box(y=hourly[f"eth_60_pmin_{pmin:g}"].to_numpy(),
                             name=f"60 · p_min={pmin:.2f}",
                             marker_color=PMIN_COLORS[pmin],
                             boxpoints="outliers", showlegend=False),
                      row=box_row2, col=1)
    fig.update_yaxes(title_text="ETH/hour", row=box_row2, col=1, type="log")
    fig.add_trace(go.Box(y=d_51, name="ArbOS 51", marker_color=COL51,
                         boxpoints="outliers", showlegend=False),
                  row=box_row2, col=2)
    for pmin in PMIN_SWEEP:
        fig.add_trace(go.Box(y=daily[f"eth_60_pmin_{pmin:g}"].to_numpy(),
                             name=f"60 · p_min={pmin:.2f}",
                             marker_color=PMIN_COLORS[pmin],
                             boxpoints="outliers", showlegend=False),
                      row=box_row2, col=2)
    fig.update_yaxes(title_text="ETH/day", row=box_row2, col=2, type="log")

    # ── Section 6: set 1 vs set 2 boxplots (51 + 60 set 1 + 60 set 2) ───────
    box_row3 = n_rows
    for y, name, color in (
        (h_51,  "ArbOS 51",          COL51),
        (h_60,  "ArbOS 60 — set 1",  COL60),
        (h_60v2,"ArbOS 60 — set 2",  COL60V2),
    ):
        fig.add_trace(go.Box(y=y, name=name, marker_color=color,
                             boxpoints="outliers", showlegend=False),
                      row=box_row3, col=1)
    fig.update_yaxes(title_text="ETH/hour", row=box_row3, col=1, type="log")
    for y, name, color in (
        (d_51,  "ArbOS 51",          COL51),
        (d_60,  "ArbOS 60 — set 1",  COL60),
        (d_60v2,"ArbOS 60 — set 2",  COL60V2),
    ):
        fig.add_trace(go.Box(y=y, name=name, marker_color=color,
                             boxpoints="outliers", showlegend=False),
                      row=box_row3, col=2)
    fig.update_yaxes(title_text="ETH/day", row=box_row3, col=2, type="log")

    # ── Layout ──────────────────────────────────────────────────────────────
    total_height = 220 * (n_rows + 2)
    fig.update_layout(
        title=dict(
            text=(
                "<b>Revenue comparison — ArbOS 51 vs ArbOS 60</b><br>"
                "<sub><b>Static-demand simulation</b> · gas usage held at "
                "historical levels — <b>demand elasticity is NOT modelled</b> "
                "(see docs/demand_elasticity_plan.md for the elastic version).<br>"
                f"{daily.height} days · {hourly.height:,} hourly buckets</sub>"
            ),
            x=0.0, xanchor="left", y=0.995, yanchor="top",
            font=dict(size=18),
        ),
        template="plotly_white",
        height=total_height,
        margin=dict(l=80, r=80, t=140, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.005,
                    xanchor="right", x=1.0),
        font=dict(size=12, color="#222"),
        hovermode="x unified",
    )
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    return fig


# ── Cumulative-revenue grid (separate figure: rows × p_min columns) ─────────
def build_cumulative_grid(daily: pl.DataFrame) -> go.Figure:
    """Cumulative ETH revenue, one row per period × one column per p_min.
    Each panel shows two lines starting at 0 ETH on the first day of the slice:
      - ArbOS 51 (red, baseline — does NOT depend on p_min)
      - ArbOS 60 set 1 at the column's p_min (blue)
    p_min only affects ArbOS 60 (the chain's price = p_min · exp(...)),
    so ArbOS 51 is identical across columns.  We still draw it in every
    cell so the eye can read each cell standalone.
    """
    n_rows = len(CUM_PERIODS)
    n_cols = len(PMIN_SWEEP)

    titles = []
    for label, _ in CUM_PERIODS:
        for pmin in PMIN_SWEEP:
            titles.append(
                f"{label} · ArbOS 60 p_min = {pmin:.2f} gwei"
            )

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
        shared_xaxes=False, shared_yaxes=False,
    )

    legend_done = False
    for r, (label, days) in enumerate(CUM_PERIODS, start=1):
        sub = daily if days is None else daily.tail(days)
        if sub.is_empty():
            continue
        x = sub["day"].to_list()
        cum51 = np.cumsum(sub["eth_51"].to_numpy())
        for c, pmin in enumerate(PMIN_SWEEP, start=1):
            col_60 = f"eth_60_pmin_{pmin:g}"
            cum60  = np.cumsum(sub[col_60].to_numpy())

            fig.add_trace(go.Scatter(
                x=x, y=cum51, mode="lines",
                name="ArbOS 51",
                line=dict(color=COL51, width=1.6),
                showlegend=not legend_done,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=x, y=cum60, mode="lines",
                name=f"ArbOS 60 (p_min={pmin:.2f})",
                line=dict(color=PMIN_COLORS[pmin], width=1.6),
                showlegend=not legend_done,
            ), row=r, col=c)
            legend_done = True
            fig.update_yaxes(title_text="cum ETH", row=r, col=c)
            fig.update_xaxes(title_text="day" if r == n_rows else None,
                             row=r, col=c)

    fig.update_layout(
        title=dict(
            text=("<b>Cumulative ETH revenue per period × p_min</b><br>"
                  "<sub>both lines start at 0 ETH on the first day of the slice; "
                  "ArbOS 51 (red) does not depend on p_min</sub>"),
            x=0.0, xanchor="left", y=0.995, yanchor="top",
            font=dict(size=16),
        ),
        template="plotly_white",
        height=240 * n_rows + 120,
        margin=dict(l=70, r=80, t=120, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.005,
                    xanchor="right", x=1.0),
        font=dict(size=11, color="#222"),
        hovermode="x unified",
    )
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    return fig


# ── Driver ──────────────────────────────────────────────────────────────────
def build_pmin_sweep_table(hourly: pl.DataFrame) -> str:
    """Total ETH for the full window — ArbOS 51 baseline + ArbOS 60 set 1
    p_min sweep + ArbOS 60 set 2 (default p_min)."""
    e51 = float(hourly["eth_51"].sum())
    e60v2 = float(hourly["eth_60_v2"].sum())
    th = ("padding:6px 10px; border-bottom:2px solid #333; "
          "background:#f4f4f4; text-align:right; font-size:13px;")
    td = ("padding:5px 10px; border-bottom:1px solid #ddd; "
          "text-align:right; font-size:13px; font-family:monospace;")
    label = td + " text-align:left; font-family:inherit;"

    rows = [
        f"<tr><td style='{label}'>ArbOS 51 (baseline)</td>"
        f"<td style='{td}'>{e51:,.2f}</td>"
        f"<td style='{td}'>—</td><td style='{td}'>—</td></tr>"
    ]
    for pmin in PMIN_SWEEP:
        e = float(hourly[f"eth_60_pmin_{pmin:g}"].sum())
        rows.append(
            f"<tr><td style='{label}'>ArbOS 60 set 1 · p_min = {pmin:.2f} gwei</td>"
            f"<td style='{td}'>{e:,.2f}</td>"
            f"<td style='{td}'>{e - e51:+,.2f}</td>"
            f"<td style='{td}'>{(e - e51)/e51*100:+.1f}%</td></tr>"
        )
    rows.append(
        f"<tr><td style='{label}'>ArbOS 60 <b>set 2</b> · p_min = 0.02 gwei</td>"
        f"<td style='{td}'>{e60v2:,.2f}</td>"
        f"<td style='{td}'>{e60v2 - e51:+,.2f}</td>"
        f"<td style='{td}'>{(e60v2 - e51)/e51*100:+.1f}%</td></tr>"
    )
    head = (
        f"<tr><th style='{th}; text-align:left'>Regime</th>"
        f"<th style='{th}'>Total ETH (full window)</th>"
        f"<th style='{th}'>Δ vs 51</th>"
        f"<th style='{th}'>Δ %</th></tr>"
    )
    return (
        "<table style='border-collapse:collapse; margin:20px auto; "
        "max-width:900px; box-shadow:0 1px 3px rgba(0,0,0,0.08);'>"
        + head + "".join(rows) + "</table>"
    )


def main():
    hourly = compute_hourly_revenue()
    hourly = add_pmin_sweep(hourly)
    daily  = hourly_to_daily(hourly)

    print(f"\nhourly: {hourly.height:,} rows  |  daily: {daily.height:,} rows")
    e51   = float(hourly["eth_51"].sum())
    e60v2 = float(hourly["eth_60_v2"].sum())
    print(f"\nfull-window total ETH:")
    print(f"  ArbOS 51:                       {e51:>12,.2f}")
    for pmin in PMIN_SWEEP:
        e = float(hourly[f"eth_60_pmin_{pmin:g}"].sum())
        print(f"  ArbOS 60 set 1 (p_min={pmin:.2f}): {e:>12,.2f}   "
              f"Δ {e - e51:+,.2f}  ({(e-e51)/e51*100:+.1f}%)")
    print(f"  ArbOS 60 set 2 (p_min=0.02): {e60v2:>12,.2f}   "
          f"Δ {e60v2 - e51:+,.2f}  ({(e60v2-e51)/e51*100:+.1f}%)")

    fig         = build_figure(hourly, daily)
    cum_fig     = build_cumulative_grid(daily)
    table_html  = build_stats_table(hourly, daily)
    sweep_html  = build_pmin_sweep_table(hourly)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig_html = fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    cum_html = cum_fig.to_html(
        include_plotlyjs=False, full_html=False,    # no second plotly bundle
        config={"displaylogo": False, "responsive": True},
    )
    inject = (
        "<h2 style='font-family:sans-serif; max-width:1400px; "
        "margin:30px auto 0; padding:0 20px'>Cumulative ETH revenue "
        "by period × p_min</h2>"
        + cum_html
        + "<h2 style='font-family:sans-serif; max-width:1400px; "
        "margin:30px auto 0; padding:0 20px'>Summary stats</h2>"
        + table_html
        + "<h2 style='font-family:sans-serif; max-width:900px; "
        "margin:30px auto 0; padding:0 20px'>ArbOS 60 p_min sweep "
        "— total revenue (full window)</h2>"
        + sweep_html
    )
    fig_html = fig_html.replace("</body>", inject + "</body>")
    OUT_HTML.write_text(fig_html)
    print(f"\nsaved {OUT_HTML}")


if __name__ == "__main__":
    main()
