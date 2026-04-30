"""
Build the reveal.js presentation deck for the ArbOS 60 historical
simulation.  Each slide embeds an interactive Plotly figure (plotly.js
loaded once via CDN, charts injected as HTML fragments).

Initial slide set:
  1. Title + dataset scope (numeric stat block — no chart).
  2. Hourly priced gas usage by resource kind (stacked, absolute Mgas/hr).
  3. Resource composition over time (stacked, % share of priced gas).

Run:
    python presentation/build.py

Output: presentation/index.html
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import polars as pl
import plotly.graph_objects as go

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent

# Reuse historical_sim's data-loading helpers — single source of truth for
# what counts as "the dataset" for the simulation.
sys.path.insert(0, str(_ROOT / "scripts"))
import historical_sim as hs                     # noqa: E402

OUT_HTML = _HERE / "index.html"

# Cached uniform reservoir sample of per-tx rows (no min-gas filter), used
# only for descriptive statistics on slide 5.  Cached so the deck rebuilds
# in seconds instead of re-streaming 594 M rows every time.
TX_SAMPLE_PARQUET = _ROOT / "data" / "presentation_tx_stats_sample.parquet"
TX_SAMPLE_N       = 500_000

# Plotly + reveal.js CDN endpoints.  Pin reveal so the deck doesn't
# silently change behaviour; plotly stays "latest" since charts are built
# at render time and any 2.x release renders identically.
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
REVEAL_CDN = "https://cdn.jsdelivr.net/npm/reveal.js@5.1.0"


# ── Data loading ────────────────────────────────────────────────────────────
def load_blocks() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(blocks_wide, blocks) — same convention as historical_sim.main()."""
    blocks_pq = _ROOT / "data" / "onchain_blocks_transactions" / "per_block.parquet"
    per_block_res = hs.build_per_block_resources()

    cutoff = datetime.strptime(hs._DEFAULT_START, "%Y-%m-%d").date()
    blocks_wide = (
        hs.load_per_block(str(blocks_pq))
          .filter(pl.col("block_date") >= cutoff)
          .with_columns([
              pl.col("block_time").dt.truncate("1h").alias("hour"),
              pl.col("block_date").cast(pl.Utf8).alias("day_str"),
          ])
    )
    blocks = blocks_wide.join(
        per_block_res.rename({"block": "block_number"}),
        on="block_number", how="inner",
    )
    return blocks_wide, blocks


# ── Charts ──────────────────────────────────────────────────────────────────
def _layout_common(title: str, ytitle: str) -> dict:
    return dict(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=18, color="#111")),
        margin=dict(l=70, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0, font=dict(size=11)),
        hovermode="x",
        font=dict(size=12, color="#222"),
        autosize=True,
        yaxis=dict(title=ytitle, showline=True, linewidth=1.0,
                   linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside"),
        xaxis=dict(showline=True, linewidth=1.0,
                   linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside"),
    )


CONGESTED_SURPLUS_SHARE = 0.05   # daily threshold: surplus / l2_total

# ArbOS DIA went live on Arbitrum One at 2026-01-08 17:00 UTC; the same
# upgrade bumped p_min from 0.01 → 0.02 gwei (the only p_min change visible
# in the on-chain block table over our window).
from datetime import datetime, date    # noqa: E402
DIA_LAUNCH_TS  = datetime(2026, 1, 8, 17, 0, 0)
# For day-level splits we treat any block on Jan 8 as "pre-DIA" (most of
# that day was still ArbOS 51 pre-Dia), so post-DIA days start Jan 9.
DIA_DAY_CUTOFF = date(2026, 1, 9)


def _daily_l2_fees(blocks_wide: pl.DataFrame) -> pl.DataFrame:
    return (
        blocks_wide
          .with_columns(pl.col("block_date").cast(pl.Date).alias("day"))
          .group_by("day")
          .agg([
              pl.col("l2_base").sum().alias("l2_base"),
              pl.col("l2_surplus").sum().alias("l2_surplus"),
          ])
          .with_columns([
              (pl.col("l2_base") + pl.col("l2_surplus")).alias("l2_total"),
          ])
          .with_columns(
              (pl.col("l2_surplus") /
               pl.col("l2_total").clip(lower_bound=1e-12)).alias("surplus_share")
          )
          .sort("day")
    )


def fig_l2_fees_daily(daily: pl.DataFrame) -> go.Figure:
    """Single-panel stacked daily bar: L2 base (the p_min component) and L2
    surplus (the dynamic-pricing premium above p_min) in ETH."""
    x = daily["day"].to_list()
    base    = daily["l2_base"].to_numpy()
    surplus = daily["l2_surplus"].to_numpy()
    total   = daily["l2_total"].to_numpy()
    sshare  = daily["surplus_share"].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=base, name="L2 base (p_min)",
        marker_color="#1f77b4", marker_line_width=0,
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>base = %{y:,.2f} ETH<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=x, y=surplus, name="L2 surplus (above p_min)",
        marker_color="#ff7f0e", marker_line_width=0,
        customdata=np.column_stack([total, sshare * 100.0]),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>surplus = %{y:,.2f} ETH<br>"
            "total = %{customdata[0]:,.2f} ETH "
            "(%{customdata[1]:.1f}% surplus share)<extra></extra>"
        ),
    ))
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        autosize=True,
        margin=dict(l=70, r=30, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0, font=dict(size=11)),
        font=dict(size=12, color="#222"),
        hovermode="x",
    )
    fig.update_yaxes(title_text="ETH",
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")

    # Annotate the DIA launch (= the on-chain p_min step from 0.01 → 0.02
    # gwei).  Plotly's add_vline annotation maths chokes on datetime
    # objects in some versions, so add the line + annotation as separate
    # `add_shape` / `add_annotation` calls keyed off the millisecond
    # timestamp — works regardless of plotly version.
    dia_ms = int(DIA_LAUNCH_TS.timestamp() * 1000)
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=dia_ms, x1=dia_ms, y0=0, y1=1,
        line=dict(color="#444", width=1.4, dash="dash"),
        layer="above",
    )
    fig.add_annotation(
        x=dia_ms, xref="x", y=1.0, yref="paper",
        text=("<b>ArbOS DIA launch</b><br>"
              "<span style='font-size:0.85em'>"
              "p_min: 0.01 → 0.02 gwei</span>"),
        showarrow=False, yanchor="bottom",
        font=dict(size=11, color="#222"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#888", borderwidth=0.5,
    )
    return fig


def l2_fee_stats_html(daily: pl.DataFrame) -> str:
    """Stats cards: window split (pre/post DIA + delta), daily L2 fees
    table (pre/post/all-time × mean/median), congested-day breakdown,
    and top-3 peak days.  Pre/post comparison is normalised by computing
    per-day means and medians, since the two windows have different
    lengths (~100 vs ~110 days)."""
    n_days = daily.height
    mean_all   = float(daily["l2_total"].mean())
    median_all = float(daily["l2_total"].median())

    pre  = daily.filter(pl.col("day") <  DIA_DAY_CUTOFF)
    post = daily.filter(pl.col("day") >= DIA_DAY_CUTOFF)

    def stats(df: pl.DataFrame) -> tuple[float, float, int]:
        if df.height == 0:
            return float("nan"), float("nan"), 0
        return (
            float(df["l2_total"].mean()),
            float(df["l2_total"].median()),
            df.height,
        )

    mean_pre,  median_pre,  n_pre  = stats(pre)
    mean_post, median_post, n_post = stats(post)

    def delta_pct(post: float, pre: float) -> str:
        if not (pre and pre == pre):  # nan guard
            return "n/a"
        d = (post - pre) / pre * 100.0
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    d_mean   = delta_pct(mean_post,   mean_pre)
    d_median = delta_pct(median_post, median_pre)

    def split_counts(df: pl.DataFrame) -> tuple[int, int]:
        if df.height == 0:
            return 0, 0
        c = int((df["surplus_share"] >= CONGESTED_SURPLUS_SHARE).sum())
        return df.height - c, c

    n_norm_pre,  n_cong_pre  = split_counts(pre)
    n_norm_post, n_cong_post = split_counts(post)

    pre_range  = (f"{pre['day'].min()} → {pre['day'].max()}"
                  if pre.height else "—")
    post_range = (f"{post['day'].min()} → {post['day'].max()}"
                  if post.height else "—")

    def peak_section(title: str, df: pl.DataFrame, n: int = 3) -> str:
        if df.height == 0:
            return ""
        top = df.sort("l2_total", descending=True).head(n)
        rows = []
        for r in top.iter_rows(named=True):
            rows.append(
                f'<div class="peak-row">'
                f'<span class="peak-day">{r["day"]}</span>'
                f'<span class="peak-amt">{r["l2_total"]:,.1f} ETH</span>'
                f'<span class="peak-sub">surplus '
                f'{r["surplus_share"]*100:.1f}%</span>'
                f'</div>'
            )
        return (
            f'<div class="peak-section">'
            f'  <div class="peak-section-tag">{title}</div>'
            f'  {"".join(rows)}'
            f'</div>'
        )

    peak_html = (
        peak_section("Pre-DIA",  pre)
        + peak_section("Post-DIA", post)
        + peak_section("All-time", daily)
    )

    def pill_row(label: str, n_norm: int, n_cong: int) -> str:
        total = n_norm + n_cong
        if total == 0:
            return ""
        pct_norm = 100.0 * n_norm / total
        pct_cong = 100.0 * n_cong / total
        return (
            f'<div class="dia-row">'
            f'  <div class="dia-tag">{label}</div>'
            f'  <div class="dia-pills">'
            f'    <span class="fee-pill normal">'
            f'      {n_norm} normal ({pct_norm:.0f}%)'
            f'    </span>'
            f'    <span class="fee-pill congested">'
            f'      {n_cong} congested ({pct_cong:.0f}%)'
            f'    </span>'
            f'  </div>'
            f'</div>'
        )

    # Mini-table for the L2 fees breakdown card.
    def fees_row(label: str, mean_v: float, median_v: float,
                 emphasise: bool = False) -> str:
        cls = "mt-total" if emphasise else ""
        return (
            f'<tr class="{cls}">'
            f'<th>{label}</th>'
            f'<td>{mean_v:,.2f}</td>'
            f'<td>{median_v:,.2f}</td>'
            f'</tr>'
        )

    fees_table = (
        '<table class="mini-stats">'
        '  <thead><tr><th></th><th>Mean</th><th>Median</th></tr></thead>'
        '  <tbody>'
        f'  {fees_row("Pre-DIA",  mean_pre,  median_pre)}'
        f'  {fees_row("Post-DIA", mean_post, median_post)}'
        f'  {fees_row("All-time", mean_all,  median_all, emphasise=True)}'
        '  </tbody>'
        '</table>'
    )

    return (
        '<div class="fee-stats-row">'
        # Card 1 — window split with pre/post delta
        f'  <div class="fee-card">'
        f'    <div class="fee-label">Window split</div>'
        f'    <div class="period-block">'
        f'      <div class="period-name">Pre-DIA</div>'
        f'      <div class="period-days">{n_pre} days</div>'
        f'      <div class="period-range">{pre_range}</div>'
        f'    </div>'
        f'    <div class="period-block">'
        f'      <div class="period-name">Post-DIA</div>'
        f'      <div class="period-days">{n_post} days</div>'
        f'      <div class="period-range">{post_range}</div>'
        f'    </div>'
        f'    <div class="delta-block">'
        f'      <div class="delta-row"><span class="kv-key">'
        f'        &Delta; daily mean</span>'
        f'      <span class="kv-val">{d_mean}</span></div>'
        f'      <div class="delta-row"><span class="kv-key">'
        f'        &Delta; daily median</span>'
        f'      <span class="kv-val">{d_median}</span></div>'
        f'    </div>'
        f'  </div>'
        # Card 2 — daily L2 fees, pre vs post vs all-time
        f'  <div class="fee-card">'
        f'    <div class="fee-label">Daily L2 fees (ETH)</div>'
        f'    {fees_table}'
        f'  </div>'
        # Card 3 — day type
        f'  <div class="fee-card">'
        f'    <div class="fee-label">Day type, before vs after ArbOS DIA</div>'
        f'    <div class="fee-note">'
        f'      Congested = at least '
        f'{int(CONGESTED_SURPLUS_SHARE*100)}% of the day&rsquo;s L2 fees came from '
        f'surplus pricing (price above p_min). Otherwise the day is normal.'
        f'    </div>'
        f'    {pill_row("Pre-DIA",  n_norm_pre,  n_cong_pre)}'
        f'    {pill_row("Post-DIA", n_norm_post, n_cong_post)}'
        f'  </div>'
        # Card 4 — top peak days, broken down by pre-DIA / post-DIA / all-time
        f'  <div class="fee-card peak-card">'
        f'    <div class="fee-label">Top peak days</div>'
        f'    {peak_html}'
        f'  </div>'
        '</div>'
    )


# ── Per-tx sample for descriptive stats (slide 5) ───────────────────────────
def _build_tx_sample(n_target: int) -> pl.DataFrame:
    """Stream every per-tx parquet in arrow batches; per-batch coin-flip at
    `keep_frac` keeps memory bounded.  No min-gas filter, so small txs (the
    bulk of the on-chain population) are represented honestly in the
    descriptive stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = [
        "computation", "wasmComputation",
        "storageAccessRead", "storageAccessWrite",
        "storageGrowth", "historyGrowth",
        "l2Calldata", "l1Calldata", "total",
    ]
    paths = sorted((_ROOT / "data" / "multigas_usage_extracts")
                   .glob("*/per_tx.parquet"))
    total_rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)
    keep_frac  = min(1.0, (n_target * 1.4) / max(total_rows, 1))
    print(f"  sampling {n_target:,} from {total_rows:,} rows "
          f"(keep_frac={keep_frac:.4g})")

    rng = np.random.default_rng(42)
    parts: list[pl.DataFrame] = []
    n_kept = 0
    for p in paths:
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=1_000_000, columns=cols):
            mask = rng.random(batch.num_rows) < keep_frac
            if mask.any():
                tbl = batch.filter(pa.array(mask))
                parts.append(pl.from_arrow(tbl))
                n_kept += int(mask.sum())
    sample = pl.concat(parts)
    if sample.height > n_target:
        sample = sample.sample(n=n_target, seed=42)
    return sample


def load_or_build_tx_sample() -> pl.DataFrame:
    if TX_SAMPLE_PARQUET.exists():
        print(f"  loading cached tx sample: {TX_SAMPLE_PARQUET}")
        return pl.read_parquet(TX_SAMPLE_PARQUET)
    print(f"  building tx sample (target {TX_SAMPLE_N:,})")
    s = _build_tx_sample(TX_SAMPLE_N)
    TX_SAMPLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    s.write_parquet(TX_SAMPLE_PARQUET)
    print(f"  cached → {TX_SAMPLE_PARQUET}")
    return s


def tx_resource_stats_html(sample: pl.DataFrame, total_pool: int) -> str:
    """Per-resource descriptive stats on the per-tx sample.  Returns a
    self-contained HTML <table> + a one-line caption above it."""
    s = sample.with_columns(
        (pl.col("computation") + pl.col("wasmComputation")).alias("comp_total")
    )
    rows: list[dict] = []
    spec = [
        ("Computation",     "comp_total"),
        ("Storage Read",    "storageAccessRead"),
        ("Storage Write",   "storageAccessWrite"),
        ("Storage Growth",  "storageGrowth"),
        ("History Growth",  "historyGrowth"),
        ("L2 Calldata",     "l2Calldata"),
        ("L1 Calldata",     "l1Calldata"),
        ("Total (per tx)",  "total"),
    ]
    n = s.height
    for name, col in spec:
        v = s[col].cast(pl.Float64)
        rows.append({
            "name":     name,
            "mean":     float(v.mean()),
            "p25":      float(v.quantile(0.25)),
            "median":   float(v.quantile(0.50)),
            "p75":      float(v.quantile(0.75)),
            "p99":      float(v.quantile(0.99)),
            "pct_zero": 100.0 * float((v == 0).sum()) / max(n, 1),
        })

    def fmt(v: float) -> str:
        if v >= 1_000_000:
            return f"{v/1e6:,.2f}M"
        if v >= 1_000:
            return f"{v/1e3:,.1f}K"
        return f"{v:,.0f}"

    headers = ["Resource", "Mean", "P25", "Median", "P75", "P99", "% zero"]
    th = "".join(
        f'<th>{h}</th>' if h == "Resource" else f'<th class="num">{h}</th>'
        for h in headers
    )
    body = []
    for r in rows:
        emphasise = "row-total" if r["name"].startswith("Total") else ""
        body.append(
            f'<tr class="{emphasise}">'
            f'<td>{r["name"]}</td>'
            f'<td class="num">{fmt(r["mean"])}</td>'
            f'<td class="num">{fmt(r["p25"])}</td>'
            f'<td class="num">{fmt(r["median"])}</td>'
            f'<td class="num">{fmt(r["p75"])}</td>'
            f'<td class="num">{fmt(r["p99"])}</td>'
            f'<td class="num">{r["pct_zero"]:.1f}%</td>'
            f'</tr>'
        )
    table = (
        f'<table class="stats-table"><thead><tr>{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
    caption = (
        f'<p class="stats-caption">Distribution of per-transaction gas '
        f"usage across the seven priced resources, computed on a uniform "
        f"{n:,}-tx sample drawn from the full {total_pool:,}-tx pool. "
        f"The %&nbsp;zero column shows the share of transactions that "
        f"consume none of that resource. All values are in raw gas units.</p>"
    )
    return caption + table


def fig_hourly_combined(rk_hr: pl.DataFrame) -> go.Figure:
    """Two stacked panels — absolute Mgas/hr (top) and % share (bottom) —
    sharing one hourly x-axis.  One legend on top covers both panels."""
    from plotly.subplots import make_subplots

    x = rk_hr["hour"].to_list()
    priced_total = np.zeros(rk_hr.height)
    for k in hs.PRICED_KINDS:
        priced_total = priced_total + rk_hr[f"gas_{k}"].to_numpy()
    safe = np.where(priced_total > 0, priced_total, 1.0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "Hourly priced gas usage, by resource (Mgas)",
            "Resource composition: share of priced gas (%)",
        ),
    )

    cum_abs = np.zeros(rk_hr.height)
    cum_pct = np.zeros(rk_hr.height)
    for k in hs.PRICED_KINDS:
        y_abs = rk_hr[f"gas_{k}"].to_numpy() / 1e6
        share = rk_hr[f"gas_{k}"].to_numpy() / safe * 100.0
        share = np.where(priced_total > 0, share, 0.0)
        label = "Computation (+ WASM)" if k == "Computation" else k
        color = hs.RESOURCE_COLORS.get(k, "#888")

        fig.add_trace(go.Bar(
            x=x, y=y_abs, base=cum_abs, name=label,
            marker_color=color, marker_line_width=0,
            legendgroup=k, showlegend=True,
            hovertemplate=f"{label}: %{{y:,.0f}} Mgas<extra></extra>",
        ), row=1, col=1)
        cum_abs = cum_abs + y_abs

        fig.add_trace(go.Bar(
            x=x, y=share, base=cum_pct, name=label,
            marker_color=color, marker_line_width=0,
            legendgroup=k, showlegend=False,
            hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
        ), row=2, col=1)
        cum_pct = cum_pct + share

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        autosize=True,
        margin=dict(l=70, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0, font=dict(size=11)),
        hovermode="x",
        font=dict(size=12, color="#222"),
    )
    fig.update_yaxes(title_text="Mgas / hour", row=1, col=1,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    fig.update_yaxes(title_text="% of priced gas", row=2, col=1,
                     range=[0, 100],
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside")
    return fig


# ── Stat block (slide 2) ────────────────────────────────────────────────────
def _total_txs() -> int:
    """Sum row counts across every per_tx.parquet via parquet metadata
    (no row scan)."""
    import pyarrow.parquet as pq
    paths = sorted((_ROOT / "data" / "multigas_usage_extracts")
                   .glob("*/per_tx.parquet"))
    return sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)


def stat_html(blocks_wide: pl.DataFrame, blocks: pl.DataFrame) -> str:
    date_min = blocks_wide["block_date"].min()
    date_max = blocks_wide["block_date"].max()
    n_days   = (date_max - date_min).days + 1

    n_blocks_full = blocks_wide.height
    n_txs         = _total_txs()

    # Total priced gas (Tgas) — 7 resources counted (incl L1 calldata,
    # which is tracked even though ArbOS 60 doesn't price it dynamically).
    total_priced_gas = sum(
        float(blocks[f"{c}"].sum()) if c != "computation"
        else float((blocks["computation"] + blocks["wasmComputation"]).sum())
        for c in [
            "computation", "storageAccessRead", "storageAccessWrite",
            "storageGrowth", "historyGrowth", "l2Calldata", "l1Calldata",
        ]
    )
    total_gas_tgas = total_priced_gas / 1e12

    resource_names = (
        "Computation, Storage Read, Storage Write, Storage Growth, "
        "History Growth, L2 Calldata, L1 Calldata"
    )

    rows = [
        ("Window",              f"{date_min} → {date_max}"),
        ("Days",                f"{n_days:,}"),
        ("Total blocks",        f"{n_blocks_full:,}"),
        ("Total transactions",  f"{n_txs:,}"),
        ("Total gas",           f"{total_gas_tgas:,.2f} Tgas"),
        ("Resources tracked",
                                f"7: {resource_names}"),
    ]
    body = "\n".join(
        f'<div class="label">{k}</div><div class="val">{v}</div>'
        for k, v in rows
    )
    return f'<div class="stat-grid">{body}</div>'


# ── HTML rendering ──────────────────────────────────────────────────────────
def fig_div(fig: go.Figure, div_id: str) -> str:
    """Return a Plotly div + script (no plotly.js — that's CDN-loaded once).
    `responsive=True` lets the chart fill its parent on every resize."""
    return fig.to_html(
        include_plotlyjs=False, full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Arbitrum Dynamic Pricing Update: Revenue Impact Analysis</title>

  <link rel="stylesheet" href="{REVEAL_CDN}/dist/reset.css">
  <link rel="stylesheet" href="{REVEAL_CDN}/dist/reveal.css">
  <link rel="stylesheet" href="{REVEAL_CDN}/dist/theme/white.css">

  <script src="{PLOTLY_CDN}"></script>

  <style>
    :root {{
      --r-main-font-size: 28px;
      --r-heading-color: #111;
      --r-main-color:    #222;
    }}
    .reveal .slides {{ text-align: left; }}
    .reveal h1 {{ font-size: 1.9em; margin-bottom: 0.3em; }}
    .reveal h2 {{ font-size: 1.3em; margin: 0 0 0.6em; color: #333;
                  font-weight: 500; }}
    .reveal h3 {{ font-size: 1.0em; color: #555; margin-top: 0; }}

    .stat-grid {{
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 0.4em 1.6em;
      font-size: 0.65em;
      max-width: 900px;
    }}
    .stat-grid .label {{ color: #666; }}
    .stat-grid .val   {{ color: #111; font-weight: 500;
                         font-variant-numeric: tabular-nums; }}

    .plotly-frame {{ width: 100%; height: 78vh; }}
    .reveal .plotly-graph-div {{ width: 100% !important;
                                  height: 100% !important; }}

    /* Chart slides: title block on top, chart fills the rest of the
       slide via flexbox.  Reveal applies `display: block` on the
       active section through a class selector with high specificity
       (`.reveal .slides > section.present`), so we bump specificity
       and use !important to win the cascade. */
    .reveal .slides > section.chart-slide,
    .reveal .slides > section.chart-slide.present {{
      display: flex !important;
      flex-direction: column !important;
      height: 100% !important;
      padding: 0 !important;
      box-sizing: border-box !important;
    }}
    .reveal .slides > section.chart-slide > h2,
    .reveal .slides > section.chart-slide > h3 {{
      flex: 0 0 auto !important;
    }}
    .reveal .slides > section.chart-slide > .plotly-frame {{
      flex: 1 1 0 !important;
      min-height: 0 !important;
      position: relative !important;
      width: 100% !important;
      height: auto !important;
    }}
    .reveal .slides > section.chart-slide > .plotly-frame > div {{
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
    }}

    .source-list {{
      display: flex; flex-direction: column; gap: 0.7em;
      margin-top: 0.5em; max-width: 1100px;
    }}
    .source {{
      border-left: 3px solid #1f77b4;
      padding: 0.4em 0.9em;
      background: rgba(31, 119, 180, 0.04);
    }}
    .source-name {{
      font-weight: 600; font-size: 0.78em; color: #111;
    }}
    .src-tag {{
      display: inline-block; margin-left: 0.7em;
      font-size: 0.78em; font-weight: 500;
      color: #555; background: #f0f0f0;
      border: 1px solid #d0d0d0; border-radius: 3px;
      padding: 0.05em 0.45em; letter-spacing: 0.04em;
    }}
    .source-desc {{
      font-size: 0.6em; color: #444; margin-top: 0.2em;
      line-height: 1.45;
    }}
    .source-tables {{
      margin-top: 0.45em; display: flex; flex-wrap: wrap; gap: 0.4em;
    }}
    .source-tables .entity {{
      font-size: 0.55em; font-weight: 500;
      color: #1f3a5f; background: #e9f0fa;
      border: 1px solid #c8d6ec; border-radius: 3px;
      padding: 0.1em 0.55em;
    }}

    .stats-caption {{
      font-size: 0.65em; color: #444; max-width: 1100px;
      line-height: 1.5; margin: 0.4em 0 1em;
    }}
    table.stats-table {{
      border-collapse: collapse; font-size: 0.62em;
      max-width: 1100px; margin-top: 0.2em;
    }}
    table.stats-table th, table.stats-table td {{
      padding: 0.35em 0.9em; border-bottom: 1px solid #e0e0e0;
      text-align: left;
    }}
    table.stats-table th {{
      background: #f4f4f4; border-bottom: 2px solid #333;
      font-weight: 600; color: #111;
    }}
    table.stats-table td.num, table.stats-table th.num {{
      text-align: right; font-variant-numeric: tabular-nums;
    }}
    table.stats-table tr.row-total td {{
      font-weight: 600; background: #fafafa;
      border-top: 2px solid #aaa;
    }}

    /* Single visual language for every metric card.  All numbers use the
       slide's default sans-serif with tabular figures so digits align
       cleanly without the visual weight of monospace.  Sizes/weights are
       coherent across cards. */
    .fee-stats-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1em; margin-top: 0.6em;
      font-variant-numeric: tabular-nums;
    }}
    .fee-card {{
      border-left: 3px solid #1f77b4;
      padding: 0.55em 0.95em;
      background: rgba(31, 119, 180, 0.04);
    }}
    .fee-label {{
      font-size: 0.55em; color: #555;
      text-transform: uppercase; letter-spacing: 0.05em;
      font-weight: 500;
    }}
    /* Equal-rank key/value rows inside a card: parallel stats (e.g. mean
       and median) read at the same scale, just with the key gray and
       the number darker.  Replaces the old big-value + tiny-sub pattern. */
    .kv-row {{
      display: flex; justify-content: space-between; gap: 0.6em;
      align-items: baseline; margin-top: 0.3em; line-height: 1.25;
    }}
    .kv-key {{ font-size: 0.6em; color: #666; }}
    .kv-val {{ font-size: 0.65em; font-weight: 600; color: #111; }}

    /* Long-form descriptive note inside a card (no key/value structure). */
    .fee-note {{
      font-size: 0.5em; color: #555; margin-top: 0.25em;
      font-weight: 400; line-height: 1.4;
      text-transform: none; letter-spacing: 0;
    }}

    /* Window-split card: stacked pre/post period blocks + delta row. */
    .period-block {{ margin-top: 0.45em; line-height: 1.2; }}
    .period-name  {{ font-size: 0.6em; font-weight: 600; color: #333; }}
    .period-days  {{ font-size: 0.65em; font-weight: 600; color: #111; }}
    .period-range {{ font-size: 0.5em; color: #777; }}
    .delta-block  {{ margin-top: 0.5em; padding-top: 0.4em;
                     border-top: 1px solid #ddd; }}
    .delta-row    {{ display: flex; justify-content: space-between;
                     align-items: baseline; margin-top: 0.2em; }}

    /* Mini stats table for the daily-L2-fees card. */
    table.mini-stats {{
      width: 100%; border-collapse: collapse;
      margin-top: 0.45em; font-variant-numeric: tabular-nums;
    }}
    table.mini-stats th, table.mini-stats td {{
      padding: 0.2em 0.3em; text-align: right;
    }}
    table.mini-stats thead th {{
      font-size: 0.55em; font-weight: 500; color: #666;
      border-bottom: 1px solid #ccc;
    }}
    table.mini-stats tbody th {{
      text-align: left; font-size: 0.6em; font-weight: 500; color: #444;
    }}
    table.mini-stats tbody td {{
      font-size: 0.65em; font-weight: 600; color: #111;
    }}
    table.mini-stats tr.mt-total th,
    table.mini-stats tr.mt-total td {{
      border-top: 1px solid #ccc; padding-top: 0.3em;
      font-weight: 700;
    }}
    .fee-pill  {{
      font-size: 0.55em; font-weight: 600;
      padding: 0.15em 0.55em; border-radius: 3px;
      border: 1px solid;
    }}
    .fee-pill.normal    {{ color: #2ca02c; border-color: #2ca02c;
                           background: rgba(44, 160, 44, 0.06); }}
    .fee-pill.congested {{ color: #d62728; border-color: #d62728;
                           background: rgba(214, 39, 40, 0.06); }}
    .peak-card {{ border-left-color: #ff7f0e;
                   background: rgba(255, 127, 14, 0.04); }}
    .peak-row  {{
      display: flex; justify-content: space-between;
      font-size: 0.6em; margin-top: 0.25em; gap: 0.5em;
      align-items: baseline; line-height: 1.25;
    }}
    .peak-day  {{ color: #111; font-weight: 600; }}
    .peak-amt  {{ color: #111; font-weight: 500; }}
    .peak-sub  {{ color: #777; font-size: 0.85em; }}
    /* Peak-days card runs tall (3 sections × 3 rows + tags), so shrink
       its internals only — other cards stay at their normal scale. */
    .peak-card .peak-section     {{ margin-top: 0.3em; }}
    .peak-card .peak-section-tag {{
      font-size: 0.5em; font-weight: 600;
      color: #b54a00; text-transform: uppercase;
      letter-spacing: 0.05em; margin-bottom: 0.05em;
    }}
    .peak-card .peak-row {{
      font-size: 0.5em; margin-top: 0.1em; line-height: 1.2;
    }}
    .peak-card .fee-label {{ margin-bottom: 0.1em; }}
    /* Tag stacks above its pill row so the two pills always sit
       side-by-side regardless of card width. */
    .dia-row   {{ margin-top: 0.5em; }}
    .dia-tag   {{
      font-size: 0.6em; font-weight: 600; color: #333;
      margin-bottom: 0.2em;
    }}
    .dia-pills {{ display: flex; gap: 0.4em; flex-wrap: nowrap; }}

    .footer {{ font-size: 0.6em; color: #888; margin-top: 1.2em; }}

    .logo-row {{
      display: flex; align-items: center; justify-content: flex-start;
      gap: 3em; margin-top: 1.6em;
    }}
    .logo-row img {{
      height: 64px; width: auto; object-fit: contain;
      opacity: 0.92;
    }}

    .reveal section.cover {{ padding-top: 4vh; position: relative; }}
    .reveal section.cover h1 {{ font-size: 2.4em; margin-bottom: 0.1em; }}
    .reveal section.cover h2 {{ font-size: 1.5em; color: #555;
                                  font-weight: 400; }}
    .reveal section.cover .logo-row {{ margin-top: 6vh; gap: 4em; }}
    .reveal section.cover .logo-row img {{ height: 80px; }}

    .internal-badge {{
      position: absolute; top: 1.5vh; right: 0;
      font-size: 0.55em; font-weight: 600; letter-spacing: 0.18em;
      color: #b54a00; border: 1.5px solid #b54a00;
      padding: 0.25em 0.7em; border-radius: 3px;
      text-transform: uppercase;
    }}

    .version-line {{
      font-size: 0.7em; color: #888;
      margin-top: 0.4em; letter-spacing: 0.05em;
    }}
  </style>
</head>
<body>
<div class="reveal">
  <div class="slides">

    <!-- Slide 1: cover -->
    <section class="cover">
      <span class="internal-badge">Internal</span>
      <h1>Arbitrum Dynamic Pricing Update</h1>
      <h2>Revenue Impact Analysis</h2>
      <p class="version-line">V 1.0 &middot; In progress</p>
      <div class="logo-row">
        <img src="assets/arbitrum.png"         alt="Arbitrum">
        <img src="assets/offchain_labs.png"    alt="Offchain Labs">
        <img src="assets/entropy_advisors.png" alt="Entropy Advisors">
      </div>
    </section>

    <!-- Slide 2: dataset scope -->
    <section>
      <h2>Historical simulation: data scope</h2>
      {STATS}
      <h3 style="margin-top:1.4em">Data sources</h3>
      <div class="source-list">
        <div class="source">
          <div class="source-name">
            Arbitrum on-chain data
            <span class="src-tag">Internal EA db</span>
          </div>
          <div class="source-desc">
            Block headers, tx counts, base fees, L1 calldata costs and
            reverts. Wide window. Drives the observed-on-chain reference
            series, the ArbOS 51 baseline and the spam-classification flags.
          </div>
          <div class="source-tables">
            <span class="entity">Blocks</span>
            <span class="entity">Transactions</span>
            <span class="entity">Reverted transactions</span>
          </div>
        </div>
        <div class="source">
          <div class="source-name">
            Per-tx multi-gas extracts
            <span class="src-tag">Offchain Labs node extracts</span>
          </div>
          <div class="source-desc">
            Per-transaction breakdown of every priced resource
            (computation, storage read/write/growth, history growth,
            L2/L1 calldata, WASM compute). Drives the ArbOS 60 + per-
            resource backlog panels.
          </div>
          <div class="source-tables">
            <span class="entity">Blocks</span>
            <span class="entity">Transactions (with per-resource gas)</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Slide 3: daily L2 fees + breakdown stats -->
    <section class="chart-slide">
      <h2>Daily L2 fees: base and surplus</h2>
      <h3>After ArbOS DIA, fees dropped and congestion eased.</h3>
      <div class="plotly-frame">{FIG_FEES}</div>
      {FEE_STATS}
    </section>

    <!-- Slide 4: hourly resource (absolute + share, stacked) -->
    <section class="chart-slide">
      <h2>Hourly priced gas: absolute and composition</h2>
      <h3>Top: Mgas/hr by resource. Bottom: same data, normalised to 100 %.</h3>
      <div class="plotly-frame">{FIG1}</div>
    </section>

    <!-- Slide 5: per-resource per-tx stats table -->
    <section>
      <h2>Per-transaction gas distribution</h2>
      {STATS_TABLE}
    </section>

  </div>
</div>

<script src="{REVEAL_CDN}/dist/reveal.js"></script>
<script>
  Reveal.initialize({{
    hash: true,
    controls: true,
    progress: true,
    slideNumber: 'c/t',
    width: 1400,
    height: 880,
    margin: 0.04,
    transition: 'fade',
  }});

  // Reveal hides slides until they are active; plotly measures 0×0 on
  // initial render and needs a resize once a slide becomes visible.
  // Two-step resize handles the race where the first call lands before
  // the slide-transition transform has finished.
  function resizeVisiblePlots() {{
    document.querySelectorAll('.plotly-graph-div').forEach(el => {{
      if (el.offsetParent !== null) Plotly.Plots.resize(el);
    }});
  }}
  Reveal.on('ready slidechanged', () => {{
    resizeVisiblePlots();
    setTimeout(resizeVisiblePlots, 250);
  }});
  window.addEventListener('resize', () => setTimeout(resizeVisiblePlots, 50));
</script>
</body>
</html>
"""


def main() -> None:
    print("Loading blocks + per-block resources...")
    blocks_wide, blocks = load_blocks()
    rk_hr = hs.hourly_gas_per_kind(blocks)
    print(f"  blocks_wide: {blocks_wide.height:,}  "
          f"blocks (priced): {blocks.height:,}  "
          f"hours: {rk_hr.height:,}")

    print("Building figures...")
    daily_fees = _daily_l2_fees(blocks_wide)
    f_fees     = fig_l2_fees_daily(daily_fees)
    f1         = fig_hourly_combined(rk_hr)

    print("Loading per-tx sample for stats table...")
    tx_sample = load_or_build_tx_sample()
    stats_table = tx_resource_stats_html(tx_sample, total_pool=_total_txs())

    print("Rendering deck...")
    page = PAGE_TEMPLATE.format(
        REVEAL_CDN=REVEAL_CDN,
        PLOTLY_CDN=PLOTLY_CDN,
        STATS=stat_html(blocks_wide, blocks),
        FIG_FEES=fig_div(f_fees, "fig-l2-fees"),
        FEE_STATS=l2_fee_stats_html(daily_fees),
        FIG1=fig_div(f1, "fig-hourly-combined"),
        STATS_TABLE=stats_table,
    )
    OUT_HTML.write_text(page)
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
