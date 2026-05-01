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
import time
from datetime import datetime, timedelta

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

# Cached uniform reservoir sample of per-tx rows.  Kept as a small fallback
# for ad-hoc stats; not used by slide 5 anymore.
TX_SAMPLE_PARQUET = _ROOT / "data" / "presentation_tx_stats_sample.parquet"
TX_SAMPLE_N       = 500_000

# Full-dataset histogram + exact-stats cache built by one streaming pass
# over every per_tx.parquet.  Per-resource per-regime arrays so slide 5
# can render the on-disk distribution of all 594 M txs instead of a
# sample.  Rebuilds automatically when missing.
TX_FULL_HIST_NPZ = _ROOT / "data" / "presentation_tx_full_histograms.npz"
HIST_LOG_HI      = float(np.log1p(2e7))   # upper bound for log1p(gas) bins
HIST_N_DISP      = 60                      # bins shown on screen
HIST_N_FINE      = 500                     # bins used to estimate percentiles

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
                    xanchor="center", x=0.5, font=dict(size=11)),
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
                    xanchor="center", x=0.5, font=dict(size=11)),
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
    # Annotation sits *inside* the plot area, just below the top edge,
    # so it doesn't overlap with the top-center legend on hover.
    fig.add_annotation(
        x=dia_ms, xref="x", y=0.96, yref="paper",
        text=("<b>ArbOS DIA launch</b><br>"
              "<span style='font-size:0.85em'>"
              "p_min: 0.01 → 0.02 gwei</span>"),
        showarrow=False, yanchor="top", xanchor="left",
        xshift=6,
        font=dict(size=11, color="#222"),
        bgcolor="rgba(255,255,255,0.92)",
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
                  if pre.height else "n/a")
    post_range = (f"{post['day'].min()} → {post['day'].max()}"
                  if post.height else "n/a")

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

        def pill(cls: str, n: int, name: str, pct: float) -> str:
            return (
                f'<span class="fee-pill {cls}">'
                f'  <span class="pill-main">'
                f'    <span class="pill-n">{n}</span>'
                f'    <span class="pill-name">{name}</span>'
                f'  </span>'
                f'  <span class="pill-pct">{pct:.0f}%</span>'
                f'</span>'
            )

        return (
            f'<div class="dia-row">'
            f'  <div class="dia-tag">{label}</div>'
            f'  <div class="dia-pills">'
            f'    {pill("normal",    n_norm, "normal",    pct_norm)}'
            f'    {pill("congested", n_cong, "congested", pct_cong)}'
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
    descriptive stats.  `block` is included so downstream code can split
    pre/post-DIA via the block-time mapping."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = [
        "block",
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


def load_taylor_figure() -> str:
    """Read the standalone arbos51_taylor_comparison.html and return just
    the plotly <div> + script.  Prefer this over re-running the 35 s × 4
    sim every time the deck builds."""
    src = _ROOT / "figures" / "arbos51_taylor_comparison.html"
    if not src.exists():
        return ('<div style="padding:1em;color:#a00;">'
                f'figure not built yet — run '
                '<code>python scripts/arbos51_taylor_comparison.py</code></div>')
    html = src.read_text()
    # Extract the body content (everything inside <body>...</body>).
    import re
    m = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL)
    return m.group(1) if m else html


def load_taylor_stats_table() -> str:
    """Render the Δ%-vs-Taylor-4 stats table from the JSON the comparison
    script writes.  Returns an empty string if the JSON isn't there yet
    (the script hasn't been run since the last refactor)."""
    import json
    src = _ROOT / "data" / "arbos51_taylor_stats.json"
    if not src.exists():
        return ""
    blob = json.loads(src.read_text())
    rows = blob["rows"]
    label = {"taylor5": "Taylor-5", "taylor6": "Taylor-6", "exp": "true exp"}

    head = (
        '<thead><tr>'
        '<th rowspan="2">Method <span class="hint">(vs Taylor-4)</span></th>'
        '<th colspan="2" class="grp">Normal hours</th>'
        '<th colspan="2" class="grp">Peak hours</th>'
        '</tr><tr>'
        '<th class="num sub">median</th><th class="num sub">mean</th>'
        '<th class="num sub">median</th><th class="num sub">mean</th>'
        '</tr></thead>'
    )

    def cell(v):
        return f'<td class="num">{v:+.2f}%</td>'

    body_rows = []
    for r in rows:
        body_rows.append(
            f'<tr><td>{label[r["method"]]}</td>'
            f'{cell(r["median_normal"])}{cell(r["mean_normal"])}'
            f'{cell(r["median_peak"])}{cell(r["mean_peak"])}</tr>'
        )
    n_peak = blob["n_peak_hours"]
    n_total = blob["n_total_hours"]
    thresh = blob["peak_threshold_eth"]
    return (
        '<div class="taylor-stats">'
        '  <div class="taylor-stats-caption">'
        f'    Δ% of hourly L2 fees vs the on-chain Taylor-4 baseline. '
        f'    Peak hour := observed fee in the top 10 % '
        f'    (≥ {thresh:.2f} ETH/hour, {n_peak} of {n_total} hours).'
        '  </div>'
        '  <table class="stats-table">'
        f'   {head}<tbody>{"".join(body_rows)}</tbody>'
        '  </table>'
        '</div>'
    )


def arbos60_code_slide_html() -> str:
    """Slide 6: spec block on top (p_min, constraint inequality, per-set
    weight + ladder tables) followed by the code blocks pairing each
    Python signature from arbos60.py with its equation in LaTeX.

    All LaTeX subscripts use explicit braces (e.g. `\\sum_{k}` not
    `\\sum_k`) — the un-braced form caused MathJax to throw 'Double
    subscripts: use braces to clarify' on backlog_per_second / fee_per_tx
    when the very next token also carried a subscript."""
    methods = [
        (
            "backlog_per_second(self, inflow_i_per_t, T_i)",
            r"B_{i,j}(t+1) = \max(0, B_{i,j}(t) + \sum_{k} a_{i,k} \, g_{k}(t) "
            r"- T_{i,j} \, \Delta t)",
        ),
        (
            "compute_set_exponents(self, g_per_block, block_t)",
            r"E_{i}(t) = \sum_{j} \frac{B_{i,j}(t)}{A_{i,j} \, T_{i,j}}",
        ),
        (
            "price_per_resource(self, g_per_block, block_t)",
            r"e_{k}(t) = \max_{i} \{ a_{i,k} \, E_{i}(t) \} "
            r"\qquad p_{k}(t) = p_{\min} \, \exp(e_{k}(t))",
        ),
        (
            "fee_per_tx(self, p_per_resource, tx_block_idx, tx_g_per_resource)",
            r"F_{tx} = \sum_{k} g_{tx,k} \, p_{k}(t_{tx})",
        ),
    ]
    blocks: list[str] = []
    for sig, eq in methods:
        blocks.append(
            '<div class="method-block">'
            '  <div class="method-sig">'
            f'    <span class="kw">def</span> '
            f'<span class="nm">{sig.split("(",1)[0]}</span>'
            f'<span class="args">({sig.split("(",1)[1]}</span>'
            '  </div>'
            f'  <div class="method-eq">\\[{eq}\\]</div>'
            '</div>'
        )
    class_header = (
        '<div class="method-block class-header">'
        '  <div class="method-sig">'
        '    <span class="kw">class</span> '
        '<span class="nm">Arbos60GasPricing</span>:'
        '  </div>'
        '  <div class="class-doc">'
        '    Per-resource dynamic pricing. '
        '    <span class="ix">i</span> = set, '
        '    <span class="ix">j</span> = constraint, '
        '    <span class="ix">k</span> = resource, '
        '    <span class="ix">t</span> = wall-clock second '
        '    (backlog ticks every <b>1 s</b>, which aggregates '
        '    <b>4 blocks</b> at Arbitrum&rsquo;s 0.25 s block time).'
        '  </div>'
        '</div>'
    )

    # Constants & per-set tables sourced from arbos60.py (Set 1 & Set 2
    # configurations from the proposal — kept verbatim so it stays in sync
    # with the simulator if the constants change there).
    # Mirrors arbos60.py SET_WEIGHTS_1/SET_LADDERS_1 (Set 1) and
    # SET_WEIGHTS_2/SET_LADDERS_2 (Set 2 — short 2-constraint ladder).
    sets_data = [
        (
            "Set 1 (default)",
            # Per-i: (label, weights dict, ladder list[(T, A)])
            [
                ("Storage/Compute mix 1",
                 {"c": 1.0, "sw": 0.67, "sr": 0.14, "sg": 0.06},
                 [(15.40, 10_000), (20.41, 2_861), (27.06, 819),
                  (35.86, 234), (47.53, 67), (63.00, 19)]),
                ("Storage/Compute mix 2",
                 {"c": 0.0625, "sw": 1.0, "sr": 0.21, "sg": 0.09},
                 [(3.13, 10_000), (4.16, 4_488), (5.53, 2_014),
                  (7.35, 904), (9.77, 406), (12.99, 182)]),
                ("History Growth",
                 {"hg": 1.0},
                 [(67.30, 10_000), (81.27, 1_591), (98.14, 253),
                  (118.50, 40), (143.10, 6), (172.80, 1)]),
                ("Long-term Disk Growth",
                 {"sw": 0.8812, "sg": 0.2526, "hg": 0.301, "l2": 1.0},
                 [(2.30, 36_000)]),
            ],
        ),
        (
            "Set 2 (alternative)",
            [
                ("Storage/Compute mix 1",
                 {"c": 1.0, "sw": 0.6714, "sr": 0.141, "sg": 0.0604},
                 [(15.40, 10_000), (55.12, 14)]),
                ("Storage/Compute mix 2",
                 {"c": 0.0625, "sw": 1.0, "sr": 0.21, "sg": 0.09},
                 [(3.13, 10_000), (10.09, 102)]),
                ("History Growth",
                 {"hg": 1.0},
                 [(67.30, 10_000), (166.04, 1)]),
                ("Long-term Disk Growth",
                 {"sw": 0.8812, "sg": 0.2526, "hg": 0.301, "l2": 1.0},
                 [(2.30, 36_000)]),
            ],
        ),
    ]

    def _fmt_a(a: int) -> str:
        return f"{a/1000:.1f}K" if a >= 1000 else f"{a}"

    def _weights_str(w: dict) -> str:
        return ", ".join(f"{k}:{v:g}" for k, v in w.items())

    def _ladder_str(ladder: list) -> str:
        return ", ".join(f"({T:g}, {_fmt_a(A)})" for T, A in ladder)

    def _fmt_coeff(c: float) -> str:
        """LaTeX coefficient.  Drops 1.0, integer-formats whole numbers."""
        if abs(c - 1.0) < 1e-12:
            return ""
        if abs(c - round(c)) < 1e-12:
            return f"{int(round(c))}\\,"
        return f"{c:g}\\,"

    def _ineq_latex(weights: dict, i_idx: int) -> str:
        """Build a per-set weighted inequality in LaTeX."""
        terms = []
        for k, coeff in weights.items():
            if coeff == 0:
                continue
            terms.append(f"{_fmt_coeff(coeff)}g_{{{k}}}")
        body = " + ".join(terms) if terms else "0"
        return f"i = {i_idx} : \\quad & {body} \\;\\le\\; T_{{{i_idx},j}}"

    def _ladder_table(label: str, ladder: list, i_idx: int) -> str:
        rows_html = []
        for j, (T, A) in enumerate(ladder):
            rows_html.append(
                f'<tr><td class="lj">{j}</td>'
                f'<td class="lT">{T:g}</td>'
                f'<td class="lA">{A:,}</td></tr>'
            )
        return (
            '<div class="ladder-table">'
            f'  <div class="ladder-title">'
            f'    Set {i_idx} <span class="ladder-note">{label}</span>'
            f'  </div>'
            '  <table class="set-table">'
            '    <thead><tr>'
            '      <th class="lj"><i>j</i></th>'
            '      <th class="lT"><i>T<sub>i,j</sub></i> (Mgas/s)</th>'
            '      <th class="lA"><i>A<sub>i,j</sub></i> (s)</th>'
            '    </tr></thead>'
            f'   <tbody>{"".join(rows_html)}</tbody>'
            '  </table>'
            '</div>'
        )

    set_blocks: list[str] = []
    for set_idx, (set_name, rows) in enumerate(sets_data, start=1):
        # ── Weighted inequalities (LaTeX aligned environment) ───────────
        ineq_lines = [
            _ineq_latex(weights, i_idx + 1)
            for i_idx, (_label, weights, _l) in enumerate(rows)
        ]
        ineq_latex = (
            r"\begin{aligned}" + r" \\ ".join(ineq_lines) + r"\end{aligned}"
        )
        # ── 4 ladder tables side-by-side ───────────────────────────────
        ladder_tables = "".join(
            _ladder_table(label, ladder, i_idx + 1)
            for i_idx, (label, _w, ladder) in enumerate(rows)
        )
        set_blocks.append(
            f'<div class="set-card set-card-{set_idx}">'
            f'  <div class="set-title">{set_name}</div>'
            f'  <div class="set-ineq method-eq">\\[{ineq_latex}\\]</div>'
            f'  <div class="ladder-row">{ladder_tables}</div>'
            '</div>'
        )

    inequality_eq = (
        r"\sum_{k} a_{i,k} \, g_{k}(t) \le T_{i,j}"
        r"\quad \text{(sustained over a window of } A_{i,j} \text{ seconds)}"
    )
    notation_block = (
        '<div class="notation-block">'
        '  <span class="notation-label">Notation</span>'
        '  <span><span class="sym">g<sub>c</sub></span> Computation</span>'
        '  <span><span class="sym">g<sub>sw</sub></span> Storage Write</span>'
        '  <span><span class="sym">g<sub>sr</sub></span> Storage Read</span>'
        '  <span><span class="sym">g<sub>sg</sub></span> Storage Growth</span>'
        '  <span><span class="sym">g<sub>hg</sub></span> History Growth</span>'
        '  <span><span class="sym">g<sub>l2</sub></span> L2 Calldata</span>'
        '  <span class="notation-note">all in gas / second</span>'
        '</div>'
    )
    spec_block = (
        '<div class="spec-block">'
        '  <div class="const-pmin">'
        '    p<sub>min</sub> = <b>0.02 gwei</b>'
        '    <span class="const-note">post-DIA</span>'
        '    <span class="const-sep">·</span>'
        '    <b>0.01 gwei</b>'
        '    <span class="const-note">pre-DIA</span>'
        '  </div>'
        f'  {notation_block}'
        '  <div class="spec-ineq">'
        '    <div class="spec-label">'
        '      Constraint inequality, for each set <i>i</i> and constraint <i>j</i>:'
        '    </div>'
        f'   \\[{inequality_eq}\\]'
        '  </div>'
        f'  {"".join(set_blocks)}'
        '</div>'
    )

    github_icon = (
        '<svg class="gh-icon" viewBox="0 0 16 16" width="14" height="14" '
        'fill="currentColor" aria-hidden="true">'
        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07'
        '.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94'
        '-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58'
        ' 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-'
        '3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-'
        '2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27'
        ' 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27'
        '.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-'
        '.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-'
        '4.42-3.58-8-8-8z"/></svg>'
    )
    arbos51_github_url = (
        "https://github.com/Entropy-Advisors/arbos60_dynamic_pricing_sim"
        "/blob/main/scripts/arbos51.py"
    )
    arbos51_source_link = (
        '<div class="source-link">'
        f'  <a href="{arbos51_github_url}" target="_blank" rel="noopener">'
        f'    {github_icon}'
        '    <span>scripts/arbos51.py on GitHub</span>'
        '  </a>'
        '</div>'
    )
    arbos51_taylor_section = (
        '<section class="chart-stats-slide">'
        '  <h2>ArbOS 51 also implemented in code</h2>'
        '  <div class="ladder-row">'
        '    <table class="ladder-table"><thead>'
        '      <tr><th>j</th><th>0</th><th>1</th><th>2</th>'
        '          <th>3</th><th>4</th><th>5</th></tr>'
        '    </thead><tbody>'
        '      <tr><th>T<sub>j</sub> (Mgas/s)</th>'
        '          <td>10</td><td>14</td><td>20</td>'
        '          <td>29</td><td>41</td><td>60</td></tr>'
        '      <tr><th>A<sub>j</sub> (s)</th>'
        '          <td>86,400</td><td>13,485</td><td>2,105</td>'
        '          <td>329</td><td>52</td><td>9</td></tr>'
        '    </tbody></table>'
        '    <div class="ladder-caption">'
        '      ArbOS 51 DIA ladder: 6 constraints (T<sub>j</sub> = drain rate, '
        '      A<sub>j</sub> = response time).'
        '    </div>'
        '  </div>'
        f' <div class="hist-grid">{load_taylor_figure()}</div>'
        f' {arbos51_source_link}'
        '</section>'
    )

    github_url = (
        "https://github.com/Entropy-Advisors/arbos60_dynamic_pricing_sim"
        "/blob/main/scripts/arbos60.py"
    )
    source_link = (
        '<div class="source-link">'
        f'  <a href="{github_url}" target="_blank" rel="noopener">'
        f'    {github_icon}'
        '    <span>scripts/arbos60.py on GitHub</span>'
        '  </a>'
        '</div>'
    )
    # Three nested <section>s = a vertical group: ArbOS 60 spec → ArbOS 60
    # class methods → ArbOS 51 implementation + Taylor comparison.
    return (
        '<section>\n'
        '  <section>\n'
        '    <h2>ArbOS 60 dynamic pricing implemented in code</h2>\n'
        f'    {spec_block}\n'
        '  </section>\n'
        '  <section>\n'
        '    <h2>ArbOS 60: class methods and equations</h2>\n'
        '    <div class="method-list">\n'
        f'      {class_header}\n'
        + "".join(blocks) +
        '    </div>\n'
        f'    {source_link}\n'
        '  </section>\n'
        f'  {arbos51_taylor_section}\n'
        '</section>'
    )


# ── ArbOS 60 vs 51 revenue (no demand elasticity) ───────────────────────────
# Cache rendered Plotly divs so repeated build.py runs (which re-import the
# heavy historical_sim helpers anyway) skip the figure-building cost too.
_REVENUE_FIG_CACHE: dict[str, go.Figure] = {}


def _load_revenue_hourly() -> pl.DataFrame:
    """Load the cached ArbOS 51 / 60 hourly ETH revenue table (with p_min
    sweep columns added).  Streamed once by revenue_no_elasticity.py and
    persisted at data/revenue_comparison_cache/hourly.parquet."""
    import revenue_no_elasticity as rne                          # noqa: E402
    hourly = rne.compute_hourly_revenue(use_cache=True)
    return rne.add_pmin_sweep(hourly)


def _cumulative_panel(
    fig, daily: pl.DataFrame, *,
    series: list[tuple[str, str, str, str]], row: int, col: int,
    show_legend: bool,
):
    """Add cumulative-ETH lines for `series` (column, label, colour, dash)
    to `fig` at (row, col).  Each line re-zeroes at the first point of
    `daily` so 51 and 60 always start at 0."""
    x = daily["day"].to_list()
    for column, label, colour, dash in series:
        cum = daily[column].cum_sum().to_numpy()
        fig.add_trace(go.Scatter(
            x=x, y=cum, name=label, mode="lines",
            line=dict(color=colour, width=1.8, dash=dash),
            legendgroup=label, showlegend=show_legend,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{label}: " "%{y:,.1f} ETH<extra></extra>"
            ),
        ), row=row, col=col)


def _performance_panel(
    fig, daily: pl.DataFrame, *,
    baseline_col: str,
    series: list[tuple[str, str, str, str]],
    row: int, col: int,
    show_legend: bool,
):
    """Cumulative % delta of each regime over ArbOS 51, normalised by
    the slice's final ArbOS 51 cumulative so the line starts at 0 %
    (both regimes had 0 ETH before day 1) and traces the running
    excess as a % of the slice's total ArbOS 51 revenue:
        y(t) = (cum_X(t) − cum_51(t)) / cum_51(slice end) × 100
    ArbOS 51 stays flat at 0 %.  The other lines start at 0 % and
    widen toward the final % gain by the slice end."""
    days = daily["day"].to_list()
    cum_base = daily[baseline_col].cum_sum().to_numpy().astype(np.float64)
    final_base = float(cum_base[-1]) if len(cum_base) and cum_base[-1] > 0 else 1.0

    # Prepend a day-0 anchor at 0 ETH gap so every line literally
    # starts on the chart at 0 %.
    if len(days):
        first_day = days[0]
        try:
            anchor_day = first_day - timedelta(days=1)
        except Exception:
            anchor_day = first_day
    x = [anchor_day] + days if len(days) else days

    for column, label, colour, dash in series:
        if column == baseline_col:
            y_inner = np.zeros_like(cum_base)
        else:
            cum = daily[column].cum_sum().to_numpy().astype(np.float64)
            y_inner = (cum - cum_base) / final_base * 100.0
        y = np.concatenate([[0.0], y_inner]) if len(days) else y_inner
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label, mode="lines",
            line=dict(color=colour, width=1.8, dash=dash),
            legendgroup=label, showlegend=show_legend,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{label}: " "%{y:+.2f}% of slice 51 revenue<extra></extra>"
            ),
        ), row=row, col=col)


def fig_cum_revenue_overview(hourly: pl.DataFrame) -> go.Figure:
    """2 rows × 3 cols cumulative ETH revenue.
        Row 1: cumulative ETH (Full / 90D / 30D), all lines start at 0 ETH.
        Row 2: cumulative performance (% gain vs ArbOS 51), all lines
               start at 0 % on the first day of the slice.
    ArbOS 51 vs ArbOS 60 set 1 vs ArbOS 60 set 2."""
    if "cum_overview" in _REVENUE_FIG_CACHE:
        return _REVENUE_FIG_CACHE["cum_overview"]

    from plotly.subplots import make_subplots                    # noqa: E402
    import revenue_no_elasticity as rne                          # noqa: E402
    daily = rne.hourly_to_daily(hourly)

    end = daily["day"].max()
    daily_90 = daily.filter(pl.col("day") >= end - timedelta(days=90))
    daily_30 = daily.filter(pl.col("day") >= end - timedelta(days=30))

    series = [
        ("eth_51",    "ArbOS 51",        "#d62728", "solid"),
        ("eth_60",    "ArbOS 60 set 1",  "#1f77b4", "solid"),
        ("eth_60_v2", "ArbOS 60 set 2",  "#17becf", "dash"),
    ]
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Last 30 days", "Last 90 days", "Full window",
            "", "", "",
        ),
        horizontal_spacing=0.07, vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )
    # Columns: 30d (left) | 90d (middle) | full window (right).
    # Row 1 — cumulative ETH.
    _cumulative_panel(fig, daily_30, series=series,
                      row=1, col=1, show_legend=True)
    _cumulative_panel(fig, daily_90, series=series,
                      row=1, col=2, show_legend=False)
    _cumulative_panel(fig, daily,    series=series,
                      row=1, col=3, show_legend=False)
    # Row 2 — performance (% gain vs ArbOS 51).
    _performance_panel(fig, daily_30, baseline_col="eth_51",
                       series=series, row=2, col=1, show_legend=False)
    _performance_panel(fig, daily_90, baseline_col="eth_51",
                       series=series, row=2, col=2, show_legend=False)
    _performance_panel(fig, daily,    baseline_col="eth_51",
                       series=series, row=2, col=3, show_legend=False)

    for c in (1, 2, 3):
        fig.update_yaxes(
            title_text=("Cumulative ETH" if c == 1 else ""),
            showline=True, linewidth=1.0, linecolor="rgba(0,0,0,0.45)",
            mirror=True, ticks="outside", row=1, col=c,
        )
        fig.update_yaxes(
            title_text=("% gain vs ArbOS 51" if c == 1 else ""),
            ticksuffix="%",
            showline=True, linewidth=1.0, linecolor="rgba(0,0,0,0.45)",
            mirror=True, ticks="outside", row=2, col=c,
        )
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside", row=1, col=c)
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside", row=2, col=c)
    fig.update_layout(
        template="plotly_white", autosize=True, height=720,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    _REVENUE_FIG_CACHE["cum_overview"] = fig
    return fig


def fig_revenue_timeseries(hourly: pl.DataFrame,
                           daily: pl.DataFrame) -> go.Figure:
    """Headline hourly + daily ETH revenue, full window. 4-panel:
        row 1: hourly 51/60 lines
        row 2: hourly Δ (60 − 51), filled area
        row 3: daily 51/60 lines
        row 4: daily Δ (60 − 51), filled area
    """
    if "ts_full" in _REVENUE_FIG_CACHE:
        return _REVENUE_FIG_CACHE["ts_full"]
    from plotly.subplots import make_subplots                    # noqa: E402
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=[0.30, 0.18, 0.30, 0.22],
        subplot_titles=(
            "Hourly ETH revenue",
            "Δ hourly ETH (60 − 51)",
            "Daily ETH revenue",
            "Δ daily ETH (60 − 51)",
        ),
    )
    h_x  = hourly["hour"].to_list()
    h_51 = hourly["eth_51"].to_numpy()
    h_60 = hourly["eth_60"].to_numpy()
    d_x  = daily["day"].to_list()
    d_51 = daily["eth_51"].to_numpy()
    d_60 = daily["eth_60"].to_numpy()

    fig.add_trace(go.Scatter(x=h_x, y=h_51, name="ArbOS 51",
                              line=dict(color="#d62728", width=1.2),
                              legendgroup="51", showlegend=True,
                              hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                              "51 = %{y:,.3f} ETH/h<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=h_x, y=h_60, name="ArbOS 60 set 1",
                              line=dict(color="#1f77b4", width=1.2),
                              legendgroup="60", showlegend=True,
                              hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                              "60 = %{y:,.3f} ETH/h<extra></extra>"),
                  row=1, col=1)
    delta_h = h_60 - h_51
    fig.add_trace(go.Scatter(x=h_x, y=delta_h, name="Δ (60 − 51)",
                              line=dict(color="#888", width=0.6),
                              fill="tozeroy",
                              fillcolor="rgba(120,120,120,0.35)",
                              legendgroup="delta", showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#444", width=0.6, dash="dot"),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=d_x, y=d_51, name="ArbOS 51 (daily)",
                              line=dict(color="#d62728", width=1.5),
                              legendgroup="51", showlegend=False,
                              hovertemplate="%{x|%Y-%m-%d}<br>"
                              "51 = %{y:,.2f} ETH/day<extra></extra>"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=d_x, y=d_60, name="ArbOS 60 (daily)",
                              line=dict(color="#1f77b4", width=1.5),
                              legendgroup="60", showlegend=False,
                              hovertemplate="%{x|%Y-%m-%d}<br>"
                              "60 = %{y:,.2f} ETH/day<extra></extra>"),
                  row=3, col=1)
    delta_d = d_60 - d_51
    fig.add_trace(go.Scatter(x=d_x, y=delta_d, name="Δ daily",
                              line=dict(color="#888", width=0.8),
                              fill="tozeroy",
                              fillcolor="rgba(120,120,120,0.35)",
                              showlegend=False),
                  row=4, col=1)
    fig.add_hline(y=0, line=dict(color="#444", width=0.6, dash="dot"),
                  row=4, col=1)

    fig.update_yaxes(title_text="ETH/h",  row=1, col=1)
    fig.update_yaxes(title_text="Δ ETH/h", row=2, col=1)
    fig.update_yaxes(title_text="ETH/day", row=3, col=1)
    fig.update_yaxes(title_text="Δ ETH/day", row=4, col=1)
    fig.update_xaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_layout(
        template="plotly_white", autosize=True, height=820,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    _REVENUE_FIG_CACHE["ts_full"] = fig
    return fig


def fig_cum_grid(daily: pl.DataFrame) -> go.Figure:
    """4 periods × 4 p_min cumulative ETH grid.  Subplot titles trimmed
    to just "p_min = X gwei" on the top row + a row-label annotation on
    the left edge of each row, so titles don't collide with axis labels."""
    if "cum_grid" in _REVENUE_FIG_CACHE:
        return _REVENUE_FIG_CACHE["cum_grid"]
    import revenue_no_elasticity as rne                           # noqa: E402
    fig = rne.build_cumulative_grid(daily)

    n_rows = len(rne.CUM_PERIODS)
    n_cols = len(rne.PMIN_SWEEP)
    new_titles: list[str] = []
    for r in range(n_rows):
        for c, pmin in enumerate(rne.PMIN_SWEEP):
            # Only the top row gets the p_min label; lower rows blank.
            new_titles.append(f"p<sub>min</sub> = {pmin:.2f} gwei"
                              if r == 0 else "")
    # Walk fig.layout.annotations to find subplot-title annotations and
    # rewrite their text in row-major order.
    if fig.layout.annotations:
        title_anns = [a for a in fig.layout.annotations
                      if getattr(a, "xref", "").startswith("x")
                      or "subplot" in str(getattr(a, "xref", ""))]
        # Subplot titles always come first in `annotations` for
        # make_subplots; the count matches n_rows*n_cols.
        for ann, new_text in zip(fig.layout.annotations[:n_rows * n_cols],
                                  new_titles):
            ann.text = new_text

    # Add a left-side label per row (Full window / Last 90D / ...).
    # x=-0.04 in paper coords + right-anchor parks the label inside the
    # left margin without going off-screen at narrower viewport widths.
    annotations = list(fig.layout.annotations)
    for r, (label, _) in enumerate(rne.CUM_PERIODS):
        y_center = 1.0 - (r + 0.5) / n_rows
        annotations.append(dict(
            text=f"<b>{label}</b>",
            xref="paper", yref="paper",
            x=-0.04, y=y_center,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(size=11, color="#444"),
        ))
    fig.layout.annotations = annotations

    # Drop redundant per-panel y/x titles; keep "cum ETH" on every
    # leftmost panel (tiny so it doesn't clash with the row label).
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_yaxes(title_text=("cum ETH" if c == 1 else ""),
                             title_font=dict(size=10),
                             row=r, col=c)
            fig.update_xaxes(title_text=("day" if r == n_rows else ""),
                             row=r, col=c)

    fig.update_layout(
        title_text="",
        height=260 * n_rows + 120,
        # ~150 px left margin: hosts row label + y-axis tick labels +
        # "cum ETH" title without overflowing the figure box.
        margin=dict(l=150, r=40, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        hovermode="x",
    )
    _REVENUE_FIG_CACHE["cum_grid"] = fig
    return fig


def fig_revenue_spike_zoom(hourly: pl.DataFrame,
                            label: str, t0, t1) -> go.Figure:
    """2-panel zoom on a spike window: 51/60 hourly line + delta."""
    cache_key = f"spike_{label}"
    if cache_key in _REVENUE_FIG_CACHE:
        return _REVENUE_FIG_CACHE[cache_key]
    from plotly.subplots import make_subplots                    # noqa: E402

    sub = hourly.filter((pl.col("hour") >= t0) & (pl.col("hour") <= t1))
    if sub.is_empty():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white", height=520,
            annotations=[dict(text=f"no data in {label}",
                              showarrow=False, xref="paper", yref="paper",
                              x=0.5, y=0.5)],
        )
        _REVENUE_FIG_CACHE[cache_key] = fig
        return fig

    x   = sub["hour"].to_list()
    y51 = sub["eth_51"].to_numpy()
    y60 = sub["eth_60"].to_numpy()
    delta = y60 - y51

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            f"Hourly ETH revenue: {label}",
            "Δ L2 fee (60 − 51)",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=y51, name="ArbOS 51",
                              line=dict(color="#d62728", width=1.4),
                              hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                              "51 = %{y:,.3f} ETH/h<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y60, name="ArbOS 60 set 1",
                              line=dict(color="#1f77b4", width=1.4),
                              hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                              "60 = %{y:,.3f} ETH/h<extra></extra>"),
                  row=1, col=1)
    # Δ split into positive (60 above 51, grey fill) and negative
    # (60 below 51, red fill = "underwater") so loss windows pop out.
    delta_pos = np.where(delta >= 0, delta, 0.0)
    delta_neg = np.where(delta <  0, delta, 0.0)
    fig.add_trace(go.Scatter(
        x=x, y=delta_pos, name="Δ ≥ 0", showlegend=False,
        line=dict(color="rgba(120,120,120,0.7)", width=0.6),
        fill="tozeroy",
        fillcolor="rgba(120,120,120,0.35)",
        hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                       "Δ = +%{y:,.3f} ETH/h<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=delta_neg, name="Δ < 0", showlegend=False,
        line=dict(color="rgba(214,39,40,0.85)", width=0.6),
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.30)",
        hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                       "Δ = %{y:,.3f} ETH/h<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#444", width=0.6, dash="dot"),
                  row=2, col=1)
    fig.update_yaxes(title_text="ETH/h",  row=1, col=1)
    fig.update_yaxes(title_text="Δ L2 fee (ETH/h)", row=2, col=1)
    fig.update_xaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_layout(
        template="plotly_white", autosize=True, height=520,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    _REVENUE_FIG_CACHE[cache_key] = fig
    return fig


def fig_distribution_boxplots(hourly: pl.DataFrame,
                                daily: pl.DataFrame) -> go.Figure:
    """3 rows × 2 cols of boxplots:
        row 1: 51 vs 60 set 1 vs 60 set 2  (hourly | daily)
        row 2: p_min range — 51 + 60 at 0.02/0.03/0.04/0.05  (hourly | daily)
    """
    if "boxplots" in _REVENUE_FIG_CACHE:
        return _REVENUE_FIG_CACHE["boxplots"]
    from plotly.subplots import make_subplots                    # noqa: E402
    import revenue_no_elasticity as rne                           # noqa: E402

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.13,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Hourly ETH (set 1 vs set 2)",
            "Daily ETH (set 1 vs set 2)",
            "Hourly ETH, p_min range",
            "Daily ETH, p_min range",
        ),
    )
    # ── Row 1: set 1 vs set 2 ──────────────────────────────────────
    for arr, name, color in (
        (hourly["eth_51"].to_numpy(),    "ArbOS 51",         "#d62728"),
        (hourly["eth_60"].to_numpy(),    "ArbOS 60 set 1",   "#1f77b4"),
        (hourly["eth_60_v2"].to_numpy(), "ArbOS 60 set 2",   "#17becf"),
    ):
        fig.add_trace(go.Box(y=arr, name=name, marker_color=color,
                              boxpoints="outliers", showlegend=False),
                       row=1, col=1)
    for arr, name, color in (
        (daily["eth_51"].to_numpy(),    "ArbOS 51",         "#d62728"),
        (daily["eth_60"].to_numpy(),    "ArbOS 60 set 1",   "#1f77b4"),
        (daily["eth_60_v2"].to_numpy(), "ArbOS 60 set 2",   "#17becf"),
    ):
        fig.add_trace(go.Box(y=arr, name=name, marker_color=color,
                              boxpoints="outliers", showlegend=False),
                       row=1, col=2)
    fig.update_yaxes(title_text="ETH/h",  row=1, col=1, type="log")
    fig.update_yaxes(title_text="ETH/day", row=1, col=2, type="log")

    # ── Row 2: p_min range ─────────────────────────────────────────
    fig.add_trace(go.Box(y=hourly["eth_51"].to_numpy(),
                          name="ArbOS 51", marker_color="#d62728",
                          boxpoints="outliers", showlegend=False),
                   row=2, col=1)
    for pmin in rne.PMIN_SWEEP:
        fig.add_trace(go.Box(
            y=hourly[f"eth_60_pmin_{pmin:g}"].to_numpy(),
            name=f"60, p_min={pmin:g}",
            marker_color=rne.PMIN_COLORS[pmin],
            boxpoints="outliers", showlegend=False,
        ), row=2, col=1)
    fig.add_trace(go.Box(y=daily["eth_51"].to_numpy(),
                          name="ArbOS 51", marker_color="#d62728",
                          boxpoints="outliers", showlegend=False),
                   row=2, col=2)
    for pmin in rne.PMIN_SWEEP:
        fig.add_trace(go.Box(
            y=daily[f"eth_60_pmin_{pmin:g}"].to_numpy(),
            name=f"60, p_min={pmin:g}",
            marker_color=rne.PMIN_COLORS[pmin],
            boxpoints="outliers", showlegend=False,
        ), row=2, col=2)
    fig.update_yaxes(title_text="ETH/h",  row=2, col=1, type="log")
    fig.update_yaxes(title_text="ETH/day", row=2, col=2, type="log")

    fig.update_xaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_layout(
        template="plotly_white", autosize=True, height=820,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
    )
    _REVENUE_FIG_CACHE["boxplots"] = fig
    return fig


def revenue_summary_tables_html(hourly: pl.DataFrame,
                                  daily: pl.DataFrame) -> str:
    """Stats table (per window) + p_min range table — both produced by
    revenue_no_elasticity.py.  Side by side under .revenue-tables."""
    import revenue_no_elasticity as rne                           # noqa: E402
    stats = rne.build_stats_table(hourly, daily)
    sweep = rne.build_pmin_sweep_table(hourly)
    return (
        '<div class="revenue-tables">'
        '  <div class="rev-table-block">'
        '    <h3>Window summary, 51 vs 60 set 1 (ETH, p<sub>min</sub> = 0.02 gwei)</h3>'
        f'    {stats}'
        '  </div>'
        '  <div class="rev-table-block">'
        '    <h3>Full-window totals, p<sub>min</sub> range</h3>'
        f'    {sweep}'
        '  </div>'
        '</div>'
    )


def fig_cum_revenue_pmin_sweep(hourly: pl.DataFrame) -> go.Figure:
    """ArbOS 60 p_min range cumulative ETH, 2 rows × 3 cols.
        Cols:  Last 30D (left)  |  Last 90D  |  Full window (right).
        Row 1: cumulative ETH per p_min vs ArbOS 51.
        Row 2: % gain vs ArbOS 51, all lines start at 0 % at slice start."""
    if "cum_pmin" in _REVENUE_FIG_CACHE:
        return _REVENUE_FIG_CACHE["cum_pmin"]

    from plotly.subplots import make_subplots                    # noqa: E402
    import revenue_no_elasticity as rne                          # noqa: E402
    daily = rne.hourly_to_daily(hourly)
    end = daily["day"].max()
    daily_90 = daily.filter(pl.col("day") >= end - timedelta(days=90))
    daily_30 = daily.filter(pl.col("day") >= end - timedelta(days=30))

    series = [("eth_51", "ArbOS 51", "#d62728", "solid")]
    for pmin in rne.PMIN_SWEEP:
        series.append((
            f"eth_60_pmin_{pmin:g}",
            f"ArbOS 60, p_min = {pmin:g} gwei",
            rne.PMIN_COLORS[pmin], "solid",
        ))

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Last 30 days", "Last 90 days", "Full window",
            "", "", "",
        ),
        horizontal_spacing=0.07, vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )
    # Row 1 — cumulative ETH per p_min (and ArbOS 51 baseline).
    _cumulative_panel(fig, daily_30, series=series,
                      row=1, col=1, show_legend=True)
    _cumulative_panel(fig, daily_90, series=series,
                      row=1, col=2, show_legend=False)
    _cumulative_panel(fig, daily,    series=series,
                      row=1, col=3, show_legend=False)
    # Row 2 — performance (% gain over ArbOS 51).
    _performance_panel(fig, daily_30, baseline_col="eth_51",
                       series=series, row=2, col=1, show_legend=False)
    _performance_panel(fig, daily_90, baseline_col="eth_51",
                       series=series, row=2, col=2, show_legend=False)
    _performance_panel(fig, daily,    baseline_col="eth_51",
                       series=series, row=2, col=3, show_legend=False)
    for c in (1, 2, 3):
        fig.update_yaxes(
            title_text=("Cumulative ETH" if c == 1 else ""),
            showline=True, linewidth=1.0, linecolor="rgba(0,0,0,0.45)",
            mirror=True, ticks="outside", row=1, col=c,
        )
        fig.update_yaxes(
            title_text=("% gain vs ArbOS 51" if c == 1 else ""),
            ticksuffix="%",
            showline=True, linewidth=1.0, linecolor="rgba(0,0,0,0.45)",
            mirror=True, ticks="outside", row=2, col=c,
        )
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside", row=1, col=c)
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside", row=2, col=c)
    fig.update_layout(
        template="plotly_white", autosize=True, height=720,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    _REVENUE_FIG_CACHE["cum_pmin"] = fig
    return fig


def revenue_no_elasticity_slide_html() -> str:
    """Top-level <section> for the ArbOS 60 vs 51 first-glance comparison
    (no demand elasticity).  Vertical sub-slides cover:
        1. Intro / what we are doing.
        2. Hourly + daily revenue time series + deltas.
        3. Cumulative ETH revenue overview.
        4. ArbOS 60 p_min range, cumulative ETH.
        5. Cumulative grid (4 periods × 4 p_min).
        6. Spike-window zoom (Jan 29 to Feb 7).
        7. Distribution boxplots.
        8. Summary tables.
    """
    print("Loading revenue-comparison hourly cache (slide 7)...")
    t0 = time.time()
    hourly = _load_revenue_hourly()
    import revenue_no_elasticity as rne                           # noqa: E402
    daily  = rne.hourly_to_daily(hourly)
    # ts figure dropped — slide removed.  Function kept for reference.
    # f_ts     = fig_revenue_timeseries(hourly, daily)
    f_cum    = fig_cum_revenue_overview(hourly)
    f_pmin   = fig_cum_revenue_pmin_sweep(hourly)
    f_grid   = fig_cum_grid(daily)
    f_box    = fig_distribution_boxplots(hourly, daily)
    spike_label, spike_t0, spike_t1 = rne.SPIKE_WINDOWS[0]
    f_spike  = fig_revenue_spike_zoom(hourly, spike_label,
                                       spike_t0, spike_t1)
    tables_html = revenue_summary_tables_html(hourly, daily)
    print(f"  built revenue figures in {time.time()-t0:.2f}s")

    intro_section = (
        '<section class="chart-stats-slide intro-revenue">'
        '  <h2>ArbOS 60 vs ArbOS 51: revenue, no demand elasticity modeling</h2>'
        '  <div class="intro-card">'
        '    <div class="intro-card-tag">What this section does</div>'
        '    <p>We replay every priced transaction in the dataset and '
        '       re-compute its fee under each pricing regime, keeping '
        '       the realised gas usage fixed. Only the pricing function '
        '       changes, users transact identically: this is a clean '
        '       counterfactual on the actual on-chain workload.</p>'
        '    <p><b>Demand elasticity is off.</b> Price feedback into '
        '       user behaviour is added in the next section.</p>'
        '  </div>'
        '  <div class="intro-card">'
        '    <div class="intro-card-tag">What follows</div>'
        '    <ul class="intro-points">'
        '      <li><b>Spike-window zoom</b>: Jan 29 to Feb 7, the '
        '          most pronounced congestion in the dataset.</li>'
        '      <li><b>Cumulative ETH</b> per window: full / 90D / 30D, '
        '          ArbOS 51 vs 60 set 1 vs set 2.</li>'
        '      <li><b>p<sub>min</sub> range</b>: 0.02, 0.03, 0.04, '
        '          0.05 gwei across ArbOS 60.</li>'
        '      <li><b>Cumulative grid</b>: 4 windows × 4 '
        '          p<sub>min</sub> values.</li>'
        '      <li><b>Distribution boxplots</b>: hourly + daily ETH, '
        '          set 1 vs set 2 + p<sub>min</sub> range.</li>'
        '      <li><b>Summary tables</b>: window-by-window totals + '
        '          p<sub>min</sub> range totals.</li>'
        '    </ul>'
        '  </div>'
        '</section>'
    )
    cum_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Cumulative ETH revenue, 51 vs 60</h2>'
        '  <div class="slide-note">'
        '    Cumulative ETH from day zero of each slice. '
        '    51 (red), 60 set 1 (blue), 60 set 2 (cyan, dashed). '
        '    Both ArbOS 60 lines at p<sub>min</sub> = 0.02 gwei.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_cum, "fig-cum-overview")}</div>'
        '</section>'
    )
    pmin_section = (
        '<section class="chart-stats-slide">'
        '  <h2>ArbOS 60 p<sub>min</sub> range, cumulative ETH</h2>'
        '  <div class="slide-note">'
        '    Same usage, ArbOS 60 set 1 priced at p<sub>min</sub> ∈ '
        '    {0.02, 0.03, 0.04, 0.05} gwei, vs 51 baseline (red).'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_pmin, "fig-cum-pmin")}</div>'
        '</section>'
    )
    grid_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Cumulative ETH: 4 windows × 4 p<sub>min</sub></h2>'
        '  <div class="slide-note">'
        '    Each panel: one (window, p<sub>min</sub>) pair, 51 vs '
        '    60 set 1.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_grid, "fig-cum-grid")}</div>'
        '</section>'
    )
    spike_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Hourly fees re-priced with ArbOS 60: spike window</h2>'
        '  <div class="slide-note">'
        '    We replay every priced tx in the network and re-compute '
        '    its fee under ArbOS 60 with the realised gas usage. '
        '    Below: hourly aggregate of the re-priced tx fees across '
        '    the whole network, for the Jan 29 to Feb 7 spike window. '
        '    51 (red), 60 set 1 (blue), gap (60 − 51) at the bottom.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_spike, "fig-rev-spike")}</div>'
        '</section>'
    )
    box_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Distribution shape: hourly and daily ETH</h2>'
        '  <div class="slide-note">'
        '    Top row, set 1 vs set 2 (vs 51). Bottom row, 51 vs 60 '
        '    across the p<sub>min</sub> range. Log-y on all boxes.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_box, "fig-rev-box")}</div>'
        '</section>'
    )
    tables_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Summary tables</h2>'
        '  <div class="slide-note">'
        '    Window-by-window ETH totals + p<sub>min</sub> range '
        '    totals over the full window.'
        '  </div>'
        f' {tables_html}'
        '</section>'
    )
    return (
        '<section>\n'
        f'  {intro_section}\n'
        f'  {spike_section}\n'
        f'  {cum_section}\n'
        f'  {pmin_section}\n'
        f'  {grid_section}\n'
        f'  {box_section}\n'
        f'  {tables_section}\n'
        '</section>'
    )


# ── ArbOS 60 capacity headroom (slide 8) ────────────────────────────────────
_CAPACITY_FIG_CACHE: dict[str, go.Figure] = {}


def _load_capacity_summaries():
    """Returns (prices_hr, cap_hr, cap_hr_mix) — first call may take a few
    minutes if the per-second caches aren't already on disk; subsequent
    calls hit only the parquet caches."""
    import capacity_estimator as cap                              # noqa: E402
    return cap.compute_or_load_capacity_summaries()


def fig_capacity_prices(prices_hr: pl.DataFrame) -> go.Figure:
    """Per-resource simulated ArbOS 60 prices (gwei, log y)."""
    if "prices" in _CAPACITY_FIG_CACHE:
        return _CAPACITY_FIG_CACHE["prices"]
    import capacity_estimator as cap                              # noqa: E402

    fig = go.Figure()
    x = prices_hr["hour"].to_list()
    for k in cap.RESOURCES:
        fig.add_trace(go.Scatter(
            x=x, y=prices_hr[f"p_{k}"].to_numpy(),
            name=cap.RESOURCE_LABEL[k],
            line=dict(color=cap.RESOURCE_COLOR[k], width=1.2),
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"p_{{{k}}} = " "%{y:.4f} gwei<extra></extra>"),
        ))
    fig.add_hline(y=cap.P_MIN_GWEI,
                  line=dict(color="#444", width=0.8, dash="dot"),
                  annotation_text=f"p_min = {cap.P_MIN_GWEI} gwei",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#444"))
    dia_ms = int(cap.DIA_LAUNCH_TS.timestamp() * 1000)
    fig.add_vline(x=dia_ms, line=dict(color="#444", width=1.0, dash="dash"),
                  annotation_text="DIA", annotation_position="top right",
                  annotation_font=dict(size=10, color="#444"))
    fig.update_yaxes(title_text="gwei", type="log",
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_layout(
        template="plotly_white", autosize=True, height=520,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    _CAPACITY_FIG_CACHE["prices"] = fig
    return fig


def _capacity_view_panel(
    cap_hr: pl.DataFrame, *, mix_label: str,
) -> go.Figure:
    """3-row figure: capacity, headroom, gain Δ%.  Each panel shows
    ArbOS 60 Set 1 (solid blue) and Set 2 (dashed teal); ArbOS 51
    (red dashed) appears in capacity + headroom as a baseline.  Window
    is clipped to ArbOS 51 DIA activation."""
    import capacity_estimator as cap                              # noqa: E402
    from plotly.subplots import make_subplots                     # noqa: E402

    cap_hr = cap_hr.filter(pl.col("hour") >= cap.DIA_LAUNCH_TS)

    has_v2 = "cap_60_v2" in cap_hr.columns
    x = cap_hr["hour"].to_list()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.34, 0.33, 0.33],
        subplot_titles=(
            f"Capacity (Mgas/s, {mix_label})",
            f"Spare capacity % ({mix_label})",
            "Capacity gain Δ% vs ArbOS 51",
        ),
    )

    # ── Row 1: capacity in Mgas/s ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=cap_hr["cap_51"].to_numpy(), name="ArbOS 51",
        line=dict(color="#d62728", width=1.4, dash="dash"),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "cap 51 = %{y:.1f} Mgas/s<extra></extra>"),
        legendgroup="51", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=cap_hr["cap_60"].to_numpy(), name="ArbOS 60 set 1",
        line=dict(color="#1f77b4", width=1.6),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "cap 60 set 1 = %{y:.1f} Mgas/s<extra></extra>"),
        legendgroup="60v1", showlegend=True,
    ), row=1, col=1)
    if has_v2:
        fig.add_trace(go.Scatter(
            x=x, y=cap_hr["cap_60_v2"].to_numpy(), name="ArbOS 60 set 2",
            line=dict(color="#17becf", width=1.6, dash="dot"),
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           "cap 60 set 2 = %{y:.1f} Mgas/s<extra></extra>"),
            legendgroup="60v2", showlegend=True,
        ), row=1, col=1)

    # ── Row 2: headroom % ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=cap_hr["headroom_51"].to_numpy(), name="ArbOS 51",
        line=dict(color="#d62728", width=1.4, dash="dash"),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "spare 51 = %{y:.1f}%<extra></extra>"),
        legendgroup="51", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=cap_hr["headroom_60"].to_numpy(), name="ArbOS 60 set 1",
        line=dict(color="#1f77b4", width=1.6),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "spare 60 set 1 = %{y:.1f}%<extra></extra>"),
        legendgroup="60v1", showlegend=False,
    ), row=2, col=1)
    if has_v2:
        fig.add_trace(go.Scatter(
            x=x, y=cap_hr["headroom_60_v2"].to_numpy(), name="ArbOS 60 set 2",
            line=dict(color="#17becf", width=1.6, dash="dot"),
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           "spare 60 set 2 = %{y:.1f}%<extra></extra>"),
            legendgroup="60v2", showlegend=False,
        ), row=2, col=1)

    # ── Row 3: gain Δ% over ArbOS 51 ─────────────────────────────────
    gain_v1 = (cap_hr["gain_median"].to_numpy()
               if "gain_median" in cap_hr.columns
               else cap_hr["gain_60"].to_numpy())
    fig.add_trace(go.Scatter(
        x=x, y=gain_v1, name="ArbOS 60 set 1 gain",
        line=dict(color="#1f77b4", width=1.6),
        hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                       "gain set 1 = %{y:+.1f}%<extra></extra>"),
        legendgroup="60v1", showlegend=False,
    ), row=3, col=1)
    if has_v2:
        gain_v2 = (cap_hr["gain_median_v2"].to_numpy()
                   if "gain_median_v2" in cap_hr.columns
                   else cap_hr["gain_60_v2"].to_numpy())
        fig.add_trace(go.Scatter(
            x=x, y=gain_v2, name="ArbOS 60 set 2 gain",
            line=dict(color="#17becf", width=1.6, dash="dot"),
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           "gain set 2 = %{y:+.1f}%<extra></extra>"),
            legendgroup="60v2", showlegend=False,
        ), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1,
                  line=dict(color="#444", width=0.8, dash="dot"),
                  annotation_text="parity",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#666"))

    fig.update_yaxes(title_text="Mgas/s", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="% spare capacity",
                     row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Δ% vs ArbOS 51", row=3, col=1)
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_layout(
        template="plotly_white", autosize=True, height=760,
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    return fig


def fig_capacity_per_second_mix(cap_hr: pl.DataFrame) -> go.Figure:
    if "per_sec_mix" in _CAPACITY_FIG_CACHE:
        return _CAPACITY_FIG_CACHE["per_sec_mix"]
    fig = _capacity_view_panel(cap_hr, mix_label="per-second mix")
    _CAPACITY_FIG_CACHE["per_sec_mix"] = fig
    return fig


def fig_capacity_daily_ma(
    cap_hr: pl.DataFrame, *, ma_window: int = 7,
) -> go.Figure:
    """Same 3-row capacity figure, but each series is the daily mean
    smoothed by a centred moving average (default 7 days).  Reads the
    same per-second-mix hourly summary; aggregation is hour → day mean,
    then rolling mean."""
    if "daily_ma" in _CAPACITY_FIG_CACHE:
        return _CAPACITY_FIG_CACHE["daily_ma"]
    import capacity_estimator as cap                              # noqa: E402

    df = cap_hr.filter(pl.col("hour") >= cap.DIA_LAUNCH_TS)
    metric_cols = [
        "cap_51", "cap_60", "cap_60_v2",
        "headroom_51", "headroom_60", "headroom_60_v2",
        "gain_median", "gain_median_v2",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    daily = (
        df.with_columns(pl.col("hour").dt.truncate("1d").alias("day"))
          .group_by("day")
          .agg([pl.col(c).mean().alias(c) for c in metric_cols])
          .sort("day")
    )
    smoothed = daily.with_columns([
        pl.col(c).rolling_mean(window_size=ma_window, center=True,
                                min_samples=1).alias(c)
        for c in metric_cols
    ])
    fig = _capacity_view_panel(
        smoothed.rename({"day": "hour"}),
        mix_label=f"daily mean, {ma_window}-day moving average",
    )
    _CAPACITY_FIG_CACHE["daily_ma"] = fig
    return fig


def fig_capacity_hourly_mix(cap_hr_mix: pl.DataFrame) -> go.Figure:
    if "hourly_mix" in _CAPACITY_FIG_CACHE:
        return _CAPACITY_FIG_CACHE["hourly_mix"]
    fig = _capacity_view_panel(cap_hr_mix, mix_label="hourly-averaged mix")
    _CAPACITY_FIG_CACHE["hourly_mix"] = fig
    return fig


def capacity_slide_html() -> str:
    """Top-level <section> for ArbOS 60 capacity headroom.  Vertical
    sub-slides:
        1. Methodology / Ben's framing + math.
        2. Per-resource simulated prices.
        3. Capacity gain + headroom + saturation, per-second mix.
        4. Capacity gain + headroom + saturation, hourly-averaged mix.
    """
    print("Loading capacity caches (slide 8)...")
    t0 = time.time()
    prices_hr, cap_hr, cap_hr_mix = _load_capacity_summaries()
    print(f"  capacity caches loaded in {time.time()-t0:.2f}s")
    f_persec = fig_capacity_per_second_mix(cap_hr)
    f_daily  = fig_capacity_daily_ma(cap_hr)
    print(f"  built capacity figures in {time.time()-t0:.2f}s")

    def eq(latex: str) -> str:
        """Wrap a display equation in `.method-eq` (already in MathJax's
        processHtmlClass list, so it always renders inside the deck)."""
        return f'<div class="method-eq">\\[{latex}\\]</div>'

    methodology = (
        '<section class="chart-stats-slide capacity-intro">'
        '  <h2>Capacity, spare and gain: ArbOS 60 vs ArbOS 51</h2>'
        '  <div class="capacity-definition">'
        '    <div class="who">Definition</div>'
        '    Capacity is the gas-per-second throughput at which the '
        '    price first starts rising above <code>p<sub>min</sub></code>. '
        '    In ArbOS 51 it is constant (10 Mgas/s post-DIA). In '
        '    ArbOS 60 it depends on the workload mix: on set 2 the '
        '    price starts rising around G ≈ 15 Mgas/s, about a 50 % '
        '    capacity gain on average.'
        '  </div>'
        '  <div class="methodology">'
        '    <ol>'
        '      <li><b>Per-second aggregation.</b> Bucket every block '
        '          into 1 s windows. Per second t, per-resource gas '
        '          g<sub>k</sub>(t), total gas G(t) and mix '
        '          α<sub>k</sub>(t) = g<sub>k</sub>(t) / G(t).</li>'
        '      <li><b>ArbOS 60 capacity</b> (mix-dependent). Smallest '
        '          constraint j = 0 binds for sustained throughput:'
        f'         {eq(r"\text{capacity}_{60}(t) = \min_{i} \frac{T_{i,0}}{\sum_{k} a_{i,k} \, \alpha_{k}(t)}")}'
        '      </li>'
        '      <li><b>ArbOS 51 capacity</b> is constant per regime: '
        '          7 Mgas/s pre-DIA, 10 Mgas/s post-DIA.</li>'
        '      <li><b>Spare capacity</b> is the share of the ceiling '
        '          left unused at a given second:'
        f'         {eq(r"\text{spare}(t) = \frac{\text{capacity}(t) - G(t)}{\text{capacity}(t)} \times 100\%")}'
        '      </li>'
        '      <li><b>Capacity gain Δ%</b> is the relative gain of '
        '          ArbOS 60 over the constant ArbOS 51 ceiling:'
        f'         {eq(r"\Delta(t) = \frac{\text{capacity}_{60}(t) - \text{capacity}_{51}}{\text{capacity}_{51}} \times 100\%")}'
        '      </li>'
        '    </ol>'
        '  </div>'
        '</section>'
    )
    persec_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Capacity, per-second mix</h2>'
        '  <div class="slide-note">'
        '    Capacity recomputed every second from the realised '
        '    α<sub>k</sub>(t); panel metrics are hourly means of '
        '    the per-second values. Dashed line = ArbOS 51, solid '
        '    line = ArbOS 60.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_persec, "fig-cap-per-sec")}</div>'
        '</section>'
    )
    daily_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Capacity, daily 7-day moving average</h2>'
        '  <div class="slide-note">'
        '    Same three metrics as the per-second view, aggregated to '
        '    daily means and smoothed with a 7-day centred rolling '
        '    average. Dashed line = ArbOS 51, solid = ArbOS 60 set 1, '
        '    dotted = ArbOS 60 set 2.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_daily, "fig-cap-daily-ma")}</div>'
        '</section>'
    )
    return (
        '<section>\n'
        f'  {methodology}\n'
        f'  {persec_section}\n'
        f'  {daily_section}\n'
        '</section>'
    )


# ── Transaction clustering (slide 9) ────────────────────────────────────────
_CLUSTERING_FIG_CACHE: dict[str, go.Figure] = {}
TSNE_CACHE       = _ROOT / "data" / "clustering_cache" / "tsne_xy.npz"
TSNE_LARGE_CACHE = _ROOT / "data" / "clustering_cache" / "tsne_large.npz"
# t-SNE sample size — kept small for fast iteration.  20 K pts at the
# default perplexity converges in ~30-60 s on this machine.  Bump to
# 100 K+ when you want higher visual resolution and can wait.
TSNE_LARGE_N    = 20_000


def _stream_clr_sample(n_target: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Reservoir-sample `n_target` txs from the per-tx parquets and
    return their X_clr feature matrix.  Streams once over the full
    ~594 M-row dataset (~3 min cold cache).  Re-uses the same CLR
    transform as `tx_clustering._featurize_batch` so the resulting
    vectors are interchangeable with the saved fit's feature space."""
    import tx_clustering as txc                                   # noqa: E402

    rng    = np.random.default_rng(seed)
    paths  = txc._per_tx_files()
    if not paths:
        raise FileNotFoundError("no per-tx parquets found "
                                 f"under {txc.MULTIGAS_DIR}")
    res_X  = np.empty((n_target, 7), dtype=np.float64)
    n_seen = 0   # priced txs encountered so far across all chunks
    n_kept = 0
    print(f"  streaming {n_target:,} priced-tx CLR sample "
           "from per-tx parquets...")
    t0 = time.time()
    for batch in txc._iter_chunks(paths):
        out = txc._featurize_batch(batch)
        if out is None:
            continue
        _, _, _, X_clr, _ = out
        # Apply the same outlier trim as the original fit pass.
        # (The per-dim CLR bounds are saved in fit.pkl as `clr_lo`/`clr_hi`.)
        n_b = X_clr.shape[0]
        # Reservoir sampling, vectorised.
        if n_kept < n_target:
            take = min(n_b, n_target - n_kept)
            res_X[n_kept:n_kept + take] = X_clr[:take]
            n_kept += take
            # Vitter-style replacement for the rest of the chunk.
            if take < n_b:
                rest = X_clr[take:]
                # Each row in `rest` should replace a random current
                # element with prob target/n_seen_after_inclusion.
                idx_global = np.arange(n_seen + take, n_seen + n_b)
                replace_idx = rng.integers(0, idx_global + 1)
                # Only positions < n_target replace; pick the matching
                # rows from `rest`.
                mask = replace_idx < n_target
                res_X[replace_idx[mask]] = rest[mask]
        else:
            idx_global = np.arange(n_seen, n_seen + n_b)
            replace_idx = rng.integers(0, idx_global + 1)
            mask = replace_idx < n_target
            res_X[replace_idx[mask]] = X_clr[mask]
        n_seen += n_b
    print(f"  reservoir done in {time.time()-t0:.1f}s "
           f"(scanned {n_seen:,} priced txs, kept {n_kept:,})")
    return res_X[:n_kept]


def _compute_or_load_tsne_large(n_target: int = TSNE_LARGE_N
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Cached Barnes-Hut t-SNE on a large fresh CLR reservoir sample.
    Returns (xy, labels).  Cache: data/clustering_cache/tsne_large.npz."""
    if TSNE_LARGE_CACHE.exists():
        z = np.load(TSNE_LARGE_CACHE)
        if int(z["n"]) == n_target:
            print(f"  loading cached large t-SNE: {TSNE_LARGE_CACHE}")
            return z["xy"], z["labels"]
        print(f"  large-t-SNE cache size mismatch (cached n={int(z['n'])}, "
              f"requested {n_target}) — recomputing")

    import tx_clustering as txc                                   # noqa: E402
    import pickle
    from sklearn.manifold import TSNE                              # noqa: E402

    X_clr = _stream_clr_sample(n_target)
    with open(_ROOT / "data" / "clustering_cache" / "fit.pkl", "rb") as f:
        fit = pickle.load(f)
    K_chosen = int(txc.K_LOG)
    mbk      = fit["mbk"][K_chosen]
    labels   = mbk.predict(X_clr).astype(np.int8)

    print(f"  computing t-SNE on {X_clr.shape[0]:,} pts "
           "(Barnes-Hut, this takes 10-30 min)...")
    t0 = time.time()
    # Plain default-ish t-SNE.  20 K pts converges in ~30-60 s; using
    # default perplexity / exaggeration / iters since aggressive tuning
    # didn't separate the CLR clusters any better and just slowed the
    # whole thing down.
    xy = TSNE(
        n_components=2, perplexity=50,
        init="pca", learning_rate="auto",
        method="barnes_hut",
        random_state=42, n_jobs=-1,
    ).fit_transform(X_clr)
    print(f"  large t-SNE done in {time.time()-t0:.1f}s")
    TSNE_LARGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TSNE_LARGE_CACHE,
        xy=xy.astype(np.float32),
        labels=labels.astype(np.int8),
        n=np.int64(X_clr.shape[0]),
    )
    print(f"  cached → {TSNE_LARGE_CACHE}")
    return xy.astype(np.float32), labels.astype(np.int8)


def _load_clustering_cache():
    """Returns dict with: features (X_clr, Lg arrays), sample (polars df),
    fit (mbk dict, metrics dict, auto_K), aggs (n_txs, vol_sum,
    n_spam_label, loggas_hist, loggas_edges), K (the locked K used by
    plot_phase, defaults to tx_clustering.K_LOG)."""
    import pickle
    import tx_clustering as txc                                   # noqa: E402

    feats = np.load(_ROOT / "data" / "clustering_cache" / "features.npz")
    sample = pl.read_parquet(_ROOT / "data" / "clustering_cache" / "sample.parquet")
    with open(_ROOT / "data" / "clustering_cache" / "fit.pkl", "rb") as f:
        fit = pickle.load(f)
    aggs = np.load(_ROOT / "data" / "clustering_cache" / "aggs.npz")
    return {
        "X_clr":   feats["X_clr"],
        "Lg":      feats["Lg"],
        "sample":  sample,
        "fit":     fit,
        "aggs":    {k: aggs[k] for k in aggs.files},
        "K":       int(txc.K_LOG),
        "txc":     txc,
    }


def _compute_or_load_tsne(X_clr: np.ndarray, n_tsne: int = 30_000):
    """Cached Barnes-Hut t-SNE on a sub-sample of X_clr (deterministic seed).
    Cache file: data/clustering_cache/tsne_xy.npz."""
    if TSNE_CACHE.exists():
        z = np.load(TSNE_CACHE)
        if int(z["n"]) == n_tsne:
            print(f"  loading cached t-SNE: {TSNE_CACHE}")
            return z["xy"], z["idx"]
        print(f"  t-SNE cache size mismatch (cached n={int(z['n'])}, "
              f"requested {n_tsne}) — recomputing")

    print(f"  computing t-SNE on {n_tsne:,} pts (Barnes-Hut)...")
    from sklearn.manifold import TSNE                              # noqa: E402
    rng = np.random.default_rng(42)
    n = min(n_tsne, X_clr.shape[0])
    idx = rng.choice(X_clr.shape[0], size=n, replace=False)
    t0 = time.time()
    xy = TSNE(n_components=2, perplexity=50, init="pca",
              learning_rate="auto", random_state=42, n_jobs=-1).fit_transform(
                  X_clr[idx])
    print(f"  t-SNE done in {time.time()-t0:.1f}s")
    TSNE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TSNE_CACHE, xy=xy.astype(np.float32),
                         idx=idx.astype(np.int64), n=np.int64(n))
    print(f"  cached → {TSNE_CACHE}")
    return xy.astype(np.float32), idx.astype(np.int64)


# Cluster palette is intentionally disjoint from the resource palette
# (Computation = #1f77b4, Storage Write = #2ca02c, Storage Read = #98df8a,
#  Storage Growth = #d62728, History Growth = #ff7f0e, L2 Calldata = #9467bd,
#  L1 Calldata = #e377c2) so the eye doesn't read a cluster colour as a
# resource. Material-design / Tableau extended hues, all distinct from
# the resource set.
_CLUSTER_PALETTE = [
    "#00838f",   # c0 — dark teal
    "#5e35b1",   # c1 — deep purple
    "#c2185b",   # c2 — magenta
    "#fbc02d",   # c3 — gold
    "#455a64",   # c4 — slate
    "#3949ab",   # c5 — indigo (only used if K > 5)
    "#6d4c41",   # c6 — brown
    "#827717",   # c7 — olive
    "#0288d1",   # c8 — cyan
    "#7b1fa2",   # c9 — violet
]


def fig_clustering_tsne_and_k() -> go.Figure:
    """Combined viz: row 1 = t-SNE scatter (taller, colspan=2), row 2 =
    silhouette + WCSS K-selection curves side by side."""
    if "tsne_and_k" in _CLUSTERING_FIG_CACHE:
        return _CLUSTERING_FIG_CACHE["tsne_and_k"]
    from plotly.subplots import make_subplots                     # noqa: E402

    cache    = _load_clustering_cache()
    K_chosen = cache["K"]
    xy, labels = _compute_or_load_tsne_large()

    metrics = cache["fit"]["metrics"]
    Ks   = sorted(metrics["silhouette"])
    sil  = [metrics["silhouette"][k] for k in Ks]
    wcss = [metrics["wcss"][k] for k in Ks]

    # 4-col grid: t-SNE centred in cols 2-3 (60 % width), silhouette in
    # row 2 cols 1-2 (50 %), WCSS in row 2 cols 3-4 (50 %) — both
    # K-selection panels span their respective half of the slide.
    fig = make_subplots(
        rows=2, cols=4,
        column_widths=[0.20, 0.30, 0.30, 0.20],
        row_heights=[0.72, 0.28],
        vertical_spacing=0.13, horizontal_spacing=0.10,
        specs=[
            [None, {"colspan": 2}, None, None],
            [{"colspan": 2}, None, {"colspan": 2}, None],
        ],
        subplot_titles=(
            "",                      # t-SNE: empty — title goes in slide H2
            "Silhouette ↑",
            "WCSS / inertia ↓ (elbow)",
        ),
    )

    # Row 1: t-SNE scatter — one trace per cluster id.
    for c in range(K_chosen):
        mask = labels == c
        if not mask.any():
            continue
        color = _CLUSTER_PALETTE[c % len(_CLUSTER_PALETTE)]
        fig.add_trace(go.Scattergl(
            x=xy[mask, 0], y=xy[mask, 1], mode="markers",
            marker=dict(size=4.5, color=color, opacity=0.6,
                         line=dict(width=0)),
            name=f"C{c+1} (n={int(mask.sum()):,})",
            hovertemplate=(f"C{c+1}<br>tsne1=%{{x:.2f}} "
                           "tsne2=%{y:.2f}<extra></extra>"),
            legendgroup=f"C{c+1}",
        ), row=1, col=2)
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside",
                     showgrid=False, zeroline=False)
    # Lock aspect ratio so the t-SNE map renders as a square — sklearn's
    # output isn't isotropic by default and wide aspect ratios stretch
    # cluster shapes.
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2,
                     scaleanchor="x", scaleratio=1.0,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside",
                     showgrid=False, zeroline=False)

    # Row 2: silhouette spans cols 1-2 (50 % width), WCSS spans cols 3-4.
    fig.add_trace(go.Scatter(
        x=Ks, y=sil, mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=7),
        hovertemplate="K=%{x}<br>silhouette=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=Ks, y=wcss, mode="lines+markers",
        line=dict(color="#9467bd", width=2),
        marker=dict(size=7),
        hovertemplate="K=%{x}<br>WCSS=%{y:,.0f}<extra></extra>",
        showlegend=False,
    ), row=2, col=3)
    for col in (1, 3):
        fig.add_vline(x=K_chosen, row=2, col=col,
                      line=dict(color="#d62728", width=1.0, dash="dash"),
                      annotation_text=f"K = {K_chosen}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color="#d62728"))
        fig.update_xaxes(title_text="K", row=2, col=col,
                         showline=True, linewidth=1.0,
                         linecolor="rgba(0,0,0,0.45)",
                         mirror=True, ticks="outside")
    fig.update_yaxes(title_text="silhouette", row=2, col=1,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_yaxes(title_text="WCSS", row=2, col=3,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")

    fig.update_layout(
        template="plotly_white", autosize=True, height=920,
        margin=dict(l=70, r=30, t=80, b=50),
        font=dict(size=12, color="#222"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    _CLUSTERING_FIG_CACHE["tsne_and_k"] = fig
    return fig


def _load_centroid_aggs():
    """Common loader for the cluster-aggregate figures."""
    import pickle
    import tx_clustering as txc                                    # noqa: E402
    cache = _load_clustering_cache()
    aggs  = cache["aggs"]
    K     = int(aggs["K"])
    with open(_ROOT / "data" / "clustering_cache" / "aggregate_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return cache, aggs, K, txc, int(meta["min_hour"])


def fig_clustering_overview() -> go.Figure:
    """Slide-9 overview: cluster sizes (left) + volume-weighted resource
    composition per cluster (right, stacked horizontal bar).  Two panels
    side-by-side; nothing else competes for vertical space."""
    if "overview" in _CLUSTERING_FIG_CACHE:
        return _CLUSTERING_FIG_CACHE["overview"]
    from plotly.subplots import make_subplots                     # noqa: E402

    cache, aggs, K, txc, _ = _load_centroid_aggs()
    n_txs = aggs["n_txs"].astype(np.int64)
    vol_full = aggs["vol_sum"].astype(np.float64)
    keep  = vol_full.sum(axis=0) > 0
    vol   = vol_full[:, keep]
    res_keys   = [r for r, k in zip(txc.RESOURCES, keep) if k]
    res_labels = [txc.RESOURCE_LABEL[r] for r in res_keys]
    res_colors = [txc.RESOURCE_COLOR[lab] for lab in res_labels]
    comp = vol / vol.sum(axis=1, keepdims=True)

    # Cluster names are kept short ("c0", "c1", ...) on the axes; the full
    # workload tag is shown in the description cards below the figure.
    cluster_names  = [f"C{c+1}" for c in range(K)]
    cluster_colors = [_CLUSTER_PALETTE[c] for c in range(K)]
    sizes_pct = (n_txs / max(int(n_txs.sum()), 1)) * 100.0

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.30, 0.70],
        horizontal_spacing=0.10,
        subplot_titles=(
            "Cluster size (% of all transactions)",
            "Resource composition per cluster (volume-weighted)",
        ),
    )

    fig.add_trace(go.Bar(
        x=cluster_names, y=sizes_pct,
        marker_color=cluster_colors,
        text=[f"{p:.1f}%" for p in sizes_pct],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#fff", size=12),
        showlegend=False,
        hovertemplate="%{x}<br>size = %{y:.1f}% of txs<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="% of txs", row=1, col=1,
                     ticksuffix="%", rangemode="tozero",
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_xaxes(row=1, col=1,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")

    cum_x = np.zeros(K)
    for r_idx, (lab, color) in enumerate(zip(res_labels, res_colors)):
        seg_pct = comp[:, r_idx] * 100.0
        # Only show the in-bar label when the segment is wide enough to
        # fit "12.3%" cleanly; smaller segments get an empty string so
        # they don't crowd the chart.
        seg_text = [f"{p:.0f}%" if p >= 6.0 else "" for p in seg_pct]
        fig.add_trace(go.Bar(
            x=seg_pct, y=cluster_names,
            base=cum_x.copy(),
            orientation="h", name=lab, marker_color=color,
            text=seg_text, textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#fff", size=11),
            hovertemplate=(f"%{{y}}<br>{lab}: " "%{x:.1f}%<extra></extra>"),
        ), row=1, col=2)
        cum_x = cum_x + seg_pct
    fig.update_xaxes(range=[0, 100], row=1, col=2, ticksuffix="%",
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_yaxes(autorange="reversed", row=1, col=2,
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")

    fig.update_layout(
        template="plotly_white", autosize=True, height=340,
        barmode="overlay",
        margin=dict(l=70, r=30, t=60, b=40),
        font=dict(size=11, color="#222"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
    )
    _CLUSTERING_FIG_CACHE["overview"] = fig
    return fig


def _cluster_descriptions_html(*, compact: bool = False) -> str:
    """One-card-per-cluster description block.  Shows size, spam %,
    workload tag (top two resources by volume share).  The `compact`
    variant uses tighter padding + smaller font so the cards work as
    a header strip on a slide that already has a big figure."""
    cache, aggs, K, txc, _ = _load_centroid_aggs()
    n_txs = aggs["n_txs"].astype(np.int64)
    vol   = aggs["vol_sum"].astype(np.float64)
    spam  = aggs["n_spam_label"].astype(np.float64)
    n_total = max(int(n_txs.sum()), 1)

    full_comp = vol / np.where(vol.sum(axis=1, keepdims=True) > 0,
                                vol.sum(axis=1, keepdims=True), 1.0)
    cards: list[str] = []
    for c in range(K):
        size_pct = (int(n_txs[c]) / n_total) * 100.0
        sl = spam[c]
        n_spam_total = float(sl[txc.SPAM_CODE[txc.SPAM_VOL]]
                              + sl[txc.SPAM_CODE[txc.SPAM_REV]]
                              + sl[txc.SPAM_CODE[txc.SPAM_BOTH]])
        spam_pct = (n_spam_total / max(int(n_txs[c]), 1)) * 100.0
        tag = txc.label_cluster(full_comp[c])
        color = _CLUSTER_PALETTE[c]
        cards.append(
            '<div class="cluster-card">'
            f'  <div class="cluster-card-tag" style="background:{color}">'
            f'    C{c+1}'
            '  </div>'
            '  <div class="cluster-card-body">'
            f'    <div class="cluster-card-line1">{tag}</div>'
            f'    <div class="cluster-card-line2">'
            f'      {size_pct:.1f}% of txs · {spam_pct:.1f}% spam · '
            f'      {int(n_txs[c]):,} txs'
            '    </div>'
            '  </div>'
            '</div>'
        )
    grid_class = (
        "cluster-card-grid cluster-card-grid--compact"
        if compact else "cluster-card-grid"
    )
    return f'<div class="{grid_class}">' + "".join(cards) + '</div>'


def fig_clustering_per_cluster() -> go.Figure:
    """Slide-9 per-cluster details: K rows × 3 cols.  Wider hourly chart
    on the left, narrower median + spam on the right.  Cols are aligned
    across rows so the eye can scan vertically.
        col 1 : hourly resource-share stacked bar (% over time)
        col 2 : median per-tx resource share (vertical bars)
        col 3 : spam classification (Non-spam | High vol | High rev | Both)
    """
    if "per_cluster" in _CLUSTERING_FIG_CACHE:
        return _CLUSTERING_FIG_CACHE["per_cluster"]
    from plotly.subplots import make_subplots                     # noqa: E402

    cache, aggs, K, txc, min_hour = _load_centroid_aggs()
    hourly_gas    = aggs["hourly_gas"]
    share_hist    = aggs["share_hist"]
    n_txs         = aggs["n_txs"].astype(np.int64)
    n_spam_label  = aggs["n_spam_label"].astype(np.int64)

    any_gas = (hourly_gas.sum(axis=(0, 2)) > 0)
    hour_indices = np.nonzero(any_gas)[0]
    hour_dt = (
        ((min_hour + hour_indices).astype(np.int64) * 3600)
        .astype("datetime64[s]")
    )

    def _median_share(c: int, k: int) -> float:
        counts = share_hist[c, :, k]
        total  = counts.sum()
        if total == 0:
            return 0.0
        cum = np.cumsum(counts)
        b = int(np.searchsorted(cum, total / 2.0))
        return 0.5 * (txc.SHARE_EDGES[b]
                       + txc.SHARE_EDGES[min(b + 1, txc.N_SHARE_BINS)])

    titles: list[str] = []
    for c in range(K):
        n = int(n_txs[c])
        size_pct = (n / max(int(n_txs.sum()), 1)) * 100.0
        titles.append(
            f"<b>C{c+1}</b>  ·  {size_pct:.1f}% · "
            f"{n:,} txs"
        )
        titles += ["", ""]

    fig = make_subplots(
        rows=K, cols=3,
        column_widths=[0.62, 0.20, 0.18],
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
        subplot_titles=titles,
    )

    legend_ts: set[str] = set()
    legend_spam: set[str] = set()

    for c in range(K):
        row = c + 1

        # Col 1 — hourly resource-share stacked bar.
        cluster_gas = hourly_gas[c, hour_indices, :]
        row_total = cluster_gas.sum(axis=1).clip(min=1.0)
        shares = cluster_gas / row_total[:, None]
        cum = np.zeros(len(hour_indices))
        for k, r in enumerate(txc.RESOURCES):
            lab   = txc.RESOURCE_LABEL[r]
            color = txc.RESOURCE_COLOR[lab]
            y_pct = shares[:, k] * 100.0
            show  = lab not in legend_ts
            if show:
                legend_ts.add(lab)
            fig.add_trace(go.Bar(
                x=hour_dt, y=y_pct, base=cum,
                name=lab, marker_color=color, marker_line_width=0,
                legendgroup="resource",
                legendgrouptitle_text=("Hourly resource share"
                                       if show else None),
                showlegend=show,
                hovertemplate=(f"{lab}: " "%{y:.1f}%<br>"
                                "%{x|%Y-%m-%d %H:00}<extra></extra>"),
            ), row=row, col=1)
            cum = cum + y_pct
        fig.update_yaxes(title_text=("% share" if c == 0 else ""),
                         range=[0, 100], row=row, col=1)

        # Col 2 — median per-tx resource share (vertical bar).
        meds_pct = [_median_share(c, k) * 100.0
                    for k in range(len(txc.RESOURCES))]
        labels_x = [txc.RESOURCE_LABEL[r] for r in txc.RESOURCES]
        colors_x = [txc.RESOURCE_COLOR[lab] for lab in labels_x]
        fig.add_trace(go.Bar(
            x=labels_x, y=meds_pct,
            marker_color=colors_x, showlegend=False,
            hovertemplate="%{x}<br>median = %{y:.1f}%<extra></extra>",
        ), row=row, col=2)
        fig.update_yaxes(title_text=("median %" if c == 0 else ""),
                         range=[0, 100], row=row, col=2)
        fig.update_xaxes(tickangle=-30, row=row, col=2)

        # Col 3 — spam classification stacked by signal.
        if int(n_txs[c]) > 0:
            sl = n_spam_label[c]
            n_non  = int(sl[txc.SPAM_CODE[txc.SPAM_NOT]]
                          + sl[txc.SPAM_CODE[txc.SPAM_UNKNOWN]])
            n_vol  = int(sl[txc.SPAM_CODE[txc.SPAM_VOL]])
            n_rev  = int(sl[txc.SPAM_CODE[txc.SPAM_REV]])
            n_both = int(sl[txc.SPAM_CODE[txc.SPAM_BOTH]])
            denom = max(n_non + n_vol + n_rev + n_both, 1)
            pct_non  = 100.0 * n_non  / denom
            pct_vol  = 100.0 * n_vol  / denom
            pct_rev  = 100.0 * n_rev  / denom
            pct_both = 100.0 * n_both / denom

            non_show = "Non-spam" not in legend_spam
            if non_show:
                legend_spam.add("Non-spam")
            fig.add_trace(go.Bar(
                x=["Non-spam"], y=[pct_non], name="Non-spam",
                marker_color=txc.SPAM_COLOR[txc.SPAM_NOT],
                legendgroup="spam_class",
                legendgrouptitle_text=("Spam classification"
                                       if non_show else None),
                showlegend=non_show,
                hovertemplate="Non-spam: %{y:.1f}%<extra></extra>",
            ), row=row, col=3)

            base = 0.0
            for seg_label, val, color in [
                ("High volume", pct_vol,  txc.SPAM_COLOR[txc.SPAM_VOL]),
                ("High revert", pct_rev,  txc.SPAM_COLOR[txc.SPAM_REV]),
                ("Both",        pct_both, txc.SPAM_COLOR[txc.SPAM_BOTH]),
            ]:
                seg_show = seg_label not in legend_spam
                if seg_show:
                    legend_spam.add(seg_label)
                fig.add_trace(go.Bar(
                    x=["Spam"], y=[val], base=base,
                    name=seg_label, marker_color=color,
                    legendgroup="spam_class",
                    showlegend=seg_show,
                    hovertemplate=(f"Spam, {seg_label}: "
                                    "%{y:.1f}%<extra></extra>"),
                ), row=row, col=3)
                base += val
        fig.update_yaxes(title_text=("% of cluster" if c == 0 else ""),
                         range=[0, 100], row=row, col=3)

    fig.update_xaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    fig.update_layout(
        template="plotly_white", autosize=True,
        height=280 * K + 120,
        barmode="overlay",
        margin=dict(l=80, r=240, t=60, b=60),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            groupclick="togglegroup",
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="rgba(0,0,0,0.20)", borderwidth=1,
            font=dict(size=10),
        ),
        font=dict(size=11, color="#222"),
        hovermode="x",
    )
    _CLUSTERING_FIG_CACHE["per_cluster"] = fig
    return fig


def _render_simple_table(headers: list[str], rows: list[list[str]],
                          *, num_classes: list[bool] | None = None) -> str:
    """Tiny HTML table used for dataframe-style snapshots.  Numeric cells
    get tabular figures + right alignment when `num_classes[col]` is True;
    the matching header gets the same class so header and body align in
    the same column."""
    th_cells = []
    for i, h in enumerate(headers):
        cls = "num" if num_classes and num_classes[i] else ""
        th_cells.append(f'<th class="{cls}">{h}</th>')
    th = "".join(th_cells)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            cls = ("num" if num_classes and num_classes[i] else "")
            cells.append(f'<td class="{cls}">{c}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="snapshot-table">'
        f'  <thead><tr>{th}</tr></thead>'
        f'  <tbody>{"".join(body)}</tbody>'
        '</table>'
    )


def _clustering_feature_snapshots() -> str:
    """Two side-by-side dataframe heads: raw per-tx gas (sample.parquet)
    and the CLR-transformed feature vectors (features.npz X_clr)."""
    cache = _load_clustering_cache()
    sample = cache["sample"].head(5)
    X_clr  = cache["X_clr"][:5]

    gas_cols = ["gas_c", "gas_sw", "gas_sr", "gas_sg",
                "gas_hg", "gas_l2", "gas_l1"]
    raw_rows = []
    for i, r in enumerate(sample.iter_rows(named=True), 1):
        cells = [f"tx {i}"]
        cells += [f"{int(r[c]):,}" for c in gas_cols]
        raw_rows.append(cells)
    raw_table = _render_simple_table(
        ["", "c", "sw", "sr", "sg", "hg", "l2", "l1"],
        raw_rows,
        num_classes=[False] + [True] * 7,
    )

    clr_rows = []
    for i in range(X_clr.shape[0]):
        cells = [f"tx {i+1}"]
        cells += [f"{X_clr[i, j]:+.2f}" for j in range(X_clr.shape[1])]
        clr_rows.append(cells)
    clr_table = _render_simple_table(
        ["", "c", "sw", "sr", "sg", "hg", "l2", "l1"],
        clr_rows,
        num_classes=[False] + [True] * 7,
    )

    return (
        '<div class="snapshot-row">'
        '  <div class="snapshot-block">'
        '    <div class="snapshot-title">Raw per-tx gas (5 rows)</div>'
        f'    {raw_table}'
        '  </div>'
        '  <div class="snapshot-block">'
        '    <div class="snapshot-title">CLR features (same 5 rows)</div>'
        f'    {clr_table}'
        '  </div>'
        '</div>'
    )


def _spam_data_preview() -> str:
    """Head of wallet_spam_classification.parquet — what the per-tx
    cluster aggregator joins on."""
    df = pl.read_parquet(SPAM_PARQUET).head(5).select([
        "address", "tx_count", "revert_count", "revert_ratio",
        "n_days_active", "n_days_spam", "frac_spam_days",
        "is_spam", "is_spam_ever",
    ])
    rows = []
    for r in df.iter_rows(named=True):
        addr = r["address"]
        addr_short = addr[:6] + "…" + addr[-4:] if len(addr) > 12 else addr
        rows.append([
            addr_short,
            f"{int(r['tx_count']):,}",
            f"{int(r['revert_count']):,}",
            f"{r['revert_ratio']:.4f}",
            str(int(r['n_days_active'])),
            str(int(r['n_days_spam'])),
            f"{r['frac_spam_days']:.3f}",
            "✓" if r['is_spam'] else "·",
            "✓" if r['is_spam_ever'] else "·",
        ])
    table = _render_simple_table(
        ["address", "tx_count", "rev_count", "rev_ratio",
         "n_days_active", "n_days_spam", "frac_spam",
         "is_spam", "is_spam_ever"],
        rows,
        num_classes=[False, True, True, True, True, True, True, False, False],
    )
    return (
        '<div class="snapshot-row">'
        '  <div class="snapshot-block snapshot-wide">'
        '    <div class="snapshot-title">'
        '      wallet_spam_classification.parquet (5 rows)'
        '    </div>'
        f'    {table}'
        '  </div>'
        '</div>'
    )


SPAM_PARQUET       = _ROOT / "data" / "wallet_spam_classification.parquet"
DAILY_SPAM_PARQUET = _ROOT / "data" / "daily_spam_breakdown.parquet"


def fig_daily_spam_share() -> go.Figure:
    """Daily share of transactions, stacked into 4 segments:
    non-spam, high-volume only, high-revert only, both.  Sourced
    from the daily breakdown parquet (per-signal SQL added in
    spam_insights.py)."""
    if "daily_spam_share" in _CLUSTERING_FIG_CACHE:
        return _CLUSTERING_FIG_CACHE["daily_spam_share"]

    df = (
        pl.read_parquet(DAILY_SPAM_PARQUET)
          .filter(pl.col("day") >= datetime(2025, 10, 1))
          .filter(pl.col("total_txs") > 0)
          .sort("day")
    )
    days     = df["day"].to_list()
    nonspam  = df["nonspammer_txs"].to_numpy().astype(np.float64)
    vol_only = df["spammer_txs_vol_only"].to_numpy().astype(np.float64)
    rev_only = df["spammer_txs_rev_only"].to_numpy().astype(np.float64)
    both     = df["spammer_txs_both"].to_numpy().astype(np.float64)
    total    = nonspam + vol_only + rev_only + both
    safe     = np.where(total > 0, total, 1.0)
    nonspam_pct  = nonspam  / safe * 100.0
    vol_pct      = vol_only / safe * 100.0
    rev_pct      = rev_only / safe * 100.0
    both_pct     = both     / safe * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=days, y=nonspam_pct, name="Non-spam",
        marker_color="#2ca02c",
        hovertemplate="%{x|%Y-%m-%d}<br>"
                       "non-spam = %{y:.1f}% of daily txs<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=days, y=vol_pct, name="High volume",
        marker_color="#ff7f0e",
        hovertemplate="%{x|%Y-%m-%d}<br>"
                       "high-vol = %{y:.1f}% of daily txs<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=days, y=rev_pct, name="High revert",
        marker_color="#d62728",
        hovertemplate="%{x|%Y-%m-%d}<br>"
                       "high-rev = %{y:.1f}% of daily txs<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=days, y=both_pct, name="Both",
        marker_color="#8c2d04",
        hovertemplate="%{x|%Y-%m-%d}<br>"
                       "both = %{y:.1f}% of daily txs<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", autosize=True, height=460,
        barmode="stack", bargap=0.0,
        margin=dict(l=70, r=30, t=20, b=40),
        font=dict(size=11, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    fig.update_yaxes(title_text="% of daily txs",
                     range=[0, 100], ticksuffix="%",
                     showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)",
                     mirror=True, ticks="outside")
    fig.update_xaxes(showline=True, linewidth=1.0,
                      linecolor="rgba(0,0,0,0.45)",
                      mirror=True, ticks="outside")
    _CLUSTERING_FIG_CACHE["daily_spam_share"] = fig
    return fig


def _spam_summary_stats_html() -> str:
    """Wallet-level stats card for the spam-flag slide.  Counts wallets
    flagged in each spam class (volume-only / revert-only / both) using
    the per-wallet rollup."""
    df = pl.read_parquet(SPAM_PARQUET)
    only_vol  = df.filter((pl.col("n_days_high_vol") > 0)
                           & (pl.col("n_days_high_rev") == 0)
                           & pl.col("is_spam_ever").cast(bool))
    only_rev  = df.filter((pl.col("n_days_high_rev") > 0)
                           & (pl.col("n_days_high_vol") == 0)
                           & pl.col("is_spam_ever").cast(bool))
    both      = df.filter((pl.col("n_days_high_vol") > 0)
                           & (pl.col("n_days_high_rev") > 0))
    not_spam  = df.filter(~pl.col("is_spam_ever").cast(bool))

    def _txs(slc: pl.DataFrame) -> int:
        return int(slc["tx_count"].sum())

    rows = [
        ("Volume only (top 0.1 %)", only_vol.height, _txs(only_vol),
         "#ff7f0e"),
        ("Revert only (≥30 % rev)", only_rev.height, _txs(only_rev),
         "#d62728"),
        ("Both signals",            both.height,     _txs(both),
         "#8c2d04"),
        ("Non-spam (any-day)",      not_spam.height, _txs(not_spam),
         "#2ca02c"),
    ]
    body_rows = []
    for label, n_w, n_tx, color in rows:
        body_rows.append(
            "<tr>"
            f'<td><span class="spam-pill" '
            f'style="background:{color}"></span>{label}</td>'
            f'<td class="num">{n_w:,}</td>'
            f'<td class="num">{n_tx:,}</td>'
            "</tr>"
        )
    return (
        '<div class="snapshot-row">'
        '  <div class="snapshot-block">'
        '    <div class="snapshot-title">'
        '      Spam-wallet rollup (full window)'
        '    </div>'
        '    <table class="snapshot-table snapshot-table--text-first">'
        '      <thead><tr>'
        '        <th>category</th>'
        '        <th class="num">wallets</th>'
        '        <th class="num">tx_count</th>'
        '      </tr></thead>'
        f'    <tbody>{"".join(body_rows)}</tbody>'
        '    </table>'
        '  </div>'
        '</div>'
    )


def clustering_slide_html() -> str:
    """Top-level <section> for the per-tx clustering result. Vertical
    sub-slides:
        1. Methodology / steps + feature snapshots.
        2. Spam-classification methodology + data head.
        3. t-SNE scatter, K = 5 clusters.
        4. K-selection diagnostics.
        5. Cluster characteristics (size, composition, spam mix).
    """
    print("Loading clustering cache (slide 9)...")
    t0 = time.time()
    f_tsne_k   = fig_clustering_tsne_and_k()
    f_overview = fig_clustering_overview()
    f_per_c    = fig_clustering_per_cluster()
    feat_snap  = _clustering_feature_snapshots()
    spam_snap  = _spam_data_preview()
    cluster_descriptions         = _cluster_descriptions_html()
    cluster_descriptions_compact = _cluster_descriptions_html(compact=True)
    print(f"  built clustering figures in {time.time()-t0:.2f}s")

    clr_eq = (
        r"\text{CLR}(x_k) = \ln \frac{x_k}{g(x)}"
        r", \qquad "
        r"g(x) = \left( x_1 \cdot x_2 \cdots x_D \right)^{1/D}"
    )

    intro_section = (
        '<section class="chart-stats-slide cluster-intro">'
        '  <h2>Per-tx clustering: workload regimes by gas mix</h2>'
        '  <div class="cluster-method">'
        '    <ol>'
        '      <li><b>Features.</b> Centred-log-ratio (CLR) of per-tx '
        '          gas usage across the 7 priced resources '
        '          (compute, sw, sr, sg, hg, l2, l1). For each tx we '
        '          divide every component by the <i>geometric mean</i> '
        '          of all components, then take the natural log. Two '
        '          txs with the same resource ratios land at the same '
        '          point regardless of total gas, so clusters describe '
        '          <i>what kind of work</i> a tx does, not how much.'
        f'         <div class="method-eq">\\[{clr_eq}\\]</div>'
        '      </li>'
        '      <li><b>KMeans (mini-batch).</b> Streaming partial-fit '
        '          over every priced tx in the multigas dataset '
        '          (~416 M txs).</li>'
        '      <li><b>K selection.</b> Target K ≈ number of priced '
        '          resources (one cluster per dominant resource), '
        '          and reject K values where two clusters end up '
        '          looking near-identical. Silhouette ↑ and WCSS '
        '          elbow ↓ are sanity checks; final pick is '
        '          <b>K = 5</b>.</li>'
        '      <li><b>Aggregate + visualise.</b> Per-cluster tx '
        '          count, volume-weighted resource centroid, '
        '          spam-label split, plus a 12 K-pt t-SNE map.</li>'
        '    </ol>'
        '  </div>'
        f' {feat_snap}'
        '</section>'
    )
    f_spam_share = fig_daily_spam_share()
    spam_stats   = _spam_summary_stats_html()
    dune_url = "https://dune.com/queries/5555110/9052858"
    spam_intro_section = (
        '<section class="chart-stats-slide cluster-intro spam-flag-slide">'
        '  <h2>Spam wallets flag</h2>'
        '  <div class="slide-note" '
        '       style="text-align:left">'
        '    Inspired by Entropy Advisors&rsquo; internal spam-flagging '
        f'   query on Dune: <a href="{dune_url}" target="_blank" '
        '    rel="noopener">'
        '    dune.com/queries/5555110</a>. '
        '    Percentile + revert rule, no separate clustering pass.'
        '  </div>'
        '  <div class="cluster-method">'
        '    <ol>'
        '      <li><b>Per (wallet, day)</b>: count txs and reverts.</li>'
        '      <li><b>Top 0.1 % percentile cutoff</b> on per-wallet '
        '          tx_count, recomputed each day.</li>'
        '      <li><b>Two flags per day</b>: '
        '          <code>high_vol</code> = '
        '          <code>n_tx &gt; top 0.1 % percentile</code>; '
        '          <code>high_rev</code> = '
        '          <code>revert_ratio &ge; 30 %</code> AND '
        '          <code>n_tx &ge; 50</code>. '
        '          <code>is_spam_day = high_vol OR high_rev</code>.</li>'
        '      <li><b>Wallet label</b>: '
        '          <code>is_spam = (n_spam_days / n_active_days &ge; 0.5)</code>.</li>'
        '    </ol>'
        '  </div>'
        f' {spam_stats}'
        f' <div class="hist-grid">{fig_div(f_spam_share, "fig-daily-spam-share")}</div>'
        '</section>'
    )

    # CSS knobs to keep the spam slide one-screen: shrink methodology
    # font + tighten paddings + lower chart height so the bar chart at
    # the bottom isn't clipped by Reveal.js's 1080 px viewport.
    tsne_section = (
        '<section class="chart-stats-slide">'
        '  <h2>t-SNE map and K-selection diagnostics</h2>'
        '  <div class="slide-note">'
        '    t-SNE is a non-linear dimensionality reduction from the '
        '    7-D CLR feature space to 2D, used purely to visualise '
        '    cluster separation. <b>For the slide we run t-SNE on a '
        f'   {TSNE_LARGE_N:,}-tx sub-sample</b> so it renders fast; '
        '    the per-cluster figures on the next slides are computed '
        '    on the <b>full 498 M-tx dataset</b>. Below the map, '
        '    silhouette and WCSS confirm the choice of K = 5.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_tsne_k, "fig-cluster-tsne-k")}</div>'
        '</section>'
    )
    centroid_overview_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Cluster characteristics: size and composition</h2>'
        '  <div class="slide-note">'
        '    Per-cluster size and volume-weighted resource '
        '    composition. Aggregated over all 498 M priced txs '
        '    in the dataset.'
        '  </div>'
        f' <div class="hist-grid">{fig_div(f_overview, "fig-cluster-overview")}</div>'
        f' {cluster_descriptions}'
        '</section>'
    )
    centroid_per_section = (
        '<section class="chart-stats-slide">'
        '  <h2>Per-cluster details: hourly share, median, spam</h2>'
        f' {cluster_descriptions_compact}'
        f' <div class="hist-grid">{fig_div(f_per_c, "fig-cluster-per")}</div>'
        '</section>'
    )
    return (
        '<section>\n'
        f'  {intro_section}\n'
        f'  {tsne_section}\n'
        f'  {spam_intro_section}\n'
        f'  {centroid_overview_section}\n'
        f'  {centroid_per_section}\n'
        '</section>'
    )


def demand_elasticity_slide_html() -> str:
    """Slide 10: short methodology preview for the demand-elasticity sim
    that uses the K=5 workload archetypes from the clustering pass."""
    def eq(latex: str) -> str:
        return f'<div class="method-eq">\\[{latex}\\]</div>'

    archetype_eq = (
        r"M[k, c] = \frac{\sum_{tx \in c} g_{tx,k}}{|c|}"
    )
    nnls_eq = (
        r"n(t) = \arg\min_{n \,\ge\, 0}\;"
        r"\bigl\| \, M \, n - g(t) \, \bigr\|_{2}^{2}"
    )
    elasticity_eq = (
        r"g'_{c}(t) = D_c \cdot \bar p'_{c}(t)^{-\alpha}"
        r", \quad \alpha = 1"
    )

    return (
        '<section class="chart-stats-slide capacity-intro elasticity-intro">'
        '  <h2>Demand elasticity model on workload clusters</h2>'
        '  <div class="wip-banner">'
        '    🚧 <b>Work in progress.</b> This section describes the '
        '    research direction we are exploring for the elastic-demand '
        '    counterfactual; the figures and numbers are not finalised.'
        '  </div>'
        '  <div class="capacity-definition">'
        '    <div class="who">Idea</div>'
        '    Use the K = 5 KMeans clusters as workload types '
        '    (compute-heavy, storage-write-heavy, ...). At every '
        '    hour, decompose the realised per-resource gas into a '
        '    non-negative number of txs per cluster; under '
        '    ArbOS 60 prices, each cluster reacts with constant-'
        '    elasticity demand and re-spends its budget across its '
        '    own resource mix. The result is a counterfactual '
        '    gas usage that <i>does</i> respond to price changes.'
        '  </div>'
        '  <div class="methodology">'
        '    <ol>'
        '      <li><b>Cluster centroid matrix</b> M (resources × '
        '          K clusters) from the cluster aggregator: each '
        '          column is a cluster’s average per-tx gas vector.'
        f'         {eq(archetype_eq)}'
        '      </li>'
        '      <li><b>Decompose</b> each hour’s realised resource '
        '          gas vector g(t) into a non-negative tx count '
        '          per cluster:'
        f'         {eq(nnls_eq)}'
        '      </li>'
        '      <li><b>Calibrate elasticity</b>: the per-cluster '
        '          budget D<sub>c</sub> is the historical product '
        '          of cluster gas and effective price.</li>'
        '      <li><b>Predict counterfactual</b> per-cluster gas '
        '          under ArbOS 60 prices, at unit elasticity '
        '          (α = 1):'
        f'         {eq(elasticity_eq)}'
        '      </li>'
        '      <li><b>Project back</b> to per-resource gas via M, '
        '          giving the predicted post-ArbOS-60 workload.</li>'
        '    </ol>'
        '  </div>'
        '</section>'
    )


def thank_you_slide_html() -> str:
    """Slide 11: closer."""
    return (
        '<section class="ty-slide">'
        '  <h1>Thank you</h1>'
        '  <h2>Questions?</h2>'
        '  <div class="logo-row">'
        '    <img src="assets/arbitrum.png"         alt="Arbitrum">'
        '    <img src="assets/offchain_labs.png"    alt="Offchain Labs">'
        '    <img src="assets/entropy_advisors.png" alt="Entropy Advisors">'
        '  </div>'
        '</section>'
    )


def _lighten_hex(hex_color: str, factor: float = 0.55) -> str:
    """Mix `hex_color` with white.  factor=0 → original, 1 → white."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    nr = int(r + (255 - r) * factor)
    ng = int(g + (255 - g) * factor)
    nb = int(b + (255 - b) * factor)
    return f"#{nr:02x}{ng:02x}{nb:02x}"


def _dia_cutoff_block() -> int:
    """Lowest block_number whose `block_time` is at or after ArbOS DIA
    activation.  Used to split the per-tx sample into pre/post regimes
    in the histogram grid on slide 5."""
    df = (
        pl.scan_parquet(
            _ROOT / "data" / "onchain_blocks_transactions" / "per_block.parquet"
        )
        .filter(pl.col("block_time") >= DIA_LAUNCH_TS)
        .select(pl.col("block_number").min().alias("b"))
        .collect()
    )
    return int(df["b"][0])


RESOURCE_SPEC = [
    ("Computation",    ("computation", "wasmComputation")),
    ("Storage Read",   ("storageAccessRead",)),
    ("Storage Write",  ("storageAccessWrite",)),
    ("Storage Growth", ("storageGrowth",)),
    ("History Growth", ("historyGrowth",)),
    ("L2 Calldata",    ("l2Calldata",)),
    ("L1 Calldata",    ("l1Calldata",)),
    ("Total per tx",   ("total",)),
]
RESOURCE_PALETTE = {
    "Computation":    "#1f77b4",
    "Storage Read":   "#98df8a",
    "Storage Write":  "#2ca02c",
    "Storage Growth": "#d62728",
    "History Growth": "#ff7f0e",
    "L2 Calldata":    "#9467bd",
    "L1 Calldata":    "#e377c2",
    "Total per tx":   "#555555",
}


def _build_full_histograms(dia_cutoff_block: int) -> dict:
    """Stream every per-tx parquet once; accumulate, per (resource, regime):
      • coarse-bin counts (HIST_N_DISP) for the on-screen histogram,
      • fine-bin counts (HIST_N_FINE) for percentile estimation,
      • exact gas-sum, zero count, total count.
    Returns a dict of arrays + scalars; saved to .npz by the caller."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["block",
            "computation", "wasmComputation",
            "storageAccessRead", "storageAccessWrite",
            "storageGrowth", "historyGrowth",
            "l2Calldata", "l1Calldata", "total"]
    edges_disp = np.linspace(0.0, HIST_LOG_HI, HIST_N_DISP + 1)
    edges_fine = np.linspace(0.0, HIST_LOG_HI, HIST_N_FINE + 1)

    resources = [name for name, _ in RESOURCE_SPEC]
    regimes   = ("pre", "post")
    cnt_disp  = {n: {r: np.zeros(HIST_N_DISP, dtype=np.int64) for r in regimes}
                 for n in resources}
    cnt_fine  = {n: {r: np.zeros(HIST_N_FINE, dtype=np.int64) for r in regimes}
                 for n in resources}
    sum_g     = {n: {r: 0.0 for r in regimes} for n in resources}
    n_zero    = {n: {r: 0   for r in regimes} for n in resources}
    n_total   = {n: {r: 0   for r in regimes} for n in resources}

    paths = sorted((_ROOT / "data" / "multigas_usage_extracts")
                   .glob("*/per_tx.parquet"))
    print(f"  streaming {len(paths)} per_tx parquets for full-data histograms")
    rows_seen = 0
    t0 = time.time()
    for path in paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=1_000_000, columns=cols):
            block = batch.column("block").to_numpy(zero_copy_only=False)
            mask_pre  = block <  dia_cutoff_block
            mask_post = ~mask_pre

            for name, src_cols in RESOURCE_SPEC:
                arr = batch.column(src_cols[0]).to_numpy(zero_copy_only=False).astype(np.float64)
                for extra in src_cols[1:]:
                    arr = arr + batch.column(extra).to_numpy(zero_copy_only=False).astype(np.float64)
                lg = np.log1p(arr)
                for regime, m in (("pre", mask_pre), ("post", mask_post)):
                    if not m.any():
                        continue
                    a, l = arr[m], lg[m]
                    cnt_disp[name][regime] += np.histogram(l, bins=edges_disp)[0]
                    cnt_fine[name][regime] += np.histogram(l, bins=edges_fine)[0]
                    sum_g [name][regime] += float(a.sum())
                    n_zero[name][regime] += int((a == 0).sum())
                    n_total[name][regime] += int(len(a))

            rows_seen += batch.num_rows
        elapsed = time.time() - t0
        print(f"    {path.parent.name}: {rows_seen:>11,} rows seen "
              f"({rows_seen/max(elapsed,1e-9)/1e6:.2f} M/s)")

    out: dict = {
        "edges_disp":  edges_disp,
        "edges_fine":  edges_fine,
        "dia_cutoff_block": np.int64(dia_cutoff_block),
    }
    for name in resources:
        for r in regimes:
            key = name.replace(" ", "_")
            out[f"disp_{key}_{r}"]  = cnt_disp[name][r]
            out[f"fine_{key}_{r}"]  = cnt_fine[name][r]
            out[f"sum_{key}_{r}"]   = np.float64(sum_g[name][r])
            out[f"nzero_{key}_{r}"] = np.int64(n_zero[name][r])
            out[f"ntot_{key}_{r}"]  = np.int64(n_total[name][r])
    return out


def load_or_build_full_histograms(dia_cutoff_block: int) -> dict:
    if TX_FULL_HIST_NPZ.exists():
        z = np.load(TX_FULL_HIST_NPZ)
        if int(z["dia_cutoff_block"]) == dia_cutoff_block:
            print(f"  loading cached full histograms: {TX_FULL_HIST_NPZ}")
            return {k: z[k] for k in z.files}
        print("  cached cutoff differs; rebuilding")
    print("  building full-dataset histograms (one streaming pass)")
    out = _build_full_histograms(dia_cutoff_block)
    TX_FULL_HIST_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TX_FULL_HIST_NPZ, **out)
    print(f"  cached → {TX_FULL_HIST_NPZ}")
    return out


def _percentile_from_hist(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Linear-interpolation quantile from a histogram in log1p(gas) space.
    Returns the value in original gas units (expm1 of the log estimate)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(counts)
    target = q * total
    idx = int(np.searchsorted(cum, target, side="left"))
    if idx >= len(counts):
        return float(np.expm1(edges[-1]))
    prev_cum = cum[idx - 1] if idx > 0 else 0
    bin_count = counts[idx]
    if bin_count == 0:
        return float(np.expm1(edges[idx]))
    frac = (target - prev_cum) / bin_count
    log_val = edges[idx] + frac * (edges[idx + 1] - edges[idx])
    return float(np.expm1(log_val))


def fig_per_resource_violins(sample: pl.DataFrame,
                              dia_cutoff_block: int) -> go.Figure:
    """Same 2x4 layout as the histogram grid, but each panel is a split
    violin (Pre-DIA on the left half, Post-DIA on the right) drawn from
    the cached 500 K-tx sample.  Each violin is per-trace down-sampled
    to keep the rendered HTML small while preserving KDE shape."""
    from plotly.subplots import make_subplots

    s = sample.with_columns(
        (pl.col("computation") + pl.col("wasmComputation")).alias("comp_total")
    )
    spec = [
        ("Computation",    "comp_total"),
        ("Storage Read",   "storageAccessRead"),
        ("Storage Write",  "storageAccessWrite"),
        ("Storage Growth", "storageGrowth"),
        ("History Growth", "historyGrowth"),
        ("L2 Calldata",    "l2Calldata"),
        ("L1 Calldata",    "l1Calldata"),
        ("Total per tx",   "total"),
    ]

    pre  = s.filter(pl.col("block") <  dia_cutoff_block)
    post = s.filter(pl.col("block") >= dia_cutoff_block)
    rng  = np.random.default_rng(42)
    n_per_violin = 8_000              # ≤ 8K points per trace keeps file small

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[name for name, _ in spec],
        vertical_spacing=0.18, horizontal_spacing=0.05,
    )
    tick_gas  = [0, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    tick_pos  = [float(np.log1p(v)) for v in tick_gas]
    tick_text = ["0", "100", "1K", "10K", "100K", "1M", "10M"]

    legend_added = {"pre": False, "post": False}
    for i, (name, col) in enumerate(spec):
        r, c = i // 4 + 1, i % 4 + 1
        color_post = RESOURCE_PALETTE[name]
        color_pre  = _lighten_hex(color_post, 0.55)

        log_pre  = np.log1p(pre[col].cast(pl.Float64).to_numpy())
        log_post = np.log1p(post[col].cast(pl.Float64).to_numpy())
        if len(log_pre) > n_per_violin:
            log_pre  = rng.choice(log_pre,  size=n_per_violin, replace=False)
        if len(log_post) > n_per_violin:
            log_post = rng.choice(log_post, size=n_per_violin, replace=False)

        fig.add_trace(go.Violin(
            y=log_pre, x=["dist"] * len(log_pre),
            name="Pre-DIA", side="negative",
            line_color=color_pre, fillcolor=color_pre, opacity=0.85,
            box_visible=False,
            # Render points beyond 1.5×IQR as small dots so the long
            # right-tail (whales / heavy txs) is visible.
            points="outliers",
            jitter=0.4, pointpos=-0.5,
            marker=dict(size=2.2, color=color_pre,
                        line=dict(width=0)),
            meanline_visible=True, meanline=dict(color=color_pre, width=1.2),
            scalemode="width", scalegroup=name,
            legendgroup="pre", showlegend=not legend_added["pre"],
            hoverinfo="skip",
        ), row=r, col=c)
        legend_added["pre"] = True

        fig.add_trace(go.Violin(
            y=log_post, x=["dist"] * len(log_post),
            name="Post-DIA", side="positive",
            line_color=color_post, fillcolor=color_post, opacity=0.7,
            box_visible=False,
            points="outliers",
            jitter=0.4, pointpos=0.5,
            marker=dict(size=2.2, color=color_post,
                        line=dict(width=0)),
            meanline_visible=True, meanline=dict(color=color_post, width=1.4),
            scalemode="width", scalegroup=name,
            legendgroup="post", showlegend=not legend_added["post"],
            hoverinfo="skip",
        ), row=r, col=c)
        legend_added["post"] = True

        fig.update_yaxes(
            tickvals=tick_pos, ticktext=tick_text,
            range=[0, HIST_LOG_HI],
            showline=True, linewidth=0.8,
            linecolor="rgba(0,0,0,0.35)", ticks="outside",
            row=r, col=c,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=40, b=20),
        font=dict(size=10, color="#222"),
        violinmode="overlay",
        violingap=0,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#333")
    return fig


def fig_per_resource_histograms(hist: dict) -> go.Figure:
    """2x4 grid of per-tx gas distributions, one per resource (+ tx total).
    Each panel overlays Pre-DIA and Post-DIA distributions in two shades
    of the resource's brand color, with dashed vertical lines marking
    each regime's median.  Counts come from the full-dataset histogram
    accumulator (594 M txs, one streaming pass).  Binning is on
    log1p(gas) so the heavy tail spans cleanly across magnitudes."""
    from plotly.subplots import make_subplots

    edges_disp = hist["edges_disp"]
    edges_fine = hist["edges_fine"]
    mids   = 0.5 * (edges_disp[:-1] + edges_disp[1:])
    width  = float(edges_disp[1] - edges_disp[0])

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[name for name, _ in RESOURCE_SPEC],
        vertical_spacing=0.18, horizontal_spacing=0.05,
    )
    tick_gas  = [0, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    tick_pos  = [float(np.log1p(v)) for v in tick_gas]
    tick_text = ["0", "100", "1K", "10K", "100K", "1M", "10M"]

    legend_added = {"pre": False, "post": False}
    for i, (name, _src) in enumerate(RESOURCE_SPEC):
        r, c = i // 4 + 1, i % 4 + 1
        key = name.replace(" ", "_")
        color_post = RESOURCE_PALETTE[name]
        color_pre  = _lighten_hex(color_post, 0.55)

        counts_pre  = hist[f"disp_{key}_pre"]
        counts_post = hist[f"disp_{key}_post"]
        fine_pre    = hist[f"fine_{key}_pre"]
        fine_post   = hist[f"fine_{key}_post"]

        fig.add_trace(go.Bar(
            x=mids, y=counts_pre, width=width,
            marker_color=color_pre, marker_line_width=0,
            opacity=0.85, name="Pre-DIA",
            legendgroup="pre", showlegend=not legend_added["pre"],
            hovertemplate=(
                f"{name} (pre-DIA)<br>"
                "ln(1+gas) ≈ %{x:.2f}<br>n = %{y:,}<extra></extra>"
            ),
        ), row=r, col=c)
        legend_added["pre"] = True

        fig.add_trace(go.Bar(
            x=mids, y=counts_post, width=width,
            marker_color=color_post, marker_line_width=0,
            opacity=0.75, name="Post-DIA",
            legendgroup="post", showlegend=not legend_added["post"],
            hovertemplate=(
                f"{name} (post-DIA)<br>"
                "ln(1+gas) ≈ %{x:.2f}<br>n = %{y:,}<extra></extra>"
            ),
        ), row=r, col=c)
        legend_added["post"] = True

        # Median lines on the log1p axis from the fine-bin histogram.
        # `_percentile_from_hist` returns gas units, but we plot on log1p.
        if fine_pre.sum():
            med_pre_log = float(np.log1p(_percentile_from_hist(fine_pre, edges_fine, 0.5)))
            fig.add_vline(x=med_pre_log, row=r, col=c,
                          line=dict(color=color_pre, width=1.4, dash="dot"))
        if fine_post.sum():
            med_post_log = float(np.log1p(_percentile_from_hist(fine_post, edges_fine, 0.5)))
            fig.add_vline(x=med_post_log, row=r, col=c,
                          line=dict(color=color_post, width=1.6, dash="dash"))
        fig.update_xaxes(
            tickvals=tick_pos, ticktext=tick_text,
            range=[0, float(np.log1p(2e7))],
            showline=True, linewidth=0.8,
            linecolor="rgba(0,0,0,0.35)", ticks="outside",
            row=r, col=c,
        )
        fig.update_yaxes(
            showline=True, linewidth=0.8,
            linecolor="rgba(0,0,0,0.35)", ticks="outside",
            row=r, col=c,
        )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=40, b=30),
        font=dict(size=10, color="#222"),
        hovermode="x",
        barmode="overlay",
        bargap=0.02,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )
    # Subplot titles default to size 16; trim them.
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#333")
    return fig


def tx_resource_stats_html(hist: dict) -> str:
    """Per-resource descriptive stats from the full-dataset histogram
    accumulator.  Mean and %zero are exact (streamed sums / counts);
    the percentile columns are linear-interpolated from the fine-bin
    histogram (500 bins on log1p, ≤ 0.5% relative error in practice).
    Mean and Median are split into Pre-DIA / Post-DIA / All-time."""
    edges_fine = hist["edges_fine"]
    rows: list[dict] = []
    n_grand_total = 0
    for name, _src in RESOURCE_SPEC:
        key = name.replace(" ", "_")
        fine_pre  = hist[f"fine_{key}_pre"]
        fine_post = hist[f"fine_{key}_post"]
        fine_all  = fine_pre + fine_post
        s_pre  = float(hist[f"sum_{key}_pre"])
        s_post = float(hist[f"sum_{key}_post"])
        n_pre  = int(hist[f"ntot_{key}_pre"])
        n_post = int(hist[f"ntot_{key}_post"])
        n_z    = int(hist[f"nzero_{key}_pre"]) + int(hist[f"nzero_{key}_post"])
        n_t    = n_pre + n_post

        z_pre  = int(hist[f"nzero_{key}_pre"])
        z_post = int(hist[f"nzero_{key}_post"])

        def q(fine, qv):
            return _percentile_from_hist(fine, edges_fine, qv)

        rows.append({
            "name":           name,
            "mean_pre":       s_pre / max(n_pre, 1),
            "mean_post":      s_post / max(n_post, 1),
            "mean_all":       (s_pre + s_post) / max(n_t, 1),
            "median_pre":     q(fine_pre,  0.50),
            "median_post":    q(fine_post, 0.50),
            "median_all":     q(fine_all,  0.50),
            "p25_pre":        q(fine_pre,  0.25),
            "p25_post":       q(fine_post, 0.25),
            "p25_all":        q(fine_all,  0.25),
            "p75_pre":        q(fine_pre,  0.75),
            "p75_post":       q(fine_post, 0.75),
            "p75_all":        q(fine_all,  0.75),
            "p99_pre":        q(fine_pre,  0.99),
            "p99_post":       q(fine_post, 0.99),
            "p99_all":        q(fine_all,  0.99),
            "pct_zero_pre":   100.0 * z_pre  / max(n_pre,  1),
            "pct_zero_post":  100.0 * z_post / max(n_post, 1),
            "pct_zero_all":   100.0 * n_z    / max(n_t,    1),
        })
        n_grand_total = max(n_grand_total, n_t)
    n = n_grand_total

    def fmt(v: float) -> str:
        if v >= 1_000_000:
            return f"{v/1e6:,.2f}M"
        if v >= 1_000:
            return f"{v/1e3:,.1f}K"
        return f"{v:,.0f}"

    def cell(v: float, extra: str = "") -> str:
        """Format a numeric cell.  Zero values get a `zero` class so the
        cell dims — heavy-tailed resources push 50%+ of txs to 0 and the
        clusters of bare `0` were hard to scan otherwise."""
        cls = " ".join(filter(None, ["num", extra, "zero" if v == 0 else ""]))
        return f'<td class="{cls}">{fmt(v)}</td>'

    # Two-row header: top groups Mean / Median into Pre / Post / All
    # tri-columns; P25 / P75 / P99 / %zero stay as single all-time
    # columns spanning both rows (rolled back from the wider variant).
    head = (
        '<thead>'
        '  <tr>'
        '    <th rowspan="2">Resource</th>'
        '    <th colspan="3" class="grp">Mean</th>'
        '    <th colspan="3" class="grp">Median</th>'
        '    <th rowspan="2" class="num">P25</th>'
        '    <th rowspan="2" class="num">P75</th>'
        '    <th rowspan="2" class="num">P99</th>'
        '    <th rowspan="2" class="num">% zero</th>'
        '  </tr>'
        '  <tr>'
        '    <th class="num sub">Pre</th>'
        '    <th class="num sub">Post</th>'
        '    <th class="num sub">All</th>'
        '    <th class="num sub">Pre</th>'
        '    <th class="num sub">Post</th>'
        '    <th class="num sub">All</th>'
        '  </tr>'
        '</thead>'
    )
    body = []
    for r in rows:
        emphasise = "row-total" if r["name"].startswith("Total") else ""
        body.append(
            f'<tr class="{emphasise}">'
            f'<td>{r["name"]}</td>'
            f'{cell(r["mean_pre"])}'
            f'{cell(r["mean_post"])}'
            f'{cell(r["mean_all"], extra="bold")}'
            f'{cell(r["median_pre"])}'
            f'{cell(r["median_post"])}'
            f'{cell(r["median_all"], extra="bold")}'
            f'{cell(r["p25_all"])}'
            f'{cell(r["p75_all"])}'
            f'{cell(r["p99_all"])}'
            f'<td class="num">{r["pct_zero_all"]:.1f}%</td>'
            f'</tr>'
        )
    table = (
        f'<table class="stats-table">{head}'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
    def _human(n: int) -> str:
        if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
        if n >= 1_000_000:     return f"{n/1e6:.0f}M"
        if n >= 1_000:         return f"{n/1e3:.0f}K"
        return f"{n:,}"

    caption = (
        f'<p class="stats-caption">Per-transaction gas distribution across '
        f"the 7 resources, computed over all <b>{_human(n)}</b> "
        f"transactions. Values in raw gas units.</p>"
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
                    xanchor="center", x=0.5, font=dict(size=11)),
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


def _total_wallets() -> int:
    """Distinct sender count over the analysis window. Uses the
    wallet_spam_classification.parquet rollup (one row per wallet that
    sent ≥ 1 tx in the window) when available — that's a metadata-only
    read, much cheaper than scanning per-tx senders."""
    import pyarrow.parquet as pq
    if SPAM_PARQUET.exists():
        return pq.ParquetFile(str(SPAM_PARQUET)).metadata.num_rows
    return 0


def _multigas_extract_max_month() -> str | None:
    """Latest YYYY-MM folder under data/multigas_usage_extracts that has
    a per_tx.parquet — used to flag when the per-tx coverage trails the
    per-block window.  Returns None if the directory is empty."""
    base = _ROOT / "data" / "multigas_usage_extracts"
    months = sorted(p.parent.name for p in base.glob("*/per_tx.parquet")
                    if len(p.parent.name) == 7)
    return months[-1] if months else None


def stat_html(blocks_wide: pl.DataFrame, blocks: pl.DataFrame) -> str:
    date_min = blocks_wide["block_date"].min()
    date_max = blocks_wide["block_date"].max()
    n_days   = (date_max - date_min).days + 1

    n_blocks_full = blocks_wide.height
    n_txs         = _total_txs()
    n_wallets     = _total_wallets()
    mg_month_max  = _multigas_extract_max_month()
    # Format the per-tx (multigas) coverage end as e.g. "2026-03-31".
    mg_window_note = ""
    if mg_month_max:
        from calendar import monthrange
        y, m = int(mg_month_max[:4]), int(mg_month_max[5:7])
        mg_end = f"{y:04d}-{m:02d}-{monthrange(y, m)[1]:02d}"
        if mg_end < str(date_max)[:10]:
            mg_window_note = (
                f' <span class="src-tag">per-tx extract: → {mg_end} '
                'so far, full April catch-up pending</span>'
            )

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
        ("Window",              f"{date_min} → {date_max}{mg_window_note}"),
        ("Days",                f"{n_days:,}"),
        ("Total blocks",        f"{n_blocks_full:,}"),
        ("Total transactions",  f"{n_txs:,}"),
        ("Active wallets",      f"{n_wallets:,}"),
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

  <!-- MathJax: renders the LaTeX equations on the ArbOS 60 slide.
       `$...$` is intentionally NOT a delimiter — Python identifiers like
       `inflow_i_per_t` would otherwise be parsed as TeX subscripts and
       throw "Double subscripts" warnings into the rendered slide.

       Strategy: skip the entire `reveal` container by default so plain
       text like "(computation, storage read/write/growth, ...)" can't
       be mis-parsed by MathJax's auto-detect, then re-enable only the
       specific equation containers (`method-eq`, `spec-ineq`, `set-ineq`)
       via `processHtmlClass`. -->
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\(', '\\)']] }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea',
                        'pre', 'code'],
        ignoreHtmlClass: 'tex2jax_ignore|reveal',
        processHtmlClass: 'tex2jax_process|method-eq|spec-ineq|set-ineq|methodology',
      }},
      svg: {{ fontCache: 'global' }},
    }};
  </script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

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

    /* Slide-scoped pixel heights, NOT viewport-scoped — reveal scales
       these correctly with the slide.  `vh` blows past slide bounds on
       tall monitors. */
    .plotly-frame {{ width: 100%; height: 850px; }}
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
    /* Distinct accent per source so the two providers read at a glance.
       Order: Arbitrum on-chain (Internal EA db) / Multi-gas extracts. */
    .source-list .source:nth-child(1) {{
      border-left-color: #08519c;
      background: rgba(8, 81, 156, 0.05);
    }}
    .source-list .source:nth-child(1) .src-tag {{
      color: #08519c; border-color: #b6cae6;
      background: rgba(8, 81, 156, 0.08);
    }}
    .source-list .source:nth-child(2) {{
      border-left-color: #ff7f0e;
      background: rgba(255, 127, 14, 0.05);
    }}
    .source-list .source:nth-child(2) .src-tag {{
      color: #b54a00; border-color: #f3c8a3;
      background: rgba(255, 127, 14, 0.10);
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
      line-height: 1.5; margin: 0.4em 0 0.6em;
    }}
    .hist-grid {{
      width: 100%; height: 580px;
      margin: 0.4em 0 0.8em;
    }}

    /* Slides that pair a chart with a caption + stats table beneath it.
       Use flexbox so the chart absorbs whatever height is left after
       the title block + caption + table.  Selector uses `.slides
       section.chart-stats-slide` (no `>`) so nested sections — i.e. the
       vertical-pair wrapper for slides 5a/5b — also match. */
    .reveal .slides section.chart-stats-slide,
    .reveal .slides section.chart-stats-slide.present {{
      display: flex !important;
      flex-direction: column !important;
      height: 100% !important;
      padding: 0 !important;
      box-sizing: border-box !important;
      overflow: hidden !important;
    }}
    .reveal .slides section.chart-stats-slide > h2,
    .reveal .slides section.chart-stats-slide > h3,
    .reveal .slides section.chart-stats-slide > .stats-caption,
    .reveal .slides section.chart-stats-slide > table.stats-table {{
      flex: 0 0 auto !important;
    }}
    .reveal .slides section.chart-stats-slide > .hist-grid {{
      flex: 1 1 0 !important;
      height: auto !important;
      min-height: 0 !important;
      margin: 0.3em 0 0.5em !important;
      position: relative !important;
      overflow: hidden !important;
    }}
    .reveal .slides section.chart-stats-slide > .hist-grid > div {{
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
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
    /* Group / sub-header styling for the two-row tx stats table. */
    table.stats-table th.grp {{
      text-align: center; color: #444; font-weight: 600;
      border-bottom: 1px solid #ccc;
    }}
    table.stats-table th.sub {{
      font-weight: 500; font-size: 0.92em; color: #777;
    }}
    table.stats-table td.bold {{ font-weight: 600; }}
    table.stats-table td.zero {{ color: #b0b0b0; font-weight: 400; }}

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
    /* Distinct accent per card so the four panels read at a glance.
       Order: window split / daily L2 fees / day type / top peak days. */
    .fee-stats-row .fee-card:nth-child(1) {{
      border-left-color: #08519c;
      background: rgba(8, 81, 156, 0.05);
    }}
    .fee-stats-row .fee-card:nth-child(2) {{
      border-left-color: #2ca02c;
      background: rgba(44, 160, 44, 0.05);
    }}
    .fee-stats-row .fee-card:nth-child(3) {{
      border-left-color: #9467bd;
      background: rgba(148, 103, 189, 0.05);
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

    /* Slide 6: method-block list pairing each Python signature with its
       spec equation rendered in LaTeX (MathJax). */
    .method-list  {{ margin-top: 0.6em; }}
    .method-block {{
      background: #f8f9fb; border-left: 3px solid #1f77b4;
      padding: 0.55em 1em; margin-bottom: 0.6em; border-radius: 2px;
    }}
    .method-block.class-header {{
      background: rgba(31, 119, 180, 0.07);
      border-left-color: #08519c;
    }}
    .method-sig {{
      font-family: ui-monospace, Menlo, Monaco, monospace;
      font-size: 0.75em; line-height: 1.5; color: #111;
    }}
    .method-sig .kw   {{ color: #a626a4; font-weight: 600; }}
    .method-sig .nm   {{ color: #1f77b4; font-weight: 700; }}
    .method-sig .args {{ color: #555; }}
    /* All MathJax-rendered blocks across slide 6 use the same font-size
       so SVG output scales coherently across the whole slide.  Includes
       `.spec-ineq` so the top-of-slide inequality matches the per-set
       ones (the wrapper isn't a `.method-eq`). */
    .method-eq,
    .set-ineq,
    .spec-ineq {{
      margin-top: 0.3em; padding-left: 0.4em;
      font-size: 0.65em; color: #111; line-height: 1.25;
    }}
    .spec-block .method-eq mjx-container[display="true"],
    .set-ineq mjx-container[display="true"] {{
      margin: 0.15em 0 !important;
    }}
    .class-doc {{
      font-size: 0.7em; color: #444; margin-top: 0.3em;
      line-height: 1.4;
    }}
    .class-doc .ix {{
      font-family: ui-monospace, Menlo, monospace;
      font-style: italic; color: #1f3a5f; font-weight: 600;
    }}

    /* Slide 6a (spec): p_min + inequality + per-set tables. */
    .spec-block  {{ margin-top: 0.4em; }}
    .const-pmin  {{
      font-size: 0.75em; color: #222; margin-bottom: 0.6em;
    }}
    .spec-ineq   {{
      margin: 0.4em 0 0.8em;
    }}
    .spec-ineq .spec-label {{
      font-size: 0.7em; color: #555; font-weight: 500;
      margin-bottom: 0.3em;
    }}
    .const-pmin b {{ color: #111; font-weight: 700; }}
    .const-note {{ color: #777; font-weight: 400;
                   margin-left: 0.25em; }}
    .const-sep  {{ color: #aaa; margin: 0 0.45em; }}

    /* Slide 8 — DIA ladder reminder at top of slide. */
    .ladder-row {{
      display: flex; align-items: center;
      justify-content: center; gap: 1.4em;
      margin: 0.3em auto 0.5em;
    }}
    table.ladder-table {{
      border-collapse: collapse; font-size: 0.6em;
      color: #111; margin: 0;
    }}
    table.ladder-table th, table.ladder-table td {{
      padding: 0.32em 0.85em;
      border-bottom: 1px solid #e0e0e0;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }}
    table.ladder-table thead th {{
      background: #f4f4f4;
      border-bottom: 2px solid #333;
      font-weight: 600; color: #111;
    }}
    table.ladder-table tbody th {{
      background: #fafafa;
      text-align: right;
      font-weight: 600; color: #111;
      border-right: 1px solid #d0d0d0;
    }}
    .ladder-caption {{
      font-size: 0.55em; color: #666; line-height: 1.45;
      max-width: 360px;
    }}

    /* Dataframe-style snapshot tables on the cluster + spam intro slides. */
    .snapshot-row {{
      display: flex; flex-direction: column; gap: 0.6em;
      align-items: center;
      max-width: 1300px; margin: 0.6em auto 0;
    }}
    .snapshot-block {{ width: 100%; max-width: 1100px; }}
    .snapshot-block.snapshot-wide {{ flex: 1 1 1100px; }}
    .snapshot-title {{
      font-size: 0.5em; font-weight: 600; color: #444;
      letter-spacing: 0.04em; text-transform: uppercase;
      margin: 0 0 0.3em 0.1em;
    }}
    table.snapshot-table {{
      border-collapse: collapse;
      font-size: 0.5em; color: #111;
      width: 100%;
      table-layout: fixed;
    }}
    table.snapshot-table th, table.snapshot-table td {{
      padding: 0.3em 0.6em;
      border-bottom: 1px solid #e0e0e0;
      text-align: left; font-weight: 400;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    table.snapshot-table thead th {{
      background: #f4f4f4; font-weight: 600;
      border-bottom: 2px solid #333; color: #111;
    }}
    table.snapshot-table td.num, table.snapshot-table th.num {{
      text-align: right; font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }}
    /* Coloured pill chip used in the spam summary stats card. */
    .spam-pill {{
      display: inline-block; width: 9px; height: 9px;
      margin-right: 7px; vertical-align: middle;
      border-radius: 2px;
    }}

    /* Cluster description card grid (size + composition slide). */
    .cluster-card-grid {{
      display: flex; flex-wrap: wrap; gap: 0.6em;
      justify-content: center; margin: 0.5em auto 0;
      max-width: 1300px;
    }}
    .cluster-card {{
      display: flex; align-items: center;
      flex: 1 1 240px; min-width: 240px; max-width: 280px;
      background: #fafafa; border: 1px solid #e0e0e0;
      border-radius: 4px; padding: 0.45em 0.6em;
    }}
    .cluster-card-tag {{
      display: inline-block;
      width: 2.6em; min-width: 2.6em;
      padding: 0.25em 0;
      margin-right: 0.7em;
      color: #fff; font-weight: 700;
      text-align: center; border-radius: 3px;
      font-size: 0.62em;
    }}
    .cluster-card-body {{
      flex: 1; line-height: 1.35; min-width: 0;
    }}
    .cluster-card-line1 {{
      font-size: 0.5em; font-weight: 600; color: #111;
    }}
    .cluster-card-line2 {{
      font-size: 0.45em; color: #555; margin-top: 1px;
      font-variant-numeric: tabular-nums;
    }}

    /* Compact variant: header strip on the per-cluster details slide. */
    .cluster-card-grid--compact {{
      gap: 0.4em; margin: 0.2em auto 0.4em;
    }}
    .cluster-card-grid--compact .cluster-card {{
      flex: 1 1 200px; min-width: 200px; max-width: 250px;
      padding: 0.3em 0.45em;
    }}
    .cluster-card-grid--compact .cluster-card-tag {{
      width: 2.0em; min-width: 2.0em;
      padding: 0.18em 0;
      margin-right: 0.5em;
      font-size: 0.5em;
    }}
    .cluster-card-grid--compact .cluster-card-line1 {{
      font-size: 0.4em;
    }}
    .cluster-card-grid--compact .cluster-card-line2 {{
      font-size: 0.36em;
    }}
    /* First column is the tx-row label ("tx 1"…); fix it narrow so the
       7 numeric columns get equal share. */
    table.snapshot-table th:first-child,
    table.snapshot-table td:first-child {{
      width: 4em; color: #777;
    }}
    /* Override for tables whose first column is a free-form text label
       (e.g. the spam summary card) — let the column auto-size and wrap. */
    table.snapshot-table--text-first {{
      table-layout: auto;
    }}
    table.snapshot-table--text-first th:first-child,
    table.snapshot-table--text-first td:first-child {{
      width: auto; min-width: 16em; color: #111;
      white-space: normal;
    }}

    /* Per-tx clustering methodology slide. */
    .cluster-intro .cluster-method {{
      background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
      padding: 0.7em 1.3em; max-width: 1100px;
      margin: 0.4em auto 0.5em;
      font-size: 0.6em; color: #333; line-height: 1.55;
    }}
    .cluster-intro .cluster-method ol {{
      padding-left: 1.4em; margin: 0.2em 0;
    }}
    .cluster-intro .cluster-method li {{ margin: 0.4em 0; }}
    .cluster-intro .cluster-method b {{ color: #08519c; }}
    .cluster-intro .cluster-method code {{
      background: rgba(0,0,0,0.05); padding: 1px 4px;
      border-radius: 2px; font-size: 0.95em;
    }}
    .cluster-intro .cluster-method ul.cluster-method-rules {{
      margin: 0.3em 0 0.3em 0.4em; padding-left: 1.1em;
      list-style: disc; color: #444;
    }}
    .cluster-intro .cluster-method ul.cluster-method-rules li {{
      margin: 0.18em 0;
    }}

    /* Spam-flag slide: compact everything so the methodology + spam-
       wallet rollup card + daily-share bar chart all fit the 1080 px
       viewport without the chart being clipped at the bottom. */
    .spam-flag-slide .slide-note {{
      padding: 0.4em 0.9em; margin: 0.25em auto;
      font-size: 0.55em;
    }}
    .spam-flag-slide .cluster-method {{
      padding: 0.5em 1.1em; margin: 0.25em auto;
      font-size: 0.55em;
    }}
    .spam-flag-slide .cluster-method ol {{ margin: 0.1em 0; }}
    .spam-flag-slide .cluster-method li {{ margin: 0.18em 0; }}
    .spam-flag-slide .snapshot-row {{
      margin: 0.2em auto 0;
    }}
    .spam-flag-slide table.snapshot-table {{
      font-size: 0.5em;
    }}
    .spam-flag-slide table.snapshot-table th,
    .spam-flag-slide table.snapshot-table td {{
      padding: 0.22em 0.55em;
    }}
    .spam-flag-slide .snapshot-title {{
      margin: 0.25em auto 0.1em !important;
    }}
    /* Display equations inside cluster-method need a fudge factor so
       MathJax doesn't inherit the small list font size. */
    .cluster-intro .cluster-method .method-eq {{
      margin: 0.5em 0 0.3em;
    }}
    .cluster-intro .cluster-method .method-eq mjx-container[display="true"] {{
      font-size: 1.45em !important;
      margin: 0.2em 0 !important;
    }}

    /* Thank-you closer slide — mirrors cover styling. */
    .reveal .slides section.ty-slide {{
      text-align: center;
    }}
    .reveal .slides section.ty-slide h1 {{
      font-size: 2.6em; margin: 0.2em 0 0.1em; color: #08519c;
      letter-spacing: 0.02em;
    }}
    .reveal .slides section.ty-slide h2 {{
      font-size: 1.2em; color: #555; font-weight: 400;
      margin: 0.2em 0 1.2em;
    }}
    .reveal .slides section.ty-slide .logo-row {{
      display: flex; justify-content: center; align-items: center;
      gap: 3.5em; margin-top: 1.5em;
    }}
    .reveal .slides section.ty-slide .logo-row img {{
      max-height: 80px; max-width: 240px; object-fit: contain;
    }}

    /* Capacity-headroom methodology slide. */
    .capacity-intro .capacity-definition {{
      background: rgba(31, 119, 180, 0.06);
      border-left: 3px solid #1f77b4;
      padding: 0.7em 1.1em; max-width: 1100px;
      margin: 0.4em auto 0.6em;
      font-size: 0.62em; color: #333; line-height: 1.55;
    }}
    .capacity-intro .capacity-definition .who {{
      font-weight: 600; color: #08519c;
      font-size: 0.92em; margin-bottom: 0.3em;
    }}
    .capacity-intro .methodology {{
      background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
      padding: 0.8em 1.4em; max-width: 1100px;
      margin: 0.4em auto 0.5em;
      font-size: 0.62em; color: #333; line-height: 1.55;
    }}
    .capacity-intro .methodology ol {{
      padding-left: 1.4em; margin: 0.2em 0;
    }}
    .capacity-intro .methodology li {{ margin: 0.5em 0; }}
    .capacity-intro .methodology code {{
      background: rgba(0,0,0,0.05); padding: 1px 4px;
      border-radius: 2px; font-size: 0.95em;
    }}
    /* Display equations inside the methodology block: scale to match
       the surrounding text and avoid the giant default MathJax size. */
    .capacity-intro .methodology .method-eq {{
      margin: 0.4em 0 0.2em;
    }}
    .capacity-intro .methodology .method-eq mjx-container[display="true"] {{
      font-size: 1.6em !important;
      margin: 0.3em 0 !important;
    }}
    /* Demand-elasticity slide: same body / equation sizing as the
       capacity intro slide.  Reset rules from earlier attempts that
       shrank the card. */
    .elasticity-intro .methodology {{
      font-size: 0.62em;
    }}
    .elasticity-intro .methodology li {{ margin: 0.45em 0; }}
    .elasticity-intro .methodology .method-eq {{
      margin: 0.4em 0 0.3em;
      text-align: center;
    }}
    .elasticity-intro .methodology .method-eq mjx-container[display="true"] {{
      font-size: 1.5em !important;
      margin: 0.2em 0 !important;
    }}
    .elasticity-intro .capacity-definition {{
      font-size: 0.62em;
      padding: 0.7em 1.1em;
      margin: 0.3em auto 0.5em;
    }}
    /* Under-construction banner on the demand-elasticity slide. */
    .elasticity-intro .wip-banner {{
      max-width: 1100px; margin: 0.4em auto;
      padding: 0.55em 1em;
      font-size: 0.62em; color: #6a4f00; line-height: 1.5;
      background: rgba(251, 192, 45, 0.18);
      border-left: 3px solid #fbc02d;
      border-radius: 3px;
    }}
    .elasticity-intro .wip-banner b {{ color: #6a4f00; }}

    /* Short data + interpretation note above/below figures. */
    .slide-note {{
      max-width: 1200px; margin: 0.4em auto 0.6em;
      padding: 0.6em 1em;
      font-size: 0.62em; color: #333; line-height: 1.55;
      background: rgba(31, 119, 180, 0.05);
      border-left: 3px solid #1f77b4;
      border-radius: 2px;
    }}
    .slide-note b {{ color: #08519c; font-weight: 600; }}

    /* Stacked summary tables on the revenue summary slide. */
    .revenue-tables {{
      display: flex; flex-direction: column; gap: 1.2em;
      align-items: center; margin: 0.5em auto 0;
      max-width: 1300px;
    }}
    .revenue-tables .rev-table-block {{
      width: 100%; max-width: 1200px;
    }}
    .revenue-tables .rev-table-block h3 {{
      font-size: 0.6em; color: #444; font-weight: 600;
      margin: 0 0 0.4em 0; text-align: center;
    }}
    .revenue-tables table {{
      font-size: 0.62em !important;
      margin: 0 auto !important;
    }}
    /* Reveal.js applies row-striping to .reveal table tr:nth-child(even),
       which clobbers per-cell heatmap backgrounds.  Strip it on these
       tables so the inline `background: ... !important` heat colours
       show up. */
    .reveal .slides .revenue-tables table tr {{
      background: transparent !important;
    }}
    .reveal .slides .revenue-tables table th,
    .reveal .slides .revenue-tables table td {{
      background-clip: padding-box;
    }}

    /* Revenue-comparison intro slide: two side-by-side cards. */
    .intro-revenue {{
      display: flex; flex-direction: column;
    }}
    .intro-revenue h2 {{ margin-bottom: 0.5em; }}
    .intro-revenue .intro-card {{
      max-width: 1100px; margin: 0.4em auto;
      padding: 0.8em 1.2em;
      background: #fafafa;
      border: 1px solid #e0e0e0; border-left: 3px solid #1f77b4;
      border-radius: 4px;
      font-size: 0.62em; color: #333; line-height: 1.55;
    }}
    .intro-revenue .intro-card-tag {{
      font-weight: 600; color: #08519c;
      font-size: 0.92em; margin-bottom: 0.4em;
      text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .intro-revenue .intro-card p {{
      margin: 0 0 0.6em 0;
    }}
    .intro-revenue .intro-card p:last-child {{
      margin-bottom: 0;
    }}
    .intro-revenue .intro-points {{
      margin: 0.2em 0 0 1.1em; padding-left: 0;
    }}
    .intro-revenue .intro-points li {{
      margin: 0.3em 0;
    }}
    .intro-revenue .intro-points b {{ color: #08519c; }}

    /* Slide 8 — short Taylor-4 explainer above the comparison chart. */
    .taylor-note {{
      font-size: 0.6em; color: #444; line-height: 1.5;
      max-width: 1200px; margin: 0.3em 0 0.5em;
      padding: 0.55em 0.85em;
      background: rgba(214, 39, 40, 0.04);
      border-left: 3px solid #d62728;
      border-radius: 2px;
    }}
    .taylor-note code {{
      background: rgba(255,255,255,0.7);
      padding: 1px 4px; border-radius: 2px;
      font-size: 0.95em;
    }}
    .taylor-note b {{ color: #d62728; }}

    /* Stats table at the foot of slide 8. */
    .taylor-stats {{ margin-top: 0.5em; }}
    .taylor-stats-caption {{
      font-size: 0.55em; color: #666; margin-bottom: 0.3em;
      line-height: 1.4;
    }}
    .taylor-stats .hint {{
      font-weight: 400; color: #888;
      text-transform: none; letter-spacing: 0;
    }}

    /* Resource-symbol notation block on slide 7a. */
    .notation-block {{
      display: flex; flex-wrap: wrap;
      gap: 0.4em 1.4em;
      font-size: 0.6em; color: #444;
      background: #f6f7f9; border: 1px solid #e1e4e8;
      border-radius: 3px;
      padding: 0.45em 0.85em;
      margin: 0.4em 0 0.6em;
    }}
    .notation-block .notation-label {{
      font-weight: 600; color: #333;
      text-transform: uppercase; letter-spacing: 0.05em;
      font-size: 0.85em;
    }}
    .notation-block .sym {{
      font-family: ui-monospace, Menlo, Monaco, monospace;
      color: #1f3a5f; font-weight: 600;
      margin-right: 0.3em;
    }}
    .notation-block .notation-note {{
      color: #888; font-style: italic;
    }}

    /* Slide 6b footer: GitHub icon + short text link. */
    .source-link {{
      margin-top: 1em; font-size: 0.7em;
      padding-top: 0.6em; border-top: 1px solid #e0e0e0;
    }}
    .source-link a {{
      display: inline-flex; align-items: center; gap: 0.4em;
      color: #24292f; text-decoration: none; font-weight: 500;
    }}
    .source-link a:hover {{ color: #0969da; }}
    .source-link a:hover .gh-icon {{ color: #0969da; }}
    .gh-icon {{ color: #24292f; vertical-align: middle; }}
    .set-card {{
      margin-top: 0.6em;
      border-left: 3px solid #888;
      padding: 0.4em 0.7em 0.5em 0.8em;
      border-radius: 2px;
    }}
    /* Distinct accent colors so the two configs read at a glance. */
    .set-card-1 {{
      border-left-color: #1f77b4;
      background: rgba(31, 119, 180, 0.05);
    }}
    .set-card-1 .set-title {{ color: #08519c; }}
    .set-card-2 {{
      border-left-color: #ff7f0e;
      background: rgba(255, 127, 14, 0.06);
    }}
    .set-card-2 .set-title {{ color: #b54a00; }}
    .set-title {{
      font-size: 0.75em; font-weight: 700; color: #111;
      margin-bottom: 0.3em; margin-top: 0.6em;
    }}
    .ladder-row {{
      display: flex; align-items: flex-start;
      gap: 1.5em; flex-wrap: wrap;
      font-variant-numeric: tabular-nums;
      margin-top: 0.3em;
    }}
    .ladder-table {{ flex: 0 0 auto; }}
    .ladder-title {{
      font-size: 0.7em; font-weight: 600; color: #111;
      margin-bottom: 0.2em;
    }}
    .ladder-note {{ font-weight: 400; color: #666; font-style: italic; }}
    table.set-table {{
      border-collapse: collapse; font-size: 0.65em;
      font-variant-numeric: tabular-nums;
    }}
    table.set-table th, table.set-table td {{
      padding: 0.18em 0.55em;
      border-bottom: 1px solid #e5e5e5;
    }}
    table.set-table th {{
      color: #666; font-weight: 500; border-bottom: 1.5px solid #888;
      text-align: right;
    }}
    table.set-table th.lj, table.set-table td.lj {{
      text-align: center; color: #555;
    }}
    table.set-table td {{
      text-align: right; color: #111;
    }}
    table.set-table td.lA {{ color: #444; }}

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
      display: inline-flex; align-items: baseline; gap: 0.4em;
      font-size: 0.55em;
      padding: 0.18em 0.6em; border-radius: 3px;
      border: 1px solid;
    }}
    .fee-pill .pill-main {{ font-weight: 600; }}
    .fee-pill .pill-n    {{ font-weight: 700; }}
    .fee-pill .pill-name {{ font-weight: 500; margin-left: 0.25em; }}
    .fee-pill .pill-pct  {{
      font-weight: 500; opacity: 0.7;
      padding-left: 0.45em; border-left: 1px solid currentColor;
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

    .reveal section.cover {{ padding-top: 40px; position: relative; }}
    .reveal section.cover h1 {{ font-size: 2.4em; margin-bottom: 0.1em; }}
    .reveal section.cover h2 {{ font-size: 1.5em; color: #555;
                                  font-weight: 400; }}
    .reveal section.cover .logo-row {{ margin-top: 60px; gap: 4em; }}
    .reveal section.cover .logo-row img {{ height: 80px; }}

    .internal-badge {{
      position: absolute; top: 16px; right: 0;
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
      <h3>Following ArbOS DIA, fees and congestion both declined</h3>
      <div class="plotly-frame">{FIG_FEES}</div>
      {FEE_STATS}
    </section>

    <!-- Slide 4: hourly resource (absolute + share, stacked) -->
    <section class="chart-slide">
      <h2>Hourly gas consumption by resource</h2>
      <h3>Top: Mgas/hr by resource. Bottom: % share by resource</h3>
      <div class="plotly-frame">{FIG1}</div>
    </section>

    <!-- Slide 5: per-tx gas distribution (vertical pair) -->
    <section>
      <section class="chart-stats-slide">
        <h2>Per-transaction gas distribution</h2>
        <div class="hist-grid">{HIST_FIG}</div>
        {STATS_TABLE}
      </section>
      <section class="chart-stats-slide">
        <h2>Per-transaction gas distribution</h2>
        <div class="hist-grid">{VIOLIN_FIG}</div>
        {STATS_TABLE_2}
      </section>
    </section>

    <!-- Slide 6: ArbOS 60 + ArbOS 51 implementations (vertical group) -->
    {SLIDE6}

    <!-- Slide 7: ArbOS 60 vs 51 revenue (no demand elasticity) -->
    {SLIDE7}

    <!-- Slide 8: ArbOS 60 capacity headroom -->
    {SLIDE8}

    <!-- Slide 9: per-tx clustering (CLR + KMeans) -->
    {SLIDE9}

    <!-- Slide 10: demand elasticity (archetype-based) -->
    {SLIDE10}

    <!-- Slide 11: thank you / closer -->
    {SLIDE11}

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
    height: 1080,
    margin: 0.04,
    transition: 'fade',
  }});

  // Reveal hides slides until they are active; Plotly measures 0×0 on
  // initial render and needs a resize once a slide becomes visible. The
  // resize must run after Reveal has finished its CSS transform AND
  // Plotly has had a chance to lay out (its own internal layout is
  // async). One delayed call doesn't always cover both — schedule a
  // burst of retries at 0/80/200/450/900 ms so even a slow first render
  // catches up. Each Plotly.Plots.resize is idempotent and cheap when
  // already at the right size, so the burst is safe.
  const RESIZE_DELAYS_MS = [0, 80, 200, 450, 900];
  function resizeVisiblePlots() {{
    document.querySelectorAll('.plotly-graph-div').forEach(el => {{
      if (el.offsetParent === null) return;          // not on screen
      const r = el.getBoundingClientRect();
      if (r.width < 8 || r.height < 8) return;       // not laid out yet
      try {{ Plotly.Plots.resize(el); }} catch (e) {{}}
    }});
  }}
  function scheduleResizeBurst() {{
    RESIZE_DELAYS_MS.forEach(d => setTimeout(resizeVisiblePlots, d));
  }}
  // Reveal fires `slidechanged` once the new slide is current — but the
  // CSS slide-transition is still running.  `slidetransitionend` fires
  // when it lands, which is when the plot containers actually have
  // their final size.  Listening to both covers fast-jump (no
  // transition) AND animated transitions.
  Reveal.on('ready slidechanged slidetransitionend', () => {{
    scheduleResizeBurst();
    if (window.MathJax && window.MathJax.typesetPromise) {{
      window.MathJax.typesetPromise();
    }}
  }});
  window.addEventListener('resize', scheduleResizeBurst);
  // Final safety net: any plot that ends up with the wrong size after
  // the burst will be picked up the moment its container size changes
  // (eg. when a sibling vertical sub-slide finishes its transform).
  if (typeof ResizeObserver !== 'undefined') {{
    const ro = new ResizeObserver(entries => {{
      for (const e of entries) {{
        if (e.target.offsetParent === null) continue;
        const r = e.contentRect;
        if (r.width >= 8 && r.height >= 8) {{
          try {{ Plotly.Plots.resize(e.target); }} catch (err) {{}}
        }}
      }}
    }});
    document.querySelectorAll('.plotly-graph-div').forEach(el => ro.observe(el));
  }}
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

    print("Loading full-dataset per-tx histograms (slide 5)...")
    cutoff_block = _dia_cutoff_block()
    print(f"  DIA cutoff block: {cutoff_block:,}")
    full_hist = load_or_build_full_histograms(cutoff_block)
    stats_table = tx_resource_stats_html(full_hist)
    f_hist = fig_per_resource_histograms(full_hist)

    print("Loading 500K-tx sample for slide 6 violins...")
    tx_sample = load_or_build_tx_sample()
    f_violin = fig_per_resource_violins(tx_sample, cutoff_block)

    print("Rendering deck...")
    page = PAGE_TEMPLATE.format(
        REVEAL_CDN=REVEAL_CDN,
        PLOTLY_CDN=PLOTLY_CDN,
        STATS=stat_html(blocks_wide, blocks),
        FIG_FEES=fig_div(f_fees, "fig-l2-fees"),
        FEE_STATS=l2_fee_stats_html(daily_fees),
        FIG1=fig_div(f1, "fig-hourly-combined"),
        HIST_FIG=fig_div(f_hist, "fig-tx-resource-hist"),
        VIOLIN_FIG=fig_div(f_violin, "fig-tx-resource-violin"),
        STATS_TABLE=stats_table,
        # Plotly auto-deduplicates element IDs across the page; the table
        # is structured HTML (not a Plotly fig) so embedding it twice is
        # safe — both copies render independently.
        STATS_TABLE_2=stats_table,
        SLIDE6=arbos60_code_slide_html(),
        SLIDE7=revenue_no_elasticity_slide_html(),
        SLIDE8=capacity_slide_html(),
        SLIDE9=clustering_slide_html(),
        SLIDE10=demand_elasticity_slide_html(),
        SLIDE11=thank_you_slide_html(),
    )
    OUT_HTML.write_text(page)
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
