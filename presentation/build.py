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
            "Set 2 (alt)",
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
    spec_block = (
        '<div class="spec-block">'
        '  <div class="const-pmin">'
        '    p<sub>min</sub> = <b>0.02 gwei</b>'
        '    <span class="const-note">post-DIA</span>'
        '    <span class="const-sep">·</span>'
        '    <b>0.01 gwei</b>'
        '    <span class="const-note">pre-DIA</span>'
        '  </div>'
        '  <div class="spec-ineq">'
        '    <div class="spec-label">'
        '      Constraint inequality (for each set i, constraint j):'
        '    </div>'
        f'   \\[{inequality_eq}\\]'
        '  </div>'
        f'  {"".join(set_blocks)}'
        '</div>'
    )

    github_url = (
        "https://github.com/Entropy-Advisors/arbos60_dynamic_pricing_sim"
        "/blob/main/scripts/arbos60.py"
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
    source_link = (
        '<div class="source-link">'
        f'  <a href="{github_url}" target="_blank" rel="noopener">'
        f'    {github_icon}'
        '    <span>scripts/arbos60.py on GitHub</span>'
        '  </a>'
        '</div>'
    )
    # Two nested <section>s = a vertical-slide pair so each half breathes.
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
        processHtmlClass: 'tex2jax_process|method-eq|spec-ineq|set-ineq',
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

    <!-- Slide 6: ArbOS 60 implementation -->
    {SLIDE6}

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
    // Re-typeset MathJax for the active slide so equations show even if
    // the slide was hidden when MathJax first loaded.
    if (window.MathJax && window.MathJax.typesetPromise) {{
      window.MathJax.typesetPromise();
    }}
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
    )
    OUT_HTML.write_text(page)
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
