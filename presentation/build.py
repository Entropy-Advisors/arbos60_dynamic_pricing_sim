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
        hovermode="x unified",
        font=dict(size=12, color="#222"),
        autosize=True,
        yaxis=dict(title=ytitle, showline=True, linewidth=1.0,
                   linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside"),
        xaxis=dict(showline=True, linewidth=1.0,
                   linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside"),
    )


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
        hovermode="x unified",
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
    .stat-grid .val   {{ font-family: ui-monospace, Menlo, monospace;
                         color: #111; font-weight: 500; }}

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

    <!-- Slide 3: hourly resource — absolute + share, stacked -->
    <section class="chart-slide">
      <h2>Hourly priced gas: absolute and composition</h2>
      <h3>Top: Mgas/hr by resource. Bottom: same data, normalised to 100 %.</h3>
      <div class="plotly-frame">{FIG1}</div>
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
    f1 = fig_hourly_combined(rk_hr)

    print("Rendering deck...")
    page = PAGE_TEMPLATE.format(
        REVEAL_CDN=REVEAL_CDN,
        PLOTLY_CDN=PLOTLY_CDN,
        STATS=stat_html(blocks_wide, blocks),
        FIG1=fig_div(f1, "fig-hourly-combined"),
    )
    OUT_HTML.write_text(page)
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
