"""
Spam-classification insights dashboard.

Sections in the output HTML:
  1. Description + headline stats (totals, category split, distributions).
  2. Methodology — how each wallet got flagged (text).
  3. Daily reverts + total txs (line chart).
  4. Daily count of spammers vs non-spammers (stacked bar).

Inputs:
  • data/wallet_spam_classification.parquet  (per-wallet rollup)
  • data/daily_spam_breakdown.parquet         (daily, fetched from CH on
                                               first run via the inline
                                               query in fetch_daily(...))

Output: figures/spam_insights.html

Run:
    python scripts/spam_insights.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent

SPAM_PARQUET   = _ROOT / "data" / "wallet_spam_classification.parquet"
DAILY_PARQUET  = _ROOT / "data" / "daily_spam_breakdown.parquet"
OUT_HTML       = _ROOT / "figures" / "spam_insights.html"

ENV_CANDIDATES = [
    pathlib.Path(p) for p in [
        os.environ.get("ARBOS_ENV_PATH"),
        _ROOT / ".env",
        pathlib.Path.home() / ".config" / "arbos60" / ".env",
    ] if p
]

START_DATE = "2025-10-01"
END_DATE   = "2026-03-01"

COL_TOTAL    = "#1f77b4"
COL_REVERT   = "#d62728"
COL_SPAMMER  = "#d62728"
COL_NORMAL   = "#2ca02c"


# ── CH client ──────────────────────────────────────────────────────────────
def _ch_client():
    from dotenv import dotenv_values
    import clickhouse_connect
    needed = ("CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD")
    for path in ENV_CANDIDATES:
        if not path.exists():
            continue
        cfg = dotenv_values(path)
        if all(cfg.get(k) for k in needed):
            return clickhouse_connect.get_client(
                host=cfg["CLICKHOUSE_HOST"], user=cfg["CLICKHOUSE_USER"],
                password=cfg["CLICKHOUSE_PASSWORD"],
                port=int(cfg.get("CLICKHOUSE_PORT", 8443)),
                secure=str(cfg.get("CLICKHOUSE_SECURE", "true")).lower() == "true",
                settings={"max_execution_time": 1800},
            )
    sys.exit(f"No usable .env with CH credentials in {ENV_CANDIDATES}")


# ── Daily-breakdown fetcher (cached) ────────────────────────────────────────
# Spam-flagging done in-CH using the same per-day percentile logic as
# arbitrum_wallet_spam_classification.sql.  Avoids the 256 KB max_query_size
# limit that we'd hit if we inlined the ~7K spammer addresses in an IN()
# clause.  Daily totals are saved alongside the wallet parquet so the
# dashboard never re-queries CH after the first run.
DAILY_SQL = """
WITH
  block_window AS (
    SELECT
      toUInt64({block_min:UInt64}) AS bmin,
      toUInt64({block_max:UInt64}) AS bmax
  ),
  success_dw AS (
    SELECT block_date AS day, "from" AS address, COUNT(*) AS success_count
    FROM raw_arbitrum.v_transactions, block_window
    WHERE block_date >= toDate({start:String})
      AND block_date <  toDate({end:String})
      AND block_number >= bmin AND block_number <= bmax
    GROUP BY day, "from"
  ),
  failed_dw AS (
    SELECT toDate(DATETIME) AS day, FROM_ADDRESS AS address,
           COUNT(*) AS revert_count
    FROM raw_arbitrum.transactions_failed, block_window
    WHERE toDate(DATETIME) >= toDate({start:String})
      AND toDate(DATETIME) <  toDate({end:String})
      AND BLOCK_NUMBER >= bmin AND BLOCK_NUMBER <= bmax
    GROUP BY day, FROM_ADDRESS
  ),
  per_day_wallet AS (
    SELECT
      if(s.day IS NULL, f.day, s.day)                                AS day,
      if(s.address = '', f.address, s.address)                      AS address,
      coalesce(s.success_count, 0)                                  AS success_count,
      coalesce(f.revert_count, 0)                                   AS revert_count,
      coalesce(s.success_count, 0) + coalesce(f.revert_count, 0)    AS tx_count
    FROM success_dw s
    FULL OUTER JOIN failed_dw f ON s.day = f.day AND s.address = f.address
  ),
  daily_buckets AS (
    SELECT
      day,
      quantileExact(0.999)(tx_count) AS p001
    FROM per_day_wallet
    GROUP BY day
  ),
  flagged AS (
    SELECT
      pw.day                                    AS day,
      pw.address                                AS address,
      pw.tx_count                               AS tx_count,
      pw.revert_count                           AS revert_count,
      (pw.tx_count > b.p001
       OR (pw.tx_count >= toUInt64({revert_min_txs:UInt64})
           AND pw.revert_count / toFloat64(pw.tx_count)
               >= toFloat64({revert_ratio:Float64})))   AS is_spam_day
    FROM per_day_wallet pw
    INNER JOIN daily_buckets b ON pw.day = b.day
  )
SELECT
  day,
  SUM(tx_count)                                  AS total_txs,
  SUM(tx_count - revert_count)                   AS success_txs,
  SUM(revert_count)                              AS revert_txs,
  SUM(if(is_spam_day, tx_count, 0))              AS spammer_txs,
  SUM(if(NOT is_spam_day, tx_count, 0))          AS nonspammer_txs,
  SUM(if(is_spam_day, revert_count, 0))          AS spammer_reverts,
  SUM(if(NOT is_spam_day, revert_count, 0))      AS nonspammer_reverts,
  countIf(is_spam_day)                           AS spammer_active_wallets,
  countIf(NOT is_spam_day)                       AS nonspammer_active_wallets
FROM flagged
WHERE day >= toDate({start:String}) AND day < toDate({end:String})
GROUP BY day
ORDER BY day
"""


def _block_range_from_multigas() -> tuple[int, int]:
    """Same convention as scripts/fetch_wallet_spam.py — narrow CH scan to
    the block range covered by our local multigas data."""
    import pyarrow.parquet as pq
    paths = sorted((_ROOT / "data" / "multigas_usage_extracts").glob("*/per_tx.parquet"))
    if not paths:
        return 0, 2**63 - 1
    block_min = block_max = None
    for p in paths:
        pf = pq.ParquetFile(str(p))
        idx = pf.schema_arrow.get_field_index("block")
        for rg in range(pf.num_row_groups):
            stats = pf.metadata.row_group(rg).column(idx).statistics
            if stats is None:
                continue
            mn, mx = int(stats.min), int(stats.max)
            block_min = mn if block_min is None else min(block_min, mn)
            block_max = mx if block_max is None else max(block_max, mx)
    return block_min or 0, block_max or 2**63 - 1


def fetch_daily(wallet_df: pl.DataFrame) -> pl.DataFrame:
    """Per-day totals + spammer-vs-non-spammer breakdown.  Cached at
    DAILY_PARQUET so the dashboard re-renders without going to CH."""
    # ClickHouse's FULL OUTER JOIN returns the type-default (1970-01-01 for
    # Date) instead of NULL on the missing side — leak through here would
    # produce one bogus epoch-dated row that stretches the x-axis to 1970.
    start_dt = np.datetime64(START_DATE)
    end_dt   = np.datetime64(END_DATE)
    window_filter = (pl.col("day") >= start_dt) & (pl.col("day") < end_dt)

    if DAILY_PARQUET.exists():
        print(f"loading daily cache: {DAILY_PARQUET}")
        return pl.read_parquet(DAILY_PARQUET).filter(window_filter)

    block_min, block_max = _block_range_from_multigas()
    print(f"fetching daily breakdown from CH "
          f"(blocks [{block_min:,}, {block_max:,}], "
          f"flagging done server-side via per-day p99.9 + revert thresholds)...")
    client = _ch_client()
    t = time.time()
    df_pd = client.query_df(
        DAILY_SQL,
        parameters={
            "start": START_DATE,
            "end": END_DATE,
            "block_min": block_min,
            "block_max": block_max,
            "revert_min_txs": 50,
            "revert_ratio": 0.30,
        },
    )
    print(f"  CH query: {time.time()-t:.1f}s, {len(df_pd):,} days")

    df = pl.from_pandas(df_pd).filter(window_filter)
    DAILY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(DAILY_PARQUET)
    print(f"  cached → {DAILY_PARQUET}")
    return df


# ── Headline stats (rendered as an HTML table) ──────────────────────────────
def stats_block(wallet_df: pl.DataFrame) -> str:
    n_total = wallet_df.height
    is_spam       = wallet_df["is_spam"].sum()       # frac_spam_days >= 0.5
    is_spam_ever  = wallet_df["is_spam_ever"].sum()  # any flagged day
    n_vol_only    = wallet_df.filter(
        (pl.col("n_days_high_vol") > 0) & (pl.col("n_days_high_rev") == 0)
    ).height
    n_rev_only    = wallet_df.filter(
        (pl.col("n_days_high_vol") == 0) & (pl.col("n_days_high_rev") > 0)
    ).height
    n_both        = wallet_df.filter(
        (pl.col("n_days_high_vol") > 0) & (pl.col("n_days_high_rev") > 0)
    ).height
    n_never       = wallet_df.filter(
        (pl.col("n_days_high_vol") == 0) & (pl.col("n_days_high_rev") == 0)
    ).height

    total_txs  = int(wallet_df["tx_count"].sum())
    revert_txs = int(wallet_df["revert_count"].sum())
    revert_pct = revert_txs / total_txs * 100.0 if total_txs else 0.0

    rows = [
        ("Total wallets seen in window",             f"{n_total:,}"),
        ("Total txs (success + revert)",             f"{total_txs:,}"),
        ("Revert txs (in `transactions_failed`)",    f"{revert_txs:,} ({revert_pct:.2f}% of total)"),
        ("Wallets flagged as <b>persistent spam</b> "
         "(<code>frac_spam_days ≥ 0.5</code>)",      f"{int(is_spam):,}"),
        ("Wallets flagged on ≥ 1 day "
         "(<code>is_spam_ever</code>)",              f"{int(is_spam_ever):,}"),
        ("&nbsp;&nbsp;⤷ high-volume only",           f"{n_vol_only:,}"),
        ("&nbsp;&nbsp;⤷ high-revert only",           f"{n_rev_only:,}"),
        ("&nbsp;&nbsp;⤷ both signals at any point",  f"{n_both:,}"),
        ("Never-flagged wallets",                    f"{n_never:,}"),
    ]
    th = ("padding:6px 12px; border-bottom:2px solid #333; "
          "background:#f4f4f4; text-align:left; font-size:14px;")
    td = ("padding:5px 12px; border-bottom:1px solid #ddd; font-size:14px;")
    body = "\n".join(
        f"<tr><td style='{td}'>{name}</td>"
        f"<td style='{td}; text-align:right; font-family:monospace'>{val}</td></tr>"
        for name, val in rows
    )
    return (
        f"<table style='border-collapse:collapse; max-width:900px; margin:0 auto;"
        f" box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"<tr><th style='{th}'>Metric</th>"
        f"<th style='{th}; text-align:right'>Value</th></tr>"
        f"{body}</table>"
    )


METHODOLOGY_HTML = """
<div style='max-width:900px; margin:0 auto; font-size:14px; line-height:1.55;'>
<h3>How a wallet gets flagged</h3>
<p>The classification runs on every wallet that sent at least one transaction
in the analysis window (<b>2025-10-01 → 2026-03-01</b>). For each wallet we
aggregate <em>per day</em> first, then roll up to a single label per wallet.</p>

<ol>
  <li><b>Per-(day, wallet) aggregation</b> over <code>raw_arbitrum.v_transactions</code> (success)
      <code>FULL OUTER JOIN</code> <code>raw_arbitrum.transactions_failed</code> (reverts), keyed by
      <code>(block_date, &quot;from&quot;)</code>. Reverts live in the second table — they aren't in
      <code>v_transactions.success</code>.</li>

  <li><b>Per-day percentile cutoffs</b> on <code>tx_count</code>:
      <code>p10</code>, <code>p01</code>, <code>p001</code>. The 0.1% most-active wallets on a given day
      cross the <code>p001</code> threshold. Cutoffs are recomputed every day so weekend
      vs weekday quantiles don't bleed into each other.</li>

  <li><b>Two independent per-day signals</b>:
    <ul>
      <li><code>is_high_volume</code> = the wallet's <code>tx_count</code> that day exceeds the day's <code>p001</code> cutoff.</li>
      <li><code>is_high_revert</code> = the wallet's revert ratio that day is ≥ 30 % AND it had ≥ 50 txs that day
          (so a wallet with one failed tx doesn't trip the flag).</li>
    </ul>
    <code>is_spam_day = is_high_volume OR is_high_revert</code>.</li>

  <li><b>Wallet-level rollup</b>: for each address we count
      <code>n_days_active</code>, <code>n_days_high_vol</code>, <code>n_days_high_rev</code>, <code>n_days_spam</code>,
      and compute <code>frac_spam_days = n_days_spam / n_days_active</code>.
      The final tags are
      <code>is_spam = (frac_spam_days ≥ 0.5)</code> (persistent spammer)
      and <code>is_spam_ever = (n_days_spam ≥ 1)</code> (any-day flagged).</li>
</ol>

<p>SQL: <code>sql/arbitrum_wallet_spam_classification.sql</code>. Fetcher:
<code>scripts/fetch_wallet_spam.py</code>. Window-wide thresholds are exposed
as placeholders (<code>revert_min_txs</code>, <code>revert_ratio_threshold</code>,
<code>spam_day_frac</code>) so they can be tightened or relaxed without
re-querying the per-day intermediate.</p>
</div>
"""


# ── Build dashboard ────────────────────────────────────────────────────────
def build_figure(daily: pl.DataFrame) -> go.Figure:
    days       = daily["day"].to_list()
    total_txs  = daily["total_txs"].to_numpy()
    revert_txs = daily["revert_txs"].to_numpy()
    revert_pct = np.where(total_txs > 0, revert_txs / total_txs * 100.0, 0.0)

    spammer_active    = daily["spammer_active_wallets"].to_numpy()
    nonspammer_active = daily["nonspammer_active_wallets"].to_numpy()
    spammer_txs       = daily["spammer_txs"].to_numpy()
    nonspammer_txs    = daily["nonspammer_txs"].to_numpy()

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.25, 0.20, 0.30, 0.25],
        vertical_spacing=0.07,
        subplot_titles=[
            "Daily total txs vs reverts",
            "Daily revert ratio (%)",
            "Daily active wallets — spammers vs non-spammers (stacked)",
            "Daily tx count — spammers vs non-spammers (stacked)",
        ],
    )

    # Row 1: total vs revert txs (line + line)
    fig.add_trace(go.Scatter(
        x=days, y=total_txs, mode="lines",
        name="Total txs",
        line=dict(color=COL_TOTAL, width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>total = %{y:,}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=days, y=revert_txs, mode="lines",
        name="Revert txs",
        line=dict(color=COL_REVERT, width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>reverts = %{y:,}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="txs / day", row=1, col=1)

    # Row 2: revert ratio
    fig.add_trace(go.Scatter(
        x=days, y=revert_pct, mode="lines",
        name="Revert %",
        line=dict(color=COL_REVERT, width=1.4),
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>revert = %{y:.2f}%<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="revert %", row=2, col=1)

    # Row 3: active wallets stacked
    fig.add_trace(go.Bar(
        x=days, y=nonspammer_active, name="Non-spammer wallets",
        marker_color=COL_NORMAL,
        hovertemplate="%{x|%Y-%m-%d}<br>non-spam = %{y:,}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=days, y=spammer_active, name="Spammer wallets",
        marker_color=COL_SPAMMER,
        hovertemplate="%{x|%Y-%m-%d}<br>spam = %{y:,}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(title_text="active wallets", row=3, col=1)

    # Row 4: tx count stacked
    fig.add_trace(go.Bar(
        x=days, y=nonspammer_txs, name="Non-spammer txs",
        marker_color=COL_NORMAL, showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>non-spam = %{y:,}<extra></extra>",
    ), row=4, col=1)
    fig.add_trace(go.Bar(
        x=days, y=spammer_txs, name="Spammer txs",
        marker_color=COL_SPAMMER, showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>spam = %{y:,}<extra></extra>",
    ), row=4, col=1)
    fig.update_yaxes(title_text="txs / day", row=4, col=1)

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=1200,
        margin=dict(l=70, r=80, t=80, b=60),
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


# ── Driver ──────────────────────────────────────────────────────────────────
def main():
    if not SPAM_PARQUET.exists():
        sys.exit(f"missing {SPAM_PARQUET} — run scripts/fetch_wallet_spam.py first")
    print(f"loading {SPAM_PARQUET}")
    wallet_df = pl.read_parquet(SPAM_PARQUET)
    print(f"  {wallet_df.height:,} wallets")

    daily = fetch_daily(wallet_df)

    # Build figure + assemble HTML.
    fig = build_figure(daily)
    fig_html = fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"displaylogo": False, "responsive": True},
    )

    stats_html = stats_block(wallet_df)

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Spam classification — insights</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 30px auto; max-width: 1200px; color: #222; }}
  h1, h2 {{ font-weight: 600; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 32px 0 12px; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
          font-size: 13px; }}
  .subtitle {{ color: #555; font-size: 14px; margin-bottom: 24px; }}
</style>
</head><body>

<h1>Spam classification — insights</h1>
<div class="subtitle">
  Window: {START_DATE} → {END_DATE} ({wallet_df.height:,} wallets,
  {wallet_df['tx_count'].sum():,} txs).
  Source SQL: <code>sql/arbitrum_wallet_spam_classification.sql</code>.
</div>

<h2>1. Headline stats</h2>
{stats_html}

<h2>2. Methodology</h2>
{METHODOLOGY_HTML}

<h2>3. Daily transaction volume + revert ratio</h2>
<h2>4. Daily breakdown — spammers vs non-spammers</h2>
{fig_html}

</body></html>
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page)
    print(f"saved {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
