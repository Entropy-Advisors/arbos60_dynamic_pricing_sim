"""
ArbOS 51 simulated hourly L2 fees, observed on-chain vs the four exp
approximations exposed on `arbos51.Arbos51GasPricing` (taylor4, taylor5,
taylor6, true exp).

Why this exists: the on-chain ArbOS 51 controller computes
    p(t) = p_min · ApproxExpBasisPoints(E(t), 4)
which is a degree-4 Taylor expansion of exp.  Solidity has no fixed-point
exp, so the protocol can't use np.exp.  This chart shows that any other
choice — even higher-order Taylor — drifts away from the realised
on-chain prices, especially during high-backlog windows where E(t) is
not small.  Taylor-4 is therefore not a coincidence; it's the spec.

Inputs:
  data/onchain_blocks_transactions/per_block.parquet
        block_time, total_l2_gas, l2_base, l2_surplus

Output:
  figures/arbos51_taylor_comparison.html
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
import arbos51                                                    # noqa: E402

_ROOT     = _HERE.parent
OUT_HTML  = _ROOT / "figures" / "arbos51_taylor_comparison.html"
# Run the simulation over the full window (so backlogs warm up correctly),
# then crop the display to three short peak-days slices where the four exp
# approximations actually diverge.  Plotted side-by-side in a 1×3 subplot.
START_DT = datetime(2025, 10, 1)
# Tight peak-day windows so the divergence between exp approximations
# stays visible — wider windows pad with floor-price hours and flatten
# the contrast.  Adjust freely; backlog warm-up runs over the full
# window before any cropping.
DISPLAY_WINDOWS: list[tuple[datetime, datetime, str]] = [
    (datetime(2026, 1, 31, 15), datetime(2026, 1, 31, 21), "Jan 31 peak (15-20 UTC)"),
    (datetime(2026, 2,  5),     datetime(2026, 2,  7),     "Feb 5 → Feb 6"),
    (datetime(2026, 3, 23,  6), datetime(2026, 3, 23, 18), "Mar 23 peak (06-17 UTC)"),
]

EXP_METHODS = ["taylor4", "taylor5", "taylor6", "exp"]
METHOD_STYLE = {
    # (color, dash, marker, label)
    "taylor4": ("#d62728", "solid",      "circle",       "Taylor-4"),
    "taylor5": ("#2ca02c", "dash",       "diamond",      "Taylor-5"),
    "taylor6": ("#9467bd", "dot",        "triangle-up",  "Taylor-6"),
    "exp":     ("#08519c", "longdashdot","x",            "Exp (python)"),
}
OBSERVED_COLOR = "#111111"


def load_blocks() -> pl.DataFrame:
    pq = _ROOT / "data" / "onchain_blocks_transactions" / "per_block.parquet"
    print(f"Loading {pq}...")
    df = (
        pl.scan_parquet(str(pq))
          .filter(pl.col("block_date") >= START_DT)
          .select([
              "block_number", "block_time",
              "total_l2_gas", "l2_base", "l2_surplus",
          ])
          .sort("block_number")
          .collect()
    )
    print(f"  {df.height:,} blocks from {df['block_time'][0]} → "
          f"{df['block_time'][-1]}")
    return df


def per_block_t(df: pl.DataFrame) -> np.ndarray:
    """UTC seconds-since-epoch per block (int64), the t-axis arbos51 expects."""
    return arbos51.Arbos51GasPricing.block_seconds_utc(
        df["block_time"].cast(pl.Datetime("us")).to_numpy()
    )


def hourly_observed_fee_eth(df: pl.DataFrame) -> pl.DataFrame:
    """Observed hourly L2 fees (= base + surplus) summed per hour."""
    return (
        df.with_columns([
            pl.col("block_time").dt.truncate("1h").alias("hour"),
            (pl.col("l2_base") + pl.col("l2_surplus")).alias("l2_fee_eth"),
        ])
        .group_by("hour")
        .agg(pl.col("l2_fee_eth").sum())
        .sort("hour")
    )


def hourly_total_gas_mgas(df: pl.DataFrame) -> pl.DataFrame:
    """Hourly total L2 gas in Mgas — used as a low-opacity context bar
    behind the fee lines so the reader can correlate spikes with usage."""
    return (
        df.with_columns(pl.col("block_time").dt.truncate("1h").alias("hour"))
        .group_by("hour")
        .agg((pl.col("total_l2_gas").sum() / 1e6).alias("gas_mgas"))
        .sort("hour")
    )


def post_dia_backlog_state(df: pl.DataFrame):
    """Per-second post-DIA backlog state — B_j(t) for each ladder rung j
    plus the raw exponent E(t) = Σ_j B_j / (A_j · T_j).  Backlog warms up
    from DIA activation so by the time we hit Jan 31 / Feb 5 / Mar 23 the
    state is fully converged.  All our display windows are post-DIA."""
    df_dia = df.filter(
        pl.col("block_time") >= arbos51.ARBOS_DIA_ACTIVATION_UTC
    )
    block_t = arbos51.Arbos51GasPricing.block_seconds_utc(
        df_dia["block_time"].cast(pl.Datetime("us")).to_numpy()
    )
    total_g = df_dia["total_l2_gas"].cast(pl.Float64).to_numpy()
    engine = arbos51.Arbos51GasPricing()
    t_axis, inflow = engine.compute_inflow_per_t(total_g, block_t)
    T_j = np.array([T for T, _ in engine.ladder], dtype=np.float64)
    A_j = np.array([A for _, A in engine.ladder], dtype=np.float64)
    B = engine.backlog_per_second(inflow, T_j)               # (n_j, n_t)
    E = (B / (A_j * T_j * 1e6)[:, None]).sum(axis=0)        # (n_t,)
    return t_axis, B, E, T_j, A_j


def crop_state(t_axis: np.ndarray, *arrays, start: datetime, end: datetime):
    """Return slice of t_axis + each array between [start, end) in seconds."""
    t_start = int(start.timestamp())
    t_end   = int(end.timestamp())
    mask = (t_axis >= t_start) & (t_axis < t_end)
    out_t = t_axis[mask].astype("datetime64[s]").astype("datetime64[ms]")
    return (out_t,) + tuple(a[..., mask] if a.ndim == 2 else a[mask]
                             for a in arrays)


def hourly_simulated_fee_eth(
    df: pl.DataFrame, p_per_t: np.ndarray, t_axis: np.ndarray,
) -> pl.DataFrame:
    """Per-block simulated fee = p(block_t) · gas, gather to ETH/hour.
    p_per_t is in gwei (i.e. 1e-9 ETH per gas)."""
    block_t = per_block_t(df)
    # Per-block t-axis index.  arbos51's historical sim returns t_axis as
    # the chronological union of pre-Dia + Dia regimes, which is sorted.
    t_idx = np.searchsorted(t_axis, block_t, side="left")
    p_block = p_per_t[t_idx]                                # gwei / gas
    gas     = df["total_l2_gas"].to_numpy()                 # gas units
    fee_eth = p_block * gas * 1e-9                          # gwei·gas → ETH

    return (
        df.with_columns([
            pl.col("block_time").dt.truncate("1h").alias("hour"),
            pl.lit(fee_eth).alias("fee_eth"),
        ])
        .group_by("hour")
        .agg(pl.col("fee_eth").sum().alias("l2_fee_eth"))
        .sort("hour")
    )


def main() -> None:
    df = load_blocks()
    block_t  = per_block_t(df)
    total_g  = df["total_l2_gas"].cast(pl.Float64).to_numpy()

    obs_hr_full = hourly_observed_fee_eth(df)
    gas_hr_full = hourly_total_gas_mgas(df)
    print(f"  observed hourly fees: {obs_hr_full.height:,} hours")
    print(f"  hourly total gas:     {gas_hr_full.height:,} hours, "
          f"max {gas_hr_full['gas_mgas'].max():,.1f} Mgas/h")

    print("Computing post-DIA backlog state per second (6 rungs)...")
    t_axis_full, B_full, E_full, T_j, A_j = post_dia_backlog_state(df)
    print(f"  state covers {len(t_axis_full):,} seconds, "
          f"max E = {E_full.max():.3f}")

    sims_full = {}
    for m in EXP_METHODS:
        t0 = time.time()
        t_axis, p = arbos51.Arbos51GasPricing.historical_price_per_second(
            total_g, block_t, exp_method=m,
        )
        sims_full[m] = hourly_simulated_fee_eth(df, p, t_axis)
        print(f"  {m:>8s}: {sims_full[m].height:,} hours (full window), "
              f"sim {time.time()-t0:.1f}s, "
              f"sum = {sims_full[m]['l2_fee_eth'].sum():,.1f} ETH")
    print(f"  observed total (full) = {obs_hr_full['l2_fee_eth'].sum():,.1f} ETH")

    # Crop to each display window separately for the side-by-side plot.
    def _crop(d: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
        return d.filter(
            (pl.col("hour") >= start) & (pl.col("hour") < end)
        )
    obs_windows: list[pl.DataFrame] = []
    sims_windows: list[dict[str, pl.DataFrame]] = []
    gas_windows: list[pl.DataFrame] = []
    state_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for ws, we, label in DISPLAY_WINDOWS:
        obs_w  = _crop(obs_hr_full, ws, we)
        sims_w = {m: _crop(d, ws, we) for m, d in sims_full.items()}
        gas_w  = _crop(gas_hr_full, ws, we)
        t_w, B_w, E_w = crop_state(t_axis_full, B_full, E_full,
                                     start=ws, end=we)
        obs_windows.append(obs_w)
        sims_windows.append(sims_w)
        gas_windows.append(gas_w)
        state_windows.append((t_w, B_w, E_w))
        print(f"\nWindow {label} ({ws.date()} → {we.date()}): "
              f"{obs_w.height} hours")
        print(f"  observed total = {obs_w['l2_fee_eth'].sum():,.2f} ETH")
        for m in EXP_METHODS:
            print(f"  {m:>8s} total = "
                  f"{sims_w[m]['l2_fee_eth'].sum():,.2f} ETH")

    # ── Δ% stats (vs Taylor-4) — full window, split normal vs peak ────────
    # Peak hour := observed L2 fee is in the top 10 % of all hours.
    obs_full = obs_hr_full["l2_fee_eth"].to_numpy()
    peak_thresh = float(np.quantile(obs_full, 0.90))
    is_peak = obs_full >= peak_thresh
    print(f"\nPeak-hour cutoff (top 10 %): {peak_thresh:.3f} ETH/hour, "
          f"{int(is_peak.sum())} peak hours / {len(is_peak)} total")

    base = sims_full["taylor4"]["l2_fee_eth"].to_numpy()
    active = base > 1e-9
    stats = []
    for m in ("taylor5", "taylor6", "exp"):
        v = sims_full[m]["l2_fee_eth"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(active, (v - base) / np.maximum(base, 1e-12) * 100.0, np.nan)
        norm_mask = active & ~is_peak
        peak_mask = active &  is_peak
        stats.append({
            "method":        m,
            "median_normal": float(np.nanmedian(d[norm_mask]))
                              if norm_mask.any() else 0.0,
            "mean_normal":   float(np.nanmean(d[norm_mask]))
                              if norm_mask.any() else 0.0,
            "median_peak":   float(np.nanmedian(d[peak_mask]))
                              if peak_mask.any() else 0.0,
            "mean_peak":     float(np.nanmean(d[peak_mask]))
                              if peak_mask.any() else 0.0,
        })
    print("\nΔ% vs Taylor-4 (full window):")
    for s in stats:
        print(f"  {s['method']:>8s}: "
              f"normal med={s['median_normal']:+.2f}% mean={s['mean_normal']:+.2f}% "
              f"peak med={s['median_peak']:+.2f}% mean={s['mean_peak']:+.2f}%")

    # Persist for the deck builder.
    import json
    stats_path = _ROOT / "data" / "arbos51_taylor_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "peak_threshold_eth": peak_thresh,
        "n_peak_hours": int(is_peak.sum()),
        "n_total_hours": int(len(is_peak)),
        "rows": stats,
    }, indent=2))
    print(f"  stats → {stats_path}")

    # ── Figure: 4 rows × 3 cols ──────────────────────────────────────────
    # Row 1: only middle col — exp vs Taylor-{4,5,6} reference, E ∈ [0, 8]
    # Row 2: per-constraint backlog B_j(t) (6 lines per panel, decimated)
    # Row 3: raw exponent  E(t) = Σ_j B_j / (A_j · T_j)  (single line)
    # Row 4: hourly L2 fees + total-gas bars (secondary y).  Same Y-axis
    #        range across the 3 fee panels: 0..20 ETH/h primary,
    #        0..90K Mgas/h secondary.
    n_panels   = len(DISPLAY_WINDOWS)
    titles_top = [w[2] for w in DISPLAY_WINDOWS]
    BACKLOG_COLORS = ["#08306b", "#2171b5", "#6baed6",
                       "#fdae6b", "#e6550d", "#a63603"]

    fig = make_subplots(
        rows=4, cols=n_panels,
        shared_yaxes=False, horizontal_spacing=0.07,
        vertical_spacing=0.07,
        column_widths=[1.0 / n_panels] * n_panels,
        row_heights=[0.18, 0.24, 0.16, 0.42],
        subplot_titles=(
            "Exp vs Taylor-4  (E ∈ [0, 8])",
            *titles_top,
            *([""] * n_panels),
            *([""] * n_panels),
        ),
        specs=[
            [None, {}, None],
            [{}] * n_panels,
            [{}] * n_panels,
            [{"secondary_y": True}] * n_panels,
        ],
    )

    # ── Row 1 (middle col only): Exp vs Taylor-4 reference, E ∈ [0, 8] ─
    E_ref = np.linspace(0.0, 8.0, 401)
    series_ref = [
        ("exp",     arbos51.Arbos51GasPricing.exp_exact(E_ref)),
        ("taylor4", arbos51.Arbos51GasPricing.taylor4_exp(E_ref)),
    ]
    for m, y_ref in series_ref:
        color, dash, _, label = METHOD_STYLE[m]
        fig.add_trace(go.Scatter(
            x=E_ref, y=y_ref,
            name=label, mode="lines",
            line=dict(color=color, width=1.8, dash=dash),
            legendgroup=m, showlegend=False,
            hovertemplate=(
                f"{label}: E = " "%{x:.2f}<br>"
                f"{label}(E) = " "%{y:,.4g}<extra></extra>"
            ),
        ), row=1, col=2)
    fig.update_yaxes(
        title_text="approx(E)",
        showline=True, linewidth=1.0,
        linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside",
        row=1, col=2,
    )
    fig.update_xaxes(
        title_text="E",
        showline=True, linewidth=1.0,
        linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside",
        row=1, col=2,
    )

    for col_idx, (obs_w, sims_w, gas_w, state_w) in enumerate(
        zip(obs_windows, sims_windows, gas_windows, state_windows), 1,
    ):
        show_legend = (col_idx == 1)
        t_w, B_w, E_w = state_w

        # ── Row 2: per-constraint backlog (Mgas) ────────────────────────────
        # Decimate per-second backlog to 1-minute resolution so the
        # embedded HTML stays small (per-second × 6 rungs × 3 panels was
        # ~4 M points = 50 MB).
        stride = 60
        t_dec = t_w[::stride]
        B_dec = B_w[:, ::stride]
        for j in range(B_dec.shape[0]):
            fig.add_trace(go.Scatter(
                x=t_dec, y=B_dec[j] / 1e6,
                name=f"B_j (j={j})",
                line=dict(color=BACKLOG_COLORS[j], width=1.0),
                legendgroup=f"backlog_j{j}",
                showlegend=show_legend,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M:%S}<br>"
                    f"B_{j} = " "%{y:,.1f} Mgas<extra></extra>"
                ),
            ), row=2, col=col_idx)
        fig.update_yaxes(
            title_text=("Backlog B_j (Mgas)" if col_idx == 1 else ""),
            showline=True, linewidth=1.0,
            linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside",
            row=2, col=col_idx,
        )
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside",
                          row=2, col=col_idx)

        # ── Row 3: raw exponent E(t) = Σ_j B_j / (A_j · T_j) ─────────
        # Same per-second sequence underlying the Taylor approx; one line
        # per panel since E(t) is independent of the exp method (it's just
        # backlog state).
        E_dec = E_w[::stride]
        fig.add_trace(go.Scatter(
            x=t_dec, y=E_dec,
            name="E(t) = Σ B_j / (A_j·T_j)",
            line=dict(color="#111111", width=1.2),
            legendgroup="Et", showlegend=show_legend,
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M:%S}<br>"
                "E(t) = %{y:.4f}<extra></extra>"
            ),
        ), row=3, col=col_idx)
        fig.update_yaxes(
            title_text=("E(t)" if col_idx == 1 else ""),
            showline=True, linewidth=1.0,
            linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside",
            row=3, col=col_idx,
        )
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside",
                          row=3, col=col_idx)

        # ── Row 4: fees + gas-context bars ───────────────────────────
        # Gas-context bars FIRST so the line traces render on top.
        fig.add_trace(go.Bar(
            x=gas_w["hour"].to_list(),
            y=gas_w["gas_mgas"].to_numpy(),
            name="Total gas (Mgas/h)",
            marker=dict(color="#9aa3ad", line=dict(width=0)),
            opacity=0.25,
            legendgroup="gas", showlegend=show_legend,
            hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                          "total gas = %{y:,.1f} Mgas<extra></extra>",
        ), row=4, col=col_idx, secondary_y=True)
        fig.update_yaxes(
            title_text=("Mgas / h" if col_idx == n_panels else ""),
            range=[0, 90_000],
            autorange=False,
            fixedrange=True,
            showgrid=False, zeroline=False,
            tickfont=dict(color="#888", size=10),
            color="#888",
            row=4, col=col_idx, secondary_y=True,
        )

        fig.add_trace(go.Scatter(
            x=obs_w["hour"].to_list(),
            y=obs_w["l2_fee_eth"].to_numpy(),
            name="Observed on-chain",
            line=dict(color=OBSERVED_COLOR, width=2.0),
            legendgroup="obs", showlegend=show_legend,
            hovertemplate="%{x|%Y-%m-%d %H:00}<br>"
                          "observed = %{y:,.2f} ETH<extra></extra>",
        ), row=4, col=col_idx)

        for m in EXP_METHODS:
            color, dash, marker, label = METHOD_STYLE[m]
            d = sims_w[m]
            if d.height == 0:
                continue
            fig.add_trace(go.Scatter(
                x=d["hour"].to_list(), y=d["l2_fee_eth"].to_numpy(),
                name=label, mode="lines",
                line=dict(color=color, width=1.4, dash=dash),
                legendgroup=m, showlegend=show_legend,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:00}<br>"
                    f"{label} = " "%{y:,.2f} ETH<extra></extra>"
                ),
            ), row=4, col=col_idx)
            n = d.height
            idx = np.arange(0, n, max(n // 30, 1))
            fig.add_trace(go.Scatter(
                x=d["hour"].to_numpy()[idx],
                y=d["l2_fee_eth"].to_numpy()[idx],
                mode="markers",
                marker=dict(color=color, size=5, symbol=marker, opacity=0.85),
                legendgroup=m, showlegend=False, hoverinfo="skip",
            ), row=4, col=col_idx)

        fig.update_yaxes(
            title_text=("ETH / hour" if col_idx == 1 else ""),
            range=[0, 20],
            autorange=False,
            fixedrange=True,
            showline=True, linewidth=1.0,
            linecolor="rgba(0,0,0,0.45)", mirror=True, ticks="outside",
            row=4, col=col_idx, secondary_y=False,
        )
        fig.update_xaxes(showline=True, linewidth=1.0,
                          linecolor="rgba(0,0,0,0.45)",
                          mirror=True, ticks="outside",
                          row=4, col=col_idx)

    # No internal title — the slide's <h2>/<h3> already labels the chart.
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=920,
        margin=dict(l=70, r=70, t=70, b=50),
        font=dict(size=12, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(OUT_HTML),
        include_plotlyjs="cdn", full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
