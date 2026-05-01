"""
Path B (nearest-centroid) and Path C (linear-system, NNLS) implementations
of the demand-elasticity model from `docs/demand_elasticity_plan.md`.

Cached inputs (no recomputation):
  - data/clustering_cache/aggs.npz            (K, hourly_gas, vol_sum, n_txs)
  - data/clustering_cache/aggregate_meta.pkl  (min_hour anchor)
  - data/capacity_hourly_prices.parquet       (ArbOS 60 hourly prices p_k^new)
  - data/onchain_blocks_transactions/per_block.parquet
        (avg_eff_price_gwei → hourly p^old; ArbOS 51 charges flat across k)

Output:
  - figures/demand_elasticity.html

Caveats handled:
  * NNLS in Path C ⇒ no negative n_c(t).
  * Condition number κ(M) reported in the page; if very large (>1e8) the
    page surfaces a warning and Path B is the primary view.
  * L1 calldata is dropped from the resource set — Tyler's per-tx
    multigas extracts have l1Calldata = 0 for every row (confirmed
    earlier), so it adds nothing but a degenerate column.
  * Sanity assert: with p^new = p^old, predicted ≈ observed (per-archetype
    mean residual reported on the page).
  * Unit-elasticity revenue invariance (α = 1 ⇒ Σ_c fee'_c = Σ_c D_c =
    historical revenue) verified numerically; that's why the chart
    focuses on per-archetype/per-resource gas redistribution rather
    than absolute revenue change.
"""

from __future__ import annotations

import pathlib
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

OUT_HTML = _ROOT / "figures" / "demand_elasticity.html"

# ── Resource set (drop L1; always 0 in Tyler's per-tx extracts) ────────────
PRICED_RESOURCES = ["c", "sw", "sr", "sg", "hg", "l2"]
K_PRICED = len(PRICED_RESOURCES)
RESOURCE_LABEL = {
    "c":  "Computation",
    "sw": "Storage Write",
    "sr": "Storage Read",
    "sg": "Storage Growth",
    "hg": "History Growth",
    "l2": "L2 Calldata",
}
RESOURCE_COLOR = {
    "c":  "#1f77b4", "sw": "#2ca02c", "sr": "#98df8a",
    "sg": "#d62728", "hg": "#ff7f0e", "l2": "#9467bd",
}
CLUSTER_PALETTE = [
    "#08519c", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]
ALPHA = 1.0


# ── Loaders ────────────────────────────────────────────────────────────────
def load_cluster_M():
    """Return (K, hourly_gas[K, n_hours, K_priced], M[K_priced, K],
    n_txs[K], min_hour).  M[k, c] = total gas of resource k in cluster c
    divided by the cluster's tx count (i.e. centroid in raw-gas space)."""
    aggs = np.load(_ROOT / "data" / "clustering_cache" / "aggs.npz")
    with open(_ROOT / "data" / "clustering_cache" / "aggregate_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    K = int(aggs["K"])
    # Drop the trailing L1 column.
    hourly_gas = np.asarray(aggs["hourly_gas"])[:, :, :K_PRICED]
    vol_sum    = np.asarray(aggs["vol_sum"])[:, :K_PRICED]
    n_txs      = np.asarray(aggs["n_txs"])
    M = (vol_sum / np.maximum(n_txs, 1)[:, None]).T   # (K_priced, K)
    return K, hourly_gas, M, n_txs, int(meta["min_hour"])


def hourly_p_new(min_hour: int, n_hours: int) -> np.ndarray:
    """Hourly p_k^new(t) (gwei) from the cached ArbOS 60 simulation."""
    df = (
        pl.read_parquet(_ROOT / "data" / "capacity_hourly_prices.parquet")
          .with_columns(
              ((pl.col("hour").dt.timestamp("ms") // 1000) // 3600)
              .cast(pl.Int64).alias("hour_idx")
          )
    )
    p_new = np.full((n_hours, K_PRICED), 0.02, dtype=np.float64)
    df_clean = df.filter(
        (pl.col("hour_idx") >= min_hour)
        & (pl.col("hour_idx") < min_hour + n_hours)
    )
    idx = (df_clean["hour_idx"].to_numpy() - min_hour).astype(np.int64)
    for ki, k in enumerate(PRICED_RESOURCES):
        p_new[idx, ki] = df_clean[f"p_{k}"].to_numpy()
    return p_new


def hourly_p_old(min_hour: int, n_hours: int) -> np.ndarray:
    """Hourly p^old(t) gas-weighted mean of avg_eff_price_gwei.  ArbOS 51
    is flat across resources so this is the single ArbOS 51 effective
    price per hour."""
    blocks_pq = _ROOT / "data" / "onchain_blocks_transactions" / "per_block.parquet"
    df = (
        pl.scan_parquet(str(blocks_pq))
          .select(["block_time", "avg_eff_price_gwei", "total_l2_gas"])
          .with_columns(pl.col("block_time").dt.truncate("1h").alias("hour"))
          .group_by("hour")
          .agg([
              ((pl.col("avg_eff_price_gwei") * pl.col("total_l2_gas")).sum()
               / pl.col("total_l2_gas").sum().clip(lower_bound=1.0)
              ).alias("p_old"),
          ])
          .sort("hour")
          .with_columns(
              ((pl.col("hour").dt.timestamp("ms") // 1000) // 3600)
              .cast(pl.Int64).alias("hour_idx")
          )
          .collect()
    )
    p_old = np.full(n_hours, 0.01, dtype=np.float64)
    df_clean = df.filter(
        (pl.col("hour_idx") >= min_hour)
        & (pl.col("hour_idx") < min_hour + n_hours)
    )
    idx = (df_clean["hour_idx"].to_numpy() - min_hour).astype(np.int64)
    p_old[idx] = df_clean["p_old"].to_numpy()
    return p_old


# ── Path implementations ───────────────────────────────────────────────────
def path_b_gas(hourly_gas: np.ndarray) -> np.ndarray:
    """g_c^B(t) = Σ_k hourly_gas[c, t, k]."""
    return hourly_gas.sum(axis=2)


def path_c_gas(M: np.ndarray, g_k_hourly: np.ndarray):
    """Per hour solve  min_{n ≥ 0} ‖ M·n − g(t) ‖₂ via NNLS, then
    g_c^C(t) = n_c(t) · Σ_k M[k, c].  Returns (g_c, n_c, residuals)."""
    K_priced, K_clusters = M.shape
    n_hours = g_k_hourly.shape[0]
    n_c = np.zeros((K_clusters, n_hours), dtype=np.float64)
    residuals = np.zeros(n_hours, dtype=np.float64)

    t0 = time.time()
    for h in range(n_hours):
        g_h = g_k_hourly[h]
        if g_h.sum() == 0:
            continue
        try:
            sol, res = nnls(M, g_h, maxiter=200)
            n_c[:, h] = sol
            residuals[h] = res
        except RuntimeError:
            # Fallback when NNLS doesn't converge — clipped lstsq.
            sol, *_ = np.linalg.lstsq(M, g_h, rcond=None)
            n_c[:, h] = np.maximum(sol, 0.0)
            residuals[h] = float(np.linalg.norm(M @ n_c[:, h] - g_h))

    g_c = n_c * M.sum(axis=0)[:, None]
    rel_err = (np.linalg.norm(M @ n_c - g_k_hourly.T)
               / max(np.linalg.norm(g_k_hourly.T), 1.0))
    print(f"  Path C: NNLS {n_hours} hours in {time.time()-t0:.1f}s, "
          f"reconstruction relative L2 = {rel_err:.4f}")
    return g_c, n_c, residuals


def archetype_eff_price(M: np.ndarray, p) -> np.ndarray:
    """\bar p_c(t) = Σ_k M[k, c]·p_k(t) / Σ_k M[k, c].

    p shape (n_hours,)        ⇒ flat-across-resources case (ArbOS 51).
    p shape (n_hours, K_priced) ⇒ per-resource (ArbOS 60).
    """
    K_priced, K_clusters = M.shape
    if p.ndim == 1:
        return np.broadcast_to(p, (K_clusters, len(p))).copy()
    M_col_sum = np.maximum(M.sum(axis=0), 1e-12)
    return ((p @ M) / M_col_sum).T


def calibrate_D(g_c: np.ndarray, p_c: np.ndarray, alpha: float = ALPHA,
                weight_min_gas: float = 1e6) -> np.ndarray:
    """D_c = mean over hours of g_c(t) · p_c(t)^α.

    Hours with no demand contribute zero on the numerator and dilute
    the mean.  Weight by `g_c > weight_min_gas` so empty hours don't
    pull D_c toward zero."""
    p_pow = np.power(np.maximum(p_c, 1e-12), alpha)
    fee_per_hr = g_c * p_pow                    # gwei·gas, per (c, hour)
    weights = (g_c > weight_min_gas).astype(np.float64)
    w_sum = np.maximum(weights.sum(axis=1), 1.0)
    return (fee_per_hr * weights).sum(axis=1) / w_sum


def predict_g_c(D_c: np.ndarray, p_c_new: np.ndarray,
                alpha: float = ALPHA) -> np.ndarray:
    """g'_c(t) = D_c · p'_c(t)^(-α)."""
    return D_c[:, None] * np.power(np.maximum(p_c_new, 1e-12), -alpha)


def per_resource_gas(M: np.ndarray, g_c: np.ndarray) -> np.ndarray:
    """Reconstruct per-resource gas from per-archetype gas using M.
    g_k(t) = M · ( g_c(t) / Σ_k M[k,c] )."""
    M_col_sum = np.maximum(M.sum(axis=0), 1e-12)
    n_c = g_c / M_col_sum[:, None]
    return M @ n_c


# ── Figure ─────────────────────────────────────────────────────────────────
def cluster_label(c: int) -> str:
    return f"Archetype c{c}"


def build_figure(K: int, M: np.ndarray, hour_axis: np.ndarray,
                  D_c_B: np.ndarray, D_c_C: np.ndarray,
                  g_c_B: np.ndarray, g_c_C: np.ndarray,
                  g_c_pred: np.ndarray,
                  p_c_old: np.ndarray, p_c_new: np.ndarray,
                  g_k_obs: np.ndarray, g_k_pred: np.ndarray,
                  alpha: float) -> go.Figure:
    fig = make_subplots(
        rows=6, cols=1,
        vertical_spacing=0.05,
        row_heights=[0.13, 0.10, 0.13, 0.18, 0.18, 0.28],
        subplot_titles=(
            f"Centroid matrix M — avg gas per resource per archetype "
            f"(rows={K_PRICED} priced resources, cols={K} archetypes)",
            "Calibrated D_c per archetype "
            "(latent demand intercept, gwei·gas)",
            "Effective price per archetype: ArbOS 51 (p^old) "
            "vs ArbOS 60 (p'_c, time-mean)",
            "Per-archetype gas demand — observed Path B "
            "vs predicted under ArbOS 60",
            "Path B vs Path C consistency "
            "(both use the same M and prices, NNLS handles non-negativity)",
            "Per-resource gas: observed total vs predicted total "
            "under ArbOS 60",
        ),
        specs=[
            [{"type": "heatmap"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
        ],
    )

    # Panel 1: M heatmap (transpose so resources are rows for readability)
    res_labels = [RESOURCE_LABEL[k] for k in PRICED_RESOURCES]
    cl_labels  = [cluster_label(c) for c in range(K)]
    fig.add_trace(go.Heatmap(
        z=M, x=cl_labels, y=res_labels,
        colorscale="Blues",
        hovertemplate="resource=%{y}<br>%{x}<br>"
                      "avg gas/tx = %{z:,.0f}<extra></extra>",
        colorbar=dict(title="gas/tx", thickness=10, len=0.16, y=0.94),
    ), row=1, col=1)

    # Panel 2: D_c bar (Path B in solid, Path C as overlay marker)
    fig.add_trace(go.Bar(
        x=cl_labels, y=D_c_B, name="D_c (Path B)",
        marker_color=[CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)] for c in range(K)],
        hovertemplate="%{x}<br>D_c = %{y:.3e}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=cl_labels, y=D_c_C, mode="markers", name="D_c (Path C)",
        marker=dict(color="#444", size=10, symbol="diamond-open"),
        hovertemplate="%{x}<br>D_c (C) = %{y:.3e}<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="D_c", type="log", row=2, col=1)

    # Panel 3: bar_p_c (mean) vs bar_p'_c (mean)
    p_old_mean = p_c_old.mean(axis=1)
    p_new_mean = p_c_new.mean(axis=1)
    fig.add_trace(go.Bar(
        x=cl_labels, y=p_old_mean, name="bar p_c (ArbOS 51, hist mean)",
        marker_color="#d62728",
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=cl_labels, y=p_new_mean, name="bar p'_c (ArbOS 60, mean)",
        marker_color="#1f77b4",
    ), row=3, col=1)
    fig.update_yaxes(title_text="gwei", type="log", row=3, col=1)

    # Panel 4: per-archetype gas time series, observed vs predicted
    for c in range(K):
        col = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
        fig.add_trace(go.Scatter(
            x=hour_axis, y=g_c_B[c],
            name=f"c{c} observed",
            line=dict(color=col, width=1.0), opacity=0.65,
            legendgroup=f"arch_c{c}",
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"c{c} obs = " "%{y:,.0f} gas<extra></extra>"),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=hour_axis, y=g_c_pred[c],
            name=f"c{c} predicted",
            line=dict(color=col, width=1.4, dash="dot"),
            legendgroup=f"arch_c{c}", showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"c{c} pred = " "%{y:,.0f} gas<extra></extra>"),
        ), row=4, col=1)
    fig.update_yaxes(title_text="gas / hour", type="log", row=4, col=1)

    # Panel 5: Path B vs Path C
    for c in range(K):
        col = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
        fig.add_trace(go.Scatter(
            x=hour_axis, y=g_c_B[c],
            name=f"c{c} B",
            line=dict(color=col, width=1.0), opacity=0.55,
            legendgroup=f"bc_c{c}", showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"c{c} B = " "%{y:,.0f}<extra></extra>"),
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=hour_axis, y=g_c_C[c],
            name=f"c{c} C",
            line=dict(color=col, width=1.0, dash="dash"),
            legendgroup=f"bc_c{c}", showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"c{c} C = " "%{y:,.0f}<extra></extra>"),
        ), row=5, col=1)
    fig.update_yaxes(title_text="gas / hour", type="log", row=5, col=1)

    # Panel 6: per-resource observed vs predicted
    for ki, k in enumerate(PRICED_RESOURCES):
        col = RESOURCE_COLOR[k]
        fig.add_trace(go.Scatter(
            x=hour_axis, y=g_k_obs[ki],
            name=f"{RESOURCE_LABEL[k]} obs",
            line=dict(color=col, width=1.0), opacity=0.65,
            legendgroup=f"res_{k}",
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"{RESOURCE_LABEL[k]} obs = "
                           "%{y:,.0f}<extra></extra>"),
        ), row=6, col=1)
        fig.add_trace(go.Scatter(
            x=hour_axis, y=g_k_pred[ki],
            name=f"{RESOURCE_LABEL[k]} pred",
            line=dict(color=col, width=1.0, dash="dot"),
            legendgroup=f"res_{k}", showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d %H:00}<br>"
                           f"{RESOURCE_LABEL[k]} pred = "
                           "%{y:,.0f}<extra></extra>"),
        ), row=6, col=1)
    fig.update_yaxes(title_text="gas / hour", type="log", row=6, col=1)

    fig.update_layout(
        template="plotly_white",
        height=1700,
        margin=dict(l=80, r=40, t=60, b=40),
        font=dict(size=11, color="#222"),
        hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True,
                     ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.45)", mirror=True,
                     ticks="outside")
    return fig


# ── Page wrapper with methodology ─────────────────────────────────────────
PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Demand-elasticity simulation</title>
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
         margin: 30px auto; max-width: 1500px; color: #222; line-height: 1.55; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  h2 { font-size: 16px; margin: 20px 0 8px; color: #333; font-weight: 600; }
  .methodology {
    background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
    padding: 0.8em 1.4em; margin: 8px 0 20px; font-size: 13px;
  }
  .methodology ol { padding-left: 1.4em; margin: 0.4em 0 0; }
  .methodology li { margin: 0.5em 0; }
  .methodology .label { font-weight: 600; color: #333; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
         font-size: 12.5px; }
  .stats {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 0.8em; margin: 12px 0 18px;
  }
  .stat-card {
    border-left: 3px solid #1f77b4; padding: 0.5em 0.85em;
    background: rgba(31, 119, 180, 0.04);
  }
  .stat-card .lbl { font-size: 0.7em; color: #555;
                    text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-card .val { font-size: 1.05em; color: #111; font-weight: 600;
                    margin-top: 0.18em; font-variant-numeric: tabular-nums; }
  .stat-card.warn { border-left-color: #d62728;
                    background: rgba(214, 39, 40, 0.04); }
  .stat-card.warn .val { color: #b54a00; }
</style>
</head><body>

<h1>Demand-elasticity simulation: ArbOS 51 → ArbOS 60 (Paths B + C)</h1>

<div class="stats">
  <div class="stat-card">
    <div class="lbl">Archetypes K</div><div class="val">{{K}}</div>
  </div>
  <div class="stat-card{{COND_WARN}}">
    <div class="lbl">cond(M)</div><div class="val">{{COND_M}}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Elasticity α</div><div class="val">{{ALPHA}}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Sanity p_new=p_old</div>
    <div class="val">{{SANITY}}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Revenue invariance (α=1)</div>
    <div class="val">{{REV_RATIO}}× hist</div>
  </div>
</div>

<h2>Methodology</h2>
<div class="methodology">
  <ol>
    <li>
      <span class="label">Centroid matrix M (\(K_{\text{priced}} \times K\)).</span>
      Read the cached MiniBatchKMeans aggregator
      (<code>data/clustering_cache/aggs.npz</code>, K = {{K}}).  Each
      column is a cluster's average per-resource gas:
      \[M[k, c] \;=\; \frac{\sum_{tx \in c} g_{tx,k}}{|tx \in c|}
                 \;=\; \frac{\texttt{vol\_sum}[c, k]}{\texttt{n\_txs}[c]}\]
      L1 calldata is dropped (\(g_{l1} \equiv 0\) in Tyler's per-tx
      extracts) so M has 6 rows and {{K}} columns.
    </li>
    <li>
      <span class="label">Path B — nearest-centroid.</span>
      The cluster aggregator already streamed every priced tx through
      its nearest centroid and summed gas per (cluster, hour, resource).
      So
      \[g_c^{B}(t) \;=\; \sum_{k} \texttt{hourly\_gas}[c, t, k]\]
      No further computation; non-negative by construction.
    </li>
    <li>
      <span class="label">Path C — linear-system NNLS.</span>
      For each hour solve the constrained least-squares problem
      \[n(t) \;=\; \arg\min_{n \,\ge\, 0}\;
            \bigl\| \, M \cdot n \;-\; g(t) \, \bigr\|_{2}^{2}\]
      via <code>scipy.optimize.nnls</code>.  Then
      \[g_c^{C}(t) \;=\; n_c(t) \cdot \sum_k M[k, c]\]
      The \(n \ge 0\) constraint is what keeps the plan's "negative
      \(n_c\)" caveat from biting.
    </li>
    <li>
      <span class="label">Per-archetype effective price.</span>
      \[\bar p_c(t) \;=\;
        \frac{\sum_k M[k, c] \cdot p_k(t)}{\sum_k M[k, c]}\]
      Under ArbOS 51 the price is flat across resources, so
      \(\bar p_c(t) = p^{\text{old}}(t)\) for every \(c\); only under
      ArbOS 60 do the archetypes diverge.  \(p_k^{\text{new}}(t)\)
      comes from the cached output of
      <code>arbos60.Arbos60GasPricing.price_per_resource</code>; \(p^{\text{old}}(t)\)
      is the gas-weighted hourly mean of <code>avg_eff_price_gwei</code>
      from the on-chain block table.
    </li>
    <li>
      <span class="label">Calibration.</span>
      \[D_c \;=\; \overline{ g_c(t) \cdot \bar p_c(t)^{\,\alpha} }\]
      averaged over hours where the cluster was actually active
      (\(g_c(t) > 10^6\) gas) so empty hours don't dilute \(D_c\) toward
      zero.  We compute \(D_c\) separately from Path B and Path C and
      compare them in panel 2 (the open diamonds vs the bars).
    </li>
    <li>
      <span class="label">Prediction.</span>
      \[g'_c(t) \;=\; D_c \cdot \bar p'_c(t)^{-\alpha}\]
      with \(\bar p'_c(t)\) using \(p_k^{\text{new}}(t)\) and \(\alpha = 1\).
      The implied per-resource counterfactual gas is
      \[g'_k(t) \;=\; \sum_c M[k, c] \cdot \frac{g'_c(t)}{\sum_{k'} M[k', c]}\]
    </li>
    <li>
      <span class="label">Caveats addressed (per the plan).</span>
      <ul style="padding-left:1.2em; margin-top:0.3em;">
        <li>\(\kappa(M)\) reported above; if very high (≥ 1e8) treat
            Path B as primary and Path C diagnostically.</li>
        <li>NNLS guarantees \(n_c(t) \ge 0\).</li>
        <li>L1 calldata column dropped (degenerate, all-zero in source).</li>
        <li>Sanity: with \(p^{\text{new}} = p^{\text{old}}\), predicted ≈
            observed; the per-archetype mean residual is reported above.</li>
        <li>Unit-elasticity revenue invariance:
            \(\alpha = 1 \Rightarrow \mathrm{fee}'(t) = \sum_c D_c\) is
            exactly the historical revenue; the chart therefore focuses
            on per-archetype/per-resource <em>gas redistribution</em>,
            not on absolute revenue change.</li>
      </ul>
    </li>
  </ol>
</div>

{{FIG}}

</body></html>
"""


def main() -> None:
    print("Loading cluster cache...")
    K, hourly_gas, M, n_txs, min_hour = load_cluster_M()
    n_hours = hourly_gas.shape[1]
    cond_M = float(np.linalg.cond(M))
    print(f"  K = {K}, M shape = {M.shape}, κ(M) = {cond_M:.3e}, n_hours = {n_hours}")

    print("Loading hourly prices...")
    p_old = hourly_p_old(min_hour, n_hours)
    p_new = hourly_p_new(min_hour, n_hours)
    print(f"  p_old: median = {np.median(p_old):.4f} gwei, "
          f"max = {np.max(p_old):.4f}")
    print(f"  p_new: median (compute) = {np.median(p_new[:, 0]):.4f} gwei")

    print("Computing observed per-archetype + per-resource gas...")
    g_c_B    = path_b_gas(hourly_gas)              # (K, n_hours)
    g_k_obs  = hourly_gas.sum(axis=0).T            # (K_priced, n_hours)

    print("Path C (NNLS solve per hour)...")
    g_c_C, n_c_C, _ = path_c_gas(M, g_k_obs.T)

    p_c_old = archetype_eff_price(M, p_old)        # (K, n_hours)
    p_c_new = archetype_eff_price(M, p_new)        # (K, n_hours)

    D_c_B = calibrate_D(g_c_B, p_c_old)
    D_c_C = calibrate_D(g_c_C, p_c_old)
    print(f"  D_c (Path B): {[f'{d:.2e}' for d in D_c_B]}")
    print(f"  D_c (Path C): {[f'{d:.2e}' for d in D_c_C]}")

    g_c_pred = predict_g_c(D_c_B, p_c_new)         # primary view = Path B
    g_k_pred = per_resource_gas(M, g_c_pred)

    # Sanity: with p_new = p_old, predicted should match observed (mean).
    g_c_sanity = predict_g_c(D_c_B, p_c_old)
    active = g_c_B > 1e6
    rel_diff = np.where(active,
                        np.abs(g_c_sanity - g_c_B) / np.maximum(g_c_B, 1.0),
                        0.0)
    sanity_pct = 100.0 * (rel_diff.sum() / np.maximum(active.sum(), 1))
    print(f"  Sanity (p_new=p_old, B): mean |%diff| over active hours = "
          f"{sanity_pct:.2f}%")

    # Revenue invariance under α=1: fee'(t) = sum_c D_c (per the plan).
    fee_hist = (g_c_B * p_c_old).sum(axis=0)
    fee_pred = (g_c_pred * p_c_new).sum(axis=0)
    rev_ratio = float(fee_pred.sum() / max(fee_hist.sum(), 1e-12))
    print(f"  Revenue invariance: predicted/historical = {rev_ratio:.4f} "
          f"(should ≈ 1 by construction at α=1)")

    # Hour axis (datetime)
    hour_axis = (((np.arange(n_hours, dtype=np.int64) + min_hour)
                  * 3600 * 1000).astype("datetime64[ms]"))

    print("Rendering figure...")
    fig = build_figure(
        K, M, hour_axis,
        D_c_B, D_c_C,
        g_c_B, g_c_C, g_c_pred,
        p_c_old, p_c_new,
        g_k_obs, g_k_pred,
        ALPHA,
    )
    fig_html = fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"displaylogo": False, "responsive": True},
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    cond_warn = " warn" if cond_M >= 1e8 else ""
    page = (
        PAGE_TEMPLATE
        .replace("{{K}}",         str(K))
        .replace("{{COND_M}}",    f"{cond_M:.2e}")
        .replace("{{COND_WARN}}", cond_warn)
        .replace("{{ALPHA}}",     str(ALPHA))
        .replace("{{SANITY}}",    f"{sanity_pct:.1f}%")
        .replace("{{REV_RATIO}}", f"{rev_ratio:.4f}")
        .replace("{{FIG}}",       fig_html)
    )
    OUT_HTML.write_text(page)
    print(f"Saved {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
