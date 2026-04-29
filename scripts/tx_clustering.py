"""
Multi-K, multi-algorithm clustering of Arbitrum transactions by their
per-resource gas mix, with an internal-validation diagnostic dashboard.

Two feature sets clustered side-by-side:
  • CLR — centered log-ratio of the per-tx weight vector.
          Answers: "what TYPE of tx is this?"  (mix only, scale-invariant)
  • LOG — log1p(g_k) standardized to z-score with first-chunk bootstrap stats.
          Answers: "what KIND of tx including its size?"  (mix + magnitude)

Four algorithms per feature set:
  • MiniBatchKMeans (baseline) — fit in a single streaming pass on FULL data
  • GaussianMixture                — sample-only (no sklearn streaming variant)
  • BisectingKMeans                — sample-only
  • HDBSCAN                        — sample-only, no fixed K (natural count)

Internal validation metrics on a 10 K subset:
  • Silhouette (higher better)
  • Calinski-Harabasz (higher better)
  • Davies-Bouldin (lower better)
  • Inertia (KMeans elbow, lower better)

Output: single HTML dashboard with side-by-side t-SNE scatters, 8 metric
diagnostic curves (4 metrics × 2 feature sets), per-cluster centroid bars,
and cluster-size distribution.

Run:
    python scripts/tx_clustering.py
    python scripts/tx_clustering.py --k 4 --algo kmeans --features clr

Output:
    figures/clustering.html
"""

from __future__ import annotations

import argparse
import pathlib
import time
from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import (
    BisectingKMeans,
    HDBSCAN,
    MiniBatchKMeans,
)
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


# ── Paths ───────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
MULTIGAS_DIR   = _HERE.parent / "data" / "multigas_usage_extracts"
BLOCKS_PARQUET = _HERE.parent / "data" / "onchain_blocks_transactions" / "per_block.parquet"
OUT_HTML       = _HERE.parent / "figures" / "clustering.html"

# ── Hyperparameters ─────────────────────────────────────────────────────────
K_RANGE        = list(range(2, 9))   # 2..8
N_SAMPLE       = 30_000              # reservoir-sampled rows for sample-only algos
N_TSNE         = 12_000              # rows embedded by t-SNE (per feature set)
N_METRIC       = 10_000              # rows used to compute O(N²) silhouette
N_HDB          = 15_000              # rows used by HDBSCAN
CHUNKSIZE      = 1_000_000           # rows per chunk
MBK_BATCH      = 8_192               # MiniBatchKMeans inner batch
RNG_SEED       = 42
MIN_PRICED_GAS = 30_000              # drop trivially small txs (degenerate corner)
CLR_EPS        = 1e-3                # additive smoothing for CLR log(0)
HDB_MIN_CLUSTER = 200                # ~1.3% of 15K — smallest meaningful cluster
# Conservative upper bound used for reservoir-fraction sizing. Exact row
# counts are inferred at runtime from the parquet metadata.
TOTAL_TX_ROWS_DEFAULT = 100_000_000

WEIGHT_COLS = ["w_c", "w_sw", "w_sr", "w_sg", "w_hg", "w_l2", "w_l1"]
CLR_COLS    = ["x_c", "x_sw", "x_sr", "x_sg", "x_hg", "x_l2", "x_l1"]
LOG_COLS    = ["l_c", "l_sw", "l_sr", "l_sg", "l_hg", "l_l2", "l_l1"]
RESOURCE_LABELS = {
    "w_c":  "Computation",
    "w_sw": "Storage Write",
    "w_sr": "Storage Read",
    "w_sg": "Storage Growth",
    "w_hg": "History Growth",
    "w_l2": "L2 Calldata",
    "w_l1": "L1 Calldata",
}
RESOURCE_COLORS = {
    "Computation":     "#1f77b4",
    "Storage Write":   "#2ca02c",
    "Storage Read":    "#98df8a",
    "Storage Growth":  "#d62728",
    "History Growth":  "#ff7f0e",
    "L2 Calldata":     "#9467bd",
    "L1 Calldata":     "#e377c2",
}
ALGO_COLORS = {
    "kmeans":    "#1f77b4",
    "gmm":       "#ff7f0e",
    "bk":        "#2ca02c",
    "hdbscan":   "#d62728",
}
ALGO_LABEL = {
    "kmeans":  "MiniBatchKMeans",
    "gmm":     "GMM (diag)",
    "bk":      "BisectingKMeans",
    "hdbscan": "HDBSCAN",
}


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--k", type=int, default=None,
                   help="Override the auto-selected K (e.g. 4)")
    p.add_argument("--algo", choices=["kmeans", "gmm", "bk", "hdbscan"], default=None,
                   help="Override the auto-selected algorithm")
    p.add_argument("--features", choices=["clr", "log"], default=None,
                   help="Override the highlighted feature set in the dashboard")
    return p.parse_args()


# ── Compositional / scaling utilities ───────────────────────────────────────
def _clr_transform(W: np.ndarray, eps: float = CLR_EPS) -> np.ndarray:
    """
    Centered log-ratio: x_k = log(w_k + eps) - mean_k log(w_k + eps).
    Maps the simplex into ℝ^d so euclidean distance becomes Aitchison
    distance (the right geometry for proportions).
    """
    L = np.log(W + eps)
    return L - L.mean(axis=1, keepdims=True)


def _featurize_chunk(
    chunk: pd.DataFrame,
    log_mean: np.ndarray | None,
    log_std: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    For one chunk: drop tiny txs, compute weights + CLR + log1p(gas) features.

    `log_mean`/`log_std` are bootstrap stats from the first chunk; if None,
    log features are returned un-standardized (caller handles bootstrapping).

    Returns (sample_df, X_clr, X_log) where X_clr/X_log are float64 arrays
    aligned with sample_df rows.
    """
    g_c   = (chunk["computation"] + chunk["wasmComputation"]).astype(np.float64)
    g_sw  = chunk["storageAccessWrite"].astype(np.float64)
    g_sr  = chunk["storageAccessRead"].astype(np.float64)
    g_sg  = chunk["storageGrowth"].astype(np.float64)
    g_hg  = chunk["historyGrowth"].astype(np.float64)
    g_l2  = chunk["l2Calldata"].astype(np.float64)
    g_l1  = chunk["l1Calldata"].astype(np.float64)
    total = g_c + g_sw + g_sr + g_sg + g_hg + g_l2 + g_l1
    keep  = total >= MIN_PRICED_GAS
    if not keep.any():
        empty = pd.DataFrame()
        z = np.empty((0, len(WEIGHT_COLS)))
        return empty, z, z

    inv_t = 1.0 / total[keep].to_numpy()
    G = np.column_stack([
        g_c[keep].to_numpy(), g_sw[keep].to_numpy(), g_sr[keep].to_numpy(),
        g_sg[keep].to_numpy(), g_hg[keep].to_numpy(), g_l2[keep].to_numpy(),
        g_l1[keep].to_numpy(),
    ])
    W = G * inv_t[:, None]
    X_clr = _clr_transform(W)
    L = np.log1p(G)
    X_log = (L - log_mean) / log_std if log_mean is not None else L

    df = pd.DataFrame({
        **{WEIGHT_COLS[i]: W[:, i] for i in range(len(WEIGHT_COLS))},
        **{CLR_COLS[i]:    X_clr[:, i] for i in range(len(CLR_COLS))},
        **{LOG_COLS[i]:    X_log[:, i] for i in range(len(LOG_COLS))},
        "gas_total": total[keep].to_numpy(),
        "tx_hash":   chunk["tx_hash"][keep].to_numpy(),
        "block":     chunk["block"][keep].to_numpy(),
    })
    return df.reset_index(drop=True), X_clr, X_log


# ── Streaming I/O ───────────────────────────────────────────────────────────
def _per_tx_files() -> list[pathlib.Path]:
    """All `per_tx.parquet` files under MULTIGAS_DIR — one per month."""
    return sorted(MULTIGAS_DIR.glob("*/per_tx.parquet"))


def _iter_chunks(paths: list[pathlib.Path]) -> Iterator[pd.DataFrame]:
    """Yield CHUNKSIZE-row pandas frames from each per-tx parquet in turn."""
    import pyarrow.parquet as pq

    cols = [
        "block", "tx_hash",
        "computation", "wasmComputation",
        "historyGrowth",
        "storageAccessRead", "storageAccessWrite",
        "storageGrowth", "l2Calldata", "l1Calldata",
    ]
    for path in paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=CHUNKSIZE, columns=cols):
            yield batch.to_pandas()


def _total_rows(paths: list[pathlib.Path]) -> int:
    """Cheap row-count summed across every parquet — for reservoir sizing."""
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)


# ── Main streaming pass ─────────────────────────────────────────────────────
def stream_fit_and_sample(
    paths: list[pathlib.Path],
    rng: np.random.Generator,
) -> tuple[
    pd.DataFrame,
    dict[int, MiniBatchKMeans],   # CLR kmeans bank
    dict[int, MiniBatchKMeans],   # log kmeans bank
    int,
    np.ndarray,                   # log_mean (for downstream z-scoring)
    np.ndarray,                   # log_std
]:
    """
    Single streaming pass that:
      • bootstraps log-feature mean/std from the FIRST chunk
      • partial_fit's 14 MiniBatchKMeans models (K=2..8 × {CLR, log})
      • reservoir-samples ~N_SAMPLE featurized rows for sample-only work

    Returns (sample_df, mbk_clr_models, mbk_log_models, n_priced_seen,
             log_mean, log_std).
    """
    mbk_clr = {
        K: MiniBatchKMeans(
            n_clusters=K, random_state=RNG_SEED, batch_size=MBK_BATCH,
            n_init=10, max_iter=300, reassignment_ratio=0.01,
        )
        for K in K_RANGE
    }
    mbk_log = {
        K: MiniBatchKMeans(
            n_clusters=K, random_state=RNG_SEED, batch_size=MBK_BATCH,
            n_init=10, max_iter=300, reassignment_ratio=0.01,
        )
        for K in K_RANGE
    }

    total_rows = _total_rows(paths) or TOTAL_TX_ROWS_DEFAULT
    keep_frac = min(1.0, (N_SAMPLE * 1.4) / max(total_rows, 1))
    sample_parts: list[pd.DataFrame] = []
    n_priced = 0
    n_scanned = 0
    log_mean = log_std = None
    t0 = time.time()

    for i, chunk in enumerate(_iter_chunks(paths)):
        # Bootstrap log scaling stats from the first chunk only. Use the
        # un-standardized log values to derive mean/std, then re-featurize
        # the chunk with the stats applied so it matches all subsequent
        # chunks' scaling.
        if log_mean is None:
            df0, X_clr0, L_raw = _featurize_chunk(chunk, None, None)
            if len(df0):
                log_mean = L_raw.mean(axis=0)
                log_std  = L_raw.std(axis=0).clip(min=1e-9)
                feats, X_clr, X_log = _featurize_chunk(chunk, log_mean, log_std)
            else:
                feats, X_clr, X_log = df0, X_clr0, L_raw
        else:
            feats, X_clr, X_log = _featurize_chunk(chunk, log_mean, log_std)

        n_scanned += len(chunk)
        n_priced  += len(feats)
        if len(feats):
            for K in K_RANGE:
                mbk_clr[K].partial_fit(X_clr)
                mbk_log[K].partial_fit(X_log)
            mask = rng.random(len(feats)) < keep_frac
            if mask.any():
                sample_parts.append(feats.loc[mask])

        elapsed = time.time() - t0
        rate = n_scanned / max(elapsed, 1e-6)
        print(
            f"  chunk {i+1:>3d}: scanned {n_scanned:>11,}, "
            f"kept (priced) {n_priced:>10,}, "
            f"sampled {sum(len(p) for p in sample_parts):>7,}, "
            f"{rate/1e6:.2f} M rows/s"
        )

    sample = (
        pd.concat(sample_parts, ignore_index=True)
        if sample_parts else pd.DataFrame()
    )
    if len(sample) > N_SAMPLE:
        sample = sample.sample(
            n=N_SAMPLE, random_state=int(rng.integers(0, 1 << 31)),
        ).reset_index(drop=True)

    return sample, mbk_clr, mbk_log, n_priced, log_mean, log_std


# ── Sample-only algorithm fitting ───────────────────────────────────────────
def fit_sample_algos(
    sample: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[tuple[str, str, int], np.ndarray]:
    """
    Fit GMM + BisectingKMeans (each K=2..8) and HDBSCAN (single point) on
    the reservoir sample for both CLR and log feature sets.

    Returns dict keyed by (algo, feature_set, K) → label array.
    HDBSCAN's natural-K is used as its key.
    """
    out: dict[tuple[str, str, int], np.ndarray] = {}
    feature_arrays = {
        "clr": sample[CLR_COLS].to_numpy(dtype=np.float64),
        "log": sample[LOG_COLS].to_numpy(dtype=np.float64),
    }

    for fset, X in feature_arrays.items():
        for K in K_RANGE:
            print(f"  fitting GMM (K={K}, features={fset})...")
            gmm = GaussianMixture(
                n_components=K, covariance_type="diag", n_init=1,
                max_iter=100, random_state=RNG_SEED, init_params="k-means++",
            )
            out[("gmm", fset, K)] = gmm.fit_predict(X)

            print(f"  fitting BisectingKMeans (K={K}, features={fset})...")
            bk = BisectingKMeans(
                n_clusters=K, random_state=RNG_SEED, n_init=1,
                init="k-means++",
            )
            out[("bk", fset, K)] = bk.fit_predict(X)

        # HDBSCAN: no K parameter; runs on a 15K subset for speed.
        idx = rng.choice(len(X), size=min(N_HDB, len(X)), replace=False)
        Xh  = X[idx]
        print(f"  fitting HDBSCAN (features={fset}, n={len(Xh):,})...")
        hdb = HDBSCAN(
            min_cluster_size=HDB_MIN_CLUSTER, min_samples=20, n_jobs=-1,
        )
        labels_sub = hdb.fit_predict(Xh)
        # Project subset labels back onto the full sample using nearest
        # centroid in the same feature space, so we have a label for every
        # row in `sample` (matches the other algos' shape).
        labels = _propagate_labels(X, Xh, labels_sub)
        n_clusters = len(set(labels) - {-1})
        out[("hdbscan", fset, max(n_clusters, 1))] = labels

    return out


def _propagate_labels(
    X_full: np.ndarray, X_sub: np.ndarray, labels_sub: np.ndarray,
) -> np.ndarray:
    """Nearest-centroid label propagation from the HDBSCAN subset back to
    the full sample. Noise points (-1) get their own label."""
    uniq = sorted(set(labels_sub) - {-1})
    if not uniq:
        return np.full(len(X_full), -1, dtype=np.int64)
    centroids = np.vstack([
        X_sub[labels_sub == lab].mean(axis=0) for lab in uniq
    ])
    # squared euclidean distance — argmin
    d2 = ((X_full[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    return np.array([uniq[i] for i in nearest], dtype=np.int64)


# ── Internal validation metrics ─────────────────────────────────────────────
def compute_metrics(
    sample: pd.DataFrame,
    mbk_clr: dict[int, MiniBatchKMeans],
    mbk_log: dict[int, MiniBatchKMeans],
    sample_labels: dict[tuple[str, str, int], np.ndarray],
    rng: np.random.Generator,
) -> dict[tuple[str, str, int], dict[str, float]]:
    """
    For each (algo, feature_set, K), compute silhouette, Calinski-Harabasz,
    Davies-Bouldin, and inertia (KMeans only) on a fixed N_METRIC subset.
    """
    feature_arrays = {
        "clr": sample[CLR_COLS].to_numpy(dtype=np.float64),
        "log": sample[LOG_COLS].to_numpy(dtype=np.float64),
    }
    n = len(sample)
    if n > N_METRIC:
        idx = rng.choice(n, size=N_METRIC, replace=False)
    else:
        idx = np.arange(n)

    metrics: dict[tuple[str, str, int], dict[str, float]] = {}

    # Predict KMeans labels first (these were trained on FULL data).
    kmeans_labels: dict[tuple[str, int], np.ndarray] = {}
    for fset, X in feature_arrays.items():
        bank = mbk_clr if fset == "clr" else mbk_log
        for K in K_RANGE:
            kmeans_labels[(fset, K)] = bank[K].predict(X)
            sample_labels[("kmeans", fset, K)] = kmeans_labels[(fset, K)]

    for (algo, fset, K), labels in sample_labels.items():
        Xs = feature_arrays[fset][idx]
        ls = labels[idx]
        # HDBSCAN may put a chunk of points in -1 (noise). Filter for metrics.
        mask = ls != -1
        if mask.sum() < 50 or len(set(ls[mask])) < 2:
            continue
        Xv, lv = Xs[mask], ls[mask]
        try:
            sil = float(silhouette_score(Xv, lv, sample_size=min(5000, len(lv)),
                                         random_state=RNG_SEED))
            ch  = float(calinski_harabasz_score(Xv, lv))
            db  = float(davies_bouldin_score(Xv, lv))
        except Exception as e:
            print(f"  metric error for {algo}/{fset}/K={K}: {e}")
            continue

        m: dict[str, float] = {"silhouette": sil, "ch": ch, "db": db}
        if algo == "kmeans":
            bank = mbk_clr if fset == "clr" else mbk_log
            m["inertia"] = float(bank[K].inertia_)
        metrics[(algo, fset, K)] = m

    return metrics


def select_winner(
    metrics: dict[tuple[str, str, int], dict[str, float]],
    feature_set: str,
) -> tuple[str, int, dict[str, float]]:
    """
    Rank-average across silhouette (↑), CH (↑), DB (↓). Returns
    (algo, K, scores). `K` is 0 for HDBSCAN's natural-K (label preserved
    in `metrics` dict key — caller maps back).
    """
    rows = [
        (algo, K, m) for (algo, fs, K), m in metrics.items()
        if fs == feature_set
    ]
    if not rows:
        return "kmeans", K_RANGE[0], {}

    sil = np.array([r[2]["silhouette"] for r in rows])
    ch  = np.array([r[2]["ch"] for r in rows])
    db  = np.array([r[2]["db"] for r in rows])

    # Higher is better → ascending rank. For DB, lower is better → invert.
    rank_sil = sil.argsort().argsort()                       # 0 = worst
    rank_ch  = ch.argsort().argsort()
    rank_db  = (-db).argsort().argsort()                     # invert
    score    = rank_sil + rank_ch + rank_db                  # higher = better

    best = int(np.argmax(score))
    algo, K, m = rows[best]
    return algo, K, m


# ── t-SNE ───────────────────────────────────────────────────────────────────
def embed_tsne(
    X: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Barnes-Hut 2-D t-SNE with input jitter (breaks duplicate-point
    artifacts on compositional data)."""
    Xj = X + rng.normal(0.0, 0.02, size=X.shape)
    tsne = TSNE(
        n_components=2, random_state=RNG_SEED,
        perplexity=50, init="pca", learning_rate="auto",
        max_iter=1000, method="barnes_hut",
    )
    return tsne.fit_transform(Xj)


# ── Cluster labelling ───────────────────────────────────────────────────────
def compute_cluster_timeseries(
    sample: pd.DataFrame, labels: np.ndarray,
) -> tuple[pd.DataFrame, list]:
    """
    For each (cluster, hour), compute the share (0..1) of total gas going to
    each resource on the reservoir sample. Joins `block` → `block_time` from
    the per-block revenue csv and truncates to hour.

    Returns (long_df, sorted_hours) where long_df has columns
    [cluster, hour, share_c, share_sw, share_sr, share_sg, share_hg, share_l2]
    with the 6 share columns summing to 1 per row.
    """
    bt = pd.read_parquet(BLOCKS_PARQUET, columns=["block_number", "block_time"])
    bt["block_time"] = pd.to_datetime(bt["block_time"])
    bt["hour"] = bt["block_time"].dt.floor("h")

    s = sample[["block", "gas_total"] + WEIGHT_COLS].copy()
    s["cluster"] = labels
    s = s.merge(
        bt[["block_number", "hour"]],
        left_on="block", right_on="block_number", how="left",
    ).dropna(subset=["hour"])

    # Reconstruct absolute gas per resource from weights × gas_total, then
    # aggregate to (cluster, hour) and normalize each row to a share that
    # sums to 1.0 across resources.
    gas_cols = []
    for w in WEIGHT_COLS:
        gcol = f"gas_{w[2:]}"   # w_c → gas_c
        s[gcol] = s[w] * s["gas_total"]
        gas_cols.append(gcol)

    grouped = (
        s.groupby(["cluster", "hour"], as_index=False)[gas_cols]
         .sum()
         .sort_values(["cluster", "hour"])
         .reset_index(drop=True)
    )
    row_total = grouped[gas_cols].sum(axis=1).clip(lower=1.0)
    for w, gcol in zip(WEIGHT_COLS, gas_cols):
        scol = f"share_{w[2:]}"
        grouped[scol] = grouped[gcol] / row_total
    keep_cols = ["cluster", "hour"] + [f"share_{w[2:]}" for w in WEIGHT_COLS]
    sorted_hours = sorted(grouped["hour"].unique())
    return grouped[keep_cols], sorted_hours


def cluster_label_from_weights(centroid_w: np.ndarray) -> str:
    order = np.argsort(centroid_w)[::-1]
    top1, top2 = order[0], order[1]
    p1, p2 = centroid_w[top1], centroid_w[top2]
    name1 = RESOURCE_LABELS[WEIGHT_COLS[top1]]
    name2 = RESOURCE_LABELS[WEIGHT_COLS[top2]]
    if p1 >= 0.80:
        return f"{name1}-heavy ({p1:.0%})"
    return f"{name1}+{name2} ({p1:.0%}/{p2:.0%})"


def volume_weighted_centroid(
    W: np.ndarray, gas_total: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """
    Volume-weighted resource share for txs in the cluster:
        share_k = Σ_tx g_{tx,k} / Σ_tx Σ_k g_{tx,k}

    This matches what the hourly stacked-bar time-series visually displays
    (gas summed by resource, normalized to 100%). Use this — not the
    unweighted mean of per-tx weights — when the label needs to align with
    the volume-share plots.
    """
    if not mask.any():
        return np.full(W.shape[1], np.nan)
    Wm = W[mask]
    gm = gas_total[mask]
    return (Wm * gm[:, None]).sum(axis=0) / gm.sum()


# ── Dashboard ───────────────────────────────────────────────────────────────
def build_dashboard(
    sample: pd.DataFrame,
    metrics: dict[tuple[str, str, int], dict[str, float]],
    sample_labels: dict[tuple[str, str, int], np.ndarray],
    winners: dict[str, tuple[str, int, dict[str, float]]],
    tsne_xy: dict[str, np.ndarray],
    n_priced: int,
    user_override: dict[str, str | int | None],
    ts_long: pd.DataFrame,
    ts_dates: list,
    clr_full_labels: np.ndarray,
) -> go.Figure:
    """
    4-row, 2-col layout:
      Row 1 (cols 1-2): t-SNE scatter, CLR | LOG
      Row 2 (cols 1-2): 4-metric grids per feature set (we collapse to 2
                         columns at the make_subplots level and lay out the
                         4 metrics horizontally inside annotations — each
                         column has its own metric-vs-K subplot stack
                         realised as 4 small subplots in a sub-row.)
    To keep things simple we render Row 2 as a *separate* row group of
    4 thin subplots × 2 columns = 8 total panels.
    """
    set_label = {"clr": "CLR features (mix only)",
                 "log": "Log-gas features (mix + magnitude)"}

    # 1 t-SNE winner + 1 t-SNE HDBSCAN + 4 metrics + 1 centroid + 1 sizes
    # + ⌈clr_K / 2⌉ time-series rows (one panel per CLR-winner cluster)
    clr_K = winners["clr"][1]
    n_ts_rows = (clr_K + 1) // 2
    n_rows = 1 + 1 + 4 + 1 + 1 + n_ts_rows
    titles: list[str] = []
    # Row 1: winner t-SNE
    for fset in ("clr", "log"):
        algo, K, m = winners[fset]
        titles.append(
            f"t-SNE — {set_label[fset]}<br>"
            f"<sub>winner: {ALGO_LABEL.get(algo, algo)} · K={K} · "
            f"sil={m.get('silhouette', float('nan')):.3f}, "
            f"CH={m.get('ch', float('nan')):,.0f}, "
            f"DB={m.get('db', float('nan')):.3f}</sub>"
        )
    # Row 2: HDBSCAN t-SNE
    for fset in ("clr", "log"):
        # Find the HDBSCAN entry for this feature set.
        hdb_K = next(
            (K for (a, fs, K) in sample_labels if a == "hdbscan" and fs == fset),
            None,
        )
        m = metrics.get(("hdbscan", fset, hdb_K), {})
        titles.append(
            f"t-SNE — HDBSCAN ({set_label[fset]})<br>"
            f"<sub>natural K={hdb_K} · "
            f"sil={m.get('silhouette', float('nan')):.3f}, "
            f"CH={m.get('ch', float('nan')):,.0f}, "
            f"DB={m.get('db', float('nan')):.3f}</sub>"
        )
    # Rows 3-6: 4 metrics × 2 columns (CLR / LOG)
    metric_titles = {
        "silhouette": "Silhouette (↑ better)",
        "ch":         "Calinski-Harabasz (↑ better)",
        "db":         "Davies-Bouldin (↓ better)",
        "inertia":    "KMeans inertia (↓ better)",
    }
    for metric_key in ("silhouette", "ch", "db", "inertia"):
        for fset in ("clr", "log"):
            titles.append(f"{metric_titles[metric_key]} — {fset.upper()}")
    # Row 7: centroid bar
    titles.append("Per-cluster mean weight (CLR winner)")
    titles.append("Per-cluster mean weight (LOG winner)")
    # Row 8: cluster sizes
    titles.append("Cluster sizes (CLR winner)")
    titles.append("Cluster sizes (LOG winner)")
    # Rows 9..(9+n_ts_rows-1): hourly resource-share stacked bars,
    # one panel per CLR-winner cluster (5 panels in 3 rows × 2 cols if K=5).
    # Cluster centroid labels — derived from the 12K t-SNE subset (length
    # matches `sample` arg). `clr_full_labels` is on the 30K reservoir and
    # is used only by the time-series aggregation, NOT for labelling.
    # Use volume-weighted centroids so labels match the % share plots.
    clr_algo, _, _ = winners["clr"]
    clr_labels_subset = sample_labels[(clr_algo, "clr", clr_K)]
    Ws_subset = sample[WEIGHT_COLS].to_numpy()
    gas_total_subset = sample["gas_total"].to_numpy()
    cluster_names_clr: dict[int, str] = {}
    for cidx in range(clr_K):
        mask = clr_labels_subset == cidx
        if not mask.any():
            cluster_names_clr[cidx] = f"cluster {cidx} (empty)"
        else:
            wcen = volume_weighted_centroid(Ws_subset, gas_total_subset, mask)
            cluster_names_clr[cidx] = (
                f"cluster {cidx} — {cluster_label_from_weights(wcen)}"
            )
    for ts_row in range(n_ts_rows):
        for col in range(2):
            cidx = ts_row * 2 + col
            if cidx < clr_K:
                titles.append(
                    f"{cluster_names_clr[cidx]} — hourly resource share (%)"
                )
            else:
                titles.append("")   # empty cell

    # Row-height ratios (Plotly normalizes them internally). Tuned so that
    # at total_height ≈ 4500 px each panel type lands at a usable size:
    #   t-SNE  ~440 px, metric ~170 px, bar  ~280 px, cluster-TS  ~310 px.
    row_heights = (
        [0.20, 0.20]                 # t-SNE rows
        + [0.08] * 4                 # metric rows
        + [0.13, 0.13]               # bar rows (centroid + size)
        + [0.14] * n_ts_rows         # cluster time-series rows
    )

    fig = make_subplots(
        rows=n_rows, cols=2,
        row_heights=row_heights,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        # 0.025 × (n_rows-1) gaps. With 11 rows that's 0.25 of the plot area
        # for spacing — anything larger squeezes panels to ~half height.
        vertical_spacing=0.025,
    )

    # ── Helper: render one t-SNE panel ──────────────────────────────────────
    Ws = sample[WEIGHT_COLS].to_numpy()
    gas_total_arr = sample["gas_total"].to_numpy()
    palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel

    def _plot_tsne_panel(fset: str, labels: np.ndarray, panel_row: int,
                         col: int, lg_prefix: str, show_centroid_in_name: bool):
        XY = tsne_xy[fset]
        unique = sorted(set(labels))
        for ci, lab in enumerate(unique):
            mask = labels == lab
            if mask.sum() == 0:
                continue
            if show_centroid_in_name and lab != -1:
                # Volume-weighted centroid → label matches the % share plots.
                wcen = volume_weighted_centroid(Ws, gas_total_arr, mask)
                name = (
                    f"{lab}: {cluster_label_from_weights(wcen)} "
                    f"(n={mask.sum():,})"
                )
            elif lab == -1:
                name = f"noise (n={mask.sum():,})"
            else:
                name = f"c{lab} (n={mask.sum():,})"
            color = "#888888" if lab == -1 else palette[ci % len(palette)]
            fig.add_trace(
                go.Scattergl(
                    x=XY[mask, 0], y=XY[mask, 1],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.65,
                                line=dict(width=0)),
                    name=name,
                    legendgroup=f"{lg_prefix}_{fset}",
                    legendgrouptitle_text=(
                        f"{lg_prefix.upper()} — {set_label[fset]}"
                        if ci == 0 else None
                    ),
                    showlegend=show_centroid_in_name,  # only winner in legend
                    hovertemplate=(
                        f"cluster {lab}<br>"
                        "tsne1=%{x:.2f} tsne2=%{y:.2f}<extra></extra>"
                    ),
                ),
                row=panel_row, col=col,
            )
        fig.update_xaxes(title_text="t-SNE 1", row=panel_row, col=col,
                         showgrid=False, zeroline=False)
        fig.update_yaxes(title_text="t-SNE 2", row=panel_row, col=col,
                         showgrid=False, zeroline=False)

    # Row 1: winner t-SNE; Row 2: HDBSCAN t-SNE.
    for col, fset in enumerate(("clr", "log"), start=1):
        algo, K, _ = winners[fset]
        _plot_tsne_panel(
            fset, sample_labels[(algo, fset, K)],
            panel_row=1, col=col,
            lg_prefix="winner", show_centroid_in_name=True,
        )
        hdb_K = next(
            (K for (a, fs, K) in sample_labels if a == "hdbscan" and fs == fset),
            None,
        )
        if hdb_K is not None:
            _plot_tsne_panel(
                fset, sample_labels[("hdbscan", fset, hdb_K)],
                panel_row=2, col=col,
                lg_prefix="hdb", show_centroid_in_name=False,
            )

    # ── Rows 3-6: metric curves per feature set ─────────────────────────────
    # X-axis is clamped to K_RANGE so K=2..8 isn't squashed by HDBSCAN's
    # natural-K (often 20+). HDBSCAN's score is shown as a corner annotation
    # instead of a far-right marker.
    legend_emitted: set[str] = set()
    for ridx, metric_key in enumerate(("silhouette", "ch", "db", "inertia"),
                                      start=3):
        for col, fset in enumerate(("clr", "log"), start=1):
            for algo in ("kmeans", "gmm", "bk"):
                xs, ys = [], []
                for K in K_RANGE:
                    m = metrics.get((algo, fset, K))
                    if m is None or metric_key not in m:
                        continue
                    xs.append(K); ys.append(m[metric_key])
                if not xs:
                    continue
                show_in_legend = (algo not in legend_emitted)
                if show_in_legend:
                    legend_emitted.add(algo)
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=ALGO_LABEL[algo],
                        line=dict(color=ALGO_COLORS[algo], width=1.5),
                        marker=dict(size=6),
                        legendgroup=f"metric_{algo}",
                        showlegend=show_in_legend,
                        hovertemplate=(
                            f"{ALGO_LABEL[algo]} K=%{{x}}: %{{y:.4g}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=ridx, col=col,
                )

            # HDBSCAN as corner annotation rather than off-axis marker.
            if metric_key != "inertia":
                hdb_entry = next(
                    ((K, m[metric_key]) for (a, fs, K), m in metrics.items()
                     if a == "hdbscan" and fs == fset and metric_key in m),
                    None,
                )
                if hdb_entry is not None:
                    K_nat, val = hdb_entry
                    # Position annotation in the panel's upper-right; xref/yref
                    # paper coords scoped to the subplot via axis ref.
                    fig.add_annotation(
                        text=(f"HDBSCAN<br>K_nat={K_nat}: "
                              f"{val:.3f}" if metric_key != "ch"
                              else f"HDBSCAN<br>K_nat={K_nat}: {val:,.0f}"),
                        xref=f"x{(ridx - 1) * 2 + col} domain",
                        yref=f"y{(ridx - 1) * 2 + col} domain",
                        x=0.98, y=0.98, xanchor="right", yanchor="top",
                        showarrow=False,
                        font=dict(size=10, color=ALGO_COLORS["hdbscan"]),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor=ALGO_COLORS["hdbscan"],
                        borderwidth=1, borderpad=3,
                    )

            fig.update_xaxes(
                title_text="K" if ridx == 6 else None,
                tickmode="linear", dtick=1,
                range=[K_RANGE[0] - 0.3, K_RANGE[-1] + 0.3],
                row=ridx, col=col,
            )

    # ── Row 7: per-cluster volume-weighted resource share (horizontal bar) ──
    # Uses the same volume-weighted projection as the cluster labels and the
    # hourly stacked bars, so all three views agree numerically.
    for col, fset in enumerate(("clr", "log"), start=1):
        algo, K, _ = winners[fset]
        labels = sample_labels[(algo, fset, K)]
        Ws = sample[WEIGHT_COLS].to_numpy()
        gt = sample["gas_total"].to_numpy()
        unique = sorted(set(labels))
        # Pre-compute per-cluster volume-weighted centroids.
        centroids = {
            u: volume_weighted_centroid(Ws, gt, labels == u) for u in unique
        }
        cum = np.zeros(len(unique))
        for j, w_col in enumerate(WEIGHT_COLS):
            label = RESOURCE_LABELS[w_col]
            xs = np.array([
                float(centroids[u][j]) if not np.isnan(centroids[u][j]) else 0.0
                for u in unique
            ])
            fig.add_trace(
                go.Bar(
                    y=[f"c{u}" for u in unique],
                    x=xs, base=cum, orientation="h",
                    name=label,
                    marker_color=RESOURCE_COLORS[label],
                    legendgroup="centroid",
                    showlegend=(col == 1),
                    hovertemplate=(
                        f"{label}: %{{x:.2f}}<extra></extra>"
                    ),
                ),
                row=7, col=col,
            )
            cum = cum + xs
        fig.update_xaxes(range=[0, 1.0], title_text="mean weight",
                         row=7, col=col)
        fig.update_yaxes(title_text="cluster", row=7, col=col)

    # ── Row 8: cluster size bars ────────────────────────────────────────────
    for col, fset in enumerate(("clr", "log"), start=1):
        algo, K, _ = winners[fset]
        labels = sample_labels[(algo, fset, K)]
        unique = sorted(set(labels))
        sizes = [int((labels == u).sum()) for u in unique]
        pct = [s / len(labels) * 100.0 for s in sizes]
        bar_colors = [
            "#888888" if u == -1 else
            (px.colors.qualitative.Bold + px.colors.qualitative.Pastel)
            [i % (len(px.colors.qualitative.Bold) + len(px.colors.qualitative.Pastel))]
            for i, u in enumerate(unique)
        ]
        fig.add_trace(
            go.Bar(
                x=[f"c{u}" for u in unique],
                y=pct,
                marker_color=bar_colors,
                showlegend=False,
                hovertemplate=(
                    "cluster %{x}: %{y:.1f}% (%{customdata:,} txs)"
                    "<extra></extra>"
                ),
                customdata=sizes,
            ),
            row=8, col=col,
        )
        fig.update_yaxes(title_text="% of sample", row=8, col=col)
        fig.update_xaxes(title_text="cluster", row=8, col=col)

    # ── Rows 9..: per-cluster hourly resource-share stacked bars ────────────
    # All panels share the same legend group ("clusters_ts") so the resource
    # legend appears once and toggles all clusters in sync.
    share_cols = [f"share_{w[2:]}" for w in WEIGHT_COLS]
    legend_resource_emitted: set[str] = set()
    for cidx in range(clr_K):
        ts_row_idx = cidx // 2
        col = cidx % 2 + 1
        panel_row = 1 + 1 + 4 + 1 + 1 + ts_row_idx + 1   # 8 + ts_row_idx + 1
        sub = ts_long[ts_long["cluster"] == cidx].sort_values("hour")
        if sub.empty:
            continue
        x_h = sub["hour"].to_numpy()
        cum = np.zeros(len(sub))
        for w, scol in zip(WEIGHT_COLS, share_cols):
            label = RESOURCE_LABELS[w]
            y_pct = sub[scol].to_numpy() * 100.0
            show_legend = label not in legend_resource_emitted
            if show_legend:
                legend_resource_emitted.add(label)
            fig.add_trace(
                go.Bar(
                    x=x_h, y=y_pct, base=cum, name=label,
                    marker_color=RESOURCE_COLORS[label],
                    marker_line_width=0,
                    legendgroup="clusters_ts",
                    legendgrouptitle_text=(
                        "Resource share (cluster time-series)"
                        if show_legend else None
                    ),
                    showlegend=show_legend,
                    hovertemplate=(
                        f"{label}: %{{y:.1f}}%<br>"
                        "%{x|%Y-%m-%d %H:00}<extra></extra>"
                    ),
                ),
                row=panel_row, col=col,
            )
            cum = cum + y_pct
        fig.update_yaxes(
            title_text="% share", range=[0, 100],
            row=panel_row, col=col,
        )
        fig.update_xaxes(
            title_text="hour (UTC)" if ts_row_idx == n_ts_rows - 1 else None,
            row=panel_row, col=col,
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    # Total height drives the absolute panel sizes; row_heights determine
    # the relative split. Spacing is 0.025 × (n_rows-1) of the plot area.
    # We size total_height so that panels feel comfortable: ≈ 320 px per
    # "average" row including its share of spacing & margins overhead.
    total_height = 320 * n_rows + 260

    title_text = (
        "<b>Per-tx clustering — multi-K / multi-algorithm dashboard</b>"
        f"<br><sub>{n_priced:,} priced txs streamed · {len(sample):,}-row "
        f"reservoir sample · K range {K_RANGE[0]}..{K_RANGE[-1]}"
    )
    if user_override.get("k") is not None or user_override.get("algo") is not None:
        title_text += (
            f" · <b>override:</b> "
            f"k={user_override.get('k')} algo={user_override.get('algo')}"
        )
    title_text += "</sub>"
    fig.update_layout(
        title=dict(
            text=title_text, x=0.0, xanchor="left",
            y=0.995, yanchor="top",
            font=dict(size=18, color="#111"),
        ),
        template="plotly_white",
        height=total_height,
        margin=dict(l=80, r=320, t=180, b=80),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            groupclick="togglegroup",
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="rgba(0,0,0,0.20)", borderwidth=1,
            font=dict(size=11),
        ),
        font=dict(size=12, color="#222"),
        hovermode="closest",
        barmode="stack",
    )

    # Bump subplot title font + nudge them up so they don't sit on top of
    # the first row of axes.
    for ann in fig.layout.annotations:
        if ann.text and ("t-SNE" in ann.text or "Silhouette" in ann.text
                         or "Calinski" in ann.text or "Davies" in ann.text
                         or "inertia" in ann.text or "Per-cluster" in ann.text
                         or "Cluster sizes" in ann.text):
            ann.font = dict(size=12, color="#111")
    fig.update_xaxes(
        showline=True, linewidth=1.0, linecolor="rgba(0,0,0,0.55)",
        mirror=True, ticks="outside",
    )
    fig.update_yaxes(
        showline=True, linewidth=1.0, linecolor="rgba(0,0,0,0.55)",
        mirror=True, ticks="outside",
    )
    return fig


# ── Driver ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    rng = np.random.default_rng(RNG_SEED)

    paths = _per_tx_files()
    if not paths:
        raise SystemExit(f"No per_tx parquets found under {MULTIGAS_DIR}")
    print(f"streaming {len(paths)} per_tx parquet(s) "
          f"({', '.join(p.parent.name for p in paths)}) → "
          f"14× MiniBatchKMeans + reservoir sample...")
    t_stream = time.time()
    sample, mbk_clr, mbk_log, n_priced, log_mean, log_std = stream_fit_and_sample(
        paths, rng,
    )
    print(f"  streaming pass: {time.time() - t_stream:.1f}s, "
          f"{n_priced:,} priced txs, sample {len(sample):,}")

    print("\nfitting sample-only algorithms (GMM, BK, HDBSCAN)...")
    t_sample = time.time()
    sample_labels = fit_sample_algos(sample, rng)
    print(f"  sample-only: {time.time() - t_sample:.1f}s")

    print("\ncomputing internal validation metrics...")
    t_metric = time.time()
    metrics = compute_metrics(sample, mbk_clr, mbk_log, sample_labels, rng)
    print(f"  metrics: {time.time() - t_metric:.1f}s")

    # Print metric table for inspection.
    print("\nMetric summary (per algo, feature set, K):")
    print(f"  {'algo':>10s} {'fset':>4s} {'K':>3s}   "
          f"{'sil':>6s}  {'CH':>10s}  {'DB':>6s}  {'inertia':>10s}")
    for key in sorted(metrics.keys()):
        algo, fset, K = key
        m = metrics[key]
        print(
            f"  {algo:>10s} {fset:>4s} {K:>3d}   "
            f"{m.get('silhouette', float('nan')):>6.3f}  "
            f"{m.get('ch', float('nan')):>10,.0f}  "
            f"{m.get('db', float('nan')):>6.3f}  "
            f"{m.get('inertia', float('nan')):>10,.0f}"
        )

    # Pick winners.
    winners: dict[str, tuple[str, int, dict[str, float]]] = {}
    for fset in ("clr", "log"):
        algo, K, m = select_winner(metrics, fset)
        winners[fset] = (algo, K, m)
        print(
            f"\nwinner [{fset}]: {ALGO_LABEL.get(algo, algo)} K={K}  "
            f"sil={m.get('silhouette', float('nan')):.3f}, "
            f"CH={m.get('ch', float('nan')):,.0f}, "
            f"DB={m.get('db', float('nan')):.3f}"
        )

    # Apply user override if any.
    user_override = {"k": args.k, "algo": args.algo, "features": args.features}
    if args.algo and args.k:
        for fset in ("clr", "log"):
            if (args.algo, fset, args.k) in metrics:
                winners[fset] = (args.algo, args.k, metrics[(args.algo, fset, args.k)])
        print(f"applied override → {args.algo} K={args.k}")

    # t-SNE for both feature sets.
    print(f"\nrunning t-SNE on {min(N_TSNE, len(sample)):,} txs × 2 feature sets...")
    if len(sample) > N_TSNE:
        idx = rng.choice(len(sample), size=N_TSNE, replace=False)
    else:
        idx = np.arange(len(sample))
    sample_t = sample.iloc[idx].reset_index(drop=True)
    # Subset labels too.
    sample_labels_t = {k: v[idx] for k, v in sample_labels.items()}

    tsne_xy: dict[str, np.ndarray] = {}
    for fset, cols in (("clr", CLR_COLS), ("log", LOG_COLS)):
        t0 = time.time()
        Xt = sample_t[cols].to_numpy(dtype=np.float64)
        tsne_xy[fset] = embed_tsne(Xt, rng)
        print(f"  t-SNE {fset}: {time.time() - t0:.1f}s")

    # Cluster time-series (CLR winner): hourly resource share per cluster.
    print("\ncomputing per-cluster hourly resource share (CLR winner)...")
    clr_algo, clr_K, _ = winners["clr"]
    clr_full_labels = sample_labels[(clr_algo, "clr", clr_K)]
    ts_long, ts_dates = compute_cluster_timeseries(sample, clr_full_labels)
    print(f"  {len(ts_long):,} (cluster, hour) bins, {len(ts_dates):,} hours")

    print("\nbuilding dashboard...")
    fig = build_dashboard(
        sample_t, metrics, sample_labels_t, winners, tsne_xy,
        n_priced, user_override,
        ts_long=ts_long, ts_dates=ts_dates, clr_full_labels=clr_full_labels,
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(OUT_HTML),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    print(f"saved {OUT_HTML}")


if __name__ == "__main__":
    main()
