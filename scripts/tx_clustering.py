"""
Single-pipeline transaction clustering by CLR (centered log-ratio)
of per-resource gas usage (7 resources including L1 calldata).

Pipeline:
  • Features: CLR(g_k + ε) per priced resource, k ∈
        {c, sw, sr, sg, hg, l2, l1}.  CLR is per-row (composition-only
        — magnitude is removed).  ε = 1.0 gas unit handles structural
        zeros (well below MIN_PRICED_GAS = 30 000).
  • Outlier trim: drop rows whose CLR coords fall outside the per-
        dimension [CLR_TAIL_PCT, 100 − CLR_TAIL_PCT] bounds (default
        0.1 % per tail).  Bounds are estimated from the first priced
        chunk (≈ 1 M rows is more than enough for tail estimates),
        frozen, and reused in pass 2 so the two passes see the same
        in-bounds set.
  • KMeans (MiniBatch) on CLR features over the FULL multigas dataset.
  • Sweep K = K_RANGE; pick K* using a composite of two standard
    internal-validation metrics (silhouette ↑, WCSS elbow ↓).
    The chosen K is locked via K_LOG; the silhouette/WCSS curves
    are rendered in the dashboard so the choice is defensible.

Two streaming passes, orchestrated as fresh subprocesses for memory
isolation:
  Pass 1 — partial_fit MiniBatchKMeans models for every K in K_RANGE
           on CLR features, reservoir-sample N_SAMPLE rows for sample-
           only work (silhouette metric, t-SNE, log-gas histograms,
           spam-label join).
  Pass 2 — predict cluster labels on every chunk and accumulate
           per-cluster aggregations on the FULL data: tx counts,
           hourly per-resource gas, log-gas histogram counts,
           spam-label distribution, volume-weighted resource centroid.

t-SNE runs on the reservoir sample (Barnes-Hut, O(N log N)).

Output: figures/clustering.html
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


# ── Paths ───────────────────────────────────────────────────────────────────
_HERE          = pathlib.Path(__file__).resolve().parent
MULTIGAS_DIR   = _HERE.parent / "data" / "multigas_usage_extracts"
BLOCKS_PARQUET = _HERE.parent / "data" / "onchain_blocks_transactions" / "per_block.parquet"
SPAM_LABELS    = _HERE.parent / "data" / "wallet_spam_classification.parquet"
CACHE_DIR      = _HERE.parent / "data" / "clustering_cache"
OUT_HTML       = _HERE.parent / "figures" / "clustering.html"

# ── Hyperparameters ─────────────────────────────────────────────────────────
K_RANGE        = list(range(3, 11))   # sweep span (3..10)
K_LOG          = 5                    # locked K — composite-best of sil / WCSS
N_SAMPLE       = 80_000               # reservoir sample (used for metrics + t-SNE)
N_TSNE         = 30_000               # t-SNE size — ~2-3 min Barnes-Hut
N_METRIC       = 8_000                # silhouette on this subset
CHUNKSIZE      = 1_000_000
PASS2_CHUNK    = 500_000
MBK_BATCH      = 8_192
RNG_SEED       = 42
MIN_PRICED_GAS = 30_000
CLR_EPS        = 1.0                 # gas-unit floor for zero-replacement in CLR
CLR_TAIL_PCT   = 0.1                 # drop rows outside per-dim [p, 100-p] CLR bounds
N_LOGGAS_BINS  = 40
N_SHARE_BINS   = 50                   # for median-share histogram (0..1, step 0.02)
SHARE_EDGES    = np.linspace(0.0, 1.0, N_SHARE_BINS + 1)

# 7 priced resources (incl L1 calldata).
RESOURCES = ["c", "sw", "sr", "sg", "hg", "l2", "l1"]
RESOURCE_LABEL = {
    "c":  "Computation",
    "sw": "Storage Write",
    "sr": "Storage Read",
    "sg": "Storage Growth",
    "hg": "History Growth",
    "l2": "L2 Calldata",
    "l1": "L1 Calldata",
}
RESOURCE_COLOR = {
    "Computation":     "#1f77b4",
    "Storage Write":   "#2ca02c",
    "Storage Read":    "#98df8a",
    "Storage Growth":  "#d62728",
    "History Growth":  "#ff7f0e",
    "L2 Calldata":     "#9467bd",
    "L1 Calldata":     "#e377c2",
}

GAS_COLS = [f"gas_{r}" for r in RESOURCES]


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-sample", type=int, default=N_SAMPLE,
                   help="Reservoir sample size (default 80k)")
    p.add_argument("--phase", choices=["fit", "aggregate", "plot", "all"],
                   default="all",
                   help="Single-phase entry point (for memory isolation).")
    return p.parse_args()


# ── I/O ─────────────────────────────────────────────────────────────────────
def _per_tx_files() -> list[pathlib.Path]:
    return sorted(MULTIGAS_DIR.glob("*/per_tx.parquet"))


PASS_COLS = [
    "block", "tx_sender",
    "computation", "wasmComputation",
    "historyGrowth",
    "storageAccessRead", "storageAccessWrite",
    "storageGrowth", "l2Calldata", "l1Calldata",
]


def _iter_chunks(paths: list[pathlib.Path], batch_size: int = CHUNKSIZE):
    """Yield pyarrow RecordBatch objects (avoid pandas allocation)."""
    import pyarrow.parquet as pq
    for path in paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=batch_size, columns=PASS_COLS):
            yield batch


def _total_rows(paths: list[pathlib.Path]) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)


def _featurize_batch(batch):
    """Featurize one arrow batch into raw numpy + arrow sender array.
    Returns (block_arr, sender_arrow, G, X_clr, Lg) or None when no rows
    survive the priced-gas filter.

    X_clr is the centered log-ratio of (g_k + CLR_EPS) — composition-only
    features (per-row centering removes magnitude).  Lg = log1p(g_k) is
    kept for the per-cluster log-gas histogram panel."""
    g_c  = (batch.column("computation").to_numpy(zero_copy_only=False)
            + batch.column("wasmComputation").to_numpy(zero_copy_only=False)
            ).astype(np.float64)
    g_sw = batch.column("storageAccessWrite").to_numpy(zero_copy_only=False).astype(np.float64)
    g_sr = batch.column("storageAccessRead").to_numpy(zero_copy_only=False).astype(np.float64)
    g_sg = batch.column("storageGrowth").to_numpy(zero_copy_only=False).astype(np.float64)
    g_hg = batch.column("historyGrowth").to_numpy(zero_copy_only=False).astype(np.float64)
    g_l2 = batch.column("l2Calldata").to_numpy(zero_copy_only=False).astype(np.float64)
    g_l1 = batch.column("l1Calldata").to_numpy(zero_copy_only=False).astype(np.float64)
    gtot = g_c + g_sw + g_sr + g_sg + g_hg + g_l2 + g_l1
    keep = gtot >= MIN_PRICED_GAS
    if not keep.any():
        return None
    G = np.column_stack([
        g_c[keep], g_sw[keep], g_sr[keep], g_sg[keep],
        g_hg[keep], g_l2[keep], g_l1[keep],
    ])
    Lg = np.log1p(G)

    # CLR: log(g_k + ε) − mean_k log(g_k + ε), per row.
    log_g = np.log(G + CLR_EPS)
    X_clr = log_g - log_g.mean(axis=1, keepdims=True)

    block_arr = batch.column("block").to_numpy(zero_copy_only=False)[keep].astype(np.int64)
    sender_arrow = batch.column("tx_sender").filter(pa.array(keep))
    return block_arr, sender_arrow, G, X_clr, Lg


# ── Pass 1: stream-fit MBK bank + reservoir sample ──────────────────────────
def _apply_clr_bounds(
    block_arr, sender_arrow, G, X_clr, Lg,
    clr_lo: np.ndarray, clr_hi: np.ndarray,
):
    """Drop rows whose CLR features fall outside the per-dim bounds."""
    keep = ((X_clr >= clr_lo) & (X_clr <= clr_hi)).all(axis=1)
    if keep.all():
        return block_arr, sender_arrow, G, X_clr, Lg, int(len(keep))
    if not keep.any():
        return None
    return (
        block_arr[keep],
        sender_arrow.filter(pa.array(keep)),
        G[keep],
        X_clr[keep],
        Lg[keep],
        int(keep.sum()),
    )


def stream_fit(
    paths: list[pathlib.Path], n_target: int, rng: np.random.Generator,
) -> tuple[
    dict[int, MiniBatchKMeans],
    np.ndarray, np.ndarray,
    pl.DataFrame, np.ndarray, np.ndarray, int, int,
]:
    """Pass 1: arrow batches → numpy featurize → partial_fit + reservoir
    sample.  Returns the MBK bank (one model per K in K_RANGE), the CLR
    per-dim bounds (clr_lo, clr_hi), the reservoir sample (polars
    DataFrame), the sample's CLR features `X_clr`, the raw log-gas `Lg`
    (for the histogram panel), the total priced-row count, and the
    in-bounds row count.  CLR bounds are estimated from the first priced
    chunk and frozen for the rest of the run."""
    mbk = {
        K: MiniBatchKMeans(
            n_clusters=K, random_state=RNG_SEED, batch_size=MBK_BATCH,
            n_init=5, max_iter=200, reassignment_ratio=0.01,
        )
        for K in K_RANGE
    }
    total_rows = _total_rows(paths) or n_target * 100
    keep_frac = min(1.0, (n_target * 1.4) / max(total_rows, 1))

    sample_arrow_parts: list[pa.Table] = []
    Xclr_parts: list[np.ndarray] = []
    Lg_parts:   list[np.ndarray] = []
    n_priced = 0
    n_kept_inbounds = 0
    n_scanned = 0
    clr_lo = clr_hi = None

    t0 = time.time()
    for i, batch in enumerate(_iter_chunks(paths, batch_size=CHUNKSIZE)):
        n_scanned += batch.num_rows
        out = _featurize_batch(batch)
        if out is None:
            continue
        block_arr, sender_arrow, G, X_clr, Lg = out
        n_priced += G.shape[0]

        if clr_lo is None:
            clr_lo = np.percentile(X_clr, CLR_TAIL_PCT,         axis=0)
            clr_hi = np.percentile(X_clr, 100.0 - CLR_TAIL_PCT, axis=0)
            print(f"  CLR bounds (per-dim, {CLR_TAIL_PCT}%/{100-CLR_TAIL_PCT}%):")
            for k, r in enumerate(RESOURCES):
                print(f"    {r:>3s}: [{clr_lo[k]:+.3f}, {clr_hi[k]:+.3f}]")

        trimmed = _apply_clr_bounds(
            block_arr, sender_arrow, G, X_clr, Lg, clr_lo, clr_hi,
        )
        if trimmed is None:
            continue
        block_arr, sender_arrow, G, X_clr, Lg, n_keep = trimmed
        n_kept_inbounds += n_keep

        for K in K_RANGE:
            mbk[K].partial_fit(X_clr)

        mask = rng.random(G.shape[0]) < keep_frac
        if mask.any():
            mask_arr = pa.array(mask)
            tbl = pa.table({
                "block":     pa.array(block_arr[mask], type=pa.int64()),
                "tx_sender": sender_arrow.filter(mask_arr),
                **{GAS_COLS[k]: pa.array(G[mask, k]) for k in range(7)},
                "gas_total": pa.array(G[mask].sum(axis=1)),
            })
            sample_arrow_parts.append(tbl)
            Xclr_parts.append(X_clr[mask])
            Lg_parts.append(Lg[mask])

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = n_scanned / max(elapsed, 1e-6)
            n_kept = sum(t.num_rows for t in sample_arrow_parts)
            print(f"  pass1 batch {i+1:>4d}: scanned {n_scanned:>11,}, "
                  f"priced {n_priced:>10,}, sampled {n_kept:>7,}  "
                  f"({rate/1e6:.2f} M rows/s)")

    sample_table = pa.concat_tables(sample_arrow_parts)
    sample = pl.from_arrow(sample_table)
    X_clr_sample = np.vstack(Xclr_parts)
    Lg_sample    = np.vstack(Lg_parts)
    sample_arrow_parts.clear(); Xclr_parts.clear(); Lg_parts.clear()

    if sample.height > n_target:
        idx = rng.choice(sample.height, size=n_target, replace=False)
        idx = np.sort(idx)
        sample = sample[idx]
        X_clr_sample = X_clr_sample[idx]
        Lg_sample    = Lg_sample[idx]

    return (mbk, clr_lo, clr_hi,
            sample, X_clr_sample, Lg_sample, n_priced, n_kept_inbounds)


# ── Diagnostics: silhouette + WCSS + CH + DB per K ──────────────────────────
def compute_diagnostics(
    mbk_bank: dict[int, MiniBatchKMeans],
    X_sample: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[int, float]]:
    """Per-K internal-validation metrics, computed on a fixed N_METRIC
    subset of the reservoir.  Two standard metrics — silhouette
    (higher is better) and WCSS / inertia (lower at any K, look for
    the elbow)."""
    sils, wcss = {}, {}
    idx = rng.choice(len(X_sample),
                     size=min(N_METRIC, len(X_sample)), replace=False)
    Xm  = X_sample[idx]
    for K in K_RANGE:
        labels_full = mbk_bank[K].predict(X_sample)
        labels      = labels_full[idx]
        sils[K] = float(silhouette_score(Xm, labels))
        wcss[K] = float(mbk_bank[K].inertia_)
        print(f"    K={K}: sil={sils[K]:.3f}  WCSS={wcss[K]:,.0f}")
    return {"silhouette": sils, "wcss": wcss}


def composite_best_k(metrics: dict[str, dict[int, float]]) -> int:
    """Rank-aggregate silhouette ↑ and WCSS ↓.  Each metric ranks K_RANGE
    1..n; sum the ranks; pick the K with the best total.  This is a
    lightweight consensus pick, not a ground truth — the dashboard plots
    both curves so the user can verify."""
    Ks = sorted(metrics["silhouette"])
    sil  = np.array([metrics["silhouette"][K] for K in Ks])
    wcss = np.array([metrics["wcss"][K]       for K in Ks])

    # Higher = better; argsort then argsort gives ranks (0 = worst).
    rank_sil  = sil.argsort().argsort()
    rank_wcss = (-wcss).argsort().argsort()    # lower better → invert
    score = rank_sil + rank_wcss
    return Ks[int(np.argmax(score))]


# ── Spam labels & block→hour map ────────────────────────────────────────────
SPAM_NOT     = "Not spam"
SPAM_VOL     = "High volume"
SPAM_REV     = "High revert"
SPAM_BOTH    = "High volume + revert"
SPAM_UNKNOWN = "Unknown"
SPAM_ORDER  = [SPAM_NOT, SPAM_VOL, SPAM_REV, SPAM_BOTH, SPAM_UNKNOWN]
SPAM_COLOR  = {
    SPAM_NOT:     "#2ca02c",
    SPAM_VOL:     "#ff7f0e",
    SPAM_REV:     "#d62728",
    SPAM_BOTH:    "#8c2d04",
    SPAM_UNKNOWN: "#cccccc",
}
SPAM_CODE = {
    SPAM_NOT: 0, SPAM_VOL: 1, SPAM_REV: 2, SPAM_BOTH: 3, SPAM_UNKNOWN: 4,
}
N_SPAM_CODES = 5


def load_spam_table() -> pl.DataFrame:
    """Polars DataFrame [tx_sender, spam_code] for fast/low-mem joins."""
    if not SPAM_LABELS.exists():
        print(f"  spam labels not found at {SPAM_LABELS}")
        return pl.DataFrame({
            "tx_sender": pl.Series([], dtype=pl.Utf8),
            "spam_code": pl.Series([], dtype=pl.UInt8),
        })
    df = pl.read_parquet(
        SPAM_LABELS, columns=["address", "n_days_high_vol", "n_days_high_rev"],
    )
    df = df.with_columns(
        pl.col("address").str.to_lowercase().alias("tx_sender")
    ).with_columns(
        pl.when((pl.col("n_days_high_vol") > 0)
                & (pl.col("n_days_high_rev") > 0))
          .then(SPAM_CODE[SPAM_BOTH])
        .when(pl.col("n_days_high_vol") > 0)
          .then(SPAM_CODE[SPAM_VOL])
        .when(pl.col("n_days_high_rev") > 0)
          .then(SPAM_CODE[SPAM_REV])
        .otherwise(SPAM_CODE[SPAM_NOT])
        .cast(pl.UInt8)
        .alias("spam_code")
    ).select(["tx_sender", "spam_code"]).unique(subset=["tx_sender"])
    summary = df.group_by("spam_code").len().sort("spam_code")
    counts = {int(r["spam_code"]): int(r["len"]) for r in summary.iter_rows(named=True)}
    print(f"  spam labels: {df.height:,} wallets — "
          f"vol={counts.get(SPAM_CODE[SPAM_VOL], 0):,}, "
          f"rev={counts.get(SPAM_CODE[SPAM_REV], 0):,}, "
          f"both={counts.get(SPAM_CODE[SPAM_BOTH], 0):,}, "
          f"none={counts.get(SPAM_CODE[SPAM_NOT], 0):,}")
    return df


def load_block_to_hour() -> tuple[np.ndarray, int, int]:
    """Returns (hour_offsets, min_block, min_hour_int) where
    hour_offsets[block - min_block] = (epoch-hour - min_hour_int)."""
    df = pl.read_parquet(BLOCKS_PARQUET,
                         columns=["block_number", "block_time"])
    block_arr = df["block_number"].to_numpy().astype(np.int64)
    seconds = df["block_time"].cast(pl.Datetime("us")).dt.timestamp("ms").to_numpy() // 1000
    hours = (seconds // 3600).astype(np.int64)
    del df
    block_min = int(block_arr.min())
    block_max = int(block_arr.max())
    n = block_max - block_min + 1
    min_hour = int(hours.min())
    offsets = np.zeros(n, dtype=np.uint32)
    offsets[block_arr - block_min] = (hours - min_hour).astype(np.uint32)
    return offsets, block_min, min_hour


# ── Per-cluster aggregation accumulator ─────────────────────────────────────
class Aggs:
    """Per-cluster accumulator for the second streaming pass."""
    def __init__(self, K: int, n_hours: int, loggas_edges: np.ndarray):
        self.K = K
        self.n_hours = n_hours
        self.loggas_edges = loggas_edges
        self.n_bins = len(loggas_edges) - 1
        self.n_txs        = np.zeros(K, dtype=np.int64)
        self.n_spam_label = np.zeros((K, N_SPAM_CODES), dtype=np.int64)
        self.hourly_gas   = np.zeros((K, n_hours, 7), dtype=np.float64)
        self.loggas_hist  = np.zeros((K, self.n_bins, 7), dtype=np.int64)
        self.vol_sum      = np.zeros((K, 7), dtype=np.float64)
        # Per-tx share histogram per (cluster, share-bin, resource).  Used to
        # compute the EXACT median per-tx share of each resource within a
        # cluster — bin width 1/N_SHARE_BINS so the median is precise to that.
        self.share_hist   = np.zeros((K, N_SHARE_BINS, 7), dtype=np.int64)


def update_aggs(
    aggs: Aggs,
    labels: np.ndarray,
    G: np.ndarray, Lg: np.ndarray,
    hour_idx: np.ndarray, spam_codes: np.ndarray,
) -> None:
    K = aggs.K
    cluster_counts = np.bincount(labels, minlength=K)
    aggs.n_txs += cluster_counts.astype(np.int64)

    flat = labels.astype(np.int64) * N_SPAM_CODES + spam_codes.astype(np.int64)
    sl = np.bincount(flat, minlength=K * N_SPAM_CODES).reshape(K, N_SPAM_CODES)
    aggs.n_spam_label += sl

    flat_hour = labels.astype(np.int64) * aggs.n_hours + hour_idx.astype(np.int64)
    n_cells = K * aggs.n_hours
    for r in range(7):
        sums = np.bincount(flat_hour, weights=G[:, r],
                           minlength=n_cells)[:n_cells]
        aggs.hourly_gas[:, :, r] += sums.reshape(K, aggs.n_hours)

    edges = aggs.loggas_edges
    n_bins = aggs.n_bins
    n_cells_h = K * n_bins
    for r in range(7):
        bin_idx = np.clip(
            np.searchsorted(edges, Lg[:, r], side="right") - 1,
            0, n_bins - 1,
        )
        flat = labels.astype(np.int64) * n_bins + bin_idx.astype(np.int64)
        counts = np.bincount(flat, minlength=n_cells_h)[:n_cells_h]
        aggs.loggas_hist[:, :, r] += counts.reshape(K, n_bins)

    for r in range(7):
        sums = np.bincount(labels, weights=G[:, r], minlength=K)
        aggs.vol_sum[:, r] += sums

    # Per-tx share histogram (for median-share computation).
    # share_{i,k} = g_{i,k} / Σ_k' g_{i,k'} for each tx i.
    G_total = G.sum(axis=1).clip(min=1.0)
    n_cells_s = K * N_SHARE_BINS
    for r in range(7):
        w = G[:, r] / G_total
        bin_idx = np.clip(
            np.searchsorted(SHARE_EDGES, w, side="right") - 1,
            0, N_SHARE_BINS - 1,
        )
        flat = labels.astype(np.int64) * N_SHARE_BINS + bin_idx.astype(np.int64)
        counts = np.bincount(flat, minlength=n_cells_s)[:n_cells_s]
        aggs.share_hist[:, :, r] += counts.reshape(K, N_SHARE_BINS)


# ── Pass 2: stream-predict + aggregate ──────────────────────────────────────
def stream_aggregate(
    paths: list[pathlib.Path],
    mbk: MiniBatchKMeans, K: int,
    clr_lo: np.ndarray, clr_hi: np.ndarray,
    spam_table: pl.DataFrame,
    block_to_hour_offsets: np.ndarray, block_min: int, min_hour: int,
    loggas_edges: np.ndarray,
) -> tuple[Aggs, int]:
    """Stream every chunk, predict labels with the locked-K model,
    accumulate aggregations on the FULL dataset.  Applies the same
    per-dim CLR bounds as pass 1 so both passes share the in-bounds set."""
    n_hours_total = int(block_to_hour_offsets.max() - block_to_hour_offsets.min() + 1)
    aggs = Aggs(K, n_hours_total, loggas_edges)

    import pyarrow.compute as pc
    n_priced = 0
    t0 = time.time()
    for i, batch in enumerate(_iter_chunks(paths, batch_size=PASS2_CHUNK)):
        out = _featurize_batch(batch)
        if out is None:
            continue
        block_arr, sender_arrow, G, X_clr, Lg = out

        trimmed = _apply_clr_bounds(
            block_arr, sender_arrow, G, X_clr, Lg, clr_lo, clr_hi,
        )
        if trimmed is None:
            continue
        block_arr, sender_arrow, G, X_clr, Lg, _ = trimmed

        # Hour index per row.
        hour_idx = block_to_hour_offsets[block_arr - block_min].astype(np.int64)

        # Spam codes via polars left-join.
        sender_lower = pc.utf8_lower(sender_arrow)
        senders_pl = pl.from_arrow(pa.table({"tx_sender": sender_lower}))
        joined = senders_pl.join(spam_table, on="tx_sender", how="left")
        spam_codes = joined["spam_code"].fill_null(
            SPAM_CODE[SPAM_UNKNOWN]
        ).cast(pl.UInt8).to_numpy()
        del joined, senders_pl, sender_lower

        labels = mbk.predict(X_clr)
        update_aggs(aggs, labels, G, Lg, hour_idx, spam_codes)

        n_priced += G.shape[0]
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = n_priced / max(elapsed, 1e-6)
            print(f"  pass2 batch {i+1:>4d}: priced {n_priced:>11,}  "
                  f"({rate/1e6:.2f} M priced/s)")

    return aggs, n_priced


# ── Cluster naming ──────────────────────────────────────────────────────────
def label_cluster(centroid_share: np.ndarray) -> str:
    if np.any(np.isnan(centroid_share)):
        return "(empty)"
    order = np.argsort(centroid_share)[::-1]
    n1 = RESOURCE_LABEL[RESOURCES[order[0]]]
    n2 = RESOURCE_LABEL[RESOURCES[order[1]]]
    p1, p2 = float(centroid_share[order[0]]), float(centroid_share[order[1]])
    if p1 >= 0.80:
        return f"{n1}-heavy ({p1:.0%})"
    return f"{n1}+{n2} ({p1:.0%}/{p2:.0%})"


# ── t-SNE ───────────────────────────────────────────────────────────────────
def embed_tsne(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xj = X + rng.normal(0.0, 0.02, size=X.shape)
    tsne = TSNE(
        n_components=2, random_state=RNG_SEED, perplexity=50,
        init="pca", learning_rate="auto", max_iter=1000, method="barnes_hut",
    )
    return tsne.fit_transform(Xj)


# ── Dashboard ───────────────────────────────────────────────────────────────
def build_dashboard(
    K: int,
    metrics: dict[str, dict[int, float]],
    aggs: Aggs,
    tsne_xy: np.ndarray, tsne_labels: np.ndarray,
    n_priced_total: int, sample_size: int, min_hour: int,
) -> go.Figure:
    """One-pipeline dashboard.
       Row 1     : t-SNE scatter (full width)
       Row 2     : 2 K-selection metric panels (silhouette, WCSS elbow)
       Per-cluster rows: hourly resource share | log-gas histogram | spam dist
    """
    n_rows = 1 + 1 + K
    titles: list[str] = []

    sil = metrics["silhouette"]
    titles.append(
        f"t-SNE — K = {K}, silhouette = {sil[K]:.3f} · "
        f"sample = {sample_size:,}"
    )
    # Row 2: 2 metric panels
    titles += [
        "Silhouette ↑  (higher = better)",
        "WCSS / inertia ↓ (elbow)",
    ]

    # Pre-compute cluster meta (tag + size + spam %).
    n_total = int(aggs.n_txs.sum())
    cluster_meta: dict[int, dict] = {}
    for c in range(K):
        n = int(aggs.n_txs[c])
        sl = aggs.n_spam_label[c]
        n_vol  = int(sl[SPAM_CODE[SPAM_VOL]])
        n_rev  = int(sl[SPAM_CODE[SPAM_REV]])
        n_both = int(sl[SPAM_CODE[SPAM_BOTH]])
        n_unk  = int(sl[SPAM_CODE[SPAM_UNKNOWN]])
        n_spam = n_vol + n_rev + n_both
        denom = aggs.vol_sum[c].sum()
        centroid = aggs.vol_sum[c] / denom if denom > 0 else np.full(7, np.nan)
        cluster_meta[c] = {
            "tag": label_cluster(centroid),
            "size_pct": (n / n_total * 100.0) if n_total else 0.0,
            "spam_pct": (n_spam / n * 100.0) if n else 0.0,
            "n_txs":  n,
            "n_vol": n_vol, "n_rev": n_rev,
            "n_both": n_both, "n_unk": n_unk,
        }

    def _fmt_n(n: int) -> str:
        if n >= 1_000_000_000: return f"{n/1e9:.1f}B"
        if n >= 1_000_000:     return f"{n/1e6:.1f}M"
        if n >= 1_000:         return f"{n/1e3:.0f}K"
        return f"{n:,}"

    # Per-cluster row titles (5 cols each).
    for c in range(K):
        meta = cluster_meta[c]
        titles.append(
            f"<b>c{c}</b> — {meta['tag']}<br>"
            f"<sub>{meta['size_pct']:.1f}% · "
            f"{meta['spam_pct']:.1f}% spam · "
            f"{_fmt_n(meta['n_txs'])} txs</sub>"
        )
        titles.append("median per-tx resource share")    # NEW panel
        titles.append("log-gas distribution")
        titles.append(
            f"spam distribution<br>"
            f"<sub>vol={_fmt_n(meta['n_vol'])} · "
            f"rev={_fmt_n(meta['n_rev'])} · "
            f"both={_fmt_n(meta['n_both'])}</sub>"
        )

    row_heights = [0.20, 0.10] + [0.10] * K

    # Layout: 5 cols overall.
    # Row 1 (t-SNE):    colspan=5
    # Row 2 (metrics):  2 metric panels — silhouette (cols 1-2 colspan=2)
    #                   and WCSS elbow (cols 3-4 colspan=2), col 5 empty
    # Cluster rows:     hourly stacked bar (cols 1-2, colspan=2)
    #                 | median share bar (col 3)
    #                 | log-gas hist (col 4)
    #                 | spam dist (col 5)
    fig = make_subplots(
        rows=n_rows, cols=5,
        row_heights=row_heights,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.025,
        specs=(
            [[{"colspan": 5}, None, None, None, None]]
            + [[{"colspan": 2}, None, {"colspan": 2}, None, None]]
            + [[{"colspan": 2}, None, {}, {}, {}]] * K
        ),
    )

    palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel

    # ── Row 1: t-SNE ────────────────────────────────────────────────────────
    for c in range(K):
        mask = tsne_labels == c
        if not mask.any():
            continue
        color = palette[c % len(palette)]
        fig.add_trace(
            go.Scattergl(
                x=tsne_xy[mask, 0], y=tsne_xy[mask, 1], mode="markers",
                marker=dict(size=3, color=color, opacity=0.55,
                            line=dict(width=0)),
                name=f"c{c} (n={int(mask.sum()):,})",
                showlegend=True,
                hovertemplate=(
                    f"cluster {c}<br>tsne1=%{{x:.2f}} "
                    "tsne2=%{y:.2f}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=1,
                     showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=1,
                     showgrid=False, zeroline=False)

    # ── Row 2: 2 K-selection curves (silhouette + WCSS elbow) ───────────────
    Ks = sorted(metrics["silhouette"])
    for col, (key, color) in [
        (1, ("silhouette", "#1f77b4")),
        (3, ("wcss",       "#9467bd")),
    ]:
        ys = [metrics[key][k] for k in Ks]
        fig.add_trace(
            go.Scatter(x=Ks, y=ys, mode="lines+markers",
                       line=dict(color=color, width=1.8),
                       marker=dict(size=8),
                       showlegend=False),
            row=2, col=col,
        )
        # Star marker at K*.
        fig.add_trace(
            go.Scatter(x=[K], y=[metrics[key][K]], mode="markers",
                       marker=dict(size=14, symbol="star", color="#000",
                                   line=dict(color="#fff", width=1)),
                       showlegend=False),
            row=2, col=col,
        )
        fig.update_xaxes(title_text="K", tickmode="linear", dtick=1,
                         row=2, col=col)
    fig.update_yaxes(row=2, col=3, type="log")    # WCSS log scale

    # ── Per-cluster rows ────────────────────────────────────────────────────
    legend_ts_emitted: set[str] = set()
    legend_hist_emitted: set[str] = set()
    legend_spam_emitted: set[str] = set()

    any_gas = (aggs.hourly_gas.sum(axis=(0, 2)) > 0)
    hour_indices = np.nonzero(any_gas)[0]
    hour_dt = (
        (min_hour + hour_indices).astype("int64") * 3600
    ).astype("datetime64[s]")
    mid = 0.5 * (aggs.loggas_edges[:-1] + aggs.loggas_edges[1:])

    def _median_share(c: int, k: int) -> float:
        """Exact per-tx median share for (cluster, resource) from
        the `share_hist` cumulative distribution."""
        counts = aggs.share_hist[c, :, k]
        total = counts.sum()
        if total == 0:
            return 0.0
        cum = np.cumsum(counts)
        # Bin where cumulative >= total/2 → median falls in that bin.
        b = int(np.searchsorted(cum, total / 2.0))
        # Use bin midpoint as the median estimate.
        return 0.5 * (SHARE_EDGES[b] + SHARE_EDGES[min(b + 1, N_SHARE_BINS)])

    for c in range(K):
        row = 3 + c

        # Cols 1-2: hourly resource-share stacked bar
        cluster_gas = aggs.hourly_gas[c, hour_indices, :]
        row_total = cluster_gas.sum(axis=1).clip(min=1.0)
        shares = cluster_gas / row_total[:, None]
        cum = np.zeros(len(hour_indices))
        for k, r in enumerate(RESOURCES):
            lab = RESOURCE_LABEL[r]
            y_pct = shares[:, k] * 100.0
            show = lab not in legend_ts_emitted
            if show:
                legend_ts_emitted.add(lab)
            fig.add_trace(
                go.Bar(
                    x=hour_dt, y=y_pct, base=cum, name=lab,
                    marker_color=RESOURCE_COLOR[lab],
                    marker_line_width=0,
                    legendgroup="resource_share",
                    legendgrouptitle_text=(
                        "Hourly resource share" if show else None
                    ),
                    showlegend=show,
                    hovertemplate=(
                        f"{lab}: %{{y:.1f}}%<br>"
                        "%{x|%Y-%m-%d %H:00}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )
            cum = cum + y_pct
        fig.update_yaxes(title_text="% share", range=[0, 100],
                         row=row, col=1)
        fig.update_xaxes(
            title_text=("hour (UTC)" if c == K - 1 else None),
            row=row, col=1,
        )

        # Col 3: median per-tx resource share (horizontal bar)
        meds_pct = [_median_share(c, k) * 100.0 for k in range(7)]
        labels_x = [RESOURCE_LABEL[r] for r in RESOURCES]
        colors_x = [RESOURCE_COLOR[lab] for lab in labels_x]
        fig.add_trace(
            go.Bar(
                x=labels_x, y=meds_pct,
                marker_color=colors_x,
                showlegend=False,
                hovertemplate="%{x}<br>median share = %{y:.1f}%<extra></extra>",
            ),
            row=row, col=3,
        )
        fig.update_yaxes(title_text="median %", range=[0, 100],
                         row=row, col=3)
        fig.update_xaxes(tickangle=-30,
                         title_text=("resource" if c == K - 1 else None),
                         row=row, col=3)

        # Col 4: log-gas histogram per resource (overlap)
        for k, r in enumerate(RESOURCES):
            lab = RESOURCE_LABEL[r]
            counts = aggs.loggas_hist[c, :, k]
            if counts.sum() == 0:
                continue
            show = lab not in legend_hist_emitted
            if show:
                legend_hist_emitted.add(lab)
            fig.add_trace(
                go.Bar(
                    x=mid, y=counts, name=lab,
                    marker_color=RESOURCE_COLOR[lab],
                    marker_line_width=0,
                    opacity=0.55,
                    legendgroup="log_gas_hist",
                    legendgrouptitle_text=(
                        "Log-gas distribution" if show else None
                    ),
                    showlegend=show,
                    hovertemplate=(
                        f"{lab}<br>ln(1+g)≈%{{x:.2f}}<br>"
                        "n=%{y:,}<extra></extra>"
                    ),
                ),
                row=row, col=4,
            )
        fig.update_yaxes(title_text="count", type="log", row=row, col=4)
        fig.update_xaxes(
            title_text=("ln(1 + gas)" if c == K - 1 else None),
            row=row, col=4,
        )

        # Col 5: spam distribution — 2 x-axis bars (Non-spam, Spam).
        # Spam bar is stacked: high-volume / high-revert / both
        # (manual `base=` since global barmode is "overlay" for the
        # log-gas histogram).  Unknown wallets fold into Non-spam since
        # they have no flag in the spam table.
        n_total_c = aggs.n_txs[c]
        if n_total_c > 0:
            sl = aggs.n_spam_label[c]
            n_non  = int(sl[SPAM_CODE[SPAM_NOT]] + sl[SPAM_CODE[SPAM_UNKNOWN]])
            n_vol  = int(sl[SPAM_CODE[SPAM_VOL]])
            n_rev  = int(sl[SPAM_CODE[SPAM_REV]])
            n_both = int(sl[SPAM_CODE[SPAM_BOTH]])
            denom = max(n_non + n_vol + n_rev + n_both, 1)
            pct_non  = 100.0 * n_non  / denom
            pct_vol  = 100.0 * n_vol  / denom
            pct_rev  = 100.0 * n_rev  / denom
            pct_both = 100.0 * n_both / denom

            # Non-spam bar (single segment).
            non_show = "Non-spam" not in legend_spam_emitted
            if non_show:
                legend_spam_emitted.add("Non-spam")
            fig.add_trace(
                go.Bar(
                    x=["Non-spam"], y=[pct_non],
                    name="Non-spam",
                    marker_color=SPAM_COLOR[SPAM_NOT],
                    legendgroup="spam_class",
                    legendgrouptitle_text=(
                        "Spam classification" if non_show else None
                    ),
                    showlegend=non_show,
                    hovertemplate="Non-spam: %{y:.1f}%<extra></extra>",
                ),
                row=row, col=5,
            )

            # Spam bar — stacked via explicit base offsets.
            base_y = 0.0
            for seg_label, val, color in [
                ("High volume", pct_vol,  SPAM_COLOR[SPAM_VOL]),
                ("High revert", pct_rev,  SPAM_COLOR[SPAM_REV]),
                ("Both",        pct_both, SPAM_COLOR[SPAM_BOTH]),
            ]:
                seg_show = seg_label not in legend_spam_emitted
                if seg_show:
                    legend_spam_emitted.add(seg_label)
                fig.add_trace(
                    go.Bar(
                        x=["Spam"], y=[val], base=base_y,
                        name=seg_label,
                        marker_color=color,
                        legendgroup="spam_class",
                        showlegend=seg_show,
                        hovertemplate=(
                            f"Spam — {seg_label}: %{{y:.1f}}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row, col=5,
                )
                base_y += val
        fig.update_yaxes(title_text="% of cluster", row=row, col=5)
        fig.update_xaxes(
            title_text=("sender class" if c == K - 1 else None),
            row=row, col=5,
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    total_height = 320 * n_rows + 240
    title_text = (
        "<b>Per-tx clustering — KMeans on CLR (centered log-ratio) gas usage (7 resources, incl L1)</b>"
        f"<br><sub>{n_priced_total:,} priced txs (FULL dataset) · "
        f"K = {K} (composite of silhouette + WCSS elbow over K∈{K_RANGE[0]}..{K_RANGE[-1]}) · "
        f"silhouette / metrics / t-SNE on {sample_size:,}-row sample</sub>"
    )
    fig.update_layout(
        title=dict(text=title_text, x=0.0, xanchor="left",
                   y=0.995, yanchor="top",
                   font=dict(size=18, color="#111")),
        template="plotly_white",
        height=total_height,
        margin=dict(l=80, r=320, t=160, b=80),
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
        hovermode="x unified",
        barmode="overlay",
    )
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.55)", mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.55)", mirror=True, ticks="outside")
    return fig, cluster_meta


# ── Phase 1: fit MBK bank + reservoir sample, save to cache ─────────────────
def fit_phase(n_sample: int) -> None:
    import pickle
    rng = np.random.default_rng(RNG_SEED)

    paths = _per_tx_files()
    if not paths:
        raise SystemExit(f"No per_tx parquets found under {MULTIGAS_DIR}")
    print(f"[fit] streaming pass 1 over {len(paths)} per_tx parquets — "
          f"partial_fit MBK bank on CLR features + reservoir "
          f"(target {n_sample:,})...")
    t0 = time.time()
    (mbk, clr_lo, clr_hi,
     sample, X_clr_s, Lg_s, n_priced, n_kept) = stream_fit(paths, n_sample, rng)
    drop_pct = 100.0 * (n_priced - n_kept) / max(n_priced, 1)
    print(f"[fit] pass 1: {time.time()-t0:.1f}s, "
          f"{n_priced:,} priced rows ({n_kept:,} in-bounds, "
          f"{drop_pct:.2f}% trimmed), sample {len(sample):,}")

    print("[fit] computing diagnostics (silhouette + WCSS)...")
    metrics = compute_diagnostics(mbk, X_clr_s, rng)
    auto_K  = composite_best_k(metrics)
    print(f"[fit] composite-best K (rank-aggregate of silhouette + WCSS): K = {auto_K}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "fit.pkl", "wb") as f:
        pickle.dump({
            "mbk": mbk,
            "clr_lo": clr_lo, "clr_hi": clr_hi,
            "metrics": metrics, "auto_K": auto_K,
            "n_priced": n_priced, "n_kept": n_kept,
        }, f)
    sample.write_parquet(CACHE_DIR / "sample.parquet")
    np.savez(CACHE_DIR / "features.npz",
             X_clr=X_clr_s, Lg=Lg_s)
    print(f"[fit] cached → {CACHE_DIR}")


# ── Phase 2: predict + aggregate on full data ───────────────────────────────
def aggregate_phase() -> None:
    import pickle
    print("[aggregate] loading fit cache...")
    with open(CACHE_DIR / "fit.pkl", "rb") as f:
        ck = pickle.load(f)
    feats = np.load(CACHE_DIR / "features.npz")
    Lg_s = feats["Lg"]
    feats.close()

    print("[aggregate] loading spam labels + block→hour map...")
    spam_table = load_spam_table()
    bt_offsets, block_min, min_hour = load_block_to_hour()
    print(f"[aggregate] block range: [{block_min:,}, "
          f"{block_min + len(bt_offsets) - 1:,}]")

    loggas_lo = float(np.percentile(Lg_s, 0.5))
    loggas_hi = float(np.percentile(Lg_s, 99.5))
    loggas_edges = np.linspace(loggas_lo, loggas_hi, N_LOGGAS_BINS + 1)
    del Lg_s

    paths = _per_tx_files()
    print(f"[aggregate] streaming pass 2 — predict + aggregate at K = {K_LOG}...")
    t0 = time.time()
    aggs, n_priced = stream_aggregate(
        paths,
        ck["mbk"][K_LOG], K_LOG,
        ck["clr_lo"], ck["clr_hi"],
        spam_table, bt_offsets, block_min, min_hour, loggas_edges,
    )
    print(f"[aggregate] pass 2: {time.time()-t0:.1f}s, "
          f"{n_priced:,} priced rows aggregated")

    np.savez(CACHE_DIR / "aggs.npz",
             K=K_LOG,
             n_txs=aggs.n_txs, n_spam_label=aggs.n_spam_label,
             hourly_gas=aggs.hourly_gas,
             loggas_hist=aggs.loggas_hist, vol_sum=aggs.vol_sum,
             share_hist=aggs.share_hist,
             loggas_edges=aggs.loggas_edges)
    with open(CACHE_DIR / "aggregate_meta.pkl", "wb") as f:
        pickle.dump({"min_hour": min_hour, "n_priced_full": n_priced}, f)
    print(f"[aggregate] cached → {CACHE_DIR}")


def _aggs_from_npz(path: pathlib.Path) -> Aggs:
    d = np.load(path)
    K = int(d["K"])
    n_hours = d["hourly_gas"].shape[1]
    a = Aggs(K, n_hours, d["loggas_edges"])
    a.n_txs        = d["n_txs"]
    a.n_spam_label = d["n_spam_label"]
    a.hourly_gas   = d["hourly_gas"]
    a.loggas_hist  = d["loggas_hist"]
    a.vol_sum      = d["vol_sum"]
    a.share_hist   = d["share_hist"] if "share_hist" in d.files else \
                     np.zeros((K, N_SHARE_BINS, 7), dtype=np.int64)
    return a


# ── Phase 3: t-SNE + dashboard ──────────────────────────────────────────────
def plot_phase() -> None:
    import pickle
    rng = np.random.default_rng(RNG_SEED)

    print("[plot] loading caches...")
    with open(CACHE_DIR / "fit.pkl", "rb") as f:
        ck = pickle.load(f)
    feats = np.load(CACHE_DIR / "features.npz")
    X_clr_s = feats["X_clr"]
    sample  = pl.read_parquet(CACHE_DIR / "sample.parquet")
    aggs    = _aggs_from_npz(CACHE_DIR / "aggs.npz")
    with open(CACHE_DIR / "aggregate_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    n_tsne = min(N_TSNE, len(sample))
    if len(sample) > n_tsne:
        idx = rng.choice(len(sample), size=n_tsne, replace=False)
    else:
        idx = np.arange(len(sample))

    print(f"[plot] running t-SNE on {len(idx):,} pts (Barnes-Hut)...")
    t = time.time()
    tsne_xy = embed_tsne(X_clr_s[idx], rng)
    tsne_labels = ck["mbk"][K_LOG].predict(X_clr_s[idx])
    print(f"[plot] t-SNE: {time.time()-t:.1f}s")

    print(f"[plot] building dashboard at K = {K_LOG} "
          f"(composite-auto K from fit was {ck.get('auto_K', '?')})...")
    fig, cluster_meta = build_dashboard(
        K=K_LOG,
        metrics=ck["metrics"],
        aggs=aggs,
        tsne_xy=tsne_xy, tsne_labels=tsne_labels,
        n_priced_total=meta["n_priced_full"],
        sample_size=len(sample),
        min_hour=meta["min_hour"],
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(OUT_HTML), include_plotlyjs="cdn", full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    print(f"[plot] saved {OUT_HTML}")

    print(f"\nKMeans on log-gas (K = {K_LOG}) — clusters by spam concentration:")
    ranked = sorted(cluster_meta.items(), key=lambda kv: -kv[1]["spam_pct"])
    for c, m in ranked:
        print(f"  c{c}: {m['size_pct']:5.1f}% size, "
              f"{m['spam_pct']:5.1f}% spam "
              f"(vol={m['n_vol']:,} rev={m['n_rev']:,} "
              f"both={m['n_both']:,})  — {m['tag']}  "
              f"[{m['n_txs']:,} txs]")


# ── Driver ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.phase == "fit":
        fit_phase(args.n_sample)
    elif args.phase == "aggregate":
        aggregate_phase()
    elif args.phase == "plot":
        plot_phase()
    else:
        # Orchestrator — one fresh subprocess per phase for memory isolation.
        import subprocess, sys
        for phase in ("fit", "aggregate", "plot"):
            cmd = [sys.executable, "-u", str(_HERE / "tx_clustering.py"),
                   "--phase", phase, "--n-sample", str(args.n_sample)]
            print(f"\n=== orchestrator: launching `{' '.join(cmd)}` ===\n")
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
