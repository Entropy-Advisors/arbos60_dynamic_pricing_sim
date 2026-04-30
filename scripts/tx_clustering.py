"""
Two-pipeline transaction clustering by per-resource log-gas (incl L1 calldata):

  • LOG  — K-Means on standardized log1p(g_k); baseline (mix + magnitude),
           locked at K_LOG.
  • CLR  — K-Means on CLR-transformed weights; sophisticated (compositional
           geometry, scale-invariant), locked at K_CLR.

Pipeline runs over the FULL multigas dataset via two streaming passes:

  Pass 1 — bootstrap log z-score stats from the first chunk, partial_fit the
           bank of MiniBatchKMeans models (K = 3..8 × {log, clr} = 12 models),
           and build a small reservoir sample (used only for t-SNE +
           silhouette + log-gas-histogram bin edges).

  Pass 2 — predict cluster labels on every chunk and accumulate per-cluster
           aggregations: tx counts, spam-label counts, hourly per-resource
           gas, log-gas histogram counts, volume-weighted resource centroid.
           These drive the dashboard panels — every per-cluster panel
           reflects the entire dataset, not the reservoir.

t-SNE and silhouette stay sample-based (O(N²) and O(N) cost respectively).

Output: figures/clustering.html

Run:
    python scripts/tx_clustering.py
    python scripts/tx_clustering.py --n-sample 100000
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
K_RANGE        = list(range(3, 9))   # 3..8 — sweep span (silhouette + WCSS)
K_LOG          = 7                   # log-gas pipeline locked at K_LOG
K_CLR          = 4                   # CLR pipeline locked at K_CLR
N_SAMPLE       = 60_000              # reservoir sample for t-SNE / silhouette
N_TSNE         = 10_000
N_SILHOUETTE   = 6_000
CHUNKSIZE      = 1_000_000          # pass-1 chunk (pandas DF; needs sample)
PASS2_CHUNK    = 500_000            # pass-2 chunk (arrow→numpy; smaller=less peak)
MBK_BATCH      = 8_192
RNG_SEED       = 42
MIN_PRICED_GAS = 30_000
CLR_EPS        = 1e-3
N_LOGGAS_BINS  = 40                  # log-gas histogram bin count

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
PIPELINE_LABEL = {
    "log": "K-Means on log-gas",
    "clr": "K-Means on CLR features",
}

GAS_COLS = [f"gas_{r}" for r in RESOURCES]


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-sample", type=int, default=N_SAMPLE,
                   help="Reservoir sample size (default 60k)")
    p.add_argument("--phase", choices=["fit", "aggregate", "plot", "all"],
                   default="all",
                   help="Run a single phase (for memory isolation) — "
                        "'all' (default) orchestrates the three phases as "
                        "fresh subprocesses so pass-1 heap is reclaimed "
                        "before pass 2 starts.")
    return p.parse_args()


# ── Streaming I/O ───────────────────────────────────────────────────────────
def _per_tx_files() -> list[pathlib.Path]:
    return sorted(MULTIGAS_DIR.glob("*/per_tx.parquet"))


PASS1_COLS = [
    "block", "tx_sender",
    "computation", "wasmComputation",
    "historyGrowth",
    "storageAccessRead", "storageAccessWrite",
    "storageGrowth", "l2Calldata", "l1Calldata",
]


def _iter_chunks(paths: list[pathlib.Path], batch_size: int = CHUNKSIZE):
    """Yield pyarrow RecordBatch (zero-copy where possible).  Avoids pandas
    DataFrame allocation; consumers pull columns as numpy / arrow."""
    import pyarrow.parquet as pq
    for path in paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=batch_size, columns=PASS1_COLS):
            yield batch


def _total_rows(paths: list[pathlib.Path]) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)


def _iter_arrow_batches(paths: list[pathlib.Path], batch_size: int):
    """Pyarrow batches for pass 2 — same columns as pass 1, smaller chunks."""
    return _iter_chunks(paths, batch_size=batch_size)


def _featurize_batch(
    batch,                                     # pyarrow.RecordBatch
    log_mean: np.ndarray | None,
    log_std:  np.ndarray | None,
):
    """Featurize one arrow batch into raw numpy.
    Returns (block_arr, sender_arrow, G, X_log, X_clr, Lg, gtot) where:
      - block_arr is int64 numpy of priced rows
      - sender_arrow is pyarrow Array of priced senders (no pandas overhead)
      - G, Lg, X_log, X_clr are float64 (n_priced, 7)
      - gtot is float64 (n_priced,)
    Returns None if no rows survive the priced-gas filter.  When
    `log_mean` is None, X_log is the un-standardised Lg (caller bootstraps)."""
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
    G = np.column_stack([g_c[keep], g_sw[keep], g_sr[keep],
                         g_sg[keep], g_hg[keep], g_l2[keep], g_l1[keep]])
    gtot_kept = gtot[keep]
    Lg = np.log1p(G)
    X_log = Lg.copy() if log_mean is None else ((Lg - log_mean) / log_std)
    W = G / gtot_kept[:, None]
    Lw = np.log(W + CLR_EPS)
    X_clr = Lw - Lw.mean(axis=1, keepdims=True)
    block_arr = batch.column("block").to_numpy(zero_copy_only=False)[keep].astype(np.int64)
    sender_arrow = batch.column("tx_sender").filter(pa.array(keep))
    return block_arr, sender_arrow, G, X_log, X_clr, Lg, gtot_kept


# ── Pass 1: stream-fit MBK bank + reservoir sample ──────────────────────────
def stream_fit(
    paths: list[pathlib.Path], n_target: int, rng: np.random.Generator,
) -> tuple[
    dict[int, MiniBatchKMeans],
    dict[int, MiniBatchKMeans],
    np.ndarray, np.ndarray,
    pl.DataFrame,                 # reservoir sample as polars (no pandas)
    np.ndarray, np.ndarray, np.ndarray,
    int,
]:
    """Pass 1: arrow batches → numpy featurize → partial_fit + reservoir.
    Sample is held as polars (cheap strings, low overhead)."""
    mbk_log = {
        K: MiniBatchKMeans(
            n_clusters=K, random_state=RNG_SEED, batch_size=MBK_BATCH,
            n_init=5, max_iter=200, reassignment_ratio=0.01,
        )
        for K in K_RANGE
    }
    mbk_clr = {
        K: MiniBatchKMeans(
            n_clusters=K, random_state=RNG_SEED, batch_size=MBK_BATCH,
            n_init=5, max_iter=200, reassignment_ratio=0.01,
        )
        for K in K_RANGE
    }
    total_rows = _total_rows(paths) or n_target * 100
    keep_frac = min(1.0, (n_target * 1.4) / max(total_rows, 1))

    sample_arrow_parts: list[pa.Table] = []
    Xlog_parts: list[np.ndarray] = []
    Xclr_parts: list[np.ndarray] = []
    Lg_parts:   list[np.ndarray] = []
    n_priced = 0
    n_scanned = 0
    log_mean = log_std = None

    t0 = time.time()
    for i, batch in enumerate(_iter_chunks(paths, batch_size=CHUNKSIZE)):
        n_scanned += batch.num_rows
        out = _featurize_batch(batch, log_mean, log_std)
        if out is None:
            continue
        block_arr, sender_arrow, G, X_log_raw, X_clr, Lg, gtot_kept = out

        if log_mean is None:
            log_mean = Lg.mean(axis=0)
            log_std  = Lg.std(axis=0).clip(min=1e-9)
            X_log = (Lg - log_mean) / log_std
        else:
            X_log = X_log_raw

        for K in K_RANGE:
            mbk_log[K].partial_fit(X_log)
            mbk_clr[K].partial_fit(X_clr)

        n_priced += G.shape[0]

        mask = rng.random(G.shape[0]) < keep_frac
        if mask.any():
            mask_arr = pa.array(mask)
            tbl = pa.table({
                "block":     pa.array(block_arr[mask], type=pa.int64()),
                "tx_sender": sender_arrow.filter(mask_arr),
                **{GAS_COLS[k]: pa.array(G[mask, k]) for k in range(7)},
                "gas_total": pa.array(gtot_kept[mask]),
            })
            sample_arrow_parts.append(tbl)
            Xlog_parts.append(X_log[mask])
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
    X_log_sample = np.vstack(Xlog_parts)
    X_clr_sample = np.vstack(Xclr_parts)
    Lg_sample    = np.vstack(Lg_parts)
    sample_arrow_parts.clear(); Xlog_parts.clear()
    Xclr_parts.clear();         Lg_parts.clear()

    if sample.height > n_target:
        idx = rng.choice(sample.height, size=n_target, replace=False)
        idx = np.sort(idx)
        sample = sample[idx]
        X_log_sample = X_log_sample[idx]
        X_clr_sample = X_clr_sample[idx]
        Lg_sample    = Lg_sample[idx]

    return (mbk_log, mbk_clr, log_mean, log_std,
            sample, X_log_sample, X_clr_sample, Lg_sample, n_priced)


# ── Diagnostics: silhouette + WCSS per K ────────────────────────────────────
def compute_diagnostics(
    mbk_bank: dict[int, MiniBatchKMeans],
    X_sample: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-K silhouette (on N_SILHOUETTE-row subset of the reservoir) and
    inertia (WCSS — straight from the fitted model).  Inertia is the
    model's last-batch value; for elbow purposes the relative shape is
    correct."""
    sils, wcss = {}, {}
    sil_idx = rng.choice(len(X_sample),
                         size=min(N_SILHOUETTE, len(X_sample)),
                         replace=False)
    for K in K_RANGE:
        labels = mbk_bank[K].predict(X_sample)
        sils[K] = float(silhouette_score(X_sample[sil_idx], labels[sil_idx]))
        wcss[K] = float(mbk_bank[K].inertia_)
        print(f"    K={K}: sil={sils[K]:.3f}  WCSS={wcss[K]:,.0f}")
    return sils, wcss


# ── Spam labels & block→hour map ────────────────────────────────────────────
# Wallet-level category derived from the SQL day-counts:
#   n_days_high_vol > 0  AND  n_days_high_rev > 0   → "High volume + revert"
#   n_days_high_vol > 0  AND  n_days_high_rev == 0  → "High volume"
#   n_days_high_vol == 0 AND  n_days_high_rev > 0   → "High revert"
#   n_days_high_vol == 0 AND  n_days_high_rev == 0  → "Not spam"
#   wallet absent from the spam parquet              → "Unknown"
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
    SPAM_NOT:     0,
    SPAM_VOL:     1,
    SPAM_REV:     2,
    SPAM_BOTH:    3,
    SPAM_UNKNOWN: 4,
}
N_SPAM_CODES = 5


def load_spam_table() -> pl.DataFrame:
    """Polars DataFrame [tx_sender, spam_code] for fast/low-mem per-chunk
    joins.  Polars uses Arrow's contiguous UTF-8 string layout (no Python
    string objects), so 13.4M-row joins fit comfortably in <1 GB."""
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
    hour_offsets[block - min_block] = (epoch-hour - min_hour_int).  Polars
    keeps RSS bounded to ~1 GB for the 72M-row per_block parquet (pandas
    retained ~7 GB of heap doing the same load)."""
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
    """Per-pipeline (per-cluster) accumulator for the second streaming pass.
    Holds totals, hourly resource gas, log-gas histogram counts, spam-label
    counts, and a running volume-weighted resource centroid."""
    def __init__(self, K: int, n_hours: int, loggas_edges: np.ndarray):
        self.K = K
        self.n_hours = n_hours
        self.loggas_edges = loggas_edges
        self.n_bins = len(loggas_edges) - 1
        # Tx counts.
        self.n_txs        = np.zeros(K, dtype=np.int64)
        self.n_spam_label = np.zeros((K, N_SPAM_CODES), dtype=np.int64)
        # Hourly per-resource gas sums (cluster, hour, resource).
        self.hourly_gas = np.zeros((K, n_hours, 7), dtype=np.float64)
        # Log-gas histogram counts (cluster, bin, resource).
        self.loggas_hist = np.zeros((K, self.n_bins, 7), dtype=np.int64)
        # Volume-weighted centroid (running sum of gas per resource).
        self.vol_sum    = np.zeros((K, 7), dtype=np.float64)


def init_aggs(K: int, n_hours: int, loggas_edges: np.ndarray) -> Aggs:
    return Aggs(K, n_hours, loggas_edges)


def update_aggs(
    aggs: Aggs,
    labels: np.ndarray,
    G: np.ndarray,             # raw gas (n × 7)
    Lg: np.ndarray,            # log1p(gas) (n × 7)
    hour_idx: np.ndarray,      # int per-row hour offset (within global hour range)
    spam_codes: np.ndarray,    # uint8 per-row SPAM_CODE
) -> None:
    K = aggs.K
    # 1) tx counts per cluster
    cluster_counts = np.bincount(labels, minlength=K)
    aggs.n_txs += cluster_counts.astype(np.int64)

    # 2) spam-label counts per cluster (cluster × N_SPAM_CODES)
    flat = labels.astype(np.int64) * N_SPAM_CODES + spam_codes.astype(np.int64)
    sl = np.bincount(flat, minlength=K * N_SPAM_CODES).reshape(K, N_SPAM_CODES)
    aggs.n_spam_label += sl

    # 3) hourly gas per (cluster, hour, resource)
    # vectorised per-resource np.add.at — 7 calls
    flat_hour = labels.astype(np.int64) * aggs.n_hours + hour_idx.astype(np.int64)
    n_cells = K * aggs.n_hours
    for r in range(7):
        sums = np.bincount(flat_hour, weights=G[:, r],
                           minlength=n_cells)[:n_cells]
        aggs.hourly_gas[:, :, r] += sums.reshape(K, aggs.n_hours)

    # 4) log-gas histogram counts (cluster, bin, resource)
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

    # 5) volume-weighted centroid running sum
    for r in range(7):
        sums = np.bincount(labels, weights=G[:, r], minlength=K)
        aggs.vol_sum[:, r] += sums


# ── Pass 2: stream-predict + aggregate ──────────────────────────────────────
def stream_aggregate(
    paths: list[pathlib.Path],
    mbk_log: MiniBatchKMeans, K_log: int,
    mbk_clr: MiniBatchKMeans, K_clr: int,
    log_mean: np.ndarray, log_std: np.ndarray,
    spam_table: pl.DataFrame,
    block_to_hour_offsets: np.ndarray, block_min: int, min_hour: int,
    loggas_edges: np.ndarray,
) -> tuple[Aggs, Aggs, int]:
    """Stream every chunk, predict labels with the chosen K models, accumulate
    aggregations on the FULL dataset.  Returns (aggs_log, aggs_clr,
    global_min_hour) — caller turns aggs into dashboard structures."""
    n_hours_total = int(block_to_hour_offsets.max() - block_to_hour_offsets.min() + 1)
    # Use the hour offset directly as the index — but it must fit within
    # the sample's actual hour range, which is already captured by uint32.
    # We use the per_block hour offset directly (already 0-indexed against
    # min block_time across all blocks).  n_hours_total covers the full span.
    aggs_log = init_aggs(K_log, n_hours_total, loggas_edges)
    aggs_clr = init_aggs(K_clr, n_hours_total, loggas_edges)

    import pyarrow.compute as pc
    n_priced = 0
    t0 = time.time()
    for i, batch in enumerate(_iter_arrow_batches(paths, PASS2_CHUNK)):
        # Materialise gas columns as numpy directly from arrow (no pandas).
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
            continue
        G = np.column_stack([g_c[keep], g_sw[keep], g_sr[keep],
                             g_sg[keep], g_hg[keep], g_l2[keep], g_l1[keep]])
        Lg = np.log1p(G)
        X_log = (Lg - log_mean) / log_std
        W = G / gtot[keep, None]
        Lw = np.log(W + CLR_EPS)
        X_clr = Lw - Lw.mean(axis=1, keepdims=True)

        # Hour index — block_to_hour offsets indexed by (block - block_min).
        block_arr = batch.column("block").to_numpy(zero_copy_only=False)[keep]
        hour_idx = block_to_hour_offsets[block_arr - block_min].astype(np.int64)

        # Spam codes — keep tx_sender as polars/arrow (no pandas StringArray),
        # lowercase + left-join against spam_table.
        sender_arrow = pc.utf8_lower(batch.column("tx_sender")).filter(keep)
        senders_pl = pl.from_arrow(
            pa.table({"tx_sender": sender_arrow})
        )
        joined = senders_pl.join(spam_table, on="tx_sender", how="left")
        spam_codes = joined["spam_code"].fill_null(
            SPAM_CODE[SPAM_UNKNOWN]
        ).cast(pl.UInt8).to_numpy()
        del joined, senders_pl, sender_arrow

        # KMeans predict with the chosen K models.
        labels_log = mbk_log.predict(X_log)
        labels_clr = mbk_clr.predict(X_clr)

        update_aggs(aggs_log, labels_log, G, Lg, hour_idx, spam_codes)
        update_aggs(aggs_clr, labels_clr, G, Lg, hour_idx, spam_codes)

        n_priced += int(keep.sum())
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = n_priced / max(elapsed, 1e-6)
            print(f"  pass2 batch {i+1:>4d}: priced {n_priced:>11,}  "
                  f"({rate/1e6:.2f} M priced/s)")

    return aggs_log, aggs_clr, n_priced


# ── Cluster naming ──────────────────────────────────────────────────────────
def label_cluster(centroid_share: np.ndarray) -> str:
    """Human-readable tag from the top-2 resources of a normalised centroid."""
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
    pipelines: dict[str, dict],          # per-fset: K, sils, wcss, aggs, tsne, ...
    n_priced_total: int,
    sample_size: int,
    min_hour: int,
) -> go.Figure:
    """3-col layout (n_rows = 3 + K_log + K_clr):
        Row 1: t-SNE log | t-SNE clr | (empty)
        Row 2: silhouette log | silhouette clr | (empty)
        Row 3: WCSS log | WCSS clr | (empty)
        Per-cluster rows: hourly resource share | log-gas hist | spam dist
    """
    K_log = pipelines["log"]["K"]
    K_clr = pipelines["clr"]["K"]
    n_rows = 3 + K_log + K_clr

    # Pre-compute cluster meta: tag, size %, spam %, total tx count, persistent count.
    cluster_meta: dict[tuple[str, int], dict] = {}
    for fset in ("log", "clr"):
        aggs: Aggs = pipelines[fset]["aggs"]
        K = pipelines[fset]["K"]
        n_txs_total = int(aggs.n_txs.sum())
        for c in range(K):
            n = int(aggs.n_txs[c])
            sl = aggs.n_spam_label[c]   # vector indexed by SPAM_CODE
            n_vol  = int(sl[SPAM_CODE[SPAM_VOL]])
            n_rev  = int(sl[SPAM_CODE[SPAM_REV]])
            n_both = int(sl[SPAM_CODE[SPAM_BOTH]])
            n_unk  = int(sl[SPAM_CODE[SPAM_UNKNOWN]])
            n_spam = n_vol + n_rev + n_both
            spam_pct = (n_spam / n * 100.0) if n else 0.0
            size_pct = (n / n_txs_total * 100.0) if n_txs_total else 0.0
            denom = aggs.vol_sum[c].sum()
            centroid = aggs.vol_sum[c] / denom if denom > 0 else np.full(7, np.nan)
            cluster_meta[(fset, c)] = {
                "tag": label_cluster(centroid),
                "size_pct": size_pct,
                "spam_pct": spam_pct,
                "n_txs":  n,
                "n_vol":  n_vol,
                "n_rev":  n_rev,
                "n_both": n_both,
                "n_unk":  n_unk,
            }
        pipelines[fset]["cluster_meta"] = {
            c: cluster_meta[(fset, c)] for c in range(K)
        }

    titles: list[str] = []
    # Row 1 — t-SNE
    for fset in ("log", "clr"):
        K   = pipelines[fset]["K"]
        sil = pipelines[fset]["sils"][K]
        titles.append(
            f"t-SNE — {PIPELINE_LABEL[fset]}<br>"
            f"<sub>K = {K} · silhouette = {sil:.3f} · "
            f"sample = {sample_size:,}</sub>"
        )
    titles.append("")
    # Row 2 — silhouette
    for fset in ("log", "clr"):
        titles.append(f"Silhouette vs K — {PIPELINE_LABEL[fset]}")
    titles.append("")
    # Row 3 — WCSS / inertia
    for fset in ("log", "clr"):
        titles.append(f"WCSS (inertia) vs K — {PIPELINE_LABEL[fset]}")
    titles.append("")
    # Per-cluster rows.
    for fset, K in (("log", K_log), ("clr", K_clr)):
        prefix = "LOG" if fset == "log" else "CLR"
        for c in range(K):
            meta = cluster_meta[(fset, c)]
            titles.append(
                f"<b>{prefix} · c{c}</b> — {meta['tag']}  "
                f"<span style='color:#666'>"
                f"({meta['size_pct']:.1f}% size, "
                f"{meta['spam_pct']:.1f}% spam, "
                f"{meta['n_txs']:,} txs)</span>"
            )
            titles.append("log-gas distribution")
            titles.append(
                f"spam distribution<br>"
                f"<sub>vol={meta['n_vol']:,} · "
                f"rev={meta['n_rev']:,} · "
                f"both={meta['n_both']:,}</sub>"
            )

    row_heights = (
        [0.18, 0.08, 0.08]                    # t-SNE, silhouette, WCSS
        + [0.10] * (K_log + K_clr)            # per-cluster
    )

    fig = make_subplots(
        rows=n_rows, cols=3,
        row_heights=row_heights,
        subplot_titles=titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.025,
    )

    palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel

    # ── Row 1: t-SNE ────────────────────────────────────────────────────────
    for col, fset in enumerate(("log", "clr"), start=1):
        K       = pipelines[fset]["K"]
        labels  = pipelines[fset]["labels_tsne"]
        XY      = pipelines[fset]["tsne"]
        for c in range(K):
            mask = labels == c
            if not mask.any():
                continue
            color = palette[c % len(palette)]
            fig.add_trace(
                go.Scattergl(
                    x=XY[mask, 0], y=XY[mask, 1], mode="markers",
                    marker=dict(size=4, color=color, opacity=0.65,
                                line=dict(width=0)),
                    name=f"c{c} (n={int(mask.sum()):,})",
                    legendgroup=f"tsne_{fset}",
                    legendgrouptitle_text=(
                        PIPELINE_LABEL[fset] if c == 0 else None
                    ),
                    showlegend=True,
                    hovertemplate=(
                        f"cluster {c}<br>tsne1=%{{x:.2f}} "
                        "tsne2=%{y:.2f}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=col,
                         showgrid=False, zeroline=False)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=col,
                         showgrid=False, zeroline=False)

    # ── Row 2: silhouette curves ────────────────────────────────────────────
    for col, fset in enumerate(("log", "clr"), start=1):
        sils = pipelines[fset]["sils"]
        Ks   = sorted(sils)
        ys   = [sils[K] for K in Ks]
        K_star = pipelines[fset]["K"]
        fig.add_trace(
            go.Scatter(
                x=Ks, y=ys, mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate="K=%{x}: silhouette=%{y:.3f}<extra></extra>",
            ),
            row=2, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[K_star], y=[sils[K_star]], mode="markers",
                marker=dict(size=15, symbol="star", color="#d62728"),
                showlegend=False,
                hovertemplate=f"K* = {K_star}<extra></extra>",
            ),
            row=2, col=col,
        )
        fig.update_xaxes(title_text="K", tickmode="linear", dtick=1,
                         row=2, col=col)
        fig.update_yaxes(title_text="silhouette", row=2, col=col)

    # ── Row 3: WCSS / inertia curves ────────────────────────────────────────
    for col, fset in enumerate(("log", "clr"), start=1):
        wcss = pipelines[fset]["wcss"]
        Ks   = sorted(wcss)
        ys   = [wcss[K] for K in Ks]
        K_star = pipelines[fset]["K"]
        fig.add_trace(
            go.Scatter(
                x=Ks, y=ys, mode="lines+markers",
                line=dict(color="#9467bd", width=2),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate="K=%{x}: WCSS=%{y:,.0f}<extra></extra>",
            ),
            row=3, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[K_star], y=[wcss[K_star]], mode="markers",
                marker=dict(size=15, symbol="star", color="#d62728"),
                showlegend=False,
                hovertemplate=f"K = {K_star}<extra></extra>",
            ),
            row=3, col=col,
        )
        fig.update_xaxes(title_text="K", tickmode="linear", dtick=1,
                         row=3, col=col)
        fig.update_yaxes(title_text="WCSS", row=3, col=col, type="log")

    # Hide unused 3rd column on top three rows.
    for r in (1, 2, 3):
        fig.update_xaxes(visible=False, row=r, col=3)
        fig.update_yaxes(visible=False, row=r, col=3)

    # ── Per-cluster rows ────────────────────────────────────────────────────
    legend_ts_emitted: set[str] = set()
    legend_hist_emitted: set[str] = set()

    def render_pipeline(fset: str, start_row: int) -> None:
        aggs: Aggs = pipelines[fset]["aggs"]
        K = pipelines[fset]["K"]

        # Compute hour x-axis labels once per pipeline.
        # Only show hours where any cluster has gas to keep x-axis dense.
        any_gas = (aggs.hourly_gas.sum(axis=(0, 2)) > 0)
        hour_indices = np.nonzero(any_gas)[0]
        hour_dt = (
            (min_hour + hour_indices).astype("int64") * 3600
        ).astype("datetime64[s]")

        # Bin midpoints for log-gas histogram x-axis.
        mid = 0.5 * (aggs.loggas_edges[:-1] + aggs.loggas_edges[1:])

        for c in range(K):
            row = start_row + c
            # ── Left: hourly resource-share stacked bar ─────────────────────
            cluster_gas = aggs.hourly_gas[c, hour_indices, :]   # (n_hours, 7)
            row_total = cluster_gas.sum(axis=1).clip(min=1.0)
            shares = cluster_gas / row_total[:, None]
            cum = np.zeros(len(hour_indices))
            for k, r in enumerate(RESOURCES):
                lab   = RESOURCE_LABEL[r]
                y_pct = shares[:, k] * 100.0
                show  = lab not in legend_ts_emitted
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

            # ── Middle: log-gas histogram per resource (bar overlay) ────────
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
                    row=row, col=2,
                )
            fig.update_yaxes(title_text="count", type="log", row=row, col=2)
            fig.update_xaxes(
                title_text=("ln(1 + gas)" if c == K - 1 else None),
                row=row, col=2,
            )

            # ── Right: spam distribution (% within cluster) ─────────────────
            n_total = aggs.n_txs[c]
            if n_total > 0:
                pct = []
                for label in SPAM_ORDER:
                    code = SPAM_CODE[label]
                    pct.append(aggs.n_spam_label[c, code] / n_total * 100.0)
                fig.add_trace(
                    go.Bar(
                        x=SPAM_ORDER, y=pct,
                        marker_color=[SPAM_COLOR[t] for t in SPAM_ORDER],
                        showlegend=False,
                        hovertemplate=(
                            "%{x}: %{y:.1f}%<extra></extra>"
                        ),
                    ),
                    row=row, col=3,
                )
            fig.update_yaxes(title_text="% of cluster", row=row, col=3)
            fig.update_xaxes(
                tickangle=-25,
                title_text=("spam class" if c == K - 1 else None),
                row=row, col=3,
            )

    render_pipeline("log", start_row=4)
    render_pipeline("clr", start_row=4 + K_log)

    # ── Layout ──────────────────────────────────────────────────────────────
    total_height = 320 * n_rows + 240
    title_text = (
        "<b>Per-tx clustering — KMeans on log-gas vs KMeans on CLR features</b>"
        f"<br><sub>{n_priced_total:,} priced txs (FULL dataset) · "
        f"K range {K_RANGE[0]}..{K_RANGE[-1]} · "
        f"silhouette / t-SNE on {sample_size:,}-row sample</sub>"
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
        hovermode="closest",
        barmode="overlay",
    )
    fig.update_xaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.55)", mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.0,
                     linecolor="rgba(0,0,0,0.55)", mirror=True, ticks="outside")
    return fig


# ── Phase 1: fit MBK bank + reservoir sample, save to cache ─────────────────
def fit_phase(n_sample: int) -> None:
    import pickle
    rng = np.random.default_rng(RNG_SEED)

    paths = _per_tx_files()
    if not paths:
        raise SystemExit(f"No per_tx parquets found under {MULTIGAS_DIR}")
    print(f"[fit] streaming pass 1 over {len(paths)} per_tx parquets — "
          f"partial_fit MBK bank + reservoir (target {n_sample:,})...")
    t0 = time.time()
    (mbk_log, mbk_clr, log_mean, log_std,
     sample, X_log_s, X_clr_s, Lg_s, n_priced) = stream_fit(
        paths, n_sample, rng,
    )
    print(f"[fit] pass 1: {time.time()-t0:.1f}s, "
          f"{n_priced:,} priced rows, sample {len(sample):,}")

    diagnostics: dict[str, dict] = {}
    for fset, X_sample, mbk_bank in (("log", X_log_s, mbk_log),
                                     ("clr", X_clr_s, mbk_clr)):
        print(f"[fit] computing silhouette + WCSS for {fset}...")
        sils, wcss = compute_diagnostics(mbk_bank, X_sample, rng)
        diagnostics[fset] = {"sils": sils, "wcss": wcss}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "fit.pkl", "wb") as f:
        pickle.dump({
            "mbk_log": mbk_log, "mbk_clr": mbk_clr,
            "log_mean": log_mean, "log_std": log_std,
            "sils_log": diagnostics["log"]["sils"],
            "wcss_log": diagnostics["log"]["wcss"],
            "sils_clr": diagnostics["clr"]["sils"],
            "wcss_clr": diagnostics["clr"]["wcss"],
            "n_priced": n_priced,
        }, f)
    sample.write_parquet(CACHE_DIR / "sample.parquet")
    np.savez(CACHE_DIR / "features.npz",
             X_log=X_log_s, X_clr=X_clr_s, Lg=Lg_s)
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
    print("[aggregate] streaming pass 2 — predict + aggregate on full data...")
    t0 = time.time()
    aggs_log, aggs_clr, n_priced = stream_aggregate(
        paths,
        ck["mbk_log"][K_LOG], K_LOG,
        ck["mbk_clr"][K_CLR], K_CLR,
        ck["log_mean"], ck["log_std"],
        spam_table, bt_offsets, block_min, min_hour,
        loggas_edges,
    )
    print(f"[aggregate] pass 2: {time.time()-t0:.1f}s, "
          f"{n_priced:,} priced rows aggregated")

    np.savez(CACHE_DIR / "aggs_log.npz",
             K=K_LOG,
             n_txs=aggs_log.n_txs, n_spam_label=aggs_log.n_spam_label,
             hourly_gas=aggs_log.hourly_gas,
             loggas_hist=aggs_log.loggas_hist, vol_sum=aggs_log.vol_sum,
             loggas_edges=aggs_log.loggas_edges)
    np.savez(CACHE_DIR / "aggs_clr.npz",
             K=K_CLR,
             n_txs=aggs_clr.n_txs, n_spam_label=aggs_clr.n_spam_label,
             hourly_gas=aggs_clr.hourly_gas,
             loggas_hist=aggs_clr.loggas_hist, vol_sum=aggs_clr.vol_sum,
             loggas_edges=aggs_clr.loggas_edges)
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
    return a


# ── Phase 3: t-SNE + dashboard ──────────────────────────────────────────────
def plot_phase() -> None:
    import pickle
    rng = np.random.default_rng(RNG_SEED)

    print("[plot] loading caches...")
    with open(CACHE_DIR / "fit.pkl", "rb") as f:
        ck = pickle.load(f)
    feats = np.load(CACHE_DIR / "features.npz")
    X_log_s = feats["X_log"]
    X_clr_s = feats["X_clr"]
    sample = pl.read_parquet(CACHE_DIR / "sample.parquet")
    aggs_log = _aggs_from_npz(CACHE_DIR / "aggs_log.npz")
    aggs_clr = _aggs_from_npz(CACHE_DIR / "aggs_clr.npz")
    with open(CACHE_DIR / "aggregate_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    pipelines: dict[str, dict] = {
        "log": {"K": K_LOG, "sils": ck["sils_log"], "wcss": ck["wcss_log"],
                "aggs": aggs_log},
        "clr": {"K": K_CLR, "sils": ck["sils_clr"], "wcss": ck["wcss_clr"],
                "aggs": aggs_clr},
    }

    if len(sample) > N_TSNE:
        idx = rng.choice(len(sample), size=N_TSNE, replace=False)
    else:
        idx = np.arange(len(sample))
    for fset, X in (("log", X_log_s), ("clr", X_clr_s)):
        print(f"[plot] running t-SNE on {fset} ({len(idx):,} pts)...")
        t = time.time()
        pipelines[fset]["tsne"] = embed_tsne(X[idx], rng)
        mbk_bank = ck["mbk_log"] if fset == "log" else ck["mbk_clr"]
        pipelines[fset]["labels_tsne"] = mbk_bank[
            pipelines[fset]["K"]
        ].predict(X[idx])
        print(f"[plot] t-SNE {fset}: {time.time()-t:.1f}s")

    print("[plot] building dashboard...")
    fig = build_dashboard(
        pipelines,
        n_priced_total=meta["n_priced_full"],
        sample_size=len(sample), min_hour=meta["min_hour"],
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(OUT_HTML), include_plotlyjs="cdn", full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    print(f"[plot] saved {OUT_HTML}")

    for fset in ("log", "clr"):
        K = pipelines[fset]["K"]
        cm = pipelines[fset]["cluster_meta"]
        print(f"\n{PIPELINE_LABEL[fset]} (K={K}) — clusters ranked by "
              f"spam concentration (any signal):")
        ranked = sorted(cm.items(), key=lambda kv: -kv[1]["spam_pct"])
        for c, meta in ranked:
            print(f"  c{c}: {meta['size_pct']:5.1f}% size, "
                  f"{meta['spam_pct']:5.1f}% spam "
                  f"(vol={meta['n_vol']:,} rev={meta['n_rev']:,} "
                  f"both={meta['n_both']:,})  — {meta['tag']}  "
                  f"[{meta['n_txs']:,} txs]")


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
        # Orchestrator — run each phase in a fresh subprocess so the OS
        # reclaims the previous phase's heap before the next one starts.
        # That's the only reliable way to keep RSS bounded across the
        # full-data pipeline on macOS (Python doesn't return retained
        # memory to the OS after pandas-heavy operations).
        import subprocess, sys
        for phase in ("fit", "aggregate", "plot"):
            cmd = [sys.executable, "-u", str(_HERE / "tx_clustering.py"),
                   "--phase", phase, "--n-sample", str(args.n_sample)]
            print(f"\n=== orchestrator: launching `{' '.join(cmd)}` ===\n")
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
