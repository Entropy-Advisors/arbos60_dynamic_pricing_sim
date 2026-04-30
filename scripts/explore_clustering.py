"""
K=6 clustering bake-off on the per-tx multigas sample.

Goal: find the (algorithm, feature-set) combo that yields six *distinct* and
*balanced* clusters. Reports silhouette / CH / DB / size-balance / interpretable
resource signature per cluster, ranked.

Usage:
    python scripts/explore_clustering.py
    python scripts/explore_clustering.py --n-sample 50000
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    BisectingKMeans,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from tx_clustering import (
    GAS_COLS,
    RESOURCES,
    RESOURCE_LABEL,
    _per_tx_files,
    reservoir_sample,
)

K        = 6
N_AGG    = 12_000      # AgglomerativeClustering is O(N²) — fit on subset, propagate
N_METRIC = 8_000       # silhouette eval subset
RNG_SEED = 42


# ── Featurization ───────────────────────────────────────────────────────────
def featurize_all(sample: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build several candidate feature representations of the same sample."""
    G = sample[GAS_COLS].to_numpy(dtype=np.float64)
    total = sample["gas_total"].to_numpy(dtype=np.float64)

    # log-gas z-scored — magnitude + mix
    Lg = np.log1p(G)
    X_log = (Lg - Lg.mean(axis=0)) / Lg.std(axis=0).clip(min=1e-9)

    # CLR — pure compositional, scale-invariant
    W = G / total[:, None]
    Lw = np.log(W + 1e-3)
    X_clr = Lw - Lw.mean(axis=1, keepdims=True)

    # CLR + log(total_gas) z-scored as an extra dimension — mix + scale
    log_total = np.log1p(total)
    log_total_z = ((log_total - log_total.mean()) / log_total.std()).reshape(-1, 1)
    X_clr_scale = np.hstack([X_clr, log_total_z])

    # log-gas projected onto top-3 PCs — denoise / decorrelate
    pca = PCA(n_components=3, random_state=RNG_SEED)
    X_log_pca = pca.fit_transform(X_log)

    # Pure weights (simplex) — KMeans-naïve baseline; expected to be worst
    X_w = W

    return {
        "log_z":       X_log,
        "clr":         X_clr,
        "clr+scale":   X_clr_scale,
        "log_pca3":    X_log_pca,
        "weights":     X_w,
    }


# ── Algorithms ──────────────────────────────────────────────────────────────
def fit_kmeans(X: np.ndarray) -> np.ndarray:
    return KMeans(
        n_clusters=K, n_init=10, max_iter=300, random_state=RNG_SEED,
    ).fit_predict(X)


def fit_mbk(X: np.ndarray) -> np.ndarray:
    return MiniBatchKMeans(
        n_clusters=K, n_init=10, batch_size=8192, max_iter=300,
        random_state=RNG_SEED, reassignment_ratio=0.01,
    ).fit_predict(X)


def fit_bisect(X: np.ndarray) -> np.ndarray:
    return BisectingKMeans(
        n_clusters=K, n_init=5, random_state=RNG_SEED,
    ).fit_predict(X)


def fit_gmm_diag(X: np.ndarray) -> np.ndarray:
    return GaussianMixture(
        n_components=K, covariance_type="diag", n_init=3,
        max_iter=200, random_state=RNG_SEED,
    ).fit_predict(X)


def fit_gmm_full(X: np.ndarray) -> np.ndarray:
    return GaussianMixture(
        n_components=K, covariance_type="full", n_init=2,
        max_iter=200, random_state=RNG_SEED,
    ).fit_predict(X)


def fit_ward(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ward-linkage agglomerative on a subset, then nearest-centroid propagate."""
    n = len(X)
    sub_idx = rng.choice(n, size=min(N_AGG, n), replace=False)
    Xs = X[sub_idx]
    labels_sub = AgglomerativeClustering(
        n_clusters=K, linkage="ward",
    ).fit_predict(Xs)
    centroids = np.vstack([Xs[labels_sub == k].mean(axis=0) for k in range(K)])
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def fit_clr_then_split(
    X_clr: np.ndarray, log_total: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Hierarchical: CLR-KMeans K=3, then split each by log-total magnitude
    into 2 quantile bins → 6 clusters.  Asks 'first what TYPE of tx, then is
    it big or small?'  Useful sanity check vs flat methods."""
    base = KMeans(
        n_clusters=3, n_init=10, random_state=RNG_SEED,
    ).fit_predict(X_clr)
    out = np.empty(len(X_clr), dtype=np.int64)
    cidx = 0
    for b in range(3):
        mask = base == b
        if not mask.any():
            continue
        med = np.median(log_total[mask])
        big = log_total >= med
        out[mask & ~big] = cidx
        out[mask & big]  = cidx + 1
        cidx += 2
    return out


# ── Quality metrics ────────────────────────────────────────────────────────
def evaluate(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(X)
    idx = rng.choice(n, size=min(N_METRIC, n), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sil = float(silhouette_score(X[idx], labels[idx]))
        ch  = float(calinski_harabasz_score(X, labels))
        db  = float(davies_bouldin_score(X, labels))
    sizes = pd.Series(labels).value_counts(normalize=True).sort_index() * 100
    # Balance: std of cluster %-sizes around perfect 100/K. Lower = more even.
    balance = float(sizes.std())
    return {"sil": sil, "ch": ch, "db": db, "balance": balance,
            "sizes": sizes.to_dict()}


def describe_clusters(
    sample: pd.DataFrame, labels: np.ndarray,
) -> tuple[list[str], int]:
    """Return per-cluster tags and count of unique top-resource signatures.
    Volume-weighted resource mix per cluster — matches what the dashboard
    bars show."""
    G = sample[GAS_COLS].to_numpy(dtype=np.float64)
    total = sample["gas_total"].to_numpy(dtype=np.float64)
    descs: list[str] = []
    top_resources: list[tuple[str, str]] = []
    for c in range(K):
        mask = labels == c
        if not mask.any():
            descs.append(f"    c{c}: (empty)")
            top_resources.append(("", ""))
            continue
        gm = total[mask]
        vw = (G[mask] * 1.0).sum(axis=0) / gm.sum()       # share by volume
        order = np.argsort(vw)[::-1]
        n1 = RESOURCE_LABEL[RESOURCES[order[0]]]
        n2 = RESOURCE_LABEL[RESOURCES[order[1]]]
        p1, p2 = vw[order[0]], vw[order[1]]
        pct = mask.mean() * 100
        if p1 >= 0.80:
            tag = f"{n1}-heavy ({p1:.0%})"
        else:
            tag = f"{n1}+{n2} ({p1:.0%}/{p2:.0%})"
        descs.append(f"    c{c}: {pct:5.1f}%  {tag}")
        top_resources.append((n1, n2))
    n_distinct_top1 = len({t[0] for t in top_resources if t[0]})
    return descs, n_distinct_top1


# ── Driver ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-sample", type=int, default=30_000)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(RNG_SEED)

    paths = _per_tx_files()
    print(f"Streaming → reservoir sample N={args.n_sample:,}...")
    t0 = time.time()
    sample = reservoir_sample(paths, args.n_sample, rng)
    print(f"  {time.time()-t0:.1f}s, {len(sample):,} priced txs")

    feats = featurize_all(sample)
    log_total = np.log1p(sample["gas_total"].to_numpy(dtype=np.float64))

    # (algo_name, fn, applicable_feature_sets) — Ward needs the subset trick;
    # GMM-full is too slow on 8-D with N>20K, restrict to 7-D feature sets.
    algos: list[tuple[str, callable, list[str]]] = [
        ("kmeans10",      fit_kmeans,    list(feats)),
        ("mbk",           fit_mbk,       list(feats)),
        ("bisect_kmeans", fit_bisect,    list(feats)),
        ("gmm_diag",      fit_gmm_diag,  list(feats)),
        ("gmm_full",      fit_gmm_full,  ["log_z", "clr", "log_pca3", "weights"]),
        ("ward",          lambda X: fit_ward(X, rng), list(feats)),
    ]

    print(f"\nEvaluating combinations at K={K}...")
    rows = []
    for aname, fn, fset_list in algos:
        for fname in fset_list:
            X = feats[fname]
            t = time.time()
            try:
                labels = fn(X)
                m = evaluate(X, labels, rng)
                descs, n_top1 = describe_clusters(sample, labels)
                rows.append({
                    "algo": aname, "feat": fname,
                    "sil": m["sil"], "ch": m["ch"], "db": m["db"],
                    "balance": m["balance"], "n_top1_distinct": n_top1,
                    "elapsed": time.time() - t,
                    "labels": labels, "descs": descs,
                })
                print(f"  {aname:<14}/{fname:<10} "
                      f"sil={m['sil']:.3f} CH={m['ch']:>9,.0f} "
                      f"DB={m['db']:.3f} bal={m['balance']:>5.1f}% "
                      f"top1_uniq={n_top1}/{K}  ({time.time()-t:.1f}s)")
            except Exception as e:
                print(f"  {aname:<14}/{fname:<10} FAILED — {e}")

    # Hierarchical CLR→split-by-magnitude — its own row.
    t = time.time()
    labels_h = fit_clr_then_split(feats["clr"], log_total, rng)
    m = evaluate(feats["clr+scale"], labels_h, rng)
    descs_h, n_top1 = describe_clusters(sample, labels_h)
    rows.append({
        "algo": "clr_split2", "feat": "clr+scale",
        "sil": m["sil"], "ch": m["ch"], "db": m["db"],
        "balance": m["balance"], "n_top1_distinct": n_top1,
        "elapsed": time.time() - t,
        "labels": labels_h, "descs": descs_h,
    })
    print(f"  {'clr_split2':<14}/clr+scale "
          f"sil={m['sil']:.3f} CH={m['ch']:>9,.0f} "
          f"DB={m['db']:.3f} bal={m['balance']:>5.1f}% "
          f"top1_uniq={n_top1}/{K}  ({time.time()-t:.1f}s)")

    # ── Composite ranking ───────────────────────────────────────────────────
    # Rank by silhouette desc, but break ties favoring more-distinct clusters
    # (n_top1_distinct) and better size-balance (lower std).
    rows.sort(key=lambda r: (-r["sil"], -r["n_top1_distinct"], r["balance"]))
    print("\nTop 10 by silhouette (tie-break: distinct top-resources, balance):")
    print(f"  {'#':>2} {'algo/feat':<26} {'sil':>6} {'CH':>10} {'DB':>6} "
          f"{'bal':>5} {'top1':>4}")
    for i, r in enumerate(rows[:10], 1):
        print(f"  {i:>2} {r['algo']+'/'+r['feat']:<26} "
              f"{r['sil']:>6.3f} {r['ch']:>10,.0f} {r['db']:>6.3f} "
              f"{r['balance']:>4.1f}% {r['n_top1_distinct']:>2}/{K}")

    # ── Top-3 detailed cluster signatures ───────────────────────────────────
    print("\nDetailed cluster signatures for top 3 candidates:")
    for i, r in enumerate(rows[:3], 1):
        print(f"\n  [{i}] {r['algo']} on {r['feat']}  "
              f"sil={r['sil']:.3f}  CH={r['ch']:,.0f}  "
              f"DB={r['db']:.3f}  balance_std={r['balance']:.1f}%  "
              f"distinct_top_resources={r['n_top1_distinct']}/{K}")
        for line in r["descs"]:
            print(line)

    # ── A also report the candidates with most distinct top-resources ──────
    # — interpretability matters even if silhouette is mid.
    by_distinct = sorted(rows,
                         key=lambda r: (-r["n_top1_distinct"], -r["sil"]))[:5]
    print("\nMost-interpretable (distinct top-resources, then silhouette):")
    for r in by_distinct:
        print(f"  {r['algo']:<14}/{r['feat']:<10}  "
              f"top1_uniq={r['n_top1_distinct']}/{K}  sil={r['sil']:.3f}  "
              f"bal={r['balance']:.1f}%")


if __name__ == "__main__":
    main()
