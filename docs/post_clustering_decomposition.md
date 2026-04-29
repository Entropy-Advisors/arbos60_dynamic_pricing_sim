# Post-clustering: decomposing observed gas into transaction-type counts

After we've clustered transactions into $J$ types (e.g. our $J = 5$ from
`tx_clustering.py`), the natural next step is to use those clusters as a
**linear basis** to back out *how many transactions of each type arrived*
in any given period — block, minute, hour — from the observed per-resource
gas totals.

## TL;DR

We build a *menu* matrix $A$ where each column is one transaction type's
typical resource fingerprint, observe the period's total gas vector
$\mathbf{b}$, and solve $A\mathbf{x} = \mathbf{b}$ for tx-type counts
$\mathbf{x}$. The solver is non-negative least squares (`scipy.optimize.nnls`).

## Why we'd want this

Clustering tells us *what kinds of transactions exist*. The decomposition
step tells us *how many of each kind ran in a given window*. That unlocks:

1. **Workload composition over time** — stacked time series showing when
   each transaction type spikes (arbitrage, NFT mints, DEX swaps, …).
2. **Counterfactual pricing** — pick a hypothetical mix
   ("what if arbitrage doubles?") and feed it through the ArbOS 60
   simulator to see what the prices would have been.
3. **Per-type revenue attribution** — split L2 fee revenue across types.
4. **Better forecasting** — each type may have its own seasonality and
   growth rate. Forecasting them separately and summing is usually
   more accurate than forecasting one aggregate gas number.

## What $A$ actually is — concrete example first

**$A$ is a "menu" of transaction types. Each column is one menu item. Each
row tells you how much gas that menu item uses on one specific resource.**

Imagine three transaction types:

| Type | Description |
|---|---|
| **Type 1** — Token swap | small compute, lots of storage reads, some storage writes |
| **Type 2** — NFT mint | huge storage writes (creating new state), moderate compute |
| **Type 3** — Simple ETH transfer | tiny compute, almost nothing else |

For each type we measure the **average gas used per resource** on a typical
transaction of that type. The numbers below are illustrative:

| | **Type 1**<br>(Token swap) | **Type 2**<br>(NFT mint) | **Type 3**<br>(ETH transfer) |
|---|---:|---:|---:|
| Computation | 80,000 | 50,000 | 21,000 |
| Storage Write | 10,000 | 100,000 | 0 |
| Storage Read | 60,000 | 5,000 | 0 |
| Storage Growth | 1,000 | 25,000 | 0 |
| History Growth | 2,000 | 3,000 | 500 |
| L2 Calldata | 4,000 | 1,000 | 0 |

That table **is** $A$. Six rows (one per resource), three columns (one per
type). Each *column* is the resource fingerprint of one type — *"what
does a typical NFT mint look like?"* → read column 2.

In math:

$$A \;=\; \begin{pmatrix}
 80{,}000 & 50{,}000 & 21{,}000 \\
 10{,}000 & 100{,}000 & 0 \\
 60{,}000 & 5{,}000 & 0 \\
 1{,}000 & 25{,}000 & 0 \\
 2{,}000 & 3{,}000 & 500 \\
 4{,}000 & 1{,}000 & 0
\end{pmatrix}$$

## How we'd build $A$ from our actual clusters

We already have 5 clusters from `tx_clustering.py`. For each cluster, look
at all the transactions in it and compute the **average gas used per
resource**:

```python
# Build a 6 × 5 matrix from our existing cluster labels.
A = np.zeros((6, J))                          # K=6 resources, J=5 clusters
for j in range(J):
    txs_in_cluster_j = sample[labels == j]
    for k, resource in enumerate(PRICED_SYMBOLS):   # c, sw, sr, sg, hg, l2
        A[k, j] = txs_in_cluster_j[f"g_{resource}"].mean()
```

Reading the resulting columns:

```
                    Cluster0  Cluster1  Cluster2  Cluster3  Cluster4
Computation         ~120,000  ~100,000  ~80,000   ~80,000   ~45,000
Storage Write         ~5,000   ~10,000  ~12,000   ~25,000      ~500
Storage Read         ~50,000   ~65,000  ~50,000   ~75,000    ~2,000
Storage Growth          ~200      ~400  ~50,000      ~400      ~300
History Growth        ~3,000    ~4,000   ~5,000    ~4,000    ~1,000
L2 Calldata           ~3,000    ~2,000   ~2,000    ~1,500    ~1,000
```

(Numbers above are illustrative; actual ones come out of the data once we
compute them.)

## The math, in one paragraph

For some period (block, minute, hour), let
$\mathbf{b} = (G_c,\, G_{sw},\, G_{sr},\, G_{sg},\, G_{hg},\, G_{l2})$ be
the **observed totals per resource** in that period — a vector of 6
numbers. If $x_j$ counts the number of type-$j$ txs that arrived, then by
conservation:

$$A\,\mathbf{x} \;=\; \mathbf{b}, \qquad \mathbf{x} \in \mathbb{R}_{\geq 0}^J$$

i.e., *"$x_1$ copies of column 1, plus $x_2$ copies of column 2, …, summed
up, should equal the observed totals."* We solve for $\mathbf{x}$.

### Three regimes

| | Condition | Solver | What it means |
|---|---|---|---|
| Square | $K = J$ | $\mathbf{x} = A^{-1}\mathbf{b}$ | Unique solution. Ben's example: 5 dimensions, 5 types. |
| Overdetermined | $K > J$ | NNLS: $\min_{\mathbf{x}\geq 0}\|A\mathbf{x} - \mathbf{b}\|_2$ | Fewer types than resources — find the closest fit. **Our case** ($K=6$, $J=5$). |
| Underdetermined | $K < J$ | Multiple solutions exist; needs regularization | Too many types — can't disambiguate from totals alone. |

Counts can't be negative, so all three regimes use the constraint
$\mathbf{x} \geq 0$. Standard tool: `scipy.optimize.nnls(A, b)`.

## Worked example

Suppose in **hour H** we observe these total gas usages:

```
b = [
    Computation     50,000,000,
    Storage Write    5,000,000,
    Storage Read    30,000,000,
    Storage Growth  10,000,000,
    History Growth   2,000,000,
    L2 Calldata      1,500,000,
]
```

Question: **how many transactions of each type ran during hour H?**

We solve $A\mathbf{x} = \mathbf{b}$ for $\mathbf{x} = (x_0, x_1, x_2, x_3, x_4)$:

```python
from scipy.optimize import nnls
x_hat, residual = nnls(A, b)

# x_hat = [n_cluster0, n_cluster1, n_cluster2, n_cluster3, n_cluster4]
# residual = ||A·x_hat − b||_2 (how well the basis fits the period)
```

The solver returns the count vector that best reconstructs $\mathbf{b}$
from a non-negative combination of the $A$ columns. With our illustrative
numbers, hour H's heavy storage growth ($G_{sg} = 10\,\text{Mgas}$) only
gets explained if $x_2$ (the cluster with $a_{sg,2} \approx 50{,}000$) is
non-trivial — so the solver assigns count there, and the high storage
read $G_{sr}$ similarly forces $x_0$ or $x_3$ counts. Roughly:
"hour H was 200 × cluster-0 + 50 × cluster-1 + 100 × cluster-2 + 30 ×
cluster-3 + 1500 × cluster-4."

## Validating the basis

Three diagnostics tell us whether the 5 clusters are a usable basis:

1. **Per-period reconstruction error**
   $\|A\mathbf{x} - \mathbf{b}\|_2 / \|\mathbf{b}\|_2$.
   If this is consistently $<5\%$, the basis spans the observed workload.
2. **Conditioning of $A$**: `np.linalg.cond(A)`. Large condition numbers
   ($> 1000$) mean the basis vectors are nearly collinear → unstable
   decomposition.
3. **Per-resource reconstruction error**: do we systematically
   under-predict any one resource? That'd point to a missing type.

## What can break

1. **Mixed-character transactions** — a single tx might be a "blend" of
   types. NNLS gives count estimates that work *in aggregate*; per-tx
   assignments stay ambiguous. Aggregate hourly forecasts are still fine.
2. **Drift** — a cluster basis fitted on Jan 2026 will eventually misfit
   later workloads. Either refit the clusters periodically, or keep the
   types fixed and watch residuals climb over time.
3. **Sparse types** — if cluster $j$ has very few transactions, its
   centroid is noisy and its column in $A$ is unreliable. Filter tiny
   clusters before building $A$.
4. **Non-uniqueness in degenerate hours** — periods with very low gas may
   fit equally well with several different $\mathbf{x}$'s. The solver's
   residual flags those.

## What we'd need to implement

A new script `scripts/tx_type_decomposition.py` (≈ 100 lines):

1. Load existing cluster labels and the raw multigas → build $A$
   (mean per-resource gas vector per cluster) and cache it.
2. Stream per-block (or per-hour) aggregates → run NNLS per period.
3. Output:
   - Cached time series of $\mathbf{x}_t$ (parquet)
   - Stacked-area Plotly chart: $x_j(t)$ over time
   - Residual diagnostics: $\|A\mathbf{x}_t - \mathbf{b}_t\|$ vs
     $\|\mathbf{b}_t\|$ per period

Once that exists, counterfactual sweeps become a one-liner: pick an
$\mathbf{x}'$, compute $\mathbf{b}' = A\mathbf{x}'$, hand to the existing
pricing engine.

## Cheat-sheet

| Symbol | Meaning | Shape |
|---|---|---|
| $A$ | "menu" — column $j$ = average resource-gas fingerprint of tx type $j$ | $K \times J$ |
| $\mathbf{b}$ | observed total gas per resource in the period | $K$ |
| $\mathbf{x}$ | unknown — count of each tx type in the period | $J \geq 0$ |
| $A\mathbf{x} = \mathbf{b}$ | "$x_j$ copies of fingerprint $j$, summed, equals what we saw" | — |

## References

- `scripts/tx_clustering.py` — produces the cluster labels we'd build $A$ from.
- `scripts/arbos60.py` — the pricing engine that consumes $\mathbf{b}'$.
- `scripts/arbos51.py` — same, for the ArbOS 51 baseline comparison.
- `scipy.optimize.nnls` — non-negative least squares solver.
