# ArbOS 51 vs ArbOS 60 — pricing equations

Reference for the equations implemented in `scripts/arbos51.py`
(`Arbos51GasPricing`) and `scripts/arbos60.py` (`Arbos60GasPricing`). Mirrors
the formula footer rendered at the bottom of `figures/historical_sim.html`.

## Notation (shared)

- $i \in \{1,2,3,4\}$ — set index. ArbOS 60 has four sets; ArbOS 51 has one
  (index suppressed).
- $j \in \{0,\dots,5\}$ — constraint index within a set. ArbOS 51 has 6
  constraints; ArbOS 60 has 6 per set except set 4 (one constraint).
- $k \in \{c, sw, sr, sg, hg, l2\}$ — priced resource index.
- $s$ — integer wall-clock UTC second (the **drain tick**). $t$ — block time.
  Multiple blocks can land in the same $s$; they share the same backlog.
- $T_{i,j}$ — target throughput / drain rate (Mgas/s) for constraint $(i,j)$.
- $A_{i,j}$ — adjustment window (seconds) for constraint $(i,j)$.
- $B_{i,j}(s)$ — backlog at the **start** of second $s$ (gas units, above-target
  accumulation).
- $a_{i,k}$ — weight of resource $k$ in set $i$'s weighted inequality.
- $p_{\min} = 0.02$ gwei — same floor for every resource (since ArbOS 51 Dia).
- Resources:
  $g_c$ = Computation (+ WasmComputation),
  $g_{sw}$ = Storage Write,
  $g_{sr}$ = Storage Read,
  $g_{sg}$ = Storage Growth,
  $g_{hg}$ = History Growth,
  $g_{l2}$ = L2 Calldata.

The $\exp$ in every price equation is approximated by the degree-4 Taylor
series, matching Nitro's `ApproxExpBasisPoints(x, 4)` in
[`arbos/l2pricing/model.go`](https://github.com/OffchainLabs/nitro/blob/master/arbos/l2pricing/model.go):

$$
\exp(x) \approx 1 + x + \tfrac{x^2}{2} + \tfrac{x^3}{6} + \tfrac{x^4}{24}.
$$

## ArbOS 51 — single-dim, 6-constraint geometric ladder

One backlog per constraint, one base fee per block, applied uniformly to
every gas unit.

**Backlog (1-second tick).** For each constraint $j$, with
$\text{inflow}_j(s) = \sum_{\text{blocks in }s} g_{\text{total}}$:

$$
B_j(s+1) \;=\; \max\!\big(0,\; B_j(s) + \text{inflow}_j(s) - T_j \cdot 1\big).
$$

Per-block backlog is the **start-of-second** value (i.e. $B_j$ at end of the
previous second).

**Per-block base fee.**

$$
p_{\text{block}} \;=\; p_{\min}\,\exp\!\Big(\sum_{j=0}^{5}\frac{B_j}{A_j\,T_j}\Big).
$$

No upper cap on the exponent (only a floor at 0).

**Per-tx pricing.** Every tx in block $N$ pays exactly $p_{\text{block}}[N]$:

$$
p_{tx} \;=\; p_{\text{block}}(\text{block of }tx),
\qquad
\text{fee}_{tx} \;=\; p_{tx} \cdot G_{tx},\;\;G_{tx} = \sum_k g_{tx,k}.
$$

**Ladder (on-chain Dia values).**

| $j$ | $T_j$ (Mgas/s) | $A_j$ (s) |
|---:|---:|---:|
| 0 | 10  | 86,400 |
| 1 | 14  | 13,485 |
| 2 | 20  |  2,105 |
| 3 | 29  |    329 |
| 4 | 41  |     52 |
| 5 | 60  |      9 |

## ArbOS 60 — per-resource pricing (4 sets, 19 constraints)

Each set has its own weighted inequality on the resource vector
$(g_c, g_{sw}, g_{sr}, g_{sg}, g_{hg}, g_{l2})$, its own per-constraint
backlogs, and contributes a raw exponent. Resource prices are obtained by a
**max-over-sets** projection.

**(0) Per-(set, constraint) backlog (1-second tick).**

$$
B_{i,j}(s+1) \;=\; \max\!\Big(0,\; B_{i,j}(s) + \sum_k a_{i,k}\,g_k(s) - T_{i,j}\cdot 1\Big).
$$

**(1) Per-set raw exponent and per-resource price.**

$$
E_i \;=\; \sum_j \frac{B_{i,j}}{A_{i,j}\,T_{i,j}},
\qquad
p_k \;=\; p_{\min}\,\exp\!\Big(\max_i\big\{\,a_{i,k}\,E_i\big\}\Big).
$$

Only the **binding** set lifts $p_k$ — sets that give resource $k$ a zero
weight ($a_{i,k}=0$) drop out of the max for that resource.

**(2) Per-tx fee — unnormalized inner product.**

$$
\text{fee}_{tx} \;=\; \sum_k g_{tx,k}\,p_k(\text{block of }tx).
$$

Dimensionally a fee (gwei · gas), not a per-gas price. Equivalent to
`price_per_tx` in the class.

**(3) Hourly gas-weighted average price.**

$$
\bar p_{\text{hr}} \;=\; \frac{\sum_{tx \in \text{hr}} \text{fee}_{tx}}
                              {\sum_{tx \in \text{hr}} G_{tx}},
\qquad G_{tx} = \sum_k g_{tx,k}.
$$

For the ArbOS 60 denominator we use **priced** gas only (excludes
`l1Calldata`, which ArbOS 60 prices separately via the L1 fee mechanism).

### Set inequalities

Let $\sum_k a_{i,k}\,g_k \le T_{i,j}$ for each constraint $j$ in set $i$:

- **Set 1 — Storage/Compute mix 1** ($i=1$):
  $\;1.00\,g_c + 0.67\,g_{sw} + 0.14\,g_{sr} + 0.06\,g_{sg} \le T_{1,j}$
- **Set 2 — Storage/Compute mix 2** ($i=2$):
  $\;0.0625\,g_c + 1.00\,g_{sw} + 0.21\,g_{sr} + 0.09\,g_{sg} \le T_{2,j}$
- **Set 3 — History Growth** ($i=3$):
  $\;1.00\,g_{hg} \le T_{3,j}$
- **Set 4 — Long-term Disk Growth** ($i=4$):
  $\;0.8812\,g_{sw} + 0.2526\,g_{sg} + 0.301\,g_{hg} + 1.00\,g_{l2} \le T_{4,j}$

### Per-set ladders

$T$ in Mgas/s, $A$ in seconds.

**Set 1 — Storage/Compute mix 1** ($i=1$)

| $j$ | $T_{1,j}$ | $A_{1,j}$ |
|---:|---:|---:|
| 0 | 15.40 | 10,000 |
| 1 | 20.41 |  2,861 |
| 2 | 27.06 |    819 |
| 3 | 35.86 |    234 |
| 4 | 47.53 |     67 |
| 5 | 63.00 |     19 |

**Set 2 — Storage/Compute mix 2** ($i=2$)

| $j$ | $T_{2,j}$ | $A_{2,j}$ |
|---:|---:|---:|
| 0 |  3.13 | 10,000 |
| 1 |  4.16 |  4,488 |
| 2 |  5.53 |  2,014 |
| 3 |  7.35 |    904 |
| 4 |  9.77 |    406 |
| 5 | 12.99 |    182 |

**Set 3 — History Growth** ($i=3$)

| $j$ | $T_{3,j}$ | $A_{3,j}$ |
|---:|---:|---:|
| 0 |  67.30 | 10,000 |
| 1 |  81.27 |  1,591 |
| 2 |  98.14 |    253 |
| 3 | 118.50 |     40 |
| 4 | 143.10 |      6 |
| 5 | 172.80 |      1 |

**Set 4 — Long-term Disk Growth** ($i=4$)

| $j$ | $T_{4,j}$ | $A_{4,j}$ |
|---:|---:|---:|
| 0 | 2.30 | 36,000 |

Set 4 has a single constraint.

## Implementation map

| Equation | ArbOS 51 method | ArbOS 60 method |
|---|---|---|
| 1-s backlog tick | `backlog_per_second` | `backlog_per_second` |
| Start-of-second per-block lookup | `backlog_per_block` | `backlog_per_block` |
| Sum of $B/(A\,T)$ over a ladder | `_ladder_exponent_from_inflow` | `_ladder_exponent_from_inflow` |
| Per-set raw $E_i$ | — | `compute_set_exponents` |
| Per-block / per-resource price | `price_per_block` | `price_per_resource` |
| Per-tx fee | `fee_per_tx` | `fee_per_tx` |
| Diagnostic per-(set, constraint) backlog | `backlogs_all_constraints` | `backlogs_all_constraints` |

All ladder operations are batched: one $\sum_k a_{i,k}\,g_k$ aggregation per
set, then a single 2-D `cumsum` + `minimum.accumulate` runs the backlog
recursion across every $T$ in that set's ladder simultaneously.
