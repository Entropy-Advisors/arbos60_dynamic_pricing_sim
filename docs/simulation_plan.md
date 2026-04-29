# Plan: ArbOS 60 dynamic-pricing impact simulation

How we'll use the existing simulator to answer the business questions
raised in the EA / Offchain Labs thread (Tyler / Ed / Ben Berger).
One page — each question has a companion doc with the math.

## The four questions

1. **Revenue.** DAO revenue under ArbOS 60 vs ArbOS 51 vs the real
   on-chain outcome, replayed against actual Oct 2025 → today demand.
2. **`p_min` sensitivity.** How (1) shifts when the floor moves
   (0.01 → 0.02 → 0.05 → … gwei).
3. **Capacity.** Same demand at lower price, or more demand at the
   same price? Bar from Ed: ≥ 50 % uplift on the binding resource is
   interesting; ≤ 10 % is not.
4. **Workload-mix counterfactuals.** Cluster txs into types; perturb
   the mix; re-answer (1) and (3) under stress regimes (read-heavy,
   write-heavy, NFT-mint surges, bridge floods, …).

## Where each question lands in the pipeline

| # | Driver | Output | Status |
|---|---|---|---|
| 1 | `historical_sim.py` (already running, Dia-split honored) | `figures/historical_sim_oct_today.html` + period-total revenue Δ table | ✓ done |
| 2 | Sweep wrapper around (1): re-run with `Arbos5{1,60}GasPricing(p_min_gwei=...)` for `p_min ∈ {0.01, 0.02, 0.05, 0.10}` | revenue Δ vs `p_min` table + small chart | ~ 30 LOC |
| 3 | `scripts/capacity_sweep.py` per `capacity_simulation_plan.md` | `figures/capacity_sweep.html` + `min_k s_break(τ=0.95)` | ~ 120 LOC |
| 4 | `scripts/tx_type_decomposition.py` per `post_clustering_decomposition.md` + a perturbation driver that re-runs (1) and (3) on each scenario | revenue + capacity per mix | ~ 200 LOC |

## Phasing

**Phase 1 — revenue + `p_min` sensitivity (≈ 1 day).**
Extend `historical_sim.py` with a multi-config driver. Output: a small
HTML per `(p_min, engine)` pair and one comparison table giving total
DAO revenue (ETH, %) for each. Answers (1) and (2).

**Phase 2 — capacity (≈ 1 day).**
Build `scripts/capacity_sweep.py` per `capacity_simulation_plan.md` —
linear demand scaling, per-resource `frac_floor(s)` curves, headline
`min_k s_break`. Answers (3) under linear scaling (the right first
cut).

**Phase 3 — workload-mix counterfactuals (≈ 3 days).**
Build `scripts/tx_type_decomposition.py` (the clustering already
exists in `tx_clustering.py`; we need the NNLS step that maps hourly
gas vectors back to per-cluster tx counts). Then a thin perturbation
driver reuses Phase 1 + 2 engines on stress mixes. Removes the
linear-scaling assumption from (3) and lets us answer (4) directly.

## Decisions to nail before writing code

- **Window.** Oct 2025 → today.
  ArbOS 51 sim crosses the Dia activation boundary (2026-01-08 17:00 UTC)
  correctly via `Arbos51GasPricing.historical_price_per_block`.
  ArbOS 60 sim is bounded to Oct → Jan until Tyler ships Feb / Mar /
  Apr multigas extracts; it auto-extends as those land in
  `data/raw_tyler_archives/`.
- **R/W split.** Native `storageAccessRead` / `storageAccessWrite`
  columns from the new extracts; the 3.22:1 fixed ratio is gone.
  Ed's "make one cluster read-heavy, one write-heavy" falls out of
  fitting clusters on the natively-split data.
- **Headline metric per question.**
  (1) Δ revenue (ETH and %); (2) Δ revenue per Δ`p_min`;
  (3) `min_k s_break(τ = 0.95)`; (4) (1) + (3) at each mix.

## Decision criteria

- **Capacity.** ≥ 50 % uplift on the binding resource → ship the
  proposed targets; ≤ 10 % → leave the feature off; in between →
  re-tune.
- **Revenue.** No hard bar in the thread, but flag any config that
  drops DAO revenue by > 5 % at the same demand — that needs a
  counter-argument before sign-off.

## Companion docs

- `docs/arbos51_vs_arbos60_equations.md` — math both engines run.
- `docs/capacity_simulation_plan.md` — Phase 2 detail.
- `docs/post_clustering_decomposition.md` — Phase 3 detail.
