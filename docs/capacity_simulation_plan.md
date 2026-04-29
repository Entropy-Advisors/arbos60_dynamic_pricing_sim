# Capacity uplift — simulation plan

Plan for answering: *"would the proposed ArbOS 60 ladder/target change give
≥ X% capacity uplift over the current configuration?"* — written against
the existing simulator pipeline so we can quote a number quickly when
Tyler asks for one.

The current bar is `< 10 %` is not worth shipping; `≥ 50 %` is.

## What "capacity" means here

**Capacity = the gas/sec rate at which the dynamic-pricing controller's
backlog stays drained** — the soft limit users feel as `p = p_min`, not
the hard `MaxTxGasLimit` block ceiling.

Two equivalent operational framings:

| Framing | Question it answers |
|---|---|
| **(A) Free-flow throughput** | "What is the highest sustained gas rate at which `frac(p_k = p_min) ≥ τ`?" (typically `τ = 0.95`) |
| **(B) Headroom multiplier** | "How much can we scale the historical workload by before that fraction drops below `τ`?" |

(B) is the right framing for Tyler's question because the answer maps
directly onto demand growth. `s_break = 1.5` means *the chain can absorb
1.5× the actual Oct–Jan workload at the same at-floor fraction*. A new
ladder gives "+50 % capacity" iff its `s_break` is `1.5×` the baseline's.

**Per-resource caveat.** Under ArbOS 60, capacity is a 6-vector — one
number per priced resource. The headline figure is `min_k s_break_k`,
because that's the one users hit first.

## Metrics

For each `(config, s, resource k)` triple we record:

1. **`frac_floor_k(s)`** — fraction of hours with `p_k = p_min` (i.e.
   `e_k ≈ 0`). Discrete, easy to read.
2. **`mean_excess_k(s)`** — `mean(p_k / p_min) − 1`. Continuous version
   of "how far above floor".
3. **`p99_k(s)`** — 99th-percentile `p_k`. Surfaces tail spikes (DEX
   bursts, mint waves) that mean/median hide.

Aggregate over `s`:

4. **`s_break_k(τ)`** — smallest `s` with `frac_floor_k(s) < τ`. The
   per-resource headroom.
5. **`s_break(τ) = min_k s_break_k(τ)`** — headline capacity headroom
   for the config.
6. **`binding_resource(s) = argmin_k frac_floor_k(s)`** — which
   dimension the controller binds on first as we scale demand. Tells
   us *where* a ladder change actually buys capacity.

Comparison metric: `capacity_uplift = s_break_proposed / s_break_current − 1`.
This is the "+X %" number that goes against the 10 / 50 % thresholds.

## Pipeline mapping (no new math, all parameters)

Everything we need already exists; the sweep is a parameter loop.

| Need | Already in repo |
|---|---|
| 1-second-tick backlog model | `scripts/arbos60.py:Arbos60GasPricing` |
| Configurable ladders / weights | `Arbos60GasPricing(set_ladders=..., set_weights=...)` constructor |
| Per-block resource gas (Oct → Jan) | `data/per_block_resources.parquet` (built by `historical_sim.py`) |
| Per-block-second timestamps | `Arbos60GasPricing.block_seconds_utc(block_time)` |
| Vectorised batched ladder pass | `_ladder_exponent_from_inflow` (one call per set; ~milliseconds) |
| Reference for the existing config | `Arbos60GasPricing.SET_LADDERS` |

What needs writing: a sweep driver + a small plot. ~120 lines total.

## Implementation — `scripts/capacity_sweep.py`

```python
import polars as pl
import numpy as np
from arbos60 import Arbos60GasPricing

P_MIN = Arbos60GasPricing.P_MIN_GWEI
PRICED_SYMBOLS = Arbos60GasPricing.PRICED_SYMBOLS


def capacity_sweep(
    parts: dict[str, np.ndarray],          # per-block resource gas (gas units)
    block_seconds: np.ndarray,
    candidate_configs: dict[str, dict],    # name → kwargs for Arbos60GasPricing
    s_values: list[float] = (0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
) -> pl.DataFrame:
    rows = []
    for name, kwargs in candidate_configs.items():
        a60 = Arbos60GasPricing(**kwargs)
        for s in s_values:
            parts_s = {k: v * s for k, v in parts.items()}
            prices, _ = a60.price_per_resource(parts_s, block_seconds)
            for k in PRICED_SYMBOLS:
                p = prices[k]
                rows.append({
                    "config":       name,
                    "s":            s,
                    "resource":     k,
                    "frac_floor":   float(np.mean(np.isclose(p, P_MIN, rtol=1e-9))),
                    "mean_excess":  float((p / P_MIN).mean() - 1.0),
                    "p99":          float(np.quantile(p, 0.99)),
                })
    return pl.DataFrame(rows)


def s_break_table(
    df: pl.DataFrame, threshold: float = 0.95,
) -> pl.DataFrame:
    """Smallest `s` per (config, resource) at which frac_floor drops
    below `threshold`. Returns a table ready for the headline number."""
    breaks = (
        df.filter(pl.col("frac_floor") < threshold)
          .group_by(["config", "resource"])
          .agg(pl.col("s").min().alias("s_break"))
    )
    return breaks
```

Driver: load `parts` from `per_block_resources.parquet` exactly the way
`historical_sim.py` does, define the configs to compare, write the
output to `data/cache/capacity_sweep.parquet`, render the figure.

## Configs to compare (initial)

- **`current_dia`** — `Arbos60GasPricing()` defaults (the 6-rung ladder
  documented in `docs/arbos51_vs_arbos60_equations.md`).
- **`tyler_proposed`** — TBD from Tyler. Plug into the sweep when we
  have the candidate `set_ladders`.
- **`single_constraint_baseline`** — degenerate config with one rung
  per set, useful as a sanity floor.
- **`tighter_long_window`** / **`looser_burst`** — small perturbations
  of the current ladder that bracket the proposed change.

## Outputs

1. **`figures/capacity_sweep.html`** — 6 panels (one per priced
   resource) showing `frac_floor(s)` curves, one trace per config,
   with `s_break(τ)` annotated. Plus a 7th panel summarising
   `min_k frac_floor_k(s)`.
2. **`data/cache/capacity_sweep.parquet`** — the long-form `(config, s,
   resource) → metrics` table. For ad-hoc analysis.
3. **A two-line summary** in the report, e.g.
   *"Proposed ladder buys +37 % capacity on Storage Write (binding on
   NFT-mint hours), +12 % on Computation, +5 % on the rest. Headline
   `min_k`: +12 %. Below the 50 % bar."*

## Validation

Independent of the sweep:

1. At `s = 1` the `frac_floor` per resource should match what we see in
   `historical_sim_oct_today.html`'s `p_k` panel (same data, same
   backlog math).
2. For the current Dia config at `s = 1`, the *binding resource by
   week* should match the per-resource backlog panels of
   `historical_sim_oct_today.html`. (Compute-bound during DEX-heavy
   weeks; storage-write during mint waves.)
3. As `τ → 1.0`, `s_break → s` at which the very first hour exceeds
   floor — useful as an extreme-case sanity check.

If 1+2 agree, the sweep numbers are trustworthy.

## Caveats / open questions

1. **Linear scaling assumption.** `s = 1.5×` multiplies every resource
   by `1.5×` uniformly. Real demand growth is rarely uniform — more
   DeFi → compute-heavy; more bridges → calldata-heavy; etc. Linear
   scaling is the right *first cut*; for a more honest counterfactual
   we need the tx-type decomposition (see
   `docs/post_clustering_decomposition.md`) so we can scale individual
   workload buckets independently.
2. **Backlog warm-up.** Larger `s` builds larger initial backlogs that
   take longer to drain — the at-floor fraction in the first ~ day of
   simulation is artificially low. Either drop the first 24 h before
   computing `frac_floor`, or compute on a stable mid-window slice.
3. **Threshold choice.** `τ = 0.95` is arbitrary. Plot `s_break` vs
   `τ ∈ {0.90, 0.95, 0.99}` so the answer doesn't depend on one knob.
4. **Time granularity.** Capacity is naturally per-second (drain rate
   `T` is per-second). Our simulator already ticks at 1-s, so we get
   this for free — just make sure `frac_floor` is computed *before*
   the hourly aggregation, not after, so a single congested second
   inside an otherwise-quiet hour doesn't get smoothed away.
5. **The single number.** Tyler's "50 %" almost certainly means the
   *binding-resource* number, not an average. Report `min_k s_break_k`
   as the headline; show the per-resource breakdown right next to it
   so we know which resource the gain came from.

## What to do next

If we want a deliverable to put against the 10 / 50 % bars:

1. Get candidate `set_ladders` from Tyler.
2. Write `scripts/capacity_sweep.py` (skeleton above) — ~ 1 hour.
3. Run the sweep (estimated ~ 5 min wall-clock at current scale).
4. Render `figures/capacity_sweep.html`, paste headline into report.

If the linear-scaling caveat is a blocker:

1. Implement `scripts/tx_type_decomposition.py` per
   `docs/post_clustering_decomposition.md`.
2. Construct counterfactual workloads by perturbing tx-type counts
   instead of uniform-scaling resource gas.
3. Re-run the sweep on each counterfactual.
