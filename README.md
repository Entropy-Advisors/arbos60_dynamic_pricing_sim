# ArbOS 60 — clean export

Standalone dynamic-pricing equations for ArbOS 51 and ArbOS 60, the two
end-to-end figures we've been iterating on, and the clustering scaffolding.

## Layout

```
arbos60_clean/
├── README.md
├── requirements.txt
├── .venv/                       (created by `make venv` below)
└── scripts/
│   ├── arbos51.py               core equations — ArbOS 51 (single-dim ladder)
│   ├── arbos60.py               core equations — ArbOS 60 (per-resource pricing)
│   ├── plot_main.py             builds figures/main.html (uses both modules)
│   ├── tx_clustering.py         multi-K / multi-algo tx clustering dashboard
│   └── wallet_clustering.py     placeholder for wallet-level clustering
└── figures/
    ├── main.html                ArbOS 51 vs ArbOS 60 multi-panel chart
    └── clustering.html          tx-clustering dashboard (k-search + t-SNE)
```

## Quickstart

```bash
cd /Users/mohammedbenseddik/Documents/Dev/EA/arbos60_clean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Re-generate the main figure (~5 min, reads from the original data folder)
python scripts/plot_main.py

# Re-generate the clustering dashboard (~5 min)
python scripts/tx_clustering.py
```

## Source data

The scripts read from `/Users/mohammedbenseddik/Documents/Dev/EA/arbos60/data/`
(the original repo). Paths are absolute inside `plot_main.py` /
`tx_clustering.py`. Move or symlink the data if you relocate the project.

## Module APIs

### `arbos51.py`
- `ARBOS51_LADDER` — published Dia (T, A) values (6 constraints).
- `backlog_per_block(inflow, dt_s, T)` — Lindley-floored backlog (gas).
- `exponent_contribution(inflow, dt_s, T, A)` — `B_eff / (A·T)` (uncapped).
- `price_per_block(total_gas, dt_s)` — per-block base fee (gwei).
- `taylor4_exp(x)` — degree-4 Taylor exp matching nitro's
  `ApproxExpBasisPoints`.

### `arbos60.py`
- `SET_WEIGHTS`, `SET_LADDERS`, `PRICED_SYMBOLS` — proposal-doc constants.
- `per_block_resource_gas(blocks)` — split per-block gas into 6 resources.
- `compute_set_exponents(parts, dt_s)` — per-set raw `E_i` over time.
- `price_per_resource(parts, dt_s)` — per-block 6-vector `p_k` (BB
  max-over-sets form).
- `block_fee_and_gas(parts, dt_s)` — `(fee_block, priced_gas_block)`,
  the building block for any time-window aggregation.
- `backlogs_all_constraints(parts, dt_s)` — per-(set, constraint) backlog
  for diagnostics.

Both modules are pure NumPy / Polars; no I/O, no plotting.
