# Arbitrum Dynamic Pricing — Revenue Impact Simulation

Standalone dynamic-pricing equations for ArbOS 51 and ArbOS 60, the main
multi-panel figure comparing them against on-chain data (internal EA
ClickHouse), and the transaction-clustering dashboard.

See `docs/arbos51_vs_arbos60_equations.md` for the full equation reference.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull source data (ClickHouse → data/ caches)
python scripts/fetch_data.py

# ArbOS 51 vs ArbOS 60 historical sim figure → figures/historical_sim.html
python scripts/historical_sim.py

# Tx-clustering dashboard → figures/clustering.html
python scripts/tx_clustering.py
```
