"""
Fetch Arbitrum revenue data from ClickHouse — both per-block and per-tx.

Runs the SQL queries in `sql/arbitrum_revenue_per_block.sql` and
`sql/arbitrum_revenue_per_tx.sql` for the configured date window, saves the
results to `data/onchain_blocks_transactions/`, and skips re-querying if the
cache already exists.

    sql/arbitrum_revenue_per_block.sql  → data/onchain_blocks_transactions/per_block.csv
    sql/arbitrum_revenue_per_tx.sql     → data/onchain_blocks_transactions/per_tx.parquet

Per-tx is fetched day by day to keep ClickHouse memory bounded; the per-block
query is a single round-trip (~5M rows).

Connection settings come from a .env file (CLICKHOUSE_HOST, _USER,
_PASSWORD; defaults port=8443, secure=True). Override the path via the
ARBOS_ENV_PATH environment variable.

Run:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --force      # re-fetch even if cache exists
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
SQL_DIR  = _ROOT / "sql"
DATA_DIR = _ROOT / "data" / "onchain_blocks_transactions"
ENV_PATH = pathlib.Path(
    os.environ.get(
        "ARBOS_ENV_PATH",
        str(pathlib.Path.home() / ".config" / "arbos60" / ".env"),
    )
)

START_DATE = "2026-01-09"
END_DATE   = "2026-02-01"   # exclusive

PER_BLOCK_SQL  = SQL_DIR / "arbitrum_revenue_per_block.sql"
PER_TX_SQL     = SQL_DIR / "arbitrum_revenue_per_tx.sql"
PER_BLOCK_CSV  = DATA_DIR / "per_block.csv"
PER_TX_PARQUET = DATA_DIR / "per_tx.parquet"


def _client():
    from dotenv import dotenv_values
    import clickhouse_connect
    cfg = dotenv_values(ENV_PATH)
    needed = ("CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD")
    missing = [k for k in needed if not cfg.get(k)]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {ENV_PATH}. "
            "Cannot connect to ClickHouse."
        )
    return clickhouse_connect.get_client(
        host=cfg["CLICKHOUSE_HOST"],
        user=cfg["CLICKHOUSE_USER"],
        password=cfg["CLICKHOUSE_PASSWORD"],
        port=int(cfg.get("CLICKHOUSE_PORT", 8443)),
        secure=str(cfg.get("CLICKHOUSE_SECURE", "true")).lower() == "true",
    )


def _read_sql(path: pathlib.Path, **subs) -> str:
    """Read a SQL file and substitute {{var}} placeholders."""
    sql = path.read_text()
    for k, v in subs.items():
        sql = sql.replace(f"{{{{{k}}}}}", str(v))
    return sql


def fetch_per_block(client, force: bool = False) -> pathlib.Path:
    """Run the per-block SQL once, write CSV."""
    if PER_BLOCK_CSV.exists() and not force:
        print(f"  per-block cache exists: {PER_BLOCK_CSV.name} "
              f"({PER_BLOCK_CSV.stat().st_size / 1e6:.1f} MB) — skip")
        return PER_BLOCK_CSV
    print("  running per-block query (single round-trip)...")
    sql = _read_sql(PER_BLOCK_SQL, start_date=START_DATE, end_date=END_DATE)
    df = client.query_df(sql)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PER_BLOCK_CSV, index=False)
    print(f"  wrote {PER_BLOCK_CSV.name}: {len(df):,} blocks, "
          f"{PER_BLOCK_CSV.stat().st_size / 1e6:.1f} MB")
    return PER_BLOCK_CSV


def fetch_per_tx(client, force: bool = False) -> pathlib.Path:
    """Run the per-tx SQL day-by-day (memory-bounded), write parquet."""
    if PER_TX_PARQUET.exists() and not force:
        print(f"  per-tx cache exists: {PER_TX_PARQUET.name} "
              f"({PER_TX_PARQUET.stat().st_size / 1e6:.1f} MB) — skip")
        return PER_TX_PARQUET

    chunks: list[pd.DataFrame] = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D", inclusive="left"):
        ds = f"{day:%Y-%m-%d}"
        next_ds = f"{day + pd.Timedelta(days=1):%Y-%m-%d}"
        sql = _read_sql(PER_TX_SQL, start_date=ds, end_date=next_ds)
        d = client.query_df(sql)
        # Compact dtypes before holding in memory.
        for col, t in [
            ("block_number", np.int64),
            ("eff_price_gwei", np.float64),
            ("gas_used", np.float64),
            ("gas_used_for_l1", np.float64),
            ("l2_gas", np.float64),
            ("l2_base", np.float64),
            ("l2_surplus", np.float64),
            ("l1_fees", np.float64),
            ("total_fees", np.float64),
        ]:
            if col in d.columns:
                d[col] = d[col].astype(t)
        chunks.append(d)
        print(f"    {ds}: {len(d):,} txs")

    df = pd.concat(chunks, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PER_TX_PARQUET, compression="snappy")
    print(f"  wrote {PER_TX_PARQUET.name}: {len(df):,} txs, "
          f"{PER_TX_PARQUET.stat().st_size / 1e6:.1f} MB")
    return PER_TX_PARQUET


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if cache exists")
    p.add_argument("--only", choices=["per-block", "per-tx"], default=None,
                   help="Fetch only one of the two (default: both)")
    args = p.parse_args()

    print(f"window: {START_DATE} → {END_DATE} (exclusive)")
    print(f"data dir: {DATA_DIR}")

    client = _client()
    try:
        if args.only != "per-tx":
            print("\nper-block:")
            fetch_per_block(client, force=args.force)
        if args.only != "per-block":
            print("\nper-tx:")
            fetch_per_tx(client, force=args.force)
    finally:
        client.close()


if __name__ == "__main__":
    main()
