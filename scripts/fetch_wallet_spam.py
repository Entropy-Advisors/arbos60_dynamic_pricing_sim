"""
Fetch per-wallet spam / bot classification from ClickHouse, scoped to the
block-number range our multigas data covers.

Single round-trip:
  1. Read min/max block_number from the per_tx parquet metadata (~1s).
  2. Substitute the bounds into sql/arbitrum_wallet_spam_classification.sql
     and run it once.
  3. Write the result to data/wallet_spam_classification.parquet.

CH credentials come from `~/.config/arbos60/.env` (override via
ARBOS_ENV_PATH), same convention as scripts/fetch_data.py.

Run:
    python scripts/fetch_wallet_spam.py
    python scripts/fetch_wallet_spam.py --start 2025-10-01 --end 2026-03-01
    python scripts/fetch_wallet_spam.py --revert-ratio 0.5 --revert-min-txs 100
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import pyarrow.parquet as pq

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
SQL_PATH = _ROOT / "sql" / "arbitrum_wallet_spam_classification.sql"
MULTIGAS_DIR = _ROOT / "data" / "multigas_usage_extracts"
OUT_PATH = _ROOT / "data" / "wallet_spam_classification.parquet"

# Look for credentials in the repo-local .env first, then ~/.config/arbos60/.env.
# Override either via ARBOS_ENV_PATH.
ENV_CANDIDATES = [
    pathlib.Path(p) for p in [
        os.environ.get("ARBOS_ENV_PATH"),
        _ROOT / ".env",
        pathlib.Path.home() / ".config" / "arbos60" / ".env",
    ] if p
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", default="2025-10-01",
                   help="Window start date (default 2025-10-01)")
    p.add_argument("--end", default="2026-03-01",
                   help="Window end date — exclusive (default 2026-03-01)")
    p.add_argument("--revert-min-txs", type=int, default=50,
                   help="Min same-day tx_count for revert-ratio flag (default 50)")
    p.add_argument("--revert-ratio", type=float, default=0.30,
                   help="Revert-ratio threshold for is_high_revert (default 0.30)")
    p.add_argument("--spam-day-frac", type=float, default=0.5,
                   help="Wallet `is_spam` if flagged on >= this fraction of "
                        "its active days (default 0.5)")
    p.add_argument("--block-min", type=int, default=None,
                   help="Override the auto-detected min block_number")
    p.add_argument("--block-max", type=int, default=None,
                   help="Override the auto-detected max block_number")
    return p.parse_args()


def block_range_from_multigas() -> tuple[int, int]:
    """Read min/max block_number across every per_tx parquet via Arrow
    column statistics — orders-of-magnitude cheaper than scanning the data."""
    paths = sorted(MULTIGAS_DIR.glob("*/per_tx.parquet"))
    if not paths:
        sys.exit(f"No per_tx parquets under {MULTIGAS_DIR}")
    block_min, block_max = None, None
    for p in paths:
        pf = pq.ParquetFile(str(p))
        schema = pf.schema_arrow
        block_idx = schema.get_field_index("block")
        for rg in range(pf.num_row_groups):
            stats = pf.metadata.row_group(rg).column(block_idx).statistics
            if stats is None:
                continue
            mn, mx = int(stats.min), int(stats.max)
            block_min = mn if block_min is None else min(block_min, mn)
            block_max = mx if block_max is None else max(block_max, mx)
    if block_min is None or block_max is None:
        sys.exit("Could not derive block range — parquet stats missing")
    return block_min, block_max


def load_ch_client():
    """dotenv_values across the candidate .env paths, first one wins."""
    from dotenv import dotenv_values
    import clickhouse_connect

    needed = ("CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD")
    cfg: dict[str, str] = {}
    for path in ENV_CANDIDATES:
        if not path.exists():
            continue
        loaded = dotenv_values(path)
        if all(loaded.get(k) for k in needed):
            cfg = loaded
            print(f"  CH credentials loaded from {path}")
            break
    missing = [k for k in needed if not cfg.get(k)]
    if missing:
        searched = ", ".join(str(p) for p in ENV_CANDIDATES)
        sys.exit(f"Missing {missing} — checked: {searched}")
    return clickhouse_connect.get_client(
        host=cfg["CLICKHOUSE_HOST"],
        user=cfg["CLICKHOUSE_USER"],
        password=cfg["CLICKHOUSE_PASSWORD"],
        port=int(cfg.get("CLICKHOUSE_PORT", 8443)),
        secure=str(cfg.get("CLICKHOUSE_SECURE", "true")).lower() == "true",
        settings={"max_execution_time": 1800},
    )


def render_sql(template: str, **subs: object) -> str:
    out = template
    for k, v in subs.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out


def main():
    args = parse_args()
    template = SQL_PATH.read_text()

    if args.block_min is not None and args.block_max is not None:
        block_min, block_max = args.block_min, args.block_max
        print(f"using user-specified block range: [{block_min:,}, {block_max:,}]")
    else:
        print("deriving block range from multigas parquet stats...")
        block_min, block_max = block_range_from_multigas()
        print(f"  blocks: [{block_min:,}, {block_max:,}]  "
              f"(span = {block_max - block_min + 1:,})")

    sql = render_sql(
        template,
        start_date=args.start,
        end_date=args.end,
        block_min=block_min,
        block_max=block_max,
        revert_min_txs=args.revert_min_txs,
        revert_ratio_threshold=f"{args.revert_ratio:.6f}",
        spam_day_frac=f"{args.spam_day_frac:.6f}",
    )

    client = load_ch_client()
    print(f"\nrunning CH query  date={args.start}→{args.end}  "
          f"blocks=[{block_min:,},{block_max:,}]  "
          f"revert>={args.revert_ratio:.0%} after {args.revert_min_txs} txs ...")
    t = time.time()
    df = client.query_df(sql)
    print(f"  fetched {len(df):,} wallet rows in {time.time()-t:.1f}s")

    n_spam       = int(df["is_spam"].sum())          if len(df) else 0
    n_spam_ever  = int(df["is_spam_ever"].sum())     if len(df) else 0
    print(f"\nclassification summary (wallet-level rollup):")
    print(f"  total wallets:                 {len(df):>10,}")
    print(f"  flagged on >= {args.spam_day_frac:.0%} of active days "
          f"(is_spam):  {n_spam:>10,}")
    print(f"  flagged on >=1 day (is_spam_ever):    "
          f"{n_spam_ever:>10,}")
    if len(df):
        print(f"  median active days / wallet:    "
              f"{df['n_days_active'].median():>10.0f}")
        print(f"  total revert txs in window:      "
              f"{int(df['revert_count'].sum()):>10,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
