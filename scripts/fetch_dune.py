"""
Fetch results of a Dune Analytics query and save as parquet.

Reads `DUNE_API_KEY` from a .env file (path picked via `ARBOS_ENV_PATH`,
default `~/.config/arbos60/.env`) or from the process environment.

Default query: https://dune.com/queries/4537706 — pulled via the latest
cached execution; pass `--execute` to force a fresh run (counts against
your Dune execution credits).

Run:
    python scripts/fetch_dune.py
    python scripts/fetch_dune.py --query-id 4537706
    python scripts/fetch_dune.py --query-id 4537706 --execute
    python scripts/fetch_dune.py --query-id 4537706 \\
        --out data/dune/custom_name.parquet

Output:
    data/dune/<query_id>.parquet  (zstd)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import dotenv_values


_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT       = _HERE.parent
DEFAULT_OUT_DIR  = _REPO_ROOT / "data" / "dune_exports"
DEFAULT_QUERY_ID = 4537706
DUNE_API_BASE    = "https://api.dune.com/api/v1"

# Search order for .env: explicit ARBOS_ENV_PATH → project-local → XDG.
ENV_CANDIDATES: list[pathlib.Path] = [
    pathlib.Path(os.environ["ARBOS_ENV_PATH"])
        if os.environ.get("ARBOS_ENV_PATH") else None,  # type: ignore[list-item]
    _REPO_ROOT / ".env",
    pathlib.Path.home() / ".config" / "arbos60" / ".env",
]
ENV_CANDIDATES = [p for p in ENV_CANDIDATES if p is not None]


def _get_api_key() -> str:
    """DUNE_API_KEY lookup — first .env hit wins, then process env."""
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            cfg = dotenv_values(env_path)
            if cfg.get("DUNE_API_KEY"):
                return str(cfg["DUNE_API_KEY"])
    if os.environ.get("DUNE_API_KEY"):
        return os.environ["DUNE_API_KEY"]
    candidates = "\n  ".join(str(p) for p in ENV_CANDIDATES)
    raise SystemExit(
        "DUNE_API_KEY not found. Add it to one of:\n"
        f"  {candidates}\n"
        "or `export DUNE_API_KEY=...` in the shell."
    )


def _headers(api_key: str) -> dict[str, str]:
    return {"X-DUNE-API-KEY": api_key}


def stream_latest_to_parquet(
    query_id: int,
    api_key: str,
    out: pathlib.Path,
    page_size: int = 32_000,
) -> int:
    """Page through `/query/{id}/results` and write each page directly to a
    zstd parquet via pyarrow.ParquetWriter — keeps peak memory bounded to
    one page (~tens of MB) regardless of the total result size.

    Returns the total row count written.
    """
    url     = f"{DUNE_API_BASE}/query/{query_id}/results"
    h       = _headers(api_key)
    writer: pq.ParquetWriter | None = None
    total_rows  = 0
    offset      = 0
    api_total: int | None = None
    t0 = time.time()
    try:
        while True:
            r = requests.get(
                url, headers=h, timeout=180,
                params={"limit": page_size, "offset": offset},
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"Dune API {r.status_code} at offset={offset}: {r.text[:300]}"
                )
            payload = r.json()
            state   = payload.get("state")
            if state and state != "QUERY_STATE_COMPLETED":
                raise RuntimeError(
                    f"Latest result not in COMPLETED state (got {state}). "
                    "Re-run with --execute to force a fresh run."
                )
            result = payload.get("result", {}) or {}
            rows   = result.get("rows", [])
            meta   = result.get("metadata", {}) or {}
            if api_total is None:
                api_total = (
                    meta.get("total_row_count")
                    or meta.get("row_count")
                    or payload.get("total_row_count")
                )
                if api_total:
                    print(f"  total rows reported by Dune: {int(api_total):,}")
            if not rows:
                break
            df    = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    str(out), table.schema, compression="zstd",
                )
            writer.write_table(table)
            total_rows += len(rows)
            offset     += len(rows)
            elapsed = time.time() - t0
            print(
                f"    offset {offset:>10,}  (+{len(rows):>6,}) "
                f"  total {total_rows:>10,}"
                + (f" / {int(api_total):>10,}" if api_total else "")
                + f"  {elapsed:6.1f}s"
            )
            if len(rows) < page_size:
                break
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def execute_and_wait(
    query_id: int, api_key: str, poll_s: float = 5.0,
) -> pd.DataFrame:
    """Trigger a fresh execution, poll until completed, return the rows."""
    h = _headers(api_key)

    r = requests.post(
        f"{DUNE_API_BASE}/query/{query_id}/execute",
        headers=h, timeout=60,
    )
    r.raise_for_status()
    eid = r.json()["execution_id"]
    print(f"  execution_id: {eid}")

    while True:
        s = requests.get(
            f"{DUNE_API_BASE}/execution/{eid}/status",
            headers=h, timeout=60,
        ).json()
        state = s.get("state", "?")
        print(f"    state: {state}")
        if state == "QUERY_STATE_COMPLETED":
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise SystemExit(f"Dune execution {state}: {s}")
        time.sleep(poll_s)

    r = requests.get(
        f"{DUNE_API_BASE}/execution/{eid}/results",
        headers=h, timeout=180,
    )
    r.raise_for_status()
    rows = r.json().get("result", {}).get("rows", [])
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Fetch a Dune query into a parquet.")
    p.add_argument("--query-id", type=int, default=DEFAULT_QUERY_ID,
                   help=f"Dune query id (default {DEFAULT_QUERY_ID} — "
                        f"https://dune.com/queries/{DEFAULT_QUERY_ID})")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Output parquet path. "
                        "Default: data/dune/<query_id>.parquet")
    p.add_argument("--execute", action="store_true",
                   help="Force a fresh execution (uses Dune credits) instead "
                        "of the latest cached results")
    p.add_argument("--poll-s", type=float, default=5.0,
                   help="Polling interval in seconds (only with --execute)")
    args = p.parse_args()

    api_key = _get_api_key()
    out = args.out or DEFAULT_OUT_DIR / f"{args.query_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"fetching Dune query {args.query_id} "
          f"(https://dune.com/queries/{args.query_id})...")
    t0 = time.time()
    if args.execute:
        df = execute_and_wait(args.query_id, api_key, poll_s=args.poll_s)
        print(f"  rows: {len(df):,}, cols: {list(df.columns)}, "
              f"in {time.time()-t0:.1f}s")
        df.to_parquet(str(out), compression="zstd")
    else:
        n = stream_latest_to_parquet(args.query_id, api_key, out)
        print(f"  wrote {n:,} rows in {time.time()-t0:.1f}s")
    size_mb = out.stat().st_size / 1e6
    print(f"saved {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
