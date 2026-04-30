"""
Convert Tyler's per-tx multigas extracts (tar.gz of CSVs) into compressed
parquet files. Processes one tar at a time so the temporary CSV never
coexists with the next one — keeps peak disk usage bounded.

Input  : data/raw_tyler_archives/<one-tar-per-month>.tar.gz
Output : data/multigas_usage_extracts/<YYYY-MM>/per_tx.parquet  (zstd, level 6)
         data/multigas_usage_extracts/<YYYY-MM>/blocks.parquet  (zstd, level 6)

Each tar's inner CSV is deleted as soon as its parquet is written. The
input tar.gz is left untouched.

Add a new month: append an entry to TARS with (label, tar.gz filename,
inner dir, tx csv name, blocks csv name). All four months processed so far
share the same per-tx schema, so no schema work is needed for new months
unless Tyler changes the column set.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import time

import polars as pl

_HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"
EXTRACTS_DIR = DATA_DIR / "raw_tyler_archives"
OUT_DIR = DATA_DIR / "multigas_usage_extracts"

# (label, tar.gz filename, inner dir, tx csv name, blocks csv name)
TARS = [
    ("2025-10",
     "384761330_395488344_oct_2025_per_tx_gas_dim.tar.gz",
     "384761330_395488344",
     "384761330_395488344_tx.csv",
     "384761330_395488344_tx.blocks.csv"),
    ("2025-11",
     "Entropy Offchain Nov 2025 Gas Dim.tar.gz",
     "395488345_405863595",
     "395488345_405863595_tx.csv",
     "395488345_405863595_tx.blocks.csv"),
    ("2025-12",
     "Entropy Offchain Dec 2025 Gas Dim.tar.gz",
     "405863596_416593973_dec_2025_per_tx_gas_dim",
     "405863596_416593973_tx.csv",
     "405863596_416593973_tx.blocks.csv"),
    ("2026-01",
     "Entropy Offchain Jan 2026 Gas Dim.tar.gz",
     "416593974_427315178_jan_2026_per_tx_gas_dim",
     "416593974_427315178_per_tx_multigas.csv",
     "416593974_427315178_per_tx_multigas.blocks.csv"),
    ("2026-02",
     "Feb 2026 TX Gas Dimensions.tar.gz",
     "427315179_437025050_feb_2026_per_tx_gas_dim",
     "427315179_437025050_tx.csv",
     "427315179_437025050_tx.blocks.csv"),
    ("2026-03",
     "Gas Dimension Transactions Mar 2026.tar.gz",
     "437025051_447736930_mar_2026_per_tx_gas_dim",
     "437025051_447736930_per_tx_multigas.csv",
     "437025051_447736930_per_tx_multigas.blocks.csv"),
]

# Explicit tx-csv schema — avoids polars dtype-inference overhead and
# guards against silently casting a column to the wrong type.
TX_SCHEMA: dict[str, type] = {
    "block":             pl.Int64,
    "tx_hash":           pl.Utf8,
    "tx_sender":         pl.Utf8,
    "unknown":           pl.Int64,
    "computation":       pl.Int64,
    "historyGrowth":     pl.Int64,
    "storageAccessRead": pl.Int64,
    "storageAccessWrite":pl.Int64,
    "storageAccess":     pl.Int64,
    "storageGrowth":     pl.Int64,
    "singleDim":         pl.Int64,
    "l1Calldata":        pl.Int64,
    "l2Calldata":        pl.Int64,
    "wasmComputation":   pl.Int64,
    "refund":            pl.Int64,
    "total":             pl.Int64,
}
BLOCKS_SCHEMA: dict[str, type] = {
    "block_num":  pl.Int64,
    "block_time": pl.Utf8,        # parsed downstream; kept as string here
    "base_fee":   pl.Int64,
}


def _human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PB"


def _disk_free(path: pathlib.Path) -> int:
    return shutil.disk_usage(path).free


def _convert_one(label: str, tar_name: str, inner_dir: str,
                 tx_csv: str, blocks_csv: str) -> None:
    tar_path  = EXTRACTS_DIR / tar_name
    out_dir   = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    tx_pq      = out_dir / "per_tx.parquet"
    blocks_pq  = out_dir / "blocks.parquet"

    if tx_pq.exists() and blocks_pq.exists():
        print(f"[{label}] already converted — skipping")
        return

    extracted_dir   = EXTRACTS_DIR / inner_dir
    tx_csv_path     = extracted_dir / tx_csv
    blocks_csv_path = extracted_dir / blocks_csv

    print(f"\n[{label}] free disk before: {_human(_disk_free(DATA_DIR))}")
    if tx_csv_path.exists() and blocks_csv_path.exists():
        print(f"[{label}] CSVs already extracted — reusing")
    else:
        print(f"[{label}] extracting {tar_name} ...")
        t0 = time.time()
        subprocess.run(
            ["tar", "-xzf", str(tar_path), "-C", str(EXTRACTS_DIR)],
            check=True,
        )
        print(f"[{label}] extracted in {time.time() - t0:.1f}s, free: {_human(_disk_free(DATA_DIR))}")

    # Per-tx: convert, then delete the CSV immediately to free the bulk of
    # the disk before touching the (much smaller) blocks CSV.
    print(f"[{label}] converting per-tx CSV → parquet (zstd) ...")
    t0 = time.time()
    csv_size = tx_csv_path.stat().st_size
    (
        pl.scan_csv(str(tx_csv_path), schema_overrides=TX_SCHEMA)
          .sink_parquet(str(tx_pq), compression="zstd", compression_level=6)
    )
    print(
        f"[{label}] per_tx.parquet: {_human(tx_pq.stat().st_size)} "
        f"in {time.time() - t0:.1f}s (from {_human(csv_size)} CSV)"
    )
    tx_csv_path.unlink()
    print(f"[{label}] deleted per-tx CSV; free: {_human(_disk_free(DATA_DIR))}")

    # Blocks: same flow.
    print(f"[{label}] converting blocks CSV → parquet (zstd) ...")
    t0 = time.time()
    csv_size = blocks_csv_path.stat().st_size
    (
        pl.scan_csv(str(blocks_csv_path), schema_overrides=BLOCKS_SCHEMA)
          .sink_parquet(str(blocks_pq), compression="zstd", compression_level=6)
    )
    print(
        f"[{label}] blocks.parquet: {_human(blocks_pq.stat().st_size)} "
        f"in {time.time() - t0:.1f}s (from {_human(csv_size)} CSV)"
    )
    blocks_csv_path.unlink()
    shutil.rmtree(extracted_dir, ignore_errors=True)
    print(f"[{label}] free disk after: {_human(_disk_free(DATA_DIR))}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tar_args in TARS:
        try:
            _convert_one(*tar_args)
        except Exception as e:
            print(f"[{tar_args[0]}] FAILED: {e}", file=sys.stderr)
            raise

    print("\n=== Summary ===")
    for label, *_ in TARS:
        d = OUT_DIR / label
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            print(f"  {f.relative_to(DATA_DIR)}: {_human(f.stat().st_size)}")
    print(f"\nFree disk: {_human(_disk_free(DATA_DIR))}")


if __name__ == "__main__":
    main()
