"""
Multi-panel chart comparing L2 gas consumption to ArbOS 60 constraint
targets and simulating the resulting base fee against the ArbOS 51 baseline.

Source: Tyler's Jan 2026 per-tx multigas parquet
(`data/multigas_usage_extracts/2026-01/per_tx.parquet`), which has the
Storage Access read/write split natively — no R:W ratio assumed anywhere.

Run:
    python scripts/historical_sim.py

Outputs:
    figures/historical_sim.html
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# Import the standalone equation modules.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arbos51 import Arbos51GasPricing      # noqa: E402
from arbos60 import Arbos60GasPricing      # noqa: E402

# Default-config pricing engines; instantiated once and reused everywhere.
a51 = Arbos51GasPricing()
a60 = Arbos60GasPricing()


# ── Visual style ────────────────────────────────────────────────────────────
RESOURCE_COLORS = {
    "Computation":     "#1f77b4",   # blue
    "StorageWrite":    "#2ca02c",   # green
    "StorageRead":     "#98df8a",   # light green
    "StorageGrowth":   "#d62728",   # red
    "HistoryGrowth":   "#ff7f0e",   # orange
    "L2Calldata":      "#9467bd",   # purple
}


def _hours_x(df: pl.DataFrame, col: str = "hour") -> list:
    """Polars Datetime column → Python datetime list for Plotly."""
    return df[col].to_list()


# ── Data loading ────────────────────────────────────────────────────────────
# Per-tx multigas parquets — one per month under multigas_usage_extracts/.
# scan_parquet accepts the glob, treating all months as one lazy frame.
_DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
MULTIGAS_TX_GLOB = str(_DATA_ROOT / "multigas_usage_extracts" / "*" / "per_tx.parquet")
# Per-block resource sums derived once from every monthly parquet, cached
# at the data-root level so the cache spans the full multigas window.
PER_BLOCK_RES_PQ = _DATA_ROOT / "per_block_resources.parquet"
_DEFAULT_START = "2025-10-01"   # first day of multigas extract coverage

# Resource columns aggregated from per-tx → per-block. l1Calldata is included
# for total-gas accounting but not priced under ArbOS 60.
_RESOURCE_COLS = [
    "computation", "wasmComputation",
    "storageAccessRead", "storageAccessWrite",
    "storageGrowth", "historyGrowth",
    "l2Calldata", "l1Calldata",
]


def load_per_block(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if df["block_time"].dtype.base_type() == pl.Datetime and \
       getattr(df["block_time"].dtype, "time_zone", None):
        df = df.with_columns(pl.col("block_time").dt.replace_time_zone(None))
    if df["block_date"].dtype.base_type() == pl.Datetime:
        df = df.with_columns(pl.col("block_date").cast(pl.Date))
    return df


def build_per_block_resources(
    per_tx_glob: str = MULTIGAS_TX_GLOB,
    cache: pathlib.Path = PER_BLOCK_RES_PQ,
    force: bool = False,
) -> pl.DataFrame:
    """Sum per-tx multigas columns into per-block totals across every monthly
    parquet matching `per_tx_glob`. Cached as a single zstd parquet — one
    streaming-engine groupby pass."""
    if cache.exists() and not force:
        return pl.read_parquet(cache)
    print(f"  building per-block resources cache from {per_tx_glob} ...")
    res = (
        pl.scan_parquet(per_tx_glob)
          .select(["block"] + _RESOURCE_COLS)
          .group_by("block")
          .agg([pl.col(c).sum() for c in _RESOURCE_COLS])
          .sort("block")
          .collect(engine="streaming")
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(str(cache), compression="zstd")
    print(f"  cached {res.height:,} rows → {cache}")
    return res


def merge_blocks_with_resources(
    blocks: pl.DataFrame,
    per_block_res: pl.DataFrame,
    start_date: str = _DEFAULT_START,
) -> pl.DataFrame:
    if start_date:
        cutoff = datetime.strptime(start_date, "%Y-%m-%d").date()
        blocks = blocks.filter(pl.col("block_date") >= cutoff)
    blocks = blocks.with_columns([
        pl.col("block_time").dt.truncate("1h").alias("hour"),
        pl.col("block_date").cast(pl.Utf8).alias("day_str"),
    ])
    # Inner-join: only blocks that have multigas data (i.e. that fall inside
    # the Tyler-extract coverage window) get priced. Blocks past the latest
    # available month — e.g. Feb-onwards on-chain data with no multigas yet —
    # drop out cleanly until a new monthly extract is converted.
    merged = blocks.join(
        per_block_res.rename({"block": "block_number"}),
        on="block_number", how="inner",
    )
    return merged

# ── Re-exports from the pricing-engine classes ──────────────────────────────
# All equation logic lives in scripts/arbos51.py and scripts/arbos60.py.
# These thin wrappers adapt the polars-DataFrame call signature used in
# build_fig() to the numpy-array signatures the classes expect.

P_MIN_GWEI          = a51.p_min_gwei
ARBOS51_LADDER      = a51.ladder
SET_WEIGHTS         = a60.set_weights
SET_LADDERS         = a60.set_ladders
PRICED_SYMBOLS      = a60.PRICED_SYMBOLS
PRICED_SYMBOL_LABELS = a60.PRICED_SYMBOL_LABELS

PRICED_KINDS = [
    "Computation",      # = computation + wasmComputation
    "StorageWrite",     # storageAccessWrite
    "StorageRead",      # storageAccessRead
    "StorageGrowth",
    "HistoryGrowth",
    "L2Calldata",
]
PRICED_SYMBOL_COLORS = {
    "c":  "#1f77b4",   # Computation — blue
    "sw": "#2ca02c",   # Storage Write — green
    "sr": "#98df8a",   # Storage Read — light green
    "sg": "#d62728",   # Storage Growth — red
    "hg": "#ff7f0e",   # History Growth — orange
    "l2": "#9467bd",   # L2 Calldata — purple
}
SET_COLORS = {
    "First Set — Storage/Compute mix 1":  "#0a2d6e",
    "Second Set — Storage/Compute mix 2": "#b54a00",
    "Third Set — History Growth":         "#0f5f0f",
    "Fourth Set — Long-term Disk Growth": "#7a1b1b",
}
SEC_PER_HR = 3600.0


# Direct re-exports of the math primitives (used inside this script).
taylor4_exp            = a51.taylor4_exp
_backlog_per_block     = a51.backlog_per_block
_exponent_contribution = a51.exponent_contribution


def _block_seconds(blocks: pl.DataFrame) -> np.ndarray:
    """Per-block integer-second UTC timestamps (input to the 1s-tick model)."""
    return a51.block_seconds_utc(blocks["block_time"].to_numpy())


def price_arbos51_per_block(blocks: pl.DataFrame) -> np.ndarray:
    """ArbOS 51 per-block price honoring the Dia activation boundary —
    pre-Dia (single T = 7 Mgas/s, A = 102 s, p_min = 0.01 gwei) for blocks
    before 2026-01-08 17:00 UTC; Dia ladder + 0.02 gwei after."""
    return Arbos51GasPricing.historical_price_per_block(
        blocks["total_l2_gas"].to_numpy(),
        _block_seconds(blocks),
    )


def price_arbos60_per_resource(
    blocks: pl.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    parts  = a60.per_block_resource_gas(blocks)
    bs     = _block_seconds(blocks)
    prices, e_per_set = a60.price_per_resource(parts, bs)
    return prices, parts, e_per_set


def compute_arbos51_backlogs(
    blocks: pl.DataFrame,
) -> dict[tuple[float, float], np.ndarray]:
    return a51.backlogs_all_constraints(
        blocks["total_l2_gas"].to_numpy().astype(np.float64),
        _block_seconds(blocks),
    )


def compute_backlogs(
    blocks: pl.DataFrame,
) -> dict[str, dict[tuple[float, float], np.ndarray]]:
    parts = a60.per_block_resource_gas(blocks)
    bs    = _block_seconds(blocks)
    return a60.backlogs_all_constraints(parts, bs)



def hourly_gas_per_kind(blocks: pl.DataFrame) -> pl.DataFrame:
    """Hourly raw gas per priced ResourceKind, keyed by `hour` (gas units)."""
    df = blocks.with_columns([
        (pl.col("computation") + pl.col("wasmComputation")).alias("gas_Computation"),
        pl.col("storageAccessWrite").alias("gas_StorageWrite"),
        pl.col("storageAccessRead").alias("gas_StorageRead"),
        pl.col("storageGrowth").alias("gas_StorageGrowth"),
        pl.col("historyGrowth").alias("gas_HistoryGrowth"),
        pl.col("l2Calldata").alias("gas_L2Calldata"),
    ])
    return (
        df.group_by("hour")
          .agg([pl.col(f"gas_{k}").sum() for k in PRICED_KINDS])
          .sort("hour")
    )


def weighted_inflow_mgas_hr(
    rk_hr: pl.DataFrame, weights: dict[str, float],
) -> np.ndarray:
    """Hourly weighted inflow Σ_k w_k · g_k in Mgas/hour."""
    parts = {
        "c":  rk_hr["gas_Computation"].to_numpy(),
        "sw": rk_hr["gas_StorageWrite"].to_numpy(),
        "sr": rk_hr["gas_StorageRead"].to_numpy(),
        "sg": rk_hr["gas_StorageGrowth"].to_numpy(),
        "hg": rk_hr["gas_HistoryGrowth"].to_numpy(),
        "l2": rk_hr["gas_L2Calldata"].to_numpy(),
    }
    inflow = np.zeros(rk_hr.height, dtype=np.float64)
    for sym, w in weights.items():
        inflow = inflow + w * parts[sym]
    return inflow / 1e6  # Mgas/hour


def _ladder_line_style(i: int, n: int) -> tuple[str, float]:
    if n == 1:
        return "solid", 2.2
    frac = i / (n - 1)
    dash = "solid" if frac == 0 else ("dash" if frac < 0.5 else "dot")
    width = 2.0 - 0.8 * frac
    return dash, width


def _weights_latex(weights: dict[str, float]) -> str:
    """Return the LHS of the set's weighted inequality as LaTeX (no $ delimiters)."""
    pretty = {"c": "g_c", "sw": "g_{sw}", "sr": "g_{sr}",
              "sg": "g_{sg}", "hg": "g_{hg}", "l2": "g_{l2}"}
    parts = []
    for sym, w in weights.items():
        if w == 1.0:
            parts.append(pretty[sym])
        else:
            parts.append(f"{w:g}\\,{pretty[sym]}")
    return " + ".join(parts)


def _set_ladder_table_latex(set_name: str) -> str:
    """LaTeX `array` of the (j, T_{i,j}, A_{i,j}) constraints for one ArbOS 60 set."""
    ladder = SET_LADDERS[set_name]
    rows = []
    for j, (T, A) in enumerate(ladder):
        T_str = f"{T:g}"
        A_str = f"{A:,}".replace(",", "{,}")
        rows.append(rf"{j} & {T_str} & {A_str} \\ \hline")
    body = "".join(rows)
    return (
        r"\begin{array}{|c|r|r|}"
        r"\hline{}"
        r"j & T_{i,j}\ (\text{Mgas/s}) & A_{i,j}\ (\text{s}) \\ \hline{}"
        + body +
        r"\end{array}"
    )


def _arbos60_tables_latex() -> str:
    """4 per-set ladder tables side-by-side, with set name + index above each."""
    cells_titles = []
    cells_tables = []
    for i, set_name in enumerate(SET_WEIGHTS.keys()):
        short = set_name.split(" — ")[0]
        cells_titles.append(rf"\textbf{{{short}}}\ (i={i+1})")
        cells_tables.append(_set_ladder_table_latex(set_name))
    return (
        r"\begin{array}{cccc}"
        + " & ".join(cells_titles) + r" \\[2pt]"
        + " & ".join(cells_tables)
        + r"\end{array}"
    )


def _arbos60_inequalities_latex() -> str:
    """Bulleted LaTeX lines per set, with explicit i=… index annotation."""
    lines = [
        r"\quad{}\text{Let Set }i\in\{1,2,3,4\}\text{:}"
    ]
    for i, (set_name, weights) in enumerate(SET_WEIGHTS.items()):
        lhs = _weights_latex(weights)
        # Set name uses an em-dash in source — render as LaTeX --- so it
        # doesn't get eaten as a math operator.
        short = set_name.split(" — ")[0]
        lines.append(
            rf"\quad{{}}\quad{{}}\bullet\;\text{{{short} }}(i={i+1})\text{{: }}\;"
            rf"{lhs} \le T_{{{i+1},j}}"
        )
    return r" \\[2pt] ".join(lines)


# ── Per-tx hourly price aggregation (single lazy polars pipeline) ────────────
def aggregate_per_tx_hourly(
    blocks: pl.DataFrame,
    p51_pb: np.ndarray,
    p60_prices: dict[str, np.ndarray],
) -> pl.DataFrame:
    """
    Compute hourly gas-weighted prices and revenues by streaming the per-tx
    parquet through one lazy polars pipeline:

        fee_real_tx = real_block_price · g_tx_total
        fee_51_tx   = p51_block         · g_tx_total
        fee_60_tx   = Σ_k g_tx,k · p_k(block)
        p̄_hr       = Σ_tx fee_tx / Σ_tx gas_tx                 (per channel)

    Mathematically equivalent to the block-level fee-sum / gas-sum
    aggregation when prices are uniform within a block (always true for
    ArbOS 51 and 60). Stated explicitly here so the per-tx model is plain
    in code. ArbOS 60 denominator uses *priced* gas (excludes l1Calldata).
    """
    import time as _time

    # Per-block keyed prices, broadcast to per-tx via a lazy join.
    prices_pb = pl.DataFrame({
        "block":  blocks["block_number"].cast(pl.Int64),
        "hour":   blocks["hour"],
        "p_real": blocks["avg_eff_price_gwei"].fill_null(0.0).cast(pl.Float64),
        "p51":    pl.Series("p51", p51_pb, dtype=pl.Float64),
        "p_c":    pl.Series("p_c",  p60_prices["c"],  dtype=pl.Float64),
        "p_sw":   pl.Series("p_sw", p60_prices["sw"], dtype=pl.Float64),
        "p_sr":   pl.Series("p_sr", p60_prices["sr"], dtype=pl.Float64),
        "p_sg":   pl.Series("p_sg", p60_prices["sg"], dtype=pl.Float64),
        "p_hg":   pl.Series("p_hg", p60_prices["hg"], dtype=pl.Float64),
        "p_l2":   pl.Series("p_l2", p60_prices["l2"], dtype=pl.Float64),
    })

    print("  streaming per-tx parquet → hourly fees / gas / fee_k (lazy polars)...")
    t0 = _time.time()

    g_priced_expr = (
        pl.col("g_c") + pl.col("g_sw") + pl.col("g_sr")
        + pl.col("g_sg") + pl.col("g_hg") + pl.col("g_l2")
    )
    fee_p60_expr = (
        pl.col("g_c")  * pl.col("p_c")
        + pl.col("g_sw") * pl.col("p_sw")
        + pl.col("g_sr") * pl.col("p_sr")
        + pl.col("g_sg") * pl.col("p_sg")
        + pl.col("g_hg") * pl.col("p_hg")
        + pl.col("g_l2") * pl.col("p_l2")
    )

    df = (
        pl.scan_parquet(MULTIGAS_TX_GLOB)
          .select([
              "block",
              "computation", "wasmComputation",
              "storageAccessRead", "storageAccessWrite",
              "storageGrowth", "historyGrowth",
              "l1Calldata", "l2Calldata",
          ])
          # Inner-join trims rows for blocks outside the revenue-CSV window.
          .join(prices_pb.lazy(), on="block", how="inner")
          .with_columns([
              (pl.col("computation") + pl.col("wasmComputation")).cast(pl.Float64).alias("g_c"),
              pl.col("storageAccessWrite").cast(pl.Float64).alias("g_sw"),
              pl.col("storageAccessRead").cast(pl.Float64).alias("g_sr"),
              pl.col("storageGrowth").cast(pl.Float64).alias("g_sg"),
              pl.col("historyGrowth").cast(pl.Float64).alias("g_hg"),
              pl.col("l2Calldata").cast(pl.Float64).alias("g_l2"),
              pl.col("l1Calldata").cast(pl.Float64).alias("g_l1"),
          ])
          .with_columns([
              g_priced_expr.alias("priced_gas"),
              (g_priced_expr + pl.col("g_l1")).alias("total_gas"),
              fee_p60_expr.alias("p60_fee"),
          ])
          .with_columns([
              (pl.col("p_real") * pl.col("total_gas")).alias("real_fee"),
              (pl.col("p51")    * pl.col("total_gas")).alias("p51_fee"),
              # Per-resource fee_k = g_tx,k · p_k(block) — used for the
              # hourly per-resource price panel.
              (pl.col("g_c")  * pl.col("p_c")).alias("fee_c"),
              (pl.col("g_sw") * pl.col("p_sw")).alias("fee_sw"),
              (pl.col("g_sr") * pl.col("p_sr")).alias("fee_sr"),
              (pl.col("g_sg") * pl.col("p_sg")).alias("fee_sg"),
              (pl.col("g_hg") * pl.col("p_hg")).alias("fee_hg"),
              (pl.col("g_l2") * pl.col("p_l2")).alias("fee_l2"),
          ])
          .group_by("hour")
          .agg([
              pl.col("real_fee").sum().alias("_real_fee"),
              pl.col("p51_fee").sum().alias("_p51_fee"),
              pl.col("p60_fee").sum().alias("_p60_fee"),
              pl.col("total_gas").sum().alias("_total_gas"),
              pl.col("priced_gas").sum().alias("_priced_gas"),
              *[pl.col(f"fee_{k}").sum().alias(f"_fee_{k}") for k in PRICED_SYMBOLS],
              *[pl.col(f"g_{k}").sum().alias(f"_gas_{k}")  for k in PRICED_SYMBOLS],
          ])
          .sort("hour")
          .collect(engine="streaming")
    )
    print(f"    done in {_time.time() - t0:.1f}s, {df.height:,} hourly rows")

    df = df.with_columns([
        (pl.col("_real_fee") / pl.col("_total_gas")).alias("p_real_gwei"),
        (pl.col("_p51_fee")  / pl.col("_total_gas")).alias("p51_gwei"),
        (pl.col("_p60_fee")  / pl.col("_priced_gas")).alias("p60_gwei"),
        *[(pl.col(f"_fee_{k}") / pl.col(f"_gas_{k}")).alias(f"p_{k}_gwei")
          for k in PRICED_SYMBOLS],
    ])
    return df


def aggregate_per_block_hourly_wide(
    blocks_wide: pl.DataFrame,
    p51_pb_wide: np.ndarray,
) -> pl.DataFrame:
    """Hourly Real + ArbOS 51 aggregation across the full per_block.parquet
    window. Block-level math (avg_eff_price × total_gas, p51 × total_gas
    summed per hour, divided by total gas) — equivalent to the per-tx
    aggregation when prices are uniform within a block (always true here),
    but doesn't need per-tx multigas, so it covers blocks past the Tyler
    extract cutoff (Feb-onwards) too."""
    df = (
        blocks_wide
            .select([
                "hour", "block_number", "total_l2_gas", "total_l1_gas",
                "avg_eff_price_gwei",
            ])
            .with_columns([
                pl.Series("p51", p51_pb_wide, dtype=pl.Float64),
                pl.col("avg_eff_price_gwei").fill_null(0.0).alias("p_real"),
                (pl.col("total_l2_gas") + pl.col("total_l1_gas")).alias("total_gas"),
            ])
            .with_columns([
                (pl.col("p_real") * pl.col("total_gas")).alias("real_fee_gwei_gas"),
                (pl.col("p51")    * pl.col("total_gas")).alias("p51_fee_gwei_gas"),
            ])
            .group_by("hour")
            .agg([
                pl.col("real_fee_gwei_gas").sum().alias("_real_fee"),
                pl.col("p51_fee_gwei_gas").sum().alias("_p51_fee"),
                pl.col("total_gas").sum().alias("_total_gas"),
            ])
            .with_columns([
                (pl.col("_real_fee") / pl.col("_total_gas")).alias("p_real_gwei"),
                (pl.col("_p51_fee")  / pl.col("_total_gas")).alias("p51_gwei"),
            ])
            .sort("hour")
    )
    return df


def build_fig(blocks: pl.DataFrame, blocks_wide: pl.DataFrame | None = None) -> go.Figure:
    rk_hr = hourly_gas_per_kind(blocks)
    x = _hours_x(rk_hr)
    x_span = [x[0], x[-1]]

    # Per-block simulated prices, aggregated to gas-weighted hourly mean
    # alongside the observed on-chain price.
    #
    # Aggregation model: per tx, fee_tx = Σ_k g_{tx,k}·p_k(block); within an
    # hour Σ fee_tx / Σ gas_tx. Since all txs in a block share the same p_k
    # vector, this reduces per-block to (Σ_k p_k·G_k) / (Σ_k G_k) and at the
    # hourly level to Σ_block fee_block / Σ_block gas_block.
    print("  simulating ArbOS 51 prices (per-block, applied to every tx)...")
    p51_pb = price_arbos51_per_block(blocks)
    print("  simulating ArbOS 60 prices (per-resource max-over-sets)...")
    p60_prices, _, _ = price_arbos60_per_resource(blocks)
    print("  computing per-(set, constraint) backlogs...")
    backlogs_pb = compute_backlogs(blocks)
    print("  computing ArbOS 51 backlogs...")
    arbos51_backlogs_pb = compute_arbos51_backlogs(blocks)

    # Per-tx → hourly aggregation (multigas window only — Oct-Jan):
    # streams the tx-level multigas parquets, computes fee_tx for each tx
    # (ArbOS 60 inner-product needs per-resource gas), then Σ fee / Σ gas
    # per hour. ArbOS 60 + per-resource lines come from this DF.
    prices_hr = aggregate_per_tx_hourly(blocks, p51_pb, p60_prices)
    x_price60 = _hours_x(prices_hr)
    p60_hr    = prices_hr["p60_gwei"].to_numpy()
    pk_hr     = {k: prices_hr[f"p_{k}_gwei"].to_numpy() for k in PRICED_SYMBOLS}
    rev_60_eth = prices_hr["_p60_fee"].to_numpy() / 1e9

    # Wide-window block-level aggregation (Oct→Apr) for the Real + ArbOS 51
    # lines. ArbOS 51 needs only total_l2_gas (already in per_block.parquet),
    # so it can be priced for blocks past the multigas cutoff. Uses
    # `blocks_wide` if provided; otherwise falls back to the narrow `blocks`.
    print("  aggregating Real + ArbOS 51 hourly (wide window)...")
    if blocks_wide is None:
        blocks_wide = blocks
        p51_pb_wide = p51_pb
    else:
        p51_pb_wide = price_arbos51_per_block(blocks_wide)
    wide_hr = aggregate_per_block_hourly_wide(blocks_wide, p51_pb_wide)
    x_wide   = _hours_x(wide_hr)
    x_span_wide = [x_wide[0], x_wide[-1]]
    p51_hr   = wide_hr["p51_gwei"].to_numpy()
    preal_hr = wide_hr["p_real_gwei"].to_numpy()
    rev_real_eth = wide_hr["_real_fee"].to_numpy() / 1e9
    rev_51_eth   = wide_hr["_p51_fee"].to_numpy()  / 1e9

    # Hourly mean backlog per (set, constraint) in Mgas — uses a flat schema with
    # generated column names to round-trip through the polars groupby.
    bl_col: dict[tuple[str, float, float], str] = {}
    bl_series: list[pl.Series] = []
    for s_idx, (set_name, b_per_constraint) in enumerate(backlogs_pb.items()):
        for j, ((T, A), B) in enumerate(b_per_constraint.items()):
            col = f"bl_{s_idx}_{j}"
            bl_col[(set_name, T, A)] = col
            bl_series.append(pl.Series(col, B / 1e6))   # gas → Mgas

    bl_hr = (
        blocks.select(["hour"])
              .with_columns(bl_series)
              .group_by("hour")
              .agg([pl.col(c).mean() for c in bl_col.values()])
              .sort("hour")
    )
    x_bl = _hours_x(bl_hr)

    # ArbOS 51 backlogs — same schema, single namespace.
    bl51_col: dict[tuple[float, float], str] = {}
    bl51_series: list[pl.Series] = []
    for j, ((T, A), B) in enumerate(arbos51_backlogs_pb.items()):
        col = f"bl51_{j}"
        bl51_col[(T, A)] = col
        bl51_series.append(pl.Series(col, B / 1e6))

    bl51_hr = (
        blocks.select(["hour"])
              .with_columns(bl51_series)
              .group_by("hour")
              .agg([pl.col(c).mean() for c in bl51_col.values()])
              .sort("hour")
    )

    set_names = list(SET_WEIGHTS.keys())
    # gas bars + weight bars + 4 set inflow panels + ArbOS 51 backlog panel
    # + 4 ArbOS 60 set backlog panels + per-resource p_k panel + price panel
    # gas bars + weights bars + 4 inflow panels + ArbOS 51 backlog + 4
    # ArbOS 60 backlog panels + per-resource p_k panel + price-compare
    # panel + revenue-compare panel  =  14 panels.
    n_panels = 1 + 1 + len(set_names) + 1 + len(set_names) + 1 + 1 + 1

    # Equal height for every subplot — Plotly normalizes these internally.
    row_heights = [1.0] * n_panels

    # Each subplot title is wrapped as a single LaTeX math region with
    # \text{} for prose — mixing inline math and plain text in one title
    # drops the prose when rendered to a static PNG via Kaleido.
    titles = [
        r"$\text{Hourly L2 gas per priced ResourceKind "
        r"(Computation includes WASM; L1Calldata excluded)}$",
        r"$\text{Hourly resource weights "
        r"(\% of priced gas; same color scheme as above)}$"
    ]
    for s in set_names:
        w_latex = _weights_latex(SET_WEIGHTS[s])
        title_text = s.replace("—", "---")
        titles.append(
            rf"$\text{{{title_text}: }}"
            rf"{w_latex}\;\text{{(Mgas/hr) vs ladder }}T_{{i,j}}$"
        )
    titles.append(
        r"$\text{ArbOS 51 backlogs }B_j\text{ "
        r"(Mgas, hourly mean; dashed = }A_j\!\cdot\!T_j\text{)}$"
    )
    for s in set_names:
        title_text = s.replace("—", "---")
        titles.append(
            rf"$\text{{Backlog }}B_{{i,j}}\text{{ for {title_text} }}"
            rf"\text{{(Mgas, hourly mean; dashed = }}A_{{i,j}}\!\cdot\!T_{{i,j}}\text{{)}}$"
        )
    titles.append(
        r"$\text{ArbOS 60 per-resource price }p_k\text{ "
        r"(hourly gas-weighted mean, gwei)}$"
    )
    titles.append(
        r"$\text{Gas price --- observed on-chain vs simulated ArbOS 51 "
        r"vs simulated ArbOS 60. Hourly gas-weighted mean.}$"
    )
    titles.append(
        r"$\text{Hourly L2 revenue (ETH) --- observed on-chain vs simulated "
        r"ArbOS 51 vs simulated ArbOS 60}$"
    )

    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        # 0.020 × (n_panels-1) gaps. With 13 panels that's 0.24 of plot area
        # for spacing. Smaller spacing → larger, more equal-looking panels.
        vertical_spacing=0.020,
        subplot_titles=titles,
    )

    # ── Panel 1: stacked ResourceKind bars ──────────────────────────────────
    cum = np.zeros(rk_hr.height)
    for k in PRICED_KINDS:
        y = rk_hr[f"gas_{k}"].to_numpy() / 1e6
        label = "Computation (+ WASM)" if k == "Computation" else k
        fig.add_trace(
            go.Bar(
                x=x, y=y, base=cum, name=label,
                marker_color=RESOURCE_COLORS.get(k, "#888"),
                marker_line_width=0,
                hovertemplate=f"{label}: %{{y:,.0f}} Mgas<extra></extra>",
                legendgroup="resources",
                legendgrouptitle_text="Resource kinds",
            ),
            row=1, col=1,
        )
        cum = cum + y

    # ── Panel 2: stacked weights (% of priced gas) ─────────────────────────
    # Same colors and stacking as panel 1, but each bar normalized so the
    # column reaches 100%. Hides the absolute scale and surfaces the mix.
    priced_total_hr = np.zeros(rk_hr.height)
    for k in PRICED_KINDS:
        priced_total_hr = priced_total_hr + rk_hr[f"gas_{k}"].to_numpy()
    priced_total_safe = np.where(priced_total_hr > 0, priced_total_hr, 1.0)

    cum_pct = np.zeros(rk_hr.height)
    for k in PRICED_KINDS:
        share_pct = rk_hr[f"gas_{k}"].to_numpy() / priced_total_safe * 100.0
        share_pct = np.where(priced_total_hr > 0, share_pct, 0.0)
        label = "Computation (+ WASM)" if k == "Computation" else k
        fig.add_trace(
            go.Bar(
                x=x, y=share_pct, base=cum_pct, name=f"{label} (%)",
                marker_color=RESOURCE_COLORS.get(k, "#888"),
                marker_line_width=0,
                hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
                legendgroup="resources",
                showlegend=False,
            ),
            row=2, col=1,
        )
        cum_pct = cum_pct + share_pct
    fig.update_yaxes(
        title_text="% of priced gas",
        range=[0, 100],
        row=2, col=1,
    )

    # ── Panels 2..5: weighted inflow line (ladder targets shown only as a
    #                  fitted y-axis ceiling, no horizontal reference lines) ─
    for panel_idx, set_name in enumerate(set_names, start=3):
        weights = SET_WEIGHTS[set_name]
        color   = SET_COLORS[set_name]

        inflow = weighted_inflow_mgas_hr(rk_hr, weights)

        fig.add_trace(
            go.Scatter(
                x=x, y=inflow,
                mode="lines+markers",
                name=set_name,
                line=dict(color=color, width=1.3),
                marker=dict(
                    color=color,
                    size=4,
                    symbol="circle",
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"<b>{set_name}</b><br>"
                    f"Weighted inflow: %{{y:,.1f}} Mgas/hr<extra></extra>"
                ),
                legendgroup=set_name,
                legendgrouptitle_text=set_name,
                showlegend=True,
            ),
            row=panel_idx, col=1,
        )

        # Y-axis ceiling: just the inflow series with a bit of headroom.
        y_ceiling = float(np.max(inflow)) * 1.10 if np.max(inflow) > 0 else 1.0
        fig.update_yaxes(
            title_text="Mgas/hr",
            range=[0, y_ceiling],
            row=panel_idx, col=1,
        )

    # ── ArbOS 51 backlog panel (single set, 6 constraints) ───────────────────────
    # Layout: gas (1) + weights (2) + 4 inflows (3..6) → ArbOS 51 backlog at 7
    arbos51_backlog_row = 1 + 1 + len(set_names) + 1
    arbos51_color = "#d62728"
    arbos51_constraints = list(arbos51_backlogs_pb.keys())
    n51 = len(arbos51_constraints)
    bl51_panel_max = 0.0
    for j, (T, A) in enumerate(arbos51_constraints):
        y = bl51_hr[bl51_col[(T, A)]].to_numpy()
        bl51_panel_max = max(bl51_panel_max, float(np.nanmax(y)) if y.size else 0.0)
        dash, width = _ladder_line_style(j, n51)
        fig.add_trace(
            go.Scatter(
                x=x_bl, y=y, mode="lines",
                name=f"ArbOS 51 C{j} (T={T} Mgas/s, A={A}s)",
                line=dict(color=arbos51_color, dash=dash, width=width),
                hovertemplate=(
                    f"ArbOS 51 C{j} (T={T} Mgas/s, A={A}s): "
                    f"B = %{{y:,.1f}} Mgas<extra></extra>"
                ),
                legendgroup="backlog_arbos51",
                legendgrouptitle_text=("Backlog — ArbOS 51" if j == 0 else None),
            ),
            row=arbos51_backlog_row, col=1,
        )
    # Cap y-axis at ~3× the max actual backlog so the ladder constraints that
    # actually accumulate backlog (high j) stay readable. Constraint 0's A·T is
    # 864,000 Mgas and would otherwise dominate; the threshold lines for
    # high-A·T constraints extend past the visible range and are clipped.
    bl51_y_cap = max(bl51_panel_max * 3.0, 1.0)
    for j, (T, A) in enumerate(arbos51_constraints):
        AT_mgas = T * A
        dash, width = _ladder_line_style(j, n51)
        fig.add_trace(
            go.Scatter(
                x=x_span, y=[AT_mgas, AT_mgas], mode="lines",
                name=f"ArbOS 51 C{j} A·T = {AT_mgas:,.0f} Mgas",
                line=dict(color=arbos51_color, dash="dash",
                          width=max(width - 0.4, 0.7)),
                opacity=0.55,
                hovertemplate=(
                    f"ArbOS 51 C{j} threshold "
                    f"A·T = {AT_mgas:,.0f} Mgas<extra></extra>"
                ),
                legendgroup="backlog_arbos51",
                showlegend=True,
            ),
            row=arbos51_backlog_row, col=1,
        )
    fig.update_yaxes(
        title_text="Mgas",
        range=[0, bl51_y_cap],
        row=arbos51_backlog_row, col=1,
    )

    # ── ArbOS 60 backlog panels: one per set ───────────────────────────────
    # Layout: gas (1) + weights (2) + 4 inflows (3..6) + ArbOS 51 backlog (7)
    # → first ArbOS 60 backlog panel at row 8.
    backlog_row_start = arbos51_backlog_row + 1
    for s_idx, (set_name, b_per_constraint) in enumerate(backlogs_pb.items()):
        panel_idx = backlog_row_start + s_idx
        color     = SET_COLORS[set_name]
        constraints     = list(b_per_constraint.keys())
        n_constraint    = len(constraints)
        short     = set_name.split(" — ")[0]

        bl_panel_max = 0.0
        for j, (T, A) in enumerate(constraints):
            col = bl_col[(set_name, T, A)]
            y   = bl_hr[col].to_numpy()
            bl_panel_max = max(bl_panel_max, float(np.nanmax(y)) if y.size else 0.0)
            dash, width = _ladder_line_style(j, n_constraint)
            fig.add_trace(
                go.Scatter(
                    x=x_bl, y=y,
                    mode="lines",
                    name=f"{short} C{j} (T={T} Mgas/s, A={A}s)",
                    line=dict(color=color, dash=dash, width=width),
                    hovertemplate=(
                        f"{set_name} C{j} "
                        f"(T={T} Mgas/s, A={A}s): "
                        f"B = %{{y:,.1f}} Mgas<extra></extra>"
                    ),
                    legendgroup=f"backlog_{s_idx}",
                    legendgrouptitle_text=(
                        f"Backlog — {short}" if j == 0 else None
                    ),
                ),
                row=panel_idx, col=1,
            )

        # Horizontal reference line at A·T (Mgas) per constraint — same dash/width
        # so each backlog line is paired visually with its A·T threshold.
        for j, (T, A) in enumerate(constraints):
            AT_mgas     = T * A   # T(Mgas/s) × A(s) = Mgas
            dash, width = _ladder_line_style(j, n_constraint)
            bl_panel_max = max(bl_panel_max, AT_mgas)
            fig.add_trace(
                go.Scatter(
                    x=x_span, y=[AT_mgas, AT_mgas],
                    mode="lines",
                    name=f"{short} C{j} A·T = {AT_mgas:,.0f} Mgas",
                    line=dict(color=color, dash="dash", width=max(width - 0.4, 0.7)),
                    opacity=0.55,
                    hovertemplate=(
                        f"{set_name} C{j} threshold "
                        f"A·T = {AT_mgas:,.0f} Mgas<extra></extra>"
                    ),
                    legendgroup=f"backlog_{s_idx}",
                    showlegend=True,
                ),
                row=panel_idx, col=1,
            )

        fig.update_yaxes(
            title_text="Mgas",
            range=[0, bl_panel_max * 1.10 if bl_panel_max > 0 else 1.0],
            row=panel_idx, col=1,
        )

    # ── Penultimate panel: per-resource ArbOS 60 prices p_k ────────────────
    pk_row = n_panels - 2
    pk_max = P_MIN_GWEI
    for k in PRICED_SYMBOLS:
        y = pk_hr[k]
        # Hours with zero gas for this resource → NaN; replace with floor so
        # the line stays visible at P_min instead of breaking.
        y = np.where(np.isnan(y), P_MIN_GWEI, y)
        pk_max = max(pk_max, float(np.nanmax(y)))
        fig.add_trace(
            go.Scatter(
                x=x_price60, y=y,
                mode="lines",
                name=f"p_{k} — {PRICED_SYMBOL_LABELS[k]}",
                line=dict(color=PRICED_SYMBOL_COLORS[k], width=1.6),
                hovertemplate=(
                    f"p_{k} ({PRICED_SYMBOL_LABELS[k]}): "
                    f"%{{y:.4f}} gwei<extra></extra>"
                ),
                legendgroup="pk",
                legendgrouptitle_text="ArbOS 60 per-resource price",
            ),
            row=pk_row, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=x_span, y=[P_MIN_GWEI, P_MIN_GWEI],
            mode="lines",
            name=f"p_min = {P_MIN_GWEI} gwei",
            line=dict(color="rgba(0,0,0,0.4)", dash="dot", width=1),
            hovertemplate=f"Floor: {P_MIN_GWEI} gwei<extra></extra>",
            legendgroup="pk",
            showlegend=False,
        ),
        row=pk_row, col=1,
    )
    fig.update_yaxes(
        title_text="gwei",
        range=[P_MIN_GWEI * 0.95, pk_max * 1.10],
        row=pk_row, col=1,
    )

    # ── Final panel: observed on-chain vs simulated ArbOS 51 / 60 ──────────
    price_row = n_panels - 1
    # Observed on-chain L2 effective gas price, plotted first so the
    # simulated lines render on top of it.
    fig.add_trace(
        go.Scatter(
            x=x_wide, y=preal_hr,
            mode="lines",
            name="Observed on-chain",
            line=dict(color="#d62728", width=2.0),
            hovertemplate="Observed: %{y:.4f} gwei<extra></extra>",
            legendgroup="price",
            legendgrouptitle_text="Gas price — hourly mean",
        ),
        row=price_row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_wide, y=p51_hr,
            mode="lines",
            name="ArbOS 51 price (sim)",
            line=dict(color="#555", width=1.4, dash="dash"),
            hovertemplate="ArbOS 51 sim: %{y:.4f} gwei<extra></extra>",
            legendgroup="price",
        ),
        row=price_row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_price60, y=p60_hr,
            mode="lines",
            name="ArbOS 60 price (sim)",
            line=dict(color="#1f77b4", width=1.6),
            hovertemplate="ArbOS 60 sim: %{y:.4f} gwei<extra></extra>",
            legendgroup="price",
        ),
        row=price_row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_span_wide, y=[P_MIN_GWEI, P_MIN_GWEI],
            mode="lines",
            name=f"p_min = {P_MIN_GWEI} gwei",
            line=dict(color="rgba(0,0,0,0.4)", dash="dot", width=1),
            hovertemplate=f"Floor: {P_MIN_GWEI} gwei<extra></extra>",
            legendgroup="price",
            showlegend=False,
        ),
        row=price_row, col=1,
    )
    price_max = max(
        float(np.nanmax(preal_hr)) if not np.all(np.isnan(preal_hr)) else P_MIN_GWEI,
        float(np.nanmax(p51_hr))   if not np.all(np.isnan(p51_hr))   else P_MIN_GWEI,
        float(np.nanmax(p60_hr))   if not np.all(np.isnan(p60_hr))   else P_MIN_GWEI,
    )
    fig.update_yaxes(
        title_text="gwei",
        range=[0, price_max * 1.10],
        row=price_row, col=1,
    )

    # ── Hourly revenue (ETH) — observed vs simulated 51 / 60 ────────────────
    revenue_row = n_panels
    fig.add_trace(
        go.Scatter(
            x=x_wide, y=rev_real_eth,
            mode="lines",
            name="Observed on-chain (ETH/hr)",
            line=dict(color="#d62728", width=2.0),
            hovertemplate="Observed: %{y:.4f} ETH/hr<extra></extra>",
            legendgroup="revenue",
            legendgrouptitle_text="Hourly L2 revenue (ETH)",
        ),
        row=revenue_row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_wide, y=rev_51_eth,
            mode="lines",
            name="ArbOS 51 sim (ETH/hr)",
            line=dict(color="#555", width=1.4, dash="dash"),
            hovertemplate="ArbOS 51 sim: %{y:.4f} ETH/hr<extra></extra>",
            legendgroup="revenue",
        ),
        row=revenue_row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_price60, y=rev_60_eth,
            mode="lines",
            name="ArbOS 60 sim (ETH/hr)",
            line=dict(color="#1f77b4", width=1.6),
            hovertemplate="ArbOS 60 sim: %{y:.4f} ETH/hr<extra></extra>",
            legendgroup="revenue",
        ),
        row=revenue_row, col=1,
    )
    rev_max = max(
        float(np.nanmax(rev_real_eth)) if not np.all(np.isnan(rev_real_eth)) else 0.0,
        float(np.nanmax(rev_51_eth))   if not np.all(np.isnan(rev_51_eth))   else 0.0,
        float(np.nanmax(rev_60_eth))   if not np.all(np.isnan(rev_60_eth))   else 0.0,
    )
    fig.update_yaxes(
        title_text="ETH / hr",
        range=[0, rev_max * 1.10] if rev_max > 0 else [0, 1],
        row=revenue_row, col=1,
    )

    # ── Layout ──────────────────────────────────────────────────────────────
    panel_height_px = 400
    total_height    = panel_height_px * n_panels + 140   # + title/margin slack

    fig.update_layout(
        title=dict(
            text=(
                "<b>L2 Gas vs ArbOS 60 Constraint Targets — Hourly</b>"
                "<br>"
                "<sub>Real + ArbOS 51 sim: full per_block.parquet window. "
                "ArbOS 60 + per-resource backlog panels: bounded to the "
                "multigas extract coverage. ArbOS 51 honors the Dia "
                "activation (2026-01-08 17:00 UTC) — pre-Dia: T = 7 Mgas/s, "
                "A = 102 s, p<sub>min</sub> = 0.01 gwei; Dia: 6-rung ladder, "
                "p<sub>min</sub> = 0.02 gwei.</sub>"
            ),
            x=0.0, xanchor="left",
            font=dict(size=20, color="#111"),
        ),
        template="plotly_white",
        barmode="overlay",
        hovermode="x",               # per-trace tooltips aligned on the x-axis
        # Scope tooltip to the panel under the cursor (Plotly 5.13+) — without
        # this, hovermode="x" lumps every panel's traces at the cursor's x
        # into one tooltip. With it, the price panel shows only 3 gwei lines
        # and the revenue panel shows only 3 ETH/hr lines.
        hoversubplots="single",
        height=total_height + 1480,  # extra space reserved for the formula + table footer
        margin=dict(l=90, r=360, t=110, b=1540),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            groupclick="togglegroup",
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="rgba(0,0,0,0.20)",
            borderwidth=1,
            font=dict(size=12),
        ),
        font=dict(size=13, color="#222"),
    )

    # Draw the axis frame on all four sides of every subplot so each reads
    # as a bounded box.
    fig.update_xaxes(
        showline=True, linewidth=1.2, linecolor="rgba(0,0,0,0.55)",
        mirror=True,
        ticks="outside", tickcolor="rgba(0,0,0,0.55)",
        tickformat="%b %d",
        tickfont=dict(size=12),
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
    )
    fig.update_yaxes(
        showline=True, linewidth=1.2, linecolor="rgba(0,0,0,0.55)",
        mirror=True,
        ticks="outside", tickcolor="rgba(0,0,0,0.55)",
        tickfont=dict(size=12),
        title_font=dict(size=14),
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
    )

    fig.update_yaxes(title_text="Mgas/hr", row=1, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=n_panels, col=1)

    # Bump the default subplot-title font size.
    for a in fig.layout.annotations:
        a.font = dict(size=17, color="#111")

    # ── Formula footer ──────────────────────────────────────────────────────
    # Bordered panel at the bottom showing both gas-price equations.
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        xanchor="center", yanchor="top",
        showarrow=False, align="left",
        font=dict(size=16, color="#111"),
        bgcolor="rgba(245,245,250,1)",
        bordercolor="rgba(0,0,0,0.55)",
        borderwidth=1.2,
        borderpad=16,
        text=(
            r"$\begin{array}{l}"
            + r"\textbf{Notation (applies to both ArbOS 51 and ArbOS 60):} \\[2pt]"
            r"\quad{}\bullet\;\;i\in\{1,2,3,4\}=\text{set index (ArbOS 60 only; ArbOS 51 has one set, index suppressed)};\;\;"
            r"j\in\{0,..,5\}=\text{constraint index within a set};\;\;"
            r"k=\text{resource index} \\[2pt]"
            r"\quad{}\bullet\;\;t=\text{block timestamp in UTC seconds (one value per block)};\;\;"
            r"\Delta t = t_N - t_{N-1}\;\text{(integer seconds — the drain step)} \\[2pt]"
            r"\quad{}\bullet\;\;T_{i,j}=\text{target throughput / drain rate for constraint }(i,j)\;(\text{Mgas/s});\;\;"
            r"A_{i,j}=\text{adjustment window }(\text{s}) \\[2pt]"
            r"\quad{}\bullet\;\;B_{i,j}(t)=\text{backlog at block-time }t\text{ (accumulated gas above target, in gas units)} \\[2pt]"
            r"\quad{}\bullet\;\;a_{i,k}=\text{weight of resource }k\text{ in set }i\text{'s weighted inequality} \\[2pt]"
            r"\quad{}\bullet\;\;\text{Floor }p_{\min}=0.02\text{ gwei (every resource)} \\[2pt]"
            r"\quad{}\bullet\;\;\text{Resources: }"
            r"g_c=\text{Computation},\;"
            r"g_{sw}=\text{Storage Write},\;"
            r"g_{sr}=\text{Storage Read},\;"
            r"g_{sg}=\text{Storage Growth},\;"
            r"g_{hg}=\text{History Growth},\;"
            r"g_{l2}=\text{L2 Calldata} \\[12pt]"
            r"\textbf{ArbOS 51 --- single-dim, 6-constraint ladder (on-chain Dia):}\\[4pt]"
            r"\quad{}\text{backlog: }"
            r"B_j(t{+}1) = \max\!\big(0,\,B_j(t) + g_{\text{total}}(t) - T_j\,\Delta t\big)"
            r"\\[4pt]"
            r"\quad{}\text{price: }"
            r"p_{tx} = p_{\min}\,\exp\!\Big(\sum_{j=0}^{5}\tfrac{B_j}{A_j\,T_j}\Big)"
            r"\quad{}\text{(same for all txs in a block)} \\[8pt]"
            r"\quad{}"
            r"\begin{array}{|c|r|r|}"
            r"\hline{}"
            r"j & T_j\ (\text{Mgas/s}) & A_j\ (\text{s}) \\ \hline{}"
            r"0 & 10 & 86{,}400 \\ \hline"
            r"1 & 14 & 13{,}485 \\ \hline"
            r"2 & 20 & 2{,}105 \\ \hline"
            r"3 & 29 & 329 \\ \hline"
            r"4 & 41 & 52 \\ \hline"
            r"5 & 60 & 9 \\ \hline"
            r"\end{array} \\[12pt]"
            r"\textbf{ArbOS 60 --- per-resource pricing (4 sets, 19 constraints):}\\[4pt]"
            r"\quad{}\text{(0) per-(set, constraint) backlog: }"
            r"B_{i,j}(t{+}1) = \max\!\big(0,\,B_{i,j}(t) + \sum_k a_{i,k}\,g_k(t) - T_{i,j}\,\Delta t\big)"
            r"\\[6pt]"
            r"\quad{}\text{(1) per-resource price:}\quad{}p_k = p_{\min}\,\exp\!"
            r"\Big(\max_i\big\{\,a_{i,k}\sum_j \tfrac{B_{i,j}}{A_{i,j}\,T_{i,j}}\big\}\Big) \\[6pt]"
            r"\quad{}\text{(2) per-tx price (unnormalized inner product):}\quad{}"
            r"p_{tx}=\sum_k g_{tx,k}\,p_k \\[6pt]"
            r"\quad{}\text{(3) hourly avg (gas-weighted):}\quad{}"
            r"\bar p_{\text{hr}} = "
            r"\dfrac{\sum_{tx\in\text{hr}} p_{tx}}{\sum_{tx\in\text{hr}} G_{tx}},\quad{}"
            r"G_{tx}=\sum_k g_{tx,k} \\[8pt]"
            + _arbos60_inequalities_latex() + r" \\[10pt]"
            r"\quad{}" + _arbos60_tables_latex() +
            r"\end{array}$"
        ),
    )

    return fig


def main():
    # All inputs are local: per-block fees (from sql/arbitrum_revenue_per_block.sql,
    # cached by scripts/fetch_data.py) + per-tx multigas (Tyler extracts
    # converted to parquet by scripts/convert_tyler_extracts.py).
    blocks_pq = _DATA_ROOT / "onchain_blocks_transactions" / "per_block.parquet"

    print(f"Loading blocks: {blocks_pq}")
    print(f"Loading per-block resources (cached): {PER_BLOCK_RES_PQ}")
    per_block_res = build_per_block_resources()

    # Wide window: every block in per_block.parquet from _DEFAULT_START on —
    # used to extend the Real + ArbOS 51 lines past the multigas cutoff.
    cutoff = datetime.strptime(_DEFAULT_START, "%Y-%m-%d").date()
    blocks_wide = (
        load_per_block(str(blocks_pq))
        .filter(pl.col("block_date") >= cutoff)
        .with_columns([
            pl.col("block_time").dt.truncate("1h").alias("hour"),
            pl.col("block_date").cast(pl.Utf8).alias("day_str"),
        ])
    )
    # Narrow window: inner-joined with multigas → only blocks where the
    # per-resource breakdown exists, so ArbOS 60 + per-resource backlog
    # panels stay correct.
    blocks = blocks_wide.join(
        per_block_res.rename({"block": "block_number"}),
        on="block_number", how="inner",
    )
    print(f"  blocks_wide (Real + ArbOS 51): {blocks_wide.height:,}")
    print(f"  blocks (with multigas, ArbOS 60 + backlogs): {blocks.height:,}")

    print("Building chart...")
    fig = build_fig(blocks, blocks_wide)

    out_html = _HERE.parent / "figures" / "historical_sim_oct_today.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Interactive HTML version — plotly.js + MathJax pulled from CDN so LaTeX
    # renders and the chart stays interactive in any browser.
    fig.write_html(
        str(out_html),
        include_plotlyjs="cdn",
        include_mathjax="cdn",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    print(f"Saved {out_html}")


if __name__ == "__main__":
    main()
