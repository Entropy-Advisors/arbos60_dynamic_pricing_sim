"""
Multi-panel chart comparing L2 gas consumption to ArbOS 60 constraint
targets and simulating the resulting base fee against the ArbOS 51 baseline.

Panels:
    1       stacked bars of hourly L2 gas per priced ResourceKind
            (Computation = computation + wasmComputation; L1Calldata excluded)
    2–5     one per constraint set — hourly weighted inflow vs ladder targets
    6       hourly mean gas price: observed on-chain vs simulated ArbOS 51
            vs simulated ArbOS 60 (log-y)

`storageAccess` is not split into read/write in the source data, so we apply
a fixed R:W ratio (tuneable via R_OVER_W_RATIO) when computing per-set inflows.

Run:
    python report/scripts/plot_constraints_overlay.py

Outputs:
    report/figures/fig_rk_gas_vs_constraints.png   (static)
    report/figures/fig_rk_gas_vs_constraints.html  (interactive)
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# Import the standalone equation modules.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import arbos51 as a51   # noqa: E402
import arbos60 as a60   # noqa: E402


# ── Visual style (was figures.RESOURCE_COLORS) ──────────────────────────────
RESOURCE_COLORS = {
    "Computation":     "#1f77b4",   # blue
    "StorageAccess":   "#2ca02c",   # green
    "StorageGrowth":   "#d62728",   # red
    "HistoryGrowth":   "#ff7f0e",   # orange
    "L2Calldata":      "#9467bd",   # purple
    "WasmComputation": "#8c564b",   # brown
    "L1Calldata":      "#e377c2",   # pink
}


def _hours_x(df: pl.DataFrame, col: str = "hour") -> list:
    """Polars Datetime/string column → Python datetime list for Plotly."""
    s = df[col]
    if s.dtype == pl.Utf8:
        return pd.to_datetime(s.to_list()).tolist()
    return s.to_list()


# ── Data loading (was report/data.py — inlined here) ────────────────────────
_WEIGHT_COLS = [
    "w_Computation",
    "w_StorageAccess",
    "w_StorageGrowth",
    "w_HistoryGrowth",
    "w_L2Calldata",
    "w_WasmComputation",
    "w_L1Calldata",
]
_DEFAULT_START = "2026-01-18"   # first day of full Multigas coverage


def _strip_tz(df: pl.DataFrame, col: str) -> pl.DataFrame:
    dtype = df[col].dtype
    if dtype.base_type() == pl.Datetime and getattr(dtype, "time_zone", None):
        df = df.with_columns(pl.col(col).dt.replace_time_zone(None))
    return df


def load_per_block(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    df = _strip_tz(df, "block_time")
    if df["block_date"].dtype.base_type() == pl.Datetime:
        df = df.with_columns(pl.col("block_date").cast(pl.Date))
    return df


def load_per_block_weights(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    return df.with_columns(pl.col("block_number").cast(pl.Int64))


def load_hourly_weights(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    if df["hour"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("hour").str.to_datetime(format=None, strict=False)
        )
    df = _strip_tz(df, "hour")
    return df.with_columns(pl.col("hour").dt.truncate("1h"))


def merge_blocks_with_weights(
    blocks: pl.DataFrame,
    weights: pl.DataFrame,
    start_date: str = _DEFAULT_START,
    per_block_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if start_date:
        cutoff = datetime.strptime(start_date, "%Y-%m-%d").date()
        blocks = blocks.filter(pl.col("block_date") >= cutoff)
    blocks = blocks.with_columns([
        pl.col("block_time").dt.truncate("1h").alias("hour"),
        pl.col("block_date").cast(pl.Utf8).alias("day_str"),
    ])
    if per_block_weights is not None:
        available_w = [c for c in _WEIGHT_COLS if c in per_block_weights.columns]
        merged = blocks.join(
            per_block_weights.select(["block_number"] + available_w),
            on="block_number", how="left",
        )
    else:
        available_w = [c for c in _WEIGHT_COLS if c in weights.columns]
        merged = blocks.join(
            weights.select(["hour"] + available_w), on="hour", how="left",
        )
    merged = merged.with_columns([pl.col(c).fill_null(0.0) for c in available_w])
    for c in _WEIGHT_COLS:
        if c not in merged.columns:
            merged = merged.with_columns(pl.lit(0.0).alias(c))
    return merged

# ── Re-export from the standalone equation modules ──────────────────────────
# All equation logic lives in scripts/arbos51.py and scripts/arbos60.py.
# These thin wrappers just adapt the polars-DataFrame call signature used
# throughout build_fig() to the numpy-array signatures of the modules.

P_MIN_GWEI          = a51.P_MIN_GWEI
EXPONENT_CAP        = a51.EXPONENT_CAP
BACKLOG_TOLERANCE_S = a51.BACKLOG_TOLERANCE_S
R_OVER_W_RATIO      = a60.R_OVER_W_RATIO
SA_WRITE_SHARE      = a60.SA_WRITE_SHARE
SA_READ_SHARE       = a60.SA_READ_SHARE
ARBOS51_LADDER      = a51.ARBOS51_LADDER
SET_WEIGHTS         = a60.SET_WEIGHTS
SET_LADDERS         = a60.SET_LADDERS
PRICED_SYMBOLS      = a60.PRICED_SYMBOLS
PRICED_SYMBOL_LABELS = a60.PRICED_SYMBOL_LABELS

PRICED_KINDS = [
    "Computation",      # = computation + wasmComputation
    "StorageAccess",    # combined Read + Write (not split in the source data)
    "StorageGrowth",
    "HistoryGrowth",
    "L2Calldata",
]
PRICED_SYMBOL_COLORS = {
    "c":  "#1f77b4",   # Computation — blue
    "sw": "#2ca02c",   # Storage Write — StorageAccess green
    "sr": "#98df8a",   # Storage Read — light green companion
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


def _dt_seconds_per_block(blocks: pl.DataFrame) -> np.ndarray:
    """Per-block integer-second deltas, derived from `block_time`."""
    return a51.dt_seconds_per_block(blocks["block_time"].to_numpy())


def _per_block_resource_gas(blocks: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per-block resource gas split across the 6 priced symbols."""
    return a60.per_block_resource_gas(blocks)


def price_arbos51_per_block(
    blocks: pl.DataFrame, p_min_gwei: float = P_MIN_GWEI,
) -> np.ndarray:
    return a51.price_per_block(
        blocks["total_l2_gas"].to_numpy(),
        _dt_seconds_per_block(blocks),
        p_min_gwei=p_min_gwei,
    )


def price_arbos60_per_resource(
    blocks: pl.DataFrame, p_min_gwei: float = P_MIN_GWEI,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    parts  = a60.per_block_resource_gas(blocks)
    dt_s   = _dt_seconds_per_block(blocks)
    prices, E_per_set = a60.price_per_resource(parts, dt_s, p_min_gwei)
    return prices, parts, E_per_set


def arbos60_block_fee_and_gas(
    blocks: pl.DataFrame, p_min_gwei: float = P_MIN_GWEI,
) -> tuple[np.ndarray, np.ndarray]:
    parts = a60.per_block_resource_gas(blocks)
    dt_s  = _dt_seconds_per_block(blocks)
    return a60.block_fee_and_gas(parts, dt_s, p_min_gwei)


def compute_arbos51_backlogs(
    blocks: pl.DataFrame,
) -> dict[tuple[float, float], np.ndarray]:
    return a51.backlogs_all_constraints(
        blocks["total_l2_gas"].to_numpy().astype(np.float64),
        _dt_seconds_per_block(blocks),
    )


def compute_backlogs(
    blocks: pl.DataFrame,
) -> dict[str, dict[tuple[float, float], np.ndarray]]:
    parts = a60.per_block_resource_gas(blocks)
    dt_s  = _dt_seconds_per_block(blocks)
    return a60.backlogs_all_constraints(parts, dt_s)



def hourly_gas_per_kind(blocks: pl.DataFrame) -> pl.DataFrame:
    """Hourly raw gas (Mgas) per priced ResourceKind, keyed by `hour`."""
    df = blocks.with_columns([
        ((pl.col("w_Computation") + pl.col("w_WasmComputation"))
         * pl.col("total_l2_gas")).alias("gas_Computation"),
        (pl.col("w_StorageAccess") * pl.col("total_l2_gas")).alias("gas_StorageAccess"),
        (pl.col("w_StorageGrowth") * pl.col("total_l2_gas")).alias("gas_StorageGrowth"),
        (pl.col("w_HistoryGrowth") * pl.col("total_l2_gas")).alias("gas_HistoryGrowth"),
        (pl.col("w_L2Calldata")    * pl.col("total_l2_gas")).alias("gas_L2Calldata"),
    ])
    return (
        df.group_by("hour")
          .agg([pl.col(f"gas_{k}").sum() for k in PRICED_KINDS])
          .sort("hour")
    )


def weighted_inflow_mgas_hr(
    rk_hr: pl.DataFrame, weights: dict[str, float],
) -> np.ndarray:
    """
    Compute hourly weighted inflow Σ_k w_k · g_k in Mgas/hour.

    Symbol map:
        c  → gas_Computation
        sw → gas_StorageAccess × write_share
        sr → gas_StorageAccess × read_share
        sg → gas_StorageGrowth
        hg → gas_HistoryGrowth
        l2 → gas_L2Calldata
    """
    sa = rk_hr["gas_StorageAccess"].to_numpy()
    parts = {
        "c":  rk_hr["gas_Computation"].to_numpy(),
        "sw": sa * SA_WRITE_SHARE,
        "sr": sa * SA_READ_SHARE,
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


def build_fig(blocks: pl.DataFrame) -> go.Figure:
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
    print("  simulating ArbOS 51 prices...")
    p51_pb = price_arbos51_per_block(blocks)
    print("  simulating ArbOS 60 prices (per-resource max-over-sets)...")
    p60_prices, p60_parts, _ = price_arbos60_per_resource(blocks)
    p60_fee_pb, p60_gas_pb = arbos60_block_fee_and_gas(blocks)
    print("  computing per-(set, constraint) backlogs...")
    backlogs_pb = compute_backlogs(blocks)
    print("  computing ArbOS 51 backlogs...")
    arbos51_backlogs_pb = compute_arbos51_backlogs(blocks)

    total_gas_pb = blocks["total_l2_gas"].to_numpy().astype(np.float64)
    # Empty blocks have null avg_eff_price; null × 0 propagates as null and
    # poisons the hourly sum. Fill with 0 so they contribute nothing.
    real_pb      = (
        blocks["avg_eff_price_gwei"].fill_null(0.0).to_numpy().astype(np.float64)
    )
    real_pb      = np.nan_to_num(real_pb, nan=0.0)

    # Build a single dataframe carrying all per-block (fee, gas) columns
    # we need for hourly gas-weighted means: real, ArbOS 51, ArbOS 60 total,
    # and ArbOS 60 per-resource k.
    cols = [
        pl.Series("real_fee",  real_pb * total_gas_pb),
        pl.Series("p51_fee",   p51_pb  * total_gas_pb),
        pl.Series("total_gas", total_gas_pb),
        pl.Series("p60_fee",   p60_fee_pb),
        pl.Series("p60_gas",   p60_gas_pb),
    ]
    for k in PRICED_SYMBOLS:
        cols.append(pl.Series(f"fee_{k}", p60_prices[k] * p60_parts[k]))
        cols.append(pl.Series(f"gas_{k}", p60_parts[k]))

    agg_exprs = [
        pl.col("real_fee").sum().alias("real_fee_sum"),
        pl.col("p51_fee").sum().alias("p51_fee_sum"),
        pl.col("total_gas").sum().alias("total_gas_sum"),
        pl.col("p60_fee").sum().alias("p60_fee_sum"),
        pl.col("p60_gas").sum().alias("p60_gas_sum"),
    ]
    for k in PRICED_SYMBOLS:
        agg_exprs.append(pl.col(f"fee_{k}").sum().alias(f"fee_{k}_sum"))
        agg_exprs.append(pl.col(f"gas_{k}").sum().alias(f"gas_{k}_sum"))

    derive_exprs = [
        (pl.col("real_fee_sum") / pl.col("total_gas_sum")).alias("p_real_gwei"),
        (pl.col("p51_fee_sum")  / pl.col("total_gas_sum")).alias("p51_gwei"),
        (pl.col("p60_fee_sum")  / pl.col("p60_gas_sum")).alias("p60_gwei"),
    ]
    for k in PRICED_SYMBOLS:
        derive_exprs.append(
            (pl.col(f"fee_{k}_sum") / pl.col(f"gas_{k}_sum")).alias(f"p_{k}_gwei")
        )

    prices_hr = (
        blocks.select(["hour"])
              .with_columns(cols)
              .group_by("hour")
              .agg(agg_exprs)
              .with_columns(derive_exprs)
              .sort("hour")
    )
    x_price  = _hours_x(prices_hr)
    p51_hr   = prices_hr["p51_gwei"].to_numpy()
    p60_hr   = prices_hr["p60_gwei"].to_numpy()
    preal_hr = prices_hr["p_real_gwei"].to_numpy()
    pk_hr    = {k: prices_hr[f"p_{k}_gwei"].to_numpy() for k in PRICED_SYMBOLS}

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
    n_panels = 1 + 1 + len(set_names) + 1 + len(set_names) + 1 + 1

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
    pk_row = n_panels - 1
    pk_max = P_MIN_GWEI
    for k in PRICED_SYMBOLS:
        y = pk_hr[k]
        # Hours with zero gas for this resource → NaN; replace with floor so
        # the line stays visible at P_min instead of breaking.
        y = np.where(np.isnan(y), P_MIN_GWEI, y)
        pk_max = max(pk_max, float(np.nanmax(y)))
        fig.add_trace(
            go.Scatter(
                x=x_price, y=y,
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
    price_row = n_panels
    # Observed on-chain L2 effective gas price, plotted first so the
    # simulated lines render on top of it.
    fig.add_trace(
        go.Scatter(
            x=x_price, y=preal_hr,
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
            x=x_price, y=p51_hr,
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
            x=x_price, y=p60_hr,
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
            x=x_span, y=[P_MIN_GWEI, P_MIN_GWEI],
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

    # ── Layout ──────────────────────────────────────────────────────────────
    panel_height_px = 400
    total_height    = panel_height_px * n_panels + 140   # + title/margin slack

    fig.update_layout(
        title=dict(
            text=(
                "<b>L2 Gas vs ArbOS 60 Constraint Targets — Hourly (Jan 2026)</b>"
                "<br>"
                f"<sub>p<sub>min</sub> = {P_MIN_GWEI} gwei (same floor for all resources). "
                f"Storage-access R:W split = {R_OVER_W_RATIO:.2f}:1.</sub>"
            ),
            x=0.0, xanchor="left",
            font=dict(size=20, color="#111"),
        ),
        template="plotly_white",
        barmode="overlay",
        hovermode="x",               # per-trace tooltips aligned on the x-axis
        height=total_height + 1320,  # extra space reserved for the formula + table footer
        margin=dict(l=90, r=360, t=110, b=1380),
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
            + r"\textbf{ArbOS 51 --- single-dim, 6-constraint ladder (on-chain Dia):}\\[4pt]"
            r"\quad{}\text{backlog (Lindley): }"
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
            r"\quad{}\text{(0) per-(set, constraint) backlog (Lindley): }"
            r"B_{i,j}(t{+}1) = \max\!\big(0,\,B_{i,j}(t) + \sum_k a_{i,k}\,g_k(t) - T_{i,j}\,\Delta t\big)"
            r"\\[6pt]"
            r"\quad{}\text{(1) per-resource price:}\quad{}p_k = p_{\min}\,\exp\!"
            r"\Big(\max_i\big\{\,a_{i,k}\sum_j \tfrac{B_{i,j}}{A_{i,j}\,T_{i,j}}\big\}\Big) \\[6pt]"
            r"\quad{}\text{(2) per-tx price (inner product):}\quad{}"
            r"p_{tx}=\langle\mathbf{w}_{tx},\,\mathbf{p}\rangle="
            r"\sum_k w_{tx,k}\,p_k,\quad{}"
            r"w_{tx,k}=\tfrac{g_{tx,k}}{\sum_k g_{tx,k}} \\[6pt]"
            r"\quad{}\text{(3) hourly avg (gas-weighted):}\quad{}"
            r"\bar p_{\text{hr}} = "
            r"\dfrac{\sum_{tx\in\text{hr}} G_{tx}\,p_{tx}}{\sum_{tx\in\text{hr}} G_{tx}},\quad{}"
            r"G_{tx}=\sum_k g_{tx,k} \\[6pt]"
            r"\quad{}\text{Set }i\text{ with constraints }j;\;a_{i,k}=\text{weight of }k\text{ in set }i \\[2pt]"
            r"\quad{}\text{Floor: }p_{\min}=0.02\text{ gwei (same for all resources)} \\[2pt]"
            r"\quad{}\text{Resources }k:\;"
            r"g_c=\text{Computation},\;"
            r"g_{sw}=\text{Storage Write},\;"
            r"g_{sr}=\text{Storage Read},\;"
            r"g_{sg}=\text{Storage Growth},\;"
            r"g_{hg}=\text{History Growth},\;"
            r"g_{l2}=\text{L2 Calldata} \\[8pt]"
            + _arbos60_inequalities_latex() + r" \\[10pt]"
            r"\quad{}" + _arbos60_tables_latex() +
            r"\end{array}$"
        ),
    )

    return fig


def main():
    # Source data still lives in the original repo (see README.md). Point the
    # paths there directly — the clean folder holds only code + HTML output.
    src_data = pathlib.Path(
        "/Users/mohammedbenseddik/Documents/Dev/EA/arbos60/data"
    )
    src_report_data = pathlib.Path(
        "/Users/mohammedbenseddik/Documents/Dev/EA/arbos60/report/data"
    )
    blocks_csv    = src_data / "arbitrum_revenue_per_block.csv"
    hourly_csv    = src_report_data / "hourly_multigas_weights_corrected.csv"
    per_block_csv = src_report_data / "per_block_multigas_weights.csv"

    print(f"Loading blocks: {blocks_csv}")
    per_block_weights = (
        load_per_block_weights(str(per_block_csv)) if per_block_csv.exists() else None
    )
    blocks = merge_blocks_with_weights(
        load_per_block(str(blocks_csv)),
        load_hourly_weights(str(hourly_csv)),
        per_block_weights=per_block_weights,
    )
    print(f"  {blocks.height:,} blocks")

    print("Building chart...")
    fig = build_fig(blocks)

    out_html = _HERE.parent / "figures" / "main.html"
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
