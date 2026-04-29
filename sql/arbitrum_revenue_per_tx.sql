-- Arbitrum revenue — per-transaction (ClickHouse)
--
-- Receipt-grain companion to `arbitrum_revenue_per_block.sql`. One row per tx,
-- with the same fee decomposition (L2 base / L2 surplus / L1 fees / total)
-- but unaggregated — the per-block CSV is just a GROUP BY of this query.
--
-- Used by check_calibration.py to validate the ArbOS 51 sim against real
-- on-chain effective_gas_price at the transaction level. The within-block
-- effective price is uniform under ArbOS 51 (no priority fees on Arbitrum),
-- so the per-tx variation here is in `gas_used` and `gas_used_for_l1` —
-- not in `effective_gas_price`.
--
-- Source table: raw_arbitrum.v_receipts
--   (NOT v_transactions — effective_gas_price is unpopulated there;
--    v_receipts.effective_gas_price is UInt64 and correctly filled)
--
-- Output: data/arbitrum_revenue_per_tx.parquet
--
-- Date window is parameterised by {{start_date}} / {{end_date}} — the
-- fetcher (scripts/fetch_data.py) substitutes these and runs day by day to
-- keep memory bounded (each day is ~2-5M rows).

SELECT
    block_number,
    tx_hash,
    block_time,
    block_date,

    -- Per-gas effective price the tx actually paid (gwei).
    toFloat64(effective_gas_price) / 1e9                              AS eff_price_gwei,

    -- Gas decomposition.
    toFloat64(gas_used)                                               AS gas_used,
    toFloat64(gas_used_for_l1)                                        AS gas_used_for_l1,
    toFloat64(gas_used - gas_used_for_l1)                             AS l2_gas,

    -- Fee decomposition (ETH).
    -- l2_base = P_min × g_L2  (P_min = 0.02 gwei, fixed in this range)
    0.02e9 * toFloat64(gas_used - gas_used_for_l1) / 1e18             AS l2_base,

    -- l2_surplus = (P_eff − P_min) × g_L2
    (toFloat64(effective_gas_price) - 0.02e9)
        * toFloat64(gas_used - gas_used_for_l1) / 1e18                AS l2_surplus,

    -- l1_fees = P_eff × g_L1
    toFloat64(effective_gas_price) * toFloat64(gas_used_for_l1) / 1e18 AS l1_fees,

    -- total_fees = P_eff × g_total
    toFloat64(effective_gas_price) * toFloat64(gas_used) / 1e18       AS total_fees

FROM raw_arbitrum.v_receipts
WHERE
    block_date >= toDate('{{start_date}}')
    AND block_date <  toDate('{{end_date}}')
ORDER BY block_number, tx_hash
