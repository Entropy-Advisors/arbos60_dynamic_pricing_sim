-- Arbitrum Revenue — Per-Block (templated date window) [ClickHouse]
--
-- Date window driven by {{start_date}} / {{end_date}} placeholders that
-- scripts/fetch_data.py substitutes at run time.
--
-- p_min_gwei is time-varying across the analysis window:
--   ArbOS 20 "Atlas"  (Mar 18 2024 → Jan 8 2026 17:00 UTC):  0.01 gwei
--   ArbOS 51 "Dia"    (Jan 8 2026 17:00 UTC →):              0.02 gwei
--
-- Source table: raw_arbitrum.v_receipts
--   (NOT v_transactions — effective_gas_price is unpopulated there;
--    v_receipts.effective_gas_price is UInt64 and correctly filled)
-- Output: data/onchain_blocks_transactions/per_block.parquet

WITH block_fees AS (
    SELECT
        block_number,
        block_date,
        MIN(block_time)                                                         AS block_time_min,

        COUNT(*)                                                                AS tx_count,

        -- ── Fee decomposition (ETH) ───────────────────────────────────────────
        -- p_min flips at the ArbOS 51 Dia activation (Jan 8 2026 17:00 UTC).

        -- l2_base = P_min × g_L2
        SUM(
            (CASE WHEN block_time < toDateTime('2026-01-08 17:00:00')
                  THEN 0.01e9 ELSE 0.02e9 END)
            * toFloat64(gas_used - gas_used_for_l1)
        ) / 1e18                                                                AS l2_base,

        -- l2_surplus = (P_eff - P_min) × g_L2
        SUM(
            (toFloat64(effective_gas_price) -
             (CASE WHEN block_time < toDateTime('2026-01-08 17:00:00')
                   THEN 0.01e9 ELSE 0.02e9 END))
            * toFloat64(gas_used - gas_used_for_l1)
        ) / 1e18                                                                AS l2_surplus,

        -- l1_fees = P_eff × g_L1
        SUM(
            toFloat64(effective_gas_price)
            * toFloat64(gas_used_for_l1)
        ) / 1e18                                                                AS l1_fees,

        -- total_fees = P_eff × g_total
        SUM(
            toFloat64(effective_gas_price)
            * toFloat64(gas_used)
        ) / 1e18                                                                AS total_fees,

        -- ── Gas volumes ───────────────────────────────────────────────────────
        SUM(toFloat64(gas_used - gas_used_for_l1))                              AS total_l2_gas,
        SUM(toFloat64(gas_used_for_l1))                                         AS total_l1_gas,

        -- ── L2-gas-weighted average effective price ───────────────────────────
        SUM(
            toFloat64(effective_gas_price)
            * toFloat64(gas_used - gas_used_for_l1)
        ) / nullIf(SUM(toFloat64(gas_used - gas_used_for_l1)), 0)              AS avg_eff_price

    FROM raw_arbitrum.v_receipts
    WHERE
        block_date >= toDate('{{start_date}}')
        AND block_date <  toDate('{{end_date}}')
    GROUP BY
        block_number,
        block_date
)

SELECT
    block_number,
    block_date,
    block_time_min                                                             AS block_time,
    tx_count,

    -- Fee decomposition (ETH)
    l2_base,
    l2_surplus,
    l1_fees,
    total_fees,
    l2_base + l2_surplus + l1_fees - total_fees                                AS fee_check,

    -- Gas volumes
    total_l2_gas,
    total_l1_gas,

    -- Pricing
    avg_eff_price / 1e9                                                        AS avg_eff_price_gwei,
    CASE WHEN block_time_min < toDateTime('2026-01-08 17:00:00')
         THEN 0.01 ELSE 0.02 END                                               AS p_min_gwei,

    -- Backlog pressure proxy: (P_eff / P_min) − 1
    -- 0 = at floor, >0 = congested
    avg_eff_price / (CASE WHEN block_time_min < toDateTime('2026-01-08 17:00:00')
                          THEN 0.01e9 ELSE 0.02e9 END) - 1.0                   AS surplus_ratio

FROM block_fees
ORDER BY block_number
