-- Arbitrum Revenue — Per-Block (Jan 9–31 2026) [ClickHouse]
--
-- ClickHouse port of arbitrum_revenue_per_block.sql.
-- Key dialect differences from Trino/DuneSQL:
--   TRY_CAST(x AS DOUBLE)  →  toFloat64(x)
--   NULLIF(x, 0)           →  nullIf(x, 0)
--   DATE 'yyyy-mm-dd'      →  toDate('yyyy-mm-dd')
--   GROUP BY 1, 2          →  GROUP BY block_number, block_date  (positional ok too)
--
-- Source table: raw_arbitrum.v_receipts
--   (NOT v_transactions — effective_gas_price is unpopulated there;
--    v_receipts.effective_gas_price is UInt64 and correctly filled)
-- Output: data/arbitrum_revenue_per_block.csv

WITH block_fees AS (
    SELECT
        block_number,
        block_date,
        MIN(block_time)                                                         AS block_time,

        COUNT(*)                                                                AS tx_count,

        -- ── Fee decomposition (ETH) ───────────────────────────────────────────

        -- l2_base = P_min × g_L2  (P_min = 0.02 gwei, constant in this range)
        SUM(
            0.02e9
            * toFloat64(gas_used - gas_used_for_l1)
        ) / 1e18                                                                AS l2_base,

        -- l2_surplus = (P_eff - P_min) × g_L2
        SUM(
            (toFloat64(effective_gas_price) - 0.02e9)
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
        block_date >= toDate('2026-01-09')
        AND block_date <  toDate('2026-02-01')
    GROUP BY
        block_number,
        block_date
)

SELECT
    block_number,
    block_date,
    block_time,
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
    0.02                                                                       AS p_min_gwei,

    -- Backlog pressure proxy: (P_eff / P_min) − 1
    -- 0 = at floor, >0 = congested
    (avg_eff_price / 0.02e9) - 1.0                                             AS surplus_ratio

FROM block_fees
ORDER BY block_number
