-- Arbitrum wallet spam / bot classification (ClickHouse)
--
-- Per-(day, wallet) aggregation, with two orthogonal spam signals layered
-- on top of each other.  Matches the Dune query the user shared: a wallet
-- can be flagged "high volume" on a busy MEV day and unflagged the next.
-- Revert-ratio is also evaluated per-day so a wallet that briefly
-- malfunctioned for a few hours gets caught.
--
--   1. tx_count percentile bucketing PER DAY — high-volume sender on that
--      day (Dune heuristic).  p10 / p001 cutoffs are recomputed each day
--      so weekend and Monday quantiles don't bleed into each other.
--
--   2. revert ratio PER DAY — fraction of the wallet's txs on that day
--      that reverted.  Requires a minimum same-day tx_count so a wallet
--      with one tx that reverted doesn't trip.
--
-- A row is `is_spam` if it crosses EITHER threshold on that day
-- (high-volume OR high-revert).  Output is grain (day, address) — the
-- clustering pipeline joins per-tx by (block_date, tx_sender).
--
-- Source tables on this cluster:
--   raw_arbitrum.v_transactions      — SUCCESSFUL txs only (success always
--     true).  Snake-case columns: block_number, block_date, "from",
--     gas_used, effective_gas_price.
--   raw_arbitrum.transactions_failed — REVERTED txs only.  SCREAMING_SNAKE
--     columns: BLOCK_NUMBER, DATETIME, FROM_ADDRESS, GAS_USED,
--     EFFECTIVE_GAS_PRICE.  No block_date column — derive via
--     toDate(DATETIME).
--
-- Reverts are NOT in v_transactions.success — they're in the failed table.
-- We FULL OUTER JOIN the two on (day, address) to get the complete picture.
--
-- Date + block-number window via {{start_date}} / {{end_date}} /
-- {{block_min}} / {{block_max}}.  block_date scopes the daily partition;
-- block_number narrows further to the exact range covered by the multigas
-- data (so we don't classify wallets that never show up downstream).
--
-- Per-day spam thresholds: {{revert_min_txs}} / {{revert_ratio_threshold}}.
-- Wallet-level rollup uses {{spam_day_frac}} — a wallet is `is_spam` if it
-- was flagged on at least that fraction of its active days (default 0.5).
--
-- Output: data/wallet_spam_classification.parquet
--   one row per address active in the window, with:
--     tx_count          — total txs in window
--     revert_count      — total reverted txs
--     gas_used_eth      — total fees paid
--     n_days_active     — distinct days the wallet sent any tx
--     n_days_high_vol   — days the wallet was in that day's top-0.1% by tx_count
--     n_days_high_rev   — days the wallet's revert ratio cleared the threshold
--     n_days_spam       — days flagged by either signal
--     frac_spam_days    — n_days_spam / n_days_active
--     is_spam           — frac_spam_days >= {{spam_day_frac}}
--     first_seen / last_seen

WITH

-- ── 1a. Successful txs per (day, wallet) ───────────────────────────────────
success_per_day_wallet AS (
    SELECT
        block_date                                                   AS day,
        "from"                                                       AS address,
        COUNT(*)                                                     AS success_count,
        SUM(toFloat64(gas_used) * toFloat64(effective_gas_price))
            / 1e18                                                   AS gas_used_eth_succ
    FROM raw_arbitrum.v_transactions
    WHERE
        block_date >= toDate('{{start_date}}')
        AND block_date <  toDate('{{end_date}}')
        AND block_number >= toUInt64('{{block_min}}')
        AND block_number <= toUInt64('{{block_max}}')
    GROUP BY day, "from"
),

-- ── 1b. Reverted txs per (day, wallet) ─────────────────────────────────────
failed_per_day_wallet AS (
    SELECT
        toDate(DATETIME)                                             AS day,
        FROM_ADDRESS                                                 AS address,
        COUNT(*)                                                     AS revert_count,
        SUM(toFloat64(GAS_USED) * toFloat64(EFFECTIVE_GAS_PRICE))
            / 1e18                                                   AS gas_used_eth_fail
    FROM raw_arbitrum.transactions_failed
    WHERE
        toDate(DATETIME) >= toDate('{{start_date}}')
        AND toDate(DATETIME) <  toDate('{{end_date}}')
        AND BLOCK_NUMBER >= toUInt64('{{block_min}}')
        AND BLOCK_NUMBER <= toUInt64('{{block_max}}')
    GROUP BY day, address
),

-- ── 1c. Combined per-(day, wallet) view ────────────────────────────────────
-- FULL JOIN: a wallet active on a given day may appear in success-only,
-- failed-only, or both.  COALESCE fills missing sides with 0.
per_day_wallet AS (
    SELECT
        if(s.day IS NULL, f.day, s.day)                              AS day,
        if(s.address = '', f.address, s.address)                     AS address,
        coalesce(s.success_count, 0)
            + coalesce(f.revert_count, 0)                            AS tx_count,
        coalesce(f.revert_count, 0)                                  AS revert_count,
        coalesce(s.gas_used_eth_succ, 0.0)
            + coalesce(f.gas_used_eth_fail, 0.0)                     AS gas_used_eth
    FROM success_per_day_wallet s
    FULL OUTER JOIN failed_per_day_wallet f
        ON s.day = f.day
       AND s.address = f.address
),

-- ── 2. Per-day tx_count percentiles ────────────────────────────────────────
-- Same percentile schedule as the Dune query (p10 / p001 / above), computed
-- per day so a quiet Sunday's spam cutoff isn't dragged up by a busy
-- Wednesday.
daily_buckets AS (
    SELECT
        day,
        quantileExact(0.90)(tx_count)  AS p10,
        quantileExact(0.99)(tx_count)  AS p01,
        quantileExact(0.999)(tx_count) AS p001
    FROM per_day_wallet
    GROUP BY day
)

-- ── 3. Per-(day, wallet) flags using each day's own cutoffs ────────────────
,
flagged AS (
    SELECT
        pw.day                                                       AS day,
        pw.address                                                   AS address,
        pw.tx_count                                                  AS tx_count,
        pw.revert_count                                              AS revert_count,
        pw.gas_used_eth                                              AS gas_used_eth,
        (pw.tx_count > b.p001)                                       AS is_high_volume,
        (pw.tx_count >= toUInt64('{{revert_min_txs}}')
            AND (pw.revert_count / toFloat64(pw.tx_count))
                >= toFloat64('{{revert_ratio_threshold}}'))          AS is_high_revert,
        (pw.tx_count > b.p001
            OR
            (pw.tx_count >= toUInt64('{{revert_min_txs}}')
                AND (pw.revert_count / toFloat64(pw.tx_count))
                    >= toFloat64('{{revert_ratio_threshold}}')))     AS is_spam_day
    FROM per_day_wallet pw
    INNER JOIN daily_buckets b
           ON pw.day = b.day
)

-- ── 4. Wallet-level rollup (intermediate) ──────────────────────────────────
-- Aggregates the day-level flags up to one row per wallet.  Aliases are
-- distinct from the inner `flagged` CTE column names to avoid CH's
-- "aggregate inside aggregate" error on collision.
,
rollup AS (
    SELECT
        address,
        SUM(tx_count)                                                AS sum_tx,
        SUM(revert_count)                                            AS sum_rev,
        SUM(gas_used_eth)                                            AS sum_gas_eth,
        COUNT(*)                                                     AS days_active,
        SUM(toUInt32(is_high_volume))                                AS days_hv,
        SUM(toUInt32(is_high_revert))                                AS days_hr,
        SUM(toUInt32(is_spam_day))                                   AS days_spam,
        MIN(day)                                                     AS first_seen,
        MAX(day)                                                     AS last_seen
    FROM flagged
    GROUP BY address
)

-- ── 5. Final per-wallet output ─────────────────────────────────────────────
-- `is_spam` requires the wallet to be flagged on at least `spam_day_frac`
-- of its active days; `is_spam_ever` is the looser companion (any day).
SELECT
    address,
    sum_tx                                                           AS tx_count,
    sum_rev                                                          AS revert_count,
    if(sum_tx > 0, sum_rev / toFloat64(sum_tx), 0.0)                 AS revert_ratio,
    sum_gas_eth                                                      AS gas_used_eth,
    days_active                                                      AS n_days_active,
    days_hv                                                          AS n_days_high_vol,
    days_hr                                                          AS n_days_high_rev,
    days_spam                                                        AS n_days_spam,
    if(days_active > 0,
       days_spam / toFloat64(days_active),
       0.0)                                                          AS frac_spam_days,
    first_seen,
    last_seen,
    (days_active > 0
        AND days_spam / toFloat64(days_active)
            >= toFloat64('{{spam_day_frac}}'))                       AS is_spam,
    (days_spam > 0)                                                  AS is_spam_ever
FROM rollup
ORDER BY days_spam DESC, sum_tx DESC
