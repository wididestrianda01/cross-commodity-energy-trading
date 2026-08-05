-- rolling_correlation.sql: pairwise {window}-day rolling correlations
WITH returns AS (
    SELECT date, commodity_key,
        LN(price_native / LAG(price_native) OVER (
            PARTITION BY commodity_key ORDER BY date
        )) AS log_return
    FROM fact_prices
    WHERE date >= '{start_date}'
),
pairs AS (
    SELECT r1.date, r1.commodity_key AS c1, r2.commodity_key AS c2,
        r1.log_return AS r1, r2.log_return AS r2
    FROM returns r1
    JOIN returns r2 ON r1.date = r2.date AND r1.commodity_key < r2.commodity_key
)
SELECT date, c1, c2,
    (SUM(r1 * r2) OVER w - SUM(r1) OVER w * SUM(r2) OVER w / {window})
    / NULLIF(
        SQRT(SUM(r1 * r1) OVER w - SUM(r1) OVER w * SUM(r1) OVER w / {window})
        * SQRT(SUM(r2 * r2) OVER w - SUM(r2) OVER w * SUM(r2) OVER w / {window}),
        0
    ) AS rolling_corr
FROM pairs
WINDOW w AS (PARTITION BY c1, c2 ORDER BY date ROWS BETWEEN {window} PRECEDING AND CURRENT ROW)
