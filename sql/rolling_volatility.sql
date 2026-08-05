-- rolling_volatility.sql: {window}-day rolling annualized volatility
WITH returns AS (
    SELECT date, commodity_key,
        LN(price_native / LAG(price_native) OVER (
            PARTITION BY commodity_key ORDER BY date
        )) AS log_return
    FROM fact_prices
    WHERE date >= '{start_date}'
)
SELECT date, commodity_key,
    SQRT(SUM(log_return * log_return) OVER (
        PARTITION BY commodity_key ORDER BY date
        ROWS BETWEEN {window} PRECEDING AND CURRENT ROW
    ) / {window}) * SQRT(252) AS rolling_vol_annualized
FROM returns
WHERE log_return IS NOT NULL
