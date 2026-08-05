-- daily_returns.sql: log returns per commodity
SELECT
    p.date,
    p.commodity_key,
    c.name,
    c.category,
    LN(p.price_native / LAG(p.price_native) OVER (
        PARTITION BY p.commodity_key ORDER BY p.date
    )) AS log_return
FROM fact_prices p
JOIN dim_commodity c ON p.commodity_key = c.commodity_key
WHERE p.date >= '{start_date}'
ORDER BY p.date, c.category
