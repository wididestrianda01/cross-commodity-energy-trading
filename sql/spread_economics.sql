-- spread_economics.sql: clean spark and dark spreads with fuel-switching signal
WITH daily AS (
    SELECT date,
        MAX(CASE WHEN commodity_key = 'DE_POWER' THEN price_eur_mwh END) AS power,
        MAX(CASE WHEN commodity_key = 'TTF' THEN price_eur_mwh END) AS gas,
        MAX(CASE WHEN commodity_key = 'API2' THEN price_eur_mwh END) AS coal,
        MAX(CASE WHEN commodity_key = 'EUA' THEN price_native END) AS carbon
    FROM fact_prices
    WHERE commodity_key IN ('DE_POWER', 'TTF', 'API2', 'EUA')
    GROUP BY date
)
SELECT date, power, gas, coal, carbon,
    power - (gas / {efficiency_gas}) - (carbon * {ef_gas}) AS clean_spark_spread,
    power - (coal / {efficiency_coal}) - (carbon * {ef_coal}) AS clean_dark_spread,
    (power - (gas / {efficiency_gas}) - (carbon * {ef_gas}))
    - (power - (coal / {efficiency_coal}) - (carbon * {ef_coal})) AS fuel_switch_signal
FROM daily
WHERE power IS NOT NULL AND gas IS NOT NULL AND coal IS NOT NULL AND carbon IS NOT NULL
