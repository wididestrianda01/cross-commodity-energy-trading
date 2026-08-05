-- var_breaches.sql: identify days where realized P&L exceeded VaR estimate
SELECT date,
    pnl,
    var_estimate,
    CASE WHEN pnl < -var_estimate THEN 1 ELSE 0 END AS is_breach
FROM var_backtest
WHERE date >= '{start_date}'
