"""Verify backtest data extraction for Tab 3 backtesting chart."""

import duckdb
from energy_cross_commodity.db import init_db
from energy_cross_commodity.risk.var_engine import kupiec_test


def test_backtest_kupiec_on_sample_data():
    """Kupiec test computes correctly on known breach count."""
    result = kupiec_test(breaches=10, total=252, confidence=0.95)
    assert result["p_value"] > 0.01  # 10/252 ~ 4%, expected 5% — well within range
    assert result["breaches"] == 10
    assert result["total"] == 252


def test_backtest_empty_table_graceful():
    """Empty var_backtest returned as empty DataFrame — no crash."""
    conn = duckdb.connect(":memory:")
    init_db(conn)
    df = conn.execute("SELECT * FROM var_backtest").df()
    assert len(df) == 0
    conn.close()
