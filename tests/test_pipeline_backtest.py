"""Verify pipeline populates var_backtest table with backtest data."""

import duckdb
import numpy as np
import pandas as pd
from energy_cross_commodity.db import init_db, seed_commodities
from energy_cross_commodity.risk.var_engine import compute_rolling_var


def test_rolling_var_populates_backtest_table():
    """compute_rolling_var returns valid date/pnl/var rows for storage."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.DataFrame({
        "TTF": rng.standard_normal(n) * 0.03,
        "BRENT": rng.standard_normal(n) * 0.02,
        "EUA": rng.standard_normal(n) * 0.025,
    }, index=dates)
    positions = {"TTF": 8_000_000, "BRENT": 9_200_000, "EUA": 3_000_000}

    result = compute_rolling_var(returns, positions, window=100, n_simulations=500)

    assert len(result) > 0
    assert set(result.columns) == {"date", "var_95", "var_99", "realized_pnl"}
    assert result["var_95"].notna().all()
    assert result["realized_pnl"].notna().all()
    assert (result["var_95"] > 0).all()


def test_backtest_rows_insertable():
    """Backtest DataFrame can be inserted into DuckDB var_backtest table."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.DataFrame({
        "X": rng.standard_normal(n) * 0.02,
    }, index=dates)
    # Single-asset book: no dependence structure, so no copula.
    result = compute_rolling_var(
        returns, {"X": 1_000_000}, window=60, fit_fn=lambda _w: (None, None)
    )

    conn = duckdb.connect(":memory:")
    init_db(conn)
    seed_commodities(conn)
    insert_df = result[["date", "realized_pnl", "var_95"]].rename(
        columns={"realized_pnl": "pnl", "var_95": "var_estimate"}
    )
    conn.execute("INSERT INTO var_backtest SELECT * FROM insert_df")
    count = conn.execute("SELECT COUNT(*) FROM var_backtest").fetchone()[0]
    assert count == len(insert_df)
    conn.close()
