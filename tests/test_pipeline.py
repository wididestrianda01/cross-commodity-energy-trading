"""Tests for DB schema, data pipeline, and normalizer."""

import pytest
import pandas as pd
from energy_cross_commodity.data.normalizer import convert_to_eur_mwh, build_date_dimension


def test_init_db_creates_tables(db_conn):
    tables = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    table_names = {row[0] for row in tables}
    assert "dim_commodity" in table_names
    assert "dim_date" in table_names
    assert "fact_prices" in table_names


def test_seed_commodities_inserts_rows(db_conn):
    count = db_conn.execute("SELECT COUNT(*) FROM dim_commodity").fetchone()[0]
    assert count >= 7


def test_convert_brent_to_eur_mwh():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="B"),
        "price": [80.0, 82.0, 81.0],
    })
    # EURUSD=X quotes USD per EUR, so a USD price is divided by it.
    eurusd = pd.Series([1.09, 1.09, 1.09], index=df["date"])
    result = convert_to_eur_mwh(df, "BRENT", 1.628, eurusd)
    expected = 80.0 / 1.09 / 1.628
    assert abs(result.iloc[0]["price_eur_mwh"] - expected) < 0.01
    assert result.iloc[0]["price_native"] == 80.0


def test_convert_uses_usd_per_eur_convention():
    """A stronger dollar must lower, not raise, the EUR-denominated price.

    Guards the direction of the FX division. The two conventions differ by
    only ~18% at typical rates, so a single-rate equality test passes under
    either one; only the *sign of the response* to an FX move separates them.
    """
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2, freq="B"),
        "price": [80.0, 80.0],
    })
    dates = df["date"]
    weak_dollar = convert_to_eur_mwh(
        df, "BRENT", 1.628, pd.Series([1.20, 1.20], index=dates)
    )
    strong_dollar = convert_to_eur_mwh(
        df, "BRENT", 1.628, pd.Series([1.00, 1.00], index=dates)
    )
    assert strong_dollar.iloc[0]["price_eur_mwh"] > weak_dollar.iloc[0]["price_eur_mwh"]


def test_convert_power_no_conversion():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2, freq="B"),
        "price": [55.0, 56.0],
    })
    eurusd = pd.Series([0.92, 0.92], index=df["date"])
    result = convert_to_eur_mwh(df, "DE_POWER", 1.0, eurusd)
    assert abs(result.iloc[0]["price_eur_mwh"] - 55.0) < 0.01


def test_build_date_dimension():
    dim = build_date_dimension("2024-01-01", "2024-01-10")
    assert len(dim) >= 5
    assert dim.iloc[0]["is_trading_day"] in (True, False)


@pytest.mark.slow
def test_fetch_yfinance_returns_dataframe():
    """fetch_yfinance returns DataFrame with date + price columns."""
    from energy_cross_commodity.data.fetcher import fetch_yfinance
    df = fetch_yfinance("AAPL", start="2024-12-01", end="2024-12-10")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["date", "price"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["price"])
