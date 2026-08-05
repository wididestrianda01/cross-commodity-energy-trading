from pathlib import Path
import duckdb
import pandas as pd

SQL_DIR = Path(__file__).parent.parent.parent.parent / "sql"


def get_connection(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_commodity (
            commodity_key VARCHAR(10) PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            category VARCHAR(10) NOT NULL,
            unit_native VARCHAR(10) NOT NULL,
            mwh_per_unit DOUBLE,
            source VARCHAR(20) NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_date (
            date DATE PRIMARY KEY,
            year SMALLINT,
            month SMALLINT,
            quarter SMALLINT,
            is_trading_day BOOLEAN
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_prices (
            date DATE NOT NULL,
            commodity_key VARCHAR(10) NOT NULL,
            price_native DOUBLE NOT NULL,
            price_eur_mwh DOUBLE NOT NULL,
            source VARCHAR(20) NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, commodity_key),
            FOREIGN KEY (commodity_key) REFERENCES dim_commodity(commodity_key)
        )
    """)


def seed_commodities(conn: duckdb.DuckDBPyConnection) -> None:
    commodities = [
        ("BRENT", "Brent Crude", "crude", "USD/bbl", 1.628, "yfinance"),
        ("TTF", "TTF Natural Gas", "gas", "EUR/MWh", 1.0, "yfinance"),
        ("EUA", "EUA Carbon Allowance", "carbon", "EUR/tCO2", None, "carbon_ets"),
        ("DE_POWER", "German Baseload Power", "power", "EUR/MWh", 1.0, "entsoe"),
        ("NP_SYS", "Nord Pool System Price", "power", "EUR/MWh", 1.0, "entsoe"),
        ("API2", "API2 Rotterdam Coal", "coal", "USD/tonne", 8.14, "yfinance"),
        ("RBOB", "RBOB Gasoline", "product", "USD/gal", None, "yfinance"),
        ("GASOIL", "ICE Gasoil", "product", "USD/tonne", None, "yfinance"),
        ("EURUSD", "EUR/USD FX", "fx", "EUR per USD", None, "yfinance"),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO dim_commodity VALUES (?, ?, ?, ?, ?, ?)",
        commodities,
    )


def query(
    conn: duckdb.DuckDBPyConnection,
    sql_file: str,
    params: dict | None = None,
) -> pd.DataFrame:
    path = SQL_DIR / sql_file
    sql = path.read_text()
    if params:
        for key, value in params.items():
            sql = sql.replace(f"{{{key}}}", str(value))
    return conn.execute(sql).df()
