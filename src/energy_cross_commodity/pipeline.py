"""Pipeline: fetch, normalize, and load energy commodity prices into DuckDB.

Usage:
    python -m energy_cross_commodity.pipeline               # full yfinance refresh
    python -m energy_cross_commodity.pipeline --synthetic   # synthetic fallback
    python -m energy_cross_commodity.pipeline --update      # incremental (last 30d)
"""

import argparse
import sys

from energy_cross_commodity.utils.config import load_config
from energy_cross_commodity.db import init_db, seed_commodities, get_connection
from energy_cross_commodity.data.fetcher import fetch_all_yfinance, fetch_all_entsoe, fetch_all_carbon_ets
from energy_cross_commodity.data.normalizer import convert_to_eur_mwh, build_date_dimension
from energy_cross_commodity.data.synthetic import generate_synthetic_prices


def run_synthetic(conn, cfg) -> None:
    """Load synthetic data into DuckDB as a fallback when real sources are unavailable."""
    df = generate_synthetic_prices(cfg.data.start_date, cfg.data.end_date or "2025-12-31")
    df["source"] = "synthetic"
    conn.execute("INSERT OR REPLACE INTO fact_prices(date, commodity_key, price_native, price_eur_mwh, source) SELECT * FROM df")
    end_date = cfg.data.end_date or "2025-12-31"
    dim_date = build_date_dimension(cfg.data.start_date, end_date)  # noqa: F841
    conn.execute("INSERT OR REPLACE INTO dim_date SELECT * FROM dim_date")


def run_yfinance(conn, cfg) -> None:
    """Fetch real market data via yfinance and load into DuckDB."""
    yf_data = fetch_all_yfinance(cfg)
    eurusd_df = yf_data.get("EURUSD")
    eurusd_series = (
        eurusd_df.set_index("date")["price"]
        if eurusd_df is not None and not eurusd_df.empty
        else None
    )
    for key, df in yf_data.items():
        if key == "EURUSD":
            spec = cfg.commodities[key]
            normalized = convert_to_eur_mwh(df, key, spec.mwh_per_unit, None)
            normalized["source"] = spec.source
            conn.execute("INSERT OR REPLACE INTO fact_prices(date, commodity_key, price_native, price_eur_mwh, source) SELECT * FROM normalized")
            continue
        spec = cfg.commodities[key]
        normalized = convert_to_eur_mwh(df, key, spec.mwh_per_unit, eurusd_series)
        normalized["source"] = spec.source
        conn.execute("INSERT OR REPLACE INTO fact_prices(date, commodity_key, price_native, price_eur_mwh, source) SELECT * FROM normalized")
    end_date = cfg.data.end_date or "2025-12-31"
    dim_date = build_date_dimension(cfg.data.start_date, end_date)  # noqa: F841
    conn.execute("INSERT OR REPLACE INTO dim_date SELECT * FROM dim_date")


def run_entsoe(conn, cfg) -> None:
    """Fetch ENTSO-E day-ahead power prices and load into DuckDB."""
    entsoe_data = fetch_all_entsoe(cfg)
    for key, df in entsoe_data.items():
        spec = cfg.commodities[key]
        normalized = convert_to_eur_mwh(df, key, spec.mwh_per_unit)
        normalized["source"] = spec.source
        conn.execute(
            "INSERT OR REPLACE INTO fact_prices(date, commodity_key, price_native, price_eur_mwh, source) "
            "SELECT * FROM normalized"
        )
    if entsoe_data:
        print(f"ENTSO-E: loaded {len(entsoe_data)} zones.")


def run_carbon_ets(conn, cfg) -> None:
    """Fetch EUA carbon prices from EEX and load into DuckDB."""
    carbon_data = fetch_all_carbon_ets(cfg)
    for key, df in carbon_data.items():
        spec = cfg.commodities[key]
        normalized = convert_to_eur_mwh(df, key, spec.mwh_per_unit)
        normalized["source"] = spec.source
        conn.execute(
            "INSERT OR REPLACE INTO fact_prices(date, commodity_key, price_native, price_eur_mwh, source) "
            "SELECT * FROM normalized"
        )
    if carbon_data:
        print("carbon-ets: EUA prices loaded.")


def run() -> None:
    """Execute the full data pipeline: fetch, normalize, and load into DuckDB.

    Tries yfinance, ENTSO-E, and carbon-ets in sequence. Fails hard if any
    real source cannot be reached — no synthetic fallback.

    Use ``--synthetic`` to load synthetic data instead (for testing only).
    """
    parser = argparse.ArgumentParser(description="Energy cross-commodity data pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of live sources")
    parser.add_argument("--update", action="store_true", help="Incremental update (last 30 days only)")
    args = parser.parse_args()

    cfg = load_config()
    conn = get_connection(cfg.data.db_path)
    init_db(conn)
    seed_commodities(conn)

    if args.synthetic:
        run_synthetic(conn, cfg)
    else:
        errors: list[str] = []

        try:
            run_yfinance(conn, cfg)
        except Exception as e:
            errors.append(f"yfinance: {e}")

        try:
            run_entsoe(conn, cfg)
        except Exception as e:
            errors.append(f"ENTSO-E: {e}")

        try:
            run_carbon_ets(conn, cfg)
        except Exception as e:
            errors.append(f"carbon-ets: {e}")

        if errors:
            failed = "; ".join(errors)
            msg = f"Pipeline failed — real data fetch errors: {failed}"
            print(msg, file=sys.stderr)
            conn.close()
            sys.exit(1)

    # Ensure date dimension is populated
    end_date = cfg.data.end_date or "2025-12-31"
    dim_df = build_date_dimension(cfg.data.start_date, end_date)
    conn.register("_dim_date_src", dim_df)
    conn.execute("INSERT OR REPLACE INTO dim_date SELECT * FROM _dim_date_src")

    count = conn.execute("SELECT COUNT(*) FROM fact_prices").fetchone()[0]
    print(f"Pipeline complete. {count} rows in fact_prices.")
    conn.close()


if __name__ == "__main__":
    run()
