"""Pipeline: fetch, normalize, and load energy commodity prices into DuckDB."""

from energy_cross_commodity.utils.config import load_config
from energy_cross_commodity.db import init_db, seed_commodities, get_connection
from energy_cross_commodity.data.fetcher import fetch_all_yfinance
from energy_cross_commodity.data.normalizer import convert_to_eur_mwh, build_date_dimension


def run() -> None:
    cfg = load_config()
    conn = get_connection(cfg.data.db_path)
    init_db(conn)
    seed_commodities(conn)

    yf_data = fetch_all_yfinance(cfg)

    eurusd_df = yf_data.get("EURUSD")
    eurusd_series = (
        eurusd_df.set_index("date")["price"]
        if eurusd_df is not None and not eurusd_df.empty
        else None
    )

    for key, df in yf_data.items():
        if key == "EURUSD":
            continue
        spec = cfg.commodities[key]
        normalized = convert_to_eur_mwh(df, key, spec.mwh_per_unit, eurusd_series)
        normalized["source"] = spec.source
        conn.execute("INSERT OR REPLACE INTO fact_prices SELECT * FROM normalized")

    end_date = cfg.data.end_date or "2025-12-31"
    dim_date = build_date_dimension(cfg.data.start_date, end_date)
    conn.execute("INSERT OR REPLACE INTO dim_date SELECT * FROM dim_date")

    count = conn.execute("SELECT COUNT(*) FROM fact_prices").fetchone()[0]
    print(f"Pipeline complete. {count} rows in fact_prices.")
    conn.close()


if __name__ == "__main__":
    run()
