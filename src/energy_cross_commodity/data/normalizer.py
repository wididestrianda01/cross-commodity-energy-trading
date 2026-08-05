import pandas as pd
import numpy as np


def convert_to_eur_mwh(
    df: pd.DataFrame,
    commodity_key: str,
    mwh_per_unit: float | None,
    eurusd_series: pd.Series,
) -> pd.DataFrame:
    result = df.copy()
    result["price_native"] = result["price"]
    if mwh_per_unit and mwh_per_unit > 0 and mwh_per_unit != 1.0:
        eurusd_aligned = eurusd_series.reindex(result["date"], method="ffill")
        result["price_eur_mwh"] = result["price"] * eurusd_aligned.values / mwh_per_unit
    elif mwh_per_unit == 1.0:
        result["price_eur_mwh"] = result["price"]
    else:
        result["price_eur_mwh"] = result["price"]
    result["commodity_key"] = commodity_key
    return result[["date", "commodity_key", "price_native", "price_eur_mwh"]]


def build_date_dimension(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    return pd.DataFrame({
        "date": dates,
        "year": dates.year.astype("int16"),
        "month": dates.month.astype("int16"),
        "quarter": dates.quarter.astype("int16"),
        "is_trading_day": True,
    })
