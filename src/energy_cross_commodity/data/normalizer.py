import pandas as pd


def convert_to_eur_mwh(
    df: pd.DataFrame,
    commodity_key: str,
    mwh_per_unit: float | None,
    eurusd_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert native prices to EUR/MWh using an FX series and unit conversion factor.

    ``eurusd_series`` follows the market convention of the ``EURUSD=X``
    quote: USD per one EUR, around 1.05-1.10 over the sample. A USD price
    is therefore *divided* by it to obtain EUR. Multiplying instead would
    inflate every USD-denominated commodity by roughly 10-20% and, worse,
    would reverse the sign of the FX contribution to its returns.
    """
    result = df.copy()
    result["price_native"] = result["price"]
    if mwh_per_unit and mwh_per_unit > 0 and mwh_per_unit != 1.0:
        if eurusd_series is not None:
            eurusd_aligned = eurusd_series.reindex(result["date"], method="ffill")
            usd_per_eur = eurusd_aligned.to_numpy(dtype=float)
            if not (usd_per_eur > 0).all():
                raise ValueError(
                    f"Non-positive or missing EURUSD rate while converting {commodity_key}."
                )
            result["price_eur_mwh"] = result["price"] / usd_per_eur / mwh_per_unit
        else:
            result["price_eur_mwh"] = result["price"] / mwh_per_unit
    else:
        result["price_eur_mwh"] = result["price"]
    result["commodity_key"] = commodity_key
    return result[["date", "commodity_key", "price_native", "price_eur_mwh"]]


def build_date_dimension(start: str, end: str) -> pd.DataFrame:
    """Build a business-day date dimension for the given range."""
    dates = pd.date_range(start, end, freq="B")
    return pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "quarter": dates.quarter,
        "is_trading_day": True,
    })
