from datetime import date
import pandas as pd
import yfinance as yf
from omegaconf import DictConfig


def fetch_yfinance(
    ticker: str,
    start: str,
    end: str | None = None,
    column: str = "Close",
) -> pd.DataFrame:
    """Download daily price data for a single ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "BZ=F" for Brent).
        start: Start date string (YYYY-MM-DD).
        end: End date string. Defaults to today.
        column: OHLC column to extract. Defaults to "Close".

    Returns:
        DataFrame with columns [date, price].
    """
    end_date = end or date.today().isoformat()
    data = yf.download(ticker, start=start, end=end_date, progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame(columns=["date", "price"])
    close_data = data[column]
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]
    prices = close_data.rename("price")
    df = prices.reset_index()
    df.columns = ["date", "price"]
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "price"]]


def fetch_all_yfinance(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Fetch all yfinance-sourced commodities listed in the config.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict mapping commodity keys (e.g. "BRENT") to DataFrames with columns
        [date, price, commodity_key].
    """
    results = {}
    yf_commodities = {k: v for k, v in cfg.commodities.items() if v.source == "yfinance"}
    for key, spec in yf_commodities.items():
        if spec.ticker:
            df = fetch_yfinance(spec.ticker, cfg.data.start_date, cfg.data.end_date)
            df["commodity_key"] = key
            results[key] = df
    return results


def fetch_entsoe(
    bidding_zone: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download day-ahead power prices for a bidding zone from ENTSO-E.

    Uses the ``ENTSOE_API_KEY`` environment variable for authentication.
    Returns hourly prices resampled to a daily average.

    Args:
        bidding_zone: ENTSO-E EIC bidding-zone code (e.g. ``"10YDE-VE-------2"``).
        start: Start date string (YYYY-MM-DD).
        end: End date string. Defaults to today.

    Returns:
        DataFrame with columns [date, price] where price is the daily
        average EUR/MWh.
    """
    import os
    from entsoe import EntsoePandasClient

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY environment variable is not set. "
            "Get a key from https://transparency.entsoe.eu/"
        )

    end_date = end or date.today().isoformat()
    client = EntsoePandasClient(api_key=api_key)
    ts = client.query_day_ahead_prices(
        country_code=bidding_zone,
        start=pd.Timestamp(start, tz="Europe/Berlin"),
        end=pd.Timestamp(end_date, tz="Europe/Berlin"),
    )
    if ts is None or ts.empty:
        return pd.DataFrame(columns=["date", "price"])

    daily = ts.resample("D").mean().reset_index()
    daily.columns = ["date", "price"]
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    return daily[["date", "price"]]

def fetch_all_entsoe(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Fetch all ENTSO-E-sourced power prices listed in the config.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict mapping commodity keys (e.g. ``"DE_POWER"``) to DataFrames
        with columns [date, price, commodity_key].
    """
    results: dict[str, pd.DataFrame] = {}
    entsoe_commodities = {k: v for k, v in cfg.commodities.items() if v.source == "entsoe"}
    for key, spec in entsoe_commodities.items():
        zone = getattr(spec, "bidding_zone", None)
        if not zone:
            continue
        df = fetch_entsoe(zone, cfg.data.start_date, cfg.data.end_date)
        df["commodity_key"] = key
        results[key] = df
    return results


def fetch_carbon_ets(
    start: str,
    end: str | None = None,
    cache_dir: str = ".carbon_cache",
) -> pd.DataFrame:
    """Download EUA auction settlement prices from EEX public files.

    Downloads individual yearly XLSX files from the public EEX Group URL
    and parses them with carbon-ets's Excel parser. No API key required.

    Args:
        start: Start date string (YYYY-MM-DD).
        end: End date string. Defaults to today.
        cache_dir: Directory for caching downloaded EEX files.

    Returns:
        DataFrame with columns [date, price] in EUR/tCO2.
    """
    import io
    import os
    import requests as _requests
    from carbon_ets.data import _parse_eex_excel

    EEX_PUBLIC = "https://public.eex-group.com/eex/eua-auction-report"
    YEARS = list(range(2020, 2027))  # 2020–2026 available

    end_date = end or date.today().isoformat()
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for year in YEARS:
        filename = f"emission-spot-primary-market-auction-report-{year}-data.xlsx"
        cache_path = os.path.join(cache_dir, filename)

        content: bytes | None = None
        if os.path.exists(cache_path):
            content = open(cache_path, "rb").read()
        else:
            url = f"{EEX_PUBLIC}/{filename}"
            try:
                resp = _requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                resp.raise_for_status()
                content = resp.content
                with open(cache_path, "wb") as f:
                    f.write(content)
            except Exception:
                continue

        if content is None:
            continue

        try:
            parsed = _parse_eex_excel(io.BytesIO(content), filename)
            frames.append(parsed)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("carbon-ets: could not fetch any EEX EUA data")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date")

    df = df.reset_index(drop=True)
    df.columns = ["date", "price"] if len(df.columns) == 2 else ["date", "price"] + list(df.columns[2:])
    df = df[["date", "price"]]
    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"] >= start) & (df["date"] <= end_date)
    return df.loc[mask].copy()


def fetch_all_carbon_ets(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Fetch EUA carbon prices for all carbon-ets-sourced commodities.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict mapping commodity keys (e.g. ``"EUA"``) to DataFrames
        with columns [date, price, commodity_key].
    """
    results: dict[str, pd.DataFrame] = {}
    carbon_commodities = {k: v for k, v in cfg.commodities.items() if v.source == "carbon_ets"}
    for key in carbon_commodities:
        df = fetch_carbon_ets(cfg.data.start_date, cfg.data.end_date)
        if not df.empty:
            df["commodity_key"] = key
            results[key] = df
    return results
