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
    end_date = end or date.today().isoformat()
    data = yf.download(ticker, start=start, end=end_date, progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame(columns=["date", "price"])
    prices = data[column].rename("price")
    df = prices.reset_index()
    df.columns = ["date", "price"]
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "price"]]


def fetch_all_yfinance(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    results = {}
    yf_commodities = {k: v for k, v in cfg.commodities.items() if v.source == "yfinance"}
    for key, spec in yf_commodities.items():
        if spec.ticker:
            df = fetch_yfinance(spec.ticker, cfg.data.start_date, cfg.data.end_date)
            df["commodity_key"] = key
            results[key] = df
    return results
