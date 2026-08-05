"""Synthetic data generator for testing and fallback."""

import numpy as np
import pandas as pd
from scipy import stats


def generate_synthetic_prices(
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate correlated synthetic commodity prices via t-copula GBM.

    Produces 9 correlated series using a pre-specified correlation matrix
    and t-distributed innovations.

    Args:
        start_date: First business day of the series.
        end_date: Last business day of the series.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [date, commodity_key, price_native, price_eur_mwh, source].
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="B")
    n = len(dates)

    commodities = {
        "BRENT": {"start": 65, "vol": 0.025, "drift": 0.0001},
        "TTF": {"start": 18, "vol": 0.035, "drift": 0.0002},
        "EUA": {"start": 25, "vol": 0.030, "drift": 0.0003},
        "DE_POWER": {"start": 45, "vol": 0.022, "drift": 0.0001},
        "NP_SYS": {"start": 40, "vol": 0.020, "drift": 0.0001},
        "API2": {"start": 70, "vol": 0.022, "drift": 0.0000},
        "RBOB": {"start": 1.80, "vol": 0.025, "drift": 0.0001},
        "GASOIL": {"start": 600, "vol": 0.020, "drift": 0.0001},
        "EURUSD": {"start": 1.12, "vol": 0.006, "drift": 0.0000},
    }

    corr = np.array([
        [1.00, 0.25, 0.30, 0.20, 0.15, 0.20, 0.80, 0.75, -0.10],
        [0.25, 1.00, 0.45, 0.40, 0.30, 0.35, 0.10, 0.15, -0.05],
        [0.30, 0.45, 1.00, 0.50, 0.30, 0.20, 0.15, 0.20, -0.05],
        [0.20, 0.40, 0.50, 1.00, 0.60, 0.30, 0.10, 0.10, -0.05],
        [0.15, 0.30, 0.30, 0.60, 1.00, 0.25, 0.05, 0.05, -0.05],
        [0.20, 0.35, 0.20, 0.30, 0.25, 1.00, 0.05, 0.10, -0.05],
        [0.80, 0.10, 0.15, 0.10, 0.05, 0.05, 1.00, 0.70, -0.05],
        [0.75, 0.15, 0.20, 0.10, 0.05, 0.10, 0.70, 1.00, -0.05],
        [-0.10, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, 1.00],
    ])

    nu = 5.0
    L = np.linalg.cholesky(corr)
    z = stats.t.rvs(df=nu, size=(n, 9), random_state=rng)
    returns = z @ L.T

    rows = []
    keys = list(commodities.keys())
    for i, key in enumerate(keys):
        spec = commodities[key]
        vol = spec["vol"]
        drift = spec["drift"]
        ret = returns[:, i] * vol + drift
        price = spec["start"] * np.exp(np.cumsum(ret))
        for t, date in enumerate(dates):
            rows.append({
                "date": date,
                "commodity_key": key,
                "price_native": round(float(price[t]), 4),
                "price_eur_mwh": round(float(price[t]), 4),
                "source": "synthetic",
            })

    return pd.DataFrame(rows)
