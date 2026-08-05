"""Verify correlation time-series extraction for Tab 2."""

import numpy as np
import pandas as pd
from energy_cross_commodity.risk.correlation import compute_rolling_correlation


def test_rolling_corr_time_series_shape():
    """Rolling correlation cube -> TTF-DE_POWER pair extracted correctly."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = pd.DataFrame({
        "TTF": rng.standard_normal(n) * 0.03,
        "DE_POWER": rng.standard_normal(n) * 0.02,
        "BRENT": rng.standard_normal(n) * 0.015,
    }, index=dates)

    corr_cube = compute_rolling_correlation(rets, window=60)
    ttf_idx = list(corr_cube.coords["c1"].values).index("TTF")
    power_idx = list(corr_cube.coords["c2"].values).index("DE_POWER")
    pair_corr = corr_cube.values[ttf_idx, power_idx, :]

    assert len(pair_corr) == n - 60 + 1
    assert -1.0 <= float(pair_corr[-1]) <= 1.0
    assert list(corr_cube.coords["date"].values)[0] == dates[59]
