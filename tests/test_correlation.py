"""Tests for GARCH and correlation modules."""

import numpy as np
import pandas as pd
from energy_cross_commodity.risk.garch import fit_univariate_garch, GARCHResult
from energy_cross_commodity.risk.correlation import compute_rolling_correlation


def test_garch_fit_returns_result():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.02, name="test")
    result = fit_univariate_garch(returns)
    assert isinstance(result, GARCHResult)
    assert len(result.cond_vol) == len(returns)
    assert len(result.std_residuals) == len(returns)
    assert "omega" in result.params
    alpha_key = [k for k in result.params if "alpha" in k][0]
    beta_key = [k for k in result.params if "beta" in k][0]
    assert result.params[alpha_key] + result.params[beta_key] < 1.0


def test_garch_std_residuals_have_unit_variance():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(1000) * 0.02, name="test")
    result = fit_univariate_garch(returns)
    assert abs(float(np.std(result.std_residuals)) - 1.0) < 0.3


def test_rolling_correlation_identical_series():
    rng = np.random.default_rng(42)
    n = 200
    r = rng.standard_normal(n) * 0.02
    df = pd.DataFrame({"A": r, "B": r}, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    corr = compute_rolling_correlation(df, window=60)
    last_corr = float(corr.sel(c1="A", c2="B").values[-1])
    assert abs(last_corr - 1.0) < 0.05
