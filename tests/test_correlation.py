"""Tests for GARCH and correlation modules."""

import pytest
import numpy as np
import pandas as pd
from energy_cross_commodity.risk.garch import fit_univariate_garch, GARCHResult
from energy_cross_commodity.risk.correlation import compute_rolling_correlation, analyze_dependence, fit_dcc_garch


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_dcc_identical_series_correlation_near_one():
    rng = np.random.default_rng(42)
    n = 300
    r = rng.standard_normal(n) * 0.02
    df = pd.DataFrame({"A": r, "B": r}, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    dcc = fit_dcc_garch(df)
    assert dcc.shape == (2, 2, n)
    last_corr = float(dcc.sel(c1="A", c2="B").values[-1])
    assert abs(last_corr - 1.0) < 0.05


@pytest.mark.slow
def test_dcc_shape_matches_input():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.02,
        "B": rng.standard_normal(n) * 0.02,
        "C": rng.standard_normal(n) * 0.02,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    dcc = fit_dcc_garch(df)
    assert dcc.shape == (3, 3, n)
    assert list(dcc.dims) == ["c1", "c2", "date"]
    assert list(dcc.coords["c1"].values) == ["A", "B", "C"]


def test_analyze_dependence_returns_dcc():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.02,
        "B": rng.standard_normal(n) * 0.02,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    result = analyze_dependence(df)
    assert result.dcc_corr is not None
    assert result.dcc_corr.shape == (2, 2, n)
    assert result.rolling_corr is not None
    assert len(result.garch_fits) == 2
