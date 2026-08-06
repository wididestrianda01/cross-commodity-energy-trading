"""Tests for the VaR engine and its backtests."""

import pytest
import numpy as np
import pandas as pd
from energy_cross_commodity.risk.copula import fit_t_copula
from energy_cross_commodity.risk.var_engine import (
    basel_traffic_light,
    christoffersen_test,
    compute_portfolio_var,
    compute_rolling_var,
    fit_fhs_copula,
    kupiec_test,
)


def test_single_asset_var_positive():
    rng = np.random.default_rng(42)
    n = 1000
    r = rng.standard_normal(n) * 0.02
    rets = pd.DataFrame({"X": r})
    # A one-dimensional copula carries no dependence information.
    result = compute_portfolio_var(rets, {"X": 1_000_000}, None, confidence=[0.95])
    assert result.var_99 > result.var_95
    assert result.es_975 >= result.var_95


@pytest.mark.slow
def test_component_var_summarizes_risk():
    rng = np.random.default_rng(42)
    n = 500
    rets = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.02,
        "B": rng.standard_normal(n) * 0.01,
    })
    copula = fit_t_copula(rets)
    positions = {"A": 1_000_000, "B": 500_000}
    result = compute_portfolio_var(rets, positions, copula)
    assert len(result.component_var) == 2
    assert "A" in result.component_var


def test_net_zero_book_var_near_zero():
    """Long + short on nearly identical assets -> net-zero VaR ~ 0."""
    rng = np.random.default_rng(42)
    n = 500
    r = rng.standard_normal(n) * 0.02
    r_short = r + rng.standard_normal(n) * 0.0002
    rets = pd.DataFrame({"X_LONG": r, "X_SHORT": r_short})
    copula = fit_t_copula(rets)
    result = compute_portfolio_var(rets, {"X_LONG": 1_000_000, "X_SHORT": -1_000_000}, copula)
    assert abs(result.var_95) < 500.0
    assert abs(result.var_99) < 1000.0
    assert abs(result.es_975) < 1500.0


def test_kupiec_no_breaches():
    """5 breaches out of 100 at 95% -> p-value > 0.05 (matches expected rate)."""
    result = kupiec_test(breaches=5, total=100, confidence=0.95)
    assert result["p_value"] > 0.05
    assert result["lr_stat"] >= 0
    assert result["breaches"] == 5
    assert result["total"] == 100


def test_rolling_var_output():
    """Rolling VaR produces expected columns and length."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = pd.DataFrame(
        {"X": rng.standard_normal(n) * 0.02}, index=dates
    )
    positions = {"X": 1_000_000}
    result = compute_rolling_var(
        rets, positions, window=100, fit_fn=lambda _w: (None, None), n_simulations=500
    )
    assert list(result.columns) == ["date", "var_95", "var_99", "realized_pnl"]
    # One row per window that has a *following* day to be scored against,
    # so the series is one shorter than the number of windows.
    assert len(result) == n - 100


def test_rolling_var_is_dated_by_the_realised_day():
    """A breach must land on the day the loss happened, not the forecast origin."""
    rng = np.random.default_rng(7)
    n = 160
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = pd.DataFrame({"X": rng.standard_normal(n) * 0.01}, index=dates)
    result = compute_rolling_var(
        rets, {"X": 1_000_000}, window=120, fit_fn=lambda _w: (None, None), n_simulations=500
    )
    first_row = result.iloc[0]
    assert first_row["date"] == dates[120]
    assert first_row["realized_pnl"] == pytest.approx(
        float(rets["X"].iloc[120]) * 1_000_000
    )


def test_fhs_copula_is_fitted_on_residuals_not_raw_returns():
    """The FHS quantile step reads residuals, so the copula must describe residuals."""
    rng = np.random.default_rng(3)
    n = 400
    vol = np.abs(np.sin(np.linspace(0, 12, n))) + 0.2
    common = rng.standard_normal(n)
    rets = pd.DataFrame(
        {
            "A": (common + 0.5 * rng.standard_normal(n)) * vol * 0.02,
            "B": (common + 0.5 * rng.standard_normal(n)) * vol * 0.02,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    copula, fits = fit_fhs_copula(rets)
    assert set(fits) == {"A", "B"}

    residuals = pd.DataFrame(
        {c: np.asarray(fits[c].std_residuals, dtype=float) for c in rets.columns}
    ).dropna()
    on_residuals = fit_t_copula(residuals)
    on_raw = fit_t_copula(rets)

    assert copula.correlation[0, 1] == pytest.approx(on_residuals.correlation[0, 1])
    assert copula.correlation[0, 1] != pytest.approx(on_raw.correlation[0, 1])


def test_christoffersen_flags_clustered_breaches():
    """Same breach count, different arrival pattern: only clustering is rejected."""
    n = 250
    clustered = np.zeros(n, dtype=int)
    clustered[100:113] = 1
    spread = np.zeros(n, dtype=int)
    spread[::19][:13] = 1

    clustered_result = christoffersen_test(clustered, 0.95)
    spread_result = christoffersen_test(spread, 0.95)

    assert clustered_result["lr_ind"] > spread_result["lr_ind"]
    assert clustered_result["p_ind"] < 0.05


def test_christoffersen_cc_exceeds_kupiec():
    """LR_cc = LR_uc + LR_ind, so it can never fall below the Kupiec statistic."""
    breaches = np.zeros(250, dtype=int)
    breaches[[10, 11, 12, 90, 91, 200]] = 1
    cc = christoffersen_test(breaches, 0.95)
    uc = kupiec_test(int(breaches.sum()), breaches.size, 0.95)
    assert cc["lr_cc"] >= uc["lr_stat"]


@pytest.mark.parametrize(
    "breaches, zone",
    [(0, "GREEN"), (4, "GREEN"), (5, "YELLOW"), (9, "YELLOW"), (10, "RED")],
)
def test_basel_traffic_light_zones(breaches, zone):
    """Basel zone boundaries for a 99% VaR over 250 days."""
    assert basel_traffic_light(breaches) == zone
