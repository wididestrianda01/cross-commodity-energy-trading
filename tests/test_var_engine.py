"""Tests for VaR engine and scenarios."""

import pytest
import numpy as np
import pandas as pd
from energy_cross_commodity.risk.copula import fit_t_copula
from energy_cross_commodity.risk.var_engine import (
    compute_portfolio_var,
    kupiec_test,
    compute_rolling_var,
)
from energy_cross_commodity.risk.scenarios import run_scenario, ScenarioDefinition


def test_single_asset_var_positive():
    rng = np.random.default_rng(42)
    n = 1000
    r = rng.standard_normal(n) * 0.02
    rets = pd.DataFrame({"X": r})
    copula = fit_t_copula(rets)
    result = compute_portfolio_var(rets, {"X": 1_000_000}, copula, confidence=[0.95])
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
    result = compute_rolling_var(rets, positions, window=100, copula_fit_fn=fit_t_copula)
    assert list(result.columns) == ["date", "var_95", "var_99", "realized_pnl"]
    assert len(result) == n - 100 + 1


def test_scenario_pnl_matches_hand_calc():
    scenario = ScenarioDefinition(
        name="test", description="test",
        price_shocks={"TTF": 3.0},
    )
    positions = {"TTF": 1_000_000}
    result = run_scenario(positions, scenario, {"TTF": 40.0})
    assert abs(result.pnl_by_position["TTF"] - 3_000_000) < 1_000
    assert abs(result.total_pnl - 3_000_000) < 1_000
