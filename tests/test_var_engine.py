"""Tests for VaR engine and scenarios."""

import numpy as np
import pandas as pd
from energy_cross_commodity.risk.copula import fit_t_copula
from energy_cross_commodity.risk.var_engine import compute_portfolio_var
from energy_cross_commodity.risk.scenarios import run_scenario, ScenarioDefinition


def test_single_asset_var_positive():
    rng = np.random.default_rng(42)
    n = 1000
    r = rng.standard_normal(n) * 0.02
    rets = pd.DataFrame({"X": r})
    copula = fit_t_copula(rets)
    result = compute_portfolio_var(rets, {"X": 1_000_000}, copula, confidence=[0.95])
    assert result.var_99 > result.var_95
    assert result.es_975 >= result.var_95  # ES >= VaR at same or lower quantile



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


def test_scenario_pnl_matches_hand_calc():
    scenario = ScenarioDefinition(
        name="test", description="test",
        price_shocks={"TTF": 3.0},
    )
    positions = {"TTF": 1_000_000}
    result = run_scenario(positions, scenario, {"TTF": 40.0})
    assert abs(result.pnl_by_position["TTF"] - 3_000_000) < 1_000
    assert abs(result.total_pnl - 3_000_000) < 1_000
