"""Tests for stress scenario engine."""

import numpy as np

from energy_cross_commodity.risk.copula import CopulaFit
from energy_cross_commodity.risk.scenarios import (
    SCENARIOS,
    ScenarioDefinition,
    run_scenario,
    stressed_copula,
)


def test_scenario_pnl_matches_hand_calc():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"TTF": 3.0})
    positions = {"TTF": 1_000_000}
    result = run_scenario(positions, scenario)
    assert abs(result.pnl_by_position["TTF"] - 3_000_000) < 1_000
    assert abs(result.total_pnl - 3_000_000) < 1_000


def test_scenario_short_position_pnl():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"BRENT": -0.40})
    positions = {"BRENT": -1_000_000}
    result = run_scenario(positions, scenario)
    assert result.pnl_by_position["BRENT"] > 0


def test_zero_shock_zero_pnl():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"X": 0.0})
    result = run_scenario({"X": 1_000_000}, scenario)
    assert abs(result.total_pnl) < 1


def test_unshocked_factor_contributes_nothing():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"TTF": 0.5})
    result = run_scenario({"TTF": 1_000_000, "EUA": 2_000_000}, scenario)
    assert result.pnl_by_position["EUA"] == 0.0
    assert abs(result.total_pnl - 500_000) < 1


def test_scenario_pnl_is_deterministic():
    """Full revaluation has no simulation, so repeated runs must agree exactly."""
    scenario = SCENARIOS["recession"]
    positions = {"BRENT": 13_800_000, "TTF": 12_000_000, "DE_POWER": -4_000_000}
    first = run_scenario(positions, scenario)
    second = run_scenario(positions, scenario)
    assert first.total_pnl == second.total_pnl


def test_recession_scenario_loses_money_on_a_long_book():
    """Every shock is negative, so a net long book cannot profit."""
    positions = {"BRENT": 13_800_000, "TTF": 12_000_000, "EUA": 3_000_000}
    result = run_scenario(positions, SCENARIOS["recession"])
    assert result.total_pnl < 0


def test_stressed_copula_raises_correlations():
    base = np.array([[1.0, 0.2], [0.2, 1.0]])
    copula = CopulaFit(correlation=base, df=6.0, tail_dep=np.zeros((2, 2)))
    stressed = stressed_copula(copula, ["TTF", "BRENT"], "all_to_one")
    assert stressed.correlation[0, 1] == 0.90
    assert stressed.df == 6.0
    assert base[0, 1] == 0.2  # the fitted copula is left untouched
