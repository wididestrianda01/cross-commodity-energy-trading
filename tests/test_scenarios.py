"""Tests for stress scenario engine."""

from energy_cross_commodity.risk.scenarios import run_scenario, ScenarioDefinition


def test_scenario_pnl_matches_hand_calc():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"TTF": 3.0})
    positions = {"TTF": 1_000_000}
    result = run_scenario(positions, scenario, {"TTF": 40.0})
    assert abs(result.pnl_by_position["TTF"] - 3_000_000) < 1_000
    assert abs(result.total_pnl - 3_000_000) < 1_000


def test_scenario_short_position_pnl():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"BRENT": -0.40})
    positions = {"BRENT": -1_000_000}
    result = run_scenario(positions, scenario, {"BRENT": 80.0})
    assert result.pnl_by_position["BRENT"] > 0


def test_zero_shock_zero_pnl():
    scenario = ScenarioDefinition(name="test", description="test", price_shocks={"X": 0.0})
    result = run_scenario({"X": 1_000_000}, scenario, {"X": 100.0})
    assert abs(result.total_pnl) < 1
