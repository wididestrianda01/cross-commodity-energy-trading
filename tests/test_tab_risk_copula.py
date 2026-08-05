"""Verify Tab 3 VaR uses copula engine, not standalone Gaussian."""

import numpy as np
import pandas as pd
from energy_cross_commodity.risk.copula import fit_t_copula
from energy_cross_commodity.risk.garch import fit_univariate_garch
from energy_cross_commodity.risk.var_engine import compute_portfolio_var


def test_copula_var_differs_from_standalone():
    """Component VaR from copula simulation != standalone position*vol*1.645."""
    rng = np.random.default_rng(42)
    n = 500
    returns = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.02,
        "B": rng.standard_normal(n) * 0.015,
    })
    positions = {"A": 1_000_000, "B": 500_000}

    # Standalone
    vol = returns.std() * np.sqrt(252)
    standalone_a = abs(positions["A"]) * vol["A"] * 1.645
    standalone_b = abs(positions["B"]) * vol["B"] * 1.645

    # Copula engine
    std_resids = pd.DataFrame({
        "A": fit_univariate_garch(returns["A"]).std_residuals,
        "B": fit_univariate_garch(returns["B"]).std_residuals,
    })
    copula = fit_t_copula(std_resids)
    result = compute_portfolio_var(returns, positions, copula)

    assert result.var_95 > 0
    assert result.es_975 >= result.var_95
    # Euler component VaR should differ from standalone (diversification effect)
    assert abs(result.component_var["A"] - standalone_a) > 1.0
    assert abs(result.component_var["B"] - standalone_b) > 1.0


def test_component_var_sums_approximately_to_total():
    """Euler component VaR ≈ total VaR (within 5% tolerance)."""
    rng = np.random.default_rng(42)
    n = 500
    returns = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.02,
        "B": rng.standard_normal(n) * 0.015,
        "C": rng.standard_normal(n) * 0.01,
    })
    positions = {"A": 1_000_000, "B": 500_000, "C": 300_000}

    std_resids = pd.DataFrame({
        c: fit_univariate_garch(returns[c]).std_residuals for c in returns.columns
    })
    copula = fit_t_copula(std_resids)
    result = compute_portfolio_var(returns, positions, copula)

    comp_sum = sum(result.component_var.values())
    assert abs(comp_sum - result.var_95) / result.var_95 < 0.05
