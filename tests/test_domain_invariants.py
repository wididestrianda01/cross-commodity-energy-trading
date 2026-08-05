"""Domain-correctness invariants.

These tests guard the conventions that make the numbers mean what the
notebooks say they mean: energy-equivalence factors, the closed form for
t-copula tail dependence, DCC stationarity, and the coherence ordering
between VaR and Expected Shortfall. They are regression guards against
silently reverting to a plausible-looking but wrong constant or estimator.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy import stats

from energy_cross_commodity.risk.copula import compute_tail_dependence, fit_t_copula
from energy_cross_commodity.risk.correlation import fit_dcc_garch
from energy_cross_commodity.risk.var_engine import compute_portfolio_var

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"

KCAL_TO_KJ = 4.1868
KJ_PER_MWH = 3.6e6


@pytest.fixture(scope="module")
def config():
    return OmegaConf.load(CONFIG_PATH)


def test_api2_energy_content_matches_6000_kcal_nar(config):
    """API2 is quoted on 6,000 kcal/kg NAR, not on the 7,000 kcal/kg tce basis.

    Using the tce figure (8.141 MWh/t) overstates coal energy content by
    about 17% and inflates every dark-spread margin computed from it.
    """
    expected = 6000.0 * 1000.0 * KCAL_TO_KJ / KJ_PER_MWH  # ~6.978 MWh/tonne
    actual = float(config.commodities.API2.mwh_per_unit)

    assert actual == pytest.approx(expected, rel=5e-3)
    assert actual < 7.5, "value looks like a 7,000 kcal/kg tce factor"


def test_crack_leg_conversions_are_barrel_based(config):
    """RBOB and Gasoil must reach USD/bbl before entering the 3-2-1 crack."""
    crack = config.spreads.crack

    assert float(crack.rbob_gal_per_bbl) == pytest.approx(42.0)
    assert float(crack.gasoil_bbl_per_tonne) == pytest.approx(7.45, rel=1e-3)
    # Refined products are priced per gallon/tonne, so an MWh factor would
    # short-circuit the barrel conversion the crack spread depends on.
    assert config.commodities.RBOB.mwh_per_unit is None
    assert config.commodities.GASOIL.mwh_per_unit is None


def test_tail_dependence_matches_closed_form():
    """lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))).

    Pins the value quoted in the correlation notebook: nu = 5, rho = 0.6
    gives approximately 0.27, not the 0.20 that belongs to rho = 0.5.
    """
    rho, df = 0.6, 5.0
    corr = np.array([[1.0, rho], [rho, 1.0]])

    lam = compute_tail_dependence(corr, df)[0, 1]
    closed_form = 2.0 * stats.t.cdf(-np.sqrt((df + 1) * (1 - rho) / (1 + rho)), df + 1)

    assert lam == pytest.approx(closed_form, rel=1e-9)
    assert lam == pytest.approx(0.267, abs=5e-3)


def test_tail_dependence_vanishes_in_the_gaussian_limit():
    """As nu grows the t-copula approaches the Gaussian, where lambda = 0."""
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])

    assert compute_tail_dependence(corr, 4.0)[0, 1] > compute_tail_dependence(corr, 30.0)[0, 1]
    assert compute_tail_dependence(corr, 200.0)[0, 1] < 0.02


def test_tail_dependence_is_symmetric_with_zero_diagonal():
    corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.55], [-0.2, 0.55, 1.0]])

    lam = compute_tail_dependence(corr, 6.0)

    assert np.allclose(lam, lam.T)
    assert np.allclose(np.diag(lam), 0.0)


def test_copula_correlation_is_a_valid_correlation_matrix():
    """Kendall inversion must return a unit-diagonal, positive semi-definite matrix."""
    rng = np.random.default_rng(7)
    n = 600
    common = rng.standard_normal(n)
    data = pd.DataFrame(
        {
            "A": common + 0.6 * rng.standard_normal(n),
            "B": common + 0.6 * rng.standard_normal(n),
            "C": rng.standard_normal(n),
        }
    )

    fit = fit_t_copula(data)

    assert np.allclose(np.diag(fit.correlation), 1.0)
    assert np.allclose(fit.correlation, fit.correlation.T)
    assert np.linalg.eigvalsh(fit.correlation).min() > -1e-8
    assert fit.correlation[0, 1] > fit.correlation[0, 2]
    assert fit.df > 2.0


@pytest.mark.slow
def test_dcc_parameters_satisfy_stationarity():
    """a >= 0, b >= 0, a + b < 1, and every R_t is a correlation matrix."""
    rng = np.random.default_rng(11)
    n = 400
    common = rng.standard_normal(n) * 0.015
    returns = pd.DataFrame(
        {
            "A": common + rng.standard_normal(n) * 0.01,
            "B": common + rng.standard_normal(n) * 0.01,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )

    path = fit_dcc_garch(returns)
    a, b = float(path.attrs["dcc_a"]), float(path.attrs["dcc_b"])

    assert a >= 0.0 and b >= 0.0
    assert a + b < 1.0

    values = np.asarray(path)
    assert np.allclose(values[0, 0, :], 1.0, atol=1e-8)
    assert np.allclose(values[0, 1, :], values[1, 0, :])
    assert np.abs(values[0, 1, :]).max() <= 1.0 + 1e-8


@pytest.mark.slow
def test_expected_shortfall_dominates_var_at_the_same_level():
    """ES_97.5 >= VaR_97.5 >= VaR_95 — the coherence ordering FRTB relies on."""
    rng = np.random.default_rng(3)
    n = 600
    common = rng.standard_normal(n) * 0.02
    returns = pd.DataFrame(
        {
            "A": common + rng.standard_normal(n) * 0.01,
            "B": common + rng.standard_normal(n) * 0.015,
        }
    )
    positions = {"A": 2_000_000.0, "B": -1_000_000.0}

    result = compute_portfolio_var(
        returns, positions, fit_t_copula(returns), n_simulations=20000, seed=1
    )
    var_975 = float(-np.quantile(result.pnl_simulations, 0.025))

    assert result.var_95 <= var_975 <= result.var_99
    assert result.es_975 >= var_975


@pytest.mark.slow
def test_component_var_adds_up_to_total_var():
    """Euler allocation is exact for a positively homogeneous risk measure.

    The components are conditional expectations, so they carry Monte Carlo
    noise; the tolerance is on the aggregation identity, not on any single
    component.
    """
    rng = np.random.default_rng(5)
    n = 600
    common = rng.standard_normal(n) * 0.02
    returns = pd.DataFrame(
        {
            "A": common + rng.standard_normal(n) * 0.01,
            "B": common + rng.standard_normal(n) * 0.012,
            "C": rng.standard_normal(n) * 0.018,
        }
    )
    positions = {"A": 1_500_000.0, "B": 900_000.0, "C": -600_000.0}

    result = compute_portfolio_var(
        returns, positions, fit_t_copula(returns), n_simulations=20000, seed=2
    )

    assert sum(result.component_var.values()) == pytest.approx(result.var_95, rel=0.05)
