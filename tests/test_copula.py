"""Tests for t-copula module."""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from energy_cross_commodity.risk.copula import fit_t_copula, simulate_t_copula


@pytest.mark.slow
def test_fit_t_copula_returns_tail_dep():
    rng = np.random.default_rng(42)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = stats.multivariate_t(loc=[0, 0], shape=cov, df=5).rvs(2000, random_state=rng)
    df = pd.DataFrame(samples, columns=["A", "B"])
    copula = fit_t_copula(df)
    assert 0.4 < copula.correlation[0, 1] < 0.8
    assert copula.df >= 2
    assert copula.tail_dep[0, 1] > 0.0


def test_simulate_t_copula_shape():
    df = pd.DataFrame(np.random.randn(500, 3), columns=["X", "Y", "Z"])
    copula = fit_t_copula(df)
    sim = simulate_t_copula(copula, n=1000)
    assert sim.shape == (1000, 3)
    assert np.all((sim >= 0) & (sim <= 1))


def test_copula_dof3_tail_dep():
    """nu=3, rho=0.6 -> tail_dep ~0.37 per Section 8.1 formula."""
    nu = 3.0
    rho = 0.6
    t_quant = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    lam = 2 * stats.t.cdf(-t_quant, nu + 1)
    assert 0.30 < lam < 0.45, f"Expected tail_dep ~0.374, got {lam:.4f}"


@pytest.mark.slow
def test_high_df_approaches_gaussian():
    rng = np.random.default_rng(42)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = stats.multivariate_t(loc=[0, 0], shape=cov, df=30).rvs(2000, random_state=rng)
    df = pd.DataFrame(samples, columns=["A", "B"])
    copula = fit_t_copula(df)
    assert copula.tail_dep[0, 1] < 0.08
