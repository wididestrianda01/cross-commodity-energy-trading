"""t-Copula fit and simulation for tail dependence modeling."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats, optimize


@dataclass
class CopulaFit:
    correlation: np.ndarray
    df: float
    tail_dep: np.ndarray


def fit_t_copula(std_residuals: pd.DataFrame) -> CopulaFit:
    """Fit a multivariate t-copula via pseudo-MLE on standardized residuals.

    Marginal CDFs are estimated per column via univariate t fits, then the
    copula degrees-of-freedom is optimised by maximum likelihood. Tail
    dependence coefficients are computed analytically.

    Args:
        std_residuals: DataFrame of GARCH-standardised residuals,
            one column per commodity.

    Returns:
        CopulaFit with correlation matrix, degrees of freedom, and tail
        dependence matrix.
    """
    n = len(std_residuals.columns)
    correlation = std_residuals.corr().values

    uniforms = pd.DataFrame(index=std_residuals.index)
    for col in std_residuals.columns:
        col_data = std_residuals[col].dropna()
        params = stats.t.fit(col_data)
        uniforms[col] = stats.t.cdf(col_data, *params)

    uniforms_clean = uniforms.dropna()
    u = uniforms_clean.values
    n_obs, k = u.shape

    def _neg_loglik(nu: float) -> float:
        if nu <= 2:
            return 1e10
        try:
            x = stats.t.ppf(u, nu)
            log_pdf_sum = np.sum(stats.t.logpdf(x, nu))
            log_det = np.linalg.slogdet(np.cov(x.T))[1]
            return -(log_pdf_sum - 0.5 * n_obs * log_det)
        except Exception:
            return 1e10

    result = optimize.minimize_scalar(_neg_loglik, bounds=(2.1, 30), method="bounded")
    df = float(result.x) if result.success else 5.0

    tail_dep = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = correlation[i, j]
            if abs(rho) < 0.999:
                t_quant = np.sqrt((df + 1) * (1 - rho) / (1 + rho))
                lam = 2 * stats.t.cdf(-t_quant, df + 1)
                tail_dep[i, j] = tail_dep[j, i] = max(0.0, lam)

    return CopulaFit(correlation=correlation, df=df, tail_dep=tail_dep)


def simulate_t_copula(copula: CopulaFit, n: int = 10000) -> np.ndarray:
    """Draw uniform samples from a fitted t-copula.

    Args:
        copula: Fitted copula with correlation matrix and df.
        n: Number of draws (default 10,000).

    Returns:
        Array of shape (n, k) with uniform marginals in [0, 1].
    """
    k = copula.correlation.shape[0]
    rng = np.random.default_rng(42)
    samples = stats.multivariate_t(
        loc=np.zeros(k), shape=copula.correlation, df=copula.df
    ).rvs(n, random_state=rng)
    return stats.t.cdf(samples, df=copula.df)
