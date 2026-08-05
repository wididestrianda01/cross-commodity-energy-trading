"""t-Copula fit and simulation for tail dependence modeling."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats, optimize

#: Pseudo-observations are kept strictly inside (0, 1); t.ppf(0) is -inf.
_U_EPS = 1e-6

#: Search bracket for the degrees-of-freedom parameter. Above ~50 the
#: t-copula is numerically indistinguishable from a Gaussian copula.
_DF_BOUNDS = (2.05, 50.0)


@dataclass
class CopulaFit:
    correlation: np.ndarray
    df: float
    tail_dep: np.ndarray
    loglik: float = float("nan")
    pseudo_obs: np.ndarray | None = field(default=None, repr=False)


def _pseudo_observations(data: pd.DataFrame) -> np.ndarray:
    """Rank-transform each margin to pseudo-observations in (0, 1).

    Uses the scaled empirical CDF ``rank / (n + 1)``. This is the
    canonical maximum pseudo-likelihood approach (Genest, Ghoudi &
    Rivest 1995): it leaves the marginal distributions unspecified so
    that a misspecified parametric margin cannot contaminate the
    estimated dependence structure.
    """
    ranks = data.rank(method="average").to_numpy()
    n_obs = ranks.shape[0]
    return np.clip(ranks / (n_obs + 1.0), _U_EPS, 1.0 - _U_EPS)


def _nearest_positive_definite(corr: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the positive-definite correlation cone.

    Kendall's-tau inversion is applied element-wise, so the resulting
    matrix is not guaranteed to be positive definite. Eigenvalues are
    floored and the matrix is renormalised to a unit diagonal.
    """
    sym = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if eigvals.min() >= min_eig:
        return sym

    rebuilt = eigvecs @ np.diag(np.maximum(eigvals, min_eig)) @ eigvecs.T
    d = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(d, d)


def _kendall_correlation(u: np.ndarray) -> np.ndarray:
    """Estimate the copula correlation matrix by Kendall's-tau inversion.

    For an elliptical copula, rho = sin(pi * tau / 2) (Lindskog, McNeil &
    Schmock 2003; Demarta & McNeil 2005). Kendall's tau is invariant to
    strictly increasing marginal transforms, so this estimator is robust
    to heavy tails, whereas the Pearson correlation of the raw residuals
    is biased for the copula parameter under exactly the fat-tailed
    conditions the t-copula is chosen to represent.
    """
    k = u.shape[1]
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            tau = stats.kendalltau(u[:, i], u[:, j]).statistic
            if not np.isfinite(tau):
                tau = 0.0
            corr[i, j] = corr[j, i] = np.sin(0.5 * np.pi * tau)

    return _nearest_positive_definite(corr)


def _copula_neg_loglik(nu: float, u: np.ndarray, corr: np.ndarray) -> float:
    """Negative log-likelihood of the t-copula density.

    The copula density separates the joint t density from the product of
    its own t margins:

        c(u) = f_{t,nu,R}(x) / prod_i f_{t,nu}(x_i),   x_i = t_nu^{-1}(u_i)

    Dropping the denominator (as a plain joint-density likelihood would)
    does not estimate a copula: the margins would then drive nu, and the
    result would not be invariant to the marginal transform.
    """
    if nu <= 2.0:
        return 1e10

    x = stats.t.ppf(u, nu)
    if not np.isfinite(x).all():
        return 1e10

    try:
        joint = stats.multivariate_t(
            loc=np.zeros(corr.shape[0]), shape=corr, df=nu
        ).logpdf(x)
    except (np.linalg.LinAlgError, ValueError):
        return 1e10

    margins = stats.t.logpdf(x, nu).sum(axis=1)
    total = float(np.sum(joint - margins))
    return -total if np.isfinite(total) else 1e10


def fit_t_copula(std_residuals: pd.DataFrame) -> CopulaFit:
    """Fit a multivariate t-copula by maximum pseudo-likelihood.

    The margins are handled non-parametrically via rank transform, the
    correlation matrix is estimated by Kendall's-tau inversion, and the
    degrees-of-freedom parameter is then obtained by maximising the
    t-copula log-likelihood at that correlation matrix. Lower- and
    upper-tail dependence coefficients follow analytically.

    Args:
        std_residuals: DataFrame of GARCH-standardised residuals,
            one column per commodity.

    Returns:
        CopulaFit with the correlation matrix, degrees of freedom, tail
        dependence matrix, the maximised log-likelihood, and the
        pseudo-observations used for the fit.

    Raises:
        ValueError: If fewer than two columns or fewer than 20 complete
            observations remain after dropping missing values.
    """
    data = std_residuals.dropna()
    n_assets = data.shape[1]
    if n_assets < 2:
        raise ValueError(f"t-copula requires at least 2 series, got {n_assets}.")
    if len(data) < 20:
        raise ValueError(
            f"t-copula requires at least 20 complete observations, got {len(data)}."
        )

    u = _pseudo_observations(data)
    correlation = _kendall_correlation(u)

    optimum = optimize.minimize_scalar(
        _copula_neg_loglik,
        bounds=_DF_BOUNDS,
        args=(u, correlation),
        method="bounded",
        options={"xatol": 1e-4},
    )
    df = float(optimum.x)
    loglik = -float(optimum.fun)

    return CopulaFit(
        correlation=correlation,
        df=df,
        tail_dep=compute_tail_dependence(correlation, df),
        loglik=loglik,
        pseudo_obs=u,
    )


def compute_tail_dependence(correlation: np.ndarray, df: float) -> np.ndarray:
    """Coefficient of tail dependence for a bivariate t-copula pair.

    lambda = 2 * t_{nu+1}( -sqrt((nu + 1)(1 - rho) / (1 + rho)) )

    The t-copula is radially symmetric, so the lower- and upper-tail
    coefficients are equal; this single matrix describes both. Unlike a
    Gaussian copula — where lambda = 0 for any rho < 1 — this is strictly
    positive for finite nu, which is the reason the t-copula is used for
    joint commodity stress.

    Args:
        correlation: Copula correlation matrix.
        df: Degrees of freedom.

    Returns:
        Symmetric matrix of tail dependence coefficients, zero diagonal.
    """
    n_assets = correlation.shape[0]
    tail_dep = np.zeros((n_assets, n_assets))

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            rho = float(np.clip(correlation[i, j], -0.999999, 0.999999))
            quantile = -np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho))
            lam = 2.0 * float(stats.t.cdf(quantile, df + 1.0))
            tail_dep[i, j] = tail_dep[j, i] = max(0.0, lam)

    return tail_dep


def simulate_t_copula(
    copula: CopulaFit,
    n: int = 10000,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw uniform samples from a fitted t-copula.

    Args:
        copula: Fitted copula with correlation matrix and df.
        n: Number of draws (default 10,000).
        seed: Seed or Generator for reproducibility. ``None`` draws a
            fresh stream, so Monte Carlo error can be assessed across
            runs instead of being hidden by a fixed internal seed.

    Returns:
        Array of shape (n, k) with uniform marginals in (0, 1).
    """
    k = copula.correlation.shape[0]
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    samples = stats.multivariate_t(
        loc=np.zeros(k), shape=copula.correlation, df=copula.df
    ).rvs(n, random_state=rng)
    samples = np.atleast_2d(samples).reshape(n, k)

    return np.clip(stats.t.cdf(samples, df=copula.df), _U_EPS, 1.0 - _U_EPS)
