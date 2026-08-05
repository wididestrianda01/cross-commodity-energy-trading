"""Rolling correlation and dependence analysis."""

from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
import xarray as xr
from scipy import optimize
from energy_cross_commodity.risk.garch import fit_univariate_garch, GARCHResult

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    rolling_corr: xr.DataArray
    garch_fits: dict[str, GARCHResult]
    dcc_corr: xr.DataArray | None


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
) -> xr.DataArray:
    """Compute rolling Pearson correlation over a fixed window.

    Args:
        returns: DataFrame of log returns.
        window: Lookback window in trading days.

    Returns:
        xr.DataArray of shape (n_assets, n_assets, T-window+1).
    """
    commodities = returns.columns.tolist()
    dates = returns.index[window - 1 :]
    n_comm = len(commodities)
    corr_cube = np.zeros((n_comm, n_comm, len(dates)))

    for t_idx, _ in enumerate(dates):
        end = t_idx + window
        start = end - window
        corr_cube[:, :, t_idx] = returns.iloc[start:end].corr().values

    return xr.DataArray(
        corr_cube,
        dims=["c1", "c2", "date"],
        coords={"c1": commodities, "c2": commodities, "date": dates},
    )


#: Upper bound on a + b. The DCC correlation process is mean-reverting only
#: for a + b < 1; the optimiser is kept strictly inside that region.
_DCC_PERSISTENCE_CAP = 0.9999


def _dcc_correlation_path(
    a: float,
    b: float,
    eps: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Run the DCC(1,1) recursion and return the correlation path.

    Q_t = (1 - a - b) * S + a * e_{t-1} e_{t-1}' + b * Q_{t-1}, with
    R_t obtained by normalising Q_t to unit diagonal.

    Args:
        a: News (ARCH) coefficient.
        b: Persistence (GARCH) coefficient.
        eps: (T, n) standardised residuals.
        S: (n, n) unconditional correlation target.

    Returns:
        Array of shape (n, n, T) of conditional correlation matrices.
    """
    T, n_series = eps.shape
    Q = S.copy()
    R_cube = np.zeros((n_series, n_series, T))

    for t in range(T):
        if t > 0:
            e_prev = eps[t - 1, :].reshape(-1, 1)
            Q = (1 - a - b) * S + a * (e_prev @ e_prev.T) + b * Q
        d = np.sqrt(np.maximum(np.diag(Q), 1e-12))
        R_cube[:, :, t] = Q / np.outer(d, d)

    return R_cube


def _dcc_neg_loglik(theta: np.ndarray, eps: np.ndarray, S: np.ndarray) -> float:
    """Negative DCC quasi-log-likelihood for the second-stage estimation.

    Engle's two-step estimator maximises only the correlation part of the
    likelihood; the univariate volatility terms are constant with respect
    to (a, b) and are dropped:

        -2 log L(a, b) = sum_t [ log|R_t| + e_t' R_t^-1 e_t - e_t' e_t ]
    """
    a, b = float(theta[0]), float(theta[1])
    if a < 0 or b < 0 or a + b >= _DCC_PERSISTENCE_CAP:
        return 1e10

    R_cube = _dcc_correlation_path(a, b, eps, S)
    total = 0.0
    for t in range(eps.shape[0]):
        R = R_cube[:, :, t]
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e10
        e_t = eps[t, :]
        total += logdet + e_t @ np.linalg.solve(R, e_t) - e_t @ e_t

    return float(0.5 * total) if np.isfinite(total) else 1e10


def fit_dcc_garch(
    returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
) -> xr.DataArray:
    """Fit DCC-GARCH via the Engle (2002) two-step estimator.

    Step 1 fits a univariate GARCH(p, q) per series and extracts the
    standardised residuals. Step 2 estimates the DCC parameters (a, b) by
    quasi-maximum likelihood, subject to a >= 0, b >= 0 and a + b < 1, with
    the unconditional correlation of the residuals used as the targeting
    matrix S.

    Estimating (a, b) rather than asserting them is what makes this a
    fitted model: hard-coded coefficients would produce a correlation path
    that reacts to the data only through S, not through the dynamics.

    Args:
        returns: DataFrame of log returns (one column per commodity).
        p: GARCH lag order for the first stage.
        q: ARCH lag order for the first stage.

    Returns:
        xr.DataArray of shape (n_assets, n_assets, T) holding the
        conditional correlation path. The estimated parameters and the
        log-likelihood are attached as ``.attrs`` (``dcc_a``, ``dcc_b``,
        ``loglik``).

    Raises:
        ValueError: If fewer than two series or fewer than 50 usable
            observations are supplied — DCC is not identified there, and
            silently substituting a rolling correlation would misreport
            a different estimator as DCC output.
    """
    returns_clean = returns.dropna()
    n_series = returns_clean.shape[1]
    T = len(returns_clean)

    if n_series < 2:
        raise ValueError(f"DCC-GARCH requires at least 2 series, got {n_series}.")
    if T < 50:
        raise ValueError(
            f"DCC-GARCH requires at least 50 observations, got {T}."
        )

    eps = np.column_stack([
        fit_univariate_garch(returns_clean[col], p=p, q=q).std_residuals
        for col in returns_clean.columns
    ])

    # Correlation targeting: S must be a correlation matrix, not a
    # covariance matrix, or R_t is not normalised to a unit diagonal.
    S = np.corrcoef(eps, rowvar=False)

    optimum = optimize.minimize(
        _dcc_neg_loglik,
        x0=np.array([0.02, 0.95]),
        args=(eps, S),
        method="SLSQP",
        bounds=[(1e-6, 0.5), (1e-6, 0.999)],
        constraints=[{
            "type": "ineq",
            "fun": lambda th: _DCC_PERSISTENCE_CAP - th[0] - th[1],
        }],
        options={"maxiter": 200, "ftol": 1e-8},
    )
    a, b = (float(optimum.x[0]), float(optimum.x[1]))

    R_cube = np.clip(_dcc_correlation_path(a, b, eps, S), -1.0, 1.0)

    commodities = returns_clean.columns.tolist()
    return xr.DataArray(
        R_cube,
        dims=["c1", "c2", "date"],
        coords={
            "c1": commodities,
            "c2": commodities,
            "date": returns_clean.index,
        },
        attrs={
            "dcc_a": a,
            "dcc_b": b,
            "persistence": a + b,
            "loglik": -float(optimum.fun),
            "converged": bool(optimum.success),
        },
    )


def analyze_dependence(
    returns: pd.DataFrame,
    window: int = 60,
) -> CorrelationResult:
    """Run the full dependence analysis: rolling correlation + DCC-GARCH.

    Args:
        returns: DataFrame of log returns.
        window: Window for the rolling correlation fallback.

    Returns:
        CorrelationResult with rolling correlation, per-commodity GARCH fits,
        and the DCC correlation cube.
    """
    garch_fits = {}
    for col in returns.columns:
        garch_fits[col] = fit_univariate_garch(returns[col])

    rolling_corr = compute_rolling_correlation(returns, window)

    try:
        dcc_corr = fit_dcc_garch(returns)
    except (ValueError, np.linalg.LinAlgError) as exc:
        # DCC is unidentified or numerically degenerate for this sample.
        # Report it rather than swallowing it: the caller must be able to
        # tell "no DCC estimate" apart from "DCC estimate of zero".
        logger.warning("DCC-GARCH estimation failed, dcc_corr=None: %s", exc)
        dcc_corr = None

    return CorrelationResult(
        rolling_corr=rolling_corr,
        garch_fits=garch_fits,
        dcc_corr=dcc_corr,
    )
