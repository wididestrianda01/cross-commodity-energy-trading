"""Rolling correlation and dependence analysis."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
import xarray as xr
from energy_cross_commodity.risk.garch import fit_univariate_garch, GARCHResult


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


def fit_dcc_garch(
    returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    a: float = 0.05,
    b: float = 0.93,
) -> xr.DataArray:
    """Fit DCC-GARCH via Engle (2002) two-step estimator.

    Step 1: univariate GARCH -> standardized residuals.
    Step 2: DCC dynamics on the correlation targeting matrix S.
    """
    returns_clean = returns.dropna()
    n_series = returns_clean.shape[1]
    T = len(returns_clean)

    if n_series < 2 or T < 10:
        return compute_rolling_correlation(returns)

    eps = np.zeros((T, n_series))
    for i, col in enumerate(returns_clean.columns):
        try:
            garch_res = fit_univariate_garch(returns_clean[col], p=p, q=q)
            eps[:, i] = garch_res.std_residuals
        except Exception:
            return compute_rolling_correlation(returns)

    S = np.cov(eps, rowvar=False)
    Q = S.copy()
    R_cube = np.zeros((n_series, n_series, T))

    for t in range(T):
        if t > 0:
            e_prev = eps[t - 1, :].reshape(-1, 1)
            Q = (1 - a - b) * S + a * (e_prev @ e_prev.T) + b * Q
        d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12))
        D_inv = np.diag(d_inv)
        R = D_inv @ Q @ D_inv
        R_cube[:, :, t] = np.clip(R, -1.0, 1.0)

    commodities = returns_clean.columns.tolist()
    return xr.DataArray(
        R_cube,
        dims=["c1", "c2", "date"],
        coords={
            "c1": commodities,
            "c2": commodities,
            "date": returns_clean.index,
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
    except Exception:
        dcc_corr = None

    return CorrelationResult(
        rolling_corr=rolling_corr,
        garch_fits=garch_fits,
        dcc_corr=dcc_corr,
    )
