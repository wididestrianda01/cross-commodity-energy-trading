"""Univariate GARCH(1,1) wrapper via arch library."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from arch import arch_model


#: Daily log returns are O(1e-2); GARCH optimisers are better conditioned on
#: percentage-scale data. Fit on returns * 100, then divide volatilities back.
_SCALE = 100.0


@dataclass
class GARCHResult:
    params: dict
    cond_vol: np.ndarray
    std_residuals: np.ndarray
    nu: float
    sigma_forecast: float
    mu: float


def fit_univariate_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> GARCHResult:
    """Fit a univariate GARCH(p,q) model with Student-t errors.

    The model is estimated on percentage-scale returns for numerical
    stability and the fitted volatilities are converted back to the
    caller's return units. ``rescale=False`` is set explicitly so that
    ``arch`` cannot apply a second, hidden rescaling that would leave
    ``cond_vol`` and ``sigma_forecast`` in different units.

    Standardised residuals are taken from the fitted model
    (``(r_t - mu) / sigma_t``) rather than ``r_t / sigma_t``: omitting the
    conditional mean leaves a drift in the residuals that propagates into
    the copula fit and the simulated tail.

    Args:
        returns: Time series of log returns.
        p: GARCH lag order.
        q: ARCH lag order.
        dist: Error distribution ("t", "normal", "skewt").

    Returns:
        GARCHResult with fitted parameters, conditional volatility,
        standardised residuals, degrees of freedom, the one-step-ahead
        volatility forecast, and the conditional mean — all expressed in
        the caller's return units.
    """
    returns_clean = returns.dropna()
    model = arch_model(
        returns_clean * _SCALE,
        mean="constant",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )
    result = model.fit(disp="off")

    params = {k: float(v) for k, v in result.params.items()}
    nu = float(params.get("nu", 10))

    cond_vol = result.conditional_volatility.values / _SCALE
    std_residuals = result.std_resid.values

    forecast_var = result.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    sigma_forecast = float(np.sqrt(forecast_var)) / _SCALE
    mu = float(params.get("mu", 0.0)) / _SCALE

    return GARCHResult(
        params=params,
        cond_vol=cond_vol,
        std_residuals=std_residuals,
        nu=nu,
        sigma_forecast=sigma_forecast,
        mu=mu,
    )
