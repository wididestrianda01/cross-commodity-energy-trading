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
    commodities = returns.columns.tolist()
    dates = returns.index[window - 1:]
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


def analyze_dependence(
    returns: pd.DataFrame,
    window: int = 60,
) -> CorrelationResult:
    garch_fits = {}
    for col in returns.columns:
        garch_fits[col] = fit_univariate_garch(returns[col])

    rolling_corr = compute_rolling_correlation(returns, window)
    dcc_corr = None

    return CorrelationResult(
        rolling_corr=rolling_corr,
        garch_fits=garch_fits,
        dcc_corr=dcc_corr,
    )
