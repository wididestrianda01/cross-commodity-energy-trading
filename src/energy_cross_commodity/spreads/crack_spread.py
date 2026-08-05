"""3-2-1 Crack Spread — crude-to-products refinery margin."""

import numpy as np


from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

def compute_321_crack(
    rbob: np.ndarray,
    gasoil: np.ndarray,
    brent: np.ndarray,
) -> np.ndarray:
    """Compute the 3-2-1 crack spread (refinery margin).

    The 3-2-1 ratio means 3 barrels of crude yield 2 barrels of gasoline
    and 1 barrel of distillate. Margin = (2*RBOB + 1*Gasoil - 3*Brent) / 3.

    Args:
        rbob: RBOB gasoline price array (USD/gal).
        gasoil: ICE Gasoil price array (USD/tonne).
        brent: Brent crude price array (USD/bbl).

    Returns:
        Crack spread array in the same units as the inputs.
    """
    return (2.0 * rbob + 1.0 * gasoil - 3.0 * brent) / 3.0



def decompose_crack_spread(
    dates: pd.DatetimeIndex,
    crack: np.ndarray,
    period: int = 252,
) -> dict[str, np.ndarray]:
    """Decompose the crack spread into trend, seasonal, and residual components.

    Uses additive STL decomposition via statsmodels. The seasonal period
    defaults to 252 trading days (approximately one calendar year).

    Args:
        dates: DatetimeIndex aligned with the crack spread series.
        crack: Crack spread array (same length as dates).
        period: Seasonal period in trading days. Defaults to 252.

    Returns:
        Dict with keys ``trend``, ``seasonal``, ``resid`` — each a numpy array
        the same length as the input.
    """
    series = pd.Series(crack, index=dates).dropna()
    if len(series) < 2 * period:
        period = max(2, len(series) // 2)

    result = seasonal_decompose(series, model="additive", period=period)

    trend = result.trend.to_numpy()
    seasonal = result.seasonal.to_numpy()
    resid = result.resid.to_numpy()

    nan_mask = np.isnan(trend) | np.isnan(seasonal) | np.isnan(resid)
    trend = np.where(nan_mask, np.nan, trend)
    seasonal = np.where(nan_mask, np.nan, seasonal)
    resid = np.where(nan_mask, np.nan, resid)

    return {"trend": trend, "seasonal": seasonal, "resid": resid}
