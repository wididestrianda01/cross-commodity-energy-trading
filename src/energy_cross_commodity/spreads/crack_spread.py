"""3-2-1 Crack Spread — crude-to-products refinery margin."""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


GAL_PER_BBL = 42.0
"""US gallons per barrel — the RBOB contract is quoted in USD/gal."""

GASOIL_BBL_PER_TONNE = 7.45
"""Barrels per tonne for ICE Low Sulphur Gasoil (USD/tonne quote basis)."""


def compute_321_crack(
    rbob: np.ndarray,
    gasoil: np.ndarray,
    brent: np.ndarray,
    gal_per_bbl: float = GAL_PER_BBL,
    gasoil_bbl_per_tonne: float = GASOIL_BBL_PER_TONNE,
) -> np.ndarray:
    """Compute the 3-2-1 crack spread (refinery margin) in USD/bbl.

    The 3-2-1 ratio means 3 barrels of crude yield 2 barrels of gasoline
    and 1 barrel of distillate, so the per-barrel margin is
    ``(2*gasoline + 1*distillate - 3*crude) / 3``.

    The three legs trade in different units, so each product leg is
    converted to USD/bbl before the ratio is applied: RBOB is multiplied
    by 42 gal/bbl and Gasoil is divided by 7.45 bbl/tonne. Applying the
    ratio to the raw quotes would add USD/gal to USD/tonne and is
    dimensionally meaningless.

    Args:
        rbob: RBOB gasoline price array (USD/gal).
        gasoil: ICE Gasoil price array (USD/tonne).
        brent: Brent crude price array (USD/bbl).
        gal_per_bbl: Gallons per barrel for the RBOB conversion.
        gasoil_bbl_per_tonne: Barrels per tonne for the Gasoil conversion.

    Returns:
        Crack spread array in USD/bbl.
    """
    rbob_bbl = np.asarray(rbob, dtype=float) * gal_per_bbl
    gasoil_bbl = np.asarray(gasoil, dtype=float) / gasoil_bbl_per_tonne
    brent_bbl = np.asarray(brent, dtype=float)
    return (2.0 * rbob_bbl + 1.0 * gasoil_bbl - 3.0 * brent_bbl) / 3.0



def decompose_crack_spread(
    dates: pd.DatetimeIndex,
    crack: np.ndarray,
    period: int = 252,
) -> dict[str, np.ndarray]:
    """Decompose the crack spread into trend, seasonal, and residual components.

    Uses robust STL (Seasonal-Trend decomposition using LOESS, Cleveland
    et al. 1990) via statsmodels. STL is preferred over classical
    moving-average decomposition here because it tolerates a
    time-varying seasonal shape and, in robust mode, down-weights the
    price spikes that are common in energy spreads rather than letting
    them bleed into the trend.

    The seasonal period defaults to 252 trading days (approximately one
    calendar year). If the sample is shorter than two full periods, the
    period is reduced so the decomposition remains identified.

    Args:
        dates: DatetimeIndex aligned with the crack spread series.
        crack: Crack spread array (same length as dates).
        period: Seasonal period in trading days. Defaults to 252.

    Returns:
        Dict with keys ``trend``, ``seasonal``, ``resid`` — each a numpy
        array the same length as ``dates``, NaN where the input was NaN.
    """
    full = pd.Series(crack, index=dates, dtype=float)
    series = full.dropna()
    if len(series) < 2 * period:
        period = max(2, len(series) // 2)

    result = STL(series, period=period, robust=True).fit()

    # Reindex back onto the caller's dates so every component aligns with
    # the input, with NaN wherever the crack spread itself was missing.
    return {
        name: component.reindex(full.index).to_numpy()
        for name, component in (
            ("trend", result.trend),
            ("seasonal", result.seasonal),
            ("resid", result.resid),
        )
    }
