"""Univariate GARCH(1,1) wrapper via arch library."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from arch import arch_model


@dataclass
class GARCHResult:
    params: dict
    cond_vol: np.ndarray
    std_residuals: np.ndarray
    nu: float


def fit_univariate_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> GARCHResult:
    returns_clean = returns.dropna()
    model = arch_model(returns_clean, mean="constant", vol="GARCH", p=p, q=q, dist=dist)
    result = model.fit(disp="off")

    params = {k: float(v) for k, v in result.params.items()}
    nu = float(params.get("nu", 10))

    cond_vol = result.conditional_volatility.values
    std_residuals = returns_clean.values / cond_vol

    return GARCHResult(params=params, cond_vol=cond_vol, std_residuals=std_residuals, nu=nu)
