"""Multi-commodity VaR/ES engine with t-copula simulation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from energy_cross_commodity.risk.copula import CopulaFit, simulate_t_copula


@dataclass
class PortfolioVaR:
    var_95: float
    var_99: float
    es_975: float
    component_var: dict[str, float]
    pnl_simulations: np.ndarray


def compute_portfolio_var(
    returns: pd.DataFrame,
    positions: dict[str, float],
    copula: CopulaFit,
    confidence: list[float] | None = None,
    n_simulations: int = 10000,
) -> PortfolioVaR:
    if confidence is None:
        confidence = [0.95, 0.99]

    uniforms = simulate_t_copula(copula, n=n_simulations)
    n_assets = returns.shape[1]
    commodities = list(returns.columns)

    garch_vols = np.std(returns.values, axis=0)
    z_scores = stats_norm_ppf(uniforms)
    simulated_returns = z_scores * garch_vols

    position_array = np.array([positions.get(c, 0.0) for c in commodities])
    if n_assets == 1:
        pnl = simulated_returns.flatten() * position_array[0]
    else:
        pnl = simulated_returns @ position_array

    var_95 = float(-np.quantile(pnl, 1 - 0.95))
    var_99 = float(-np.quantile(pnl, 1 - 0.99))
    es_975 = float(-np.mean(pnl[pnl <= np.quantile(pnl, 1 - 0.975)]))

    component_var = {}
    eps = 1.0
    for i, comm in enumerate(commodities):
        pos_perturbed = position_array.copy()
        pos_perturbed[i] += eps
        if n_assets == 1:
            pnl_perturbed = simulated_returns.flatten() * pos_perturbed[0]
        else:
            pnl_perturbed = simulated_returns @ pos_perturbed
        var_perturbed = float(-np.quantile(pnl_perturbed, 1 - 0.95))
        component_var[comm] = var_perturbed - var_95

    return PortfolioVaR(
        var_95=var_95, var_99=var_99, es_975=es_975,
        component_var=component_var, pnl_simulations=pnl,
    )


def stats_norm_ppf(u: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
