"""Multi-commodity VaR/ES engine with t-copula simulation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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
    """Simulate portfolio P&L via t-copula and compute VaR/ES/component VaR.

    Draws correlated uniform samples from the fitted copula, maps them to
    returns via inverse normal CDF scaled by historical volatility, computes
    portfolio P&L, and extracts VaR at 95%/99%, expected shortfall at 97.5%,
    and Euler-allocated component VaR via finite differences.

    Args:
        returns: Historical log returns (one column per asset).
        positions: Notional exposures keyed by commodity.
        copula: Fitted t-copula for dependence structure.
        confidence: VaR confidence levels (default [0.95, 0.99]).
        n_simulations: Number of Monte Carlo draws.

    Returns:
        PortfolioVaR with var_95, var_99, es_975, component_var, and
        the full P&L simulation array.
    """
    if confidence is None:
        confidence = [0.95, 0.99]

    uniforms = simulate_t_copula(copula, n=n_simulations)
    n_assets = returns.shape[1]
    commodities = list(returns.columns)

    garch_vols = np.std(returns.values, axis=0)
    z_scores = _norm_ppf(uniforms)
    simulated_returns = z_scores * garch_vols

    position_array = np.array([positions.get(c, 0.0) for c in commodities])
    if n_assets == 1:
        pnl = simulated_returns.flatten() * position_array[0]
    else:
        pnl = simulated_returns @ position_array

    var_95 = float(-np.quantile(pnl, 1 - 0.95))
    var_99 = float(-np.quantile(pnl, 1 - 0.99))
    es_975 = float(-np.mean(pnl[pnl <= np.quantile(pnl, 1 - 0.975)]))

    # Euler component VaR via finite-difference marginal contributions
    component_var: dict[str, float] = {}
    for i, comm in enumerate(commodities):
        pos_i = position_array[i]
        h = max(abs(pos_i) * 0.0001, 1e-4)
        pos_up = position_array.copy()
        pos_up[i] += h
        pos_down = position_array.copy()
        pos_down[i] -= h
        pnl_up = simulated_returns @ pos_up if n_assets > 1 else simulated_returns.flatten() * pos_up[0]
        pnl_down = simulated_returns @ pos_down if n_assets > 1 else simulated_returns.flatten() * pos_down[0]
        var_up = float(-np.quantile(pnl_up, 1 - 0.95))
        var_down = float(-np.quantile(pnl_down, 1 - 0.95))
        marginal_var = (var_up - var_down) / (2.0 * h)
        component_var[comm] = float(pos_i * marginal_var)

    return PortfolioVaR(
        var_95=var_95, var_99=var_99, es_975=es_975,
        component_var=component_var, pnl_simulations=pnl,
    )


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    return scipy_stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))


def kupiec_test(breaches: int, total: int, confidence: float) -> dict:
    """Kupiec POF (proportion of failures) backtest.

    H0: observed breach rate = expected breach rate (1 - confidence).
    Returns LR statistic and p-value.
    """
    if total == 0:
        return {"lr_stat": 0.0, "p_value": 1.0, "breaches": 0, "total": 0}
    expected_rate = 1.0 - confidence
    observed_rate = breaches / total
    # Avoid log(0)
    if breaches == 0:
        lr_stat = 2.0 * total * np.log(1.0 / (1.0 - expected_rate))
    elif breaches == total:
        lr_stat = 2.0 * total * np.log(1.0 / expected_rate)
    else:
        lr_stat = 2.0 * (
            breaches * np.log(observed_rate / expected_rate)
            + (total - breaches) * np.log((1 - observed_rate) / (1 - expected_rate))
        )
    lr_stat = float(max(0.0, lr_stat))
    p_value = float(1.0 - scipy_stats.chi2.cdf(lr_stat, 1))
    return {
        "lr_stat": lr_stat,
        "p_value": p_value,
        "breaches": breaches,
        "total": total,
    }


def compute_rolling_var(
    returns: pd.DataFrame,
    positions: dict[str, float],
    window: int,
    copula_fit_fn,
) -> pd.DataFrame:
    """Rolling VaR over a fixed-size window.

    Returns DataFrame with columns: date, var_95, var_99, realized_pnl.
    """

    commodities = list(returns.columns)
    position_array = np.array([positions.get(c, 0.0) for c in commodities])
    results: list[dict] = []

    for end in range(window, len(returns) + 1):
        start = end - window
        window_rets = returns.iloc[start:end]
        date = returns.index[end - 1]

        copula = copula_fit_fn(window_rets)
        pv = compute_portfolio_var(window_rets, positions, copula)

        if end < len(returns):
            realized = float(returns.iloc[end].values @ position_array)
        else:
            realized = float(returns.iloc[end - 1].values @ position_array)

        results.append({
            "date": date,
            "var_95": pv.var_95,
            "var_99": pv.var_99,
            "realized_pnl": realized,
        })

    return pd.DataFrame(results)
