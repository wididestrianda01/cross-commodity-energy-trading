"""Multi-commodity VaR/ES engine using Filtered Historical Simulation."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from energy_cross_commodity.risk.copula import CopulaFit, simulate_t_copula
from energy_cross_commodity.risk.garch import GARCHResult, fit_univariate_garch


@dataclass
class PortfolioVaR:
    var_95: float
    var_99: float
    es_975: float
    component_var: dict[str, float]
    pnl_simulations: np.ndarray
    sigma_forecast: dict[str, float] = field(default_factory=dict)


def compute_portfolio_var(
    returns: pd.DataFrame,
    positions: dict[str, float],
    copula: CopulaFit | None,
    confidence: list[float] | None = None,
    n_simulations: int = 10000,
    garch_fits: dict[str, GARCHResult] | None = None,
    es_confidence: float = 0.975,
    seed: int | None = None,
) -> PortfolioVaR:
    """Compute VaR/ES/component VaR by Filtered Historical Simulation.

    Filtered Historical Simulation (Barone-Adesi, Giannopoulos & Vosper
    1999) combines a conditional volatility model with the empirical
    distribution of the standardised residuals:

    1. Fit a univariate GARCH per commodity and take the one-step-ahead
       volatility forecast sigma_{T+1} and the standardised residuals.
    2. Draw rank-correlated uniforms from the fitted t-copula, so the
       joint tail behaviour is preserved across commodities.
    3. Map each uniform through the *empirical* quantile function of that
       commodity's residuals, rather than a normal inverse CDF.
    4. Rescale by sigma_{T+1} and add the conditional mean to obtain
       simulated one-day returns, then aggregate to portfolio P&L.

    Steps 1 and 3 are what make the result conditional and fat-tailed. A
    normal inverse CDF applied to an unconditional sample standard
    deviation — the textbook shortcut — discards both the current
    volatility state and the excess kurtosis of the residuals, and
    understates the tail on exactly the days that matter.

    Args:
        returns: Historical log returns (one column per asset).
        positions: Notional exposures keyed by commodity.
        copula: Fitted t-copula for the dependence structure. May be
            ``None`` only for a single-asset book, where there is no
            dependence structure to model; passing ``None`` with several
            assets would silently assert independence.
        confidence: VaR confidence levels (default [0.95, 0.99]).
        n_simulations: Number of Monte Carlo draws.
        garch_fits: Pre-computed GARCH fits keyed by column name. Fitted
            on demand when omitted.
        es_confidence: Confidence level for expected shortfall.
        seed: Seed for the copula draw, for reproducible runs.

    Returns:
        PortfolioVaR with var_95, var_99, es_975, component VaR, the
        simulated P&L vector, and the per-commodity volatility forecast.

    Raises:
        ValueError: If ``copula`` is None for a multi-asset book.
    """
    if confidence is None:
        confidence = [0.95, 0.99]

    commodities = list(returns.columns)
    if garch_fits is None:
        garch_fits = {c: fit_univariate_garch(returns[c]) for c in commodities}

    if copula is not None:
        uniforms = simulate_t_copula(copula, n=n_simulations, seed=seed)
    elif len(commodities) == 1:
        uniforms = np.random.default_rng(seed).random((n_simulations, 1))
    else:
        raise ValueError(
            f"A copula is required for {len(commodities)} assets; passing None "
            "would impose independence rather than estimate dependence."
        )

    # Step 3-4: empirical residual quantiles, rescaled to the forecast vol.
    simulated_returns = np.zeros((n_simulations, len(commodities)))
    sigma_forecast: dict[str, float] = {}
    for i, comm in enumerate(commodities):
        fit = garch_fits[comm]
        residuals = np.asarray(fit.std_residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        shocks = np.quantile(residuals, uniforms[:, i])
        simulated_returns[:, i] = fit.mu + fit.sigma_forecast * shocks
        sigma_forecast[comm] = float(fit.sigma_forecast)

    position_array = np.array([positions.get(c, 0.0) for c in commodities])
    pnl = simulated_returns @ position_array

    var_95 = float(-np.quantile(pnl, 1 - 0.95))
    var_99 = float(-np.quantile(pnl, 1 - 0.99))
    es_975 = float(-np.mean(pnl[pnl <= np.quantile(pnl, 1 - es_confidence)]))

    component_var = _component_var(
        simulated_returns, position_array, commodities, pnl, var_95
    )

    return PortfolioVaR(
        var_95=var_95,
        var_99=var_99,
        es_975=es_975,
        component_var=component_var,
        pnl_simulations=pnl,
        sigma_forecast=sigma_forecast,
    )


def _component_var(
    simulated_returns: np.ndarray,
    position_array: np.ndarray,
    commodities: list[str],
    pnl: np.ndarray,
    var: float,
) -> dict[str, float]:
    """Euler-allocate VaR via a kernel-smoothed conditional expectation.

    VaR is positively homogeneous of degree one in the position vector,
    so Euler's theorem gives an exact additive decomposition with
    marginal contribution ``d VaR / d w_i = -E[r_i | r_p = -VaR]``.

    That conditional expectation is estimated with a Gaussian kernel
    around the VaR quantile (Hallerbach 2003) rather than by bumping each
    position and re-taking the quantile. Finite differences on an order
    statistic are unreliable: a small bump usually selects the *same*
    simulated scenario, so the derivative comes back as zero or as pure
    Monte Carlo noise. The kernel estimator uses every scenario near the
    quantile and therefore sums back to total VaR by construction.
    """
    spread = float(np.std(pnl))
    if spread <= 0.0:
        return {c: 0.0 for c in commodities}

    # Silverman-type bandwidth; widened slightly because the target is a
    # conditional mean in the tail, where scenarios are sparse.
    bandwidth = 1.06 * spread * len(pnl) ** (-0.2)
    weights = np.exp(-0.5 * ((pnl + var) / bandwidth) ** 2)
    total_weight = weights.sum()
    if total_weight <= 0.0:
        return {c: 0.0 for c in commodities}

    weights = weights / total_weight
    conditional_returns = simulated_returns.T @ weights

    return {
        comm: float(-position_array[i] * conditional_returns[i])
        for i, comm in enumerate(commodities)
    }


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
    n_simulations: int = 2000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Rolling out-of-sample VaR for backtesting.

    Each row pairs a VaR estimated from the trailing ``window`` of
    returns with the P&L actually realised on the *following* day. The
    series therefore stops one day before the end of the sample: a final
    row reusing the last in-sample return as its own realised P&L would
    compare a forecast against data it was fitted on and bias the breach
    count.

    Args:
        returns: Historical log returns (one column per asset).
        positions: Notional exposures keyed by commodity.
        window: Estimation window length in trading days.
        copula_fit_fn: Callable mapping a window of returns to a CopulaFit.
        n_simulations: Draws per window. Lower than the single-shot
            default because this runs once per trading day.
        seed: Base seed; each window is offset from it so the windows are
            reproducible without sharing one common random draw.

    Returns:
        DataFrame with columns date, var_95, var_99, realized_pnl.
    """
    commodities = list(returns.columns)
    position_array = np.array([positions.get(c, 0.0) for c in commodities])
    results: list[dict] = []

    for end in range(window, len(returns)):
        window_rets = returns.iloc[end - window : end]
        copula = copula_fit_fn(window_rets)
        portfolio_var = compute_portfolio_var(
            window_rets,
            positions,
            copula,
            n_simulations=n_simulations,
            seed=None if seed is None else seed + end,
        )

        results.append({
            "date": returns.index[end - 1],
            "var_95": portfolio_var.var_95,
            "var_99": portfolio_var.var_99,
            "realized_pnl": float(returns.iloc[end].to_numpy() @ position_array),
        })

    return pd.DataFrame(results)
