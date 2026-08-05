"""Stress scenario definitions and P&L impact calculation."""

from dataclasses import dataclass
import numpy as np
from scipy import stats as scipy_stats
from energy_cross_commodity.risk.copula import CopulaFit, simulate_t_copula


@dataclass
class ScenarioDefinition:
    name: str
    description: str
    price_shocks: dict[str, float]
    correlation_override: str | None = None


@dataclass
class ScenarioResult:
    pnl_by_position: dict[str, float]
    total_pnl: float


SCENARIOS: dict[str, ScenarioDefinition] = {
    "gas_crisis": ScenarioDefinition(
        name="Nord Stream Zero",
        description="Russian gas supply cut to zero. TTF spikes 300%, power follows, carbon rises.",
        price_shocks={
            "TTF": 3.00, "DE_POWER": 2.00, "EUA": 0.50, "BRENT": 0.30,
            "RBOB": 0.20, "GASOIL": 0.25, "API2": 1.00,
        },
        correlation_override="gas_power_spike",
    ),
    "recession": ScenarioDefinition(
        name="Global Recession",
        description="Demand destruction across all commodities. Risk-off convergence.",
        price_shocks={
            "BRENT": -0.40, "TTF": -0.30, "DE_POWER": -0.25,
            "EUA": -0.20, "RBOB": -0.45, "GASOIL": -0.40, "API2": -0.35,
        },
        correlation_override="all_to_one",
    ),
    "energy_transition": ScenarioDefinition(
        name="Energy Transition Accelerates",
        description="Carbon at 150 EUR/t. Coal destroyed. Renewables cannibalize power.",
        price_shocks={
            "EUA": 2.00, "API2": -0.40, "BRENT": -0.30,
            "DE_POWER": -0.10, "TTF": -0.20, "RBOB": -0.35, "GASOIL": -0.35,
        },
        correlation_override=None,
    ),
}


def _commodity_universe() -> list[str]:
    return ["BRENT", "TTF", "DE_POWER", "EUA", "RBOB", "GASOIL", "API2"]


def apply_correlation_override(
    base_corr: np.ndarray,
    commodities: list[str],
    override_type: str,
) -> np.ndarray:
    """Return modified correlation matrix with scenario override applied.

    - "all_to_one": sets all off-diagonals to 0.90.
    - "gas_power_spike": sets TTF-DE_POWER and TTF-EUA correlations to 0.85.
    """
    n = len(commodities)
    if base_corr.shape != (n, n):
        raise ValueError(f"base_corr shape {base_corr.shape} != ({n},{n})")
    result = base_corr.copy()

    if override_type == "all_to_one":
        for i in range(n):
            for j in range(i + 1, n):
                result[i, j] = result[j, i] = 0.90
    elif override_type == "gas_power_spike":
        try:
            ttf_idx = commodities.index("TTF")
            power_idx = commodities.index("DE_POWER")
            eua_idx = commodities.index("EUA")
            result[ttf_idx, power_idx] = result[power_idx, ttf_idx] = 0.85
            result[ttf_idx, eua_idx] = result[eua_idx, ttf_idx] = 0.85
        except ValueError:
            pass

    return result


def run_scenario(
    positions: dict[str, float],
    scenario: ScenarioDefinition,
    current_prices: dict[str, float],
    copula: CopulaFit | None = None,
    commodities: list[str] | None = None,
) -> ScenarioResult:
    """Calculate P&L impact of a stress scenario, optionally copula-aware."""
    # Copula-aware path: simulate correlated shocks
    if copula is not None and scenario.correlation_override is not None:
        if commodities is None:
            commodities = _commodity_universe()
        overridden_corr = apply_correlation_override(
            copula.correlation, commodities, scenario.correlation_override
        )
        overridden_copula = CopulaFit(
            correlation=overridden_corr,
            df=copula.df,
            tail_dep=np.zeros(overridden_corr.shape),
        )
        uniforms = simulate_t_copula(overridden_copula, n=10000)
        z = scipy_stats.norm.ppf(np.clip(uniforms, 1e-10, 1 - 1e-10))
        pnl_samples = np.zeros(z.shape[0])
        for idx, comm in enumerate(commodities):
            if comm in positions and comm in scenario.price_shocks:
                shock = scenario.price_shocks[comm]
                notional = positions[comm]
                pnl_samples += z[:, idx] * abs(notional) * shock * (1 if notional >= 0 else -1)
        pnl_by_position: dict[str, float] = {}
        for comm in positions:
            pnl_by_position[comm] = 0.0
        total_pnl = float(np.mean(pnl_samples))
        return ScenarioResult(pnl_by_position=pnl_by_position, total_pnl=total_pnl)

    # Simple deterministic path: shock * notional
    pnl_by_position = {}
    for key, notional in positions.items():
        if key in scenario.price_shocks:
            shock = scenario.price_shocks[key]
            pnl = abs(notional) * shock
            if notional < 0:
                pnl = -pnl
            pnl_by_position[key] = pnl
        else:
            pnl_by_position[key] = 0.0

    total_pnl = sum(pnl_by_position.values())
    return ScenarioResult(pnl_by_position=pnl_by_position, total_pnl=total_pnl)
