"""Stress scenario definitions and P&L impact calculation."""

from dataclasses import dataclass
import numpy as np
from energy_cross_commodity.risk.copula import CopulaFit


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


def apply_correlation_override(
    base_corr: np.ndarray,
    commodities: list[str],
    override_type: str,
) -> np.ndarray:
    """Return modified correlation matrix with scenario override applied.

    - "all_to_one": sets all off-diagonals to 0.90.
    - "gas_power_spike": sets TTF-DE_POWER and TTF-EUA correlations to 0.85.

    Args:
        base_corr: Fitted copula correlation matrix.
        commodities: Column order of ``base_corr``.
        override_type: Which stressed regime to impose.

    Returns:
        A copy of the matrix with the stressed entries substituted.
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
        # Missing legs must not pass silently: a matrix returned unstressed but
        # labelled "stressed" reports a correlation shock that never happened.
        missing = [c for c in ("TTF", "DE_POWER", "EUA") if c not in commodities]
        if missing:
            raise ValueError(
                f"gas_power_spike needs {missing} in the book; got {commodities}"
            )
        ttf_idx = commodities.index("TTF")
        power_idx = commodities.index("DE_POWER")
        eua_idx = commodities.index("EUA")
        result[ttf_idx, power_idx] = result[power_idx, ttf_idx] = 0.85
        result[ttf_idx, eua_idx] = result[eua_idx, ttf_idx] = 0.85
    else:
        raise ValueError(
            f"Unknown override_type {override_type!r}; "
            "expected 'all_to_one' or 'gas_power_spike'"
        )

    return result


def stressed_copula(
    copula: CopulaFit,
    commodities: list[str],
    override_type: str,
) -> CopulaFit:
    """Rebuild a copula under a scenario's stressed correlation regime.

    Deterministic scenario P&L answers "what if these prices move by this
    much". It cannot answer "how much could we lose if the book's
    diversification stops working", because a fixed set of shocks has no
    distribution. Re-estimating VaR with the stressed correlation
    substituted for the fitted one does answer that, and it is where a
    correlation override belongs.

    Args:
        copula: The fitted copula.
        commodities: Column order matching the copula correlation matrix.
        override_type: Stressed regime, see ``apply_correlation_override``.

    Returns:
        A copula with the stressed correlation and the fitted degrees of
        freedom. Tail dependence is recomputed by the caller if needed.
    """
    correlation = apply_correlation_override(copula.correlation, commodities, override_type)
    return CopulaFit(
        correlation=correlation,
        df=copula.df,
        tail_dep=np.zeros_like(correlation),
    )


def run_scenario(
    positions: dict[str, float],
    scenario: ScenarioDefinition,
) -> ScenarioResult:
    """Full-revaluation P&L for a deterministic stress scenario.

    A scenario states where each price goes, so the revaluation is
    deterministic: for a book of linear positions the P&L of each leg is
    its signed exposure times the shock. Nothing here is simulated, and
    the correlation between the factors plays no part — the scenario has
    already fixed every price jointly, which is the whole point of
    specifying one. Correlation enters the framework through VaR, where
    the moves are unknown, not through stress testing, where they are
    assumed.

    Args:
        positions: Signed EUR exposures keyed by price factor. Use
            ``expand_spread_positions`` first so spread legs are present.
        scenario: The scenario to apply. Factors it does not shock
            contribute zero.

    Returns:
        Per-position and total P&L in EUR.
    """
    pnl_by_position = {
        key: float(notional) * scenario.price_shocks.get(key, 0.0)
        for key, notional in positions.items()
    }
    return ScenarioResult(
        pnl_by_position=pnl_by_position,
        total_pnl=float(sum(pnl_by_position.values())),
    )
