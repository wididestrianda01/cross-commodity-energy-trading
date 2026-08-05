"""Stress scenario definitions and P&L impact calculation."""

from dataclasses import dataclass, field
import numpy as np


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
        correlation_override=None,
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


def run_scenario(
    positions: dict[str, float],
    scenario: ScenarioDefinition,
    current_prices: dict[str, float],
    copula=None,
) -> ScenarioResult:
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
