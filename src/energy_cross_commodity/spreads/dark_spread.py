"""Clean Dark Spread — coal-to-power profitability."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DarkSpreadResult:
    cds: np.ndarray
    fuel_cost: np.ndarray
    carbon_cost: np.ndarray


def compute_dark_spread(
    power: np.ndarray,
    coal: np.ndarray,
    carbon: np.ndarray,
    efficiency: float = 0.38,
    emission_factor: float = 0.90,
) -> DarkSpreadResult:
    """Compute the Clean Dark Spread (CDS): power revenue minus coal and carbon costs.

    CDS = Power - Coal/efficiency - Carbon * emission_factor.

    Args:
        power: Baseload power price array (EUR/MWh).
        coal: API2 coal price array (EUR/MWh equivalent).
        carbon: EUA carbon price array (EUR/tCO2).
        efficiency: Coal plant thermal efficiency (default 0.38).
        emission_factor: tCO2 emitted per MWh of coal generation (default 0.90).

    Returns:
        DarkSpreadResult with cds, fuel_cost, and carbon_cost arrays.
    """
    fuel_cost = coal / efficiency
    carbon_cost = carbon * emission_factor
    cds = power - fuel_cost - carbon_cost

    return DarkSpreadResult(
        cds=cds, fuel_cost=fuel_cost, carbon_cost=carbon_cost,
    )
