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
    fuel_cost = coal / efficiency
    carbon_cost = carbon * emission_factor
    cds = power - fuel_cost - carbon_cost

    return DarkSpreadResult(
        cds=cds, fuel_cost=fuel_cost, carbon_cost=carbon_cost,
    )
