"""Clean Spark Spread — gas-to-power profitability."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SparkSpreadResult:
    css: np.ndarray
    uss: np.ndarray
    regime: np.ndarray
    carbon_cost: np.ndarray
    fuel_cost: np.ndarray


@dataclass
class FuelSwitchResult:
    signal: np.ndarray
    regime: np.ndarray
    spark_spread: np.ndarray
    dark_spread: np.ndarray


def compute_spark_spread(
    power: np.ndarray,
    gas: np.ndarray,
    carbon: np.ndarray,
    efficiency: float = 0.55,
    emission_factor: float = 0.37,
) -> SparkSpreadResult:
    fuel_cost = gas / efficiency
    carbon_cost = carbon * emission_factor
    css = power - fuel_cost - carbon_cost
    uss = power - fuel_cost

    regime = np.full_like(css, "RUN", dtype=object)
    regime[(css > -20) & (css <= 0)] = "MARGINAL"
    regime[css <= -20] = "IDLE"

    return SparkSpreadResult(
        css=css, uss=uss, regime=regime,
        carbon_cost=carbon_cost, fuel_cost=fuel_cost,
    )


def compute_fuel_switch(
    css: np.ndarray,
    cds: np.ndarray,
) -> FuelSwitchResult:
    signal = css - cds
    regime = np.full_like(signal, "GAS_FAVORED", dtype=object)
    regime[signal < -5.0] = "COAL_FAVORED"
    regime[(signal >= -5.0) & (signal <= 5.0)] = "SWITCHING_ZONE"

    return FuelSwitchResult(
        signal=signal, regime=regime,
        spark_spread=css, dark_spread=cds,
    )
