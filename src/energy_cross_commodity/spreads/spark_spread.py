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
    """Compute the Clean Spark Spread (CSS): power revenue minus gas and carbon costs.

    CSS = Power - Gas/efficiency - Carbon * emission_factor.
    USS (Unclean Spark Spread) excludes carbon cost.
    Assets are classified into RUN (>0), MARGINAL (-20 to 0), or IDLE (<-20).

    Args:
        power: Baseload power price array (EUR/MWh).
        gas: TTF natural gas price array (EUR/MWh).
        carbon: EUA carbon price array (EUR/tCO2).
        efficiency: CCGT thermal efficiency (default 0.55).
        emission_factor: tCO2 per MWh of gas generation (default 0.37).

    Returns:
        SparkSpreadResult with css, uss, regime, carbon_cost, and fuel_cost.
    """
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
    """Compute the fuel-switching signal: CSS minus CDS.

    When CSS > CDS gas is favoured; when CDS > CSS by more than 5 EUR/MWh
    coal is favoured; the 5 EUR/MWh band is the switching zone.

    Args:
        css: Clean Spark Spread array.
        cds: Clean Dark Spread array.

    Returns:
        FuelSwitchResult with signal, regime, spark_spread, and dark_spread.
    """
    signal = css - cds
    regime = np.full_like(signal, "GAS_FAVORED", dtype=object)
    regime[signal < -5.0] = "COAL_FAVORED"
    regime[(signal >= -5.0) & (signal <= 5.0)] = "SWITCHING_ZONE"

    return FuelSwitchResult(
        signal=signal, regime=regime,
        spark_spread=css, dark_spread=cds,
    )


@dataclass
class BreakEvenCarbonResult:
    """Result of the break-even carbon price calculation.

    The carbon price at which the Clean Spark Spread equals the Clean Dark
    Spread — i.e., the point where gas and coal generation are equally
    profitable on a clean basis.
    """
    value: np.ndarray
    """Break-even carbon price in EUR/tCO2."""


def compute_break_even_carbon(
    gas: np.ndarray,
    coal: np.ndarray,
    spark_eff: float = 0.55,
    spark_ef: float = 0.37,
    dark_eff: float = 0.38,
    dark_ef: float = 0.90,
) -> np.ndarray:
    """Compute the carbon price at which CSS equals CDS.

    Solves CSS = CDS for the carbon price:

    .. math::
        P_{carbon} = \\frac{
            P_{coal}/\\eta_{coal} - P_{gas}/\\eta_{gas}
        }{
            \\varepsilon_{coal} - \\varepsilon_{gas}
        }

    where :math:`\\eta` is thermal efficiency and :math:`\\varepsilon` is
    the emission factor (tCO2 per MWh of fuel input).

    Args:
        gas: TTF natural gas price array (EUR/MWh).
        coal: API2 coal price array (EUR/MWh equivalent).
        spark_eff: CCGT thermal efficiency (default 0.55).
        spark_ef: Gas emission factor (default 0.37 tCO2/MWh).
        dark_eff: Coal plant thermal efficiency (default 0.38).
        dark_ef: Coal emission factor (default 0.90 tCO2/MWh).

    Returns:
        Break-even carbon price array in EUR/tCO2. Negative values indicate
        that coal always wins (given the spread term); very large positive
        values indicate gas always wins.
    """
    gas_fuel_cost = gas / spark_eff
    coal_fuel_cost = coal / dark_eff
    delta_emission = dark_ef - spark_ef
    delta_fuel = gas_fuel_cost - coal_fuel_cost

    return np.divide(delta_fuel, delta_emission)
