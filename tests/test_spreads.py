"""Tests for spread economics engine."""

import numpy as np
from energy_cross_commodity.spreads.spark_spread import compute_spark_spread, compute_fuel_switch
from energy_cross_commodity.spreads.dark_spread import compute_dark_spread
from energy_cross_commodity.spreads.crack_spread import compute_321_crack


def test_spark_spread_hand_calculation():
    """CSS = 100 - 30/0.5 - 80*0.4 = 8"""
    power = np.array([100.0])
    gas = np.array([30.0])
    carbon = np.array([80.0])
    result = compute_spark_spread(power, gas, carbon, efficiency=0.5, emission_factor=0.4)
    assert abs(result.css[0] - 8.0) < 0.01


def test_zero_carbon_gives_unclean_spread():
    power = np.array([100.0])
    gas = np.array([30.0])
    carbon = np.array([0.0])
    result = compute_spark_spread(power, gas, carbon, efficiency=0.5, emission_factor=0.4)
    assert abs(result.css[0] - 40.0) < 0.01
    assert abs(result.uss[0] - 40.0) < 0.01


def test_spark_spread_regime_classification():
    power = np.array([100, 100, 100, 100])
    gas = np.array([10, 30, 50, 70])
    carbon = np.array([0, 0, 0, 0])
    result = compute_spark_spread(power, gas, carbon, efficiency=0.5, emission_factor=0.4)
    assert result.regime[0] == "RUN"
    assert result.regime[1] == "RUN"
    assert result.regime[2] == "MARGINAL"  # CSS = 100-100 = 0
    assert result.regime[3] == "IDLE"      # CSS = 100-140 = -40


def test_dark_spread_hand_calculation():
    """CDS = 100 - 30/0.4 - 80*0.8 = 100 - 75 - 64 = -39"""
    power = np.array([100.0])
    coal = np.array([30.0])
    carbon = np.array([80.0])
    result = compute_dark_spread(power, coal, carbon, efficiency=0.4, emission_factor=0.8)
    assert abs(result.cds[0] - (-39.0)) < 0.01


def test_321_crack_spread():
    """Legs must be converted to USD/bbl before the 2:1:3 ratio is applied.

    RBOB 2.10 USD/gal * 42        = 88.2 USD/bbl
    ULSD (HO=F) 2.30 USD/gal * 42 = 96.6 USD/bbl
    (2*88.2 + 96.6 - 3*82) / 3    = 9.0 USD/bbl
    """
    rbob = np.array([2.10])
    gasoil = np.array([2.30])
    brent = np.array([82.0])
    result = compute_321_crack(rbob, gasoil, brent)
    assert abs(result[0] - 9.0) < 0.01


def test_321_crack_rejects_raw_unit_arithmetic():
    """A crack spread built from unconverted quotes is off by orders of magnitude.

    Guards the specific regression of summing raw USD/gal quotes as if they
    were already in USD/bbl, which yields a plausible-looking but meaningless
    number rather than an obvious error.
    """
    rbob, gasoil, brent = np.array([2.10]), np.array([2.30]), np.array([82.0])
    naive = (2.0 * rbob + gasoil - 3.0 * brent) / 3.0
    assert abs(compute_321_crack(rbob, gasoil, brent)[0] - naive[0]) > 50.0


def test_fuel_switch_gas_favored():
    css = np.array([10.0])
    cds = np.array([-5.0])
    result = compute_fuel_switch(css, cds)
    assert result.signal[0] == 15.0
    assert result.regime[0] == "GAS_FAVORED"


def test_fuel_switch_switching_zone():
    css = np.array([2.0])
    cds = np.array([0.0])
    result = compute_fuel_switch(css, cds)
    assert result.regime[0] == "SWITCHING_ZONE"


def test_fuel_switch_coal_favored():
    css = np.array([-10.0])
    cds = np.array([5.0])
    result = compute_fuel_switch(css, cds)
    assert result.regime[0] == "COAL_FAVORED"
