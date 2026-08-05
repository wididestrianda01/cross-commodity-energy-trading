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
    """(2*100 + 200 - 3*80) / 3 = (200+200-240)/3 = 53.33"""
    rbob = np.array([100.0])
    gasoil = np.array([200.0])
    brent = np.array([80.0])
    result = compute_321_crack(rbob, gasoil, brent)
    assert abs(result[0] - 53.33) < 0.1


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
