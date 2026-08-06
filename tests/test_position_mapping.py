"""Tests for mapping traded positions and prices onto risk-model inputs."""

import numpy as np
import pandas as pd
import pytest

from energy_cross_commodity.risk.portfolio import expand_spread_positions
from energy_cross_commodity.risk.returns import compute_log_returns

CRACK_LEGS = {
    "CRACK_3_2_1": {"BRENT": -1.0, "RBOB": 2 / 3, "GASOIL": 1 / 3},
    "SPARK_SPREAD": {"DE_POWER": 1.0, "TTF": -1.0},
}


def test_outright_positions_pass_through():
    exposures = expand_spread_positions({"BRENT": 10_000_000.0}, CRACK_LEGS)
    assert exposures == {"BRENT": 10_000_000.0}


def test_spread_is_replaced_by_its_legs():
    """A short crack is long crude against short products."""
    exposures = expand_spread_positions({"CRACK_3_2_1": -4_600_000.0}, CRACK_LEGS)
    assert exposures["BRENT"] == pytest.approx(4_600_000.0)
    assert exposures["RBOB"] == pytest.approx(-3_066_666.7, rel=1e-6)
    assert exposures["GASOIL"] == pytest.approx(-1_533_333.3, rel=1e-6)
    assert "CRACK_3_2_1" not in exposures


def test_spread_legs_net_against_outrights():
    """The crude leg of a short crack partly hedges an outright long."""
    exposures = expand_spread_positions(
        {"BRENT": 10_000_000.0, "CRACK_3_2_1": -4_600_000.0}, CRACK_LEGS
    )
    assert exposures["BRENT"] == pytest.approx(14_600_000.0)


def test_spreads_introduce_shorts():
    """Dropping spreads instead of expanding them would leave a long-only book."""
    exposures = expand_spread_positions(
        {"CRACK_3_2_1": -4_600_000.0, "SPARK_SPREAD": -4_000_000.0}, CRACK_LEGS
    )
    assert any(v < 0 for v in exposures.values())
    assert exposures["DE_POWER"] == pytest.approx(-4_000_000.0)
    assert exposures["TTF"] == pytest.approx(4_000_000.0)


def test_log_returns_match_plain_formula_without_displacement():
    prices = pd.DataFrame({"BRENT": [80.0, 82.0, 81.0]})
    returns = compute_log_returns(prices)
    assert returns["BRENT"].iloc[0] == pytest.approx(np.log(82.0 / 80.0))
    assert len(returns) == 2


def test_negative_power_price_survives_displacement():
    """The oversupply day must stay in the sample, not be dropped as NaN."""
    prices = pd.DataFrame(
        {"DE_POWER": [90.0, -20.0, 70.0], "BRENT": [80.0, 81.0, 82.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    returns = compute_log_returns(prices, {"DE_POWER": 100.0})
    assert len(returns) == 2
    assert np.isfinite(returns.to_numpy()).all()
    assert returns["DE_POWER"].iloc[0] == pytest.approx(np.log(80.0 / 190.0))
    # The other columns keep their ordinary log returns.
    assert returns["BRENT"].iloc[0] == pytest.approx(np.log(81.0 / 80.0))


def test_insufficient_displacement_raises():
    prices = pd.DataFrame({"DE_POWER": [90.0, -120.0, 70.0]})
    with pytest.raises(ValueError, match="DE_POWER"):
        compute_log_returns(prices, {"DE_POWER": 100.0})
