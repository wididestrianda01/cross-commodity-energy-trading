"""Mapping traded positions onto the price factors the risk model uses."""

from __future__ import annotations


def expand_spread_positions(
    positions: dict[str, float],
    spread_legs: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Restate spread positions as signed exposures to their underlyings.

    A desk trades the crack and the spark as single instruments, but a
    spread level is a price *difference*: it is routinely negative and has
    no log return, so it cannot be a risk factor. Standard practice is to
    map each position onto the factors that actually drive it — the delta
    of a spread is the sum of the deltas of its legs — and to run the VaR
    engine on those factors.

    Doing so also restores the offsets that make the book a book: a short
    crack contributes long crude against short products, so its crude leg
    partly hedges an outright long. Dropping the spreads instead (the
    tempting shortcut, since they break ``np.log``) silently deletes every
    short in the portfolio and overstates diversification.

    Args:
        positions: Signed EUR notionals keyed by instrument. Keys absent
            from ``spread_legs`` are outright positions and pass through.
        spread_legs: Leg weights per spread instrument. Each weight
            multiplies the spread notional to give that leg's exposure.

    Returns:
        Signed EUR exposures keyed by price factor, with the contributions
        of outright and spread positions netted together.
    """
    exposures: dict[str, float] = {}
    for instrument, notional in positions.items():
        legs = spread_legs.get(instrument)
        if legs is None:
            exposures[instrument] = exposures.get(instrument, 0.0) + float(notional)
            continue
        for factor, weight in legs.items():
            exposures[factor] = exposures.get(factor, 0.0) + float(notional) * float(weight)
    return exposures
