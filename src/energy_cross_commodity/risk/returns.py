"""Return construction for price series that may cross zero."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(
    prices: pd.DataFrame,
    displacements: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Log returns, displaced where a price can go non-positive.

    Day-ahead power clears below zero whenever inflexible generation
    exceeds demand, so ``ln(P_t / P_{t-1})`` is undefined on those days.
    Taking it anyway yields NaN, and a panel-wide ``dropna`` then deletes
    the whole cross-section for that date — quietly discarding exactly the
    oversupply days a power book cares about.

    The fix is a displaced log return, ``ln((P_t + k) / (P_{t-1} + k))``
    with a fixed k large enough to keep both arguments positive. This is
    the same displacement idea used to price options on rates and spreads
    that admit negative values; it converges to the ordinary log return
    when P is large relative to k, and stays finite when P is not.

    Args:
        prices: Wide price panel, one column per series.
        displacements: EUR displacement per column. Columns omitted here
            are treated as strictly positive and use plain log returns.

    Returns:
        Log returns indexed by date, with the first observation dropped
        and rows carrying missing prices removed.

    Raises:
        ValueError: If a column is non-positive after displacement, which
            means the configured k is too small for the sample.
    """
    displacements = displacements or {}
    shifted = prices.copy()
    for column, displacement in displacements.items():
        if column in shifted.columns:
            shifted[column] = shifted[column] + float(displacement)

    non_positive = [c for c in shifted.columns if (shifted[c].dropna() <= 0).any()]
    if non_positive:
        raise ValueError(
            f"Non-positive prices remain in {non_positive} after displacement; "
            "increase risk.price_displacement_eur for those series."
        )

    return np.log(shifted / shifted.shift(1)).dropna()
