"""3-2-1 Crack Spread — crude-to-products refinery margin."""

import numpy as np


def compute_321_crack(
    rbob: np.ndarray,
    gasoil: np.ndarray,
    brent: np.ndarray,
) -> np.ndarray:
    return (2.0 * rbob + 1.0 * gasoil - 3.0 * brent) / 3.0
