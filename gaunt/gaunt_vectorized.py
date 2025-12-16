"""
Vectorized Gaunt coefficient helper using SciPy's batched wigner_3j.

Provides gaunt_coeff_vectorized(L, M, l0, m0, l, m) which accepts NumPy
array-like inputs and returns a broadcasted NumPy array of Gaunt
coefficients matching the scalar `_gaunt_coeff` in europa.solvers.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.special import wigner_3j  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy.special.wigner_3j is required for gaunt_coeff_vectorized") from exc


def gaunt_coeff_vectorized(
    L: np.ndarray,
    M: np.ndarray,
    l0: np.ndarray,
    m0: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Gaunt coefficient <Y_{L,M} | Y_{l0,m0} Y_{l,m}> using SciPy wigner_3j.

    Inputs are broadcastable NumPy arrays of integer quantum numbers.
    Returns float64 NumPy array with the broadcasted shape.
    """
    L = np.asarray(L, dtype=int)
    M = np.asarray(M, dtype=int)
    l0 = np.asarray(l0, dtype=int)
    m0 = np.asarray(m0, dtype=int)
    l = np.asarray(l, dtype=int)
    m = np.asarray(m, dtype=int)

    # Selection rule mask
    sel = (-M + m0 + m) == 0

    pref = np.sqrt((2 * L + 1) * (2 * l0 + 1) * (2 * l + 1) / (4 * np.pi))
    w1 = wigner_3j(L, l0, l, 0, 0, 0)
    w2 = wigner_3j(L, l0, l, -M, m0, m)

    out = ((-1.0) ** M) * pref * w1 * w2
    out = np.where(sel, out, 0.0)
    return np.asarray(out, dtype=np.float64)
