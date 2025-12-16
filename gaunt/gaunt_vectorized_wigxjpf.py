"""
Vectorized Gaunt coefficients using pywigxjpf (WIGXJPF backend).

Relies on WIGXJPF's C implementation of Wigner 3j. Inputs are broadcastable
NumPy integer arrays; outputs float64 matching the scalar _gaunt_coeff.
"""
from __future__ import annotations

import numpy as np

try:
    from pywigxjpf import (
        wig_table_init,
        wig_table_free,
        wig_temp_init,
        wig_temp_free,
        wig3jj,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("pywigxjpf is required for gaunt_coeff_vectorized_wigxjpf") from exc


def _init_tables(max_l: int) -> None:
    max_two_j = 2 * max_l
    # wigner_type=3 allocates for 3j symbols
    wig_table_init(max_two_j, 3)
    wig_temp_init(max_two_j)


def _cleanup_tables() -> None:
    try:
        wig_temp_free()
    finally:
        wig_table_free()


def gaunt_coeff_vectorized_wigxjpf(
    L: np.ndarray,
    M: np.ndarray,
    l0: np.ndarray,
    m0: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Gaunt coefficient <Y_{L,M} | Y_{l0,m0} Y_{l,m}> using pywigxjpf.

    Inputs are broadcastable NumPy integer arrays; returns float64 array.
    """
    L = np.asarray(L, dtype=int)
    M = np.asarray(M, dtype=int)
    l0 = np.asarray(l0, dtype=int)
    m0 = np.asarray(m0, dtype=int)
    l = np.asarray(l, dtype=int)
    m = np.asarray(m, dtype=int)

    # Broadcast to common shape
    L, M, l0, m0, l, m = np.broadcast_arrays(L, M, l0, m0, l, m)
    out = np.zeros(L.shape, dtype=np.float64)

    max_l = int(np.max([L.max(initial=0), l0.max(initial=0), l.max(initial=0)]))
    _init_tables(max_l)
    try:
        it = np.nditer([L, M, l0, m0, l, m, out], op_flags=[['readonly']]*6 + [['writeonly']])
        for L_i, M_i, l0_i, m0_i, l_i, m_i, out_i in it:
            Li = int(L_i)
            Mi = int(M_i)
            l0i = int(l0_i)
            m0i = int(m0_i)
            li = int(l_i)
            mi = int(m_i)
            if (-Mi + m0i + mi) != 0:
                out_i[...] = 0.0
                continue
            pref = np.sqrt((2 * Li + 1) * (2 * l0i + 1) * (2 * li + 1) / (4 * np.pi))
            w1 = wig3jj(2 * Li, 2 * l0i, 2 * li, 0, 0, 0)
            w2 = wig3jj(2 * Li, 2 * l0i, 2 * li, -2 * Mi, 2 * m0i, 2 * mi)
            out_i[...] = ((-1.0) ** Mi) * pref * w1 * w2
    finally:
        _cleanup_tables()
    return out
