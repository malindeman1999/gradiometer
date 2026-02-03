"""
Compare vectorized Gaunt coefficients against the scalar _gaunt_coeff.

Runs over all valid (L,M,l0,m0,l,m) combinations up to lmax=4 by default.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa_model.solvers import _gaunt_coeff

try:
from gaunt.gaunt_vectorized import gaunt_coeff_vectorized
except ImportError as exc:
    print(f"Skipping test: {exc}")
    sys.exit(0)


def compare(lmax: int = 4) -> None:
    combos = []
    for L in range(lmax + 1):
        for M in range(-L, L + 1):
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        for m in range(-l, l + 1):
                            combos.append((L, M, l0, m0, l, m))

    arr = np.array(combos, dtype=int)
    L, M, l0, m0, l, m = arr.T

    g_vec = gaunt_coeff_vectorized(L, M, l0, m0, l, m)
    g_scalar = np.array([_gaunt_coeff(*t) if (-t[1] + t[3] + t[5]) == 0 else 0.0 for t in combos])

    max_abs = np.max(np.abs(g_vec - g_scalar))
    max_rel = max_abs / max(np.max(np.abs(g_scalar)), 1e-30)

    print(f"Compared {len(combos)} combinations up to lmax={lmax}")
    print(f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")


if __name__ == "__main__":
    compare()
