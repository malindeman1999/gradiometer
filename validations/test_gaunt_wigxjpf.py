"""
Validate pywigxjpf-based Gaunt against scalar _gaunt_coeff and time it.
Skips if pywigxjpf is not available.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa.solvers import _gaunt_coeff

try:
from gaunt.gaunt_vectorized_wigxjpf import gaunt_coeff_vectorized_wigxjpf
except ImportError as exc:
    print(f"Skipping test: {exc}")
    sys.exit(0)


def compare_and_time(lmax: int = 4) -> None:
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

    t0 = time.perf_counter()
    g_vec = gaunt_coeff_vectorized_wigxjpf(L, M, l0, m0, l, m)
    t_vec = time.perf_counter() - t0

    t1 = time.perf_counter()
    g_scalar = np.array([_gaunt_coeff(*t) if (-t[1] + t[3] + t[5]) == 0 else 0.0 for t in combos])
    t_scalar = time.perf_counter() - t1

    max_abs = np.max(np.abs(g_vec - g_scalar))
    max_rel = max_abs / max(np.max(np.abs(g_scalar)), 1e-30)

    print(f"Compared {len(combos)} combinations up to lmax={lmax}")
    print(f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")
    print(f"time_vectorized={t_vec:.4f}s, time_scalar={t_scalar:.4f}s, speedup={t_scalar/t_vec if t_vec>0 else float('inf'):.1f}x")


if __name__ == "__main__":
    compare_and_time()
