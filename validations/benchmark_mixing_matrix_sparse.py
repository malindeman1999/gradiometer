"""
Benchmark mixing-matrix build: dense vs sparse accumulation from Gaunt tensor.

- Loads Gaunt tensor from the pywigxjpf cache.
- Builds mixing matrix using dense NumPy loops (baseline).
- Builds mixing matrix using sparse COO accumulation over nonzero Gaunt entries.
- Reports timing and max difference between the two.

Requires SciPy.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gaunt.gaunt_cache_wigxjpf import load_gaunt_tensor_wigxjpf


def lm_index_map(lmax: int):
    mapping = {}
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            mapping[(l, m)] = idx
            idx += 1
    return mapping


def build_faraday_diag(lmax: int, omega: float, radius: float) -> np.ndarray:
    F = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    for l in range(1, lmax + 1):
        factor = -(1j * omega * radius) / (l * (l + 1))
        F[l] = factor
    return F


def build_dense(G: np.ndarray, Y: np.ndarray, F: np.ndarray, lmax: int) -> np.ndarray:
    lm_map = lm_index_map(lmax)
    n = len(lm_map)
    M = np.zeros((n, n), dtype=np.complex128)
    for L in range(lmax + 1):
        for Mval in range(-L, L + 1):
            row = lm_map[(L, Mval)]
            for l in range(lmax + 1):
                Fl = F[l, 0]  # same across m for given l
                for m in range(-l, l + 1):
                    col = lm_map[(l, m)]
                    accum = 0.0 + 0.0j
                    for l0 in range(lmax + 1):
                        for m0 in range(-l0, l0 + 1):
                            val = G[L, L + Mval, l0, lmax + m0, l, lmax + m]
                            if val == 0.0:
                                continue
                            accum += Y[l0, lmax + m0] * Fl * val
                    M[row, col] = accum
    return M


def build_sparse(G: np.ndarray, Y: np.ndarray, F: np.ndarray, lmax: int) -> np.ndarray:
    lm_map = lm_index_map(lmax)
    n = len(lm_map)

    rows = []
    cols = []
    data = []

    for L in range(lmax + 1):
        for Mval in range(-L, L + 1):
            row = lm_map[(L, Mval)]
            for l in range(lmax + 1):
                Fl = F[l, 0]
                for m in range(-l, l + 1):
                    col = lm_map[(l, m)]
                    accum = 0.0 + 0.0j
                    for l0 in range(lmax + 1):
                        for m0 in range(-l0, l0 + 1):
                            gval = G[L, L + Mval, l0, lmax + m0, l, lmax + m]
                            if gval == 0.0:
                                continue
                            accum += Y[l0, lmax + m0] * Fl * gval
                    if accum != 0.0:
                        rows.append(row)
                        cols.append(col)
                        data.append(accum)

    coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.complex128)
    return coo.toarray()


def main(lmax: int = 10, omega: float = 1.0, radius: float = 1.0, cache_dir: str = "data/gaunt_cache_wigxjpf"):
    path = Path(cache_dir) / "gaunt_wigxjpf.pt"
    if not path.exists():
        print(f"Gaunt cache not found at {path}; run save_gaunt_tensor_wigxjpf first.")
        return
    G = load_gaunt_tensor_wigxjpf(lmax_out=lmax, lmax_y=lmax, lmax_b=lmax, cache_dir=cache_dir).numpy()
    rng = np.random.default_rng(0)
    Y = rng.standard_normal(G.shape[2:4]) + 1j * rng.standard_normal(G.shape[2:4])
    F = build_faraday_diag(lmax, omega, radius)

    t0 = time.perf_counter()
    M_dense = build_dense(G, Y, F, lmax)
    t_dense = time.perf_counter() - t0

    t1 = time.perf_counter()
    M_sparse = build_sparse(G, Y, F, lmax)
    t_sparse = time.perf_counter() - t1

    max_abs = np.max(np.abs(M_dense - M_sparse))
    denom = max(np.max(np.abs(M_dense)), 1e-30)
    max_rel = max_abs / denom

    print(f"lmax={lmax}")
    print(f"dense_time={t_dense:.4f}s, sparse_time={t_sparse:.4f}s")
    print(f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")


if __name__ == "__main__":
    main()
