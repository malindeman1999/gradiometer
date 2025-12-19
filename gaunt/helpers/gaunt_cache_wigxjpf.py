"""
Gaunt tensor cache using pywigxjpf (WIGXJPF backend), sparse-only.

Single lmax is assumed for output (L/M), admittance (l0,m0), and input field (l,m);
cached files are expected to store all axes at the same maximum degree.
Default lmax=10. No pruning; all nonzero Gaunt coefficients are stored sparsely.
"""
from __future__ import annotations

import numpy as np
import torch

from gaunt.helpers.gaunt_vectorized_wigxjpf import gaunt_coeff_vectorized_wigxjpf


def compute_gaunt_tensor_wigxjpf(lmax: int) -> torch.Tensor:
    """Legacy dense build (small lmax only)."""
    G = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
    )
    for L in range(lmax + 1):
        for M in range(-L, L + 1):
            Ls: list[int] = []
            Ms: list[int] = []
            l0s: list[int] = []
            m0s: list[int] = []
            ls: list[int] = []
            ms: list[int] = []
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    m = M - m0
                    l_min = max(abs(m), 0)
                    for l in range(l_min, lmax + 1):
                        Ls.append(L)
                        Ms.append(M)
                        l0s.append(l0)
                        m0s.append(m0)
                        ls.append(l)
                        ms.append(m)
            if Ls:
                vals = gaunt_coeff_vectorized_wigxjpf(
                    np.array(Ls, dtype=int),
                    np.array(Ms, dtype=int),
                    np.array(l0s, dtype=int),
                    np.array(m0s, dtype=int),
                    np.array(ls, dtype=int),
                    np.array(ms, dtype=int),
                )
                for v, l0i, m0i, li, mi in zip(vals, l0s, m0s, ls, ms):
                    G[L, L + M, l0i, lmax + m0i, li, lmax + mi] = v
    return G


def compute_gaunt_tensor_wigxjpf_sparse(
    lmax: int = 10, verbose: bool = False
) -> torch.Tensor:
    """
    Compute Gaunt tensor as a sparse COO without saving to disk (no pruning).
    """
    if verbose:
        print(f"Computing Gaunt tensor (sparse) lmax={lmax}")
    rows: list[tuple[int, int, int, int, int, int]] = []
    vals: list[float] = []
    for L in range(lmax + 1):
        if verbose:
            print(f"  L={L}")
        for M in range(-L, L + 1):
            l0_all = []
            m0_all = []
            for l0 in range(lmax + 1):
                m0_range = np.arange(-l0, l0 + 1, dtype=int)
                l0_all.append(np.full_like(m0_range, l0))
                m0_all.append(m0_range)
            if not l0_all:
                continue
            l0_vec = np.concatenate(l0_all)
            m0_vec = np.concatenate(m0_all)
            m_vec = M - m0_vec
            Ls = []
            Ms = []
            l0s = []
            m0s = []
            ls = []
            ms = []
            for l0i, m0i, m_val in zip(l0_vec, m0_vec, m_vec):
                l_arr = np.arange(abs(int(m_val)), lmax + 1, dtype=int)
                mask = ((L + l0i + l_arr) % 2) == 0
                l_arr = l_arr[mask]
                if l_arr.size == 0:
                    continue
                Ls.append(np.full_like(l_arr, L))
                Ms.append(np.full_like(l_arr, M))
                l0s.append(np.full_like(l_arr, l0i))
                m0s.append(np.full_like(l_arr, m0i))
                ls.append(l_arr)
                ms.append(np.full_like(l_arr, m_val))
            if not Ls:
                continue
            Ls_arr = np.concatenate(Ls)
            Ms_arr = np.concatenate(Ms)
            l0_arr = np.concatenate(l0s)
            m0_arr = np.concatenate(m0s)
            l_arr = np.concatenate(ls)
            m_arr = np.concatenate(ms)
            vals_block = gaunt_coeff_vectorized_wigxjpf(Ls_arr, Ms_arr, l0_arr, m0_arr, l_arr, m_arr)
            nz_mask = vals_block != 0.0
            if np.any(nz_mask):
                vb = vals_block[nz_mask]
                l0_keep = l0_arr[nz_mask]
                m0_keep = m0_arr[nz_mask]
                l_keep = l_arr[nz_mask]
                m_keep = m_arr[nz_mask]
                rows.extend(
                    zip(
                        np.full_like(vb, L, dtype=int),
                        (L + Ms_arr[nz_mask]).astype(int),
                        l0_keep.astype(int),
                        (lmax + m0_keep).astype(int),
                        l_keep.astype(int),
                        (lmax + m_keep).astype(int),
                    )
                )
                vals.extend(vb.tolist())
    if not rows:
        raise RuntimeError("Sparse build produced no entries.")
    idx = torch.tensor(list(zip(*rows)), dtype=torch.int64)
    G_sparse = torch.sparse_coo_tensor(
        idx,
        torch.tensor(vals, dtype=torch.float64),
        size=(
            lmax + 1,
            2 * lmax + 1,
            lmax + 1,
            2 * lmax + 1,
            lmax + 1,
            2 * lmax + 1,
        ),
    ).coalesce()
    return G_sparse
