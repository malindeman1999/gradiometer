"""
Gaunt tensor cache using pywigxjpf (WIGXJPF backend), sparse-only.

Supports independent lmax values for:
- L/M (output): lmax_out
- admittance Y_s (l0,m0): lmax_y
- input field (l,m): lmax_b

Defaults: lmax_out=10, lmax_y=10, lmax_b=1.
No pruning; all nonzero Gaunt coefficients are stored sparsely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .gaunt_vectorized_wigxjpf import gaunt_coeff_vectorized_wigxjpf
from europa.legacy_variants.gaunt_cache_wigxjpf_sym import compute_gaunt_tensor_wigxjpf_sym2_sparse


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


def _cache_path(cache_dir: Union[str, Path]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "gaunt_wigxjpf.pt"


def save_gaunt_tensor_wigxjpf(
    lmax_out: int = 10,
    lmax_y: int | None = None,
    lmax_b: int | None = None,
    cache_dir: Union[str, Path] = "data/gaunt_cache_wigxjpf",
) -> None:
    """
    Compute and save Gaunt tensor as a sparse COO (no pruning), using the faster sym2 builder.
    """
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = lmax_out if lmax_b is None else lmax_b
    path = _cache_path(cache_dir)
    print(
        f"Computing Gaunt tensor (sym2) up to lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b} "
        f"(saving to {path})"
    )
    G_sparse = compute_gaunt_tensor_wigxjpf_sym2_sparse(
        lmax_out=lmax_out, lmax_y=lmax_y, lmax_b=lmax_b, verbose=True
    )
    meta = {
        "lmax_out": lmax_out,
        "lmax_y": lmax_y,
        "lmax_b": lmax_b,
        "sparse": True,
        "prune_tol": None,
        "symmetric": True,
    }
    torch.save({"G_sparse": G_sparse, **meta}, path)
    print(f"Finished Gaunt tensor lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}, saved to {path} (sym2 sparse)")


def compute_gaunt_tensor_wigxjpf_sparse(
    lmax_out: int = 10, lmax_y: int | None = None, lmax_b: int | None = None, verbose: bool = False
) -> torch.Tensor:
    """
    Compute Gaunt tensor as a sparse COO without saving to disk (no pruning).
    """
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = lmax_out if lmax_b is None else lmax_b
    if verbose:
        print(f"Computing Gaunt tensor (sparse) lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}")
    rows: list[tuple[int, int, int, int, int, int]] = []
    vals: list[float] = []
    for L in range(lmax_out + 1):
        if verbose:
            print(f"  L={L}")
        for M in range(-L, L + 1):
            l0_all = []
            m0_all = []
            for l0 in range(lmax_y + 1):
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
                l_arr = np.arange(abs(int(m_val)), lmax_b + 1, dtype=int)
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
                        (lmax_y + m0_keep).astype(int),
                        l_keep.astype(int),
                        (lmax_b + m_keep).astype(int),
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
            lmax_out + 1,
            2 * lmax_out + 1,
            lmax_y + 1,
            2 * lmax_y + 1,
            lmax_b + 1,
            2 * lmax_b + 1,
        ),
    ).coalesce()
    return G_sparse


def load_gaunt_tensor_wigxjpf(
    lmax_out: int = 10, lmax_y: int | None = None, lmax_b: int | None = None, cache_dir: Union[str, Path] = "data/gaunt_cache_wigxjpf"
) -> torch.Tensor:
    """
    Load Gaunt tensor for requested lmax_out/lmax_y/lmax_b. Raises if cache missing or insufficient.
    """
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = lmax_out if lmax_b is None else lmax_b
    path = _cache_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Gaunt cache not found at {path}; run save_gaunt_tensor_wigxjpf first.")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not (isinstance(obj, dict) and ("G" in obj or "G_sparse" in obj) and "lmax_out" in obj and "lmax_y" in obj and "lmax_b" in obj):
        raise ValueError(f"Invalid cache format at {path}")
    cached_out = int(obj["lmax_out"])
    cached_y = int(obj["lmax_y"])
    cached_b = int(obj["lmax_b"])
    if cached_out < lmax_out or cached_y < lmax_y or cached_b < lmax_b:
        raise ValueError(
            f"Cached lmax_out/y/b=({cached_out},{cached_y},{cached_b}) < requested ({lmax_out},{lmax_y},{lmax_b})"
        )
    if "G" in obj:
        G = obj["G"]
    else:
        G = obj["G_sparse"].to_dense()
    if cached_out > lmax_out or cached_y > lmax_y or cached_b > lmax_b:
        G = G[
            : lmax_out + 1,
            : 2 * lmax_out + 1,
            : lmax_y + 1,
            : 2 * lmax_y + 1,
            : lmax_b + 1,
            : 2 * lmax_b + 1,
        ]
    return G
