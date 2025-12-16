"""
Symmetry-aware Gaunt tensor cache using pywigxjpf (WIGXJPF backend), sparse-only.

- Uses selection, parity, triangle, and m-flip symmetry: G(L,-M,l0,-m0,l,-m)=G(L,M,l0,m0,l,m).
- Computes only M >= 0 and mirrors to negative M.
- Supports independent lmax values for output (L/M), admittance (l0,m0), and input field (l,m).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from gaunt.gaunt_vectorized_wigxjpf import gaunt_coeff_vectorized_wigxjpf


def _cache_path(cache_dir: Union[str, Path]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "gaunt_wigxjpf.pt"


def save_gaunt_tensor_wigxjpf_sym(
    lmax_out: int = 10,
    lmax_y: int | None = None,
    lmax_b: int | None = None,
    cache_dir: Union[str, Path] = "data/gaunt_cache_wigxjpf",
) -> None:
    """
    Compute and save Gaunt tensor as a sparse COO using m-flip symmetry to halve work.
    No pruning; checkpoints per L.
    """
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = lmax_out if lmax_b is None else lmax_b
    path = _cache_path(cache_dir)
    print(f"[sym] Computing Gaunt tensor lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b} -> {path}")

    rows: list[tuple[int, int, int, int, int, int]] = []
    vals: list[float] = []
    processed = 0
    keep = 0
    zero_count = 0

    for L in range(lmax_out + 1):
        print(f"  L={L}")
        for M in range(0, L + 1):  # m-flip symmetry, compute M>=0
            # Build l0,m0 grid
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
            l_blocks = []
            for m_val in m_vec:
                l_min = max(abs(int(m_val)), 0)
                # triangle/parity: |L-l0| <= l <= L+l0 and even sum
                l_range = np.arange(l_min, lmax_b + 1, dtype=int)
                mask = (np.abs(L - l0_vec[0]) <= l_range) if l_range.size else np.array([], dtype=bool)
                l_blocks.append(l_range)

            Ls = []
            Ms = []
            l0s = []
            m0s = []
            ls = []
            ms = []
            for l0i, m0i, m_val, l_arr in zip(l0_vec, m0_vec, m_vec, l_blocks):
                if l_arr.size == 0:
                    continue
                # parity filter: L + l0 + l even
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
            processed += len(vals_block)
            nz_mask = vals_block != 0.0
            zero_count += int(np.count_nonzero(~nz_mask))
            if np.any(nz_mask):
                vb = vals_block[nz_mask]
                l0_keep = l0_arr[nz_mask]
                m0_keep = m0_arr[nz_mask]
                l_keep = l_arr[nz_mask]
                m_keep = m_arr[nz_mask]
                keep += vb.size

                # positive M entries
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

                # mirror to negative M if M>0
                if M > 0:
                    rows.extend(
                        zip(
                            np.full_like(vb, L, dtype=int),
                            (L - Ms_arr[nz_mask]).astype(int),
                            l0_keep.astype(int),
                            (lmax_y - m0_keep).astype(int),
                            l_keep.astype(int),
                            (lmax_b - m_keep).astype(int),
                        )
                    )
                    vals.extend(vb.tolist())
                    keep += vb.size  # mirrored entries

        if rows:
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
            meta = {
                "lmax_out": lmax_out,
                "lmax_y": lmax_y,
                "lmax_b": lmax_b,
                "sparse": True,
                "prune_tol": None,
                "symmetric": True,
            }
            torch.save({"G_sparse": G_sparse, **meta}, path)
            print(f"Completed L={L}/{lmax_out}, checkpoint saved (keep={keep}, zeros={zero_count}, processed={processed})")

    if not rows:
        raise RuntimeError("Sparse build produced no entries.")

    # Final save
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
    meta = {
        "lmax_out": lmax_out,
        "lmax_y": lmax_y,
        "lmax_b": lmax_b,
        "sparse": True,
        "prune_tol": None,
        "symmetric": True,
    }
    torch.save({"G_sparse": G_sparse, **meta}, path)
    print(
        f"Finished Gaunt tensor lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}, saved to {path} (symmetric sparse)"
        f" keep={keep}, zeros={zero_count}, processed={processed}"
    )


def compute_gaunt_tensor_wigxjpf_sym_sparse(
    lmax_out: int = 10, lmax_y: int | None = None, lmax_b: int | None = None, verbose: bool = False
) -> torch.Tensor:
    """
    Compute Gaunt tensor as a sparse COO using symmetry, without saving to disk.
    Set verbose=True to print progress; default is silent for benchmarking.
    """
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = lmax_out if lmax_b is None else lmax_b

    if verbose:
        print(f"[sym] Computing Gaunt tensor (sparse, no save) lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}")

    rows: list[tuple[int, int, int, int, int, int]] = []
    vals: list[float] = []

    for L in range(lmax_out + 1):
        if verbose:
            print(f"  L={L}")
        for M in range(0, L + 1):  # m-flip symmetry, compute M>=0
            # Build l0,m0 grid
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
                # parity: L + l0 + l even
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

                if M > 0:
                    rows.extend(
                        zip(
                            np.full_like(vb, L, dtype=int),
                            (L - Ms_arr[nz_mask]).astype(int),
                            l0_keep.astype(int),
                            (lmax_y - m0_keep).astype(int),
                            l_keep.astype(int),
                            (lmax_b - m_keep).astype(int),
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


def compute_gaunt_tensor_wigxjpf_sym2_sparse(
    lmax_out: int = 10, lmax_y: int | None = None, lmax_b: int | None = None, verbose: bool = False
) -> torch.Tensor:
    """
    Symmetry+rules accelerated builder (no save):
    - M >= 0 with m-flip mirroring
    - Triangle: |L-l0| <= l <= L+l0 and |m|<=l
    - Parity: L+l0+l even (implemented via stride-2 arange)
    - Skips zero entries before storing
    """
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = lmax_out if lmax_b is None else lmax_b

    if verbose:
        print(f"[sym2] Computing Gaunt tensor (sparse, no save) lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}")

    rows: list[tuple[int, int, int, int, int, int]] = []
    vals: list[float] = []

    for L in range(lmax_out + 1):
        if verbose:
            print(f"  L={L}")
        for M in range(0, L + 1):  # m-flip symmetry
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
                # triangle + |m| constraint
                l_min = max(abs(int(m_val)), abs(L - l0i))
                l_max = min(L + l0i, lmax_b)
                if l_min > l_max:
                    continue
                # parity stride: ensure L+l0+l even => adjust start to correct parity and step 2
                start = l_min if (L + l0i + l_min) % 2 == 0 else l_min + 1
                if start > l_max:
                    continue
                l_arr = np.arange(start, l_max + 1, 2, dtype=int)
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

                if M > 0:
                    rows.extend(
                        zip(
                            np.full_like(vb, L, dtype=int),
                            (L - Ms_arr[nz_mask]).astype(int),
                            l0_keep.astype(int),
                            (lmax_y - m0_keep).astype(int),
                            l_keep.astype(int),
                            (lmax_b - m_keep).astype(int),
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
