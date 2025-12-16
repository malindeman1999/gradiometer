"""
Gaunt cache builder that saves one checkpoint per L (complete over M) with no meta sidecars.

Naming: gaunt_wigxjpf_L##.pt under CACHE_DIR. If a file already exists for a given L,
it is skipped. This avoids partial ranges and keeps bookkeeping simple.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from gaunt.gaunt_vectorized_wigxjpf import gaunt_coeff_vectorized_wigxjpf

# Target settings (symmetric)
LMAX = 50
LMAX_OUT = LMAX
LMAX_Y = LMAX
LMAX_B = LMAX
# Store checkpoints in the shared Gaunt cache directory (no L suffix).
CACHE_DIR = Path("data/gaunt_cache_wigxjpf")


def _checkpoint_path(cache_dir: Path, L: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gaunt_wigxjpf_L{L:02d}.pt"


def _append_block(
    L: int, M: int, rows: list[tuple[int, int, int, int, int, int]], vals: list[float]
) -> int:
    """
    Append Gaunt entries for a single (L, M>=0) block, using symmetry to mirror to -M.
    """
    l0_all = []
    m0_all = []
    for l0 in range(LMAX_Y + 1):
        m0_range = np.arange(-l0, l0 + 1, dtype=int)
        l0_all.append(np.full_like(m0_range, l0))
        m0_all.append(m0_range)
    if not l0_all:
        return 0
    l0_vec = np.concatenate(l0_all)
    m0_vec = np.concatenate(m0_all)
    m_vec = M - m0_vec

    Ls: list[np.ndarray] = []
    Ms: list[np.ndarray] = []
    l0s: list[np.ndarray] = []
    m0s: list[np.ndarray] = []
    ls: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    for l0i, m0i, m_val in zip(l0_vec, m0_vec, m_vec):
        l_min = max(abs(int(m_val)), abs(L - l0i))
        l_max = min(L + l0i, LMAX_B)
        if l_min > l_max:
            continue
        start = l_min if (L + l0i + l_min) % 2 == 0 else l_min + 1  # parity stride
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
        return 0

    Ls_arr = np.concatenate(Ls)
    Ms_arr = np.concatenate(Ms)
    l0_arr = np.concatenate(l0s)
    m0_arr = np.concatenate(m0s)
    l_arr = np.concatenate(ls)
    m_arr = np.concatenate(ms)

    vals_block = gaunt_coeff_vectorized_wigxjpf(Ls_arr, Ms_arr, l0_arr, m0_arr, l_arr, m_arr)
    nz_mask = vals_block != 0.0
    if not np.any(nz_mask):
        return 0
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
            (LMAX_Y + m0_keep).astype(int),
            l_keep.astype(int),
            (LMAX_B + m_keep).astype(int),
        )
    )
    vals.extend(vb.tolist())

    added = vb.size

    if M > 0:
        rows.extend(
            zip(
                np.full_like(vb, L, dtype=int),
                (L - Ms_arr[nz_mask]).astype(int),
                l0_keep.astype(int),
                (LMAX_Y - m0_keep).astype(int),
                l_keep.astype(int),
                (LMAX_B - m_keep).astype(int),
            )
        )
        vals.extend(vb.tolist())
        added += vb.size

    return int(added)


def main() -> None:
    print(f"Saving Gaunt cache with checkpoints under {CACHE_DIR}")
    print(f"Target lmax_out={LMAX_OUT}, lmax_y={LMAX_Y}, lmax_b={LMAX_B}")

    for L in range(LMAX_OUT + 1):
        ckpt_path = _checkpoint_path(CACHE_DIR, L)
        if ckpt_path.exists():
            print(f"Skipping L={L}: {ckpt_path.name} already exists.")
            continue

        rows_delta: list[tuple[int, int, int, int, int, int]] = []
        vals_delta: list[float] = []
        for M in range(0, L + 1):
            _append_block(L, M, rows_delta, vals_delta)

        if not rows_delta:
            print(f"L={L} produced no entries; writing empty placeholder.")

        idx = torch.tensor(list(zip(*rows_delta)), dtype=torch.int64) if rows_delta else torch.empty((6, 0), dtype=torch.int64)
        vals = torch.tensor(vals_delta, dtype=torch.float64) if vals_delta else torch.empty((0,), dtype=torch.float64)
        G_sparse = torch.sparse_coo_tensor(
            idx,
            vals,
            size=(
                LMAX_OUT + 1,
                2 * LMAX_OUT + 1,
                LMAX_Y + 1,
                2 * LMAX_Y + 1,
                LMAX_B + 1,
                2 * LMAX_B + 1,
            ),
        ).coalesce()
        meta = {
            "target_lmax_out": LMAX_OUT,
            "target_lmax_y": LMAX_Y,
            "target_lmax_b": LMAX_B,
            "lmax_out": LMAX_OUT,
            "lmax_y": LMAX_Y,
            "lmax_b": LMAX_B,
            "sparse": True,
            "prune_tol": None,
            "symmetric": True,
            "last_L": L,
            "last_M": L,
            "range_start_L": L,
            "range_end_L": L,
        }
        torch.save({"G_sparse": G_sparse, **meta}, ckpt_path)
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"Saved L={L} checkpoint to {ckpt_path} (entries={G_sparse._nnz()}, size={size_mb:.2f} MB)")

    print(f"Finished Gaunt tensor lmax_out={LMAX_OUT}, lmax_y={LMAX_Y}, lmax_b={LMAX_B}, saved under {CACHE_DIR}")


if __name__ == "__main__":
    main()
