"""
Gaunt cache builder that saves one checkpoint per L shell, storing only a 1/12
canonical branch and regenerating the rest via symmetry at load time.

Naming: gaunt_wigxjpf_L##.pt under CACHE_DIR. Each file stores a "shell" containing
canonical entries where max(L, l0, l) == shell_L. If a file already exists for a given
shell_L, it is skipped. This avoids partial ranges and keeps bookkeeping simple.

Usage:
    python gaunt/run_gaunt_calculator.py
    python gaunt/run_gaunt_calculator.py --plot

This script resumes automatically: it skips any L checkpoint files that already exist
and only computes missing L values. Output is written under CACHE_DIR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Allow running this file directly (python gaunt/run_gaunt_calculator.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaunt.helpers.gaunt_vectorized_wigxjpf import gaunt_coeff_vectorized_wigxjpf

# Target settings (symmetric)
LMAX = 71  # corresponds to 4 subdivisions (5120 faces) and 5184 coeffs
LMAX_OUT = LMAX
LMAX_Y = LMAX
LMAX_B = LMAX
# Store checkpoints in the shared Gaunt cache directory (no L suffix).
CACHE_DIR = Path("data/gaunt_cache_wigxjpf")


def _checkpoint_path(cache_dir: Path, L: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gaunt_wigxjpf_L{L:02d}.pt"


def _append_block(
    L: int,
    M: int,
    rows: list[tuple[int, int, int, int, int, int]],
    vals: list[float],
    lmax_y: int,
    lmax_b: int,
    shell_L: int,
    triple_cache: dict[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], float],
) -> int:
    """
    Append Gaunt entries for a single (L, M>=0) block, storing only the canonical branch.
    Only canonical entries satisfying max(L, l0, l) == shell_L are included.
    """
    l0_all = []
    m0_all = []
    for l0 in range(lmax_y + 1):
        m0_range = np.arange(-l0, l0 + 1, dtype=int)
        l0_all.append(np.full_like(m0_range, l0))
        m0_all.append(m0_range)
    if not l0_all:
        return 0
    l0_vec = np.concatenate(l0_all)
    m0_vec = np.concatenate(m0_all)
    m_vec = M - m0_vec

    Ls: list[np.ndarray] = []
    l0s: list[np.ndarray] = []
    m0s: list[np.ndarray] = []
    ls: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    for l0i, m0i, m_val in zip(l0_vec, m0_vec, m_vec):
        base = max(L, int(l0i))
        if base > shell_L:
            continue
        l_min = max(abs(int(m_val)), abs(L - l0i))
        l_max = min(L + l0i, lmax_b)
        if l_min > l_max:
            continue
        start = l_min if (L + l0i + l_min) % 2 == 0 else l_min + 1  # parity stride
        if start > l_max:
            continue
        l_arr = np.arange(start, l_max + 1, 2, dtype=int)
        if base < shell_L:
            l_arr = l_arr[l_arr == shell_L]
        if l_arr.size == 0:
            continue
        Ls.append(np.full_like(l_arr, L))
        l0s.append(np.full_like(l_arr, l0i))
        m0s.append(np.full_like(l_arr, m0i))
        ls.append(l_arr)
        ms.append(np.full_like(l_arr, m_val))

    if not Ls:
        return 0

    Ls_arr = np.concatenate(Ls)
    l0_arr = np.concatenate(l0s)
    m0_arr = np.concatenate(m0s)
    l_arr = np.concatenate(ls)
    m_arr = np.concatenate(ms)

    triple_keys: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    new_keys: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    new_key_set: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = set()

    def parity_sign(m_val: int) -> int:
        return -1 if (m_val & 1) else 1

    def sorted_triple_key(
        l1: int, m1: int, l2: int, m2: int, l3: int, m3: int
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        a = (int(l1), int(m1))
        b = (int(l2), int(m2))
        c = (int(l3), int(m3))
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        return (a, b, c)

    for l0i, m0i, li, mi in zip(l0_arr, m0_arr, l_arr, m_arr):
        key = sorted_triple_key(L, -M, int(l0i), int(m0i), int(li), int(mi))
        triple_keys.append(key)
        if key not in triple_cache and key not in new_key_set:
            new_keys.append(key)
            new_key_set.add(key)

    if new_keys:
        l1 = np.array([k[0][0] for k in new_keys], dtype=int)
        m1 = np.array([k[0][1] for k in new_keys], dtype=int)
        l2 = np.array([k[1][0] for k in new_keys], dtype=int)
        m2 = np.array([k[1][1] for k in new_keys], dtype=int)
        l3 = np.array([k[2][0] for k in new_keys], dtype=int)
        m3 = np.array([k[2][1] for k in new_keys], dtype=int)

        vals_raw = gaunt_coeff_vectorized_wigxjpf(l1, -m1, l2, m2, l3, m3)
        signs = np.where((m1 & 1) == 0, 1.0, -1.0)
        vals_T = signs * vals_raw
        for key, val in zip(new_keys, vals_T):
            triple_cache[key] = float(val)

    vals_block = np.array([triple_cache[key] for key in triple_keys], dtype=np.float64)
    vals_block *= parity_sign(M)
    nz_mask = vals_block != 0.0
    if not np.any(nz_mask):
        return 0
    vb = vals_block[nz_mask]
    l0_keep = l0_arr[nz_mask]
    m0_keep = m0_arr[nz_mask]
    l_keep = l_arr[nz_mask]
    m_keep = m_arr[nz_mask]

    base = 2 * shell_L + 3
    shift = shell_L + 1
    key1 = np.full_like(l0_keep, L * base + (-M + shift))
    key2 = l0_keep * base + (m0_keep + shift)
    key3 = l_keep * base + (m_keep + shift)
    order_mask = (key1 <= key2) & (key2 <= key3)

    keep_mask = order_mask
    if not np.any(keep_mask):
        return 0
    vb = vb[keep_mask]
    l0_keep = l0_keep[keep_mask]
    m0_keep = m0_keep[keep_mask]
    l_keep = l_keep[keep_mask]
    m_keep = m_keep[keep_mask]

    rows.extend(
        zip(
            np.full_like(vb, L, dtype=int),
            np.full_like(vb, L + M, dtype=int),
            l0_keep.astype(int),
            (lmax_y + m0_keep).astype(int),
            l_keep.astype(int),
            (lmax_b + m_keep).astype(int),
        )
    )
    vals.extend(vb.tolist())

    added = vb.size

    return int(added)


def _update_shell_plot(fig, ax, counts: np.ndarray, max_L: int, shell_L: int, title: str, cbar_state, plt):
    ax.clear()
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    grid = np.arange(shell_L + 1)
    Y, Z = np.meshgrid(grid, grid, indexing="ij")
    X = np.full_like(Y, shell_L)
    vals_x = counts[shell_L, : shell_L + 1, : shell_L + 1]

    X2, Z2 = np.meshgrid(grid, grid, indexing="ij")
    Y2 = np.full_like(X2, shell_L)
    vals_y = counts[: shell_L + 1, shell_L, : shell_L + 1]

    X3, Y3 = np.meshgrid(grid, grid, indexing="ij")
    Z3 = np.full_like(X3, shell_L)
    vals_z = counts[: shell_L + 1, : shell_L + 1, shell_L]

    max_val = int(max(np.max(vals_x), np.max(vals_y), np.max(vals_z)))
    if max_val == 0:
        max_val = 1
    norm = mcolors.Normalize(vmin=0, vmax=max_val)
    cmap = cm.get_cmap("viridis")

    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(vals_x)), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.plot_surface(X2, Y2, Z2, facecolors=cmap(norm(vals_y)), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.plot_surface(X3, Y3, Z3, facecolors=cmap(norm(vals_z)), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.set_xlabel("L")
    ax.set_ylabel("l0")
    ax.set_zlabel("l")
    ax.set_xlim(0, max_L)
    ax.set_ylim(0, max_L)
    ax.set_zlim(0, max_L)
    ax.view_init(elev=22, azim=40)
    ax.set_title(title)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    if cbar_state["cbar"] is None:
        cbar_state["cbar"] = fig.colorbar(mappable, ax=ax, shrink=0.6)
    else:
        try:
            cbar_state["cbar"].update_normal(mappable)
        except Exception:
            try:
                cbar_state["cbar"].remove()
            except Exception:
                pass
            cbar_state["cbar"] = fig.colorbar(mappable, ax=ax, shrink=0.6)
    fig.canvas.draw_idle()
    plt.pause(0.001)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Gaunt cache shell checkpoints.")
    parser.add_argument("--plot", action="store_true", help="Show shell plot animation.")
    args = parser.parse_args()

    plot = bool(args.plot)
    if plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        plt.show(block=False)
        cbar_state = {"cbar": None}
        counts = np.zeros((LMAX_OUT + 1, LMAX_OUT + 1, LMAX_OUT + 1), dtype=np.int32)
    else:
        plt = None
        fig = None
        ax = None
        cbar_state = None
        counts = None

    print(f"Saving Gaunt cache with checkpoints under {CACHE_DIR}")
    print(f"Target lmax_out={LMAX_OUT}, lmax_y={LMAX_Y}, lmax_b={LMAX_B}")

    for shell_L in range(LMAX_OUT + 1):
        ckpt_path = _checkpoint_path(CACHE_DIR, shell_L)
        if ckpt_path.exists():
            print(f"Skipping shell L={shell_L}: {ckpt_path.name} already exists.")
            continue

        rows_delta: list[tuple[int, int, int, int, int, int]] = []
        vals_delta: list[float] = []
        triple_cache: dict[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], float] = {}
        for L in range(shell_L + 1):
            for M in range(0, L + 1):
                _append_block(L, M, rows_delta, vals_delta, shell_L, shell_L, shell_L, triple_cache)

        if not rows_delta:
            print(f"Shell L={shell_L} produced no entries; writing empty placeholder.")

        idx = torch.tensor(list(zip(*rows_delta)), dtype=torch.int64) if rows_delta else torch.empty((6, 0), dtype=torch.int64)
        vals = torch.tensor(vals_delta, dtype=torch.float64) if vals_delta else torch.empty((0,), dtype=torch.float64)
        G_sparse = torch.sparse_coo_tensor(
            idx,
            vals,
            size=(
                shell_L + 1,
                2 * shell_L + 1,
                shell_L + 1,
                2 * shell_L + 1,
                shell_L + 1,
                2 * shell_L + 1,
            ),
        ).coalesce()
        meta = {
            "format": "shell",
            "shell_L": shell_L,
            "target_lmax_out": LMAX_OUT,
            "target_lmax_y": LMAX_Y,
            "target_lmax_b": LMAX_B,
            "lmax_out": shell_L,
            "lmax_y": shell_L,
            "lmax_b": shell_L,
            "sparse": True,
            "prune_tol": None,
            "symmetric": True,
            "symmetry_mode": "canonical12",
            "last_L": shell_L,
            "last_M": shell_L,
            "range_start_L": 0,
            "range_end_L": shell_L,
        }
        torch.save({"G_sparse": G_sparse, **meta}, ckpt_path)
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"Saved shell L={shell_L} checkpoint to {ckpt_path} (entries={G_sparse._nnz()}, size={size_mb:.2f} MB)")

        if plot and rows_delta:
            idx_np = G_sparse.indices().cpu().numpy()
            np.add.at(counts, (idx_np[0], idx_np[2], idx_np[4]), 1)
            _update_shell_plot(
                fig,
                ax,
                counts,
                LMAX_OUT,
                shell_L,
                title=f"Gaunt shells filled up to L={shell_L}",
                cbar_state=cbar_state,
                plt=plt,
            )

    print(f"Finished Gaunt tensor shells up to L={LMAX_OUT}, saved under {CACHE_DIR}")


if __name__ == "__main__":
    main()
