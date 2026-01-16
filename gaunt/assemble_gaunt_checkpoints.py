"""
Assemble Gaunt checkpoints into a single sparse tensor.

Behavior (new format):
- Reads per-L checkpoint files named gaunt_wigxjpf_L##.pt in the cache directory.
  Each file contains a shell where max(L, l0, l) == shell_L (no sidecar metadata).
  We assume shells are complete and non-overlapping.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def _all_checkpoints(cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return sorted(cache_dir.glob("gaunt_wigxjpf_L*.pt"))


def _require_all_L(ckpts: list[Path], assemble_L: int) -> None:
    """Ensure a checkpoint exists for each L in [0, assemble_L]."""
    have = {int(p.stem.split("_L")[-1]) for p in ckpts if "_L" in p.stem}
    missing = [L for L in range(assemble_L + 1) if L not in have]
    if missing:
        raise FileNotFoundError(f"Missing Gaunt checkpoint(s) for L={missing}")


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


def assemble_in_memory(cache_dir: Path, lmax_limit: Optional[int], verbose: bool = False, plot: bool = True):
    """
    Assemble Gaunt checkpoints into a sparse tensor (optionally trimmed to lmax_limit) without saving.
    Returns (tensor, meta).
    """
    ckpts = _all_checkpoints(cache_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cache_dir}")
    total_ckpts = len(ckpts)
    print(f"Found {total_ckpts} checkpoint file(s) in {cache_dir}; assuming coverage is complete.", flush=True)

    if lmax_limit is None:
        assemble_L = max(int(p.stem.split("_L")[-1]) for p in ckpts if "_L" in p.stem)
        lmax_y_lim = assemble_L
        lmax_b_lim = assemble_L
        target_lmax = assemble_L
    else:
        assemble_L = lmax_limit
        lmax_y_lim = lmax_limit
        lmax_b_lim = lmax_limit
        target_lmax = lmax_limit
    _require_all_L(ckpts, assemble_L)
    print(f"Assembling checkpoints up to L={assemble_L} (lmax_out=lmax_y=lmax_b={target_lmax})...", flush=True)

    if plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        plt.show(block=False)
        cbar_state = {"cbar": None}
        counts = np.zeros((assemble_L + 1, assemble_L + 1, assemble_L + 1), dtype=np.int32)
    else:
        plt = None
        fig = None
        ax = None
        cbar_state = None
        counts = None

    # Merge tensors up to assemble_L / lmax limits
    idx_all: List[List[int]] = [[], [], [], [], [], []]
    vals_all: List[float] = []
    t_start = time.perf_counter()
    for i, path in enumerate(ckpts, start=1):
        L_val = int(path.stem.split("_L")[-1])
        if L_val > assemble_L:
            continue
        if verbose:
            print(f"[{i}/{total_ckpts}] Loading {path.name} (L={L_val})...", flush=True)
        t0 = time.perf_counter()
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        t_load = time.perf_counter() - t0
        G_sparse = obj["G_sparse"] if "G_sparse" in obj else obj["G"].to_sparse()
        G_sparse = G_sparse.coalesce()
        idx = G_sparse.indices()
        vals = G_sparse.values()
        # Infer source lmax offsets from tensor shape to trim m indices
        lmax_y_full = (G_sparse.shape[3] - 1) // 2
        lmax_b_full = (G_sparse.shape[5] - 1) // 2
        mask = (
            (idx[0] <= assemble_L)
            & (idx[2] <= lmax_y_lim)
            & (idx[4] <= lmax_b_lim)
        )
        if lmax_y_full >= lmax_y_lim:
            shift_y = lmax_y_full - lmax_y_lim
            mask &= (idx[3] >= shift_y) & (idx[3] <= shift_y + 2 * lmax_y_lim)
        else:
            shift_y = lmax_y_lim - lmax_y_full
        if lmax_b_full >= lmax_b_lim:
            shift_b = lmax_b_full - lmax_b_lim
            mask &= (idx[5] >= shift_b) & (idx[5] <= shift_b + 2 * lmax_b_lim)
        else:
            shift_b = lmax_b_lim - lmax_b_full
        if not torch.any(mask):
            continue
        idx_f = idx[:, mask].clone()
        # Re-center m0/m indices to the trimmed lmax offsets
        if lmax_y_full >= lmax_y_lim:
            idx_f[3] = idx_f[3] - shift_y
        else:
            idx_f[3] = idx_f[3] + shift_y
        if lmax_b_full >= lmax_b_lim:
            idx_f[5] = idx_f[5] - shift_b
        else:
            idx_f[5] = idx_f[5] + shift_b
        vals_f = vals[mask]
        added = int(mask.sum())
        for d in range(6):
            idx_all[d].extend(idx_f[d].tolist())
        vals_all.extend(vals_f.tolist())
        if verbose:
            t_done = time.perf_counter() - t0
            print(
                f"[{i}/{total_ckpts}] Added {added} entries from {path.name} "
                f"(L={L_val}); "
                f"load={t_load:.2f}s, total step={t_done:.2f}s; nnz so far={len(vals_all)}",
                flush=True,
            )
        if plot:
            idx_np = idx_f[[0, 2, 4]].cpu().numpy()
            np.add.at(counts, (idx_np[0], idx_np[1], idx_np[2]), 1)
            _update_shell_plot(
                fig,
                ax,
                counts,
                assemble_L,
                L_val,
                title=f"Gaunt shells assembled up to L={L_val}",
                cbar_state=cbar_state,
                plt=plt,
            )

    if not vals_all:
        raise RuntimeError("No entries collected for assembled tensor.")

    idx_tensor = torch.tensor(idx_all, dtype=torch.int64)
    vals_tensor = torch.tensor(vals_all, dtype=torch.float64)
    max_L_collected = int(idx_tensor[0].max().item()) if idx_tensor.numel() > 0 else -1
    if max_L_collected < assemble_L:
        raise RuntimeError(
            f"Expected entries up to L={assemble_L} but only collected up to L={max_L_collected}. "
            "Check that high-L checkpoints are present and non-empty."
        )
    G_assembled = torch.sparse_coo_tensor(
        idx_tensor,
        vals_tensor,
        size=(
            assemble_L + 1,
            2 * assemble_L + 1,
            lmax_y_lim + 1,
            2 * lmax_y_lim + 1,
            lmax_b_lim + 1,
            2 * lmax_b_lim + 1,
        ),
    ).coalesce()

    meta_out = {
        "complete_L": assemble_L,
        "assembled_L": assemble_L,
        "target_lmax_out": target_lmax,
        "target_lmax_y": target_lmax,
        "target_lmax_b": target_lmax,
        "assembled_lmax_y": lmax_y_lim,
        "assembled_lmax_b": lmax_b_lim,
        "sparse": True,
        "prune_tol": None,
        "symmetric": True,
    }
    return G_assembled, meta_out


def assemble(cache_dir: Path, output_path: Path, verbose: bool = False, plot: bool = True) -> None:
    G_assembled, meta_out = assemble_in_memory(cache_dir, lmax_limit=None, verbose=verbose, plot=plot)
    torch.save({"G_sparse": G_assembled, **meta_out}, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved assembled tensor to {output_path} "
        f"(size={size_mb:.2f} MB, entries={G_assembled._nnz()}, complete_L={meta_out['complete_L']})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble Gaunt checkpoints into a single sparse tensor.")
    parser.add_argument("--cache-dir", default="data/gaunt_cache_wigxjpf", type=Path, help="Directory containing checkpoint files.")
    parser.add_argument("--output", default="data/gaunt_cache_wigxjpf/gaunt_wigxjpf_assembled.pt", type=Path, help="Output path for assembled tensor.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    parser.add_argument("--no-plot", action="store_true", help="Suppress shell plot animation.")
    args = parser.parse_args()
    assemble(args.cache_dir, args.output, verbose=args.verbose, plot=not args.no_plot)


if __name__ == "__main__":
    main()
