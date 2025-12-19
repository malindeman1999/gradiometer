"""
Assemble Gaunt checkpoints into a single sparse tensor.

Behavior (new format):
- Reads per-L checkpoint files named gaunt_wigxjpf_L##.pt in the cache directory.
  Each file contains the entries for that exact L (no sidecar metadata).
  We assume files are complete for their L and non-overlapping.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

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

def assemble_in_memory(cache_dir: Path, lmax_limit: int, verbose: bool = False):
    """
    Assemble Gaunt checkpoints into a sparse tensor (optionally trimmed to lmax_limit) without saving.
    Returns (tensor, meta).
    """
    ckpts = _all_checkpoints(cache_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cache_dir}")
    total_ckpts = len(ckpts)
    print(f"Found {total_ckpts} checkpoint file(s) in {cache_dir}; assuming coverage is complete.", flush=True)

    assemble_L = lmax_limit
    _require_all_L(ckpts, assemble_L)
    lmax_y_lim = lmax_limit
    lmax_b_lim = lmax_limit
    print(f"Assembling checkpoints up to L={assemble_L} (lmax_out=lmax_y=lmax_b={lmax_limit})...", flush=True)

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
        shift_y = lmax_y_full - lmax_y_lim
        shift_b = lmax_b_full - lmax_b_lim
        mask = (
            (idx[0] <= assemble_L)
            & (idx[2] <= lmax_y_lim)
            & (idx[4] <= lmax_b_lim)
            & (idx[3] >= shift_y)
            & (idx[3] <= shift_y + 2 * lmax_y_lim)
            & (idx[5] >= shift_b)
            & (idx[5] <= shift_b + 2 * lmax_b_lim)
        )
        if not torch.any(mask):
            continue
        idx_f = idx[:, mask].clone()
        # Re-center m0/m indices to the trimmed lmax offsets
        idx_f[3] = idx_f[3] - shift_y
        idx_f[5] = idx_f[5] - shift_b
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
        "target_lmax_out": lmax_limit,
        "target_lmax_y": lmax_limit,
        "target_lmax_b": lmax_limit,
        "assembled_lmax_y": lmax_y_lim,
        "assembled_lmax_b": lmax_b_lim,
        "sparse": True,
        "prune_tol": None,
        "symmetric": True,
    }
    return G_assembled, meta_out


def assemble(cache_dir: Path, output_path: Path, verbose: bool = False) -> None:
    G_assembled, meta_out = assemble_in_memory(cache_dir, lmax_limit=None, verbose=verbose)
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
    args = parser.parse_args()
    assemble(args.cache_dir, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
