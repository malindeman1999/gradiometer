"""
Assemble Gaunt checkpoints into a single sparse tensor.

Behavior (new format):
- Reads per-L checkpoint files named gaunt_wigxjpf_L##.pt in the cache directory.
  Each file contains the entries for that exact L (no .meta sidecar).
- Verifies lmax targets match across all files.
- Tracks coverage per L using recorded last_M/range values to detect completeness.
- Confirms all L <= L_complete are fully covered (M=0..L) before saving.
- Saves the merged sparse tensor to an output file with meta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _all_checkpoints(cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return sorted(cache_dir.glob("gaunt_wigxjpf_L*.pt"))


def _load_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".meta.json")
    try:
        with meta_path.open("r", encoding="ascii") as f:
            return json.load(f)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass

    # Fallback to reading the tensor file if meta is missing/broken
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or ("G_sparse" not in obj and "G" not in obj):
        raise ValueError(f"Invalid checkpoint format: {path}")
    fallback = {
        "target_lmax_out": int(obj.get("target_lmax_out", obj.get("lmax_out", -1))),
        "target_lmax_y": int(obj.get("target_lmax_y", obj.get("lmax_y", -1))),
        "target_lmax_b": int(obj.get("target_lmax_b", obj.get("lmax_b", -1))),
        "last_L": int(obj.get("last_L", -1)),
        "last_M": int(obj.get("last_M", -1)),
        "range_start_L": int(obj.get("range_start_L", -1)),
        "range_end_L": int(obj.get("range_end_L", -1)),
    }
    return fallback


def _validate_targets(metas: list[dict]) -> tuple[int, int, int]:
    outs = [int(m.get("target_lmax_out", -1)) for m in metas if "target_lmax_out" in m]
    ys = [int(m.get("target_lmax_y", -1)) for m in metas if "target_lmax_y" in m]
    bs = [int(m.get("target_lmax_b", -1)) for m in metas if "target_lmax_b" in m]
    if not outs or not ys or not bs:
        raise ValueError("Missing target lmax metadata in checkpoints.")
    # Allow mixing runs with different targets; take the maximum to size the output.
    return max(outs), max(ys), max(bs)


def _coverage_from_meta(meta: dict, coverage: List[int]) -> None:
    start = int(meta.get("range_start_L", -1))
    end = int(meta.get("range_end_L", -1))
    last_M = int(meta.get("last_M", -1))
    if start < 0 or end < start:
        return
    end_inclusive = min(end, len(coverage) - 1)
    for L in range(start, end_inclusive + 1):
        coverage[L] = max(coverage[L], last_M if L == end_inclusive else L)


def _compute_complete_L(coverage: List[int]) -> int:
    complete_L = -1
    for L, m in enumerate(coverage):
        if m == L:
            complete_L = L
        else:
            break
    return complete_L


def assemble_in_memory(cache_dir: Path, lmax_limit: int | None = None, verbose: bool = False):
    """
    Assemble Gaunt checkpoints into a sparse tensor (optionally trimmed to lmax_limit) without saving.
    Returns (tensor, meta).
    """
    ckpts = _all_checkpoints(cache_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cache_dir}")
    total_ckpts = len(ckpts)
    print(f"Found {total_ckpts} checkpoint file(s) in {cache_dir}; scanning coverage...", flush=True)

    metas = [_load_meta(p) for p in ckpts]
    lmax_out, lmax_y, lmax_b = _validate_targets(metas)

    coverage = [-1] * (lmax_out + 1)
    for meta in metas:
        _coverage_from_meta(meta, coverage)
    complete_L = _compute_complete_L(coverage)
    if complete_L < 0:
        raise RuntimeError("No complete L found across checkpoints.")

    assemble_L = complete_L if lmax_limit is None else min(complete_L, lmax_limit)
    lmax_y_lim = lmax_y if lmax_limit is None else min(lmax_y, lmax_limit)
    lmax_b_lim = lmax_b if lmax_limit is None else min(lmax_b, lmax_limit)
    print(
        f"Coverage complete through L={complete_L} (targets: lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}); "
        f"assembling up to L={assemble_L}...",
        flush=True,
    )

    # Merge tensors up to assemble_L / lmax limits
    idx_all: List[List[int]] = [[], [], [], [], [], []]
    vals_all: List[float] = []
    shift_y = lmax_y - lmax_y_lim
    shift_b = lmax_b - lmax_b_lim
    for i, path in enumerate(ckpts, start=1):
        meta = _load_meta(path)
        start_L = int(meta.get("range_start_L", -1))
        end_L = int(meta.get("range_end_L", -1))
        if end_L < 0 or start_L > assemble_L:
            continue
        if verbose:
            print(f"[{i}/{total_ckpts}] Loading {path.name} (L range {start_L}-{end_L})...", flush=True)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        G_sparse = obj["G_sparse"] if "G_sparse" in obj else obj["G"].to_sparse()
        G_sparse = G_sparse.coalesce()
        idx = G_sparse.indices()
        vals = G_sparse.values()
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
        print(
            f"[{i}/{total_ckpts}] Added {added} entries from {path.name} "
            f"(L range {meta.get('range_start_L', '?')}-{meta.get('range_end_L', '?')})",
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
        "complete_L": complete_L,
        "assembled_L": assemble_L,
        "target_lmax_out": lmax_out,
        "target_lmax_y": lmax_y,
        "target_lmax_b": lmax_b,
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
