"""
Verify Gaunt shell checkpoints against an in-memory sparse build.

Computes Gaunt coefficients up to LMAX in memory, assembles shell checkpoints
from disk up to the same LMAX, and compares the sparse tensors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Allow running this file directly (python gaunt/verify_gaunt_shells.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaunt.assemble_gaunt_checkpoints import assemble_in_memory
from gaunt.helpers.gaunt_cache_wigxjpf import compute_gaunt_tensor_wigxjpf_sparse


def _compare_sparse(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> tuple[bool, str]:
    a = a.coalesce()
    b = b.coalesce()
    if a.shape != b.shape:
        return False, f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    if a._nnz() != b._nnz():
        return False, f"NNZ mismatch: {a._nnz()} vs {b._nnz()}"
    if not torch.equal(a.indices(), b.indices()):
        return False, "Index mismatch between sparse tensors."
    if not torch.allclose(a.values(), b.values(), atol=atol, rtol=rtol):
        max_diff = torch.max(torch.abs(a.values() - b.values())).item()
        return False, f"Value mismatch; max abs diff={max_diff}"
    return True, "Sparse tensors match."


def _build_value_map(a: torch.Tensor) -> dict[tuple[int, int, int, int, int, int], float]:
    a = a.coalesce()
    idx = a.indices()
    vals = a.values()
    out: dict[tuple[int, int, int, int, int, int], float] = {}
    for i in range(vals.numel()):
        key = (
            int(idx[0, i].item()),
            int(idx[1, i].item()),
            int(idx[2, i].item()),
            int(idx[3, i].item()),
            int(idx[4, i].item()),
            int(idx[5, i].item()),
        )
        out[key] = float(vals[i].item())
    return out


def _print_sample_matches(
    a: torch.Tensor,
    b: torch.Tensor,
    lmax: int,
    nonzero_samples: int,
    zero_samples: int,
    seed: int,
) -> None:
    a = a.coalesce()
    b = b.coalesce()
    idx = a.indices()
    total = a.values().numel()
    if total == 0 and zero_samples == 0:
        print("No entries to sample.")
        return
    map_a = _build_value_map(a)
    map_b = _build_value_map(b)
    keys = list(map_a.keys())
    gen = torch.Generator()
    gen.manual_seed(seed)
    nonzero_count = min(nonzero_samples, len(keys))
    if nonzero_count > 0:
        picks = torch.randperm(len(keys), generator=gen)[:nonzero_count]
        print("Nonzero samples:")
        for i, p in enumerate(picks.tolist(), start=1):
            key = keys[p]
            va = map_a.get(key, 0.0)
            vb = map_b.get(key, 0.0)
            print(
                f"[{i}] (L,M,l0,m0,l,m)=({key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{key[5]}) "
                f"mem={va:.12e} shell={vb:.12e}"
            )
    if zero_samples > 0:
        print("Zero samples:")
        seen = set(keys)
        tries = 0
        found = 0
        while found < zero_samples and tries < zero_samples * 100:
            L = int(torch.randint(0, lmax + 1, (1,), generator=gen).item())
            M = int(torch.randint(0, 2 * lmax + 1, (1,), generator=gen).item())
            l0 = int(torch.randint(0, lmax + 1, (1,), generator=gen).item())
            m0 = int(torch.randint(0, 2 * lmax + 1, (1,), generator=gen).item())
            l = int(torch.randint(0, lmax + 1, (1,), generator=gen).item())
            m = int(torch.randint(0, 2 * lmax + 1, (1,), generator=gen).item())
            key = (L, M, l0, m0, l, m)
            if key in seen:
                tries += 1
                continue
            va = map_a.get(key, 0.0)
            vb = map_b.get(key, 0.0)
            found += 1
            print(
                f"[{found}] (L,M,l0,m0,l,m)=({L},{M},{l0},{m0},{l},{m}) "
                f"mem={va:.12e} shell={vb:.12e}"
            )
            tries += 1
        if found < zero_samples:
            print(f"Only found {found} zero samples after {tries} attempts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Gaunt shell checkpoints against in-memory build.")
    parser.add_argument("--lmax", type=int, default=10, help="Max L for verification (default: 10).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/gaunt_cache_wigxjpf"),
        help="Directory containing shell checkpoint files.",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for value comparison.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for value comparison.")
    parser.add_argument("--nonzero-samples", type=int, default=3, help="Number of nonzero coefficients to print.")
    parser.add_argument("--zero-samples", type=int, default=3, help="Number of zero coefficients to print.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    args = parser.parse_args()

    print(f"Computing in-memory sparse Gaunt tensor up to L={args.lmax}...")
    G_mem = compute_gaunt_tensor_wigxjpf_sparse(lmax=args.lmax, verbose=True)

    print(f"Assembling shell checkpoints from {args.cache_dir} up to L={args.lmax}...")
    G_shell, meta = assemble_in_memory(args.cache_dir, lmax_limit=args.lmax, verbose=True, plot=False)

    ok, msg = _compare_sparse(G_mem, G_shell, atol=args.atol, rtol=args.rtol)
    if ok:
        print("OK:", msg)
        _print_sample_matches(
            G_mem,
            G_shell,
            lmax=args.lmax,
            nonzero_samples=args.nonzero_samples,
            zero_samples=args.zero_samples,
            seed=args.seed,
        )
    else:
        print("FAIL:", msg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
