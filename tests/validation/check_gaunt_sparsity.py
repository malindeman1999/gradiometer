"""
Report sparsity of Gaunt tensor assembled from per-L checkpoints.
"""
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gaunt.assemble_gaunt_checkpoints import assemble_in_memory


def main(cache_dir: str = "gaunt/data/gaunt_cache_wigxjpf") -> None:
    path = Path(cache_dir)
    ckpts = sorted(path.glob("gaunt_wigxjpf_L*.pt"))
    if not ckpts:
        print(f"No Gaunt checkpoints found under {path}; run run_gaunt_calculator.py first.")
        return

    G_sparse, meta = assemble_in_memory(cache_dir=path, verbose=True)
    total = 1
    for dim in G_sparse.shape:
        total *= dim
    nonzero = int(G_sparse._nnz())
    zero_frac = 1.0 - nonzero / total
    print(f"Assembled Gaunt tensor from {len(ckpts)} checkpoints in {path}")
    print(f"lmax_out={meta.get('lmax_out')}, lmax_y={meta.get('lmax_y')}, lmax_b={meta.get('lmax_b')}")
    print(f"shape={tuple(G_sparse.shape)}")
    print(f"total entries={total}, nonzero={nonzero}, zero_fraction={zero_frac:.6f}")


if __name__ == "__main__":
    main()
