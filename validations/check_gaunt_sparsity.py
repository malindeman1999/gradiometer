"""
Report sparsity of saved Gaunt tensor (pywigxjpf cache).

Loads gaunt_wigxjpf.pt and reports fraction of zero entries and nonzero count.
"""
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gaunt.gaunt_cache_wigxjpf import _cache_path  # type: ignore


def main(cache_dir: str = "data/gaunt_cache_wigxjpf") -> None:
    path = _cache_path(cache_dir)
    if not path.exists():
        print(f"Cache not found at {path}; run save_gaunt_tensor_wigxjpf first.")
        return
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not (isinstance(obj, dict) and "G" in obj and "lmax" in obj):
        print(f"Invalid cache format at {path}")
        return
    G: torch.Tensor = obj["G"]
    total = G.numel()
    nonzero = torch.count_nonzero(G).item()
    zero_frac = 1.0 - nonzero / total
    print(f"Loaded Gaunt cache {path}")
    print(f"lmax={obj['lmax']}, shape={tuple(G.shape)}")
    print(f"total entries={total}, nonzero={nonzero}, zero_fraction={zero_frac:.6f}")


if __name__ == "__main__":
    main()
