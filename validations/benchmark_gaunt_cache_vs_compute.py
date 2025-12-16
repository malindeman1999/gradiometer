"""
Benchmark Gaunt tensor compute vs load for the pywigxjpf backend.

Measures:
- compute_gaunt_tensor_wigxjpf(lmax)
- save_gaunt_tensor_wigxjpf(lmax, cache_dir) (first run: compute+save)
- load_gaunt_tensor_wigxjpf(lmax, cache_dir) (subsequent load)
"""
import os
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

import torch

# Allow running from validations/ by adding repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gaunt.gaunt_cache_wigxjpf import (
    compute_gaunt_tensor_wigxjpf,
    save_gaunt_tensor_wigxjpf,
    load_gaunt_tensor_wigxjpf,
)


def time_fn(fn: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    return elapsed, out


def benchmark(lmax_values) -> None:
    print(f"{'lmax':>4s} | {'compute':>10s} | {'save':>10s} | {'load':>10s}")
    print("-" * 50)
    for lmax in lmax_values:
        cache_dir = Path(f"data/gaunt_cache_wigxjpf_bench_lmax{lmax}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "gaunt_wigxjpf.pt"
        if cache_path.exists():
            cache_path.unlink()

        t_compute, Gc = time_fn(lambda: compute_gaunt_tensor_wigxjpf(lmax))
        t_save, Gs = time_fn(lambda: save_gaunt_tensor_wigxjpf(lmax, cache_dir=cache_dir))
        t_load, Gl = time_fn(lambda: load_gaunt_tensor_wigxjpf(lmax, cache_dir=cache_dir))

        max_diff = float((Gc - Gl).abs().max())
        print(f"{lmax:4d} | {t_compute:10.3f} | {t_save:10.3f} | {t_load:10.3f}  (max_diff vs compute={max_diff:.3e})")


def main() -> None:
    lmax_values = [4, 5, 6]
    benchmark(lmax_values)


if __name__ == "__main__":
    main()
