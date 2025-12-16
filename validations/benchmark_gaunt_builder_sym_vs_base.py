"""
Benchmark Gaunt builder (compute-only): base pywigxjpf sparse vs symmetry-aware variant.
Uses in-memory builders only (no saving) and keeps timing clean (no prints inside timing window).
"""
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gaunt.gaunt_cache_wigxjpf import compute_gaunt_tensor_wigxjpf_sparse
from europa.legacy_variants.gaunt_cache_wigxjpf_sym import (
    compute_gaunt_tensor_wigxjpf_sym_sparse,
    compute_gaunt_tensor_wigxjpf_sym2_sparse,
)


def benchmark(lmax_out: int = 6, lmax_y: int | None = None, lmax_b: int | None = 1) -> None:
    lmax_y = lmax_out if lmax_y is None else lmax_y
    lmax_b = 1 if lmax_b is None else lmax_b

    t0 = time.perf_counter()
    G_base = compute_gaunt_tensor_wigxjpf_sparse(lmax_out=lmax_out, lmax_y=lmax_y, lmax_b=lmax_b, verbose=False)
    t_base = time.perf_counter() - t0

    t1 = time.perf_counter()
    G_sym = compute_gaunt_tensor_wigxjpf_sym_sparse(lmax_out=lmax_out, lmax_y=lmax_y, lmax_b=lmax_b, verbose=False)
    t_sym = time.perf_counter() - t1

    t2 = time.perf_counter()
    G_sym2 = compute_gaunt_tensor_wigxjpf_sym2_sparse(lmax_out=lmax_out, lmax_y=lmax_y, lmax_b=lmax_b, verbose=False)
    t_sym2 = time.perf_counter() - t2

    G_base_d = G_base.to_dense()
    G_sym_d = G_sym.to_dense()
    diff = (G_base_d - G_sym_d).abs()
    diff2 = (G_base_d - G_sym2.to_dense()).abs()
    max_diff = float(diff.max())
    max_diff2 = float(diff2.max())
    denom = float(torch.max(torch.stack([G_base_d.abs(), G_sym_d.abs(), G_sym2.to_dense().abs()])).cpu()) if G_base_d.numel() > 0 else 1.0
    max_rel = max_diff / denom if denom != 0 else 0.0
    max_rel2 = max_diff2 / denom if denom != 0 else 0.0

    print(f"lmax_out={lmax_out}, lmax_y={lmax_y}, lmax_b={lmax_b}")
    print(f"  base_time={t_base:.3f}s, sym_time={t_sym:.3f}s, sym2_time={t_sym2:.3f}s")
    print(f"  sym  max_diff={max_diff:.3e},  max_rel={max_rel:.3e}")
    print(f"  sym2 max_diff={max_diff2:.3e}, max_rel={max_rel2:.3e}")


def main() -> None:
    for n in range(6):
        lmax = 2**n
        benchmark(lmax_out=lmax, lmax_y=lmax, lmax_b=1)


if __name__ == "__main__":
    main()
