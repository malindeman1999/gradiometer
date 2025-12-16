"""
Benchmark Gaunt tensor precompute variants (naive vs symmetry-aware).

Times:
- naive builder (fully dense loops with selection rule)
- symmetry-aware builder from solver_variant_precomputed_sym.load_gaunt_tensor

Each variant uses its own temporary cache directory to avoid mixing cached results.
"""
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import torch

# Allow running from validations/ by adding repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa.solvers import _gaunt_coeff, _build_gaunt_tensor
from europa.legacy_variants.solver_variant_precomputed_sym import load_gaunt_tensor as load_gaunt_tensor_sym
from europa.legacy_variants.solver_variant_precomputed_sym_adv import (
    load_gaunt_tensor_adv,
    build_gaunt_tensor_numpy_adv,
    load_gaunt_tensor_adv2,
    build_gaunt_tensor_numpy_adv2,
    load_gaunt_tensor_adv3,
    build_gaunt_tensor_numpy_adv3,
)


def build_gaunt_tensor_naive(lmax: int, device: str = "cpu") -> torch.Tensor:
    """Reference builder matching the original all-loops implementation."""
    G = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
        device=device,
    )
    for L in range(lmax + 1):
        for M in range(-L, L + 1):
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        for m in range(-l, l + 1):
                            if -M + m0 + m != 0:
                                continue
                            val = _gaunt_coeff(L, M, l0, m0, l, m)
                            G[L, L + M, l0, lmax + m0, l, lmax + m] = val
    return G


def build_gaunt_tensor_numpy_sym(lmax: int) -> torch.Tensor:
    """
    Symmetry-aware NumPy builder; mirrors the torch symmetry logic but in NumPy.
    Returns a torch tensor for consistent comparison.
    """
    G = np.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=np.float64,
    )
    for L in range(lmax + 1):
        for M in range(0, L + 1):  # non-negative M
            M_idx = L + M
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        # Parity: vanishes when L + l0 + l is odd
                        if (L + l0 + l) % 2 == 1:
                            continue
                        val = _gaunt_coeff(L, M, l0, m0, l, m)
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return torch.from_numpy(G)


def time_fn(fn: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    return elapsed, out


def clear_cache(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def benchmark(lmax_values: Iterable[int]) -> None:
    headers = [
        "lmax",
        "torch_naive",
        "torch_sym",
        "torch_adv",
        "torch_adv2",
        "torch_adv3",
        "numpy_naive",
        "numpy_sym",
        "numpy_adv",
        "numpy_adv2",
        "numpy_adv3",
        "max_diff(all)",
    ]
    rows = []
    print(" | ".join(f"{h:>12s}" for h in headers))
    print("-" * (len(headers) * 15))
    for lmax in lmax_values:
        t_naive, t_sym, t_adv, t_adv2, t_adv3 = None, None, None, None, None
        t_np_naive, t_np_sym, t_np_adv, t_np_adv2, t_np_adv3 = None, None, None, None, None
        cache_dir = Path(f"data/gaunt_cache_bench_lmax{lmax}")
        clear_cache(cache_dir)

        # Torch naive (no cache)
        t_naive, G_naive = time_fn(lambda: build_gaunt_tensor_naive(lmax, device="cpu"))

        # Torch symmetry-aware (builds once, caches; cache cleared above)
        t_sym, G_sym = time_fn(lambda: load_gaunt_tensor_sym(lmax, cache_dir=cache_dir, device="cpu"))

        # Torch advanced symmetry (more reuse)
        t_adv, G_adv = time_fn(lambda: load_gaunt_tensor_adv(lmax, cache_dir=cache_dir / "adv", device="cpu"))
        # Torch advanced2 symmetry (triangle + sign-flip canonical)
        t_adv2, G_adv2 = time_fn(lambda: load_gaunt_tensor_adv2(lmax, cache_dir=cache_dir / "adv2", device="cpu"))
        # Torch advanced3 symmetry (adds wigner caching and tighter bounds)
        t_adv3, G_adv3 = time_fn(lambda: load_gaunt_tensor_adv3(lmax, cache_dir=cache_dir / "adv3", device="cpu"))

        # NumPy naive builder from solvers.py
        t_np_naive, G_np_naive = time_fn(lambda: torch.from_numpy(_build_gaunt_tensor(lmax)))

        # NumPy symmetry-aware
        t_np_sym, G_np_sym = time_fn(lambda: build_gaunt_tensor_numpy_sym(lmax))

        # NumPy advanced symmetry
        t_np_adv, G_np_adv = time_fn(lambda: build_gaunt_tensor_numpy_adv(lmax))
        # NumPy advanced2 symmetry
        t_np_adv2, G_np_adv2 = time_fn(lambda: build_gaunt_tensor_numpy_adv2(lmax))
        # NumPy advanced3 symmetry
        t_np_adv3, G_np_adv3 = time_fn(lambda: build_gaunt_tensor_numpy_adv3(lmax))

        # Validate equality (float64; tolerance zero expected)
        max_diffs = [
            float((G_naive - G_sym).abs().max()),
            float((G_naive - G_adv).abs().max()),
            float((G_naive - G_adv2).abs().max()),
            float((G_naive - G_adv3).abs().max()),
            float((G_naive - G_np_naive).abs().max()),
            float((G_naive - G_np_sym).abs().max()),
            float((G_naive - G_np_adv).abs().max()),
            float((G_naive - G_np_adv2).abs().max()),
            float((G_naive - G_np_adv3).abs().max()),
        ]
        max_diff_all = max(max_diffs)

        row = [
            f"{lmax:>4d}",
            f"{t_naive:>10.3f}",
            f"{t_sym:>10.3f}",
            f"{t_adv:>10.3f}",
            f"{t_adv2:>11.3f}",
            f"{t_adv3:>11.3f}",
            f"{t_np_naive:>11.3f}",
            f"{t_np_sym:>10.3f}",
            f"{t_np_adv:>10.3f}",
            f"{t_np_adv2:>10.3f}",
            f"{t_np_adv3:>10.3f}",
            f"{max_diff_all:>12.1e}",
        ]
        rows.append([lmax, t_naive, t_sym, t_adv, t_adv2, t_adv3,
                     t_np_naive, t_np_sym, t_np_adv, t_np_adv2, t_np_adv3,
                     max_diff_all])
        print(" | ".join(row))

        # Clean up cache to avoid clutter
        clear_cache(cache_dir)

    # Write HTML table for easier viewing
    out_path = Path("html/benchmark_gaunt_precompute.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def fmt(val, digits=3):
        return f"{val:.{digits}f}"
    html_rows = []
    for r in rows:
        # Identify the fastest timing among the timing columns (indices 1..10)
        time_vals = r[1:11]
        min_time = min(time_vals)
        tds = [f"<td>{r[0]}</td>"]
        for i, val in enumerate(time_vals, start=1):
            if val == min_time:
                tds.append(f'<td style="color:green;font-weight:bold">{fmt(val)}</td>')
            else:
                tds.append(f"<td>{fmt(val)}</td>")
        # single max diff column (index 11)
        tds.append(f"<td>{r[11]:.1e}</td>")
        html_rows.append("<tr>" + "".join(tds) + "</tr>")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gaunt Precompute Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1000px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
    th {{ background: #f2f2f2; }}
    td:first-child, th:first-child {{ text-align: center; }}
  </style>
  <body>
    <h2>Gaunt Precompute Benchmark</h2>
    <p>Timing columns are in seconds; max_diff columns report max absolute element-wise difference vs. torch_naive.</p>
    <table>
      <tr>{"".join([f"<th>{h}</th>" for h in headers])}</tr>
      {"".join(html_rows)}
    </table>
  </body>
</html>
"""
    out_path.write_text(html)
    print(f"\nHTML table written to {out_path}")


def main() -> None:
    # Keep lmax modest so runtime stays reasonable; adjust as needed.
    lmax_values = [4, 5, 6, 7, 8]
    benchmark(lmax_values)


if __name__ == "__main__":
    main()
