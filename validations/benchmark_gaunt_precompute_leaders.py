"""
Benchmark only the "leader" Gaunt precompute variants:
- torch_adv (permutation symmetry)
- torch_adv2 (triangle + sign-invariant cache)
- numpy_adv
- numpy_adv2

Uses torch_adv as the reference for max-diff comparisons.
"""
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Tuple

import torch

# Allow running from validations/ by adding repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa.legacy_variants.solver_variant_precomputed_sym_adv import (
    build_gaunt_tensor_torch_adv,
    build_gaunt_tensor_torch_adv2,
    build_gaunt_tensor_numpy_adv,
    build_gaunt_tensor_numpy_adv2,
)


def time_fn(fn: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    return elapsed, out


def clear_cache(path: Path) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)


def benchmark(lmax_values: Iterable[int]) -> None:
    headers = [
        "lmax",
        "torch_adv",
        "torch_adv2",
        "numpy_adv",
        "numpy_adv2",
        "max_diff(all vs torch_adv)",
    ]
    print(" | ".join(f"{h:>18s}" for h in headers))
    print("-" * (len(headers) * 20))
    rows = []

    for lmax in lmax_values:
        # Reference: torch_adv (build in-memory; no cache write)
        t_t_adv, G_t_adv = time_fn(lambda: build_gaunt_tensor_torch_adv(lmax, device="cpu"))

        t_t_adv2, G_t_adv2 = time_fn(lambda: build_gaunt_tensor_torch_adv2(lmax, device="cpu"))
        t_np_adv, G_np_adv = time_fn(lambda: build_gaunt_tensor_numpy_adv(lmax))
        t_np_adv2, G_np_adv2 = time_fn(lambda: build_gaunt_tensor_numpy_adv2(lmax))

        max_diff = max(
            float((G_t_adv - G_t_adv2).abs().max()),
            float((G_t_adv - G_np_adv).abs().max()),
            float((G_t_adv - G_np_adv2).abs().max()),
        )

        row = [
            f"{lmax:>4d}",
            f"{t_t_adv:>10.3f}",
            f"{t_t_adv2:>10.3f}",
            f"{t_np_adv:>10.3f}",
            f"{t_np_adv2:>10.3f}",
            f"{max_diff:>18.1e}",
        ]
        print(" | ".join(row))
        rows.append((lmax, t_t_adv, t_t_adv2, t_np_adv, t_np_adv2, max_diff))

    # Write HTML table
    out_path = Path("html/benchmark_gaunt_precompute_leaders.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(val: float) -> str:
        return f"{val:.3f}"

    html_rows = []
    for (lmax, t_adv, t_adv2, n_adv, n_adv2, mdiff) in rows:
        time_vals = [t_adv, t_adv2, n_adv, n_adv2]
        min_time = min(time_vals)
        cells = [f"<td>{lmax}</td>"]
        for v in time_vals:
            if v == min_time:
                cells.append(f'<td style="color:green;font-weight:bold">{fmt(v)}</td>')
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        cells.append(f"<td>{mdiff:.1e}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gaunt Precompute Benchmark (Leaders)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
    th {{ background: #f2f2f2; }}
    td:first-child, th:first-child {{ text-align: center; }}
  </style>
</head>
<body>
  <h2>Gaunt Precompute Benchmark (Leader Variants)</h2>
  <p>Timings in seconds; max_diff is max absolute difference vs. torch_adv.</p>
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
    lmax_values = [13, 14]
    benchmark(lmax_values)


if __name__ == "__main__":
    main()
