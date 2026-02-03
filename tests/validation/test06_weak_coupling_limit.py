"""
Test 6: Weak-coupling limit (self-consistent -> first-order as Y_s -> 0).

Setup:
- Choose a nontrivial spectral admittance pattern (random complex coefficients).
- Scale it by eps in {1e-3, 1e-4}.
- Drive with a simple ambient field (e.g., +X).

Expectation:
- Self-consistent and first-order solutions converge as eps -> 0.
- The difference between them scales linearly with eps (roughly diff_small ≈ diff_big * eps_small/eps_big).
"""
import os
import sys
from typing import Dict, List, Tuple

import torch

# Allow running from the tests/validation/ directory by adding repo root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from workflow.ambient_field.ambient_driver import build_ambient_driver_x
from europa_model import solvers
from europa_model.config import AmbientConfig, GridConfig, ModelConfig
from europa_model.simulation import Simulation
from workflow.data_objects.phasor_data import PhasorSimulation


def _clone_ps(base: PhasorSimulation) -> PhasorSimulation:
    """Deep-ish clone via serializable dict."""
    return PhasorSimulation.from_serializable(base.to_serializable())


def _max_abs_and_rel(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """Return max absolute and relative difference between two tensors."""
    diff = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
    max_abs = float(torch.max(diff).cpu())
    max_ref = float(
        torch.max(torch.stack([a.abs(), b.abs()])).cpu()
    ) if a.numel() > 0 else 1.0
    max_ref = max(max_ref, 1e-30)
    max_rel = max_abs / max_ref
    return max_abs, max_rel


def run_check(seed: int = 0) -> Tuple[Dict[float, float], List[str], List[str]]:
    """Run weak-coupling scaling check for a given random seed."""
    torch.manual_seed(seed)
    lmax = 3
    grid_cfg = GridConfig(nside=8, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver_x(grid_cfg)
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()

    # Base random complex admittance (nontrivial pattern)
    Y_base = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_base += 1j * torch.randn_like(Y_base)
    Y_base[0] = 0.0  # no DC to avoid uniform dominance

    eps_list = [1e-3, 1e-4]
    deltas: Dict[float, float] = {}
    notes: List[str] = []
    failures: List[str] = []

    for eps in eps_list:
        Y_scaled = eps * Y_base
        base = PhasorSimulation.from_model_and_grid(
            model=model,
            grid=sim.grid,
            solver_variant="",
            admittance_uniform=None,
            admittance_spectral=Y_scaled,
            B_radial=B_radial_spec.cpu(),
            period_sec=period_sec,
        )

        first = _clone_ps(base)
        solvers.solve_spectral_first_order_sim(first)
        selfc = _clone_ps(base)
        solvers.solve_spectral_self_consistent_sim(selfc)

        max_abs, _ = _max_abs_and_rel(first.K_toroidal, selfc.K_toroidal)
        deltas[eps] = max_abs
        notes.append(f"eps={eps:.0e}: max |K_first - K_sc| = {max_abs:.3e}")

    # Scaling check: diff should reduce roughly in proportion to eps
    if deltas[eps_list[1]] >= deltas[eps_list[0]] * (eps_list[1] / eps_list[0]) * 2.0:
        failures.append(
            f"Difference did not shrink ~linearly: "
            f"delta[{eps_list[0]:.0e}]={deltas[eps_list[0]]:.3e}, "
            f"delta[{eps_list[1]:.0e}]={deltas[eps_list[1]]:.3e}"
        )

    # Absolute smallness at the smallest eps
    if deltas[eps_list[-1]] > 1e-4:
        failures.append(f"Residual at eps={eps_list[-1]:.0e} too large: {deltas[eps_list[-1]]:.3e}")

    return deltas, notes, failures


def main() -> None:
    seeds = [0, 1, 2]
    overall_fail = False
    for seed in seeds:
        print(f"\nCase: seed = {seed}")
        deltas, notes, failures = run_check(seed)
        if failures:
            overall_fail = True
            print("[FAIL] weak-coupling limit:")
            for msg in failures:
                print(f"  - {msg}")
        else:
            print("[OK]   weak-coupling limit: self-consistent ~ first-order as eps -> 0")
        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
