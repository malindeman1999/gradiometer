"""
Test 7: Linearity in driver amplitude.

Setup:
- Fix spectral admittance (random complex pattern).
- Scale driver amplitude by factors in {0.1, 1, 10}.

Expectation:
- Outputs (E, K, emitted B) scale linearly with driver amplitude for both
  first-order and self-consistent spectral solvers.
"""
import os
import sys
from typing import Dict, List, Tuple

import torch

# Allow running from the validations/ directory by adding repo root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ambient_driver import build_ambient_driver_x
from europa import solvers
from europa.config import AmbientConfig, GridConfig, ModelConfig
from europa.simulation import Simulation
from phasor_data import PhasorSimulation


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


def _check_scaling(
    ref: PhasorSimulation,
    test: PhasorSimulation,
    scale: float,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-3,
) -> List[str]:
    """Check test fields ≈ scale * ref fields."""
    failures: List[str] = []
    fields = [
        "E_toroidal",
        "K_toroidal",
        "K_poloidal",
        "B_tor_emit",
        "B_pol_emit",
        "B_rad_emit",
    ]
    for fname in fields:
        a = getattr(ref, fname)
        b = getattr(test, fname)
        if a is None or b is None:
            failures.append(f"{fname} missing")
            continue
        target = a.to(torch.complex128) * scale
        max_abs, max_rel = _max_abs_and_rel(target, b)
        if max_abs > abs_tol and max_rel > rel_tol:
            failures.append(f"{fname}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    return failures


def run_check(seed: int = 0) -> Tuple[Dict[str, List[str]], List[str]]:
    torch.manual_seed(seed)
    lmax = 3
    grid_cfg = GridConfig(nside=8, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg, B_radial_spec_base, period_sec = build_ambient_driver_x(grid_cfg)
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()

    # Random spectral admittance (nontrivial), keep l=0 small to avoid dominance
    Y_s = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_s += 1j * torch.randn_like(Y_s)
    Y_s[0] *= 0.1

    # Reference solve at scale=1
    base_ref = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s,
        B_radial=B_radial_spec_base.cpu(),
        period_sec=period_sec,
    )
    ref_first = _clone_ps(base_ref)
    solvers.solve_spectral_first_order_sim(ref_first)
    ref_self = _clone_ps(base_ref)
    solvers.solve_spectral_self_consistent_sim(ref_self)

    scales = [0.1, 10.0]
    failures: Dict[str, List[str]] = {}
    notes: List[str] = [f"seed={seed}"]

    for scale in scales:
        B_scaled = B_radial_spec_base * scale
        base = PhasorSimulation.from_model_and_grid(
            model=model,
            grid=sim.grid,
            solver_variant="",
            admittance_uniform=None,
            admittance_spectral=Y_s,
            B_radial=B_scaled.cpu(),
            period_sec=period_sec,
        )
        test_first = _clone_ps(base)
        solvers.solve_spectral_first_order_sim(test_first)
        test_self = _clone_ps(base)
        solvers.solve_spectral_self_consistent_sim(test_self)

        key_first = f"first_order scale={scale}"
        key_self = f"self_consistent scale={scale}"
        failures[key_first] = _check_scaling(ref_first, test_first, scale)
        failures[key_self] = _check_scaling(ref_self, test_self, scale)

    return failures, notes


def main() -> None:
    seeds = [0, 1, 2]
    overall_fail = False
    for seed in seeds:
        print(f"\nCase: seed = {seed}")
        failures, notes = run_check(seed)
        any_fail = any(msgs for msgs in failures.values() if msgs)
        overall_fail = overall_fail or any_fail

        for key, msgs in failures.items():
            if msgs:
                print(f"[FAIL] {key}:")
                for msg in msgs:
                    print(f"  - {msg}")
            else:
                print(f"[OK]   {key} linear scaling")
        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
