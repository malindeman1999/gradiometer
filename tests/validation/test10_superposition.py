"""
Superposition test: linearity in the driver for fixed admittance.

Pick a random spectral admittance Y_s, two random drivers B1 and B2, and verify:
    solve(B1 + B2) == solve(B1) + solve(B2)
for both spectral first-order and spectral self-consistent solvers.
"""
import os
import sys
from typing import Dict, List, Tuple

import torch

# Allow running from the tests/validation/ directory by adding repo root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


def _check_superposition(a: PhasorSimulation, b: PhasorSimulation, ab: PhasorSimulation, abs_tol=1e-8, rel_tol=1e-3) -> List[str]:
    """Verify ab ≈ a + b for key fields."""
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
        fa = getattr(a, fname)
        fb = getattr(b, fname)
        fab = getattr(ab, fname)
        if fa is None or fb is None or fab is None:
            failures.append(f"{fname} missing")
            continue
        target = fa.to(torch.complex128) + fb.to(torch.complex128)
        max_abs, max_rel = _max_abs_and_rel(target, fab)
        if max_abs > abs_tol and max_rel > rel_tol:
            failures.append(f"{fname}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    return failures


def run_check(seed: int = 0) -> Tuple[Dict[str, List[str]], List[str]]:
    torch.manual_seed(seed)
    lmax = 5
    grid_cfg = GridConfig(nside=8, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=0.0, phase_rad=0.0, spatial_mode="custom")
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()

    # Random admittance and two random drivers
    Y_s = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_s += 1j * torch.randn_like(Y_s)
    B1 = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128) * 1e-6
    B1 += 1j * torch.randn_like(B1) * 1e-6
    B2 = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128) * 1e-6
    B2 += 1j * torch.randn_like(B2) * 1e-6
    Bsum = B1 + B2

    base_kwargs = dict(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s,
        period_sec=0.0,
    )

    def solve_with(B_radial: torch.Tensor) -> Tuple[PhasorSimulation, PhasorSimulation]:
        base = PhasorSimulation.from_model_and_grid(**base_kwargs, B_radial=B_radial)
        first = _clone_ps(base)
        solvers.solve_spectral_first_order_sim(first)
        selfc = _clone_ps(base)
        solvers.solve_spectral_self_consistent_sim(selfc)
        return first, selfc

    a_first, a_self = solve_with(B1)
    b_first, b_self = solve_with(B2)
    ab_first, ab_self = solve_with(Bsum)

    failures: Dict[str, List[str]] = {
        "first_order": _check_superposition(a_first, b_first, ab_first),
        "self_consistent": _check_superposition(a_self, b_self, ab_self),
    }
    notes = [f"seed={seed}", f"lmax={lmax}, nside={grid_cfg.nside}"]
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
                print(f"[FAIL] {key} superposition:")
                for msg in msgs:
                    print(f"  - {msg}")
            else:
                print(f"[OK]   {key} superposition holds")
        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
