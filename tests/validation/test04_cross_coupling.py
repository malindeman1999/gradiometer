"""
Test 4: Cross-coupling with Y20 admittance and Y10 driver.

Admittance: Y_s = Y_2,0 only.
Driver:     B_r = Y_1,0.

Expected (first-order):
- Outputs only at m = 0.
- Allowed degrees l in {1, 3}; l = 2 is forbidden by 3j parity.
- Ratio K_{1,0} / K_{3,0} matches Gaunt G(1,0;2,0;1,0) / G(3,0;2,0;1,0).

Expected (self-consistent):
- m-selection preserved (m = 0 only); higher l with m=0 may appear via feedback.
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


def _max_off_modes(tensor: torch.Tensor, allowed: set[tuple[int, int]], lmax: int) -> float:
    """Maximum magnitude outside allowed (l, m)."""
    mask = torch.ones_like(tensor, dtype=torch.bool)
    for l, m in allowed:
        mask[..., l, lmax + m] = False
    if torch.all(~mask):
        return 0.0
    return float(tensor.abs()[mask].max().cpu())


def _get_mode(tensor: torch.Tensor, l: int, m: int, lmax: int) -> complex:
    """Return a single (l, m) coefficient."""
    return complex(tensor[l, lmax + m].item())


def _gaunt_ratio() -> complex:
    """Gaunt ratio G(L=1)/G(L=3) for Y20 x Y10."""
    g1 = solvers._gaunt_coeff(1, 0, 2, 0, 1, 0)  # type: ignore[attr-defined]
    g3 = solvers._gaunt_coeff(3, 0, 2, 0, 1, 0)  # type: ignore[attr-defined]
    return complex(g1 / g3)


def _check_first_order(sim: PhasorSimulation, lmax: int, abs_tol=1e-9, rel_tol=1e-3) -> List[str]:
    """Check first-order sparsity (l in {1,3}, m=0) and Gaunt ratio."""
    failures: List[str] = []
    allowed = {(1, 0), (3, 0)}
    fields = ["K_toroidal", "B_pol_emit", "B_rad_emit"]
    for fname in fields:
        tensor = getattr(sim, fname)
        if tensor is None:
            failures.append(f"{fname} missing")
            continue
        off = _max_off_modes(tensor, allowed, lmax)
        tol = max(abs_tol, rel_tol * float(tensor.abs().max().cpu()))
        if off > tol:
            failures.append(f"{fname}: off-mode max {off:.3e} exceeds tol {tol:.3e}")

    K = sim.K_toroidal
    if K is not None:
        k1 = _get_mode(K, 1, 0, lmax)
        k3 = _get_mode(K, 3, 0, lmax)
        if abs(k3) <= abs_tol:
            failures.append("K_toroidal: |K_3,0| too small for ratio check")
        else:
            ratio = k1 / k3
            expected = _gaunt_ratio()
            diff = abs(ratio - expected)
            tol_ratio = max(abs_tol, rel_tol * abs(expected))
            if diff > tol_ratio:
                failures.append(
                    f"K_toroidal: ratio K10/K30={ratio:.6e} differs from Gaunt {expected:.6e} by {diff:.2e}"
                )
    else:
        failures.append("K_toroidal missing")
    return failures


def _check_self_consistent(sim: PhasorSimulation, lmax: int, abs_tol=1e-9, rel_tol=1e-3) -> List[str]:
    """
    Check self-consistent solution preserves m-selection (only m=0).
    Allows all l with m=0 since feedback can populate higher degrees.
    """
    failures: List[str] = []
    allowed = {(l, 0) for l in range(lmax + 1)}
    fields = ["K_toroidal", "B_pol_emit", "B_rad_emit"]
    for fname in fields:
        tensor = getattr(sim, fname)
        if tensor is None:
            failures.append(f"{fname} missing")
            continue
        off = _max_off_modes(tensor, allowed, lmax)
        tol = max(abs_tol, rel_tol * float(tensor.abs().max().cpu()))
        if off > tol:
            failures.append(f"{fname}: off-mode max {off:.3e} exceeds tol {tol:.3e}")
    return failures


def run_check(ocean_thickness_m: float) -> Tuple[Dict[str, List[str]], List[str]]:
    """Run the cross-coupling test for a given ocean thickness."""
    grid_cfg = GridConfig(nside=4, lmax=4, radius_m=1.56e6, device="cpu", ocean_thickness_m=ocean_thickness_m)
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=0.0, phase_rad=0.0, spatial_mode="custom")
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    sigma_2d = float(sim.grid.surface_conductivity_s)

    # Admittance: only Y_2,0 non-zero
    Y_s_spectral = torch.zeros((grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1), dtype=torch.float64)
    Y_s_spectral[2, grid_cfg.lmax] = sigma_2d

    # Driver: B_r = Y_1,0
    B_radial_spec = torch.zeros((grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1), dtype=torch.complex128)
    B_radial_spec[1, grid_cfg.lmax] = 1e-6

    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec.cpu(),
        period_sec=0.0,
    )

    first = _clone_ps(base)
    solvers.solve_spectral_first_order_sim(first)

    selfc = _clone_ps(base)
    solvers.solve_spectral_self_consistent_sim(selfc)

    failures: Dict[str, List[str]] = {
        "first_order": _check_first_order(first, grid_cfg.lmax),
        "self_consistent": _check_self_consistent(selfc, grid_cfg.lmax),
    }
    notes: List[str] = []
    notes.append(f"B_r driver peak |Y1,0| = {float(B_radial_spec.abs().max().cpu()):.3e} T")
    notes.append(f"Gaunt ratio G(1,0;2,0;1,0)/G(3,0;2,0;1,0) = {_gaunt_ratio():.6e}")
    return failures, notes


def main() -> None:
    cases = [
        (10.0, "expected weak response"),
        (100_000.0, "expected strong response"),
    ]
    overall_fail = False
    for thickness, desc in cases:
        print(f"\nCase: ocean_thickness = {thickness:.0f} m ({desc})")
        failures, notes = run_check(thickness)
        any_fail = any(msgs for msgs in failures.values() if msgs)
        overall_fail = overall_fail or any_fail

        for key, msgs in failures.items():
            if key == "first_order":
                label = "first_order (Y20 x Y10 -> l=1,3; m=0)"
            else:
                label = "self_consistent (m=0 selection)"
            if msgs:
                print(f"[FAIL] {label}:")
                for msg in msgs:
                    print(f"  - {msg}")
            else:
                print(f"[OK]   {label} passed")
        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
