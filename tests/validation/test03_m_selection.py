"""
Test 3: m-selection with Y10 admittance and Y1,±1 driver.

Admittance: Y_s = Y_1,0 only.
Drivers: (a) B_r = Y_1,+1, (b) B_r = Y_1,-1.

Expected (first-order):
- Outputs only at M = m_driver.
- Allowed degrees l in {1, 2}; l = 0 is forbidden by m-selection.
- Ratio K_{1,m} / K_{2,m} matches Gaunt G(1,m;1,0;1,m) / G(2,m;1,0;1,m).

Expected (self-consistent):
- m-selection preserved (only the driven m). Higher l can appear via feedback.
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


def _gaunt_ratio(m: int) -> complex:
    """Gaunt coefficient ratio G(L=1)/G(L=2) for Y10 x Y1,m."""
    g1 = solvers._gaunt_coeff(1, m, 1, 0, 1, m)  # type: ignore[attr-defined]
    g2 = solvers._gaunt_coeff(2, m, 1, 0, 1, m)  # type: ignore[attr-defined]
    return complex(g1 / g2)


def _check_first_order(sim: PhasorSimulation, lmax: int, m: int, abs_tol=1e-9, rel_tol=1e-3) -> List[str]:
    """Check first-order sparsity (l in {1,2}, fixed m) and Gaunt ratio."""
    failures: List[str] = []
    allowed = {(1, m), (2, m)}
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

    # Gaunt ratio on K_toroidal
    K = sim.K_toroidal
    if K is not None:
        k1 = _get_mode(K, 1, m, lmax)
        k2 = _get_mode(K, 2, m, lmax)
        if abs(k2) <= abs_tol:
            failures.append("K_toroidal: |K_2,m| too small for ratio check")
        else:
            ratio = k1 / k2
            expected = _gaunt_ratio(m)
            diff = abs(ratio - expected)
            tol_ratio = max(abs_tol, rel_tol * abs(expected))
            if diff > tol_ratio:
                failures.append(
                    f"K_toroidal: ratio K1{m:+d}/K2{m:+d}={ratio:.6e} differs from Gaunt {expected:.6e} by {diff:.2e}"
                )
    else:
        failures.append("K_toroidal missing")
    return failures


def _check_self_consistent(sim: PhasorSimulation, lmax: int, m: int, abs_tol=1e-9, rel_tol=1e-3) -> List[str]:
    """
    Check self-consistent solution preserves m-selection (only driven m).
    Allows all l with that m since feedback can populate higher degrees.
    """
    failures: List[str] = []
    allowed = {(l, m) for l in range(lmax + 1)}
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


def _build_driver(lmax: int, m: int, amplitude_t: float = 1e-6, device: str = "cpu") -> torch.Tensor:
    """Construct B_r spectrum with only Y_1,m."""
    B = torch.zeros((lmax + 1, 2 * lmax + 1), device=device, dtype=torch.complex128)
    B[1, lmax + m] = amplitude_t
    return B


def run_check(ocean_thickness_m: float) -> Tuple[Dict[str, Dict[int, List[str]]], List[str]]:
    """Run m-selection test for m=+1 and m=-1 at a given ocean thickness."""
    grid_cfg = GridConfig(nside=4, lmax=3, radius_m=1.56e6, device="cpu", ocean_thickness_m=ocean_thickness_m)
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=0.0, phase_rad=0.0, spatial_mode="custom")
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    sigma_2d = float(sim.grid.surface_conductivity_s)

    # Admittance: only Y_1,0 non-zero
    Y_s_spectral = torch.zeros((grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1), dtype=torch.float64)
    Y_s_spectral[1, grid_cfg.lmax] = sigma_2d

    results: Dict[str, Dict[int, List[str]]] = {"first_order": {}, "self_consistent": {}}
    notes: List[str] = []

    for m in (+1, -1):
        B_radial_spec = _build_driver(grid_cfg.lmax, m, device="cpu")
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

        results["first_order"][m] = _check_first_order(first, grid_cfg.lmax, m)
        results["self_consistent"][m] = _check_self_consistent(selfc, grid_cfg.lmax, m)
        notes.append(f"m={m:+d}: |B_r| peak = {float(B_radial_spec.abs().max().cpu()):.3e} T")
        notes.append(f"m={m:+d}: Gaunt ratio G(1,m)/G(2,m) = {_gaunt_ratio(m):.6e}")

    return results, notes


def main() -> None:
    cases = [
        (10.0, "expected weak response"),
        (100_000.0, "expected strong response"),
    ]
    overall_fail = False
    for thickness, desc in cases:
        print(f"\nCase: ocean_thickness = {thickness:.0f} m ({desc})")
        results, notes = run_check(thickness)
        any_fail = any(msgs for group in results.values() for msgs in group.values() if msgs)
        overall_fail = overall_fail or any_fail

        for variant, m_map in results.items():
            for m, msgs in m_map.items():
                if variant == "first_order":
                    label = f"{variant} (Y10 x Y1{m:+d} -> m={m:+d}, l in {{1,2}})"
                else:
                    label = f"{variant} (m-selection, m={m:+d})"
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
