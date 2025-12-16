"""
Verify spectral solver with uniform admittance matches the uniform solver.
Checks both first-order and self-consistent variants on a simple +X driver.
"""
import sys
import os
from typing import Dict, List, Tuple

import torch

# Allow running from the validations/ directory by adding repo root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa.config import GridConfig, ModelConfig
from europa.simulation import Simulation
from europa import solvers
from ambient_driver import build_ambient_driver_x
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


def _compare_fields(label: str, a: PhasorSimulation, b: PhasorSimulation, abs_tol=1e-6, rel_tol=1e-2) -> List[str]:
    """Compare key phasor fields; return list of failures for this pair."""
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
        if fa is None or fb is None:
            failures.append(f"{label}:{fname} missing")
            continue
        max_abs, max_rel = _max_abs_and_rel(fa, fb)
        if max_abs > abs_tol and max_rel > rel_tol:
            failures.append(f"{label}:{fname} max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    return failures


def _emitted_to_ambient_ratio(sim: PhasorSimulation) -> float:
    """Return max |B_emit| / max |B_ambient| to detect weak self-field regime."""
    if sim.B_rad_emit is None or sim.B_radial is None:
        return float("inf")
    b_emit = torch.max(sim.B_rad_emit.abs()).item()
    b_amb = torch.max(sim.B_radial.abs()).item()
    if b_amb <= 0:
        return float("inf")
    return b_emit / b_amb


def run_check(ocean_thickness_m: float) -> Tuple[Dict[str, List[str] | None], List[str]]:
    """Run uniform vs spectral comparisons for a given ocean thickness and collect failures/notes."""
    grid_cfg = GridConfig(nside=4, lmax=4, radius_m=1.56e6, device="cpu", ocean_thickness_m=ocean_thickness_m)
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver_x(grid_cfg)
    omega = ambient_cfg.omega_jovian
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    sigma_2d = float(sim.grid.surface_conductivity_s)

    Y_s_spectral = torch.zeros((grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1), dtype=torch.float64)
    Y_s_spectral[0, grid_cfg.lmax] = sigma_2d * (4 * torch.pi) ** 0.5

    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=sigma_2d,
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec.cpu(),
        period_sec=period_sec,
    )

    # Uniform solvers
    u_first = _clone_ps(base)
    u_first.admittance_spectral = None
    solvers.solve_uniform_first_order_sim(u_first)

    u_sc = _clone_ps(base)
    u_sc.admittance_spectral = None
    solvers.solve_uniform_self_consistent_sim(u_sc)

    # Spectral solvers with uniform Y00 only
    s_first = _clone_ps(base)
    solvers.solve_spectral_first_order_sim(s_first)

    s_sc = _clone_ps(base)
    solvers.solve_spectral_self_consistent_sim(s_sc)

    failures: Dict[str, List[str] | None] = {
        "first_order": None,
        "self_consistent": None,
    }
    notes: List[str] = []

    # Verify admittance is uniform (only Y00 non-zero) before comparing spectral vs uniform
    nonzero_mask = (Y_s_spectral.abs() > 0).nonzero(as_tuple=False)
    if nonzero_mask.shape[0] == 1 and tuple(nonzero_mask[0].tolist()) == (0, grid_cfg.lmax):
        notes.append("admittance is uniform (Y00 only)")
        failures["first_order"] = _compare_fields("first_order", u_first, s_first)
        failures["self_consistent"] = _compare_fields("self_consistent", u_sc, s_sc)
    else:
        notes.append("admittance is NOT uniform (additional harmonics present)")
        failures["first_order"] = None
        failures["self_consistent"] = None
        notes.append("uniform vs spectral skipped: admittance not purely Y00")

    # In weak self-field regime, self-consistent should match first-order
    weak_ratio_thresh = 1e-3
    tol_abs = 1e-6
    tol_rel = 1e-2

    ratio_u = _emitted_to_ambient_ratio(u_sc)
    if ratio_u < weak_ratio_thresh:
        failures["weak_uniform"] = _compare_fields("weak_uniform", u_first, u_sc, abs_tol=tol_abs, rel_tol=tol_rel)
    else:
        failures["weak_uniform"] = None
        notes.append(
            f"weak_uniform: skipped (emitted/ambient={ratio_u:.3e} >= {weak_ratio_thresh}; not in weak-field regime)"
        )

    ratio_s = _emitted_to_ambient_ratio(s_sc)
    if ratio_s < weak_ratio_thresh:
        failures["weak_spectral"] = _compare_fields("weak_spectral", s_first, s_sc, abs_tol=tol_abs, rel_tol=tol_rel)
    else:
        failures["weak_spectral"] = None
        notes.append(
            f"weak_spectral: skipped (emitted/ambient={ratio_s:.3e} >= {weak_ratio_thresh}; not in weak-field regime)"
        )
    return failures, notes


def main() -> None:
    cases = [
        (10.0, "expected weak response"),
        (100_000.0, "expected strong response"),
    ]
    overall_fail = False
    label_map = {
        "first_order": "uniform vs spectral (first-order)",
        "self_consistent": "uniform vs spectral (self-consistent)",
        "weak_uniform": "weak response, uniform admittance: first order matches self-consistent",
        "weak_spectral": "weak response, spectral admittance: first order matches self-consistent",
    }

    for thickness, desc in cases:
        print(f"\nCase: ocean_thickness = {thickness:.0f} m ({desc})")
        failures, notes = run_check(thickness)
        any_fail = any(msgs for msgs in failures.values() if msgs)
        overall_fail = overall_fail or any_fail

        for key, msgs in failures.items():
            label = label_map.get(key, key)
            if msgs is None:
                continue  # skipped; reported in notes
            if msgs:
                print(f"[FAIL] {label}:")
                for msg in msgs:
                    print(f"  - {msg}")
            else:
                print(f"[OK]   {label} within tolerance")
        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
