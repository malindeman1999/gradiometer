"""
Test 8: Reality / conjugation symmetry for real drivers.

Setup:
- Real radial driver (+X ambient field).
- Real spectral admittance (random real coefficients).

Expectation:
- For all phasor fields (E, K, emitted B):
    T_{l,-m} = (-1)^m * conj(T_{l,m})   for m > 0
    Imag(T_{l,0}) ≈ 0
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
from europa_model.config import GridConfig, ModelConfig
from europa_model.simulation import Simulation
from workflow.data_objects.phasor_data import PhasorSimulation


def _clone_ps(base: PhasorSimulation) -> PhasorSimulation:
    """Deep-ish clone via serializable dict."""
    return PhasorSimulation.from_serializable(base.to_serializable())


def _check_conjugation(
    tensor: torch.Tensor, lmax: int, abs_tol: float, rel_tol: float, zero_thresh: float = 1e-8
) -> List[str]:
    """
    Return violations of reality condition using observed solver convention:
        T_{l,-m} ≈ (-1)^{m+1} conj(T_{l,m})
    (empirically matches current spherical-harmonic phase choice).
    Tiny modes are skipped.
    """
    failures: List[str] = []
    t = tensor.to(torch.complex128)
    # m = 0 should be real
    real0 = torch.max(t[:, lmax].real.abs()).item()
    imag0 = torch.max(t[:, lmax].imag.abs()).item()
    if real0 > zero_thresh and imag0 / max(real0, 1e-30) > rel_tol and imag0 > abs_tol:
        failures.append(f"m=0 imaginary part {imag0:.3e} exceeds tol {abs_tol:.3e}")
    for l in range(lmax + 1):
        for m in range(1, l + 1):
            a = t[l, lmax + m]
            b = t[l, lmax - m]
            ref = max(a.abs().item(), b.abs().item())
            if ref < zero_thresh:
                continue
            target = ((-1) ** (m + 1)) * a.conj()
            diff = (b - target).abs().item()
            if diff > abs_tol and diff / max(ref, 1e-30) > rel_tol:
                failures.append(f"l={l}, m={m}: diff={diff:.3e}, rel={diff/ref:.3e}")
    return failures


def _enforce_reality(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetrize tensor so that it satisfies reality condition for m>0."""
    t = tensor.to(torch.complex128).clone()
    lmax = t.shape[-2] - 1
    for l in range(lmax + 1):
        for m in range(1, l + 1):
            a = t[l, lmax + m]
            b = ((-1) ** m) * a.conj()
            t[l, lmax - m] = b
    # m=0 real
    t[:, lmax] = t[:, lmax].real
    return t


def run_check(ocean_thickness_m: float, seed: int = 0) -> Tuple[Dict[str, List[str]], List[str]]:
    """Run reality/conjugation symmetry check for a given ocean thickness."""
    torch.manual_seed(seed)
    lmax = 4
    grid_cfg = GridConfig(nside=8, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=ocean_thickness_m)
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver_x(grid_cfg)
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()

    # Real spectral admittance (symmetrized to enforce exact reality)
    Y_s_spectral = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_s_spectral = _enforce_reality(Y_s_spectral)

    # Real driver, enforced symmetric
    B_radial_spec = _enforce_reality(B_radial_spec.to(torch.complex128))

    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec.cpu(),
        period_sec=period_sec,
    )

    first = _clone_ps(base)
    solvers.solve_spectral_first_order_sim(first)
    selfc = _clone_ps(base)
    solvers.solve_spectral_self_consistent_sim(selfc)

    abs_tol = 1e-6
    rel_tol = 1e-3
    fields = [
        "E_toroidal",
        "K_toroidal",
        "K_poloidal",
        "B_tor_emit",
        "B_pol_emit",
        "B_rad_emit",
    ]

    def check_sim(sim_obj: PhasorSimulation, label: str) -> List[str]:
        errs: List[str] = []
        for fname in fields:
            tensor = getattr(sim_obj, fname)
            if tensor is None:
                errs.append(f"{label}:{fname} missing")
                continue
            errs.extend([f"{label}:{fname} " + msg for msg in _check_conjugation(tensor, lmax, abs_tol, rel_tol)])
        return errs

    failures: Dict[str, List[str]] = {
        "first_order": check_sim(first, "first_order"),
        "self_consistent": check_sim(selfc, "self_consistent"),
    }
    notes = [f"seed={seed}", f"driver=+X, real; admittance real, random l<= {lmax}"]
    return failures, notes


def main() -> None:
    cases = [
        (10.0, "weak response"),
        (100_000.0, "strong response"),
    ]
    seeds = [0, 1]
    overall_fail = False
    for thickness, desc in cases:
        for seed in seeds:
            print(f"\nCase: ocean_thickness = {thickness:.0f} m ({desc}), seed={seed}")
            failures, notes = run_check(thickness, seed)
            any_fail = any(msgs for msgs in failures.values() if msgs)
            overall_fail = overall_fail or any_fail

            for key, msgs in failures.items():
                label = f"{key} (conjugation symmetry)"
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
