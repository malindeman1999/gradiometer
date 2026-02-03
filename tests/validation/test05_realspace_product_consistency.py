"""
Test 5: Real-space vs spectral product consistency.

Goal: Verify that the spectral solver matches an independent Gaunt-based
convolution of Y_s(lm) with E_tor(lm) computed analytically (first-order).
"""
import os
import sys
from typing import List, Tuple

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


def _gaunt_convolution_reference(
    Y_s: torch.Tensor, B: torch.Tensor, omega: float, radius: float
) -> torch.Tensor:
    """Compute K_tor via explicit Gaunt convolution (first-order)."""
    lmax = B.shape[-2] - 1
    dtype = torch.complex128
    K = torch.zeros_like(B, dtype=dtype)
    for L in range(lmax + 1):
        for M in range(-L, L + 1):
            accum = 0.0 + 0.0j
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    Y_val = Y_s[l0, lmax + m0]
                    if Y_val == 0:
                        continue
                    for l in range(1, lmax + 1):
                        for m in range(-l, l + 1):
                            if -M + m0 + m != 0:
                                continue
                            G = solvers._gaunt_coeff(L, M, l0, m0, l, m)  # type: ignore[attr-defined]
                            F = -(1j * omega * radius) / (l * (l + 1))
                            accum = accum + Y_val * F * B[l, lmax + m] * G
            K[L, lmax + M] = accum
    return K


def run_check(seed: int = 0) -> Tuple[List[str], List[str]]:
    """Run one random-case check."""
    torch.manual_seed(seed)
    lmax = 2
    grid_cfg = GridConfig(nside=32, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=0.0, phase_rad=0.0, spatial_mode="custom")
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()

    # Random complex admittance and driver (low order)
    Y_s_spectral = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    B_radial_spec = torch.zeros_like(Y_s_spectral)
    for l in range(0, 2):  # l=0,1 to keep product within l<=2
        real = torch.randn(2 * l + 1, dtype=torch.float64)
        imag = torch.randn(2 * l + 1, dtype=torch.float64)
        Y_s_spectral[l, lmax - l:lmax + l + 1] = real + 1j * imag

        real_b = torch.randn(2 * l + 1, dtype=torch.float64)
        imag_b = torch.randn(2 * l + 1, dtype=torch.float64)
        B_radial_spec[l, lmax - l:lmax + l + 1] = (real_b + 1j * imag_b) * 1e-6

    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec,
        period_sec=0.0,
    )

    spec = _clone_ps(base)
    solvers.solve_spectral_first_order_sim(spec)

    # Reference via explicit Gaunt convolution
    K_tor_ref = _gaunt_convolution_reference(Y_s_spectral, B_radial_spec, omega=base.omega, radius=base.radius_m)

    failures: List[str] = []
    notes: List[str] = [f"seed={seed}"]

    max_abs, max_rel = _max_abs_and_rel(spec.K_toroidal, K_tor_ref)
    if max_abs > 1e-6 and max_rel > 1e-2:
        failures.append(f"K_toroidal mismatch: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

    return failures, notes


def main() -> None:
    seeds = [0, 1, 2]
    overall_fail = False
    for seed in seeds:
        print(f"\nCase: random seed = {seed}")
        failures, notes = run_check(seed)
        if failures:
            overall_fail = True
            print("[FAIL] real-space vs spectral product consistency:")
            for msg in failures:
                print(f"  - {msg}")
        else:
            print("[OK]   real-space vs spectral product match")
        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
