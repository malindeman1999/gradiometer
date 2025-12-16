"""
Test 11: Complex admittance consistency.

Checks:
1) First-order spectral solver matches an explicit Gaunt-convolution reference
   when Y_s is complex-valued.
2) In the weak-coupling limit (Y_s -> eps * Y_s), self-consistent approaches
   first-order with differences shrinking ~eps.
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
    """Compute K_tor via explicit Gaunt convolution (first-order) for complex Y_s."""
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


def run_check(seed: int = 0) -> Tuple[Dict[str, List[str]], List[str]]:
    torch.manual_seed(seed)
    lmax = 3
    grid_cfg = GridConfig(nside=8, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver_x(grid_cfg)
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()

    # Complex spectral admittance
    Y_s = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_s += 1j * torch.randn_like(Y_s)
    Y_s[0] = 0.0  # remove DC

    eps_list = [1.0, 1e-3]
    failures: Dict[str, List[str]] = {
        "first_order_vs_gaunt": [],
        "weak_coupling": [],
    }
    notes: List[str] = [f"seed={seed}", f"lmax={lmax}", "driver=+X", "complex Y_s"]

    deltas: Dict[float, float] = {}

    for eps in eps_list:
        Y_scaled = eps * Y_s
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

        # Reference for first-order at eps=1 only
        if eps == 1.0:
            K_ref = _gaunt_convolution_reference(Y_scaled, base.B_radial, omega=base.omega, radius=base.radius_m)
            max_abs, max_rel = _max_abs_and_rel(first.K_toroidal, K_ref)
            if max_abs > 1e-6 and max_rel > 1e-2:
                failures["first_order_vs_gaunt"].append(
                    f"eps=1: K_toroidal mismatch vs Gaunt ref (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"
                )

        # Difference between self-consistent and first-order
        diff_abs, _ = _max_abs_and_rel(selfc.K_toroidal, first.K_toroidal)
        deltas[eps] = diff_abs

    # Weak-coupling scaling: delta_small should shrink ~ eps_small/eps_big
    eps_big, eps_small = eps_list[0], eps_list[1]
    if deltas[eps_small] >= deltas[eps_big] * (eps_small / eps_big) * 2.0:
        failures["weak_coupling"].append(
            f"delta did not shrink ~linearly: delta[{eps_big:.0e}]={deltas[eps_big]:.3e}, "
            f"delta[{eps_small:.0e}]={deltas[eps_small]:.3e}"
        )

    return failures, notes


def main() -> None:
    seeds = [0, 1, 2]
    overall_fail = False
    for seed in seeds:
        print(f"\nCase: seed = {seed}")
        failures, notes = run_check(seed)
        any_fail = any(msgs for msgs in failures.values() if msgs)
        overall_fail = overall_fail or any_fail

        if failures["first_order_vs_gaunt"]:
            print("[FAIL] first-order vs Gaunt reference:")
            for msg in failures["first_order_vs_gaunt"]:
                print(f"  - {msg}")
        else:
            print("[OK]   first-order matches Gaunt reference (eps=1)")

        if failures["weak_coupling"]:
            print("[FAIL] weak-coupling scaling (complex Y_s):")
            for msg in failures["weak_coupling"]:
                print(f"  - {msg}")
        else:
            print("[OK]   weak-coupling scaling holds (self-consistent -> first-order)")

        for note in notes:
            print(f"[INFO] {note}")

    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
