"""
Compare baseline vs precomputed solver for lmax=1 (b-spectrum limited) using saved Gaunt cache.
Runs both solvers on a random PhasorSimulation and reports max diffs.
Skips if the Gaunt cache is missing.
"""
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa_model.config import GridConfig, ModelConfig, AmbientConfig
from europa_model.simulation import Simulation
from workflow.data_objects.phasor_data import PhasorSimulation
from europa_model import solvers
from europa_model.solver_variants.solver_variant_precomputed import solve_spectral_self_consistent_sim_precomputed


def _build_sim(lmax: int = 1) -> PhasorSimulation:
    grid_cfg = GridConfig(nside=4, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=0.0, phase_rad=0.0, spatial_mode="custom")
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    torch.manual_seed(0)
    Y_s = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_s += 1j * torch.randn_like(Y_s)
    B_rad = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128) * 1e-6
    B_rad += 1j * torch.randn_like(B_rad) * 1e-6
    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s,
        B_radial=B_rad,
        period_sec=0.0,
    )
    return base


def _max_abs_and_rel(a: torch.Tensor, b: torch.Tensor):
    diff = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
    max_abs = float(torch.max(diff).cpu())
    max_ref = float(torch.max(torch.stack([a.abs(), b.abs()])).cpu()) if a.numel() > 0 else 1.0
    max_ref = max(max_ref, 1e-30)
    max_rel = max_abs / max_ref
    return max_abs, max_rel


def main() -> None:
    cache_path = os.path.join("gaunt", "data", "gaunt_cache_wigxjpf", "gaunt_wigxjpf.pt")
    if not os.path.exists(cache_path):
        print(f"Gaunt cache not found at {cache_path}; run gaunt/run_gaunt_calculator.py first.")
        return

    base = _build_sim(lmax=1)
    ref = PhasorSimulation.from_serializable(base.to_serializable())
    test = PhasorSimulation.from_serializable(base.to_serializable())

    solvers.solve_spectral_self_consistent_sim(ref)
    solve_spectral_self_consistent_sim_precomputed(test, cache_dir="gaunt/data/gaunt_cache_wigxjpf")

    fields = [
        "E_toroidal",
        "K_toroidal",
        "K_poloidal",
        "B_tor_emit",
        "B_pol_emit",
        "B_rad_emit",
    ]
    failures = []
    for fname in fields:
        a = getattr(ref, fname)
        b = getattr(test, fname)
        if a is None or b is None:
            failures.append(f"{fname}: missing")
            continue
        max_abs, max_rel = _max_abs_and_rel(a, b)
        failures.append(f"{fname}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

    print("Comparison baseline vs precomputed (lmax=1): max_abs/max_rel differences")
    for msg in failures:
        print("  ", msg)


if __name__ == "__main__":
    main()
