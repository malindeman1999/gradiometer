"""
Benchmark harmonic vs. Bempp (placeholder) pipeline.

- Runs the existing harmonic precomputed solver on a small grid.
- Attempts to run the Bempp-based solver (stub); skips if Bempp is not installed.
- Reports timing and basic norm differences when both are available.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from europa_model.config import GridConfig, ModelConfig, AmbientConfig
from europa_model.simulation import Simulation
from europa_model.solvers import toroidal_e_from_radial_b
from europa_model.solver_variants.solver_variant_precomputed import solve_spectral_self_consistent_sim_precomputed
from gaunt.gaunt_cache_wigxjpf import load_gaunt_tensor_wigxjpf
from fem_bempp import bempp_available, solve_self_consistent_bempp


def _build_sim(lmax: int = 3, nside: int = 4) -> Simulation:
    grid_cfg = GridConfig(nside=nside, lmax=lmax, radius_m=1.56e6, device="cpu")
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=1.0, phase_rad=0.0)
    model_cfg = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model_cfg)
    sim.build_grid()
    return sim


def _run_harmonic(sim: Simulation) -> Simulation:
    # Use dummy uniform admittance and ambient radial field
    sim.admittance_uniform = 1.0
    sim.B_radial = torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128)
    # Build toroidal E from B_radial
    sim.E_toroidal = toroidal_e_from_radial_b(sim.B_radial, sim.omega, sim.radius_m)
    solve_spectral_self_consistent_sim_precomputed(sim, cache_dir="gaunt/data/gaunt_cache_wigxjpf")
    return sim


def _run_bempp(sim: Simulation):
    if not bempp_available():
        print("Bempp not installed; skipping Bempp path.")
        return None
    # Placeholder call; actual Bempp coupling not implemented.
    res = solve_self_consistent_bempp(
        positions=sim.grid.positions,
        normals=sim.grid.normals,
        admittance=torch.full((sim.grid.positions.shape[0],), 1.0, dtype=torch.complex128),
        ambient_B_tor=torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128),
        ambient_B_pol=torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128),
        radius_m=sim.radius_m,
    )
    return res


def main() -> int:
    sim = _build_sim(lmax=3, nside=4)

    t0 = time.perf_counter()
    sim_h = _run_harmonic(sim)
    t_h = time.perf_counter() - t0

    t1 = time.perf_counter()
    bempp_res = _run_bempp(sim)
    t_b = time.perf_counter() - t1 if bempp_res is not None else None

    print(f"Harmonic path time: {t_h:.3f}s")
    if bempp_res is None:
        print("Bempp path skipped (not installed).")
    else:
        print(f"Bempp path time: {t_b:.3f}s")
        # Compare emitted radial field norm as a crude check
        b_emit_h = sim_h.B_rad_emit if sim_h.B_rad_emit is not None else torch.zeros(1)
        b_emit_bempp = bempp_res["B_emit"]["radial"]
        diff = (b_emit_h.flatten()[: b_emit_bempp.numel()].cpu() - b_emit_bempp.cpu()).abs().max()
        print(f"Max |B_emit_rad_harmonic - Bempp|: {float(diff):.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
