"""
Variant of run_demo_compare focusing on non-uniform (spectral) admittance only.

- Uses complex spectral admittance up to lmax=50.
- Uses a normal magnetic driver (+X) from ambient_driver.
- Runs spectral first-order and self-consistent solvers using precomputed Gaunt coefficients.
- Saves the self-consistent solution and renders spectral plots with the magnitude
  of the self-consistent fields on the sphere.
"""
import argparse
from pathlib import Path
import math
import time
from contextlib import contextmanager
from typing import Tuple

import torch
import numpy as np

from workflow.ambient_field.ambient_driver import build_ambient_driver_x
from europa_model import inductance
from europa_model.transforms import sh_forward
from europa_model.config import GridConfig, ModelConfig
from europa_model.simulation import Simulation
from workflow.data_objects.phasor_data import PhasorSimulation
from workflow.plotting.render_demo_overview import render_demo_overview
from europa_model.solvers import _flatten_lm, _unflatten_lm, toroidal_e_from_radial_b
from europa_model.solver_variants.solver_variant_precomputed import (
    solve_spectral_self_consistent_sim_precomputed,
    _build_mixing_matrix_precomputed_sparse,
)
from europa_model.gradient_utils import rss_gradient_from_emit, render_gradient_map
from europa_model.old.harmonic_mapping import harmonics_to_points, points_to_harmonics
from gaunt.assemble_gaunt_checkpoints import assemble_in_memory

def _log(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def _time_block(label: str):
    start = time.perf_counter()
    _log(f"{label}...")
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        _log(f"{label} done in {dur:.2f}s")


def _clone_ps(base: PhasorSimulation) -> PhasorSimulation:
    """Deep-ish clone via serializable dict."""
    return PhasorSimulation.from_serializable(base.to_serializable())


def _checkerboard_admittance(positions: torch.Tensor, sigma_low: float, sigma_high: float, divisions: int) -> torch.Tensor:
    """Assign a checkerboard pattern over (theta, phi) with the given number of divisions."""
    # positions are on a sphere; compute angles
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = torch.linalg.norm(positions, dim=1)
    theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))  # [0, pi]
    phi = torch.atan2(y, x)
    phi = torch.remainder(phi, 2 * math.pi)  # [0, 2pi)

    div = max(1, divisions)
    theta_bin = torch.floor(theta / (math.pi / div))
    phi_bin = torch.floor(phi / (2 * math.pi / div))
    parity = (theta_bin + phi_bin).to(torch.int64) % 2
    return torch.where(parity == 0, torch.full_like(theta, sigma_low), torch.full_like(theta, sigma_high))



def run(
    compute_gradients: bool = False,
    checker_divisions: int = 6,
    first_order_only: bool = False,
) -> Tuple[PhasorSimulation, str, int, bool, str]:
    # Grid and driver
    # Increase spatial resolution (double nside) and allow higher spectral order
    lmax_target = 10
    grid_cfg = GridConfig(nside=32, lmax=lmax_target, radius_m=1.56e6, device="cpu")
    _log(f"Building grid with nside={grid_cfg.nside}, lmax={grid_cfg.lmax}...")
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver_x(grid_cfg)
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    cache_dir = Path("gaunt/data/gaunt_cache_wigxjpf")
    with _time_block(f"Assembling Gaunt tensor from checkpoints in {cache_dir}"):
        G_sparse, gaunt_meta = assemble_in_memory(cache_dir=cache_dir, lmax_limit=grid_cfg.lmax, verbose=True)
    complete_L = int(gaunt_meta["complete_L"])
    _log(
        f"Gaunt tensor assembled in-memory with nnz={G_sparse._nnz()}, "
        f"complete_L={complete_L}, assembled_L={gaunt_meta.get('assembled_L')}"
    )

    # Real (conductive) spectral admittance: random conductance map projected to SH
    sigma_3d = grid_cfg.seawater_conductivity_s_per_m
    thickness = grid_cfg.ocean_thickness_m
    sigma_2d_max = 2.0 * sigma_3d * thickness
    _log(
        f"Assigning checkerboard conductance: low=0, high={sigma_2d_max:.2e} S "
        f"(divisions={checker_divisions}, up to L={grid_cfg.lmax})..."
    )
    conductance_grid = _checkerboard_admittance(
        sim.grid.positions.to(dtype=torch.float64), sigma_low=0.0, sigma_high=sigma_2d_max, divisions=checker_divisions
    )
    weights = sim.grid.areas.to(dtype=torch.float64)
    positions = sim.grid.positions.to(dtype=torch.float64)
    Y_s_spectral = sh_forward(conductance_grid, positions, lmax=grid_cfg.lmax, weights=weights)

    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec.cpu(),
        period_sec=period_sec,
    )

    # Prebuild mixing matrix once using sparse Gaunt tensor
    with _time_block("Building sparse mixing matrix from Gaunt tensor and admittance"):
        mixing_matrix = _build_mixing_matrix_precomputed_sparse(
            grid_cfg.lmax, model.ambient.omega_jovian, model.grid.radius_m, Y_s_spectral, G_sparse
        )
    _log(f"Mixing matrix ready with shape {tuple(mixing_matrix.shape)}")

    sim_out: PhasorSimulation
    label: str
    if first_order_only:
        sim_out = _clone_ps(base)
        label = "first_order"
        with _time_block("Solving first-order (no feedback) currents with precomputed mixing matrix"):
            sim_out.E_toroidal = toroidal_e_from_radial_b(sim_out.B_radial, sim_out.omega, sim_out.radius_m)
            b_flat = _flatten_lm(sim_out.B_radial.to(torch.complex128))
            k_flat = mixing_matrix @ b_flat
            sim_out.K_toroidal = _unflatten_lm(k_flat, grid_cfg.lmax)
            sim_out.K_poloidal = torch.zeros_like(sim_out.K_toroidal)
            sim_out.B_tor_emit, sim_out.B_pol_emit, sim_out.B_rad_emit = inductance.spectral_b_from_surface_currents(
                sim_out.K_toroidal, sim_out.K_poloidal, radius=sim_out.radius_m
            )
            sim_out.solver_variant = "spectral_first_order_precomputed_gaunt_sparse"
    else:
        sim_out = _clone_ps(base)
        label = "self_consistent"
        # Use precomputed Gaunt cache (loader) instead of on-the-fly computation
        with _time_block("Solving self-consistent system (matrix inversion) using precomputed mixing matrix"):
            solve_spectral_self_consistent_sim_precomputed(sim_out, gaunt_sparse=G_sparse, mixing_matrix=mixing_matrix)
        _log("Self-consistent solve complete.")

    _log("Saving phasor snapshot and launching render pipeline...")

    out_path = Path(f"figures/nonuniform_{label}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Saving phasor snapshot to {out_path}...")
    torch.save({"phasor_sim": sim_out}, out_path)
    # Choose sphere sampling commensurate with spectral bandwidth (~4*(lmax+1)^2 pixels).
    desired_faces = max(20, 4 * (grid_cfg.lmax + 1) ** 2)
    subdivisions = max(1, math.ceil(math.log(desired_faces / 20, 4)))
    return sim_out, str(out_path), subdivisions, compute_gradients, label


def main() -> None:
    print("Starting.")
    parser = argparse.ArgumentParser(description="Non-uniform spectral admittance demo using precomputed Gaunt cache.")
    parser.add_argument("--gradients", action="store_true", help="Render gradient magnitude maps.")
    parser.add_argument(
        "--checker-divisions",
        type=int,
        default=6,
        help="Number of theta/phi divisions for checkerboard admittance pattern.",
    )
    parser.add_argument(
        "--first-order-only",
        action="store_true",
        help="Compute only the first-order (no feedback) response instead of the self-consistent solution.",
    )
    args = parser.parse_args()

    sim_out, out_path, subdivisions, compute_gradients, label = run(
        compute_gradients=args.gradients, checker_divisions=args.checker_divisions, first_order_only=args.first_order_only
    )
    # Print quick comparison metric
    if label == "self_consistent":
        _log("Solved self-consistent response (first-order comparison skipped).")
    else:
        _log("Solved first-order response only.")
    # Render spectral plots and sphere map of the chosen solution
    face_count = 20 * (4 ** subdivisions)
    _log(
        f"Rendering overview plots with subdivisions={subdivisions} (~{face_count} sphere pixels per map, lmax={sim_out.lmax})..."
    )
    with _time_block("Rendering overview plots"):
        render_demo_overview(
            data_path=out_path,
            subdivisions=subdivisions,
            save_path=f"figures/nonuniform_{label}.png",
            show=False,
        )
    if compute_gradients:
        _log("Rendering RSS |grad_B_emit| maps at surface and +100 km...")
        # Gradient maps at surface and at observation altitude
        obs_alt_m = 100e3
        surface_title = f"RSS |grad_B_emit| at surface (alt=0 km)"
        alt_title = f"RSS |grad_B_emit| at alt={obs_alt_m/1000:.0f} km"
        surface_path = f"figures/nonuniform_grad_surface_{label}.png"
        alt_path = f"figures/nonuniform_grad_alt_{label}.png"
        with _time_block("Rendering surface gradient map"):
            render_gradient_map(sim_out, altitude_m=0.0, subdivisions=subdivisions, save_path=surface_path, title=surface_title)
        with _time_block("Rendering altitude gradient map"):
            render_gradient_map(sim_out, altitude_m=obs_alt_m, subdivisions=subdivisions, save_path=alt_path, title=alt_title)
    _log("Saved snapshot and all overview/gradient figures.")


if __name__ == "__main__":
    main()
