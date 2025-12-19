"""
Run gradient_sanity_check to compare analytic vs finite-difference gradients using the closed-form
toroidal field evaluator. Synthesizes a random toroidal spectrum and reports RSS at one point.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Allow running from repository root without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_data import PhasorSimulation
from europa.gradient_utils import gradient_sanity_check


def _build_dummy_sim(lmax: int = 3, radius: float = 1.0) -> PhasorSimulation:
    """
    Create a minimal PhasorSimulation carrying a random toroidal spectrum.
    Grid fields are placeholders; only K_toroidal, lmax, and radius_m are required by the check.
    """
    positions = torch.zeros((1, 3), dtype=torch.float64)
    normals = torch.zeros_like(positions)
    areas = torch.ones((1,), dtype=torch.float64)
    K_tor = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    # zero out l=0 toroidal mode (non-physical)
    K_tor[0] = 0.0

    return PhasorSimulation(
        omega=0.0,
        period_sec=0.0,
        lmax=lmax,
        radius_m=radius,
        ambient_amplitude_t=0.0,
        ambient_phase_rad=0.0,
        grid_positions=positions,
        grid_normals=normals,
        grid_areas=areas,
        grid_neighbors=None,
        solver_variant="gradient_sanity_check",
        K_toroidal=K_tor,
    )


def main() -> int:
    sim = _build_dummy_sim(lmax=4, radius=1.56e6)
    r_obs = sim.radius_m  # altitude 0 -> observation radius = source radius
    delta_angle = 1.0 / r_obs  # 1 meter arc length

    delta_cart_candidates = [0.1, 1.0, 5.0]
    # Pick a single random point and reuse it across delta_cart variations
    rand_seed = torch.randint(0, 2**31 - 1, (1,)).item()
    print("Single-point RSS gradients (random point each run):")
    for delta_cart in delta_cart_candidates:
        result = gradient_sanity_check(
            sim,
            altitude_m=0.0,
            n_points=1,
            seed=rand_seed,  # same point for each delta_cart
            delta_cart_m=delta_cart,
            delta_r=1.0,
            delta_theta=delta_angle,
            delta_phi=delta_angle,
            theta_fd_step=1e-6,
            verbose=False,
            use_autograd=False,
        )

        rss_A = result["rss_A_analytic_spherical"].squeeze()
        rss_B = result["rss_B_fd_spherical"].squeeze()
        rss_fd_cart = result["rss_fd_cartesian"].squeeze()
        pos_sph = result["positions_sph"]

        print(f"\nDelta_cart = {delta_cart} m")
        print(f"  Closed-form spherical (analytic):   {rss_A}")
        print(f"  Finite-diff spherical:              {rss_B}")
        print(f"  Finite-diff Cartesian:              {rss_fd_cart}")
        print(
            f"  Point spherical coords: r={float(pos_sph['r'][0]):.6f} m, "
            f"theta={float(pos_sph['theta'][0]):.6f} rad, phi={float(pos_sph['phi'][0]):.6f} rad"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
