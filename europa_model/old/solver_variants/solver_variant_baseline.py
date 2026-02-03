"""
Baseline copy of the spectral self-consistent solver (non-uniform admittance).

Kept in its own module so additional variants can live in separate files.
"""
import torch

from europa_model import inductance
from europa_model.solvers import (
    _flatten_lm,
    _unflatten_lm,
    _build_self_field_diag,
    _build_mixing_matrix_spectral,
    toroidal_e_from_radial_b,
)
from workflow.data_objects.phasor_data import PhasorSimulation


def solve_spectral_self_consistent_sim_baseline(sim: PhasorSimulation) -> PhasorSimulation:
    """
    Baseline copy of the spectral (non-uniform) admittance solver including self-field feedback.
    Matches europa.solvers.solve_spectral_self_consistent_sim.
    """
    omega0 = sim.omega
    lmax = sim.lmax
    dtype = torch.complex128
    B_radial = sim.B_radial.to(dtype)
    Y_s_spectral = (
        sim.admittance_spectral.to(dtype) if sim.admittance_spectral is not None else torch.zeros_like(B_radial)
    )
    E_tor = toroidal_e_from_radial_b(B_radial, omega0, sim.radius_m)

    b_ext_flat = _flatten_lm(B_radial)
    M = _build_mixing_matrix_spectral(lmax, omega0, sim.radius_m, Y_s_spectral)
    S_diag = _build_self_field_diag(lmax, sim.grid_positions.device, dtype)
    I = torch.eye(M.shape[0], device=M.device, dtype=dtype)
    A = I - torch.diag(S_diag) @ M
    b_tot = torch.linalg.solve(A, b_ext_flat)
    k_flat = M @ b_tot
    K_tor = _unflatten_lm(k_flat, lmax)
    K_pol = torch.zeros_like(K_tor)
    B_tor_emit, B_pol_emit, B_rad_emit = inductance.spectral_b_from_surface_currents(
        K_tor, K_pol, radius=sim.radius_m
    )
    sim.E_toroidal = E_tor
    sim.K_toroidal = K_tor
    sim.K_poloidal = K_pol
    sim.B_tor_emit = B_tor_emit
    sim.B_pol_emit = B_pol_emit
    sim.B_rad_emit = B_rad_emit
    sim.solver_variant = "spectral_self_consistent_baseline_copy"
    return sim
