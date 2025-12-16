"""
Precomputed-coefficient variant of the spectral self-consistent solver.

- Precomputes Gaunt coefficients and caches them on disk keyed by lmax.
- Builds the mixing matrix using tensor operations (limited Python loops) and
  the cached Gaunt tensor.

Baseline solver in solvers.py remains unchanged.
"""
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from europa import inductance
from europa.solvers import (
    _flatten_lm,
    _unflatten_lm,
    _build_faraday_diag,
    _build_self_field_diag,
    toroidal_e_from_radial_b,
)
from phasor_data import PhasorSimulation
from gaunt.gaunt_cache_wigxjpf import load_gaunt_tensor_wigxjpf


def _all_lm(lmax: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


def _build_mixing_matrix_precomputed(
    lmax: int, omega: float, radius: float, Y_s_spectral: torch.Tensor, G: torch.Tensor
) -> torch.Tensor:
    """
    Build mixing matrix using cached Gaunt tensor.
    M[L_flat, q_flat] = sum_{l0,m0} Y_s(l0,m0) * F(l') * G[L,M,l0,m0,l',m']
    """
    device = Y_s_spectral.device
    dtype = torch.complex128
    Y = Y_s_spectral.to(dtype)
    # Faraday diagonal unflattened
    F_flat = _build_faraday_diag(lmax, omega, radius, device=device, dtype=dtype)
    F = _unflatten_lm(F_flat, lmax)  # shape [lmax+1, 2*lmax+1]

    G = G.to(device=device, dtype=torch.float64)
    Gc = G.to(dtype=dtype)

    # Broadcast: Gc[L,M,l0,m0,l,m] * Y[l0,m0] * F[l,m]
    # Align Y on (l0, m0) axes of G, then add trailing dims for (l, m)
    Y_exp = Y.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,1,l0,m0,1,1]
    F_exp = F.unsqueeze(0).unsqueeze(0).unsqueeze(2).unsqueeze(2)  # [1,1,1,1,l,m]
    M_tensor = (Gc * Y_exp * F_exp).sum(dim=(2, 3))  # sum over l0,m0 -> [L,M,l,m]

    # Map to flattened matrix directly
    lm_list = _all_lm(lmax)
    n = len(lm_list)
    M_flat = torch.zeros((n, n), device=device, dtype=dtype)
    for r, (L, M_val) in enumerate(lm_list):
        M_idx = L + M_val
        for c, (l_c, m_c) in enumerate(lm_list):
            M_flat[r, c] = M_tensor[L, M_idx, l_c, lmax + m_c]
    return M_flat


def _build_mixing_matrix_precomputed_sparse(
    lmax: int, omega: float, radius: float, Y_s_spectral: torch.Tensor, G_sparse: torch.Tensor
) -> torch.Tensor:
    """
    Build mixing matrix using a sparse Gaunt tensor to avoid densification at high lmax.
    """
    device = Y_s_spectral.device
    dtype = torch.complex128
    n = (lmax + 1) ** 2
    F_flat = _build_faraday_diag(lmax, omega, radius, device=device, dtype=dtype)
    M = torch.zeros((n, n), device=device, dtype=dtype)

    G_sparse = G_sparse.coalesce()
    idx = G_sparse.indices().to(device)
    vals = G_sparse.values().to(device=device, dtype=dtype)

    lmax_b = G_sparse.shape[4] - 1

    L = idx[0]
    M_idx = idx[1]
    l0 = idx[2]
    m0_idx = idx[3]
    l_in = idx[4]
    m_in_idx = idx[5]

    row_flat = L * L + M_idx  # M_idx = M + L
    m_in = m_in_idx - lmax_b
    col_flat = l_in * l_in + (m_in + l_in)

    mask = (
        (L <= lmax)
        & (l_in <= lmax)
        & (l0 <= lmax)
        & (m0_idx >= 0)
        & (m0_idx < Y_s_spectral.shape[1])
        & (l0 < Y_s_spectral.shape[0])
        & (col_flat >= 0)
        & (col_flat < n)
        & (row_flat >= 0)
        & (row_flat < n)
    )
    if not torch.all(mask):
        L = L[mask]
        M_idx = M_idx[mask]
        l0 = l0[mask]
        m0_idx = m0_idx[mask]
        l_in = l_in[mask]
        m_in = m_in[mask]
        col_flat = col_flat[mask]
        row_flat = row_flat[mask]
        vals = vals[mask]

    y_vals = Y_s_spectral.to(device=device, dtype=dtype)[l0, m0_idx]
    f_vals = F_flat[col_flat]
    contrib = vals * y_vals * f_vals

    flat_idx = row_flat * n + col_flat
    M.view(-1).index_add_(0, flat_idx, contrib)
    return M


def _trim_gaunt_sparse(G_sparse: torch.Tensor, lmax_out: int, lmax_y: int, lmax_b: int) -> torch.Tensor:
    """
    Trim a sparse Gaunt tensor to the requested lmax bounds without densifying.
    """
    G_sparse = G_sparse.coalesce()
    idx = G_sparse.indices()
    vals = G_sparse.values()
    orig_lmax_y = (G_sparse.shape[3] - 1) // 2
    orig_lmax_b = (G_sparse.shape[5] - 1) // 2
    shift_y = orig_lmax_y - lmax_y
    shift_b = orig_lmax_b - lmax_b
    mask = (
        (idx[0] <= lmax_out)
        & (idx[2] <= lmax_y)
        & (idx[4] <= lmax_b)
        & (idx[3] >= shift_y)
        & (idx[3] <= shift_y + 2 * lmax_y)
        & (idx[5] >= shift_b)
        & (idx[5] <= shift_b + 2 * lmax_b)
    )
    size = (
        lmax_out + 1,
        2 * lmax_out + 1,
        lmax_y + 1,
        2 * lmax_y + 1,
        lmax_b + 1,
        2 * lmax_b + 1,
    )
    if not torch.any(mask):
        raise RuntimeError("Trimmed Gaunt tensor has no entries.")
    idx_t = idx[:, mask].clone()
    idx_t[3] = idx_t[3] - shift_y
    idx_t[5] = idx_t[5] - shift_b
    vals_t = vals[mask]
    return torch.sparse_coo_tensor(idx_t, vals_t, size=size).coalesce()


def solve_spectral_self_consistent_sim_precomputed(
    sim: PhasorSimulation,
    cache_dir: str | Path = "data/gaunt_cache_wigxjpf",
    gaunt_sparse: torch.Tensor | None = None,
    mixing_matrix: torch.Tensor | None = None,
) -> PhasorSimulation:
    """
    Self-consistent spectral solver using cached Gaunt tensor for mixing matrix build.
    """
    omega0 = sim.omega
    lmax = sim.lmax
    dtype = torch.complex128
    B_radial = sim.B_radial.to(dtype)
    Y_s_spectral = (
        sim.admittance_spectral.to(dtype) if sim.admittance_spectral is not None else torch.zeros_like(B_radial)
    )
    E_tor = toroidal_e_from_radial_b(B_radial, omega0, sim.radius_m)

    if mixing_matrix is None:
        if gaunt_sparse is None:
            print("Loading dense Gaunt tensor from cache and building mixing matrix...", flush=True)
            G = load_gaunt_tensor_wigxjpf(lmax_out=lmax, lmax_y=lmax, lmax_b=lmax, cache_dir=cache_dir)
            M = _build_mixing_matrix_precomputed(lmax, omega0, sim.radius_m, Y_s_spectral, G)
        else:
            print("Building mixing matrix from provided sparse Gaunt tensor...", flush=True)
            G = _trim_gaunt_sparse(gaunt_sparse, lmax_out=lmax, lmax_y=lmax, lmax_b=lmax)
            M = _build_mixing_matrix_precomputed_sparse(lmax, omega0, sim.radius_m, Y_s_spectral, G)
    else:
        print("Using provided mixing matrix (skipping Gaunt build)...", flush=True)
        M = mixing_matrix
    print(f"Mixing matrix built with shape {tuple(M.shape)}", flush=True)
    b_ext_flat = _flatten_lm(B_radial)
    S_diag = _build_self_field_diag(lmax, sim.grid_positions.device, dtype)
    I = torch.eye(M.shape[0], device=M.device, dtype=dtype)
    print("Solving self-consistent linear system...", flush=True)
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
    sim.solver_variant = "spectral_self_consistent_precomputed_gaunt"
    return sim
