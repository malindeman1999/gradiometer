"""
Phasor-domain solvers using complex VSH.

Variants:
- Uniform admittance, first-order (no self-field feedback)
- Uniform admittance, self-consistent (includes normal self-field)
- Spectral (non-uniform) admittance, first-order (mode mixing via Gaunt coefficients)
- Spectral (non-uniform) admittance, self-consistent (matrix solve including self-field)

Outputs: toroidal E, toroidal/poloidal currents, and emitted B field phasors.
All inputs/outputs use spherical-harmonic layout [lmax+1, 2*lmax+1] with m index = m+lmax.
"""
from typing import Tuple

import numpy as np
import torch
try:
    from scipy.special import sph_harm  # type: ignore[attr-defined]
except Exception:
    sph_harm = None

try:
    from sympy.physics.wigner import wigner_3j as sympy_wigner_3j
    def wigner_3j(j1, j2, j3, m1, m2, m3):
        val = sympy_wigner_3j(j1, j2, j3, m1, m2, m3)
        return float(val)
except ImportError as exc:
    raise ImportError("wigner_3j not available; install scipy>=1.11 or sympy") from exc

from .config import ModelConfig
from . import inductance
from phasor_data import PhasorSimulation


def _ell_term_like(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast l(l+1) to match a spectral tensor."""
    lmax = tensor.shape[-2] - 1
    l = torch.arange(lmax + 1, device=tensor.device, dtype=torch.float64)
    ell = l * (l + 1)
    # avoid divide by zero; we'll explicitly zero l=0 contributions later
    ell = torch.where(ell == 0, torch.ones_like(ell), ell)
    ell = ell.view(lmax + 1, 1)
    while ell.dim() < tensor.dim():
        ell = ell.unsqueeze(0)
    return ell


def _flatten_lm(coeffs: torch.Tensor) -> torch.Tensor:
    """Flatten [lmax+1, 2*lmax+1] spectral coefficients to [N] with canonical (l,m) ordering."""
    lmax = coeffs.shape[-2] - 1
    flat = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            flat.append(coeffs[..., l, lmax + m])
    flat = torch.stack(flat, dim=-1)
    return flat


def _unflatten_lm(vec: torch.Tensor, lmax: int) -> torch.Tensor:
    """Inverse of _flatten_lm: [N] -> [lmax+1, 2*lmax+1]."""
    out = torch.zeros(vec.shape[:-1] + (lmax + 1, 2 * lmax + 1), dtype=vec.dtype, device=vec.device)
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            out[..., l, lmax + m] = vec[..., idx]
            idx += 1
    return out


def _gaunt_coeff(L: int, M: int, l0: int, m0: int, l: int, m: int) -> float:
    """
    Gaunt coefficient <Y_{L,M} | Y_{l0,m0} Y_{l,m}> using Wigner 3j (complex SH).

    Note: the bra Y_{L,M} carries a complex conjugate, so the magnetic index
    selection rule is -M + m0 + m = 0. Violating this rule is treated as an
    error rather than silently returning 0.
    """
    if -M + m0 + m != 0:
        raise ValueError(
            f"Gaunt selection rule violated: -M + m0 + m != 0 "
            f"(L={L}, M={M}, l0={l0}, m0={m0}, l={l}, m={m})"
        )
    pref = np.sqrt((2 * L + 1) * (2 * l0 + 1) * (2 * l + 1) / (4 * np.pi))
    w1 = wigner_3j(L, l0, l, 0, 0, 0)
    w2 = wigner_3j(L, l0, l, -M, m0, m)
    # (-1)^M appears because the bra carries a complex conjugate.
    return float(((-1.0) ** M) * pref * w1 * w2)


def _build_gaunt_tensor(lmax: int) -> np.ndarray:
    """Precompute Gaunt coefficients G[L,M,l0,m0,l,m] = <Y_{L,M} | Y_{l0,m0} Y_{l,m}>."""
    G = np.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=np.float64,
    )
    for L in range(lmax + 1):
        for M in range(-L, L + 1):
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        for m in range(-l, l + 1):
                            # Skip combinations that violate the m-selection rule
                            if -M + m0 + m != 0:
                                continue
                            G[L, L + M, l0, lmax + m0, l, lmax + m] = _gaunt_coeff(
                                L, M, l0, m0, l, m
                            )
    return G


def toroidal_e_from_radial_b(B_radial: torch.Tensor, omega: float, radius: float) -> torch.Tensor:
    """
    Faraday (phasor) mapping on the sphere:
        ∇ × E = -i ω B   (e^{i ω t} convention)

    For a single spherical-harmonic mode of the radial field at r = R,

        E_{ℓm} = -(i ω R / [ℓ (ℓ+1)]) B_{r,ℓm}.

    This returns the toroidal VSH coefficients E_{ℓm} as a complex tensor.

    Units:
        B_radial : tesla
        omega    : s^{-1}
        radius   : m
        E_tor    : V/m
    """
    ell = _ell_term_like(B_radial)  # ℓ(ℓ+1) for each degree ℓ, broadcast to tensor shape
    factor = -(1j * omega * radius) / ell
    E_tor = factor * B_radial.to(torch.complex128)
    # ℓ = 0 mode carries no toroidal field; zero it explicitly
    E_tor[..., 0, :] = 0.0
    return E_tor


def first_order_current(
    B_radial_ext: torch.Tensor,
    omega: float,
    radius: float,
    surface_admittance_s: torch.Tensor | float,
) -> torch.Tensor:
    """
    First-order surface current (no self-field feedback) with sheet admittance Y_s.

    Faraday (phasor) mapping on the sphere:
        ∇ × E = -i ω B   (e^{i ω t} convention)

    For a single (ℓ, m) mode of the radial field at r = R,

        E_{ℓm} = -(i ω R / [ℓ (ℓ+1)]) B_{r,ℓm}.

    With a (possibly complex) surface admittance Y_s(θ, φ) and Ohmic boundary
    condition K = Y_s E, a *uniform* Y_s gives

        J_{ℓm} = -Y_s * (i ω R / [ℓ (ℓ+1)]) B_{r,ℓm}.

    This function implements that relation, allowing `surface_admittance_s`
    to be either a scalar (uniform sheet admittance) or a spectral tensor
    [lmax+1, 2*lmax+1].

    Units:
        Y_s : S  (A/V)
        B_r : T
        ω   : s^{-1}
        R   : m
        J   : A/m  (surface current density on the sphere)
    """
    ell = _ell_term_like(B_radial_ext)
    Y_s = surface_admittance_s
    if not torch.is_tensor(Y_s):
        Y_s = torch.tensor(Y_s, device=B_radial_ext.device, dtype=torch.float64)
    while Y_s.dim() < B_radial_ext.dim():
        Y_s = Y_s.unsqueeze(0)

    # Faraday + Ohm:
    #   E_ℓm = -(1j * ω * R / ℓ(ℓ+1)) B_ℓm
    #   J_ℓm = Y_s E_ℓm
    prefactor = -(Y_s * 1j * omega * radius) / ell
    prefactor = prefactor.to(torch.complex128)
    J = prefactor * B_radial_ext.to(torch.complex128)
    # ℓ = 0 mode must not carry toroidal current
    J[..., 0, :] = 0.0
    return J


def self_consistent_current(
    B_radial_ext: torch.Tensor,
    omega: float,
    radius: float,
    surface_admittance_s: torch.Tensor | float,
) -> torch.Tensor:
    """
    Modal surface current including geometric self-field feedback
    (uniform or spectral admittance).

    Start from the first-order relation (no feedback):

        J_{ℓm}^{(1)} = -Y_s * (i ω R / [ℓ (ℓ+1)]) B_{r,ℓm}^{(ext)}.

    Toroidal surface currents also produce a normal magnetic field at the
    surface, which induces additional E via Faraday's law. In a quasi-static
    approximation, this leads to a scalar correction for each degree ℓ:

        J_{ℓm} = J_{ℓm}^{(1)} / (1 - α_ℓ),

    with the dimensionless loading parameter

        α_ℓ ≈ i ω μ₀ R Y_s /(2ℓ+1),

    using the usual μ₀ R/(2ℓ+1) geometric factor for degree-ℓ harmonics.

    Units:
        Y_s : S  (A/V)
        B_r : T
        ω   : s^{-1}
        R   : m
        J   : A/m  (surface current density)
    """
    ell = _ell_term_like(B_radial_ext)  # ℓ(ℓ+1)
    lmax = B_radial_ext.shape[-2] - 1
    l = torch.arange(lmax + 1, device=B_radial_ext.device, dtype=torch.float64)

    Y_s = surface_admittance_s
    if not torch.is_tensor(Y_s):
        Y_s = torch.tensor(Y_s, device=B_radial_ext.device, dtype=torch.float64)
    while Y_s.dim() < B_radial_ext.dim():
        Y_s = Y_s.unsqueeze(0)

    # Faraday diagonal F and self-field diagonal S (matching spectral solve)
    F = -(1j * omega * radius) / ell  # shape like ell
    S = inductance.MU0 / ((2 * l + 1).view(lmax + 1, 1) * ell)  # μ0 / [(2ℓ+1) ℓ(ℓ+1)]
    while S.dim() < B_radial_ext.dim():
        S = S.unsqueeze(0)

    prefactor = (Y_s.to(torch.complex128) * F.to(torch.complex128))
    feedback = S.to(torch.complex128) * prefactor  # elementwise

    J = prefactor * B_radial_ext.to(torch.complex128)
    J = J / (1.0 - feedback)

    # Remove ℓ = 0 mode which cannot support toroidal current
    J[..., 0, :] = 0.0
    return J


def _build_faraday_diag(lmax: int, omega: float, radius: float, device, dtype) -> torch.Tensor:
    """Diagonal Faraday operator F mapping b -> e (toroidal), flattened."""
    F = torch.zeros((lmax + 1, 2 * lmax + 1), device=device, dtype=dtype)
    for l in range(1, lmax + 1):
        factor = -(1j * omega * radius) / (l * (l + 1))
        F[l] = factor
    return _flatten_lm(F)


def _build_self_field_diag(lmax: int, device, dtype) -> torch.Tensor:
    """Diagonal map S: k -> b_self (normal component on surface) flattened."""
    S = torch.zeros((lmax + 1, 2 * lmax + 1), device=device, dtype=dtype)
    for l in range(1, lmax + 1):
        S[l] = inductance.MU0 / ((2 * l + 1) * l * (l + 1))
    return _flatten_lm(S)


def _build_mixing_matrix_spectral(
    lmax: int,
    omega: float,
    radius: float,
    Y_s_spectral: torch.Tensor,
) -> torch.Tensor:
    """
    Build spectral mixing matrix M mapping external radial field b_ext_flat
    to toroidal current k_flat for non-uniform surface admittance Y_s(θ,φ).

    k_flat = M @ b_ext_flat

    where M encodes convolution of Y_s(lm) with Faraday diagonal F(l) using
    Gaunt coefficients.
    """
    device = Y_s_spectral.device
    dtype = torch.complex128
    G = _build_gaunt_tensor(lmax)
    F_flat = _build_faraday_diag(lmax, omega, radius, device=device, dtype=dtype)

    n_harm = (lmax + 1) * (lmax + 1)  # (l+1)^2 canonical count
    M = torch.zeros((n_harm, n_harm), device=device, dtype=dtype)
    G_t = torch.from_numpy(G).to(dtype=dtype, device=device)

    def lm_index(l: int, m: int) -> int:
        idx = 0
        for l2 in range(lmax + 1):
            for m2 in range(-l2, l2 + 1):
                if l2 == l and m2 == m:
                    return idx
                idx += 1
        return -1

    # indices: p = (L,M_idx), q = (l',m'), r = (l0,m0)
    for L in range(lmax + 1):
        for M_idx in range(-L, L + 1):
            p_flat = lm_index(L, M_idx)
            for lprime in range(lmax + 1):
                # Faraday diagonal depends only on l'
                F_lprime = F_flat[lm_index(lprime, 0)]
                for mprime in range(-lprime, lprime + 1):
                    q_flat = lm_index(lprime, mprime)
                    accum = 0.0 + 0.0j
                    for l0 in range(lmax + 1):
                        for m0 in range(-l0, l0 + 1):
                            Y_r = Y_s_spectral[l0, lmax + m0]
                            G_val = G_t[
                                L,
                                L + M_idx,
                                l0,
                                lmax + m0,
                                lprime,
                                lmax + mprime,
                            ]
                            accum = accum + Y_r * F_lprime * G_val
                    M[p_flat, q_flat] = accum
    return M


def solve_uniform_first_order_sim(sim: PhasorSimulation) -> PhasorSimulation:
    """
    Uniform admittance, first-order (no self-field feedback).
    Uses only data from PhasorSimulation; returns updated instance.
    """
    omega0 = sim.omega
    Y_s = sim.admittance_uniform if sim.admittance_uniform is not None else 0.0
    B_radial = sim.B_radial.to(torch.complex128)
    E_tor = toroidal_e_from_radial_b(B_radial, omega0, sim.radius_m)
    K_tor = first_order_current(B_radial, omega0, sim.radius_m, Y_s)
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
    sim.solver_variant = "uniform_first_order"
    return sim


def solve_uniform_self_consistent_sim(sim: PhasorSimulation) -> PhasorSimulation:
    """
    Self-consistent, uniform admittance (includes self-field feedback).
    Uses only data from PhasorSimulation; returns updated instance.
    """
    omega0 = sim.omega
    Y_s = sim.admittance_uniform if sim.admittance_uniform is not None else 0.0
    B_radial = sim.B_radial.to(torch.complex128)
    E_tor = toroidal_e_from_radial_b(B_radial, omega0, sim.radius_m)
    K_tor = self_consistent_current(B_radial, omega0, sim.radius_m, Y_s)
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
    sim.solver_variant = "uniform_self_consistent"
    return sim


def solve_spectral_first_order_sim(sim: PhasorSimulation) -> PhasorSimulation:
    """
    Spectral (non-uniform) admittance, first-order (no self-field feedback).
    Uses only data from PhasorSimulation; returns updated instance.
    """
    omega0 = sim.omega
    lmax = sim.lmax
    dtype = torch.complex128
    B_radial = sim.B_radial.to(dtype)
    Y_s_spectral = (
        sim.admittance_spectral.to(dtype) if sim.admittance_spectral is not None else torch.zeros_like(B_radial)
    )
    E_tor = toroidal_e_from_radial_b(B_radial, omega0, sim.radius_m)

    b_flat = _flatten_lm(B_radial)
    M = _build_mixing_matrix_spectral(lmax, omega0, sim.radius_m, Y_s_spectral)
    k_flat = M @ b_flat
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
    sim.solver_variant = "spectral_first_order"
    return sim


def solve_spectral_self_consistent_sim(sim: PhasorSimulation) -> PhasorSimulation:
    """
    Spectral (non-uniform) admittance, including self-field feedback.
    Uses only data from PhasorSimulation; returns updated instance.
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
    sim.solver_variant = "spectral_self_consistent"
    return sim


def run_uniform_first_order(
    model: ModelConfig,
    grid,
    B_radial: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper to run uniform first-order solver for a given model & grid.
    Returns (K_tor, K_pol).
    """
    ph = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=grid,
        solver_variant="uniform_first_order",
        admittance_uniform=grid.surface_conductivity_s,
        admittance_spectral=None,
        B_radial=B_radial,
        period_sec=0.0,
    )
    ph = solve_uniform_first_order_sim(ph)
    K_tor = ph.K_toroidal
    K_pol = ph.K_poloidal
    return K_tor, K_pol


def run_uniform_self_consistent(
    model: ModelConfig,
    grid,
    B_radial: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper to run uniform self-consistent solver for a given model & grid.
    Returns (K_tor, K_pol).
    """
    ph = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=grid,
        solver_variant="uniform_self_consistent",
        admittance_uniform=grid.surface_conductivity_s,
        admittance_spectral=None,
        B_radial=B_radial,
        period_sec=0.0,
    )
    ph = solve_uniform_self_consistent_sim(ph)
    K_tor = ph.K_toroidal
    K_pol = ph.K_poloidal
    return K_tor, K_pol
