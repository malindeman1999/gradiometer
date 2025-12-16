"""
Modal inductance and field mapping using the phasor/VSH formulas from the revised method notes.
"""

from typing import Tuple

import torch

# Permeability of free space
MU0 = 4e-7 * torch.pi


def spectral_operator_factors(lmax: int, radius: float, device: str):
    """
    Retained for callers that still need simple l-dependent factors (e.g., grad/curl scalings).
    """
    l = torch.arange(lmax + 1, device=device, dtype=torch.float64)
    ell_term = l * (l + 1)
    inv_r = 1.0 / radius
    factors = {
        "ell": ell_term,
        "grad": torch.sqrt(ell_term) * inv_r,
        "curl": torch.sqrt(ell_term) * inv_r,
        "laplace": -ell_term * (inv_r ** 2),
    }
    return factors


def _expand_l_factor(factor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast an l-dependent factor to match a spectral tensor rank."""
    while factor.dim() < target.dim():
        factor = factor.unsqueeze(0)
    return factor


def modal_radial_self_field(toroidal: torch.Tensor) -> torch.Tensor:
    """
    Radial self-field on the surface from a toroidal surface-current mode (phasor domain).
    From the magnetoquasistatic solution: B_n,self = -mu0 * J_lm / ((2l+1) l(l+1)).
    """
    lmax = toroidal.shape[-2] - 1
    l = torch.arange(lmax + 1, device=toroidal.device, dtype=torch.float64)
    ell = l * (l + 1)
    mask = (l > 0).to(toroidal.dtype).view(-1, 1)
    ell = torch.where(ell == 0, torch.ones_like(ell), ell)
    factor = (-MU0 / (2 * l + 1) / ell).view(-1, 1)
    factor = _expand_l_factor(factor, toroidal)
    return factor * toroidal * mask


def spectral_b_from_surface_currents(
    toroidal: torch.Tensor,
    poloidal: torch.Tensor,
    radius: float,
    r_eval: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Analytic magnetic field from a toroidal surface current mode (phasor, exterior field).

    Per the PDF derivation (Compute magnetic field.pdf):
        B_r(out)  = -mu0 * J_lm / [(2l+1) l(l+1)] (r/R)^(l+2) Y_lm
        B_t(out)  =  mu0 * l /(2l+1) (r/R)^(l+2) grad Y_lm   (poloidal)
    Toroidal magnetic components vanish for these currents.
    """
    if r_eval is None:
        r_eval = radius
    lmax = toroidal.shape[-2] - 1
    device = toroidal.device
    dtype = torch.promote_types(toroidal.dtype, torch.complex128)
    toroidal = toroidal.to(dtype)
    poloidal = poloidal.to(dtype)
    l = torch.arange(lmax + 1, device=device, dtype=torch.float64)
    ell = l * (l + 1)
    ell_safe = torch.where(ell == 0, torch.ones_like(ell), ell)
    # Exterior decay scales as (R / r)^(l+2); r_eval defaults to the surface radius.
    ratio = (torch.as_tensor(radius / r_eval, device=device, dtype=torch.float64) ** (l + 2)).view(-1, 1)

    # Radial component (scalar SH)
    b_rad_factor = (-MU0 / (2 * l + 1) / ell_safe).view(-1, 1) * ratio
    b_rad_factor = _expand_l_factor(b_rad_factor, toroidal)
    B_rad = b_rad_factor * toroidal

    # Tangential: purely poloidal for toroidal current
    b_pol_factor = (MU0 * l / (2 * l + 1)).view(-1, 1) * ratio
    b_pol_factor = _expand_l_factor(b_pol_factor, toroidal)
    B_pol = b_pol_factor * toroidal

    # Toroidal magnetic field from these modes is zero in the MQS limit
    B_tor = torch.zeros_like(B_pol)
    # Zero l=0 row to avoid numerical junk
    B_rad[0] = 0.0
    B_pol[0] = 0.0
    B_tor[0] = 0.0
    return B_tor, B_pol, B_rad
