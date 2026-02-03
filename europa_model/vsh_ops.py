"""
Vector spherical harmonic operator utilities.
Provides analytic grad/div/curl/Laplace-Beltrami factors in spectral space.
"""
import torch


def divergence(toroidal: torch.Tensor, poloidal: torch.Tensor, radius: float) -> torch.Tensor:
    """Surface divergence of VSH field: only poloidal contributes."""
    lmax = poloidal.shape[-2] - 1
    l = torch.arange(lmax + 1, device=poloidal.device, dtype=torch.float64)
    ell = l * (l + 1)
    scale = (-ell / (radius ** 2)).view(-1, 1)  # [lmax+1,1]
    while scale.dim() < poloidal.dim():
        scale = scale.unsqueeze(0)
    return scale * poloidal


def curl_toroidal_to_poloidal(toroidal: torch.Tensor, radius: float) -> torch.Tensor:
    """Surface curl maps toroidal -> poloidal with sqrt(l(l+1))/R factor."""
    lmax = toroidal.shape[-2] - 1
    l = torch.arange(lmax + 1, device=toroidal.device, dtype=torch.float64)
    factor = (torch.sqrt(l * (l + 1)) / radius).view(-1, 1)
    while factor.dim() < toroidal.dim():
        factor = factor.unsqueeze(0)
    return factor * toroidal


def curl_poloidal_to_toroidal(poloidal: torch.Tensor, radius: float) -> torch.Tensor:
    """Surface curl maps poloidal -> toroidal with sqrt(l(l+1))/R factor."""
    lmax = poloidal.shape[-2] - 1
    l = torch.arange(lmax + 1, device=poloidal.device, dtype=torch.float64)
    factor = (torch.sqrt(l * (l + 1)) / radius).view(-1, 1)
    while factor.dim() < poloidal.dim():
        factor = factor.unsqueeze(0)
    return factor * poloidal


def laplace(toroidal: torch.Tensor, poloidal: torch.Tensor, radius: float):
    """Laplace-Beltrami on tangential field: scale both components by -l(l+1)/R^2."""
    lmax = toroidal.shape[-2] - 1
    l = torch.arange(lmax + 1, device=toroidal.device, dtype=torch.float64)
    factor = (-l * (l + 1) / (radius ** 2)).view(-1, 1)
    while factor.dim() < toroidal.dim():
        factor = factor.unsqueeze(0)
    return toroidal * factor, poloidal * factor


def grad_scalar_to_poloidal(scalar_coeffs: torch.Tensor, radius: float) -> torch.Tensor:
    """Surface gradient of scalar harmonic coefficients into poloidal VSH."""
    lmax = scalar_coeffs.shape[-2] - 1
    l = torch.arange(lmax + 1, device=scalar_coeffs.device, dtype=torch.float64)
    factor = (torch.sqrt(l * (l + 1)) / radius).view(-1, 1)
    while factor.dim() < scalar_coeffs.dim():
        factor = factor.unsqueeze(0)
    return factor * scalar_coeffs
