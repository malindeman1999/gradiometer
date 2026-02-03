"""
Helpers to map between spherical-harmonic phasors and point-sampled phasors.
"""
from __future__ import annotations

import torch

from europa_model import transforms


def harmonics_to_points(B_rad: torch.Tensor, B_tor: torch.Tensor, B_pol: torch.Tensor, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Convert harmonic phasors (radial/toroidal/poloidal) to Cartesian phasors at positions."""
    tangential = transforms.vsh_inverse(B_tor, B_pol, positions, weights)
    radial_vals = transforms.sh_inverse(B_rad, positions, weights)
    normals = torch.nn.functional.normalize(positions, dim=-1)
    return tangential + normals * radial_vals[..., None]


def points_to_harmonics(B_grid: torch.Tensor, positions: torch.Tensor, weights: torch.Tensor, lmax: int):
    """Project Cartesian phasors at positions back to harmonic components."""
    normals = torch.nn.functional.normalize(positions, dim=-1)
    radial_vals = (B_grid * normals).sum(dim=-1)
    tangential = B_grid - radial_vals[..., None] * normals
    B_rad = transforms.sh_forward(radial_vals, positions, lmax=lmax, weights=weights)
    B_tor, B_pol = transforms.vsh_forward(tangential, positions, lmax=lmax, weights=weights)
    return B_rad, B_tor, B_pol
