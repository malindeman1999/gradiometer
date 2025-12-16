"""
Grid utilities for HEALPix-like equal-area sampling and helper estimates for l_max.
Implementations are placeholders; to be filled with actual HEALPix routines.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import numpy as np
from astropy_healpix import HEALPix
from astropy import units as u

from .config import GridConfig


@dataclass(frozen=True)
class Grid:
    """Grid data container."""
    positions: torch.Tensor  # [N_nodes, 3]
    normals: torch.Tensor    # [N_nodes, 3]
    areas: torch.Tensor      # [N_nodes]
    neighbors: torch.Tensor  # [N_nodes, k] integer neighbor indices
    surface_conductivity_s: float  # 2D surface conductivity (S) = sigma_3d * thickness


def estimate_lmax_from_nside(nside: int) -> int:
    """Heuristic l_max for a given HEALPix nside to avoid aliasing."""
    return max(1, 3 * nside - 1)


def estimate_nside_from_lmax(lmax: int) -> int:
    """Heuristic nside for a desired l_max."""
    return max(1, (lmax + 1) // 3)


def estimate_lmax_from_pixels(npix: int) -> int:
    """Estimate l_max from a target pixel/facet count."""
    nside = int((npix / 12.0) ** 0.5)
    return estimate_lmax_from_nside(nside)


def make_grid(config: GridConfig) -> Grid:
    """
    Build a HEALPix grid using astropy-healpix for positions and neighbors.
    Returns positions, normals, equal-area weights, neighbor indices, and surface conductivity (σ·t).
    """
    nside = config.nside
    # Convert 3D conductivity (S/m) to 2D surface conductivity (S) using ocean thickness.
    surface_sigma = float(config.seawater_conductivity_s_per_m * config.ocean_thickness_m)
    hp = HEALPix(nside=nside, order="nested", frame=None)
    npix = hp.npix
    idx = np.arange(npix)
    lon, lat = hp.healpix_to_lonlat(idx)
    # Convert to Cartesian
    lon_rad = lon.to_value(u.rad)
    lat_rad = lat.to_value(u.rad)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    xyz = np.stack((x, y, z), axis=-1)
    xyz = torch.tensor(xyz, device=config.device, dtype=torch.float32) * config.radius_m
    positions = xyz
    normals = torch.nn.functional.normalize(xyz, dim=-1)
    areas = torch.full((npix,), 4.0 * torch.pi * (config.radius_m ** 2) / npix, device=config.device)
    # Neighbors: astropy-healpix returns up to 8 neighbors with -1 for missing
    neigh_np = hp.neighbours(idx)  # shape [8, npix]
    # neighbours may return nan for missing; replace with self index
    neigh_np = np.where(np.isnan(neigh_np), idx[None, :], neigh_np)
    neigh_np = neigh_np.T  # -> [npix, 8]
    neigh = torch.tensor(neigh_np, device=config.device, dtype=torch.long)
    # Replace any negative with self index to keep shape consistent
    mask_missing = neigh < 0
    if torch.any(mask_missing):
        idx_tensor = torch.tensor(idx, device=config.device, dtype=torch.long).unsqueeze(1).expand_as(neigh)
        neigh[mask_missing] = idx_tensor[mask_missing]
    return Grid(positions=positions, normals=normals, areas=areas, neighbors=neigh, surface_conductivity_s=surface_sigma)
