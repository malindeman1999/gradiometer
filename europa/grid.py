"""
Grid utilities for equal-area spherical sampling.
Uses a triangular icosphere mesh (trimesh) so grid sampling matches the rendering mesh.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import numpy as np
import trimesh
import math

from .config import GridConfig


@dataclass(frozen=True)
class Grid:
    """Grid data container."""
    positions: torch.Tensor  # [N_nodes, 3] (face centers)
    normals: torch.Tensor    # [N_nodes, 3]
    areas: torch.Tensor      # [N_nodes]
    neighbors: torch.Tensor  # [N_nodes, k] integer neighbor indices
    surface_conductivity_s: float  # 2D surface conductivity (S) = sigma_3d * thickness


def estimate_lmax_from_nside(nside: int) -> int:
    """Heuristic l_max for a given subdivision level to avoid aliasing."""
    return max(1, 3 * nside - 1)


def estimate_nside_from_lmax(lmax: int) -> int:
    """Heuristic subdivision level for a desired l_max."""
    return max(1, (lmax + 1) // 3)


def estimate_lmax_from_pixels(npix: int) -> int:
    """Estimate l_max from a target face count (approximate)."""
    nside = int((npix / 20.0) ** 0.5)  # rough inverse for icosphere faces
    return estimate_lmax_from_nside(nside)


def make_grid(config: GridConfig) -> Grid:
    """
    Build a triangular icosphere grid using trimesh.
    Positions are face centers; neighbors are adjacent faces (up to 3, padded with self).
    """
    def _subdivisions_from_nside(n: int) -> int:
        """
        Choose subdivision level whose face count (20*4^s) best matches a target proportional to nside.
        Here we treat nside as a rough face-count proxy (target_faces ≈ 2*nside) rather than HEALPix nside^2.
        """
        if n <= 0:
            return 0
        target = max(20, 2 * n)
        best_s = 0
        best_err = float("inf")
        for s in range(0, 9):  # 0..8
            faces = 20 * (4 ** s)
            err = abs(faces - target)
            if err < best_err:
                best_err = err
                best_s = s
        return best_s

    subdivisions = _subdivisions_from_nside(config.nside)
    surface_sigma = float(config.seawater_conductivity_s_per_m * config.ocean_thickness_m)

    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=config.radius_m)
    faces = ico.faces  # (F, 3)
    vertices = ico.vertices  # (V, 3)
    tri_pts = vertices[faces]
    centers = tri_pts.mean(axis=1)
    positions = torch.tensor(centers, device=config.device, dtype=torch.float32)
    normals = torch.nn.functional.normalize(positions, dim=-1)
    areas = torch.tensor(ico.area_faces, device=config.device, dtype=torch.float32)

    # Face adjacency -> neighbors per face (up to 3)
    adj = ico.face_adjacency  # (M, 2)
    F = faces.shape[0]
    max_degree = 3
    neigh = torch.full((F, max_degree), -1, device=config.device, dtype=torch.long)
    for a, b in adj:
        for src, dst in ((a, b), (b, a)):
            row = neigh[src]
            idx_free = (row == -1).nonzero(as_tuple=False)
            if idx_free.numel():
                row[idx_free[0]] = int(dst)
    # Fill missing with self to keep shape consistent
    mask_missing = neigh < 0
    if torch.any(mask_missing):
        self_idx = torch.arange(F, device=config.device, dtype=torch.long).unsqueeze(1).expand_as(neigh)
        neigh[mask_missing] = self_idx[mask_missing]

    return Grid(positions=positions, normals=normals, areas=areas, neighbors=neigh, surface_conductivity_s=surface_sigma)
