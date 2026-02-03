"""
Observation shell evaluation stubs.
"""
from typing import Optional

import torch

from .transforms import vsh_inverse
from . import vsh_ops, transforms
from . import inductance


def evaluate_B_observation(toroidal: torch.Tensor, poloidal: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Inverse transform tangential spectral field to observation grid (Cartesian components)."""
    # Assume unit weights for observation nodes
    n_nodes = positions.shape[0]
    weights = torch.ones((n_nodes,), device=positions.device)
    return vsh_inverse(toroidal, poloidal, positions, weights)


def compute_gradients(B_grid: torch.Tensor, positions: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """
    Approximate gradients using neighbor differences (least squares).
    Args:
        B_grid: [N_nodes, 3]
        positions: [N_nodes, 3]
        neighbors: [N_nodes, k] integer neighbor indices
    Returns:
        gradB: [N_nodes, 3, 3] (dB_i/dx_j)
    """
    n_nodes, k = neighbors.shape
    grad = torch.zeros((n_nodes, 3, 3), device=B_grid.device, dtype=B_grid.dtype)
    for i in range(n_nodes):
        nbrs = neighbors[i]
        pos_i = positions[i]
        Bi = B_grid[i]
        diffs_pos = positions[nbrs] - pos_i
        diffs_B = B_grid[nbrs] - Bi
        # Least squares gradient fit
        A = diffs_pos.to(dtype=diffs_B.dtype)  # [k,3]
        Y = diffs_B    # [k,3]
        # Solve (A^T A) grad = A^T Y
        ATA = A.T @ A
        ATY = A.T @ Y
        sol = torch.linalg.lstsq(ATA, ATY).solution
        grad[i] = sol
    return grad


def compute_rms_gradient(gradB_grid: torch.Tensor) -> torch.Tensor:
    """Compute RMS gradient magnitude from gradients [N_obs, 3, 3] or [T/F, N_obs, 3, 3]."""
    comps = gradB_grid ** 2
    rms = torch.sqrt(comps.mean(dim=(-2, -1)))
    return rms


def evaluate_field_from_spectral(
    B_tor: torch.Tensor,
    B_pol: torch.Tensor,
    B_rad: torch.Tensor,
    positions: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evaluate magnetic field (Cartesian components) from spectral coefficients at arbitrary positions.
    When weights are not provided, unit weights are assumed (acts like interpolation).
    """
    n_nodes = positions.shape[0]
    if weights is None:
        weights = torch.ones((n_nodes,), device=positions.device, dtype=positions.dtype)
    tangential = vsh_inverse(B_tor, B_pol, positions, weights)
    normals = torch.nn.functional.normalize(positions, dim=-1)
    radial_vals = transforms.sh_inverse(B_rad, positions, weights)
    B_grid = tangential + normals * radial_vals[..., None]
    return B_grid


def evaluate_B_from_currents(
    toroidal: torch.Tensor,
    poloidal: torch.Tensor,
    positions: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute full 3D B on an observation grid directly from spectral surface currents.
    Uses spectral jump-condition mapping to avoid node-level singularities.
    """
    radius = float(torch.norm(positions, dim=-1).mean())
    B_tor, B_pol, B_rad = inductance.spectral_b_from_surface_currents(toroidal, poloidal, radius=radius)
    B_grid = evaluate_field_from_spectral(B_tor, B_pol, B_rad, positions, weights)
    if torch.any(B_grid.abs() > 1.0):
        raise ValueError("Computed B field component exceeds 1 Tesla")
    return B_grid


def spectral_to_gradients(toroidal: torch.Tensor, poloidal: torch.Tensor, positions: torch.Tensor, lmax: int, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of B from spectral coefficients using VSH operators (approximate).
    Returns gradients evaluated on grid.
    """
    tor_from_pol, pol_from_tor = vsh_ops.curl_spectral(toroidal, poloidal, radius=1.0)
    n_nodes = positions.shape[0]
    weights = torch.ones((n_nodes,), device=positions.device)
    curl_vec = vsh_inverse(pol_from_tor, tor_from_pol, positions, weights)
    # Derive gradients of components approximately via curl relationships
    n_nodes = positions.shape[0]
    grad = torch.zeros((n_nodes, 3, 3), device=curl_vec.device, dtype=curl_vec.dtype)
    grad[:, 0, :] = curl_vec
    return grad
