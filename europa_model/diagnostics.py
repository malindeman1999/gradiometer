"""
Diagnostics helpers for divergence and Faraday checks across domains.
"""
import torch

from . import transforms, inductance, vsh_ops


def divergence_grid(vector_grid: torch.Tensor, positions: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """
    Estimate surface divergence using least-squares fit over neighbor differences.
    Args:
        vector_grid: [N,3]
        positions: [N,3]
        neighbors: [N,k]
    Returns:
        divergence: [N]
    """
    n_nodes, k = neighbors.shape
    div = torch.zeros((n_nodes,), device=vector_grid.device, dtype=vector_grid.dtype)
    for i in range(n_nodes):
        nbrs = neighbors[i]
        pos_i = positions[i]
        vec_i = vector_grid[i]
        diffs_pos = positions[nbrs] - pos_i  # [k,3]
        diffs_vec = vector_grid[nbrs] - vec_i  # [k,3]
        # Solve least squares for Jacobian J: diffs_vec ≈ diffs_pos @ J^T
        ATA = diffs_pos.T @ diffs_pos
        ATY = diffs_pos.T @ diffs_vec
        J = torch.linalg.lstsq(ATA, ATY).solution  # [3,3]
        div[i] = torch.trace(J)
    return div


def divergence_spectral(toroidal: torch.Tensor, poloidal: torch.Tensor, radius: float) -> torch.Tensor:
    """Divergence from spectral coefficients via VSH operators."""
    return vsh_ops.divergence(toroidal, poloidal, radius)


def faraday_error(B_time: torch.Tensor, E_time: torch.Tensor, positions: torch.Tensor, neighbors: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute Faraday consistency error: || dB/dt + curl(E) ||_RMS using neighbor-based curl.
    Args:
        B_time: [T, N, 3] magnetic field over time.
        E_time: [T, N, 3] electric field over time.
        positions: [N,3]
        neighbors: [N,k]
        dt: timestep.
    Returns:
        scalar RMS error over time and space.
    """
    def curl_grid(E_grid):
        n_nodes, k = neighbors.shape
        curl = torch.zeros((n_nodes, 3), device=E_grid.device, dtype=E_grid.dtype)
        for i in range(n_nodes):
            nbrs = neighbors[i]
            pos_i = positions[i]
            diffs_pos = positions[nbrs] - pos_i  # [k,3]
            diffs_E = E_grid[nbrs] - E_grid[i]   # [k,3]
            ATA = diffs_pos.T @ diffs_pos
            ATY = diffs_pos.T @ diffs_E
            J = torch.linalg.lstsq(ATA, ATY).solution  # [3,3]
            # Curl from Jacobian
            curl_vec = torch.stack([
                J[2,1] - J[1,2],
                J[0,2] - J[2,0],
                J[1,0] - J[0,1],
            ])
            curl[i] = curl_vec
        return curl

    dB_dt = torch.diff(B_time, dim=0) / dt  # [T-1,N,3]
    curlE = []
    for t in range(E_time.shape[0]):
        curlE.append(curl_grid(E_time[t]))
    curlE = torch.stack(curlE, dim=0)[1:]  # align with dB_dt
    mismatch = dB_dt + curlE
    return torch.sqrt(torch.mean(mismatch ** 2))


def faraday_error_spectral(
    dB_rad_dt: torch.Tensor,
    dB_tor_dt: torch.Tensor,
    dB_pol_dt: torch.Tensor,
    E_tor: torch.Tensor,
    E_pol: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """
    Compute spectral Faraday mismatch RMS: || dB/dt + curl(E) ||_RMS in VSH space.
    Args:
        dB_rad_dt: [l,m] radial SH time derivative
        dB_tor_dt: [l,m] toroidal time derivative
        dB_pol_dt: [l,m] poloidal time derivative
        E_tor: [l,m] toroidal electric field coefficients
        E_pol: [l,m] poloidal electric field coefficients
        radius: sphere radius
    Returns:
        scalar RMS mismatch
    """
    # Ambient tangential drivers may be NaN sentinels; exclude them from the error metric.
    dB_rad_dt = torch.nan_to_num(dB_rad_dt, nan=0.0)
    dB_tor_dt = torch.nan_to_num(dB_tor_dt, nan=0.0)
    dB_pol_dt = torch.nan_to_num(dB_pol_dt, nan=0.0)
    E_tor = torch.nan_to_num(E_tor, nan=0.0)
    E_pol = torch.nan_to_num(E_pol, nan=0.0)
    curl_tor_from_pol = vsh_ops.curl_poloidal_to_toroidal(E_pol, radius)
    curl_pol_from_tor = vsh_ops.curl_toroidal_to_poloidal(E_tor, radius)
    # radial component has no curl contribution from purely tangential E in this approximation
    rad_mismatch = dB_rad_dt
    tor_mismatch = dB_tor_dt + curl_tor_from_pol
    pol_mismatch = dB_pol_dt + curl_pol_from_tor
    rms = torch.sqrt(
        torch.mean(torch.abs(rad_mismatch) ** 2 + torch.abs(tor_mismatch) ** 2 + torch.abs(pol_mismatch) ** 2)
    )
    return rms


def curl_spectral(toroidal: torch.Tensor, poloidal: torch.Tensor, radius: float):
    """Curl in spectral domain returning (toroidal_from_pol, poloidal_from_tor)."""
    tor_from_pol = vsh_ops.curl_poloidal_to_toroidal(poloidal, radius)
    pol_from_tor = vsh_ops.curl_toroidal_to_poloidal(toroidal, radius)
    return tor_from_pol, pol_from_tor


def curl_grid(vector_grid: torch.Tensor, positions: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """Approximate curl on grid via neighbor-based Jacobian fit."""
    n_nodes, k = neighbors.shape
    curl = torch.zeros((n_nodes, 3), device=vector_grid.device, dtype=vector_grid.dtype)
    for i in range(n_nodes):
        nbrs = neighbors[i]
        pos_i = positions[i]
        diffs_pos = positions[nbrs] - pos_i  # [k,3]
        diffs_vec = vector_grid[nbrs] - vector_grid[i]   # [k,3]
        ATA = diffs_pos.T @ diffs_pos
        ATY = diffs_pos.T @ diffs_vec
        J = torch.linalg.lstsq(ATA, ATY).solution  # [3,3]
        curl_vec = torch.stack([
            J[2,1] - J[1,2],
            J[0,2] - J[2,0],
            J[1,0] - J[0,1],
        ])
        curl[i] = curl_vec
    return curl
