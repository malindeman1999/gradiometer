"""
Spherical harmonic transforms with weighted pseudoinverse for HEALPix sampling.
Vector transforms use true poloidal/toroidal vector spherical harmonics (approximate numerical basis).
"""
from typing import Dict, Tuple

import numpy as np
import torch

# Use SciPy's complex spherical harmonics; prefer sph_harm for consistent signature (m, l, phi, theta)
from scipy.special import sph_harm as sph_harm_fn


def _angles_from_positions(positions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian positions to spherical angles (theta, phi)."""
    pos_np = positions.detach().cpu().numpy()
    x, y, z = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    return theta, phi


def _get_scalar_matrices(positions: torch.Tensor, lmax: int, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build weighted spherical harmonic matrix and its pseudoinverse for given nodes."""
    theta, phi = _angles_from_positions(positions)
    n_nodes = positions.shape[0]
    n_coeff = (lmax + 1) ** 2
    Y = np.zeros((n_nodes, n_coeff), dtype=np.complex128)
    col = 0
    for l in range(lmax + 1):
        m_vals = np.arange(-l, l + 1)
        # sph_harm signature: sph_harm(m, l, phi, theta) with phi=azimuth, theta=polar
        Y[:, col:col + (2 * l + 1)] = np.column_stack([sph_harm_fn(m, l, phi, theta) for m in m_vals])
        col += 2 * l + 1
    w = weights.detach().cpu().numpy()
    W_sqrt = np.sqrt(w)[:, None]
    Y_w = Y * W_sqrt
    pinv = np.linalg.pinv(Y_w)
    Y_w_torch = torch.from_numpy(Y_w).to(device=positions.device)
    pinv_torch = torch.from_numpy(pinv).to(device=positions.device)
    # Preserve a 1D shape even for a single node (avoids scalar squeeze issues downstream).
    W_sqrt_torch = torch.from_numpy(W_sqrt.reshape(-1)).to(device=positions.device, dtype=torch.float64)
    return Y_w_torch, pinv_torch, W_sqrt_torch


def _pad_coeffs(coeffs_flat: torch.Tensor, lmax: int, template_shape) -> torch.Tensor:
    """Reshape flattened SH coefficients into (l,m) grid with symmetric m index layout."""
    out = torch.zeros(template_shape, device=coeffs_flat.device, dtype=coeffs_flat.dtype)
    offset = 0
    for l in range(lmax + 1):
        count = 2 * l + 1
        sl = coeffs_flat[..., offset:offset + count]
        out[..., l, lmax - l:lmax + l + 1] = sl
        offset += count
    return out


def _unpad_coeffs(coeffs: torch.Tensor) -> torch.Tensor:
    """Flatten (l,m) layout back to contiguous coefficient vector."""
    lmax = coeffs.shape[-2] - 1
    parts = []
    for l in range(lmax + 1):
        parts.append(coeffs[..., l, lmax - l:lmax + l + 1].reshape(coeffs.shape[:-2] + (2 * l + 1,)))
    return torch.cat(parts, dim=-1)


def sh_forward(grid_values: torch.Tensor, positions: torch.Tensor, lmax: int, weights: torch.Tensor) -> torch.Tensor:
    """Project scalar grid values to SH coefficients using weighted pseudoinverse.
    Args:
        grid_values: [..., N_nodes]
        positions: [N_nodes,3]
        lmax: maximum degree
        weights: [N_nodes] quadrature weights
    Returns:
        coeffs: [..., lmax+1, 2*lmax+1] with m aligned to index m+lmax
    """
    Y_w, pinv, w_sqrt = _get_scalar_matrices(positions, lmax, weights)
    n_nodes = positions.shape[0]
    vals = grid_values.reshape(-1, n_nodes).to(dtype=torch.complex128)
    vals_w = vals * w_sqrt[None, :].to(dtype=torch.complex128)
    coeffs_flat = vals_w @ pinv.T
    full_shape = grid_values.shape[:-1] + (lmax + 1, 2 * lmax + 1)
    coeffs = _pad_coeffs(coeffs_flat, lmax, full_shape)
    return coeffs


def sh_inverse(coeffs: torch.Tensor, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reconstruct scalar values on grid from SH coefficients (complex-aware)."""
    lmax = coeffs.shape[-2] - 1
    Y_w, _, w_sqrt = _get_scalar_matrices(positions, lmax, weights)
    n_nodes = positions.shape[0]
    coeffs_flat = _unpad_coeffs(coeffs)
    coeffs_flat = coeffs_flat.to(dtype=Y_w.dtype)
    recon_w = coeffs_flat @ torch.conj(Y_w.T)
    recon = recon_w / w_sqrt[None, :].to(dtype=recon_w.dtype)
    return recon.reshape(coeffs.shape[:-2] + (n_nodes,))


def _compute_vsh_basis(positions: torch.Tensor, lmax: int):
    """
    Compute toroidal/poloidal VSH basis vectors at grid positions (numerical gradient).
    Returns complex basis arrays shaped [lmax+1, 2*lmax+1, N, 3] (CPU numpy).
    """
    pos_np = positions.detach().cpu().numpy()
    x, y, z = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    r = np.linalg.norm(pos_np, axis=1) + 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    r_hat = (pos_np / r[:, None]).astype(np.float64)

    theta_hat = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        axis=1,
    )
    phi_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)

    basis_tor = np.zeros((lmax + 1, 2 * lmax + 1, pos_np.shape[0], 3), dtype=np.complex128)
    basis_pol = np.zeros_like(basis_tor)

    eps = 1e-6
    sin_theta = np.sin(theta) + 1e-12
    for l in range(lmax + 1):
        m_vals = np.arange(-l, l + 1)
        for idx_m, m in enumerate(m_vals):
            Y = sph_harm_fn(m, l, phi, theta)
            # derivatives
            Y_theta = (sph_harm_fn(m, l, phi, theta + eps) - sph_harm_fn(m, l, phi, theta - eps)) / (2 * eps)
            Y_phi = 1j * m * Y
            g_theta = Y_theta
            g_phi = Y_phi / sin_theta
            grad = g_theta[:, None] * theta_hat + g_phi[:, None] * phi_hat
            tor_vec = np.cross(r_hat, grad)
            basis_tor[l, lmax - l + idx_m] = tor_vec
            basis_pol[l, lmax - l + idx_m] = grad
    return basis_tor, basis_pol


def vsh_forward(vector_grid: torch.Tensor, positions: torch.Tensor, lmax: int, weights: torch.Tensor):
    """
    Project tangential vector field onto toroidal/poloidal VSH via weighted inner products.
    Returns complex coefficients shaped [lmax+1, 2*lmax+1].
    """
    vec = np.asarray(vector_grid.detach().cpu().numpy(), dtype=np.complex128)
    w = weights.detach().cpu().numpy().astype(np.float64)
    basis_tor, basis_pol = _compute_vsh_basis(positions, lmax)
    tor_coeffs = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    pol_coeffs = np.zeros_like(tor_coeffs)
    for l in range(lmax + 1):
        for m_idx in range(2 * l + 1):
            b_tor = basis_tor[l, lmax - l + m_idx]
            b_pol = basis_pol[l, lmax - l + m_idx]
            denom_t = np.sum(w * np.einsum("ij,ij->i", np.conj(b_tor), b_tor).real) + 1e-12
            denom_p = np.sum(w * np.einsum("ij,ij->i", np.conj(b_pol), b_pol).real) + 1e-12
            num_t = np.sum(w * np.einsum("ij,ij->i", np.conj(b_tor), vec))
            num_p = np.sum(w * np.einsum("ij,ij->i", np.conj(b_pol), vec))
            tor_coeffs[l, lmax - l + m_idx] = num_t / denom_t
            pol_coeffs[l, lmax - l + m_idx] = num_p / denom_p
    tor_t = torch.from_numpy(tor_coeffs).to(positions.device)
    pol_t = torch.from_numpy(pol_coeffs).to(positions.device)
    return tor_t, pol_t


def vsh_inverse(toroidal: torch.Tensor, poloidal: torch.Tensor, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct tangential vector field from toroidal/poloidal coefficients using VSH basis.
    Returns complex reconstruction (tangential vector in Cartesian components).
    """
    lmax = toroidal.shape[-2] - 1
    basis_tor, basis_pol = _compute_vsh_basis(positions, lmax)
    vec = np.zeros((positions.shape[0], 3), dtype=np.complex128)
    tor_np = toroidal.detach().cpu().numpy()
    pol_np = poloidal.detach().cpu().numpy()
    for l in range(lmax + 1):
        for m_idx in range(2 * l + 1):
            c_t = tor_np[l, lmax - l + m_idx]
            c_p = pol_np[l, lmax - l + m_idx]
            vec += c_t * basis_tor[l, lmax - l + m_idx]
            vec += c_p * basis_pol[l, lmax - l + m_idx]
    vec_torch = torch.from_numpy(vec).to(positions.device, dtype=torch.complex128)
    return vec_torch
