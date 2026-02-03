#!/usr/bin/env python3
"""
Compute poloidal surface-current coefficients K_pol(l,m) given:

    - Toroidal E-field coefficients E_tor(l,m) (in T_lm basis)
    - Surface admittance coefficients Y_s(l,m) (scalar Ylm basis)

using real-space quadrature on the sphere.

All spectral arrays use the layout:
    shape = (lmax+1, 2*lmax+1)
with m-index = m + lmax and only entries |m| <= l for each l used.
"""

import numpy as np
from scipy.special import sph_harm


def lm_list(lmax: int):
    """
    Return ordered list of (l,m) pairs and a mapping functions:
      - lm_pairs: list of (l,m) in canonical order
      - lm_to_index(l,m): -> flat index 0..N-1
    """
    lm_pairs = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))
    index_map = { (l, m): i for i, (l, m) in enumerate(lm_pairs) }

    def lm_to_index(l: int, m: int) -> int:
        return index_map[(l, m)]

    return lm_pairs, lm_to_index


def flatten_lm(arr: np.ndarray, lmax: int) -> np.ndarray:
    """
    Flatten spectral array [lmax+1, 2*lmax+1] to [N] in canonical (l,m) order.
    """
    lm_pairs, lm_to_index = lm_list(lmax)
    flat = np.zeros(len(lm_pairs), dtype=arr.dtype)
    for l, m in lm_pairs:
        flat[lm_to_index(l, m)] = arr[l, lmax + m]
    return flat


def unflatten_lm(vec: np.ndarray, lmax: int) -> np.ndarray:
    """
    Inverse of flatten_lm: [N] -> [lmax+1, 2*lmax+1].
    """
    arr = np.zeros((lmax + 1, 2 * lmax + 1), dtype=vec.dtype)
    lm_pairs, lm_to_index = lm_list(lmax)
    for l, m in lm_pairs:
        arr[l, lmax + m] = vec[lm_to_index(l, m)]
    return arr


def compute_poloidal_current(
    E_tor_lm: np.ndarray,
    Y_s_lm: np.ndarray,
    lmax: int,
    n_theta: int = 64,
    n_phi: int = 128,
) -> np.ndarray:
    """
    Compute K_pol(l,m) given:
        E_tor_lm: complex array [lmax+1, 2*lmax+1] of toroidal E-field coeffs
        Y_s_lm:   complex/real  [lmax+1, 2*lmax+1] of scalar admittance coeffs
        lmax:     maximum degree
    Returns:
        K_pol_lm: complex array [lmax+1, 2*lmax+1] of poloidal current coeffs
    """

    # Basic checks
    assert E_tor_lm.shape == (lmax + 1, 2 * lmax + 1)
    assert Y_s_lm.shape == (lmax + 1, 2 * lmax + 1)

    # Build (theta, phi) grid
    # theta in [0, pi], phi in [0, 2pi)
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    # Broadcast to 2D grids
    TH, PH = np.meshgrid(theta, phi, indexing="ij")  # shape (n_theta, n_phi)

    # Precompute sin(theta) for weights and for gradient
    sinTH = np.sin(TH)
    # Avoid divide-by-zero at poles: clamp
    sinTH_safe = np.where(sinTH == 0, 1e-15, sinTH)

    # Build list of (l,m) and their indices
    lm_pairs, lm_to_index = lm_list(lmax)
    n_harm = len(lm_pairs)

    # 1) Evaluate all scalar Y_lm(theta,phi) on grid
    #    Y_grid: shape (n_theta, n_phi, n_harm)
    Y_grid = np.zeros((n_theta, n_phi, n_harm), dtype=np.complex128)

    for idx, (l, m) in enumerate(lm_pairs):
        # scipy sph_harm signature: m, l, phi, theta
        Y_grid[:, :, idx] = sph_harm(m, l, PH, TH)

    # 2) Build surface gradients to get P_lm and T_lm at every grid point
    #
    #   ∇_s Y = θ-hat ∂Y/∂θ + φ-hat (1/sinθ) ∂Y/∂φ
    #
    #   P_lm = (P_theta, P_phi) = (A, B)
    #   T_lm = (T_theta, T_phi) = (-B, A)
    #
    # We compute:
    #   A = ∂Y/∂θ via finite differences in theta
    #   B = (1/sinθ) * ∂Y/∂φ, with ∂/∂φ = i m Y
    #

    # Precompute m-values in a vector of length n_harm
    m_vals = np.array([m for (l, m) in lm_pairs])

    # dY/dphi = i m Y
    dY_dphi = 1j * m_vals[np.newaxis, np.newaxis, :] * Y_grid

    # B = (1/sinθ) * dY/dphi
    B = dY_dphi / sinTH_safe[:, :, np.newaxis]  # shape (n_theta, n_phi, n_harm)

    # A = dY/dtheta via finite differences along theta axis
    A = np.zeros_like(Y_grid)
    # central differences for interior points
    A[1:-1, :, :] = (Y_grid[2:, :, :] - Y_grid[:-2, :, :]) / (theta[2:, np.newaxis, np.newaxis] - theta[:-2, np.newaxis, np.newaxis])
    # forward/backward differences at the ends
    A[0, :, :]  = (Y_grid[1, :, :] - Y_grid[0, :, :]) / (theta[1] - theta[0])
    A[-1, :, :] = (Y_grid[-1, :, :] - Y_grid[-2, :, :]) / (theta[-1] - theta[-2])

    # Poloidal and toroidal basis components
    P_theta = A
    P_phi   = B
    T_theta = -B
    T_phi   = A

    # 3) Reconstruct E_tor(θ,φ) from E_tor_lm
    E_flat = flatten_lm(E_tor_lm, lmax)  # shape (n_harm,)

    # E_theta(phi,theta) = sum_{idx} E_flat[idx] * T_theta[..., idx]
    # E_phi(phi,theta)   = sum_{idx} E_flat[idx] * T_phi[..., idx]
    E_theta = np.tensordot(T_theta, E_flat, axes=([2], [0]))  # (n_theta, n_phi)
    E_phi   = np.tensordot(T_phi,   E_flat, axes=([2], [0]))  # (n_theta, n_phi)

    # 4) Reconstruct Y_s(θ,φ) from Y_s_lm
    Y_flat = flatten_lm(Y_s_lm, lmax)
    Y_s = np.tensordot(Y_grid, Y_flat, axes=([2], [0]))  # (n_theta, n_phi)

    # 5) Compute physical current K(θ,φ) = Y_s * E_tor
    K_theta = Y_s * E_theta
    K_phi   = Y_s * E_phi

    # 6) Project K onto P_lm to get K_pol(l,m)
    #    K_pol_{LM} = ∫ P_{LM}^* · K dΩ
    #
    # Discretization:
    #   dΩ ≈ sinθ_j dθ dφ
    #   sum over j,k: [P_theta^* K_theta + P_phi^* K_phi] sinθ_j dθ dφ
    #
    weight = sinTH * dtheta * dphi  # shape (n_theta, n_phi)

    K_pol_flat = np.zeros(n_harm, dtype=np.complex128)
    for idx, (L, M) in enumerate(lm_pairs):
        # Pick P_{LM}(θ,φ) components
        Pth = P_theta[:, :, idx]
        Pph = P_phi[:, :, idx]
        integrand = (np.conjugate(Pth) * K_theta + np.conjugate(Pph) * K_phi) * weight
        K_pol_flat[idx] = np.sum(integrand)

    # Convert flat to [lmax+1, 2*lmax+1]
    K_pol_lm = unflatten_lm(K_pol_flat, lmax)
    return K_pol_lm


if __name__ == "__main__":
    # Example usage / smoke test with small lmax
    lmax = 3

    # Dummy example: single toroidal mode E_tor_lm (say l=2,m=1)
    E_tor_lm = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    E_tor_lm[2, lmax + 1] = 1.0 + 0.0j  # E_{21}^{(T)} = 1

    # Dummy admittance: Y_s = Y00 + epsilon * Y10
    Y_s_lm = np.zeros_like(E_tor_lm)
    Y_s_lm[0, lmax + 0] = 1.0  # uniform piece
    Y_s_lm[1, lmax + 0] = 0.1  # small Y10 modulation

    K_pol_lm = compute_poloidal_current(E_tor_lm, Y_s_lm, lmax)
    print("K_pol(l,m) coefficients:")
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            val = K_pol_lm[l, lmax + m]
            if abs(val) > 1e-6:
                print(f"  l={l}, m={m}: {val}")
