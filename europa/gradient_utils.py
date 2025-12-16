"""
Helpers for evaluating emitted-field gradients at arbitrary radii.
Uses simple finite differences on Cartesian axes; a higher-order stencil would
improve accuracy at additional cost.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
import warnings

import torch

from europa import inductance
from europa.observation import evaluate_field_from_spectral
from europa.transforms import sph_harm_fn

if TYPE_CHECKING:  # pragma: no cover
    from phasor_data import PhasorSimulation


def _cart_to_sph_components(B_cart: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Project Cartesian B vectors to spherical components (B_r, B_theta, B_phi) using the local basis
    defined by (theta, phi). Shapes: B_cart [..., 3]; theta/phi broadcastable to B_cart.
    """
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    e_r = torch.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dim=-1)
    e_theta = torch.stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], dim=-1)
    e_phi = torch.stack([-sin_phi, cos_phi, torch.zeros_like(sin_phi)], dim=-1)
    B_r = (B_cart * e_r).sum(dim=-1)
    B_theta = (B_cart * e_theta).sum(dim=-1)
    B_phi = (B_cart * e_phi).sum(dim=-1)
    return torch.stack([B_r, B_theta, B_phi], dim=-1)


def _sph_to_cart_coords(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Convert spherical coordinates to Cartesian positions."""
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    sin_p = torch.sin(phi)
    cos_p = torch.cos(phi)
    x = r * sin_t * cos_p
    y = r * sin_t * sin_p
    z = r * cos_t
    return torch.stack([x, y, z], dim=-1)


def sph_to_cart_coords(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Public helper: convert spherical coordinates to Cartesian positions."""
    return _sph_to_cart_coords(r, theta, phi)


def _cart_basis(positions: torch.Tensor):
    """Unit vectors (r̂, θ̂, φ̂) at each Cartesian position."""
    r = torch.linalg.norm(positions, dim=-1, keepdim=True)
    r_safe = torch.where(r == 0, torch.full_like(r, 1e-30), r)
    rhat = positions / r_safe
    theta = torch.acos(torch.clamp(positions[..., 2] / r_safe[..., 0], -1.0, 1.0))
    phi = torch.atan2(positions[..., 1], positions[..., 0])
    theta_hat = torch.stack(
        [torch.cos(theta) * torch.cos(phi), torch.cos(theta) * torch.sin(phi), -torch.sin(theta)],
        dim=-1,
    )
    phi_hat = torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)], dim=-1)
    return rhat, theta_hat, phi_hat


def spherical_components_to_cart(Br: torch.Tensor, Btheta: torch.Tensor, Bphi: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Transform spherical B components at given positions into Cartesian components."""
    rhat, theta_hat, phi_hat = _cart_basis(positions)
    return Br[..., None] * rhat + Btheta[..., None] * theta_hat + Bphi[..., None] * phi_hat


def _sph_harm_with_derivs(positions: torch.Tensor, lmax: int, eps: float = 1e-6):
    """
    Compute Y_lm and its first/second theta derivatives for all positions (NumPy/CPU).
    TODO: Replace with a torch-native spherical harmonics implementation to enable end-to-end GPU execution.
    Returns numpy arrays:
        theta, phi: [N]
        sin_theta: [N]
        Y: [N, lmax+1, 2*lmax+1] (complex)
        dY_dtheta: same shape
        d2Y_dtheta2: same shape
    """
    import numpy as np

    pos_np = positions.detach().cpu().numpy()
    x, y, z = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    r = np.linalg.norm(pos_np, axis=1) + 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    sin_theta = np.sin(theta)

    n = pos_np.shape[0]
    Y = np.zeros((lmax + 1, 2 * lmax + 1, n), dtype=np.complex128)
    dY = np.zeros_like(Y)
    d2Y = np.zeros_like(Y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="`scipy.special.sph_harm` is deprecated")
        for l in range(lmax + 1):
            m_vals = np.arange(-l, l + 1)
            for idx, m in enumerate(m_vals):
                base = sph_harm_fn(m, l, phi, theta)
                plus = sph_harm_fn(m, l, phi, theta + eps)
                minus = sph_harm_fn(m, l, phi, theta - eps)
                dtheta = (plus - minus) / (2 * eps)
                d2theta = (plus - 2 * base + minus) / (eps ** 2)
                Y[l, lmax - l + idx] = base
                dY[l, lmax - l + idx] = dtheta
                d2Y[l, lmax - l + idx] = d2theta

    # Reorder to [N, l, m] for easier broadcasting with point dimension first
    Y_np = np.transpose(Y, (2, 0, 1))
    dY_np = np.transpose(dY, (2, 0, 1))
    d2Y_np = np.transpose(d2Y, (2, 0, 1))
    return theta, phi, sin_theta, Y_np, dY_np, d2Y_np


def finite_diff_gradients(
    B_tor: torch.Tensor,
    B_pol: torch.Tensor,
    B_rad: torch.Tensor,
    positions: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Finite-difference gradients of B by evaluating the field at +/- delta along Cartesian axes.
    Uses a simple central difference; a fully symmetric higher-order stencil would be more accurate
    but more expensive, and at meter-scale steps this approximation should be adequate.
    Returns [N, 3, 3] with dB_i/dx_j in Cartesian coordinates.
    """
    device = positions.device
    dtype = positions.dtype
    deltas = torch.eye(3, device=device, dtype=dtype) * delta
    grads = []
    for axis in range(3):
        shift = deltas[axis]
        pos_plus = positions + shift
        pos_minus = positions - shift
        B_plus = evaluate_field_from_spectral(B_tor, B_pol, B_rad, pos_plus)
        B_minus = evaluate_field_from_spectral(B_tor, B_pol, B_rad, pos_minus)
        grad_axis = (B_plus - B_minus) / (2.0 * delta)
        grads.append(grad_axis)
    grad_tensor = torch.stack(grads, dim=-1)  # [N,3,3] with last dim = axis
    return grad_tensor


def finite_diff_gradients_spherical(
    B_tor: torch.Tensor,
    B_pol: torch.Tensor,
    B_rad: torch.Tensor,
    positions: torch.Tensor,
    delta_r: float = 1.0,
    delta_theta: float = 1e-3,
    delta_phi: float = 1e-3,
) -> torch.Tensor:
    """
    Finite-difference gradients of B in spherical coordinates (r, theta, phi).
    Returns [N, 3, 3] with partials of (B_r, B_theta, B_phi) w.r.t (r, theta, phi).
    Perturbations: radial step delta_r (meters), angular steps delta_theta/delta_phi (radians).
    """
    device = positions.device
    dtype = positions.dtype
    # Spherical coords of base positions
    r = torch.linalg.norm(positions, dim=-1)
    r_safe = torch.where(r == 0, torch.ones_like(r), r)
    theta = torch.acos(torch.clamp(positions[:, 2] / r_safe, -1.0, 1.0))
    phi = torch.atan2(positions[:, 1], positions[:, 0])

    def _to_cart(r_val, theta_val, phi_val):
        sin_t = torch.sin(theta_val)
        cos_t = torch.cos(theta_val)
        sin_p = torch.sin(phi_val)
        cos_p = torch.cos(phi_val)
        return torch.stack([r_val * sin_t * cos_p, r_val * sin_t * sin_p, r_val * cos_t], dim=-1)

    grads = []
    # Perturbations for r, theta, phi
    shifts = [
        (delta_r, 0.0, 0.0),
        (0.0, delta_theta, 0.0),
        (0.0, 0.0, delta_phi),
    ]
    for dr, dt, dp in shifts:
        r_plus = r + dr
        r_minus = r - dr
        theta_plus = theta + dt
        theta_minus = theta - dt
        phi_plus = phi + dp
        phi_minus = phi - dp

        pos_plus = _to_cart(r_plus, theta_plus, phi_plus).to(device=device, dtype=dtype)
        pos_minus = _to_cart(r_minus, theta_minus, phi_minus).to(device=device, dtype=dtype)

        B_plus_cart = evaluate_field_from_spectral(B_tor, B_pol, B_rad, pos_plus)
        B_minus_cart = evaluate_field_from_spectral(B_tor, B_pol, B_rad, pos_minus)

        B_plus_sph = _cart_to_sph_components(B_plus_cart, theta_plus, phi_plus)
        B_minus_sph = _cart_to_sph_components(B_minus_cart, theta_minus, phi_minus)

        step = dr if dr != 0 else dt if dt != 0 else dp
        grad_axis = (B_plus_sph - B_minus_sph) / (2.0 * step)
        grads.append(grad_axis)

    grad_tensor = torch.stack(grads, dim=-1)  # [N,3,3] components x (r,theta,phi)
    return grad_tensor


def _toroidal_field_and_gradients_spherical_core(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
    """
    Analytic spherical components and gradients for toroidal surface currents (MQS exterior).
    NOTE: This implementation is NumPy/SciPy-backed. TODO: provide a torch-native spherical harmonic path
    to enable GPU execution end-to-end.
    Returns (B_r, B_theta, B_phi, grad_Br, grad_Btheta, grad_Bphi), with gradients expressed
    as [dr f, (1/r) dtheta f, (1/(r sin(theta))) dphi f].
    """
    import numpy as np

    device = positions.device
    lmax = J_tor.shape[-2] - 1
    if J_tor.shape[-1] != 2 * lmax + 1:
        raise ValueError(f"Expected J_tor shape (lmax+1, 2*lmax+1); got {tuple(J_tor.shape)}")

    # Positions to numpy
    pos_np = positions.detach().cpu().numpy()
    r = np.linalg.norm(pos_np, axis=1)
    r_safe = np.where(r == 0.0, 1e-30, r)
    theta = np.arccos(np.clip(pos_np[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(pos_np[:, 1], pos_np[:, 0])
    sin_th = np.sin(theta)
    sin_th_safe = np.where(sin_th == 0.0, 1e-30, sin_th)
    cos_th = np.cos(theta)

    # Harmonics and theta derivatives at points (shapes: [N, l, m])
    _, _, _, Y, dY, d2Y = _sph_harm_with_derivs(positions, lmax, eps=theta_fd_step)

    # Broadcast grids for l and m
    ls = np.arange(0, lmax + 1, dtype=np.float64).reshape(1, lmax + 1, 1)
    ms = np.arange(-lmax, lmax + 1, dtype=np.float64).reshape(1, 1, 2 * lmax + 1)

    # Radial factor F_l(r) = (R/r)^(l+2)
    R = float(radius)
    r_ratio = (R / r_safe).reshape(-1, 1, 1)
    F_lm = np.power(r_ratio, ls + 2.0)  # [N, l, m] via broadcast below
    F_lm = np.broadcast_to(F_lm, (r.shape[0], lmax + 1, 2 * lmax + 1)).astype(np.complex128)

    MU0_val = float(inductance.MU0)
    ell = ls * (ls + 1.0)
    two_l1 = 2.0 * ls + 1.0
    ell_safe = np.where(ell == 0.0, 1.0, ell)
    A_l = (-MU0_val / (two_l1 * ell_safe)).astype(np.complex128)
    C_l = (MU0_val * ls / two_l1).astype(np.complex128)
    A_lm = np.broadcast_to(A_l, (r.shape[0], lmax + 1, 2 * lmax + 1))
    C_lm = np.broadcast_to(C_l, (r.shape[0], lmax + 1, 2 * lmax + 1))

    J = np.asarray(J_tor.detach().cpu().numpy(), dtype=np.complex128)
    J_lm = np.broadcast_to(J, (r.shape[0],) + J.shape)  # [N,l,m]

    # Field components
    Br = np.sum(A_lm * J_lm * F_lm * Y, axis=(-2, -1))
    Btheta = np.sum(C_lm * J_lm * F_lm * dY, axis=(-2, -1))
    im_over_sin = 1j * ms / sin_th_safe[:, None, None]
    Bphi = np.sum(C_lm * J_lm * F_lm * (im_over_sin * Y), axis=(-2, -1))

    # Radial derivative of F_l
    dF_dr = (-(ls + 2.0) / r_safe[:, None, None]) * F_lm

    # grad Br
    dBr_dr = np.sum(A_lm * J_lm * dF_dr * Y, axis=(-2, -1))
    dBr_dth = np.sum(A_lm * J_lm * F_lm * dY, axis=(-2, -1))
    dBr_dph = np.sum(A_lm * J_lm * F_lm * (1j * ms * Y), axis=(-2, -1))
    grad_Br = np.stack(
        [
            dBr_dr,
            dBr_dth / r_safe,
            dBr_dph / (r_safe * sin_th_safe),
        ],
        axis=-1,
    )

    # grad Btheta
    dBth_dr = np.sum(C_lm * J_lm * dF_dr * dY, axis=(-2, -1))
    dBth_dth = np.sum(C_lm * J_lm * F_lm * d2Y, axis=(-2, -1))
    dBth_dph = np.sum(C_lm * J_lm * F_lm * (1j * ms * dY), axis=(-2, -1))
    grad_Btheta = np.stack(
        [
            dBth_dr,
            dBth_dth / r_safe,
            dBth_dph / (r_safe * sin_th_safe),
        ],
        axis=-1,
    )

    # grad Bphi
    term_theta = (1.0 / sin_th_safe)[:, None, None] * dY - (cos_th / (sin_th_safe ** 2))[:, None, None] * Y
    d_dth_im_over_sin_Y = (1j * ms) * term_theta
    m2_over_sin = (-(ms * ms) / sin_th_safe[:, None, None])

    dBph_dr = np.sum(C_lm * J_lm * dF_dr * (im_over_sin * Y), axis=(-2, -1))
    dBph_dth = np.sum(C_lm * J_lm * F_lm * d_dth_im_over_sin_Y, axis=(-2, -1))
    dBph_dph = np.sum(C_lm * J_lm * F_lm * (m2_over_sin * Y), axis=(-2, -1))
    grad_Bphi = np.stack(
        [
            dBph_dr,
            dBph_dth / r_safe,
            dBph_dph / (r_safe * sin_th_safe),
        ],
        axis=-1,
    )

    # Convert results back to torch on the original device
    def _to_tensor(arr):
        return torch.as_tensor(arr, device=device, dtype=torch.complex128)

    return (
        _to_tensor(Br),
        _to_tensor(Btheta),
        _to_tensor(Bphi),
        _to_tensor(grad_Br),
        _to_tensor(grad_Btheta),
        _to_tensor(grad_Bphi),
    )


def toroidal_field_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
    """
    Analytic spherical components (B_r, B_theta, B_phi) for toroidal surface currents (MQS exterior).
    """
    Br, Btheta, Bphi, *_ = _toroidal_field_and_gradients_spherical_core(J_tor, radius, positions, theta_fd_step)
    return Br, Btheta, Bphi


def toroidal_gradients_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
    """
    Analytic spherical gradients (∂/∂r, (1/r)∂/∂θ, (1/(r sinθ))∂/∂φ) of the toroidal field components.
    Returns grad_Br, grad_Btheta, grad_Bphi stacked as [..., 3].
    """
    *_, grad_Br, grad_Btheta, grad_Bphi = _toroidal_field_and_gradients_spherical_core(
        J_tor, radius, positions, theta_fd_step
    )
    return grad_Br, grad_Btheta, grad_Bphi


def toroidal_field_and_gradients_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
    """
    Analytic spherical components and gradients for toroidal surface currents (MQS exterior).
    Returns (B_r, B_theta, B_phi, grad_Br, grad_Btheta, grad_Bphi), with gradients expressed
    as [dr f, (1/r) dtheta f, (1/(r sin(theta))) dphi f].
    """
    return _toroidal_field_and_gradients_spherical_core(J_tor, radius, positions, theta_fd_step)


def toroidal_field_fd_gradients_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    delta_r: float = 1e-4,
    delta_theta: float = 1e-4,
    delta_phi: float = 1e-4,
) -> torch.Tensor:
    """
    Finite-difference spherical gradients of the toroidal field by re-evaluating the closed-form field.
    Returns [N,3,3] with partials of (B_r, B_theta, B_phi) w.r.t (r, theta, phi).
    """
    device = positions.device
    dtype = positions.dtype

    r = torch.linalg.norm(positions, dim=-1)
    r_safe = torch.where(r == 0, torch.full_like(r, 1e-30), r)
    theta = torch.acos(torch.clamp(positions[:, 2] / r_safe, -1.0, 1.0))
    phi = torch.atan2(positions[:, 1], positions[:, 0])

    deltas = (delta_r, delta_theta, delta_phi)
    grads_fd = []
    for axis, delta in enumerate(deltas):
        r_plus = r + (delta if axis == 0 else 0.0)
        r_minus = r - (delta if axis == 0 else 0.0)
        theta_plus = theta + (delta if axis == 1 else 0.0)
        theta_minus = theta - (delta if axis == 1 else 0.0)
        phi_plus = phi + (delta if axis == 2 else 0.0)
        phi_minus = phi - (delta if axis == 2 else 0.0)

        pos_plus = _sph_to_cart_coords(r_plus, theta_plus, phi_plus).to(device=device, dtype=dtype)
        pos_minus = _sph_to_cart_coords(r_minus, theta_minus, phi_minus).to(device=device, dtype=dtype)

        Br_p, Bth_p, Bph_p = toroidal_field_spherical(J_tor, radius, pos_plus)
        Br_m, Bth_m, Bph_m = toroidal_field_spherical(J_tor, radius, pos_minus)

        grad_axis = torch.stack(
            [
                (Br_p - Br_m) / (2 * delta),
                (Bth_p - Bth_m) / (2 * delta),
                (Bph_p - Bph_m) / (2 * delta),
            ],
            dim=-1,
        )
        grads_fd.append(grad_axis)

    grad_tensor = torch.stack(grads_fd, dim=-2)  # [N,3,3] d(component)/d(r,theta,phi)
    return grad_tensor


def rss_gradient_from_emit(sim: "PhasorSimulation", positions: torch.Tensor, obs_radius: float | None = None, delta: float = 1.0) -> torch.Tensor:
    """
    Compute RSS of emitted B-field gradient (|∇B|) at specified positions by re-evaluating field at obs_radius.
    """
    obs_r = sim.radius_m if obs_radius is None else float(obs_radius)
    K_tor = sim.K_toroidal
    K_pol = sim.K_poloidal if sim.K_poloidal is not None else torch.zeros_like(K_tor)
    if K_tor is None:
        raise ValueError("K_toroidal is required to evaluate emitted field at new radius.")
    B_tor, B_pol, B_rad = inductance.spectral_b_from_surface_currents(K_tor, K_pol, radius=obs_r)
    grad = finite_diff_gradients(B_tor, B_pol, B_rad, positions, delta=delta)
    rss = torch.sqrt((grad.abs() ** 2).sum(dim=(-2, -1)))
    return rss


def render_gradient_map(sim: "PhasorSimulation", altitude_m: float, subdivisions: int, save_path: str, title: str) -> None:
    """Render a sphere heat map of RSS(|∇B_emit|) at a given altitude."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from render_phasor_maps import _build_mesh

    radius = sim.radius_m + altitude_m
    scale = radius / sim.radius_m
    positions = (sim.grid_positions * scale).to(dtype=torch.float64)
    rss = rss_gradient_from_emit(sim, positions, obs_radius=radius, delta=1.0)

    vertices, faces, centers = _build_mesh(radius, subdivisions=subdivisions, stride=1)
    positions = positions.to(dtype=centers.dtype)
    dists = torch.cdist(centers, positions)
    nearest = dists.argmin(dim=1)
    face_vals = rss[nearest].cpu().numpy()

    tri_verts = vertices[faces].cpu().numpy()
    vmax = float(face_vals.max()) if face_vals.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    colors = cmap(norm(face_vals))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    collection = Poly3DCollection(
        tri_verts,
        facecolors=colors,
        edgecolor="none",
        linewidth=0.05,
        antialiased=True,
    )
    ax.add_collection3d(collection)
    lim = radius * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(title, pad=12)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(face_vals)
    fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.05, label="|grad_B_emit| RSS (T/m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
