"""
Helpers for evaluating emitted-field gradients at arbitrary radii.

Key notes:
- finite_diff_gradients_cartesian_closed_form(): central differences on Cartesian axes using closed-form toroidal field.
- finite_diff_gradients_spherical(): central differences in (r, theta, phi) using closed-form toroidal field with angular derivatives scaled to per-meter via 1/r and 1/(r sin theta).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
import math

import torch

from europa import inductance
from europa.observation import evaluate_field_from_spectral
from europa.transforms import sph_harm_fn

if TYPE_CHECKING:  # pragma: no cover
    from phasor_data import PhasorSimulation


def _cart_to_sph_components(B_cart: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
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
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    sin_p = torch.sin(phi)
    cos_p = torch.cos(phi)
    x = r * sin_t * cos_p
    y = r * sin_t * sin_p
    z = r * cos_t
    return torch.stack([x, y, z], dim=-1)


def sph_to_cart_coords(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return _sph_to_cart_coords(r, theta, phi)


def _cart_basis(positions: torch.Tensor):
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
    rhat, theta_hat, phi_hat = _cart_basis(positions)
    return Br[..., None] * rhat + Btheta[..., None] * theta_hat + Bphi[..., None] * phi_hat


def _sph_harm_with_derivs(positions: torch.Tensor, lmax: int, eps: float = 1e-6):
    import numpy as np

    pos_np = positions.detach().cpu().numpy()
    x, y, z = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    r = np.linalg.norm(pos_np, axis=1) + 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
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
    Y_np = np.transpose(Y, (2, 0, 1))
    dY_np = np.transpose(dY, (2, 0, 1))
    d2Y_np = np.transpose(d2Y, (2, 0, 1))
    return theta, phi, np.sin(theta), Y_np, dY_np, d2Y_np


def finite_diff_gradients_cartesian_closed_form(
    J_tor: torch.Tensor, radius: float, positions: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """
    Cartesian finite differences using the closed-form toroidal field evaluator.
    Returns [N,3,3] with dB_i/dx_j in Cartesian coordinates.
    """
    device = positions.device
    dtype = positions.dtype
    deltas = torch.eye(3, device=device, dtype=dtype) * delta
    grads = []
    for axis in range(3):
        shift = deltas[axis]
        pos_plus = positions + shift
        pos_minus = positions - shift
        Br_p, Bth_p, Bph_p = toroidal_field_spherical(J_tor, radius, pos_plus)
        Br_m, Bth_m, Bph_m = toroidal_field_spherical(J_tor, radius, pos_minus)
        Bp_cart = spherical_components_to_cart(Br_p, Bth_p, Bph_p, pos_plus)
        Bm_cart = spherical_components_to_cart(Br_m, Bth_m, Bph_m, pos_minus)
        grad_axis = (Bp_cart - Bm_cart) / (2.0 * delta)
        grads.append(grad_axis)
    return torch.stack(grads, dim=-1)  # [N,3,3]


def finite_diff_gradients_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    delta_r: float = 1.0,
    delta_theta: float = 1e-3,
    delta_phi: float = 1e-3,
) -> torch.Tensor:
    """
    Finite-difference gradients in spherical coordinates using closed-form toroidal field.
    Returns [N,3,3] with components [d/dr, (1/r)d/dtheta, (1/(r sin theta))d/dphi].
    """
    device = positions.device
    dtype = positions.dtype
    r = torch.linalg.norm(positions, dim=-1)
    r_safe = torch.where(r == 0, torch.ones_like(r), r)
    theta = torch.acos(torch.clamp(positions[:, 2] / r_safe, -1.0, 1.0))
    phi = torch.atan2(positions[:, 1], positions[:, 0])

    grads = []

    # d/dr
    pos_plus = _sph_to_cart_coords(r + delta_r, theta, phi).to(device=device, dtype=dtype)
    pos_minus = _sph_to_cart_coords(r - delta_r, theta, phi).to(device=device, dtype=dtype)
    Br_p, Bth_p, Bph_p = toroidal_field_spherical(J_tor, radius, pos_plus)
    Br_m, Bth_m, Bph_m = toroidal_field_spherical(J_tor, radius, pos_minus)
    grads.append((torch.stack([Br_p, Bth_p, Bph_p], dim=-1) - torch.stack([Br_m, Bth_m, Bph_m], dim=-1)) / (2.0 * delta_r))

    # d/dtheta -> per-meter
    pos_plus = _sph_to_cart_coords(r, theta + delta_theta, phi).to(device=device, dtype=dtype)
    pos_minus = _sph_to_cart_coords(r, theta - delta_theta, phi).to(device=device, dtype=dtype)
    Br_p, Bth_p, Bph_p = toroidal_field_spherical(J_tor, radius, pos_plus)
    Br_m, Bth_m, Bph_m = toroidal_field_spherical(J_tor, radius, pos_minus)
    grad_theta = (torch.stack([Br_p, Bth_p, Bph_p], dim=-1) - torch.stack([Br_m, Bth_m, Bph_m], dim=-1)) / (2.0 * delta_theta)
    grad_theta = grad_theta / r_safe[..., None]
    grads.append(grad_theta)

    # d/dphi -> per-meter
    pos_plus = _sph_to_cart_coords(r, theta, phi + delta_phi).to(device=device, dtype=dtype)
    pos_minus = _sph_to_cart_coords(r, theta, phi - delta_phi).to(device=device, dtype=dtype)
    Br_p, Bth_p, Bph_p = toroidal_field_spherical(J_tor, radius, pos_plus)
    Br_m, Bth_m, Bph_m = toroidal_field_spherical(J_tor, radius, pos_minus)
    grad_phi = (torch.stack([Br_p, Bth_p, Bph_p], dim=-1) - torch.stack([Br_m, Bth_m, Bph_m], dim=-1)) / (2.0 * delta_phi)
    sin_theta = torch.sin(theta)
    sin_theta_safe = torch.where(sin_theta == 0, torch.full_like(sin_theta, 1e-30), sin_theta)
    grad_phi = grad_phi / (r_safe[..., None] * sin_theta_safe[..., None])
    grads.append(grad_phi)

    return torch.stack(grads, dim=-1)  # [N,3,3]


def _toroidal_field_and_gradients_spherical_core(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
    import numpy as np

    device = positions.device
    lmax = J_tor.shape[-2] - 1
    if J_tor.shape[-1] != 2 * lmax + 1:
        raise ValueError(f"Expected J_tor shape (lmax+1, 2*lmax+1); got {tuple(J_tor.shape)}")

    pos_np = positions.detach().cpu().numpy()
    r = np.linalg.norm(pos_np, axis=1)
    r_safe = np.where(r == 0.0, 1e-30, r)
    theta = np.arccos(np.clip(pos_np[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(pos_np[:, 1], pos_np[:, 0])
    sin_th = np.sin(theta)
    sin_th_safe = np.where(sin_th == 0.0, 1e-30, sin_th)
    cos_th = np.cos(theta)

    _, _, _, Y, dY, d2Y = _sph_harm_with_derivs(positions, lmax, eps=theta_fd_step)

    ls = np.arange(0, lmax + 1, dtype=np.float64).reshape(1, lmax + 1, 1)
    ms = np.arange(-lmax, lmax + 1, dtype=np.float64).reshape(1, 1, 2 * lmax + 1)

    R = float(radius)
    r_ratio = (R / r_safe).reshape(-1, 1, 1)
    F_lm = np.power(r_ratio, ls + 2.0)
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
    J_lm = np.broadcast_to(J, (r.shape[0],) + J.shape)

    Br = np.sum(A_lm * J_lm * F_lm * Y, axis=(-2, -1))
    Btheta = np.sum(C_lm * J_lm * F_lm * dY, axis=(-2, -1))
    im_over_sin = 1j * ms / sin_th_safe[:, None, None]
    Bphi = np.sum(C_lm * J_lm * F_lm * (im_over_sin * Y), axis=(-2, -1))

    dF_dr = (-(ls + 2.0) / r_safe[:, None, None]) * F_lm

    dBr_dr = np.sum(A_lm * J_lm * dF_dr * Y, axis=(-2, -1))
    dBr_dth = np.sum(A_lm * J_lm * F_lm * dY, axis=(-2, -1))
    dBr_dph = np.sum(A_lm * J_lm * F_lm * (1j * ms * Y), axis=(-2, -1))
    grad_Br = np.stack([dBr_dr, dBr_dth / r_safe, dBr_dph / (r_safe * sin_th_safe)], axis=-1)

    dBth_dr = np.sum(C_lm * J_lm * dF_dr * dY, axis=(-2, -1))
    dBth_dth = np.sum(C_lm * J_lm * F_lm * d2Y, axis=(-2, -1))
    dBth_dph = np.sum(C_lm * J_lm * F_lm * (1j * ms * dY), axis=(-2, -1))
    grad_Btheta = np.stack([dBth_dr, dBth_dth / r_safe, dBth_dph / (r_safe * sin_th_safe)], axis=-1)

    term_theta = (1.0 / sin_th_safe)[:, None, None] * dY - (cos_th / (sin_th_safe ** 2))[:, None, None] * Y
    d_dth_im_over_sin_Y = (1j * ms) * term_theta
    m2_over_sin = (-(ms * ms) / sin_th_safe[:, None, None])

    dBph_dr = np.sum(C_lm * J_lm * dF_dr * (im_over_sin * Y), axis=(-2, -1))
    dBph_dth = np.sum(C_lm * J_lm * F_lm * d_dth_im_over_sin_Y, axis=(-2, -1))
    dBph_dph = np.sum(C_lm * J_lm * F_lm * (m2_over_sin * Y), axis=(-2, -1))
    grad_Bphi = np.stack([dBph_dr, dBph_dth / r_safe, dBph_dph / (r_safe * sin_th_safe)], axis=-1)

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
    Br, Btheta, Bphi, *_ = _toroidal_field_and_gradients_spherical_core(J_tor, radius, positions, theta_fd_step)
    return Br, Btheta, Bphi


def toroidal_gradients_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
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
    return _toroidal_field_and_gradients_spherical_core(J_tor, radius, positions, theta_fd_step)


def rss_gradient_from_emit(sim: "PhasorSimulation", positions: torch.Tensor, obs_radius: float | None = None) -> torch.Tensor:
    obs_r = sim.radius_m if obs_radius is None else float(obs_radius)
    K_tor = sim.K_toroidal
    if K_tor is None:
        raise ValueError("K_toroidal is required to evaluate emitted field at new radius.")
    _, _, _, grad_Br, grad_Btheta, grad_Bphi = toroidal_field_and_gradients_spherical(
        K_tor, radius=obs_r, positions=positions
    )
    grad_tensor = torch.stack([grad_Br, grad_Btheta, grad_Bphi], dim=1)  # [N,3,3]
    return torch.linalg.norm(grad_tensor, dim=(1, 2))


def rss_gradient_cartesian_autograd(J_tor: torch.Tensor, radius: float, positions: torch.Tensor) -> torch.Tensor:
    B_tor, B_pol, B_rad = inductance.spectral_b_from_surface_currents(J_tor, torch.zeros_like(J_tor), radius=radius)
    pos = positions.detach().requires_grad_(True)
    B_cart = evaluate_field_from_spectral(B_tor, B_pol, B_rad, pos)
    grads = []
    for comp in range(3):
        g_real = torch.autograd.grad(B_cart[:, comp].real.sum(), pos, retain_graph=True)[0]
        g_imag = torch.autograd.grad(B_cart[:, comp].imag.sum(), pos, retain_graph=True)[0]
        grads.append(g_real + 1j * g_imag)
    grad_tensor = torch.stack(grads, dim=1)
    return torch.linalg.norm(grad_tensor, dim=(1, 2))


def gradient_sanity_check(
    sim: "PhasorSimulation",
    altitude_m: float = 0.0,
    n_points: int = 24,
    seed: int | None = 0,
    delta_cart_m: float = 0.25,
    delta_r: float = 1e-4,
    delta_theta: float = 1e-4,
    delta_phi: float = 1e-4,
    theta_fd_step: float = 1e-6,
    device: str | None = None,
    verbose: bool = True,
    use_autograd: bool = False,
):
    if sim.K_toroidal is None:
        raise ValueError("sim.K_toroidal is required for this sanity check.")
    if device is None:
        device = sim.grid_positions.device if hasattr(sim, "grid_positions") else "cpu"
    if seed is not None:
        torch.manual_seed(seed)
    eps = torch.as_tensor(1e-30, device=device, dtype=torch.float64)

    u = torch.rand(n_points, device=device, dtype=torch.float64)
    v = torch.rand(n_points, device=device, dtype=torch.float64)
    theta = torch.acos(2.0 * u - 1.0)
    phi = 2.0 * torch.pi * v
    r_obs = float(sim.radius_m + altitude_m)
    positions = sph_to_cart_coords(
        torch.full((n_points,), r_obs, device=device, dtype=torch.float64),
        theta,
        phi,
    )

    R_source = float(sim.radius_m)
    J_tor = sim.K_toroidal.to(device=device)

    _, _, _, grad_Br, grad_Btheta, grad_Bphi = toroidal_field_and_gradients_spherical(
        J_tor, radius=R_source, positions=positions, theta_fd_step=theta_fd_step
    )
    rss_A = torch.linalg.norm(torch.stack([grad_Br, grad_Btheta, grad_Bphi], dim=1), dim=(1, 2))

    grad_sph_fd = finite_diff_gradients_spherical(
        J_tor, R_source, positions, delta_r=delta_r, delta_theta=delta_theta, delta_phi=delta_phi
    )
    rss_B = torch.linalg.norm(grad_sph_fd, dim=(1, 2))

    grad_cart_fd = finite_diff_gradients_cartesian_closed_form(J_tor, R_source, positions, delta=delta_cart_m)
    rss_cart_fd = torch.linalg.norm(grad_cart_fd, dim=(1, 2))

    rss_C = None
    rel_BC_cart = None
    if use_autograd:
        rss_C = rss_gradient_cartesian_autograd(J_tor, radius=R_source, positions=positions)
        rel_BC_cart = torch.abs(rss_cart_fd - rss_C) / torch.maximum(torch.abs(rss_C), eps)

    rel_AB = torch.abs(rss_A - rss_B) / torch.maximum(torch.abs(rss_B), eps)

    def _summary(x: torch.Tensor):
        x = x.detach().cpu()
        return {
            "min": float(x.min()),
            "median": float(x.median()),
            "mean": float(x.mean()),
            "max": float(x.max()),
        }

    out = {
        "rss_A_analytic_spherical": rss_A.detach().cpu(),
        "rss_B_fd_spherical": rss_B.detach().cpu(),
        "rss_fd_cartesian": rss_cart_fd.detach().cpu(),
        "rss_C_autograd_cartesian": rss_C.detach().cpu() if rss_C is not None else None,
        "rel_AB_vs_B": rel_AB.detach().cpu(),
        "rel_BC_cart_vs_C": rel_BC_cart.detach().cpu() if rel_BC_cart is not None else None,
        "summary": {
            "rel_AB_vs_B": _summary(rel_AB),
            "rel_BC_cart_vs_C": _summary(rel_BC_cart) if rel_BC_cart is not None else None,
        },
        "params": {
            "altitude_m": float(altitude_m),
            "n_points": int(n_points),
            "seed": None if seed is None else int(seed),
            "delta_cart_m": float(delta_cart_m),
            "delta_r": float(delta_r),
            "delta_theta": float(delta_theta),
            "delta_phi": float(delta_phi),
            "theta_fd_step": float(theta_fd_step),
            "R_source_m": float(R_source),
            "r_obs_m": float(r_obs),
            "device": str(device),
        },
        "positions_cart": positions.detach().cpu(),
        "positions_sph": {
            "r": torch.full((n_points,), r_obs, device=device, dtype=torch.float64).detach().cpu(),
            "theta": theta.detach().cpu(),
            "phi": phi.detach().cpu(),
        },
    }

    if verbose:
        print("\n=== Gradient sanity check ===")
        for k, v in out["params"].items():
            print(f"{k}: {v}")
        print("\nRelative errors (RSS gradients):")
        print("A vs B (relative to B):", out["summary"]["rel_AB_vs_B"])
        if rel_BC_cart is not None:
            print("FD Cartesian vs autograd Cartesian:", out["summary"]["rel_BC_cart_vs_C"])

    return out


def render_gradient_map(sim: "PhasorSimulation", altitude_m: float, subdivisions: int, save_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from render_phasor_maps import _build_mesh

    radius = sim.radius_m + altitude_m
    scale = radius / sim.radius_m
    positions = (sim.grid_positions * scale).to(dtype=torch.float64)

    rss = rss_gradient_from_emit(sim, positions, obs_radius=radius)

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
