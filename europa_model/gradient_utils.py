"""
Helpers for evaluating emitted-field gradients at arbitrary radii.

Key notes:
- finite_diff_gradients_cartesian_closed_form(): central differences on Cartesian axes using closed-form toroidal field.
- finite_diff_gradients_spherical(): central differences in (r, theta, phi) using closed-form toroidal field with angular derivatives scaled to per-meter via 1/r and 1/(r sin theta).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import math
import time

import torch
import numpy as np

from europa_model import inductance
from europa_model.observation import evaluate_field_from_spectral
from workflow.plotting.sphere_roundtrip import DEFAULT_SPHERE_ELEV, DEFAULT_SPHERE_AZIM

if TYPE_CHECKING:  # pragma: no cover
    from workflow.data_objects.phasor_data import PhasorSimulation


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


def finite_diff_gradients_cartesian_closed_form(
    J_tor: torch.Tensor, radius: float, positions: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """
    Cartesian finite differences using the closed-form toroidal field evaluator.
    Returns [N,3,3] with dB_i/dx_j in Cartesian coordinates.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]
    deltas = torch.eye(3, device=device, dtype=dtype) * delta

    shifted = []
    for axis in range(3):
        shift = deltas[axis]
        shifted.append(positions + shift)
        shifted.append(positions - shift)
    shifted_all = torch.cat(shifted, dim=0)  # [6N, 3]

    Br, Bth, Bph = toroidal_field_spherical(J_tor, radius, shifted_all)
    B_cart_all = spherical_components_to_cart(Br, Bth, Bph, shifted_all).reshape(3, 2, n, 3)

    grads = []
    for axis in range(3):
        Bp_cart = B_cart_all[axis, 0]
        Bm_cart = B_cart_all[axis, 1]
        grads.append((Bp_cart - Bm_cart) / (2.0 * delta))
    return torch.stack(grads, dim=-1)  # [N,3,3]


def finite_diff_gradients_cartesian_closed_form_forward(
    J_tor: torch.Tensor, radius: float, positions: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """
    Forward differences using a shared base point:
    dB_i/dx_j ~= (B_i(x + delta e_j) - B_i(x)) / delta
    Returns [N,3,3] with dB_i/dx_j in Cartesian coordinates.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]
    deltas = torch.eye(3, device=device, dtype=dtype) * delta

    shifted = [positions]
    for axis in range(3):
        shifted.append(positions + deltas[axis])
    shifted_all = torch.cat(shifted, dim=0)  # [4N, 3]

    Br, Bth, Bph = toroidal_field_spherical(J_tor, radius, shifted_all)
    B_cart_all = spherical_components_to_cart(Br, Bth, Bph, shifted_all).reshape(4, n, 3)
    B0 = B_cart_all[0]
    grads = []
    for axis in range(3):
        Bp = B_cart_all[axis + 1]
        grads.append((Bp - B0) / delta)
    return torch.stack(grads, dim=-1)  # [N,3,3]


def finite_diff_gradients_cartesian_from_spectral(
    B_tor: torch.Tensor,
    B_pol: torch.Tensor,
    B_rad: torch.Tensor,
    positions: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Cartesian finite differences of the emitted field evaluated directly from spectra.
    Returns [N,3,3] with dB_i/dx_j in Cartesian coordinates.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]
    deltas = torch.eye(3, device=device, dtype=dtype) * delta

    shifted = []
    for axis in range(3):
        shift = deltas[axis]
        shifted.append(positions + shift)
        shifted.append(positions - shift)
    shifted_all = torch.cat(shifted, dim=0)  # [6N, 3]

    B_cart_all = evaluate_field_from_spectral(B_tor, B_pol, B_rad, shifted_all).reshape(3, 2, n, 3)
    grads = []
    for axis in range(3):
        Bp_cart = B_cart_all[axis, 0]
        Bm_cart = B_cart_all[axis, 1]
        grads.append((Bp_cart - Bm_cart) / (2.0 * delta))
    return torch.stack(grads, dim=-1)  # [N,3,3]


def finite_diff_gradients_cartesian_from_spectral_forward(
    B_tor: torch.Tensor,
    B_pol: torch.Tensor,
    B_rad: torch.Tensor,
    positions: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Forward differences from spectral field evaluation using a shared base point.
    Returns [N,3,3] with dB_i/dx_j in Cartesian coordinates.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]
    deltas = torch.eye(3, device=device, dtype=dtype) * delta

    shifted = [positions]
    for axis in range(3):
        shifted.append(positions + deltas[axis])
    shifted_all = torch.cat(shifted, dim=0)  # [4N, 3]

    B_cart_all = evaluate_field_from_spectral(B_tor, B_pol, B_rad, shifted_all).reshape(4, n, 3)
    B0 = B_cart_all[0]
    grads = []
    for axis in range(3):
        Bp = B_cart_all[axis + 1]
        grads.append((Bp - B0) / delta)
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


def toroidal_field_spherical(
    J_tor: torch.Tensor,
    radius: float,
    positions: torch.Tensor,
    theta_fd_step: float = 1e-6,
):
    _ = theta_fd_step
    zeros = torch.zeros_like(J_tor)
    B_tor, B_pol, B_rad = inductance.spectral_b_from_surface_currents(J_tor, zeros, radius=radius)
    B_cart = evaluate_field_from_spectral(B_tor, B_pol, B_rad, positions)
    r = torch.linalg.norm(positions, dim=-1)
    r_safe = torch.where(r == 0, torch.full_like(r, 1e-30), r)
    theta = torch.acos(torch.clamp(positions[..., 2] / r_safe, -1.0, 1.0))
    phi = torch.atan2(positions[..., 1], positions[..., 0])
    B_sph = _cart_to_sph_components(B_cart, theta, phi)
    Br, Btheta, Bphi = B_sph[..., 0], B_sph[..., 1], B_sph[..., 2]
    return Br, Btheta, Bphi


def rss_gradient_from_emit(
    sim: "PhasorSimulation",
    positions: torch.Tensor,
    obs_radius: float | None = None,
    *,
    fd_delta_m: float = 1000.0,
    method: str = "cartesian_spectral",
    fd_scheme: str = "forward",
) -> torch.Tensor:
    _ = sim.radius_m if obs_radius is None else float(obs_radius)
    K_tor = sim.K_toroidal
    if K_tor is None:
        raise ValueError("K_toroidal is required to evaluate emitted field at new radius.")

    method_key = str(method).strip().lower()
    fd_scheme_key = str(fd_scheme).strip().lower()
    if fd_scheme_key not in {"forward", "central"}:
        raise ValueError(f"Unknown finite-difference scheme: {fd_scheme}")
    delta = float(max(1e-6, fd_delta_m))
    source_radius = float(sim.radius_m)
    if method_key == "cartesian_spectral":
        if sim.B_tor_emit is None or sim.B_pol_emit is None or sim.B_rad_emit is None:
            method_key = "cartesian_closed_form"
        else:
            if fd_scheme_key == "forward":
                grad_cart = finite_diff_gradients_cartesian_from_spectral_forward(
                    sim.B_tor_emit,
                    sim.B_pol_emit,
                    sim.B_rad_emit,
                    positions=positions,
                    delta=delta,
                )
            else:
                grad_cart = finite_diff_gradients_cartesian_from_spectral(
                    sim.B_tor_emit,
                    sim.B_pol_emit,
                    sim.B_rad_emit,
                    positions=positions,
                    delta=delta,
                )
            return torch.linalg.norm(grad_cart, dim=(1, 2))

    if method_key == "cartesian_closed_form":
        if fd_scheme_key == "forward":
            grad_cart = finite_diff_gradients_cartesian_closed_form_forward(
                K_tor,
                radius=source_radius,
                positions=positions,
                delta=delta,
            )
        else:
            grad_cart = finite_diff_gradients_cartesian_closed_form(
                K_tor,
                radius=source_radius,
                positions=positions,
                delta=delta,
            )
        return torch.linalg.norm(grad_cart, dim=(1, 2))

    raise ValueError(f"Unknown gradient RSS method: {method}")


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

    _ = theta_fd_step
    grad_sph_fd = finite_diff_gradients_spherical(
        J_tor, R_source, positions, delta_r=delta_r, delta_theta=delta_theta, delta_phi=delta_phi
    )
    rss_sph_fd = torch.linalg.norm(grad_sph_fd, dim=(1, 2))

    grad_cart_fd = finite_diff_gradients_cartesian_closed_form(J_tor, R_source, positions, delta=delta_cart_m)
    rss_cart_fd = torch.linalg.norm(grad_cart_fd, dim=(1, 2))

    rss_C = None
    rel_BC_cart = None
    if use_autograd:
        rss_C = rss_gradient_cartesian_autograd(J_tor, radius=R_source, positions=positions)
        rel_BC_cart = torch.abs(rss_cart_fd - rss_C) / torch.maximum(torch.abs(rss_C), eps)

    rel_AB = torch.abs(rss_cart_fd - rss_sph_fd) / torch.maximum(torch.abs(rss_sph_fd), eps)

    def _summary(x: torch.Tensor):
        x = x.detach().cpu()
        return {
            "min": float(x.min()),
            "median": float(x.median()),
            "mean": float(x.mean()),
            "max": float(x.max()),
        }

    out = {
        "rss_fd_spherical": rss_sph_fd.detach().cpu(),
        "rss_fd_cartesian": rss_cart_fd.detach().cpu(),
        "rss_C_autograd_cartesian": rss_C.detach().cpu() if rss_C is not None else None,
        "rel_cart_vs_sph_fd": rel_AB.detach().cpu(),
        "rel_BC_cart_vs_C": rel_BC_cart.detach().cpu() if rel_BC_cart is not None else None,
        "summary": {
            "rel_cart_vs_sph_fd": _summary(rel_AB),
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
        print("FD Cartesian vs FD spherical (relative to spherical):", out["summary"]["rel_cart_vs_sph_fd"])
        if rel_BC_cart is not None:
            print("FD Cartesian vs autograd Cartesian:", out["summary"]["rel_BC_cart_vs_C"])

    return out


def render_gradient_map(
    sim: "PhasorSimulation",
    altitude_m: float,
    save_path: str,
    title: str,
    faces: torch.Tensor | None = None,
    plotter: str = "pyvista",
    subdivisions: int = 0,
    log_scale: bool = False,
    timing_log=None,
    fd_scheme: str = "forward",
    rss_values: torch.Tensor | np.ndarray | None = None,
    positions_override: torch.Tensor | np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if positions_override is None:
        radius = sim.radius_m + altitude_m
        scale = radius / sim.radius_m
        positions = (sim.grid_positions * scale).to(dtype=torch.float64)
    else:
        positions = positions_override
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).to(dtype=torch.float64)
        else:
            positions = positions.to(dtype=torch.float64)

    if rss_values is None:
        t_grad0 = time.perf_counter()
        rss = rss_gradient_from_emit(sim, positions, obs_radius=float(sim.radius_m + altitude_m), fd_scheme=fd_scheme).cpu().numpy()
        grad_dt = time.perf_counter() - t_grad0
        if timing_log is not None:
            try:
                timing_log(
                    f"Gradient compute only: {grad_dt:.2f}s "
                    f"(nodes={positions.shape[0]}, method=cartesian_spectral, scheme={fd_scheme})"
                )
            except Exception:
                pass
    elif isinstance(rss_values, np.ndarray):
        rss = rss_values
    else:
        rss = rss_values.detach().cpu().numpy()

    pts = positions.detach().cpu().numpy()
    if faces is None:
        from scipy.spatial import ConvexHull
        face_np = ConvexHull(pts).simplices.astype(np.int64)
    else:
        face_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else faces
    show_matplotlib = plotter != "pyvista"
    vmin_data = float(rss.min()) if rss.size else 0.0
    vmax_data = float(rss.max()) if rss.size else 1.0
    if vmax_data <= vmin_data:
        vmax_data = vmin_data + max(abs(vmin_data) * 1e-6, 1e-30)
    if plotter == "pyvista":
        pyvista_shown = False
        try:
            import pyvista as pv

            face_prefix = np.full((face_np.shape[0], 1), 3, dtype=np.int64)
            pv_faces = np.hstack((face_prefix, face_np)).reshape(-1)
            mesh = pv.PolyData(pts, pv_faces)
            mesh.point_data["value"] = rss
            pl = pv.Plotter(off_screen=False, window_size=(1100, 900))
            pl.set_background("white")
            pl.add_mesh(
                mesh,
                scalars="value",
                cmap="rainbow",
                clim=[vmin_data, vmax_data],
                show_edges=False,
                smooth_shading=False,
                lighting=False,
            )
            pl.add_title(title)
            pl.view_isometric()
            pl.show(auto_close=False)
            pyvista_shown = True
            try:
                pl.screenshot(save_path)
            except Exception:
                pass
            pl.close()
            pyvista_shown = True
        except Exception:
            if pyvista_shown:
                pass

    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    face_vals = rss[face_np].mean(axis=1)
    tri_verts = pts[face_np]
    vmin = float(face_vals.min()) if face_vals.size else 0.0
    vmax = float(face_vals.max()) if face_vals.size else 1.0
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 1e-6, 1e-30)
    if log_scale:
        positive = face_vals[face_vals > 0.0]
        if positive.size:
            vmin = float(np.quantile(positive, 0.01))
            vmin = max(vmin, float(np.min(positive)))
        else:
            vmin = 1e-30
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("rainbow")
    colors = cmap(norm(face_vals))

    lim = float(np.max(np.abs(pts)))
    lim_save = lim * 0.8  # zoom in so spheres appear ~25% larger in saved figure

    # Saved figure: two views with shared colorbar.
    fig_save = plt.figure(figsize=(10.5, 6))
    grid = fig_save.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=-0.02)
    axes_save = [
        fig_save.add_subplot(grid[0, 0], projection="3d"),
        fig_save.add_subplot(grid[0, 1], projection="3d"),
    ]
    cax = fig_save.add_subplot(grid[0, 2])
    for ax, view_label, azim in (
        (axes_save[0], "Front", DEFAULT_SPHERE_AZIM),
        (axes_save[1], "Back", DEFAULT_SPHERE_AZIM + 180.0),
    ):
        collection = Poly3DCollection(
            tri_verts,
            facecolors=colors,
            edgecolor="none",
            linewidth=0.05,
            antialiased=True,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(-lim_save, lim_save)
        ax.set_ylim(-lim_save, lim_save)
        ax.set_zlim(-lim_save, lim_save)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(view_label, pad=8)
        ax.view_init(elev=DEFAULT_SPHERE_ELEV, azim=azim)
    fig_save.suptitle(title, y=0.98)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(face_vals)
    fig_save.colorbar(
        mappable,
        cax=cax,
        label="|grad_B_emit| RSS (T/m)" + (" [log]" if log_scale else ""),
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig_save)

    if show_matplotlib:
        # On-screen plot: match the original single-view display.
        fig_show = plt.figure(figsize=(7, 6))
        ax = fig_show.add_subplot(1, 1, 1, projection="3d")
        collection = Poly3DCollection(
            tri_verts,
            facecolors=colors,
            edgecolor="none",
            linewidth=0.05,
            antialiased=True,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(title, pad=12)
        ax.view_init(elev=DEFAULT_SPHERE_ELEV, azim=DEFAULT_SPHERE_AZIM)
        fig_show.colorbar(
            mappable,
            ax=ax,
            shrink=0.8,
            pad=0.05,
            label="|grad_B_emit| RSS (T/m)" + (" [log]" if log_scale else ""),
        )
        plt.tight_layout()
        plt.show()


def render_b_magnitude_map(
    sim: "PhasorSimulation",
    altitude_m: float,
    save_path: str,
    title: str,
    faces: torch.Tensor | None = None,
    plotter: str = "pyvista",
    subdivisions: int = 0,
    log_scale: bool = False,
    rss_values: torch.Tensor | np.ndarray | None = None,
    positions_override: torch.Tensor | np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if sim.K_toroidal is None:
        raise ValueError("K_toroidal is required to render emitted-field magnitude.")

    if positions_override is None:
        obs_radius = float(sim.radius_m + altitude_m)
        scale = obs_radius / float(sim.radius_m)
        positions = (sim.grid_positions * scale).to(dtype=torch.float64)
    else:
        positions = positions_override
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).to(dtype=torch.float64)
        else:
            positions = positions.to(dtype=torch.float64)

    if rss_values is None:
        Br, Btheta, Bphi = toroidal_field_spherical(sim.K_toroidal, radius=float(sim.radius_m), positions=positions)
        rss = torch.sqrt(torch.abs(Br) ** 2 + torch.abs(Btheta) ** 2 + torch.abs(Bphi) ** 2).cpu().numpy()
    elif isinstance(rss_values, np.ndarray):
        rss = rss_values
    else:
        rss = rss_values.detach().cpu().numpy()

    pts = positions.detach().cpu().numpy()
    if faces is None:
        from scipy.spatial import ConvexHull
        face_np = ConvexHull(pts).simplices.astype(np.int64)
    else:
        face_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else faces
    show_matplotlib = plotter != "pyvista"
    if plotter == "pyvista":
        pyvista_shown = False
        try:
            import pyvista as pv

            face_prefix = np.full((face_np.shape[0], 1), 3, dtype=np.int64)
            pv_faces = np.hstack((face_prefix, face_np)).reshape(-1)
            mesh = pv.PolyData(pts, pv_faces)
            mesh.point_data["value"] = rss
            pl = pv.Plotter(off_screen=False, window_size=(1100, 900))
            pl.set_background("white")
            pl.add_mesh(
                mesh,
                scalars="value",
                cmap="rainbow",
                show_edges=False,
                smooth_shading=False,
                lighting=False,
            )
            pl.add_title(title)
            pl.view_isometric()
            pl.show(auto_close=False)
            pyvista_shown = True
            try:
                pl.screenshot(save_path)
            except Exception:
                pass
            pl.close()
            pyvista_shown = True
        except Exception:
            if pyvista_shown:
                pass

    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    face_vals = rss[face_np].mean(axis=1)
    tri_verts = pts[face_np]
    vmin = float(face_vals.min()) if face_vals.size else 0.0
    vmax = float(face_vals.max()) if face_vals.size else 1.0
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 1e-6, 1e-30)
    if log_scale:
        positive = face_vals[face_vals > 0.0]
        if positive.size:
            vmin = float(np.quantile(positive, 0.01))
            vmin = max(vmin, float(np.min(positive)))
        else:
            vmin = 1e-30
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("rainbow")
    colors = cmap(norm(face_vals))

    lim = float(np.max(np.abs(pts)))
    lim_save = lim * 0.8

    fig_save = plt.figure(figsize=(10.5, 6))
    grid = fig_save.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=-0.02)
    axes_save = [
        fig_save.add_subplot(grid[0, 0], projection="3d"),
        fig_save.add_subplot(grid[0, 1], projection="3d"),
    ]
    cax = fig_save.add_subplot(grid[0, 2])
    for ax, view_label, azim in (
        (axes_save[0], "Front", DEFAULT_SPHERE_AZIM),
        (axes_save[1], "Back", DEFAULT_SPHERE_AZIM + 180.0),
    ):
        collection = Poly3DCollection(
            tri_verts,
            facecolors=colors,
            edgecolor="none",
            linewidth=0.05,
            antialiased=True,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(-lim_save, lim_save)
        ax.set_ylim(-lim_save, lim_save)
        ax.set_zlim(-lim_save, lim_save)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(view_label, pad=8)
        ax.view_init(elev=DEFAULT_SPHERE_ELEV, azim=azim)
    fig_save.suptitle(title, y=0.98)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(face_vals)
    cb_save = fig_save.colorbar(
        mappable,
        cax=cax,
        label="|B_emit| RSS (T)" + (" [log]" if log_scale else ""),
    )
    if not log_scale:
        ticks = np.linspace(vmin, vmax, 6)
        cb_save.set_ticks(ticks)
        cb_save.ax.set_yticklabels([f"{t:.3e}" for t in ticks])
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig_save)

    if show_matplotlib:
        fig_show = plt.figure(figsize=(7, 6))
        ax = fig_show.add_subplot(1, 1, 1, projection="3d")
        collection = Poly3DCollection(
            tri_verts,
            facecolors=colors,
            edgecolor="none",
            linewidth=0.05,
            antialiased=True,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(title, pad=12)
        ax.view_init(elev=DEFAULT_SPHERE_ELEV, azim=DEFAULT_SPHERE_AZIM)
        cb = fig_show.colorbar(
            mappable,
            ax=ax,
            shrink=0.8,
            pad=0.05,
            label="|B_emit| RSS (T)" + (" [log]" if log_scale else ""),
        )
        if not log_scale:
            ticks = np.linspace(vmin, vmax, 6)
            cb.set_ticks(ticks)
            cb.ax.set_yticklabels([f"{t:.3e}" for t in ticks])
        plt.tight_layout()
        plt.show()
