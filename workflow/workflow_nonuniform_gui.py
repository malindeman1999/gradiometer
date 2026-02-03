"""
GUI for running the non-uniform demo pipeline in four stages:
1) Build grid + admittance (single-mode conductivity)
2) Build ambient field
3) Solve currents (self-consistent by default, or first-order)
4) Render overview and gradient plots

Each step saves its state so runs can be resumed later.
"""
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import time
import math
import threading
import shutil

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from workflow.ambient_field.ambient_driver import build_ambient_driver_x
from europa_model.config import GridConfig, ModelConfig
from europa_model.grid import make_grid
from europa_model.transforms import sh_forward, sh_inverse
from europa_model.simulation import Simulation
from europa_model.solvers import _flatten_lm, _unflatten_lm, toroidal_e_from_radial_b, _build_self_field_diag
from europa_model.solver_variants.solver_variant_precomputed import (
    solve_spectral_self_consistent_sim_precomputed,
    _build_mixing_matrix_precomputed_sparse,
)
from europa_model import inductance
from europa_model.gradient_utils import render_gradient_map
from workflow.plotting.render_demo_overview import render_demo_overview
from workflow.plotting.render_phasor_maps import _build_mesh
from gaunt.assemble_gaunt_checkpoints import assemble_in_memory
from workflow.data_objects.phasor_data import PhasorSimulation

STATE_DIR = Path("workflow/artifacts/nonuniform_workflow")
FIG_DIR = Path("workflow/figures/nonuniform")
GAUNT_CACHE = Path("gaunt/data/gaunt_cache_wigxjpf")


def _log(text_widget: tk.Text, msg: str) -> None:
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)
    text_widget.update_idletasks()


def _synthesize_sigma_field(
    positions: torch.Tensor,
    weights: torch.Tensor,
    lmax: int,
    mean: float,
    frac_rms: float,
    mode_l: int,
    mode_m: int,
    log=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a real conductivity field with target RMS and single (l,±m) modes."""
    mode_l = int(max(0, min(mode_l, lmax)))
    mode_m = int(max(0, min(abs(mode_m), mode_l)))
    frac_rms = max(0.0, float(frac_rms))
    delta_coeffs = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    rng = np.random.default_rng()
    phase = rng.uniform(0.0, 2 * math.pi)
    c = math.cos(phase) + 1j * math.sin(phase)
    delta_coeffs[mode_l, lmax + mode_m] = c
    delta_coeffs[mode_l, lmax - mode_m] = ((-1) ** mode_m) * np.conj(c)
    delta = sh_inverse(delta_coeffs, positions, weights)
    imag_max = float(delta.imag.abs().max().item())
    real_max = float(delta.real.abs().max().item())
    tol = max(1e-12, 1e-9 * max(real_max, 1e-30))
    if imag_max > tol:
        raise RuntimeError(
            f"Conductivity synthesis produced significant imaginary values: "
            f"max|imag|={imag_max:.3e} (tol={tol:.3e})."
        )
    delta = delta.real
    delta = delta - delta.mean()
    current_rms = float(torch.sqrt((delta ** 2).mean()).item())
    target_rms = mean * frac_rms
    if current_rms > 0.0 and target_rms > 0.0:
        scale = target_rms / current_rms
        delta_coeffs = delta_coeffs * scale
    else:
        delta_coeffs = torch.zeros_like(delta_coeffs)

    sigma_coeffs = delta_coeffs.clone()
    # SciPy-normalized Y_00 = 1/(2*sqrt(pi)); to realize a constant mean "mean",
    # the l=0,m=0 coefficient must be mean / Y_00 = mean * 2*sqrt(pi).
    c00 = mean * (2.0 * math.sqrt(math.pi))
    sigma_coeffs[0, lmax] = c00
    sigma = sh_inverse(sigma_coeffs, positions, weights)
    imag_max = float(sigma.imag.abs().max().item())
    real_max = float(sigma.real.abs().max().item())
    tol = max(1e-12, 1e-9 * max(real_max, 1e-30))
    if imag_max > tol:
        raise RuntimeError(
            f"Conductivity synthesis produced significant imaginary values: "
            f"max|imag|={imag_max:.3e} (tol={tol:.3e})."
        )
    sigma = sigma.real
    if float(sigma.min().item()) <= 0.0:
        if log is not None:
            log(
                "Warning: conductivity synthesis produced non-positive values on the grid. "
                "Plots may show unphysical regions."
            )
    return sigma, sigma_coeffs


def _subdivisions_from_faces(target_faces: int) -> int:
    """Pick the subdivision level whose face count best matches the target."""
    target_faces = max(1, int(target_faces))
    best_s = 0
    best_err = float("inf")
    for s in range(0, 9):
        faces = 20 * (4 ** s)
        err = abs(faces - target_faces)
        if err < best_err:
            best_err = err
            best_s = s
    return best_s


def _nside_from_subdivisions(subdiv: int) -> int:
    """Choose an nside that maps to the same subdivision in make_grid."""
    return max(1, 10 * (4 ** max(0, subdiv)))


def _faces_for_subdiv(subdiv: int) -> int:
    return 20 * (4 ** max(0, subdiv))


def _lmax_for_target_faces(faces: int) -> int:
    """Choose lmax so (lmax+1)^2 is as close as possible to target faces."""
    faces = max(1, int(faces))
    cand = max(1, int(round(math.sqrt(faces)) - 1))
    # check neighbors to minimize absolute difference
    best = cand
    best_err = abs((cand + 1) ** 2 - faces)
    for delta in (-1, 1):
        alt = max(1, cand + delta)
        err = abs((alt + 1) ** 2 - faces)
        if err < best_err:
            best_err = err
            best = alt
    return best + 1


def _mean_face_center_spacing_km(subdivisions: int, radius_m: float) -> float:
    """Approximate mean nearest-neighbor spacing between face centers."""
    _, _, centers = _build_mesh(radius_m, subdivisions=subdivisions, stride=1)
    centers = centers.to(dtype=torch.float64)
    count = int(centers.shape[0])
    if count < 2:
        return 0.0
    sample_cap = 2000
    if count > sample_cap:
        idx = torch.randperm(count)[:sample_cap]
        centers = centers[idx]
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(float("inf"))
    nn = dists.min(dim=1).values
    return float(nn.mean().item() / 1000.0)


def _save_state(name: str, payload) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / name
    torch.save(payload, path)
    return path


def _load_state(name: str):
    path = STATE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing state file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _complex_sheet_admittance(
    sigma_s: torch.Tensor,
    omega: float,
    radius_m: float,
    inductance_scale: float = 1.0,
) -> torch.Tensor:
    """Compute complex sheet admittance from thin-shell impedance model."""
    sigma_s = sigma_s.to(torch.float64)
    X_s = float(inductance_scale) * omega * float(inductance.MU0) * radius_m / 2.0
    R_s = torch.where(sigma_s > 0, 1.0 / sigma_s, torch.zeros_like(sigma_s))
    Z = R_s + 1j * X_s
    Y = torch.where(sigma_s > 0, 1.0 / Z, torch.zeros_like(Z))
    return Y.to(torch.complex128)


def step1_build_grid_admittance(
    subdiv: int,
    lmax: int,
    mean_cond: float,
    frac_rms: float,
    mode_l: int,
    mode_m: int,
    inductance_scale: float,
    log,
) -> Path:
    subdiv = max(0, int(subdiv))
    actual_faces = _faces_for_subdiv(subdiv)
    nside = _nside_from_subdivisions(subdiv)
    grid_cfg = GridConfig(nside=nside, lmax=lmax, radius_m=1.56e6, device="cpu")
    grid = make_grid(grid_cfg)
    mean_val = max(0.0, float(mean_cond))
    frac_rms = max(0.0, float(frac_rms))
    mode_l = int(mode_l)
    mode_m = int(mode_m)
    cond_real, sigma_coeffs = _synthesize_sigma_field(
        grid.positions.to(torch.float64),
        grid.areas.to(torch.float64),
        grid_cfg.lmax,
        mean_val,
        frac_rms,
        mode_l,
        mode_m,
        log,
    )
    realized_mean = float(cond_real.mean().item())
    realized_rms = float(torch.sqrt(((cond_real - realized_mean) ** 2).mean()).item())
    target_rms = mean_val * frac_rms
    min_val = float(cond_real.min().item())
    max_val = float(cond_real.max().item())

    def _relative_error(actual: float, target: float) -> float:
        if abs(target) > 1e-12:
            return abs(actual - target) / abs(target)
        return 0.0 if abs(actual) <= 1e-12 else float("inf")

    mean_rel_err = _relative_error(realized_mean, mean_val)
    rms_rel_err = _relative_error(realized_rms, target_rms)
    if mean_rel_err > 0.10 or rms_rel_err > 0.10:
        raise RuntimeError(
            "Conductivity synthesis missed requested statistics by more than 10%: "
            f"mean target={mean_val:.6e}, realized={realized_mean:.6e}, rel_err={mean_rel_err:.2%}; "
            f"rms target={target_rms:.6e}, realized={realized_rms:.6e}, rel_err={rms_rel_err:.2%}."
        )

    log(
        f"Sigma_s stats: mean={realized_mean:.3e}, rms={realized_rms:.3e} "
        f"(frac={realized_rms/mean_val if mean_val > 0 else 0.0:.2%}), "
        f"min={min_val:.3e}, max={max_val:.3e}"
    )
    omega = 2.0 * math.pi / (9.925 * 3600.0)
    cond = _complex_sheet_admittance(cond_real, omega, grid_cfg.radius_m, inductance_scale=inductance_scale)
    Y_s = sh_forward(cond, grid.positions.to(torch.float64), lmax=grid_cfg.lmax, weights=grid.areas.to(torch.float64))
    sigma_proj = sh_forward(
        cond_real.to(torch.float64),
        grid.positions.to(torch.float64),
        lmax=grid_cfg.lmax,
        weights=grid.areas.to(torch.float64),
    )
    state = {
        "grid_cfg": grid_cfg,
        "positions": grid.positions,
        "normals": grid.normals,
        "areas": grid.areas,
        "neighbors": grid.neighbors,
        "admittance_spectral": Y_s,
        "admittance_grid": cond,
        "sigma_spectral": sigma_proj,
        "sigma_spectral_target": sigma_coeffs,
        "sigma_grid": cond_real,
        "sigma_mean": mean_val,
        "sigma_frac_rms": frac_rms,
        "sigma_mode_l": int(mode_l),
        "sigma_mode_m": int(mode_m),
        "inductance_scale": float(inductance_scale),
        "subdivisions": subdiv,
    }
    path = _save_state("grid_admittance.pt", state)
    log(f"Step 1 complete (subdivisions={subdiv}, faces={grid.positions.shape[0]}). Saved grid+admittance to {path}")
    return path, subdiv, int(grid.positions.shape[0]), actual_faces


def step1b_plot_roundtrip(log) -> None:
    state = _load_state("grid_admittance.pt")
    positions = state["positions"].to(torch.float64)
    weights = state["areas"].to(torch.float64)
    coeffs = state["admittance_spectral"]
    recon = sh_inverse(coeffs, positions, weights)
    recon = recon.reshape(-1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    idx = np.arange(recon.size)
    for ax, a, label in (
        (axes[0], recon.real, "Real"),
        (axes[1], recon.imag, "Imag"),
    ):
        ax.scatter(idx, a, s=8, alpha=0.7, label="Spectral reconstruction")
        ax.set_title(f"{label} part: spectral reconstruction")
        ax.set_xlabel("Grid point index")
        ax.set_ylabel("Admittance (S)")
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    plt.show()
    sigma_grid = state.get("sigma_grid")
    if sigma_grid is not None:
        sigma_vals = sigma_grid.to(torch.float64).reshape(-1).cpu().numpy()
        _plot_admittance_and_conductivity_spheres(
            sigma_vals,
            recon,
            positions=positions,
            subdivisions=int(state["subdivisions"]),
            radius=float(state["grid_cfg"].radius_m),
        )
    log("Step 1b complete. Displayed spectral reconstruction plots.")


def step1b_plot_admittance_power(log) -> None:
    state = _load_state("grid_admittance.pt")
    coeffs = state.get("admittance_spectral")
    sigma_coeffs = state.get("sigma_spectral")
    if coeffs is None or sigma_coeffs is None:
        raise RuntimeError("Missing admittance_spectral or sigma_spectral. Run Step 1 before plotting magnitudes.")
    mode_l = state.get("sigma_mode_l", None)
    mode_m = state.get("sigma_mode_m", None)
    frac_rms = state.get("sigma_frac_rms", None)
    mode_l_str = f"{int(mode_l)}" if mode_l is not None else "?"
    mode_m_str = f"{int(mode_m)}" if mode_m is not None else "?"
    frac_rms_str = f"{float(frac_rms):.2%}" if frac_rms is not None else "?"
    title_suffix = f"(l={mode_l_str}, m=±{mode_m_str}, frac RMS {frac_rms_str})"

    l_b, m_b, mag = _flatten_harmonics(coeffs.to(torch.complex128))
    _, _, mag_sigma = _flatten_harmonics(sigma_coeffs.to(torch.complex128))
    active = (mag > 0.0) | (mag_sigma > 0.0)
    active_ls = l_b[active]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    keep = l_b <= l_cut
    x = np.arange(int(np.sum(keep)))
    tick_idx = np.where(m_b[keep] == 0)[0]
    tick_labels = [f"({l},0)" for l in l_b[keep][tick_idx]]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    sigma_vals = mag_sigma[keep]
    sigma_vals_plot = np.where(sigma_vals > 0.0, sigma_vals, np.nan)
    axes[0].bar(x, sigma_vals_plot, color="#5c9bd5")
    axes[0].set_ylabel("|σ_s| (S)")
    axes[0].set_title(f"Conductivity mode magnitude (S) {title_suffix}")
    axes[0].grid(True, which="both", alpha=0.3)

    y_vals = mag[keep]
    y_vals_plot = np.where(y_vals > 0.0, y_vals, np.nan)
    axes[1].bar(x, y_vals_plot, color="#ff9c43")
    axes[1].set_xlabel("(l,m) ordering; ticks at m=0")
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels(tick_labels, rotation=90)
    axes[1].set_ylabel("|Y_s| (S)")
    axes[1].set_title(f"Admittance mode magnitude (S) {title_suffix}")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.show()

    lmax = int(l_b.max()) if l_b.size else 0
    l_vals = np.arange(lmax + 1)
    rss_sigma = np.zeros(lmax + 1, dtype=np.float64)
    rss_y = np.zeros(lmax + 1, dtype=np.float64)
    for l in range(lmax + 1):
        mask = l_b == l
        rss_sigma[l] = float(np.sqrt(np.sum(mag_sigma[mask] ** 2)))
        rss_y[l] = float(np.sqrt(np.sum(mag[mask] ** 2)))

    fig2, axes2 = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    rss_sigma_plot = np.where(rss_sigma > 0.0, rss_sigma, np.nan)
    axes2[0].plot(l_vals, rss_sigma_plot, marker="o", linewidth=1.2, color="#5c9bd5")
    axes2[0].set_ylabel("RSS |σ_s| (S)")
    axes2[0].set_title(f"Conductivity by degree l (RSS, S) {title_suffix}")
    axes2[0].grid(True, which="both", alpha=0.3)

    rss_y_plot = np.where(rss_y > 0.0, rss_y, np.nan)
    axes2[1].plot(l_vals, rss_y_plot, marker="o", linewidth=1.2, color="#ff9c43")
    axes2[1].set_xlabel("Spherical harmonic degree l")
    axes2[1].set_ylabel("RSS |Y_s| (S)")
    axes2[1].set_title(f"Admittance by degree l (RSS, S) {title_suffix}")
    axes2[1].grid(True, which="both", alpha=0.3)

    fig2.tight_layout()
    plt.show()
    log("Step 1b complete. Plotted conductivity/admittance magnitudes and per-l RSS.")


def _plot_sphere_map_single(
    values: np.ndarray,
    label: str,
    positions: torch.Tensor,
    subdivisions: int,
    radius: float,
    elev: float = 20.0,
    azim: float = 30.0,
) -> None:
    vertices, faces, centers = _build_mesh(radius, subdivisions=subdivisions, stride=1)
    face_count = faces.shape[0]
    node_count = int(positions.shape[0])
    if values.size != node_count:
        raise RuntimeError(
            "Sphere map values do not match grid positions "
            f"(nodes={node_count}, values={values.size})."
        )

    tri_verts = vertices[faces].cpu().numpy()
    vals = values.reshape(-1)
    pos = positions.detach().cpu().to(torch.float64)
    cen = centers.detach().cpu().to(torch.float64)
    nearest = torch.cdist(cen, pos).argmin(dim=1).cpu().numpy()
    if len(nearest) != face_count:
        raise RuntimeError("Failed to map grid values to mesh faces.")
    vals = vals[nearest]

    real_vmax = float(max(np.max(np.abs(vals.real)), 1e-12))
    imag_vmax = float(max(np.max(np.abs(vals.imag)), 1e-12))
    norm_real = mcolors.Normalize(vmin=-real_vmax, vmax=real_vmax)
    norm_imag = mcolors.Normalize(vmin=-imag_vmax, vmax=imag_vmax)
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.05, hspace=0.2)
    ax_real = fig.add_subplot(gs[0, 0], projection="3d")
    _add_sphere_subplot(
        ax_real,
        vals.real,
        norm_real,
        cmap,
        tri_verts,
        radius,
        f"{label} real(Y_s)",
        elev=elev,
        azim=azim,
    )
    ax_imag = fig.add_subplot(gs[0, 1], projection="3d")
    _add_sphere_subplot(
        ax_imag,
        vals.imag,
        norm_imag,
        cmap,
        tri_verts,
        radius,
        f"{label} imag(Y_s)",
        elev=elev,
        azim=azim,
    )

    plt.tight_layout()
    plt.show()


def _plot_admittance_and_conductivity_spheres(
    sigma_real: np.ndarray,
    admittance: np.ndarray,
    positions: torch.Tensor,
    subdivisions: int,
    radius: float,
    elev: float = 20.0,
    azim: float = 30.0,
) -> None:
    vertices, faces, centers = _build_mesh(radius, subdivisions=subdivisions, stride=1)
    face_count = faces.shape[0]
    node_count = int(positions.shape[0])
    if sigma_real.size != node_count or admittance.size != node_count:
        raise RuntimeError(
            "Sphere map values do not match grid positions "
            f"(nodes={node_count}, sigma={sigma_real.size}, admittance={admittance.size})."
        )

    tri_verts = vertices[faces].cpu().numpy()
    pos = positions.detach().cpu().to(torch.float64)
    cen = centers.detach().cpu().to(torch.float64)
    nearest = torch.cdist(cen, pos).argmin(dim=1).cpu().numpy()
    if len(nearest) != face_count:
        raise RuntimeError("Failed to map grid values to mesh faces.")

    sigma_vals = sigma_real.reshape(-1)[nearest]
    adm_vals = admittance.reshape(-1)[nearest]

    sigma_min = float(np.min(sigma_vals))
    sigma_max = float(np.max(sigma_vals))
    real_min = float(np.min(adm_vals.real))
    real_max = float(np.max(adm_vals.real))
    imag_min = float(np.min(adm_vals.imag))
    imag_max = float(np.max(adm_vals.imag))
    if sigma_max <= sigma_min:
        sigma_max = sigma_min + 1e-12
    if real_max <= real_min:
        real_max = real_min + 1e-12
    if imag_max <= imag_min:
        imag_max = imag_min + 1e-12
    norm_sigma = mcolors.Normalize(vmin=sigma_min, vmax=sigma_max)
    norm_real = mcolors.Normalize(vmin=real_min, vmax=real_max)
    norm_imag = mcolors.Normalize(vmin=imag_min, vmax=imag_max)
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.05)

    ax_sigma = fig.add_subplot(gs[0, 0], projection="3d")
    _add_sphere_subplot(
        ax_sigma,
        sigma_vals,
        norm_sigma,
        cmap,
        tri_verts,
        radius,
        "Conductivity real(σ_s)",
        elev=elev,
        azim=azim,
    )
    ax_real = fig.add_subplot(gs[0, 1], projection="3d")
    _add_sphere_subplot(
        ax_real,
        adm_vals.real,
        norm_real,
        cmap,
        tri_verts,
        radius,
        "Admittance real(Y_s)",
        elev=elev,
        azim=azim,
    )
    ax_imag = fig.add_subplot(gs[0, 2], projection="3d")
    _add_sphere_subplot(
        ax_imag,
        adm_vals.imag,
        norm_imag,
        cmap,
        tri_verts,
        radius,
        "Admittance imag(Y_s)",
        elev=elev,
        azim=azim,
    )

    plt.tight_layout()
    plt.show()


def _add_sphere_subplot(
    ax,
    face_vals: np.ndarray,
    norm: mcolors.Normalize,
    cmap,
    tri_verts: np.ndarray,
    radius: float,
    title: str,
    elev: float,
    azim: float,
) -> None:
    collection = Poly3DCollection(
        tri_verts,
        facecolors=cmap(norm(face_vals)),
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
    ax.set_title(title, pad=8)
    ax.view_init(elev=elev, azim=azim)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(face_vals)
    ax.figure.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02)


def step2_build_ambient(log) -> Path:
    state1 = _load_state("grid_admittance.pt")
    grid_cfg: GridConfig = state1["grid_cfg"]
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver_x(grid_cfg)
    state1.update(
        {
            "ambient_cfg": ambient_cfg,
            "B_radial_spec": B_radial_spec,
            "period_sec": period_sec,
        }
    )
    path = _save_state("ambient.pt", state1)
    log(f"Step 2 complete. Saved ambient + B_radial to {path}")
    return path


def _build_phasor_base(state) -> PhasorSimulation:
    grid_cfg: GridConfig = state["grid_cfg"]
    ambient_cfg = state["ambient_cfg"]
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    grid_ns = Simulation(model).grid  # type: ignore[attr-defined]
    # Reuse prebuilt tensors instead of rebuilding the grid
    grid_ns = type("GridNS", (), {})()
    grid_ns.positions = state["positions"]
    grid_ns.normals = state["normals"]
    grid_ns.areas = state["areas"]
    grid_ns.neighbors = state["neighbors"]
    return PhasorSimulation.from_model_and_grid(
        model=model,
        grid=grid_ns,
        solver_variant="",
        admittance_spectral=state["admittance_spectral"],
        B_radial=state["B_radial_spec"],
        period_sec=state["period_sec"],
    )


def step3_solve_currents(first_order_only: bool, log) -> Path:
    state = _load_state("ambient.pt")
    grid_cfg: GridConfig = state["grid_cfg"]
    base = _build_phasor_base(state)

    log(f"Assembling Gaunt tensor from {GAUNT_CACHE} (lmax_limit={grid_cfg.lmax})...")
    G_sparse, gaunt_meta = assemble_in_memory(
        cache_dir=GAUNT_CACHE,
        lmax_limit=grid_cfg.lmax,
        verbose=True,
        plot=False,
    )
    complete_L = gaunt_meta.get("complete_L")
    log(f"Gaunt tensor nnz={G_sparse._nnz()}, complete_L={complete_L}")
    if complete_L is None or int(complete_L) < grid_cfg.lmax:
        raise RuntimeError(
            f"Gaunt cache incomplete: complete_L={complete_L}, required lmax={grid_cfg.lmax}. "
            "Rebuild the Gaunt cache to at least the requested lmax or lower lmax."
        )

    log("Building sparse mixing matrix...")
    mixing_matrix = _build_mixing_matrix_precomputed_sparse(
        grid_cfg.lmax, base.omega, base.radius_m, base.admittance_spectral, G_sparse
    )

    def _log_matrix_diagnostics(name: str, A: torch.Tensor) -> None:
        with torch.no_grad():
            A = A.to(torch.complex128)
            max_abs = float(A.abs().max().item())
            log(f"{name}: shape={tuple(A.shape)}, max|A|={max_abs:.3e}")
            try:
                s = torch.linalg.svdvals(A)
                s_max = float(s.max().item())
                s_min = float(s.min().item())
                cond = float(s_max / s_min) if s_min != 0.0 else float("inf")
                log(f"{name}: svd s_max={s_max:.3e}, s_min={s_min:.3e}, cond={cond:.3e}")
            except Exception as exc:  # noqa: BLE001
                log(f"{name}: svdvals failed: {exc}")

    def _matrix_condition(A: torch.Tensor) -> float:
        with torch.no_grad():
            s = torch.linalg.svdvals(A)
            s_max = float(s.max().item())
            s_min = float(s.min().item())
            return float(s_max / s_min) if s_min != 0.0 else float("inf")

    def _log_vec_diagnostics(name: str, v: torch.Tensor) -> None:
        with torch.no_grad():
            max_abs = float(v.abs().max().item())
            any_nan = bool(torch.isnan(v).any().item())
            any_inf = bool(torch.isinf(v).any().item())
            log(f"{name}: max|v|={max_abs:.3e}, has_nan={any_nan}, has_inf={any_inf}")

    def _log_kl_power(label: str, K_tor: torch.Tensor) -> None:
        with torch.no_grad():
            K = K_tor.to(torch.complex128)
            lmax = K.shape[-2] - 1
            power = []
            for l in range(lmax + 1):
                row = K[l]
                power.append(float((row.abs() ** 2).sum().item()))
            total = sum(power) if power else 0.0
            if total <= 0.0:
                log(f"{label}: total power=0 (no currents).")
                return
            top = sorted(range(len(power)), key=lambda i: power[i], reverse=True)[:6]
            top_str = ", ".join([f"l={i}:{power[i]/total:.2%}" for i in top])
            log(f"{label}: total power={total:.3e}, top l fractions: {top_str}")

    sim_out = PhasorSimulation.from_serializable(base.to_serializable())
    if first_order_only:
        log("Solving first-order currents (no feedback)...")
        sim_out.E_toroidal = toroidal_e_from_radial_b(sim_out.B_radial, sim_out.omega, sim_out.radius_m)
        b_flat = _flatten_lm(sim_out.B_radial.to(torch.complex128))
        k_flat = mixing_matrix @ b_flat
        _log_vec_diagnostics("b_ext_flat", b_flat)
        _log_vec_diagnostics("k_flat (first_order)", k_flat)
        sim_out.K_toroidal = _unflatten_lm(k_flat, grid_cfg.lmax)
        # Toroidal l=0 is unphysical; explicitly zero to avoid numerical leakage.
        sim_out.K_toroidal[0, :] = 0.0
        _log_kl_power("K_tor (first_order)", sim_out.K_toroidal)
        sim_out.K_poloidal = torch.zeros_like(sim_out.K_toroidal)
        sim_out.B_tor_emit, sim_out.B_pol_emit, sim_out.B_rad_emit = inductance.spectral_b_from_surface_currents(
            sim_out.K_toroidal, sim_out.K_poloidal, radius=sim_out.radius_m
        )
        src_energy = float((sim_out.B_radial.abs() ** 2).sum().item())
        resp_energy = float((sim_out.B_rad_emit.abs() ** 2).sum().item())
        if resp_energy > src_energy:
            log(
                "Warning: first-order response energy exceeds source energy "
                f"(resp={resp_energy:.3e} > src={src_energy:.3e})."
            )
        sim_out.solver_variant = "spectral_first_order_precomputed_gaunt_sparse"
        label = "first_order"
    else:
        log("Solving self-consistent system (matrix inversion)...")
        sim_out.E_toroidal = toroidal_e_from_radial_b(sim_out.B_radial, sim_out.omega, sim_out.radius_m)
        b_ext_flat = _flatten_lm(sim_out.B_radial.to(torch.complex128))
        S_diag = _build_self_field_diag(grid_cfg.lmax, sim_out.grid_positions.device, torch.complex128)
        I = torch.eye(mixing_matrix.shape[0], device=mixing_matrix.device, dtype=torch.complex128)
        A = I - torch.diag(S_diag) @ mixing_matrix
        _log_vec_diagnostics("b_ext_flat", b_ext_flat)
        _log_matrix_diagnostics("M (mixing_matrix)", mixing_matrix)
        _log_matrix_diagnostics("A (I - S*M)", A)
        try:
            cond = _matrix_condition(A)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Condition check failed: {exc}") from exc
        if not np.isfinite(cond) or cond > 1e8:
            raise RuntimeError(f"Ill-conditioned A (cond={cond:.3e}); aborting self-consistent solve.")
        try:
            b_tot = torch.linalg.solve(A, b_ext_flat)
        except Exception as exc:  # noqa: BLE001
            log(f"Linear solve failed: {exc}")
            raise
        k_flat = mixing_matrix @ b_tot
        _log_vec_diagnostics("b_tot (self_consistent)", b_tot)
        _log_vec_diagnostics("k_flat (self_consistent)", k_flat)
        sim_out.K_toroidal = _unflatten_lm(k_flat, grid_cfg.lmax)
        # Toroidal l=0 is unphysical; explicitly zero to avoid numerical leakage.
        sim_out.K_toroidal[0, :] = 0.0
        _log_kl_power("K_tor (self_consistent)", sim_out.K_toroidal)
        sim_out.K_poloidal = torch.zeros_like(sim_out.K_toroidal)
        sim_out.B_tor_emit, sim_out.B_pol_emit, sim_out.B_rad_emit = inductance.spectral_b_from_surface_currents(
            sim_out.K_toroidal, sim_out.K_poloidal, radius=sim_out.radius_m
        )
        sim_out.solver_variant = "spectral_self_consistent_precomputed_gaunt_sparse"
        sim_out.solver_variant = "spectral_self_consistent_precomputed_gaunt_sparse"
        label = "self_consistent"
        log("Self-consistent solve complete.")

    payload = {
        "label": label,
        "phasor_sim": sim_out,
    }
    path = _save_state(f"solution_{label}.pt", payload)
    _save_state("solution_latest.pt", payload)
    log(f"Step 3 complete. Saved solution to {path}")
    return path


def _calc_subdivisions(lmax: int) -> int:
    desired_faces = max(20, (lmax + 1) ** 2)
    return max(1, math.ceil(math.log(desired_faces / 20, 4)))


def _load_solution(label: str):
    return _load_state(f"solution_{label}.pt")


def step4_render_overview(label: str, log) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    subdivisions = int(grid_state["subdivisions"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"nonuniform_{label}_overview.png"
    log(f"Step 4 overview: label={label}, lmax={sim_out.lmax}, subdivisions={subdivisions}")
    log("Step 4 overview: assembling input state for renderer...")
    t0 = time.perf_counter()
    render_demo_overview(
        data_path=_save_state("overview_input.pt", payload),  # save tmp input for renderer
        subdivisions=subdivisions,
        save_path=str(out_path),
        show=False,
        grid_state_path=str(STATE_DIR / "grid_admittance.pt"),
    )
    dt = time.perf_counter() - t0
    log(f"Step 4 overview: rendered in {dt:.1f}s -> {out_path}")
    return out_path


def step4_render_gradient(label: str, altitude_m: float, log) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    subdivisions = int(grid_state["subdivisions"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    title = f"RSS |grad_B_emit| at alt={altitude_m/1000:.0f} km"
    save_path = FIG_DIR / f"nonuniform_grad_{int(altitude_m):d}m_{label}.png"
    render_gradient_map(sim_out, altitude_m=altitude_m, subdivisions=subdivisions, save_path=str(save_path), title=title)
    log(f"Rendered gradient map to {save_path}")
    return save_path


def _flatten_harmonics(coeffs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return l, m, |coeff| arrays in canonical (l,m) order."""
    lmax = coeffs.shape[-2] - 1
    ls, ms, mags = [], [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ls.append(l)
            ms.append(m)
            mags.append(torch.abs(coeffs[l, lmax + m]).item())
    return np.array(ls), np.array(ms), np.array(mags)


def step4_plot_harmonics(label: str, log) -> None:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    if sim_out.B_radial is None or sim_out.B_rad_emit is None:
        raise RuntimeError("Missing B_radial or B_rad_emit; run the solve before plotting harmonics.")

    l_b, m_b, mag_b = _flatten_harmonics(sim_out.B_radial)
    _, _, mag_emit = _flatten_harmonics(sim_out.B_rad_emit)
    peak = float(max(np.max(mag_b), np.max(mag_emit), 1e-30))
    eps = peak * 1e-9
    active = (mag_b > eps) | (mag_emit > eps)
    active_ls = l_b[active]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    keep = l_b <= l_cut
    x = np.arange(int(np.sum(keep)))
    tick_idx = np.where(m_b[keep] == 0)[0]
    tick_labels = [f"({l},0)" for l in l_b[keep][tick_idx]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.42
    ax.bar(x - width / 2, np.maximum(mag_b[keep], eps), width=width, label="ambient |B_rad|")
    ax.bar(x + width / 2, np.maximum(mag_emit[keep], eps), width=width, label="emitted |B_rad_emit|")
    ax.set_yscale("log")
    ax.set_xlabel("(l,m) ordering; ticks at m=0")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_ylabel("RSS magnitude")
    ax.set_title(f"Harmonics magnitude (ambient vs emitted) [{label}]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
    log(f"Plotted harmonics magnitude for {label}.")


def _clear_outputs(log) -> None:
    for path in (FIG_DIR, STATE_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    log(f"Cleared outputs in {FIG_DIR} and {STATE_DIR}.")


def step6_iterative_solve(order: int, log) -> Path:
    state = _load_state("ambient.pt")
    grid_cfg: GridConfig = state["grid_cfg"]
    base = _build_phasor_base(state)

    log(f"Assembling Gaunt tensor from {GAUNT_CACHE} (lmax_limit={grid_cfg.lmax})...")
    G_sparse, gaunt_meta = assemble_in_memory(
        cache_dir=GAUNT_CACHE,
        lmax_limit=grid_cfg.lmax,
        verbose=True,
        plot=False,
    )
    complete_L = gaunt_meta.get("complete_L")
    log(f"Gaunt tensor nnz={G_sparse._nnz()}, complete_L={complete_L}")
    if complete_L is None or int(complete_L) < grid_cfg.lmax:
        raise RuntimeError(
            f"Gaunt cache incomplete: complete_L={complete_L}, required lmax={grid_cfg.lmax}. "
            "Rebuild the Gaunt cache to at least the requested lmax or lower lmax."
        )

    log("Building sparse mixing matrix...")
    mixing_matrix = _build_mixing_matrix_precomputed_sparse(
        grid_cfg.lmax, base.omega, base.radius_m, base.admittance_spectral, G_sparse
    )

    sim_out = PhasorSimulation.from_serializable(base.to_serializable())
    sim_out.E_toroidal = toroidal_e_from_radial_b(sim_out.B_radial, sim_out.omega, sim_out.radius_m)

    max_order = max(1, int(order))
    log(f"Iterative solve: max_order={max_order}")
    b_ext_flat = _flatten_lm(sim_out.B_radial.to(torch.complex128))
    S_diag = _build_self_field_diag(grid_cfg.lmax, sim_out.grid_positions.device, torch.complex128)
    SM = torch.diag(S_diag) @ mixing_matrix

    b_tot = b_ext_flat.clone()
    term = b_ext_flat.clone()
    prev_norm = float(term.abs().max().item())
    log(f"Iterative order 0: max|term|={prev_norm:.3e}")
    for n in range(1, max_order + 1):
        term = SM @ term
        term_norm = float(term.abs().max().item())
        log(f"Iterative order {n}: max|term|={term_norm:.3e}")
        if term_norm > prev_norm:
            log(
                "Warning: iterative series not converging at this order "
                f"(order {n} term grew {term_norm:.3e} > {prev_norm:.3e})."
            )
        b_tot = b_tot + term
        prev_norm = term_norm

    k_flat = mixing_matrix @ b_tot
    sim_out.K_toroidal = _unflatten_lm(k_flat, grid_cfg.lmax)
    # Toroidal l=0 is unphysical; explicitly zero to avoid numerical leakage.
    sim_out.K_toroidal[0, :] = 0.0
    sim_out.K_poloidal = torch.zeros_like(sim_out.K_toroidal)
    sim_out.B_tor_emit, sim_out.B_pol_emit, sim_out.B_rad_emit = inductance.spectral_b_from_surface_currents(
        sim_out.K_toroidal, sim_out.K_poloidal, radius=sim_out.radius_m
    )
    sim_out.solver_variant = "spectral_iterative_series"
    label = "iterative"

    payload = {
        "label": label,
        "phasor_sim": sim_out,
    }
    path = _save_state("solution_iterative.pt", payload)
    log(f"Step 6 complete. Saved iterative solution to {path}")
    return path


def main():
    root = tk.Tk()
    root.title("Non-uniform Demo Workflow")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Log output (placed early so lambdas can close over it)
    log_widget = tk.Text(frm, height=18, width=45)
    log_widget.grid(row=11, column=0, columnspan=10, pady=8, sticky="nsew")

    def run_step(button: tk.Button, task, on_success=None):
        """Run a task in a thread, coloring the button yellow while running, green on success, red on error."""
        def worker():
            try:
                root.after(0, lambda: button.config(state=tk.DISABLED, bg="yellow"))
                result = task()
                if on_success is not None:
                    root.after(0, lambda: on_success(result))
                root.after(0, lambda: button.config(state=tk.NORMAL, bg="pale green"))
            except Exception as exc:  # noqa: BLE001
                root.after(0, lambda: button.config(state=tk.NORMAL, bg="tomato"))
                _log(log_widget, f"Error: {exc}")
            finally:
                root.after(0, _update_button_states)
        threading.Thread(target=worker, daemon=True).start()

    def run_step_ui(button: tk.Button, task, on_success=None):
        """Run a task on the main UI thread (needed for matplotlib/Tk)."""
        try:
            button.config(state=tk.DISABLED, bg="yellow")
            result = task()
            if on_success is not None:
                on_success(result)
            button.config(state=tk.NORMAL, bg="pale green")
        except Exception as exc:  # noqa: BLE001
            button.config(state=tk.NORMAL, bg="tomato")
            _log(log_widget, f"Error: {exc}")
        finally:
            _update_button_states()

    def _solution_exists(label: str) -> bool:
        return (STATE_DIR / f"solution_{label}.pt").exists()

    def _grid_exists() -> bool:
        return (STATE_DIR / "grid_admittance.pt").exists()

    def _ambient_exists() -> bool:
        return (STATE_DIR / "ambient.pt").exists()

    state_files = {
        "grid_admittance": "grid_admittance.pt",
        "ambient": "ambient.pt",
        "solution_first_order": "solution_first_order.pt",
        "solution_self_consistent": "solution_self_consistent.pt",
        "solution_iterative": "solution_iterative.pt",
        "solution_latest": "solution_latest.pt",
        "overview_input": "overview_input.pt",
    }

    def _standard_state_path(state_key: str) -> Path:
        if state_key not in state_files:
            raise KeyError(f"Unknown state key: {state_key}")
        return STATE_DIR / state_files[state_key]

    def _load_state_from_path(state_key: str, source_path: str, log) -> Path:
        src = Path(source_path).expanduser()
        dst = _standard_state_path(state_key)
        if not src.exists():
            raise FileNotFoundError(f"Source state file not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            log(f"Step 0 load: {state_key} already at standard path {dst}")
            return dst
        shutil.copy2(src, dst)
        log(f"Step 0 load: copied {src} -> {dst}")
        return dst

    def _save_state_to_path(state_key: str, target_path: str, log) -> Path:
        src = _standard_state_path(state_key)
        dst = Path(target_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Standard state file missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            log(f"Step 0 save: {state_key} stays at standard path {src}")
            return dst
        shutil.copy2(src, dst)
        log(f"Step 0 save: copied {src} -> {dst}")
        return dst

    def _set_button_state(btn: tk.Button, enabled: bool, completed: bool = False) -> None:
        if enabled:
            btn.config(state=tk.NORMAL, bg="pale green" if completed else "SystemButtonFace")
        else:
            btn.config(state=tk.DISABLED, bg="light gray")

    def _update_lmax_button_color() -> None:
        try:
            faces = _faces_for_subdiv(int(subdiv_var.get()))
            best = _lmax_for_target_faces(faces)
            current = int(lmax_var.get())
            btn_l_from_faces.config(bg="pale green" if current == best else "SystemButtonFace")
        except Exception:
            btn_l_from_faces.config(bg="SystemButtonFace")

    def _update_button_states() -> None:
        grid_ok = _grid_exists()
        ambient_ok = _ambient_exists()
        first_ok = _solution_exists("first_order")
        self_ok = _solution_exists("self_consistent")
        iter_ok = _solution_exists("iterative")
        _set_button_state(btn_step1, True, completed=grid_ok)
        _set_button_state(btn_step1b, grid_ok, completed=grid_ok)
        _set_button_state(btn_step1b_power, grid_ok, completed=grid_ok)
        _set_button_state(btn_step2, grid_ok, completed=ambient_ok)
        _set_button_state(btn_step4_first, ambient_ok, completed=first_ok)
        _set_button_state(btn_step5_self, ambient_ok, completed=self_ok)
        _set_button_state(btn_step6_iter, ambient_ok, completed=iter_ok)
        _set_button_state(btn_overview_first, first_ok)
        _set_button_state(btn_grad0_first, first_ok)
        _set_button_state(btn_grad100_first, first_ok)
        _set_button_state(btn_harm_first, first_ok)
        _set_button_state(btn_overview_self, self_ok)
        _set_button_state(btn_grad0_self, self_ok)
        _set_button_state(btn_grad100_self, self_ok)
        _set_button_state(btn_harm_self, self_ok)
        _set_button_state(btn_overview_iter, iter_ok)
        _set_button_state(btn_grad0_iter, iter_ok)
        _set_button_state(btn_grad100_iter, iter_ok)
        _set_button_state(btn_harm_iter, iter_ok)

    # Step 0: load/save state files
    ttk.Label(frm, text="Step 0: Load/save state").grid(row=0, column=0, sticky="w")
    ttk.Label(frm, text="state key").grid(row=0, column=1, sticky="e")
    state_key_var = tk.StringVar(value="grid_admittance")
    state_keys = list(state_files.keys())
    state_key_box = ttk.Combobox(
        frm,
        textvariable=state_key_var,
        values=state_keys,
        state="readonly",
        width=22,
    )
    state_key_box.grid(row=0, column=2, sticky="w")
    state_path_var = tk.StringVar(value=str(_standard_state_path(state_key_var.get())))
    ttk.Entry(frm, textvariable=state_path_var, width=44).grid(row=0, column=3, columnspan=4, sticky="we", padx=4)

    def _sync_state_path_default(*_) -> None:
        state_path_var.set(str(_standard_state_path(state_key_var.get())))

    state_key_var.trace_add("write", _sync_state_path_default)

    btn_step0_load = tk.Button(
        frm,
        text="Load selected",
        command=lambda: run_step(
            btn_step0_load,
            lambda: _load_state_from_path(
                state_key_var.get(),
                state_path_var.get(),
                lambda msg: _log(log_widget, msg),
            ),
            on_success=lambda _: _refresh_inputs_from_loaded_state(),
        ),
    )
    btn_step0_load.grid(row=0, column=7, padx=4, sticky="w")
    btn_step0_save_as = tk.Button(
        frm,
        text="Save selected as",
        command=lambda: run_step(
            btn_step0_save_as,
            lambda: _save_state_to_path(
                state_key_var.get(),
                state_path_var.get(),
                lambda msg: _log(log_widget, msg),
            ),
        ),
    )
    btn_step0_save_as.grid(row=0, column=8, padx=4, sticky="w")

    # Inputs for step 1
    ttk.Label(frm, text="Step 1: Grid + admittance").grid(row=1, column=0, sticky="w")
    ttk.Label(frm, text="effective subdivisions").grid(row=1, column=1, sticky="e")
    subdiv_var = tk.StringVar(value="3")
    ttk.Entry(frm, textvariable=subdiv_var, width=8).grid(row=1, column=2, sticky="w")
    ttk.Label(frm, text="faces (exact)").grid(row=1, column=3, sticky="e")
    faces_var = tk.StringVar(value=str(_faces_for_subdiv(int(subdiv_var.get()))))
    ttk.Label(frm, textvariable=faces_var).grid(row=1, column=4, sticky="w")
    ttk.Label(frm, text="face spacing (km)").grid(row=1, column=5, sticky="e")
    spacing_var = tk.StringVar(
        value=f"{_mean_face_center_spacing_km(int(subdiv_var.get()), 1.56e6):.1f}"
    )
    ttk.Label(frm, textvariable=spacing_var).grid(row=1, column=6, sticky="w")
    ttk.Label(frm, text="lmax").grid(row=1, column=7, sticky="e")
    lmax_var = tk.StringVar(value="36")
    ttk.Entry(frm, textvariable=lmax_var, width=6).grid(row=1, column=8, sticky="w")
    sh_count_var = tk.StringVar(value="1296")
    ttk.Label(frm, text="# SH coeffs=").grid(row=2, column=3, sticky="e")
    ttk.Label(frm, textvariable=sh_count_var).grid(row=2, column=4, sticky="w")
    default_cfg = GridConfig(nside=1, lmax=1, radius_m=1.56e6, device="cpu")
    default_mean = 2.0 * default_cfg.seawater_conductivity_s_per_m * default_cfg.ocean_thickness_m
    ttk.Label(frm, text="mean conductivity (S)").grid(row=3, column=1, sticky="e")
    mean_cond_var = tk.StringVar(value=f"{default_mean:.3e}")
    ttk.Entry(frm, textvariable=mean_cond_var, width=10).grid(row=3, column=2, sticky="w")
    ttk.Label(frm, text="target l").grid(row=3, column=3, sticky="e")
    mode_l_var = tk.StringVar(value="10")
    ttk.Entry(frm, textvariable=mode_l_var, width=6).grid(row=3, column=4, sticky="w")
    ttk.Label(frm, text="target |m|").grid(row=3, column=5, sticky="e")
    mode_m_var = tk.StringVar(value="2")
    ttk.Entry(frm, textvariable=mode_m_var, width=6).grid(row=3, column=6, sticky="w")
    ttk.Label(frm, text="frac RMS").grid(row=3, column=7, sticky="e")
    frac_rms_var = tk.StringVar(value="0.05")
    ttk.Entry(frm, textvariable=frac_rms_var, width=6).grid(row=3, column=8, sticky="w")
    ttk.Label(frm, text="inductance scale (x)").grid(row=2, column=6, sticky="e")
    inductance_scale_var = tk.StringVar(value="0.0")
    ttk.Entry(frm, textvariable=inductance_scale_var, width=6).grid(row=2, column=7, sticky="w")
    btn_l_from_faces = tk.Button(
        frm,
        text="Set lmax from faces",
        command=lambda: [
            faces_var.set(str(_faces_for_subdiv(int(subdiv_var.get())))),
            spacing_var.set(
                f"{_mean_face_center_spacing_km(int(subdiv_var.get()), 1.56e6):.1f}"
            ),
            lmax_var.set(
                str(
                    _lmax_for_target_faces(
                        int(faces_var.get())
                    )
                )
            ),
            sh_count_var.set(str((int(lmax_var.get()) + 1) ** 2)),
            _update_lmax_button_color(),
        ],
    )
    btn_l_from_faces.grid(row=2, column=5, padx=4, sticky="w")
    btn_step1 = tk.Button(
        frm,
        text="Clear + Run Step 1",
        command=lambda: run_step(
            btn_step1,
            lambda: (
                _clear_outputs(lambda msg: _log(log_widget, msg)),
                step1_build_grid_admittance(
                    int(subdiv_var.get()),
                    int(lmax_var.get()),
                    float(mean_cond_var.get()),
                    float(frac_rms_var.get()),
                    int(mode_l_var.get()),
                    int(mode_m_var.get()),
                    float(inductance_scale_var.get()),
                    lambda msg: _log(log_widget, msg),
                ),
            )[1],
            on_success=lambda res: (
                subdiv_var.set(str(res[1]) if isinstance(res, tuple) and len(res) > 1 else "?"),
                faces_var.set(str(res[3]) if isinstance(res, tuple) and len(res) > 3 else "?"),
                spacing_var.set(
                    f"{_mean_face_center_spacing_km(int(subdiv_var.get()), 1.56e6):.1f}"
                ),
                sh_count_var.set(str((int(lmax_var.get()) + 1) ** 2)),
            ),
        ),
    )
    btn_step1.grid(row=3, column=9, padx=6, sticky="w")
    ttk.Label(frm, text="iter order").grid(row=1, column=4, sticky="e")
    iter_order_var = tk.StringVar(value="3")
    ttk.Entry(frm, textvariable=iter_order_var, width=6).grid(row=1, column=5, sticky="w")

    def _refresh_inputs_from_loaded_state() -> None:
        if not _grid_exists():
            return
        try:
            state = _load_state("grid_admittance.pt")
            subdiv = int(state.get("subdivisions", subdiv_var.get()))
            lmax = int(getattr(state.get("grid_cfg"), "lmax", lmax_var.get()))
            subdiv_var.set(str(subdiv))
            lmax_var.set(str(lmax))
            faces_var.set(str(_faces_for_subdiv(subdiv)))
            spacing_var.set(f"{_mean_face_center_spacing_km(subdiv, 1.56e6):.1f}")
            sh_count_var.set(str((lmax + 1) ** 2))
            if "sigma_mean" in state:
                mean_cond_var.set(f"{float(state['sigma_mean']):.3e}")
            if "sigma_frac_rms" in state:
                frac_rms_var.set(str(float(state["sigma_frac_rms"])))
            if "sigma_mode_l" in state:
                mode_l_var.set(str(int(state["sigma_mode_l"])))
            if "sigma_mode_m" in state:
                mode_m_var.set(str(int(state["sigma_mode_m"])))
            if "inductance_scale" in state:
                inductance_scale_var.set(str(float(state["inductance_scale"])))
            _log(log_widget, "Step 0: refreshed GUI inputs from loaded state.")
        except Exception as exc:  # noqa: BLE001
            _log(log_widget, f"Step 0: unable to refresh GUI inputs ({exc})")

    # Step 1b
    ttk.Label(frm, text="Step 1b: Admittance check").grid(row=4, column=0, sticky="w")
    btn_step1b = tk.Button(
        frm,
        text="Admittance plots",
        command=lambda: run_step_ui(btn_step1b, lambda: step1b_plot_roundtrip(lambda msg: _log(log_widget, msg))),
    )
    btn_step1b.grid(row=4, column=2, padx=6, sticky="w")
    btn_step1b_power = tk.Button(
        frm,
        text="Admittance magnitude (l,m)",
        command=lambda: run_step_ui(
            btn_step1b_power,
            lambda: step1b_plot_admittance_power(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step1b_power.grid(row=4, column=3, padx=6, sticky="w")

    # Step 2
    ttk.Label(frm, text="Step 2: Ambient field").grid(row=5, column=0, sticky="w")
    btn_step2 = tk.Button(
        frm,
        text="Build ambient",
        command=lambda: run_step(btn_step2, lambda: step2_build_ambient(lambda msg: _log(log_widget, msg))),
    )
    btn_step2.grid(row=5, column=2, padx=6, sticky="w")

    # Step 4: First-order solve + plots
    ttk.Label(frm, text="Step 4: First-order solve").grid(row=6, column=0, sticky="w")
    btn_step4_first = tk.Button(
        frm,
        text="Solve first-order",
        command=lambda: run_step(
            btn_step4_first,
            lambda: step3_solve_currents(True, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step4_first.grid(row=6, column=2, padx=4, sticky="w")
    btn_overview_first = tk.Button(
        frm,
        text="Overview (first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_first,
            lambda: step4_render_overview("first_order", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_first.grid(row=6, column=3, padx=4, sticky="w")
    btn_grad0_first = tk.Button(
        frm,
        text="Gradients @ surface (first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_first,
            lambda: step4_render_gradient("first_order", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_first.grid(row=6, column=4, padx=4, sticky="w")
    btn_grad100_first = tk.Button(
        frm,
        text="Gradients @ 100 km (first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_first,
            lambda: step4_render_gradient("first_order", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_first.grid(row=6, column=5, padx=4, sticky="w")
    btn_harm_first = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_first,
            lambda: step4_plot_harmonics("first_order", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_first.grid(row=6, column=6, padx=4, sticky="w")

    # Step 5: Self-consistent solve + plots
    ttk.Label(frm, text="Step 5: Self-consistent solve").grid(row=7, column=0, sticky="w")
    btn_step5_self = tk.Button(
        frm,
        text="Solve self-consistent",
        command=lambda: run_step(
            btn_step5_self,
            lambda: step3_solve_currents(False, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step5_self.grid(row=7, column=2, padx=4, sticky="w")
    btn_overview_self = tk.Button(
        frm,
        text="Overview (self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_self,
            lambda: step4_render_overview("self_consistent", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_self.grid(row=7, column=3, padx=4, sticky="w")
    btn_grad0_self = tk.Button(
        frm,
        text="Gradients @ surface (self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_self,
            lambda: step4_render_gradient("self_consistent", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_self.grid(row=7, column=4, padx=4, sticky="w")
    btn_grad100_self = tk.Button(
        frm,
        text="Gradients @ 100 km (self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_self,
            lambda: step4_render_gradient("self_consistent", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_self.grid(row=7, column=5, padx=4, sticky="w")
    btn_harm_self = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_self,
            lambda: step4_plot_harmonics("self_consistent", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_self.grid(row=7, column=6, padx=4, sticky="w")

    # Step 6: Iterative series solve + plots
    ttk.Label(frm, text="Step 6: Iterative solve").grid(row=8, column=0, sticky="w")
    btn_step6_iter = tk.Button(
        frm,
        text="Solve iterative",
        command=lambda: run_step(
            btn_step6_iter,
            lambda: step6_iterative_solve(int(iter_order_var.get()), lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step6_iter.grid(row=8, column=2, padx=4, sticky="w")
    btn_overview_iter = tk.Button(
        frm,
        text="Overview (iterative)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_iter,
            lambda: step4_render_overview("iterative", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_iter.grid(row=8, column=3, padx=4, sticky="w")
    btn_grad0_iter = tk.Button(
        frm,
        text="Gradients @ surface (iterative)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_iter,
            lambda: step4_render_gradient("iterative", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_iter.grid(row=8, column=4, padx=4, sticky="w")
    btn_grad100_iter = tk.Button(
        frm,
        text="Gradients @ 100 km (iterative)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_iter,
            lambda: step4_render_gradient("iterative", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_iter.grid(row=8, column=5, padx=4, sticky="w")
    btn_harm_iter = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_iter,
            lambda: step4_plot_harmonics("iterative", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_iter.grid(row=8, column=6, padx=4, sticky="w")

    frm.rowconfigure(11, weight=1)
    frm.columnconfigure(6, weight=1)

    subdiv_var.trace_add("write", lambda *_: _update_lmax_button_color())
    lmax_var.trace_add("write", lambda *_: _update_lmax_button_color())
    _update_lmax_button_color()
    _update_button_states()

    root.mainloop()


if __name__ == "__main__":
    main()
