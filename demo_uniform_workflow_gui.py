"""
GUI for comparing uniform and spectral (non-uniform) solvers using the same
uniform thin-shell admittance.

Steps:
1) Build grid + uniform admittance
2) Build ambient field
3) Uniform solves (first-order, self-consistent)
4) Spectral first-order solve + plots
5) Spectral self-consistent solve + plots
6) Spectral iterative solve + plots
7) Compare solves against uniform first-order
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

from ambient_driver import build_ambient_driver_x
from europa.config import GridConfig, ModelConfig
from europa.grid import make_grid
from europa.transforms import sh_forward, sh_inverse
from europa.simulation import Simulation
from europa.solvers import (
    _flatten_lm,
    _unflatten_lm,
    _build_self_field_diag,
    toroidal_e_from_radial_b,
    solve_uniform_first_order_sim,
    solve_uniform_self_consistent_sim,
)
from europa.solver_variants.solver_variant_precomputed import (
    _build_mixing_matrix_precomputed_sparse,
)
from europa import inductance
from europa.gradient_utils import render_gradient_map
from render_demo_overview import render_demo_overview
from render_phasor_maps import _build_mesh
from gaunt.assemble_gaunt_checkpoints import assemble_in_memory
from phasor_data import PhasorSimulation

STATE_DIR = Path("artifacts/uniform_workflow")
FIG_DIR = Path("figures/uniform")
GAUNT_CACHE = Path("data/gaunt_cache_wigxjpf")


def _log(text_widget: tk.Text, msg: str) -> None:
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)
    text_widget.update_idletasks()




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
    return best


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
) -> torch.Tensor:
    """Compute complex sheet admittance from thin-shell impedance model."""
    sigma_s = sigma_s.to(torch.float64)
    X_s = omega * float(inductance.MU0) * radius_m / 2.0
    R_s = torch.where(sigma_s > 0, 1.0 / sigma_s, torch.zeros_like(sigma_s))
    Z = R_s + 1j * X_s
    Y = torch.where(sigma_s > 0, 1.0 / Z, torch.zeros_like(Z))
    return Y.to(torch.complex128)


def _uniformize_spectral_admittance(Y_s: torch.Tensor) -> torch.Tensor:
    """Keep only the l=0, m=0 admittance coefficient (force perfectly uniform Y_s)."""
    Y = torch.zeros_like(Y_s)
    lmax = Y_s.shape[0] - 1
    Y[0, lmax] = Y_s[0, lmax]
    return Y


def step1_build_grid_uniform_admittance(subdiv: int, lmax: int, sigma_2d_max: float, log) -> Path:
    subdiv = max(0, int(subdiv))
    actual_faces = _faces_for_subdiv(subdiv)
    nside = _nside_from_subdivisions(subdiv)
    grid_cfg = GridConfig(nside=nside, lmax=lmax, radius_m=1.56e6, device="cpu")
    grid = make_grid(grid_cfg)
    sigma_2d_max = max(0.0, float(sigma_2d_max))
    omega = 2.0 * math.pi / (9.925 * 3600.0)
    cond_real = torch.full(
        (grid.positions.shape[0],),
        float(sigma_2d_max),
        dtype=torch.float64,
        device=grid.positions.device,
    )
    cond = _complex_sheet_admittance(cond_real, omega, grid_cfg.radius_m)
    Y_s = sh_forward(cond, grid.positions.to(torch.float64), lmax=grid_cfg.lmax, weights=grid.areas.to(torch.float64))
    state = {
        "grid_cfg": grid_cfg,
        "positions": grid.positions,
        "normals": grid.normals,
        "areas": grid.areas,
        "neighbors": grid.neighbors,
        "admittance_uniform": cond[0].clone().detach(),
        "admittance_spectral": Y_s,
        "admittance_grid": cond,
        "sigma_2d_max": sigma_2d_max,
        "omega": omega,
        "subdivisions": subdiv,
    }
    path = _save_state("grid_admittance.pt", state)
    log(f"Step 1 complete (uniform admittance, subdivisions={subdiv}, faces={grid.positions.shape[0]}). Saved grid to {path}")
    return path, subdiv, int(grid.positions.shape[0]), actual_faces


def step1b_plot_roundtrip(log) -> None:
    state = _load_state("grid_admittance.pt")
    positions = state["positions"].to(torch.float64)
    weights = state["areas"].to(torch.float64)
    coeffs = state["admittance_spectral"]
    orig = state.get("admittance_grid")
    if orig is None:
        raise RuntimeError("Missing admittance grid. Run Step 1 before Step 1b.")

    recon = sh_inverse(coeffs, positions, weights)

    orig = orig.to(torch.complex128).reshape(-1).cpu().numpy()
    recon = recon.reshape(-1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    idx = np.arange(orig.size)
    real_a = orig.real
    real_b = recon.real
    imag_a = orig.imag
    imag_b = recon.imag

    for ax, a, b, label in (
        (axes[0], real_a, real_b, "Real"),
        (axes[1], imag_a, imag_b, "Imag"),
    ):
        ax.scatter(idx, a, s=8, alpha=0.7, label="Roundtrip 0")
        ax.scatter(idx, b, s=8, alpha=0.7, label="Roundtrip 1")
        ax.vlines(idx, a, b, colors="gray", alpha=0.3, linewidth=0.5)
        ax.set_title(f"{label} part: roundtrip 0 vs 1")
        ax.set_xlabel("Grid point index")
        ax.set_ylabel("Admittance (S)")
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    plt.show()
    _plot_roundtrip_sphere_maps(
        orig,
        "Roundtrip 0",
        recon,
        "Roundtrip 1",
        positions=positions,
        subdivisions=int(state["subdivisions"]),
        radius=float(state["grid_cfg"].radius_m),
    )
    log("Step 1b complete. Displayed round-trip scatter plots.")


def step1b_plot_admittance_power(log) -> None:
    state = _load_state("grid_admittance.pt")
    coeffs = state.get("admittance_spectral")
    if coeffs is None:
        raise RuntimeError("Missing admittance_spectral. Run Step 1 before plotting admittance power.")

    l_b, m_b, mag = _flatten_harmonics(coeffs.to(torch.complex128))
    magnitude = mag
    peak = float(max(np.max(magnitude), 1e-30))
    eps = peak * 1e-9
    active = magnitude > eps
    active_ls = l_b[active]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    keep = l_b <= l_cut
    x = np.arange(int(np.sum(keep)))
    tick_idx = np.where(m_b[keep] == 0)[0]
    tick_labels = [f"({l},0)" for l in l_b[keep][tick_idx]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, np.maximum(magnitude[keep], eps), color="#ff9c43")
    ax.set_yscale("log")
    ax.set_xlabel("(l,m) ordering; ticks at m=0")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_ylabel("|Y_s| (S)")
    ax.set_title("Admittance mode magnitude")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.show()
    log("Step 1b complete. Plotted admittance mode power.")


def step1c_plot_roundtrip_stability(log) -> None:
    state = _load_state("grid_admittance.pt")
    positions = state["positions"].to(torch.float64)
    weights = state["areas"].to(torch.float64)
    coeffs = state["admittance_spectral"]
    orig = state.get("admittance_grid")
    if orig is None:
        raise RuntimeError("Missing admittance grid. Run Step 1 before Step 1c.")

    lmax = coeffs.shape[-2] - 1
    round1 = sh_inverse(coeffs, positions, weights)
    coeffs2 = sh_forward(round1, positions, lmax=lmax, weights=weights)
    round2 = sh_inverse(coeffs2, positions, weights)

    round1 = round1.reshape(-1).cpu().numpy()
    round2 = round2.reshape(-1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    idx = np.arange(round1.size)
    real_a = round1.real
    real_b = round2.real
    imag_a = round1.imag
    imag_b = round2.imag

    for ax, a, b, label in (
        (axes[0], real_a, real_b, "Real"),
        (axes[1], imag_a, imag_b, "Imag"),
    ):
        ax.scatter(idx, a, s=8, alpha=0.7, label="Roundtrip 1")
        ax.scatter(idx, b, s=8, alpha=0.7, label="Roundtrip 2")
        ax.vlines(idx, a, b, colors="gray", alpha=0.3, linewidth=0.5)
        ax.set_title(f"{label} part: roundtrip 1 vs 2")
        ax.set_xlabel("Grid point index")
        ax.set_ylabel("Admittance (S)")
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    plt.show()
    _plot_roundtrip_sphere_maps(
        round1,
        "Roundtrip 1",
        round2,
        "Roundtrip 2",
        positions=positions,
        subdivisions=int(state["subdivisions"]),
        radius=float(state["grid_cfg"].radius_m),
    )
    log("Step 1c complete. Displayed roundtrip stability scatter plots.")


def _plot_roundtrip_sphere_maps(
    values_a: np.ndarray,
    label_a: str,
    values_b: np.ndarray,
    label_b: str,
    positions: torch.Tensor,
    subdivisions: int,
    radius: float,
    elev: float = 20.0,
    azim: float = 30.0,
) -> None:
    vertices, faces, centers = _build_mesh(radius, subdivisions=subdivisions, stride=1)
    face_count = faces.shape[0]
    node_count = int(positions.shape[0])
    if values_a.size != node_count or values_b.size != node_count:
        raise RuntimeError(
            "Roundtrip values do not match grid positions "
            f"(nodes={node_count}, roundtrip0={values_a.size}, roundtrip1={values_b.size})."
        )

    tri_verts = vertices[faces].cpu().numpy()
    vals_a = values_a.reshape(-1)
    vals_b = values_b.reshape(-1)
    pos = positions.detach().cpu().to(torch.float64)
    cen = centers.detach().cpu().to(torch.float64)
    nearest = torch.cdist(cen, pos).argmin(dim=1).cpu().numpy()
    if len(nearest) != face_count:
        raise RuntimeError("Failed to map grid values to mesh faces.")
    vals_a = vals_a[nearest]
    vals_b = vals_b[nearest]

    real_vmax = float(max(np.max(np.abs(vals_a.real)), np.max(np.abs(vals_b.real)), 1e-12))
    imag_vmax = float(max(np.max(np.abs(vals_a.imag)), np.max(np.abs(vals_b.imag)), 1e-12))
    norm_real = mcolors.Normalize(vmin=-real_vmax, vmax=real_vmax)
    norm_imag = mcolors.Normalize(vmin=-imag_vmax, vmax=imag_vmax)
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.2)

    for row, (label, vals) in enumerate(((label_a, vals_a), (label_b, vals_b))):
        ax_real = fig.add_subplot(gs[row, 0], projection="3d")
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
        ax_imag = fig.add_subplot(gs[row, 1], projection="3d")
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
        admittance_uniform=state.get("admittance_uniform"),
        admittance_spectral=state["admittance_spectral"],
        B_radial=state["B_radial_spec"],
        period_sec=state["period_sec"],
    )


def _log_matrix_diagnostics(name: str, A: torch.Tensor, log) -> None:
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


def _build_mixing_matrix(state, log) -> torch.Tensor:
    grid_cfg: GridConfig = state["grid_cfg"]
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
    Y_s_uniform = _uniformize_spectral_admittance(state["admittance_spectral"])
    return _build_mixing_matrix_precomputed_sparse(
        grid_cfg.lmax,
        state["ambient_cfg"].omega_jovian,
        float(state["grid_cfg"].radius_m),
        Y_s_uniform,
        G_sparse,
    )


def step3_uniform_first_order(log) -> Path:
    state = _load_state("ambient.pt")
    sim_out = _build_phasor_base(state)
    log("Uniform first-order solve...")
    sim_out = solve_uniform_first_order_sim(sim_out)
    payload = {"label": "uniform_first_order", "phasor_sim": sim_out}
    path = _save_state("solution_uniform_first_order.pt", payload)
    log(f"Step 3 complete. Saved uniform first-order solution to {path}")
    return path


def step3_uniform_self_consistent(log) -> Path:
    state = _load_state("ambient.pt")
    sim_out = _build_phasor_base(state)
    log("Uniform self-consistent solve...")
    sim_out = solve_uniform_self_consistent_sim(sim_out)
    payload = {"label": "uniform_self_consistent", "phasor_sim": sim_out}
    path = _save_state("solution_uniform_self_consistent.pt", payload)
    log(f"Step 3 complete. Saved uniform self-consistent solution to {path}")
    return path


def step4_spectral_first_order(log) -> Path:
    state = _load_state("ambient.pt")
    grid_cfg: GridConfig = state["grid_cfg"]
    base = _build_phasor_base(state)
    mixing_matrix = _build_mixing_matrix(state, log)
    log("Spectral first-order solve...")
    sim_out = PhasorSimulation.from_serializable(base.to_serializable())
    sim_out.E_toroidal = toroidal_e_from_radial_b(sim_out.B_radial, sim_out.omega, sim_out.radius_m)
    b_flat = _flatten_lm(sim_out.B_radial.to(torch.complex128))
    k_flat = mixing_matrix @ b_flat
    sim_out.K_toroidal = _unflatten_lm(k_flat, grid_cfg.lmax)
    sim_out.K_toroidal[0, :] = 0.0
    sim_out.K_poloidal = torch.zeros_like(sim_out.K_toroidal)
    sim_out.B_tor_emit, sim_out.B_pol_emit, sim_out.B_rad_emit = inductance.spectral_b_from_surface_currents(
        sim_out.K_toroidal, sim_out.K_poloidal, radius=sim_out.radius_m
    )
    src_energy = float((sim_out.B_radial.abs() ** 2).sum().item())
    resp_energy = float((sim_out.B_rad_emit.abs() ** 2).sum().item())
    if resp_energy > src_energy:
        log(
            "Warning: spectral first-order response energy exceeds source energy "
            f"(resp={resp_energy:.3e} > src={src_energy:.3e})."
        )
    sim_out.solver_variant = "spectral_first_order_precomputed_gaunt_sparse"
    payload = {"label": "spectral_first_order", "phasor_sim": sim_out}
    path = _save_state("solution_spectral_first_order.pt", payload)
    log(f"Step 4 complete. Saved spectral first-order solution to {path}")
    return path


def step5_spectral_self_consistent(log) -> Path:
    state = _load_state("ambient.pt")
    grid_cfg: GridConfig = state["grid_cfg"]
    base = _build_phasor_base(state)
    mixing_matrix = _build_mixing_matrix(state, log)
    log("Spectral self-consistent solve...")
    sim_out = PhasorSimulation.from_serializable(base.to_serializable())
    sim_out.E_toroidal = toroidal_e_from_radial_b(sim_out.B_radial, sim_out.omega, sim_out.radius_m)
    b_ext_flat = _flatten_lm(sim_out.B_radial.to(torch.complex128))
    S_diag = _build_self_field_diag(grid_cfg.lmax, sim_out.grid_positions.device, torch.complex128)
    I = torch.eye(mixing_matrix.shape[0], device=mixing_matrix.device, dtype=torch.complex128)
    A = I - torch.diag(S_diag) @ mixing_matrix
    _log_matrix_diagnostics("A (I - S*M)", A, log)
    cond = _matrix_condition(A)
    if not np.isfinite(cond) or cond > 1e8:
        raise RuntimeError(f"Ill-conditioned A (cond={cond:.3e}); aborting spectral self-consistent solve.")
    b_tot = torch.linalg.solve(A, b_ext_flat)
    k_flat = mixing_matrix @ b_tot
    sim_out.K_toroidal = _unflatten_lm(k_flat, grid_cfg.lmax)
    sim_out.K_toroidal[0, :] = 0.0
    sim_out.K_poloidal = torch.zeros_like(sim_out.K_toroidal)
    sim_out.B_tor_emit, sim_out.B_pol_emit, sim_out.B_rad_emit = inductance.spectral_b_from_surface_currents(
        sim_out.K_toroidal, sim_out.K_poloidal, radius=sim_out.radius_m
    )
    sim_out.solver_variant = "spectral_self_consistent_precomputed_gaunt_sparse"
    payload = {"label": "spectral_self_consistent", "phasor_sim": sim_out}
    path = _save_state("solution_spectral_self_consistent.pt", payload)
    log(f"Step 5 complete. Saved spectral self-consistent solution to {path}")
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
    out_path = FIG_DIR / f"uniform_{label}_overview.png"
    log(f"Overview: label={label}, lmax={sim_out.lmax}, subdivisions={subdivisions}")
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
    save_path = FIG_DIR / f"uniform_grad_{int(altitude_m):d}m_{label}.png"
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


def step6_spectral_iterative(order: int, log) -> Path:
    state = _load_state("ambient.pt")
    grid_cfg: GridConfig = state["grid_cfg"]
    base = _build_phasor_base(state)
    mixing_matrix = _build_mixing_matrix(state, log)

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
    label = "spectral_iterative"

    payload = {
        "label": label,
        "phasor_sim": sim_out,
    }
    path = _save_state("solution_spectral_iterative.pt", payload)
    log(f"Step 6 complete. Saved spectral iterative solution to {path}")
    return path


def step7_compare_to_uniform_first_order(log) -> None:
    def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
        diff = (a - b).reshape(-1)
        denom = b.reshape(-1)
        num = float(torch.linalg.norm(diff).item())
        den = float(torch.linalg.norm(denom).item())
        return float("inf") if den == 0.0 else num / den

    ref_payload = _load_solution("uniform_first_order")
    ref = ref_payload["phasor_sim"]
    if ref.K_toroidal is None or ref.B_rad_emit is None:
        raise RuntimeError("Uniform first-order solution missing currents or emitted field.")

    candidates = [
        "uniform_self_consistent",
        "spectral_first_order",
        "spectral_self_consistent",
        "spectral_iterative",
    ]
    for label in candidates:
        path = STATE_DIR / f"solution_{label}.pt"
        if not path.exists():
            log(f"Compare: missing {label} (no solution file).")
            continue
        payload = _load_solution(label)
        sim = payload["phasor_sim"]
        if sim.K_toroidal is None or sim.B_rad_emit is None:
            log(f"Compare: {label} missing K_toroidal or B_rad_emit.")
            continue
        k_err = _rel_error(sim.K_toroidal.to(torch.complex128), ref.K_toroidal.to(torch.complex128))
        b_err = _rel_error(sim.B_rad_emit.to(torch.complex128), ref.B_rad_emit.to(torch.complex128))
        log(f"Compare vs uniform first-order: {label} -> rel_err K_tor={k_err:.3e}, B_rad={b_err:.3e}")


def main():
    root = tk.Tk()
    root.title("Uniform vs Spectral Workflow")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Log output (placed early so lambdas can close over it)
    log_widget = tk.Text(frm, height=18, width=45)
    log_widget.grid(row=12, column=0, columnspan=10, pady=8, sticky="nsew")

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

    btn_clear = tk.Button(
        frm,
        text="Clear figures/data",
        command=lambda: run_step(
            btn_clear,
            lambda: _clear_outputs(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_clear.grid(row=0, column=0, padx=4, pady=(0, 6), sticky="w")

    def _solution_exists(label: str) -> bool:
        return (STATE_DIR / f"solution_{label}.pt").exists()

    def _grid_exists() -> bool:
        return (STATE_DIR / "grid_admittance.pt").exists()

    def _ambient_exists() -> bool:
        return (STATE_DIR / "ambient.pt").exists()

    def _set_button_state(btn: tk.Button, enabled: bool) -> None:
        if enabled:
            btn.config(state=tk.NORMAL)
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
        uniform_first_ok = _solution_exists("uniform_first_order")
        uniform_self_ok = _solution_exists("uniform_self_consistent")
        spectral_first_ok = _solution_exists("spectral_first_order")
        spectral_self_ok = _solution_exists("spectral_self_consistent")
        spectral_iter_ok = _solution_exists("spectral_iterative")
        _set_button_state(btn_step1b, grid_ok)
        _set_button_state(btn_step1b_power, grid_ok)
        _set_button_state(btn_step1c, grid_ok)
        _set_button_state(btn_step2, grid_ok)
        _set_button_state(btn_step3_uniform_first, ambient_ok)
        _set_button_state(btn_step3_uniform_self, ambient_ok)
        _set_button_state(btn_step4_spectral_first, ambient_ok)
        _set_button_state(btn_step5_spectral_self, ambient_ok)
        _set_button_state(btn_step6_spectral_iter, ambient_ok)
        _set_button_state(btn_overview_uniform_first, uniform_first_ok)
        _set_button_state(btn_grad0_uniform_first, uniform_first_ok)
        _set_button_state(btn_grad100_uniform_first, uniform_first_ok)
        _set_button_state(btn_harm_uniform_first, uniform_first_ok)
        _set_button_state(btn_overview_uniform_self, uniform_self_ok)
        _set_button_state(btn_grad0_uniform_self, uniform_self_ok)
        _set_button_state(btn_grad100_uniform_self, uniform_self_ok)
        _set_button_state(btn_harm_uniform_self, uniform_self_ok)
        _set_button_state(btn_overview_spectral_first, spectral_first_ok)
        _set_button_state(btn_grad0_spectral_first, spectral_first_ok)
        _set_button_state(btn_grad100_spectral_first, spectral_first_ok)
        _set_button_state(btn_harm_spectral_first, spectral_first_ok)
        _set_button_state(btn_overview_spectral_self, spectral_self_ok)
        _set_button_state(btn_grad0_spectral_self, spectral_self_ok)
        _set_button_state(btn_grad100_spectral_self, spectral_self_ok)
        _set_button_state(btn_harm_spectral_self, spectral_self_ok)
        _set_button_state(btn_overview_spectral_iter, spectral_iter_ok)
        _set_button_state(btn_grad0_spectral_iter, spectral_iter_ok)
        _set_button_state(btn_grad100_spectral_iter, spectral_iter_ok)
        _set_button_state(btn_harm_spectral_iter, spectral_iter_ok)
        _set_button_state(btn_step7_compare, uniform_first_ok)

    # Inputs for step 1
    ttk.Label(frm, text="Step 1: Grid + uniform admittance").grid(row=1, column=0, sticky="w")
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
    lmax_var = tk.StringVar(value="35")
    ttk.Entry(frm, textvariable=lmax_var, width=6).grid(row=1, column=8, sticky="w")
    sh_count_var = tk.StringVar(value="1296")
    ttk.Label(frm, text="# SH coeffs=").grid(row=2, column=1, sticky="e")
    ttk.Label(frm, textvariable=sh_count_var).grid(row=2, column=2, sticky="w")
    ttk.Label(frm, text="iter order").grid(row=2, column=3, sticky="e")
    iter_order_var = tk.StringVar(value="3")
    ttk.Entry(frm, textvariable=iter_order_var, width=6).grid(row=2, column=4, sticky="w")
    default_cfg = GridConfig(nside=1, lmax=1, radius_m=1.56e6, device="cpu")
    default_sigma_2d = 2.0 * default_cfg.seawater_conductivity_s_per_m * default_cfg.ocean_thickness_m
    ttk.Label(frm, text="sheet conductivity").grid(row=2, column=6, sticky="e")
    sigma_2d_var = tk.StringVar(value=f"{default_sigma_2d:.3e}")
    ttk.Entry(frm, textvariable=sigma_2d_var, width=10).grid(row=2, column=7, sticky="w")
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
        text="Run Step 1",
        command=lambda: run_step(
            btn_step1,
            lambda: step1_build_grid_uniform_admittance(
                int(subdiv_var.get()),
                int(lmax_var.get()),
                float(sigma_2d_var.get()),
                lambda msg: _log(log_widget, msg),
            ),
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
    btn_step1.grid(row=1, column=3, padx=6, sticky="w")

    # Step 1b
    ttk.Label(frm, text="Step 1b: Roundtrip check").grid(row=3, column=0, sticky="w")
    btn_step1b = tk.Button(
        frm,
        text="Plot admittance roundtrip",
        command=lambda: run_step_ui(btn_step1b, lambda: step1b_plot_roundtrip(lambda msg: _log(log_widget, msg))),
    )
    btn_step1b.grid(row=3, column=2, padx=6, sticky="w")
    btn_step1b_power = tk.Button(
        frm,
        text="Admittance power (l,m)",
        command=lambda: run_step_ui(
            btn_step1b_power,
            lambda: step1b_plot_admittance_power(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step1b_power.grid(row=3, column=3, padx=6, sticky="w")

    # Step 1c
    ttk.Label(frm, text="Step 1c: Roundtrip stability").grid(row=4, column=0, sticky="w")
    btn_step1c = tk.Button(
        frm,
        text="Plot roundtrip 1 vs 2",
        command=lambda: run_step_ui(btn_step1c, lambda: step1c_plot_roundtrip_stability(lambda msg: _log(log_widget, msg))),
    )
    btn_step1c.grid(row=4, column=2, padx=6, sticky="w")

    # Step 2
    ttk.Label(frm, text="Step 2: Ambient field").grid(row=5, column=0, sticky="w")
    btn_step2 = tk.Button(
        frm,
        text="Build ambient",
        command=lambda: run_step(btn_step2, lambda: step2_build_ambient(lambda msg: _log(log_widget, msg))),
    )
    btn_step2.grid(row=5, column=2, padx=6, sticky="w")

    # Step 3: Uniform solves + plots
    ttk.Label(frm, text="Step 3: Uniform first-order").grid(row=6, column=0, sticky="w")
    btn_step3_uniform_first = tk.Button(
        frm,
        text="Solve uniform first-order",
        command=lambda: run_step(
            btn_step3_uniform_first,
            lambda: step3_uniform_first_order(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step3_uniform_first.grid(row=6, column=2, padx=4, sticky="w")
    btn_overview_uniform_first = tk.Button(
        frm,
        text="Overview (uniform first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_uniform_first,
            lambda: step4_render_overview("uniform_first_order", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_uniform_first.grid(row=6, column=3, padx=4, sticky="w")
    btn_grad0_uniform_first = tk.Button(
        frm,
        text="Gradients @ surface (uniform first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_uniform_first,
            lambda: step4_render_gradient("uniform_first_order", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_uniform_first.grid(row=6, column=4, padx=4, sticky="w")
    btn_grad100_uniform_first = tk.Button(
        frm,
        text="Gradients @ 100 km (uniform first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_uniform_first,
            lambda: step4_render_gradient("uniform_first_order", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_uniform_first.grid(row=6, column=5, padx=4, sticky="w")
    btn_harm_uniform_first = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_uniform_first,
            lambda: step4_plot_harmonics("uniform_first_order", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_uniform_first.grid(row=6, column=6, padx=4, sticky="w")

    ttk.Label(frm, text="Step 3b: Uniform self-consistent").grid(row=7, column=0, sticky="w")
    btn_step3_uniform_self = tk.Button(
        frm,
        text="Solve uniform self-consistent",
        command=lambda: run_step(
            btn_step3_uniform_self,
            lambda: step3_uniform_self_consistent(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step3_uniform_self.grid(row=7, column=2, padx=4, sticky="w")
    btn_overview_uniform_self = tk.Button(
        frm,
        text="Overview (uniform self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_uniform_self,
            lambda: step4_render_overview("uniform_self_consistent", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_uniform_self.grid(row=7, column=3, padx=4, sticky="w")
    btn_grad0_uniform_self = tk.Button(
        frm,
        text="Gradients @ surface (uniform self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_uniform_self,
            lambda: step4_render_gradient("uniform_self_consistent", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_uniform_self.grid(row=7, column=4, padx=4, sticky="w")
    btn_grad100_uniform_self = tk.Button(
        frm,
        text="Gradients @ 100 km (uniform self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_uniform_self,
            lambda: step4_render_gradient("uniform_self_consistent", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_uniform_self.grid(row=7, column=5, padx=4, sticky="w")
    btn_harm_uniform_self = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_uniform_self,
            lambda: step4_plot_harmonics("uniform_self_consistent", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_uniform_self.grid(row=7, column=6, padx=4, sticky="w")

    # Step 4: Spectral first-order + plots
    ttk.Label(frm, text="Step 4: Spectral first-order").grid(row=8, column=0, sticky="w")
    btn_step4_spectral_first = tk.Button(
        frm,
        text="Solve spectral first-order",
        command=lambda: run_step(
            btn_step4_spectral_first,
            lambda: step4_spectral_first_order(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step4_spectral_first.grid(row=8, column=2, padx=4, sticky="w")
    btn_overview_spectral_first = tk.Button(
        frm,
        text="Overview (spectral first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_spectral_first,
            lambda: step4_render_overview("spectral_first_order", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_spectral_first.grid(row=8, column=3, padx=4, sticky="w")
    btn_grad0_spectral_first = tk.Button(
        frm,
        text="Gradients @ surface (spectral first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_spectral_first,
            lambda: step4_render_gradient("spectral_first_order", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_spectral_first.grid(row=8, column=4, padx=4, sticky="w")
    btn_grad100_spectral_first = tk.Button(
        frm,
        text="Gradients @ 100 km (spectral first-order)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_spectral_first,
            lambda: step4_render_gradient("spectral_first_order", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_spectral_first.grid(row=8, column=5, padx=4, sticky="w")
    btn_harm_spectral_first = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_spectral_first,
            lambda: step4_plot_harmonics("spectral_first_order", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_spectral_first.grid(row=8, column=6, padx=4, sticky="w")

    # Step 5: Spectral self-consistent + plots
    ttk.Label(frm, text="Step 5: Spectral self-consistent").grid(row=9, column=0, sticky="w")
    btn_step5_spectral_self = tk.Button(
        frm,
        text="Solve spectral self-consistent",
        command=lambda: run_step(
            btn_step5_spectral_self,
            lambda: step5_spectral_self_consistent(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step5_spectral_self.grid(row=9, column=2, padx=4, sticky="w")
    btn_overview_spectral_self = tk.Button(
        frm,
        text="Overview (spectral self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_spectral_self,
            lambda: step4_render_overview("spectral_self_consistent", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_spectral_self.grid(row=9, column=3, padx=4, sticky="w")
    btn_grad0_spectral_self = tk.Button(
        frm,
        text="Gradients @ surface (spectral self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_spectral_self,
            lambda: step4_render_gradient("spectral_self_consistent", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_spectral_self.grid(row=9, column=4, padx=4, sticky="w")
    btn_grad100_spectral_self = tk.Button(
        frm,
        text="Gradients @ 100 km (spectral self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_spectral_self,
            lambda: step4_render_gradient("spectral_self_consistent", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_spectral_self.grid(row=9, column=5, padx=4, sticky="w")
    btn_harm_spectral_self = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_spectral_self,
            lambda: step4_plot_harmonics("spectral_self_consistent", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_spectral_self.grid(row=9, column=6, padx=4, sticky="w")

    # Step 6: Spectral iterative + plots
    ttk.Label(frm, text="Step 6: Spectral iterative").grid(row=10, column=0, sticky="w")
    btn_step6_spectral_iter = tk.Button(
        frm,
        text="Solve spectral iterative",
        command=lambda: run_step(
            btn_step6_spectral_iter,
            lambda: step6_spectral_iterative(int(iter_order_var.get()), lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step6_spectral_iter.grid(row=10, column=2, padx=4, sticky="w")
    btn_overview_spectral_iter = tk.Button(
        frm,
        text="Overview (spectral iterative)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview_spectral_iter,
            lambda: step4_render_overview("spectral_iterative", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_overview_spectral_iter.grid(row=10, column=3, padx=4, sticky="w")
    btn_grad0_spectral_iter = tk.Button(
        frm,
        text="Gradients @ surface (spectral iterative)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad0_spectral_iter,
            lambda: step4_render_gradient("spectral_iterative", 0.0, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad0_spectral_iter.grid(row=10, column=4, padx=4, sticky="w")
    btn_grad100_spectral_iter = tk.Button(
        frm,
        text="Gradients @ 100 km (spectral iterative)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_spectral_iter,
            lambda: step4_render_gradient("spectral_iterative", 100e3, lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_grad100_spectral_iter.grid(row=10, column=5, padx=4, sticky="w")
    btn_harm_spectral_iter = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm_spectral_iter,
            lambda: step4_plot_harmonics("spectral_iterative", lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm_spectral_iter.grid(row=10, column=6, padx=4, sticky="w")

    # Step 7: Compare solves
    ttk.Label(frm, text="Step 7: Compare to uniform first-order").grid(row=11, column=0, sticky="w")
    btn_step7_compare = tk.Button(
        frm,
        text="Compare solutions",
        command=lambda: run_step(
            btn_step7_compare,
            lambda: step7_compare_to_uniform_first_order(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step7_compare.grid(row=11, column=2, padx=4, sticky="w")

    frm.rowconfigure(12, weight=1)
    frm.columnconfigure(6, weight=1)

    subdiv_var.trace_add("write", lambda *_: _update_lmax_button_color())
    lmax_var.trace_add("write", lambda *_: _update_lmax_button_color())
    _update_lmax_button_color()
    _update_button_states()

    root.mainloop()


if __name__ == "__main__":
    main()
