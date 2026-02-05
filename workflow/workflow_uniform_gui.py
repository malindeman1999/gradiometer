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
from tkinter import ttk, filedialog
from pathlib import Path
import time
import math
import threading
import shutil
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import numpy as np

from workflow.ambient_field.ambient_driver import build_ambient_driver_x
from europa_model.config import GridConfig, ModelConfig
from europa_model.transforms import sh_forward, sh_inverse
from europa_model.solvers import (
    _flatten_lm,
    _unflatten_lm,
    _build_self_field_diag,
    toroidal_e_from_radial_b,
    solve_uniform_first_order_sim,
    solve_uniform_self_consistent_sim,
)
from europa_model.solver_variants.solver_variant_precomputed import (
    _build_mixing_matrix_precomputed_sparse,
)
from europa_model import inductance
from europa_model.gradient_utils import render_gradient_map
from workflow.plotting.render_demo_overview import render_demo_overview
from workflow.plotting.sphere_roundtrip import build_roundtrip_grid, sphere_image
from gaunt.assemble_gaunt_checkpoints import assemble_in_memory
from workflow.data_objects.phasor_data import PhasorSimulation

BASE_RUN_DIR = Path("workflow/artifacts/uniform_workflow")
STATE_DIR = BASE_RUN_DIR
FIG_DIR = BASE_RUN_DIR / "figures"
LOG_PATH = STATE_DIR / "log.txt"
GAUNT_CACHE = Path("gaunt/data/gaunt_cache_wigxjpf")


def _log(text_widget: tk.Text, msg: str) -> None:
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)
    text_widget.update_idletasks()
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.open("a", encoding="utf-8").write(msg + "\n")
    except Exception:
        pass


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_run_dir(prefix: str) -> Path:
    clean = (prefix or "run").strip() or "run"
    return BASE_RUN_DIR / f"{clean}_{_timestamp()}"


def _latest_run_dir() -> Path | None:
    if not BASE_RUN_DIR.exists():
        return None
    candidates = [p for p in BASE_RUN_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _set_run_dirs(run_dir: Path) -> None:
    global STATE_DIR, FIG_DIR, LOG_PATH
    STATE_DIR = run_dir
    FIG_DIR = run_dir / "figures"
    LOG_PATH = run_dir / "log.txt"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_log_into_widget(text_widget: tk.Text) -> None:
    text_widget.delete("1.0", tk.END)
    if LOG_PATH.exists():
        try:
            content = LOG_PATH.read_text(encoding="utf-8")
            if content:
                text_widget.insert(tk.END, content)
                text_widget.see(tk.END)
        except Exception:
            pass


def _rename_run_prefix(new_prefix: str, log) -> None:
    run_dir = STATE_DIR
    name = run_dir.name
    parts = name.split("_")
    stamp = parts[-1] if len(parts) >= 2 and len(parts[-1]) == 6 else _timestamp()
    if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
        stamp = f"{parts[-2]}_{parts[-1]}"
    clean = (new_prefix or "run").strip() or "run"
    target = run_dir.parent / f"{clean}_{stamp}"
    if target == run_dir:
        return
    run_dir.rename(target)
    _set_run_dirs(target)
    log(f"Renamed run folder to {target}")


def _start_new_run(prefix: str, log) -> None:
    run_dir = _new_run_dir(prefix)
    _set_run_dirs(run_dir)
    log(f"Started new run folder: {run_dir}")




def _node_count_from_lmax(lmax: int) -> int:
    return max(1, (int(lmax) + 1) ** 2)


def _mean_node_spacing_km(node_count: int, radius_m: float) -> float:
    node_count = max(1, int(node_count))
    area_per_node = (4.0 * math.pi * (float(radius_m) ** 2)) / node_count
    return math.sqrt(area_per_node) / 1000.0



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


def step1_build_grid_uniform_admittance(lmax: int, sigma_2d_max: float, log) -> Path:
    lmax = max(1, int(lmax))
    grid_cfg = GridConfig(nside=_node_count_from_lmax(lmax), lmax=lmax, radius_m=1.56e6, device="cpu")
    grid = build_roundtrip_grid(lmax=lmax, radius_m=grid_cfg.radius_m, device=grid_cfg.device)

    positions = grid["positions"].to(torch.float64)
    weights = grid["areas"].to(torch.float64)
    sigma_2d_max = max(0.0, float(sigma_2d_max))
    omega = 2.0 * math.pi / (9.925 * 3600.0)
    cond_real = torch.full((positions.shape[0],), float(sigma_2d_max), dtype=torch.float64)
    cond = _complex_sheet_admittance(cond_real, omega, grid_cfg.radius_m)
    Y_s = sh_forward(cond, positions, lmax=grid_cfg.lmax, weights=weights)
    state = {
        "grid_cfg": grid_cfg,
        "positions": positions,
        "normals": grid["normals"],
        "areas": weights,
        "neighbors": None,
        "faces": grid["faces"],
        "node_count": int(grid["n_points"]),
        "face_count": int(grid["n_faces"]),
        "admittance_uniform": cond[0].clone().detach(),
        "admittance_spectral": Y_s,
        "admittance_grid": cond,
        "sigma_2d_max": sigma_2d_max,
        "omega": omega,
    }
    path = _save_state("grid_admittance.pt", state)
    log(f"Step 1 complete (lmax={lmax}, nodes={grid['n_points']}, faces={grid['n_faces']}). Saved grid to {path}")
    return path, int(grid["n_points"]), int(grid["n_faces"])


def step1b_plot_roundtrip(log, plotter: str) -> None:
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
        faces=state["faces"],
        plotter=plotter,
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


def step1c_plot_roundtrip_stability(log, plotter: str) -> None:
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
        faces=state["faces"],
        plotter=plotter,
    )
    log("Step 1c complete. Displayed roundtrip stability scatter plots.")


def _plot_roundtrip_sphere_maps(
    values_a: np.ndarray,
    label_a: str,
    values_b: np.ndarray,
    label_b: str,
    positions: torch.Tensor,
    faces: torch.Tensor,
    plotter: str,
) -> None:
    pts = positions.detach().cpu().numpy()
    fcs = faces.detach().cpu().numpy()

    vals_a = np.asarray(values_a).reshape(-1)
    vals_b = np.asarray(values_b).reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (f"{label_a} real(Y_s)", vals_a.real),
        (f"{label_a} imag(Y_s)", vals_a.imag),
        (f"{label_b} real(Y_s)", vals_b.real),
        (f"{label_b} imag(Y_s)", vals_b.imag),
    ]
    for ax, (title, vals) in zip(axes.reshape(-1), panels):
        img = sphere_image(values=vals, positions=pts, faces=fcs, title=title, plotter=plotter, cmap="coolwarm", symmetric=True)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


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
    # Reuse prebuilt tensors instead of rebuilding the grid.
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
    _save_state("solution_latest.pt", payload)
    log(f"Step 3 complete. Saved uniform first-order solution to {path}")
    return path


def step3_uniform_self_consistent(log) -> Path:
    state = _load_state("ambient.pt")
    sim_out = _build_phasor_base(state)
    log("Uniform self-consistent solve...")
    sim_out = solve_uniform_self_consistent_sim(sim_out)
    payload = {"label": "uniform_self_consistent", "phasor_sim": sim_out}
    path = _save_state("solution_uniform_self_consistent.pt", payload)
    _save_state("solution_latest.pt", payload)
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
    _save_state("solution_latest.pt", payload)
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
    _save_state("solution_latest.pt", payload)
    log(f"Step 5 complete. Saved spectral self-consistent solution to {path}")
    return path


def _load_solution(label: str):
    return _load_state(f"solution_{label}.pt")


def step4_render_overview(label: str, log, plotter: str) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"uniform_{label}_overview.png"
    log(f"Overview: label={label}, lmax={sim_out.lmax}, plotter={plotter}")
    log("Step 4 overview: assembling input state for renderer...")
    t0 = time.perf_counter()
    render_demo_overview(
        data_path=_save_state("overview_input.pt", payload),  # save tmp input for renderer
        save_path=str(out_path),
        show=False,
        grid_state_path=str(STATE_DIR / "grid_admittance.pt"),
        plotter=plotter,
    )
    dt = time.perf_counter() - t0
    log(f"Step 4 overview: rendered in {dt:.1f}s -> {out_path}")
    return out_path


def step4_render_gradient(label: str, altitude_m: float, log, plotter: str) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    title = f"RSS |grad_B_emit| at alt={altitude_m/1000:.0f} km"
    save_path = FIG_DIR / f"uniform_grad_{int(altitude_m):d}m_{label}.png"
    render_gradient_map(sim_out, altitude_m=altitude_m, save_path=str(save_path), title=title, faces=grid_state["faces"], plotter=plotter)
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
    _save_state("solution_latest.pt", payload)
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

    state_files = {
        "grid_admittance": "grid_admittance.pt",
        "ambient": "ambient.pt",
        "solution_uniform_first_order": "solution_uniform_first_order.pt",
        "solution_uniform_self_consistent": "solution_uniform_self_consistent.pt",
        "solution_spectral_first_order": "solution_spectral_first_order.pt",
        "solution_spectral_self_consistent": "solution_spectral_self_consistent.pt",
        "solution_spectral_iterative": "solution_spectral_iterative.pt",
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

    ttk.Label(frm, text="Step 0: Run folder").grid(row=0, column=1, sticky="e")
    prefix_var = tk.StringVar(value="run")
    ttk.Label(frm, text="prefix").grid(row=0, column=2, sticky="e")
    ttk.Entry(frm, textvariable=prefix_var, width=10).grid(row=0, column=3, sticky="w")
    run_dir = _latest_run_dir()
    if run_dir is None:
        run_dir = _new_run_dir(prefix_var.get())
    _set_run_dirs(run_dir)
    run_dir_var = tk.StringVar(value=str(STATE_DIR))
    ttk.Entry(frm, textvariable=run_dir_var, width=42, state="readonly").grid(
        row=0, column=4, columnspan=2, sticky="we", padx=4
    )
    _load_log_into_widget(log_widget)

    def _load_run_folder_dialog() -> None:
        selection = filedialog.askdirectory(initialdir=str(BASE_RUN_DIR), title="Select run folder")
        if selection:
            _set_run_dirs(Path(selection))
            run_dir_var.set(str(STATE_DIR))
            _refresh_inputs_from_loaded_state()
            _load_log_into_widget(log_widget)

    btn_step0_load = tk.Button(
        frm,
        text="Load run folder",
        command=lambda: run_step_ui(btn_step0_load, _load_run_folder_dialog),
    )
    btn_step0_load.grid(row=0, column=6, padx=4, pady=(0, 6), sticky="w")

    btn_step0_rename = tk.Button(
        frm,
        text="Rename prefix",
        command=lambda: run_step_ui(
            btn_step0_rename,
            lambda: (
                _rename_run_prefix(prefix_var.get(), lambda msg: _log(log_widget, msg)),
                run_dir_var.set(str(STATE_DIR)),
            ),
        ),
    )
    btn_step0_rename.grid(row=0, column=7, padx=4, pady=(0, 6), sticky="w")

    def _set_button_state(btn: tk.Button, enabled: bool, completed: bool = False) -> None:
        if enabled:
            btn.config(state=tk.NORMAL, bg="pale green" if completed else "SystemButtonFace")
        else:
            btn.config(state=tk.DISABLED, bg="light gray")

    def _update_grid_counts() -> None:
        try:
            lmax = max(1, int(lmax_var.get()))
            nodes = _node_count_from_lmax(lmax)
            node_count_var.set(str(nodes))
            face_count_var.set(str(max(4, nodes * 2 - 4)))
            spacing_var.set(f"{_mean_node_spacing_km(nodes, 1.56e6):.1f}")
            sh_count_var.set(str((lmax + 1) ** 2))
        except Exception:
            pass

    def _update_button_states() -> None:
        grid_ok = _grid_exists()
        ambient_ok = _ambient_exists()
        uniform_first_ok = _solution_exists("uniform_first_order")
        uniform_self_ok = _solution_exists("uniform_self_consistent")
        spectral_first_ok = _solution_exists("spectral_first_order")
        spectral_self_ok = _solution_exists("spectral_self_consistent")
        spectral_iter_ok = _solution_exists("spectral_iterative")
        _set_button_state(btn_step1, True, completed=grid_ok)
        _set_button_state(btn_step1b, grid_ok, completed=grid_ok)
        _set_button_state(btn_step1b_power, grid_ok, completed=grid_ok)
        _set_button_state(btn_step1c, grid_ok, completed=grid_ok)
        _set_button_state(btn_step2, grid_ok, completed=ambient_ok)
        _set_button_state(btn_step3_uniform_first, ambient_ok, completed=uniform_first_ok)
        _set_button_state(btn_step3_uniform_self, ambient_ok, completed=uniform_self_ok)
        _set_button_state(btn_step4_spectral_first, ambient_ok, completed=spectral_first_ok)
        _set_button_state(btn_step5_spectral_self, ambient_ok, completed=spectral_self_ok)
        _set_button_state(btn_step6_spectral_iter, ambient_ok, completed=spectral_iter_ok)
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
    ttk.Label(frm, text="lmax").grid(row=1, column=1, sticky="e")
    lmax_var = tk.StringVar(value="36")
    ttk.Entry(frm, textvariable=lmax_var, width=6).grid(row=1, column=2, sticky="w")

    ttk.Label(frm, text="# nodes").grid(row=1, column=3, sticky="e")
    node_count_var = tk.StringVar(value=str(_node_count_from_lmax(int(lmax_var.get()))))
    ttk.Label(frm, textvariable=node_count_var).grid(row=1, column=4, sticky="w")

    ttk.Label(frm, text="# faces").grid(row=1, column=5, sticky="e")
    face_count_var = tk.StringVar(value=str(max(4, int(node_count_var.get()) * 2 - 4)))
    ttk.Label(frm, textvariable=face_count_var).grid(row=1, column=6, sticky="w")

    ttk.Label(frm, text="mean node spacing (km)").grid(row=1, column=7, sticky="e")
    spacing_var = tk.StringVar(value=f"{_mean_node_spacing_km(int(node_count_var.get()), 1.56e6):.1f}")
    ttk.Label(frm, textvariable=spacing_var).grid(row=1, column=8, sticky="w")

    sh_count_var = tk.StringVar(value=str((int(lmax_var.get()) + 1) ** 2))
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

    ttk.Label(frm, text="Sphere plotter").grid(row=2, column=8, sticky="e")
    plotter_var = tk.StringVar(value="matplotlib")
    tk.Radiobutton(frm, text="PyVista", variable=plotter_var, value="pyvista").grid(row=2, column=9, sticky="w")
    tk.Radiobutton(frm, text="Matplotlib", variable=plotter_var, value="matplotlib").grid(row=2, column=10, sticky="w")

    btn_step1 = tk.Button(
        frm,
        text="Run Step 1",
        command=lambda: run_step(
            btn_step1,
            lambda: (
                _start_new_run(prefix_var.get(), lambda msg: _log(log_widget, msg)),
                step1_build_grid_uniform_admittance(
                    int(lmax_var.get()),
                    float(sigma_2d_var.get()),
                    lambda msg: _log(log_widget, msg),
                ),
            )[1],
            on_success=lambda res: (
                node_count_var.set(str(res[1]) if isinstance(res, tuple) and len(res) > 1 else "?"),
                face_count_var.set(str(res[2]) if isinstance(res, tuple) and len(res) > 2 else "?"),
                _update_grid_counts(),
                run_dir_var.set(str(STATE_DIR)),
                _load_log_into_widget(log_widget),
            ),
        ),
    )
    btn_step1.grid(row=1, column=11, padx=6, sticky="w")
    def _refresh_inputs_from_loaded_state() -> None:
        if not _grid_exists():
            return
        try:
            state = _load_state("grid_admittance.pt")
            lmax = int(getattr(state.get("grid_cfg"), "lmax", lmax_var.get()))
            lmax_var.set(str(lmax))
            node_count_var.set(str(int(state.get("node_count", _node_count_from_lmax(lmax)))))
            face_count_var.set(str(int(state.get("face_count", max(4, int(node_count_var.get()) * 2 - 4)))))
            _update_grid_counts()
            if "sigma_2d_max" in state:
                sigma_2d_var.set(f"{float(state['sigma_2d_max']):.3e}")
            _log(log_widget, "Step 0: refreshed GUI inputs from loaded state.")
        except Exception as exc:  # noqa: BLE001
            _log(log_widget, f"Step 0: unable to refresh GUI inputs ({exc})")

    # Step 1b
    ttk.Label(frm, text="Step 1b: Roundtrip check").grid(row=3, column=0, sticky="w")
    btn_step1b = tk.Button(
        frm,
        text="Plot admittance roundtrip",
        command=lambda: run_step_ui(btn_step1b, lambda: step1b_plot_roundtrip(lambda msg: _log(log_widget, msg), plotter_var.get())),
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
        command=lambda: run_step_ui(btn_step1c, lambda: step1c_plot_roundtrip_stability(lambda msg: _log(log_widget, msg), plotter_var.get())),
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
            lambda: step4_render_overview("uniform_first_order", lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("uniform_first_order", 0.0, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("uniform_first_order", 100e3, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_overview("uniform_self_consistent", lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("uniform_self_consistent", 0.0, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("uniform_self_consistent", 100e3, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_overview("spectral_first_order", lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("spectral_first_order", 0.0, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("spectral_first_order", 100e3, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_overview("spectral_self_consistent", lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("spectral_self_consistent", 0.0, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("spectral_self_consistent", 100e3, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_overview("spectral_iterative", lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("spectral_iterative", 0.0, lambda msg: _log(log_widget, msg), plotter_var.get()),
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
            lambda: step4_render_gradient("spectral_iterative", 100e3, lambda msg: _log(log_widget, msg), plotter_var.get()),
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

    lmax_var.trace_add("write", lambda *_: _update_grid_counts())
    _update_grid_counts()
    _update_button_states()

    root.mainloop()


if __name__ == "__main__":
    main()
