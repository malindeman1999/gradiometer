"""
GUI for running the non-uniform demo pipeline in staged steps:
1) Run folder management
2) Build grid + admittance (selectable conductivity model)
3) Admittance checks/plots
4) Build ambient field
5) Solve + renders/diagnostics

Each step saves its state so runs can be resumed later.
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
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from workflow.ambient_field.ambient_driver import (
    build_ambient_driver_x,
    build_ambient_driver_y,
    build_ambient_driver_z,
)
from europa_model.config import GridConfig, ModelConfig
from europa_model.transforms import sh_forward, sh_inverse
from europa_model.solvers import _flatten_lm, _unflatten_lm, toroidal_e_from_radial_b, _build_self_field_diag
from europa_model.solver_variants.solver_variant_precomputed import (
    solve_spectral_self_consistent_sim_precomputed,
    _build_mixing_matrix_precomputed_sparse,
)
from europa_model import inductance
from europa_model.gradient_utils import (
    render_gradient_map,
    render_b_magnitude_map,
    rss_gradient_from_emit,
    toroidal_field_spherical,
)
from europa_model.observation import evaluate_field_from_spectral
from workflow.plotting.render_demo_overview import render_demo_overview
from workflow.plotting.sphere_roundtrip import build_roundtrip_grid, sphere_image
from gaunt.assemble_gaunt_checkpoints import assemble_in_memory
from workflow.data_objects.phasor_data import PhasorSimulation
from workflow.conductivity_models import EuropaSnapshotConfig, build_europa_snapshot_conductivity

BASE_RUN_DIR = Path("workflow/artifacts/nonuniform_workflow")
STATE_DIR = BASE_RUN_DIR
FIG_DIR = BASE_RUN_DIR / "figures"
LOG_PATH = STATE_DIR / "log.txt"
GAUNT_CACHE = Path("gaunt/data/gaunt_cache_wigxjpf")
STEP6_CUBE_HALF_M = 2.0e6  # Display/sample cube extent: +/-2 Mm


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
    if not run_dir.exists():
        log("Rename prefix skipped: run folder does not exist yet.")
        return
    name = run_dir.name
    parts = name.split("_")
    stamp = parts[-1] if len(parts) >= 2 and len(parts[-1]) == 6 else _timestamp()
    if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
        stamp = f"{parts[-2]}_{parts[-1]}"
    clean = (new_prefix or "run").strip() or "run"
    target = run_dir.parent / f"{clean}_{stamp}"
    if target == run_dir:
        log("Rename prefix skipped: new prefix matches current folder.")
        return
    try:
        run_dir.rename(target)
    except Exception as exc:  # noqa: BLE001
        log(f"Rename prefix failed: {exc}")
        return
    _set_run_dirs(target)
    log(f"Renamed run folder to {target}")


def _start_new_run(prefix: str, log) -> None:
    run_dir = _new_run_dir(prefix)
    _set_run_dirs(run_dir)
    log(f"Started new run folder: {run_dir}")


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
    """Build a real conductivity field with target RMS and single (l,Â±m) modes."""
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


def _project_and_verify_real_field_roundtrip(
    field_real: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    lmax: int,
    log,
    *,
    label: str,
    rel_l2_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Project real grid field to SH and verify inverse round-trip fidelity."""
    field = field_real.to(torch.float64)
    coeffs = sh_forward(field, positions, lmax=lmax, weights=weights)
    recon = sh_inverse(coeffs, positions, weights)
    imag_max = float(recon.imag.abs().max().item())
    recon_real = recon.real

    w = weights.to(torch.float64)
    wsum = float(w.sum().item())
    diff = recon_real - field
    num = float(torch.sum(w * diff * diff).item())
    den = float(torch.sum(w * field * field).item())
    rel_l2 = math.sqrt(num / max(den, 1e-30))
    max_abs = float(torch.max(torch.abs(diff)).item())
    ref_max = float(torch.max(torch.abs(field)).item())
    rel_max = max_abs / max(ref_max, 1e-30)

    stats = {
        "rel_l2": rel_l2,
        "rel_max": rel_max,
        "imag_max": imag_max,
        "weight_sum": wsum,
    }
    log(
        f"{label} round-trip: rel_l2={rel_l2:.3e}, rel_max={rel_max:.3e}, "
        f"max|imag(recon)|={imag_max:.3e}"
    )
    if rel_l2 > float(rel_l2_tol):
        raise RuntimeError(
            f"{label} SH round-trip rel_l2={rel_l2:.3e} exceeds tolerance {float(rel_l2_tol):.3e}. "
            "Increase lmax or smooth/broaden spatial structure."
        )
    return coeffs, recon_real, stats


def _node_count_from_lmax(lmax: int) -> int:
    return max(1, (int(lmax) + 1) ** 2)


def _mean_node_spacing_km(node_count: int, radius_m: float) -> float:
    node_count = max(1, int(node_count))
    area_per_node = (4.0 * math.pi * (float(radius_m) ** 2)) / node_count
    spacing = math.sqrt(area_per_node)
    return spacing / 1000.0


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


def _clear_solution_states(log) -> None:
    stale = [
        "solution_first_order.pt",
        "solution_self_consistent.pt",
        "solution_iterative.pt",
        "solution_latest.pt",
        "overview_input.pt",
    ]
    removed = 0
    for name in stale:
        p = STATE_DIR / name
        if p.exists():
            p.unlink()
            removed += 1
    if removed > 0:
        log(f"Step 4: cleared {removed} stale Step 5 state file(s); solve must be rerun.")
    _clear_overview_cache(log, reason="upstream Step 4 changes")
    _clear_step4_field_cache(log, reason="upstream Step 4 changes")
    _clear_step6_field_cache(log, reason="upstream Step 4 changes")


def _clear_overview_cache(log, reason: str | None = None, label: str | None = None) -> None:
    pattern = "overview_cache_*.pt" if label is None else f"overview_cache_{label}.pt"
    removed = 0
    for p in STATE_DIR.glob(pattern):
        if p.is_file():
            p.unlink()
            removed += 1
    if removed > 0:
        suffix = f" ({reason})" if reason else ""
        log(f"Cleared {removed} overview cache file(s){suffix}.")


def _clear_step4_field_cache(log, reason: str | None = None, label: str | None = None) -> None:
    pattern = "step4_field_cache_*.pt" if label is None else f"step4_field_cache_{label}_*.pt"
    removed = 0
    for p in STATE_DIR.glob(pattern):
        if p.is_file():
            p.unlink()
            removed += 1
    if removed > 0:
        suffix = f" ({reason})" if reason else ""
        log(f"Cleared {removed} Step 4 field cache file(s){suffix}.")


def _clear_step6_field_cache(log, reason: str | None = None, label: str | None = None) -> None:
    patterns = (
        ["step6_field_phasors_*.pt", "step6_gradient_cache_*.pt"]
        if label is None
        else [f"step6_field_phasors_{label}_*.pt", f"step6_gradient_cache_{label}_*.pt"]
    )
    removed = 0
    for pattern in patterns:
        for p in STATE_DIR.glob(pattern):
            if p.is_file():
                p.unlink()
                removed += 1
    if removed > 0:
        suffix = f" ({reason})" if reason else ""
        log(f"Cleared {removed} Step 6 cache file(s){suffix}.")


def _state_file_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return int(st.st_mtime_ns), int(st.st_size)


def _step4_dependency_signature(label: str) -> dict[str, tuple[int, int]]:
    deps = {
        "grid_admittance.pt": _state_file_signature(STATE_DIR / "grid_admittance.pt"),
        "ambient.pt": _state_file_signature(STATE_DIR / "ambient.pt"),
        f"solution_{label}.pt": _state_file_signature(STATE_DIR / f"solution_{label}.pt"),
    }
    return deps


def _step4_gradient_cache_name(label: str, altitude_m: float, fd_scheme: str) -> str:
    alt_key = int(round(float(altitude_m)))
    fd_key = str(fd_scheme or "forward").strip().lower()
    return f"step4_field_cache_{label}_gradient_{alt_key}m_{fd_key}.pt"


def _step4_bmag_cache_name(label: str, altitude_m: float) -> str:
    alt_key = int(round(float(altitude_m)))
    return f"step4_field_cache_{label}_bmag_{alt_key}m.pt"


def _step6_field_cache_name(label: str, field_mode: str, n_edge: int) -> str:
    mode = str(field_mode).strip().lower()
    return f"step6_field_phasors_{label}_{mode}_{int(n_edge)}edge.pt"


def _step6_gradient_cache_name(label: str, field_mode: str, altitude_m: float, fd_scheme: str) -> str:
    mode = str(field_mode).strip().lower()
    alt_key = int(round(float(altitude_m)))
    fd_key = str(fd_scheme or "forward").strip().lower()
    return f"step6_gradient_cache_{label}_{mode}_{alt_key}m_{fd_key}.pt"


def _latest_step4_gradient_altitude_m(label: str, fd_scheme: str) -> float:
    fd_key = str(fd_scheme or "forward").strip().lower()
    pattern = f"step4_field_cache_{label}_gradient_*m_{fd_key}.pt"
    latest_path = None
    latest_mtime = -1.0
    for p in STATE_DIR.glob(pattern):
        if p.is_file():
            mt = p.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
                latest_path = p
    if latest_path is None:
        return 100e3
    stem = latest_path.stem
    token = "_gradient_"
    i0 = stem.find(token)
    if i0 < 0:
        return 100e3
    tail = stem[i0 + len(token):]
    i1 = tail.find("m_")
    if i1 < 0:
        return 100e3
    try:
        return float(int(tail[:i1]))
    except Exception:
        return 100e3


def _load_latest_step4_gradient_cache(label: str, fd_scheme: str) -> dict:
    fd_key = str(fd_scheme or "forward").strip().lower()
    pattern = f"step4_field_cache_{label}_gradient_*m_{fd_key}.pt"
    latest_path = None
    latest_mtime = -1.0
    for p in STATE_DIR.glob(pattern):
        if p.is_file():
            mt = p.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
                latest_path = p
    if latest_path is None:
        raise RuntimeError(
            f"No cached gradient data found for label={label}, fd={fd_key}. "
            "Run a Step 5 gradient plot first."
        )
    data = _load_state(latest_path.name)
    if not isinstance(data, dict) or "positions" not in data or "rss" not in data:
        raise RuntimeError(f"Gradient cache format invalid: {latest_path}")
    return data


def _complex_sheet_admittance(
    sigma_s: torch.Tensor,
    omega: float,
    radius_m: float,
) -> torch.Tensor:
    """Compute complex sheet admittance from thin-shell impedance model."""
    sigma_s = sigma_s.to(torch.float64)
    X_s = 0.0
    R_s = torch.where(sigma_s > 0, 1.0 / sigma_s, torch.zeros_like(sigma_s))
    Z = R_s + 1j * X_s
    Y = torch.where(sigma_s > 0, 1.0 / Z, torch.zeros_like(Z))
    return Y.to(torch.complex128)


def _component_sigma_map_from_x(
    sigma0: float,
    x_component: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Convert one log-space component into a conductivity map with mean sigma0."""
    sigma = float(sigma0) * torch.exp(x_component.to(torch.float64))
    w = weights.to(torch.float64)
    wsum = float(w.sum().item())
    mean_now = float((w * sigma).sum().item() / max(wsum, 1e-30))
    if mean_now > 0.0:
        sigma = sigma * (float(sigma0) / mean_now)
    return sigma.to(torch.float64)


def step1_build_grid_admittance(
    lmax: int,
    mean_cond: float,
    frac_rms: float,
    mode_l: int,
    mode_m: int,
    conductivity_model: str,
    log,
) -> Path:
    _clear_overview_cache(log, reason="Step 2 grid/admittance rebuild")
    _clear_step4_field_cache(log, reason="Step 2 grid/admittance rebuild")
    _clear_step6_field_cache(log, reason="Step 2 grid/admittance rebuild")
    lmax = max(1, int(lmax))
    grid_cfg = GridConfig(nside=_node_count_from_lmax(lmax), lmax=lmax, radius_m=1.56e6, device="cpu")
    grid = build_roundtrip_grid(lmax=lmax, radius_m=grid_cfg.radius_m, device=grid_cfg.device)

    positions = grid["positions"].to(torch.float64)
    weights = grid["areas"].to(torch.float64)
    mean_val = max(0.0, float(mean_cond))
    frac_rms = max(0.0, float(frac_rms))
    mode_l = int(mode_l)
    mode_m = int(mode_m)
    model_key = str(conductivity_model or "europa_snapshot").strip().lower()
    if model_key not in {"uniform", "synthetic_sh", "europa_snapshot"}:
        raise RuntimeError(f"Unknown conductivity_model: {conductivity_model}")
    log(f"Conductivity model: {model_key}")

    sigma_coeffs_target = None
    model_components: dict[str, torch.Tensor | float | int] = {}
    if model_key == "uniform":
        cond_real = torch.full_like(weights, fill_value=mean_val, dtype=torch.float64)
        roundtrip_tol = 1e-10
    elif model_key == "synthetic_sh":
        cond_real, sigma_coeffs_target = _synthesize_sigma_field(
            positions,
            weights,
            grid_cfg.lmax,
            mean_val,
            frac_rms,
            mode_l,
            mode_m,
            log,
        )
        roundtrip_tol = 5e-6
    else:
        cfg = EuropaSnapshotConfig(seed=7)
        _cond_model, model_components = build_europa_snapshot_conductivity(
            positions=positions,
            weights=weights,
            sigma0=mean_val,
            cfg=cfg,
        )
        x_conv = model_components.get("x_chem")
        x_exchange = model_components.get("x_exchange")
        x_flow = model_components.get("x_flow")
        x_bg = model_components.get("x_bg")
        if not all(isinstance(xi, torch.Tensor) for xi in (x_conv, x_exchange, x_flow, x_bg)):
            raise RuntimeError("Europa snapshot components missing (x_chem/x_exchange/x_flow/x_bg).")
        sigma_conv = _component_sigma_map_from_x(mean_val, x_conv, weights)
        sigma_exchange = _component_sigma_map_from_x(mean_val, x_exchange, weights)
        sigma_flow = _component_sigma_map_from_x(mean_val, x_flow, weights)
        sigma_bg = _component_sigma_map_from_x(mean_val, x_bg, weights)
        cond_real = 0.25 * (sigma_conv + sigma_exchange + sigma_flow + sigma_bg)
        # Enforce that the final Europa snapshot map mean matches the GUI target.
        mean_now = float(cond_real.mean().item())
        if mean_now > 0.0 and mean_val >= 0.0:
            scale_mean = mean_val / mean_now
            cond_real = cond_real * scale_mean
            sigma_conv = sigma_conv * scale_mean
            sigma_exchange = sigma_exchange * scale_mean
            sigma_flow = sigma_flow * scale_mean
            sigma_bg = sigma_bg * scale_mean
            log(
                f"Europa snapshot mean normalization: applied scale={scale_mean:.6g} "
                f"to match target mean={mean_val:.3e}"
            )
        model_components["sigma_conv_only"] = sigma_conv
        model_components["sigma_exchange_only"] = sigma_exchange
        model_components["sigma_flow_only"] = sigma_flow
        model_components["sigma_bg_only"] = sigma_bg
        model_components["sigma_combined_rule"] = "avg_components"
        log("Europa snapshot combine rule: sigma = average of component-only maps")
        roundtrip_tol = 3e-2
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
    if model_key == "synthetic_sh":
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
    sigma_proj, sigma_recon, sigma_rt = _project_and_verify_real_field_roundtrip(
        cond_real,
        positions,
        weights,
        grid_cfg.lmax,
        log,
        label="conductivity",
        rel_l2_tol=roundtrip_tol,
    )

    omega = 2.0 * math.pi / (9.925 * 3600.0)
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
        "admittance_spectral": Y_s,
        "admittance_grid": cond,
        "sigma_spectral": sigma_proj,
        "sigma_spectral_target": sigma_coeffs_target,
        "sigma_roundtrip_recon_grid": sigma_recon,
        "sigma_roundtrip_rel_l2": float(sigma_rt["rel_l2"]),
        "sigma_roundtrip_rel_max": float(sigma_rt["rel_max"]),
        "sigma_roundtrip_imag_max": float(sigma_rt["imag_max"]),
        "sigma_grid": cond_real,
        "conductivity_model": model_key,
        "sigma_mean": mean_val,
        "sigma_frac_rms": frac_rms,
        "sigma_mode_l": int(mode_l),
        "sigma_mode_m": int(mode_m),
    }
    if model_components:
        state.update(model_components)
    path = _save_state("grid_admittance.pt", state)
    log(
        f"Step 2 complete (lmax={lmax}, nodes={grid['n_points']}, faces={grid['n_faces']}). "
        f"Saved grid+admittance to {path}"
    )
    return path, int(grid["n_points"]), int(grid["n_faces"])


def step1b_plot_roundtrip(log, plotter: str) -> None:
    state = _load_state("grid_admittance.pt")
    positions = state["positions"].to(torch.float64)
    weights = state["areas"].to(torch.float64)
    coeffs = state["admittance_spectral"]
    recon = sh_inverse(coeffs, positions, weights)
    recon = recon.reshape(-1).cpu().numpy()

    sigma_grid = state.get("sigma_grid")
    if sigma_grid is not None:
        sigma_vals = sigma_grid.to(torch.float64).reshape(-1).cpu().numpy()
        _plot_admittance_and_conductivity_spheres(
            sigma_vals,
            recon,
            positions=positions,
            faces=state["faces"],
            plotter=plotter,
        )
    log("Step 3 complete. Displayed conductivity/admittance sphere plots.")


def step1b_plot_admittance_power(log) -> None:
    state = _load_state("grid_admittance.pt")
    coeffs = state.get("admittance_spectral")
    sigma_coeffs = state.get("sigma_spectral")
    if coeffs is None or sigma_coeffs is None:
        raise RuntimeError("Missing admittance_spectral or sigma_spectral. Run Step 2 before plotting magnitudes.")
    model_key = str(state.get("conductivity_model", "synthetic_sh"))
    if model_key == "synthetic_sh":
        mode_l = state.get("sigma_mode_l", None)
        mode_m = state.get("sigma_mode_m", None)
        frac_rms = state.get("sigma_frac_rms", None)
        mode_l_str = f"{int(mode_l)}" if mode_l is not None else "?"
        mode_m_str = f"{int(mode_m)}" if mode_m is not None else "?"
        frac_rms_str = f"{float(frac_rms):.2%}" if frac_rms is not None else "?"
        title_suffix = f"(model=synthetic_sh, l={mode_l_str}, m=+/-{mode_m_str}, frac RMS {frac_rms_str})"
    elif model_key == "europa_snapshot":
        n_sites = int(state.get("snapshot_n_exchange_sites", 0))
        seed = int(state.get("snapshot_seed", -1))
        width = float(state.get("snapshot_exchange_width_deg", 0.0))
        title_suffix = f"(model=europa_snapshot, sites={n_sites}, width={width:.1f} deg, seed={seed})"
    else:
        title_suffix = f"(model={model_key})"

    l_b, _, mag = _flatten_harmonics(coeffs.to(torch.complex128))
    _, _, mag_sigma = _flatten_harmonics(sigma_coeffs.to(torch.complex128))

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
    axes2[0].set_yscale("log")
    axes2[0].set_ylabel("RSS |Ïƒ_s| (S)")
    axes2[0].set_title(f"Conductivity by degree l (RSS, S) {title_suffix}")
    axes2[0].grid(True, which="both", alpha=0.3)

    rss_y_plot = np.where(rss_y > 0.0, rss_y, np.nan)
    axes2[1].plot(l_vals, rss_y_plot, marker="o", linewidth=1.2, color="#ff9c43")
    axes2[1].set_yscale("log")
    axes2[1].set_xlabel("Spherical harmonic degree l")
    axes2[1].set_ylabel("RSS |Y_s| (S)")
    axes2[1].set_title(f"Admittance by degree l (RSS, S) {title_suffix}")
    axes2[1].grid(True, which="both", alpha=0.3)

    fig2.tight_layout()
    plt.show()
    log("Step 3 complete. Plotted conductivity/admittance power grouped by degree l.")


def _plot_admittance_and_conductivity_spheres(
    sigma_real: np.ndarray,
    admittance: np.ndarray,
    positions: torch.Tensor,
    faces: torch.Tensor,
    plotter: str,
) -> None:
    pts = positions.detach().cpu().numpy()
    fcs = faces.detach().cpu().numpy()
    sigma_vals = np.asarray(sigma_real).reshape(-1)
    adm_real = np.asarray(admittance).reshape(-1).real
    adm_imag = np.asarray(admittance).reshape(-1).imag
    sigma_vmin = float(np.min(sigma_vals))
    sigma_vmax = float(np.max(sigma_vals))
    panels = [
        ("Conductivity real(sigma_s)", sigma_vals, False, "viridis", sigma_vmin, sigma_vmax),
        # Force real(Y_s) to the same color scale as real conductivity for direct visual comparison.
        ("Admittance real(Y_s)", adm_real, False, "coolwarm", sigma_vmin, sigma_vmax),
        ("Admittance imag(Y_s)", adm_imag, True, "coolwarm", None, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, vals, sym, cmap, vmin, vmax) in zip(axes, panels):
        img = sphere_image(
            values=vals,
            positions=pts,
            faces=fcs,
            title=title,
            plotter=plotter,
            cmap=cmap,
            symmetric=sym,
            vmin=vmin,
            vmax=vmax,
        )
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def step2_build_ambient(
    direction_axis: str,
    amplitude_t: float,
    period_hours: float,
    log,
) -> Path:
    state1 = _load_state("grid_admittance.pt")
    grid_cfg: GridConfig = state1["grid_cfg"]
    axis = str(direction_axis or "x").strip().lower()
    if axis not in {"x", "y", "z"}:
        raise RuntimeError(f"Unknown ambient direction axis: {direction_axis}")
    amp_val = max(0.0, float(amplitude_t))
    period_val = float(period_hours)
    if period_val <= 0.0:
        raise RuntimeError(f"Ambient period must be > 0 hours (got {period_hours}).")
    builders = {
        "x": build_ambient_driver_x,
        "y": build_ambient_driver_y,
        "z": build_ambient_driver_z,
    }
    ambient_cfg, B_radial_spec, period_sec = builders[axis](
        grid_cfg,
        period_hours=period_val,
        amplitude_t=amp_val,
        phase_rad=0.0,
    )
    state1.update(
        {
            "ambient_cfg": ambient_cfg,
            "B_radial_spec": B_radial_spec,
            "period_sec": period_sec,
            "ambient_direction": axis,
            "ambient_amplitude_t": amp_val,
            "ambient_period_hours": period_val,
        }
    )
    path = _save_state("ambient.pt", state1)
    _clear_solution_states(log)
    log(
        f"Step 4 complete. Saved ambient + B_radial to {path} "
        f"(axis={axis.upper()}, amplitude={amp_val:.3e} T, period={period_val:.6g} h)"
    )
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
        admittance_spectral=state["admittance_spectral"],
        B_radial=state["B_radial_spec"],
        period_sec=state["period_sec"],
    )


def step3_solve_currents(first_order_only: bool, log) -> Path:
    label_for_clear = "first_order" if bool(first_order_only) else "self_consistent"
    _clear_overview_cache(log, reason="Step 5 solve rerun", label=label_for_clear)
    _clear_step4_field_cache(log, reason="Step 5 solve rerun", label=label_for_clear)
    _clear_step6_field_cache(log, reason="Step 5 solve rerun", label=label_for_clear)
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

    log("Building sparse mixing matrix (v_toroidal)...")
    mixing_matrix = _build_mixing_matrix_precomputed_sparse(
        grid_cfg.lmax,
        base.omega,
        base.radius_m,
        base.admittance_spectral,
        G_sparse,
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
        sim_out.solver_variant = "spectral_first_order_precomputed_v_toroidal_sparse"
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
        sim_out.solver_variant = "spectral_self_consistent_precomputed_v_toroidal_sparse"
        label = "self_consistent"
        log("Self-consistent solve complete.")

    payload = {
        "label": label,
        "phasor_sim": sim_out,
    }
    path = _save_state(f"solution_{label}.pt", payload)
    _save_state("solution_latest.pt", payload)
    log(f"Step 5 complete. Saved solution to {path}")
    return path


def _load_solution(label: str):
    return _load_state(f"solution_{label}.pt")


def step4_render_overview(label: str, log, plotter: str) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"nonuniform_{label}_overview.png"
    log(f"Step 5 overview: label={label}, lmax={sim_out.lmax}, plotter={plotter}")
    log("Step 5 overview: assembling input state for renderer...")
    t0 = time.perf_counter()
    render_demo_overview(
        data_path=_save_state("overview_input.pt", payload),  # save tmp input for renderer
        save_path=str(out_path),
        show=True,
        grid_state_path=str(STATE_DIR / "grid_admittance.pt"),
        plotter=plotter,
        cache_path=str(STATE_DIR / f"overview_cache_{label}.pt"),
        cache_deps=_step4_dependency_signature(label),
    )
    dt = time.perf_counter() - t0
    log(f"Step 5 overview: rendered in {dt:.1f}s -> {out_path}")
    return out_path


def step4_render_gradient(label: str, altitude_m: float, log, plotter: str, gradient_fd_scheme: str) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    title = f"RSS |grad_B_emit| at alt={altitude_m/1000:.0f} km"
    save_path = FIG_DIR / f"nonuniform_grad_{int(altitude_m):d}m_{label}.png"
    cache_name = _step4_gradient_cache_name(label, altitude_m, gradient_fd_scheme)
    cache_path = STATE_DIR / cache_name
    deps = _step4_dependency_signature(label)
    cache_data = None
    if cache_path.exists():
        try:
            candidate = _load_state(cache_name)
            if (
                isinstance(candidate, dict)
                and candidate.get("kind") == "gradient_rss"
                and candidate.get("deps") == deps
                and int(candidate.get("altitude_m", -1)) == int(round(float(altitude_m)))
                and str(candidate.get("fd_scheme", "")).strip().lower() == str(gradient_fd_scheme).strip().lower()
            ):
                cache_data = candidate
        except Exception:
            cache_data = None

    if cache_data is None:
        radius = float(sim_out.radius_m + altitude_m)
        scale = radius / float(sim_out.radius_m)
        positions = (sim_out.grid_positions * scale).to(dtype=torch.float64)
        t0 = time.perf_counter()
        rss = rss_gradient_from_emit(sim_out, positions, obs_radius=radius, fd_scheme=gradient_fd_scheme).to(torch.float64)
        dt = time.perf_counter() - t0
        cache_data = {
            "kind": "gradient_rss",
            "label": label,
            "altitude_m": int(round(float(altitude_m))),
            "fd_scheme": str(gradient_fd_scheme).strip().lower(),
            "deps": deps,
            "positions": positions.cpu(),
            "rss": rss.cpu(),
        }
        _save_state(cache_name, cache_data)
        log(f"Cached gradient field data to {cache_path} (compute {dt:.2f}s).")
    else:
        log(f"Loaded cached gradient field data from {cache_path}.")

    render_gradient_map(
        sim_out,
        altitude_m=altitude_m,
        save_path=str(save_path),
        title=title,
        faces=grid_state["faces"],
        plotter=plotter,
        timing_log=log,
        fd_scheme=gradient_fd_scheme,
        positions_override=cache_data["positions"],
        rss_values=cache_data["rss"],
    )
    log(f"Rendered gradient map to {save_path}")
    return save_path


def step4_render_gradient_log100(label: str, log, plotter: str, gradient_fd_scheme: str) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    altitude_m = 100e3
    title = "RSS |grad_B_emit| at alt=100 km (log scale)"
    save_path = FIG_DIR / f"nonuniform_grad_{int(altitude_m):d}m_{label}_log.png"
    cache_name = _step4_gradient_cache_name(label, altitude_m, gradient_fd_scheme)
    cache_path = STATE_DIR / cache_name
    deps = _step4_dependency_signature(label)
    cache_data = None
    if cache_path.exists():
        try:
            candidate = _load_state(cache_name)
            if (
                isinstance(candidate, dict)
                and candidate.get("kind") == "gradient_rss"
                and candidate.get("deps") == deps
                and int(candidate.get("altitude_m", -1)) == int(round(float(altitude_m)))
                and str(candidate.get("fd_scheme", "")).strip().lower() == str(gradient_fd_scheme).strip().lower()
            ):
                cache_data = candidate
        except Exception:
            cache_data = None

    if cache_data is None:
        radius = float(sim_out.radius_m + altitude_m)
        scale = radius / float(sim_out.radius_m)
        positions = (sim_out.grid_positions * scale).to(dtype=torch.float64)
        t0 = time.perf_counter()
        rss = rss_gradient_from_emit(sim_out, positions, obs_radius=radius, fd_scheme=gradient_fd_scheme).to(torch.float64)
        dt = time.perf_counter() - t0
        cache_data = {
            "kind": "gradient_rss",
            "label": label,
            "altitude_m": int(round(float(altitude_m))),
            "fd_scheme": str(gradient_fd_scheme).strip().lower(),
            "deps": deps,
            "positions": positions.cpu(),
            "rss": rss.cpu(),
        }
        _save_state(cache_name, cache_data)
        log(f"Cached gradient field data to {cache_path} (compute {dt:.2f}s).")
    else:
        log(f"Loaded cached gradient field data from {cache_path}.")

    render_gradient_map(
        sim_out,
        altitude_m=altitude_m,
        save_path=str(save_path),
        title=title,
        faces=grid_state["faces"],
        plotter=plotter,
        log_scale=True,
        timing_log=log,
        fd_scheme=gradient_fd_scheme,
        positions_override=cache_data["positions"],
        rss_values=cache_data["rss"],
    )
    log(f"Rendered log-scale gradient map to {save_path}")
    return save_path


def step4_render_bmag100(label: str, log, plotter: str) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    altitude_m = 100e3
    faces = grid_state["faces"]
    title = "RSS |B_emit| at alt=100 km"
    save_path = FIG_DIR / f"nonuniform_bmag_{int(altitude_m):d}m_{label}.png"
    cache_name = _step4_bmag_cache_name(label, altitude_m)
    cache_path = STATE_DIR / cache_name
    deps = _step4_dependency_signature(label)
    cache_data = None
    if cache_path.exists():
        try:
            candidate = _load_state(cache_name)
            if (
                isinstance(candidate, dict)
                and candidate.get("kind") == "bmag_rss"
                and candidate.get("deps") == deps
                and int(candidate.get("altitude_m", -1)) == int(round(float(altitude_m)))
            ):
                cache_data = candidate
        except Exception:
            cache_data = None

    if cache_data is None:
        obs_radius = float(sim_out.radius_m + altitude_m)
        scale = obs_radius / float(sim_out.radius_m)
        positions = (sim_out.grid_positions * scale).to(dtype=torch.float64)
        t0 = time.perf_counter()
        Br, Btheta, Bphi = toroidal_field_spherical(sim_out.K_toroidal, radius=float(sim_out.radius_m), positions=positions)
        rss = torch.sqrt(torch.abs(Br) ** 2 + torch.abs(Btheta) ** 2 + torch.abs(Bphi) ** 2).to(torch.float64)
        dt = time.perf_counter() - t0
        cache_data = {
            "kind": "bmag_rss",
            "label": label,
            "altitude_m": int(round(float(altitude_m))),
            "deps": deps,
            "positions": positions.cpu(),
            "rss": rss.cpu(),
        }
        _save_state(cache_name, cache_data)
        log(f"Cached B-field magnitude data to {cache_path} (compute {dt:.2f}s).")
    else:
        log(f"Loaded cached B-field magnitude data from {cache_path}.")

    render_b_magnitude_map(
        sim_out,
        altitude_m=altitude_m,
        save_path=str(save_path),
        title=title,
        faces=faces,
        plotter=plotter,
        positions_override=cache_data["positions"],
        rss_values=cache_data["rss"],
    )
    log(f"Rendered emitted-field magnitude map to {save_path}")
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
    grid_state = _load_state("grid_admittance.pt")
    sigma_coeffs = grid_state.get("sigma_spectral")
    if sigma_coeffs is None:
        raise RuntimeError("Missing sigma_spectral in grid_admittance state; run Step 2 before plotting harmonics.")

    l_b, m_b, mag_b = _flatten_harmonics(sim_out.B_radial)
    _, _, mag_emit = _flatten_harmonics(sim_out.B_rad_emit)
    _, _, mag_sigma = _flatten_harmonics(sigma_coeffs.to(torch.complex128))
    lmax = int(l_b.max()) if l_b.size else 0
    l_vals = np.arange(lmax + 1)
    rss_b = np.zeros(lmax + 1, dtype=np.float64)
    rss_emit = np.zeros(lmax + 1, dtype=np.float64)
    rss_sigma = np.zeros(lmax + 1, dtype=np.float64)
    for l in range(lmax + 1):
        mask = l_b == l
        rss_b[l] = float(np.sqrt(np.sum(mag_b[mask] ** 2)))
        rss_emit[l] = float(np.sqrt(np.sum(mag_emit[mask] ** 2)))
        rss_sigma[l] = float(np.sqrt(np.sum(mag_sigma[mask] ** 2)))

    peak_b = float(max(np.max(rss_b), np.max(rss_emit), 1e-30))
    eps_b = peak_b * 1e-9
    rss_b_plot = np.maximum(rss_b, eps_b)
    rss_emit_plot = np.maximum(rss_emit, eps_b)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2))
    ax_top, ax_bottom = axes
    ax_top.plot(l_vals, rss_b_plot, marker="o", linewidth=2.8, color="#1f77b4", label="ambient RSS |B_rad|")
    ax_top.plot(l_vals, rss_emit_plot, marker="o", linewidth=1.2, color="#d62728", label="emitted RSS |B_rad_emit|")
    ax_top.set_yscale("log")
    ax_top.set_xlabel("Spherical harmonic degree l")
    ax_top.set_ylabel("RSS |B| [T]")
    ax_top.set_title(f"Magnetic harmonics by degree l (RSS) [{label}]")
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend(frameon=False)

    peak_sigma = float(max(np.max(rss_sigma), 1e-30))
    eps_sigma = peak_sigma * 1e-9
    rss_sigma_plot = np.maximum(rss_sigma, eps_sigma)

    ax_bottom.plot(l_vals, rss_sigma_plot, marker="o", linewidth=1.2, color="#5c9bd5", label="RSS |sigma_spectral|")
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("Spherical harmonic degree l")
    ax_bottom.set_ylabel("RSS |sigma| [S]")
    ax_bottom.set_title("Conductivity harmonic magnitude (RSS by degree l)")
    ax_bottom.grid(True, which="both", alpha=0.3)
    ax_bottom.legend(frameon=False)

    fig.tight_layout()
    plt.show()
    log(f"Plotted harmonics magnitude and conductivity harmonics for {label}.")


def _step6_cube_points(radius_m: float, n_edge: int) -> tuple[torch.Tensor, float]:
    n_edge = max(3, int(n_edge))
    _ = radius_m
    cube_half = float(STEP6_CUBE_HALF_M)
    axis = torch.linspace(-cube_half, cube_half, n_edge, dtype=torch.float64)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    keep = torch.linalg.norm(pts, dim=-1) > (float(radius_m) * 1.02)
    return pts[keep], cube_half


def _step6_build_emitted_field_spectra(sim_out: PhasorSimulation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sim_out.B_tor_emit is None or sim_out.B_pol_emit is None or sim_out.B_rad_emit is None:
        raise RuntimeError("Missing emitted field phasors; rerun solve before Step 6 plot.")
    b_tor_emit = sim_out.B_tor_emit.to(torch.complex128)
    b_pol_emit = sim_out.B_pol_emit.to(torch.complex128)
    b_rad_emit = sim_out.B_rad_emit.to(torch.complex128)
    return b_tor_emit, b_pol_emit, b_rad_emit


def _step6_uniform_applied_field_phasor(sim_out: PhasorSimulation, ambient_state: dict, n_points: int) -> torch.Tensor:
    axis = str(ambient_state.get("ambient_direction", "z")).strip().lower()
    if axis not in {"x", "y", "z"}:
        axis = "z"
    amp = float(ambient_state.get("ambient_amplitude_t", sim_out.ambient_amplitude_t))
    phase0 = float(getattr(ambient_state.get("ambient_cfg", None), "phase_rad", sim_out.ambient_phase_rad))
    u = torch.zeros((3,), dtype=torch.complex128)
    if axis == "x":
        u[0] = 1.0
    elif axis == "y":
        u[1] = 1.0
    else:
        u[2] = 1.0
    ph = torch.exp(torch.tensor(1j * phase0, dtype=torch.complex128))
    vec = (amp * ph) * u
    return vec.unsqueeze(0).repeat(max(1, int(n_points)), 1)



def _step6_integrate_flowlines(
    vec_pos: np.ndarray,
    vec_dir: np.ndarray,
    cube_half: float,
    step_len: float,
    n_steps: int,
    n_seeds: int,
) -> list[np.ndarray]:
    if vec_pos.size == 0 or vec_dir.size == 0:
        return []
    n = vec_pos.shape[0]
    seed_count = max(1, min(int(n_seeds), n))
    seed_idx = np.linspace(0, n - 1, seed_count, dtype=int)

    def _nearest_dir(p: np.ndarray) -> np.ndarray:
        d2 = np.sum((vec_pos - p[None, :]) ** 2, axis=1)
        i = int(np.argmin(d2))
        return vec_dir[i]

    lines: list[np.ndarray] = []
    for i in seed_idx:
        seed = vec_pos[i]
        for sign in (1.0, -1.0):
            pts = [seed.copy()]
            p = seed.copy()
            for _ in range(max(1, int(n_steps))):
                d = _nearest_dir(p)
                p = p + sign * step_len * d
                if np.any(np.abs(p) > cube_half):
                    break
                pts.append(p.copy())
            if len(pts) >= 2:
                lines.append(np.asarray(pts))
    return lines


def _visible_points_for_3d_view(
    points_xyz: np.ndarray,
    sphere_radius: float,
    elev_deg: float,
    azim_deg: float,
) -> np.ndarray:
    """Mask points occluded by an opaque sphere at origin for the current 3D view."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    view = np.array(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=np.float64,
    )
    view = view / max(np.linalg.norm(view), 1e-30)
    depth = pts @ view
    perp = pts - np.outer(depth, view)
    rho2 = np.sum(perp * perp, axis=1)
    r2 = float(sphere_radius) ** 2
    in_disc = rho2 < r2
    front_depth = np.sqrt(np.clip(r2 - rho2, 0.0, None))
    occluded = in_disc & (depth < front_depth)
    return ~occluded


def step6_render_magnetic_vectors(
    label: str,
    field_mode: str,
    display_mode: str,
    show_gradient_shell: bool,
    gradient_fd_scheme: str,
    gradient_alpha: float,
    t_sec: float,
    n_edge: int,
    log,
) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    ambient_state = _load_state("ambient.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mode = str(field_mode).strip().lower()
    if mode not in {"applied", "emitted", "combined"}:
        raise RuntimeError(f"Unknown Step 6 field mode: {field_mode}")
    display = str(display_mode).strip().lower()
    if display not in {"vectors", "flow"}:
        raise RuntimeError(f"Unknown Step 6 display mode: {display_mode}")
    fd_key = str(gradient_fd_scheme or "forward").strip().lower()
    if fd_key not in {"forward", "central"}:
        raise RuntimeError(f"Unknown finite-difference scheme: {gradient_fd_scheme}")
    grad_alpha = float(max(0.0, min(1.0, gradient_alpha)))
    n_edge = max(3, int(n_edge))
    t_sec = float(t_sec)

    cache_name = _step6_field_cache_name(label, mode, n_edge)
    cache_path = STATE_DIR / cache_name
    deps = _step4_dependency_signature(label)
    cache_data = None
    if cache_path.exists():
        try:
            candidate = _load_state(cache_name)
            if (
                isinstance(candidate, dict)
                and candidate.get("kind") == "step6_field_phasors"
                and candidate.get("deps") == deps
                and int(candidate.get("n_edge", -1)) == int(n_edge)
                and str(candidate.get("field_mode", "")).strip().lower() == mode
                and abs(float(candidate.get("cube_half_m", -1.0)) - float(STEP6_CUBE_HALF_M)) <= 1e-9
            ):
                cache_data = candidate
        except Exception:
            cache_data = None

    if cache_data is None:
        sample_pts, cube_half = _step6_cube_points(sim_out.radius_m, n_edge)
        t0 = time.perf_counter()
        b_applied = _step6_uniform_applied_field_phasor(sim_out, ambient_state, sample_pts.shape[0])
        if mode == "applied":
            B_phasor = b_applied
        elif mode == "emitted":
            b_tor, b_pol, b_rad = _step6_build_emitted_field_spectra(sim_out)
            B_phasor = evaluate_field_from_spectral(
                b_tor,
                b_pol,
                b_rad,
                sample_pts.to(dtype=torch.float64),
            ).to(torch.complex128)
        else:
            b_tor, b_pol, b_rad = _step6_build_emitted_field_spectra(sim_out)
            b_emit = evaluate_field_from_spectral(
                b_tor,
                b_pol,
                b_rad,
                sample_pts.to(dtype=torch.float64),
            ).to(torch.complex128)
            B_phasor = b_emit + b_applied
        dt = time.perf_counter() - t0
        cache_data = {
            "kind": "step6_field_phasors",
            "label": label,
            "field_mode": mode,
            "n_edge": int(n_edge),
            "cube_half_m": float(cube_half),
            "deps": deps,
            "positions": sample_pts.cpu(),
            "B_phasor": B_phasor.cpu(),
        }
        _save_state(cache_name, cache_data)
        log(f"Step 6 cached phasors to {cache_path} (compute {dt:.2f}s, n={sample_pts.shape[0]}).")
    else:
        log(f"Step 6 loaded cached phasors from {cache_path}.")

    positions = cache_data["positions"].to(torch.float64)
    b_phasor = cache_data["B_phasor"].to(torch.complex128)
    phase = np.exp(1j * float(sim_out.omega) * t_sec)
    b_real = torch.real(b_phasor * phase).to(torch.float64)

    norms = torch.linalg.norm(b_real, dim=-1)
    keep = norms > 0.0
    vec_pos = positions[keep]
    vec_dir = b_real[keep]
    vec_norm = torch.linalg.norm(vec_dir, dim=-1, keepdim=True).clamp_min(1e-30)
    vec_dir = vec_dir / vec_norm

    surf_pts_m = grid_state["positions"].to(torch.float64).cpu().numpy()
    faces = grid_state["faces"]
    face_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)
    sigma = grid_state.get("sigma_grid")
    if sigma is None:
        sigma_vals = np.zeros((surf_pts_m.shape[0],), dtype=np.float64)
    else:
        sigma_vals = sigma.detach().cpu().numpy().reshape(-1)
    face_sigma = sigma_vals[face_np].mean(axis=1)
    tri_verts = (surf_pts_m / 1.0e6)[face_np]
    sphere_radius_mm = float(np.median(np.linalg.norm(surf_pts_m / 1.0e6, axis=1)))
    vmin = float(np.nanmin(face_sigma)) if face_sigma.size else 0.0
    vmax = float(np.nanmax(face_sigma)) if face_sigma.size else 1.0
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 1e-6, 1e-30)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("Greys")
    face_colors = cmap(norm(face_sigma))

    cube_half_m = float(cache_data["cube_half_m"])
    step_m = (2.0 * cube_half_m) / max(int(n_edge) - 1, 1)
    arrow_len = (0.35 * step_m) / 1.0e6
    cube_half_mm = cube_half_m / 1.0e6
    title_mode = {"applied": "Applied", "emitted": "Emitted", "combined": "Combined"}[mode]
    display_title = "Vectors" if display == "vectors" else "Flow lines"
    t_tag = f"{t_sec:.3f}".replace(".", "p")
    grad_tag = "_withgrad" if bool(show_gradient_shell) else ""
    save_path = FIG_DIR / f"nonuniform_field_{display}{grad_tag}_{label}_{mode}_{int(n_edge)}edge_t{t_tag}s.png"

    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    surface = Poly3DCollection(
        tri_verts,
        facecolors=face_colors,
        edgecolor="none",
        linewidth=0.05,
        alpha=1.0,
        antialiased=True,
    )
    ax.add_collection3d(surface)
    grad_map = None
    grad_alt_km = None
    if bool(show_gradient_shell):
        grad_cache = _load_latest_step4_gradient_cache(label, fd_key)
        grad_pos = grad_cache["positions"].to(torch.float64).cpu().numpy()
        grad_rss = grad_cache["rss"].to(torch.float64).cpu().numpy().reshape(-1)
        tri_grad = (grad_pos / 1.0e6)[face_np]
        face_grad = grad_rss[face_np].mean(axis=1) * 1.0e12
        centers = tri_grad.mean(axis=1)
        hemi_mask = centers[:, 0] >= 0.0
        tri_grad_half = tri_grad[hemi_mask]
        face_grad_half = face_grad[hemi_mask]
        gmin = float(np.nanmin(face_grad_half)) if face_grad_half.size else 0.0
        gmax = float(np.nanmax(face_grad_half)) if face_grad_half.size else 1.0
        if gmax <= gmin:
            gmax = gmin + max(abs(gmin) * 1e-6, 1e-30)
        grad_norm = colors.Normalize(vmin=gmin, vmax=gmax)
        grad_colors = cm.get_cmap("rainbow")(grad_norm(face_grad_half))
        grad_surface = Poly3DCollection(
            tri_grad_half,
            facecolors=grad_colors,
            edgecolor="none",
            linewidth=0.05,
            alpha=grad_alpha,
            antialiased=True,
        )
        ax.add_collection3d(grad_surface)
        grad_map = cm.ScalarMappable(norm=grad_norm, cmap=cm.get_cmap("rainbow"))
        grad_map.set_array(face_grad_half)
        grad_alt_km = float(grad_cache.get("altitude_m", _latest_step4_gradient_altitude_m(label, fd_key))) / 1000.0

    if display == "vectors" and vec_pos.shape[0] > 0:
        p = vec_pos.cpu().numpy() / 1.0e6
        d = vec_dir.cpu().numpy()
        vis = _visible_points_for_3d_view(
            p,
            sphere_radius=sphere_radius_mm,
            elev_deg=float(getattr(ax, "elev", 30.0)),
            azim_deg=float(getattr(ax, "azim", -60.0)),
        )
        p = p[vis]
        d = d[vis]
        if p.shape[0] > 0:
            ax.quiver(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                d[:, 0],
                d[:, 1],
                d[:, 2],
                length=arrow_len,
                normalize=True,
                color="#202020",
                linewidth=0.8,
            )
    elif display == "flow" and vec_pos.shape[0] > 0:
        p = vec_pos.cpu().numpy()
        d = vec_dir.cpu().numpy()
        lines = _step6_integrate_flowlines(
            vec_pos=p,
            vec_dir=d,
            cube_half=cube_half_m,
            step_len=0.30 * step_m,
            n_steps=14,
            n_seeds=max(12, min(80, p.shape[0] // 2)),
        )
        for line in lines:
            line_mm = line / 1.0e6
            ax.plot(
                line_mm[:, 0],
                line_mm[:, 1],
                line_mm[:, 2],
                color="#202020",
                linewidth=1.0,
                alpha=0.9,
            )

    lim = cube_half_mm
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("y (Mm)")
    ax.set_zlabel("z (Mm)")
    title = f"Step 6 magnetic field {display_title} ({title_mode}) at t={t_sec:.3f} s"
    if bool(show_gradient_shell):
        title += f", gradients at alt={grad_alt_km:.0f} km"
    ax.set_title(title)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(face_sigma)
    fig.colorbar(mappable, ax=ax, pad=0.04, shrink=0.72, label="Conductivity sigma_s (S)")
    if grad_map is not None:
        fig.colorbar(grad_map, ax=ax, pad=0.12, shrink=0.72, label="|grad_B_emit| RSS (pT/m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    log(f"Step 6 rendered magnetic field {display_title.lower()} to {save_path}")
    return save_path


def step6_render_gradient_shell(label: str, field_mode: str, altitude_m: float, fd_scheme: str, log) -> Path:
    payload = _load_solution(label)
    sim_out: PhasorSimulation = payload["phasor_sim"]
    grid_state = _load_state("grid_admittance.pt")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mode = str(field_mode).strip().lower()
    if mode not in {"applied", "emitted", "combined"}:
        raise RuntimeError(f"Unknown Step 6 field mode: {field_mode}")
    altitude_m = max(0.0, float(altitude_m))
    fd_key = str(fd_scheme or "forward").strip().lower()
    if fd_key not in {"forward", "central"}:
        raise RuntimeError(f"Unknown finite-difference scheme: {fd_scheme}")

    cache_name = _step6_gradient_cache_name(label, mode, altitude_m, fd_key)
    cache_path = STATE_DIR / cache_name
    deps = _step4_dependency_signature(label)
    cache_data = None
    if cache_path.exists():
        try:
            candidate = _load_state(cache_name)
            if (
                isinstance(candidate, dict)
                and candidate.get("kind") == "step6_gradient_rss"
                and candidate.get("deps") == deps
                and str(candidate.get("field_mode", "")).strip().lower() == mode
                and int(candidate.get("altitude_m", -1)) == int(round(altitude_m))
                and str(candidate.get("fd_scheme", "")).strip().lower() == fd_key
            ):
                cache_data = candidate
        except Exception:
            cache_data = None

    if cache_data is None:
        scale = float(sim_out.radius_m + altitude_m) / float(sim_out.radius_m)
        positions_obs = (sim_out.grid_positions * scale).to(dtype=torch.float64)
        t0 = time.perf_counter()
        if mode == "applied":
            rss = torch.zeros((positions_obs.shape[0],), dtype=torch.float64)
        else:
            rss = rss_gradient_from_emit(
                sim_out,
                positions_obs,
                obs_radius=float(sim_out.radius_m + altitude_m),
                fd_scheme=fd_key,
            ).to(torch.float64)
        dt = time.perf_counter() - t0
        cache_data = {
            "kind": "step6_gradient_rss",
            "label": label,
            "field_mode": mode,
            "altitude_m": int(round(altitude_m)),
            "fd_scheme": fd_key,
            "deps": deps,
            "positions_obs": positions_obs.cpu(),
            "rss": rss.cpu(),
        }
        _save_state(cache_name, cache_data)
        log(f"Step 6 cached gradient shell data to {cache_path} (compute {dt:.2f}s).")
    else:
        log(f"Step 6 loaded cached gradient shell data from {cache_path}.")

    faces = grid_state["faces"]
    face_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)

    surf_pts_m = grid_state["positions"].to(torch.float64).cpu().numpy()
    tri_cond = (surf_pts_m / 1.0e6)[face_np]
    sigma = grid_state.get("sigma_grid")
    sigma_vals = np.zeros((surf_pts_m.shape[0],), dtype=np.float64) if sigma is None else sigma.detach().cpu().numpy().reshape(-1)
    face_sigma = sigma_vals[face_np].mean(axis=1)
    cond_vmin = float(np.nanmin(face_sigma)) if face_sigma.size else 0.0
    cond_vmax = float(np.nanmax(face_sigma)) if face_sigma.size else 1.0
    if cond_vmax <= cond_vmin:
        cond_vmax = cond_vmin + max(abs(cond_vmin) * 1e-6, 1e-30)
    cond_norm = colors.Normalize(vmin=cond_vmin, vmax=cond_vmax)
    cond_colors = cm.get_cmap("Greys")(cond_norm(face_sigma))

    obs_pts_m = cache_data["positions_obs"].to(torch.float64).cpu().numpy()
    rss = cache_data["rss"].to(torch.float64).cpu().numpy().reshape(-1)
    tri_grad = (obs_pts_m / 1.0e6)[face_np]
    face_grad = rss[face_np].mean(axis=1) * 1.0e12
    centers = tri_grad.mean(axis=1)
    hemi_mask = centers[:, 0] >= 0.0
    tri_grad_half = tri_grad[hemi_mask]
    face_grad_half = face_grad[hemi_mask]
    grad_vmin = float(np.nanmin(face_grad_half)) if face_grad_half.size else 0.0
    grad_vmax = float(np.nanmax(face_grad_half)) if face_grad_half.size else 1.0
    if grad_vmax <= grad_vmin:
        grad_vmax = grad_vmin + max(abs(grad_vmin) * 1e-6, 1e-30)
    grad_norm = colors.Normalize(vmin=grad_vmin, vmax=grad_vmax)
    grad_colors = cm.get_cmap("rainbow")(grad_norm(face_grad_half))

    title_mode = {"applied": "Applied", "emitted": "Emitted", "combined": "Combined"}[mode]
    alt_km = altitude_m / 1000.0
    save_path = FIG_DIR / (
        f"nonuniform_gradient_shell_{label}_{mode}_{int(round(altitude_m))}m_{fd_key}.png"
    )

    fig = plt.figure(figsize=(11.0, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    cond_surface = Poly3DCollection(
        tri_cond,
        facecolors=cond_colors,
        edgecolor="none",
        linewidth=0.05,
        alpha=1.0,
        antialiased=True,
    )
    ax.add_collection3d(cond_surface)
    grad_surface = Poly3DCollection(
        tri_grad_half,
        facecolors=grad_colors,
        edgecolor="none",
        linewidth=0.05,
        alpha=0.75,
        antialiased=True,
    )
    ax.add_collection3d(grad_surface)

    lim = float(STEP6_CUBE_HALF_M / 1.0e6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("y (Mm)")
    ax.set_zlabel("z (Mm)")
    ax.set_title(
        f"Step 6 gradient shell ({title_mode}) at alt={alt_km:.0f} km; "
        "conductivity at ocean radius"
    )
    cond_map = cm.ScalarMappable(norm=cond_norm, cmap=cm.get_cmap("Greys"))
    cond_map.set_array(face_sigma)
    grad_map = cm.ScalarMappable(norm=grad_norm, cmap=cm.get_cmap("rainbow"))
    grad_map.set_array(face_grad_half)
    fig.colorbar(cond_map, ax=ax, pad=0.02, shrink=0.62, label="Conductivity sigma_s (S)")
    fig.colorbar(grad_map, ax=ax, pad=0.10, shrink=0.62, label="|grad_B_emit| RSS (pT/m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    log(f"Step 6 rendered gradient shell to {save_path}")
    return save_path


def _clear_outputs(log) -> None:
    for path in (FIG_DIR, STATE_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    log(f"Cleared outputs in {FIG_DIR} and {STATE_DIR}.")


def step6_iterative_solve(order: int, log) -> Path:
    _clear_overview_cache(log, reason="Step 5 solve rerun", label="iterative")
    _clear_step4_field_cache(log, reason="Step 5 solve rerun", label="iterative")
    _clear_step6_field_cache(log, reason="Step 5 solve rerun", label="iterative")
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

    log("Building sparse mixing matrix (v_toroidal)...")
    mixing_matrix = _build_mixing_matrix_precomputed_sparse(
        grid_cfg.lmax,
        base.omega,
        base.radius_m,
        base.admittance_spectral,
        G_sparse,
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
    log(f"Step 5 complete. Saved iterative solution to {path}")
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
    solve_mode_var = tk.StringVar(value="self_consistent")
    mode_labels = {
        "first_order": "first-order",
        "self_consistent": "self-consistent",
        "iterative": "iterative",
    }
    mode_accents = {
        "first_order": "light sky blue",
        "self_consistent": "pale green",
        "iterative": "khaki1",
    }

    def _selected_mode() -> str:
        mode = solve_mode_var.get()
        if mode not in mode_labels:
            return "self_consistent"
        return mode

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
            log(f"Step 1 load: {state_key} already at standard path {dst}")
            return dst
        shutil.copy2(src, dst)
        log(f"Step 1 load: copied {src} -> {dst}")
        return dst

    def _save_state_to_path(state_key: str, target_path: str, log) -> Path:
        src = _standard_state_path(state_key)
        dst = Path(target_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"Standard state file missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            log(f"Step 1 save: {state_key} stays at standard path {src}")
            return dst
        shutil.copy2(src, dst)
        log(f"Step 1 save: copied {src} -> {dst}")
        return dst

    def _set_button_state(
        btn: tk.Button, enabled: bool, completed: bool = False, color: str | None = None
    ) -> None:
        if enabled:
            if color is not None:
                bg = color
            else:
                bg = "pale green" if completed else "SystemButtonFace"
            btn.config(state=tk.NORMAL, bg=bg)
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
        first_ok = _solution_exists("first_order")
        self_ok = _solution_exists("self_consistent")
        iter_ok = _solution_exists("iterative")
        selected_mode = _selected_mode()
        selected_ok = {
            "first_order": first_ok,
            "self_consistent": self_ok,
            "iterative": iter_ok,
        }[selected_mode]
        selected_color = mode_accents[selected_mode]
        _set_button_state(btn_step1, True, completed=grid_ok)
        _set_button_state(btn_step1b, grid_ok, completed=grid_ok)
        _set_button_state(btn_step1b_power, grid_ok, completed=grid_ok)
        _set_button_state(btn_step2, grid_ok, completed=ambient_ok)
        _set_button_state(btn_step_solve, ambient_ok, completed=selected_ok, color=selected_color if ambient_ok else None)
        _set_button_state(btn_overview, selected_ok, color=selected_color if selected_ok else None)
        _set_button_state(btn_grad100, selected_ok, color=selected_color if selected_ok else None)
        _set_button_state(btn_grad100_log, selected_ok, color=selected_color if selected_ok else None)
        _set_button_state(btn_harm, selected_ok, color=selected_color if selected_ok else None)
        _set_button_state(btn_bmag100, selected_ok, color=selected_color if selected_ok else None)
        _set_button_state(btn_step6_field, selected_ok, color=selected_color if selected_ok else None)

    # Step 1: load latest run folder
    ttk.Label(frm, text="Step 1: Run folder").grid(row=0, column=0, sticky="w")
    prefix_var = tk.StringVar(value="run")
    ttk.Label(frm, text="prefix").grid(row=0, column=1, sticky="e")
    ttk.Entry(frm, textvariable=prefix_var, width=10).grid(row=0, column=2, sticky="w")
    run_dir = _latest_run_dir()
    if run_dir is None:
        run_dir = _new_run_dir(prefix_var.get())
    _set_run_dirs(run_dir)
    run_dir_var = tk.StringVar(value=str(STATE_DIR))
    ttk.Entry(frm, textvariable=run_dir_var, width=44, state="readonly").grid(
        row=0, column=3, columnspan=4, sticky="we", padx=4
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
    btn_step0_load.grid(row=0, column=7, padx=4, sticky="w")
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
    btn_step0_rename.grid(row=0, column=8, padx=4, sticky="w")

    # Inputs for step 2
    ttk.Label(frm, text="Step 2: Grid + admittance").grid(row=1, column=0, sticky="w")
    ttk.Label(frm, text="lmax").grid(row=1, column=1, sticky="e")
    default_lmax = "36"
    default_iter_order = "3"
    default_plotter = "matplotlib"
    default_gradient_fd_scheme = "forward"
    default_conductivity_model = "europa_snapshot"
    default_mode_l = "10"
    default_mode_m = "2"
    default_frac_rms = "0.05"
    default_ambient_axis = "z"
    default_ambient_amplitude = "1e-6"
    default_ambient_period_hours = "9.925"
    default_step6_time_sec = "0.0"
    default_step6_grid_n = "7"
    default_step6_field_mode = "combined"
    default_step6_display_mode = "vectors"
    default_step6_show_gradient = True
    default_step6_gradient_alpha = "0.75"
    lmax_var = tk.StringVar(value=default_lmax)
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
    ttk.Label(frm, text="# SH coeffs=").grid(row=2, column=3, sticky="e")
    ttk.Label(frm, textvariable=sh_count_var).grid(row=2, column=4, sticky="w")

    ttk.Label(frm, text="iter order").grid(row=2, column=5, sticky="e")
    iter_order_var = tk.StringVar(value=default_iter_order)
    ttk.Entry(frm, textvariable=iter_order_var, width=6).grid(row=2, column=6, sticky="w")

    ttk.Label(frm, text="Sphere plotter").grid(row=2, column=7, sticky="e")
    plotter_var = tk.StringVar(value=default_plotter)
    tk.Radiobutton(frm, text="PyVista", variable=plotter_var, value="pyvista").grid(row=2, column=8, sticky="w")
    tk.Radiobutton(frm, text="Matplotlib", variable=plotter_var, value="matplotlib").grid(row=2, column=9, sticky="w")
    ttk.Label(frm, text="Gradient FD").grid(row=2, column=10, sticky="e")
    gradient_fd_var = tk.StringVar(value=default_gradient_fd_scheme)
    tk.Radiobutton(frm, text="Forward", variable=gradient_fd_var, value="forward").grid(row=2, column=11, sticky="w")
    tk.Radiobutton(frm, text="Central", variable=gradient_fd_var, value="central").grid(row=2, column=12, sticky="w")

    ttk.Label(frm, text="conductivity model").grid(row=4, column=0, sticky="w")
    conductivity_model_var = tk.StringVar(value=default_conductivity_model)
    tk.Radiobutton(
        frm, text="Uniform", variable=conductivity_model_var, value="uniform"
    ).grid(row=4, column=1, sticky="w")
    tk.Radiobutton(
        frm, text="Europa snapshot", variable=conductivity_model_var, value="europa_snapshot"
    ).grid(row=4, column=2, sticky="w")
    tk.Radiobutton(
        frm, text="Selected harmonic", variable=conductivity_model_var, value="synthetic_sh"
    ).grid(row=4, column=3, sticky="w")

    default_cfg = GridConfig(nside=1, lmax=1, radius_m=1.56e6, device="cpu")
    default_mean = 2.0 * default_cfg.seawater_conductivity_s_per_m * default_cfg.ocean_thickness_m
    ttk.Label(frm, text="mean conductivity (S)").grid(row=3, column=1, sticky="e")
    mean_cond_var = tk.StringVar(value=f"{default_mean:.3e}")
    ttk.Entry(frm, textvariable=mean_cond_var, width=10).grid(row=3, column=2, sticky="w")
    ttk.Label(frm, text="target l").grid(row=3, column=3, sticky="e")
    mode_l_var = tk.StringVar(value=default_mode_l)
    ttk.Entry(frm, textvariable=mode_l_var, width=6).grid(row=3, column=4, sticky="w")
    ttk.Label(frm, text="target |m|").grid(row=3, column=5, sticky="e")
    mode_m_var = tk.StringVar(value=default_mode_m)
    ttk.Entry(frm, textvariable=mode_m_var, width=6).grid(row=3, column=6, sticky="w")
    ttk.Label(frm, text="frac RMS").grid(row=3, column=7, sticky="e")
    frac_rms_var = tk.StringVar(value=default_frac_rms)
    ttk.Entry(frm, textvariable=frac_rms_var, width=6).grid(row=3, column=8, sticky="w")
    ambient_direction_var = tk.StringVar(value=default_ambient_axis)
    ambient_amplitude_var = tk.StringVar(value=default_ambient_amplitude)
    ambient_period_var = tk.StringVar(value=default_ambient_period_hours)
    step6_time_sec_var = tk.StringVar(value=default_step6_time_sec)
    step6_grid_n_var = tk.StringVar(value=default_step6_grid_n)
    step6_field_mode_var = tk.StringVar(value=default_step6_field_mode)
    step6_display_mode_var = tk.StringVar(value=default_step6_display_mode)
    step6_show_gradient_var = tk.BooleanVar(value=default_step6_show_gradient)
    step6_gradient_alpha_var = tk.StringVar(value=default_step6_gradient_alpha)

    def _reset_inputs_to_defaults(log_reset: bool = True) -> None:
        lmax_var.set(default_lmax)
        mean_cond_var.set(f"{default_mean:.3e}")
        mode_l_var.set(default_mode_l)
        mode_m_var.set(default_mode_m)
        frac_rms_var.set(default_frac_rms)
        conductivity_model_var.set(default_conductivity_model)
        ambient_direction_var.set(default_ambient_axis)
        ambient_amplitude_var.set(default_ambient_amplitude)
        ambient_period_var.set(default_ambient_period_hours)
        iter_order_var.set(default_iter_order)
        plotter_var.set(default_plotter)
        gradient_fd_var.set(default_gradient_fd_scheme)
        solve_mode_var.set("self_consistent")
        step6_time_sec_var.set(default_step6_time_sec)
        step6_grid_n_var.set(default_step6_grid_n)
        step6_field_mode_var.set(default_step6_field_mode)
        step6_display_mode_var.set(default_step6_display_mode)
        step6_show_gradient_var.set(default_step6_show_gradient)
        step6_gradient_alpha_var.set(default_step6_gradient_alpha)
        _update_grid_counts()
        if log_reset:
            _log(log_widget, "Reset GUI inputs to defaults.")

    btn_step0_reset = tk.Button(
        frm,
        text="Reset defaults",
        command=lambda: _reset_inputs_to_defaults(log_reset=True),
    )
    btn_step0_reset.grid(row=0, column=9, padx=4, sticky="w")

    btn_step1 = tk.Button(
        frm,
        text="Clear + Run Step 2",
        command=lambda: run_step(
            btn_step1,
            lambda: (
                _start_new_run(prefix_var.get(), lambda msg: _log(log_widget, msg)),
                step1_build_grid_admittance(
                    int(lmax_var.get()),
                    float(mean_cond_var.get()),
                    float(frac_rms_var.get()),
                    int(mode_l_var.get()),
                    int(mode_m_var.get()),
                    conductivity_model_var.get(),
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
    btn_step1.grid(row=3, column=11, padx=6, sticky="w")
    def _refresh_inputs_from_loaded_state() -> None:
        try:
            if _grid_exists():
                state = _load_state("grid_admittance.pt")
                lmax = int(getattr(state.get("grid_cfg"), "lmax", lmax_var.get()))
                lmax_var.set(str(lmax))
                node_count_var.set(str(int(state.get("node_count", _node_count_from_lmax(lmax)))))
                face_count_var.set(str(int(state.get("face_count", max(4, int(node_count_var.get()) * 2 - 4)))))
                _update_grid_counts()
                if "sigma_mean" in state:
                    mean_cond_var.set(f"{float(state['sigma_mean']):.3e}")
                if "sigma_frac_rms" in state:
                    frac_rms_var.set(str(float(state["sigma_frac_rms"])))
                if "sigma_mode_l" in state:
                    mode_l_var.set(str(int(state["sigma_mode_l"])))
                if "sigma_mode_m" in state:
                    mode_m_var.set(str(int(state["sigma_mode_m"])))
                if "conductivity_model" in state:
                    conductivity_model_var.set(str(state["conductivity_model"]))

            if _ambient_exists():
                astate = _load_state("ambient.pt")
                axis = str(astate.get("ambient_direction", "x")).strip().lower()
                if axis in {"x", "y", "z"}:
                    ambient_direction_var.set(axis)
                if "ambient_amplitude_t" in astate:
                    ambient_amplitude_var.set(f"{float(astate['ambient_amplitude_t']):.6g}")
                elif "ambient_cfg" in astate:
                    amp = getattr(astate["ambient_cfg"], "amplitude_t", None)
                    if amp is not None:
                        ambient_amplitude_var.set(f"{float(amp):.6g}")
                if "ambient_period_hours" in astate:
                    ambient_period_var.set(f"{float(astate['ambient_period_hours']):.6g}")
                elif "period_sec" in astate:
                    ambient_period_var.set(f"{float(astate['period_sec']) / 3600.0:.6g}")
                elif "ambient_cfg" in astate:
                    omega = getattr(astate["ambient_cfg"], "omega_jovian", None)
                    if omega is not None and float(omega) > 0.0:
                        ambient_period_var.set(f"{(2.0 * math.pi / float(omega)) / 3600.0:.6g}")
            _log(log_widget, "Step 1: refreshed GUI inputs from loaded state.")
        except Exception as exc:  # noqa: BLE001
            _log(log_widget, f"Step 1: unable to refresh GUI inputs ({exc})")
    _refresh_inputs_from_loaded_state()

    # Step 3
    ttk.Label(frm, text="Step 3: Admittance check").grid(row=5, column=0, sticky="w")
    btn_step1b = tk.Button(
        frm,
        text="Admittance plots",
        command=lambda: run_step_ui(btn_step1b, lambda: step1b_plot_roundtrip(lambda msg: _log(log_widget, msg), plotter_var.get())),
    )
    btn_step1b.grid(row=5, column=2, padx=6, sticky="w")
    btn_step1b_power = tk.Button(
        frm,
        text="Admittance magnitude (l,m)",
        command=lambda: run_step_ui(
            btn_step1b_power,
            lambda: step1b_plot_admittance_power(lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_step1b_power.grid(row=5, column=3, padx=6, sticky="w")

    # Step 4
    ttk.Label(frm, text="Step 4: Ambient field").grid(row=6, column=0, sticky="w")
    ttk.Label(frm, text="axis").grid(row=6, column=1, sticky="e")
    tk.Radiobutton(frm, text="X", variable=ambient_direction_var, value="x").grid(row=6, column=2, sticky="w")
    tk.Radiobutton(frm, text="Y", variable=ambient_direction_var, value="y").grid(row=6, column=3, sticky="w")
    tk.Radiobutton(frm, text="Z", variable=ambient_direction_var, value="z").grid(row=6, column=4, sticky="w")
    ttk.Label(frm, text="amplitude (T)").grid(row=6, column=5, sticky="e")
    ttk.Entry(frm, textvariable=ambient_amplitude_var, width=10).grid(row=6, column=6, sticky="w")
    ttk.Label(frm, text="period (h)").grid(row=6, column=7, sticky="e")
    ttk.Entry(frm, textvariable=ambient_period_var, width=8).grid(row=6, column=8, sticky="w")
    btn_step2 = tk.Button(
        frm,
        text="Build ambient",
        command=lambda: run_step(
            btn_step2,
            lambda: step2_build_ambient(
                ambient_direction_var.get(),
                float(ambient_amplitude_var.get()),
                float(ambient_period_var.get()),
                lambda msg: _log(log_widget, msg),
            ),
        ),
    )
    btn_step2.grid(row=6, column=9, padx=6, sticky="w")

    # Step 5: Solve mode selector + shared plots
    solve_mode_header_var = tk.StringVar(value="Step 5: Solve mode")
    ttk.Label(frm, textvariable=solve_mode_header_var).grid(row=7, column=0, sticky="w")
    tk.Radiobutton(
        frm,
        text="First-order",
        variable=solve_mode_var,
        value="first_order",
        selectcolor=mode_accents["first_order"],
    ).grid(row=7, column=1, sticky="w")
    tk.Radiobutton(
        frm,
        text="Self-consistent",
        variable=solve_mode_var,
        value="self_consistent",
        selectcolor=mode_accents["self_consistent"],
    ).grid(row=7, column=2, sticky="w")
    tk.Radiobutton(
        frm,
        text="Iterative",
        variable=solve_mode_var,
        value="iterative",
        selectcolor=mode_accents["iterative"],
    ).grid(row=7, column=3, sticky="w")
    def _run_selected_solve():
        mode = _selected_mode()
        if mode == "first_order":
            return step3_solve_currents(True, lambda msg: _log(log_widget, msg))
        if mode == "iterative":
            return step6_iterative_solve(int(iter_order_var.get()), lambda msg: _log(log_widget, msg))
        return step3_solve_currents(False, lambda msg: _log(log_widget, msg))

    btn_step_solve = tk.Button(
        frm,
        text="Solve self-consistent",
        command=lambda: run_step(btn_step_solve, _run_selected_solve),
    )
    btn_step_solve.grid(row=8, column=2, padx=4, sticky="w")
    btn_overview = tk.Button(
        frm,
        text="Overview (self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_overview,
            lambda: step4_render_overview(_selected_mode(), lambda msg: _log(log_widget, msg), plotter_var.get()),
        ),
    )
    btn_overview.grid(row=8, column=3, padx=4, sticky="w")
    btn_grad100 = tk.Button(
        frm,
        text="Gradients @ 100 km (self-consistent)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100,
            lambda: step4_render_gradient(
                _selected_mode(),
                100e3,
                lambda msg: _log(log_widget, msg),
                plotter_var.get(),
                gradient_fd_var.get(),
            ),
        ),
    )
    btn_grad100.grid(row=8, column=4, padx=4, sticky="w")
    btn_grad100_log = tk.Button(
        frm,
        text="Gradients @ 100 km (log scale)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_grad100_log,
            lambda: step4_render_gradient_log100(
                _selected_mode(),
                lambda msg: _log(log_widget, msg),
                plotter_var.get(),
                gradient_fd_var.get(),
            ),
        ),
    )
    btn_grad100_log.grid(row=8, column=5, padx=4, sticky="w")
    btn_bmag100 = tk.Button(
        frm,
        text="B magnitude @ 100 km",
        wraplength=180,
        command=lambda: run_step_ui(
            btn_bmag100,
            lambda: step4_render_bmag100(_selected_mode(), lambda msg: _log(log_widget, msg), plotter_var.get()),
        ),
    )
    btn_bmag100.grid(row=8, column=6, padx=4, sticky="w")
    btn_harm = tk.Button(
        frm,
        text="Harmonics (ambient vs emitted)",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_harm,
            lambda: step4_plot_harmonics(_selected_mode(), lambda msg: _log(log_widget, msg)),
        ),
    )
    btn_harm.grid(row=8, column=7, padx=4, sticky="w")

    # Step 6: 3D magnetic vectors around conductivity sphere
    ttk.Label(frm, text="Step 6: Field vectors around sphere").grid(row=9, column=0, sticky="w")
    ttk.Label(frm, text="t (s)").grid(row=9, column=1, sticky="e")
    ttk.Entry(frm, textvariable=step6_time_sec_var, width=8).grid(row=9, column=2, sticky="w")
    ttk.Label(frm, text="cube edge vectors").grid(row=9, column=3, sticky="e")
    ttk.Entry(frm, textvariable=step6_grid_n_var, width=6).grid(row=9, column=4, sticky="w")
    tk.Radiobutton(
        frm,
        text="Applied",
        variable=step6_field_mode_var,
        value="applied",
    ).grid(row=9, column=5, sticky="w")
    tk.Radiobutton(
        frm,
        text="Emitted",
        variable=step6_field_mode_var,
        value="emitted",
    ).grid(row=9, column=6, sticky="w")
    tk.Radiobutton(
        frm,
        text="Combined",
        variable=step6_field_mode_var,
        value="combined",
    ).grid(row=9, column=7, sticky="w")
    tk.Radiobutton(
        frm,
        text="Vectors",
        variable=step6_display_mode_var,
        value="vectors",
    ).grid(row=10, column=5, sticky="w")
    tk.Radiobutton(
        frm,
        text="Flow lines",
        variable=step6_display_mode_var,
        value="flow",
    ).grid(row=10, column=6, sticky="w")
    tk.Checkbutton(
        frm,
        text="Overlay gradient shell (use prior Step 5 gradient cache)",
        variable=step6_show_gradient_var,
        onvalue=True,
        offvalue=False,
    ).grid(row=10, column=7, sticky="w")
    ttk.Label(frm, text="grad alpha [0-1]").grid(row=10, column=3, sticky="e")
    ttk.Entry(frm, textvariable=step6_gradient_alpha_var, width=6).grid(row=10, column=4, sticky="w")
    btn_step6_field = tk.Button(
        frm,
        text="Plot field",
        wraplength=180,
        justify="left",
        command=lambda: run_step_ui(
            btn_step6_field,
            lambda: step6_render_magnetic_vectors(
                _selected_mode(),
                step6_field_mode_var.get(),
                step6_display_mode_var.get(),
                bool(step6_show_gradient_var.get()),
                gradient_fd_var.get(),
                float(step6_gradient_alpha_var.get()),
                float(step6_time_sec_var.get()),
                int(step6_grid_n_var.get()),
                lambda msg: _log(log_widget, msg),
            ),
        ),
    )
    btn_step6_field.grid(row=9, column=8, padx=4, sticky="w")

    def _update_selected_mode_labels() -> None:
        mode = _selected_mode()
        label = mode_labels[mode]
        solve_mode_header_var.set(f"Step 5: {label} solve")
        btn_step_solve.config(text=f"Solve {label}")
        btn_overview.config(text=f"Overview ({label})")
        btn_grad100.config(text=f"Gradients @ 100 km ({label})")
        btn_grad100_log.config(text=f"Gradients @ 100 km (log, {label})")
        btn_bmag100.config(text=f"B magnitude @ 100 km ({label})")
        btn_step6_field.config(text=f"Plot field ({label})")

    solve_mode_var.trace_add("write", lambda *_: (_update_selected_mode_labels(), _update_button_states()))
    _update_selected_mode_labels()

    frm.rowconfigure(11, weight=1)
    frm.columnconfigure(6, weight=1)

    lmax_var.trace_add("write", lambda *_: _update_grid_counts())
    _update_grid_counts()
    _update_button_states()

    root.mainloop()


if __name__ == "__main__":
    main()

