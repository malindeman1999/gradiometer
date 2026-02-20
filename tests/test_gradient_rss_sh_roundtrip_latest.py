import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from europa_model import transforms
from europa_model.gradient_utils import rss_gradient_from_emit
from workflow.data_objects.phasor_data import PhasorSimulation

matplotlib.use("Agg")

BASE_RUN_DIR = Path("workflow/artifacts/nonuniform_workflow")
PLOT_DIR = Path("tests/artifacts")


def _latest_nonuniform_run_dir() -> Path | None:
    if not BASE_RUN_DIR.exists():
        return None
    runs = [p for p in BASE_RUN_DIR.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _load_latest_simulation() -> tuple[PhasorSimulation, Path]:
    run_dir = _latest_nonuniform_run_dir()
    if run_dir is None:
        raise FileNotFoundError(f"No run folders found under {BASE_RUN_DIR}")

    candidates = [run_dir / "solution_latest.pt", run_dir / "overview_input.pt"]
    for data_path in candidates:
        if not data_path.exists():
            continue
        raw = torch.load(data_path, map_location="cpu", weights_only=False)
        try:
            return PhasorSimulation.from_saved(raw), data_path
        except ValueError:
            continue

    raise FileNotFoundError(
        f"Could not load a PhasorSimulation from latest run folder {run_dir}. "
        f"Checked: {', '.join(str(p.name) for p in candidates)}"
    )


def _lm_magnitude_matrix(coeffs: torch.Tensor) -> np.ndarray:
    coeffs = coeffs.to(torch.complex128).detach().cpu()
    lmax = coeffs.shape[0] - 1
    mat = np.full((lmax + 1, 2 * lmax + 1), np.nan, dtype=np.float64)
    for l in range(lmax + 1):
        lo = lmax - l
        hi = lmax + l + 1
        mat[l, lo:hi] = torch.abs(coeffs[l, lo:hi]).numpy()
    return mat


def _save_lm_plots(
    coeffs_original: torch.Tensor,
    coeffs_roundtrip: torch.Tensor,
    out_path: Path,
) -> None:
    lmax = coeffs_original.shape[0] - 1
    m_min = -lmax
    m_max = lmax
    extent = (m_min - 0.5, m_max + 0.5, lmax + 0.5, -0.5)

    orig = _lm_magnitude_matrix(coeffs_original)
    rt = _lm_magnitude_matrix(coeffs_roundtrip)
    diff = np.abs(rt - orig)

    vmax = float(np.nanmax(np.log10(np.maximum(orig, 1e-30))))
    vmin = float(vmax - 6.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(np.log10(np.maximum(orig, 1e-30)), aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title("Original |coeff(l,m)|")
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("l")
    fig.colorbar(im0, ax=axes[0], label="log10 |coeff|")

    im1 = axes[1].imshow(np.log10(np.maximum(rt, 1e-30)), aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title("Round-trip |coeff(l,m)|")
    axes[1].set_xlabel("m")
    axes[1].set_ylabel("l")
    fig.colorbar(im1, ax=axes[1], label="log10 |coeff|")

    diff_vmax = float(np.nanmax(np.log10(np.maximum(diff, 1e-30))))
    diff_vmin = float(diff_vmax - 6.0)
    im2 = axes[2].imshow(np.log10(np.maximum(diff, 1e-30)), aspect="auto", extent=extent, vmin=diff_vmin, vmax=diff_vmax)
    axes[2].set_title("Absolute Difference |delta coeff(l,m)|")
    axes[2].set_xlabel("m")
    axes[2].set_ylabel("l")
    fig.colorbar(im2, ax=axes[2], label="log10 |delta coeff|")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_grid_scatter_plot(values: torch.Tensor, recon: torch.Tensor, out_path: Path) -> None:
    v = values.detach().cpu().numpy().reshape(-1)
    r = recon.detach().cpu().numpy().reshape(-1)
    err = r - v
    lim_lo = float(min(np.min(v), np.min(r)))
    lim_hi = float(max(np.max(v), np.max(r)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    ax.scatter(v, r, s=8, alpha=0.7, color="#1f77b4", edgecolors="none")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Original gradient RSS")
    ax.set_ylabel("Round-trip gradient RSS")
    ax.set_title("RSS Gradient Round-Trip (Grid Values)")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03,
        0.97,
        f"mean abs err = {np.mean(np.abs(err)):.3e}\nmax abs err = {np.max(np.abs(err)):.3e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#888888"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def test_latest_gradient_rss_sh_roundtrip_with_plots():
    try:
        sim, source_path = _load_latest_simulation()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
        return

    positions = sim.grid_positions.to(torch.float64)
    weights = sim.grid_areas.to(torch.float64)
    lmax = int(sim.lmax)

    grad_altitude_m = 100e3
    obs_scale = float(sim.radius_m + grad_altitude_m) / float(sim.radius_m)
    obs_positions = positions * obs_scale

    grad_rss = rss_gradient_from_emit(
        sim,
        obs_positions,
        obs_radius=float(sim.radius_m + grad_altitude_m),
        fd_scheme="forward",
    ).to(torch.float64).reshape(-1)

    coeffs_original = transforms.sh_forward(grad_rss, positions, lmax=lmax, weights=weights).to(torch.complex128)
    grad_rss_roundtrip = transforms.sh_inverse(coeffs_original, positions, weights).to(torch.complex128).real.reshape(-1)
    coeffs_roundtrip = transforms.sh_forward(grad_rss_roundtrip, positions, lmax=lmax, weights=weights).to(torch.complex128)

    w = weights.reshape(-1)
    diff = grad_rss_roundtrip - grad_rss
    rel_l2 = math.sqrt(
        float(torch.sum(w * diff * diff).item())
        / max(float(torch.sum(w * grad_rss * grad_rss).item()), 1e-30)
    )
    max_abs = float(torch.max(torch.abs(diff)).item())
    coeff_rel = float(
        torch.linalg.norm(coeffs_roundtrip - coeffs_original).item()
        / max(torch.linalg.norm(coeffs_original).item(), 1e-30)
    )

    run_name = source_path.parent.name
    lm_plot_path = PLOT_DIR / f"gradient_rss_roundtrip_lm_{run_name}.png"
    scatter_plot_path = PLOT_DIR / f"gradient_rss_roundtrip_grid_{run_name}.png"
    _save_lm_plots(coeffs_original, coeffs_roundtrip, lm_plot_path)
    _save_grid_scatter_plot(grad_rss, grad_rss_roundtrip, scatter_plot_path)

    print(f"Loaded latest simulation from: {source_path}")
    print(f"Saved (l,m) coefficient comparison plot: {lm_plot_path}")
    print(f"Saved grid-value round-trip scatter plot: {scatter_plot_path}")
    print(
        f"Round-trip stats: rel_l2={rel_l2:.3e}, max_abs={max_abs:.3e}, coeff_rel={coeff_rel:.3e}"
    )

    assert torch.isfinite(grad_rss).all()
    assert torch.isfinite(grad_rss_roundtrip).all()
    assert coeff_rel < 1e-9
    assert rel_l2 < 0.25
