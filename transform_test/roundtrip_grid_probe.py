"""
Standalone transform roundtrip probe for conductivity harmonics.

This script:
1) Uses the default non-uniform demo conductivity harmonic settings.
2) Builds the minimum-point spherical sample for SH coefficient recovery.
3) Runs SH inverse -> SH forward roundtrip checks.
4) Plots original vs roundtrip harmonic magnitudes on log scale.
5) Builds a surface triangulation and plots conductivity heatmap on the sphere.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from europa_model.config import GridConfig
from europa_model.transforms import sh_forward, sh_inverse


def fibonacci_sphere_points(n: int, radius: float = 1.0) -> torch.Tensor:
    idx = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * idx / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * idx
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = radius * rho * np.cos(phi)
    y = radius * rho * np.sin(phi)
    pts = np.stack((x, y, radius * z), axis=1)
    return torch.tensor(pts, dtype=torch.float64)


def synthesize_sigma_field(
    positions: torch.Tensor,
    weights: torch.Tensor,
    lmax: int,
    mean_cond: float,
    frac_rms: float,
    mode_l: int,
    mode_m: int,
    phase_rad: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode_l = int(max(0, min(mode_l, lmax)))
    mode_m = int(max(0, min(abs(mode_m), mode_l)))
    frac_rms = max(0.0, float(frac_rms))

    delta_coeffs = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    c = math.cos(phase_rad) + 1j * math.sin(phase_rad)
    delta_coeffs[mode_l, lmax + mode_m] = c
    delta_coeffs[mode_l, lmax - mode_m] = ((-1) ** mode_m) * np.conj(c)

    delta = sh_inverse(delta_coeffs, positions, weights)
    delta = delta.real - delta.real.mean()
    current_rms = float(torch.sqrt((delta * delta).mean()).item())
    target_rms = float(mean_cond) * frac_rms
    if current_rms > 0.0 and target_rms > 0.0:
        delta_coeffs = delta_coeffs * (target_rms / current_rms)
    else:
        delta_coeffs = torch.zeros_like(delta_coeffs)

    sigma_coeffs = delta_coeffs.clone()
    sigma_coeffs[0, lmax] = float(mean_cond) * (2.0 * math.sqrt(math.pi))
    sigma = sh_inverse(sigma_coeffs, positions, weights)
    return sigma, sigma_coeffs


def flatten_harmonics(coeffs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lmax = coeffs.shape[-2] - 1
    l_list: list[int] = []
    m_list: list[int] = []
    mag_list: list[float] = []
    arr = coeffs.detach().cpu()
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            l_list.append(l)
            m_list.append(m)
            mag_list.append(float(torch.abs(arr[l, lmax + m]).item()))
    return np.array(l_list), np.array(m_list), np.array(mag_list)


def _gpu_active_from_caps(caps: str) -> bool:
    low = caps.lower()
    software_markers = ("llvmpipe", "softpipe", "software", "swiftshader", "mesa offscreen")
    return not any(marker in low for marker in software_markers)


def _report_pyvista_gpu() -> None:
    import pyvista as pv

    probe = pv.Plotter(off_screen=True, window_size=(320, 240))
    probe.add_mesh(pv.Sphere(radius=0.5))
    probe.show(auto_close=False)
    caps = ""
    if hasattr(probe, "ren_win") and probe.ren_win is not None:
        try:
            caps = str(probe.ren_win.ReportCapabilities())
        except Exception:  # noqa: BLE001
            caps = ""
    probe.close()

    renderer_line = next((ln.strip() for ln in caps.splitlines() if "OpenGL renderer string:" in ln), "").strip()
    vendor_line = next((ln.strip() for ln in caps.splitlines() if "OpenGL vendor string:" in ln), "").strip()
    print(vendor_line if vendor_line else "OpenGL vendor string: <unavailable>")
    print(renderer_line if renderer_line else "OpenGL renderer string: <unavailable>")
    if vendor_line and renderer_line:
        gpu_ok = _gpu_active_from_caps(caps)
        print(f"PyVista GPU acceleration check: {'PASS' if gpu_ok else 'FAIL'}")
    else:
        print("PyVista GPU acceleration check: UNKNOWN (OpenGL capability text unavailable)")


def _plot_sphere_matplotlib(pts: np.ndarray, faces: np.ndarray, vals: np.ndarray, title: str, no_show: bool, out_path: Path) -> None:
    face_vals = vals[faces].mean(axis=1)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=float(face_vals.min()), vmax=float(face_vals.max()))
    fig_s = plt.figure(figsize=(8, 7))
    ax_s = fig_s.add_subplot(111, projection="3d")
    poly = Poly3DCollection(pts[faces], linewidths=0.0, edgecolors="none")
    poly.set_facecolor(cmap(norm(face_vals)))
    ax_s.add_collection3d(poly)
    lim = np.max(np.abs(pts))
    ax_s.set_xlim(-lim, lim)
    ax_s.set_ylim(-lim, lim)
    ax_s.set_zlim(-lim, lim)
    ax_s.set_box_aspect((1, 1, 1))
    ax_s.set_axis_off()
    ax_s.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(face_vals)
    fig_s.colorbar(sm, ax=ax_s, shrink=0.75, pad=0.02, label="Conductivity (S)")
    fig_s.tight_layout()
    if no_show:
        fig_s.savefig(out_path, dpi=150)
    else:
        plt.show()


def _plot_sphere_pyvista(pts: np.ndarray, faces: np.ndarray, vals: np.ndarray, title: str, no_show: bool, out_path: Path) -> bool:
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001
        print(f"PyVista unavailable ({exc}); falling back to Matplotlib sphere plot.")
        _plot_sphere_matplotlib(pts, faces, vals, title, no_show, out_path)
        return False

    face_prefix = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    pv_faces = np.hstack((face_prefix, faces)).reshape(-1)
    mesh = pv.PolyData(pts, pv_faces)
    mesh.point_data["conductivity"] = vals

    plotter = pv.Plotter(off_screen=bool(no_show), window_size=(1100, 900))
    plotter.add_mesh(mesh, scalars="conductivity", cmap="viridis", smooth_shading=True, show_edges=False)
    plotter.add_axes()
    plotter.add_title(title)
    plotter.view_isometric()

    _report_pyvista_gpu()

    if no_show:
        plotter.show(auto_close=False)
        plotter.screenshot(str(out_path))
        plotter.close()
    else:
        # Interactive path: user closes the window; do not access ren_win afterward.
        plotter.show(auto_close=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmax", type=int, default=36)
    parser.add_argument("--mode-l", type=int, default=10)
    parser.add_argument("--mode-m", type=int, default=2)
    parser.add_argument("--frac-rms", type=float, default=0.05)
    parser.add_argument("--phase-rad", type=float, default=0.0)
    parser.add_argument("--tol-imag", type=float, default=1e-10)
    parser.add_argument("--tol-rel", type=float, default=1e-10)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plotter", choices=("pyvista", "matplotlib"), default="pyvista")
    parser.add_argument("--save-dir", type=Path, default=Path("transform_test/artifacts"))
    args = parser.parse_args()

    default_cfg = GridConfig(nside=1, lmax=1, radius_m=1.56e6, device="cpu")
    mean_cond = 2.0 * default_cfg.seawater_conductivity_s_per_m * default_cfg.ocean_thickness_m

    # Minimum point count to uniquely recover complex SH coefficients up to lmax.
    n_points = (args.lmax + 1) ** 2
    positions = fibonacci_sphere_points(n_points)
    weights = torch.full((n_points,), 4.0 * math.pi / n_points, dtype=torch.float64)

    sigma, sigma_coeff_target = synthesize_sigma_field(
        positions=positions,
        weights=weights,
        lmax=args.lmax,
        mean_cond=mean_cond,
        frac_rms=args.frac_rms,
        mode_l=args.mode_l,
        mode_m=args.mode_m,
        phase_rad=args.phase_rad,
    )

    imag_max = float(torch.max(torch.abs(sigma.imag)).item())
    sigma_real = sigma.real
    sigma_coeff_roundtrip = sh_forward(sigma_real, positions, lmax=args.lmax, weights=weights)
    sigma_roundtrip = sh_inverse(sigma_coeff_roundtrip, positions, weights)

    coeff_diff = sigma_coeff_roundtrip - sigma_coeff_target
    coeff_rel_l2 = float(torch.linalg.norm(coeff_diff).item() / torch.linalg.norm(sigma_coeff_target).item())
    coeff_max_abs = float(torch.max(torch.abs(coeff_diff)).item())
    field_rel_l2 = float(
        torch.linalg.norm((sigma_roundtrip.real - sigma_real)).item() / torch.linalg.norm(sigma_real).item()
    )

    print("=== Transform Roundtrip Probe ===")
    print(f"lmax={args.lmax}, n_points={n_points}, n_coeff={(args.lmax + 1) ** 2}")
    coeff_count = (args.lmax + 1) ** 2
    print(f"point-to-harmonic ratio={n_points / coeff_count:.6f} (points / coeffs)")
    print(f"default mean conductivity={mean_cond:.6e} S")
    print(f"mode=(l={args.mode_l}, |m|={args.mode_m}), frac_rms={args.frac_rms:.2%}, phase={args.phase_rad:.3f} rad")
    print(f"max|imag(sigma_grid)|={imag_max:.3e} (tol={args.tol_imag:.1e})")
    print(f"coeff rel-L2 error={coeff_rel_l2:.3e} (tol={args.tol_rel:.1e})")
    print(f"coeff max|delta|={coeff_max_abs:.3e}")
    print(f"field rel-L2 error={field_rel_l2:.3e}")
    print(f"real-valued check: {'PASS' if imag_max <= args.tol_imag else 'FAIL'}")
    print(f"roundtrip check: {'PASS' if coeff_rel_l2 <= args.tol_rel else 'FAIL'}")

    l_idx, m_idx, mag_target = flatten_harmonics(sigma_coeff_target)
    _, _, mag_rt = flatten_harmonics(sigma_coeff_roundtrip)
    active = (mag_target > 0.0) | (mag_rt > 0.0)
    x = np.arange(int(np.sum(active)))
    tick_idx = np.where(m_idx[active] == 0)[0]
    tick_labels = [f"({l},0)" for l in l_idx[active][tick_idx]]

    fig_h, ax_h = plt.subplots(figsize=(9, 4.5))
    ax_h.plot(x, np.maximum(mag_target[active], 1e-30), lw=2.0, label="Original coeffs")
    ax_h.plot(x, np.maximum(mag_rt[active], 1e-30), lw=1.5, ls="--", label="Roundtrip coeffs")
    ax_h.set_yscale("log")
    ax_h.set_xlabel("(l,m) ordering (ticks at m=0)")
    ax_h.set_ylabel("|sigma_lm|")
    ax_h.set_xticks(tick_idx)
    ax_h.set_xticklabels(tick_labels, rotation=90)
    ax_h.set_title("Conductivity Harmonics: Original vs Roundtrip")
    ax_h.grid(True, which="both", alpha=0.3)
    ax_h.legend(frameon=False)
    fig_h.tight_layout()

    pts = positions.detach().cpu().numpy()
    vals = sigma_real.detach().cpu().numpy().reshape(-1)
    hull = ConvexHull(pts)
    faces = hull.simplices
    print(f"face-to-harmonic ratio={faces.shape[0] / coeff_count:.6f} (faces / coeffs)")

    if args.no_show:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        harmonics_path = args.save_dir / "harmonics_roundtrip.png"
        sphere_path = args.save_dir / "solver_grid_heatmap.png"
        fig_h.savefig(harmonics_path, dpi=150)
        if args.plotter == "pyvista":
            _plot_sphere_pyvista(
                pts,
                faces,
                vals,
                "Conductivity Heatmap on Triangulated Solver Grid (PyVista)",
                no_show=True,
                out_path=sphere_path,
            )
        else:
            _plot_sphere_matplotlib(
                pts,
                faces,
                vals,
                "Conductivity Heatmap on Triangulated Solver Grid (Matplotlib)",
                no_show=True,
                out_path=sphere_path,
            )
        print(f"Saved plots to: {harmonics_path} and {sphere_path}")
    else:
        plt.show(block=False)
        if args.plotter == "pyvista":
            _plot_sphere_pyvista(
                pts,
                faces,
                vals,
                "Conductivity Heatmap on Triangulated Solver Grid (PyVista)",
                no_show=False,
                out_path=Path("unused.png"),
            )
        else:
            _plot_sphere_matplotlib(
                pts,
                faces,
                vals,
                "Conductivity Heatmap on Triangulated Solver Grid (Matplotlib)",
                no_show=False,
                out_path=Path("unused.png"),
            )
        plt.show()


if __name__ == "__main__":
    main()
