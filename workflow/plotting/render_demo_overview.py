"""
Compose overview with harmonic power grouped by degree l (RSS over m)
and matching sphere insets in the upper-right of each harmonic plot.
Outputs one figure with 2 columns x 3 rows (6 panels total).
"""
import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from workflow.data_objects.phasor_data import PhasorSimulation
from workflow.plotting.plot_demo_harmonics import _flatten as _flatten_lm
from workflow.plotting.render_phasor_maps import _scalar_from_sh, _toroidal_vec_mag
from workflow.plotting.sphere_roundtrip import sphere_image, DEFAULT_SPHERE_ELEV, DEFAULT_SPHERE_AZIM
from europa_model import transforms
from europa_model.gradient_utils import rss_gradient_from_emit

OVERVIEW_CACHE_SCHEMA = 2


def _tight_crop_image(img: np.ndarray, white_thresh: int = 245, pad_px: int = 1) -> np.ndarray:
    """Crop near-white borders from an RGB(A) image."""
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    rgb = img[..., :3]
    mask = np.any(rgb < white_thresh, axis=2)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img
    y0 = max(int(ys.min()) - int(pad_px), 0)
    y1 = min(int(ys.max()) + int(pad_px) + 1, img.shape[0])
    x0 = max(int(xs.min()) - int(pad_px), 0)
    x1 = min(int(xs.max()) + int(pad_px) + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def _faces_from_grid_state(grid_state: Optional[dict]) -> Optional[np.ndarray]:
    if not isinstance(grid_state, dict):
        return None
    faces = grid_state.get("faces")
    if faces is None:
        return None
    if isinstance(faces, torch.Tensor):
        return faces.detach().cpu().numpy().astype(np.int64)
    return np.asarray(faces, dtype=np.int64)


def _load_faces(sim: PhasorSimulation, grid_state_path: Optional[str]) -> np.ndarray:
    if grid_state_path:
        grid_state = torch.load(grid_state_path, map_location="cpu", weights_only=False)
        f = _faces_from_grid_state(grid_state)
        if f is not None:
            return f
    pts = sim.grid_positions.detach().cpu().numpy()
    return ConvexHull(pts).simplices.astype(np.int64)


def render_demo_overview(
    data_path: str = "demo_currents.pt",
    subdivisions: int = 0,
    stride: int = 1,
    elev: float = DEFAULT_SPHERE_ELEV,
    azim: float = DEFAULT_SPHERE_AZIM,
    save_path: Optional[str] = "demo_currents_overview.png",
    show: bool = True,
    eps: float = 1e-15,
    grid_state_path: Optional[str] = None,
    plotter: str = "pyvista",
    cache_path: Optional[str] = None,
    cache_deps: Optional[dict] = None,
) -> None:
    print("Overview: loading simulation state...", flush=True)
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    sim = PhasorSimulation.from_saved(raw)
    print(f"Overview: loaded state (lmax={sim.lmax}, n_points={sim.grid_positions.shape[0]}).", flush=True)

    print("Overview: flattening spectral coefficients...", flush=True)
    zeros = torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128)
    B_rad_ph = sim.B_radial if sim.B_radial is not None else zeros
    B_rad_emit_ph = sim.B_rad_emit if sim.B_rad_emit is not None else zeros
    E_tor_ph = sim.E_toroidal if sim.E_toroidal is not None else zeros
    K_tor_ph = sim.K_toroidal if sim.K_toroidal is not None else zeros
    Y_s_spec = sim.admittance_spectral.to(torch.complex128) if sim.admittance_spectral is not None else torch.zeros_like(B_rad_ph)
    omega = float(sim.omega)

    l_b, m_b, mag_b = _flatten_lm(B_rad_ph)
    _, _, mag_e = _flatten_lm(E_tor_ph)
    _, _, mag_k = _flatten_lm(K_tor_ph)
    _, _, mag_bemit = _flatten_lm(B_rad_emit_ph)

    grad_altitude_m = 100e3
    cache_valid = False
    cached = None
    grad_rss = None
    grad_coeffs = None
    cached_sphere_values = None
    print("Overview: preparing gradient RSS data...", flush=True)
    if cache_path:
        try:
            print(f"Overview: checking gradient cache {cache_path}...", flush=True)
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            cache_valid = (
                isinstance(cached, dict)
                and int(cached.get("schema", -1)) == int(OVERVIEW_CACHE_SCHEMA)
                and int(cached.get("lmax", -1)) == int(sim.lmax)
                and int(cached.get("n_points", -1)) == int(sim.grid_positions.shape[0])
                and abs(float(cached.get("omega", -1.0)) - float(sim.omega)) <= 1e-30
                and abs(float(cached.get("radius_m", -1.0)) - float(sim.radius_m)) <= 1e-9
                and int(cached.get("grad_altitude_m", -1)) == int(grad_altitude_m)
            )
            if cache_valid and cache_deps is not None:
                cache_valid = cached.get("deps") == cache_deps
            if cache_valid and cached.get("grad_rss") is not None and cached.get("grad_coeffs") is not None:
                grad_rss = cached["grad_rss"].to(torch.float64)
                grad_coeffs = cached["grad_coeffs"].to(torch.complex128)
                print("Overview: loaded gradient RSS from cache.", flush=True)
                cached_sphere_values = cached.get("sphere_values")
        except Exception:
            cache_valid = False
            cached = None
            grad_rss = None
            grad_coeffs = None

    if grad_rss is None or grad_coeffs is None:
        print("Overview: computing gradient RSS from emitted field...", flush=True)
        obs_scale = float(sim.radius_m + grad_altitude_m) / float(sim.radius_m)
        obs_positions = (sim.grid_positions.to(torch.float64) * obs_scale)
        grad_rss = rss_gradient_from_emit(
            sim,
            obs_positions,
            obs_radius=float(sim.radius_m + grad_altitude_m),
            fd_scheme="forward",
        ).to(torch.float64)
        grad_coeffs = transforms.sh_forward(
            grad_rss,
            sim.grid_positions.to(torch.float64),
            lmax=sim.lmax,
            weights=sim.grid_areas.to(torch.float64),
        ).to(torch.complex128)
        print("Overview: gradient RSS compute complete; projecting to SH done.", flush=True)
    def _save_overview_cache(sphere_values_payload: Optional[dict]) -> None:
        if not cache_path:
            return
        try:
            torch.save(
                {
                    "schema": int(OVERVIEW_CACHE_SCHEMA),
                    "lmax": int(sim.lmax),
                    "n_points": int(sim.grid_positions.shape[0]),
                    "omega": float(sim.omega),
                    "radius_m": float(sim.radius_m),
                    "grad_altitude_m": int(grad_altitude_m),
                    "deps": cache_deps,
                    "grad_rss": grad_rss.cpu() if grad_rss is not None else None,
                    "grad_coeffs": grad_coeffs.cpu() if grad_coeffs is not None else None,
                    "sphere_values": sphere_values_payload,
                },
                cache_path,
            )
            print(f"Overview: saved overview cache to {cache_path}.", flush=True)
        except Exception:
            pass

    if (grad_rss is not None and grad_coeffs is not None) and (
        not cache_valid or not isinstance(cached, dict) or cached.get("grad_rss") is None or cached.get("grad_coeffs") is None
    ):
        _save_overview_cache(cached_sphere_values)

    print("Overview: loading saved grid mesh state...", flush=True)
    points = sim.grid_positions.detach().cpu().to(torch.float64)
    points_np = points.numpy()
    grid_state = None
    if grid_state_path:
        try:
            grid_state = torch.load(grid_state_path, map_location="cpu", weights_only=False)
            print("Overview: loaded grid_admittance state.", flush=True)
        except Exception:
            grid_state = None
            print("Overview: failed to load saved grid state, falling back to ConvexHull.", flush=True)

    print("Overview: preparing sphere faces...", flush=True)
    faces_np = _faces_from_grid_state(grid_state)
    if faces_np is None:
        pts = sim.grid_positions.detach().cpu().numpy()
        faces_np = ConvexHull(pts).simplices.astype(np.int64)
        print("Overview: faces built from ConvexHull.", flush=True)
    else:
        print("Overview: reusing faces saved in Step 2 grid state.", flush=True)

    sigma_real = None
    sigma_spec = None
    if isinstance(grid_state, dict):
        sigma_grid = grid_state.get("sigma_grid")
        if isinstance(sigma_grid, torch.Tensor):
            sigma_real = sigma_grid.to(torch.float64).reshape(-1).cpu().numpy()
            print("Overview: using sigma_grid from saved grid state.", flush=True)
        sigma_spec_candidate = grid_state.get("sigma_spectral")
        if isinstance(sigma_spec_candidate, torch.Tensor):
            sigma_spec = sigma_spec_candidate.to(torch.complex128)
    if sigma_real is None and sigma_spec is not None and isinstance(grid_state, dict):
        print("Overview: reconstructing sigma_s from sigma_spectral on grid...", flush=True)
        sigma_real = _real_field_from_state(
            sigma_spec,
            grid_state.get("positions"),
            grid_state.get("areas"),
        )
    if sigma_real is None and sim.admittance_spectral is not None:
        if isinstance(grid_state, dict):
            print("Overview: conductivity unavailable; fallback to Re(Y_s) for panel 2.", flush=True)
            sigma_real = _real_field_from_state(
                sim.admittance_spectral,
                grid_state.get("positions"),
                grid_state.get("areas"),
            )
        elif grid_state_path:
            print("Overview: conductivity unavailable; fallback to Re(Y_s) from grid_state_path.", flush=True)
            sigma_real = _real_admittance_from_grid(sim.admittance_spectral, grid_state_path)

    print("Overview: aggregating harmonic RSS by degree...", flush=True)
    _, _, mag_grad = _flatten_lm(grad_coeffs.to(torch.complex128))
    if sigma_spec is not None:
        _, _, mag_sigma = _flatten_lm(sigma_spec)
    else:
        _, _, mag_sigma = _flatten_lm(Y_s_spec)
        print("Overview: sigma_spectral missing; panel 2 harmonic RSS uses admittance fallback.", flush=True)

    mags_all = np.stack([mag_b, mag_sigma, mag_e, mag_k, mag_bemit, mag_grad], axis=0)
    nonzero_mask = mags_all > eps
    active_ls = l_b[np.any(nonzero_mask, axis=0)]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    l_vals = np.arange(l_cut + 1)

    def _rss_by_l(mag: np.ndarray) -> np.ndarray:
        out = np.zeros((l_cut + 1,), dtype=np.float64)
        for l in range(l_cut + 1):
            mask = l_b == l
            out[l] = float(np.sqrt(np.sum((mag[mask]) ** 2)))
        return out

    rss_vals = [
        _rss_by_l(mag_sigma),
        _rss_by_l(mag_b),
        _rss_by_l(mag_e),
        _rss_by_l(mag_k),
        _rss_by_l(mag_bemit),
        _rss_by_l(mag_grad),
    ]
    bar_titles = [
        "Surface conductivity",
        "Ambient normal field phasors",
        "Toroidal E phasors",
        "Toroidal current phasors",
        "Emitted normal field phasors",
        "Gradient RSS harmonics by degree l",
    ]
    bar_units = ["S", "T", "V/m", "A/m", "T", "T/m"]
    bar_colors = ["#ff9c43", "#4472c4", "#2ca7a0", "#70ad47", "#c55a11", "#8c564b"]

    sphere_fields = None
    if isinstance(cached_sphere_values, dict):
        required = ("B_r", "sigma_s_real", "E_tor_mag", "K_tor_mag", "B_emit_r", "grad_rss")
        if all(k in cached_sphere_values and cached_sphere_values.get(k) is not None for k in required):
            print("Overview: loaded sphere scalar/vector fields from cache.", flush=True)
            b_r_cached = torch.as_tensor(cached_sphere_values["B_r"], dtype=torch.float64).cpu().numpy()
            sigma_cached = torch.as_tensor(cached_sphere_values["sigma_s_real"], dtype=torch.float64).cpu().numpy()
            e_cached = torch.as_tensor(cached_sphere_values["E_tor_mag"], dtype=torch.float64).cpu().numpy()
            k_cached = torch.as_tensor(cached_sphere_values["K_tor_mag"], dtype=torch.float64).cpu().numpy()
            b_emit_cached = torch.as_tensor(cached_sphere_values["B_emit_r"], dtype=torch.float64).cpu().numpy()
            grad_cached = torch.as_tensor(cached_sphere_values["grad_rss"], dtype=torch.float64).cpu().numpy()
            sphere_fields = [
                ("sigma_s", sigma_cached, "S", False, "rainbow"),
                ("|B_r|", b_r_cached, "T", False, "rainbow"),
                ("|E_tor|", e_cached, "V/m", False, "rainbow"),
                ("|K_tor|", k_cached, "A/m", False, "rainbow"),
                ("|B_emit,r|", b_emit_cached, "T", False, "rainbow"),
                ("|grad_B_emit| RSS @100 km", grad_cached, "T/m", False, "rainbow"),
            ]

    if sphere_fields is None:
        print("Overview: evaluating sphere scalar/vector fields...", flush=True)
        b_r_vals = _scalar_from_sh(B_rad_ph, points)
        sigma_vals = sigma_real if sigma_real is not None else _scalar_from_sh(Y_s_spec, points)
        e_tor_vals = _toroidal_vec_mag(E_tor_ph, points)
        k_tor_vals = _toroidal_vec_mag(K_tor_ph, points)
        b_emit_vals = _scalar_from_sh(B_rad_emit_ph, points)
        grad_vals = grad_rss.cpu().numpy()
        sphere_fields = [
            ("sigma_s", sigma_vals, "S", False, "rainbow"),
            ("|B_r|", b_r_vals, "T", False, "rainbow"),
            ("|E_tor|", e_tor_vals, "V/m", False, "rainbow"),
            ("|K_tor|", k_tor_vals, "A/m", False, "rainbow"),
            ("|B_emit,r|", b_emit_vals, "T", False, "rainbow"),
            ("|grad_B_emit| RSS @100 km", grad_vals, "T/m", False, "rainbow"),
        ]
        _save_overview_cache(
            {
                "B_r": torch.as_tensor(b_r_vals, dtype=torch.float64).cpu(),
                "sigma_s_real": torch.as_tensor(sigma_vals, dtype=torch.float64).cpu(),
                "E_tor_mag": torch.as_tensor(e_tor_vals, dtype=torch.float64).cpu(),
                "K_tor_mag": torch.as_tensor(k_tor_vals, dtype=torch.float64).cpu(),
                "B_emit_r": torch.as_tensor(b_emit_vals, dtype=torch.float64).cpu(),
                "grad_rss": grad_rss.to(torch.float64).cpu(),
            }
        )
    print("Overview: sphere fields prepared. Rendering 6 panels...", flush=True)

    def render_all(fig_save: Optional[str]) -> None:
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.22)

        for row in range(len(bar_titles)):
            panel_idx = row + 1
            total_panels = len(bar_titles)
            print(f"Overview panel {panel_idx}/{total_panels}: {bar_titles[row]}", flush=True)
            r = row // 2
            c = row % 2
            ax_bar = fig.add_subplot(gs[r, c])
            y = rss_vals[row]
            y_plot = np.maximum(y, eps)
            ax_bar.plot(l_vals, y_plot, marker="o", linewidth=1.5, color=bar_colors[row])
            ax_bar.set_yscale("log")
            ax_bar.set_ylabel(f"RSS ({bar_units[row]})")
            ax_bar.set_title(bar_titles[row])
            ax_bar.grid(True, which="both", alpha=0.3)
            if r < 2:
                ax_bar.tick_params(labelbottom=False)
            else:
                ax_bar.set_xlabel("Spherical harmonic degree l")

            field_title, mags, unit, symmetric, cmap = sphere_fields[row]
            img = sphere_image(
                values=np.asarray(mags),
                positions=points_np,
                faces=faces_np,
                title="",
                plotter=plotter,
                cmap=cmap,
                symmetric=symmetric,
                elev=elev,
                azim=azim,
            )
            img = _tight_crop_image(img, white_thresh=245, pad_px=1)
            # Large inset in upper-right corner with minimal internal margins.
            ax_sph = ax_bar.inset_axes([0.50, 0.15, 0.49, 0.82])
            ax_sph.set_facecolor((1.0, 1.0, 1.0, 1.0))
            # Keep sphere aspect while reserving extra right-side whitespace for unit labels.
            # Width remains +12.5% vs original, but shifted left to open label room.
            ax_sph.imshow(img, interpolation="nearest", extent=(0.00, 0.90, 0.10, 0.90))
            ax_sph.set_xlim(0.0, 1.0)
            ax_sph.set_ylim(0.0, 1.0)
            ax_sph.set_aspect("equal", adjustable="box")
            ax_sph.set_xticks([])
            ax_sph.set_yticks([])
            ax_sph.set_frame_on(True)
            # Unit marker placed next to the inset's internal colorbar.
            ax_sph.text(
                0.96,
                0.5,
                unit,
                rotation=90,
                va="center",
                ha="right",
                fontsize=8,
                color="#222222",
                transform=ax_sph.transAxes,
            )
            for spine in ax_sph.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#333333")
                spine.set_linewidth(1.2)

        if fig_save:
            plt.savefig(fig_save, dpi=200, bbox_inches="tight")
            print(f"Saved combined harmonics + phasor maps to {fig_save}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    render_all(save_path)


def main():
    parser = argparse.ArgumentParser(description="Render combined harmonic spectra and phasor sphere maps.")
    parser.add_argument("--input", type=str, default="demo_currents.pt", help="Path to saved demo file.")
    parser.add_argument("--save", type=str, default="demo_currents_overview.png", help="Output image path (None to disable).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    parser.add_argument("--grid-state", type=str, default=None, help="Optional grid state path for mesh/faces and sigma_s map.")
    parser.add_argument("--plotter", choices=("pyvista", "matplotlib"), default="pyvista")
    args = parser.parse_args()
    render_demo_overview(
        data_path=args.input,
        save_path=args.save,
        show=not args.no_show,
        grid_state_path=args.grid_state,
        plotter=args.plotter,
    )


def _real_admittance_from_grid(coeffs: torch.Tensor, grid_state_path: str) -> np.ndarray:
    grid_state = torch.load(grid_state_path, map_location="cpu", weights_only=False)
    vals = transforms.sh_inverse(
        coeffs,
        grid_state["positions"].to(torch.float64),
        grid_state["areas"].to(torch.float64),
    ).reshape(-1)
    return vals.to(torch.complex128).cpu().numpy().real


def _real_field_from_state(
    coeffs: torch.Tensor,
    positions: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
) -> Optional[np.ndarray]:
    if not isinstance(positions, torch.Tensor) or not isinstance(weights, torch.Tensor):
        return None
    vals = transforms.sh_inverse(coeffs, positions.to(torch.float64), weights.to(torch.float64)).reshape(-1)
    return vals.to(torch.complex128).cpu().numpy().real


if __name__ == "__main__":
    main()
