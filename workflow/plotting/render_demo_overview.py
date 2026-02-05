"""
Compose paired overviews: harmonic bar spectra on the left and matching
phasor sphere maps on the right for each quantity. Outputs two figures
(3 rows each) covering six quantities.
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
from workflow.plotting.sphere_roundtrip import sphere_image
from europa_model import transforms


def _load_faces(sim: PhasorSimulation, grid_state_path: Optional[str]) -> np.ndarray:
    if grid_state_path:
        grid_state = torch.load(grid_state_path, map_location="cpu", weights_only=False)
        faces = grid_state.get("faces")
        if faces is not None:
            if isinstance(faces, torch.Tensor):
                return faces.detach().cpu().numpy().astype(np.int64)
            return np.asarray(faces, dtype=np.int64)
    pts = sim.grid_positions.detach().cpu().numpy()
    return ConvexHull(pts).simplices.astype(np.int64)


def render_demo_overview(
    data_path: str = "demo_currents.pt",
    subdivisions: int = 0,
    stride: int = 1,
    elev: float = 20.0,
    azim: float = 30.0,
    save_path: Optional[str] = "demo_currents_overview.png",
    show: bool = True,
    eps: float = 1e-15,
    grid_state_path: Optional[str] = None,
    plotter: str = "pyvista",
) -> None:
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    sim = PhasorSimulation.from_saved(raw)

    zeros = torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128)
    B_rad_ph = sim.B_radial if sim.B_radial is not None else zeros
    B_rad_emit_ph = sim.B_rad_emit if sim.B_rad_emit is not None else zeros
    E_tor_ph = sim.E_toroidal if sim.E_toroidal is not None else zeros
    K_tor_ph = sim.K_toroidal if sim.K_toroidal is not None else zeros
    Y_s_spec = sim.admittance_spectral.to(torch.complex128) if sim.admittance_spectral is not None else torch.zeros_like(B_rad_ph)
    omega = float(sim.omega)

    l_b, m_b, mag_b = _flatten_lm(B_rad_ph)
    _, _, mag_dbdt = l_b, m_b, omega * mag_b
    _, _, mag_y = _flatten_lm(Y_s_spec)
    _, _, mag_e = _flatten_lm(E_tor_ph)
    _, _, mag_k = _flatten_lm(K_tor_ph)
    _, _, mag_bemit = _flatten_lm(B_rad_emit_ph)
    mags_all = np.stack([mag_b, mag_dbdt, mag_y, mag_e, mag_k, mag_bemit], axis=0)
    nonzero_mask = mags_all > eps
    active_ls = l_b[np.any(nonzero_mask, axis=0)]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    keep = l_b <= l_cut

    labels = [f"({l},{m})" for l, m in zip(l_b[keep], m_b[keep])]
    x = np.arange(len(labels))
    bar_vals = [
        mag_b[keep],
        mag_dbdt[keep],
        mag_y[keep],
        mag_e[keep],
        mag_k[keep],
        mag_bemit[keep],
    ]
    bar_titles = [
        "Ambient normal field phasors (T)",
        "Normal dB/dt phasors (T/s)",
        "Surface admittance |Y_s| (S)",
        "Toroidal E phasors (V/m)",
        "Toroidal current phasors (A/m)",
        "Emitted normal field phasors (T)",
    ]
    bar_colors = ["#4472c4", "#5c9bd5", "#ff9c43", "#2ca7a0", "#70ad47", "#c55a11"]

    points = sim.grid_positions.detach().cpu().to(torch.float64)
    points_np = points.numpy()
    faces_np = _load_faces(sim, grid_state_path)

    y_real = None
    if grid_state_path and sim.admittance_spectral is not None:
        y_real = _real_admittance_from_grid(sim.admittance_spectral, grid_state_path)

    sphere_fields = [
        ("|B_r|", _scalar_from_sh(B_rad_ph, points), "T", False, "inferno"),
        ("|dB/dt|", omega * _scalar_from_sh(B_rad_ph, points), "T/s", False, "inferno"),
        ("Re(Y_s)", y_real if y_real is not None else _scalar_from_sh(Y_s_spec, points), "S", True, "coolwarm"),
        ("|E_tor|", _toroidal_vec_mag(E_tor_ph, points), "V/m", False, "inferno"),
        ("|K_tor|", _toroidal_vec_mag(K_tor_ph, points), "A/m", False, "inferno"),
        ("|B_emit,r|", _scalar_from_sh(B_rad_emit_ph, points), "T", False, "inferno"),
    ]

    def render_chunk(start_idx: int, end_idx: int, fig_save: Optional[str]) -> None:
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, width_ratios=[2.4, 0.9], wspace=0.05, hspace=0.45)

        for local_row, row in enumerate(range(start_idx, end_idx)):
            ax_bar = fig.add_subplot(gs[local_row, 0])
            ax_bar.bar(x, bar_vals[row], color=bar_colors[row])
            ax_bar.set_ylabel("Magnitude")
            ax_bar.set_title(bar_titles[row])
            if local_row < 2:
                ax_bar.tick_params(labelbottom=False)
            else:
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(labels, rotation=90)
                ax_bar.set_xlabel("(l,m) up to active l (min l=1)")

            ax_sph = fig.add_subplot(gs[local_row, 1])
            field_title, mags, unit, symmetric, cmap = sphere_fields[row]
            img = sphere_image(
                values=np.asarray(mags),
                positions=points_np,
                faces=faces_np,
                title=f"{field_title} ({unit})",
                plotter=plotter,
                cmap=cmap,
                symmetric=symmetric,
            )
            ax_sph.imshow(img)
            ax_sph.set_title(f"{field_title} ({unit})")
            ax_sph.axis("off")

        if fig_save:
            plt.savefig(fig_save, dpi=200, bbox_inches="tight")
            print(f"Saved combined harmonics + phasor maps to {fig_save}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    if save_path and save_path.lower().endswith(".png"):
        save_path1 = save_path
        save_path2 = save_path.replace(".png", "_part2.png")
    else:
        save_path1 = save_path
        save_path2 = None if save_path is None else f"{save_path}_part2"

    render_chunk(0, 3, save_path1)
    render_chunk(3, 6, save_path2)


def main():
    parser = argparse.ArgumentParser(description="Render combined harmonic spectra and phasor sphere maps.")
    parser.add_argument("--input", type=str, default="demo_currents.pt", help="Path to saved demo file.")
    parser.add_argument("--save", type=str, default="demo_currents_overview.png", help="Output image path (None to disable).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    parser.add_argument("--grid-state", type=str, default=None, help="Optional grid state path for mesh/faces and Y_s real map.")
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
    positions = grid_state["positions"].to(torch.float64)
    weights = grid_state["areas"].to(torch.float64)
    vals = transforms.sh_inverse(coeffs, positions, weights).reshape(-1)
    return vals.to(torch.complex128).cpu().numpy().real


if __name__ == "__main__":
    main()
