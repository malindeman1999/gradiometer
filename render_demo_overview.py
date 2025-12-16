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
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from phasor_data import PhasorSimulation
from plot_demo_harmonics import _flatten as _flatten_lm
from render_phasor_maps import _build_mesh, _scalar_from_sh, _toroidal_vec_mag


def render_demo_overview(
    data_path: str = "demo_currents.pt",
    subdivisions: int = 2,
    stride: int = 1,
    elev: float = 20.0,
    azim: float = 30.0,
    save_path: Optional[str] = "demo_currents_overview.png",
    show: bool = True,
    eps: float = 1e-15,
) -> None:
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    sim = PhasorSimulation.from_saved(raw)

    # Pull phasors (fallback to zeros if missing)
    zeros = torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128)
    B_rad_ph = sim.B_radial if sim.B_radial is not None else zeros
    B_rad_emit_ph = sim.B_rad_emit if sim.B_rad_emit is not None else zeros
    E_tor_ph = sim.E_toroidal if sim.E_toroidal is not None else zeros
    K_tor_ph = sim.K_toroidal if sim.K_toroidal is not None else zeros
    Y_s_spec = sim.admittance_spectral.to(torch.complex128) if sim.admittance_spectral is not None else torch.zeros_like(B_rad_ph)
    omega = float(sim.omega)
    radius = float(sim.radius_m)

    # Bar spectra (reuse ordering from plot_demo_harmonics)
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

    # Sphere data
    vertices, faces, centers = _build_mesh(radius, subdivisions=subdivisions, stride=max(1, stride))
    tri_verts = vertices[faces].cpu().numpy()
    sphere_fields = [
        ("|B_r|", _scalar_from_sh(B_rad_ph, centers), "T"),
        ("|dB/dt|", omega * _scalar_from_sh(B_rad_ph, centers), "T/s"),
        ("|Y_s|", _scalar_from_sh(Y_s_spec, centers), "S"),
        ("|E_tor|", _toroidal_vec_mag(E_tor_ph, centers), "V/m"),
        ("|K_tor|", _toroidal_vec_mag(K_tor_ph, centers), "A/m"),
        ("|B_emit,r|", _scalar_from_sh(B_rad_emit_ph, centers), "T"),
    ]

    def render_chunk(start_idx: int, end_idx: int, fig_save: Optional[str]) -> None:
        fig = plt.figure(figsize=(14, 12))
        # Hug the columns together so spheres nearly abut the bar charts
        gs = fig.add_gridspec(3, 2, width_ratios=[2.4, 0.9], wspace=0.0, hspace=0.5)
        cmap = plt.get_cmap("inferno")

        for local_row, row in enumerate(range(start_idx, end_idx)):
            # Bar subplot
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

            # Sphere subplot
            ax_sph = fig.add_subplot(gs[local_row, 1], projection="3d")
            mags = sphere_fields[row][1]
            vmax = float(np.max(mags)) if mags.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            collection = Poly3DCollection(
                tri_verts,
                facecolors=cmap(norm(mags)),
                edgecolor="none",
                linewidth=0.05,
                antialiased=True,
            )
            ax_sph.add_collection3d(collection)
            lim = radius * 1.05
            ax_sph.set_xlim(-lim, lim)
            ax_sph.set_ylim(-lim, lim)
            ax_sph.set_zlim(-lim, lim)
            ax_sph.set_box_aspect([1, 1, 1])
            ax_sph.set_axis_off()
            ax_sph.set_title(f"{sphere_fields[row][0]} ({sphere_fields[row][2]})", pad=12)
            ax_sph.view_init(elev=elev, azim=azim)
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(mags)
            fig.colorbar(mappable, ax=ax_sph, shrink=0.7, pad=0.005, label=sphere_fields[row][2])

        if fig_save:
            plt.savefig(fig_save, dpi=200, bbox_inches="tight")
            print(f"Saved combined harmonics + phasor maps to {fig_save}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Determine save paths for the two figures
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
    parser.add_argument("--subdivisions", type=int, default=2, help="Icosphere subdivision level for sampling.")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth face to thin the mesh.")
    parser.add_argument("--elev", type=float, default=20.0, help="Elevation angle for each sphere view.")
    parser.add_argument("--azim", type=float, default=30.0, help="Azimuth angle for each sphere view.")
    parser.add_argument("--save", type=str, default="demo_currents_overview.png", help="Output image path (None to disable).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    args = parser.parse_args()
    render_demo_overview(
        data_path=args.input,
        subdivisions=args.subdivisions,
        stride=max(1, args.stride),
        elev=args.elev,
        azim=args.azim,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
