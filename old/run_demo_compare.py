"""
Compare multiple solver variants (uniform/spectral, first-order/self-consistent)
for the same uniform ambient driver. Generates a 6x2 overview:
left = stacked harmonics overlaid for four solvers; right = sphere map (baseline).
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from europa_model.config import GridConfig, ModelConfig
from europa_model.simulation import Simulation
from europa_model import solvers
from old.admittance_check import check_admittance
from workflow.data_objects.phasor_data import PhasorSimulation
from old.analytic_helpers import thin_shell_estimate
from workflow.plotting.plot_demo_harmonics import _flatten as _flatten_lm
from workflow.plotting.render_demo_overview import _build_mesh, _scalar_from_sh, _toroidal_vec_mag
from workflow.ambient_field.ambient_driver import build_ambient_driver_x as build_ambient_driver


def _clone_ps(base: PhasorSimulation) -> PhasorSimulation:
    """Deep-ish clone via serializable dict."""
    return PhasorSimulation.from_serializable(base.to_serializable())


def _compute_mag_fields(sim: PhasorSimulation, eps: float = 1e-15) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return l, m, mag arrays for key fields."""
    zeros = torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128)
    B_rad_ph = sim.B_radial if sim.B_radial is not None else zeros
    B_rad_emit_ph = sim.B_rad_emit if sim.B_rad_emit is not None else zeros
    E_tor_ph = sim.E_toroidal if sim.E_toroidal is not None else zeros
    K_tor_ph = sim.K_toroidal if sim.K_toroidal is not None else zeros
    Y_s_spec = sim.admittance_spectral if sim.admittance_spectral is not None else torch.zeros_like(B_rad_ph.real)
    l_b, m_b, mag_b = _flatten_lm(B_rad_ph)
    _, _, mag_dbdt = l_b, m_b, float(sim.omega) * mag_b
    _, _, mag_y = _flatten_lm(Y_s_spec.to(torch.complex128))
    _, _, mag_e = _flatten_lm(E_tor_ph)
    _, _, mag_k = _flatten_lm(K_tor_ph)
    _, _, mag_bemit = _flatten_lm(B_rad_emit_ph)
    return {
        "B": (l_b, m_b, mag_b),
        "dBdt": (l_b, m_b, mag_dbdt),
        "Y": (l_b, m_b, mag_y),
        "E": (l_b, m_b, mag_e),
        "K": (l_b, m_b, mag_k),
        "Bemit": (l_b, m_b, mag_bemit),
    }


def run_variants() -> Tuple[List[Tuple[str, PhasorSimulation]], Simulation, float, torch.Tensor]:
    """Run four solver variants and return sims plus grid/sigma."""
    # Build grid and ambient
    grid_cfg = GridConfig(nside=4, lmax=6, radius_m=1.56e6, device="cpu")
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver(grid_cfg)
    omega = ambient_cfg.omega_jovian
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    sigma_2d = float(sim.grid.surface_conductivity_s)

    # Baseline PhasorSimulation template
    Y_s_spectral = torch.zeros((grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1), dtype=torch.float64)
    Y_s_spectral[0, grid_cfg.lmax] = sigma_2d * (4 * torch.pi) ** 0.5
    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=sigma_2d,
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec.cpu(),
        period_sec=period_sec,
    )

    variants = [
        ("uniform_first_order", solvers.solve_uniform_first_order_sim),
        ("spectral_first_order", solvers.solve_spectral_first_order_sim),
        ("uniform_self_consistent", solvers.solve_uniform_self_consistent_sim),
        ("spectral_self_consistent", solvers.solve_spectral_self_consistent_sim),
    ]

    sims: List[Tuple[str, PhasorSimulation]] = []
    for name, fn in variants:
        ps = _clone_ps(base)
        fn(ps)
        ps.solver_variant = name
        sims.append((name, ps))

    return sims, sim, sigma_2d, B_radial_spec


def plot_comparison(
    sims: List[Tuple[str, PhasorSimulation]],
    grid_radius: float,
    omega: float,
    save_path: str,
    show: bool = False,
    eps: float = 1e-15,
) -> None:
    """
    Create 2 figures (3x2 each) with overlaid harmonics (4 solvers) and baseline spheres.
    """
    # Use first solver as baseline for sphere maps
    _, baseline = sims[0]
    mag_fields = {name: _compute_mag_fields(ps, eps=eps) for name, ps in sims}

    # Build common l/m and truncation mask
    l_ref, m_ref, _ = next(iter(mag_fields.values()))["B"]
    mags_all = []
    for fields in mag_fields.values():
        for key in ["B", "dBdt", "Y", "E", "K", "Bemit"]:
            mags_all.append(fields[key][2])
    mags_all = np.stack(mags_all, axis=0)
    nonzero_mask = mags_all > eps
    active_ls = l_ref[np.any(nonzero_mask, axis=0)]
    l_cut = int(active_ls.max()) if active_ls.size else 1
    l_cut = max(l_cut, 1)
    keep = l_ref <= l_cut
    labels = [f"({l},{m})" for l, m in zip(l_ref[keep], m_ref[keep])]
    x = np.arange(len(labels))

    # Sphere data from baseline
    vertices, faces, centers = _build_mesh(grid_radius, subdivisions=2, stride=1)
    tri_verts = vertices[faces].cpu().numpy()
    zeros = torch.zeros((baseline.lmax + 1, 2 * baseline.lmax + 1), dtype=torch.complex128)
    B_rad_ph = baseline.B_radial if baseline.B_radial is not None else zeros
    B_rad_emit_ph = baseline.B_rad_emit if baseline.B_rad_emit is not None else zeros
    E_tor_ph = baseline.E_toroidal if baseline.E_toroidal is not None else zeros
    K_tor_ph = baseline.K_toroidal if baseline.K_toroidal is not None else zeros
    Y_s_spec = baseline.admittance_spectral if baseline.admittance_spectral is not None else torch.zeros_like(B_rad_ph.real)
    sphere_fields = [
        ("|B_r|", _scalar_from_sh(B_rad_ph, centers)),
        ("|dB/dt|", omega * _scalar_from_sh(B_rad_ph, centers)),
        ("|Y_s|", _scalar_from_sh(Y_s_spec.to(torch.complex128), centers)),
        ("|E_tor|", _toroidal_vec_mag(E_tor_ph, centers)),
        ("|K_tor|", _toroidal_vec_mag(K_tor_ph, centers)),
        ("|B_emit,r|", _scalar_from_sh(B_rad_emit_ph, centers)),
    ]

    qty_order = [("B", "|B_r|"), ("dBdt", "|dB/dt|"), ("Y", "|Y_s|"), ("E", "|E_tor|"), ("K", "|K_tor|"), ("Bemit", "|B_emit,r|")]
    colors = ["#4472c4", "#c55a11", "#70ad47", "#ff6fb7"]
    cmap = plt.get_cmap("inferno")

    def render_chunk(start_idx: int, end_idx: int, fig_save: str):
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1.0], wspace=0.08, hspace=0.6)
        fig_save_path = Path(fig_save)
        fig_save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_save_path.unlink(missing_ok=True)
        for local_row, row in enumerate(range(start_idx, end_idx)):
            key, title = qty_order[row]
            ax_bar = fig.add_subplot(gs[local_row, 0])
            width = 0.18
            for idx, (name, _) in enumerate(sims):
                l, m, mags = mag_fields[name][key]
                mags = mags[keep]
                offset = (idx - 1.5) * width
                ax_bar.bar(x + offset, mags, width=width, color=colors[idx], alpha=0.85, label=name.replace("_", " "), align="center")
            if row == 0:
                ax_bar.legend(loc="upper right", fontsize=10)
            ax_bar.set_ylabel(title)
            ax_bar.set_title(title)
            ax_bar.set_ylim(bottom=0.0)
            if local_row < 2:
                ax_bar.tick_params(labelbottom=False)
            else:
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(labels, rotation=90)
                ax_bar.set_xlabel("(l,m) up to active l")

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
            lim = grid_radius * 1.05
            ax_sph.set_xlim(-lim, lim)
            ax_sph.set_ylim(-lim, lim)
            ax_sph.set_zlim(-lim, lim)
            ax_sph.set_box_aspect([1, 1, 1])
            ax_sph.set_axis_off()
            ax_sph.set_title(f"{title} (baseline)", pad=12)
            ax_sph.view_init(elev=20.0, azim=30.0)
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(mags)
            fig.colorbar(mappable, ax=ax_sph, shrink=0.7, pad=0.01)

        plt.tight_layout()
        plt.savefig(str(fig_save_path), dpi=200, bbox_inches="tight")
        os.utime(fig_save_path, None)
        print(f"Saved solver comparison overview to {fig_save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    save_path = str(save_path)
    if save_path.lower().endswith(".png"):
        save_path1 = save_path
        save_path2 = save_path.replace(".png", "_part2.png")
    else:
        save_path1 = save_path
        save_path2 = f"{save_path}_part2"

    render_chunk(0, 3, save_path1)
    render_chunk(3, 6, save_path2)


def main():
    parser = argparse.ArgumentParser(description="Compare solver variants on the same uniform driver.")
    parser.add_argument("--save", type=str, default="figures/demo_solver_compare.png", help="Output figure path.")
    parser.add_argument("--skip-check", action="store_true", help="Skip admittance check on the last solver.")
    args = parser.parse_args()

    sims, sim, sigma_2d, B_radial_spec = run_variants()
    # Admittance check on last (most complex) solver
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    torch.save({"phasor_sim": sims[-1][1]}, data_dir / "demo_compare_last.pt")
    if not args.skip_check:
        try:
            check_admittance(str(data_dir / "demo_compare_last.pt"))
        except Exception as exc:
            print(f"Admittance check failed: {exc}")

    plot_comparison(
        sims=sims,
        grid_radius=float(sim.config.grid.radius_m),
        omega=float(sim.config.ambient.omega_jovian),
        save_path=args.save,
        show=False,
    )


if __name__ == "__main__":
    main()
