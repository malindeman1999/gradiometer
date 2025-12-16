"""
Render stacked sphere maps for key phasor magnitudes:
|B_r|, |dB/dt| (radial), |E_tor|, |K_tor|, |B_emit,r|.
"""
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from phasor_data import PhasorSimulation
from europa import transforms


def _build_mesh(radius: float, subdivisions: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices = torch.from_numpy(ico.vertices).to(dtype=torch.float64)
    faces = torch.from_numpy(ico.faces).to(dtype=torch.long)
    if stride > 1:
        faces = faces[::stride]
    tri_pts = vertices[faces]  # [F,3,3]
    centers = tri_pts.mean(dim=1)  # [F,3]
    return vertices, faces, centers


def _scalar_from_sh(coeffs: torch.Tensor, points: torch.Tensor) -> np.ndarray:
    """Evaluate scalar SH field at given points and return magnitude."""
    lmax = coeffs.shape[-2] - 1
    weights = torch.ones(points.shape[0], dtype=torch.float64, device=points.device)
    vals = transforms.sh_inverse(coeffs, points, weights)  # complex
    return torch.abs(vals).cpu().numpy()


def _toroidal_vec_mag(coeffs: torch.Tensor, points: torch.Tensor) -> np.ndarray:
    """Evaluate toroidal VSH field and return |vector| at points."""
    lmax = coeffs.shape[-2] - 1
    basis_tor, _ = transforms._compute_vsh_basis(points, lmax)
    vec = np.zeros((points.shape[0], 3), dtype=np.complex128)
    tor_np = coeffs.detach().cpu().numpy()
    for l in range(lmax + 1):
        for m_idx in range(2 * l + 1):
            vec += tor_np[l, lmax - l + m_idx] * basis_tor[l, lmax - l + m_idx]
    return np.linalg.norm(vec, axis=1)


def render_phasor_maps(
    data_path: str = "demo_currents.pt",
    subdivisions: int = 2,
    stride: int = 1,
    elev: float = 20.0,
    azim: float = 30.0,
    save_path: Optional[str] = "phasor_maps.png",
):
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    sim = PhasorSimulation.from_saved(raw)
    radius = float(sim.radius_m)
    vertices, faces, centers = _build_mesh(radius, subdivisions=subdivisions, stride=max(1, stride))

    # Quantities
    B_rad_ph = sim.B_radial if sim.B_radial is not None else torch.zeros((sim.lmax + 1, 2 * sim.lmax + 1), dtype=torch.complex128)
    B_rad_emit_ph = sim.B_rad_emit if sim.B_rad_emit is not None else torch.zeros_like(B_rad_ph)
    E_tor_ph = sim.E_toroidal if sim.E_toroidal is not None else torch.zeros_like(B_rad_ph)
    K_tor_ph = sim.K_toroidal if sim.K_toroidal is not None else torch.zeros_like(B_rad_ph)
    omega = float(sim.omega)

    scalars = [
        ("|B_r|", _scalar_from_sh(B_rad_ph, centers)),
        ("|dB/dt|", omega * _scalar_from_sh(B_rad_ph, centers)),
        ("|E_tor|", _toroidal_vec_mag(E_tor_ph, centers)),
        ("|K_tor|", _toroidal_vec_mag(K_tor_ph, centers)),
        ("|B_emit,r|", _scalar_from_sh(B_rad_emit_ph, centers)),
    ]

    nplots = len(scalars)
    fig = plt.figure(figsize=(10, 2 * nplots))
    cmap = plt.get_cmap("inferno")
    for idx, (title, mags) in enumerate(scalars, start=1):
        ax = fig.add_subplot(nplots, 1, idx, projection="3d")
        vmax = float(np.max(mags)) if mags.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        vmin = 0.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        face_colors = cmap(norm(mags))
        tri_verts = vertices[faces].cpu().numpy()
        collection = Poly3DCollection(
            tri_verts,
            facecolors=face_colors,
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
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(mags)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved phasor sphere maps to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Render stacked sphere maps of key phasor magnitudes.")
    parser.add_argument("--input", type=str, default="demo_currents.pt", help="Path to saved demo file.")
    parser.add_argument("--subdivisions", type=int, default=2, help="Icosphere subdivision level for sampling.")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth face to thin the mesh.")
    parser.add_argument("--elev", type=float, default=20.0, help="Elevation angle for each view.")
    parser.add_argument("--azim", type=float, default=30.0, help="Azimuth angle for each view.")
    parser.add_argument("--save", type=str, default="phasor_maps.png", help="Output image path (None to disable).")
    args = parser.parse_args()
    render_phasor_maps(
        data_path=args.input,
        subdivisions=args.subdivisions,
        stride=max(1, args.stride),
        elev=args.elev,
        azim=args.azim,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
