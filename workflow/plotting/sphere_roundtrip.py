"""
Roundtrip-aligned spherical sampling and sphere rendering helpers.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


def fibonacci_sphere_points(n: int, radius: float) -> torch.Tensor:
    idx = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * idx / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * idx
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = radius * rho * np.cos(phi)
    y = radius * rho * np.sin(phi)
    pts = np.stack((x, y, radius * z), axis=1)
    return torch.tensor(pts, dtype=torch.float64)


def build_roundtrip_grid(lmax: int, radius_m: float, device: str = "cpu") -> dict:
    """Build a quasi-uniform Fibonacci grid with equal solid-angle weights.

    Note: Fibonacci points are not perfectly symmetric, so equal weights
    (4π/N) are an approximation. Integration errors and small spectral
    leakage are expected, but generally decrease as N grows.
    """
    n_points = int((int(lmax) + 1) ** 2)
    positions = fibonacci_sphere_points(n_points, radius=float(radius_m)).to(device=device)
    normals = torch.nn.functional.normalize(positions, dim=-1)
    # SH quadrature weights are solid-angle weights, not physical area.
    # Using equal weights is an approximation for Fibonacci points.
    weights = torch.full((n_points,), 4.0 * math.pi / n_points, dtype=torch.float64, device=device)
    faces_np = ConvexHull(positions.detach().cpu().numpy()).simplices.astype(np.int64)
    faces = torch.from_numpy(faces_np)
    return {
        "positions": positions,
        "normals": normals,
        "areas": weights,
        "neighbors": None,
        "faces": faces,
        "n_points": n_points,
        "n_faces": int(faces.shape[0]),
    }


def _face_values_from_nodes(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vals = np.asarray(values).reshape(-1)
    return vals[faces].mean(axis=1)


def sphere_image(
    values: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    title: str,
    plotter: str = "pyvista",
    cmap: str = "viridis",
    symmetric: bool = False,
    elev: float = 20.0,
    azim: float = 30.0,
) -> np.ndarray:
    face_vals = _face_values_from_nodes(values, faces)
    if symmetric:
        vmax = float(max(np.max(np.abs(face_vals)), 1e-12))
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    else:
        vmin = float(np.min(face_vals))
        vmax = float(np.max(face_vals))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if plotter == "pyvista":
        try:
            import pyvista as pv

            face_prefix = np.full((faces.shape[0], 1), 3, dtype=np.int64)
            pv_faces = np.hstack((face_prefix, faces)).reshape(-1)
            mesh = pv.PolyData(positions, pv_faces)
            mesh.cell_data["value"] = face_vals
            pl = pv.Plotter(off_screen=True, window_size=(900, 700))
            pl.set_background("white")
            pl.add_mesh(mesh, scalars="value", cmap=cmap, show_edges=False, smooth_shading=True, clim=[norm.vmin, norm.vmax])
            pl.add_title(title)
            pl.view_isometric()
            img = pl.screenshot(return_img=True)
            pl.close()
            return np.asarray(img)
        except Exception:
            pass

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    tri_verts = positions[faces]
    map_cmap = plt.get_cmap(cmap)
    poly = Poly3DCollection(tri_verts, linewidths=0.0, edgecolors="none")
    poly.set_facecolor(map_cmap(norm(face_vals)))
    ax.add_collection3d(poly)
    lim = np.max(np.abs(positions))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=map_cmap, norm=norm)
    sm.set_array(face_vals)
    fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def save_or_show_sphere(
    values: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    title: str,
    plotter: str,
    no_show: bool,
    out_path: Optional[Path] = None,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> None:
    img = sphere_image(
        values=values,
        positions=positions,
        faces=faces,
        title=title,
        plotter=plotter,
        cmap=cmap,
        symmetric=symmetric,
    )
    if no_show:
        if out_path is None:
            raise ValueError("out_path is required when no_show=True")
        plt.imsave(out_path, img)
        return
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
