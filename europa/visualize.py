"""
Visualization utilities for scalar and vector fields on the spherical grid.
Uses matplotlib for 3D scatter/quiver and animation via ffmpeg.
"""
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import warnings
import trimesh


def plot_scalar_on_sphere(values: torch.Tensor, positions: torch.Tensor, title: str = "", save_path: Optional[str] = None):
    """
    Scatter plot of a scalar field on the sphere.
    Args:
        values: [N] scalar values.
        positions: [N,3] Cartesian coords on the sphere.
    """
    vals = values.detach().cpu().numpy()
    pos = positions.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=vals, cmap="rainbow", alpha=0.8)
    fig.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_vector_field(positions: torch.Tensor, vectors: torch.Tensor, title: str = "", scale: float = 1.0, save_path: Optional[str] = None):
    """
    Quiver plot of a vector field on the sphere.
    Args:
        positions: [N,3] positions.
        vectors: [N,3] vectors.
    """
    pos = positions.detach().cpu().numpy()
    vec = vectors.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], vec[:, 0], vec[:, 1], vec[:, 2], length=scale, normalize=True, color="k", alpha=0.7)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def animate_scalar_frames(
    frames: List[torch.Tensor],
    positions: torch.Tensor,
    fps: int = 24,
    save_path: Optional[str] = None,
    title: str = "",
    writer: Optional[str] = None,
):
    """
    Animate scalar fields on the sphere across frames.
    Args:
        frames: list of [N] tensors.
        positions: [N,3] positions.
    """
    pos = positions.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_vals):
        ax.cla()
        vals_np = frame_vals.detach().cpu().numpy()
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=vals_np, cmap="rainbow", alpha=0.8)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(title)
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if save_path:
        chosen_writer = writer
        if chosen_writer is None:
            chosen_writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
        # Pillow writer cannot save mp4; switch to gif if needed
        if chosen_writer == "pillow" and save_path.lower().endswith(".mp4"):
            save_path = save_path.rsplit(".", 1)[0] + ".gif"
        ani.save(save_path, writer=chosen_writer, fps=fps)
    plt.close(fig)


def animate_scalar_surface_pyvista(
    frames: List[torch.Tensor],
    positions: torch.Tensor,
    fps: int = 24,
    save_path: Optional[str] = None,
    title: str = "",
    scale_factor: float = 1.0,
    units: str = "",
    cmap: str = "RdBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    rotate: bool = True,
    revolution_degrees: float = 360.0,
    mesh: Optional["pv.PolyData"] = None,
    frames_are_cell_data: bool = False,
    frame_labels: Optional[List[str]] = None,
):
    """
    Animate scalar fields on the sphere using PyVista (VTK) for seam-free surface rendering.
    If a mesh is provided, frames are assumed to align with that mesh (point or cell data).
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("PyVista is required for this animation; please `pip install pyvista`") from exc

    frames_np = [(f.detach().cpu().numpy() * scale_factor) for f in frames]
    all_vals = np.stack(frames_np)
    vmin_use = np.min(all_vals) if vmin is None else vmin
    vmax_use = np.max(all_vals) if vmax is None else vmax
    if vmax_use - vmin_use < 1e-12:
        vmax_use = vmin_use + 1e-12

    if mesh is None:
        # Build an icosphere (faceted) at the desired radius
        points = positions.detach().cpu().numpy()
        radius = np.linalg.norm(points, axis=1).mean()
        ico = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        vertices = ico.vertices
        faces = ico.faces
        pv_faces = np.hstack([np.c_[np.full(len(faces), 3), faces]]).astype(np.int64)
        mesh = pv.PolyData(vertices, pv_faces)
        # Compute facet centers and reproject to exact radius to avoid radial drift
        face_centers = vertices[faces].mean(axis=1)  # [F,3]
        norm_fc = np.linalg.norm(face_centers, axis=1, keepdims=True) + 1e-12
        face_centers = face_centers / norm_fc * radius
        face_pts = torch.from_numpy(face_centers).to(dtype=torch.float64)
        node_pts = torch.from_numpy(points).to(dtype=torch.float64)
        # Precompute k-NN weights for faces -> nodes (inverse-distance weighting)
        dists = torch.cdist(face_pts, node_pts) + 1e-12
        k = min(6, node_pts.shape[0])
        dist_k, idx_k = torch.topk(dists, k=k, dim=1, largest=False)
        w = 1.0 / (dist_k ** 2)
        w = w / torch.sum(w, dim=1, keepdim=True)

        def sample_frames(frames_np_local):
            sampled = []
            for f_np in frames_np_local:
                vals_grid = torch.from_numpy(f_np).to(dtype=torch.float64)
                vals_sel = vals_grid[idx_k]  # [F,k]
                face_vals = torch.sum(w * vals_sel, dim=1)
                sampled.append(face_vals.cpu().numpy())
            return sampled

        frames_sampled = sample_frames(frames_np)
        frames_np = frames_sampled
        frames_are_cell_data = True

    # Initialize mesh scalars
    if frames_are_cell_data:
        mesh.cell_data["vals"] = frames_np[0]
    else:
        mesh.point_data["vals"] = frames_np[0]

    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    text_actor = plotter.add_text(title, font_size=10)
    plotter.add_mesh(
        mesh,
        scalars="vals",
        cmap=cmap,
        clim=(vmin_use, vmax_use),
        show_scalar_bar=True,
        scalar_bar_args={"title": units},
        interpolate_before_map=False,
        lighting=False,
    )
    plotter.open_gif(save_path if save_path else "b_mag.gif", fps=fps)
    n_frames = len(frames_np)
    for idx, vals_np in enumerate(frames_np):
        if frames_are_cell_data:
            mesh.cell_data["vals"] = vals_np
        else:
            mesh.point_data["vals"] = vals_np
        if frame_labels is not None:
            label = frame_labels[idx] if idx < len(frame_labels) else str(idx)
            if text_actor is not None:
                try:
                    text_actor.SetInput(f"{title} ({label})")
                except Exception:
                    # If direct update is unavailable, remove and recreate the text actor to avoid overdraw.
                    try:
                        plotter.remove_actor(text_actor)
                    except Exception:
                        pass
                    text_actor = plotter.add_text(f"{title} ({label})", font_size=10)
            else:
                text_actor = plotter.add_text(f"{title} ({label})", font_size=10)
        if rotate:
            azim = (idx / max(n_frames, 1)) * revolution_degrees
            plotter.camera.azimuth = azim
        plotter.write_frame()
    plotter.close()


def animate_vector_frames(
    frames: List[torch.Tensor],
    positions: torch.Tensor,
    fps: int = 24,
    save_path: Optional[str] = None,
    title: str = "",
    scale: float = 1.0,
    writer: Optional[str] = None,
):
    """
    Animate vector fields on the sphere across frames.
    Args:
        frames: list of [N,3] tensors.
        positions: [N,3] positions.
    """
    pos = positions.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(vecs):
        ax.cla()
        v_np = vecs.detach().cpu().numpy()
        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], v_np[:, 0], v_np[:, 1], v_np[:, 2], length=scale, normalize=True, color="k", alpha=0.7)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(title)
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if save_path:
        chosen_writer = writer
        if chosen_writer is None:
            chosen_writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
        if chosen_writer == "pillow" and save_path.lower().endswith(".mp4"):
            save_path = save_path.rsplit(".", 1)[0] + ".gif"
        ani.save(save_path, writer=chosen_writer, fps=fps)
    plt.close(fig)


class ProgressReporter:
    """Lightweight progress reporter with callbacks."""
    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.label = label
        self.current = 0

    def step(self, increment: int = 1):
        self.current += increment
        self.report()

    def report(self):
        pct = 100.0 * self.current / max(self.total, 1)
        print(f"[{self.label}] {self.current}/{self.total} ({pct:.1f}%)")


def animate_scalar_surface(
    frames: List[torch.Tensor],
    positions: torch.Tensor,
    fps: int = 24,
    save_path: Optional[str] = None,
    title: str = "",
    alpha: float = 0.8,
    writer: Optional[str] = None,
    scale_factor: float = 1.0,
    units: str = "",
    cmap: str = "rainbow",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    normalize_per_frame: bool = False,
    rotate: bool = False,
    revolution_degrees: float = 360.0,
):
    """
    Animate scalar fields as a colored surface on the sphere using triangulation in (theta, phi).
    Args:
        frames: list of [N] tensors.
        positions: [N,3] positions.
    """
    # Precompute static geometry
    pos_np = positions.detach().cpu().numpy()
    x, y, z = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5 + 1e-12
    theta = torch.arccos(torch.clamp(positions[:, 2] / r, -1.0, 1.0)).cpu().numpy()
    phi = torch.atan2(positions[:, 1], positions[:, 0]).cpu().numpy()
    tri = mtri.Triangulation(phi, theta)

    # Convert frames to numpy to avoid accidental tensor aliasing
    frames_np = [(f.detach().cpu().numpy() * scale_factor) for f in frames]
    all_vals = np.stack(frames_np)
    global_min = np.min(all_vals) if vmin is None else vmin
    global_max = np.max(all_vals) if vmax is None else vmax
    if global_max - global_min < 1e-12:
        global_max = global_min + 1e-12
    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
    cmap_fn = plt.cm.get_cmap(cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label(units)

    def update(idx_frame):
        ax.cla()
        vals_np = frames_np[idx_frame]
        if normalize_per_frame:
            fmin, fmax = vals_np.min(), vals_np.max()
            if fmax - fmin < 1e-12:
                fmax = fmin + 1e-12
            norm_local = mcolors.Normalize(vmin=fmin, vmax=fmax)
            colors = cmap_fn(norm_local(vals_np))
        else:
            colors = cmap_fn(norm(vals_np))
        surf = ax.plot_trisurf(
            x,
            y,
            z,
            triangles=tri.triangles,
            shade=False,
            cmap=cmap,
            alpha=alpha,
            linewidth=0.2,
            antialiased=True,
            facecolors=colors,
        )
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"{title} (frame {idx_frame})")
        ax.set_axis_off()
        if rotate:
            azim = (idx_frame / max(len(frames_np), 1)) * revolution_degrees
            ax.view_init(elev=20, azim=azim)
        return [surf]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    if save_path:
        chosen_writer = writer
        if chosen_writer is None:
            chosen_writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
        if chosen_writer == "pillow" and save_path.lower().endswith(".mp4"):
            save_path = save_path.rsplit(".", 1)[0] + ".gif"
        ani.save(save_path, writer=chosen_writer, fps=fps)
    plt.close(fig)
