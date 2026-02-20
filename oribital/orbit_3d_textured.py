"""Render a 3D Europa orbit with a textured surface and repeated pixel crossings.

This mirrors the ground-track crossing logic in orbit_estimates.py and shows
the first N crossings through the same ground-pixel box in body-fixed space.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Ensure repo root is on sys.path when running this module directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oribital.orbit_estimates import (
    DEFAULT_ALTITUDE_M,
    DEFAULT_LMAX,
    EUROPA_MU_M3_S2,
    EUROPA_RADIUS_M,
    estimate_polar_orbit,
)
from workflow.plotting.sphere_roundtrip import build_roundtrip_grid

# NASA Photojournal Europa globe map (PIA03526).
DEFAULT_TEXTURE_URL = (
    "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/photojournal/pia/"
    "pia03/pia03526/PIA03526.jpg?crop=faces%2Cfocalpoint&fit=clip&h=1024&w=2048"
)


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


def _download_texture_if_needed(texture_path: Path, url: str) -> Path | None:
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    if texture_path.exists() and texture_path.stat().st_size > 0:
        return texture_path
    try:
        urllib.request.urlretrieve(url, texture_path)  # noqa: S310
        if texture_path.exists() and texture_path.stat().st_size > 0:
            return texture_path
    except Exception:
        return None
    return None


def _load_texture_rgb(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        img = plt.imread(path)
    except Exception:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype.kind in {"u", "i"}:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = np.clip(arr.astype(np.float64), 0.0, 1.0)
    return arr


def _track_body_fixed(
    orbit_omega_rad_s: float,
    rotation_omega_rad_s: float,
    inclination_rad: float,
    duration_s: float,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, duration_s, n_samples)
    nu = orbit_omega_rad_s * t
    x_i = np.cos(nu)
    y_i = np.sin(nu) * math.cos(inclination_rad)
    z_i = np.sin(nu) * math.sin(inclination_rad)
    lon_i = np.arctan2(y_i, x_i)
    lon_b = lon_i - rotation_omega_rad_s * t
    lon_b = (lon_b + math.pi) % (2.0 * math.pi) - math.pi
    lat = np.arcsin(np.clip(z_i, -1.0, 1.0))
    return t, lon_b, lat


def _find_crossings(
    orbit_omega_rad_s: float,
    rotation_omega_rad_s: float,
    inclination_rad: float,
    node_angle_rad: float,
    initial_duration_s: float,
    needed_crossings: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], float, float]:
    if needed_crossings < 2:
        raise ValueError("needed_crossings must be >= 2")
    half_width = 0.5 * float(node_angle_rad)

    def _wrap_delta(a: np.ndarray, b: float) -> np.ndarray:
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    period = 2.0 * math.pi / orbit_omega_rad_s
    max_duration = max(initial_duration_s, period * 1200.0)
    cur_duration = float(initial_duration_s)
    crossing_idx: list[int] = []
    t = lon_b = lat = None
    lon0 = lat0 = 0.0

    for _ in range(8):
        n_samples = max(25000, int(cur_duration / period * 3000))
        t, lon_b, lat = _track_body_fixed(
            orbit_omega_rad_s=orbit_omega_rad_s,
            rotation_omega_rad_s=rotation_omega_rad_s,
            inclination_rad=inclination_rad,
            duration_s=cur_duration,
            n_samples=n_samples,
        )
        lon0 = float(lon_b[0])
        lat0 = float(lat[0])
        in_box = (np.abs(_wrap_delta(lon_b, lon0)) <= half_width) & (np.abs(lat - lat0) <= half_width)
        crossing_idx = []
        for i in range(1, len(in_box)):
            if in_box[i] and not in_box[i - 1]:
                crossing_idx.append(i)
        if in_box[0]:
            crossing_idx = [0] + crossing_idx
        if len(crossing_idx) >= needed_crossings:
            break
        if cur_duration >= max_duration:
            break
        cur_duration = min(cur_duration * 2.0, max_duration)

    if t is None or lon_b is None or lat is None or len(crossing_idx) < needed_crossings:
        raise RuntimeError("Could not find enough same-pixel crossings for the requested count.")

    crossing_idx = crossing_idx[:needed_crossings]
    return t, lon_b, lat, crossing_idx, lon0, lat0


def _equator_crossings(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Return interpolated equator crossings as (time, xyz) with z=0."""
    out: list[tuple[float, np.ndarray]] = []
    for i in range(len(z) - 1):
        z0 = float(z[i])
        z1 = float(z[i + 1])
        if z0 == 0.0:
            out.append((float(t[i]), np.array([x[i], y[i], z[i]], dtype=np.float64)))
            continue
        if z0 * z1 > 0.0:
            continue
        dz = z1 - z0
        if dz == 0.0:
            continue
        a = -z0 / dz
        if 0.0 <= a <= 1.0:
            p = np.array(
                [
                    x[i] + a * (x[i + 1] - x[i]),
                    y[i] + a * (y[i + 1] - y[i]),
                    0.0,
                ],
                dtype=np.float64,
            )
            tc = float(t[i] + a * (t[i + 1] - t[i]))
            out.append((tc, p))
    return out


def _face_colors_from_texture(
    face_centers_xyz: np.ndarray,
    radius_m: float,
    texture_rgb: np.ndarray | None,
) -> np.ndarray:
    xyz = np.asarray(face_centers_xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=1)
    r = np.where(r > 0.0, r, float(radius_m))
    x = xyz[:, 0] / r
    y = xyz[:, 1] / r
    z = np.clip(xyz[:, 2] / r, -1.0, 1.0)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z)

    if texture_rgb is None:
        # Fallback Europa-like warm/cool tint.
        c1 = 0.55 + 0.2 * np.cos(lat) * np.cos(2.0 * lon)
        c2 = 0.48 + 0.2 * np.cos(lat + 0.5)
        c3 = 0.40 + 0.2 * np.sin(1.3 * lon)
        return np.clip(np.stack([c1, c2, c3], axis=1), 0.0, 1.0)

    h, w, _ = texture_rgb.shape
    u = (lon + math.pi) / (2.0 * math.pi)
    v = (0.5 * math.pi - lat) / math.pi
    ix = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    iy = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    return texture_rgb[iy, ix]


def plot_orbit_3d_textured(
    altitude_m: float = DEFAULT_ALTITUDE_M,
    lmax: int = DEFAULT_LMAX,
    inclination_deg: float = 80.0,
    crossings_to_show: int = 2,
    sphere_lmax: int | None = None,
    texture_path: Path = Path("oribital/assets/europa2out-1-1024x512.jpg"),
    texture_url: str = DEFAULT_TEXTURE_URL,
    out_path: Path = Path("figures/oribital_orbit_3d_textured.png"),
    show: bool = True,
) -> None:
    est = estimate_polar_orbit(
        altitude_m=float(altitude_m),
        lmax=int(lmax),
        radius_m=EUROPA_RADIUS_M,
        mu_m3_s2=EUROPA_MU_M3_S2,
    )
    inc_rad = math.radians(float(inclination_deg))
    t, lon_b, lat, crossing_idx, lon0, lat0 = _find_crossings(
        orbit_omega_rad_s=est.omega_rad_s,
        rotation_omega_rad_s=est.europa_rotation_omega_rad_s,
        inclination_rad=inc_rad,
        node_angle_rad=est.node_angle_rad,
        initial_duration_s=est.period_s * 200.0,
        needed_crossings=int(crossings_to_show),
    )

    r_orbit = est.radius_m + est.altitude_m
    x_orb = r_orbit * np.cos(lat) * np.cos(lon_b)
    y_orb = r_orbit * np.cos(lat) * np.sin(lon_b)
    z_orb = r_orbit * np.sin(lat)
    first_idx = crossing_idx[0]
    center_time = float(t[first_idx])
    orbit_window_s = 5.0 * float(est.period_s)
    t_min = center_time - orbit_window_s
    t_max = center_time + orbit_window_s
    eq = _equator_crossings(t, x_orb, y_orb, z_orb)
    if len(eq) >= 2:
        t_eq = np.array([v[0] for v in eq], dtype=np.float64)
        i_start = int(np.searchsorted(t_eq, t_min, side="right") - 1)
        i_end = int(np.searchsorted(t_eq, t_max, side="left"))
        i_start = max(0, min(i_start, len(eq) - 1))
        i_end = max(i_start + 1, min(i_end, len(eq) - 1))
        start_t, start_p = eq[i_start]
        end_t, end_p = eq[i_end]
    else:
        start_t, start_p = float(t[0]), np.array([x_orb[0], y_orb[0], z_orb[0]], dtype=np.float64)
        end_t, end_p = float(t[-1]), np.array([x_orb[-1], y_orb[-1], z_orb[-1]], dtype=np.float64)
    track_mask = (t >= start_t) & (t <= end_t)
    track_idx = np.where(track_mask)[0]
    if track_idx.size < 2:
        track_idx = np.arange(max(0, first_idx - 1), min(len(t), first_idx + 2))
    track_core = np.column_stack((x_orb[track_idx], y_orb[track_idx], z_orb[track_idx]))
    orbit_seg = np.vstack((start_p[None, :], track_core, end_p[None, :]))

    tex_file = _download_texture_if_needed(Path(texture_path), texture_url)
    texture_rgb = _load_texture_rgb(tex_file)

    # Build the same style of triangulated sphere mesh used by workflow sphere renders.
    # Use a denser render mesh than the orbital lmax by default for smoother texture.
    sphere_lmax_use = int(sphere_lmax) if sphere_lmax is not None else max(int(lmax) * 2, 72)
    sphere_grid = build_roundtrip_grid(lmax=sphere_lmax_use, radius_m=float(est.radius_m), device="cpu")
    sphere_pts = sphere_grid["positions"].detach().cpu().numpy()
    sphere_faces = sphere_grid["faces"].detach().cpu().numpy().astype(np.int64)
    tri_verts = sphere_pts[sphere_faces]
    face_centers = tri_verts.mean(axis=1)
    face_colors = _face_colors_from_texture(face_centers, est.radius_m, texture_rgb)

    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    europa_surface = Poly3DCollection(
        tri_verts,
        facecolors=face_colors,
        edgecolor="none",
        linewidth=0.05,
        alpha=1.0,
        antialiased=True,
        zorder=0,
    )
    europa_surface.set_zsort("average")
    ax.add_collection3d(europa_surface)
    view_elev = 24.0
    view_azim = 118.0
    vis_orbit = _visible_points_for_3d_view(
        orbit_seg,
        sphere_radius=float(est.radius_m),
        elev_deg=view_elev,
        azim_deg=view_azim,
    )
    visible_idx = np.where(vis_orbit)[0]
    if visible_idx.size > 0:
        breaks = np.where(np.diff(visible_idx) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(visible_idx) - 1]
        labeled = False
        for s_idx, e_idx in zip(starts, ends):
            seg_idx = visible_idx[s_idx : e_idx + 1]
            ax.plot3D(
                orbit_seg[seg_idx, 0],
                orbit_seg[seg_idx, 1],
                orbit_seg[seg_idx, 2],
                color="#0b2f6b",
                linewidth=1.1,
                label=f"Orbit segment ({crossings_to_show} crossings)" if not labeled else None,
                zorder=10,
            )
            labeled = True
    crossing_idx_plot = [i for i in crossing_idx if bool(track_mask[i])]
    if not crossing_idx_plot:
        crossing_idx_plot = [crossing_idx[0]]
    crossing_pts = np.column_stack((x_orb[crossing_idx_plot], y_orb[crossing_idx_plot], z_orb[crossing_idx_plot]))
    vis_cross = _visible_points_for_3d_view(
        crossing_pts,
        sphere_radius=float(est.radius_m),
        elev_deg=view_elev,
        azim_deg=view_azim,
    )
    cp = crossing_pts[vis_cross] if np.any(vis_cross) else np.empty((0, 3), dtype=np.float64)
    final_pt = orbit_seg[-1]

    # Ground-pixel center marker on surface.
    x0 = est.radius_m * math.cos(lat0) * math.cos(lon0)
    y0 = est.radius_m * math.cos(lat0) * math.sin(lon0)
    z0 = est.radius_m * math.sin(lat0)
    pixel_center = np.array([x0, y0, z0], dtype=np.float64)

    lim = r_orbit * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=view_elev, azim=view_azim)

    hours_span = float(end_t - start_t) / 3600.0
    ax.set_title(
        f"Europa 3D Orbit, alt={est.altitude_m/1000.0:.0f} km, inc={inclination_deg:.1f} deg\n"
        f"First {crossings_to_show} same-pixel crossings, span={hours_span:.2f} hr"
    )
    # Draw point markers last so they are visible above surface artists.
    if cp.shape[0] > 0:
        ax.scatter(
            cp[:, 0],
            cp[:, 1],
            cp[:, 2],
            color="#ff4040",
            s=30,
            depthshade=False,
            label="Crossings",
            zorder=11,
        )
    ax.scatter(
        [final_pt[0]],
        [final_pt[1]],
        [final_pt[2]],
        color="#ff00aa",
        s=46,
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        depthshade=False,
        label="Final point",
        zorder=12,
    )
    ax.scatter(
        [pixel_center[0]],
        [pixel_center[1]],
        [pixel_center[2]],
        color="#ffc300",
        s=42,
        depthshade=False,
        label="Pixel center",
        zorder=12,
    )
    ax.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved 3D orbit figure to {out_path}")
    if tex_file is not None:
        print(f"Texture source: {tex_file}")
    else:
        print("Texture download/load failed; used fallback procedural texture.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a textured 3D Europa orbit with repeated pixel crossings.")
    parser.add_argument("--altitude-km", type=float, default=DEFAULT_ALTITUDE_M / 1000.0)
    parser.add_argument("--lmax", type=int, default=DEFAULT_LMAX)
    parser.add_argument("--inclination-deg", type=float, default=80.0)
    parser.add_argument("--crossings", type=int, default=2)
    parser.add_argument("--sphere-lmax", type=int, default=None, help="Sphere render resolution; default uses a denser mesh.")
    parser.add_argument("--texture-path", type=str, default="oribital/assets/europa2out-1-1024x512.jpg")
    parser.add_argument("--texture-url", type=str, default=DEFAULT_TEXTURE_URL)
    parser.add_argument("--save", type=str, default="figures/oribital_orbit_3d_textured.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plot_orbit_3d_textured(
        altitude_m=float(args.altitude_km) * 1000.0,
        lmax=int(args.lmax),
        inclination_deg=float(args.inclination_deg),
        crossings_to_show=max(2, int(args.crossings)),
        sphere_lmax=int(args.sphere_lmax) if args.sphere_lmax is not None else None,
        texture_path=Path(args.texture_path),
        texture_url=str(args.texture_url),
        out_path=Path(args.save),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
