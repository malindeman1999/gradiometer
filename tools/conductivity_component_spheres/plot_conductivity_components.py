"""Plot baseline and individual Europa conductivity components on spheres.

Uses the same gridding and conductivity component generation as the
nonuniform GUI workflow:
- workflow.plotting.sphere_roundtrip.build_roundtrip_grid
- workflow.conductivity_models.europa_snapshot.build_europa_snapshot_conductivity
"""

from __future__ import annotations

import argparse
import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from europa_model.config import GridConfig
from workflow.conductivity_models import EuropaSnapshotConfig, build_europa_snapshot_conductivity
from workflow.plotting.sphere_roundtrip import build_roundtrip_grid


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    w = weights.to(torch.float64)
    v = values.to(torch.float64)
    wsum = float(w.sum().item())
    if wsum <= 0.0:
        raise RuntimeError("Non-positive quadrature weight sum.")
    return float((w * v).sum().item() / wsum)


def _component_sigma_map(sigma0: float, x_component: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Convert one log-space component into a conductivity map with mean sigma0."""
    sigma = float(sigma0) * torch.exp(x_component.to(torch.float64))
    mean_now = _weighted_mean(sigma, weights)
    if mean_now > 0.0:
        sigma = sigma * (float(sigma0) / mean_now)
    return sigma


def _face_values(node_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(node_values, dtype=np.float64)[faces].mean(axis=1)


def _plot_sphere_panel(
    ax,
    positions: np.ndarray,
    faces: np.ndarray,
    node_values: np.ndarray,
    norm: mcolors.Normalize,
    cmap: str,
    title: str,
) -> None:
    tri_verts = positions[faces]
    face_vals = _face_values(node_values, faces)
    cm = plt.get_cmap(cmap)
    poly = Poly3DCollection(tri_verts, linewidths=0.0, edgecolors="none")
    poly.set_facecolor(cm(norm(face_vals)))
    ax.add_collection3d(poly)

    lim = float(np.max(np.abs(positions)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=20.0, azim=30.0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def build_and_plot(
    lmax: int,
    sigma0: float,
    seed: int,
    output: str | None,
) -> None:
    grid = build_roundtrip_grid(lmax=int(lmax), radius_m=1.56e6, device="cpu")
    positions = grid["positions"].to(torch.float64)
    weights = grid["areas"].to(torch.float64)
    faces = grid["faces"].detach().cpu().numpy()

    cfg = EuropaSnapshotConfig(seed=int(seed))
    _, comp = build_europa_snapshot_conductivity(
        positions=positions,
        weights=weights,
        sigma0=float(sigma0),
        cfg=cfg,
    )

    sigma0 = float(sigma0)
    baseline = torch.full((positions.shape[0],), sigma0, dtype=torch.float64)
    sigma_conv = _component_sigma_map(sigma0, comp["x_chem"], weights)
    sigma_exchange = _component_sigma_map(sigma0, comp["x_exchange"], weights)
    sigma_flow = _component_sigma_map(sigma0, comp["x_flow"], weights)
    sigma_bg = _component_sigma_map(sigma0, comp["x_bg"], weights)
    sigma_combined_avg = 0.25 * (sigma_conv + sigma_exchange + sigma_flow + sigma_bg)

    maps = [
        ("Baseline", baseline),
        ("Convection only", sigma_conv),
        ("Exchange only", sigma_exchange),
        ("Flow only", sigma_flow),
        ("Background only", sigma_bg),
        ("Combined (avg of components)", sigma_combined_avg),
    ]

    pos_np = positions.detach().cpu().numpy()
    map_np = [(name, field.detach().cpu().numpy()) for name, field in maps]

    fig = plt.figure(figsize=(16, 9), dpi=120, constrained_layout=True)
    fig.suptitle(
        f"Europa Conductivity Component Maps (lmax={int(lmax)}, sigma0={sigma0:.3e} S, seed={int(seed)})",
        fontsize=14,
    )
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]

    cm = plt.get_cmap("viridis")
    for ax, (name, vals) in zip(axes, map_np):
        face_vals = _face_values(vals, faces)
        vmin = float(np.min(face_vals))
        vmax = float(np.max(face_vals))
        if not math.isfinite(vmin) or not math.isfinite(vmax):
            raise RuntimeError(f"Non-finite color scale limits for panel '{name}'.")
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        _plot_sphere_panel(
            ax=ax,
            positions=pos_np,
            faces=faces,
            node_values=vals,
            norm=norm,
            cmap="viridis",
            title=name,
        )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array(face_vals)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Conductivity (S)")

    for name, vals in map_np:
        print(
            f"{name:16s} mean={np.mean(vals):.3e} S  min={np.min(vals):.3e} S  max={np.max(vals):.3e} S"
        )

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"\nSaved figure to {output}")
    else:
        plt.show()


def main() -> None:
    default_cfg = GridConfig(nside=1, lmax=1, radius_m=1.56e6, device="cpu")
    default_sigma0 = 2.0 * default_cfg.seawater_conductivity_s_per_m * default_cfg.ocean_thickness_m

    parser = argparse.ArgumentParser(
        description="Plot baseline and individual Europa conductivity components on one figure."
    )
    parser.add_argument("--lmax", type=int, default=36, help="Spherical harmonic / grid resolution control.")
    parser.add_argument(
        "--sigma0",
        type=float,
        default=float(default_sigma0),
        help="Baseline sheet conductivity (S). Defaults to GUI baseline.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for europa snapshot component generation.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path. If omitted, opens an interactive window.",
    )
    args = parser.parse_args()

    build_and_plot(
        lmax=args.lmax,
        sigma0=args.sigma0,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
