"""Render real-valued (l,m)+(-m) conductivity harmonic sphere heatmaps and save PDFs.

Uses GUI-consistent gridding via:
- workflow.plotting.sphere_roundtrip.build_roundtrip_grid

And SH reconstruction via:
- europa_model.transforms.sh_inverse
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from europa_model.transforms import sh_inverse
from workflow.plotting.sphere_roundtrip import build_roundtrip_grid


def _weighted_standardize(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(torch.float64)
    f = field.to(torch.float64)
    wsum = float(w.sum().item())
    if wsum <= 0.0:
        raise RuntimeError("Non-positive quadrature weight sum.")
    mean = torch.sum(w * f) / wsum
    centered = f - mean
    var = torch.sum(w * centered * centered) / wsum
    std = torch.sqrt(var).clamp_min(1e-12)
    return centered / std


def _face_values(node_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(node_values, dtype=np.float64)[faces].mean(axis=1)


def _plot_sphere(
    ax,
    positions: np.ndarray,
    faces: np.ndarray,
    node_values: np.ndarray,
    title: str,
    cmap: str = "coolwarm",
) -> None:
    tri_verts = positions[faces]
    face_vals = _face_values(node_values, faces)
    vmax = float(max(np.max(np.abs(face_vals)), 1e-12))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
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
    ax.set_title(title, fontsize=9)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array(face_vals)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("delta sigma (S)")


def _build_real_mode_map(
    l: int,
    m: int,
    grid_lmax: int,
    positions: torch.Tensor,
    weights: torch.Tensor,
    rms_s: float,
) -> tuple[torch.Tensor, float]:
    coeffs = torch.zeros((grid_lmax + 1, 2 * grid_lmax + 1), dtype=torch.complex128)

    # Make a real field by pairing +m and -m coefficients with conjugate symmetry.
    c = 1.0 + 0.0j
    coeffs[l, grid_lmax + m] = c
    if m > 0:
        coeffs[l, grid_lmax - m] = ((-1) ** m) * np.conj(c)

    recon = sh_inverse(coeffs, positions, weights)
    imag_max = float(recon.imag.abs().max().item())
    real_field = recon.real.to(torch.float64)
    standardized = _weighted_standardize(real_field, weights)
    delta_sigma = float(rms_s) * standardized
    return delta_sigma, imag_max


def _layout_for_count(n: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def render_l_set(
    degrees: list[int],
    grid_lmax: int,
    rms_s: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] building GUI-default grid with lmax={grid_lmax}", flush=True)
    grid = build_roundtrip_grid(lmax=int(grid_lmax), radius_m=1.56e6, device="cpu")
    positions = grid["positions"].to(torch.float64)
    weights = grid["areas"].to(torch.float64)
    faces = grid["faces"].detach().cpu().numpy()
    pos_np = positions.detach().cpu().numpy()
    n_nodes = int(grid["n_points"])
    print(f"[grid] nodes={n_nodes}, faces={int(grid['n_faces'])}", flush=True)

    for l in degrees:
        if l < 1:
            continue
        if l > grid_lmax:
            print(f"[skip] l={l} exceeds grid_lmax={grid_lmax}", flush=True)
            continue

        print(f"[l={l}] generating m=0..{l}", flush=True)
        n_panels = l + 1
        rows, cols = _layout_for_count(n_panels)
        fig = plt.figure(figsize=(4.4 * cols, 4.2 * rows), dpi=120, constrained_layout=True)
        fig.suptitle(
            f"Real Conductivity Harmonics: l={l}, m=0..{l} (delta sigma RMS={rms_s:.3e} S)",
            fontsize=13,
        )

        axes = [fig.add_subplot(rows, cols, i + 1, projection="3d") for i in range(rows * cols)]
        used_axes = axes[:n_panels]
        extra_axes = axes[n_panels:]

        for m, ax in enumerate(used_axes):
            print(f"  [l={l}] m={m}: reconstructing and plotting", flush=True)
            delta_sigma, imag_max = _build_real_mode_map(
                l=l,
                m=m,
                grid_lmax=grid_lmax,
                positions=positions,
                weights=weights,
                rms_s=rms_s,
            )
            if imag_max > 1e-10:
                print(f"    [warn] l={l}, m={m}: max|imag|={imag_max:.3e}", flush=True)
            vals = delta_sigma.detach().cpu().numpy()
            _plot_sphere(
                ax=ax,
                positions=pos_np,
                faces=faces,
                node_values=vals,
                title=f"(l={l}, m={m})",
            )

        for ax in extra_axes:
            ax.axis("off")

        out_path = out_dir / f"conductivity_harmonics_l{l:02d}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}", flush=True)

    print("[done] finished rendering requested harmonic PDFs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot real-valued conductivity harmonic sphere heatmaps for each (l,m), "
            "using +m/-m pairing, and save one PDF per degree l."
        )
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Degrees l to render (default: 1 2 4 8 16 32).",
    )
    parser.add_argument(
        "--grid-lmax",
        type=int,
        default=36,
        help="Grid resolution control from GUI path (default: 36).",
    )
    parser.add_argument(
        "--rms-s",
        type=float,
        default=1.0,
        help="Target RMS amplitude for each harmonic map in S (default: 1.0).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/conductivity_harmonic_spheres/output"),
        help="Output directory for PDF files.",
    )
    args = parser.parse_args()

    degrees = [int(x) for x in args.degrees]
    render_l_set(
        degrees=degrees,
        grid_lmax=int(args.grid_lmax),
        rms_s=float(args.rms_s),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

