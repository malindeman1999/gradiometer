"""Plot scaled toroidal inductance vs degree for Europa-radius comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toroidal_inductance.toroidal_mode_inductance import (
    MU0,
    load_and_scale_inductance_table,
    sweep_unit_sphere_inductance,
)

GUI_EUROPA_RADIUS_M = 1.56e6


def _build_notes_profile(l_values: np.ndarray, radius_m: float, profile: str, power_p: float) -> np.ndarray:
    l = l_values.astype(float)
    if profile == "geom":
        c_l = 1.0 / (2.0 * l + 1.0)
    elif profile == "self_coupling":
        ell = l * (l + 1.0)
        c_l = 1.0 / ((2.0 * l + 1.0) * (ell**2))
    elif profile == "powerlaw":
        c_l = 1.0 / ((2.0 * l + 1.0) ** power_p)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return MU0 * radius_m * c_l


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Europa-scaled inductance curves vs L.")
    parser.add_argument("--lmax", type=int, default=71, help="Maximum degree to include in the comparison plot.")
    parser.add_argument(
        "--radius-m",
        type=float,
        default=GUI_EUROPA_RADIUS_M,
        help="Radius to use for scaled inductance (default is GUI Europa radius).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("toroidal_inductance/data/inductance_unit_sphere.csv"),
        help="CSV cache path for unit-sphere inductance data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("toroidal_inductance/data/europa_inductance_comparison.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plot only (do not display interactive window).",
    )
    parser.add_argument("--tol", type=float, default=1.0e-2, help="Tolerance for any newly computed l rows.")
    parser.add_argument(
        "--notes-profile",
        choices=["geom", "self_coupling", "powerlaw"],
        default="geom",
        help="Notes-based l-scaling profile to compare against.",
    )
    parser.add_argument("--power-p", type=float, default=1.0, help="Exponent p if --notes-profile powerlaw.")
    parser.add_argument(
        "--normalize-ell",
        action="store_true",
        help="Optional: divide computed inductance by l(l+1) before plotting.",
    )

    args = parser.parse_args()

    # Ensure data exists up to requested lmax (resume-capable).
    sweep_unit_sphere_inductance(args.lmax, data_path=args.data_path, tol=args.tol)

    scaled_rows = load_and_scale_inductance_table(radius_m=args.radius_m, data_path=args.data_path)
    scaled_rows = [row for row in scaled_rows if 1 <= int(row["l"]) <= args.lmax]
    if not scaled_rows:
        raise RuntimeError("No inductance rows available to plot.")

    l_vals = np.array([int(row["l"]) for row in scaled_rows], dtype=int)
    l_data_raw = np.array([float(row["inductance_h"]) for row in scaled_rows], dtype=float)
    if args.normalize_ell:
        ell = l_vals.astype(float) * (l_vals.astype(float) + 1.0)
        l_data = l_data_raw / ell
    else:
        l_data = l_data_raw

    l_grid = np.arange(1, args.lmax + 1, dtype=int)
    l_gui = np.full_like(l_grid, MU0 * args.radius_m / 2.0, dtype=float)
    l_notes = _build_notes_profile(l_grid.astype(float), args.radius_m, args.notes_profile, args.power_p)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if args.normalize_ell:
        computed_label = "Computed L_l / [l(l+1)]"
        y_label = "Normalized Inductance [H]"
    else:
        computed_label = "Computed toroidal mode inductance (field-energy)"
        y_label = "Inductance L_l [H]"
    ax.plot(l_vals, l_data, "o-", lw=1.8, ms=4.5, label=computed_label)
    ax.plot(l_grid, l_gui, "--", lw=2.0, label="GUI inductance_scale=1: mu0 R / 2")

    if args.notes_profile == "geom":
        notes_label = "Notes scaling: mu0 R / (2l+1)"
    elif args.notes_profile == "self_coupling":
        notes_label = "Notes scaling: mu0 R / ((2l+1)[l(l+1)]^2)"
    else:
        notes_label = f"Notes scaling: mu0 R / (2l+1)^p, p={args.power_p:g}"
    ax.plot(l_grid, l_notes, "-.", lw=2.0, label=notes_label)

    ax.set_xlabel("Spherical Harmonic Degree l")
    ax.set_ylabel(y_label)
    ax.set_title(f"Europa-Radius Toroidal Inductance Comparison (R={args.radius_m:.3e} m)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved plot: {args.output}")
    print(f"Data source: {args.data_path}")
    print(f"Radius used: {args.radius_m:.6e} m")
    print(f"GUI scale=1 level: {MU0 * args.radius_m / 2.0:.6e} H")


if __name__ == "__main__":
    main()
