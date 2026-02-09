"""Resume-capable CLI sweep for toroidal mode inductance on a unit sphere."""

from __future__ import annotations

import argparse
from pathlib import Path

from toroidal_inductance.toroidal_mode_inductance import (
    load_and_scale_inductance_table,
    sweep_unit_sphere_inductance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute toroidal inductance table up to Lmax.")
    parser.add_argument("--lmax", type=int, default=71, help="Maximum l to compute (l>=1).")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("toroidal_inductance/data/inductance_unit_sphere.csv"),
        help="CSV path for cached unit-sphere inductances.",
    )
    parser.add_argument("--tol", type=float, default=1.0e-2, help="Relative refinement tolerance.")
    parser.add_argument("--scale-radius", type=float, default=None, help="Optional radius (m) for scaled printout.")
    parser.add_argument("--scale-mu-r", type=float, default=1.0, help="Relative permeability for scaled printout.")

    args = parser.parse_args()

    rows = sweep_unit_sphere_inductance(args.lmax, data_path=args.data_path, tol=args.tol)
    print(f"Saved {len(rows)} rows to {args.data_path}")

    if rows:
        last = rows[-1]
        print(
            "Last row: "
            f"l={int(last['l'])}, "
            f"L={last['inductance_h']:.8e} H, "
            f"rel(general_vs_m0)={last['rel_diff_general_m0']:.3e}, "
            f"rel(m_test_vs_m0)={last['rel_diff_m1_m0']:.3e}"
        )

    if args.scale_radius is not None:
        scaled = load_and_scale_inductance_table(
            radius_m=args.scale_radius,
            mu_r=args.scale_mu_r,
            data_path=args.data_path,
        )
        if scaled:
            print(
                f"Scaled last row at radius={args.scale_radius} m, mu_r={args.scale_mu_r}: "
                f"l={int(scaled[-1]['l'])}, L={scaled[-1]['inductance_h']:.8e} H"
            )


if __name__ == "__main__":
    main()
