"""Sweep frequency and compare induced-current response across solver variants."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from europa_model.solvers import (
    solve_spectral_first_order_sim,
    solve_spectral_self_consistent_sim,
    solve_uniform_first_order_sim,
    solve_uniform_self_consistent_sim,
)
from workflow.data_objects.phasor_data import PhasorSimulation


def _build_b_radial_mode(lmax: int, mode_l: int, mode_m: int, amplitude_t: float) -> torch.Tensor:
    coeffs = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    coeffs[mode_l, lmax + mode_m] = complex(amplitude_t, 0.0)
    if mode_m > 0:
        coeffs[mode_l, lmax - mode_m] = ((-1) ** mode_m) * complex(amplitude_t, 0.0)
    return coeffs


def _build_uniform_admittance_spectral(lmax: int, sigma_sheet_s: float) -> torch.Tensor:
    coeffs = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    coeffs[0, lmax] = complex(float(sigma_sheet_s) * 2.0 * math.sqrt(math.pi), 0.0)
    return coeffs


def _sim_template(
    *,
    omega: float,
    freq_hz: float,
    lmax: int,
    radius_m: float,
    b_amp_t: float,
    b_radial: torch.Tensor,
    admittance_uniform: float | None,
    admittance_spectral: torch.Tensor | None,
    solver_variant: str,
) -> PhasorSimulation:
    return PhasorSimulation(
        omega=omega,
        period_sec=1.0 / freq_hz,
        lmax=lmax,
        radius_m=radius_m,
        ambient_amplitude_t=b_amp_t,
        ambient_phase_rad=0.0,
        grid_positions=torch.zeros((1, 3), dtype=torch.float64),
        grid_normals=torch.zeros((1, 3), dtype=torch.float64),
        grid_areas=torch.ones((1,), dtype=torch.float64),
        grid_neighbors=None,
        solver_variant=solver_variant,
        admittance_uniform=admittance_uniform,
        admittance_spectral=admittance_spectral,
        B_radial=b_radial,
    )


def _extract_metrics(
    sim: PhasorSimulation,
    *,
    mode_l: int,
    mode_m: int,
    lmax: int,
    e_applied_mode: float,
) -> dict[str, float]:
    mode_amp = float(torch.abs(sim.K_toroidal[mode_l, lmax + mode_m]).item())
    rms_amp = float(torch.sqrt(torch.mean(torch.abs(sim.K_toroidal) ** 2)).item())
    i_over_emf = mode_amp / max(e_applied_mode, 1.0e-30)
    return {
        "k_mode_amp_a_per_m": mode_amp,
        "k_rms_a_per_m": rms_amp,
        "i_over_emf_a_per_v": i_over_emf,
    }


def _run_single_frequency(
    *,
    freq_hz: float,
    lmax: int,
    mode_l: int,
    mode_m: int,
    b_amp_t: float,
    radius_m: float,
    sigma_sheet_s: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    omega = 2.0 * math.pi * freq_hz
    b_radial = _build_b_radial_mode(lmax=lmax, mode_l=mode_l, mode_m=mode_m, amplitude_t=b_amp_t)
    admittance_uniform = float(sigma_sheet_s)
    admittance_spectral = _build_uniform_admittance_spectral(lmax=lmax, sigma_sheet_s=sigma_sheet_s)
    e_applied_mode = abs((omega * radius_m) / (mode_l * (mode_l + 1.0)) * b_amp_t)

    solvers = {
        "uniform_first_order": solve_uniform_first_order_sim,
        "uniform_self_consistent": solve_uniform_self_consistent_sim,
        "spectral_first_order": solve_spectral_first_order_sim,
        "spectral_self_consistent": solve_spectral_self_consistent_sim,
    }

    out: dict[str, dict[str, float]] = {}
    for name, solver in solvers.items():
        sim = _sim_template(
            omega=omega,
            freq_hz=freq_hz,
            lmax=lmax,
            radius_m=radius_m,
            b_amp_t=b_amp_t,
            b_radial=b_radial,
            admittance_uniform=admittance_uniform if name.startswith("uniform") else None,
            admittance_spectral=admittance_spectral if name.startswith("spectral") else None,
            solver_variant=name,
        )
        sim = solver(sim)
        out[name] = _extract_metrics(sim, mode_l=mode_l, mode_m=mode_m, lmax=lmax, e_applied_mode=e_applied_mode)

    return e_applied_mode, out


def _decade_points(f_min: float, f_max: float) -> np.ndarray:
    if f_min <= 0.0 or f_max <= 0.0 or f_max < f_min:
        raise ValueError("Require 0 < f_min <= f_max")
    d0 = int(round(math.log10(f_min)))
    d1 = int(round(math.log10(f_max)))
    if not math.isclose(f_min, 10.0**d0, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("f_min must be a power of 10 for one-point-per-decade spacing")
    if not math.isclose(f_max, 10.0**d1, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("f_max must be a power of 10 for one-point-per-decade spacing")
    return np.array([10.0**d for d in range(d0, d1 + 1)], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency sweep of induced toroidal current across solver variants.")
    parser.add_argument("--f-min", type=float, default=1.0e-9, help="Minimum frequency [Hz], power of 10.")
    parser.add_argument("--f-max", type=float, default=1.0e-3, help="Maximum frequency [Hz], power of 10.")
    parser.add_argument("--mode-l", type=int, default=1, help="Forcing radial B mode degree l.")
    parser.add_argument("--mode-m", type=int, default=0, help="Forcing radial B mode order m.")
    parser.add_argument("--lmax", type=int, default=2, help="Spectral lmax.")
    parser.add_argument("--radius-m", type=float, default=1.56e6, help="Sphere radius [m].")
    parser.add_argument(
        "--sigma-sheet-s",
        type=float,
        default=0.3 * 100000.0,
        help="Uniform sheet conductivity [S] (default 0.3 S/m * 100 km).",
    )
    parser.add_argument("--b-amp-t", type=float, default=1.0, help="Forcing B_r mode amplitude [T].")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("toroidal_inductance/data/solver_current_vs_frequency.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("toroidal_inductance/data/solver_current_vs_frequency.png"),
        help="Output plot path.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save plot only (no interactive window).")
    args = parser.parse_args()

    if args.mode_l < 1:
        raise ValueError("mode-l must be >= 1")
    if abs(args.mode_m) > args.mode_l:
        raise ValueError("Require |mode-m| <= mode-l")

    lmax = int(args.lmax)
    freqs = _decade_points(args.f_min, args.f_max)
    solver_names = [
        "uniform_first_order",
        "uniform_self_consistent",
        "spectral_first_order",
        "spectral_self_consistent",
    ]

    rows = []
    for f in freqs:
        e_applied_mode, metrics = _run_single_frequency(
            freq_hz=float(f),
            lmax=lmax,
            mode_l=args.mode_l,
            mode_m=args.mode_m,
            b_amp_t=args.b_amp_t,
            radius_m=args.radius_m,
            sigma_sheet_s=args.sigma_sheet_s,
        )
        rows.append(
            {
                "frequency_hz": float(f),
                "omega_rad_s": float(2.0 * math.pi * f),
                "e_applied_mode_v_per_m": e_applied_mode,
            }
        )
        row = rows[-1]
        for name in solver_names:
            row[f"k_mode_amp_{name}_a_per_m"] = metrics[name]["k_mode_amp_a_per_m"]
            row[f"k_rms_{name}_a_per_m"] = metrics[name]["k_rms_a_per_m"]
            row[f"i_over_emf_{name}_a_per_v"] = metrics[name]["i_over_emf_a_per_v"]

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frequency_hz", "omega_rad_s", "e_applied_mode_v_per_m"]
    for name in solver_names:
        fieldnames.extend(
            [
                f"k_mode_amp_{name}_a_per_m",
                f"k_rms_{name}_a_per_m",
                f"i_over_emf_{name}_a_per_v",
            ]
        )
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fvals = np.array([r["frequency_hz"] for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
    styles = {
        "uniform_first_order": ("o-", "#1f77b4"),
        "uniform_self_consistent": ("s-", "#ff7f0e"),
        "spectral_first_order": ("^-", "#2ca02c"),
        "spectral_self_consistent": ("d-", "#d62728"),
    }
    for name in solver_names:
        kvals = np.array([r[f"k_mode_amp_{name}_a_per_m"] for r in rows], dtype=float)
        yvals = np.array([r[f"i_over_emf_{name}_a_per_v"] for r in rows], dtype=float)
        fmt, color = styles[name]
        ax0.loglog(fvals, kvals, fmt, lw=1.8, ms=5.5, color=color, label=name)
        ax1.loglog(fvals, yvals, fmt, lw=1.8, ms=5.0, color=color, label=name)

    ax0.set_ylabel(r"Induced |K_{lm}| [A/m]")
    ax0.set_title(
        "Induced Current vs Frequency (Solver Comparison)\n"
        f"mode=(l={args.mode_l}, m={args.mode_m}), lmax={lmax}, sigma_s={args.sigma_sheet_s:.3e} S"
    )
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(loc="best")

    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"|I/EMF| (|K_{lm}|/|E_{applied,lm}|) [A/V]")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")
    fig.tight_layout()

    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_out, dpi=180)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved CSV: {args.csv_out}")
    print(f"Saved plot: {args.plot_out}")


if __name__ == "__main__":
    main()
