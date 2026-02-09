"""Toroidal spherical-harmonic inductance via magnetic-field energy sampling."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import lpmv

MU0 = 4.0e-7 * math.pi


@dataclass
class AdaptiveResult:
    energy_joule: float
    inductance_h: float
    rel_error_estimate: float
    samples: Dict[str, int]


def _norm_complex_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Y_lm and angular derivatives (complex convention, orthonormal on sphere)."""
    if l < 1:
        raise ValueError("l must be >= 1 for toroidal modes")
    if abs(m) > l:
        raise ValueError("|m| must be <= l")

    abs_m = abs(m)
    x = np.cos(theta)
    sin_theta = np.sin(theta)
    sin_safe = np.clip(sin_theta, 1.0e-12, None)

    prefac = math.sqrt(
        ((2.0 * l + 1.0) / (4.0 * math.pi))
        * (math.factorial(l - abs_m) / math.factorial(l + abs_m))
    )
    plm = lpmv(abs_m, l, x)
    if l - 1 >= abs_m:
        plm_prev = lpmv(abs_m, l - 1, x)
    else:
        plm_prev = np.zeros_like(plm)
    dplm_dtheta = (l * x * plm - (l + abs_m) * plm_prev) / sin_safe

    phase = np.exp(1j * abs_m * phi)
    y_pos = prefac * plm[:, None] * phase[None, :]
    dy_dtheta_pos = prefac * dplm_dtheta[:, None] * phase[None, :]

    if m >= 0:
        y = y_pos
        dy_dtheta = dy_dtheta_pos
    else:
        sign = (-1) ** abs_m
        y = sign * np.conjugate(y_pos)
        dy_dtheta = sign * np.conjugate(dy_dtheta_pos)

    dy_dphi = 1j * m * y
    return y, dy_dtheta, dy_dphi


def _radial_coefficients(l: int, r: float, radius: float, current: float, mu: float) -> Tuple[complex, complex]:
    """Return scalar multipliers for B_r and tangential gradient terms at radius r."""
    a = current * (l + 1.0) / (2.0 * l + 1.0)
    b = -current * l / (2.0 * l + 1.0)

    if r <= radius:
        x = r / radius
        coeff_r = -mu * l * a / radius * (x ** (l - 1))
        coeff_t = -mu * a / radius * (x ** (l - 1))
    else:
        x = radius / r
        coeff_r = mu * (l + 1.0) * b / radius * (x ** (l + 2))
        coeff_t = -mu * b / radius * (x ** (l + 2))
    return coeff_r, coeff_t


def _energy_integral_general(
    l: int,
    m: int,
    *,
    radius: float,
    current: float,
    mu: float,
    n_r_inside: int,
    n_r_outside: int,
    n_theta: int,
    n_phi: int,
) -> float:
    """Full 3D sampled energy integral for a toroidal (l,m) mode."""
    dtheta = math.pi / n_theta
    dphi = 2.0 * math.pi / n_phi

    theta = (np.arange(n_theta) + 0.5) * dtheta
    phi = (np.arange(n_phi) + 0.5) * dphi
    sin_theta = np.sin(theta)[:, None]

    y, dy_dtheta, dy_dphi = _norm_complex_sph_harm(l, m, theta, phi)
    grad_sq = np.abs(dy_dtheta) ** 2 + (np.abs(dy_dphi) ** 2) / np.clip(sin_theta ** 2, 1.0e-12, None)
    y_sq = np.abs(y) ** 2

    angle_weight = sin_theta * dtheta * dphi
    ang_y = float(np.sum(y_sq * angle_weight))
    ang_grad = float(np.sum(grad_sq * angle_weight))

    energy = 0.0

    dr_in = radius / n_r_inside
    for i in range(n_r_inside):
        r = (i + 0.5) * dr_in
        coeff_r, coeff_t = _radial_coefficients(l, r, radius, current, mu)
        b2_avg = (abs(coeff_r) ** 2) * ang_y + (abs(coeff_t) ** 2) * ang_grad
        energy += (r ** 2) * b2_avg * dr_in

    du = 1.0 / n_r_outside
    for i in range(n_r_outside):
        u = (i + 0.5) * du
        r = radius / (1.0 - u)
        dr_du = radius / ((1.0 - u) ** 2)
        coeff_r, coeff_t = _radial_coefficients(l, r, radius, current, mu)
        b2_avg = (abs(coeff_r) ** 2) * ang_y + (abs(coeff_t) ** 2) * ang_grad
        energy += (r ** 2) * b2_avg * dr_du * du

    return 0.5 * energy / mu


def _energy_integral_m0_axisymmetric(
    l: int,
    *,
    radius: float,
    current: float,
    mu: float,
    n_r_inside: int,
    n_r_outside: int,
    n_theta: int,
) -> float:
    """Axisymmetric sampled energy integral for m=0 (phi integrated analytically)."""
    m = 0
    dtheta = math.pi / n_theta
    theta = (np.arange(n_theta) + 0.5) * dtheta
    sin_theta = np.sin(theta)

    # m=0 makes phi dependence trivial, so a single phi sample suffices.
    y, dy_dtheta, _ = _norm_complex_sph_harm(l, m, theta, np.array([0.0]))
    y = y[:, 0]
    dy_dtheta = dy_dtheta[:, 0]

    ang_y = float(2.0 * math.pi * np.sum((np.abs(y) ** 2) * sin_theta * dtheta))
    ang_grad = float(2.0 * math.pi * np.sum((np.abs(dy_dtheta) ** 2) * sin_theta * dtheta))

    energy = 0.0

    dr_in = radius / n_r_inside
    for i in range(n_r_inside):
        r = (i + 0.5) * dr_in
        coeff_r, coeff_t = _radial_coefficients(l, r, radius, current, mu)
        b2_avg = (abs(coeff_r) ** 2) * ang_y + (abs(coeff_t) ** 2) * ang_grad
        energy += (r ** 2) * b2_avg * dr_in

    du = 1.0 / n_r_outside
    for i in range(n_r_outside):
        u = (i + 0.5) * du
        r = radius / (1.0 - u)
        dr_du = radius / ((1.0 - u) ** 2)
        coeff_r, coeff_t = _radial_coefficients(l, r, radius, current, mu)
        b2_avg = (abs(coeff_r) ** 2) * ang_y + (abs(coeff_t) ** 2) * ang_grad
        energy += (r ** 2) * b2_avg * dr_du * du

    return 0.5 * energy / mu


def _adaptive_refine(run_once, initial: Dict[str, int], tol: float, max_refinements: int) -> AdaptiveResult:
    params = dict(initial)
    energy_prev = run_once(**params)
    for _ in range(max_refinements):
        params = {k: v * 2 for k, v in params.items()}
        energy_new = run_once(**params)
        rel = abs(energy_new - energy_prev) / max(abs(energy_new), 1.0e-30)
        if rel < tol:
            return AdaptiveResult(
                energy_joule=energy_new,
                inductance_h=2.0 * energy_new,
                rel_error_estimate=rel,
                samples=params,
            )
        energy_prev = energy_new

    return AdaptiveResult(
        energy_joule=energy_prev,
        inductance_h=2.0 * energy_prev,
        rel_error_estimate=float("nan"),
        samples=params,
    )


def inductance_lm_toroidal_general(
    l: int,
    m: int,
    *,
    radius: float = 1.0,
    current: float = 1.0,
    mu: float = MU0,
    tol: float = 1.0e-2,
    max_refinements: int = 5,
) -> AdaptiveResult:
    """Inductance from full 3D sampling for toroidal mode (l,m)."""
    if l < 1:
        raise ValueError("l must be >= 1")
    if abs(m) > l:
        raise ValueError("|m| must be <= l")

    def run_once(n_r_inside: int, n_r_outside: int, n_theta: int, n_phi: int) -> float:
        return _energy_integral_general(
            l,
            m,
            radius=radius,
            current=current,
            mu=mu,
            n_r_inside=n_r_inside,
            n_r_outside=n_r_outside,
            n_theta=n_theta,
            n_phi=n_phi,
        )

    return _adaptive_refine(
        run_once,
        initial={"n_r_inside": 12, "n_r_outside": 24, "n_theta": 24, "n_phi": 48},
        tol=tol,
        max_refinements=max_refinements,
    )


def inductance_l0_toroidal_axisymmetric(
    l: int,
    *,
    radius: float = 1.0,
    current: float = 1.0,
    mu: float = MU0,
    tol: float = 1.0e-2,
    max_refinements: int = 5,
) -> AdaptiveResult:
    """Inductance from axisymmetric sampling for toroidal mode (l,m=0)."""
    if l < 1:
        raise ValueError("l must be >= 1")

    def run_once(n_r_inside: int, n_r_outside: int, n_theta: int) -> float:
        return _energy_integral_m0_axisymmetric(
            l,
            radius=radius,
            current=current,
            mu=mu,
            n_r_inside=n_r_inside,
            n_r_outside=n_r_outside,
            n_theta=n_theta,
        )

    return _adaptive_refine(
        run_once,
        initial={"n_r_inside": 12, "n_r_outside": 24, "n_theta": 24},
        tol=tol,
        max_refinements=max_refinements,
    )


def verify_l_mode(
    l: int,
    *,
    radius: float = 1.0,
    current: float = 1.0,
    mu: float = MU0,
    tol: float = 1.0e-2,
) -> Dict[str, float]:
    """Check method agreement for m=0 and m-independence for m=l."""
    gen_m0 = inductance_lm_toroidal_general(l, 0, radius=radius, current=current, mu=mu, tol=tol)
    fast_m0 = inductance_l0_toroidal_axisymmetric(l, radius=radius, current=current, mu=mu, tol=tol)
    m_test = l
    gen_m1 = inductance_lm_toroidal_general(l, m_test, radius=radius, current=current, mu=mu, tol=tol)

    rel_general_vs_fast = abs(gen_m0.inductance_h - fast_m0.inductance_h) / max(abs(fast_m0.inductance_h), 1.0e-30)
    rel_m_independence = abs(gen_m1.inductance_h - gen_m0.inductance_h) / max(abs(gen_m0.inductance_h), 1.0e-30)

    return {
        "l": float(l),
        "inductance_h": float(fast_m0.inductance_h),
        "energy_joule": float(fast_m0.energy_joule),
        "rel_diff_general_m0": float(rel_general_vs_fast),
        "rel_diff_m1_m0": float(rel_m_independence),
        "m_test": float(m_test),
    }


def _load_existing_table(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _write_table(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "l",
        "energy_joule",
        "inductance_h",
        "rel_diff_general_m0",
        "rel_diff_m1_m0",
        "m_test",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: int(x["l"])):
            writer.writerow(row)


def sweep_unit_sphere_inductance(
    lmax: int,
    *,
    data_path: Path | str = Path("toroidal_inductance/data/inductance_unit_sphere.csv"),
    tol: float = 1.0e-2,
) -> List[Dict[str, float]]:
    """Resume-capable L sweep on unit sphere. Computes first uncomputed l and onward."""
    if lmax < 1:
        raise ValueError("lmax must be >= 1")

    path = Path(data_path)
    rows = _load_existing_table(path)
    by_l = {int(r["l"]): r for r in rows}

    start_l = 1
    while start_l <= lmax and start_l in by_l:
        start_l += 1

    if start_l > lmax:
        return sorted(by_l.values(), key=lambda x: int(x["l"]))

    for l in range(start_l, lmax + 1):
        if l in by_l:
            continue
        result = verify_l_mode(l, tol=tol)
        by_l[l] = result
        _write_table(path, list(by_l.values()))

    return sorted(by_l.values(), key=lambda x: int(x["l"]))


def load_and_scale_inductance_table(
    *,
    radius_m: float,
    mu_r: float = 1.0,
    unit_radius_m: float = 1.0,
    data_path: Path | str = Path("toroidal_inductance/data/inductance_unit_sphere.csv"),
) -> List[Dict[str, float]]:
    """Load unit-sphere inductance table and scale to a different sphere size/permeability."""
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if unit_radius_m <= 0.0:
        raise ValueError("unit_radius_m must be positive")
    if mu_r <= 0.0:
        raise ValueError("mu_r must be positive")

    rows = _load_existing_table(Path(data_path))
    scale = mu_r * (radius_m / unit_radius_m)

    scaled: List[Dict[str, float]] = []
    for row in rows:
        scaled.append(
            {
                "l": row["l"],
                "inductance_h": row["inductance_h"] * scale,
                "energy_joule": row["energy_joule"] * scale,
            }
        )
    return scaled
