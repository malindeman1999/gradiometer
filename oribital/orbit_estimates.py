"""
Estimate orbital parameters for a circular polar orbit around Europa
and relate the orbit to the workflow GUI sphere mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repo root is on sys.path when running this module directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.plotting.render_phasor_maps import _build_mesh

# Matches workflow_* defaults (Europa mean radius in meters)
EUROPA_RADIUS_M = 1.56e6
# Europa GM (m^3/s^2), computed from M=4.80e22 kg (NSSDC Jovian Satellite Fact Sheet):
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/joviansatfact.html
# and G=6.67430e-11 m^3 kg^-1 s^-2 (CODATA 2018; e.g., https://pmc.ncbi.nlm.nih.gov/articles/PMC9890581/).
EUROPA_MU_M3_S2 = 3.203664e12
# Europa rotation/orbital period (tidally locked) in days from NSSDC fact sheet.
EUROPA_ROTATION_DAYS = 3.551181
# Jupiter rotation period used by ambient driver (hours).
JUPITER_ROTATION_PERIOD_H = 9.925
# Ambient field amplitude used by ambient_driver (tesla).
AMBIENT_FIELD_AMPLITUDE_T = 1e-6

DEFAULT_ALTITUDE_M = 100e3
DEFAULT_SUBDIVISIONS = 3


@dataclass(frozen=True)
class OrbitEstimate:
    altitude_m: float
    radius_m: float
    mu_m3_s2: float
    omega_rad_s: float
    speed_m_s: float
    period_s: float
    face_spacing_m: float
    face_transit_time_s: float
    europa_rotation_omega_rad_s: float
    europa_rotation_angle_rad: float
    face_angle_rad: float
    face_drift_count: float
    revisit_orbits: int
    revisit_time_s: float
    field_period_s: float
    phase_coverage_orbits: int
    phase_coverage_time_s: float
    phase_coverage_gap_s: float
    phase_bin_zero_count: int
    phase_bin_mean: float
    phase_bin_std: float
    phase_bin_zero_hours: list[float]


def circular_orbit_omega(mu_m3_s2: float, radius_m: float, altitude_m: float) -> float:
    r_orbit = float(radius_m + altitude_m)
    if r_orbit <= 0.0:
        raise ValueError("Orbit radius must be positive.")
    return math.sqrt(float(mu_m3_s2) / (r_orbit ** 3))


def circular_orbit_speed(mu_m3_s2: float, radius_m: float, altitude_m: float) -> float:
    r_orbit = float(radius_m + altitude_m)
    omega = circular_orbit_omega(mu_m3_s2, radius_m, altitude_m)
    return omega * r_orbit


def _mean_face_center_spacing_m(subdivisions: int, radius_m: float) -> float:
    _, _, centers = _build_mesh(radius_m, subdivisions=subdivisions, stride=1)
    centers = centers.to(dtype=torch.float64)
    count = int(centers.shape[0])
    if count < 2:
        return 0.0
    sample_cap = 2000
    if count > sample_cap:
        idx = torch.randperm(count)[:sample_cap]
        centers = centers[idx]
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(float("inf"))
    nn = dists.min(dim=1).values
    return float(nn.mean().item())


def estimate_polar_orbit(
    altitude_m: float = DEFAULT_ALTITUDE_M,
    subdivisions: int = DEFAULT_SUBDIVISIONS,
    radius_m: float = EUROPA_RADIUS_M,
    mu_m3_s2: float = EUROPA_MU_M3_S2,
) -> OrbitEstimate:
    omega = circular_orbit_omega(mu_m3_s2, radius_m, altitude_m)
    speed = circular_orbit_speed(mu_m3_s2, radius_m, altitude_m)
    period = 2.0 * math.pi / omega

    orbit_radius_m = float(radius_m + altitude_m)
    spacing_m = _mean_face_center_spacing_m(subdivisions, orbit_radius_m)
    transit_time_s = spacing_m / speed if speed > 0.0 else float("inf")
    rot_period_s = float(EUROPA_ROTATION_DAYS) * 86400.0
    rot_omega = 2.0 * math.pi / rot_period_s
    rot_angle = rot_omega * period
    face_angle = spacing_m / orbit_radius_m if orbit_radius_m > 0.0 else 0.0
    face_drift = rot_angle / face_angle if face_angle > 0.0 else 0.0
    revisit_orbits = estimate_revisit_orbits(face_drift)
    revisit_time = revisit_orbits * period if revisit_orbits > 0 else float("inf")
    field_period_s = JUPITER_ROTATION_PERIOD_H * 3600.0
    phase_orbits, phase_time, phase_gap = estimate_phase_coverage_time(
        revisit_time, field_period_s, 0.5 * transit_time_s
    )
    zero_bins, mean_bins, std_bins, zero_hours = estimate_phase_bin_stats(
        revisit_time, field_period_s, transit_time_s, phase_time
    )

    return OrbitEstimate(
        altitude_m=float(altitude_m),
        radius_m=float(radius_m),
        mu_m3_s2=float(mu_m3_s2),
        omega_rad_s=float(omega),
        speed_m_s=float(speed),
        period_s=float(period),
        face_spacing_m=float(spacing_m),
        face_transit_time_s=float(transit_time_s),
        europa_rotation_omega_rad_s=float(rot_omega),
        europa_rotation_angle_rad=float(rot_angle),
        face_angle_rad=float(face_angle),
        face_drift_count=float(face_drift),
        revisit_orbits=int(revisit_orbits),
        revisit_time_s=float(revisit_time),
        field_period_s=float(field_period_s),
        phase_coverage_orbits=int(phase_orbits),
        phase_coverage_time_s=float(phase_time),
        phase_coverage_gap_s=float(phase_gap),
        phase_bin_zero_count=int(zero_bins),
        phase_bin_mean=float(mean_bins),
        phase_bin_std=float(std_bins),
        phase_bin_zero_hours=[float(v) for v in zero_hours],
    )


def estimate_default_orbit() -> OrbitEstimate:
    return estimate_polar_orbit(
        altitude_m=DEFAULT_ALTITUDE_M,
        subdivisions=DEFAULT_SUBDIVISIONS,
        radius_m=EUROPA_RADIUS_M,
        mu_m3_s2=EUROPA_MU_M3_S2,
    )


def estimate_revisit_orbits(
    faces_per_orbit: float, tolerance_faces: float = 0.5, max_orbits: int = 10000
) -> int:
    if faces_per_orbit <= 0.0:
        return 0
    tol = max(0.0, float(tolerance_faces))
    for n in range(1, int(max_orbits) + 1):
        frac = (faces_per_orbit * n) % 1.0
        if frac <= tol:
            return n
    return 0


def estimate_phase_coverage_time(
    revisit_interval_s: float,
    period_s: float,
    tolerance_s: float,
    max_samples: int = 20000,
) -> tuple[int, float, float]:
    if revisit_interval_s <= 0.0 or period_s <= 0.0:
        return 0, float("inf"), float("inf")
    tol = max(0.0, float(tolerance_s))
    phases = [0.0]
    for n in range(2, max_samples + 1):
        phases.append((revisit_interval_s * (n - 1)) % period_s)
        phases_sorted = np.sort(np.asarray(phases, dtype=float))
        gaps = np.diff(phases_sorted, append=phases_sorted[0] + period_s)
        max_gap = float(gaps.max()) if gaps.size else float("inf")
        if max_gap <= 2.0 * tol:
            return n - 1, (n - 1) * revisit_interval_s, max_gap
    return 0, float("inf"), float("inf")


def estimate_phase_bin_stats(
    revisit_interval_s: float,
    period_s: float,
    bin_width_s: float,
    coverage_time_s: float,
) -> tuple[int, float, float, list[float]]:
    if (
        revisit_interval_s <= 0.0
        or period_s <= 0.0
        or bin_width_s <= 0.0
        or coverage_time_s <= 0.0
    ):
        return 0, 0.0, 0.0, []
    sample_times = np.arange(0.0, coverage_time_s + 1e-9, revisit_interval_s)
    sample_mod = np.mod(sample_times, period_s)
    bin_count = int(math.ceil(period_s / bin_width_s))
    edges = np.linspace(0.0, period_s, bin_count + 1)
    counts, _ = np.histogram(sample_mod, bins=edges, range=(0.0, period_s))
    zero_bins = int((counts == 0).sum())
    mean = float(counts.mean()) if counts.size else 0.0
    std = float(counts.std()) if counts.size else 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    zero_hours = [float(c / 3600.0) for c in centers[counts == 0]]
    return zero_bins, mean, std, zero_hours


def plot_field_revisit(
    revisit_interval_s: float,
    period_s: float,
    amplitude_t: float,
    out_path: Path,
    coverage_time_s: float | None = None,
    phase_bin_s: float | None = None,
    show: bool = False,
) -> None:
    if period_s <= 0.0:
        raise ValueError("Field period must be positive.")
    t_end = 2.0 * period_s if coverage_time_s is None else float(coverage_time_s)
    t = np.linspace(0.0, t_end, 4000)
    y = amplitude_t * np.sin(2.0 * math.pi * t / period_s)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 9), sharey=False)

    if revisit_interval_s > 0.0 and math.isfinite(revisit_interval_s):
        sample_times = np.arange(0.0, t_end + 1e-9, revisit_interval_s)
        sample_vals = amplitude_t * np.sin(2.0 * math.pi * sample_times / period_s)
        ax0.scatter(
            sample_times / 3600.0,
            sample_vals,
            color="tab:red",
            s=9,
            zorder=3,
            label="same face",
        )
    else:
        sample_times = np.array([], dtype=float)
        sample_vals = np.array([], dtype=float)

    ax0.set_xlabel("Time (hours)")
    ax0.set_ylabel("Magnetic field (T)")
    ax0.set_title("Ambient field at same equatorial face (first crossing only)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    t_mod = np.mod(t, period_s)
    if revisit_interval_s > 0.0 and math.isfinite(revisit_interval_s):
        sample_mod = np.mod(sample_times, period_s)
        ax1.scatter(
            sample_mod / 3600.0,
            sample_vals,
            color="tab:red",
            s=9,
            zorder=3,
            label="same face",
        )
    else:
        sample_mod = np.array([], dtype=float)

    ax1.set_xlabel("Time modulo Jupiter period (hours)")
    ax1.set_ylabel("Magnetic field (T)")
    ax1.set_title("Ambient field modulo Jupiter rotation period (first crossing only)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    if phase_bin_s is None or phase_bin_s <= 0.0:
        phase_bin_s = period_s
    bin_count = int(math.ceil(period_s / phase_bin_s))
    edges = np.linspace(0.0, period_s, bin_count + 1)
    counts, _ = np.histogram(sample_mod, bins=edges, range=(0.0, period_s))
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    ax2.bar(
        centers / 3600.0,
        counts,
        width=widths / 3600.0,
        color="tab:gray",
        edgecolor="black",
        align="center",
    )
    ax2.set_xlabel("Phase bin center (hours)")
    ax2.set_ylabel("Measurement count")
    ax2.set_title("Counts per phase bin (first crossing only)")
    ax2.grid(True, alpha=0.25, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_ground_track(
    orbit_omega_rad_s: float,
    rotation_omega_rad_s: float,
    duration_s: float,
    out_path: Path,
    inclination_rad: float,
    show: bool = False,
) -> None:
    if duration_s <= 0.0:
        raise ValueError("Duration must be positive.")
    samples = 6000
    t = np.linspace(0.0, duration_s, samples)
    nu = orbit_omega_rad_s * t
    x = np.cos(nu)
    y = np.sin(nu) * math.cos(inclination_rad)
    z = np.sin(nu) * math.sin(inclination_rad)

    lon_inertial = np.arctan2(y, x)
    lon_body = lon_inertial - rotation_omega_rad_s * t
    lon_body = (lon_body + math.pi) % (2.0 * math.pi) - math.pi
    lat = np.arcsin(np.clip(z, -1.0, 1.0))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.degrees(lon_body), np.degrees(lat), color="tab:green", linewidth=1.0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    inc_deg = math.degrees(inclination_rad)
    ax.set_title(f"Ground track (equirectangular projection, inc={inc_deg:.1f} deg)")
    ax.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    est = estimate_default_orbit()
    plot_field_revisit(
        est.revisit_time_s,
        est.field_period_s,
        AMBIENT_FIELD_AMPLITUDE_T,
        Path("figures/oribital_field_revisit.png"),
        coverage_time_s=est.phase_coverage_time_s,
        phase_bin_s=est.face_transit_time_s,
        show=True,
    )
    plot_ground_track(
        est.omega_rad_s,
        est.europa_rotation_omega_rad_s,
        est.period_s * 50.0,
        Path("figures/oribital_ground_track.png"),
        inclination_rad=math.radians(80.0),
        show=True,
    )
    print("Europa polar circular orbit (100 km altitude)")
    print(f"omega = {est.omega_rad_s:.6e} rad/s")
    print(f"speed = {est.speed_m_s:.3f} m/s")
    print(f"period = {est.period_s/3600.0:.3f} hours")
    print(f"mesh mean spacing = {est.face_spacing_m/1000.0:.3f} km")
    print(f"face transit time = {est.face_transit_time_s:.3f} s")
    print(f"Europa rotation omega = {est.europa_rotation_omega_rad_s:.6e} rad/s")
    print(f"Europa rotation during orbit = {est.europa_rotation_angle_rad:.6e} rad")
    print(f"face angle (equator) = {est.face_angle_rad:.6e} rad")
    print(f"face drift per orbit = {est.face_drift_count:.3f} faces")
    print(f"revisit within 0.5 face = {est.revisit_orbits} orbits")
    print(f"revisit time = {est.revisit_time_s/3600.0:.3f} hours")
    print(f"field period (Jupiter rotation) = {est.field_period_s/3600.0:.3f} hours")
    print(f"phase coverage gap = {est.phase_coverage_gap_s:.3f} s")
    print(f"phase coverage time = {est.phase_coverage_time_s/3600.0:.3f} hours")
    print(f"phase coverage orbits = {est.phase_coverage_orbits}")
    print(f"phase bins with zero = {est.phase_bin_zero_count}")
    print(f"phase bin mean = {est.phase_bin_mean:.3f}")
    print(f"phase bin std = {est.phase_bin_std:.3f}")
    if est.phase_bin_zero_hours:
        print("phase bins with zero (hours): " + ", ".join(f"{h:.3f}" for h in est.phase_bin_zero_hours))
    print("Note: counts/plots use only the first equator crossing each orbit; continuous sampling would roughly double counts.")
