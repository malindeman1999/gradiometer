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

# Ensure repo root is on sys.path when running this module directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
DEFAULT_LMAX = 36


@dataclass(frozen=True)
class OrbitEstimate:
    altitude_m: float
    radius_m: float
    mu_m3_s2: float
    omega_rad_s: float
    speed_m_s: float
    period_s: float
    node_spacing_m: float
    node_transit_time_s: float
    europa_rotation_omega_rad_s: float
    europa_rotation_angle_rad: float
    node_angle_rad: float
    node_drift_count: float
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


def _node_angle_width_rad(lmax: int) -> float:
    node_count = max(1, (int(lmax) + 1) ** 2)
    solid_angle = 4.0 * math.pi / node_count
    return math.sqrt(solid_angle)


def estimate_polar_orbit(
    altitude_m: float = DEFAULT_ALTITUDE_M,
    lmax: int = DEFAULT_LMAX,
    radius_m: float = EUROPA_RADIUS_M,
    mu_m3_s2: float = EUROPA_MU_M3_S2,
) -> OrbitEstimate:
    omega = circular_orbit_omega(mu_m3_s2, radius_m, altitude_m)
    speed = circular_orbit_speed(mu_m3_s2, radius_m, altitude_m)
    period = 2.0 * math.pi / omega

    orbit_radius_m = float(radius_m + altitude_m)
    node_angle = _node_angle_width_rad(lmax)
    spacing_m = node_angle * orbit_radius_m
    transit_time_s = spacing_m / speed if speed > 0.0 else float("inf")
    rot_period_s = float(EUROPA_ROTATION_DAYS) * 86400.0
    rot_omega = 2.0 * math.pi / rot_period_s
    rot_angle = rot_omega * period
    node_drift = rot_angle / node_angle if node_angle > 0.0 else 0.0
    field_period_s = JUPITER_ROTATION_PERIOD_H * 3600.0
    phase_orbits, phase_time, phase_gap = estimate_phase_coverage_time(
        period, field_period_s, 0.5 * transit_time_s
    )
    zero_bins, mean_bins, std_bins, zero_hours = estimate_phase_bin_stats(
        period, field_period_s, transit_time_s, phase_time
    )

    return OrbitEstimate(
        altitude_m=float(altitude_m),
        radius_m=float(radius_m),
        mu_m3_s2=float(mu_m3_s2),
        omega_rad_s=float(omega),
        speed_m_s=float(speed),
        period_s=float(period),
        node_spacing_m=float(spacing_m),
        node_transit_time_s=float(transit_time_s),
        europa_rotation_omega_rad_s=float(rot_omega),
        europa_rotation_angle_rad=float(rot_angle),
        node_angle_rad=float(node_angle),
        node_drift_count=float(node_drift),
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
        lmax=DEFAULT_LMAX,
        radius_m=EUROPA_RADIUS_M,
        mu_m3_s2=EUROPA_MU_M3_S2,
    )




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
    sample_times_s: np.ndarray | None = None,
) -> tuple[int, float, float, list[float]]:
    if period_s <= 0.0 or bin_width_s <= 0.0:
        return 0, 0.0, 0.0, []
    if sample_times_s is not None and len(sample_times_s) > 0:
        sample_times = np.asarray(sample_times_s, dtype=float)
        sample_mod = np.mod(sample_times, period_s)
    else:
        if revisit_interval_s <= 0.0 or coverage_time_s <= 0.0:
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
    sample_times_s: np.ndarray | None = None,
    show: bool = False,
) -> None:
    if period_s <= 0.0:
        raise ValueError("Field period must be positive.")
    t_end = 2.0 * period_s if coverage_time_s is None else float(coverage_time_s)

    if sample_times_s is not None and len(sample_times_s) > 0:
        sample_times = np.asarray(sample_times_s, dtype=float)
        if sample_times.size:
            t_end = float(sample_times.max())
        t = np.linspace(0.0, t_end, 2000)
        y = amplitude_t * np.sin(2.0 * math.pi * t / period_s)
        sample_vals = amplitude_t * np.sin(2.0 * math.pi * sample_times / period_s)
    elif revisit_interval_s > 0.0 and math.isfinite(revisit_interval_s):
        sample_times = np.arange(0.0, t_end + 1e-9, revisit_interval_s)
        t = np.linspace(0.0, t_end, 4000)
        y = amplitude_t * np.sin(2.0 * math.pi * t / period_s)
        sample_vals = amplitude_t * np.sin(2.0 * math.pi * sample_times / period_s)
    else:
        sample_times = np.array([], dtype=float)
        sample_vals = np.array([], dtype=float)
        t = np.linspace(0.0, t_end, 4000)
        y = amplitude_t * np.sin(2.0 * math.pi * t / period_s)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 9), sharey=False)
    ax0.plot(t / 3600.0, y, color="tab:blue", linestyle=":", linewidth=1.2, label="ambient")
    if sample_times.size:
        ax0.scatter(
            sample_times / 3600.0,
            sample_vals,
            color="tab:red",
            s=9,
            zorder=3,
            label="crossings",
        )

    ax0.set_xlabel("Time (hours)")
    ax0.set_ylabel("Magnetic field (T)")
    ax0.set_title("Ambient field at crossings (ground track)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    t_mod = np.mod(t, period_s)
    if sample_times_s is not None and len(sample_times_s) > 0:
        sample_mod = np.mod(sample_times, period_s)
        ax1.scatter(
            sample_mod / 3600.0,
            sample_vals,
            color="tab:red",
            s=9,
            zorder=3,
            label="crossings",
        )
    elif revisit_interval_s > 0.0 and math.isfinite(revisit_interval_s):
        sample_mod = np.mod(sample_times, period_s)
        ax1.scatter(
            sample_mod / 3600.0,
            sample_vals,
            color="tab:red",
            s=9,
            zorder=3,
            label="crossings",
        )
    else:
        sample_mod = np.array([], dtype=float)

    ax1.set_xlabel("Time modulo Jupiter period (hours)")
    ax1.set_ylabel("Magnetic field (T)")
    ax1.set_title("Ambient field modulo Jupiter rotation period (crossings)")
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
    ax2.set_title("Counts per phase bin (crossings)")
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
    node_angle_rad: float,
    duration_s: float,
    out_path: Path,
    inclination_rad: float,
    show: bool = False,
) -> np.ndarray:
    if duration_s <= 0.0:
        raise ValueError("Duration must be positive.")
    half_width = 0.5 * node_angle_rad
    if half_width <= 0.0:
        raise ValueError("Node angle must be positive.")

    def _wrap_delta(a: np.ndarray, b: float) -> np.ndarray:
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def _compute_track(dur_s: float, n_samples: int):
        t = np.linspace(0.0, dur_s, n_samples)
        nu = orbit_omega_rad_s * t
        x = np.cos(nu)
        y = np.sin(nu) * math.cos(inclination_rad)
        z = np.sin(nu) * math.sin(inclination_rad)
        lon_inertial = np.arctan2(y, x)
        lon_body = lon_inertial - rotation_omega_rad_s * t
        lon_body = (lon_body + math.pi) % (2.0 * math.pi) - math.pi
        lat = np.arcsin(np.clip(z, -1.0, 1.0))
        return t, lon_body, lat

    # Expand duration until we find enough crossings.
    period = 2.0 * math.pi / orbit_omega_rad_s
    max_duration = max(duration_s, period * 800.0)
    cur_duration = duration_s
    crossing_idx: list[int] = []
    crossing_times: list[float] = []
    t = lon_body = lat = None
    for _ in range(6):
        samples = max(20000, int(cur_duration / period * 2000))
        t, lon_body, lat = _compute_track(cur_duration, samples)
        lon0 = lon_body[0]
        lat0 = lat[0]
        in_box = (np.abs(_wrap_delta(lon_body, lon0)) <= half_width) & (np.abs(lat - lat0) <= half_width)
        crossing_idx = []
        for i in range(1, len(in_box)):
            if in_box[i] and not in_box[i - 1]:
                crossing_idx.append(i)
        if in_box[0]:
            crossing_idx = [0] + crossing_idx
        crossing_times = [float(t[i]) for i in crossing_idx]
        if len(crossing_idx) >= 10:
            break
        if cur_duration >= max_duration:
            break
        cur_duration = min(cur_duration * 2.0, max_duration)

    if len(crossing_idx) < 2 or t is None or lon_body is None or lat is None:
        raise RuntimeError("Did not find enough node crossings within the sampled duration.")

    # Limit to first 10 crossings.
    crossing_idx = crossing_idx[:10]
    crossing_times = crossing_times[:10]
    first_idx = crossing_idx[0]
    last_idx = crossing_idx[-1]

    lon_seg = lon_body[first_idx:last_idx + 1]
    lat_seg = lat[first_idx:last_idx + 1]
    track_hours = float(t[last_idx] - t[first_idx]) / 3600.0
    if len(crossing_times) >= 2:
        avg_dt = float(np.mean(np.diff(crossing_times)))
        print(f"Average crossing interval: {avg_dt/3600.0:.3f} hours ({avg_dt:.1f} s)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.degrees(lon_seg), np.degrees(lat_seg), color="tab:green", linewidth=1.0)
    ax.scatter(
        np.degrees(lon_body[crossing_idx]),
        np.degrees(lat[crossing_idx]),
        color="tab:red",
        s=18,
        zorder=3,
        label="crossings",
    )

    # Draw node area (approx square of width node_angle_rad centered at first visit).
    lon_c = lon0
    lat_c = lat0
    width_deg = math.degrees(node_angle_rad)
    half_deg = 0.5 * width_deg
    rect = plt.Rectangle(
        (math.degrees(lon_c) - half_deg, math.degrees(lat_c) - half_deg),
        width_deg,
        width_deg,
        fill=False,
        edgecolor="tab:orange",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    inc_deg = math.degrees(inclination_rad)
    ax.set_title(
        f"Ground track (first 10 crossings, {track_hours:.2f} hr, inc={inc_deg:.1f} deg)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return np.asarray(crossing_times, dtype=float)


if __name__ == "__main__":
    est = estimate_default_orbit()
    track_duration = est.period_s * 200.0
    crossing_times = plot_ground_track(
        est.omega_rad_s,
        est.europa_rotation_omega_rad_s,
        est.node_angle_rad,
        track_duration,
        Path("figures/oribital_ground_track.png"),
        inclination_rad=math.radians(80.0),
        show=True,
    )
    plot_field_revisit(
        est.period_s,
        est.field_period_s,
        AMBIENT_FIELD_AMPLITUDE_T,
        Path("figures/oribital_field_revisit.png"),
        coverage_time_s=est.phase_coverage_time_s,
        phase_bin_s=est.node_transit_time_s,
        sample_times_s=crossing_times,
        show=True,
    )
    print("Europa polar circular orbit (100 km altitude)")
    print(f"omega = {est.omega_rad_s:.6e} rad/s")
    print(f"speed = {est.speed_m_s:.3f} m/s")
    print(f"period = {est.period_s/3600.0:.3f} hours")
    print(f"node mean spacing = {est.node_spacing_m/1000.0:.3f} km")
    print(f"node transit time = {est.node_transit_time_s:.3f} s")
    print(f"Europa rotation omega = {est.europa_rotation_omega_rad_s:.6e} rad/s")
    print(f"Europa rotation during orbit = {est.europa_rotation_angle_rad:.6e} rad")
    print(f"node angle (equator) = {est.node_angle_rad:.6e} rad")
    print(f"node drift per orbit = {est.node_drift_count:.3f} nodes")
    print(f"field period (Jupiter rotation) = {est.field_period_s/3600.0:.3f} hours")
    zero_bins, mean_bins, std_bins, zero_hours = estimate_phase_bin_stats(
        est.period_s,
        est.field_period_s,
        est.node_transit_time_s,
        est.phase_coverage_time_s,
        sample_times_s=crossing_times,
    )
    print(f"phase coverage gap = {est.phase_coverage_gap_s:.3f} s")
    print(f"phase coverage time = {est.phase_coverage_time_s/3600.0:.3f} hours")
    print(f"phase coverage orbits = {est.phase_coverage_orbits}")
    print(f"phase bin mean = {mean_bins:.3f}")
    print(f"phase bin std = {std_bins:.3f}")
