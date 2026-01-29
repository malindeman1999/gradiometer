"""
Print equator-crossing longitudes for polar and 45-deg inclined orbits.
Assumes same orbital period and initial crossing at longitude 0.
"""
from __future__ import annotations

import math

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oribital.orbit_estimates import (
    EUROPA_MU_M3_S2,
    EUROPA_RADIUS_M,
    EUROPA_ROTATION_DAYS,
    DEFAULT_ALTITUDE_M,
)


def _wrap_lon_rad(lon: float) -> float:
    return (lon + math.pi) % (2.0 * math.pi) - math.pi


def orbit_period_s(mu_m3_s2: float, radius_m: float, altitude_m: float) -> float:
    r_orbit = radius_m + altitude_m
    omega = math.sqrt(mu_m3_s2 / (r_orbit ** 3))
    return 2.0 * math.pi / omega


def equator_crossing_longitudes(
    period_s: float,
    rotation_omega_rad_s: float,
    orbits: int,
) -> list[float]:
    times = []
    for n in range(orbits):
        times.append(n * period_s)
        times.append(n * period_s + 0.5 * period_s)
    longs = []
    for t in times:
        lon_body = _wrap_lon_rad(-rotation_omega_rad_s * t)
        longs.append(lon_body)
    return longs


def main() -> None:
    period_s = orbit_period_s(EUROPA_MU_M3_S2, EUROPA_RADIUS_M, DEFAULT_ALTITUDE_M)
    rotation_omega = 2.0 * math.pi / (EUROPA_ROTATION_DAYS * 86400.0)

    orbits = 10

    polar_lons = equator_crossing_longitudes(period_s, rotation_omega, orbits)
    incl_lons = equator_crossing_longitudes(period_s, rotation_omega, orbits)

    print("Assumptions: initial equator crossing at longitude 0, same period.")
    print(f"Orbits listed: {orbits} (2 crossings per orbit)")
    print("\nPolar orbit equator crossings (deg):")
    print(", ".join(f"{math.degrees(lon):.3f}" for lon in polar_lons))
    print("\n45 deg inclined orbit equator crossings (deg):")
    print(", ".join(f"{math.degrees(lon):.3f}" for lon in incl_lons))


if __name__ == "__main__":
    main()
