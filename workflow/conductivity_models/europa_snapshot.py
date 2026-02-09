"""Europa snapshot conductivity model for nonuniform workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EuropaSnapshotConfig:
    chem_contrast: float = 0.35
    n_exchange_sites: int = 4
    exchange_amp: float = 0.45
    exchange_width_deg: float = 18.0
    flow_anisotropy: float = 0.20
    background_amp: float = 0.08
    seed: int = 7


def _unit_vectors(positions: torch.Tensor) -> torch.Tensor:
    pos = positions.to(torch.float64)
    return pos / torch.linalg.norm(pos, dim=1, keepdim=True).clamp_min(1e-30)


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


def _spherical_gaussian(uhat: torch.Tensor, center_unit: torch.Tensor, width_deg: float) -> torch.Tensor:
    width_deg = max(1.0, float(width_deg))
    sigma = math.radians(width_deg)
    cosang = torch.clamp(uhat @ center_unit.to(torch.float64), -1.0, 1.0)
    ang = torch.arccos(cosang)
    return torch.exp(-0.5 * (ang / sigma) ** 2)


def _center_from_lat_lon_deg(lat_deg: float, lon_deg: float) -> torch.Tensor:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    clat = math.cos(lat)
    return torch.tensor([
        clat * math.cos(lon),
        clat * math.sin(lon),
        math.sin(lat),
    ], dtype=torch.float64)


def build_europa_snapshot_conductivity(
    positions: torch.Tensor,
    weights: torch.Tensor,
    sigma0: float,
    cfg: EuropaSnapshotConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
    """Build a static, physically motivated conductivity snapshot on the sphere."""
    cfg = cfg or EuropaSnapshotConfig()
    sigma0 = max(0.0, float(sigma0))
    if sigma0 <= 0.0:
        return torch.zeros(positions.shape[0], dtype=torch.float64), {
            "x_chem": torch.zeros(positions.shape[0], dtype=torch.float64),
            "x_exchange": torch.zeros(positions.shape[0], dtype=torch.float64),
            "x_flow": torch.zeros(positions.shape[0], dtype=torch.float64),
            "x_bg": torch.zeros(positions.shape[0], dtype=torch.float64),
            "snapshot_seed": int(cfg.seed),
        }

    rng = np.random.default_rng(int(cfg.seed))
    uhat = _unit_vectors(positions)
    x = uhat[:, 0]
    y = uhat[:, 1]
    z = uhat[:, 2]

    lon = torch.atan2(y, x)
    lat = torch.arcsin(torch.clamp(z, -1.0, 1.0))

    p1 = float(rng.uniform(0.0, 2.0 * math.pi))
    p2 = float(rng.uniform(0.0, 2.0 * math.pi))
    p3 = float(rng.uniform(0.0, 2.0 * math.pi))

    chem_raw = (
        0.9 * torch.cos(lat) * torch.cos(2.0 * lon + p1)
        + 0.6 * torch.sin(lat) * torch.sin(lon + p2)
        + 0.4 * torch.sin(2.0 * lat + p3)
    )
    x_chem = float(cfg.chem_contrast) * _weighted_standardize(chem_raw, weights)

    equatorial_env = torch.exp(-0.5 * (lat / math.radians(35.0)) ** 2)
    flow_raw = equatorial_env * (torch.sin(2.0 * lon + p2) + 0.5 * torch.sin(3.0 * lon - p1))
    x_flow = float(cfg.flow_anisotropy) * _weighted_standardize(flow_raw, weights)

    exchange = torch.zeros_like(lat, dtype=torch.float64)
    base_centers = [
        (-45.0, 210.0),
        (-10.0, 270.0),
        (25.0, 80.0),
        (5.0, 320.0),
        (40.0, 150.0),
        (-30.0, 30.0),
    ]
    n_sites = max(0, int(cfg.n_exchange_sites))
    for i in range(n_sites):
        if i < len(base_centers):
            c_lat, c_lon = base_centers[i]
        else:
            c_lat = float(rng.uniform(-65.0, 65.0))
            c_lon = float(rng.uniform(0.0, 360.0))
        center = _center_from_lat_lon_deg(c_lat, c_lon)
        amp = float(cfg.exchange_amp) * float(rng.uniform(0.5, 1.2))
        sign = 1.0 if float(rng.uniform(0.0, 1.0)) > 0.2 else -1.0
        exchange = exchange + sign * amp * _spherical_gaussian(uhat, center, float(cfg.exchange_width_deg))
    x_exchange = _weighted_standardize(exchange, weights) if n_sites > 0 else exchange

    bg_raw = (
        torch.sin(lon + p1) * torch.cos(lat)
        + 0.5 * torch.sin(2.0 * lon - p3) * torch.sin(2.0 * lat)
    )
    x_bg = float(cfg.background_amp) * _weighted_standardize(bg_raw, weights)

    x_total = x_chem + x_exchange + x_flow + x_bg
    x_total = _weighted_standardize(x_total, weights)

    sigma = sigma0 * torch.exp(x_total)
    w = weights.to(torch.float64)
    wsum = float(w.sum().item())
    mean_now = float((w * sigma).sum().item() / max(wsum, 1e-30))
    if mean_now > 0.0:
        sigma = sigma * (sigma0 / mean_now)

    return sigma.to(torch.float64), {
        "x_chem": x_chem.to(torch.float64),
        "x_exchange": x_exchange.to(torch.float64),
        "x_flow": x_flow.to(torch.float64),
        "x_bg": x_bg.to(torch.float64),
        "snapshot_seed": int(cfg.seed),
        "snapshot_n_exchange_sites": int(n_sites),
        "snapshot_exchange_width_deg": float(cfg.exchange_width_deg),
    }
