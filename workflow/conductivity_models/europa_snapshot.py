"""Europa snapshot conductivity model for nonuniform workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from europa_model.transforms import sh_inverse


@dataclass(frozen=True)
class EuropaSnapshotConfig:
    chem_contrast: float = 0.35
    n_exchange_sites: int = 5
    exchange_amp: float = 0.45
    exchange_width_deg: float = 18.0
    exchange_target_max_s: float = 1.0e5
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
    n_nodes = int(positions.shape[0])
    grid_lmax = max(1, int(round(math.sqrt(float(n_nodes)))) - 1)

    # "Convection" component stored under x_chem key for backward compatibility.
    # Requested target: l=32, m=3 (or nearest available if grid lmax is smaller).
    l_conv = min(32, grid_lmax)
    m_conv = min(3, l_conv)
    coeffs_conv = torch.zeros((l_conv + 1, 2 * l_conv + 1), dtype=torch.complex128)
    c_conv = 1.0 + 0.0j
    coeffs_conv[l_conv, l_conv + m_conv] = c_conv
    if m_conv > 0:
        coeffs_conv[l_conv, l_conv - m_conv] = ((-1) ** m_conv) * np.conj(c_conv)
    conv_recon = sh_inverse(coeffs_conv, positions, weights)
    chem_raw = conv_recon.real.to(torch.float64)
    x_chem = float(cfg.chem_contrast) * _weighted_standardize(chem_raw, weights)

    equatorial_env = torch.exp(-0.5 * (lat / math.radians(35.0)) ** 2)
    flow_raw = equatorial_env * (torch.sin(2.0 * lon + p2) + 0.5 * torch.sin(3.0 * lon - p1))
    # Override flow proxy with a fixed large-scale real harmonic.
    # Requested target: l=4, m=0 (or nearest available if grid lmax is smaller).
    l_flow = min(4, grid_lmax)
    m_flow = 0
    coeffs_flow = torch.zeros((l_flow + 1, 2 * l_flow + 1), dtype=torch.complex128)
    coeffs_flow[l_flow, l_flow + m_flow] = 1.0 + 0.0j
    flow_recon = sh_inverse(coeffs_flow, positions, weights)
    flow_raw = flow_recon.real.to(torch.float64)
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
    # Rescale exchange so exchange-only conductivity has controlled peak magnitude.
    target_exchange_max = max(0.0, float(cfg.exchange_target_max_s))
    if n_sites > 0 and target_exchange_max > 0.0 and sigma0 > 0.0:
        w = weights.to(torch.float64)
        wsum = float(w.sum().item())

        def _exchange_only_peak(scale: float) -> float:
            s = sigma0 * torch.exp(float(scale) * x_exchange)
            mean_now = float((w * s).sum().item() / max(wsum, 1e-30))
            if mean_now > 0.0:
                s = s * (sigma0 / mean_now)
            return float(s.max().item())

        if target_exchange_max <= sigma0:
            scale_exchange = 0.0
        else:
            lo, hi = 0.0, 1.0
            peak_hi = _exchange_only_peak(hi)
            while peak_hi < target_exchange_max and hi < 1e3:
                hi *= 2.0
                peak_hi = _exchange_only_peak(hi)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if _exchange_only_peak(mid) < target_exchange_max:
                    lo = mid
                else:
                    hi = mid
            scale_exchange = hi
        x_exchange = float(scale_exchange) * x_exchange

    bg_raw = (
        torch.sin(lon + p1) * torch.cos(lat)
        + 0.5 * torch.sin(2.0 * lon - p3) * torch.sin(2.0 * lat)
    )
    # Override residual background with a fixed higher-degree real harmonic.
    # Requested target: l=16, m=8 (or nearest available if grid lmax is smaller).
    n_nodes = int(positions.shape[0])
    grid_lmax = max(1, int(round(math.sqrt(float(n_nodes)))) - 1)
    l_bg = min(16, grid_lmax)
    m_bg = min(8, l_bg)
    coeffs_bg = torch.zeros((l_bg + 1, 2 * l_bg + 1), dtype=torch.complex128)
    c_bg = 1.0 + 0.0j
    coeffs_bg[l_bg, l_bg + m_bg] = c_bg
    if m_bg > 0:
        coeffs_bg[l_bg, l_bg - m_bg] = ((-1) ** m_bg) * np.conj(c_bg)
    bg_recon = sh_inverse(coeffs_bg, positions, weights)
    bg_raw = bg_recon.real.to(torch.float64)
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
        "snapshot_exchange_target_max_s": float(target_exchange_max),
    }
