"""
Helper for configuring the ambient B driver (uniform along +X, +Y, or +Z) and attaching its spectrum to a PhasorSimulation.
"""
import math
from typing import Tuple

import torch

from europa.config import AmbientConfig
from phasor_data import PhasorSimulation


def _base_config(period_hours: float, amplitude_t: float, phase_rad: float) -> Tuple[AmbientConfig, float, float]:
    period_sec = period_hours * 3600.0
    omega = 2.0 * math.pi / period_sec
    ambient_cfg = AmbientConfig(
        omega_jovian=omega,
        amplitude_t=amplitude_t,
        phase_rad=phase_rad,
        spatial_mode="uniform",
    )
    return ambient_cfg, omega, period_sec


def build_ambient_driver_z(
    grid_cfg,
    *,
    period_hours: float = 9.925,
    amplitude_t: float = 1e-6,
    phase_rad: float = 0.0,
) -> Tuple[AmbientConfig, torch.Tensor, float]:
    """Ambient +Z: radial spectrum has only Y_1,0 non-zero."""
    ambient_cfg, _, period_sec = _base_config(period_hours, amplitude_t, phase_rad)
    lmax = grid_cfg.lmax
    B_radial_spec = torch.zeros((lmax + 1, 2 * lmax + 1), device=grid_cfg.device, dtype=torch.complex128)
    B_radial_spec[1, lmax] = amplitude_t
    return ambient_cfg, B_radial_spec, period_sec


def build_ambient_driver_x(
    grid_cfg,
    *,
    period_hours: float = 9.925,
    amplitude_t: float = 1e-6,
    phase_rad: float = 0.0,
) -> Tuple[AmbientConfig, torch.Tensor, float]:
    """
    Ambient +X: radial spectrum uses a real combination of Y_1,-1 and Y_1,+1:
      B ∝ Y_1,-1 - Y_1,+1
    """
    ambient_cfg, _, period_sec = _base_config(period_hours, amplitude_t, phase_rad)
    lmax = grid_cfg.lmax
    B_radial_spec = torch.zeros((lmax + 1, 2 * lmax + 1), device=grid_cfg.device, dtype=torch.complex128)
    # Combination gives a real +X-directed field on the sphere
    B_radial_spec[1, lmax - 1] = amplitude_t / 2.0
    B_radial_spec[1, lmax + 1] = -amplitude_t / 2.0
    return ambient_cfg, B_radial_spec, period_sec


def build_ambient_driver_y(
    grid_cfg,
    *,
    period_hours: float = 9.925,
    amplitude_t: float = 1e-6,
    phase_rad: float = 0.0,
) -> Tuple[AmbientConfig, torch.Tensor, float]:
    """
    Ambient +Y: radial spectrum uses an imaginary combination:
      B ∝ (Y_1,-1 + Y_1,+1) / (2j)
    """
    ambient_cfg, _, period_sec = _base_config(period_hours, amplitude_t, phase_rad)
    lmax = grid_cfg.lmax
    B_radial_spec = torch.zeros((lmax + 1, 2 * lmax + 1), device=grid_cfg.device, dtype=torch.complex128)
    # Imaginary coefficients yield a real +Y-directed field
    coeff = amplitude_t / (2.0j)
    B_radial_spec[1, lmax - 1] = coeff
    B_radial_spec[1, lmax + 1] = coeff
    return ambient_cfg, B_radial_spec, period_sec
