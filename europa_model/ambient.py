"""
Ambient field generation (time and frequency domains) with optional spectral ingestion.
"""
from typing import Optional, Tuple

import torch

from .config import AmbientConfig, GridConfig
from . import transforms


def generate_time_series(config: AmbientConfig, grid: GridConfig, timesteps: int) -> torch.Tensor:
    """
    Generate ambient B(t) on the grid.
    Returns:
        B: [T, N_nodes, 3] (spatial) or [T, 3] if uniform.
    """
    device = grid.device
    if config.custom_time_series is not None:
        series = torch.as_tensor(config.custom_time_series, device=device)
        return series
    t = torch.arange(timesteps, device=device, dtype=torch.float64)
    phase = config.phase_rad
    omega = config.omega_jovian
    amp = config.amplitude_t
    signal = amp * torch.sin(omega * t[:, None] + phase)
    if config.spatial_mode == "uniform":
        # Default along z-axis
        vec = torch.zeros((timesteps, 3), device=device, dtype=torch.float64)
        vec[:, 2] = signal[:, 0] if signal.ndim > 1 else signal
        return vec
    else:
        # Per-node: allow a simple dipole-like spatial variation aligned with z
        nside = grid.nside
        npix = 12 * nside * nside
        vec = torch.zeros((timesteps, npix, 3), device=device, dtype=torch.float64)
        idx = torch.arange(npix, device=device, dtype=torch.float64)
        colat_scale = torch.cos(2 * torch.pi * idx / max(npix - 1, 1))  # [npix]
        vec[..., 2] = signal.view(timesteps, 1) * colat_scale.view(1, npix)
        return vec


def generate_frequency_series(config: AmbientConfig, grid: GridConfig, freqs: int) -> torch.Tensor:
    """
    Generate ambient B(omega) on the grid (complex).
    Returns:
        B: [F, N_nodes, 3] (spatial) or [F, 3] if uniform.
    """
    device = grid.device
    if config.custom_frequency_series is not None:
        series = torch.as_tensor(config.custom_frequency_series, device=device)
        return series
    # Single line at omega_jovian: amplitude at that bin, zero elsewhere.
    omega0 = config.omega_jovian
    amps = torch.zeros((freqs,), device=device, dtype=torch.complex128)
    amps[0] = torch.tensor(config.amplitude_t, device=device, dtype=torch.complex128)  # placeholder bin
    if config.spatial_mode == "uniform":
        vec = torch.zeros((freqs, 3), device=device, dtype=torch.complex128)
        vec[:, 2] = amps
        return vec
    else:
        nside = grid.nside
        npix = 12 * nside * nside
        vec = torch.zeros((freqs, npix, 3), device=device, dtype=torch.complex128)
        idx = torch.arange(npix, device=device, dtype=torch.float64)
        colat_scale = torch.cos(2 * torch.pi * idx / max(npix - 1, 1))
        vec[..., 2] = amps[:, None] * colat_scale[None, :]
        return vec


def to_spectral_time(B_time: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, lmax: int, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform grid B(t) to spectral coefficients per timestep (radial SH + tangential VSH).
    For the ambient driver we retain only the radial part. Tangential coefficients are filled with NaN
    sentinels to detect any unintended coupling paths; downstream solvers must ignore/guard them.
    """
    # Ensure shape [T, N, 3]
    if B_time.ndim == 2:  # [T,3] uniform
        n_nodes = positions.shape[0]
        B_time = B_time[:, None, :].repeat(1, n_nodes, 1)
    rad_list, tor_list, pol_list = [], [], []
    # Fill ambient tangential drivers with NaN sentinels so any accidental use is immediately detectable.
    nan_dtype = torch.complex128 if B_time.is_complex() else torch.float64
    nan_fill = torch.full((lmax + 1, 2 * lmax + 1), float("nan"), device=positions.device, dtype=nan_dtype)
    for t_slice in B_time:
        # radial projection
        radial = (t_slice * normals).sum(dim=-1)
        rad_coeff = transforms.sh_forward(radial, positions, lmax, weights)
        rad_list.append(rad_coeff)
        tor_list.append(nan_fill)
        pol_list.append(nan_fill)
    return torch.stack(rad_list, dim=0), torch.stack(tor_list, dim=0), torch.stack(pol_list, dim=0)


def to_spectral_frequency(B_freq: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, lmax: int, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform grid B(omega) to spectral coefficients per frequency; tangential drivers are zeroed."""
    if B_freq.ndim == 2:  # [F,3] uniform
        n_nodes = positions.shape[0]
        B_freq = B_freq[:, None, :].repeat(1, n_nodes, 1)
    rad_list, tor_list, pol_list = [], [], []
    # Fill ambient tangential drivers with NaN sentinels so any accidental use is immediately detectable.
    nan_dtype = torch.complex128 if B_freq.is_complex() else torch.float64
    nan_fill = torch.full((lmax + 1, 2 * lmax + 1), float("nan"), device=positions.device, dtype=nan_dtype)
    for f_slice in B_freq:
        radial = (f_slice * normals).sum(dim=-1)
        rad_coeff = transforms.sh_forward(radial, positions, lmax, weights)
        rad_list.append(rad_coeff)
        tor_list.append(nan_fill)
        pol_list.append(nan_fill)
    return torch.stack(rad_list, dim=0), torch.stack(tor_list, dim=0), torch.stack(pol_list, dim=0)
