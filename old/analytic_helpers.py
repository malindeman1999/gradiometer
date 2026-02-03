"""
Analytic utilities used by the demo.
"""
import torch
from europa_model.old.analytic_shell import estimate_response as analytic_estimate


def thin_shell_estimate(grid_cfg, ambient_cfg, period_sec: float, positions=None):
    """
    Compute the thin-shell analytic response for a dipole drive.
    Returns (relative amplitude, phase_deg).
    """
    rel_amp, phase_rad, _ = analytic_estimate(
        sigma_s_per_m=grid_cfg.seawater_conductivity_s_per_m,
        thickness_m=grid_cfg.ocean_thickness_m,
        radius_m=grid_cfg.radius_m,
        freq_hz=1.0 / period_sec,
        B0_vec=torch.tensor([0.0, 0.0, ambient_cfg.amplitude_t], dtype=torch.float64),
        obs_xyz=positions,
    )
    phase_deg = float(phase_rad * 180.0 / torch.pi)
    return float(rel_amp), phase_deg
