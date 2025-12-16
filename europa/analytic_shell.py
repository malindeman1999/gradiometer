"""
Analytic thin-shell induction estimate (dipole, ℓ=1, lumped RL approximation).

What it does:
- Takes bulk conductivity (σ, S/m), shell thickness (t, m), ocean radius (R, m),
  and a sinusoidal uniform applied field B0(ω).
- Builds a simple RL time constant: τ ≈ μ0 * σ * t * R (thin, conducting shell).
- Uses a first-order transfer function H(ω) = 1 / (1 + i ω τ) for the dipole term.
- Computes the induced dipole moment m = -(4π R^3 / 3μ0) * H(ω) * B0.
- Evaluates the complex induced B field at arbitrary observation points using the
  static dipole kernel scaled by H(ω).
- Returns relative amplitude |H| and phase lag arg(H) along with the complex field.

Assumptions / scope:
- Dipole-only (ℓ=1) response; higher-ℓ effects are ignored.
- Thin-shell, low-to-mid frequency heuristic; at higher frequency/skin-depth
  regimes or for thicker shells, the exact spherical Bessel solution is needed.
- Shell resistance and inductance are collapsed into a single τ; no explicit L/R.
"""

from typing import Tuple

import torch


MU0 = 4e-7 * torch.pi  # permeability of free space


def shell_time_constant(sigma_s_per_m: float, thickness_m: float, radius_m: float) -> float:
    """Return the lumped RL time constant tau (seconds) for a thin conductive shell."""
    return float(MU0 * sigma_s_per_m * thickness_m * radius_m)


def transfer_function(omega: float, tau: float) -> complex:
    """Complex transfer function H(ω) = 1 / (1 + i ω tau)."""
    return 1.0 / (1.0 + 1j * omega * tau)


def induced_dipole_moment(B0_vec: torch.Tensor, radius_m: float, H: complex) -> torch.Tensor:
    """
    Compute complex dipole moment vector induced by a uniform applied field.
    Args:
        B0_vec: real-valued ambient field amplitude (Tesla), shape [3].
        radius_m: ocean radius (m).
        H: complex transfer function scalar.
    Returns:
        Complex dipole moment vector (A·m^2), shape [3], dtype complex128.
    """
    scale = - (4.0 * torch.pi * (radius_m ** 3)) / (3.0 * MU0)
    return torch.tensor(scale, dtype=torch.complex128) * B0_vec.to(torch.complex128) * torch.tensor(H, dtype=torch.complex128)


def dipole_field_at_points(m_vec: torch.Tensor, obs_xyz: torch.Tensor) -> torch.Tensor:
    """
    Evaluate complex magnetic field of dipole m_vec at observation points.
    Args:
        m_vec: complex dipole moment vector, shape [3] (complex128).
        obs_xyz: real observation coordinates [N,3] (meters) relative to center.
    Returns:
        B_obs: complex field [N,3] (Tesla), complex128.
    """
    obs = obs_xyz.to(dtype=torch.float64)
    r = torch.norm(obs, dim=-1, keepdim=True)  # [N,1]
    rhat = obs / torch.clamp(r, min=1e-12)
    m = m_vec
    rhat_c = rhat.to(torch.complex128)
    mdotr = torch.sum(m[None, :] * rhat_c, dim=-1)
    term1 = 3.0 * mdotr[..., None] * rhat_c
    term2 = m[None, :]
    factor = MU0 / (4.0 * torch.pi) / torch.clamp(r ** 3, min=1e-12)
    return (term1 - term2) * factor


def estimate_response(
    sigma_s_per_m: float,
    thickness_m: float,
    radius_m: float,
    freq_hz: float,
    B0_vec: torch.Tensor,
    obs_xyz: torch.Tensor,
) -> Tuple[float, float, torch.Tensor]:
    """
    Estimate relative amplitude, phase lag, and complex response field at observation points.
    Args:
        sigma_s_per_m: bulk conductivity (S/m)
        thickness_m: shell thickness (m)
        radius_m: ocean radius (m)
        freq_hz: driver frequency (Hz)
        B0_vec: real ambient field vector amplitude (Tesla), shape [3]
        obs_xyz: observation points (m) shape [N,3]
    Returns:
        (relative_amp, phase_lag_rad, B_resp_complex[N,3])
        where relative_amp = |H(ω)|, phase_lag_rad = arg(H(ω)) (negative => lag)
    """
    omega = 2.0 * torch.pi * float(freq_hz)
    tau = shell_time_constant(sigma_s_per_m, thickness_m, radius_m)
    H = transfer_function(omega, tau)
    rel_amp = abs(H)
    phase = torch.angle(torch.tensor(H, dtype=torch.complex128)).item()
    m_vec = induced_dipole_moment(B0_vec.to(torch.float64), radius_m, H)
    B_resp = dipole_field_at_points(m_vec, obs_xyz)
    return rel_amp, phase, B_resp
