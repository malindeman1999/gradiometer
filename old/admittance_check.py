"""
Check proportionality between ambient normal dB/dt harmonics and induced surface-current harmonics.
Works entirely in phasor space: derive dB/dt phasor via i*omega*B_phasor, then compare to current phasors.
"""
import argparse
from typing import Dict, Tuple

import torch
from workflow.data_objects.phasor_data import PhasorSimulation


def _harmonic_indices(lmax: int):
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            yield l, m


def check_admittance(data_path: str = "demo_currents.pt", eps: float = 1e-10) -> Dict[str, float]:
    results = torch.load(data_path, map_location="cpu", weights_only=False)
    sim_ph = PhasorSimulation.from_saved(results)
    omega = sim_ph.omega
    lmax = sim_ph.lmax
    phasor_Brad = sim_ph.B_radial.to(torch.complex128)
    phasor_Ktor = sim_ph.K_toroidal.to(torch.complex128) if sim_ph.K_toroidal is not None else torch.zeros_like(phasor_Brad)
    phasor_Kpol = sim_ph.K_poloidal.to(torch.complex128) if sim_ph.K_poloidal is not None else torch.zeros_like(phasor_Brad)
    phasor_Brad_emit = (sim_ph.B_rad_emit if sim_ph.B_rad_emit is not None else torch.zeros_like(phasor_Brad)).to(torch.complex128)
    phasor_Bpol_emit = (sim_ph.B_pol_emit if sim_ph.B_pol_emit is not None else torch.zeros_like(phasor_Brad)).to(torch.complex128)

    phasor_dBdt_rad = 1j * omega * phasor_Brad

    ratios = []
    for l, m in _harmonic_indices(lmax):
        idx = m + lmax
        B_val = phasor_dBdt_rad[l, idx]
        Kt = phasor_Ktor[l, idx]
        Kp = phasor_Kpol[l, idx]
        Brad_out = phasor_Brad_emit[l, idx]
        Bpol_out = phasor_Bpol_emit[l, idx]

        if torch.abs(B_val) < eps and torch.abs(Kt) < eps and torch.abs(Kp) < eps and torch.abs(Brad_out) < eps and torch.abs(Bpol_out) < eps:
            continue  # all zero -> skip

        if torch.abs(B_val) >= eps:
            ratio = Kt / B_val
            ratios.append(ratio)

    if not ratios:
        return {}
    ratio_tensor = torch.stack([torch.as_tensor(r) for r in ratios])
    mags = torch.abs(ratio_tensor)
    phases_deg = torch.angle(ratio_tensor) * 180.0 / torch.pi
    stats = {
        "ratio_mag_mean": float(torch.mean(mags)),
        "ratio_mag_std": float(torch.std(mags)) if mags.numel() > 1 else 0.0,
        "ratio_mag_min": float(torch.min(mags)),
        "ratio_mag_max": float(torch.max(mags)),
        "ratio_phase_mean_deg": float(torch.mean(phases_deg)),
        "ratio_phase_std_deg": float(torch.std(phases_deg)) if phases_deg.numel() > 1 else 0.0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Check proportionality between ambient dB/dt and induced currents.")
    parser.add_argument("--input", type=str, default="demo_currents.pt", help="Path to saved demo file.")
    parser.add_argument("--eps", type=float, default=1e-10, help="Threshold for considering a harmonic non-zero.")
    args = parser.parse_args()
    check_admittance(args.input, eps=args.eps)


if __name__ == "__main__":
    main()
