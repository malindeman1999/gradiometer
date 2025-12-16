"""
Utilities for packaging and saving demo results.
"""
from typing import Dict, Any

import torch

from phasor_data import PhasorSimulation


def build_results_dict(
    *,
    phasor_sim: PhasorSimulation,
    grid,
    B_rad_ph: torch.Tensor,
    time_domain: Dict[str, Any],
    north_idx: int,
    rel_amp: float,
    phase_deg: float,
) -> Dict[str, Any]:
    """Assemble a results dictionary for saving/visualization."""
    return {
        "config": {
            "radius_m": phasor_sim.radius_m,
            "lmax": phasor_sim.lmax,
            "dt": float(time_domain["dt"]),
            "timesteps": int(time_domain["timesteps"]),
            "period_sec": float(time_domain["period_sec"]),
            "ambient_amplitude_t": phasor_sim.ambient_amplitude_t,
            "ambient_omega": phasor_sim.omega,
            "ambient_phase_rad": phasor_sim.ambient_phase_rad,
            "solver_variant": phasor_sim.solver_variant,
            "admittance_uniform": phasor_sim.admittance_uniform,
        },
        "times": time_domain["times"].cpu(),
        "ambient": {
            "B_time": time_domain["B_time"].cpu(),
            "B_radial_phasor": B_rad_ph.cpu(),
        },
        "grid": {
            "positions": grid.positions.cpu(),
            "areas": grid.areas.cpu(),
            "normals": grid.normals.cpu(),
        },
        "currents": {
            "toroidal": time_domain["K_tor_stack"].cpu(),
            "poloidal": time_domain["K_pol_stack"].cpu(),
        },
        "phasors": {
            "B_radial": B_rad_ph.cpu(),
            "E_toroidal": phasor_sim.E_toroidal.cpu(),
            "K_toroidal": phasor_sim.K_toroidal.cpu(),
            "K_poloidal": phasor_sim.K_poloidal.cpu(),
            "B_rad_emit": phasor_sim.B_rad_emit.cpu(),
            "B_pol_emit": phasor_sim.B_pol_emit.cpu(),
            "B_tor_emit": phasor_sim.B_tor_emit.cpu(),
        },
        "phasor_sim": phasor_sim.to_serializable(),
        "north_idx": int(north_idx),
        "analytic": {"relative_amplitude": float(rel_amp), "phase_deg": float(phase_deg)},
    }


def save_demo_results(
    *,
    output_path: str,
    phasor_sim: PhasorSimulation,
    grid,
    B_rad_ph: torch.Tensor,
    time_domain: Dict[str, Any],
    north_idx: int,
    rel_amp: float,
    phase_deg: float,
) -> Dict[str, Any]:
    """Package results and save to disk."""
    results = build_results_dict(
        phasor_sim=phasor_sim,
        grid=grid,
        B_rad_ph=B_rad_ph,
        time_domain=time_domain,
        north_idx=north_idx,
        rel_amp=rel_amp,
        phase_deg=phase_deg,
    )
    torch.save(results, output_path)
    print(f"Saved current time series to {output_path} "
          f"(shape {results['currents']['toroidal'].shape}, device=cpu).")
    return results
