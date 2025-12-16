"""
Demo pipeline: build a spherical grid, prescribe an ambient normal field phasor (pure Y_1,1),
solve first-order (uniform admittance) for E/K and emitted B, and save phasors plus
synthesized time samples for visualization.
"""
import argparse
from pathlib import Path

import torch

from europa.config import GridConfig, ModelConfig
from europa.simulation import Simulation
from europa import solvers
from admittance_check import check_admittance
from phasor_data import PhasorSimulation
from demo_io import save_demo_results
from render_demo_overview import render_demo_overview
from analytic_helpers import thin_shell_estimate
from ambient_driver import build_ambient_driver_z as build_ambient_driver


def simulate_and_save(output_path: str = "demo_currents.pt"):
    """
    Phasor-only pipeline:
      - Build grid and model
      - Prescribe ambient normal field phasor (pure Y_1,1)
      - Solve first-order, uniform admittance: E_tor, K_tor, emitted B
      - Synthesize time samples for visualization (no new physics)
      - Save phasors, time series, and a harmonic bar plot
    """
    # Output locations
    data_dir = Path("data")
    fig_dir = Path("figures")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Europa mean radius ~1560 km
    grid_cfg = GridConfig(nside=4, lmax=6, radius_m=1.56e6, device="cpu")
    
    # Build ambient driver (uniform +Z) and spectrum
    ambient_cfg, B_radial_spec, period_sec = build_ambient_driver(
        grid_cfg,
    )
    omega = ambient_cfg.omega_jovian
    timesteps = 120  # only used for synthesizing display samples
    dt = period_sec / timesteps
    model = ModelConfig(
        grid=grid_cfg,
        ambient=ambient_cfg,
    )

    sim = Simulation(model)
    sim.build_grid()
    sigma_2d = sim.grid.surface_conductivity_s
    
    # Uniform surface admittance expressed spectrally: only Y_0,0 is non-zero
    Y_s_spectral = torch.zeros((grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1), dtype=torch.float64)
    Y_s_spectral[0, grid_cfg.lmax] = float(sigma_2d * (4 * torch.pi) ** 0.5)
    print(f"Using surface conductivity (sigma_2d) = {sigma_2d:.2e} S")
    
    # Analytic thin-shell estimate for comparison (dipole, thin-shell RL)
    rel_amp, phase_deg = thin_shell_estimate(sim.config.grid, ambient_cfg, period_sec, positions=sim.grid.positions)
    print(f"Analytic thin-shell estimate (dipole): relative amplitude={rel_amp:.3f}, phase={phase_deg/180*torch.pi:.3f} rad ({phase_deg:.1f} deg)")
    
    # Phasor-domain solve (single frequency): first-order, uniform admittance
    north_idx = torch.argmax(sim.grid.positions[:, 2])
    phasor_sim = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="uniform_first_order",
        admittance_uniform=float(sim.grid.surface_conductivity_s),
        admittance_spectral=Y_s_spectral,
        B_radial=B_radial_spec.cpu(),
        period_sec=period_sec,
    )
    # Solvers mutate the PhasorSimulation in-place; no reassignment needed
    solvers.solve_uniform_first_order_sim(phasor_sim)

    # Synthesize time-domain samples from phasors (for visualization only)
    times = torch.arange(timesteps, device=grid_cfg.device, dtype=torch.float64) * dt
    exp_t = torch.exp(1j * omega * times)  # [T]
    K_tor_stack = torch.stack([(phasor_sim.K_toroidal * exp) for exp in exp_t], dim=0).real
    K_pol_stack = torch.stack([(phasor_sim.K_poloidal * exp) for exp in exp_t], dim=0).real
    B_time = torch.zeros((timesteps, 3), device=grid_cfg.device, dtype=torch.float64)
    B_time[:, 2] = ambient_cfg.amplitude_t * torch.sin(ambient_cfg.omega_jovian * times + ambient_cfg.phase_rad)
    time_domain = {
        "times": times,
        "K_tor_stack": K_tor_stack,
        "K_pol_stack": K_pol_stack,
        "B_time": B_time,
        "dt": dt,
        "period_sec": period_sec,
        "timesteps": timesteps,
    }

    # Package and save results for visualization
    # Package and save via helper
    out_path = Path(output_path)
    data_file = str(data_dir / out_path.name)
    save_demo_results(
        output_path=data_file,
        phasor_sim=phasor_sim,
        grid=sim.grid,
        B_rad_ph=B_radial_spec,
        time_domain=time_domain,
        north_idx=int(north_idx),
        rel_amp=float(rel_amp),
        phase_deg=float(phase_deg),
    )
    # Render stacked sphere maps for key phasors
    render_demo_overview(
        data_path=data_file,
        save_path=str(fig_dir / f"{out_path.stem}_overview.png"),
        show=False,
    )
    return data_file


def main():
    parser = argparse.ArgumentParser(description="Run demo simulation and store current spectra for visualization.")
    parser.add_argument("--output", type=str, default="demo_currents.pt", help="Filename (saved under data/).")
    parser.add_argument("--skip-check", action="store_true", help="Skip admittance ratio check.")
    args = parser.parse_args()
    data_file = simulate_and_save(args.output)
    if not args.skip_check:
        try:
            check_admittance(data_file)
        except Exception as exc:
            print(f"Admittance check failed: {exc}")


if __name__ == "__main__":
    main()
