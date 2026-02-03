import torch

from europa_model.config import GridConfig, AmbientConfig, SolverConfig, ObservationConfig, VisualizationConfig, ModelConfig
from europa_model.simulation import Simulation


def test_simulation_runs():
    grid_cfg = GridConfig(nside=1, lmax=2, radius_m=1.0, device="cpu")
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=1e-6, phase_rad=0.0, spatial_mode="uniform")
    solver_cfg = SolverConfig(dt=0.1, method="implicit", max_steps=10)
    obs_cfg = ObservationConfig()
    viz_cfg = VisualizationConfig(enable_plots=False, enable_animation=False)
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg, solver=solver_cfg, observation=obs_cfg, visualization=viz_cfg)
    sim = Simulation(model)
    K_tor, K_pol, _ = sim.run_time_domain(timesteps=3)
    assert K_tor.shape[-2:] == (grid_cfg.lmax + 1, 2 * grid_cfg.lmax + 1)
    assert K_pol.shape == K_tor.shape
