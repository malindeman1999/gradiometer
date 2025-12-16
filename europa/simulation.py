"""
Simulation orchestrator wiring configs, drivers, solvers, diagnostics, and visualization.
"""
from dataclasses import dataclass

import torch

from .config import ModelConfig
from . import grid as grid_module
from . import ambient, solvers, diagnostics, observation, visualize


@dataclass
class Simulation:
    config: ModelConfig
    grid = None

    def build_grid(self):
        self.grid = grid_module.make_grid(self.config.grid)

    def run_time_domain(self, timesteps: int):
        if self.grid is None:
            self.build_grid()
        B_time = ambient.generate_time_series(self.config.ambient, self.config.grid, timesteps)
        B_rad, B_tor, B_pol = ambient.to_spectral_time(B_time, self.grid.positions, self.grid.normals, self.config.grid.lmax, self.grid.areas)
        K_tor, K_pol = solvers.solve_time_domain(self.config, self.grid, B_rad, B_tor, B_pol, timesteps)
        return K_tor, K_pol, B_rad

    def run_frequency_domain(self, freqs: int):
        if self.grid is None:
            self.build_grid()
        B_freq = ambient.generate_frequency_series(self.config.ambient, self.config.grid, freqs)
        B_rad, B_tor, B_pol = ambient.to_spectral_frequency(B_freq, self.grid.positions, self.grid.normals, self.config.grid.lmax, self.grid.areas)
        K_tor, K_pol = solvers.solve_frequency_domain(self.config, self.grid, B_rad, B_tor, B_pol)
        return K_tor, K_pol, B_rad

    def diagnostics_divergence(self, K_tor, K_pol):
        return diagnostics.divergence_spectral(K_tor, K_pol, self.config.grid.radius_m)

    def visualize_heatmap(self, values, title=""):
        visualize.plot_heatmap(values, title=title)
