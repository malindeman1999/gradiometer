"""
Container for phasor-domain simulation inputs/outputs plus helpers to synthesize time samples.
Keeps ambient, current, E-field, and emitted-field harmonics in spherical-harmonic form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch

from europa_model.config import ModelConfig

@dataclass
class PhasorSimulation:
    omega: float
    period_sec: float
    lmax: int
    radius_m: float
    ambient_amplitude_t: float
    ambient_phase_rad: float
    grid_positions: torch.Tensor
    grid_normals: torch.Tensor
    grid_areas: torch.Tensor
    grid_neighbors: Optional[torch.Tensor]
    solver_variant: str
    admittance_uniform: Optional[float] = None
    admittance_spectral: Optional[torch.Tensor] = None
    # Phasor fields (optional; can be filled by solvers)
    B_radial: Optional[torch.Tensor] = None
    E_toroidal: Optional[torch.Tensor] = None
    K_toroidal: Optional[torch.Tensor] = None
    K_poloidal: Optional[torch.Tensor] = None
    B_tor_emit: Optional[torch.Tensor] = None
    B_pol_emit: Optional[torch.Tensor] = None
    B_rad_emit: Optional[torch.Tensor] = None

    @classmethod
    def from_model_and_grid(
        cls,
        model: ModelConfig,
        grid,
        *,
        solver_variant: str,
        admittance_uniform: Optional[float] = None,
        admittance_spectral: Optional[torch.Tensor] = None,
        B_radial: Optional[torch.Tensor] = None,
        period_sec: Optional[float] = None,
    ) -> "PhasorSimulation":
        """Convenience constructor pulling shared fields directly from model/grid."""
        return cls(
            omega=float(model.ambient.omega_jovian),
            period_sec=float(period_sec) if period_sec is not None else 0.0,
            lmax=int(model.grid.lmax),
            radius_m=float(model.grid.radius_m),
            ambient_amplitude_t=float(model.ambient.amplitude_t),
            ambient_phase_rad=float(model.ambient.phase_rad),
        grid_positions=grid.positions,
        grid_normals=grid.normals,
        grid_areas=grid.areas,
        grid_neighbors=getattr(grid, "neighbors", None),
        solver_variant=solver_variant,
            admittance_uniform=admittance_uniform,
            admittance_spectral=admittance_spectral,
            B_radial=B_radial,
        )

    def to_serializable(self) -> Dict[str, Any]:
        """Return a dict form that can be re-hydrated."""
        return {
            "omega": self.omega,
            "period_sec": self.period_sec,
            "lmax": self.lmax,
            "radius_m": self.radius_m,
            "ambient_amplitude_t": self.ambient_amplitude_t,
            "ambient_phase_rad": self.ambient_phase_rad,
            "grid_positions": self.grid_positions,
            "grid_normals": self.grid_normals,
            "grid_areas": self.grid_areas,
            "grid_neighbors": self.grid_neighbors,
            "solver_variant": self.solver_variant,
            "admittance_uniform": self.admittance_uniform,
            "admittance_spectral": self.admittance_spectral,
            "B_radial": self.B_radial,
            "E_toroidal": self.E_toroidal,
            "K_toroidal": self.K_toroidal,
            "K_poloidal": self.K_poloidal,
            "B_tor_emit": self.B_tor_emit,
            "B_pol_emit": self.B_pol_emit,
            "B_rad_emit": self.B_rad_emit,
        }

    @classmethod
    def from_saved(cls, obj: Any) -> "PhasorSimulation":
        """Build from a saved torch.load result (dict or instance)."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            if "phasor_sim" in obj:
                ps = obj["phasor_sim"]
                if isinstance(ps, cls):
                    return ps
                elif isinstance(ps, dict):
                    return cls.from_serializable(ps)
            if "phasors" in obj:
                cfg = obj.get("config", {})
                grid = obj.get("grid", {})
                ph = obj["phasors"]
                return cls(
                    omega=float(cfg["ambient_omega"]),
                    period_sec=float(cfg.get("period_sec", 0.0)),
                    lmax=int(cfg["lmax"]),
                    radius_m=float(cfg["radius_m"]),
                    ambient_amplitude_t=float(cfg.get("ambient_amplitude_t", 0.0)),
                    ambient_phase_rad=float(cfg.get("ambient_phase_rad", 0.0)),
                    grid_positions=grid["positions"],
                    grid_normals=grid["normals"],
                    grid_areas=grid["areas"],
                    grid_neighbors=grid.get("neighbors", None),
                    solver_variant=cfg.get("solver_variant", "unknown"),
                    admittance_uniform=cfg.get("admittance_uniform", None),
                    admittance_spectral=cfg.get("admittance_spectral", None),
                    B_radial=ph.get("B_radial"),
                    E_toroidal=ph.get("E_toroidal"),
                    K_toroidal=ph.get("K_toroidal"),
                    K_poloidal=ph.get("K_poloidal"),
                    B_tor_emit=ph.get("B_tor_emit"),
                    B_pol_emit=ph.get("B_pol_emit"),
                    B_rad_emit=ph.get("B_rad_emit"),
                )
        raise ValueError("Cannot rebuild PhasorSimulation from provided object.")

    @classmethod
    def from_serializable(cls, data: Dict[str, Any]) -> "PhasorSimulation":
        return cls(
            omega=float(data["omega"]),
            period_sec=float(data["period_sec"]),
            lmax=int(data["lmax"]),
            radius_m=float(data["radius_m"]),
            ambient_amplitude_t=float(data["ambient_amplitude_t"]),
            ambient_phase_rad=float(data["ambient_phase_rad"]),
            grid_positions=data["grid_positions"],
            grid_normals=data["grid_normals"],
            grid_areas=data["grid_areas"],
            grid_neighbors=data.get("grid_neighbors", None),
            solver_variant=data.get("solver_variant", "unknown"),
            admittance_uniform=data.get("admittance_uniform", None),
            admittance_spectral=data.get("admittance_spectral", None),
            B_radial=data.get("B_radial"),
            E_toroidal=data.get("E_toroidal"),
            K_toroidal=data.get("K_toroidal"),
            K_poloidal=data.get("K_poloidal"),
            B_tor_emit=data.get("B_tor_emit"),
            B_pol_emit=data.get("B_pol_emit"),
            B_rad_emit=data.get("B_rad_emit"),
        )
