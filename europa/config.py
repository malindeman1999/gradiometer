from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class GridConfig:
    """Grid and harmonic resolution settings."""
    nside: int  # HEALPix-like resolution parameter
    lmax: int   # maximum spherical harmonic degree (derived from nside)
    radius_m: float = 1.0  # sphere radius (Europa radius in meters for scaling)
    ocean_thickness_m: float = 100_000.0  # shell thickness to convert 3D to 2D conductivity (default 100 km)
    # ocean_thickness_m: float = 10.0  # NOT PHYSICAL, FOR DEBUGGING ONLY
    seawater_conductivity_s_per_m: float = 3.3  # bulk seawater conductivity (S/m)
    device: str = "cuda"  # target device: "cuda" or "cpu"


@dataclass(frozen=True)
class AmbientConfig:
    """Ambient magnetic field driver parameters."""
    omega_jovian: float  # base angular frequency (rad/s) for Jovian rotation
    amplitude_t: float   # base amplitude of ambient field (Tesla)
    phase_rad: float = 0.0
    spatial_mode: str = "uniform"  # "uniform" or "per_node"
    custom_time_series: Optional[Sequence] = None  # optional external driver
    custom_frequency_series: Optional[Sequence] = None  # optional spectral driver


@dataclass(frozen=True)
class SolverConfig:
    """Solver parameters and stability guardrails."""
    dt: float
    method: str = "implicit"  # "implicit" or "explicit"
    max_steps: int = 0  # 0 => infer from input length
    tol: float = 1e-8
    max_iters: int = 200
    regularization: float = 0.0  # damping/regularization if conditioning is poor
    enforce_stable_dt: bool = True
    log_stability: bool = True
    solver_choice_auto: bool = True  # auto-switch direct vs iterative


@dataclass(frozen=True)
class ObservationConfig:
    """Observation shell and sampling settings."""
    altitude_m: float = 50_000.0
    custom_points: Optional[Sequence] = None  # optional custom observation points (xyz)
    include_vectors: bool = True


@dataclass(frozen=True)
class VisualizationConfig:
    """Plotting and animation toggles."""
    enable_plots: bool = True
    enable_animation: bool = True
    fps: int = 24
    overlay_vectors: bool = True
    export_path: Optional[str] = None


@dataclass(frozen=True)
class ModelConfig:
    """Top-level configuration bundle."""
    grid: GridConfig
    ambient: AmbientConfig
    solver: Optional[SolverConfig] = None
    observation: Optional[ObservationConfig] = None
    visualization: Optional[VisualizationConfig] = None
