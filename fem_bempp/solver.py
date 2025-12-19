"""
Placeholder Bempp-based solver.

This scaffolds an alternate path using a boundary-element method (Bempp) to solve
for self-consistent surface currents and emitted fields on the same grid used by
the harmonic pipeline. The actual Bempp implementation is left to be filled in.
"""
from __future__ import annotations

from typing import Any, Dict

import torch


def bempp_available() -> bool:
    try:
        import bempp.api  # type: ignore
    except ImportError:
        return False
    return True


def solve_self_consistent_bempp(
    positions: torch.Tensor,
    normals: torch.Tensor,
    admittance: torch.Tensor,
    ambient_B_tor: torch.Tensor,
    ambient_B_pol: torch.Tensor,
    radius_m: float,
) -> Dict[str, Any]:
    """
    Solve for self-consistent surface currents using Bempp (placeholder).

    Args:
        positions: [N,3] grid points on the sphere
        normals: [N,3] normals at each point
        admittance: spectral or spatial admittance data (format TBD)
        ambient_B_tor / ambient_B_pol: ambient tangential B components (VSH coeffs) or spatial samples
        radius_m: sphere radius

    Returns:
        dict with keys: K_toroidal, K_poloidal, B_emit (dict of components), timing info
    """
    if not bempp_available():
        raise ImportError("Bempp is not installed; cannot run BEM solver.")

    # NOTE: Actual Bempp formulation is not implemented. This placeholder returns zeros
    # to allow plumbing/validation scaffolding without failing hard.
    N = positions.shape[0]
    result = {
        "K_toroidal": torch.zeros((N,), dtype=torch.complex128),
        "K_poloidal": torch.zeros((N,), dtype=torch.complex128),
        "B_emit": {
            "radial": torch.zeros((N,), dtype=torch.complex128),
            "toroidal": torch.zeros((N,), dtype=torch.complex128),
            "poloidal": torch.zeros((N,), dtype=torch.complex128),
        },
        "notes": "Bempp solver not implemented; returned zeros.",
    }
    return result
