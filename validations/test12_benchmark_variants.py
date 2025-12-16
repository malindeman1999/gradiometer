"""
Benchmark and validate spectral self-consistent variants (lmax=3).

Runs baseline (solvers.py), baseline copy, vectorized NumPy, precomputed Gaunt,
torch on-the-fly, and torch with precomputed Gaunt on random inputs, and compares
outputs plus timings.
"""
import time
import os
import sys
from typing import Dict, List, Tuple

import torch

# Allow running from validations/ by adding repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa.config import AmbientConfig, GridConfig, ModelConfig
from europa.simulation import Simulation
from phasor_data import PhasorSimulation
from europa import solvers
from europa.solver_variants.solver_variant_baseline import solve_spectral_self_consistent_sim_baseline
from europa.solver_variants.solver_variant_vectorized import solve_spectral_self_consistent_sim_vectorized
from europa.solver_variants.solver_variant_precomputed import solve_spectral_self_consistent_sim_precomputed
from europa.solver_variants.solver_variant_torch import solve_spectral_self_consistent_sim_torch
from europa.solver_variants.solver_variant_torch_precomputed import solve_spectral_self_consistent_sim_torch_precomputed


def _clone_ps(base: PhasorSimulation) -> PhasorSimulation:
    return PhasorSimulation.from_serializable(base.to_serializable())


def _to_device(ps: PhasorSimulation, device: torch.device) -> PhasorSimulation:
    """Move tensor fields of PhasorSimulation to the given device."""
    for attr in [
        "grid_positions",
        "grid_normals",
        "grid_areas",
        "B_radial",
        "E_toroidal",
        "K_toroidal",
        "K_poloidal",
        "B_tor_emit",
        "B_pol_emit",
        "B_rad_emit",
        "admittance_spectral",
    ]:
        val = getattr(ps, attr, None)
        if torch.is_tensor(val):
            setattr(ps, attr, val.to(device))
    return ps


def _max_abs_and_rel(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a.to(torch.complex128) - b.to(torch.complex128)).abs()
    max_abs = float(torch.max(diff).cpu())
    max_ref = float(
        torch.max(torch.stack([a.abs(), b.abs()])).cpu()
    ) if a.numel() > 0 else 1.0
    max_ref = max(max_ref, 1e-30)
    max_rel = max_abs / max_ref
    return max_abs, max_rel


def _compare_fields(ref: PhasorSimulation, other: PhasorSimulation, abs_tol=1e-8, rel_tol=1e-3) -> List[str]:
    failures: List[str] = []
    fields = [
        "E_toroidal",
        "K_toroidal",
        "K_poloidal",
        "B_tor_emit",
        "B_pol_emit",
        "B_rad_emit",
    ]
    for fname in fields:
        ra = getattr(ref, fname)
        rb = getattr(other, fname)
        if ra is None or rb is None:
            failures.append(f"{fname} missing")
            continue
        if ra.device != rb.device:
            failures.append(f"{fname} device mismatch: ref={ra.device}, other={rb.device}")
            continue
        max_abs, max_rel = _max_abs_and_rel(ra, rb)
        if max_abs > abs_tol and max_rel > rel_tol:
            failures.append(f"{fname}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    return failures


def _build_base(seed: int, lmax: int = 3) -> PhasorSimulation:
    torch.manual_seed(seed)
    grid_cfg = GridConfig(nside=8, lmax=lmax, radius_m=1.56e6, device="cpu", ocean_thickness_m=10.0)
    ambient_cfg = AmbientConfig(omega_jovian=1.0, amplitude_t=0.0, phase_rad=0.0, spatial_mode="custom")
    model = ModelConfig(grid=grid_cfg, ambient=ambient_cfg)
    sim = Simulation(model)
    sim.build_grid()
    Y_s = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    Y_s += 1j * torch.randn_like(Y_s)
    B_rad = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128) * 1e-6
    B_rad += 1j * torch.randn_like(B_rad) * 1e-6
    base = PhasorSimulation.from_model_and_grid(
        model=model,
        grid=sim.grid,
        solver_variant="",
        admittance_uniform=None,
        admittance_spectral=Y_s,
        B_radial=B_rad,
        period_sec=0.0,
    )
    return base


def _time_solver(fn, sim: PhasorSimulation) -> Tuple[PhasorSimulation, float]:
    start = time.perf_counter()
    out = fn(sim)
    elapsed = time.perf_counter() - start
    return out, elapsed


def run_once(seed: int = 0) -> None:
    base = _build_base(seed)

    # Baseline (original)
    ref = _clone_ps(base)
    ref, t_ref = _time_solver(solvers.solve_spectral_self_consistent_sim, ref)

    variants = {
        "baseline_copy": solve_spectral_self_consistent_sim_baseline,
        "vectorized_np": solve_spectral_self_consistent_sim_vectorized,
        "precomputed_gaunt": solve_spectral_self_consistent_sim_precomputed,
        "torch_on_the_fly": solve_spectral_self_consistent_sim_torch,
        "torch_precomputed": solve_spectral_self_consistent_sim_torch_precomputed,
    }

    results: Dict[str, Tuple[PhasorSimulation, float, List[str]]] = {}
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print()
        print(f"CUDA available: using device {torch.cuda.get_device_name(0)}")

    for name, fn in variants.items():
        sim = _clone_ps(base)
        # Torch variants: optionally move to GPU
        if name.startswith("torch_") and cuda_available:
            sim = _to_device(sim, torch.device("cuda"))
            ref_dev = _to_device(_clone_ps(ref), torch.device("cuda"))
        else:
            ref_dev = ref
        out, t = _time_solver(fn, sim)
        fails = _compare_fields(ref_dev, out)
        # Verify device for torch variants when CUDA is available
        if name.startswith("torch_") and cuda_available and out.K_toroidal is not None:
            if out.K_toroidal.device.type != "cuda":
                fails.append("K_toroidal not on CUDA device")
        results[name] = (out, t, fails)

    print(f"Seed {seed}: baseline time {t_ref:.3f}s")
    for name, (_, t, fails) in results.items():
        status = "OK" if not fails else "FAIL"
        print(f"  {name:18s} {status:4s} time={t:.3f}s {'; '.join(fails) if fails else ''}")


def main() -> None:
    seeds = [0, 1, 2]
    for seed in seeds:
        run_once(seed)


if __name__ == "__main__":
    main()
