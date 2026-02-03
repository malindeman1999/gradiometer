"""
Symmetry-aware Gaunt tensor precompute.

Exploits several Gaunt symmetries to reduce the number of explicit
_gaunt_coeff evaluations:
- Magnetic selection: m = M - m0 (so only valid pairs are visited)
- Parity: L + l0 + l must be even (otherwise the coefficient vanishes)
- Magnetic flip: G(L,-M,l0,-m0,l,-m) = G(L,M,l0,m0,l,m) when the selection
  rule holds (sign is +1 because -M + m0 + m = 0 -> 2M is even)

The tensor is built for M >= 0 and mirrored to negative M using the flip
symmetry. Cached on disk keyed by lmax.
"""
from pathlib import Path
from typing import Union

import torch

from europa_model.solvers import _gaunt_coeff


def _cache_path(lmax: int, cache_dir: Union[str, Path] = "data/gaunt_cache") -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gaunt_lmax{lmax}.pt"


def _build_gaunt_tensor_torch_symmetric(lmax: int, device: str = "cpu") -> torch.Tensor:
    """
    Build Gaunt tensor G[L,M,l0,m0,l,m] as float64 using Gaunt symmetries
    to avoid redundant evaluations.
    """
    G = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
        device=device,
    )
    for L in range(lmax + 1):
        for M in range(0, L + 1):  # compute non-negative M and mirror
            M_idx = L + M
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        # Selection rule enforces m uniquely
                        m = M - m0
                        if abs(m) > l:
                            continue
                        # Parity: vanishes if L + l0 + l is odd
                        if (L + l0 + l) % 2 == 1:
                            continue
                        val = _gaunt_coeff(L, M, l0, m0, l, m)
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        # Magnetic flip symmetry to fill negative M/m/m0
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return G


def load_gaunt_tensor(
    lmax: int, cache_dir: Union[str, Path] = "data/gaunt_cache", device: str = "cpu"
) -> torch.Tensor:
    """
    Load Gaunt tensor from cache or build with symmetry-aware routine and cache it.
    """
    path = _cache_path(lmax, cache_dir)
    if path.exists():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.1
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "G" in obj:
            return obj["G"].to(device=device)
        if torch.is_tensor(obj):
            return obj.to(device=device)
    G = _build_gaunt_tensor_torch_symmetric(lmax, device="cpu")
    torch.save({"G": G}, path)
    return G.to(device=device)
