"""
More symmetry-aware Gaunt tensor precompute (torch + NumPy variants).

Uses the fully symmetric triple-product form (no conjugation) and caches
Gaunt values keyed by sorted (l, m) tuples so all permutations share the
same value. Converts back to the conjugated convention via (-1)^M factor.

Symmetries exploited:
- Selection rule m1 + m2 + m3 = 0 (encoded via m = M - m0)
- Parity: l1 + l2 + l3 even
- Magnetic flip: fill ±M
- Full permutation symmetry of the unconjugated triple product
"""
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch

from europa_model.solvers import _gaunt_coeff


def _cache_path(lmax: int, cache_dir: Union[str, Path] = "data/gaunt_cache_adv") -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gaunt_adv_lmax{lmax}.pt"


def _gaunt_no_conj_cached(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int, cache: Dict[Tuple[Tuple[int, int], ...], float]
) -> float:
    """
    Gaunt triple product without conjugation:
        ∫ Y_{l1,m1} Y_{l2,m2} Y_{l3,m3} dΩ
    symmetric under permutation of the three (l,m) pairs.
    """
    key = tuple(sorted(((l1, m1), (l2, m2), (l3, m3))))
    key_flip = tuple(sorted(((l1, -m1), (l2, -m2), (l3, -m3))))
    for k in (key, key_flip):
        if k in cache:
            return cache[k]
    # Use _gaunt_coeff with the first slot conjugated: G = (-1)^m1 * T where T is the triple product with m1 -> -m1
    val = (-1) ** m1 * _gaunt_coeff(l1, -m1, l2, m2, l3, m3)
    cache[key] = val
    cache[key_flip] = val
    return val


def build_gaunt_tensor_torch_adv(lmax: int, device: str = "cpu") -> torch.Tensor:
    """
    Torch Gaunt tensor using full permutation symmetry of the unconjugated triple product.
    """
    G = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
        device=device,
    )
    cache: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for L in range(lmax + 1):
        for M in range(0, L + 1):  # fill M>=0 and mirror
            M_idx = L + M
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        if (L + l0 + l) % 2 == 1:
                            continue
                        # Triangle inequality (Wigner 3j) pruning
                        if not (abs(L - l0) <= l <= L + l0):
                            continue
                        # Triple product value (symmetric in permutations)
                        T = _gaunt_no_conj_cached(L, -M, l0, m0, l, m, cache)
                        val = (-1) ** M * T
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return G


def build_gaunt_tensor_numpy_adv(lmax: int) -> torch.Tensor:
    """
    NumPy Gaunt tensor using full permutation symmetry; returns a torch tensor for consistency.
    """
    G = np.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=np.float64,
    )
    cache: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for L in range(lmax + 1):
        for M in range(0, L + 1):
            M_idx = L + M
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        if (L + l0 + l) % 2 == 1:
                            continue
                        if not (abs(L - l0) <= l <= L + l0):
                            continue
                        T = _gaunt_no_conj_cached(L, -M, l0, m0, l, m, cache)
                        val = (-1) ** M * T
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return torch.from_numpy(G)


def load_gaunt_tensor_adv(
    lmax: int, cache_dir: Union[str, Path] = "data/gaunt_cache_adv", device: str = "cpu"
) -> torch.Tensor:
    """
    Load or build the advanced-symmetry Gaunt tensor (torch).
    """
    path = _cache_path(lmax, cache_dir)
    if path.exists():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "G" in obj:
            return obj["G"].to(device=device)
        if torch.is_tensor(obj):
            return obj.to(device=device)
    G = build_gaunt_tensor_torch_adv(lmax, device="cpu")
    torch.save({"G": G}, path)
    return G.to(device=device)


# Advanced2: adds explicit triangle pruning and sign-flip cache re-use (already above),
# but uses a tighter canonical key: sorted by l then |m| to maximize reuse across flips.
def _gaunt_no_conj_cached_adv2(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int, cache: Dict[Tuple[Tuple[int, int], ...], float]
) -> float:
    key = tuple(sorted(((l1, abs(m1)), (l2, abs(m2)), (l3, abs(m3)))))
    if key in cache:
        return cache[key]
    val = (-1) ** m1 * _gaunt_coeff(l1, -m1, l2, m2, l3, m3)
    cache[key] = val
    return val


def build_gaunt_tensor_torch_adv2(lmax: int, device: str = "cpu") -> torch.Tensor:
    G = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
        device=device,
    )
    cache: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for L in range(lmax + 1):
        for M in range(0, L + 1):
            M_idx = L + M
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        if (L + l0 + l) % 2 == 1:
                            continue
                        if not (abs(L - l0) <= l <= L + l0):
                            continue
                        T = _gaunt_no_conj_cached_adv2(L, -M, l0, m0, l, m, cache)
                        val = (-1) ** M * T
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return G


def build_gaunt_tensor_numpy_adv2(lmax: int) -> torch.Tensor:
    G = np.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=np.float64,
    )
    cache: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for L in range(lmax + 1):
        for M in range(0, L + 1):
            M_idx = L + M
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    for l in range(lmax + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        if (L + l0 + l) % 2 == 1:
                            continue
                        if not (abs(L - l0) <= l <= L + l0):
                            continue
                        T = _gaunt_no_conj_cached_adv2(L, -M, l0, m0, l, m, cache)
                        val = (-1) ** M * T
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return torch.from_numpy(G)


def load_gaunt_tensor_adv2(
    lmax: int, cache_dir: Union[str, Path] = "data/gaunt_cache_adv2", device: str = "cpu"
) -> torch.Tensor:
    path = _cache_path(lmax, cache_dir)
    if path.exists():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "G" in obj:
            return obj["G"].to(device=device)
        if torch.is_tensor(obj):
            return obj.to(device=device)
    G = build_gaunt_tensor_torch_adv2(lmax, device="cpu")
    torch.save({"G": G}, path)
    return G.to(device=device)


# Advanced3: adds Gaunt call caching and tighter loop bounds.
def _gaunt_coeff_cached_full(
    L: int,
    M: int,
    l0: int,
    m0: int,
    l: int,
    m: int,
    cache_val: Dict[Tuple[int, int, int, int, int, int], float],
) -> float:
    key_full = (L, M, l0, m0, l, m)
    if key_full in cache_val:
        return cache_val[key_full]
    val = _gaunt_coeff(L, M, l0, m0, l, m)
    cache_val[key_full] = val
    return val


def build_gaunt_tensor_torch_adv3(lmax: int, device: str = "cpu") -> torch.Tensor:
    G = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
        device=device,
    )
    cache_full: Dict[Tuple[int, int, int, int, int, int], float] = {}
    for L in range(lmax + 1):
        for M in range(0, L + 1):
            M_idx = L + M
            # Tighten l0 range via triangle inequality with l later
            for l0 in range(max(0, L - lmax), min(lmax, L + lmax) + 1):
                for m0 in range(-l0, l0 + 1):
                    # Precompute bounds for l given triangle with L,l0
                    l_min = abs(L - l0)
                    l_max_local = min(lmax, L + l0)
                    for l in range(l_min, l_max_local + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        if (L + l0 + l) % 2 == 1:
                            continue
                        val = _gaunt_coeff_cached_full(L, M, l0, m0, l, m, cache_full)
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return G


def build_gaunt_tensor_numpy_adv3(lmax: int) -> torch.Tensor:
    G = np.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=np.float64,
    )
    cache_full: Dict[Tuple[int, int, int, int, int, int], float] = {}
    for L in range(lmax + 1):
        for M in range(0, L + 1):
            M_idx = L + M
            for l0 in range(max(0, L - lmax), min(lmax, L + lmax) + 1):
                for m0 in range(-l0, l0 + 1):
                    l_min = abs(L - l0)
                    l_max_local = min(lmax, L + l0)
                    for l in range(l_min, l_max_local + 1):
                        m = M - m0
                        if abs(m) > l:
                            continue
                        if (L + l0 + l) % 2 == 1:
                            continue
                        val = _gaunt_coeff_cached_full(L, M, l0, m0, l, m, cache_full)
                        G[L, M_idx, l0, lmax + m0, l, lmax + m] = val
                        G[L, L - M, l0, lmax - m0, l, lmax - m] = val
    return torch.from_numpy(G)


def load_gaunt_tensor_adv3(
    lmax: int, cache_dir: Union[str, Path] = "data/gaunt_cache_adv3", device: str = "cpu"
) -> torch.Tensor:
    path = _cache_path(lmax, cache_dir)
    if path.exists():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "G" in obj:
            return obj["G"].to(device=device)
        if torch.is_tensor(obj):
            return obj.to(device=device)
    G = build_gaunt_tensor_torch_adv3(lmax, device="cpu")
    torch.save({"G": G}, path)
    return G.to(device=device)
