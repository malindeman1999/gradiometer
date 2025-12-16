"""
Test 9: Small lmax direct matrix spot-check.

Goal: Verify selected mixing-matrix entries against analytic Gaunt values for
small lmax (here lmax=1 and lmax=2).

Procedure:
- Build Y_s with a single harmonic non-zero.
- Build the spectral mixing matrix M (first-order).
- Compare selected entries to analytic Gaunt-derived values.
"""
import os
import sys
from typing import List, Tuple

import torch

# Allow running from the validations/ directory by adding repo root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from europa import solvers


def lm_index(lmax: int, l: int, m: int) -> int:
    idx = 0
    for l2 in range(lmax + 1):
        for m2 in range(-l2, l2 + 1):
            if l2 == l and m2 == m:
                return idx
            idx += 1
    return -1


def expected_entry(lmax: int, L: int, M: int, l0: int, m0: int, lprime: int, mprime: int, omega: float, radius: float):
    """Analytic entry: sum over Gaunt of Y_s * F(l') * G."""
    Y = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.float64)
    Y[l0, lmax + m0] = 1.0
    F = -(1j * omega * radius) / (lprime * (lprime + 1))
    G = solvers._gaunt_coeff(L, M, l0, m0, lprime, mprime)  # type: ignore[attr-defined]
    return F * G


def check_matrix(lmax: int, omega: float, radius: float) -> List[str]:
    failures: List[str] = []
    Y_s = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.float64)
    # Single harmonic Y_1,0 or Y_2,0 depending on lmax
    if lmax >= 2:
        Y_s[2, lmax] = 1.0
    else:
        Y_s[1, lmax] = 1.0
    M = solvers._build_mixing_matrix_spectral(lmax, omega, radius, Y_s)

    tests: List[Tuple[int, int, int, int, int, int]] = []
    if lmax == 1:
        # Only one non-zero: L=0,M=0 <- l'=1,m'=0
        tests.append((0, 0, 1, 0, 1, 0))
    elif lmax == 2:
        # Only L=1 is representable at lmax=2
        tests.append((1, 0, 2, 0, 1, 0))
    else:
        # lmax>=3 with Y20: expect L=1,3 couplings from l'=1
        tests.append((1, 0, 2, 0, 1, 0))
        tests.append((3, 0, 2, 0, 1, 0))

    M_mat = M.reshape(M.shape[0], M.shape[1]) if isinstance(M, torch.Tensor) else torch.as_tensor(M)
    for (L, M_val, l0, m0, lprime, mprime) in tests:
        i = lm_index(lmax, L, M_val)
        j = lm_index(lmax, lprime, mprime)
        if i < 0 or j < 0:
            failures.append(f"Invalid index for (L={L},M={M_val}) or (l'={lprime},m'={mprime})")
            continue
        val = M_mat[int(i), int(j)].item()
        exp = expected_entry(lmax, L, M_val, l0, m0, lprime, mprime, omega, radius)
        diff = abs(val - exp)
        rel = diff / max(abs(exp), 1e-30)
        if diff > 1e-9 and rel > 1e-6:
            failures.append(
                f"L={L},M={M_val}; l'={lprime},m'={mprime}: got {val:.6e}, exp {exp:.6e}, diff {diff:.2e}, rel {rel:.2e}"
            )
    return failures


def main() -> None:
    cases = [
        (1, "Y10 on lmax=1"),
        (3, "Y20 on lmax=3"),
    ]
    omega = 1.0
    radius = 1.0
    overall_fail = False
    for lmax, desc in cases:
        print(f"\nCase: lmax={lmax} ({desc})")
        failures = check_matrix(lmax, omega, radius)
        if failures:
            overall_fail = True
            print("[FAIL] mixing matrix spot-check:")
            for msg in failures:
                print(f"  - {msg}")
        else:
            print("[OK]   mixing matrix entries match analytic Gaunt values")
    if overall_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
