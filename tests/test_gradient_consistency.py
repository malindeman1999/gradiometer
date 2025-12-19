import math

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    # Allow running the test file directly without installing the package.
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

from europa.gradient_utils import (
    sph_to_cart_coords,
    spherical_components_to_cart,
    toroidal_field_spherical,
    toroidal_gradients_spherical,
)
from europa import inductance


def _err_stats(actual, expected):
    diff = actual - expected
    abs_diff = diff.abs()
    abs_expected = expected.abs()
    max_abs = abs_diff.max().item()
    # Only compute relative error where the reference is safely away from zero.
    tol_floor = max(1e-6, (abs_expected.max().item() if abs_expected.numel() else 0.0) * 1e-6)
    mask = abs_expected > tol_floor
    if mask.any():
        max_rel = (abs_diff[mask] / abs_expected[mask]).max().item()
    else:
        max_rel = 0.0
    mean_abs = abs_diff.mean().item()
    return max_abs, max_rel, mean_abs


def test_spherical_gradients_match_numeric():
    torch.manual_seed(0)
    lmax = 3
    R = 1.0
    # Inflate input coefficients to avoid extremely tiny field/gradient magnitudes.
    J = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128) * 1e9
    # zero out invalid rows (l=0 toroidal not physical)
    J[0] = 0.0

    # fixed-radius positions on a shell (so spectral evaluation is valid)
    n_pts = 8
    theta = torch.linspace(0.3, math.pi - 0.3, n_pts)
    phi = torch.linspace(0.1, 2 * math.pi - 0.1, n_pts)
    th_grid, ph_grid = torch.meshgrid(theta, phi, indexing="ij")
    r_val = torch.full_like(th_grid, 1.4)
    positions = sph_to_cart_coords(r_val, th_grid, ph_grid).reshape(-1, 3)

    # analytic field + gradients
    Br, Bth, Bph = toroidal_field_spherical(J, R, positions)
    gBr, gBth, gBph = toroidal_gradients_spherical(J, R, positions)
    grad_true = torch.stack([gBr, gBth, gBph], dim=-2)  # [N,3,3]
    grad_true_normed = grad_true / 1e9

    # numeric spherical gradients via finite differencing on r/theta/phi by re-evaluating the field
    from europa.gradient_utils import finite_diff_gradients_spherical
    Btor, Bpol, Brad = inductance.spectral_b_from_surface_currents(J, torch.zeros_like(J), radius=R)
    grad_fd = finite_diff_gradients_spherical(
        B_tor=Btor,
        B_pol=Bpol,
        B_rad=Brad,
        positions=positions,
        delta_r=1e-4,
        delta_theta=1e-4,
        delta_phi=1e-4,
    )
    grad_fd_normed = grad_fd / 1e9

    max_abs, max_rel, mean_abs = _err_stats(grad_fd_normed, grad_true_normed)
    median_true = grad_true.abs().median().item()
    medians_by_component = grad_true.abs().median(dim=0).values.median(dim=0).values
    print(
        f"[spherical components] max_abs_err={max_abs:.2e}, "
        f"max_rel_err={max_rel:.2e}, mean_abs_err={mean_abs:.2e}, "
        f"median_true_mag={median_true:.2e}"
    )
    print(
        "[spherical component medians] Br={:.2e}, Btheta={:.2e}, Bphi={:.2e}".format(
            medians_by_component[0].item(),
            medians_by_component[1].item(),
            medians_by_component[2].item(),
        )
    )
    torch.testing.assert_close(
        grad_fd_normed,
        grad_true_normed,
        atol=1e-4,
        rtol=1e-3,
        msg=(
            f"max_abs_err={max_abs:.2e}, max_rel_err={max_rel:.2e}, "
            f"mean_abs_err={mean_abs:.2e}, median_true_mag={median_true:.2e}"
        ),
    )


def test_cartesian_gradient_norm_matches_spherical():
    torch.manual_seed(1)
    lmax = 3
    R = 1.0
    # Inflate input coefficients to avoid extremely tiny field/gradient magnitudes.
    J = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128) * 1e9
    J[0] = 0.0

    n_pts = 6
    theta = torch.linspace(0.4, math.pi - 0.4, n_pts)
    phi = torch.linspace(0.2, 2 * math.pi - 0.2, n_pts)
    th_grid, ph_grid = torch.meshgrid(theta, phi, indexing="ij")
    r_val = torch.full_like(th_grid, 1.3)
    positions = sph_to_cart_coords(r_val, th_grid, ph_grid).reshape(-1, 3)

    Br, Bth, Bph = toroidal_field_spherical(J, R, positions)
    gBr, gBth, gBph = toroidal_gradients_spherical(J, R, positions)
    grad_sph = torch.stack([gBr, gBth, gBph], dim=-2)  # [N,3,3]
    frob_sph = torch.linalg.norm(grad_sph, dim=(1, 2))
    frob_sph_normed = frob_sph / 1e9

    def B_cart_at(pos):
        Br_c, Bth_c, Bph_c = toroidal_field_spherical(J, R, pos)
        return spherical_components_to_cart(Br_c, Bth_c, Bph_c, pos)

    delta = 1e-4
    eye = torch.eye(3, device=positions.device, dtype=positions.dtype) * delta
    B0 = B_cart_at(positions)
    grads = []
    for axis in range(3):
        pos_plus = positions + eye[axis]
        pos_minus = positions - eye[axis]
        Bp = B_cart_at(pos_plus)
        Bm = B_cart_at(pos_minus)
        grads.append((Bp - Bm) / (2 * delta))
    grad_cart = torch.stack(grads, dim=-1)  # [N,3,3] dB_i/dx_j
    frob_cart = torch.linalg.norm(grad_cart, dim=(1, 2))
    frob_cart_normed = frob_cart / 1e9

    max_abs, max_rel, mean_abs = _err_stats(frob_cart_normed, frob_sph_normed)
    median_true = frob_sph.abs().median().item()
    print(
        f"[cartesian vs spherical norm] max_abs_err={max_abs:.2e}, "
        f"max_rel_err={max_rel:.2e}, mean_abs_err={mean_abs:.2e}, "
        f"median_true_mag={median_true:.2e}"
    )
    torch.testing.assert_close(
        frob_cart_normed,
        frob_sph_normed,
        atol=5e-4,
        rtol=1e-3,
        msg=(
            f"norm max_abs_err={max_abs:.2e}, max_rel_err={max_rel:.2e}, "
            f"mean_abs_err={mean_abs:.2e}, median_true_mag={median_true:.2e}"
        ),
    )


if __name__ == "__main__":
    import pytest

    # Run this file directly and print an explicit success message.
    exit_code = pytest.main(["-s", __file__])
    if exit_code == 0:
        print(f"{Path(__file__).name}: all tests passed")
    raise SystemExit(exit_code)
