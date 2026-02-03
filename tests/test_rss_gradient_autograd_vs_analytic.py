import torch

from europa_model.gradient_utils import rss_gradient_cartesian_autograd, toroidal_field_and_gradients_spherical


def test_rss_autograd_matches_analytic_toroidal():
    torch.manual_seed(0)
    lmax = 3
    radius = 1.56e6
    # Random toroidal surface currents (complex)
    J_tor = torch.randn((lmax + 1, 2 * lmax + 1), dtype=torch.complex128)
    J_tor += 1j * torch.randn_like(J_tor)

    # Sample a few points on a shell outside the surface
    n_pts = 5
    pos_cart = torch.randn((n_pts, 3), dtype=torch.float64)
    pos_cart = pos_cart / pos_cart.norm(dim=-1, keepdim=True) * (radius * 1.1)

    rss_autograd = rss_gradient_cartesian_autograd(J_tor, radius, pos_cart)

    _, _, _, grad_Br, grad_Btheta, grad_Bphi = toroidal_field_and_gradients_spherical(
        J_tor, radius=radius, positions=pos_cart
    )
    grad_tensor = torch.stack([grad_Br, grad_Btheta, grad_Bphi], dim=1)
    rss_analytic = torch.linalg.norm(grad_tensor, dim=(1, 2))

    # Gradients are ~1e-11; enforce very tight absolute tolerance.
    torch.testing.assert_close(rss_autograd, rss_analytic, rtol=1e-6, atol=1e-13)
