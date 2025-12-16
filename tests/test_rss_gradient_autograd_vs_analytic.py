import torch

from europa import inductance
from europa.gradient_utils import toroidal_field_and_gradients_spherical
from europa.observation import evaluate_field_from_spectral


def _rss_autograd(J_tor: torch.Tensor, radius: float, positions: torch.Tensor) -> torch.Tensor:
    """
    Autograd-based RSS |∇B| using the spectral field evaluation pipeline.
    """
    B_tor, B_pol, B_rad = inductance.spectral_b_from_surface_currents(
        J_tor, torch.zeros_like(J_tor), radius=radius
    )
    pos = positions.detach().requires_grad_(True)
    B_cart = evaluate_field_from_spectral(B_tor, B_pol, B_rad, pos)
    grads = []
    for comp in range(3):
        g_real = torch.autograd.grad(B_cart[:, comp].real.sum(), pos, retain_graph=True)[0]
        g_imag = torch.autograd.grad(B_cart[:, comp].imag.sum(), pos, retain_graph=True)[0]
        grads.append(g_real + 1j * g_imag)
    grad_tensor = torch.stack(grads, dim=1)  # [N,3,3]
    return torch.linalg.norm(grad_tensor, dim=(1, 2))


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

    rss_autograd = _rss_autograd(J_tor, radius, pos_cart)

    _, _, _, grad_Br, grad_Btheta, grad_Bphi = toroidal_field_and_gradients_spherical(
        J_tor, radius=radius, positions=pos_cart
    )
    grad_tensor = torch.stack([grad_Br, grad_Btheta, grad_Bphi], dim=1)
    rss_analytic = torch.linalg.norm(grad_tensor, dim=(1, 2))

    # Gradients are ~1e-11; enforce very tight absolute tolerance.
    torch.testing.assert_close(rss_autograd, rss_analytic, rtol=1e-6, atol=1e-13)
