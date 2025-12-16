import torch

from europa import diagnostics, inductance


def test_divergence_spectral_toroidal_zero():
    lmax = 2
    device = "cpu"
    tor = torch.zeros((lmax + 1, 2 * lmax + 1), device=device, dtype=torch.complex128)
    pol = torch.zeros_like(tor)
    div = diagnostics.divergence_spectral(tor, pol, radius=1.0)
    assert torch.allclose(div, torch.zeros_like(div))


def test_faraday_zero():
    B = torch.zeros((3, 4, 3))
    E = torch.zeros_like(B)
    positions = torch.randn(4, 3)
    positions = torch.nn.functional.normalize(positions, dim=-1)
    neighbors = torch.zeros((4, 1), dtype=torch.long)
    err = diagnostics.faraday_error(B, E, positions, neighbors, dt=1.0)
    assert torch.isfinite(err)
    assert err == 0
