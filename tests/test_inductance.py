import torch

from europa_model import inductance, observation


def test_spectral_mapping_swaps_components():
    lmax = 2
    tor = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.float64)
    pol = torch.zeros_like(tor)
    idx = lmax  # m = 0 slot for l = 1 when lmax=2
    tor[1, idx] = 1.0

    B_tor, B_pol, B_rad = inductance.spectral_b_from_surface_currents(tor, pol, radius=1.0)
    expected_pol = torch.tensor(float(inductance.MU0 / 3.0), dtype=torch.float64)  # l=1 => mu0 * l /(2l+1)
    expected_rad = torch.tensor(float(-inductance.MU0 / 6.0), dtype=torch.float64)  # l=1 => -mu0/[(2l+1) l(l+1)]
    assert torch.allclose(B_tor[1, idx], torch.tensor(0.0, dtype=torch.complex128))
    assert torch.allclose(B_pol[1, idx].real, expected_pol)
    assert torch.allclose(B_rad[1, idx].real, expected_rad)


def test_evaluate_B_from_currents_spectral_path():
    lmax = 2
    tor = torch.zeros((lmax + 1, 2 * lmax + 1), dtype=torch.float64)
    pol = torch.zeros_like(tor)
    tor[1, lmax] = 1.0  # simple mode

    n_nodes = 20
    positions = torch.randn(n_nodes, 3, dtype=torch.float64)
    positions = torch.nn.functional.normalize(positions, dim=-1)
    areas = torch.full((n_nodes,), 4.0 * torch.pi / n_nodes, dtype=torch.float64)

    B_grid = observation.evaluate_B_from_currents(tor, pol, positions, areas)
    assert B_grid.shape == (n_nodes, 3)
    assert torch.isfinite(B_grid).all()
