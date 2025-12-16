import torch

from europa import transforms


def test_sh_roundtrip():
    lmax = 2
    n_nodes = 50
    positions = torch.randn(n_nodes, 3)
    positions = torch.nn.functional.normalize(positions, dim=-1)
    areas = torch.full((n_nodes,), 4.0 * torch.pi / n_nodes)
    values = torch.randn(n_nodes)
    coeffs = transforms.sh_forward(values, positions, lmax, areas)
    recon = transforms.sh_inverse(coeffs, positions, areas)
    assert recon.shape[-1] == n_nodes
    assert torch.isfinite(recon).all()
