import torch

from europa import grid
from europa.config import GridConfig


def test_neighbors_no_nan():
    cfg = GridConfig(nside=1, lmax=2, radius_m=1.0, device="cpu")
    g = grid.make_grid(cfg)
    assert not torch.isnan(g.neighbors).any()
