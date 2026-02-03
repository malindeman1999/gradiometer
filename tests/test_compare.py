import torch

from europa_model import compare_td_fd


def test_compare_shapes():
    td = torch.zeros((5, 3))
    fd = torch.zeros((5, 3))
    rms, phase = compare_td_fd.compare_series(td, fd)
    assert rms.shape == (3,)
    assert phase.shape == (3,)


def test_compare_values():
    td = torch.ones((4, 2))
    fd = torch.fft.fft(td, dim=0)
    rms, phase = compare_td_fd.compare_series(td, fd)
    assert torch.allclose(rms, torch.zeros_like(rms), atol=1e-6)
