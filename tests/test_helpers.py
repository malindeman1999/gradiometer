from europa_model import helpers
import torch


def test_stable_dt_positive():
    ell = torch.tensor([1.0, 2.0, 3.0])
    dt = helpers.estimate_stable_dt_from_eigs(ell)
    assert dt > 0


def test_capacitance_ratio():
    ratio = helpers.estimate_capacitance_neglect(omega=1.0, epsilon=1e-12, sigma_s=1.0)
    assert ratio >= 0


def test_lmax_choice():
    lmax = helpers.choose_lmax_for_nside(8, safety=2.0)
    assert lmax >= 1
