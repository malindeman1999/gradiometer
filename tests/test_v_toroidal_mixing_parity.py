import torch

from europa_model.solver_variants.solver_variant_precomputed import (
    _build_mixing_matrix_precomputed,
    _build_mixing_matrix_precomputed_sparse,
)


def _random_gaunt_dense(lmax: int, seed: int = 7) -> torch.Tensor:
    g = torch.zeros(
        (lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1, lmax + 1, 2 * lmax + 1),
        dtype=torch.float64,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for L in range(lmax + 1):
        for M_idx in range(2 * L + 1):
            for l0 in range(lmax + 1):
                for m0 in range(-l0, l0 + 1):
                    m0_idx = lmax + m0
                    for l_in in range(lmax + 1):
                        for m_in in range(-l_in, l_in + 1):
                            m_in_idx = lmax + m_in
                            g[L, M_idx, l0, m0_idx, l_in, m_in_idx] = torch.randn(
                                (), generator=gen, dtype=torch.float64
                            )
    return g


def test_dense_sparse_parity_for_gaunt_and_v_toroidal() -> None:
    lmax = 2
    omega = 1.234
    radius = 1.56e6

    G_dense = _random_gaunt_dense(lmax=lmax)
    G_sparse = G_dense.to_sparse_coo().coalesce()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(17)
    Y_real = torch.randn((lmax + 1, 2 * lmax + 1), generator=gen, dtype=torch.float64)
    Y_imag = torch.randn((lmax + 1, 2 * lmax + 1), generator=gen, dtype=torch.float64)
    Y = (Y_real + 1j * Y_imag).to(torch.complex128)

    M_dense_gaunt = _build_mixing_matrix_precomputed(lmax, omega, radius, Y, G_dense, coupling="gaunt")
    M_sparse_gaunt = _build_mixing_matrix_precomputed_sparse(lmax, omega, radius, Y, G_sparse, coupling="gaunt")
    assert torch.allclose(M_dense_gaunt, M_sparse_gaunt, rtol=1e-10, atol=1e-10)

    M_dense_v = _build_mixing_matrix_precomputed(lmax, omega, radius, Y, G_dense, coupling="v_toroidal")
    M_sparse_v = _build_mixing_matrix_precomputed_sparse(lmax, omega, radius, Y, G_sparse, coupling="v_toroidal")
    assert torch.allclose(M_dense_v, M_sparse_v, rtol=1e-10, atol=1e-10)

