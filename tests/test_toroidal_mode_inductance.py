from pathlib import Path

from toroidal_inductance.toroidal_mode_inductance import (
    inductance_l0_toroidal_axisymmetric,
    inductance_lm_toroidal_general,
    sweep_unit_sphere_inductance,
)


def test_general_and_axisymmetric_match_for_m0():
    l = 2
    general = inductance_lm_toroidal_general(l, 0, tol=1.0e-2, max_refinements=4)
    axisym = inductance_l0_toroidal_axisymmetric(l, tol=1.0e-2, max_refinements=4)
    rel = abs(general.inductance_h - axisym.inductance_h) / axisym.inductance_h
    assert rel < 2.0e-2


def test_m_independence_same_l():
    l = 3
    m0 = inductance_lm_toroidal_general(l, 0, tol=1.0e-2, max_refinements=4)
    m1 = inductance_lm_toroidal_general(l, 1, tol=1.0e-2, max_refinements=4)
    rel = abs(m1.inductance_h - m0.inductance_h) / m0.inductance_h
    assert rel < 2.0e-2


def test_sweep_resume_from_first_uncomputed_l(tmp_path: Path):
    data_path = tmp_path / "inductance_unit_sphere.csv"
    rows1 = sweep_unit_sphere_inductance(3, data_path=data_path, tol=1.0e-2)
    assert [int(r["l"]) for r in rows1] == [1, 2, 3]

    rows2 = sweep_unit_sphere_inductance(4, data_path=data_path, tol=1.0e-2)
    assert [int(r["l"]) for r in rows2] == [1, 2, 3, 4]
