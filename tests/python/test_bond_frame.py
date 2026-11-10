"""Bond-frame orbital analysis (analysis.orbital_frame = bond): rotation matrices, sigma/pi channels, invariances."""
import os
from pathlib import Path

import numpy as np
import pytest

from prowl import _core
from prowl.analysis import AnalysisInput, EnergyGrid, analyze, bond_frame_axes, find_bonds, orbital_rotation

GRID = EnergyGrid(-16, 8, 2401, 0.05, 0.0)
LM = [(0, 0), (1, -1), (1, 0), (1, 1)]  # s, p_y, p_z, p_x per atom


def _rot(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q if np.linalg.det(Q) > 0 else -Q


def test_rotation_matrix_mapping_orthogonality_composition():
    D = _core.real_harmonic_rotation(1, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.]]))  # x'=y, y'=z, z'=x
    assert np.allclose(D, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-13)  # p_y'=p_z, p_z'=p_x, p_x'=p_y
    rng = np.random.default_rng(3)
    for l in range(4):
        A, B = _rot(rng), _rot(rng)
        DA = _core.real_harmonic_rotation(l, A)
        assert np.allclose(DA @ DA.T, np.eye(2 * l + 1), atol=1e-13)
        assert np.allclose(_core.real_harmonic_rotation(l, A @ B), DA @ _core.real_harmonic_rotation(l, B), atol=1e-13)
    with pytest.raises(Exception):
        _core.real_harmonic_rotation(1, np.diag([1.0, 1.0, -1.0]))  # left-handed


def _sp_dimer(u):
    """Orthonormal s+p dimer along unit vector u, defined in its bond frame (sigma: s, p_z; pi: p_x, p_y)."""
    h = np.zeros((8, 8))
    np.fill_diagonal(h, [-10, -4, -4, -4, -9, -3, -3, -3])
    for i, j, v in [(0, 4, -2.0), (0, 6, 1.4), (2, 4, -1.6), (2, 6, 2.2), (1, 5, -1.1), (3, 7, -1.1)]:
        h[i, j] = h[j, i] = v
    lat = np.eye(3) * 20.0
    fa = np.array([0.5, 0.5, 0.5])
    fb = fa + 1.3 * np.asarray(u) / 20.0
    frac = np.array([fa, fb])
    ax = bond_frame_axes(lat, frac, find_bonds(lat, frac, 0.5, 2.0)[0])
    D = np.zeros((8, 8))
    for a in range(2):
        D[4 * a:4 * a + 4, 4 * a:4 * a + 4] = orbital_rotation(LM, ax)[0]
    hg = D.T @ h @ D                           # chi_frame = D chi_global  ->  H_global = D^T H_frame D
    e, V = np.linalg.eigh(hg)
    occ = np.array([1.0] * 4 + [0.0] * 4)
    inp = AnalysisInput(kfrac=np.zeros((1, 3)), weights=np.ones(1), T=[[V.astype(complex)]], S=[np.eye(8, dtype=complex)],
                        eps=e.reshape(1, 1, 8), occ=occ.reshape(1, 1, 8), lattice=lat, frac=frac, elements=["A", "B"],
                        zval=np.full(2, 4.0), orb_offsets=[0, 4, 8], orbital_labels=["s", "p_y", "p_z", "p_x"] * 2,
                        overlap_terms=[(0, 1, (0, 0, 0), np.zeros((4, 4)))], efermi=0.0, mesh=(1, 1, 1),
                        spin_factor=2.0, orbital_lm=LM * 2)
    return inp, find_bonds(lat, frac, 0.5, 2.0), h, e, V, D


@pytest.mark.parametrize("u", [(0, 0, 1), (1, 2, 3), (-0.3, 0.9, -0.2)])
def test_sp_dimer_channels_do_not_depend_on_orientation(u):
    u = np.asarray(u, float) / np.linalg.norm(u)
    inp, bonds, h, e, V, D = _sp_dimer(u)
    rg = analyze(inp, bonds, GRID)
    rb = analyze(inp, bonds, GRID, orbital_frame="bond")
    # reference in the model's own frame: ICOHP_ij = 2 sum_n f_n h_ij c_in c_jn with c = D V
    c = D @ V
    P = 2 * c[:, :4] @ c[:, :4].T
    ref = h[:4, 4:] * P[:4, 4:]
    sigma = ref[np.ix_([0, 2], [0, 2])].sum()
    pi = ref[np.ix_([1, 3], [1, 3])].sum()
    assert rb.icohp[0, 0] == pytest.approx(rg.icohp[0, 0], abs=1e-12)          # totals do not depend on the frame
    assert rb.icohp_channel[0, 0, 0] == pytest.approx(sigma, abs=1e-10)
    assert rb.icohp_channel[0, 0, 1] == pytest.approx(pi, abs=1e-10)
    assert np.abs(rb.icohp_channel[0, 0, 2:]).max() < 1e-12                   # no delta/phi, no sigma-pi mixing
    assert rb.icohp_orbital[0][0] == pytest.approx(ref, abs=1e-10)            # orbital matrix in the bond frame
    assert np.abs(rb.cohp_channel.sum(axis=2) - rb.cohp).max() < 1e-12        # channels add up to the COHP
    assert rb.bond_axes[0][2] == pytest.approx(u, abs=1e-12)                  # z' along A -> B


def test_numpy_engine_rejects_bond_frame():
    inp, bonds, *_ = _sp_dimer(np.array([0, 0, 1.0]))
    with pytest.raises(ValueError):
        analyze(inp, bonds, GRID, engine="numpy", orbital_frame="bond")


RUNS = os.environ.get("PROWL_TEST_RUNS")


@pytest.mark.skipif(not RUNS, reason="set PROWL_TEST_RUNS")
def test_diamond_equivalent_bonds_have_equal_channels():
    from prowl.calculation import Calculation
    from prowl.config import BondingConfig

    d = Path(RUNS) / "c_diamond_k7"
    if not (d / "WAVECAR").is_file():
        pytest.skip("fixture c_diamond_k7 missing")
    cfg = BondingConfig.for_profile("prowl")
    cfg.window_sensitivity = False
    cfg.min_distance, cfg.max_distance = 1.0, 1.7
    r = Calculation.open(d).analyze(cfg).result
    cfg.orbital_frame = "bond"
    rb = Calculation.open(d).analyze(cfg).result
    assert len(rb.bonds) == 4
    ch = rb.icohp_channel.sum(axis=0)                 # (4 bonds, 5 channels)
    assert np.allclose(ch, ch[0], atol=1e-6)          # symmetry-equivalent bonds: identical sigma / pi
    assert np.abs(ch[:, 2:]).max() < 1e-6             # C3 axis along each bond: no sigma-pi mixing, no delta (sp basis)
    assert ch[0, 0] < ch[0, 1] < 0                    # sigma dominates, pi bonding
    assert rb.icohp.sum(axis=0) == pytest.approx(r.icohp.sum(axis=0), abs=1e-10)
