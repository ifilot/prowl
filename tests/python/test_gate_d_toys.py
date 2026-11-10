"""Gate D: analytic Hamiltonian / COHP fixtures (no VASP data)."""
import numpy as np
import pytest

from prowl.analysis import AnalysisInput, EnergyGrid, analyze, find_bonds
from prowl.calculation import unfold_time_reversal

GRID = EnergyGrid(-6, 6, 1201, 0.05, 0.0)
ENGINES = pytest.mark.parametrize("engine", ["cpp", "numpy"])


def _inp(kfrac, T, S, eps, occ, lattice, frac, overlap_terms, mesh, labels):
    nat = len(frac)
    return AnalysisInput(kfrac=np.array(kfrac, float), weights=np.full(len(kfrac), 1 / len(kfrac)), T=[T], S=S,
                         eps=np.array(eps, float).reshape(1, len(kfrac), -1), occ=np.array(occ, float).reshape(1, len(kfrac), -1), lattice=lattice, frac=np.array(frac),
                         elements=["X"] * nat, zval=np.ones(nat), orb_offsets=list(range(nat + 1)),
                         orbital_labels=labels, overlap_terms=overlap_terms, efermi=0.0, mesh=mesh, spin_factor=2.0)


@ENGINES
def test_two_orbital_orthogonal_dimer(engine):
    alpha, beta = -2.0, -1.0
    H = np.array([[alpha, beta], [beta, alpha]])
    eps, V = np.linalg.eigh(H)
    lat = np.eye(3) * 20
    frac = [[0, 0, 0], [0.07, 0, 0]]
    terms = [(0, 1, (0, 0, 0), np.zeros((1, 1)))]
    inp = _inp([[0, 0, 0]], [V.astype(complex)], [np.eye(2, dtype=complex)], eps, [1.0, 0.0], lat, frac, terms,
               (1, 1, 1), ["A_s", "B_s"])
    bonds = find_bonds(lat, np.array(frac), 0.1, 2.0)
    r = analyze(inp, bonds, GRID, engine=engine)
    assert len(bonds) == 1
    assert r.icohp[0, 0] == pytest.approx(beta, abs=1e-12)  # 2 electrons x (1/2) beta: negative = bonding
    assert r.icoop[0, 0] == pytest.approx(0.0, abs=1e-14)    # orthogonal basis: no overlap population
    assert r.diagnostics["band_energy_ev"][0] == pytest.approx(2 * (alpha + beta))
    # spectral sum rule: integrated COHP curve over all energies equals sum over all bands (= 0 off-diagonal)
    dE = GRID.energies[1] - GRID.energies[0]
    assert np.sum(r.cohp[0, 0]) * dE == pytest.approx(0.0, abs=1e-6)
    occ_curve = np.sum(r.cohp[0, 0][GRID.energies < -2]) * dE
    assert occ_curve == pytest.approx(beta, abs=1e-6)  # independent integration check
    assert r.mulliken_charge == pytest.approx([0.0, 0.0], abs=1e-12)


@ENGINES
def test_nonorthogonal_dimer_cohp_coop_distinct(engine):
    s, alpha, beta = 0.3, -2.0, -1.5
    S = np.array([[1, s], [s, 1.0]])
    h = np.array([[alpha, beta], [beta, alpha]])
    # generalized eigenproblem h c = e S c
    w, U = np.linalg.eigh(S)
    X = (U / np.sqrt(w)) @ U.T
    e, Y = np.linalg.eigh(X @ h @ X)
    c = X @ Y                  # S-orthonormal LCAO coefficients
    T = S @ c                  # projections <chi|psi>
    lat = np.eye(3) * 20
    frac = [[0, 0, 0], [0.07, 0, 0]]
    terms = [(0, 1, (0, 0, 0), np.array([[s]]))]
    inp = _inp([[0, 0, 0]], [T.astype(complex)], [S.astype(complex)], e, [1.0, 0.0], lat, frac, terms, (1, 1, 1),
               ["A", "B"])
    r = analyze(inp, find_bonds(lat, np.array(frac), 0.1, 2.0), GRID, engine=engine)
    Hl = X @ h @ X  # Löwdin Hamiltonian
    cb = Y[:, 0]
    assert r.icohp[0, 0] == pytest.approx(2 * cb[0] * Hl[0, 1] * cb[1], abs=1e-12)
    assert r.icoop[0, 0] == pytest.approx(2 * c[0, 0] * s * c[1, 0], abs=1e-12)  # = s/(1+s)
    assert r.icoop[0, 0] == pytest.approx(s / (1 + s), abs=1e-12)
    assert r.icoop[0, 0] > 0 and r.icohp[0, 0] < 0
    # Mulliken (nonorthogonal) and Löwdin (orthonormal) gross populations differ in definition but both sum to 2
    assert r.mulliken_gp.sum() == pytest.approx(2.0) and r.lowdin_gp.sum() == pytest.approx(2.0)


@ENGINES
@pytest.mark.parametrize("N", [40, 400])
def test_periodic_chain_bloch_phases_and_kpoint_integration(N, engine):
    t = -1.0
    a = 2.0
    lat = np.diag([a, 10.0, 10.0])
    k = np.array([[(i - N // 2) / N, 0, 0] for i in range(N)])
    epsk = 2 * t * np.cos(2 * np.pi * k[:, 0])
    rng = np.random.default_rng(1)
    T = [np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.ones((1, 1)) for _ in range(N)]  # random gauge
    occ = (epsk < 0).astype(float)
    inp = _inp(k, T, [np.ones((1, 1), complex)] * N, epsk[:, None], occ[:, None], lat, [[0, 0, 0]],
               [(0, 0, (1, 0, 0), np.zeros((1, 1)))], (N, 1, 1), ["X_s"])
    bonds = find_bonds(lat, np.zeros((1, 3)), 0.1, 2.5)
    assert [b.R for b in bonds] == [(1, 0, 0)]
    r = analyze(inp, bonds, GRID, engine=engine)
    exact = 2 * t * np.mean(occ * np.cos(2 * np.pi * k[:, 0]))
    assert r.icohp[0, 0] == pytest.approx(exact, abs=1e-12)
    if N == 400:
        assert r.icohp[0, 0] == pytest.approx(2 * t / np.pi, abs=1e-3)  # continuum limit
    # band energy = sum over +R and -R bonds
    assert r.diagnostics["band_energy_ev"][0] == pytest.approx(2 * r.icohp[0, 0], abs=1e-12)

    # time-reversal half mesh gives the same result after unfolding
    half = [i for i in range(N) if k[i, 0] >= 0 or np.isclose(k[i, 0], -0.5)]
    kw = np.full(len(half), 1.0 / N)
    for j, i in enumerate(half):
        if not (np.isclose(k[i, 0], 0) or np.isclose(abs(k[i, 0]), 0.5)):
            kw[j] = 2.0 / N
    kk, ww, TT, SS, ee, oo = unfold_time_reversal(k[half], kw, [[T[i] for i in half]], [np.ones((1, 1), complex)] * len(half),
                                                  epsk[half][None, :, None], occ[half][None, :, None], (N, 1, 1))
    assert len(kk) == N and ww.sum() == pytest.approx(1.0)
    inp2 = AnalysisInput(kfrac=kk, weights=ww, T=TT, S=SS, eps=ee, occ=oo, lattice=lat, frac=np.zeros((1, 3)),
                         elements=["X"], zval=np.ones(1), orb_offsets=[0, 1], orbital_labels=["X_s"],
                         overlap_terms=[], efermi=0.0, mesh=(N, 1, 1), spin_factor=2.0)
    r2 = analyze(inp2, bonds, GRID, engine=engine)
    assert r2.icohp[0, 0] == pytest.approx(r.icohp[0, 0], abs=1e-12)


@ENGINES
def test_spin_polarised_channels(engine):
    """Explicit spin channels: no factor 2, channels resolved."""
    beta = -1.0
    H = np.array([[-2.0, beta], [beta, -2.0]])
    eps, V = np.linalg.eigh(H)
    lat = np.eye(3) * 20
    frac = np.array([[0, 0, 0], [0.07, 0, 0]])
    inp = AnalysisInput(kfrac=np.zeros((1, 3)), weights=np.ones(1), T=[[V.astype(complex)], [V.astype(complex)]],
                        S=[np.eye(2, dtype=complex)], eps=np.array([[eps], [eps]]),
                        occ=np.array([[[1.0, 0.0]], [[0.0, 0.0]]]), lattice=lat, frac=frac, elements=["X", "X"],
                        zval=np.array([0.5, 0.5]), orb_offsets=[0, 1, 2], orbital_labels=["A", "B"],
                        overlap_terms=[], efermi=0.0, mesh=(1, 1, 1), spin_factor=1.0)
    r = analyze(inp, find_bonds(lat, frac, 0.1, 2.0), GRID, engine=engine)
    assert r.icohp[0, 0] == pytest.approx(beta / 2)
    assert r.icohp[1, 0] == pytest.approx(0.0)
    assert r.diagnostics["electrons_dft"] == pytest.approx(1.0)


@ENGINES
@pytest.mark.parametrize("case,expected", [("H2", 1.0), ("He2", 0.0)])
def test_icobi_bond_orders_closed_shell(case, expected, engine):
    """ICOBI = Wiberg bond index: H2 (bonding orbital filled) = 1, He2 (bonding + antibonding) = 0."""
    H = np.array([[-2.0, -1.0], [-1.0, -2.0]])
    eps, V = np.linalg.eigh(H)
    occ = [1.0, 0.0] if case == "H2" else [1.0, 1.0]
    lat = np.eye(3) * 20
    frac = [[0, 0, 0], [0.04, 0, 0]]
    inp = _inp([[0, 0, 0]], [V.astype(complex)], [np.eye(2, dtype=complex)], eps, occ, lat, frac, [], (1, 1, 1),
               ["A", "B"])
    r = analyze(inp, find_bonds(lat, np.array(frac), 0.1, 2.0), GRID, engine=engine)
    assert r.icobi[0, 0] == pytest.approx(expected, abs=1e-12)


@ENGINES
def test_icobi_h2_cation_spin_polarised(engine):
    """H2+ (one electron, ISPIN=2): bond order 1/2 = 0.5 (spin up) + 0 (spin down)."""
    H = np.array([[-2.0, -1.0], [-1.0, -2.0]])
    eps, V = np.linalg.eigh(H)
    lat = np.eye(3) * 20
    frac = np.array([[0, 0, 0], [0.04, 0, 0]])
    inp = AnalysisInput(kfrac=np.zeros((1, 3)), weights=np.ones(1), T=[[V.astype(complex)], [V.astype(complex)]],
                        S=[np.eye(2, dtype=complex)], eps=np.array([[eps], [eps]]),
                        occ=np.array([[[1.0, 0.0]], [[0.0, 0.0]]]), lattice=lat, frac=frac, elements=["H", "H"],
                        zval=np.array([0.5, 0.5]), orb_offsets=[0, 1, 2], orbital_labels=["A", "B"],
                        overlap_terms=[], efermi=0.0, mesh=(1, 1, 1), spin_factor=1.0)
    r = analyze(inp, find_bonds(lat, frac, 0.1, 2.0), GRID, engine=engine)
    assert r.icobi[:, 0] == pytest.approx([0.5, 0.0], abs=1e-12)
