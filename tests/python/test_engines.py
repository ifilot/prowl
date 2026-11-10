"""C++ core vs NumPy reference for the reconstruction building blocks (synthetic data)."""
import numpy as np
import pytest

from prowl import _core
from prowl import analysis as an


def _random_problem(seed, norb=6, nb=14):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    S = A @ A.conj().T / norb + np.eye(norb)
    T = rng.normal(size=(norb, nb)) + 1j * rng.normal(size=(norb, nb))
    eps = np.sort(rng.normal(size=nb) * 5)
    occ = np.r_[np.ones(3), np.zeros(nb - 3)]
    return T, S, eps, occ


@pytest.mark.parametrize("seed", range(4))
def test_reconstruct_k_matches(seed):
    T, S, eps, _ = _random_problem(seed)
    for nb in (6, 14):
        a = _core.reconstruct_k(T[:, :nb], S, eps[:nb])
        b = an.reconstruct_k(T[:, :nb], S, eps[:nb])
        for key in ("C", "Cn", "H", "spill", "completeness"):
            np.testing.assert_allclose(a[key], getattr(b, key), atol=1e-10)
        if nb == T.shape[0]:
            assert np.max(a["band_residual"]) == pytest.approx(b.residual, abs=1e-10)
        else:
            np.testing.assert_allclose(a["band_residual"], b.residual, atol=1e-10)


def test_rectangular_reconstruction_residual():
    """A wide band window retains the per-band energy error, including unoccupied bands."""
    T, S, eps, _ = _random_problem(19, norb=12, nb=192)
    a = _core.reconstruct_k(T, S, eps)
    b = an.reconstruct_k(T, S, eps)
    np.testing.assert_allclose(a["band_residual"], b.residual, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_window_helpers_match(seed):
    T, S, eps, occ = _random_problem(seed)
    assert _core.window_conditioning(T, S, 6) == pytest.approx(an.window_conditioning(T, S, 6), abs=1e-10)
    e = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    assert _core.cut_at_gap(e, 3) == an.cut_at_gap(e, 3) == 1
    Ta, ea, oa, wa, ga = _core.frozen_window_subspace(T, S, eps, occ, 6, 12)
    Tb, eb, ob, wb, gb = an.frozen_window_subspace(T, S, eps, occ, 6, ncand=12)
    np.testing.assert_allclose(ea, eb, atol=1e-10)
    # the complement basis is defined up to rotations within degenerate Rayleigh-Ritz levels;
    # compare the gauge-invariant projector
    np.testing.assert_allclose(Ta @ Ta.conj().T, Tb @ Tb.conj().T, atol=1e-10)
    assert wa == pytest.approx(wb, abs=1e-10) and ga == pytest.approx(gb, abs=1e-10)


def test_window_larger_than_band_count_is_an_error():
    T, S, eps, occ = _random_problem(0, norb=6, nb=4)
    with pytest.raises(_core.ProwlError, match="exceeds"):
        _core.window_conditioning(T, S, 6)


def test_find_bonds_with_unwrapped_coordinates():
    lat = np.eye(3) * 3.0
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    a = an.find_bonds(lat, frac, 0.1, 3.1)
    b = an.find_bonds(lat, frac + np.array([[0, 0, 0], [2.0, -3.0, 1.0]]), 0.1, 3.1)
    assert sorted(round(x.distance, 8) for x in a) == sorted(round(x.distance, 8) for x in b)


@pytest.mark.skipif("cuda" not in _core.available_backends(), reason="no CUDA device/build")
@pytest.mark.parametrize("n", [5, 300])
def test_cusolver_eigensolver_matches_lapack(n):
    rng = np.random.default_rng(n)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = A + A.conj().T
    be = _core.make_backend("cuda")
    assert be.has_hermitian_eigensolver and not _core.make_backend("cpu").has_hermitian_eigensolver
    w, V = be.hermitian_eig(A)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(A), atol=1e-10 * np.abs(w).max())
    assert np.abs(A @ V - V * w).max() < 1e-10 * np.abs(w).max()
    assert np.abs(V.conj().T @ V - np.eye(n)).max() < 1e-12


def test_tetrahedron_corner_weights_vs_monte_carlo():
    """Blöchl weights equal the barycentric-weighted volume below E (independent Monte Carlo)."""
    e = np.array([-0.4, 0.1, 0.35, 1.2])
    rng = np.random.default_rng(3)
    u = rng.dirichlet([1, 1, 1, 1], 400000)
    en = u @ e
    for E in (-0.2, 0.2, 0.6, 1.0):
        w = np.array(_core.tetra_weights(list(e), E, 1.0))
        mc = (u * (en < E)[:, None]).mean(axis=0)
        np.testing.assert_allclose(w, mc, atol=3e-3)
