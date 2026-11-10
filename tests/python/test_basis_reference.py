"""Gate B: STO basis mathematics against independent mpmath/SciPy references."""
import mpmath as mp
import numpy as np
import pytest

from prowl import _core


@pytest.mark.parametrize("N,l,zeta,q", [(2, 0, 1.3, 0.2), (3, 1, 2.1, 1.7), (4, 2, 0.9, 3.3), (5, 3, 1.6, 8.0),
                                        (6, 0, 4.0, 25.0), (4, 1, 30.0, 2.0)])
def test_sto_bessel_integral_vs_mpmath(N, l, zeta, q):
    mp.mp.dps = 30
    ref = mp.quadosc(lambda r: r ** N * mp.e ** (-zeta * r) * mp.sqrt(mp.pi / (2 * q * r)) * mp.besselj(l + 0.5, q * r),
                     [0, mp.inf], omega=q) if q > 0 else None
    assert _core.sto_bessel_integral(N, l, zeta, q) == pytest.approx(float(ref), rel=1e-9, abs=1e-14)


def test_koga_orbitals_normalised_and_nodes():
    lib = _core.StoLibrary.load_default()
    si3s = lib.shell("Si", "3s")
    r = np.linspace(1e-4, 8, 20001)
    R = np.array(si3s.values(list(r)))
    assert np.trapezoid(R * R * r * r, r) == pytest.approx(1.0, abs=2e-5)
    nodes = np.sum(np.sign(R[1:]) != np.sign(R[:-1]))
    assert nodes == 2  # 3s has n - l - 1 = 2 radial nodes
    c2p = lib.shell("C", "2p")
    assert np.sum(np.diff(np.sign(c2p.values(list(r)))) != 0) == 0


def test_fourier_transform_roundtrip():
    """Rt(q) = int r^2 j_l(qr) R(r) dr checked by direct quadrature."""
    lib = _core.StoLibrary.load_default()
    for el, lab in (("C", "2p"), ("Ga", "3d"), ("Si", "3s")):
        sh = lib.shell(el, lab)
        r = np.geomspace(1e-6, 40, 60001)
        R = np.array(sh.values(list(r)))
        for q in (0.0, 0.7, 3.0, 9.0):
            j = np.array([_core.sph_bessel(sh.l, q * x) for x in r])
            ref = np.trapezoid(r * r * j * R, r)
            assert sh.fourier(q) == pytest.approx(ref, rel=2e-5, abs=1e-8)


def test_two_centre_overlap_vs_3d_quadrature():
    """C 2p_z - C 2s overlap along z against a cylindrical 2D quadrature (independent algorithm)."""
    from scipy import integrate
    lib = _core.StoLibrary.load_default()
    s, p = lib.shell("C", "2s"), lib.shell("C", "2p")
    d = 1.4
    o = _core.TwoCenterOverlap(p, s).evaluate([0, 0, d])  # rows p_y, p_z, p_x
    Ylm1 = np.sqrt(3 / (4 * np.pi))
    Y00 = 1 / np.sqrt(4 * np.pi)

    def f(z, rho):
        r1 = np.hypot(rho, z)
        r2 = np.hypot(rho, z - d)
        return 2 * np.pi * rho * p.value(r1) * Ylm1 * z / max(r1, 1e-14) * s.value(r2) * Y00

    ref, _ = integrate.dblquad(f, 0, 12, -12, 12 + d, epsabs=1e-10)
    assert o[1, 0] == pytest.approx(ref, rel=1e-6)
    assert abs(o[0, 0]) < 1e-12 and abs(o[2, 0]) < 1e-12


def test_koga_library_regenerates_from_vendored_upstream(tmp_path):
    """data/basis/koga/koga_hf_sto.json is a pure format conversion of the vendored upstream tables."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[2]
    out = tmp_path / "koga.json"
    subprocess.run([sys.executable, str(root / "tools" / "convert_koga.py"), str(root / "data/basis/koga/upstream"),
                    str(out)], check=True, capture_output=True)
    assert out.read_bytes() == (root / "data/basis/koga/koga_hf_sto.json").read_bytes()
