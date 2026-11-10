"""Gate A: synthetic WAVECARs with conjugate-symmetric coefficients (std vs Gamma-half)."""
import struct

import numpy as np
import pytest

from prowl import _core

A = np.array([[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]])
ENCUT = 180.0


def write_wavecar(path, coeffs, kfrac, rtag=45210):
    """coeffs: list over bands of stored coefficient arrays (single k, single spin)."""
    nb = len(coeffs)
    npl = len(coeffs[0])
    csize = 8 if rtag in (45200, 53300) else 16
    recl = max(npl * csize, (4 + 3 * nb) * 8, 13 * 8)
    recl = (recl + 7) // 8 * 8
    buf = bytearray(recl * (2 + 1 + nb))
    struct.pack_into("3d", buf, 0, recl, 1, rtag)
    struct.pack_into("12d", buf, recl, 1, nb, ENCUT, *A.reshape(-1), )
    hdr = [npl, *kfrac] + sum(([0.1 * b, 0.0, 1.0] for b in range(nb)), [])
    struct.pack_into(f"{len(hdr)}d", buf, 2 * recl, *hdr)
    dt = np.complex64 if csize == 8 else np.complex128
    for b, c in enumerate(coeffs):
        raw = np.asarray(c, dtype=dt).tobytes()
        buf[(3 + b) * recl:(3 + b) * recl + len(raw)] = raw
    path.write_bytes(bytes(buf))


def test_gamma_half_equals_full_sphere(tmp_path):
    full = np.array(_core.gvectors_full(A, [0, 0, 0], ENCUT))
    half = np.array(_core.gvectors_gamma_half(A, ENCUT))
    assert len(half) == (len(full) + 1) // 2
    index = {tuple(g): i for i, g in enumerate(full)}
    rng = np.random.default_rng(5)
    nb = 4
    c_half = rng.normal(size=(nb, len(half))) + 1j * rng.normal(size=(nb, len(half)))
    c_half[:, 0] = c_half[:, 0].real  # G = 0 real in a real gauge
    c_full = np.zeros((nb, len(full)), complex)
    for j, g in enumerate(half):
        c_full[:, index[tuple(g)]] = c_half[:, j]
        c_full[:, index[tuple(-g)]] = np.conj(c_half[:, j])  # conjugate symmetry
    norm = np.sqrt(np.sum(np.abs(c_full) ** 2, axis=1))
    c_full /= norm[:, None]
    c_half /= norm[:, None]
    stored = c_half.copy()
    stored[:, 1:] *= np.sqrt(2.0)  # vasp_gam storage convention
    write_wavecar(tmp_path / "W_std", list(c_full), [0, 0, 0])
    write_wavecar(tmp_path / "W_gam", list(stored), [0, 0, 0])
    rs, rg = _core.WavecarReader(str(tmp_path / "W_std")), _core.WavecarReader(str(tmp_path / "W_gam"))
    assert rs.layout["encoding"] == "FullComplex" and rg.layout["encoding"] == "GammaHalf"
    bs, bg = rs.read_block(0, 0, 0, nb), rg.read_block(0, 0, 0, nb)
    np.testing.assert_allclose(bs["plane_wave_norms"], 1.0, atol=1e-12)
    np.testing.assert_allclose(bg["plane_wave_norms"], 1.0, atol=1e-12)
    np.testing.assert_allclose(bg["coefficients"], c_half, atol=1e-13)
    # projection-invariant check: any real localized function gives identical overlaps
    W_full = rng.normal(size=(len(full), 3)) + 1j * rng.normal(size=(len(full), 3))
    for j, g in enumerate(full):  # impose W(-G) = W(G)* (real function in real space)
        i = index[tuple(-g)]
        if i > j:
            W_full[i] = np.conj(W_full[j])
        elif i == j:
            W_full[j] = W_full[j].real
    W_half = np.array([W_full[index[tuple(g)]] for g in half])
    w = np.where(np.all(half == 0, axis=1), 1.0, 2.0)
    for be in _core.available_backends():
        b = _core.make_backend(be)
        ref = b.project_full(W_full, bs["coefficients"].T)
        got = b.project_gamma(W_half, bg["coefficients"].T, list(w))
        assert np.abs(ref.imag).max() < 1e-12
        np.testing.assert_allclose(got, ref.real, atol=1e-12)


def test_truncated_and_spinor_files_fail_cleanly(tmp_path):
    full = np.array(_core.gvectors_full(A, [0, 0, 0], ENCUT))
    c = np.ones(len(full), complex) / np.sqrt(len(full))
    write_wavecar(tmp_path / "W", [c, c], [0, 0, 0])
    data = (tmp_path / "W").read_bytes()
    (tmp_path / "W_trunc").write_bytes(data[:-100])
    with pytest.raises(_core.ProwlError, match="CorruptFile"):
        _core.WavecarReader(str(tmp_path / "W_trunc"))
    sp = np.concatenate([c, c]) / np.sqrt(2)
    write_wavecar(tmp_path / "W_ncl", [sp], [0, 0, 0])
    r = _core.WavecarReader(str(tmp_path / "W_ncl"))
    assert r.layout["encoding"] == "Spinor"
    with pytest.raises(_core.ProwlError, match="UnsupportedFeature"):
        r.read_block(0, 0, 0, 1)
