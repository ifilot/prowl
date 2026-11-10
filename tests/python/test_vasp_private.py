"""Regression and acceptance tests on private VASP fixtures (skipped when absent).

Generate fixtures with tools/run_vasp_fixture.py and set PROWL_TEST_RUNS."""
import json
from pathlib import Path

import numpy as np
import pytest

from prowl import _core
from prowl.calculation import Calculation
from prowl.config import BondingConfig

EXPECTED = json.loads((Path(__file__).parents[1] / "private_vasp_manifest" / "expected.json").read_text())
TOL = EXPECTED["tolerances"]


def _run(run_dir, mode, bond=None, backend="cpu", **kw):
    # Regression values in expected.json were produced with the 'auto' window (profile 'prowl');
    # profile behaviour is tested separately (test_profiles_on_real_data).
    cfg = BondingConfig.for_profile("prowl")
    cfg.window_sensitivity = False
    cfg.projection = mode
    if bond:
        cfg.bond_mode = "pairs"
        cfg.bond_pairs = [bond]
    for k, v in kw.items():
        setattr(cfg, k, v)
    return Calculation.open(run_dir).analyze(cfg, backend=backend)


def _need(runs_dir, name):
    d = runs_dir / name
    if not (d / "WAVECAR").is_file():
        pytest.skip(f"fixture {name} missing")
    return d


@pytest.mark.parametrize("system", ["c_diamond_k7", "gaas_k6"])
@pytest.mark.parametrize("mode", ["paw_onsite", "paw_full"])
def test_regression_and_lobster_equivalence(runs_dir, system, mode):
    ref = EXPECTED[system]
    res = _run(_need(runs_dir, system), mode, ref["bond"])
    r = res.result
    exp = ref[mode]
    assert r.diagnostics["max_paw_norm_deviation"] < TOL["paw_norm"]
    assert r.icohp[0, 0] == pytest.approx(exp["icohp"], abs=TOL["icohp_ev"])
    assert r.icoop[0, 0] == pytest.approx(exp["icoop"], abs=TOL["icoop"])
    assert 100 * r.diagnostics["abs_charge_spilling"] == pytest.approx(exp["spilling_pct"], abs=TOL["spilling_pct"])
    if "mulliken" in exp:
        np.testing.assert_allclose(r.mulliken_charge, exp["mulliken"], atol=TOL["charge"])
        np.testing.assert_allclose(r.lowdin_charge, exp["lowdin"], atol=TOL["charge"])
    if mode == "paw_onsite":  # one-centre PAW == LOBSTER's projection: identical numbers
        assert r.icohp[0, 0] == pytest.approx(ref["lobster"]["icohp"], abs=TOL["icohp_ev"])


def test_transfer_matrices_match_lobster(runs_dir):
    from prowl.validation.lobster import read_matrices

    d = _need(runs_dir, "c_diamond_k7")
    if not (d / "transferMatrices.lobster").is_file():
        pytest.skip("no LOBSTER transfer matrices")
    lt = read_matrices(d / "transferMatrices.lobster")["by_key"]
    st, pcs = _core.read_poscar(str(d / "POSCAR")), _core.read_potcar(str(d / "POTCAR"))
    basis = _core.AtomicBasis(st, _core.StoLibrary.load_default(), {"C": ["2s", "2p"]})
    opt = _core.ProjectionOptions()
    opt.include_offsite_paw = False
    m = _core.ProjectionModel(st, basis, pcs, opt)
    rd, be = _core.WavecarReader(str(d / "WAVECAR")), _core.make_backend("cpu")
    for k in (1, 2, 50, 200):
        T = m.project(rd, 0, k - 1, 0, 8, be)["a_ae"]
        L = lt[(1, k)]
        P, PL = T @ T.conj().T, L @ L.conj().T
        assert np.linalg.norm(P - PL) / np.linalg.norm(PL) < 1e-5


def test_gamma_std_vs_gam_full_pipeline(runs_dir):
    a = _run(_need(runs_dir, "si8_gamma_std"), "paw_full", max_distance=2.4).result
    b = _run(_need(runs_dir, "si8_gamma_gam"), "paw_full", max_distance=2.4).result
    assert len(a.bonds) == 16
    assert np.ptp(a.icohp) < 1e-6  # symmetry-equivalent bonds identical
    assert np.abs(a.icohp - b.icohp).max() < 1e-6
    assert np.abs(a.cohp - b.cohp).max() < 1e-5 * np.abs(a.cohp).max()
    assert np.abs(a.lowdin_charge - b.lowdin_charge).max() < 1e-6


def test_si_auto_window_is_repaired(runs_dir):
    res = _run(_need(runs_dir, "si_diamond_k6"), "paw_full", [1, 2, [0, 0, 0]])
    d = res.result.diagnostics
    assert d["band_window_note"].startswith("auto -> repaired") and "at 12 with up to 1 extra" in d["band_window_note"]
    assert d["electrons_projected_lowdin"] == pytest.approx(8.0, abs=1e-8)
    assert d["max_reconstruction_residual_ev"] < 1e-8


@pytest.mark.skipif("cuda" not in _core.available_backends(), reason="no CUDA device/build")
def test_cpu_cuda_parity(runs_dir):
    for name in ("gaas_k6", "si8_gamma_gam"):
        d = _need(runs_dir, name)
        a = _run(d, "paw_full", max_distance=2.5, backend="cpu").result
        b = _run(d, "paw_full", max_distance=2.5, backend="cuda").result
        assert np.abs(a.icohp - b.icohp).max() < 1e-10
        assert np.abs(a.cohp - b.cohp).max() < 1e-9 * max(1.0, np.abs(a.cohp).max())


def test_spinor_rejected_by_inspection(runs_dir):
    rep = Calculation.open(_need(runs_dir, "si2_ncl_gamma")).inspect()
    assert not rep.ok
    assert any(i.code == "unsupported_spinor" for i in rep.issues)


@pytest.mark.parametrize("name,budget", [("si216_gamma_gam", 0.1), ("fe_bcc_k10_spin", 1e-4)])
def test_band_block_streaming_is_exact(runs_dir, name, budget):
    """Gate E: a small memory budget (band blocks + uncached factor column blocks) reproduces the
    all-at-once result."""
    d = _need(runs_dir, name)
    full = _run(d, "paw_full", max_distance=2.5, backend="auto").result
    small = _run(d, "paw_full", max_distance=2.5, backend="auto", memory_budget_gb=budget).result
    assert small.diagnostics["band_block_size"] < full.diagnostics["band_block_size"]
    assert np.abs(full.icohp - small.icohp).max() < 1e-12
    assert np.abs(full.pdos_integral - small.pdos_integral).max() < 1e-12


def test_spin_polarised_metal(runs_dir):
    """bcc Fe, ISPIN=2: exact spin moment and electron count, symmetric bonds (LOBSTER's window is singular)."""
    d = _need(runs_dir, "fe_bcc_k10_spin")
    r = _run(d, "paw_onsite", min_distance=2.3, max_distance=2.5).result
    assert len(r.bonds) == 4
    assert np.ptp(r.icohp, axis=1).max() < 1e-6  # single-precision WAVECAR noise
    moment = r.lowdin_gp[0].sum() - r.lowdin_gp[1].sum()
    assert moment == pytest.approx(2.1345, abs=2e-3)  # VASP OSZICAR mag= 2.1345
    assert r.diagnostics["electrons_projected_lowdin"] == pytest.approx(8.0, abs=1e-6)
    assert r.icohp[1, 0] < r.icohp[0, 0] < 0  # minority channel more bonding


def test_isym0_time_reversal_unfolding_matches_full_mesh(runs_dir):
    a = _run(_need(runs_dir, "c_diamond_k7"), "paw_full", max_distance=2.6).result
    b = _run(_need(runs_dir, "c_diamond_k7_isym0"), "paw_full", max_distance=2.6).result
    assert len(a.bonds) == len(b.bonds) == 16
    assert np.abs(a.icohp - b.icohp).max() < 2e-5  # separate SCF runs, single-precision WAVECARs
    assert np.abs(a.lowdin_gp - b.lowdin_gp).max() < 1e-6


@pytest.mark.parametrize("name", ["gaas_k6", "fe_bcc_k10_spin", "si64_gamma_gam"])
def test_cpp_and_numpy_engines_agree(runs_dir, name):
    """The C++ reconstruction/analysis core equals the independent NumPy reference."""
    from prowl import analysis as an

    d = _need(runs_dir, name)
    res = _run(d, "paw_full", max_distance=2.5)
    inp, bonds = res.input, res.result.bonds
    grid = an.EnergyGrid(-20, 10, 3001, 0.05, res.input.efermi)
    a = an.analyze(inp, bonds, grid, engine="cpp")
    b = an.analyze(inp, bonds, grid, engine="numpy")
    for key in ("icohp", "icoop", "cohp", "coop", "pdos", "mulliken_gp", "lowdin_gp"):
        x, y = getattr(a, key), getattr(b, key)
        assert np.abs(x - y).max() <= 1e-10 * max(1.0, np.abs(y).max()), key
    for x, y in zip(a.icohp_orbital, b.icohp_orbital):
        assert np.abs(x - y).max() < 1e-10
    assert a.diagnostics["abs_charge_spilling"] == pytest.approx(b.diagnostics["abs_charge_spilling"], abs=1e-12)


def test_transition_metal_compound_fes2(runs_dir):
    """FeS2 pyrite (Fe d6): symmetric bonds/charges, exact electron count, spilling equals LOBSTER's
    (one-centre PAW). LOBSTER's own window is singular at 11/64 k-points here."""
    d = _need(runs_dir, "fes2_pyrite_k4")
    r = _run(d, "paw_onsite", min_distance=2.0, max_distance=2.35).result
    el = ["Fe"] * 4 + ["S"] * 8
    fes = [i for i, b in enumerate(r.bonds) if {el[b.atom_a], el[b.atom_b]} == {"Fe", "S"}]
    ss = [i for i, b in enumerate(r.bonds) if el[b.atom_a] == el[b.atom_b] == "S"]
    assert len(fes) == 24 and len(ss) == 4
    assert np.ptp(r.icohp[0, fes]) < 1e-4 and np.ptp(r.icohp[0, ss]) < 1e-4
    # repaired window (default): close to LOBSTER's own -2.215...-2.219 / -3.964...-3.972 eV
    assert r.icohp[0, fes].mean() == pytest.approx(-2.24110, abs=1e-3)
    assert r.icohp[0, ss].mean() == pytest.approx(-3.98576, abs=1e-3)
    assert 100 * r.diagnostics["abs_charge_spilling"] == pytest.approx(2.69, abs=0.01)
    assert r.diagnostics["electrons_projected_lowdin"] == pytest.approx(80.0, abs=1e-6)
    assert np.ptp(r.mulliken_charge[:4]) < 1e-4 and r.mulliken_charge[0] == pytest.approx(0.4405, abs=2e-3)  # LOBSTER 0.42-0.43


@pytest.mark.skipif("cuda" not in _core.available_backends(), reason="no CUDA device/build")
@pytest.mark.parametrize("name", ["gaas_k6", "si64_gamma_gam"])
def test_device_factor_generation_matches_host(runs_dir, name):
    """Factor matrices generated by the CUDA kernel equal the host evaluation element-wise."""
    d = _need(runs_dir, name)
    st, pcs = _core.read_poscar(str(d / "POSCAR")), _core.read_potcar(str(d / "POTCAR"))
    from prowl import load_basis_library
    from prowl.config import zval_valence_shells
    lib = load_basis_library()
    sel = {sp: zval_valence_shells(lib, sp, pc.zval) for sp, pc in zip(st.species, pcs)}
    m = _core.ProjectionModel(st, _core.AtomicBasis(st, lib, sel), pcs)
    rd = _core.WavecarReader(str(d / "WAVECAR"))
    cpu, gpu = _core.make_backend("cpu"), _core.make_backend("cuda")
    for k in (0, rd.header["nkpts"] - 1):
        for proj in (False, True):
            a = m.factors_on_backend(rd, k, proj, cpu)
            b = m.factors_on_backend(rd, k, proj, gpu)
            assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max()


def test_lobster_format_export_roundtrip(runs_dir, tmp_path):
    from prowl.lobster_export import write_lobster_format
    from prowl.validation.lobster import read_cohpcar, read_icohplist

    res = _run(_need(runs_dir, "gaas_k6"), "paw_onsite", [1, 2, [0, 0, 0]])
    h5 = res.save(tmp_path / "bonding.h5")
    write_lobster_format(h5, tmp_path / "lob")
    il = read_icohplist(tmp_path / "lob" / "ICOHPLIST.lobster")
    assert il[0]["icohp"] == pytest.approx(-4.55365, abs=2e-5)  # LOBSTER's own value
    car = read_cohpcar(tmp_path / "lob" / "COHPCAR.lobster")
    assert car["n_interactions"] == 2 and car["data"].shape[0] == len(res.result.energies)
    pmg = pytest.importorskip("pymatgen.io.lobster")
    ic = pmg.Icohplist(filename=str(tmp_path / "lob" / "ICOHPLIST.lobster"))
    assert list(ic.icohplist.values())[0]["translation"] == (0, 0, 0)


def test_review_regressions(runs_dir, tmp_path):
    """Structure choice, unwrapped coordinates and concurrent WAVECAR reads (independent review)."""
    import shutil
    import threading

    d = _need(runs_dir, "c_diamond_k7")
    assert Calculation.open(d).paths["structure"].name == "CONTCAR"
    # unwrapped coordinates: shift atom 2 by lattice vectors -> identical bonding physics
    run = tmp_path / "shifted"
    run.mkdir()
    for f in ("WAVECAR", "POTCAR", "vasprun.xml"):
        (run / f).symlink_to(d / f)
    lines = (d / "POSCAR").read_text().splitlines()
    x = [float(v) for v in lines[9].split()[:3]]
    lines[9] = f"  {x[0] + 2.0:.10f} {x[1] - 3.0:.10f} {x[2] + 1.0:.10f}"
    (run / "POSCAR").write_text("\n".join(lines) + "\n")
    a = _run(d, "paw_full", max_distance=2.6).result
    b = _run(run, "paw_full", max_distance=2.6).result
    assert sorted(np.round(a.icohp[0], 6)) == sorted(np.round(b.icohp[0], 6))
    assert np.abs(a.lowdin_gp - b.lowdin_gp).max() < 1e-8
    # concurrent reads from one reader (bindings release the GIL)
    rd = _core.WavecarReader(str(d / "WAVECAR"))
    ref = {k: rd.read_block(0, k, 0, 8)["coefficients"] for k in range(40)}
    errs = []

    def work(ks):
        for k in ks:
            if not np.array_equal(rd.read_block(0, k, 0, 8)["coefficients"], ref[k]):
                errs.append(k)

    th = [threading.Thread(target=work, args=(list(range(i, 40, 4)) * 3,)) for i in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs


def test_inconsistent_wavecar_is_rejected(runs_dir, tmp_path):
    """Design: a WAVECAR that is not the converged state of this run (MD/continuation) must be rejected."""
    import re

    d = _need(runs_dir, "c_diamond_k7")
    for f in ("WAVECAR", "POTCAR", "POSCAR"):
        (tmp_path / f).symlink_to(d / f)
    xml = (d / "vasprun.xml").read_text()
    # shift the first eigenvalue of the last calculation block by 0.1 eV
    # the direct <eigenvalues> child of the last <calculation> (LORBIT adds another inside <projected>)
    i = xml.find('<set comment="kpoint 1">', xml.find("<eigenvalues>", xml.rfind("<calculation>")))
    m = re.compile(r"<r>\s*(-?\d+\.\d+)").search(xml, i)
    val = float(m.group(1)) + 0.1
    xml = xml[:m.start(1)] + f"{val:.4f}" + xml[m.end(1):]
    (tmp_path / "vasprun.xml").write_text(xml)
    rep = Calculation.open(tmp_path).inspect()
    assert not rep.ok and any(i.code == "eigenvalue_mismatch" for i in rep.issues)
    with pytest.raises(_core.ProwlError, match="InconsistentInput"):
        Calculation.open(tmp_path).analyze(BondingConfig())


def test_tetrahedron_spectra_vs_lobster(runs_dir):
    """Linear-tetrahedron COHP reproduces LOBSTER's tetrahedron COHPCAR (ISMEAR=-5 run)."""
    from prowl import analysis as an
    from prowl.validation.lobster import read_cohpcar

    d = _need(runs_dir, "c_diamond_k7")
    if not (d / "COHPCAR.lobster").is_file():
        pytest.skip("no LOBSTER reference")
    L = read_cohpcar(d / "COHPCAR.lobster")
    E = L["energies"]
    res = _run(d, "paw_onsite", [1, 2, [0, 0, 0]], band_window="norb")
    g = an.EnergyGrid(E[0], E[-1], len(E), 0.05, L["efermi"], kind="tetrahedron", bin_offset=-1.0)
    r = an.analyze(res.input, res.result.bonds, g)
    lob = L["data"][:, 2]
    assert np.linalg.norm(r.cohp[0, 0] - lob) / np.linalg.norm(lob) < 0.03
    dE = E[1] - E[0]
    assert r.cohp[0, 0][E <= 0].sum() * dE == pytest.approx(r.icohp[0, 0], abs=2e-3)
    with pytest.raises(_core.ProwlError, match="tetrahedron"):
        gam = _run(_need(runs_dir, "si8_gamma_gam"), "paw_full", max_distance=2.4)
        an.analyze(gam.input, gam.result.bonds, an.EnergyGrid(-5, 5, 101, 0.05, 0.0, kind="tetrahedron"))


def test_ionic_solid_mgo(runs_dir):
    """MgO (rocksalt): charges and spilling agree with LOBSTER (paw_onsite); symmetric shells; 8 e-."""
    r = _run(_need(runs_dir, "mgo_k6"), "paw_onsite", max_distance=2.2).result
    assert len(r.bonds) == 6 and np.ptp(r.icohp) < 1e-6
    assert r.icohp[0].mean() == pytest.approx(-0.97589, abs=1e-3)   # LOBSTER -0.90...-0.93 (asymmetric)
    assert r.mulliken_charge[0] == pytest.approx(1.55, abs=0.01)    # LOBSTER 1.55 (2 decimals, own window ill-conditioned)
    assert r.lowdin_charge[0] == pytest.approx(1.50, abs=0.006)     # LOBSTER 1.50
    assert 100 * r.diagnostics["abs_charge_spilling"] == pytest.approx(2.21, abs=0.01)
    assert r.diagnostics["electrons_projected_lowdin"] == pytest.approx(8.0, abs=1e-6)


def test_projection_store_and_verified_reuse(runs_dir, tmp_path):
    d = _need(runs_dir, "gaas_k6")
    cfg = BondingConfig()
    cfg.write_projection = True
    first = Calculation.open(d).analyze(cfg)
    h5 = first.save(tmp_path / "bonding.h5")
    cfg2 = BondingConfig()
    cfg2.max_distance = 4.1                     # different analysis, same projection
    again = Calculation.open(d).analyze(cfg2, reuse=h5).result
    full = Calculation.open(d).analyze(cfg2).result
    assert np.abs(again.icohp - full.icohp).max() < 1e-12
    assert any("reused" in w for w in again.diagnostics["warnings"])
    cfg3 = BondingConfig()
    cfg3.projection = "paw_full"
    with pytest.raises(_core.ProwlError, match="differing inputs: projection"):
        Calculation.open(d).analyze(cfg3, reuse=h5)
    plain = Calculation.open(d).analyze(BondingConfig()).save(tmp_path / "plain.h5")
    with pytest.raises(_core.ProwlError, match="no stored projection"):
        Calculation.open(d).analyze(BondingConfig(), reuse=plain)


def test_dry_run_plan(runs_dir):
    plan = Calculation.open(_need(runs_dir, "si216_gamma_gam")).plan(BondingConfig())
    assert plan["inspection_ok"] and plan["sizes"]["N_orb"] == 864 and plan["sizes"]["encoding"] == "GammaHalf"
    small = BondingConfig()
    small.memory_budget_gb = 0.1
    p2 = Calculation.open(_need(runs_dir, "si216_gamma_gam")).plan(small)
    assert p2["streaming"]["band_blocks_per_k"] > 1 and p2["estimated_peak_GiB"] < plan["estimated_peak_GiB"]


def test_real_spanning_header_wavecar(runs_dir):
    """VASP 6 RTAG 53300 file whose k-point headers span several records."""
    d = _need(runs_dir, "si2_spanning_header")
    h = _core.WavecarReader(str(d / "WAVECAR")).header
    assert h["rtag"] == 53300 and h["header_spans_records"]
    r = _run(d, "paw_full", max_distance=2.4, band_window="frozen").result
    assert r.diagnostics["max_paw_norm_deviation"] < 1e-6
    assert r.diagnostics["electrons_projected_lowdin"] == pytest.approx(8.0, abs=1e-6)


@pytest.mark.parametrize("system,lobster_icobi", [("c_diamond_k7", 0.95035), ("gaas_k6", 0.83923)])
def test_icobi_equals_lobster(runs_dir, system, lobster_icobi):
    r = _run(_need(runs_dir, system), "paw_onsite", [1, 2, [0, 0, 0]], band_window="norb").result
    assert r.icobi[0, 0] == pytest.approx(lobster_icobi, abs=2e-5)


def test_second_review_regressions(runs_dir, tmp_path):
    import json
    import shutil

    import h5py

    from prowl import default_basis_path
    from prowl.lobster_export import write_lobster_format
    from prowl.validation.lobster import read_cohpcar

    d = _need(runs_dir, "gaas_k6")
    # (1) absolute energy reference: LOBSTER files must still be E_F-referenced
    cfg = BondingConfig()
    cfg.reference = "absolute"
    cfg.emin, cfg.emax, cfg.npoints = -10.0, 15.0, 501
    res = Calculation.open(d).analyze(cfg)
    h5 = res.save(tmp_path / "abs.h5")
    write_lobster_format(h5, tmp_path / "lob")
    car = read_cohpcar(tmp_path / "lob" / "COHPCAR.lobster")
    assert car["energies"][0] == pytest.approx(-10.0 - res.input.efermi, abs=1e-4)
    # (6) files without the COBI group still export
    old = tmp_path / "old.h5"
    shutil.copy(h5, old)
    with h5py.File(old, "a") as f:
        del f["analyses/cobi"]
    write_lobster_format(old, tmp_path / "lob_old")
    assert not (tmp_path / "lob_old" / "ICOBILIST.lobster").exists()
    # (2) bands superset accepted; basis-library content change refused
    cw = BondingConfig()
    cw.write_projection = True
    cw.band_window = "all"
    store = Calculation.open(d).analyze(cw).save(tmp_path / "store.h5")
    cn = BondingConfig()
    cn.band_window = "norb"
    r = Calculation.open(d).analyze(cn, reuse=store)
    assert r.result.icohp[0, 0] == pytest.approx(Calculation.open(d).analyze(cn).result.icohp[0, 0], abs=1e-12)
    # (3) reusing keeps the restart data in the new output
    out = r.save(tmp_path / "again.h5")
    with h5py.File(out, "r") as f:
        assert "projection" in f
    lib = tmp_path / "lib.json"
    doc = json.loads(default_basis_path().read_text())
    lib.write_text(json.dumps(doc))
    ca = BondingConfig()
    ca.basis_library, ca.write_projection = str(lib), True
    stored = Calculation.open(d).analyze(ca).save(tmp_path / "lib.h5")
    doc["elements"]["As"]["shells"][-1]["primitives"][0]["coef"] *= 1.001
    lib.write_text(json.dumps(doc))
    with pytest.raises(_core.ProwlError, match="basis_library_sha256"):
        Calculation.open(d).analyze(ca, reuse=stored)


def test_profiles_on_real_data(runs_dir, tmp_path):
    """'prowl' (default): LOBSTER's window where well conditioned, frozen fallback with an NBANDS warning,
    frozen value reported alongside; 'lobster': always LOBSTER's window, clear error where singular."""
    import csv

    from prowl.io import export

    d = _need(runs_dir, "c_diamond_k7")
    cfg = BondingConfig()
    cfg.bond_mode, cfg.bond_pairs = "pairs", [[1, 2, [0, 0, 0]]]
    res = Calculation.open(d).analyze(cfg)
    r, ws = res.result, res.result.diagnostics["window_sensitivity"]
    assert r.diagnostics["band_window_note"].startswith("auto -> norb")
    assert r.icohp[0, 0] == pytest.approx(-9.62564, abs=2e-5)                          # LOBSTER's value
    assert ws["available"] and ws["window"] == "frozen" and ws["nbands"] == 16
    assert np.asarray(ws["icohp"])[0, 0] == pytest.approx(-10.340, abs=1e-3)
    assert "depends on NBANDS (here 16)" in res.quality_report()
    files = export(res.save(tmp_path / "b.h5"), tmp_path / "csv")
    row = list(csv.DictReader(open([f for f in files if f.name == "icohp.csv"][0])))[0]
    assert float(row["ICOHP_frozen_window_spin1_eV"]) == pytest.approx(-10.340, abs=1e-3)

    lob = BondingConfig.for_profile("lobster")
    lob.bond_mode, lob.bond_pairs = "pairs", [[1, 2, [0, 0, 0]]]
    rl = Calculation.open(d).analyze(lob).result
    assert rl.icohp[0, 0] == pytest.approx(-9.62564, abs=2e-5)
    assert not rl.diagnostics["warnings"]

    si = _need(runs_dir, "si_diamond_k6")
    rs = Calculation.open(si).analyze(BondingConfig()).result
    assert any("repaired with the fewest extra bands" in w for w in rs.diagnostics["warnings"])
    assert rs.diagnostics["window_sensitivity"]["window"] == "frozen"
    with pytest.raises(_core.ProwlError, match="singular for this system"):
        Calculation.open(si).analyze(BondingConfig.for_profile("lobster"))

    few = BondingConfig()
    few.band_window, few.unconverged_band_fraction = "frozen", 0.9
    with pytest.raises(_core.ProwlError, match="profile 'lobster'"):
        Calculation.open(d).analyze(few)
    full = BondingConfig()
    full.projection = "paw_full"
    assert any("experimental" in w for w in Calculation.open(d).analyze(full).result.diagnostics["warnings"])


def test_too_few_bands_like_lobster(runs_dir):
    """NBANDS < N_orb: LOBSTER disables COHP ('you need to use at least 8 bands'); PROWL stops with
    the same requirement, and the planner reports it."""
    d = _need(runs_dir, "c_diamond_k7_nb6")
    plan = Calculation.open(d).plan(BondingConfig())
    assert plan["required_nbands"]["minimum"] == 8 and plan["sizes"]["N_band"] == 6
    assert plan["recommendations"][0].startswith("too few bands")
    with pytest.raises(_core.ProwlError, match="needs at least N_orb = 8 bands"):
        Calculation.open(d).analyze(BondingConfig())
    lob = d / "lobsterout"
    if lob.is_file():
        assert "you need to use at least" in lob.read_text() and "8 bands" in lob.read_text()


@pytest.mark.parametrize("pair,pol,expected", [
    (("mgo_k6", "mgo_k6_nb40"), "none", -0.97589),
    (("fe_bcc_k10_spin", "fe_bcc_k10_spin_nb40"), "none", -0.78741),
    (("fe_bcc_k10_spin", "fe_bcc_k10_spin_nb40"), "all", -2.51689),
])
def test_repaired_window_is_nbands_independent(runs_dir, pair, pol, expected):
    """Default window (LOBSTER's, minimally repaired): identical results for different NBANDS, with and
    without polarization functions; the frozen window differs (docs/results/nbands_and_polarization_study.md)."""
    vals, frozen = [], []
    for name in pair:
        d = _need(runs_dir, name)
        cfg = BondingConfig()
        cfg.basis_polarization, cfg.min_distance, cfg.max_distance = pol, 2.0, 2.5
        cfg.window_sensitivity = True
        r = Calculation.open(d).analyze(cfg).result
        ic = r.icohp.sum(axis=0)
        assert np.ptp(ic) < 1e-5 and r.diagnostics["band_window_note"].startswith("auto -> repaired")
        vals.append(ic.mean())
        ws = r.diagnostics["window_sensitivity"]
        frozen.append(np.asarray(ws["icohp"]).sum(axis=0).mean() if ws["available"] else np.nan)
    assert vals[0] == pytest.approx(vals[1], abs=1e-6) and vals[0] == pytest.approx(expected, abs=1e-4)
    if not np.isnan(frozen).any():
        assert abs(frozen[0] - frozen[1]) > 0.02
