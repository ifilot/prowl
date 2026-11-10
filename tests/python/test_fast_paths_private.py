"""The batched / parallel fast paths give the same numbers as the reference single-k paths (private fixtures)."""
import os
from pathlib import Path

import numpy as np
import pytest

from prowl import BondingConfig, _core

RUNS = os.environ.get("PROWL_TEST_RUNS")
pytestmark = pytest.mark.skipif(not RUNS, reason="set PROWL_TEST_RUNS")


def _model(run, selection, offsite=True):
    from prowl.calculation import Calculation

    c = Calculation.open(str(Path(RUNS) / run))
    st = _core.read_poscar(str(c.paths["structure"]))
    pcs = _core.read_potcar(str(c.paths["potcar"]))
    cfg = BondingConfig()
    cfg.basis_source, cfg.basis_selection = "paw-pbe-sto-1", selection
    lib = cfg.load_library()
    basis = _core.AtomicBasis(st, lib, cfg.resolve_basis(st, pcs, lib))
    opt = _core.ProjectionOptions()
    opt.include_offsite_paw = offsite
    return _core.ProjectionModel(st, basis, pcs, opt), _core.WavecarReader(str(c.paths["wavecar"]))


@pytest.mark.parametrize("run,sel", [("fe_bcc_k10_spin", {"Fe": ["4s", "3d", "4p"]}),
                                     ("fes2_pyrite_k4", {"Fe": ["4s", "3d", "4p"], "S": ["3s", "3p"]})])
def test_batched_overlaps_and_projection(run, sel):
    if not (Path(RUNS) / run / "WAVECAR").exists():
        pytest.skip(f"{run} not available")
    model, reader = _model(run, sel)
    nk = reader.header["nkpts"]
    ks = [list(reader.kpoint(0, k)["k_frac"]) for k in range(nk)]
    S = model.overlaps(ks)                       # mesh-folded, OpenMP over k
    for k in (0, 1, nk // 2, nk - 1):
        assert np.max(np.abs(S[k] - model.overlap(ks[k]))) < 1e-12
    nb = model.norb + 4
    pick = [0, 3, nk - 1]
    many = model.project_many(reader, pick, 0, nb)
    be = _core.make_backend("cpu")
    for k, per in zip(pick, many):
        for s, p in enumerate(model.project_k(reader, k, 0, nb, be)):
            assert np.max(np.abs(np.asarray(per[s]["a_ae"]) - np.asarray(p["a_ae"]))) < 1e-12
            assert np.max(np.abs(np.asarray(per[s]["paw_norm"]) - np.asarray(p["paw_norm"]))) < 1e-12


@pytest.mark.parametrize("run,sel", [("fes2_pyrite_k4", {"Fe": ["4s", "3d", "4p"], "S": ["3s", "3p"]}),
                                     ("si8_gamma_gam", {"Si": ["3s", "3p"]})])
def test_folded_projection(run, sel):
    """PAW fold (paw_check_stride > 1): same a_ae as the full projection; b and the PAW norm of the sampled bands."""
    if not (Path(RUNS) / run / "WAVECAR").exists():
        pytest.skip(f"{run} not available")
    model, reader = _model(run, sel, offsite=False)
    assert model.foldable
    nk = reader.header["nkpts"]
    nb = model.norb + 13
    idx = list(range(0, nb, 8)) + ([nb - 1] if (nb - 1) % 8 else [])
    backends = ["cpu"] + (["cuda"] if "cuda" in _core.available_backends() else [])
    for name in backends:
        be = _core.make_backend(name)
        for k in sorted({0, nk - 1}):
            # block_bands 7: the samples fall into several blocks, some blocks hold none
            for bb in (0, 7):
                full = model.project_k(reader, k, 0, nb, be, bb)
                fold = model.project_k(reader, k, 0, nb, be, bb, paw_check_stride=8)
                for f, g in zip(full, fold):
                    assert list(g["b_bands"]) == idx and list(f["b_bands"]) == list(range(nb))
                    assert g["a_soft"] is None and np.asarray(f["a_soft"]).shape == (model.norb, nb)
                    assert np.max(np.abs(np.asarray(f["a_ae"]) - np.asarray(g["a_ae"]))) < 1e-12
                    assert np.max(np.abs(np.asarray(f["b"])[:, idx] - np.asarray(g["b"]))) < 1e-12
                    assert np.max(np.abs(np.asarray(f["paw_norm"])[idx] - np.asarray(g["paw_norm"]))) < 1e-12
    if nk > 1:  # k-parallel CPU path
        for per_full, per_fold in zip(model.project_many(reader, [0, nk - 1], 0, nb),
                                      model.project_many(reader, [0, nk - 1], 0, nb, 0, 2147483648.0, 8)):
            for f, g in zip(per_full, per_fold):
                assert np.max(np.abs(np.asarray(f["a_ae"]) - np.asarray(g["a_ae"]))) < 1e-12
                assert list(g["b_bands"]) == idx and len(g["paw_norm"]) == len(idx)


def test_fold_requires_one_centre_paw():
    """With two-centre PAW terms the coupling is not separable: the stride is ignored (full projection)."""
    run = "si8_gamma_gam"
    if not (Path(RUNS) / run / "WAVECAR").exists():
        pytest.skip(f"{run} not available")
    from prowl.calculation import Calculation

    c = Calculation.open(str(Path(RUNS) / run))
    st = _core.read_poscar(str(c.paths["structure"]))
    pcs = _core.read_potcar(str(c.paths["potcar"]))
    cfg = BondingConfig()
    cfg.basis_source, cfg.basis_selection = "paw-pbe-sto-1", {"Si": ["3s", "3p"]}
    lib = cfg.load_library()
    opt = _core.ProjectionOptions()
    opt.include_offsite_paw = True
    model = _core.ProjectionModel(st, _core.AtomicBasis(st, lib, cfg.resolve_basis(st, pcs, lib)), pcs, opt)
    assert not model.foldable
    reader = _core.WavecarReader(str(c.paths["wavecar"]))
    p = model.project_k(reader, 0, 0, model.norb, _core.make_backend("cpu"), 0, 2147483648.0, 8)[0]
    assert list(p["b_bands"]) == list(range(model.norb)) and len(p["paw_norm"]) == model.norb and p["a_soft"] is not None


def test_paw_norm_check_config():
    assert BondingConfig.from_dict({}).paw_norm_check == "sampled"
    assert BondingConfig.from_dict({"numerical": {"paw_norm_check": "full"}}).paw_norm_check == "full"
    with pytest.raises(_core.ProwlError):
        BondingConfig.from_dict({"numerical": {"paw_norm_check": "some"}})


def _arrays(res):
    """Every numeric output of an analysis (AnalysisResult arrays and orbital-resolved ICOHP)."""
    import dataclasses

    out = {}
    for f in dataclasses.fields(res):
        v = getattr(res, f.name)
        if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
            out[f.name] = v
    for i, v in enumerate(res.icohp_orbital):
        out[f"icohp_orbital[{i}]"] = np.asarray(v)
    return out


@pytest.mark.parametrize("run", ["si8_gamma_gam", "fes2_pyrite_k4"])
def test_paw_norm_escalation(run):
    """Sampled max |PAW norm - 1| above tolerance/2 rechecks every band with the full projection: same maximum and
    the same outputs as paw_norm_check: full; without escalation the sampled result agrees to rounding."""
    from prowl.calculation import Calculation

    d = Path(RUNS) / run
    if not (d / "WAVECAR").exists():
        pytest.skip(f"{run} not available")

    def run_with(check, tol):
        cfg = BondingConfig()
        cfg.paw_norm_check, cfg.paw_norm_tolerance, cfg.strict = check, tol, False
        return Calculation.open(str(d)).analyze(cfg, backend="cpu")

    sampled = run_with("sampled", 1e-4)
    ds = sampled.result.diagnostics
    assert not any(k.endswith("_count") for k in ds["timings_s"])
    assert ds["setup_counts"]["overlap_terms_count"] > 0
    assert ds["paw_norm_check"] == "sampled" and not ds["paw_norm_escalated"]
    assert 0 < ds["paw_norm_bands_checked"] < ds["paw_norm_bands_total"]
    assert "PAW norm checked on" in sampled.quality_report()

    tol = 1e-15  # below the sampled maximum: forces the escalation (strict off, the violation is a warning)
    esc, full = run_with("sampled", tol), run_with("full", tol)
    de, df = esc.result.diagnostics, full.result.diagnostics
    assert de["paw_norm_check"] == "sampled_escalated" and de["paw_norm_escalated"]
    assert de["paw_norm_sampled_max"] > tol / 2
    assert de["paw_norm_bands_checked"] == de["paw_norm_bands_total"] == df["paw_norm_bands_checked"]
    assert df["paw_norm_check"] == "full" and not df["paw_norm_escalated"]
    assert de["max_paw_norm_deviation"] == df["max_paw_norm_deviation"]
    assert de["max_paw_norm_deviation"] >= de["paw_norm_sampled_max"]
    assert "rechecked all" in esc.quality_report()
    a, b, c = _arrays(esc.result), _arrays(full.result), _arrays(sampled.result)
    assert a.keys() == b.keys() == c.keys()
    for k in a:
        # escalated = full projection; the k-parallel analysis itself repeats only to ~1e-15 (FeS2: two full runs differ
        # by 1.8e-15), so "identical" is checked to 1e-12
        assert np.allclose(a[k], b[k], rtol=0, atol=1e-12), k
        assert np.allclose(a[k], c[k], rtol=0, atol=1e-10), k


@pytest.mark.parametrize("run,selection", [("si8_gamma_gam", {"Si": ["3s", "3p"]}),
                                          ("fe_bcc_k10_spin", {"Fe": ["4s", "3d"]})])
def test_aggregate_factor_memory_budget(run, selection):
    if not (Path(RUNS) / run / "WAVECAR").exists():
        pytest.skip(f"{run} not available")
    model, reader = _model(run, selection, offsite=False)
    pick = [0, reader.header["nkpts"] - 1] * 2
    max_ng = max(reader.kpoint(0, k)["nplw"] for k in pick)
    copies = 2 if reader.layout["encoding"] == "GammaHalf" else 1
    column = max_ng * 16 * copies
    budget = 2 * column
    workers = min(len(pick), _core.omp_max_threads(), 2)
    many = model.project_many(reader, pick, 0, 4, 1, budget, 2)
    for k, per in zip(pick, many):
        full = model.project_k(reader, k, 0, 4, _core.make_backend("cpu"), paw_check_stride=2)
        for got, ref in zip(per, full):
            assert 0 < got["factor_peak_bytes"] <= budget // workers
            np.testing.assert_allclose(got["a_ae"], ref["a_ae"], rtol=0, atol=1e-12)
    with pytest.raises(_core.ProwlError, match="OutOfMemory"):
        model.project_k(reader, 0, 0, 4, _core.make_backend("cpu"), factor_budget_bytes=1)
    for invalid in (-1.0, np.nan, np.inf):
        with pytest.raises(_core.ProwlError, match="InvalidArgument"):
            model.project_many(reader, pick, 0, 4, factor_budget_bytes=invalid)


def test_concurrent_models_coordinate_blas_policy():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    run, selection = "fe_bcc_k10_spin", {"Fe": ["4s", "3d"]}
    if not (Path(RUNS) / run / "WAVECAR").exists():
        pytest.skip(f"{run} not available")
    a, reader = _model(run, selection, offsite=False)
    b, _ = _model(run, selection, offsite=False)
    ks = [0, reader.header["nkpts"] - 1]
    reference = a.project_many(reader, ks, 0, 4, 1, paw_check_stride=2)
    ready = Barrier(2)

    def parallel_k():
        ready.wait()
        return a.project_many(reader, ks, 0, 4, 1, paw_check_stride=2)

    def single_k():
        ready.wait()
        backend = _core.make_backend("cpu")
        return [b.project_k(reader, k, 0, 4, backend, 1, paw_check_stride=2) for k in ks]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(parallel_k), pool.submit(single_k)]
        for future in futures:
            for per, ref in zip(future.result(timeout=20), reference):
                for got, expected in zip(per, ref):
                    np.testing.assert_allclose(got["a_ae"], expected["a_ae"], atol=1e-12, rtol=0)
