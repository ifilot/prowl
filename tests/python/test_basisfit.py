"""tools/basisfit: the parts of the basis-fitting method that need no POTCAR or VASP run.

The fits themselves are validated against VASP free atoms in docs/BASIS_FIT_PLAN.md (M2-M6); here: the
all-electron atom against NIST LDA data, STO algebra, the frozen selection rules and the .sto writer.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "basisfit"))

from pbefit import library  # noqa: E402
from pbefit.aeatom import Grid, solve_atom  # noqa: E402
from pbefit.fit import sto_S  # noqa: E402
from pbefit.shells import PMG_STANDARD, koga  # noqa: E402


@pytest.mark.parametrize("Z,cfg,ref", [
    # NIST atomic reference data, LDA, non-relativistic, spin-unpolarized (Hartree)
    (1, {"1s": 1}, {"1s": -0.233471}),
    (10, {"1s": 2, "2s": 2, "2p": 6}, {"1s": -30.305855, "2s": -1.322548, "2p": -0.498034}),
])
def test_ae_atom_lda_eigenvalues_vs_nist(Z, cfg, ref):
    a = solve_atom(Z, cfg, "lda", rel=False, grid=Grid())
    assert a.converged
    for lab, e in ref.items():
        assert a.energies[lab] == pytest.approx(e, abs=3e-4)


def test_sto_overlap_and_rmean():
    assert sto_S([(1, 1.7)])[0, 0] == pytest.approx(1.0)
    # 1s STO: <r> = 3 / (2 zeta)
    sh = {"primitives": [{"n": 1, "zeta": 1.5, "coef": 1.0}]}
    assert library.r_mean(sh) == pytest.approx(1.0)
    two = {"primitives": [{"n": 2, "zeta": 1.0, "coef": 0.6}, {"n": 2, "zeta": 0.4, "coef": 0.5}]}
    assert library.overlap(two, two) == pytest.approx(1.0)


def _fit(spill, passed=True):
    return {"orbital_spilling": spill, "passed": passed}


def test_auto_m_fewest_extras_within_tolerance():
    f = {"0": _fit(5e-3), "1": _fit(1.5e-4), "2": _fit(1e-4), "3": _fit(0.9e-4)}
    assert library.auto_m(f) == "1"          # within 1e-4 of the best (0.9e-4)
    f = {"0": _fit(5e-3), "1": _fit(3e-4), "2": _fit(-2e-4), "3": _fit(-3e-4)}
    assert library.auto_m(f) == "2"          # negative spilling counts as 0
    f = {"0": _fit(1e-5, passed=False), "1": _fit(2e-4)}
    assert library.auto_m(f) == "1"          # failing fits are never chosen


def _report(el, shells):
    return {"element": el, "potcar": el, "fits": {t: {} for t in shells}}


@pytest.mark.parametrize("el,fitted,kept", [
    ("Fe", ["3d", "4s", "4p"], ["3d", "4s", "4p"]),      # d metal: occupied shells refitted, 4p added
    ("Cu", ["3d", "4s", "4p"], ["4s", "4p"]),            # filled 3d10 keeps Koga
    ("O", ["2s", "2p"], []),                             # p block keeps Koga
    ("Ga", ["4s", "4p", "3d"], []),                      # p block incl. its filled d
    ("Pb", ["6s", "6p", "5d"], ["6s", "6p"]),            # 6p block refitted (relativity), 5d10 keeps Koga
    ("Na", ["3s", "3p"], ["3s", "3p"]),                  # s metal, 3p added
])
def test_rule_2026(el, fitted, kept):
    out = library.rule_2026(_report(el, fitted))
    assert sorted(out["fits"]) == sorted(kept)


def test_write_sto_koga_entry(tmp_path):
    ent = koga()["Fe"]
    p = tmp_path / "Fe.sto"
    library.write_sto(ent, "Fe", p)
    lines = p.read_text().splitlines()
    assert lines[0] == "fe, Z=26"
    norb = list(map(int, lines[2].split()[1:]))
    nprim = list(map(int, lines[3].split()[1:]))
    assert norb == [len([s for s in ent["shells"] if s["l"] == l]) for l in range(4)]
    assert len(lines) == 4 + sum(nprim)


def test_frozen_pymatgen_list():
    d = json.loads(PMG_STANDARD.read_text())
    assert "MIT" in d["license"] and (PMG_STANDARD.parent / "LICENSE.pymatgen").exists()
    assert d["shells"]["Fe_pv"] == ["3d", "3p", "4s"]       # 4p is added by our rule (shells.polarization_p)
    assert all(isinstance(v, list) and v for v in d["shells"].values())


def test_merge_variants_base_and_semicore():
    def rep(pot, shells):
        return {"element": pot.split("_")[0], "potcar": pot, "fits": {t: {"src": pot} for t in shells}}
    reps = {"Fe_sv": rep("Fe_sv", ["3s", "3p", "3d", "4s"]), "Fe_pv": rep("Fe_pv", ["3p", "3d", "4s", "4p"]),
            "Fe": rep("Fe", ["3d", "4s", "4p"]), "Fe_GW": rep("Fe_GW", ["3d", "4s", "5s"])}
    out = library.merge_variants(reps)
    assert out["potcar"] == "Fe" and out["variants"] == ["Fe", "Fe_pv", "Fe_sv"]
    assert {t: f["_potcar"] for t, f in out["fits"].items()} == \
        {"3d": "Fe", "4s": "Fe", "4p": "Fe", "3p": "Fe_pv", "3s": "Fe_sv"}       # GW variants are not used
    assert all("titel" in f["_atom"] for f in out["fits"].values())                # per-shell atom provenance
    out = library.merge_variants({"K_sv": rep("K_sv", ["3s"]), "K_pv": rep("K_pv", ["3p"])})
    assert out["potcar"] == "K_pv"                                             # no plain K: _pv is the base


def test_atom_provenance(tmp_path):
    (tmp_path / "pbefit_status.txt").write_text("converged=True setting=B seconds=12\n")
    (tmp_path / "OUTCAR").write_text(" vasp.6.5.1 10Mar25  (build X) gamma-only   \n more\n")
    (tmp_path / "POTCAR").write_text("  PAW_PBE Fe_pv 02Aug2007\n   SHA256 =  ab12cd Fe_pv/POTCAR\n")
    rep = {"potcar": "Fe_pv", "potcar_info": {"titel": "PAW_PBE Fe_pv 02Aug2007"}, "run": str(tmp_path),
           "half_box_bohr": 12.0 / library.BOHR_A}
    assert library.atom_provenance(rep) == {"titel": "PAW_PBE Fe_pv 02Aug2007", "box_angstrom": 24.0,
                                            "scf_setting": "B", "vasp": "vasp.6.5.1 10Mar25 (build X) gamma-only",
                                            "potcar_sha256": "ab12cd"}
    assert library.atom_provenance({"potcar": "Fe"}) == {"titel": ""}       # run directory not available


def test_auto_m_coefficient_cap():
    def fp(spill, c):
        return {"orbital_spilling": spill, "passed": True, "checks": {},
                "primitives": [{"n": 1, "zeta": 1.0, "coef": c}, {"n": 1, "zeta": 0.5, "coef": -0.5 * c}]}
    f = {"2": fp(1.5e-4, 1.0), "3": fp(1.4e-6, 19.0)}         # Ti 4s: 3 extras with cancelling coefficients
    assert library.auto_m(f) == "2"
    assert library.fit_checks(f["3"]) == {"coef_ok": False}


def test_merge_variants_lanthanide_5d_from_f_in_core():
    def rep(pot, shells):
        return {"element": pot.split("_")[0], "potcar": pot, "fits": {t: {"src": pot} for t in shells}}
    out = library.merge_variants({"Gd": rep("Gd", ["4f", "5d", "6s", "6p"]), "Gd_3": rep("Gd_3", ["5d", "6s"])})
    assert {t: f["_potcar"] for t, f in out["fits"].items()} == {"4f": "Gd", "5d": "Gd_3", "6s": "Gd", "6p": "Gd"}
    out = library.merge_variants({"Cm": rep("Cm", ["5f", "6d", "7s"])})       # actinides: no _3, base fit
    assert out["fits"]["6d"]["_potcar"] == "Cm"
