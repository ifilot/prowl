"""Polarization-function generator (prowl.basisgen) and the generated library."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from prowl import _core, load_basis_library
from prowl.basisgen import (RadialGrid, even_tempered, excited_configuration, fit_sto, solve_atom, solve_radial,
                             sto_value)
from prowl.config import BondingConfig, polarization_shells

ROOT = Path(__file__).parents[2]
POL = ROOT / "data/basis/polarization/prowl_lda_polarization.json"
sys.path.insert(0, str(ROOT / "tools"))


def test_hydrogen_spectrum():
    g = RadialGrid()
    for l in (0, 1, 2):
        E, R, _ = solve_radial(g, -1.0 / g.r, l, 3 - l)
        np.testing.assert_allclose(E, [-0.5 / n**2 for n in range(l + 1, 4)], atol=2e-8)
        assert g.integrate(R[:, 0] ** 2 * g.r**2) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("Z,cfg,etot,eps", [
    (2, {"1s": 2}, -2.834836, {"1s": -0.570425}),
    (10, {"1s": 2, "2s": 2, "2p": 6}, -128.233481, {"1s": -30.305855, "2s": -1.322809, "2p": -0.498034}),
])
def test_lda_atom_matches_nist(Z, cfg, etot, eps):
    """NIST atomic reference data (LDA/VWN, non-relativistic)."""
    a = solve_atom(Z, cfg)
    assert a.converged
    assert a.total_energy == pytest.approx(etot, abs=2e-5)
    for k, v in eps.items():
        assert a.energies[k] == pytest.approx(v, abs=1e-5)


def test_sto_fit_recovers_an_sto_expansion():
    g = RadialGrid(n=8000)
    prims = even_tempered(1, 0.2, 20.0, 1.4)
    c0 = np.zeros(len(prims))
    c0[3], c0[8] = 0.7, -0.4
    R = sum(c * sto_value(n, z, g.r) for c, (n, z) in zip(c0, prims))
    R /= math.sqrt(g.integrate(R * R * g.r**2))
    c, resid, _ = fit_sto(g, R, prims)
    assert resid < 1e-10


def test_excitation_rules():
    from generate_polarization import polarization_label

    fe = {"1s": 2, "2s": 2, "3s": 2, "4s": 2, "2p": 6, "3p": 6, "3d": 6}
    assert polarization_label(fe) == "4p"
    cfg, desc = excited_configuration(fe, "4p")
    assert cfg["4s"] == 1 and cfg["4p"] == 1 and desc == "4s -> 4p"
    pd = {"1s": 2, "2s": 2, "3s": 2, "4s": 2, "2p": 6, "3p": 6, "4p": 6, "3d": 10, "4d": 10}
    assert polarization_label(pd) == "5p"
    cfg, desc = excited_configuration(pd, "5p")
    assert cfg["4d"] == 9 and cfg["5p"] == 1 and desc == "4d -> 5p"
    li = {"1s": 2, "2s": 1}
    cfg, _ = excited_configuration(li, "2p")
    assert cfg["2s"] == 0 and cfg["2p"] == 1


@pytest.fixture(scope="module")
def merged():
    if not POL.is_file():
        pytest.skip("polarization library not generated")
    return load_basis_library(polarization=True)


def test_polarization_library_contents(merged):
    from generate_polarization import ELEMENTS

    doc = json.loads(POL.read_text())
    assert doc["schema"] == "prowl-sto-basis/1" and "CC-BY-SA-4.0" in doc["provenance"]["license"]
    assert set(doc["elements"]) == set(ELEMENTS)
    for el, rec in doc["elements"].items():
        (sh,) = rec["shells"]
        g = sh["generation"]
        assert sh["occupation"] == 0 and sh["l"] == 1
        assert g["fit_residual"] < 1e-8 and g["eigenvalue_hartree"] < 0
        # nearly orthogonal to the occupied Koga p shells of the same atom
        assert all(abs(v) < 0.1 for v in g["overlap_with_koga_occupied_p"].values()), (el, g)
        # analytic norm of the STO expansion as loaded by the C++ core
        assert merged.shell(el, sh["label"]).norm2() == pytest.approx(1.0, abs=1e-8)
    fe = [s.label for s in merged.shells("Fe")]
    assert "4p" in fe and "3d" in fe
    assert "extensions" in json.loads(merged.provenance_json)


def test_polarization_selection_rules(merged):
    assert polarization_shells(merged, ["Fe", "S"], "all") == {"Fe": ["4p"]}
    assert polarization_shells(merged, ["Fe", "S"], {"Fe": ["4p"]}) == {"Fe": ["4p"]}
    assert polarization_shells(merged, ["Fe"], "none") == {}
    with pytest.raises(_core.ProwlError, match="no polarization shell 5p for Fe"):
        polarization_shells(merged, ["Fe"], {"Fe": ["5p"]})
    with pytest.raises(_core.ProwlError, match="not in the structure"):
        polarization_shells(merged, ["Fe"], {"Mg": ["3p"]})
    with pytest.raises(_core.ProwlError, match="basis.polarization"):
        BondingConfig.from_dict({"basis": {"polarization": "some"}})
    c = BondingConfig.from_dict({"basis": {"polarization": {"Fe": ["4p"]}}})
    assert c.uses_polarization_library and len(c.library_paths()) == 2
    assert not BondingConfig().uses_polarization_library


def test_merge_rejects_duplicates_and_unknown_elements(tmp_path, merged):
    doc = json.loads(POL.read_text())
    dup = {**doc, "name": "dup", "elements": {"Fe": {"Z": 26, "shells": [{**doc["elements"]["Fe"]["shells"][0],
                                                                           "label": "3p"}]}}}
    (tmp_path / "dup.json").write_text(json.dumps(dup))
    lib = load_basis_library()
    with pytest.raises(_core.ProwlError, match="duplicates shell Fe 3p"):
        lib.merge(_core.StoLibrary.load(str(tmp_path / "dup.json")))
