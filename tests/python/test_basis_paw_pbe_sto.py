"""The shipped fitted basis paw-pbe-sto-1 (data/basis/paw-pbe-sto-1, docs/BASIS_FIT_REPORT.md)."""
import json
from pathlib import Path

import pytest

import prowl
from prowl import BondingConfig, ProwlError
from prowl.config import polarization_shells, zval_valence_shells

D = Path(__file__).resolve().parents[2] / "data" / "basis" / "paw-pbe-sto-1"


def test_files_and_licence():
    doc = json.loads((D / "paw-pbe-sto-1.json").read_text())
    assert doc["schema"] == "prowl-sto-basis/1" and doc["name"] == "paw-pbe-sto-1"
    assert doc["provenance"]["license"] == "CC-BY-SA-4.0" and (D / "LICENSE.md").exists()
    fitted = {el for el, e in doc["elements"].items() if any("fit" in s for s in e["shells"])}
    with_potcar = {el for el, e in doc["elements"].items() if "fit_potcar" in e}
    assert len(doc["elements"]) == 103 and len(with_potcar) == 97 and len(fitted) == 71   # Koga H-Lr; POTCARs H-Cf
    assert {p.stem for p in (D / "sto").glob("*.sto")} == with_potcar
    for el in fitted:      # every fitted shell names its source atom
        for s in doc["elements"][el]["shells"]:
            if "fit" in s:
                assert s["fit"]["atom"]["titel"].startswith("PAW_PBE") and s["fit"]["atom"]["potcar_sha256"]
    gd5d = next(s for s in doc["elements"]["Gd"]["shells"] if s["label"] == "5d")
    assert gd5d["fit"]["potcar"] == "Gd_3"          # lanthanide 5d from the f-in-core atom (M6 decision)


def test_selectable_by_name():
    c = BondingConfig.from_dict({"basis": {"source": "paw-pbe-sto-1", "polarization": "all"}})
    assert c.library_path == D / "paw-pbe-sto-1.json"
    assert c.polarization_source() is None              # its own polarization shells, no LDA library merged
    lib = c.load_library()
    assert lib.name == "paw-pbe-sto-1"
    assert zval_valence_shells(lib, "Fe", 8.0, "Fe") == ["4s", "3d"] or \
        sorted(zval_valence_shells(lib, "Fe", 8.0, "Fe")) == ["3d", "4s"]
    assert polarization_shells(lib, ["Fe", "O"], "all") == {"Fe": ["4p"]}
    with pytest.raises(ProwlError, match="not built in"):
        BondingConfig.from_dict({"basis": {"source": "nonexistent"}})


def test_koga_default_unchanged():
    c = BondingConfig()
    assert c.library_path == prowl.default_basis_path() and c.polarization_source() is None
    c.basis_polarization = "all"
    assert c.polarization_source() is True
