import pytest

import prowl
from prowl import _core
from prowl.config import BondingConfig, potcar_symbol, zval_valence_shells


def test_import_and_backends():
    assert prowl.__version__
    assert "cpu" in _core.available_backends()
    assert _core.make_backend("cpu").name == "cpu"
    with pytest.raises(_core.ProwlError, match="InvalidArgument"):
        _core.make_backend("opencl")


def test_config_schema_is_strict():
    with pytest.raises(_core.ProwlError, match="unknown config key 'analysis.energy.foo'"):
        BondingConfig.from_dict({"analysis": {"energy": {"foo": 1}}})
    with pytest.raises(_core.ProwlError, match="wrong type"):
        BondingConfig.from_dict({"numerical": {"strict_validation": "yes"}})
    with pytest.raises(_core.ProwlError, match="precision"):
        BondingConfig.from_dict({"numerical": {"precision": "balanced"}})
    c = BondingConfig.from_dict({"numerical": {"projection": "paw_onsite"},
                                 "analysis": {"energy": {"broadening": {"sigma_ev": 0.1}}}})
    assert c.paw_offsite is False and c.sigma == 0.1
    assert c.to_dict()["band_window"] == "auto"


def test_profiles_and_defaults():
    d = BondingConfig()
    assert (d.profile, d.projection, d.band_window, d.window_sensitivity) == ("prowl", "paw_onsite", "auto", True)
    lob = BondingConfig.from_dict({"profile": "lobster"})
    assert (lob.projection, lob.band_window) == ("paw_onsite", "norb")
    # explicit keys override the profile
    o = BondingConfig.from_dict({"profile": "lobster", "numerical": {"band_window": "all", "projection": "paw_full"}})
    assert (o.band_window, o.projection) == ("all", "paw_full")
    assert BondingConfig.from_dict({"analysis": {"window_sensitivity": False}}).window_sensitivity is False
    with pytest.raises(_core.ProwlError, match="unknown profile"):
        BondingConfig.from_dict({"profile": "vasp"})


@pytest.mark.parametrize("el,zval,expect", [
    ("C", 4, ["2s", "2p"]), ("Ga", 13, ["4s", "4p", "3d"]), ("As", 5, ["4s", "4p"]),
    ("Fe", 8, ["4s", "3d"]), ("Ti", 12, ["3s", "4s", "3p", "3d"]), ("O", 6, ["2s", "2p"])])
def test_zval_valence_selection(el, zval, expect):
    lib = _core.StoLibrary.load_default()
    assert zval_valence_shells(lib, el, zval) == expect


@pytest.mark.parametrize("el,zval,symbol,expect", [
    ("Hf", 10, "Hf_pv", ["6s", "5p", "5d"]),               # 4f14 is core from Hf on
    ("W", 14, "W_sv", ["5s", "6s", "5p", "5d"]),
    ("Lu", 25, "Lu", ["5s", "6s", "5p", "5d", "4f"]),      # Lu: 4f14 still valence
    ("Gd", 9, "Gd_3", ["6s", "5p", "5d"]),                 # f in core: 5p6 6s2 5d1
    ("Ce", 11, "Ce_3", ["5s", "6s", "5p", "5d"]),          # 5s2 5p6 6s2 5d1
    ("Eu", 8, "Eu_2", ["6s", "5p"])])
def test_zval_selection_special_potcars(el, zval, symbol, expect):
    lib = _core.StoLibrary.load_default()
    assert zval_valence_shells(lib, el, zval, symbol) == expect


def test_f_in_core_needs_5d_in_library():
    lib = _core.StoLibrary.load_default()          # Koga Eu ([Xe]4f7 6s2) has no 5d shell
    with pytest.raises(_core.ProwlError, match="f-in-core POTCAR Eu_3"):
        zval_valence_shells(lib, "Eu", 9, "Eu_3")
    assert potcar_symbol("PAW_PBE Eu_3 20Oct2008") == "Eu_3"


def test_zval_mismatch_is_an_error():
    lib = _core.StoLibrary.load_default()
    with pytest.raises(_core.ProwlError, match="ZVAL"):
        zval_valence_shells(lib, "C", 3)
