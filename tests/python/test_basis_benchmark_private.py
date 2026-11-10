"""Regression benchmark of the pbeVaspFit2026 basis on 17 solids (docs/BASIS_FIT_PLAN.md, M7).

Needs the private VASP runs (PROWL_TEST_RUNS/<run>, generated from tests/private_vasp_manifest/recipes) and the
basis library (PROWL_TEST_BASIS, a prowl-sto-basis/1 JSON from `tools/basisfit/basisfit.py library ... --all`).
Reference values and tolerances: tests/private_vasp_manifest/basis_benchmark.json.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

REF = json.loads((Path(__file__).resolve().parents[1] / "private_vasp_manifest" / "basis_benchmark.json").read_text())
RUNS = os.environ.get("PROWL_TEST_RUNS")
BASIS = os.environ.get("PROWL_TEST_BASIS")

pytestmark = pytest.mark.skipif(not (RUNS and BASIS), reason="set PROWL_TEST_RUNS and PROWL_TEST_BASIS")


def _species(run: Path) -> list:
    pos = run / "CONTCAR" if (run / "CONTCAR").exists() and (run / "CONTCAR").stat().st_size > 0 else run / "POSCAR"
    lines = pos.read_text().split("\n")
    return [e for e, c in zip(lines[5].split(), map(int, lines[6].split())) for _ in range(c)]


@pytest.mark.parametrize("name", sorted(REF["solids"]))
def test_basis_benchmark(name, tmp_path):
    from prowl import BondingConfig, Calculation
    ref = REF["solids"][name]
    run = Path(RUNS) / ref["run"]
    if not (run / "WAVECAR").exists():
        pytest.skip(f"run {run} not available")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": "prowl-sto-basis/1", "name": "empty", "provenance": {}, "elements": {},
                                 "units": {"length": "bohr", "energy": "hartree"}}))
    cfg = BondingConfig.for_profile(REF["profile"])
    cfg.window_sensitivity = False
    cfg.bond_mode, cfg.max_distance = "distance_cutoff", ref["max_distance"]
    cfg.basis_source, cfg.basis_library = Path(BASIS).stem, BASIS
    cfg.basis_selection, cfg.polarization_library = ref["selection"], str(empty)
    r = Calculation.open(str(run)).analyze(cfg).result

    tol = REF["tolerance"]
    assert 100 * r.diagnostics["abs_charge_spilling"] == pytest.approx(ref["spilling_pct"], abs=tol["spilling_pp"])
    ic = np.asarray(r.icohp)
    icb = ic.reshape(-1, ic.shape[-1]).sum(axis=0) if ic.ndim > 1 else ic     # spin-summed per bond
    species = _species(run)
    groups = defaultdict(list)
    for b, v in zip(r.bonds, icb):
        groups[f"{'-'.join(sorted([species[b.atom_a], species[b.atom_b]]))}@{b.distance:.3f}"].append(float(v))
    got = {k: float(np.mean(v)) for k, v in groups.items()}
    assert sorted(got) == sorted(ref["bonds"])
    for k, v in ref["bonds"].items():
        assert got[k] == pytest.approx(v, rel=tol["icohp_rel"], abs=tol["icohp_abs"]), k
