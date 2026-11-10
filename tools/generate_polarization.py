#!/usr/bin/env python3
"""Generate PROWL's polarization-function library (data/basis/polarization/prowl_lda_polarization.json).

For every s- and d-block element (groups 1-12 of periods 2-6, without the 4f series) the lowest
unoccupied p shell (e.g. Mg 3p, Fe 4p, Pd 5p) is generated with prowl.basisgen: self-consistent LDA
atom in an excited configuration with one electron in that shell, followed by an STO least-squares fit.
Everything is computed here from first principles; no third-party basis data are used.

Usage: tools/generate_polarization.py [output.json] [--elements Fe,Mg] [--workers 8]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

GENERATOR_VERSION = "1"
ELEMENTS = ("Li Be Na Mg K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
            "Cs Ba La Hf Ta W Re Os Ir Pt Au Hg").split()
# NIST atomic reference data (LDA, VWN, non-relativistic): solver validation recorded in the provenance
NIST_LDA = {"He": (2, {"1s": 2}, -2.834836, {"1s": -0.570425}),
            "Be": (4, {"1s": 2, "2s": 2}, -14.447209, {"1s": -3.856411, "2s": -0.205744}),
            "Ne": (10, {"1s": 2, "2s": 2, "2p": 6}, -128.233481,
                   {"1s": -30.305855, "2s": -1.322809, "2p": -0.498034})}
FIT_TARGET = 1e-8  # residual 2(1 - <R|fit>) of the normalized functions


def polarization_label(ground: dict) -> str:
    nmax = max(int(k[:-1]) for k, f in ground.items() if f > 0)
    lab = f"{nmax}p"
    return lab if ground.get(lab, 0) == 0 else f"{nmax + 1}p"


def generate(args):
    el, Z, ground = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np

    from prowl.basisgen import RadialGrid, even_tempered, excited_configuration, fit_sto, solve_atom

    lab = polarization_label(ground)
    cfg, desc = excited_configuration(ground, lab)
    t = time.time()
    grid = RadialGrid()
    atom = solve_atom(Z, cfg, grid)
    if not atom.converged:
        raise RuntimeError(f"{el}: SCF not converged in {atom.iterations} iterations")
    R = atom.orbitals[lab]
    r = grid.r
    eps = atom.energies[lab]
    zmin = max(0.03, 0.25 * math.sqrt(2 * abs(eps)))
    for ratio in (1.35, 1.25, 1.2):
        prims = even_tempered(1, zmin, 2.0 * Z, ratio)
        c, resid, rank = fit_sto(grid, R, prims)
        if resid < FIT_TARGET:
            break
    if resid >= FIT_TARGET:
        raise RuntimeError(f"{el} {lab}: STO fit residual {resid:.1e} >= {FIT_TARGET}")
    return {
        "element": el, "label": lab, "n": int(lab[:-1]), "l": 1, "occupation": 0.0, "energy": eps,
        "primitives": [{"n": n, "zeta": z, "coef": float(ci)} for ci, (n, z) in zip(c, prims)],
        "generation": {
            "configuration": {k: v for k, v in cfg.items() if v > 0}, "excitation": desc,
            "eigenvalue_hartree": eps, "total_energy_hartree": atom.total_energy, "scf_iterations": atom.iterations,
            "mean_radius_bohr": grid.integrate(R * R * r**3),
            "fit_residual": resid, "fit_primitives": len(prims), "fit_rank": rank,
            "orbital_energies_hartree": atom.energies, "seconds": round(time.time() - t, 1),
        },
    }


def koga_overlaps(lib: dict, rec: dict) -> dict:
    """Overlap of the generated function with Koga's occupied shells of the same l (same atom)."""
    import numpy as np

    from prowl.basisgen import RadialGrid, sto_value

    g = RadialGrid(n=8000)
    r = g.r

    def radial(prims):
        return sum(p["coef"] * sto_value(p["n"], p["zeta"], r) for p in prims)

    R = radial(rec["primitives"])
    out = {}
    for sh in lib["elements"][rec["element"]]["shells"]:
        if sh["label"][-1] == "p" and sh["occupation"] > 0:
            out[sh["label"]] = float(g.integrate(R * radial(sh["primitives"]) * r * r))
    return out


def validate_solver() -> dict:
    from prowl.basisgen import solve_atom

    out = {}
    for el, (Z, cfg, E, eps) in NIST_LDA.items():
        a = solve_atom(Z, cfg)
        out[el] = {"total_energy_error_hartree": a.total_energy - E,
                   "max_eigenvalue_error_hartree": max(abs(a.energies[k] - v) for k, v in eps.items())}
        if abs(a.total_energy - E) > 2e-5 or out[el]["max_eigenvalue_error_hartree"] > 1e-5:
            raise RuntimeError(f"solver validation failed for {el}: {out[el]}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", default=str(ROOT / "data/basis/polarization/prowl_lda_polarization.json"))
    ap.add_argument("--elements", default=",".join(ELEMENTS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    lib = json.loads((ROOT / "data/basis/koga/koga_hf_sto.json").read_text())
    from prowl.basisgen import koga_ground_configuration

    els = a.elements.split(",")
    jobs = [(el, lib["elements"][el]["Z"], koga_ground_configuration(lib, el)) for el in els]
    validation = validate_solver()
    print("solver validated against NIST LDA:", json.dumps(validation), file=sys.stderr)
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(generate, jobs))
    elements = {}
    for rec in recs:
        rec["generation"]["overlap_with_koga_occupied_p"] = koga_overlaps(lib, rec)
        el = rec.pop("element")
        elements[el] = {"Z": lib["elements"][el]["Z"], "shells": [rec]}
        g = rec["generation"]
        print(f"{el:2s} {rec['label']} eps={g['eigenvalue_hartree']:8.4f} Ha <r>={g['mean_radius_bohr']:5.2f} bohr "
              f"fit {g['fit_residual']:.1e} ({g['fit_primitives']} prim) {g['excitation']:10s} {g['seconds']}s",
              file=sys.stderr)
    doc = {
        "schema": "prowl-sto-basis/1",
        "name": "prowl_lda_polarization",
        "units": {"length": "bohr", "energy": "hartree"},
        "radial_convention": lib["radial_convention"],
        "provenance": {
            "generator": "tools/generate_polarization.py + python/prowl/basisgen.py",
            "generator_version": GENERATOR_VERSION,
            "generated": dt.date.today().isoformat(),
            "license": "CC-BY-SA-4.0 (generated by PROWL; no third-party basis data; https://creativecommons.org/licenses/by-sa/4.0/)",
            "method": ("non-relativistic spherical spin-unpolarized Kohn-Sham LDA atom (Slater exchange + VWN5), "
                       "excited configuration with one electron in the polarization shell; all-electron radial "
                       "function fitted by linear least squares to even-tempered normalized STOs"),
            "radial_grid": "x = ln r uniform, r in [1e-9, 50] bohr, 20000 points; fourth-order finite differences",
            "fit_target_residual": FIT_TARGET,
            "solver_validation_nist_lda": validation,
            "notes": ("occupation 0: never selected by the default ZVAL rule; enabled with basis.polarization. "
                      "energy = LDA eigenvalue in the excited configuration (hartree)."),
        },
        "elements": elements,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(elements)} elements to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
