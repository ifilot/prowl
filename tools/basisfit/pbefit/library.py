"""Build basis libraries from fit reports: PROWL JSON (prowl-sto-basis/1) and LOBSTER .sto files.

An element entry = the Koga entry with fitted shells replaced (2026 mode) or added (2015 mode).
Variant: number of extra primitives `m` per fitted shell (int, or dict {shell: m}).

The frozen method (M3, 2026-11-06) is `box_rule_files` (which atom run per POTCAR) + `rule_2026` (which fitted
shells replace Koga) + `auto_m` (how many diffuse extras), all on nonspin smeared atoms.
"""
from __future__ import annotations

import copy
import datetime
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from .fit import DEFAULTS, sto_S
from .shells import LETTER, koga

# p block (groups 13-18) and H/He
P_BLOCK = {1, 2} | set(range(5, 11)) | set(range(13, 19)) | set(range(31, 37)) | set(range(49, 55)) | set(range(81, 87))
BOX_RULE_EPS = -1.0      # eV: a fitted shell bound by less than this in the 20 A box -> use the 24 A box
BOHR_A = 0.529177210903
LANTHANIDES = set(range(57, 72))
# the frozen atom settings (pbefit.atom), recorded in every library
ATOM_SETTINGS = {
    "code": "VASP, gamma-only, one atom in a cubic box (20 A; 24 A by the box rule)",
    "functional": "PBE (potpaw_PBE)",
    "configuration": "nonspin: ISPIN 1, ISMEAR 0, SIGMA 0.2 eV",
    "basis": "ENCUT max(400, 1.3 ENMAX) eV rounded up to 10 eV, PREC Accurate, LREAL .FALSE., ISYM 0",
    "scf": "EDIFF 1e-7; setting A: ALGO Normal, NELM 60; setting B (only if A fails): ALGO All, NELM 250",
    "box_rule": f"24 A box if a fitted shell is bound by less than {-BOX_RULE_EPS:g} eV in the 20 A box",
}


def box_rule_files(fitdir: Path, mode: str = "2026") -> dict:
    """{POTCAR: fit report path}: the nonspin 20 A fit, or the 24 A fit if one exists and a fitted shell of the
    20 A atom is bound by less than 1 eV (2015-mode fits hold only added shells: decide from the 2026 fit)."""
    fitdir = Path(fitdir)
    out = {}
    for f in sorted(fitdir.glob(f"*_nonspin_box20_{mode}.json")):
        pot = f.name[: -len(f"_nonspin_box20_{mode}.json")]
        ref = fitdir / f"{pot}_nonspin_box20_2026.json"
        r = json.loads((ref if ref.exists() else f).read_text())
        eps = max(fs["0"]["eps_ev"][0] for fs in r["fits"].values()) if r["fits"] else -99.0
        f24 = fitdir / f"{pot}_nonspin_box24_{mode}.json"
        out[pot] = f24 if (eps > BOX_RULE_EPS and f24.exists()) else f
    return out


def needs_box24(rep2026: dict) -> bool:
    """Box rule on a 20 A 2026-mode report: is a fitted shell bound by less than 1 eV?"""
    return bool(rep2026["fits"]) and max(fs["0"]["eps_ev"][0] for fs in rep2026["fits"].values()) > BOX_RULE_EPS


def rule_2026(rep: dict) -> dict:
    """Copy of a 2026-mode report keeping only the shells that replace Koga under the frozen rule:
    added (unoccupied) shells always; occupied shells of s/d/f elements and of the 6p block (Tl-Rn: relativistic
    contraction that non-relativistic Koga lacks), except filled d10/f14 shells; the other p-block elements and
    H/He keep Koga for every occupied shell."""
    el = rep["element"]
    Z = koga()[el]["Z"]
    pblock = Z in P_BLOCK and Z <= 80
    occ = {s["label"]: s["occupation"] for s in koga()[el]["shells"] if s["occupation"] > 0}

    def keep_koga(t):
        if t not in occ:
            return False
        if pblock:
            return True
        l = LETTER.index(t[-1])
        return (l == 2 and occ[t] >= 10) or (l == 3 and occ[t] >= 14)

    out = dict(rep)
    out["fits"] = {t: f for t, f in rep["fits"].items() if not keep_koga(t)}
    return out


def fit_checks(e: dict) -> dict:
    """The checks of one fit. `coef_ok` (max |coefficient| <= coef_max, added 2026-11-06) is evaluated from the
    stored primitives when an older report lacks it: the same result as a refit."""
    ch = dict(e.get("checks", {}))
    if "coef_ok" not in ch and "primitives" in e:
        ch["coef_ok"] = max(abs(p["coef"]) for p in e["primitives"]) <= DEFAULTS["coef_max"]
    return ch


def fit_passed(e: dict) -> bool:
    return bool(e.get("passed", True)) and all(fit_checks(e).values())


def auto_m(f: dict, tol: float = 1e-4) -> str:
    """Fewest extra primitives whose fit passes all checks and is within `tol` of the best passing fit.
    Negative orbital spilling (captured fraction > 1) is PAW noise and counts as 0."""
    ms = sorted((k for k in f if k.isdigit()), key=int)
    ok = [k for k in ms if fit_passed(f[k])] or ms
    bestv = max(0.0, min(f[k]["orbital_spilling"] for k in ok))
    return next(k for k in ok if f[k]["orbital_spilling"] - bestv < tol)


BASE_SUFFIXES = ("", "_pv", "_sv", "_d")   # POTCAR variants that feed the per-element entry, in base order


def atom_provenance(rep: dict) -> dict:
    """The VASP atom behind a fit report: POTCAR TITEL, box, and (read from the run directory when it is available)
    the POTCAR's SHA256 line, SCF setting and VASP version."""
    out = {"titel": rep.get("potcar_info", {}).get("titel", "")}
    if "half_box_bohr" in rep:
        out["box_angstrom"] = round(2 * rep["half_box_bohr"] * BOHR_A, 2)
    run = Path(rep["run"]) if rep.get("run") else None
    if run and (run / "pbefit_status.txt").exists():
        out["scf_setting"] = (run / "pbefit_status.txt").read_text().split("setting=")[1].split()[0]
    if run and (run / "POTCAR").exists():      # the POTCAR's own SHA256 line (identifies the file, no content)
        with (run / "POTCAR").open() as fh:
            sha = next((ln.split("=", 1)[1].split()[0] for ln in fh if ln.strip().startswith("SHA256")), None)
        if sha:
            out["potcar_sha256"] = sha
    if run and (run / "OUTCAR").exists():
        with (run / "OUTCAR").open() as fh:
            out["vasp"] = " ".join(fh.readline().split())
    return out


def merge_variants(reps: dict) -> dict:
    """{POTCAR: report} of one element (2026 mode, after rule_2026) -> one report for the element entry:
    every shell of the base POTCAR (the plain symbol, else _pv, _sv, _d) plus the shells that only another
    variant fits (semicore 3p of Fe_pv, 3s of Fe_sv, ...), taken from the first variant in BASE_SUFFIXES order.
    Hard/soft (_h, _s), GW and f-in-core (_2, _3) POTCARs are not used, except the lanthanide 5d from _3 (below).
    Each shell records its source POTCAR.
    Justification: the same shell fitted from different variants agrees to overlap >= 0.998 outside the f block."""
    el = next(iter(reps.values()))["element"]
    use = [el + sfx for sfx in BASE_SUFFIXES if el + sfx in reps]
    if not use:
        raise ValueError(f"{el}: none of {[el + s for s in BASE_SUFFIXES]} fitted")
    base = reps[use[0]]
    out = dict(base)
    atom = atom_provenance(base)
    out["fits"] = {t: {**f, "_potcar": base["potcar"], "_atom": atom} for t, f in base["fits"].items()}
    for pot in use[1:]:
        atom = atom_provenance(reps[pot])
        for t, f in reps[pot]["fits"].items():
            if t not in out["fits"]:
                out["fits"][t] = {**f, "_potcar": pot, "_atom": atom}
    # User decision 2026-11-06 (M6): lanthanide 5d from the f-in-core _3 atom (4f^n 5d^1-like). The nonspin
    # f-in-valence atom moves the d electron into 4f; its 5d is up to 33 % more diffuse (Gd 3.40 vs 2.85 bohr,
    # all-electron 4f7 5d1: 2.85). Actinides have no _3 POTCARs and keep the base fit.
    l3 = reps.get(el + "_3")
    if koga()[el]["Z"] in LANTHANIDES and l3 and "5d" in l3["fits"]:
        out["fits"]["5d"] = {**l3["fits"]["5d"], "_potcar": l3["potcar"], "_atom": atom_provenance(l3)}
    out["variants"] = use
    return out


def element_entry(rep: dict, m="auto") -> dict:
    el = rep["element"]
    ent = copy.deepcopy(koga()[el])
    shells = ent["shells"]
    own_atom = None
    for t, f in rep["fits"].items():
        mm = m.get(t, "auto") if isinstance(m, dict) else m
        mm = auto_m(f) if mm == "auto" else str(min(int(mm), max(int(k) for k in f if k.isdigit())))
        src = f.get("_potcar", rep["potcar"])
        if "_atom" in f:
            atom = f["_atom"]
        else:
            own_atom = own_atom or atom_provenance(rep)
            atom = own_atom
        e = f[mm]
        old = next((s for s in shells if s["label"] == t), None)
        new = {"label": t, "n": e["n"], "l": e["l"],
               "occupation": old["occupation"] if old else 0.0, "energy": old["energy"] if old else 0.0,
               "primitives": e["primitives"],
               "fit": {"orbital_spilling": e["orbital_spilling"], "n_extra": int(mm), "r_mean_bohr": e["r_mean_bohr"],
                       "potcar": src, "atom": atom, "mode": rep["config"]["mode"], "share": rep["config"]["share"]}}
        if old:
            shells[shells.index(old)] = new
        else:
            shells.append(new)
    shells.sort(key=lambda s: (s["l"], s["n"]))
    ent["fit_potcar"] = rep["potcar"]
    if "variants" in rep:
        ent["fit_variants"] = rep["variants"]
    return ent


def build_library(reports: dict, name: str, m="auto", note="") -> dict:
    """reports: {element: fit report}; returns a prowl-sto-basis/1 document (Koga for unfitted elements)."""
    base = json.loads(Path(__import__("prowl").default_basis_path()).read_text())
    doc = copy.deepcopy(base)
    doc["name"] = name
    doc["provenance"]["pbefit"] = {
        "method": "STO shells fitted to VASP PBE free-atom PAW orbitals; frozen method of tools/basisfit "
                  "(docs/BASIS_FIT_PLAN.md, M3 2026-11-06)",
        "koga_base": "shells without a 'fit' block are the Koga HF STOs described above",
        "created": datetime.date.today().isoformat(), "code_commit": _code_commit(), "note": note,
        "atom_settings": ATOM_SETTINGS,
        "potcars": {el: r["potcar"] for el, r in reports.items()},
        "config": {el: r["config"] for el, r in reports.items()}}
    for el, rep in reports.items():
        doc["elements"][el] = element_entry(rep, m)
    doc["provenance"]["pbefit"]["vasp"] = sorted({s["fit"]["atom"]["vasp"] for el in reports
                                                  for s in doc["elements"][el]["shells"]
                                                  if "fit" in s and "vasp" in s["fit"]["atom"]})
    return doc


def _code_commit() -> str:
    """git commit of tools/basisfit ('+dirty' if it has local changes), or 'unknown' outside a checkout."""
    here = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_sto(entry: dict, el: str, path: Path):
    """LOBSTER custom .sto: per l one shared primitive list, one coefficient column per shell (n consecutive)."""
    lines = [f"{el.lower()}, Z={entry['Z']}"]
    first, norb, nprim, blocks = [], [], [], []
    for l in range(4):
        sh = sorted([s for s in entry["shells"] if s["l"] == l], key=lambda s: s["n"])
        if not sh:
            first.append(0); norb.append(0); nprim.append(0); continue
        ns = [s["n"] for s in sh]
        if ns != list(range(ns[0], ns[0] + len(ns))):
            raise ValueError(f"{el} l={l}: shells not consecutive {ns}")
        prims = []
        for s in sh:
            for p in s["primitives"]:
                key = (p["n"], round(p["zeta"], 6))
                if key not in prims:
                    prims.append(key)
        prims.sort(key=lambda k: -k[1])
        C = np.zeros((len(prims), len(sh)))
        for j, s in enumerate(sh):
            for p in s["primitives"]:
                C[prims.index((p["n"], round(p["zeta"], 6))), j] += p["coef"]
        first.append(ns[0]); norb.append(len(sh)); nprim.append(len(prims))
        for (n, z), row in zip(prims, C):
            blocks.append(f"{'SPDF'[l]} {n}  {z:18.10f} " + " ".join(f"{c:16.10f}" for c in row))
    lines += ["FirstOrbitals: " + " ".join(map(str, first)), "Orbitals: " + " ".join(map(str, norb)),
              "Primitives: " + " ".join(map(str, nprim))] + blocks
    Path(path).write_text("\n".join(lines) + "\n")


def overlap(sh1: dict, sh2: dict) -> float:
    """Normalized overlap of two radial STO contractions of the same l."""
    p1 = [(p["n"], p["zeta"]) for p in sh1["primitives"]]
    p2 = [(p["n"], p["zeta"]) for p in sh2["primitives"]]
    c1 = np.array([p["coef"] for p in sh1["primitives"]]); c2 = np.array([p["coef"] for p in sh2["primitives"]])
    S = sto_S(p1 + p2)
    v1 = np.concatenate([c1, np.zeros(len(c2))]); v2 = np.concatenate([np.zeros(len(c1)), c2])
    return float(abs(v1 @ S @ v2) / math.sqrt((v1 @ S @ v1) * (v2 @ S @ v2)))


def r_mean(sh: dict) -> float:
    from .fit import sto_rmat
    p = [(q["n"], q["zeta"]) for q in sh["primitives"]]
    c = np.array([q["coef"] for q in sh["primitives"]])
    return float(c @ sto_rmat(p, 1) @ c / (c @ sto_S(p) @ c))


def scale_extras(doc: dict, factor: float, shells: dict | None = None) -> dict:
    """Copy of a library with the diffuse extra primitives of fitted shells scaled by `factor`
    (coefficients kept, shell renormalized). shells: {El: [labels]} to restrict, else every fitted shell.
    Extra primitives = those not in the Koga primitive set of the element and l."""
    out = copy.deepcopy(doc)
    for el, ent in out["elements"].items():
        if "fit_potcar" not in ent:
            continue
        kz = {(p["n"], round(p["zeta"], 6)) for s in koga()[el]["shells"] for p in s["primitives"]}
        for s in ent["shells"]:
            if "fit" not in s or (shells and s["label"] not in shells.get(el, [])):
                continue
            for p in s["primitives"]:
                if (p["n"], round(p["zeta"], 6)) not in kz:
                    p["zeta"] *= factor
            pr = [(p["n"], p["zeta"]) for p in s["primitives"]]
            c = np.array([p["coef"] for p in s["primitives"]])
            nrm = math.sqrt(c @ sto_S(pr) @ c)
            for p in s["primitives"]:
                p["coef"] /= nrm
    return out
