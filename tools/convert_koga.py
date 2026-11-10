#!/usr/bin/env python3
"""Convert Koga Hartree-Fock STO tabulations into PROWL's basis JSON format.

Input: the ``literature_data`` directory of Furness & Lehtola's AtomicOrbitals
(https://github.com/JFurness1/AtomicOrbitals, CC0 1.0), which contains the
Koga et al. tabulations as received from A. Thakkar:

* ``k99l/neutral``  -- Koga, Kanayama, Watanabe, Thakkar, IJQC 71, 491 (1999), H-Xe
* ``k00heavy``      -- Koga, Kanayama, Watanabe, Imai, Thakkar, TCA 104, 411 (2000), Cs-Lr

Output schema (``schema: prowl-sto-basis/1``), atomic units throughout::

    {"schema": ..., "name": "koga_hf_sto", "units": {"length": "bohr", "energy": "hartree"},
     "radial_convention": "R(r) = sum_p c_p N_p r^(n_p-1) exp(-zeta_p r), N_p=(2 zeta_p)^(n_p+1/2)/sqrt((2 n_p)!)",
     "provenance": {...}, "elements": {"Si": {"Z": 14, "configuration": "...",
        "shells": [{"label": "3s", "n": 3, "l": 0, "occupation": 2.0, "energy": -0.54,
                    "primitives": [{"n": 1, "zeta": 40.74, "coef": 2.7e-3}, ...]}]}}}

Usage: tools/convert_koga.py <AtomicOrbitals checkout or data/basis/koga/upstream> <output.json>
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

LETTERS = "SPDFGH"
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
           "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()

CORES = {
    "K": "1S(2)", "L": "2S(2)2P(6)", "M": "3S(2)3P(6)3D(10)",
    "[HE]": "1S(2)", "[NE]": "1S(2)2S(2)2P(6)", "[AR]": "1S(2)2S(2)2P(6)3S(2)3P(6)",
}
CORES["[KR]"] = CORES["[AR]"] + "3D(10)4S(2)4P(6)"
CORES["[XE]"] = CORES["[KR]"] + "4D(10)5S(2)5P(6)"
CORES["[RN]"] = CORES["[XE]"] + "4F(14)5D(10)6S(2)6P(6)"


def parse_configuration(cfg: str) -> dict[str, float]:
    cfg = cfg.split(",")[0].strip().upper()
    for key in sorted(CORES, key=len, reverse=True):
        if key.startswith("["):
            cfg = cfg.replace(key, CORES[key])
    # K(2)L(8)M(18) style shells
    cfg = re.sub(r"\b([KLM])\(\d+\)", lambda m: CORES[m.group(1)], cfg)
    cfg = re.sub(r"(?<![0-9])([KLM])\(\d+\)", lambda m: CORES[m.group(1)], cfg)
    occ: dict[str, float] = {}
    for n, letter, o in re.findall(r"(\d+)([SPDF])\((\d+)\)", cfg):
        key = f"{n}{letter.lower()}"
        occ[key] = occ.get(key, 0.0) + float(o)
    return occ


def parse_file(path: Path) -> dict:
    lines = path.read_text().splitlines()
    header = lines[0].split(None, 1)
    name, cfg = header[0], header[1].strip()
    i = next(j for j, l in enumerate(lines) if "ORBITAL ENERGIES AND EXPANSION COEFFICIENTS" in l) + 1
    shells = []
    while i < len(lines):
        tok = lines[i].split()
        if not tok or tok[0] not in LETTERS:
            i += 1
            continue
        l = LETTERS.index(tok[0])
        labels = tok[1:]
        energies = [float(x) for x in lines[i + 1].split()[1:]]
        i += 2
        if lines[i].split()[0] == "CUSP":
            i += 1
        prims = []
        while i < len(lines) and len(lines[i].split()) == len(labels) + 2:
            e = lines[i].split()
            assert e[0][1] == tok[0], (path, lines[i])
            prims.append((int(e[0][0]), float(e[1]), [float(x) for x in e[2:]]))
            i += 1
        for c, lab in enumerate(labels):
            shells.append({
                "label": lab.lower(), "n": int(lab[:-1]), "l": l, "energy": energies[c],
                "primitives": [{"n": n, "zeta": z, "coef": cs[c]} for n, z, cs in prims],
            })
    occ = parse_configuration(cfg)
    for s in shells:
        s["occupation"] = occ.get(s["label"], 0.0)
    return {"name": name, "configuration": cfg, "shells": shells}


def main() -> int:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    lit = src / "literature_data"
    files = sorted((lit / "k99l" / "neutral").iterdir()) + sorted((lit / "k00heavy").iterdir())
    elements = {}
    hashes = {}
    for f in files:
        sym = f.name.capitalize()
        if sym not in SYMBOLS:
            continue
        rec = parse_file(f)
        rec["Z"] = SYMBOLS.index(sym) + 1
        rec["source_file"] = str(f.relative_to(src))
        elements[sym] = rec
        hashes[str(f.relative_to(src))] = hashlib.sha256(f.read_bytes()).hexdigest()
    if (src / "UPSTREAM_COMMIT").is_file():  # vendored copy (data/basis/koga/upstream)
        commit = (src / "UPSTREAM_COMMIT").read_text().strip()
    else:  # an AtomicOrbitals git checkout (not a subdirectory of another repository)
        try:
            top = subprocess.check_output(["git", "-C", str(src), "rev-parse", "--show-toplevel"], text=True).strip()
            commit = (subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
                      if Path(top).resolve() == src.resolve() else "unknown")
        except Exception:
            commit = "unknown"
    doc = {
        "schema": "prowl-sto-basis/1",
        "name": "koga_hf_sto",
        "units": {"length": "bohr", "energy": "hartree"},
        "radial_convention": "R(r) = sum_p c_p N_p r^(n_p-1) exp(-zeta_p r), N_p = (2 zeta_p)^(n_p+1/2) / sqrt((2 n_p)!)",
        "provenance": {
            "upstream": "https://github.com/JFurness1/AtomicOrbitals",
            "upstream_commit": commit,
            "upstream_license": "CC0 1.0 Universal (waiver by J. Furness and S. Lehtola)",
            "data_origin": "Tabulations received by S. Lehtola from A. J. Thakkar (literature_data/README.txt)",
            "citations": [
                "T. Koga, K. Kanayama, S. Watanabe, A. J. Thakkar, Int. J. Quantum Chem. 71, 491 (1999), doi:10.1002/(SICI)1097-461X(1999)71:6<491::AID-QUA6>3.0.CO;2-T",
                "T. Koga, K. Kanayama, S. Watanabe, T. Imai, A. J. Thakkar, Theor. Chem. Acc. 104, 411 (2000), doi:10.1007/s002140000150",
            ],
            "source_sha256": hashes,
            "converter": "tools/convert_koga.py",
        },
        "elements": elements,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {len(elements)} elements to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
