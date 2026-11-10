"""POTCAR metadata and the shell lists of a POTCAR (valence shells + polarization shells).

Valence shells: the Koga ground configuration ordered by HF orbital energy (highest first), accumulated until
the occupations add up to ZVAL. f-in-core POTCARs (`_2`, `_3`) exclude the open 4f/5f shell.
Polarization ("added") shells: shells that LOBSTER's recommended basis lists for the POTCAR (pymatgen,
MIT licence, BASIS_PBE_54_standard.yaml; shell labels frozen in data/pmg_basis_PBE54_standard.json) but that the
free atom does not occupy.  If a POTCAR is not listed, the rule of thumb of that list is applied: the lowest empty
p shell for s/d/f metals.

The POTCAR directory comes from $PROWL_POTCAR_DIR (potpaw_PBE layout: <dir>/<symbol>/POTCAR).
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import prowl

LETTER = "spdf"
PMG_STANDARD = Path(__file__).resolve().parents[1] / "data/pmg_basis_PBE54_standard.json"


def potcar_dir() -> Path:
    d = os.environ.get("PROWL_POTCAR_DIR")
    if not d:
        raise RuntimeError("set PROWL_POTCAR_DIR to the potpaw_PBE directory")
    return Path(d)


def element_of(potcar: str) -> str:
    return potcar.split("_")[0].split(".")[0]


@lru_cache(None)
def potcar_info(potcar: str) -> dict:
    text = (potcar_dir() / potcar / "POTCAR").read_text(errors="replace")
    head = text[:4000]
    m = lambda pat: re.search(pat, head)
    return {"potcar": potcar, "element": element_of(potcar),
            "titel": m(r"TITEL\s*=\s*(.+)").group(1).strip(),
            "zval": float(m(r"ZVAL\s*=\s*([\d.]+)").group(1)),
            "enmax": float(m(r"ENMAX\s*=\s*([\d.]+)").group(1)),
            "vrhfin": m(r"VRHFIN\s*=\s*(.+)").group(1).strip()}


@lru_cache(None)
def koga() -> dict:
    return json.loads(Path(prowl.default_basis_path()).read_text())["elements"]


@lru_cache(None)
def pmg_list() -> dict:
    """POTCAR symbol -> shell labels of pymatgen's BASIS_PBE_54_standard list (frozen copy)."""
    return json.loads(PMG_STANDARD.read_text())["shells"]


def f_in_core(potcar: str) -> bool:
    return bool(re.search(r"_[23]$", potcar))


def valence_shells(potcar: str) -> list[str]:
    """Occupied Koga shells that are valence for this POTCAR (by orbital energy, summing to ZVAL)."""
    info = potcar_info(potcar)
    if f_in_core(potcar):
        # lanthanide _2/_3 POTCARs: 5p6 6s2 (8 e), then 5s2 if ZVAL >= 10, then 5d with the rest
        # (ZVAL 8: 5p6 6s2; 9: 5p6 6s2 5d1; 11: 5s2 5p6 6s2 5d1). Counting Koga energies instead picks 4d/5s.
        rest = round(info["zval"]) - 8
        out = ["6s", "5p"]
        if rest >= 2:
            out.append("5s")
            rest -= 2
        if rest > 0:
            out.append("5d")
        return sorted(out, key=lambda t: ("spdf".index(t[-1]), int(t[:-1])))
    sh = [s for s in koga()[info["element"]]["shells"] if s["occupation"] > 0]
    Z = koga()[info["element"]]["Z"]
    if Z > 71:   # a full 4f shell is core in every POTCAR from Hf onwards (HF puts it above 5s/5p)
        sh = [s for s in sh if s["label"] != "4f"]
    sh.sort(key=lambda s: -s["energy"])
    out, tot = [], 0.0
    for s in sh:
        if tot >= info["zval"] - 1e-6:
            break
        out.append(s["label"])
        tot += s["occupation"]
    if f_in_core(potcar):   # valence electrons count the f electrons that are frozen: drop the check
        return out
    if abs(tot - info["zval"]) > 1e-6:
        raise ValueError(f"{potcar}: Koga occupations do not sum to ZVAL={info['zval']} ({out}, {tot})")
    return out


def polarization_p(potcar: str) -> list[str]:
    """2015 design rule: the lowest empty p shell of s, d and f metals (n = outermost valence n)."""
    el = element_of(potcar)
    Z = koga()[el]["Z"]
    pblock = Z in (1, 2) or Z in set(range(5, 11)) | set(range(13, 19)) | set(range(31, 37)) | set(range(49, 55)) \
        | set(range(81, 87))
    if pblock:
        return []
    occ = {s["label"] for s in koga()[el]["shells"] if s["occupation"] > 0}
    val = valence_shells(potcar)
    top = max(int(t[:-1]) + {"s": 0, "p": 0, "d": 1, "f": 2}[t[-1]] for t in val)
    cand = f"{top}p"
    return [] if cand in occ or top < 2 else [cand]


def selection_standard(potcar: str) -> list[str]:
    """Shells a LOBSTER user would select (pymatgen standard list; else valence + rule-based p shell)."""
    lst = pmg_list().get(potcar)
    return list(lst) if lst is not None else valence_shells(potcar) + polarization_p(potcar)


def added_shells(potcar: str) -> list[str]:
    """Unoccupied shells to fit: those of the standard selection plus the rule-based polarization p shell."""
    el = element_of(potcar)
    occ = {s["label"] for s in koga()[el]["shells"] if s["occupation"] > 0} | set(valence_shells(potcar))
    out = [t for t in selection_standard(potcar) if t not in occ]
    for t in polarization_p(potcar):
        if t not in out:
            out.append(t)
    return out


def shell_list(potcar: str) -> dict:
    """{'valence': [...], 'added': [...], 'pmg_standard': [...] or None}, sorted by (l, n)."""
    key = lambda s: (LETTER.index(s[-1]), int(s[:-1]))
    v, a = valence_shells(potcar), added_shells(potcar)
    pm = pmg_list().get(potcar)
    return {"valence": sorted(v, key=key), "added": sorted(a, key=key), "pmg_standard": pm,
            "standard": sorted(selection_standard(potcar), key=key),
            "pmg_valence_mismatch": None if pm is None else sorted(set(pm) ^ set(v) - set(a))}
