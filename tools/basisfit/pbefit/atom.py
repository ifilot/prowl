"""Free-atom VASP runs (Gamma only, one atom at the origin of a cubic box).

The same settings are used for every POTCAR:
* ENCUT = max(400, 1.3 ENMAX) rounded up to 10 eV, PREC = Accurate, LREAL = .FALSE., ISYM = 0
* configuration "nonspin": ISPIN = 1, Gaussian SIGMA = 0.2 eV (spherical average of the open shells)
  configuration "spin":    ISPIN = 2, MAGMOM = Hund's-rule moment of the Koga configuration, SIGMA = 0.05 eV
  configuration "s1":      ISPIN = 2, NUPDOWN = high-spin moment of d^(N-1) s^1 (2015's state; M2 check only)
* SCF: setting A = ALGO Normal (Davidson), NELM 60; setting B (only if A fails) = ALGO All, NELM 250.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .shells import LETTER, koga, potcar_dir, potcar_info, shell_list

WORK = Path(os.environ.get("PBEFIT_WORK", ".")).resolve() / "atoms"   # run directories
# Gamma-only VASP command; "{threads}" and "{ranks}" are filled in if present
VASP_CMD = os.environ.get("PBEFIT_VASP_CMD", "vasp_gam")


def hund_moment(el: str, shells: list[str]) -> int:
    mom = 0
    for s in koga()[el]["shells"]:
        if s["label"] in shells and s["occupation"] > 0:
            cap = 2 * (2 * s["l"] + 1)
            occ = s["occupation"]
            mom += round(occ if occ <= cap / 2 else cap - occ)
    return mom


def s1_moment(el: str) -> int:
    """High-spin moment of the d^(N-1) s^1 configuration (N = electrons in the outermost d and s shells)."""
    sh = [s for s in koga()[el]["shells"] if s["l"] in (0, 2)]
    nd = max(int(s["label"][:-1]) for s in sh if s["l"] == 2)
    n_el = sum(s["occupation"] for s in sh if (s["l"] == 2 and int(s["label"][:-1]) == nd)
               or (s["l"] == 0 and int(s["label"][:-1]) == nd + 1))
    k = round(n_el) - 1
    return min(k, 10 - k) + 1


def nbands_for(potcar: str) -> int:
    sh = shell_list(potcar)
    norb = sum(2 * LETTER.index(t[-1]) + 1 for t in sh["valence"] + sh["added"])
    nel = potcar_info(potcar)["zval"]
    return int(max(norb + 8, math.ceil(nel / 2) + 12))


def encut_for(potcar: str) -> int:
    return int(max(400, math.ceil(1.3 * potcar_info(potcar)["enmax"] / 10) * 10))


def setup(potcar: str, conf: str = "nonspin", box: float = 20.0, tag: str = "", algo="Normal", nelm=60) -> Path:
    d = WORK / f"{potcar}_{conf}_box{box:g}{tag}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    el = potcar_info(potcar)["element"]
    shutil.copy(potcar_dir() / potcar / "POTCAR", d / "POTCAR")
    (d / "POSCAR").write_text(f"{potcar} free atom\n1.0\n {box} 0 0\n 0 {box} 0\n 0 0 {box}\n{el}\n1\nCartesian\n 0 0 0\n")
    (d / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")
    if conf == "nonspin":
        spin = "ISPIN = 1\nISMEAR = 0; SIGMA = 0.2"
    elif conf == "spin":
        mom = hund_moment(el, shell_list(potcar)["valence"])
        spin = f"ISPIN = 2; MAGMOM = {mom}\nISMEAR = 0; SIGMA = 0.05"
    elif conf == "s1":
        # 2015's state (M2 check): d^(N-1) s^1 high spin, N = outer d + s electrons; fixed moment forces it
        mom = s1_moment(el)
        spin = f"ISPIN = 2; MAGMOM = {mom}; NUPDOWN = {mom}\nISMEAR = 0; SIGMA = 0.05"
    else:
        raise ValueError(conf)
    (d / "INCAR").write_text(f"""SYSTEM = {potcar} free atom ({conf}, pbefit)
PREC = Accurate; ENCUT = {encut_for(potcar)}; LREAL = .FALSE.
ISYM = 0
{spin}
EDIFF = 1E-7; NELM = {nelm}; ALGO = {algo}
NBANDS = {nbands_for(potcar)}
LWAVE = .TRUE.; LCHARG = .FALSE.
""")
    return d


def converged(d: Path) -> bool:
    out = d / "OUTCAR"
    if not out.exists():
        return False
    txt = out.read_text(errors="replace")
    nelm = int(re.search(r"NELM\s*=\s*(\d+)", txt).group(1))
    its = len(re.findall(r"^\s*LOOP:", txt, re.M))
    return "aborting loop because EDIFF is reached" in txt and its < nelm


def run(d: Path, threads: int = 10, ranks: int = 1) -> float:
    t0 = time.time()
    with open(d / "vasp.log", "w") as log:
        subprocess.call(VASP_CMD.format(threads=threads, ranks=ranks), shell=True, cwd=d, stdout=log, stderr=subprocess.STDOUT)
    return time.time() - t0


def run_atom(potcar: str, conf="nonspin", box=20.0, threads=10, tag="", ranks=1) -> dict:
    """Setting A, then setting B if A does not converge. Returns a status dict.
    The fixed-moment s1 atoms (M2 check only) go straight to B: A did not converge for Ti/Pd s1 in 60 steps."""
    t, setting = 0.0, "B"
    if conf != "s1":
        d = setup(potcar, conf, box, tag)
        t = run(d, threads, ranks)
        setting = "A"
    if conf == "s1" or not converged(d):
        d = setup(potcar, conf, box, tag, algo="All", nelm=250)
        t += run(d, threads, ranks)
        setting = "B"
    ok = converged(d)
    (d / "pbefit_status.txt").write_text(f"converged={ok} setting={setting} seconds={t:.0f}\n")
    return {"potcar": potcar, "dir": str(d), "converged": ok, "setting": setting, "seconds": t}
