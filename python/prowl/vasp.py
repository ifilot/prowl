"""VASP run-directory metadata (vasprun.xml) and consistency checks.

Only metadata is read here; wavefunctions are read by the native WAVECAR reader.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import _core


def sha256_file(path: str | Path, limit: int | None = None) -> str:
    """SHA-256 of a file (or of its first ``limit`` bytes, for very large WAVECARs)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            if limit is not None and n + len(chunk) > limit:
                h.update(chunk[: limit - n])
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest()


@dataclass
class VasprunInfo:
    path: Path
    ispin: int
    nelect: float
    nbands: int
    isym: int | None
    lreal: str | None
    lsorbit: bool
    lnoncollinear: bool
    efermi: float
    kpoints: np.ndarray  # (nk, 3) fractional
    weights: np.ndarray  # (nk,) normalised to 1
    eigenvalues: np.ndarray  # (nspin, nk, nbands)
    occupations: np.ndarray  # (nspin, nk, nbands)
    generator: dict = field(default_factory=dict)
    version: str = ""
    nsw: int = 0
    ibrion: int | None = None


def _param(root: ET.Element, name: str):
    for el in root.iter():
        if el.tag in ("i", "v") and el.get("name") == name:
            return el.text.strip() if el.text else ""
    return None


def read_vasprun(path: str | Path) -> VasprunInfo:
    """Parsed vasprun.xml (cached per path, size and mtime: inspect() and analyze() parse it once)."""
    path = Path(path)
    st = path.stat()
    return _read_vasprun_cached(str(path.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=4)
def _read_vasprun_cached(path: str, size: int, mtime_ns: int) -> VasprunInfo:
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:  # truncated / still running
        raise _core.ProwlError(f"CorruptFile: vasprun.xml cannot be parsed ({e}); was the run complete?") from e
    params = root.find("parameters")
    if params is None:
        raise _core.ProwlError("CorruptFile: vasprun.xml has no <parameters> block")

    def p(name, cast=str, default=None):
        v = _param(params, name)
        if v is None:
            return default
        try:
            return cast(v)
        except ValueError:
            return default

    def to_bool(s):
        return s.strip().upper().startswith("T")

    kp = root.find("kpoints")
    klist = np.array([[float(x) for x in v.text.split()] for v in kp.find("varray[@name='kpointlist']")])
    w = np.array([float(v.text) for v in kp.find("varray[@name='weights']")])
    w = w / w.sum()

    calc = root.findall("calculation")[-1]
    eig_block = calc.find("eigenvalues/array/set")
    spins = eig_block.findall("set")
    eig, occ = [], []
    for sp in spins:
        e_s, o_s = [], []
        for kset in sp.findall("set"):
            rows = np.array([[float(x) for x in r.text.split()] for r in kset.findall("r")])
            e_s.append(rows[:, 0])
            o_s.append(rows[:, 1])
        eig.append(e_s)
        occ.append(o_s)
    efermi_el = calc.find("dos/i[@name='efermi']")
    efermi = float(efermi_el.text) if efermi_el is not None else float("nan")
    gen = {i.get("name"): (i.text or "").strip() for i in root.find("generator").findall("i")} if root.find("generator") is not None else {}
    incar = root.find("incar")
    incar_d = {i.get("name"): (i.text or "").strip() for i in incar.findall("i")} if incar is not None else {}
    return VasprunInfo(
        path=path,
        ispin=p("ISPIN", int, 1),
        nelect=p("NELECT", float, float("nan")),
        nbands=p("NBANDS", int, 0),
        isym=int(incar_d["ISYM"]) if "ISYM" in incar_d else p("ISYM", int, None),
        lreal=incar_d.get("LREAL", p("LREAL", str, None)),
        lsorbit=to_bool(p("LSORBIT", str, "F")),
        lnoncollinear=to_bool(p("LNONCOLLINEAR", str, "F")),
        efermi=efermi,
        kpoints=klist,
        weights=w,
        eigenvalues=np.array(eig),
        occupations=np.array(occ),
        generator=gen,
        version=gen.get("version", ""),
        nsw=int(incar_d.get("NSW", "0") or 0),
        ibrion=int(incar_d["IBRION"]) if "IBRION" in incar_d else None,
    )


def detect_mesh(kpoints: np.ndarray, tol: float = 1e-6) -> tuple[tuple[int, int, int], bool, bool]:
    """Infer a uniform Gamma-containing mesh from fractional k-points.

    Returns ((n1, n2, n3), is_full_mesh, is_time_reversal_half_mesh).
    """
    n = []
    for c in range(3):
        vals = np.mod(kpoints[:, c] + tol, 1.0) - tol
        nonzero = np.abs(vals[np.abs(vals) > tol])
        n.append(int(round(1.0 / nonzero.min())) if nonzero.size else 1)
    n = tuple(n)
    grid = {tuple(np.round(np.mod(k * n, n)).astype(int) % n) for k in kpoints}
    on_grid = all(np.allclose(k * n, np.round(k * n), atol=1e-4) for k in kpoints)
    total = n[0] * n[1] * n[2]
    full = on_grid and len(grid) == total and len(kpoints) == total
    half = False
    if on_grid and not full:
        closure = set(grid) | {tuple((-np.array(g)) % n) for g in grid}
        half = len(closure) == total
    return n, full, half
