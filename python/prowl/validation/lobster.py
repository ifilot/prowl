"""Readers for LOBSTER text outputs, used only to *compare* against a locally run,
licensed LOBSTER. No LOBSTER code or data is used by PROWL's computations."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_matrices(path: str | Path) -> dict:
    """Parse transferMatrices.lobster / coefficientMatrices.lobster.

    Returns {"labels": [...], "kpoints": [(spin, k, kfrac)], "matrices": list of complex arrays
    (norb x nbands) in file order, "by_key": {(spin, k): matrix}} with 1-based spin/k labels.
    NOTE: LOBSTER may write k-points out of order; always use the labels (``by_key``).
    """
    lines = Path(path).read_text().splitlines()
    out = {"labels": None, "kpoints": [], "matrices": []}
    i = 0
    while i < len(lines):
        m = re.match(r".*matrix for spin (\d+) kpoint (\d+) at (.*)$", lines[i])
        if not m:
            m2 = re.match(r".*matrix for kpoint (\d+) at (.*)$", lines[i])
            if not m2:
                i += 1
                continue
            m = re.match(r"(\d+) (\d+) (.*)$", f"1 {m2.group(1)} {m2.group(2)}")
        part = lines[i + 1].strip()
        header = lines[i + 2].split()
        nb = len(header) - 1
        rows, labels = [], []
        j = i + 3
        while j < len(lines) and lines[j].strip() and "matrix for" not in lines[j]:
            t = lines[j].split()
            labels.append(t[0])
            rows.append([float(x) for x in t[1:]])
            j += 1
        arr = np.array(rows)
        key = (int(m.group(1)), int(m.group(2)))
        if part.startswith("Real"):
            # Gamma blocks may be written without an imaginary part
            out["matrices"].append(arr.astype(complex))
            kf = [float(x) for x in m.group(3).split()[:3]]
            out["kpoints"].append((key[0], key[1], kf))
            out["labels"] = labels
        else:
            assert out["kpoints"] and out["kpoints"][-1][:2] == key, "imaginary block without real block"
            out["matrices"][-1] = out["matrices"][-1] + 1j * arr
        i = j
    out["by_key"] = {(sp, k): M for (sp, k, _), M in zip(out["kpoints"], out["matrices"])}
    return out


def read_icohplist(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if len(t) >= 8 and t[0].isdigit():
            rows.append({"index": int(t[0]), "atom_a": t[1], "atom_b": t[2], "distance": float(t[3]),
                         "translation": tuple(int(x) for x in t[4:7]), "icohp": float(t[7])})
    return rows


def read_cohpcar(path: str | Path) -> dict:
    """COHPCAR.lobster: returns energies (relative to E_F) and the column blocks."""
    lines = Path(path).read_text().splitlines()
    # header: (interactions + 1) nspin npts emin emax efermi
    ninter, ispin, npts = (int(float(x)) for x in lines[1].split()[:3])
    efermi = float(lines[1].split()[5])
    data = np.array([[float(x) for x in l.split()] for l in lines[2 + ninter:2 + ninter + npts]])
    if data.shape[0] != npts:
        raise ValueError(f"COHPCAR has {data.shape[0]} data rows, header says {npts}")
    return {"n_interactions": ninter, "spin": ispin, "efermi": efermi, "energies": data[:, 0], "data": data[:, 1:],
            "labels": [l.strip() for l in lines[2:2 + ninter]]}
