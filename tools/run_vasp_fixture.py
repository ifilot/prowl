#!/usr/bin/env python3
"""Generate a private VASP reference fixture from a committed recipe.

Recipes live in ``tests/private_vasp_manifest/recipes/<name>/`` and contain only
INCAR/KPOINTS/POSCAR, a ``potcar.txt`` list of POTCAR symbols (one per species,
in POSCAR order), a ``binary.txt`` naming the VASP executable, and optionally a
``lobsterin`` for reference comparisons.

The POTCAR is assembled from the user's locally licensed dataset
(``--potcar-dir`` or ``$PROWL_POTCAR_DIR``) inside the private run directory and
is never written into the repository.  After the run, SHA-256 hashes of the
inputs/outputs and the POTCAR TITEL lines are written to ``manifest.json`` in the
run directory so that results can be cited without redistributing data.

Example::

    tools/run_vasp_fixture.py c_diamond_k7 --runs ./runs \
        --vasp-bin /path/to/vasp/bin --np 8 --lobster /path/to/lobster
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "tests" / "private_vasp_manifest" / "recipes"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble_potcar(symbols: list[str], potcar_dir: Path, dest: Path) -> list[str]:
    titels = []
    with open(dest, "wb") as out:
        for sym in symbols:
            src = potcar_dir / sym / "POTCAR"
            if not src.is_file():
                raise SystemExit(f"POTCAR for '{sym}' not found at {src}")
            data = src.read_bytes()
            out.write(data)
            for line in data.decode(errors="replace").splitlines():
                if "TITEL" in line:
                    titels.append(line.split("=", 1)[1].strip())
                    break
    return titels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("PROWL_RUNS", "runs")))
    ap.add_argument("--potcar-dir", type=Path, default=os.environ.get("PROWL_POTCAR_DIR"))
    ap.add_argument("--vasp-bin", type=Path, default=os.environ.get("PROWL_VASP_BIN"))
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--lobster", type=Path, default=os.environ.get("PROWL_LOBSTER"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    recipe = RECIPES / args.recipe
    if not recipe.is_dir():
        raise SystemExit(f"unknown recipe {args.recipe}")
    if args.potcar_dir is None or args.vasp_bin is None:
        raise SystemExit("--potcar-dir and --vasp-bin (or env vars) are required")

    run = args.runs / args.recipe
    if run.exists() and (run / "manifest.json").exists() and not args.force:
        print(f"{run} already complete; use --force to regenerate")
        return 0
    run.mkdir(parents=True, exist_ok=True)
    for name in ("INCAR", "KPOINTS", "POSCAR", "lobsterin"):
        if (recipe / name).exists():
            shutil.copy(recipe / name, run / name)
    symbols = (recipe / "potcar.txt").read_text().split()
    titels = assemble_potcar(symbols, Path(args.potcar_dir), run / "POTCAR")
    binary = (recipe / "binary.txt").read_text().strip()

    t0 = time.time()
    cmd = ["mpirun", "-np", str(args.np), str(Path(args.vasp_bin) / binary)]
    with open(run / "vasp.log", "w") as log:
        rc = subprocess.call(cmd, cwd=run, stdout=log, stderr=subprocess.STDOUT)
    t_vasp = time.time() - t0
    if rc != 0:
        print(f"VASP failed with exit code {rc}; see {run/'vasp.log'}", file=sys.stderr)
        return rc

    t_lob = None
    if args.lobster and (run / "lobsterin").exists():
        t1 = time.time()
        with open(run / "lobster.log", "w") as log:
            rc = subprocess.call([str(args.lobster)], cwd=run, stdout=log, stderr=subprocess.STDOUT)
        t_lob = time.time() - t1
        if rc != 0:
            print(f"LOBSTER failed with exit code {rc}", file=sys.stderr)

    files = {}
    for name in ("INCAR", "KPOINTS", "POSCAR", "WAVECAR", "vasprun.xml", "OUTCAR", "lobsterin"):
        p = run / name
        if p.exists():
            files[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    manifest = {
        "recipe": args.recipe,
        "binary": binary,
        "mpi_ranks": args.np,
        "potcar_titel": titels,
        "files": files,
        "wall_seconds": {"vasp": t_vasp, "lobster": t_lob},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
