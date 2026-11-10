#!/usr/bin/env python3
"""Isolated end-to-end benchmarks; each backend/repeat gets a fresh process.

Example: PYTHONPATH=python benchmarks/run_benchmarks.py <small-run> --backend cpu --repeat 3 --warmup 1
wall_s measures Calculation.open(...).analyze(...), excluding imports and optional warmups.
process_wall_s also includes interpreter startup, imports, warmups and serialization.
Peak RSS includes warmups of the same case, never another backend or dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from prowl import _core
from prowl.calculation import Calculation
from prowl.config import BondingConfig


def peak_rss_mib() -> float:
    """Read this worker's high-water mark, using the platform's ru_maxrss units."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def one(run: str, backend: str, max_distance: float) -> tuple[dict, dict]:
    """Measure one analysis in this worker; return metadata and parity arrays separately."""
    cfg = BondingConfig()
    cfg.max_distance = max_distance
    start = time.perf_counter()
    res = Calculation.open(run).analyze(cfg, backend=backend)
    wall = time.perf_counter() - start
    r, inp = res.result, res.input
    reader = _core.WavecarReader(str(res.provenance["inputs"]["wavecar"]["path"]))
    h = reader.header
    return {
        "run": run, "backend": backend, "device": res.provenance["backend"]["device"], "wall_s": wall,
        "phases_s": r.diagnostics["timings_s"], "peak_rss_mib": peak_rss_mib(),
        "setup_counts": r.diagnostics.get("setup_counts", {}),
        "worker_pid": os.getpid(),
        "sizes": {"N_atom": len(inp.frac), "N_k": h["nkpts"], "N_band": h["nbands"],
                  "N_G": reader.kpoint(0, 0)["nplw"], "N_orb": len(inp.orbital_labels),
                  "encoding": reader.layout["encoding"], "N_bonds": len(r.bonds)},
        "band_window": r.diagnostics["band_window_note"],
        "max_paw_norm_deviation": r.diagnostics["max_paw_norm_deviation"],
    }, {"icohp": np.asarray(r.icohp), "cohp": np.asarray(r.cohp)}


def sample_case(run: str, backend: str, max_distance: float, warmup: int, array_path: Path) -> dict:
    """Spawn one measured sample, including no state from earlier cases."""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", run, backend,
               "--max-distance", str(max_distance), "--warmup", str(warmup), "--arrays", str(array_path)]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - start
    metadata = json.loads(result.stdout)
    metadata["process_wall_s"] = wall
    metadata["warmup"] = warmup
    return metadata


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--backend", action="append", choices=("cpu", "cuda"), help="repeat to select several backends")
    ap.add_argument("--repeat", type=int, default=1, help="fresh worker processes per backend (default: 1)")
    ap.add_argument("--warmup", type=nonnegative, default=0, help="unmeasured analyses of the same case per worker (default: 0)")
    ap.add_argument("--max-distance", type=float, default=2.5)
    ap.add_argument("--json")
    ap.add_argument("--worker", nargs=2, metavar=("RUN", "BACKEND"), help=argparse.SUPPRESS)
    ap.add_argument("--arrays", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        if not args.arrays:
            ap.error("worker needs --arrays")
        run, backend = args.worker
        start = time.perf_counter()
        for _ in range(args.warmup):
            one(run, backend, args.max_distance)
        warmup_s = time.perf_counter() - start
        metadata, arrays = one(run, backend, args.max_distance)
        metadata["warmup_s"] = warmup_s
        np.savez(args.arrays, **arrays)
        metadata["peak_rss_mib"] = peak_rss_mib()  # include parity-array serialization in the worker peak
        print(json.dumps(metadata))
        return 0
    if not args.runs or args.repeat < 1:
        ap.error("provide at least one run and --repeat >= 1")
    available = _core.available_backends()
    backends = list(dict.fromkeys(args.backend or available))
    if any(b not in available for b in backends):
        ap.error(f"requested backend unavailable; available: {available}")
    output = []
    with tempfile.TemporaryDirectory(prefix="prowl-benchmark-") as directory:
        for run in args.runs:
            cases, arrays = {}, {}
            for backend in backends:
                samples = []
                for repeat in range(args.repeat):
                    target = Path(directory) / f"{backend}-{repeat}.npz"
                    samples.append(sample_case(run, backend, args.max_distance, args.warmup, target))
                    if repeat == 0:
                        with np.load(target) as saved:
                            arrays[backend] = {name: saved[name].copy() for name in saved.files}
                    target.unlink()
                median = dict(samples[0])
                for key in ("wall_s", "process_wall_s", "warmup_s", "peak_rss_mib"):
                    median[key] = statistics.median(sample[key] for sample in samples)
                median["phases_s"] = {key: statistics.median(s["phases_s"][key] for s in samples)
                                      for key in samples[0]["phases_s"]}
                median["samples"] = samples
                median.pop("worker_pid")
                cases[backend] = median
            for backend, case in cases.items():
                for name, label in (("icohp", "max_abs_icohp_dev_vs_cpu_ev"), ("cohp", "max_abs_cohp_dev_vs_cpu")):
                    case[label] = (float(np.abs(arrays[backend][name] - arrays["cpu"][name]).max(initial=0.0))
                                   if "cpu" in arrays else None)
                print(f"{run} [{backend}] analysis {case['wall_s']:.3f}s; process {case['process_wall_s']:.3f}s; "
                      f"peak RSS {case['peak_rss_mib']:.0f} MiB; repeats {args.repeat}, warmups {args.warmup}")
                output.append(case)
    if args.json:
        Path(args.json).write_text(json.dumps({"host": platform.node(), "machine": platform.machine(),
                                              "results": output}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode)
