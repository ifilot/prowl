"""Benchmark workers isolate peaks and honor filters, without private scientific fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys


def fake_package(tmp_path):
    package = tmp_path / "prowl"
    package.mkdir()
    (package / "__init__.py").write_text('''
from types import SimpleNamespace as NS
class Reader:
    def __init__(self, path): pass
    header = {"nkpts": 1, "nbands": 1}
    layout = {"encoding": "FullComplex"}
    def kpoint(self, s, k): return {"nplw": 1}
_core = NS(available_backends=lambda: ["cpu", "cuda"], WavecarReader=Reader)
''')
    (package / "config.py").write_text("class BondingConfig: pass\n")
    (package / "calculation.py").write_text('''
from types import SimpleNamespace as NS
import numpy as np
class Calculation:
    @classmethod
    def open(cls, run):
        obj = cls(); obj.run = run
        return obj
    def analyze(self, cfg, backend):
        # Touch pages so ru_maxrss independently distinguishes the two workloads.
        data = np.ones((64 if self.run == "big" else 1) * 1024 * 1024, dtype=np.uint8)
        assert data[-1] == 1
        r = NS(diagnostics={"timings_s": {"total": .001}, "band_window_note": "toy",
                            "max_paw_norm_deviation": 0.}, icohp=np.zeros((1, 0)), cohp=np.zeros((1, 0, 1)), bonds=[])
        return NS(result=r, input=NS(frac=[[0,0,0]], orbital_labels=["s"]),
                  provenance={"inputs": {"wavecar": {"path": "toy"}}, "backend": {"device": backend}})
''')
    return dict(os.environ, PYTHONPATH=str(tmp_path), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")


def test_worker_peak_isolation_and_repeats(tmp_path):
    env = fake_package(tmp_path)
    target = tmp_path / "results.json"
    script = Path(__file__).resolve().parents[2] / "benchmarks/run_benchmarks.py"
    subprocess.run([sys.executable, str(script), "big", "small", "--backend", "cpu", "--repeat", "2",
                    "--warmup", "1", "--json", str(target)], env=env, check=True, capture_output=True, text=True)
    cases = json.loads(target.read_text())["results"]
    assert [c["backend"] for c in cases] == ["cpu", "cpu"]
    assert cases[0]["peak_rss_mib"] > cases[1]["peak_rss_mib"] + 20
    samples = [s for c in cases for s in c["samples"]]
    assert len({s["worker_pid"] for s in samples}) == 4
    assert all(s["warmup"] == 1 and s["warmup_s"] > 0 and s["process_wall_s"] > s["wall_s"] > 0 for s in samples)
    assert all(c["max_abs_icohp_dev_vs_cpu_ev"] == 0 for c in cases)


def test_single_backend_does_not_run_an_unrequested_reference(tmp_path):
    env = fake_package(tmp_path)
    target = tmp_path / "results.json"
    script = Path(__file__).resolve().parents[2] / "benchmarks/run_benchmarks.py"
    subprocess.run([sys.executable, str(script), "small", "--backend", "cuda", "--json", str(target)],
                   env=env, check=True, capture_output=True, text=True)
    cases = json.loads(target.read_text())["results"]
    assert len(cases) == 1 and cases[0]["backend"] == "cuda"
    assert cases[0]["max_abs_icohp_dev_vs_cpu_ev"] is None
