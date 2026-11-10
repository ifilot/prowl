"""Gate E robustness: interrupted writes and device absence (no VASP data needed)."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prowl import _core
from prowl.io import write_hdf5


def test_native_blas_scope_context_cleanup():
    for threads in (1, 2):
        with _core.BlasThreads(threads) as scope:
            assert scope is not None
            # Nested native calls inherit the active policy, even when it differs from their default.
            _core.reconstruct_k(np.eye(2, dtype=complex), np.eye(2, dtype=complex), np.array([-1., 1.]))
            with pytest.raises(RuntimeError, match="leave scope"):
                with _core.BlasThreads(threads):
                    raise RuntimeError("leave scope")
    # Different policies can be acquired after both normal and exceptional context exits.
    with _core.BlasThreads(1):
        pass


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires Linux descriptor enumeration")
def test_invalid_wavecar_does_not_leak_descriptors(tmp_path):
    path = tmp_path / "WAVECAR"
    path.write_bytes(bytes(24))
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(8):
        with pytest.raises(_core.ProwlError, match="CorruptFile"):
            _core.WavecarReader(str(path))
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_interrupted_write_leaves_no_output(tmp_path):
    # result object that fails half-way through writing (after /meta and /structure)
    inp = SimpleNamespace(lattice=np.eye(3), frac=np.zeros((1, 3)), elements=["X"], zval=np.ones(1),
                          kfrac=np.zeros((1, 3)), weights=np.ones(1), mesh=(1, 1, 1), spin_factor=2.0,
                          orbital_labels=["X_s"], orb_offsets=[0, 1])
    br = SimpleNamespace(provenance={"prowl_version": "test"}, input=inp,
                         inspect=SimpleNamespace(to_dict=lambda: {}),
                         result=SimpleNamespace(diagnostics={}))  # lacks energies/pdos/... -> AttributeError
    target = tmp_path / "bonding.h5"
    with pytest.raises(AttributeError):
        write_hdf5(br, target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_cuda_absence_is_typed():
    code = ("from prowl import _core\n"
            "assert _core.available_backends() == ['cpu']\n"
            "assert _core.make_backend('auto').name == 'cpu'\n"
            "try:\n    _core.make_backend('cuda')\nexcept _core.ProwlError as e:\n"
            "    assert 'BackendUnavailable' in str(e)\nelse:\n    raise SystemExit('no error')\n")
    import prowl
    pkg_parent = os.path.dirname(os.path.dirname(prowl.__file__))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
