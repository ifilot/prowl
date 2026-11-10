# PROWL — PROjecting Wavefunctions onto Local orbitals

PROWL projects VASP PAW wavefunctions onto Slater-type atomic orbitals and computes COHP/ICOHP,
COOP/ICOOP, COBI/ICOBI, projected DOS, Mulliken/Löwdin charges and quality diagnostics. It reads
standard (`vasp_std`) and Γ-only (`vasp_gam`) WAVECARs. The C++20 core has an FP64 CPU backend and
a native CUDA backend; the user-facing parts are a Python API and a CLI. Results can be exported
in LOBSTER's file formats.

Version 0.1.0.

## Build

Requirements: CMake ≥ 3.24, a C++20 compiler, Eigen 3, OpenBLAS (CBLAS), nlohmann-json, Catch2 v3,
Python ≥ 3.11 with `nanobind`, `numpy`, `h5py`, `pyyaml`, `threadpoolctl`. Optional: CUDA Toolkit ≥ 12.8.

```bash
python -m venv .venv && . .venv/bin/activate
pip install nanobind scikit-build-core numpy scipy h5py pyyaml pytest mpmath threadpoolctl

# CPU-only build and tests
cmake --preset cpu-release -DPython_EXECUTABLE=$(which python)
cmake --build --preset cpu-release
ctest --preset cpu-release
pytest

# CUDA build (RTX 50 series = sm_120; PROWL_CUDA_ARCHITECTURES="80;90;120" adds A100/H100)
cmake --preset cuda-release -DPython_EXECUTABLE=$(which python) -DCUDAToolkit_ROOT=$CUDA_HOME
cmake --build --preset cuda-release

# or install the Python package (distribution name prowl-dft, import name prowl)
pip install .
```

Developer builds put the extension module into `python/prowl/`; use `PYTHONPATH=python` or
`pip install -e .`.

## Run

```bash
prowl inspect  ./vasp-run                         # what is there, is it analysable?
prowl validate ./vasp-run --config bonding.yaml   # inputs and strict config check
prowl plan     ./vasp-run --config bonding.yaml   # dry run: sizes, memory, streaming blocks
prowl run      ./vasp-run --config bonding.yaml --backend cuda
prowl run      ./vasp-run --profile lobster       # LOBSTER's conventions, no config file
prowl export   ./vasp-run/bonding.h5 --format lobster --output ./lobster-like
prowl plot     ./vasp-run/bonding.h5 --kind cohp --bonds 0,1 --output cohp.png
```

```python
from prowl.calculation import Calculation
from prowl.config import BondingConfig

res = Calculation.open("./vasp-run").analyze(BondingConfig.from_file("bonding.yaml"), backend="auto")
print(res.quality_report())
res.save("bonding.h5")
```

VASP requirements: `LWAVE = .TRUE.`, `ISYM = -1` (or 0), a Γ-centred mesh or Γ only,
`LREAL = .FALSE.` recommended, and `NBANDS` ≥ the number of basis functions in the cell
(≥ 1.5× recommended; `prowl plan` prints the numbers).

## Tests with real VASP data

POTCARs, WAVECARs and LOBSTER outputs are never committed. The real-data tests are skipped unless
the fixtures exist; generate them from the recipes in `tests/private_vasp_manifest/recipes`:

```bash
tools/run_vasp_fixture.py c_diamond_k7 --runs ./runs --vasp-bin /path/to/vasp/bin --np 8
PROWL_TEST_RUNS=./runs pytest
```

## Layout

`src/` C++ core and bindings · `python/prowl/` Python package and CLI · `data/basis/` basis sets
(Koga HF, `paw-pbe-sto-1`, generated polarization functions) · `tools/` generators of the basis
data and the test fixtures · `benchmarks/` performance runner · `tests/` C++ and Python tests.

## Licence

Code: GPL-3.0-or-later (`LICENSE`). PROWL's own basis data (`paw-pbe-sto-1` and the generated
polarization functions): CC BY-SA 4.0 (`data/basis/paw-pbe-sto-1/LICENSE.md`). The Koga basis data
are redistributed from the CC0 AtomicOrbitals project; the upstream tables are vendored verbatim
in `data/basis/koga/upstream/`. `tools/basisfit/data/` contains a pymatgen basis table (MIT,
`LICENSE.pymatgen`).

PROWL continues the COBALT development code (imported at COBALT commit e060b93).
