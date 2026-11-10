# Installation

The instructions below build PROWL from a local checkout of this repository.
The commands use a Linux or WSL shell. Native Windows and macOS installation
are not covered by this guide.

## Requirements

| Dependency | Needed for |
| --- | --- |
| Python 3.11 or newer, including development headers | Python API and CLI |
| C++20 compiler, CMake 3.24 or newer, and a build tool such as Make | Compiling the native extension |
| Eigen 3, nlohmann-json, OpenBLAS with CBLAS headers, and OpenMP | Native CPU core |
| CUDA Toolkit 12.8 or newer and a compatible NVIDIA GPU | Optional CUDA backend |
| Catch2 v3 | C++ developer tests only |

Install the native development dependencies through your operating system or
cluster modules before running pip. For example, on Ubuntu 24.04:

```bash
sudo apt-get install build-essential cmake python3-dev python3-venv \
  libeigen3-dev libopenblas-dev nlohmann-json3-dev
```

On a managed cluster, ask for equivalent compiler, Python, CMake and OpenBLAS
modules. CMake must be able to find their development headers and libraries.

## Install the CPU package

Run these commands from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m pip install matplotlib
prowl --version
prowl --help
```

`pip install .` builds the native extension, installs the Python dependencies,
and creates the `prowl` command. The distribution is named `prowl-dft`; the import
is `prowl`. Matplotlib is an additional dependency for `prowl plot`.

Re-activate the environment in each new shell. A successful version check confirms
that the command and native extension can load; it does not validate a VASP run.

## Enable CUDA

The default package build is CPU-only. To build with CUDA, use the same environment
and pass the native build options through pip:

```bash
python -m pip install . \
  -Ccmake.define.PROWL_ENABLE_CUDA=ON \
  -Ccmake.define.CUDAToolkit_ROOT=/path/to/cuda \
  -Ccmake.define.PROWL_CUDA_ARCHITECTURES=80
```

Replace the toolkit path and architecture for your machine. PROWL's build settings
identify A100 as `80`, H100 as `90`, and RTX 50-series as `120`. A toolkit must
support the architecture you request. On a machine with multiple target GPU
families, a quoted semicolon-separated list can be used, for example
`-Ccmake.define.PROWL_CUDA_ARCHITECTURES='80;90;120'`.

Select `--backend cuda` to request the CUDA backend explicitly. `--backend auto`
selects an available backend and can prefer the CPU for small per-k-point jobs.
Use the backend printed in the quality report to identify what actually ran.

## Common installation problems

| Symptom | Check |
| --- | --- |
| CMake cannot find Eigen or JSON | Install the development packages or expose the cluster module's CMake package path. |
| `cblas.h` or OpenBLAS is missing | Install OpenBLAS development headers as well as its runtime library. |
| `prowl: command not found` | Activate the environment where you installed the package. A CMake-only build does not create this Python entry point. |
| `ModuleNotFoundError: matplotlib` | Run `python -m pip install matplotlib` in that environment. |
| CUDA is unavailable | Check that the package was compiled with CUDA, the toolkit and driver are compatible, and the selected architecture matches your GPU. |

For editable installs and native unit tests, see [Development](../development.md).
