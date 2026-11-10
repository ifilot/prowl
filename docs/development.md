# Development

The [installation guide](getting-started/installation.md) covers routine use.
This page covers native tests, editable development, and maintaining the website.

## Build and test the native code

Install the native dependencies, including **Catch2 v3** for C++ tests. The
repository's preset file uses [CMake preset schema 6](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html#versions), which needs CMake 3.25 or
newer; a direct non-preset CMake configuration requires 3.24 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install nanobind scikit-build-core numpy h5py pyyaml \
  threadpoolctl pytest scipy mpmath
cmake --preset cpu-release -DPython_EXECUTABLE="$(command -v python)"
cmake --build --preset cpu-release
ctest --preset cpu-release
python -m pytest
```

The CMake development build places `_core` next to the Python sources. Pytest is
configured to find that source tree. For imports outside pytest, set
`PYTHONPATH=python`; invoke the CLI with `PYTHONPATH=python python -m prowl.cli`.
Alternatively, `python -m pip install -e '.[test]'` provides an editable package
and installs the `prowl` entry point.

Release presets enable `-march=native`; their binaries target the build CPU.
The CUDA release preset defaults to architecture `120`. Override
`PROWL_CUDA_ARCHITECTURES` for your target GPU.

## Real-data tests

POTCARs, WAVECARs and LOBSTER reference outputs are private fixtures and are not
committed. Tests requiring them are skipped when fixtures are absent. For
example, with locally licensed VASP and PAW datasets:

```bash
python tools/run_vasp_fixture.py c_diamond_k7 --runs ./runs \
  --vasp-bin /path/to/vasp/bin --potcar-dir /path/to/paw-datasets --np 8
PROWL_TEST_RUNS=./runs python -m pytest
```

These paths are placeholders. The fixture helper writes a manifest alongside
the private run. Do not add the generated licensed files to the repository.

## Build this website

The documentation environment is independent of the scientific runtime. It
does not compile PROWL, load the native module, or require VASP/CUDA.

```bash
python3 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000> in your browser. Stop the server with Ctrl+C.
Sphinx treats documentation warnings and unresolved references as build failures.
Edit Markdown in `docs/`, the theme in `docs/_static/prowl.css`, and configuration
in `docs/conf.py`. The approved owl assets live in `docs/_static/brand/`.

The homepage uses the same PyData Sphinx theme as the content pages. Search,
section navigation, mobile navigation and the light/dark toggle are provided by
the theme. The original angular owl and wordmark are used without regeneration;
CSS renders them in a light monochrome treatment in dark mode for contrast.

## Regenerate the example gallery

The figures of the [worked examples](gallery/index.md) are generated from real
VASP calculations by `tools/docs_gallery/make_gallery.py` and committed to
`docs/_static/gallery/`, so building the website needs neither PROWL nor VASP.
Regenerating them needs the PROWL runtime, Matplotlib, h5py, SciPy,
[PyTessel](https://github.com/ifilot/pytessel), the
[QASR](https://github.com/ifilot/qasr) renderer and the Inter font:

```bash
python -m pip install matplotlib scipy pytessel
PYTHONPATH=python python tools/docs_gallery/make_gallery.py \
  --runs /path/to/runs --qasr /path/to/qasr/build/qasr
```

`--runs` must contain the static runs `co_molecule`, `ch4_molecule` and
`c_diamond_k7`; the last one is generated from the recipe of the same name.
The analysis settings live in `docs/examples/gallery/`. The script writes the
figures, orbital renders and `manifest.json`, which records software versions,
VASP settings, input checksums and every value quoted in `docs/gallery/`.
After regenerating, update the quoted numbers from `manifest.json`.

## Continuous integration

`.github/workflows/ci.yml` compiles the CPU core, Python extension and C++ tests
on every push to `master`, on pull requests targeting `master`, and on manual
dispatch. It runs CTest and pytest on Ubuntu 24.04 with Python 3.11, 3.12 and
3.13. Python build/runtime/test requirements come from `pyproject.toml`.
CUDA tests and private VASP fixture tests require resources not available on
these hosted CPU runners and are skipped when those resources are absent.

The README test badge tracks push results for `master`. Its version badge reads
`project.version` from `pyproject.toml` on `master`; the licence badge identifies
the code licence, with separate dataset licences described below.

## GitHub Pages hosting

`.github/workflows/docs.yml` builds this website on pushes to `master`, pull
requests targeting `master`, and manual dispatch. A failed Sphinx build stops
publication. Pull requests only build the documentation; successful builds of
`master` upload the HTML artifact and deploy it with GitHub's Pages actions.
Deployment uses the `github-pages` environment and a job-scoped token with
`pages: write` and `id-token: write` permissions.

For the first deployment, select **Settings → Pages → Build and deployment →
Source → GitHub Actions** in the repository. Push the workflow and documentation
to `master` or manually run the Documentation workflow on `master`. The deployment
job reports the resulting site URL. Sphinx emits relative asset and navigation
links, so the site can be served under the repository's Pages subpath.
See [GitHub's custom-workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Optional Read the Docs hosting

The repository also includes `.readthedocs.yaml`, pointing to `docs/conf.py` and the
separate documentation requirements. After the source is pushed, import the
repository into your Read the Docs account, enable the versions you want to
serve, and enable pull-request previews if desired. Account connection and a
successful hosted build are required before there is a public documentation URL.
See the [Read the Docs configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).

Keep examples linked to the implementation and use `literalinclude` for shared
configuration files. When scientific defaults change, review the method,
reference and tutorial together. Add real calculation figures only with a
recorded recipe, settings and provenance.

## Licence

Code is GPL-3.0-or-later. PROWL's `paw-pbe-sto-1` and generated polarization data
are CC BY-SA 4.0. Koga tables come from the CC0 AtomicOrbitals project;
`tools/basisfit/data/` also contains an MIT-licensed pymatgen basis table.
Consult the licence files distributed with each dataset before redistribution.
