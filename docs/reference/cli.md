# Command-line reference

After installation, run `prowl --help` for commands and `prowl COMMAND --help`
for the options of one command. `prowl --version` reports the package version.

## Inspect

```text
prowl inspect RUN_DIR [--json]
```

Checks files, wavefunction encoding and consistency, spin support and mesh
recognition. Returns 0 if inspection passes and 2 otherwise.

## Validate

```text
prowl validate RUN_DIR [--config FILE]
```

Parses the strict configuration schema, inspects the run, and prints the resolved
configuration. It does not perform projection or establish numerical quality.
Without `--config`, uses default settings. Use `plan` to resolve the selected basis
and examine band requirements.

## Plan

```text
prowl plan RUN_DIR [--config FILE] [--profile {prowl,lobster}] [--json]
```

Reports selected shells, system sizes, required/recommended bands, buffer blocks,
a memory estimate and recommendations without projecting wavefunctions. A memory
estimate excludes allocator/library overhead and basis/model tables. Inspect the
recommendations even when the command exits successfully.

## Run

```text
prowl run RUN_DIR [--config FILE] [--profile {prowl,lobster}]
  [--backend {cpu,cuda,auto}] [--output FILE]
  [--max-print N] [--quiet] [--reuse H5_FILE]
```

Runs analysis, prints quality diagnostics and integrated values, and saves the
result. The default output is `RUN_DIR/bonding.h5`, subject to the configuration.
An explicit relative `--output` is relative to the current working directory.
Existing output at that path is replaced.

`--max-print` defaults to 20 bonds. `--quiet` suppresses progress reporting;
the final report is still printed. `--reuse` loads compatible saved projections
from a previous result.

`--profile` is useful without a file. If both a profile flag and configuration
are supplied, the profile must agree with the file's profile (defaulting to
`prowl` if omitted there). `--backend` overrides the configured backend.

## Plot

```text
prowl plot H5_FILE --output IMAGE
  [--kind {cohp,coop,pdos}] [--bonds INDICES] [--emin EV] [--emax EV]
```

Requires Matplotlib. The default kind is `cohp`, displayed as −COHP.
`INDICES` is a comma-separated list of zero-based bond indices, or orbital
indices for `pdos`. Without it, up to six entries are shown.

## Export

```text
prowl export H5_FILE --output DIRECTORY [--format {csv,json,lobster}]
```

Defaults to CSV. See [Results and export](../guide/results.md) for generated
files and differences between native and interchange formats.
