# Python interface

The public starting points are `Calculation` and `BondingConfig`. The reference
below is maintained explicitly so documentation can build without importing the
native extension or installing scientific runtime dependencies.

## Analyze and save

```python
from prowl import BondingConfig, Calculation

cfg = BondingConfig.from_file("bonding.yaml")
calc = Calculation.open("./vasp-run")
inspection = calc.inspect()
if not inspection.ok:
    raise RuntimeError(inspection.summary())

print(calc.plan(cfg))
result = calc.analyze(cfg, backend="cpu")
print(result.quality_report())
result.save("./vasp-run/bonding.h5")
```

Unlike CLI `run`, Python does not automatically save the result. `save()` uses
the path you pass; it does not automatically apply `cfg.output_path`.

## Configuration

| Interface | Purpose |
| --- | --- |
| `BondingConfig()` | Default configuration with flat attributes. |
| `BondingConfig.for_profile(name)` | Build a `prowl` or `lobster` configuration. |
| `BondingConfig.from_file(path)` | Load a nested YAML file using the strict schema. |
| `BondingConfig.from_dict(mapping)` | Parse a mapping in the same nested input schema. |
| `cfg.to_dict()` | Return the resolved flat settings for inspection/provenance. |

For example, to change the cutoff and request bond-frame contributions:

```python
cfg = BondingConfig.for_profile("prowl")
cfg.max_distance = 2.5
cfg.orbital_frame = "bond"
```

Use `from_dict()` or `from_file()` when you want schema validation. Direct
attribute changes do not invoke the input schema validator.

## Calculation

```text
Calculation.open(run_dir, *, wavecar=None, structure=None, potcar=None, metadata=None)
```

Opens a run directory. Optional file overrides are relative to that directory
unless absolute. `metadata` names the vasprun.xml file. Large wavefunction
projection is deferred until analysis.

| Method | Result |
| --- | --- |
| `calc.inspect()` | `InspectReport`: `.ok`, `.issues`, `.summary()` and `.to_dict()`. |
| `calc.plan(config=None)` | Dictionary of sizes, basis selection, recommendations and memory estimates. |
| `calc.analyze(config=None, backend=None, progress=None, reuse=None)` | `BondingResult`; `reuse` is an optional previous HDF5 result. |

`backend=None` follows the configuration. Explicit values are `cpu`, `cuda` and
`auto`. Invalid inputs or unsupported cases raise `prowl.ProwlError`; plan and
inspection results should also be checked for their reported errors.

## Result

`BondingResult` provides `.quality_report()`, `.save(path)`, `.provenance`,
`.inspect`, `.input` and `.result`. The latter is an `AnalysisResult` containing
the computed arrays:

| Attribute | Shape or meaning |
| --- | --- |
| `energies` | `(energy,)`, according to the selected reference |
| `bonds` | `Bond` objects with zero-based `atom_a`, `atom_b`, translation `R`, and distance |
| `cohp`, `coop`, `cobi` | `(spin, bond, energy)` |
| `icohp`, `icoop`, `icobi` | `(spin, bond)` |
| `pdos` | `(spin, orbital, energy)` |
| `pdos_integral` | `(spin, orbital)` |
| `mulliken_charge`, `lowdin_charge` | One charge per atom |
| `diagnostics` | Quality measurements, warnings and timing information |

Bond-frame calculations additionally provide `bond_axes`, `cohp_channel` and
`icohp_channel`. For shell-pair sums from an open HDF5 file, use
`prowl.io.orbital_resolved_icohp(file, bond_index)`.

The [results guide](../guide/results.md) explains spin counting, signs and the
on-disk schema.
