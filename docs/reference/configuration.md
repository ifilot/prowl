# Configuration reference

Configuration files use YAML schema version 1. The tables below use dotted paths
to identify **nested YAML keys**. Defaults are those of the `prowl` profile.
Unknown keys are rejected.

```{literalinclude} ../examples/bonding.yaml
:language: yaml
```

## Profile and basis

| Key | Default | Accepted values and meaning |
| --- | --- | --- |
| `schema_version` | `1` | Schema version; currently only 1. |
| `profile` | `prowl` | `prowl` sets one-center PAW and automatic window repair; `lobster` sets one-center PAW and `norb`. Explicit numerical keys override profile defaults. |
| `basis.source` | `koga_hf_sto` | Built-in `koga_hf_sto` or `paw-pbe-sto-1`; other names require a library. |
| `basis.library` | Built-in library | Path to a custom STO JSON library. Relative paths are resolved from the process working directory. |
| `basis.selection` | `zval_valence` | POTCAR-matched valence shells, or a mapping such as `{C: [2s, 2p]}` covering every element. |
| `basis.polarization` | `none` | `none`, `all`, or an element-to-shell mapping. Do not request a shell twice. |
| `basis.polarization_library` | Library selected for the basis | Optional custom polarization-library path. |

## Bonds and spectra

| Key | Default | Accepted values and meaning |
| --- | --- | --- |
| `analysis.bond_selection.mode` | `distance_cutoff` | `distance_cutoff` or `pairs`. |
| `analysis.bond_selection.min_distance_angstrom` | `0.0` | Exclusive lower bound for the distance search. |
| `analysis.bond_selection.max_distance_angstrom` | `3.0` | Inclusive upper bound in Å. Choose for your system. |
| `analysis.bond_selection.pairs` | `[]` | Explicit `[atom_a, atom_b, [Rx, Ry, Rz]]` rows with one-based atom numbers; translation defaults to zero if omitted. |
| `analysis.energy.reference` | `fermi` | `fermi` subtracts E_F; `absolute` retains the input energy reference. |
| `analysis.energy.min_ev` | `-20.0` | Lower end of the spectral grid in eV. |
| `analysis.energy.max_ev` | `10.0` | Upper end of the spectral grid in eV. |
| `analysis.energy.points` | `3001` | Number of grid points; use a meaningful increasing range and at least two points. |
| `analysis.energy.broadening.type` | `gaussian` | `gaussian` or native `tetrahedron`. |
| `analysis.energy.broadening.sigma_ev` | `0.05` | Gaussian standard deviation in eV; use a positive value. |
| `analysis.window_sensitivity` | `true` | Evaluate an alternative band window where available. |
| `analysis.orbital_frame` | `global` | `global` Cartesian frame, or common `bond` frame for both atoms with z′ along A → B. |
| `analysis.observables` | `[cohp, icohp, coop, pdos, mulliken, lowdin]` | Accepted and recorded, but currently does not filter the analysis: the pipeline computes the standard result set, including COBI. |

## Numerical settings

| Key | Default | Accepted values and meaning |
| --- | --- | --- |
| `numerical.projection` | `paw_onsite` | One-center `paw_onsite`, or experimental one- plus two-center `paw_full`. |
| `numerical.band_window` | `auto` | `auto`, `repaired`, `norb`, `frozen`, `all`, or an integer retained count. See [method and conventions](../science/method.md). |
| `numerical.precision` | `accurate` | Only `accurate` (FP64) is supported by the configuration validator. |
| `numerical.backend` | `auto` | `auto`, `cpu` or `cuda`; CLI `--backend` overrides it. |
| `numerical.memory_budget_gb` | `auto` | Positive number or `auto` (2 GiB working budget); this is not a total-memory cap. |
| `numerical.strict_validation` | `true` | Stop on a PAW norm deviation above tolerance. Keep enabled for routine work. |
| `numerical.paw_norm_tolerance` | `0.0001` | Tolerance for the PAW norm consistency check. |
| `numerical.paw_norm_check` | `sampled` | `sampled`, with escalation when needed, or `full`. |

## Output

| Key | Default | Meaning |
| --- | --- | --- |
| `output.path` | `bonding.h5` | CLI output path relative to the run directory, unless absolute. `--output` overrides it. |
| `output.write_intermediate_projection` | `false` | Save projected amplitudes for a later compatible `--reuse`; increases file size. |

The schema also accepts `input.wavecar`, `input.structure`, `input.potcar` and
`input.metadata`, but the current configuration loader does **not** apply these
keys to the calculation. Use the standard filenames for CLI runs, or pass the
file overrides to `Calculation.open` in Python.

## YAML and Python names differ

The YAML input is nested; `BondingConfig` exposes flat Python attributes. For
example, YAML `analysis.energy.min_ev` corresponds to `cfg.emin`, and
`numerical.strict_validation` corresponds to `cfg.strict`. `to_dict()` returns
the resolved flat record for provenance; it is not a nested YAML input template
and is not a round-trip argument for `from_dict()`.
