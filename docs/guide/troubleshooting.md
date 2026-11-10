# Troubleshooting

When something goes wrong, narrow it down step by step. Each command checks a
little more than the one before it:

1. `prowl inspect ./vasp-run`: are the VASP files present, consistent and supported?
2. `prowl validate ./vasp-run --config bonding.yaml`: is the configuration valid?
3. `prowl plan ./vasp-run --config bonding.yaml`: are there enough bands and memory?

Use the same configuration file you intend to run. Errors are printed with a
short code, such as `kmesh_not_full`, which you can look up in the table below.
For PROWL input or analysis errors, the command exits with code 2, so a script
or job file can detect the failure.

## Common messages

| Message or symptom | Meaning and next action |
| --- | --- |
| `missing_input` | A required file is absent. Check WAVECAR, POTCAR, vasprun.xml, and a nonempty CONTCAR or POSCAR. |
| `lattice_mismatch`, `species_mismatch`, `metadata_mismatch` | Inputs disagree. Restore files from the same calculation; confirm the final structure. |
| `eigenvalue_mismatch` | WAVECAR and metadata do not describe the same electronic state. Check continuation/MD outputs or run a consistent static calculation. |
| `unsupported_noncollinear` or `unsupported_spinor` | SOC/noncollinear projection is unavailable. Use a supported calculation if scientifically appropriate. |
| `kmesh_not_full` | VASP used symmetry to reduce the k-points, or the mesh is not a uniform Γ-centred grid. Rerun with `ISYM = -1` and a Γ-centred mesh (see [VASP settings](../getting-started/vasp-inputs.md#what-each-setting-does)). |
| Too few bands or repair needs more reliable bands | Increase `NBANDS` in VASP and rerun the calculation. See [how many bands?](../getting-started/vasp-inputs.md#how-many-bands) |
| PAW norm identity violated | Investigate potential/wavefunction consistency, LREAL, and encoding. Keep strict validation enabled while diagnosing. |
| High charge spilling | The basis misses part of the occupied states. Check the selected shells in `prowl plan`; try [polarization shells](choosing-an-analysis.md#add-polarization-shells) or the alternative library, with enough bands. |
| Overlap rank deficiency | The local basis is nearly linearly dependent. Reconsider redundant functions; adding more orbitals can worsen it. |
| Bond not resolvable / periodic-image aliasing | The k-mesh is too coarse to tell this bond apart from a periodic copy (see the {term}`Born–von Kármán supercell`). Use a denser k-mesh or a shorter cutoff. |
| Empty bond list | Check distances and cutoff, or supply explicit pairs. Plotting index 0 requires at least one selected bond. |
| Out of memory | Read the plan, reduce working buffers, or reduce basis/output size where scientifically justified. The working-memory budget does not cap total memory. |

## A successful run still needs interpretation

Do not treat completion as a scientific acceptance test. Review the
[quality report](../science/quality.md), investigate window sensitivity and
compare results under physically justified changes to k-mesh, basis and band
count. A low spilling value alone does not establish reliable bond energies.

## Preparing a useful issue report

Include the PROWL version, installation method, operating system, CPU/GPU backend,
exact command, configuration and full error message. Add the inspection and plan
output when available. Share a minimal recipe or synthetic reproducer rather
than licensed POTCARs or private wavefunction files.
