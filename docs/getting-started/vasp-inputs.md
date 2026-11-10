# Prepare the VASP inputs

PROWL analyzes the electronic structure of a VASP calculation that has already
finished; it does not run VASP itself. Most problems new users run into come from
the VASP run rather than from PROWL. This page explains which settings matter and
why.

:::{admonition} In short
:class: tip

Run a **static** calculation (no ionic steps) on your final structure with
`LWAVE = .TRUE.`, `ISYM = -1`, `LREAL = .FALSE.`, a Γ-centred k-point mesh and
enough bands. Then keep WAVECAR, POTCAR, vasprun.xml and CONTCAR together in one
directory.
:::

## Required files

Keep the following files together in one directory:

```text
vasp-run/
├── WAVECAR       # wavefunctions (plane-wave coefficients of every band)
├── POTCAR        # the exact PAW potentials used in this run
├── vasprun.xml   # k-points, occupations, band energies and run metadata
└── CONTCAR       # final structure; POSCAR is used if CONTCAR is absent or empty
```

All files must come from the **same calculation**. PROWL compares the lattice,
species, band and spin counts, k-points and band energies across the files and
stops if they disagree. This often happens after a geometry relaxation, when
`CONTCAR` already holds the new positions but `WAVECAR` belongs to the previous
step. The simplest fix is a separate static calculation on the relaxed
structure.

## An example static calculation

The following INCAR and KPOINTS files are taken from the diamond test
calculation in the repository
(`tests/private_vasp_manifest/recipes/c_diamond_k7/`). The settings in the
second block of the INCAR are the ones that matter for PROWL; set the others as
you would for any well-converged calculation of your system.

```text
# INCAR
SYSTEM = C diamond, static run for PROWL

# Usual electronic-structure settings: converge these for your system
PREC   = Accurate
ENCUT  = 500
EDIFF  = 1E-8
ISMEAR = -5

# Settings that matter for PROWL
NSW    = 0          # static run: no ionic steps
LWAVE  = .TRUE.     # write the WAVECAR
ISYM   = -1         # keep the full k-point mesh
LREAL  = .FALSE.    # reciprocal-space projectors
NBANDS = 16         # at least N_orb; see "How many bands?" below
LCHARG = .FALSE.    # optional: the charge density is not needed
```

```text
# KPOINTS
Gamma-centred 7x7x7
0
Gamma
7 7 7
0 0 0
```

## What each setting does

| Setting | Why PROWL needs it |
| --- | --- |
| `LWAVE = .TRUE.` | Writes the WAVECAR, which contains the wavefunctions PROWL projects. |
| `ISYM = -1` (or `0`) | Real-space bonds are reconstructed from **every** k-point of a uniform mesh. With symmetry on, VASP keeps only the symmetry-unique k-points. `ISYM = 0` still applies time-reversal symmetry and stores half the mesh, which PROWL can complete itself. |
| Uniform Γ-centred mesh | The reconstruction requires a regular grid that contains Γ. Monkhorst–Pack meshes that miss Γ, and k-point paths for band structures, are not suitable. Γ-only calculations are accepted (see below). |
| `LREAL = .FALSE.` | Recommended. It evaluates the PAW projectors in reciprocal space, consistent with what PROWL uses. |
| `NBANDS` | Must be at least the number of local orbitals, {term}`N_orb`. Spare bands give PROWL room to choose a good band window. |

## How many bands?

Count the local orbitals in the cell. PROWL's default basis uses the valence
shells that match the number of valence electrons ({term}`ZVAL`) of each POTCAR.
Each shell contributes 1 (s), 3 (p) or 5 (d) functions for every atom of that
element.

| Element and POTCAR | ZVAL | Default shells | Functions per atom |
| --- | --- | --- | --- |
| C | 4 | 2s, 2p | 1 + 3 = 4 |
| O | 6 | 2s, 2p | 1 + 3 = 4 |
| Ti | 4 | 3d, 4s | 5 + 1 = 6 |

Multiply by the number of atoms and add up the totals:

- **Diamond** (2 C): N_orb = 2 × 4 = 8.
- **Rutile TiO₂** (2 Ti, 4 O): N_orb = 2 × 6 + 4 × 4 = 28.

Use **at least** N_orb bands. PROWL recommends about **1.5 × N_orb**, or
**2 × N_orb** when you add polarization shells. For rutile this gives
`NBANDS ≥ 42`; the repository's rutile test calculation uses 64. These are
planning guidelines, not convergence guarantees.

```{note}
POTCARs with semicore states (names ending in `_pv` or `_sv`) have a larger
ZVAL, and PROWL then includes the semicore shells too. With `Ti_pv` (ZVAL 10),
for example, each Ti atom gets 3p, 3d and 4s shells, or 9 functions. VASP may
also round `NBANDS` up to suit the parallelization. Check the `NBANDS` value in
the OUTCAR, which is the value that was actually used.
```

Once you have a WAVECAR, `prowl plan` reports the selected shells, N_orb, the
minimum band count and the recommended one, so you can check your estimate
before the full analysis.

## Supported calculations

| Calculation type | Supported? |
| --- | --- |
| Standard (`vasp_std`) WAVECAR | Yes |
| Γ-only (`vasp_gam`) WAVECAR | Yes |
| Non-spin-polarized (`ISPIN = 1`) | Yes |
| Collinear spin-polarized (`ISPIN = 2`) | Yes |
| Spin–orbit coupling or noncollinear (`vasp_ncl`) | No: detected and rejected |

A Γ-only calculation, often used for large supercells, can be read. However,
a single k-point may not resolve the individual bonds you are interested in
(see [periodic-image aliasing](../science/quality.md)). Check the aliasing
warnings in the quality report, and use a denser mesh if a bond cannot be
resolved.

## Inspect before analyzing

With the files in place, check that PROWL can read them:

```bash
prowl inspect ./vasp-run
```

The report lists the files found, the wavefunction format, the spin treatment
and the k-point mesh, and ends with a status line. Continue when the status is
`OK`. If it reports `NOT ANALYSABLE`, the listed error codes are explained in
[Troubleshooting](../guide/troubleshooting.md).

Inspection only confirms that the inputs are consistent and in a supported
format. The quality of the projection itself is assessed after the analysis,
in the quality report.
