# Choose an analysis

An analysis is defined by a few decisions:

1. **Which atomic orbitals** to describe the electrons with (the basis).
2. **Which atom pairs** to treat as bonds.
3. **Which energy range** to show, and how smooth the curves should be.
4. Optionally, whether to split each bond into **σ, π and δ** contributions.

Write these choices in a YAML file and pass it to `validate`, `plan` and `run`
with `--config`. Any key you leave out keeps its default value. Unknown or
misspelled keys are reported as errors, so typos cannot go unnoticed. The
[configuration reference](../reference/configuration.md) lists every key.

## Choose a basis

The {term}`basis set` is the set of atomic orbitals onto which PROWL projects
your wavefunctions. The default is a good starting point for most systems:

```yaml
basis:
  source: koga_hf_sto        # Slater-type orbitals from atomic Hartree–Fock
  selection: zval_valence    # valence shells matching each POTCAR
  polarization: none
```

`zval_valence` picks, for each element, the outermost occupied atomic shells
that together hold the number of valence electrons in the POTCAR
({term}`ZVAL`). For a standard carbon POTCAR (ZVAL 4) that is 2s and 2p. For a
semicore POTCAR such as `Ti_pv`, the 3p semicore shell is included too.
`prowl plan` prints the shells it selected.

### Add polarization shells

{term}`Polarization functions <polarization function>` are extra shells that
are empty in the free atom but can help describe bonding in a solid, for
example a 4p shell on iron or a 3p shell on sodium. To add every polarization
shell available for the elements in your structure:

```yaml
basis:
  source: koga_hf_sto
  selection: zval_valence
  polarization: all
```

`all` only adds shells that the library provides; elements without an entry
are left unchanged. Run `plan` again after changing the basis: more orbitals
need more VASP bands, and orbitals that are too similar can make the basis
nearly {term}`linearly dependent <linear dependence>`.

### Use the alternative library

The second built-in library, `paw-pbe-sto-1`, contains functions fitted to the
VASP PBE PAW atomic data, combined with the Koga functions where no fit is
available. It has its own polarization shells. Select it with
`basis.source: paw-pbe-sto-1`. Neither library is best for every material, so
report which one you used.

### Choose shells explicitly

You can also list the shells for every element in the structure yourself. For
a carbon-only calculation:

```yaml
basis:
  source: koga_hf_sto
  selection:
    C: [2s, 2p]
```

Each shell expands into all of its orbitals: 2p gives three functions.

```{important}
Changing the basis changes the representation, not the chemistry. When you
compare two systems, use the same basis choices for both; otherwise a
difference in ICOHP may come from the basis and not from the bonding.
```

## Select bonds

### By distance

The simplest choice is to include every atom pair within a distance range:

```yaml
analysis:
  bond_selection:
    mode: distance_cutoff
    min_distance_angstrom: 0.0
    max_distance_angstrom: 3.0
```

A pair is included when `min_distance < distance <= max_distance`. Every pair
appears once: PROWL lists A–B but not the equivalent B–A in the opposite
direction.

:::{admonition} Choosing a cutoff
:class: tip

Look up the nearest-neighbour distances in your structure, for example with a
structure viewer, and set the cutoff slightly beyond the shell of neighbours you
want to study. Alternatively, start with a generous cutoff, read the distances
in the bond table that `prowl run` prints, and narrow the range for the final
analysis. A large cutoff in a big cell can select hundreds of bonds.
:::

### Specific atom pairs

To analyze particular contacts, list them as `[atom_a, atom_b, [R1, R2, R3]]`:

```yaml
analysis:
  bond_selection:
    mode: pairs
    pairs:
      - [1, 2, [0, 0, 0]]
      - [1, 2, [1, 0, 0]]
```

Atom numbers are **one-based**, in the order of your POSCAR/CONTCAR. The
third entry is the lattice translation applied to the second atom: the first
row is atom 2 in the same cell as atom 1, and the second row is the copy of
atom 2 shifted by one lattice vector along **a**₁. The
[key concepts](../getting-started/concepts.md#bonds-in-a-periodic-crystal) page
illustrates this. The translation can be omitted, and defaults to `[0, 0, 0]`.
Explicit pairs are analyzed exactly as listed, so avoid listing the same bond twice.

## Set the energy range and smoothing

```yaml
analysis:
  energy:
    reference: fermi
    min_ev: -20.0
    max_ev: 10.0
    points: 3001
    broadening:
      type: gaussian
      sigma_ev: 0.05
```

`reference: fermi` places the Fermi level at 0 eV. `reference: absolute` keeps
the energies as VASP reports them.

Each state is smeared into a Gaussian of width `sigma_ev` (the standard
deviation, in eV). Smaller widths give sharper features but noisier curves.
Use more grid `points` when you make the width smaller, so that each peak is
still sampled by several points. `type: tetrahedron` uses the
{term}`tetrahedron method` instead of Gaussians. Whatever method you choose,
check that the curves do not change much with a denser k-mesh.

```{note}
These settings only affect the **curves**. The integrated values (ICOHP,
ICOOP, ICOBI, charges) are computed directly from the VASP occupations and do
not depend on the broadening or on the plotted energy range. The energy range
is also separate from the {term}`band window`, which decides which bands are
used to build the local picture.
```

## Split bonds into σ, π and δ contributions

```yaml
analysis:
  orbital_frame: bond
```

By default, orbital labels such as p<sub>x</sub> and p<sub>y</sub> refer to the Cartesian axes of
your structure. With `orbital_frame: bond`, PROWL instead rotates the orbitals
of both atoms into a {term}`bond frame` with the z′ axis pointing from atom A to
atom B. Each bond is then split into channels by the symmetry of the orbitals
around the bond axis:

| Channel | Orbital pairs (examples) | Character |
| --- | --- | --- |
| σ | s–s, s–p<sub>z′</sub>, p<sub>z′</sub>–p<sub>z′</sub>, d<sub>z′²</sub>–p<sub>z′</sub> | Cylindrically symmetric around the bond axis |
| π | p<sub>x′</sub>–p<sub>x′</sub>, p<sub>y′</sub>–p<sub>y′</sub>, d<sub>x′z′</sub>–p<sub>x′</sub> | One nodal plane containing the bond axis |
| δ | d<sub>x′y′</sub>–d<sub>x′y′</sub>, d<sub>x′²−y′²</sub>–d<sub>x′²−y′²</sub> | Two nodal planes containing the bond axis |
| φ | f–f | Three nodal planes (f orbitals only) |
| mixed | Pairs of different symmetry | Zero for an isolated linear molecule; can be nonzero in a crystal |

The channels add up to the total COHP. Changing the frame does not change the
total; it only reveals how the total is composed. The command-line output then
includes a per-channel ICOHP table.

## Resources and reusing projections

```yaml
numerical:
  backend: auto
  memory_budget_gb: 2.0
output:
  write_intermediate_projection: true
```

`memory_budget_gb` controls the size of the main working buffers. It is **not**
a hard limit on the total memory PROWL uses: stored projections, outputs and
libraries need memory on top of it. `prowl plan` gives an estimate of the peak.

The projection is usually the most expensive step. If you plan several
analyses of the same calculation, for example with different bond selections,
store the projection and reuse it:

```bash
prowl run ./vasp-run --config bonding.yaml --output ./vasp-run/reused.h5 \
  --reuse ./vasp-run/bonding.h5
```

The earlier result must have been written with
`write_intermediate_projection: true`. PROWL checks that it is compatible: the
same input files (by checksum), basis selection and projection mode, and enough
stored bands. If you change the basis or the VASP calculation, a new projection
is needed.

Before changing the band window or profile, read
[band windows and profiles](../science/method.md#band-windows-and-profiles).
