# Key concepts

This page explains the ideas behind a PROWL analysis without heavy mathematics.
If you have run DFT calculations in VASP but have never done a COHP analysis,
read this page first. The [scientific background](../science/index.md) gives
the full definitions, and the {doc}`glossary <../reference/glossary>` defines
each term briefly.

## Why plane waves need to be translated into orbitals

VASP represents each electronic state (each {term}`band` at each k-point) as a
sum of **plane waves**. Plane waves are a good way to compute energies and
forces, but they belong to the whole crystal rather than to particular atoms.
A plane-wave coefficient tells you nothing about which atoms are bonded or
which orbitals take part.

Chemists usually describe bonding in terms of **atomic orbitals**, such as a
2s or 2p orbital on carbon and a 3d orbital on iron, combining into bonding and
antibonding states. PROWL connects the two descriptions. It takes every
Kohn–Sham state from your VASP calculation and expresses it as a linear
combination of atom-centred orbitals, in the spirit of LCAO (linear combination
of atomic orbitals). This step is called the {term}`projection`.

The atom-centred functions are {term}`Slater-type orbitals <Slater-type orbital>`
taken from a built-in {term}`basis set`. PROWL picks the orbitals that match
the valence electrons of each {term}`POTCAR`, so you rarely need to choose them
yourself.

:::{admonition} An analogy
:class: tip

Projection resembles writing a vector in a new coordinate system. If the new
axes span the whole space, nothing is lost. If they do not, part of the vector
falls outside the new axes. The fraction of the electron density that the local
orbitals cannot represent is called {term}`charge spilling`, and PROWL reports
it after every run.
:::

## Bonding and antibonding states

Two hydrogen atoms show the basic idea. Their 1s orbitals combine in two ways:

- **In phase** (same sign on both atoms): electron density builds up between
  the nuclei and the energy is lowered. This is the **bonding** orbital.
- **Out of phase** (opposite signs): there is a node between the nuclei and the
  energy is raised. This is the **antibonding** orbital.

Filling the bonding orbital stabilizes the pair; filling the antibonding
orbital as well cancels that stabilization. A solid has thousands of states
spread across an energy range, and each one can be bonding, antibonding or
nonbonding with respect to a particular pair of atoms. A crystal orbital
analysis answers one question at every energy: **do the states here bind atoms
A and B together, or push them apart?**

## Reading a COHP plot

The {term}`crystal orbital Hamilton population (COHP) <COHP>` answers that
question for one pair of atoms. It weights each state's contribution to the
A–B pair by the Hamiltonian matrix element between their orbitals. Bonding
interactions have negative COHP values. Most programs, PROWL included,
therefore plot **−COHP**, so that bonding contributions point to the right.

```{raw} html
<figure class="prowl-figure">
<svg viewBox="0 0 600 370" role="img" aria-labelledby="cohp-fig-title cohp-fig-desc">
  <title id="cohp-fig-title">Schematic −COHP plot</title>
  <desc id="cohp-fig-desc">Energy runs vertically with the Fermi level marked as a dashed line. Bonding lobes extend to the right below the Fermi level and an antibonding lobe extends to the left above it. The region below the Fermi level is shaded as occupied.</desc>
  <defs>
    <clipPath id="cohp-right"><rect x="300" y="0" width="300" height="370"/></clipPath>
    <clipPath id="cohp-left"><rect x="0" y="0" width="300" height="370"/></clipPath>
  </defs>
  <rect class="occupied" x="70" y="160" width="460" height="160"/>
  <path id="cohp-shape" class="bond-fill" clip-path="url(#cohp-right)"
        d="M300,320 C300,300 440,292 440,270 C440,250 300,252 300,238 C300,226 390,216 390,202 C390,189 300,192 300,178 L300,150 C300,140 160,130 160,110 C160,92 300,88 300,72 L300,30 Z"/>
  <path class="antibond-fill" clip-path="url(#cohp-left)"
        d="M300,320 C300,300 440,292 440,270 C440,250 300,252 300,238 C300,226 390,216 390,202 C390,189 300,192 300,178 L300,150 C300,140 160,130 160,110 C160,92 300,88 300,72 L300,30 Z"/>
  <path class="curve"
        d="M300,320 C300,300 440,292 440,270 C440,250 300,252 300,238 C300,226 390,216 390,202 C390,189 300,192 300,178 L300,150 C300,140 160,130 160,110 C160,92 300,88 300,72 L300,30"/>
  <line class="axis" x1="300" y1="30" x2="300" y2="320"/>
  <polyline class="axis" points="70,30 70,320 530,320"/>
  <polyline class="axis" points="65,40 70,30 75,40"/>
  <line class="fermi" x1="70" y1="160" x2="530" y2="160"/>
  <text x="526" y="153" text-anchor="end">E<tspan baseline-shift="sub" font-size="10">F</tspan> = 0</text>
  <text x="56" y="175" text-anchor="middle" transform="rotate(-90 56 175)">E − E<tspan baseline-shift="sub" font-size="10">F</tspan> (eV)</text>
  <text class="bond-text" x="452" y="274">bonding</text>
  <text class="antibond-text" x="150" y="114" text-anchor="end">antibonding</text>
  <text class="muted small" x="82" y="50">empty states (above E<tspan baseline-shift="sub" font-size="9">F</tspan>)</text>
  <text class="muted small" x="82" y="308">occupied states: summed into ICOHP</text>
  <text class="antibond-text small" x="290" y="340" text-anchor="end">← antibonding</text>
  <text class="bond-text small" x="310" y="340">bonding →</text>
  <text x="300" y="362" text-anchor="middle">−COHP</text>
</svg>
<figcaption>Schematic −COHP plot, not computed data. Energy runs vertically, as in <code>prowl plot</code>. Here the states below the Fermi level are bonding and the antibonding states are empty. This is the textbook picture of a stable, well-optimized bond.</figcaption>
</figure>
```

For a real example, compare this sketch with the [analysis of carbon monoxide](../gallery/co.md),
where each peak is connected to the molecular orbital that causes it.

When you look at a −COHP plot, ask three questions:

1. **Where is the Fermi level?** States below it are occupied and contribute to
   the bond; states above it are empty.
2. **Which side are the occupied states on?** Occupied states on the right
   strengthen the A–B interaction; occupied states on the left weaken it.
3. **Is there antibonding weight just below the Fermi level?** This often
   indicates a strained or electronically unstable interaction. Removing those
   electrons (for example by oxidation) or distorting the structure might
   relieve it.

## ICOHP: one number per bond

Summing the COHP over all occupied states gives the **integrated COHP**, or
{term}`ICOHP`, in eV. It summarizes the whole −COHP plot of one atom pair in a
single number. A more negative ICOHP means a larger net bonding contribution
for that pair.

```{important}
ICOHP is **not** a bond dissociation energy and not a total-energy difference.
It is a useful relative measure. Compare ICOHP values only between
calculations that use the same basis set, band window and settings.
```

PROWL also reports related quantities:

| Quantity | What it measures | Typical use |
| --- | --- | --- |
| {term}`COOP` / ICOOP | Bonding weighted by orbital *overlap* instead of the Hamiltonian | An alternative, dimensionless bonding indicator |
| {term}`COBI` / ICOBI | A bond index derived from the density matrix | Loosely comparable to a bond order |
| {term}`pDOS` | Which orbitals contribute to the states at each energy | Identifying the orbital character of bands and peaks |
| Mulliken and Löwdin charges | How electrons are shared out among atoms | Comparing charge transfer between related structures |

The [observables page](../science/interpreting-results.md) explains each in more detail.

## Bonds in a periodic crystal

In a molecule, a bond connects two atoms. In a crystal, atom B also appears in
every neighbouring unit cell. Atom A in the reference cell can therefore bond to
several copies, or {term}`periodic images <periodic image>`, of atom B. PROWL
labels each bond with three pieces of information: atom A, atom B, and the
lattice translation **R** that moves B into the right cell.

```{raw} html
<figure class="prowl-figure">
<svg viewBox="0 0 600 240" role="img" aria-labelledby="img-fig-title img-fig-desc">
  <title id="img-fig-title">Periodic images and lattice translations</title>
  <desc id="img-fig-desc">Three unit cells side by side. Atom 1 in the middle cell bonds to atom 2 in the same cell, with translation zero zero zero, and to the copy of atom 2 in the cell to the right, with translation one zero zero.</desc>
  <rect class="grid" x="40" y="35" width="170" height="155"/>
  <rect class="grid" x="210" y="35" width="170" height="155"/>
  <rect class="grid" x="380" y="35" width="170" height="155"/>
  <line class="bond-line" x1="340" y1="135" x2="250" y2="80"/>
  <line class="bond-line-alt" x1="340" y1="135" x2="420" y2="80"/>
  <circle class="atom-b-image" cx="80" cy="80" r="11"/>
  <circle class="atom-b" cx="250" cy="80" r="11"/>
  <circle class="atom-b-image" cx="420" cy="80" r="11"/>
  <circle class="atom-a" cx="340" cy="135" r="11"/>
  <text x="340" y="164" text-anchor="middle">atom 1</text>
  <text x="250" y="60" text-anchor="middle">atom 2</text>
  <text class="muted small" x="48" y="60">image of atom 2</text>
  <text class="muted small" x="388" y="60">image of atom 2</text>
  <text class="small" x="262" y="122" text-anchor="end">[0 0 0]</text>
  <text class="small" x="398" y="122">[1 0 0]</text>
  <text x="125" y="212" text-anchor="middle">R = [−1 0 0]</text>
  <text x="295" y="212" text-anchor="middle">R = [0 0 0]</text>
  <text x="465" y="212" text-anchor="middle">R = [1 0 0]</text>
  <text class="muted small" x="295" y="230" text-anchor="middle">reference cell</text>
</svg>
<figcaption>Atom 1 interacts with atom 2 in its own cell (R = [0 0 0]) and with the copy of atom 2 one lattice vector further along <b>a</b><sub>1</sub> (R = [1 0 0]). These are two different bonds, and PROWL analyzes them separately.</figcaption>
</figure>
```

PROWL prints such a bond as `C1-C2[1 0 0]`, which means carbon atom 1, carbon
atom 2 and the translation R = [1 0 0]. A bond search with a distance cutoff
finds every pair (A, B, R) within that distance and lists each bond once.

:::{admonition} One-based and zero-based numbers
:class: note

Bond labels and YAML configuration files number atoms from **1**, in the order
of your POSCAR/CONTCAR, as VASP users expect. The bond and orbital indices used
by `prowl plot --bonds`, by Python and inside the HDF5 file start from **0**, in
the usual programming convention. So `--bonds 0` means "the first bond in the list".
:::

## What controls the quality of the result

Three choices determine whether a projection is trustworthy. PROWL checks each
of them and reports the result.

The basis set
: The local orbitals must be able to describe the occupied states. A low
  {term}`charge spilling` shows that they do. If spilling is high, the basis may
  be missing important orbitals.

The number of bands
: To build the local picture, PROWL needs at least as many VASP bands as there
  are local orbitals in the cell, and preferably extra bands as a safety margin.
  You set this through `NBANDS` in the INCAR. The
  [input guide](vasp-inputs.md) shows how to estimate it.

The k-point mesh
: Bonds are reconstructed in real space from the k-points. An
  N₁×N₂×N₃ mesh acts like a periodic supercell N₁×N₂×N₃ times the size of your
  unit cell, called the {term}`Born–von Kármán supercell`. Bonds that are long
  compared with that supercell cannot be told apart from their periodic
  images. PROWL warns you when this happens.

After every run, PROWL prints a **quality report** with these diagnostics. A
successful run does not by itself guarantee a meaningful result. The
[quality guide](../science/quality.md) explains how to read the report.

## Where to go next

- [Install PROWL](installation.md) and [prepare a VASP calculation](vasp-inputs.md).
- Follow [your first analysis](first-analysis.md) from inputs to a plot.
- See real results in the [worked examples](../gallery/index.md).
- Look up unfamiliar terms in the {doc}`glossary <../reference/glossary>`.
