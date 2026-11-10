# Frequently asked questions

## Getting a calculation ready

### Can I use the output of my geometry relaxation?

It is safer not to. During a relaxation, the WAVECAR, CONTCAR and vasprun.xml
may describe slightly different geometries or electronic steps, and PROWL's
consistency checks may then fail. Run a short **static calculation**
(`NSW = 0`, `IBRION = -1`) on the relaxed structure with the
[recommended settings](../getting-started/vasp-inputs.md#an-example-static-calculation),
and analyze that run.

### Why do I have to switch off symmetry?

To build bonds in real space, PROWL combines the wavefunctions from every
k-point of a uniform mesh. With symmetry switched on, VASP computes only the
symmetry-unique k-points, so most of the mesh is missing. `ISYM = -1` keeps the
full mesh. With `ISYM = 0`, VASP still uses time-reversal symmetry (k and −k)
and stores half the mesh. PROWL can complete this half mesh itself.

### How many bands do I need?

At least as many bands as there are local orbitals in the cell, {term}`N_orb`.
Preferably use about 1.5 times as many, or 2 times with polarization functions.
For example, diamond has two carbon atoms with a 2s and a 2p shell each:

$$
N_\text{orb} = 2 \times (1 + 3) = 8, \qquad
\text{recommended } \texttt{NBANDS} \ge 1.5 \times 8 = 12.
$$

The repository's diamond test calculation uses `NBANDS = 16`. The
[input guide](../getting-started/vasp-inputs.md#how-many-bands) explains how
to count orbitals for your own system.

### Do I need a GPU?

No. The CPU backend handles everything. The optional CUDA backend speeds up
large calculations with many bands and k-points.

### Are spin–orbit coupling and noncollinear magnetism supported?

Not yet. PROWL supports non-spin-polarized calculations (`ISPIN = 1`) and
collinear spin-polarized calculations (`ISPIN = 2`). It detects SOC and
noncollinear WAVECARs and rejects them with a clear error.

## Choosing settings

### Which bond cutoff should I use?

There is no universal value. Find the nearest-neighbour distances in your
structure, for example with a structure viewer, and choose a cutoff slightly
beyond the shell of neighbours you care about. You can also start with a
generous cutoff, look at the distances `prowl run` prints for each bond, and
then narrow it down. To study only specific contacts, list them explicitly with
`mode: pairs` (see [select bonds](choosing-an-analysis.md#select-bonds)).

### Which basis set should I use?

Start with the default (`koga_hf_sto` with `zval_valence`), which selects the
valence shells matching your POTCARs. If charge spilling is high, try adding
polarization shells or the alternative `paw-pbe-sto-1` library, and check that
your conclusion does not change. Always report which basis you used.

### What is the difference between the `prowl` and `lobster` profiles?

Both use the same projection. The `lobster` profile always builds the local
picture from the lowest {term}`N_orb` bands, similar to LOBSTER's default. The
default `prowl` profile also starts from those bands but can repair the
{term}`band window` when they do not describe the local orbitals well. See
[band windows and profiles](../science/method.md#band-windows-and-profiles).

## Understanding results

### Is ICOHP a bond energy?

No. ICOHP is a useful **relative** measure of the bonding contribution of an
atom pair. It is not the energy needed to break the bond and not a total-energy
difference. Use it to compare similar bonds that were analyzed with the same
settings.

### Why is COHP negative for bonding?

The Hamiltonian matrix element between two in-phase orbitals is negative
because the electrons are stabilized. COHP inherits this sign, so bonding
contributions are negative. Plots show −COHP so that bonding appears on the
right, which most people find more intuitive.

### What charge spilling is acceptable?

PROWL does not impose a pass/fail threshold, because the acceptable value
depends on the system and on the question you are asking. As a rough guide,
spilling of a few percent or less suggests the basis captures the occupied
states well. Above 3 %, PROWL suggests polarization shells. Low spilling alone
does not prove the result is reliable; read the whole
[quality report](../science/quality.md).

### My numbers differ from LOBSTER. Is something wrong?

Not necessarily. Small differences in the basis functions, the band window,
the PAW treatment or the orbital normalization all change the integrated
values, and different broadening changes the shape of the curves. Use
`--profile lobster` and the same basis to make the comparison as close as
possible, and compare trends rather than individual values. See
[comparing with LOBSTER](../science/method.md#comparing-with-lobster).

### How do I combine spin-up and spin-down results?

For a spin-polarized run, add the two channels to get the total bond population;
the command-line output already does this. For a non-spin-polarized run, the
single stored channel already counts both spins, so do not multiply it by two.

### Why do some indices start at 0 and others at 1?

Atom numbers in bond labels and YAML files start at 1, as in VASP. Indices for
`prowl plot --bonds`, in Python and in the HDF5 file start at 0, as is usual in
programming. The [key concepts](../getting-started/concepts.md#bonds-in-a-periodic-crystal)
page shows an example.
