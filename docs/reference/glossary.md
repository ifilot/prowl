# Glossary

Short definitions of the terms used in this documentation. The
[key concepts](../getting-started/concepts.md) page introduces the most
important ones in context.

```{glossary}
Band
  One Kohn–Sham electronic state at a given k-point, with an energy
  (eigenvalue) and an occupation. VASP computes `NBANDS` bands at every
  k-point.

Band window
  The set of bands PROWL uses to build the local-orbital Hamiltonian. It
  always contains the occupied bands and usually some unoccupied ones. The band
  window is separate from the energy range you plot. See
  [band windows and profiles](../science/method.md#band-windows-and-profiles).

Basis set
  The collection of atom-centred functions onto which the wavefunctions are
  projected. PROWL ships two Slater-type libraries, `koga_hf_sto` (the
  default) and `paw-pbe-sto-1`.

Bloch sum
  A periodic combination of one atomic orbital over all unit cells, with a
  phase factor that depends on the k-point. It turns a localized orbital into a
  function that fits a periodic calculation.

Born–von Kármán supercell
  The periodic supercell implied by a uniform k-point mesh: an N₁×N₂×N₃ mesh
  corresponds to a supercell N₁×N₂×N₃ times the unit cell. Real-space
  interactions are only resolved within this supercell.

Bond frame
  A local coordinate system with z′ pointing from atom A to atom B. In this
  frame, orbital interactions separate into σ, π, δ and φ channels. Select it
  with `analysis.orbital_frame: bond`.

Broadening
  The smoothing applied to turn discrete band energies into continuous curves.
  PROWL uses Gaussian broadening (width `sigma_ev`) or the
  {term}`tetrahedron method`. Broadening changes the appearance of plots, not
  the integrated values.

Charge spilling
  The fraction of the electron density of the projected bands that the local
  basis cannot represent. Low spilling means the orbitals capture the states
  well. PROWL reports the occupied-band value and the value over all retained
  bands.

COBI
  Crystal orbital bond index: an energy-resolved bond index based on the
  density matrix. The integrated value, ICOBI, is dimensionless and loosely
  comparable to a bond order.

COHP
  Crystal orbital Hamilton population: the states' contributions to one atom
  pair, weighted by the Hamiltonian matrix elements between the two atoms'
  orbitals. Negative values are bonding. Plots show −COHP.

COOP
  Crystal orbital overlap population: like {term}`COHP`, but weighted by the
  overlap of the orbitals instead of the Hamiltonian. Positive values are
  bonding.

Fermi level
  The energy that separates occupied from unoccupied states (E_F). PROWL
  plots energies relative to it by default.

HDF5
  A portable binary file format for scientific data. PROWL's native result,
  `bonding.h5`, is an HDF5 file that you can read with `h5py` in Python.

ICOHP
  Integrated COHP: the COHP summed over occupied states, in eV. It gives one
  number per bond. More negative values mean a larger net bonding
  contribution under the same analysis settings. It is not a bond energy.

k-point mesh
  The grid of points in reciprocal space at which VASP solves for the
  electronic states. PROWL requires a uniform, Γ-containing mesh covering the
  full Brillouin zone.

Linear dependence
  The situation in which some basis functions can be written, nearly, as
  combinations of the others. It makes the overlap matrix ill-conditioned and
  is reported as a small minimum overlap eigenvalue.

LOBSTER
  An established program for COHP analysis from plane-wave calculations.
  PROWL can write results in LOBSTER's file formats and offers a `lobster`
  profile with similar conventions.

Löwdin orthonormalization
  A symmetric procedure that transforms overlapping atomic orbitals into an
  orthonormal set that stays as close as possible to the originals. PROWL's
  pDOS and Löwdin charges are expressed in this basis.

Mulliken population
  A way to divide electrons among atoms in a nonorthogonal basis, by sharing
  each overlap contribution equally between the two atoms involved. Löwdin
  populations do the same in the orthonormalized basis. Both depend on the basis.

N_orb
  The total number of local orbital functions in the cell. An s shell counts
  1, a p shell 3 and a d shell 5 for every atom. `prowl plan` reports it.

NBANDS
  The VASP INCAR tag that sets how many bands are computed. It must be at
  least {term}`N_orb`; more bands give PROWL room to choose a good band window.

Occupation
  The number of electrons in a band at a k-point, between 0 and 1 per spin
  channel. PROWL uses VASP's occupations to compute integrated quantities.

PAW
  Projector augmented-wave method: VASP's way of combining a smooth
  plane-wave wavefunction with atom-centred corrections near each nucleus.
  PROWL includes these corrections when it projects the wavefunctions.

Periodic image
  A copy of an atom in a neighbouring unit cell, displaced by a lattice
  translation R. A bond is identified by atom A, atom B and the translation R
  applied to B.

pDOS
  Projected density of states: the density of states resolved by local
  orbital, in states/eV. It shows which orbitals contribute at each energy.

Polarization function
  An extra basis shell beyond those occupied in the free atom, such as an empty
  4p shell on iron. It gives the basis more flexibility to describe bonding.

POTCAR
  The VASP file containing the PAW potentials. PROWL reads it for the PAW
  data and the number of valence electrons ({term}`ZVAL`) of each element.

Projection
  Expressing each plane-wave Kohn–Sham state as a combination of local atomic
  orbitals. The projection is the first step in every PROWL analysis.

Shell
  A set of orbitals sharing principal and angular quantum numbers, written
  like `2p` or `3d`. A shell expands into 1 (s), 3 (p), 5 (d) or 7 (f)
  functions.

Slater-type orbital
  An atomic orbital whose radial part decays as e^(−ζr), like the exact
  hydrogen-atom functions. PROWL's basis sets are made of these functions (STOs).

Spin channel
  In a spin-polarized calculation (`ISPIN = 2`), the spin-up and spin-down
  states are stored separately. In a non-spin-polarized one (`ISPIN = 1`),
  PROWL stores a single channel that already counts both spins.

Tetrahedron method
  A way of integrating over the Brillouin zone by interpolating band energies
  linearly inside small tetrahedra of the k-point mesh, as an alternative to
  Gaussian {term}`broadening`.

WAVECAR
  The binary VASP file containing the plane-wave coefficients of every band.
  It is PROWL's main input and is written when `LWAVE = .TRUE.`.

ZVAL
  The number of valence electrons treated explicitly by a POTCAR. PROWL uses
  it to select the default valence shells and to compute atomic charges.
```
