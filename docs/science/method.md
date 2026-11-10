# Method and conventions

This page describes how PROWL computes its results and which conventions it
uses. The [observables page](interpreting-results.md) explains what the
results mean chemically.

## Overview

Every analysis follows the same five steps:

1. **Build the local orbitals.** Each Slater-type atomic orbital is repeated
   over all unit cells as a {term}`Bloch sum`, so that it fits the periodic
   calculation. PROWL also computes how strongly the orbitals overlap with one
   another (the overlap matrix).
2. **Project the wavefunctions.** For every band and k-point, PROWL computes
   how much of each local orbital the VASP wavefunction contains, including
   the {term}`PAW` corrections near the nuclei.
3. **Choose the band window.** PROWL selects the bands that will be used to
   build the local picture (see [band windows](#band-windows-and-profiles)).
4. **Rebuild the Hamiltonian.** From the projections and the VASP band
   energies, PROWL constructs a Hamiltonian in the local-orbital basis, first at
   each k-point and then in real space for each selected bond.
5. **Compute the observables.** COHP, COOP, COBI, pDOS and charges follow from
   the local Hamiltonian, the coefficients and the VASP occupations.

The rest of this page gives the details of each step.

## From PAW states to local orbitals

PROWL builds Bloch sums of Slater-type atomic orbitals, evaluates the local-basis
overlap matrix, and projects VASP wavefunctions with PAW corrections. The first
term below is the overlap with the smooth plane-wave part of the wavefunction;
the sum adds the PAW correction inside the augmentation spheres. The recorded
projection convention is

$$
A_{\mu n}(\mathbf{k}) =
\langle\chi_{\mu\mathbf{k}}|\widetilde\psi_{n\mathbf{k}}\rangle
+ \sum_i
\langle\chi_{\mu\mathbf{k}}|\phi_i-\widetilde\phi_i\rangle
\langle\widetilde p_i|\widetilde\psi_{n\mathbf{k}}\rangle.
$$

The default `paw_onsite` mode uses one-center PAW terms. `paw_full` includes
two-center terms and is experimental; it changes the projection convention and
should be reported explicitly.

PROWL applies Löwdin orthonormalization to the local basis and polar
orthonormalization to projected band coefficients. With the resulting
coefficients C, the local Hamiltonian is reconstructed as

$$
H(\mathbf{k}) = C(\mathbf{k})\,
\mathrm{diag}(\varepsilon_{n\mathbf{k}})\,C(\mathbf{k})^\dagger.
$$

In words: the local Hamiltonian is the one that has the VASP band energies as
eigenvalues when expressed in the projected orbitals.

A discrete Fourier transform on the uniform mesh yields real-space terms for
selected atom/image pairs. This step is why PROWL needs the full k-point mesh.
PROWL checks whether those pairs can be distinguished within the mesh's
{term}`Born–von Kármán supercell`.

## Band windows and profiles

The **spectral energy range** controls where curves are evaluated. The **band
window** controls the states used to reconstruct the local representation.
They are separate choices.

Why is a band window needed at all? The local basis has only {term}`N_orb`
functions, while VASP usually computes more bands than that. PROWL must decide
which bands the local Hamiltonian is built from. The window always contains the
occupied states. Above the Fermi level, the choice matters: high-lying bands
often have character that the atomic orbitals cannot describe, and including
them can distort the picture. The policies below make this choice in different
ways.

| Policy | Current behavior |
| --- | --- |
| `auto` (default `prowl` profile) | Uses the lowest `N_orb` bands where they span the local space without splitting a degenerate multiplet. Otherwise keeps occupied states and minimally repairs the local completion using additional candidate band groups. |
| `repaired` | Requests the same minimal-repair policy explicitly. |
| `norb` (`lobster` profile) | Always selects the lowest `N_orb` bands. Ill conditioning is warned about; a numerically singular local span is rejected. |
| `frozen` | Preserves occupied states and selects a local complement from the reliable candidate bands. The complement can depend on NBANDS and can be ambiguous. |
| `all` | Uses the reliable candidate band range after excluding the highest band fraction and avoiding a split degenerate multiplet; it does not literally include every WAVECAR band. |
| Integer | Uses the requested retained band count, subject to the basis size and available bands. |

The default candidate pool excludes the top 15% of bands, with at least two
excluded. This is a heuristic for avoiding unreliable high bands, not proof of
convergence. Window repair needs sufficient reliable spare states. Its choices
and warnings are printed in the quality report. Once the required repair is
covered, higher unused bands do not enter that repaired representation.

The sensitivity check compares against an alternative `frozen` or `norb` window
where available. Treat a large difference as information about representation
dependence; do not hide it by selecting whichever value looks preferable.

## Bonds, spin and units

A real-space pair is `(A, B, R)`, where R translates B. Distance selection counts
each pair once. Atom numbers in YAML pairs and displayed labels are one-based;
the Python `Bond` object, HDF5 atom indices and plotting indices are zero-based.

For `ISPIN=1`, spin degeneracy is included in the stored single channel. For
`ISPIN=2`, channels are stored separately. The CLI sums the channels when printing
total ICOHP, ICOOP and ICOBI values.

Energies and ICOHP are in eV, lengths in Å, pDOS in states/eV, and integrated
overlap populations and bond indices are dimensionless. Stored COHP is negative
for bonding; plotting displays −COHP. Integrated quantities use VASP occupations.

## Comparing with LOBSTER

`--profile lobster` selects the one-center projection and lowest-`N_orb` band
window. It does not make every numerical choice identical to LOBSTER. Match the
basis, potentials, mesh, occupations, orbital conventions and normalization,
and inspect conditioning and the actual window used.

LOBSTER-format export is an interoperability feature. It preserves PROWL's bond
selection and spectra. Gaussian curves are sampled at grid points; tetrahedron
spectra use bin averages. Running integrals in curve exports are constructed from
those spectra, while integrated lists retain the exact occupation-weighted sums.
