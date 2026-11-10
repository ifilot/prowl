# Interpret the observables

PROWL turns the plane-wave states of a VASP calculation into a description in
terms of atom-centred orbitals. From that description it computes several
quantities, each answering a slightly different question about the electronic
structure. This page explains what each one means and how to read it. For a
gentler introduction, see [key concepts](../getting-started/concepts.md).

## Notation

The formulas on this page use a few symbols:

| Symbol | Meaning |
| --- | --- |
| $\mu \in A$, $\nu \in B$ | A local orbital on atom A, and one on atom B |
| $\mathbf{R}$ | The lattice translation applied to atom B ([illustration](../getting-started/concepts.md#bonds-in-a-periodic-crystal)) |
| $n$, $\mathbf{k}$, $w_\mathbf{k}$ | Band index, k-point and k-point weight |
| $\varepsilon_{n\mathbf{k}}$, $f_{n\mathbf{k}}$ | Band energy and occupation from VASP |
| $C_{\mu n}(\mathbf{k})$ | The coefficient of orbital $\mu$ in band $n$ |
| $H_{\mu\nu}(\mathbf{R})$, $S_{\mu\nu}(\mathbf{R})$ | The Hamiltonian and overlap matrix elements between $\mu$ on A and $\nu$ on B |

The formulas are shown schematically, for one spin channel. The
[method page](method.md) describes how the coefficients and matrix elements
are obtained.

## COHP and ICOHP

The **crystal orbital Hamilton population** shows, energy by energy, how the
electronic states contribute to the interaction between two atoms:

$$
\mathrm{COHP}_{AB}(E) = \sum_{\mathbf{k}} w_\mathbf{k} \sum_n
\sum_{\mu\in A}\sum_{\nu\in B}
\mathrm{Re}\!\left[C^*_{\mu n}(\mathbf{k})\,H_{\mu\nu}(\mathbf{R})\,
e^{2\pi i\,\mathbf{k}\cdot\mathbf{R}}\,C_{\nu n}(\mathbf{k})\right]
\delta(E-\varepsilon_{n\mathbf{k}}).
$$

In words, a state contributes to the A–B COHP when it has weight on both atoms,
and the contribution is weighted by the Hamiltonian coupling between their
orbitals. When the orbitals combine **in phase**, the product of the
coefficients is positive while the coupling $H_{\mu\nu}$ is negative, so the
contribution is negative. That is why **negative COHP means bonding** and
positive COHP means antibonding. PROWL stores COHP with this sign and plots
−COHP, so that bonding points to the right.

The **integrated COHP** (ICOHP) replaces the energy-resolved delta function by
the occupation $f_{n\mathbf{k}}$, summing over all occupied states. It is
reported in eV, one value per bond.

How to use ICOHP:

- Under the same analysis settings, a more negative ICOHP indicates a larger net
  bonding contribution. This is useful for ranking similar bonds, for example
  the same bond across a series of related compounds.
- ICOHP is **not** a bond dissociation energy, not a total-energy difference,
  and not a universal bond-strength scale across different bases or band windows.
- PROWL computes ICOHP from the VASP occupations directly. It need not equal a
  numerical integral of the broadened curve in the plot up to 0 eV.

The method was introduced by
[Dronskowski and Blöchl (1993)](https://doi.org/10.1021/j100135a014);
its use with plane-wave calculations is discussed by
[Deringer, Tchougréeff and Dronskowski (2011)](https://doi.org/10.1021/jp202489s).

## COOP and ICOOP

The **crystal orbital overlap population** has the same form as COHP, with the
Hamiltonian $H_{\mu\nu}$ replaced by the orbital overlap $S_{\mu\nu}$. It is
evaluated with the original, non-orthogonal orbital coefficients.

Because overlap is positive for in-phase combinations, **positive COOP means
bonding** and negative COOP means antibonding. No sign flip is needed for
plotting. The integrated value, ICOOP, is dimensionless and is closely related
to the Mulliken overlap population between the two atoms.

COOP and COHP usually agree on which states are bonding, but they weight them
differently. COHP is energy-weighted and is often preferred for comparing
interaction strengths; COOP is a population measure.

## COBI and ICOBI

The **crystal orbital bond index** extends the Wiberg bond index of molecular
quantum chemistry to crystals. It is built from the density matrix between the
orbitals of atoms A and B. PROWL uses a two-centre, Wiberg-type definition, and
the output records its normalization.

ICOBI is dimensionless and loosely comparable to a bond order. It need not be
an integer: delocalized, multicentre and polar bonds generally give fractional
values. See [the original COBI paper](https://doi.org/10.1021/acs.jpcc.1c00718)
for the periodic formalism. Before comparing ICOBI values between programs,
check that the definitions and the spin conventions agree.

## Projected density of states

The **projected density of states** (pDOS) shows how much each local orbital
contributes to the electronic states at every energy, in states/eV. It answers
questions such as "is this peak mainly O 2p or Ti 3d?". PROWL reports the pDOS
in the {term}`Löwdin-orthonormalized <Löwdin orthonormalization>` basis, so
that the orbital contributions add up without overlap terms.

The orbital labels describe the chosen local representation. Inspect the
charge spilling before treating the pDOS as a complete account of the DFT states.

## Mulliken and Löwdin charges

These population analyses divide the electrons among the atoms:

- **Mulliken** populations work in the original, non-orthogonal basis and
  divide each overlap contribution equally between the two atoms.
- **Löwdin** populations first make the basis orthonormal and then simply count
  the electrons in each atom's orbitals.

PROWL's **atomic charge** is the POTCAR valence-electron count minus the
assigned population:

$$
q_A = Z^\text{val}_A - N_A.
$$

A positive charge means that the atom has fewer electrons than its neutral
valence configuration. These charges depend on the basis and on the
partitioning scheme. They are useful for comparing trends, for example across
a series of related compounds, but they are not formal oxidation states and not
unique physical observables.

## Compare like with like

Most of these quantities depend on the analysis settings. When you study a
trend across materials, keep the following the same for every calculation:

- the POTCARs and VASP settings;
- the basis set and selected shells;
- the band-window policy;
- equivalent atom pairs (including the translation R);
- the spin treatment.

Check the quality diagnostics for every calculation. A trend is most convincing
when it persists under reasonable changes to the k-mesh, the basis and the
band count.
