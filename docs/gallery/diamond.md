# Diamond

Diamond is the textbook covalent solid: every carbon atom forms four
equivalent single bonds to its neighbours. It is also the system behind the
[first analysis](../getting-started/first-analysis.md) and the
[VASP input example](../getting-started/vasp-inputs.md#an-example-static-calculation),
so you can reproduce this page step by step.

```{figure} ../_static/gallery/diamond-light.svg
:figclass: only-light
:class: prowl-gallery
:alt: Projected DOS split into C 2s and C 2p, −COHP of the C–C bond, and a picture of the diamond structure.

PROWL analysis of diamond. **Left:** pDOS split into C 2s and C 2p.
**Middle:** −COHP of a nearest-neighbour C–C bond. **Right:** 2×2×2 primitive
cells of the structure.
```

```{figure} ../_static/gallery/diamond-dark.svg
:figclass: only-dark
:class: prowl-gallery
:alt: Projected DOS split into C 2s and C 2p, −COHP of the C–C bond, and a picture of the diamond structure.

PROWL analysis of diamond. **Left:** pDOS split into C 2s and C 2p.
**Middle:** −COHP of a nearest-neighbour C–C bond. **Right:** 2×2×2 primitive
cells of the structure.
```

## Reading the figure

In a solid, the discrete molecular levels broaden into **bands**, so the
curves are continuous instead of a few sharp peaks. Here the Fermi level lies
at the top of the valence band.

- **Every occupied state is bonding.** Below E<sub>F</sub>, the −COHP is
  positive across the whole valence band. There are no occupied antibonding
  states that would weaken the bonds.
- **All antibonding states are empty.** They form the conduction band, above
  the band gap.
- **2s at the bottom, 2p at the top.** The lowest part of the valence band has
  mostly C 2s character, the upper part almost entirely C 2p. Both mix
  throughout the band, as in sp³ bonding.

This combination of a filled bonding band and an empty antibonding band,
separated by a large gap, is the electronic reason for diamond's stability,
hardness and insulating behaviour.

## Four bonds, four translations

With a 1.6 Å cutoff, PROWL finds the four nearest-neighbour bonds of atom 1:

```text
C1-C2[-1 0 0]   d = 1.5446 Å   ICOHP = -9.626 eV   ICOBI = 0.950
C1-C2[0 -1 0]   d = 1.5446 Å   ICOHP = -9.626 eV   ICOBI = 0.950
C1-C2[0 0 -1]   d = 1.5446 Å   ICOHP = -9.626 eV   ICOBI = 0.950
C1-C2[0 0 0]    d = 1.5446 Å   ICOHP = -9.626 eV   ICOBI = 0.950
```

All four bonds go to atom 2, but each to a different
[periodic image](../getting-started/concepts.md#bonds-in-a-periodic-crystal):
one in the same cell and three in neighbouring cells. By symmetry they are
equivalent, and PROWL reproduces this to all printed digits.

## Integrated values

| Quantity | Value |
| --- | --- |
| C–C distance | 1.545 Å |
| ICOHP | −9.63 eV (σ: −9.06 eV, π: −0.57 eV) |
| ICOBI | 0.95 |
| ICOOP | 0.41 |

ICOBI is 0.95, very close to the bond order of 1 for a C–C single bond.
Almost all of the bonding is σ, as expected.

## Quality report

| Diagnostic | Value | Comment |
| --- | --- | --- |
| Occupied charge spilling | 1.0 % | The valence basis describes the occupied states well. |
| Electrons (DFT / Löwdin / Mulliken) | 8.000 / 8.000 / 8.000 | Exact agreement. |
| Band window | lowest 8 bands | Well conditioned; identical to LOBSTER's window. |
| Window sensitivity | 7.4 % | ICOHP is −9.63 eV with the default window and −10.34 eV with the frozen window. |
| Max \|PAW norm − 1\| | 1 × 10⁻⁷ | Wavefunctions and POTCAR are consistent. |

## Calculation details

The two-atom primitive cell (a = 3.567 Å), computed on a Γ-centred 7×7×7 mesh
without symmetry. The INCAR and KPOINTS are shown in the
[input guide](../getting-started/vasp-inputs.md#an-example-static-calculation),
with `PAW_PBE C 08Apr2002`. The recipe is
`tests/private_vasp_manifest/recipes/c_diamond_k7/`. The PROWL configuration is
{download}`diamond.yaml <../examples/gallery/diamond.yaml>`:

```{literalinclude} ../examples/gallery/diamond.yaml
:language: yaml
```

On a 7×7×7 mesh, the Gaussian width of 0.25 eV smooths out the noise from the
finite k-point sampling. A denser mesh would allow a narrower width.
