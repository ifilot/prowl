# Carbon monoxide

CO is the classic first example of molecular-orbital theory, and it shows every
feature of a COHP analysis in a single bond: strongly bonding, nearly
nonbonding and antibonding levels, σ and π contributions, and a polar bond.

```{figure} ../_static/gallery/co-light.svg
:figclass: only-light
:class: prowl-gallery
:alt: Projected DOS, −COHP of the C–O bond, and the Kohn–Sham orbitals of CO, each orbital connected to its energy level.

PROWL analysis of CO. **Left:** pDOS per atom. **Middle:** −COHP of the C–O
bond, with bonding contributions (right) in teal and antibonding contributions
(left) in red; the shaded region below E<sub>F</sub> is occupied.
**Right:** the Kohn–Sham orbitals of the same VASP calculation, connected to
their energies. Teal and orange mark opposite signs of the orbital; each
picture is scaled to fit its panel.
```

```{figure} ../_static/gallery/co-dark.svg
:figclass: only-dark
:class: prowl-gallery
:alt: Projected DOS, −COHP of the C–O bond, and the Kohn–Sham orbitals of CO, each orbital connected to its energy level.

PROWL analysis of CO. **Left:** pDOS per atom. **Middle:** −COHP of the C–O
bond, with bonding contributions (right) in teal and antibonding contributions
(left) in red; the shaded region below E<sub>F</sub> is occupied.
**Right:** the Kohn–Sham orbitals of the same VASP calculation, connected to
their energies. Teal and orange mark opposite signs of the orbital; each
picture is scaled to fit its panel.
```

## Reading the figure, level by level

For a molecule, VASP places the Fermi level just above the highest occupied
orbital (HOMO), so the HOMO appears at about 0 eV. The table integrates the
curves around each level; the contributions of the occupied levels add up to
the total ICOHP.

| Level | E − E<sub>F</sub> (eV) | Character in the −COHP | Contribution to ICOHP (eV) | Located mainly on |
| --- | --- | --- | --- | --- |
| 3σ | −20.3 | strongly bonding | −12.8 | O (2/3) |
| 4σ | −5.1 | weakly bonding | −0.8 | O (7/8) |
| 1π (×2) | −2.8 | strongly bonding | −7.7 | O (3/4) |
| 5σ (HOMO) | −0.1 | weakly **antibonding** | +0.9 | C (7/8) |
| 2π* (×2, LUMO) | +6.8 | strongly antibonding, **empty** | (not occupied) | C (3/4) |

Several textbook ideas appear directly in the data:

- **Polarization follows electronegativity.** The bonding 1π level lies mostly
  on oxygen, while its antibonding partner 2π* lies mostly on carbon. The pDOS
  shows this as a swap of the C and O peak heights between the two levels.
- **The HOMO is a carbon lone pair.** 5σ has almost all its weight on carbon,
  and its large lobe points away from the molecule (left, on the carbon side,
  in the picture). It is slightly antibonding for C–O, so donating these
  electrons to a metal (σ donation) does not weaken the C–O bond; if
  anything, it strengthens it slightly.
- **The LUMO weakens the bond when filled.** 2π* is strongly antibonding and
  mostly on carbon. When a metal surface donates electrons into it
  (π back-donation), the C–O bond weakens, which is why adsorbed CO has a lower
  stretching frequency than free CO.

## Integrated values

| Quantity | Value |
| --- | --- |
| C–O distance | 1.143 Å |
| ICOHP | −20.36 eV (σ: −12.64 eV, π: −7.72 eV) |
| ICOBI | 2.52 |
| ICOOP | 0.57 |
| Löwdin charges | C +0.19, O −0.19 |

The σ/π split comes from the [bond frame](../guide/choosing-an-analysis.md#split-bonds-into-σ-π-and-δ-contributions):
about 60 % of the bonding is σ and 40 % π. ICOBI is 2.52, below the formal
bond order of 3. This is expected for a polar bond, where the electrons are not
shared equally between the two atoms.

## Quality report

| Diagnostic | Value | Comment |
| --- | --- | --- |
| Occupied charge spilling | 1.8 % | The valence basis describes the occupied states well. |
| Electrons (DFT / Löwdin / Mulliken) | 10.000 / 10.000 / 10.000 | Exact agreement. |
| Band window | lowest 8 bands | Well conditioned; identical to LOBSTER's window. |
| Window sensitivity | 1.0 % | The result hardly depends on the band window. |
| Max \|PAW norm − 1\| | 4 × 10⁻⁸ | Wavefunctions and POTCAR are consistent. |

## Calculation details

A CO molecule in a 12 Å cubic box, computed at the Γ point only:

```text
PREC = Accurate ; ENCUT = 500 ; LREAL = .FALSE.
ISMEAR = 0 ; SIGMA = 0.01 ; EDIFF = 1E-7
ISYM = 0 ; NSW = 0 ; NBANDS = 32
LWAVE = .TRUE. ; LCHARG = .FALSE.
```

with `PAW_PBE C 08Apr2002` and `PAW_PBE O 08Apr2002` potentials, at the
geometry relaxed beforehand with the same settings (C–O = 1.143 Å). The PROWL configuration is
{download}`co.yaml <../examples/gallery/co.yaml>`:

```{literalinclude} ../examples/gallery/co.yaml
:language: yaml
```

The Gaussian width of 0.3 eV only widens the peaks for display; the integrated
values do not depend on it. See [how the figures were made](index.md#how-the-figures-were-made).
