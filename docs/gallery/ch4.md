# Methane

Methane connects two pictures of chemical bonding that students often find
hard to reconcile: **four equivalent C–H bonds** in the Lewis structure, and
**two energy levels** in the molecular-orbital diagram. COHP shows how both are
true at once.

```{figure} ../_static/gallery/ch4-light.svg
:figclass: only-light
:class: prowl-gallery
:alt: Projected DOS, −COHP of one C–H bond, and the two occupied Kohn–Sham levels of methane.

PROWL analysis of CH₄. **Left:** pDOS of carbon and of the four hydrogens
together. **Middle:** −COHP of one C–H bond. **Right:** the occupied
Kohn–Sham orbitals from the same VASP calculation; one of the three
degenerate 1t₂ orbitals is drawn.
```

```{figure} ../_static/gallery/ch4-dark.svg
:figclass: only-dark
:class: prowl-gallery
:alt: Projected DOS, −COHP of one C–H bond, and the two occupied Kohn–Sham levels of methane.

PROWL analysis of CH₄. **Left:** pDOS of carbon and of the four hydrogens
together. **Middle:** −COHP of one C–H bond. **Right:** the occupied
Kohn–Sham orbitals from the same VASP calculation; one of the three
degenerate 1t₂ orbitals is drawn.
```

## Two levels, four bonds

The eight valence electrons occupy two levels:

| Level | E − E<sub>F</sub> (eV) | Made from | Contribution to the ICOHP of **one** C–H bond (eV) |
| --- | --- | --- | --- |
| 2a₁ | −7.7 | C 2s + all four H 1s, in phase | −2.1 |
| 1t₂ (×3) | −0.1 (HOMO) | C 2p + H 1s | −3.6 |
| | | **Total** | **−5.6** |

Both levels are delocalized over the whole molecule, and both are bonding for
every C–H pair. The COHP of a single C–H bond collects its share of each
level. Because the molecule is symmetric, PROWL finds four identical bonds,
each with ICOHP = −5.64 eV and ICOBI = 0.95, close to a single bond.

So the MO picture (two levels, as measured in photoelectron spectroscopy) and
the Lewis picture (four equivalent two-electron bonds) describe the same
electrons. COHP translates between them.

The shell-resolved ICOHP printed by `prowl run` shows that C 2p–H 1s
interactions contribute −3.56 eV and C 2s–H 1s −2.09 eV to each bond.

## The antibonding states

Above the Fermi level, the −COHP shows a large antibonding peak around
+10 eV: the C–H σ* states. In a molecule-in-a-box calculation, these empty
states mix with the free-electron-like states of the surrounding vacuum, so no
single Kohn–Sham state looks like a clean σ* orbital. The figure therefore does
not draw them. The occupied orbitals, which are what determine the bonding,
are well localized on the molecule.

## Quality report

| Diagnostic | Value | Comment |
| --- | --- | --- |
| Occupied charge spilling | 1.4 % | The valence basis describes the occupied states well. |
| Total spilling | 46 % | High, because the retained empty states are largely vacuum states that atomic orbitals cannot describe. |
| Electrons (DFT / Löwdin / Mulliken) | 8.000 / 8.000 / 8.000 | Exact agreement. |
| Band window | 8 bands, **repaired** | See below. |
| Window sensitivity | 9.7 % | ICOHP is −5.64 eV with the default window and −6.19 eV with the frozen window. |

This example shows why the quality report matters. The lowest 8 bands, the
choice LOBSTER would make, do not describe the local orbitals well, because
some of them are vacuum states. PROWL detects this, prints a warning, and
repairs the window with one additional band. The window-sensitivity check then
shows that the ICOHP changes by about 10 % with a different window. The
qualitative conclusions (two bonding levels, four equivalent bonds, a C–H
single bond) are robust; the third digit of ICOHP is not. See
[band windows and profiles](../science/method.md#band-windows-and-profiles).

## Calculation details

A CH₄ molecule in a 12 Å cubic box, computed at the Γ point only:

```text
PREC = Accurate ; ENCUT = 500 ; LREAL = .FALSE.
ISMEAR = 0 ; SIGMA = 0.01 ; EDIFF = 1E-7
ISYM = 0 ; NSW = 0 ; NBANDS = 32
LWAVE = .TRUE. ; LCHARG = .FALSE.
```

with `PAW_PBE C 08Apr2002` and `PAW_PBE H 15Jun2001` potentials, at the
geometry relaxed beforehand with the same settings (C–H = 1.096 Å). The PROWL configuration is
{download}`ch4.yaml <../examples/gallery/ch4.yaml>`:

```{literalinclude} ../examples/gallery/ch4.yaml
:language: yaml
```
