# Worked examples

These pages show real PROWL results for three small systems whose bonding is
familiar from chemistry textbooks. Each page walks through the figures, the
integrated values and the quality report, and gives the exact VASP and PROWL
settings so you can reproduce the analysis.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Carbon monoxide
:link: co
:link-type: doc
:img-top: ../_static/gallery/mo/co-05.png
:class-img-top: prowl-gallery
:img-alt: The 5σ HOMO of CO

σ and π bonding, a carbon lone pair, and the antibonding 2π* acceptor orbital.
:::

:::{grid-item-card} Methane
:link: ch4
:link-type: doc
:img-top: ../_static/gallery/mo/ch4-02.png
:class-img-top: prowl-gallery
:img-alt: One of the three 1t₂ orbitals of CH₄

Two molecular-orbital levels, four equivalent C–H bonds.
:::

:::{grid-item-card} Diamond
:link: diamond
:link-type: doc
:img-top: ../_static/gallery/diamond-structure.png
:class-img-top: prowl-gallery
:img-alt: The diamond crystal structure

A covalent solid: bonding valence band, antibonding conduction band.
:::
::::

## Comparing the bonds

```{figure} ../_static/gallery/bonds-light.svg
:figclass: only-light
:class: prowl-gallery
:alt: Bar charts of −ICOHP split into σ and π, and of ICOBI, for the C–O, C–C and C–H bonds.

Integrated COHP (left, split into σ and π) and bond index ICOBI (right) for
the three examples. All values are from the analyses on the following pages.
```

```{figure} ../_static/gallery/bonds-dark.svg
:figclass: only-dark
:class: prowl-gallery
:alt: Bar charts of −ICOHP split into σ and π, and of ICOBI, for the C–O, C–C and C–H bonds.

Integrated COHP (left, split into σ and π) and bond index ICOBI (right) for
the three examples. All values are from the analyses on the following pages.
```

The two panels answer different questions:

- **ICOBI behaves like a bond order.** The C–C bond in diamond and the C–H bond
  in methane are single bonds, and both have ICOBI ≈ 0.95. The bond in CO,
  formally a triple bond, has ICOBI = 2.52. It is below 3 because the bond is
  polar.
- **ICOHP is an energy-weighted measure.** C–C and C–H have the same ICOBI, but
  very different ICOHP values (−9.6 and −5.6 eV). ICOHP depends on the elements
  and orbitals involved, so use it to compare **similar** bonds, for example
  C–O bonds in different molecules or on different surfaces, rather than bonds
  between different pairs of elements.
- **The σ/π split shows the bond type.** CO has a large π part from its two
  π bonds; the single bonds are almost purely σ.

## How the figures were made

Everything on these pages comes from three static VASP calculations: CO and
CH₄ each in a 12 Å box at the Γ point, and diamond on a 7×7×7 k-point mesh.
The script `tools/docs_gallery/make_gallery.py` regenerates all figures:

1. It runs the PROWL analyses with the configurations in
   `docs/examples/gallery/`, using the Python interface.
2. For the molecules, it computes each Kohn–Sham orbital from the VASP
   WAVECAR by a Fourier transform of the plane-wave coefficients at Γ.
   Degenerate orbitals are rotated so that each points along a Cartesian axis.
   The surfaces are drawn at ±0.08 Å⁻³ᐟ² with
   [PyTessel](https://github.com/ifilot/pytessel) and rendered with
   [QASR](https://github.com/ifilot/qasr).
3. It draws the plots with Matplotlib in the colours of this website, in a
   light and a dark version.

The orbital pictures show the smooth plane-wave part of the PAW wavefunction.
Close to the nuclei, the true orbitals have additional nodes; at the isovalue
used here, they hardly change the overall shape. The pictures are the same
states that enter the COHP analysis, not a separate quantum-chemistry
calculation.

The file `docs/_static/gallery/manifest.json` records the software versions,
the INCAR and KPOINTS of each run, the POTCAR identities, checksums of the
input files, and all values quoted on these pages. The VASP input files
themselves are licensed and therefore not distributed. See
[Development](../development.md#regenerate-the-example-gallery) for how to
rerun the script.

```{toctree}
:hidden:

co
ch4
diamond
```
