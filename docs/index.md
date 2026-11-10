---
html_theme:
  secondary_sidebar_items: []
---

# PROWL documentation

```{raw} html
<div class="prowl-hero">
  <div>
    <p class="prowl-eyebrow">PROWL · Scientific documentation</p>
    <h1>Chemical bonding,<br>from your wavefunctions.</h1>
    <p class="prowl-intro">Turn VASP calculations into a chemist's picture of bonding: see which orbitals bind atoms together, compare bonds, and check how trustworthy the analysis is.</p>
    <div class="prowl-actions">
      <a class="prowl-button prowl-button-primary" href="getting-started/first-analysis.html">Start an analysis →</a>
      <a class="prowl-button" href="getting-started/concepts.html">New to COHP? Key concepts</a>
    </div>
  </div>
  <div class="prowl-hero-art">
    <img class="prowl-owl" src="_static/brand/prowl-owl-symbol.png" alt="PROWL angular owl with teal diamond eyes" width="1254" height="1254">
  </div>
</div>
```

PROWL is a bonding-analysis tool for VASP calculations. It re-expresses the
plane-wave wavefunctions from your calculation in terms of familiar atomic
orbitals (2s, 2p, 3d, …) and uses that picture to show which atoms are bonded,
how strongly, and through which orbitals. It computes crystal orbital Hamilton
populations (COHP), overlap populations (COOP), bond indices (COBI), projected
densities of states and atomic charges. You can use it from the command line or
from Python.

```{raw} html
<div class="prowl-workflow" aria-label="Analysis workflow">
  <div class="prowl-step"><span class="prowl-step-number">01 / PREPARE</span><strong>Check your VASP run</strong><p>Confirm that the wavefunctions, structure, potentials and metadata belong together.</p></div>
  <div class="prowl-step"><span class="prowl-step-number">02 / PROJECT</span><strong>Build the local picture</strong><p>Choose the atomic orbitals and bonds, preview the work, then run the analysis.</p></div>
  <div class="prowl-step"><span class="prowl-step-number">03 / INTERPRET</span><strong>Read the evidence</strong><p>Check the quality report, then interpret the bonding curves and populations.</p></div>
</div>
```

:::{admonition} New to bonding analysis?
:class: tip

You do not need prior experience with COHP to use PROWL. The
[key concepts](getting-started/concepts.md) page explains, with pictures, what
the analysis does and how to read its plots. Unfamiliar terms are defined in
the [glossary](reference/glossary.md).
:::

## See it in action

```{image} _static/gallery/co-light.svg
:class: only-light prowl-gallery
:alt: PROWL analysis of carbon monoxide: projected DOS, −COHP of the C–O bond, and the Kohn–Sham orbitals at each energy level.
:target: gallery/co.html
```

```{image} _static/gallery/co-dark.svg
:class: only-dark prowl-gallery
:alt: PROWL analysis of carbon monoxide: projected DOS, −COHP of the C–O bond, and the Kohn–Sham orbitals at each energy level.
:target: gallery/co.html
```

A real PROWL analysis of carbon monoxide. The −COHP (middle) shows which
levels bind carbon and oxygen together (right, teal) and which push them apart
(left, red). Each level is connected to its orbital, computed from the same
VASP calculation. [Read the full example →](gallery/co.md)

## Find your starting point

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Key concepts
:link: getting-started/concepts
:link-type: doc

Bonding and antibonding states, reading a COHP plot, and what makes a projection trustworthy.
:::

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install PROWL, prepare your VASP files, and follow a first analysis through to a plot.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc

Select orbitals and bonds, read and export results, solve common problems, and find answers to frequent questions.
:::

:::{grid-item-card} Worked examples
:link: gallery/index
:link-type: doc

Real results for CO, methane and diamond, with the orbitals behind every peak.
:::

:::{grid-item-card} Scientific background
:link: science/index
:link-type: doc

The definitions, the projection method, sign conventions and the quality diagnostics in detail.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Configuration keys, command-line options, the Python interface and the glossary.
:::
::::

## At a glance

| Feature | Details |
| --- | --- |
| **Input** | A completed VASP calculation: WAVECAR, POTCAR, vasprun.xml and CONTCAR/POSCAR. Standard (`vasp_std`) and Γ-only (`vasp_gam`) runs, without or with collinear spin polarization. |
| **Computes** | COHP/ICOHP, COOP/ICOOP, COBI/ICOBI, projected DOS, Mulliken and Löwdin charges, σ/π/δ bond-channel decomposition, and projection-quality diagnostics. |
| **Output** | One HDF5 file (`bonding.h5`) with everything, plus CSV, JSON and LOBSTER-format exports and ready-made plots. |
| **Runs on** | CPU (double precision), with an optional NVIDIA CUDA backend. |
| **Not supported** | Spin–orbit coupling and noncollinear magnetism. |

The LOBSTER-format export lets you use existing LOBSTER scripts and plotting
tools. Matching file formats do not guarantee identical numbers between
programs; see [comparing with LOBSTER](science/method.md#comparing-with-lobster).

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
guide/index
gallery/index
science/index
reference/index
development
```
