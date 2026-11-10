# Your first analysis

This walkthrough takes one VASP calculation from the raw files to a COHP plot
and a table of bond populations. Along the way you will see what each PROWL
command checks and how to read its output.

**Before you start:** PROWL must be [installed](installation.md) and
`./vasp-run` must contain a [completed, compatible calculation](vasp-inputs.md).
Run the commands from the repository root. If you have no suitable calculation,
see [a small example system](#a-small-example-system) at the end of this page.

The analysis has five steps:

1. Write a configuration that says what to analyze.
2. Check the inputs and preview the work.
3. Run the analysis and read the quality report.
4. Plot a bond and export the numbers.
5. Record what you did.

## 1. Choose what to analyze

PROWL reads its settings from a short YAML file. Copy the supplied example:

```bash
cp docs/examples/bonding.yaml bonding.yaml
```

You can also {download}`download bonding.yaml <../examples/bonding.yaml>` directly.
It contains:

```{literalinclude} ../examples/bonding.yaml
:language: yaml
```

For a first run, the settings to look at are:

`basis`
: The atomic orbitals to project onto. `zval_valence` picks the valence
  shells that match your POTCARs, so the default works for most systems.

`analysis.bond_selection`
: Which atom pairs to analyze. Here, every pair up to 3.0 Å apart is
  included. This is an example, not a universal definition of a chemical bond:
  adapt it to the distances in your structure.

`analysis.energy`
: The energy range of the curves, here from −20 to +10 eV around the Fermi
  level. Each state is smoothed into a Gaussian with a width (standard
  deviation) of 0.05 eV.

Every key is described in the [configuration reference](../reference/configuration.md).
Any key you leave out keeps its default value. Misspelled or unknown keys are
reported as errors rather than silently ignored.

:::{tip}
For a very quick first check without a configuration file,
`prowl run ./vasp-run --backend cpu` uses the default settings. An explicit
file is easier to review, share and reuse, so use one for real work.
:::

## 2. Inspect, validate and plan

Run three quick checks before the actual analysis. None of them projects the
wavefunctions, so they finish in seconds.

```bash
prowl inspect  ./vasp-run
prowl validate ./vasp-run --config bonding.yaml
prowl plan     ./vasp-run --config bonding.yaml
```

| Command | What it checks | What to look for |
| --- | --- | --- |
| `inspect` | Files are present, consistent and in a supported format | `status  : OK` on the last line |
| `validate` | The configuration file is valid for this run | `config OK`, followed by the resolved settings |
| `plan` | Basis, system size, band count and memory | The `bands` line and any `recommend` lines |

The `bands` line of the plan compares your `NBANDS` with the minimum
({term}`N_orb`) and the recommended value:

```text
basis     : …                     # the shells selected for each element
bands     : NBANDS=…, minimum … (= N_orb, as LOBSTER), recommended >= …
memory    : … | peak ~… GiB
recommend : …                     # suggestions worth reading before you run
```

If `NBANDS` is below the minimum, rerun VASP with more bands (see
[how many bands?](vasp-inputs.md#how-many-bands)). Fix any errors before
continuing.

## 3. Run the analysis

```bash
prowl run ./vasp-run --config bonding.yaml --backend cpu
```

PROWL projects the wavefunctions, builds the bonding quantities and writes
everything to `./vasp-run/bonding.h5`. Running the command again with the same
output path overwrites that file. To keep several results, choose a different
name, for example `--output ./vasp-run/bonding-comparison.h5`.

### Read the quality report

The run first prints a **quality report**. It tells you how well the local
orbitals describe your calculation. Read it before you interpret any bond
numbers. Its layout is:

```text
PROWL quality report
  backend                      : cpu
  band window                  : … bands
  max |PAW norm - 1|           : …
  abs. charge spilling (occ.)  : … %
  abs. total spilling          : … %
  min overlap eigenvalue       : …
  max reconstruction residual  : … eV (occupied bands)
  electrons (DFT / Löwdin / Mulliken): … / … / …
  timings [s]                  : …
  window sensitivity           : …
```

The most important lines for a first analysis are:

| Line | Plain-language meaning |
| --- | --- |
| `max \|PAW norm - 1\|` | Are the wavefunctions and POTCAR consistent? This should be very small. With the default strict validation, PROWL stops if it exceeds the tolerance. |
| `abs. charge spilling (occ.)` | The percentage of occupied-state density that the local orbitals miss. Lower is better. |
| `electrons (DFT / Löwdin / Mulliken)` | The electron count from VASP and from the two population analyses. Large differences point to problems with the basis. |
| `window sensitivity` | How much the bond populations change with an alternative band window. A large value means the result depends on a technical choice. |
| `WARNING: …` | Anything PROWL thinks you should know, such as bonds that this k-mesh cannot resolve. |

The [quality guide](../science/quality.md) explains every diagnostic and what
to do when one looks wrong.

### Read the bond table

After the report, PROWL lists the integrated populations of each bond:

```text
ICOHP at E_F (eV, COHP sign: negative = bonding):
  C1-C2[0 0 0]                 d =  …  A   ICOHP = …   ICOOP = …   ICOBI = …
  …
```

Each line is one atom pair. `C1-C2[0 0 0]` means carbon atom 1 to carbon atom 2
in the same unit cell ([what the brackets mean](concepts.md#bonds-in-a-periodic-crystal)),
`d` is the distance in Å, and the three integrated values summarize the
bonding. A more negative ICOHP means a larger net bonding contribution. When
the window-sensitivity check is active, the value obtained with the alternative
window is shown in square brackets at the end of each line.

Below the table, the first bond is broken down into shell pairs, such as
`2s - 2p`. This shows which orbitals contribute most to that interaction.

## 4. Plot a bond and export the table

```bash
prowl plot ./vasp-run/bonding.h5 --kind cohp --bonds 0 --output cohp.png
prowl export ./vasp-run/bonding.h5 --format csv --output ./bonding-tables
```

`--bonds 0` plots the first bond in the table. Plot indices count from
**zero**, so the second bond is `--bonds 1`. Without `--bonds`, up to six bonds
are shown. The plot fails if the cutoff selected no bonds at all.

In the plot, energy runs **vertically** (`E − E_F` in eV) and the horizontal
axis shows **−COHP**:

- contributions to the **right** of the zero line are **bonding**;
- contributions to the **left** are **antibonding**;
- states **below** `E − E_F = 0` are occupied.

The [key concepts](concepts.md#reading-a-cohp-plot) page shows an annotated
example.

The CSV export writes the curves to `cohp.csv`, `coop.csv` and `pdos.csv`,
with one energy column followed by one column per bond or orbital.
`icohp.csv` contains the bond labels, distances and integrated values per spin
channel, and `summary.json` adds the diagnostics and a record of the settings. See
[results and export](../guide/results.md) for all files and formats.

## 5. Record the analysis

For your lab notebook or a publication, keep:

- `bonding.yaml` and the PROWL version (`prowl --version`);
- the result file `bonding.h5`, which also stores many of the settings;
- the VASP input files (INCAR, KPOINTS, POSCAR and the POTCAR names).

When you compare systems, use the same basis and band-window settings for all
of them. Remember that ICOHP describes the selected local representation and is
not a bond dissociation energy.

## A small example system

The repository contains a recipe for a two-atom diamond cell in
`tests/private_vasp_manifest/recipes/c_diamond_k7/`. If you have a licensed VASP
installation and access to the PAW potentials, a helper script can run that
calculation for you:

```bash
python tools/run_vasp_fixture.py c_diamond_k7 \
  --runs ./runs --vasp-bin /path/to/vasp/bin \
  --potcar-dir /path/to/paw-datasets --np 8
```

Replace the two paths with the locations on your machine. The calculation is
written to `./runs/c_diamond_k7`; use that directory in place of `./vasp-run`
in this walkthrough. POTCAR and WAVECAR files are not distributed with PROWL
because of VASP licensing.

## Next steps

- [Choose an analysis](../guide/choosing-an-analysis.md): change the basis,
  select specific bonds and resolve σ/π contributions.
- [Interpret the observables](../science/interpreting-results.md): what COHP,
  COOP, COBI and the charges mean.
- [Worked examples](../gallery/index.md): complete analyses of CO, methane
  and diamond, including this calculation.
- [Frequently asked questions](../guide/faq.md).
