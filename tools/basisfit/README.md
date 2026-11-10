# basisfit — STO basis fitted to VASP PBE free atoms

Fits the valence and polarization shells of a Slater-type (STO) basis to the PAW orbitals of VASP PBE free
atoms, in the same PAW metric PROWL uses for projection. The plan, targets and decisions are in
[docs/BASIS_FIT_PLAN.md](../../docs/BASIS_FIT_PLAN.md). This directory holds the method as frozen at M3
(2026-11-06). Results and VASP runs are not part of the repository.

## Method (frozen)

* **Atom:** one atom in a cubic box (20 Å), Γ only, nonspin with Gaussian smearing (SIGMA 0.2 eV).
  ENCUT = max(400, 1.3 ENMAX) eV, PREC Accurate.
  SCF setting A (Davidson, NELM 60); setting B (ALGO All, NELM 250) only if A fails.
* **Box rule:** if a fitted shell is bound by less than 1 eV in the 20 Å box, the atom is rerun in 24 Å and that
  fit is used.
* **Shells** ([pbefit/shells.py](pbefit/shells.py)):
  * Valence shells come from the Koga ground configuration, filled until ZVAL is reached; f-in-core POTCARs are a
    special case.
  * Polarization shells are taken from pymatgen's `BASIS_PBE_54_standard` list, plus the lowest empty p shell of
    s/d/f metals.
* **Fit** ([pbefit/fit.py](pbefit/fit.py)):
  * Per l: the Koga primitives plus 0–3 diffuse even-tempered primitives, shared by all shells of that l.
  * The contraction is the top generalized eigenvector of the captured band weight.
  * Automatic checks: radial nodes, conditioning, l character, tail inside the box, and max |coefficient| ≤ 10
    (added 2026-11-06: rejects contractions of cancelling diffuse extras; evaluated from the stored primitives for
    older fit reports).
  * The number of extra primitives is the fewest that come within 1e-4 spilling of the best fit (`auto_m`).
* **Which fitted shells replace Koga** (`library.rule_2026`):
  * Added (empty) shells: always.
  * Occupied shells of s/d/f elements and of the 6p block (Tl–Rn): yes, except filled d10/f14 shells.
  * Other p-block elements and H/He: no, they keep Koga.

## Requirements

* The `prowl` Python package from this repository (Koga basis and PAW projections), plus numpy, scipy and
  matplotlib (for `report`).
* `PROWL_POTCAR_DIR`: a licensed `potpaw_PBE` directory. POTCARs are copied into the run directories and
  never into the repository.
* `PBEFIT_VASP_CMD`: a Γ-only VASP command, for example `vasp_gam` or `mpirun -np {ranks} vasp_gam`.
  `{threads}` and `{ranks}` are filled in when present. The queue also exports `VASP_CORESET=<first>-<last>`
  (physical cores) per slot, for wrappers that pin threads.

## Usage

Run from a work directory, or pass `--work DIR`. Outputs go to `atoms/`, `fits/`, `libs/` and `reports/`
inside it.

```bash
B=tools/basisfit/basisfit.py
$B atoms Fe Cu_pv --threads 4 --fit          # VASP runs, then fits (2026 and 2015 mode)
$B queue jobs.txt --slots 0-3:2x2,4-7:2x2 --fit --log reports/queue.log
#   jobs.txt: "POTCAR conf box" per line, e.g. "Fe nonspin 20"; box-rule reruns are appended automatically
$B report                                    # reports/summary_boxrule.md, rmean_vs_Z_boxrule.png
$B aecheck Fe Au                             # all-electron scalar-relativistic cross-check of the fits
$B library mybasis --potcars Fe,O,Ti --sto sto/   # libs/mybasis.json (prowl-sto-basis/1) + LOBSTER .sto
$B library full --all                         # one entry per element: base POTCAR + semicore shells of _pv/_sv
                                              #   (+ lanthanide 5d from the _3 fit)
```

Every fitted shell in a library records its atom (`fit.potcar`, and `fit.atom`: POTCAR TITEL and SHA256 line,
box, SCF setting, VASP version); `provenance.pbefit` records the method, atom settings, code commit and date.

`library --scale 0.95` scales the diffuse extra exponents, for the sensitivity checks of M3 and M7.
`--mode 2015` gives the 2015-style library: Koga plus fitted added shells only.

## Scope and limits

* Pseudo-hydrogen (`H.5`, …) and GW POTCARs are out of scope.
* Open-shell f atoms: the nonspin smeared atom moves the d electron into the f shell (Gd 4f8 5d0, Cm 5f8 6d0).
  The fitted d shell is correct for that state but about 20 % more diffuse than in the d1 configuration.
  `library --all` therefore takes the lanthanide 5d from the f-in-core `_3` fit (d1-like, matches the
  all-electron 4f7 5d1 atom); the actinides have no `_3` POTCARs and keep the diffuse 6d (Np, Cm). See M6 in
  the plan.
* Comparisons with pbeVaspFit2015 and the LOBSTER/PROWL solid benchmarks are kept outside the repository:
  they read licensed or private data.

## Data

[data/pmg_basis_PBE54_standard.json](data/pmg_basis_PBE54_standard.json) is a frozen copy of the shell
labels in pymatgen's `BASIS_PBE_54_standard.yaml` (MIT licence, notice in
[data/LICENSE.pymatgen](data/LICENSE.pymatgen)). It contains orbital labels only, with no basis-set exponents
or coefficients.
