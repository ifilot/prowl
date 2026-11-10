# paw-pbe-sto-1

STO basis fitted to VASP PBE PAW free atoms (182 POTCARs, 97 elements H–Cf without Bk); one entry per element
(Es–Lr, without POTCARs, are plain Koga). Method and
validation: [docs/BASIS_FIT_REPORT.md](../../../docs/BASIS_FIT_REPORT.md); provenance:
[docs/BASIS_PROVENANCE.md](../../../docs/BASIS_PROVENANCE.md). Licence: CC BY-SA 4.0 ([LICENSE.md](LICENSE.md)).

* `paw-pbe-sto-1.json` — `prowl-sto-basis/1`; in PROWL: `basis: {source: paw-pbe-sto-1}`. Shells with a `fit`
  record are fitted (source atom in `fit.atom`); the others are Koga HF shells. Occupation-0 shells are the
  polarization shells (`basis.polarization: all`).
* `sto/<El>.sto` — the same basis for LOBSTER (`basisSet custom` with `customSTOforAtom`).

Regenerate: `tools/basisfit/basisfit.py --work <dir> library paw-pbe-sto-1 --all --sto sto/` from the fitted atoms
(VASP and POTCARs required), then add the licence fields to `provenance`.
