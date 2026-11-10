# Koga Hartree–Fock STO tabulations (vendored upstream data)

Verbatim copy of `literature_data/` from Furness & Lehtola, *AtomicOrbitals*,
https://github.com/JFurness1/AtomicOrbitals, commit in `UPSTREAM_COMMIT`
(`parser.py` omitted). Files are unmodified; their SHA-256 hashes are recorded in
`../koga_hf_sto.json` (`provenance.source_sha256`).

**Licence of the upstream repository** (from its README): "To the extent possible under law,
James Furness and Susi Lehtola have waived all copyright and related or neighboring rights to
Hartree-Fock Orbitals for Spherical Atoms" — CC0 1.0 Universal,
http://creativecommons.org/publicdomain/zero/1.0/. The tabulations were received by S. Lehtola from
A. J. Thakkar (`literature_data/README.txt`).

**Cite** (also in `literature_data/HFwfn.bib`):

* T. Koga, K. Kanayama, S. Watanabe, A. J. Thakkar, Int. J. Quantum Chem. 71, 491 (1999) — Z = 1–54
* T. Koga, K. Kanayama, S. Watanabe, T. Imai, A. J. Thakkar, Theor. Chem. Acc. 104, 411 (2000) — Z = 55–103

COBALT uses the neutral-atom tables (`k99l/neutral`, `k00heavy`); cation and anion tables are kept for
completeness. Regenerate the converted library with

    python tools/convert_koga.py data/basis/koga/upstream data/basis/koga/koga_hf_sto.json

See docs/BASIS_PROVENANCE.md for the rights review.
