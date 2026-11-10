"""Bonding-analysis configuration with a strict schema (unknown keys are errors).

Every default that affects chemistry or numerics is recorded in the output
provenance via :meth:`BondingConfig.to_dict`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import _core

SCHEMA_VERSION = 1

_SCHEMA = {
    "schema_version": int,
    "profile": str,
    "input": {"wavecar": str, "structure": str, "potcar": str, "metadata": str},
    "basis": {"source": str, "library": str, "selection": (str, dict), "polarization": (str, dict),
              "polarization_library": str},
    "analysis": {
        "observables": list,
        "window_sensitivity": bool,
        "orbital_frame": str,
        "bond_selection": {"mode": str, "min_distance_angstrom": (int, float), "max_distance_angstrom": (int, float),
                           "pairs": list},
        "energy": {"reference": str, "min_ev": (int, float), "max_ev": (int, float), "points": int,
                   "broadening": {"type": str, "sigma_ev": (int, float)}},
    },
    "numerical": {"projection": str, "band_window": (str, int), "precision": str, "backend": str,
                  "memory_budget_gb": (str, int, float), "strict_validation": bool, "paw_norm_tolerance": (int, float),
                  "paw_norm_check": str},
    "output": {"path": str, "write_intermediate_projection": bool},
}


def _check(d: dict, schema: dict, where: str) -> None:
    if not isinstance(d, dict):
        raise _core.ProwlError(f"InvalidArgument: config section '{where}' must be a mapping")
    for k, v in d.items():
        if k not in schema:
            raise _core.ProwlError(f"InvalidArgument: unknown config key '{where}{k}' "
                                    f"(allowed: {', '.join(sorted(schema))})")
        t = schema[k]
        if isinstance(t, dict):
            _check(v, t, f"{where}{k}.")
        elif not isinstance(v, t) or (t is int and isinstance(v, bool)):
            raise _core.ProwlError(f"InvalidArgument: config key '{where}{k}' has wrong type {type(v).__name__}")


PROFILES = {
    # PROWL's defaults (ADR-016): LOBSTER's lowest-N_orb window (NBANDS-independent) wherever it is
    # well conditioned; otherwise the frozen window, with a warning (its ICOHP depends on NBANDS).
    "prowl": {"projection": "paw_onsite", "band_window": "auto"},
    # LOBSTER's behaviour: always the lowest N_orb bands, with a warning where that window is
    # ill-conditioned (values of symmetry-equivalent bonds may then differ, as in LOBSTER).
    "lobster": {"projection": "paw_onsite", "band_window": "norb"},
}


@dataclass
class BondingConfig:
    profile: str = "prowl"  # prowl | lobster (sets projection and band_window; explicit keys override)
    # basis
    basis_source: str = "koga_hf_sto"
    basis_library: str | None = None
    basis_selection: str | dict = "zval_valence"
    # polarization functions (generated, docs/BASIS_PROVENANCE.md): "none" | "all" (every element in the
    # structure that has one, e.g. Fe 4p, Mg 3p) | {El: [shells]}
    basis_polarization: str | dict = "none"
    polarization_library: str | None = None
    polarization_spilling_hint: float = 0.03  # suggest polarization above this charge spilling
    # bonds
    bond_mode: str = "distance_cutoff"
    min_distance: float = 0.0
    max_distance: float = 3.0
    bond_pairs: list = field(default_factory=list)
    # energies
    reference: str = "fermi"
    emin: float = -20.0
    emax: float = 10.0
    npoints: int = 3001
    sigma: float = 0.05
    broadening: str = "gaussian"  # gaussian | tetrahedron
    # numerics
    projection: str = "paw_onsite"  # paw_onsite (one-centre PAW terms, as LOBSTER) | paw_full (+ two-centre; experimental)
    band_window: str | int = "auto"  # auto | norb | frozen | all | integer (see SCIENCE.md 5.3)
    window_condition_threshold: float = 1e-4  # min eigenvalue of Tp Tp^H accepted for the norb window
    unconverged_band_fraction: float = 0.15   # highest fraction of NBANDS excluded from the frozen-window complement
    frozen_gap_threshold: float = 1e-3        # min relative gap at the complement cut for a unique frozen window
    analysis_blas_threads: int = 1            # BLAS threads during reconstruction/analysis (small dense matrices)
    memory_budget_gb: float | str = "auto"    # wavefunction block budget; "auto" = 2 GB
    precision: str = "accurate"
    backend: str = "auto"
    strict: bool = True
    paw_norm_tolerance: float = 1e-4
    # sampled: folded projection (A_AE onto N_orb factor columns), PAW norm of every 8th band and the last one;
    # full: projector amplitudes and PAW norm of every band (docs/SCIENCE.md 4.4, PERFORMANCE.md)
    paw_norm_check: str = "sampled"
    output_path: str = "bonding.h5"
    write_projection: bool = False  # store projected amplitudes in bonding.h5 for reuse (restart)
    window_sensitivity: bool = True  # also report ICOHP under the other window (lowest-N_orb <-> frozen)
    observables: list = field(default_factory=lambda: ["cohp", "icohp", "coop", "pdos", "mulliken", "lowdin"])
    # global: orbitals in the Cartesian frame of the POSCAR; bond: per bond, both atoms' orbitals in one frame with
    # z' along A->B (sigma/pi/delta-resolved ICOHP and COHP; totals unchanged; LOBSTER's autoRotate is per atom)
    orbital_frame: str = "global"

    @property
    def paw_offsite(self) -> bool:
        if self.projection not in ("paw_full", "paw_onsite"):
            raise _core.ProwlError(f"InvalidArgument: numerical.projection must be paw_full|paw_onsite, got {self.projection}")
        return self.projection == "paw_full"

    @classmethod
    def for_profile(cls, name: str) -> "BondingConfig":
        """Configuration with the defaults of a named profile ('prowl' or 'lobster')."""
        if name not in PROFILES:
            raise _core.ProwlError(f"InvalidArgument: unknown profile '{name}' (allowed: {', '.join(PROFILES)})")
        c = cls()
        c.profile = name
        for k, v in PROFILES[name].items():
            setattr(c, k, v)
        return c

    @classmethod
    def from_dict(cls, d: dict) -> "BondingConfig":
        _check(d, _SCHEMA, "")
        if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise _core.ProwlError(f"UnsupportedFeature: config schema_version {d['schema_version']} not supported")
        c = cls.for_profile(d.get("profile", "prowl"))
        b = d.get("basis", {})
        c.basis_source = b.get("source", c.basis_source)
        from . import BUILTIN_BASES
        if c.basis_source not in BUILTIN_BASES and "library" not in b:
            raise _core.ProwlError(f"InvalidArgument: basis.source '{c.basis_source}' is not built in "
                                    f"({', '.join(BUILTIN_BASES)}) and needs basis.library")
        c.basis_library = b.get("library")
        c.basis_selection = b.get("selection", c.basis_selection)
        c.basis_polarization = b.get("polarization", c.basis_polarization)
        c.polarization_library = b.get("polarization_library", c.polarization_library)
        if isinstance(c.basis_polarization, str) and c.basis_polarization not in ("none", "all"):
            raise _core.ProwlError("InvalidArgument: basis.polarization must be none|all|{element: [shells]}")
        a = d.get("analysis", {})
        c.observables = a.get("observables", c.observables)
        c.window_sensitivity = a.get("window_sensitivity", c.window_sensitivity)
        c.orbital_frame = a.get("orbital_frame", c.orbital_frame)
        bs = a.get("bond_selection", {})
        c.bond_mode = bs.get("mode", c.bond_mode)
        c.min_distance = float(bs.get("min_distance_angstrom", c.min_distance))
        c.max_distance = float(bs.get("max_distance_angstrom", c.max_distance))
        c.bond_pairs = bs.get("pairs", c.bond_pairs)
        e = a.get("energy", {})
        c.reference = e.get("reference", c.reference)
        c.emin = float(e.get("min_ev", c.emin))
        c.emax = float(e.get("max_ev", c.emax))
        c.npoints = int(e.get("points", c.npoints))
        br = e.get("broadening", {})
        c.broadening = br.get("type", "gaussian")
        if c.broadening not in ("gaussian", "tetrahedron"):
            raise _core.ProwlError("InvalidArgument: broadening.type must be gaussian|tetrahedron")
        c.sigma = float(br.get("sigma_ev", c.sigma))
        n = d.get("numerical", {})
        c.projection = n.get("projection", c.projection)
        c.band_window = n.get("band_window", c.band_window)
        c.precision = n.get("precision", c.precision)
        if c.precision != "accurate":
            raise _core.ProwlError("UnsupportedFeature: only precision 'accurate' (FP64) is validated")
        c.backend = n.get("backend", c.backend)
        c.strict = n.get("strict_validation", c.strict)
        c.memory_budget_gb = n.get("memory_budget_gb", c.memory_budget_gb)
        if c.memory_budget_gb != "auto" and not (isinstance(c.memory_budget_gb, (int, float)) and c.memory_budget_gb > 0):
            raise _core.ProwlError("InvalidArgument: numerical.memory_budget_gb must be 'auto' or a positive number")
        c.paw_norm_tolerance = float(n.get("paw_norm_tolerance", c.paw_norm_tolerance))
        c.paw_norm_check = n.get("paw_norm_check", c.paw_norm_check)
        if c.paw_norm_check not in ("sampled", "full"):
            raise _core.ProwlError("InvalidArgument: numerical.paw_norm_check must be sampled|full")
        o = d.get("output", {})
        c.output_path = o.get("path", c.output_path)
        c.write_projection = o.get("write_intermediate_projection", c.write_projection)
        if c.reference not in ("fermi", "absolute"):
            raise _core.ProwlError("InvalidArgument: analysis.energy.reference must be fermi|absolute")
        if c.orbital_frame not in ("global", "bond"):
            raise _core.ProwlError("InvalidArgument: analysis.orbital_frame must be global|bond")
        if c.bond_mode not in ("distance_cutoff", "pairs"):
            raise _core.ProwlError("InvalidArgument: bond_selection.mode must be distance_cutoff|pairs")
        _ = c.paw_offsite
        return c

    @classmethod
    def from_file(cls, path: str | Path) -> "BondingConfig":
        import yaml

        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ resolution helpers
    def factor_budget_bytes(self) -> float:
        """Half of the memory budget is available for plane-wave factor matrices."""
        budget = 2.0 if self.memory_budget_gb == "auto" else float(self.memory_budget_gb)
        return 0.5 * budget * 2**30

    def projection_workers(self, reader) -> int:
        """Maximum CPU k workers that can each generate a factor column under the aggregate limit."""
        ng = max(reader.kpoint(0, k)["nplw"] for k in range(reader.header["nkpts"]))
        copies = 2 if reader.layout["encoding"] == "GammaHalf" else 1
        by_memory = max(1, int(self.factor_budget_bytes() // (ng * 16 * copies)))
        return min(reader.header["nkpts"], _core.omp_max_threads(), by_memory)

    def block_bands(self, reader, nbands: int) -> int:
        """Budget coefficient work per active worker, separately from the aggregate factor cache.

        Six logical coefficient copies conservatively cover input/prefetch, resident/packed and sample
        buffers, plus a raw record per band for reader conversion. A single band is the irreducible minimum;
        outputs and reconstruction storage are additional and are listed by Calculation.plan().
        """
        budget = 2.0 if self.memory_budget_gb == "auto" else float(self.memory_budget_gb)
        nplw = max(reader.kpoint(0, k)["nplw"] for k in range(reader.header["nkpts"]))
        per_band = nplw * 16 * 6 + reader.header["record_length"]
        return int(max(1, min(nbands, 0.5 * budget * 2**30 // (per_band * self.projection_workers(reader)))))

    def band_count(self, norb: int, nbands: int) -> int:
        if self.band_window in ("auto", "repaired", "frozen"):
            return nbands  # project everything; the window is chosen after conditioning checks
        if self.band_window == "norb":
            if self.window_sensitivity:
                return nbands  # the alternate (frozen) window needs the higher bands too
            n = norb
        elif self.band_window == "all":
            n = nbands
        elif isinstance(self.band_window, int):
            n = self.band_window
        else:
            raise _core.ProwlError(f"InvalidArgument: band_window must be auto|repaired|norb|frozen|all|int, "
                                    f"got {self.band_window}")
        if n < norb:
            raise _core.ProwlError(f"InvalidArgument: band window {n} < number of basis functions {norb}")
        if n > nbands:
            raise _core.ProwlError(f"InvalidArgument: band window {n} exceeds NBANDS={nbands} in WAVECAR; rerun VASP "
                                    f"with NBANDS >= {n}")
        return n

    @property
    def uses_polarization_library(self) -> bool:
        """Merge the polarization library when polarization is requested or shells are named explicitly."""
        return self.basis_polarization != "none" or isinstance(self.basis_selection, dict)

    @property
    def library_path(self) -> Path:
        """basis.library, else the built-in set named by basis.source ($PROWL_BASIS_FILE overrides Koga)."""
        from . import basis_path, default_basis_path

        if self.basis_library:
            return Path(self.basis_library)
        return default_basis_path() if self.basis_source == "koga_hf_sto" else basis_path(self.basis_source)

    def polarization_source(self, force: bool = False):
        """Polarization library to merge: a path, True (the generated default), or None. Built-in sets with their
        own polarization shells (paw-pbe-sto-1) merge nothing unless basis.polarization_library is given."""
        from . import BUILTIN_BASES

        if not (force or self.uses_polarization_library):
            return None
        if self.polarization_library:
            return self.polarization_library
        if not self.basis_library and BUILTIN_BASES.get(self.basis_source, (None, False))[1]:
            return None
        return True

    def load_library(self, force_polarization: bool = False):
        from . import load_basis_library

        return load_basis_library(self.library_path, self.polarization_source(force_polarization))

    def library_paths(self) -> list:
        from . import default_polarization_path

        paths = [self.library_path]
        pol = self.polarization_source()
        if pol is not None:
            paths.append(default_polarization_path() if pol is True else Path(pol))
        return paths

    def resolve_basis(self, st, potcars, lib) -> dict:
        if isinstance(self.basis_selection, dict):
            sel = {k: list(v) for k, v in self.basis_selection.items()}
            missing = set(st.species) - set(sel)
            if missing:
                raise _core.ProwlError(f"InvalidArgument: basis.selection lacks elements {sorted(missing)}")
        elif self.basis_selection != "zval_valence":
            raise _core.ProwlError(f"InvalidArgument: unknown basis selection rule '{self.basis_selection}'")
        else:
            sel = {}
            for sp, pc in zip(st.species, potcars):
                sel[sp] = zval_valence_shells(lib, sp, pc.zval, potcar_symbol(pc.titel))
        for sp, extra in polarization_shells(lib, list(st.species), self.basis_polarization).items():
            for lab in extra:
                if lab in sel[sp]:
                    raise _core.ProwlError(f"InvalidArgument: polarization shell {sp} {lab} is already selected")
                sel[sp].append(lab)
        return sel

    def select_bonds(self, lattice, frac, elements):
        from .analysis import Bond, find_bonds

        if self.bond_mode == "pairs":
            out = []
            for p in self.bond_pairs:
                a, b = int(p[0]) - 1, int(p[1]) - 1
                R = tuple(int(x) for x in (p[2] if len(p) > 2 else (0, 0, 0)))
                d = float(np.linalg.norm((frac[b] + R - frac[a]) @ lattice))
                out.append(Bond(a, b, R, d))
            return out
        return find_bonds(lattice, frac, self.min_distance, self.max_distance)


_F_IN_CORE = re.compile(r"_[23]$")


@lru_cache(None)
def _atomic_numbers() -> dict:
    from . import default_basis_path
    return {el: e["Z"] for el, e in json.loads(Path(default_basis_path()).read_text())["elements"].items()}


def potcar_symbol(titel: str) -> str:
    """POTCAR symbol from its TITEL ('PAW_PBE Gd_3 06Sep2000' -> 'Gd_3'); '' if the TITEL has another form."""
    f = titel.split()
    return f[1] if len(f) >= 2 else ""


def _shell_order(lab: str):
    return ("spdf".index(lab[-1]), int(lab[:-1]))


def zval_valence_shells(lib, element: str, zval: float, symbol: str = "") -> list[str]:
    """Outermost occupied Koga shells (by orbital energy) whose occupations sum to ZVAL exactly.

    Two POTCAR families need more than the energy order:
    * f-in-core lanthanides (symbol ending in _2 or _3): the open 4f shell is frozen in the core; the valence is
      5p6 6s2, plus 5s2 if ZVAL >= 10, plus 5d with the remaining electrons (5d need not be occupied in Koga);
    * from Hf (Z = 72) on, the filled 4f14 shell is core in every POTCAR (HF puts it above 5s/5p in energy).
    """
    if symbol and _F_IN_CORE.search(symbol):
        rest = round(zval) - 8
        out = ["6s", "5p"]
        if rest >= 2:
            out.append("5s")
            rest -= 2
        if rest > 0:
            out.append("5d")
        have = {s.label for s in lib.shells(element)}
        missing = [t for t in out if t not in have]
        if missing:
            raise _core.ProwlError(
                f"InvalidArgument: f-in-core POTCAR {symbol} needs the {element} shells {missing}, which the basis "
                "library lacks; use a library with these shells or specify basis.selection explicitly")
        return sorted(out, key=_shell_order)
    shells = sorted((s for s in lib.shells(element) if s.occupation > 0), key=lambda s: -s.energy_ev)
    if _atomic_numbers().get(element, 0) > 71:
        shells = [s for s in shells if s.label != "4f"]
    acc, out = 0.0, []
    for s in shells:
        if acc >= zval - 1e-6:
            break
        out.append(s.label)
        acc += s.occupation
    if abs(acc - zval) > 1e-6:
        raise _core.ProwlError(
            f"InvalidArgument: cannot match POTCAR ZVAL={zval} for {element} with Koga shell occupations "
            f"(reached {acc}); specify basis.selection explicitly")
    return sorted(out, key=_shell_order)


def polarization_shells(lib, species: list[str], spec) -> dict[str, list[str]]:
    """Polarization shells to add: spec 'none' | 'all' | {El: [labels]} (labels must exist with occupation 0)."""
    if spec == "none":
        return {}
    avail = {sp: [s.label for s in lib.shells(sp) if s.occupation == 0] for sp in species}
    if spec == "all":
        return {sp: v for sp, v in avail.items() if v}
    out = {}
    for sp, labs in spec.items():
        if sp not in species:
            raise _core.ProwlError(f"InvalidArgument: basis.polarization names {sp}, which is not in the structure")
        for lab in labs:
            if lab not in avail[sp]:
                raise _core.ProwlError(f"InvalidArgument: no polarization shell {lab} for {sp} "
                                        f"(available: {', '.join(avail[sp]) or 'none'})")
        out[sp] = list(labs)
    return out
