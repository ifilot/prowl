"""High-level API: open a VASP run directory, inspect it, run the bonding analysis."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import __version__, _core
from .analysis import AnalysisInput, AnalysisResult, EnergyGrid, analyze

# Band-window helpers from the C++ core (NumPy references with the same names live in
# prowl.analysis and are cross-checked in tests/python/test_engines.py).
cut_at_gap = _core.cut_at_gap
window_conditioning = _core.window_conditioning
frozen_window_subspace = _core.frozen_window_subspace
from .config import BondingConfig
from .vasp import detect_mesh, read_vasprun, sha256_file


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str


@dataclass
class InspectReport:
    run_dir: Path
    files: dict
    wavecar: dict | None = None
    structure: dict | None = None
    potcar: list | None = None
    vasprun: dict | None = None
    mesh: dict | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "issues"}
        d["run_dir"] = str(self.run_dir)
        d["issues"] = [i.__dict__ for i in self.issues]
        d["ok"] = self.ok
        return d

    def summary(self) -> str:
        L = [f"PROWL {__version__} inspection of {self.run_dir}"]
        if self.wavecar:
            w = self.wavecar
            L.append(f"  WAVECAR : {w['encoding']} ({w['precision']} precision, RTAG {w['rtag']}), "
                     f"{w['nspin']} spin(s), {w['nkpts']} k-point(s), {w['nbands']} bands, ENCUT {w['encut_ev']:.1f} eV")
            if w["encoding"] == "GammaHalf":
                L.append("            Gamma-compressed (vasp_gam) wavefunctions: supported (native half-sphere arithmetic)")
            elif w["encoding"] == "Spinor":
                L.append("            noncollinear spinor wavefunctions: NOT supported for projection")
        if self.structure:
            L.append(f"  POSCAR  : {self.structure['natoms']} atoms, species {self.structure['species']}")
        if self.potcar:
            L.append("  POTCAR  : " + "; ".join(p["titel"] for p in self.potcar))
        if self.vasprun:
            v = self.vasprun
            L.append(f"  vasprun : VASP {v['version']}, ISPIN={v['ispin']}, ISYM={v['isym']}, LREAL={v['lreal']}, "
                     f"E_F={v['efermi']:.4f} eV, NELECT={v['nelect']}")
        if self.mesh:
            L.append(f"  k-mesh  : {self.mesh['mesh']} full={self.mesh['full']} time-reversal-half={self.mesh['half']}")
        for i in self.issues:
            L.append(f"  [{i.level.upper():7s}] {i.code}: {i.message}")
        L.append("  status  : " + ("OK" if self.ok else "NOT ANALYSABLE (see errors)"))
        return "\n".join(L)


def _find_structure(run: Path) -> Path | None:
    c = run / "CONTCAR"
    if c.is_file() and c.stat().st_size > 0:
        return c
    return _find(run, "POSCAR")


def _find(run: Path, *names: str) -> Path | None:
    for n in names:
        p = run / n
        if p.is_file():
            return p
    return None


class Calculation:
    """A VASP run directory. Nothing large is loaded until :meth:`analyze` is called."""

    def __init__(self, run_dir: str | Path, wavecar: str | None = None, structure: str | None = None,
                 potcar: str | None = None, metadata: str | None = None):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise _core.ProwlError(f"FileNotFound: run directory {self.run_dir} does not exist")
        self.paths = {
            "wavecar": Path(self.run_dir, wavecar) if wavecar else _find(self.run_dir, "WAVECAR"),
            # CONTCAR first: after ionic steps the WAVECAR belongs to the final geometry
            "structure": Path(self.run_dir, structure) if structure else _find_structure(self.run_dir),
            "potcar": Path(self.run_dir, potcar) if potcar else _find(self.run_dir, "POTCAR"),
            "metadata": Path(self.run_dir, metadata) if metadata else _find(self.run_dir, "vasprun.xml"),
        }

    @classmethod
    def open(cls, run_dir: str | Path, **kw) -> "Calculation":
        return cls(run_dir, **kw)

    # ------------------------------------------------------------------ inspect
    def inspect(self) -> InspectReport:
        rep = InspectReport(run_dir=self.run_dir, files={k: (str(v) if v else None) for k, v in self.paths.items()})
        add = lambda lvl, code, msg: rep.issues.append(Issue(lvl, code, msg))
        for k, v in self.paths.items():
            if v is None or not Path(v).is_file():
                add("error", "missing_input", f"{k} file not found in {self.run_dir}")
        reader = st = pcs = vr = None
        if self.paths["wavecar"] and self.paths["wavecar"].is_file():
            try:
                reader = _core.WavecarReader(str(self.paths["wavecar"]))
                h, L = reader.header, reader.layout
                rep.wavecar = {**{k: v for k, v in h.items() if k != "lattice"}, "encoding": L["encoding"],
                               "gamma_half": L["gamma_half"], "notes": L["notes"],
                               "min_sphere_margin_ev": L["min_sphere_margin_ev"]}
                for n in L["notes"]:
                    add("info", "wavecar_note", n)
                if L["encoding"] == "Spinor":
                    add("error", "unsupported_spinor", "noncollinear/SOC wavefunctions are detected but not supported")
            except _core.ProwlError as e:
                add("error", "wavecar_invalid", str(e))
        if self.paths["structure"] and self.paths["structure"].is_file():
            try:
                st = _core.read_poscar(str(self.paths["structure"]))
                rep.structure = {"natoms": st.natoms, "species": list(st.species), "counts": list(st.counts),
                                 "volume": st.volume}
            except _core.ProwlError as e:
                add("error", "structure_invalid", str(e))
        if self.paths["potcar"] and self.paths["potcar"].is_file():
            try:
                pcs = _core.read_potcar(str(self.paths["potcar"]))
                rep.potcar = [{"titel": p.titel, "element": p.element, "symbol": p.symbol, "zval": p.zval,
                               "sha256_header": p.sha256, "channels": [c.l for c in p.channels]} for p in pcs]
            except _core.ProwlError as e:
                add("error", "potcar_invalid", str(e))
        if self.paths["metadata"] and self.paths["metadata"].is_file():
            try:
                vr = read_vasprun(self.paths["metadata"])
                rep.vasprun = {"version": vr.version, "ispin": vr.ispin, "isym": vr.isym, "lreal": vr.lreal,
                               "efermi": vr.efermi, "nelect": vr.nelect, "nbands": vr.nbands, "nkpts": len(vr.kpoints),
                               "lsorbit": vr.lsorbit, "lnoncollinear": vr.lnoncollinear, "nsw": vr.nsw}
            except Exception as e:  # noqa: BLE001
                add("error", "vasprun_invalid", str(e))
        # cross checks
        if reader is not None and st is not None:
            if np.abs(np.array(reader.header["lattice"]) - np.array(st.lattice)).max() > 1e-4:
                add("error", "lattice_mismatch", "POSCAR/CONTCAR lattice differs from WAVECAR header lattice")
        if pcs is not None and st is not None:
            el = [p.element for p in pcs]
            if el != list(st.species):
                add("error", "species_mismatch", f"POTCAR elements {el} differ from POSCAR species {list(st.species)}")
        if vr is not None and reader is not None:
            h = reader.header
            if len(vr.kpoints) != h["nkpts"] or vr.ispin != h["nspin"] or vr.nbands != h["nbands"]:
                add("error", "metadata_mismatch", "vasprun.xml (k-points/spins/bands) inconsistent with WAVECAR")
            else:
                kw = np.array([reader.kpoint(0, k)["k_frac"] for k in range(h["nkpts"])])
                if np.abs(kw - vr.kpoints).max() > 1e-6:
                    add("error", "kpoint_mismatch", "k-point coordinates differ between WAVECAR and vasprun.xml")
                ew = np.array([reader.kpoint(s, k)["eigenvalues"] for s in range(h["nspin"]) for k in range(h["nkpts"])])
                dev = np.abs(ew - vr.eigenvalues.reshape(ew.shape)).max()
                if dev > 1e-3:
                    add("error", "eigenvalue_mismatch",
                        f"WAVECAR and vasprun.xml eigenvalues differ by {dev:.2e} eV: WAVECAR is not the converged state "
                        "of this run (e.g. MD/continuation predicted wavefunctions)")
            if vr.lsorbit or vr.lnoncollinear:
                add("error", "unsupported_noncollinear", "LSORBIT/LNONCOLLINEAR runs are not supported")
            if vr.nsw and vr.nsw > 0 and vr.ibrion not in (None, -1):
                add("warning", "ionic_run", "NSW > 0: verify that WAVECAR and CONTCAR belong to the same final step")
            if vr.lreal and vr.lreal.strip(".").upper() not in ("F", "FALSE"):
                add("warning", "lreal", f"LREAL={vr.lreal}: reciprocal-space projectors are used here, VASP used real-space "
                    "ones; PAW norm check quantifies the inconsistency")
            mesh, full, half = detect_mesh(vr.kpoints)
            rep.mesh = {"mesh": mesh, "full": full, "half": half}
            if not (full or half):
                add("error", "kmesh_not_full", "k-points are symmetry-reduced (ISYM > 0) or not a uniform Gamma-containing "
                    "mesh; rerun VASP with ISYM = -1 (or 0) for real-space bond resolution")
            elif half:
                add("info", "kmesh_time_reversal", "time-reversal half mesh (ISYM=0) will be unfolded with T(-k) = T(k)*")
        return rep

    # ------------------------------------------------------------------ analysis
    def plan(self, config: BondingConfig | None = None) -> dict:
        """Dry run: problem sizes, memory estimates and streaming blocks, without projecting."""
        cfg = config or BondingConfig()
        rep = self.inspect()
        out = {"inspection_ok": rep.ok, "issues": [i.__dict__ for i in rep.issues]}
        if rep.wavecar is None or rep.structure is None or rep.potcar is None:
            return out
        st = _core.read_poscar(str(self.paths["structure"]))
        pcs = _core.read_potcar(str(self.paths["potcar"]))
        reader = _core.WavecarReader(str(self.paths["wavecar"]))
        lib = cfg.load_library()
        try:
            selection = cfg.resolve_basis(st, pcs, lib)
        except _core.ProwlError as e:
            out["error"] = str(e)
            return out
        norb = sum(2 * lib.shell(st.element(a), lab).l + 1 for a in range(st.natoms) for lab in selection[st.element(a)])
        nproj = sum(pcs[st.atom_type[a]].nprojectors for a in range(st.natoms))
        h = reader.header
        ng = max(reader.kpoint(0, k)["nplw"] for k in range(h["nkpts"]))
        nb = h["nbands"]
        nk_full = h["nkpts"] * (2 if (rep.mesh or {}).get("half") else 1)
        block = cfg.block_bands(reader, nb)
        workers = cfg.projection_workers(reader)
        copies = 2 if reader.layout["encoding"] == "GammaHalf" else 1
        cols = int(cfg.factor_budget_bytes() // (workers * ng * 16 * copies))
        GiB = 2.0 ** 30
        nbonds = len(cfg.select_bonds(np.asarray(st.lattice), np.asarray(st.frac),
                                     [st.element(a) for a in range(st.natoms)]))
        mem = {
            "coefficient_block_GiB": workers * ng * min(block, nb) * 16 * 6 / GiB,
            "reader_conversion_GiB": workers * min(block, nb, 64) * h["record_length"] / GiB,
            "factor_block_GiB": min(cfg.factor_budget_bytes(), workers * ng * (norb + nproj) * 16 * 5) / GiB,
            "projection_temporaries_GiB": workers * h["nspin"] * (2 * norb + nproj) * nb * 16 / GiB,
            "projections_all_k_GiB": h["nspin"] * nk_full * norb * nb * 16 / GiB,
            "overlaps_and_factors_GiB": nk_full * norb * norb * 16 * 2 / GiB,
            "reconstruction_all_k_GiB": h["nspin"] * nk_full * (3 * norb * nb + 2 * norb * norb) * 16 / GiB,
            "spectrum_outputs_GiB": h["nspin"] * (8 * nbonds + norb) * cfg.npoints * 8 / GiB,
        }
        out.update({
            "sizes": {"N_atom": st.natoms, "N_orb": norb, "N_proj": nproj, "N_band": nb, "N_G_max": ng,
                      "N_k": h["nkpts"], "N_k_after_unfolding": nk_full, "N_spin": h["nspin"],
                      "encoding": reader.layout["encoding"]},
            "basis": selection,
            "streaming": {"band_block": block, "band_blocks_per_k": -(-nb // block), "factor_columns_per_block": cols,
                          "memory_budget_gb": cfg.memory_budget_gb, "cpu_workers": workers,
                          "aggregate_factor_limit_bytes": int(cfg.factor_budget_bytes())},
            "memory_estimate": mem,
            "memory_estimate_note": "Conservative projection/analysis array estimate including concurrent workers, conversion, "
                                    "prefetch/packing, retained factors and outputs; excludes allocator and library "
                                    "overhead and basis/model tables. The aggregate factor limit is enforced; one coefficient band and "
                                    "stored results are irreducible and can exceed a very small working budget.",
            "estimated_peak_GiB": sum(mem.values()),
            "gemm_flops_per_k": 8.0 * ng * (norb + nproj) * nb * h["nspin"],
            "recommendations": [],
        })
        factor = 2.0 if cfg.basis_polarization != "none" else 1.5
        out["required_nbands"] = {"minimum": norb, "recommended": int(np.ceil(factor * norb)),
                                  "note": "minimum = N_orb (LOBSTER's requirement for the Hamiltonian reconstruction); "
                                          "recommended leaves spare bands for the window repair and the "
                                          "window-sensitivity check"}
        if nb < norb:
            out["recommendations"].insert(0, f"too few bands: NBANDS={nb} < N_orb={norb}; no COHP is possible "
                                             f"(LOBSTER's requirement too). Rerun VASP with NBANDS >= {norb}")
        elif nb < int(factor * norb):
            out["recommendations"].append(
                f"NBANDS={nb} < {factor:g} x N_orb ({int(factor * norb)}): the frozen window may lack reliable spare "
                "bands" + (" with the character of the polarization functions" if factor == 2.0 else "") +
                "; rerun VASP with more bands")
        if rep.mesh and not (rep.mesh.get("full") or rep.mesh.get("half")):
            out["recommendations"].append("rerun VASP with ISYM = -1 (or 0)")
        return out

    def projection_key(self, cfg: BondingConfig, selection: dict, nb_use: int) -> dict:
        """Everything the projected amplitudes depend on (content hashes, not paths). ``bands`` is
        compared as 'stored >= requested'; all other entries must be identical."""
        w = self.paths["wavecar"]
        return {
            "algorithm_version": PROJECTION_ALGORITHM_VERSION,
            "wavecar_sha256_head64MiB": sha256_file(w, limit=1 << 26),
            "wavecar_sha256_tail64MiB": _sha256_tail(w, 1 << 26),
            "wavecar_bytes": w.stat().st_size,
            "potcar_sha256": sha256_file(self.paths["potcar"]),
            "structure_sha256": sha256_file(self.paths["structure"]),
            "basis_selection": {k: list(v) for k, v in sorted(selection.items())},
            "basis_library_sha256": "+".join(sha256_file(p) for p in cfg.library_paths()),
            "projection": cfg.projection,
            "bands": nb_use,
        }

    def analyze(self, config: BondingConfig | None = None, backend: str | None = None, progress=None,
                reuse: str | Path | None = None) -> "BondingResult":
        """Run the analysis. ``reuse``: a bonding.h5 written with ``write_intermediate_projection``;
        its stored projections are used if (and only if) all inputs they depend on are identical."""
        cfg = config or BondingConfig()
        rep = self.inspect()
        if not rep.ok:
            raise _core.ProwlError("InconsistentInput: inputs failed inspection:\n" + rep.summary())
        t0 = time.time()
        timings = {}
        _phase(progress, "setup")
        st = _core.read_poscar(str(self.paths["structure"]))
        pcs = _core.read_potcar(str(self.paths["potcar"]))
        vr = read_vasprun(self.paths["metadata"])
        reader = _core.WavecarReader(str(self.paths["wavecar"]))
        lib = cfg.load_library()
        selection = cfg.resolve_basis(st, pcs, lib)
        basis = _core.AtomicBasis(st, lib, selection)
        opt = _core.ProjectionOptions()
        opt.include_offsite_paw = cfg.paw_offsite
        model = _core.ProjectionModel(st, basis, pcs, opt)
        timings["setup"] = time.time() - t0
        timings.update({f"setup.{k}": v for k, v in model.setup_timings if not k.endswith("_count")})
        be = _core.make_backend(backend or cfg.backend)
        h = reader.header
        norb = model.norb
        if h["nbands"] < norb:
            raise _core.ProwlError(
                f"InvalidArgument: too few bands: NBANDS={h['nbands']}, but reconstructing the Hamiltonian in the local "
                f"basis needs at least N_orb = {norb} bands (the number of basis functions in the cell; LOBSTER has the "
                f"same requirement). Rerun VASP with NBANDS >= {norb} (>= {int(np.ceil(1.5 * norb))} recommended, so "
                "that the window repair and the window-sensitivity check have spare bands)")
        nb_use = cfg.band_count(norb, h["nbands"])
        nspin, nk = h["nspin"], h["nkpts"]
        T = [[None] * nk for _ in range(nspin)]
        paw_dev = 0.0
        warnings_list: list[str] = []
        if cfg.projection == "paw_full":
            warnings_list.append("projection paw_full (one- plus two-centre PAW terms) is experimental: it is not "
                                 "LOBSTER's convention and its effect is system-dependent (SCIENCE.md 4, 9)")
        # folded projection (PAW coupling in the orbital factors, projector amplitudes of a band sample only) unless
        # numerical.paw_norm_check is 'full' or the model has two-centre PAW terms (docs/PERFORMANCE.md, round 3)
        stride = PAW_NORM_SAMPLE_STRIDE if cfg.paw_norm_check == "sampled" and model.foldable else 1
        ncols = norb + model.nproj / stride if stride > 1 else norb + model.nproj
        flop_unit = 4.0 if reader.layout["encoding"] == "GammaHalf" else 8.0
        gflop = sum(flop_unit * reader.kpoint(0, k)["nplw"] * ncols * nb_use for k in range(nk)) * nspin / 1e9
        if (backend or cfg.backend) == "auto" and be.name != "cpu" and gflop / (nk * nspin) < AUTO_GPU_MIN_GFLOP_PER_K:
            # many small k-points: the k-parallel CPU path beats one small GPU job per k (docs/PERFORMANCE.md)
            be = _core.make_backend("cpu")
        _phase(progress, "projection", nk=nk, nspin=nspin, nbands=nb_use, norb=norb, gflop=gflop, backend=be.name)
        t1 = time.time()
        store_projection = bool(cfg.write_projection or reuse)
        key = self.projection_key(cfg, selection, nb_use) if store_projection else None
        paw_checked, paw_mode, paw_sampled_max = 0, "full", None
        nbands_total = nb_use * nk * nspin

        def project(stride):
            Tp = [[None] * nk for _ in range(nspin)]
            dev, checked = 0.0, 0
            block = cfg.block_bands(reader, nb_use)
            if be.name == "cpu" and nk > 1:
                # OpenMP over k-points in C++ (one backend per thread); chunks keep progress reports flowing
                nthr = max(1, _omp_threads())
                chunk = max(4 * nthr, nk // 50)
                budget = cfg.factor_budget_bytes()  # native project_many divides the aggregate budget among workers
                for k0 in range(0, nk, chunk):
                    ks = list(range(k0, min(nk, k0 + chunk)))
                    for k, per in zip(ks, model.project_many(reader, ks, 0, nb_use, block, budget, stride)):
                        for s, p in enumerate(per):
                            Tp[s][k] = p["a_ae"]
                            dev = max(dev, float(np.max(np.abs(np.array(p["paw_norm"]) - 1.0))))
                            checked += len(p["paw_norm"])
                    _step(progress, ks[-1] + 1, nk)
            else:
                for k in range(nk):
                    for s, p in enumerate(model.project_k(reader, k, 0, nb_use, be, block, cfg.factor_budget_bytes(),
                                                          stride)):
                        Tp[s][k] = p["a_ae"]
                        dev = max(dev, float(np.max(np.abs(np.array(p["paw_norm"]) - 1.0))))
                        checked += len(p["paw_norm"])
                    _step(progress, k + 1, nk)
            return Tp, dev, checked

        if reuse:
            T, paw_dev = _load_projection(reuse, key, nspin, nk)
            paw_mode = "reused"
            warnings_list.append(f"projected amplitudes reused from {reuse} (input hashes verified)")
            # keep the restart data in the new output (it may replace the reused file)
        else:
            T, paw_dev, paw_checked = project(stride)
            if stride > 1:
                paw_mode = "sampled"
                if paw_dev > PAW_NORM_ESCALATE_FRACTION * cfg.paw_norm_tolerance:
                    # sampled maximum close to the tolerance: redo the projection with B and the norm of every band
                    # (the full path; its amplitudes are used, identical to paw_norm_check: full)
                    paw_mode, paw_sampled_max = "sampled_escalated", paw_dev
                    T, paw_dev, paw_checked = project(1)
        self._projection_store = (key, [list(Ts) for Ts in T], paw_dev) if store_projection else None
        timings["projection"] = time.time() - t1
        if paw_dev > cfg.paw_norm_tolerance:
            msg = (f"PAW norm identity violated (max |<psi|S|psi> - 1| = {paw_dev:.2e} > {cfg.paw_norm_tolerance:.1e}); "
                   "projector data inconsistent with the wavefunctions (wrong POTCAR, LREAL, or Gamma-half convention)")
            if cfg.strict:
                raise _core.ProwlError("NumericalFailure: " + msg)
            warnings_list.append(msg)
        # Small dense linear algebra below: oversubscribed BLAS pools can be >100x slower
        # (observed with multithreaded OpenBLAS wheels), so limit BLAS threads in this phase.
        limiter = _blas_limiter(cfg.analysis_blas_threads)
        limiter.__enter__()
        try:
            res = self._finish(cfg, reader, model, basis, st, pcs, vr, lib, selection, be, T, nb_use, norb, nspin, nk,
                               paw_dev, timings, t0, rep, warnings_list, progress)
            # PAW norm check coverage: 'sampled' (folded projection), 'sampled_escalated' (sampled maximum above
            # tolerance/2, rechecked on all bands with the full projection), 'full', or 'reused' (restart file)
            d = res.result.diagnostics
            d["paw_norm_check"] = paw_mode
            d["paw_norm_escalated"] = paw_mode == "sampled_escalated"
            d["paw_norm_sampled_max"] = paw_sampled_max
            d["paw_norm_bands_checked"] = paw_checked
            d["paw_norm_bands_total"] = nbands_total
            return res
        finally:
            limiter.__exit__(None, None, None)

    def _finish(self, cfg, reader, model, basis, st, pcs, vr, lib, selection, be, T, nb_use, norb, nspin, nk, paw_dev,
                timings, t0, rep, warnings_list, progress=None):
        _phase(progress, "overlaps S(k)")
        tw = time.time()
        kfrac = np.array([reader.kpoint(0, k)["k_frac"] for k in range(nk)])
        S = list(model.overlaps(kfrac.tolist()))
        timings["overlap_k"] = time.time() - tw
        eps = np.array([[reader.kpoint(s, k)["eigenvalues"][:nb_use] for k in range(nk)] for s in range(nspin)])
        occ = np.array([[reader.kpoint(s, k)["occupations"][:nb_use] for k in range(nk)] for s in range(nspin)])
        occ_all = np.array([[reader.kpoint(s, k)["occupations"] for k in range(nk)] for s in range(nspin)])
        eps_all = np.array([[reader.kpoint(s, k)["eigenvalues"] for k in range(nk)] for s in range(nspin)])
        weights = vr.weights.copy()
        nb_proj = nb_use
        _phase(progress, "band window")
        tw = time.time()
        T_proj, eps_proj, occ_proj, S_irr = T, eps, occ, S
        win = _select_window(cfg.band_window, cfg, T, S, eps, occ, eps_all, norb, nb_use, nspin, nk, warnings_list,
                             efermi=vr.efermi, device=be if be.name != "cpu" else None)
        T, eps, occ, nb_use, window_note = win["T"], win["eps"], win["occ"], win["nb"], win["note"]
        timings["band_window"] = time.time() - tw
        mesh, full, half = detect_mesh(kfrac)
        kfrac_irr, weights_irr = kfrac, weights
        if np.any(np.abs(occ_all[:, :, nb_use:]) > 1e-8):
            raise _core.ProwlError("InvalidArgument: occupied bands lie outside the analysed band window; "
                                    "increase the band window")
        if half and not full:
            kfrac, weights, T, S, eps, occ = unfold_time_reversal(kfrac, weights, T, S, eps, occ, mesh)
        lattice = np.array(st.lattice)
        frac = np.array(st.frac)
        elements = [st.element(i) for i in range(st.natoms)]
        zval = np.array([pcs[st.atom_type[i]].zval for i in range(st.natoms)])
        inp = AnalysisInput(kfrac=kfrac, weights=weights, T=T, S=S, eps=eps, occ=occ, lattice=lattice, frac=frac,
                            elements=elements, zval=zval, orb_offsets=list(model.orbital_offsets),
                            orbital_labels=[o.label for o in basis.orbitals], overlap_terms=None,
                            efermi=vr.efermi, mesh=mesh, spin_factor=2.0 if nspin == 1 else 1.0,
                            orbital_lm=[(o.l, o.m) for o in basis.orbitals])
        _phase(progress, "bonds")
        tb = time.time()
        bonds = cfg.select_bonds(lattice, frac, elements)
        overlap_terms = model.overlap_blocks([(b.atom_a, b.atom_b, *b.R) for b in bonds])
        timings["bonds"] = time.time() - tb
        inp.overlap_terms = overlap_terms
        grid = EnergyGrid(cfg.emin, cfg.emax, cfg.npoints, cfg.sigma, vr.efermi if cfg.reference == "fermi" else 0.0,
                          kind=cfg.broadening)
        _phase(progress, "analysis")
        t2 = time.time()
        res = analyze(inp, bonds, grid, eig_backend=be if be.has_hermitian_eigensolver else None,
                      orbital_frame=cfg.orbital_frame)
        timings["analysis"] = time.time() - t2
        if cfg.window_sensitivity and bonds:
            _phase(progress, "window sensitivity")
            t3 = time.time()
            res.diagnostics["window_sensitivity"] = _window_sensitivity(
                cfg, win["mode"], T_proj, S_irr, eps_proj, occ_proj, eps_all, occ_all, norb, nb_proj, nspin, nk, inp,
                bonds, half and not full, mesh, kfrac_irr, weights_irr, be, res)
            timings["window_sensitivity"] = time.time() - t3
        timings["total"] = time.time() - t0
        if progress is not None and hasattr(progress, "finish"):
            progress.finish(timings)
        res.diagnostics["max_paw_norm_deviation"] = paw_dev
        if rep.structure is not None and self.paths["structure"].name == "CONTCAR":
            pos = _find(self.run_dir, "POSCAR")
            if pos is not None:
                p0 = _core.read_poscar(str(pos))
                if np.abs(np.array(p0.frac) - frac).max() > 1e-4:
                    warnings_list.append("CONTCAR differs from POSCAR (ionic run); the analysis uses CONTCAR, the "
                                         "geometry belonging to the final WAVECAR")
        if cfg.basis_polarization == "none" and res.diagnostics["abs_charge_spilling"] > cfg.polarization_spilling_hint:
            try:
                from .config import polarization_shells

                full = cfg.load_library(force_polarization=True)
                avail = polarization_shells(full, list(st.species), "all")
            except _core.ProwlError:
                avail = {}
            if avail:
                warnings_list.append(
                    f"charge spilling {100 * res.diagnostics['abs_charge_spilling']:.1f} % > "
                    f"{100 * cfg.polarization_spilling_hint:.0f} %: the minimal basis misses part of the occupied "
                    "space; consider basis.polarization: all (adds " +
                    ", ".join(f"{k} {' '.join(v)}" for k, v in avail.items()) + ")")
        res.diagnostics["warnings"] = warnings_list
        res.diagnostics["band_window"] = nb_use
        res.diagnostics["band_block_size"] = cfg.block_bands(reader, reader.header["nbands"])
        res.diagnostics["band_window_note"] = window_note
        res.diagnostics["timings_s"] = timings
        res.diagnostics["setup_counts"] = {k: int(v) for k, v in model.setup_timings if k.endswith("_count")}
        res.diagnostics["backend"] = be.name
        prov = {
            "prowl_version": __version__,
            "inputs": {k: {"path": str(v), "sha256": sha256_file(v) if k != "wavecar" else sha256_file(v, limit=1 << 26),
                           "sha256_scope": "full" if k != "wavecar" else "first 64 MiB"} for k, v in self.paths.items()},
            "potcar_titel": [p.titel for p in pcs],
            "basis": {"library": lib.name, "selection": selection, "provenance": json.loads(lib.provenance_json)},
            "config": cfg.to_dict(),
            "backend": {"name": be.name, "device": be.device_description},
            "conventions": CONVENTIONS,
        }
        return BondingResult(result=res, input=inp, provenance=prov, inspect=rep,
                             projection=getattr(self, "_projection_store", None))


def _select_window(mode, cfg, T, S, eps, occ, eps_all, norb, nb, nspin, nk, warnings_list, strict_norb=False,
                   efermi=None, device=None):
    """Apply a band-window policy (SCIENCE.md 5.3). Returns the windowed T/eps/occ, the band count, a
    note and the mode actually used. ``strict_norb``: an ill-defined lowest-N_orb window is an error
    instead of falling back (used for the alternate-window sensitivity check)."""
    note = None
    requested = mode
    ncand = nb - max(2, int(np.ceil(cfg.unconverged_band_fraction * nb)))
    if mode in ("auto", "repaired"):
        rep = _repaired_window(cfg, T, S, eps, occ, norb, nspin, nk, ncand, efermi, device)
        if rep["nrepaired"] == 0:
            note = (f"{requested} -> norb: lowest {norb} bands well conditioned at every k-point "
                    f"(min eigenvalue {rep['min_cond']:.2e}); identical to LOBSTER's window")
            return {"T": rep["T"], "eps": rep["eps"], "occ": rep["occ"], "nb": norb, "note": note, "mode": "norb"}
        else:
            note = (f"{requested} -> repaired: lowest {norb} bands at {rep['nk_total'] - rep['nrepaired']} of "
                    f"{rep['nk_total']} (spin, k); minimal repair at {rep['nrepaired']} with up to {rep['max_extra']} "
                    f"extra band(s) (highest candidate {rep['max_energy']:+.2f} eV" +
                    (" vs E_F" if efermi is not None else "") + f"; min eigenvalue {rep['min_cond']:.2e})")
            warnings_list.append(
                f"LOBSTER's band window (lowest {norb} bands) is ill-conditioned at {rep['nrepaired']} of "
                f"{rep['nk_total']} (spin, k) points; there it was repaired with the fewest extra bands in energy "
                f"order (up to {rep['max_extra']}). Values are not identical to LOBSTER's but do not depend on "
                "NBANDS beyond the repair (SCIENCE.md 5.3)")
            return {"T": rep["T"], "eps": rep["eps"], "occ": rep["occ"], "nb": norb, "note": note, "mode": "repaired"}
    if mode == "norb":
        cond = min(window_conditioning(T[s][k], S[k], norb) for s in range(nspin) for k in range(nk))
        cond_raw = cond
        nsplit = sum(cut_at_gap(np.ascontiguousarray(eps_all[s, k]), norb) != norb
                     for s in range(nspin) for k in range(nk))
        if nsplit:
            cond = min(cond, 0.0)
        if cond >= cfg.window_condition_threshold:
            mode = "norb"
            note = f"{requested} -> norb: lowest {norb} bands span the local space (min eigenvalue {cond:.2e})"
        elif strict_norb:
            raise _core.ProwlError(f"InvalidArgument: lowest-{norb} (LOBSTER) window ill-defined (min eigenvalue "
                                    f"{cond:.2e}; degenerate multiplet split at {nsplit} k-point(s))")
        elif requested == "norb" and cond_raw < 1e-10:
            raise _core.ProwlError(
                f"NumericalFailure: LOBSTER's band window (lowest {norb} bands) is singular for this system (min "
                f"eigenvalue {cond_raw:.1e}): those bands do not span the local orbital space at some k-points. "
                "LOBSTER continues with an orthonormalization warning and inconsistent values; PROWL does not. "
                "Use profile 'prowl' (minimally repaired LOBSTER window)")
        elif requested == "norb":
            # LOBSTER's behaviour: keep the window, say that it is ill-conditioned
            note = (f"norb: lowest {norb} bands, ill-conditioned (min eigenvalue {cond:.2e}; degenerate multiplet "
                    f"split at {nsplit} k-point(s))")
            warnings_list.append(f"LOBSTER's band window (lowest {norb} bands) is ill-conditioned here (min eigenvalue "
                                 f"{cond:.1e}, degenerate multiplet split at {nsplit} k-point(s)); as in LOBSTER, values "
                                 "for symmetry-equivalent bonds may differ. Profile 'prowl' repairs the window there")
    if mode == "frozen":
        if ncand < norb:
            need = int(np.ceil((norb + 2) / (1.0 - cfg.unconverged_band_fraction)))
            raise _core.ProwlError(
                f"InvalidArgument: the frozen band window needs more bands than the WAVECAR has: only {ncand} reliable "
                f"bands (NBANDS={nb} minus the top {cfg.unconverged_band_fraction:.0%}) for {norb} basis functions. "
                f"Rerun VASP with NBANDS >= {max(need, int(1.5 * norb))}, or use profile 'lobster' (always the lowest "
                f"{norb} bands, as LOBSTER; symmetry-equivalent bonds may then differ)")
        Tn, en, on = [[None] * nk for _ in range(nspin)], np.zeros((nspin, nk, norb)), np.zeros((nspin, nk, norb))
        worst, mingap = 1.0, 1.0
        for s in range(nspin):
            for k in range(nk):
                Tn[s][k], en[s, k], on[s, k], q, g = frozen_window_subspace(
                    T[s][k], S[k], np.ascontiguousarray(eps[s, k]), np.ascontiguousarray(occ[s, k]), norb, ncand)
                worst, mingap = min(worst, q), min(mingap, g)
        if mingap >= cfg.frozen_gap_threshold or requested == "frozen":
            T, eps, occ = Tn, en, on
            nb = norb
            note = ((note + " -> frozen" if note else "frozen") +
                    f": occupied bands frozen, complement from bands <= {ncand} (weakest state weight "
                    f"{worst:.3f}, min selection gap {mingap:.1e})")
            if mingap < cfg.frozen_gap_threshold:
                note += " WARNING: complement not unique (degenerate selection); per-bond values are ambiguous"
                warnings_list.append(f"frozen window: complement selection not unique (gap {mingap:.1e} < "
                                     f"{cfg.frozen_gap_threshold:g}); per-bond values may break symmetry. Increase "
                                     "NBANDS or use band_window 'all'")
        else:
            mode = "all"
            note = ((note or "") + f" -> frozen rejected (selection gap {mingap:.1e} < "
                    f"{cfg.frozen_gap_threshold:g}, complement not unique) -> all {ncand} reliable bands, "
                    "polar orthonormalization")
    if mode == "norb":
        T = [[T[s][k][:, :norb] for k in range(nk)] for s in range(nspin)]
        eps, occ = eps[:, :, :norb], occ[:, :, :norb]
        nb = norb
    elif mode == "all":
        ncut = min(cut_at_gap(np.ascontiguousarray(eps_all[s, k]), ncand) for s in range(nspin) for k in range(nk))
        T = [[T[s][k][:, :ncut] for k in range(nk)] for s in range(nspin)]
        eps, occ = eps[:, :, :ncut], occ[:, :, :ncut]
        nb = ncut
    return {"T": T, "eps": eps, "occ": occ, "nb": nb, "note": note, "mode": mode}


def _next_gap(e, n: int, tol: float = 1e-4) -> int:
    """Smallest n' >= n such that bands n'-1 and n' are not degenerate (cut never splits a multiplet)."""
    while 0 < n < len(e) and e[n] - e[n - 1] < tol:
        n += 1
    return n


def _repaired_window(cfg, T, S, eps, occ, norb, nspin, nk, ncand, efermi=None, device=None) -> dict:
    """LOBSTER's lowest-N_orb window with minimal repair (SCIENCE.md 5.3): at each (spin, k) the lowest
    N_orb bands are used unchanged if they do not split a degenerate multiplet and span the local space
    (min eig(Tp Tp^H) >= window_condition_threshold). Otherwise the occupied bands are kept and the
    candidate set is enlarged band group by band group in energy order until the best (maximal local
    weight) completion is well conditioned and uniquely selected; it is then Rayleigh-Ritz
    re-diagonalized (frozen_window_subspace on that candidate set). Bands above the repair are never
    used, so the result is NBANDS-independent once NBANDS covers the repair."""
    thr, gthr = cfg.window_condition_threshold, cfg.frozen_gap_threshold
    Tn = [[None] * nk for _ in range(nspin)]
    en, on = np.zeros((nspin, nk, norb)), np.zeros((nspin, nk, norb))
    nrep, max_extra, max_energy, min_cond = 0, 0, -np.inf, np.inf
    # C++ (analysis.cpp repaired_window): exact fast search; on the GPU stepper when a device backend is in use
    res = _core.repaired_windows(T, list(S), [[np.ascontiguousarray(eps[s, k]) for k in range(nk)] for s in range(nspin)],
                                 [[np.ascontiguousarray(occ[s, k]) for k in range(nk)] for s in range(nspin)],
                                 norb, ncand, thr, gthr, device=device)
    for i, (tw, ew, ow, c, nc, failed) in enumerate(res):
        s, k = divmod(i, nk)
        if failed:
            need = int(np.ceil((failed + 2) / (1.0 - cfg.unconverged_band_fraction)))
            raise _core.ProwlError(
                f"InvalidArgument: LOBSTER's band window is ill-conditioned at spin {s + 1}, k-point {k + 1} and "
                f"the repair needs more than the {ncand} reliable bands; rerun VASP with NBANDS >= {need}")
        Tn[s][k], en[s, k], on[s, k] = tw, ew, ow
        min_cond = min(min_cond, c)
        if nc:
            nrep += 1
            max_extra = max(max_extra, nc - norb)
            max_energy = max(max_energy, float(eps[s, k][nc - 1]) - (efermi or 0.0))
    return {"T": Tn, "eps": en, "occ": on, "nrepaired": nrep, "nk_total": nspin * nk, "max_extra": max_extra,
            "max_energy": max_energy, "min_cond": min_cond}


def _window_sensitivity(cfg, used_mode, T, S, eps, occ, eps_all, occ_all, norb, nb, nspin, nk, inp, bonds, unfold,
                        mesh, kfrac, weights, be, res) -> dict:
    """ICOHP etc. under the other well-defined band window (frozen <-> LOBSTER's lowest-N_orb), reusing
    the projections. The window dependence is a property of minimal-basis COHP (SCIENCE.md 9), so it is
    reported with every result instead of being hidden by the default."""
    alt = "frozen" if used_mode in ("norb", "repaired") else "norb"
    out = {"window": alt, "available": False, "nbands": int(nb)}
    try:
        w = _select_window(alt, cfg, T, S, eps, occ, eps_all, norb, nb, nspin, nk, [], strict_norb=True)
    except _core.ProwlError as e:
        out["reason"] = str(e).split(": ", 1)[-1]
        return out
    if np.any(np.abs(occ_all[:, :, w["nb"]:]) > 1e-8):
        out["reason"] = "occupied bands outside the alternate window"
        return out
    T2, S2, e2, o2, kf, wt = w["T"], S, w["eps"], w["occ"], kfrac, weights
    if unfold:
        kf, wt, T2, S2, e2, o2 = unfold_time_reversal(kfrac, weights, T2, S, e2, o2, mesh)
    inp2 = replace(inp, kfrac=kf, weights=wt, T=T2, S=S2, eps=e2, occ=o2)
    try:
        r2 = analyze(inp2, bonds, EnergyGrid(-1.0, 1.0, 3, 0.1, inp.efermi),
                     eig_backend=be if be.has_hermitian_eigensolver else None)
    except _core.ProwlError as e:  # e.g. the alternate window does not span the local space
        out["reason"] = str(e).split(": ", 1)[-1]
        return out
    a, b = res.icohp.sum(axis=0), r2.icohp.sum(axis=0)
    strong = np.abs(a) > 0.1
    rel = float(np.max(np.abs(b[strong] - a[strong]) / np.abs(a[strong]))) if strong.any() else 0.0
    out.update(available=True, note=w["note"], icohp=r2.icohp, icoop=r2.icoop, icobi=r2.icobi,
               lowdin_charge=r2.lowdin_charge, mulliken_charge=r2.mulliken_charge,
               abs_charge_spilling=r2.diagnostics["abs_charge_spilling"], max_rel_icohp_change=rel)
    return out


# backend "auto" uses the GPU only when one (k, spin) carries at least this much projection work
AUTO_GPU_MIN_GFLOP_PER_K = 1.0
# numerical.paw_norm_check 'sampled': PAW norm of the bands 0, 8, 16, ... and the last band of each (spin, k);
# a sampled maximum above this fraction of paw_norm_tolerance triggers the full check of every band
PAW_NORM_SAMPLE_STRIDE = 8
PAW_NORM_ESCALATE_FRACTION = 0.5


def _paw_check_note(d: dict) -> str:
    mode = d.get("paw_norm_check")
    if mode == "sampled":
        return f"  [PAW norm checked on {d['paw_norm_bands_checked']} of {d['paw_norm_bands_total']} bands]"
    if mode == "sampled_escalated":
        return (f"  [sampled max {d['paw_norm_sampled_max']:.1e} > tolerance/2: rechecked all "
                f"{d['paw_norm_bands_checked']} bands]")
    return ""


def _phase(progress, name: str, **info) -> None:
    if progress is not None and hasattr(progress, "phase"):
        progress.phase(name, **info)


def _step(progress, done: int, total: int) -> None:
    """Projection progress: reporter.step(done, total), or a plain callable with the last finished k index."""
    if progress is None:
        return
    if hasattr(progress, "step"):
        progress.step(done, total)
    else:
        progress(done - 1)


def _omp_threads() -> int:
    """Read the native runtime's actual worker setting, including OpenMP initialization."""
    return _core.omp_max_threads()


def _blas_limiter(n: int):
    """Coordinate the native OpenBLAS policy with projections in other Python threads/models."""
    return _core.BlasThreads(n)


# Bump when anything that changes A_AE numerically changes (tables, quadratures, projector handling).
PROJECTION_ALGORITHM_VERSION = "2026-11-04.1"


CONVENTIONS = {
    "projection": "A_AE = <chi|psi~> + sum_i <chi|phi_i - phi~_i><p~_i|psi~> (PAW, see SCIENCE.md 3)",
    "bloch_gauge": "chi_mu,k = sum_R exp(i k.R) chi_mu(r - tau_mu - R)",
    "orthonormalization": "Löwdin S^{-1/2}; projected bands re-orthonormalized by polar decomposition",
    "hamiltonian": "H(k) = C diag(eps) C^H in Löwdin basis; H(R) by discrete Fourier transform on full mesh",
    "cohp_sign": "COHP (negative = bonding)",
    "cohp_bond": "pair (A, B, R) counted once: sum_{mu in A, nu in B} Re[C*_mu H_mu,nu(R) e^{ik.R} C_nu]",
    "spin": "spin_factor 2 for ISPIN=1; per channel for ISPIN=2",
    "energy_unit": "eV",
    "integrated": "occupation-weighted sums with VASP occupations (no smearing)",
}


def unfold_time_reversal(kfrac, weights, T, S, eps, occ, mesh):
    """Complete a time-reversal half mesh: psi(-k) = psi(k)*, real orbitals => T(-k) = T(k)*."""
    n = np.array(mesh)
    keys = {tuple(np.round(np.mod(k * n, n)).astype(int) % n): i for i, k in enumerate(kfrac)}
    new_k, new_w = list(kfrac), list(weights)
    T = [list(Ts) for Ts in T]
    S = list(S)
    eps_l = [list(e) for e in eps]
    occ_l = [list(o) for o in occ]
    for key, i in list(keys.items()):
        mk = tuple((-np.array(key)) % n)
        if mk in keys:
            continue
        new_w[i] = weights[i] / 2
        new_k.append(-kfrac[i])
        new_w.append(weights[i] / 2)
        for s in range(len(T)):
            T[s].append(np.conj(T[s][i]))
            eps_l[s].append(eps[s][i])
            occ_l[s].append(occ[s][i])
        S.append(np.conj(S[i]))
        keys[mk] = len(new_k) - 1
    return np.array(new_k), np.array(new_w), T, S, np.array(eps_l), np.array(occ_l)


def _load_projection(path, key: dict, nspin: int, nk: int):
    import h5py

    with h5py.File(path, "r") as f:
        if "projection" not in f:
            raise _core.ProwlError(f"InvalidArgument: {path} contains no stored projection "
                                    "(write it with output.write_intermediate_projection: true)")
        g = f["projection"]
        stored = json.loads(g.attrs["key_json"])
        diff = [k for k in key if k != "bands" and stored.get(k) != key[k]]
        if stored.get("bands", -1) < key["bands"]:
            diff.append(f"bands (stored {stored.get('bands')}, needed {key['bands']})")
        if diff:
            raise _core.ProwlError("InconsistentInput: cannot reuse projection from " + str(path) +
                                    "; differing inputs: " + ", ".join(diff))
        nb = key["bands"]
        T = [[g[f"T_s{s}_k{k}"][:, :nb] for k in range(nk)] for s in range(nspin)]
        return T, float(g.attrs["max_paw_norm_deviation"])


def _sha256_tail(path, n: int) -> str:
    import hashlib

    size = Path(path).stat().st_size
    with open(path, "rb") as fh:
        fh.seek(max(0, size - n))
        return hashlib.sha256(fh.read()).hexdigest()


@dataclass
class BondingResult:
    result: AnalysisResult
    input: AnalysisInput
    provenance: dict
    inspect: InspectReport
    projection: tuple | None = None  # (key, T[spin][k], max PAW norm deviation) when requested

    def quality_report(self) -> str:
        d = self.result.diagnostics
        L = ["PROWL quality report",
             f"  backend                      : {d['backend']}",
             f"  band window                  : {d['band_window']} bands" + (f"  [{d['band_window_note']}]" if d.get('band_window_note') else ""),
             f"  max |PAW norm - 1|           : {d['max_paw_norm_deviation']:.2e}" + _paw_check_note(d),
             f"  abs. charge spilling (occ.)  : {100 * d['abs_charge_spilling']:.3f} %",
             f"  abs. total spilling          : {100 * d['abs_total_spilling']:.3f} %",
             f"  min overlap eigenvalue       : {d['min_overlap_eigenvalue']:.4e}",
             f"  max reconstruction residual  : {d['max_reconstruction_residual_ev']:.2e} eV (occupied bands)",
             f"  electrons (DFT / Löwdin / Mulliken): {d['electrons_dft']:.4f} / {d['electrons_projected_lowdin']:.4f} / "
             f"{d['electrons_projected_mulliken']:.4f}",
             "  timings [s]                  : " + ", ".join(f"{k} {v:.2f}" for k, v in d["timings_s"].items())]
        ws = d.get("window_sensitivity")
        if ws:
            if ws["available"]:
                L.append(f"  window sensitivity           : max |dICOHP|/|ICOHP| = {100 * ws['max_rel_icohp_change']:.1f} % "
                         f"with the {ws['window']} window (bonds with |ICOHP| > 0.1 eV)" +
                         (f"; the frozen value depends on NBANDS (here {ws['nbands']})" if ws["window"] == "frozen" else ""))
            else:
                L.append(f"  window sensitivity           : {ws['window']} window not available ({ws.get('reason', '')})")
        for w in d.get("warnings", []):
            L.append(f"  WARNING: {w}")
        bad = [b for b, ok in zip(self.result.bonds, d["bonds_alias_free"]) if not ok]
        if bad:
            L.append(f"  WARNING: {len(bad)} bond(s) not resolvable on this k-mesh (periodic-image aliasing)")
        return "\n".join(L)

    def save(self, path: str | Path) -> Path:
        from .io import write_hdf5
        return write_hdf5(self, path)
