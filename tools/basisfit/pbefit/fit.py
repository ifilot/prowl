"""Fit STO shells to VASP free-atom orbitals in the PAW metric used by PROWL/LOBSTER.

For one atom at the origin of a large box (Gamma only) PROWL gives a_ae[p m, b] = <chi_pm|psi_b> (all-electron
PAW overlap incl. one-centre terms) for single-primitive "shells" p.  For a radial contraction c over primitives
of angular momentum l, the captured weight of the bands B of a shell is
    W(c) = c^T M c / c^T S c,   M_pq = sum_{m, b in B} Re(a_pmb conj(a_qmb)),
with S the analytic one-centre STO overlap; W/|B| is the mean fidelity (1 - orbital spilling).  The best contraction
is the top generalized eigenvector of (M, S).

Primitive sets.  For each l the fitted shells use the Koga primitive set of that l (shared by all Koga shells of the
l) plus m = 0..extra_max diffuse even-tempered n = l+1 primitives chosen exhaustively from a candidate ladder.
share = "l": one extra set per l, chosen to maximise the summed fidelity of all fitted shells of the l
(LOBSTER .sto compatible); share = "none": each shell has its own extras.  If an l has no Koga primitives
(e.g. Li 2p, La 4f), a base set is grown greedily from a wider ladder first.

No pbeVaspFit2015 number enters this module.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.special import gammaincc

from prowl import _core

from .shells import LETTER, koga, shell_list, potcar_info

BOHR = 0.529177210903

DEFAULTS = {
    "mode": "2026",          # "2015": fit only added shells; "2026": refit valence + added shells
    "share": "l",            # "l" (shared extras per l) or "none"
    "extra_max": 3,
    "et_count": 16, "et_zmax": 3.0,                 # diffuse candidate ladder (zmin from the box)
    "ladder_count": 28, "ladder_zmax": 25.0,         # ladder for an l without Koga primitives
    "greedy_max": 8, "greedy_tol": 5e-5,
    "tail_max": 1e-3,        # max norm fraction of a candidate beyond half the box
    "cond_max": 1e8, "fidelity_max": 1 + 5e-4, "lchar_min": 0.9,
    "coef_max": 10.0,        # max |coefficient| of the normalized contraction: larger = cancelling diffuse extras,
                             # fragile under small exponent changes (Ti 4s, Sc 4s, Th 7s; user decision 2026-11-06)
    "min_ratio": 1.3,        # extra exponents differ by at least this factor from each other / same-n base ones
    "require_nodes": True,   # reject primitive sets whose fitted shells have the wrong number of radial nodes
    "spins": "auto",         # spin-polarized atoms: "majority" (auto) or "all"; minority states of strongly
                             # polarized atoms are unbound (Li: minority 2s at -0.1 eV, 2p above the vacuum level)
}


# ----------------------------------------------------------------------------------------------- STO algebra
def _nrm(n, z):
    return (2 * z) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def sto_S(prims):
    """One-centre overlap of normalized STO primitives [(n, zeta)]."""
    k = len(prims)
    S = np.empty((k, k))
    for i, (na, za) in enumerate(prims):
        for j, (nb, zb) in enumerate(prims):
            S[i, j] = _nrm(na, za) * _nrm(nb, zb) * math.factorial(na + nb) / (za + zb) ** (na + nb + 1)
    return S


def sto_rmat(prims, power=1):
    """<p|r^power|q> for normalized primitives."""
    k = len(prims)
    R = np.empty((k, k))
    for i, (na, za) in enumerate(prims):
        for j, (nb, zb) in enumerate(prims):
            R[i, j] = _nrm(na, za) * _nrm(nb, zb) * math.factorial(na + nb + power) / (za + zb) ** (na + nb + 1 + power)
    return R


def radial(prims, c, r):
    """r * R(r) (the reduced radial function u(r)) of a contraction."""
    u = np.zeros_like(r)
    for (n, z), ci in zip(prims, c):
        u += ci * _nrm(n, z) * r ** n * np.exp(-z * r)
    return u


def count_nodes(prims, c, rmax=40.0):
    r = np.linspace(1e-4, rmax, 40001)
    u = radial(prims, c, r)
    big = np.abs(u) > 1e-3 * np.abs(u).max()
    s = np.sign(u[big])
    return int(np.sum(s[1:] != s[:-1]))


def zmin_for(n, half_bohr, tail):
    """Smallest exponent of an n STO whose norm beyond r = half box is below `tail`."""
    for z in np.arange(0.05, 5.0, 0.002):
        if gammaincc(2 * n + 1, 2 * z * half_bohr) < tail:
            return float(z)
    raise ValueError("box too small")


# ------------------------------------------------------------------------------------------------ projection
def _project_chunk(st, pcs, reader, be, el, l, chunk, opt):
    shells, sel = [], []
    for i, (n, z) in enumerate(chunk):
        lab = f"{LETTER[l]}{i:03d}"
        shells.append({"label": lab, "n": n, "l": l, "occupation": 0.0, "energy": 0.0,
                       "primitives": [{"n": n, "zeta": z, "coef": 1.0}]})
        sel.append(lab)
    doc = {"schema": "prowl-sto-basis/1", "name": "pool", "units": {"length": "bohr", "energy": "hartree"},
           "provenance": {}, "elements": {el: {"Z": 0, "shells": shells}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(doc, fh)
    try:
        lib = _core.StoLibrary.load(fh.name)
    finally:
        os.unlink(fh.name)
    basis = _core.AtomicBasis(st, lib, {el: sel})
    model = _core.ProjectionModel(st, basis, pcs, opt)
    nb = reader.header["nbands"]
    out = model.project_k(reader, 0, 0, nb, be)        # one entry per spin
    a = np.stack([np.asarray(p["a_ae"]).reshape(len(chunk), 2 * l + 1, nb) for p in out])
    pawn = np.stack([np.asarray(p["paw_norm"]) for p in out])
    return a, pawn


def project_pool(run: Path, el: str, pool: dict, chunk=6):
    """pool {l: [(n, zeta)]} -> ({l: A[s, p, m, b]}, eps[s, b], occ[s, b], paw_norm[s, b])."""
    st = _core.read_poscar(str(run / "POSCAR"))
    pcs = _core.read_potcar(str(run / "POTCAR"))
    reader = _core.WavecarReader(str(run / "WAVECAR"))
    be = _core.make_backend("cpu")
    opt = _core.ProjectionOptions()
    opt.overlap_nq = 401
    out, pawn = {}, None
    for l, prims in pool.items():
        parts = []
        for i in range(0, len(prims), chunk):
            a, pawn = _project_chunk(st, pcs, reader, be, el, l, prims[i:i + chunk], opt)
            parts.append(a)
        out[l] = np.concatenate(parts, axis=1)
    ns = reader.header["nspin"]
    eps = np.array([reader.kpoint(s, 0)["eigenvalues"] for s in range(ns)])
    occ = np.array([reader.kpoint(s, 0)["occupations"] for s in range(ns)])
    return out, eps, occ, pawn


# ---------------------------------------------------------------------------------------------- fitting
def _M(A, bands):
    """M_pq summed over spins and the shell's bands; bands = [list per spin]. Returns (M, number of states)."""
    M = 0
    nst = 0
    for s, bs in enumerate(bands):
        Ab = A[s][:, :, bs]
        M = M + np.einsum("pmb,qmb->pq", Ab, Ab.conj()).real
        nst += len(bs)
    return M, nst


def best(Mn, S):
    """Top generalized eigenpair of (Mn, S), Mn already divided by the number of states."""
    w, v = sla.eigh(Mn, S)
    c = v[:, -1] / math.sqrt(v[:, -1] @ S @ v[:, -1])
    if c[np.argmax(np.abs(c))] < 0:
        c = -c
    return float(w[-1]), c


def fixed_fid(Mn, S, c):
    return float((c @ Mn @ c) / (c @ S @ c))


class Fitter:
    def __init__(self, run: Path, potcar: str, cfg: dict | None = None):
        self.run, self.potcar = Path(run), potcar
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cfg["mode"] = str(self.cfg["mode"])
        self.el = potcar_info(potcar)["element"]
        self.koga = {s["label"]: s for s in koga()[self.el]["shells"]}
        self.shells = shell_list(potcar)
        self._caps = {}
        lat = np.array(_core.read_poscar(str(self.run / "POSCAR")).lattice)
        self.half = float(min(np.linalg.norm(lat, axis=1)) / 2 / BOHR)

    # -- pools
    def _build_pool(self):
        c = self.cfg
        vasp_shells = self.shells["valence"] + self.shells["added"]
        self.ls = sorted({LETTER.index(s[-1]) for s in vasp_shells})
        self.pool, self.idx, self.base, self.et, self.ladder, self.zmin = {}, {}, {}, {}, {}, {}
        for l in self.ls:
            prims, idx = [], {}

            def add(n, z):
                key = (n, round(z, 6))
                if key not in idx:
                    idx[key] = len(prims)
                    prims.append((n, float(z)))
                return idx[key]
            base = []
            for s in self.koga.values():
                if s["l"] == l:
                    for p in s["primitives"]:
                        j = add(p["n"], p["zeta"])
                        if j not in base:
                            base.append(j)
            zmin = zmin_for(l + 1, self.half, c["tail_max"])
            et = [add(l + 1, z) for z in np.geomspace(zmin, c["et_zmax"], c["et_count"])]
            ladder = [add(l + 1, z) for z in np.geomspace(zmin, c["ladder_zmax"], c["ladder_count"])]
            self.pool[l], self.idx[l], self.base[l], self.et[l], self.ladder[l] = prims, idx, base, et, ladder
            self.zmin[l] = zmin

    def project(self):
        self._build_pool()
        import hashlib, pickle
        key = hashlib.sha256(repr(sorted((l, tuple(p)) for l, p in self.pool.items())).encode()).hexdigest()[:16]
        cache = self.run / f"pbefit_pool_{key}.pkl"
        wav = self.run / "WAVECAR"
        if cache.exists() and cache.stat().st_mtime > wav.stat().st_mtime:
            self.A, self.eps, self.occ, self.pawn = pickle.loads(cache.read_bytes())
        else:
            self.A, self.eps, self.occ, self.pawn = project_pool(self.run, self.el, self.pool)
            cache.write_bytes(pickle.dumps((self.A, self.eps, self.occ, self.pawn)))
        self.ns, self.nb = self.eps.shape
        # l character of each band (completeness of the whole pool of that l, conditioned)
        self.char = {}
        for l in self.ls:
            S = sto_S(self.pool[l])
            w, U = np.linalg.eigh(S)
            keep = w > 1e-9 * w.max()
            X = U[:, keep] / np.sqrt(w[keep])
            self.char[l] = np.array([np.einsum("kmb,kmb->b", B, B.conj()).real
                                     for B in (np.einsum("pk,pmb->kmb", X, self.A[l][s]) for s in range(self.ns))])

    # -- band assignment: shells of one l ordered by n take consecutive groups of 2l+1 l-like bands
    def assign(self):
        vasp_shells = self.shells["valence"] + self.shells["added"]
        self.bands = {}
        for l in self.ls:
            same = sorted([t for t in vasp_shells if LETTER.index(t[-1]) == l], key=lambda t: int(t[:-1]))
            for s in range(self.ns):
                cand = sorted([b for b in range(self.nb) if self.char[l][s, b] > 0.5], key=lambda b: self.eps[s, b])
                for k, t in enumerate(same):
                    bs = cand[k * (2 * l + 1):(k + 1) * (2 * l + 1)]
                    self.bands.setdefault(t, []).append(bs)
        if self.ns == 2 and self.cfg["spins"] in ("auto", "majority"):
            nel = self.occ.sum(axis=1)
            maj = int(np.argmax(nel))
            self.bands = {t: [bs if s == maj else [] for s, bs in enumerate(b)] for t, b in self.bands.items()}
        return self.bands

    # -- fitting of one l
    def _Mn(self, l, t):
        M, nst = _M(self.A[l], self.bands[t])
        return M / nst

    def _cap(self, l, t):
        """Captured-fraction cap of shell t: max(1, PAW excess of the Koga-primitive refit) + (fidelity_max - 1).
        The Koga primitives are compact and well conditioned; if they already capture more than 1, the excess is
        PAW partial-wave incompleteness (seen for compact d shells), not overfitting."""
        key = (l, t)
        if key not in self._caps:
            base = self.base[l]
            f0 = 1.0
            if base:
                S = sto_S([self.pool[l][j] for j in base])
                f0 = max(1.0, best(self._Mn(l, t)[np.ix_(base, base)], S)[0])
            self._caps[key] = f0 + (self.cfg["fidelity_max"] - 1.0)
        return self._caps[key]

    def _ok(self, l, sub, fits):
        """Physical constraints on a candidate set: captured fraction <= cap, correct node counts."""
        c = self.cfg
        for t, (f, coef) in fits.items():
            if f > self._cap(l, t):
                return False
            if c["require_nodes"] and count_nodes([self.pool[l][j] for j in sub], coef) != int(t[:-1]) - l - 1:
                return False
        return True

    def _grow_base(self, l, targets):
        """Greedy forward selection from the wide ladder for an l without Koga primitives
        (only sets that satisfy the physical constraints; stop when the gain is below greedy_tol)."""
        c = self.cfg
        S_all = sto_S(self.pool[l])
        Mns = {t: self._Mn(l, t) for t in targets}
        sel, cur = [], 0.0
        while len(sel) < c["greedy_max"]:
            trial = []
            for j in self.ladder[l]:
                if j in sel or not self._spaced(l, sel, (j,)):
                    continue
                sub = sel + [j]
                Ss = S_all[np.ix_(sub, sub)]
                if np.linalg.cond(Ss) > c["cond_max"]:
                    continue
                fits = {t: best(Mn[np.ix_(sub, sub)], Ss) for t, Mn in Mns.items()}
                if not self._ok(l, sub, fits):
                    continue
                trial.append((sum(f for f, _ in fits.values()) / len(fits), j))
            if not trial:
                break
            f, j = max(trial)
            if sel and f - cur < c["greedy_tol"]:
                break
            sel.append(j)
            cur = f
        return sorted(sel, key=lambda j: -self.pool[l][j][1])

    def _spaced(self, l, base, combo):
        r = self.cfg["min_ratio"]
        P = self.pool[l]
        for i, a in enumerate(combo):
            for b in list(combo[:i]) + [j for j in base if P[j][0] == P[a][0]]:
                q = P[a][1] / P[b][1]
                if max(q, 1 / q) < r:
                    return False
        return True

    def fit_l(self, l, targets, share):
        """Returns {m: {"extras": [...], "shells": {t: (fidelity, coef over set)}, "set": [...]}}."""
        c = self.cfg
        S_all = sto_S(self.pool[l])
        base = list(self.base[l]) or self._grow_base(l, targets)
        cands = [j for j in self.et[l] if j not in base]
        groups = [targets] if share == "l" else [[t] for t in targets]
        res = {m: {"shells": {}, "extras": {}} for m in range(c["extra_max"] + 1)}
        for grp in groups:
            Mns = {t: self._Mn(l, t) for t in grp}
            for m in range(c["extra_max"] + 1):
                bestv, fallback = None, None
                for combo in combinations(cands, m):
                    if not self._spaced(l, base, combo):
                        continue
                    sub = base + list(combo)
                    if not sub:
                        continue
                    Ss = S_all[np.ix_(sub, sub)]
                    if m and np.linalg.cond(Ss) > c["cond_max"]:
                        continue
                    fits = {t: best(Mns[t][np.ix_(sub, sub)], Ss) for t in grp}
                    tot = sum(f for f, _ in fits.values())
                    if fallback is None or tot > fallback[0] + 1e-12:
                        fallback = (tot, sub, fits)
                    if bestv is not None and tot <= bestv[0] + 1e-12:
                        continue
                    if not self._ok(l, sub, fits):
                        continue
                    bestv = (tot, sub, fits)
                if bestv is None:
                    bestv = fallback
                if bestv is None:
                    continue
                _, sub, fits = bestv
                for t in grp:
                    res[m]["shells"][t] = {"set": sub, "fidelity": fits[t][0], "coef": fits[t][1]}
                    res[m]["extras"][t] = [j for j in sub if j not in base]
        return res, base

    # -- checks and bookkeeping
    def shell_entry(self, l, t, setidx, coef, fid):
        prims = [self.pool[l][j] for j in setidx]
        S = sto_S(prims)
        coef = np.asarray(coef) / math.sqrt(coef @ S @ coef)
        r1 = float(coef @ sto_rmat(prims, 1) @ coef)
        n = int(t[:-1])
        nodes = count_nodes(prims, coef)
        lch = [float(self.char[l][s, b]) for s, bs in enumerate(self.bands[t]) for b in bs]
        nb_ok = all(len(bs) == 2 * l + 1 for bs in self.bands[t] if bs) and any(self.bands[t])
        e = {"label": t, "n": n, "l": l, "fidelity": fid, "orbital_spilling": 1 - fid, "r_mean_bohr": r1,
             "nodes": nodes, "cond_S": float(np.linalg.cond(S)), "lchar_min": min(lch) if lch else 0.0,
             "eps_ev": [float(self.eps[s, b]) for s, bs in enumerate(self.bands[t]) for b in bs],
             "occ": [float(self.occ[s, b]) for s, bs in enumerate(self.bands[t]) for b in bs],
             "primitives": [{"n": p[0], "zeta": p[1], "coef": float(ci)} for p, ci in zip(prims, coef)]}
        if t in self.koga:   # overlap with the Koga (HF) shell
            kp = [(p["n"], p["zeta"]) for p in self.koga[t]["primitives"]]
            kc = np.array([p["coef"] for p in self.koga[t]["primitives"]])
            allp = prims + kp
            Sx = sto_S(allp)
            v1 = np.concatenate([coef, np.zeros(len(kp))]); v2 = np.concatenate([np.zeros(len(prims)), kc])
            e["overlap_koga"] = float(abs(v1 @ Sx @ v2) / math.sqrt((v1 @ Sx @ v1) * (v2 @ Sx @ v2)))
        c = self.cfg
        e["fidelity_cap"] = self._cap(l, t)
        e["checks"] = {"fidelity_le_max": fid <= e["fidelity_cap"] + 1e-12, "nodes_ok": nodes == n - l - 1,
                       "cond_ok": e["cond_S"] < c["cond_max"], "lchar_ok": e["lchar_min"] > c["lchar_min"],
                       "bands_found": nb_ok,
                       "coef_ok": max(abs(float(ci)) for ci in coef) <= c["coef_max"]}
        e["passed"] = all(e["checks"].values())
        return e

    def run_fit(self):
        c = self.cfg
        self.project()
        self.assign()
        val, add = self.shells["valence"], self.shells["added"]
        targets = add if c["mode"] == "2015" else val + add
        rep = {"potcar": self.potcar, "element": self.el, "potcar_info": potcar_info(self.potcar), "run": str(self.run),
               "config": c, "shells": self.shells, "half_box_bohr": self.half, "zmin": self.zmin,
               "paw_norm_max_dev": float(np.abs(self.pawn - 1).max()), "nspin": self.ns,
               "bands": [{"eps_ev": self.eps[s].tolist(), "occ": self.occ[s].tolist(),
                          **{LETTER[l]: self.char[l][s].tolist() for l in self.ls}} for s in range(self.ns)],
               "fits": {}}
        for l in self.ls:
            tl = [t for t in targets if LETTER.index(t[-1]) == l]
            if not tl:
                continue
            res, base = self.fit_l(l, tl, c["share"])
            S_all = sto_S(self.pool[l])
            for m, r in res.items():
                for t, sr in r["shells"].items():
                    e = self.shell_entry(l, t, sr["set"], sr["coef"], sr["fidelity"])
                    e["n_extra"] = m
                    e["base_from_koga"] = bool(self.base[l])
                    rep["fits"].setdefault(t, {})[str(m)] = e
            # reference numbers: Koga contraction (occupied shells only) on the projected bands
            for t in tl:
                if t in self.koga:
                    kp = [(p["n"], p["zeta"]) for p in self.koga[t]["primitives"]]
                    ji = [self.idx[l][(n, round(z, 6))] for n, z in kp]
                    kc = np.array([p["coef"] for p in self.koga[t]["primitives"]])
                    Mn = self._Mn(l, t)[np.ix_(ji, ji)]
                    rep["fits"][t]["koga_fidelity"] = fixed_fid(Mn, S_all[np.ix_(ji, ji)], kc)
        return rep
