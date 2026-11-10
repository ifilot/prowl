#!/usr/bin/env python3
"""basisfit.py -- fit STO valence and polarization shells to VASP PBE free atoms (docs/BASIS_FIT_PLAN.md).

Subcommands (all paths relative to the work directory, --work or $PBEFIT_WORK, default "."):
  atoms   POTCAR...   free-atom VASP runs -> atoms/<POTCAR>_<conf>_box<box>/ (setting A, then B if needed)
  fit     POTCAR...   fit those runs -> fits/<POTCAR>_<conf>_box<box>_<mode>.json (modes 2026 and 2015)
  queue   JOBS        run a jobs file ("POTCAR conf box" per line) on fixed core slots, fit each atom when it is
                      done and append 24 A reruns by the box rule; the jobs file may be extended while it runs
  boxrule JOBS        append the 24 A reruns that the box rule asks for to a jobs file
  library NAME        build libs/NAME.json (prowl-sto-basis/1) with the frozen method; --sto DIR writes .sto files
  report              summary of every fit (checks, SCF setting, <r> outliers vs Koga) + <r>-vs-Z plot
  aecheck POTCAR...   all-electron scalar-relativistic PBE atom with the VASP occupations vs the fitted shells

Environment: PROWL_POTCAR_DIR (potpaw_PBE), PBEFIT_VASP_CMD (Gamma-only VASP; may contain {threads}, {ranks}),
the `prowl` Python package (Koga basis, PAW projections).  No LOBSTER or pbeVaspFit2015 data are read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _run_tag(pot, conf, box):
    return f"{pot}_{conf}_box{float(box):g}"


# ---------------------------------------------------------------------------------------------------- atoms / fit
def cmd_atoms(a):
    from pbefit import atom

    def one(pot):
        st = atom.run_atom(pot, a.conf, a.box, a.threads, ranks=a.ranks)
        print(f"[atom] {pot:8s} converged={st['converged']} setting={st['setting']} {st['seconds']:.0f}s", flush=True)
        Path("reports").mkdir(exist_ok=True)
        with open("reports/atom_status.jsonl", "a") as fh:
            fh.write(json.dumps({**st, "conf": a.conf, "box": a.box, "time": time.ctime()}) + "\n")
        if a.fit and st["converged"]:
            fit_potcars([pot], a.conf, a.box, a.modes, [])

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, a.potcars))


def fit_potcars(potcars, conf, box, modes, settings, jobs=1):
    """Each fit in its own process (single-threaded BLAS), `jobs` at a time."""
    from pbefit import atom
    Path("fits").mkdir(exist_ok=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

    def one(job):
        pot, mode = job
        tag = _run_tag(pot, conf, box)
        cmd = [sys.executable, str(Path(__file__).resolve()), "fit-one", str(atom.WORK / tag), pot,
               f"fits/{tag}_{mode}.json", f"mode={mode}", *settings]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print("[fit] " + "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "RuntimeWarning" not in l),
              flush=True)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, [(p, m) for p in potcars for m in modes.split(",")]))


def cmd_fit(a):
    fit_potcars(a.potcars, a.conf, a.box, a.modes, a.set, a.jobs)


def cmd_fit_one(a):
    from pbefit.fit import Fitter
    cfg = {}
    for kv in a.settings:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v) if v[0] in "0123456789-[{tf" else v
    t0 = time.time()
    rep = Fitter(Path(a.run), a.potcar, cfg).run_fit()
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"{a.potcar} ({rep['config']['mode']}, share={rep['config']['share']}) {time.time() - t0:.0f}s  "
          f"PAW norm dev {rep['paw_norm_max_dev']:.1e}  zmin {rep['zmin']}")
    for t, f in rep["fits"].items():
        kf = f.get("koga_fidelity")
        ms = sorted((k for k in f if k.isdigit()), key=int)
        e = f[ms[-1]]
        print(f"  {t}: spill " + "  ".join(f"m{m}: {1 - f[m]['fidelity']:.1e}" for m in ms) +
              (f"  koga {1 - kf:.1e}" if kf is not None else "") +
              f"  <r> {e['r_mean_bohr']:.3f}  nodes {e['nodes']}  eps {[round(x, 2) for x in e['eps_ev']][:3]}  "
              f"m{ms[-1]} {'PASS' if e['passed'] else 'FAIL ' + str([k for k, v in e['checks'].items() if not v])}")


# ---------------------------------------------------------------------------------------------------- queue
def read_jobs(jobfile):
    for line in Path(jobfile).read_text().splitlines():
        f = line.split("#")[0].split()
        if f:
            yield f[0], f[1], float(f[2])


def boxrule_append(jobfile) -> list:
    """Append '<POTCAR> nonspin 24' for every 20 A 2026-mode fit that the box rule flags (once per POTCAR)."""
    from pbefit.library import needs_box24
    jobfile = Path(jobfile)
    have = {(p, c, b) for jf in jobfile.parent.glob("jobs_*.txt") if jf.exists() for p, c, b in read_jobs(jf)}
    have |= set(read_jobs(jobfile))
    new = []
    for f in sorted(Path("fits").glob("*_nonspin_box20_2026.json")):
        r = json.loads(f.read_text())
        if needs_box24(r) and (r["potcar"], "nonspin", 24.0) not in have:
            e = max(fs["0"]["eps_ev"][0] for fs in r["fits"].values())
            new.append(f"{r['potcar']} nonspin 24   # box rule: eps {e:.2f} eV")
    if new:
        with open(jobfile, "a") as fh:
            fh.write("# box-rule reruns\n" + "\n".join(new) + "\n")
    return [n.split()[0] for n in new]


def cmd_boxrule(a):
    print("appended:", boxrule_append(a.jobs))


def cmd_queue(a):
    """Slots "first-last:RANKSxTHREADS" (physical cores); each slot runs one job at a time. A job is skipped when
    its run directory has a status file or a lock file (atoms/.lock_<run>), so several queues can share a file."""
    from pbefit import atom
    lock = threading.Lock()
    atom.WORK.mkdir(parents=True, exist_ok=True)

    def next_job():
        with lock:
            for pot, conf, box in read_jobs(a.jobs):
                tag = _run_tag(pot, conf, box)
                lk = atom.WORK / f".lock_{tag}"
                if (atom.WORK / tag / "pbefit_status.txt").exists() or lk.exists():
                    continue
                lk.write_text(str(os.getpid()))
                return pot, conf, box
        return None

    def worker(slot):
        cores, rt = slot.split(":")
        ranks, threads = rt.split("x")
        env = {**os.environ, "VASP_CORESET": cores}
        while (job := next_job()) is not None:
            pot, conf, box = job
            cmd = [sys.executable, str(Path(__file__).resolve()), "atoms", pot, "--conf", conf, "--box", str(box),
                   "--threads", threads, "--ranks", ranks] + (["--fit"] if a.fit else [])
            t0 = time.time()
            out = subprocess.run(cmd, env=env, capture_output=True, text=True).stdout
            with lock, open(a.log, "a") as fh:
                fh.write(f"# slot {slot} {pot} {conf} box {box:g} {time.time() - t0:.0f}s\n" +
                         "\n".join(l for l in out.splitlines() if "RuntimeWarning" not in l) + "\n")
            if a.fit:
                with lock:
                    new = boxrule_append(a.jobs)
                    if new:
                        with open(a.log, "a") as fh:
                            fh.write(f"# box rule: appended {new}\n")

    ths = [threading.Thread(target=worker, args=(s,)) for s in a.slots.split(",")]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    with open(a.log, "a") as fh:
        fh.write(f"# QUEUE DONE {time.strftime('%Y-%m-%d %H:%M')}\n")


# ---------------------------------------------------------------------------------------------------- library
def cmd_library(a):
    from pbefit.library import auto_m, box_rule_files, build_library, rule_2026, scale_extras, write_sto
    if a.conf == "boxrule":
        files = box_rule_files(Path("fits"), a.mode)
    else:
        files = {f.name[: -len(f"_{a.conf}_{a.mode}.json")]: f for f in sorted(Path("fits").glob(f"*_{a.conf}_{a.mode}.json"))}
    want = a.potcars.split(",") if a.potcars else None
    reps, by_el = {}, {}
    for pot, f in sorted(files.items()):
        if want and pot not in want:
            continue
        rep = json.loads(Path(f).read_text())
        rep["_file"] = str(f)
        if a.rule == "2026" and a.mode == "2026":
            rep = rule_2026(rep)
        if a.all:
            by_el.setdefault(rep["element"], {})[pot] = rep
            continue
        if rep["element"] in reps:
            raise SystemExit(f"two POTCARs for {rep['element']} ({reps[rep['element']]['potcar']}, {pot}): "
                             "choose with --potcars, or merge them with --all")
        reps[rep["element"]] = rep
    if a.all:
        from pbefit.library import merge_variants
        for el, rr in sorted(by_el.items()):
            try:
                reps[el] = merge_variants(rr)
            except ValueError as e:
                print("skipped:", e)
    m = a.m if a.m == "auto" else int(a.m)
    doc = build_library(reps, a.name, m=m, note=f"mode {a.mode}, atoms {a.conf}, rule {a.rule}, extras {a.m}")
    if a.scale:
        only = None
        if a.scale_shells:
            only = {}
            for x in a.scale_shells.split(","):
                el, t = x.split(":")
                only.setdefault(el, []).append(t)
        doc = scale_extras(doc, a.scale, only)
    Path("libs").mkdir(exist_ok=True)
    Path(f"libs/{a.name}.json").write_text(json.dumps(doc, indent=1))
    if a.sto:
        Path(a.sto).mkdir(parents=True, exist_ok=True)
        for el in reps:
            write_sto(doc["elements"][el], el, Path(a.sto) / f"{el}.sto")
    for el, rep in reps.items():
        ch = {t: (auto_m(f) if m == "auto" else str(m)) for t, f in rep["fits"].items()}
        print(f"{el:3s} {rep['potcar']:8s} {Path(rep['_file']).name:40s} " +
              " ".join(f"{t}:m{v}({rep['fits'][t][v]['orbital_spilling']:.1e})" for t, v in ch.items()))
    print("wrote", f"libs/{a.name}.json")


# ---------------------------------------------------------------------------------------------------- report
def cmd_report(a):
    from pbefit.library import auto_m, box_rule_files, fit_checks, fit_passed, r_mean
    from pbefit.shells import koga
    if a.conf == "boxrule":
        files = box_rule_files(Path("fits"), "2026")
    else:
        files = {f.name[: -len(f"_{a.conf}_2026.json")]: f for f in sorted(Path("fits").glob(f"*_{a.conf}_2026.json"))}
    rows, fails, outl = [], [], []
    for pot, f in files.items():
        rep = json.loads(Path(f).read_text())
        el = rep["element"]
        Z = koga()[el]["Z"]
        run = Path(f).name[: -len("_2026.json")]
        st = Path("atoms") / run / "pbefit_status.txt"
        setting = st.read_text().split("setting=")[1].split()[0] if st.exists() else "?"
        kg = {s["label"]: s for s in koga()[el]["shells"]}
        occ = {k for k, s in kg.items() if s["occupation"] > 0}
        for t, fs in rep["fits"].items():
            m = auto_m(fs)
            e = fs[m]
            r = {"potcar": pot, "Z": Z, "shell": t, "l": e["l"], "m": int(m), "spill": e["orbital_spilling"],
                 "r": e["r_mean_bohr"], "passed": fit_passed(e), "added": t not in occ, "setting": setting,
                 "box": run.split("_box")[-1]}
            if t in kg:
                r["r_koga"] = r_mean(kg[t])
                ratio = r["r"] / r["r_koga"]
                if not 0.8 < ratio < 1.3:
                    outl.append(f"{pot} {t}: <r> {r['r']:.3f} vs Koga {r['r_koga']:.3f} (x{ratio:.2f})")
            if not fit_passed(e):
                fails.append(f"{pot} {t} (m{m}): " + ", ".join(k for k, v in fit_checks(e).items() if not v))
            rows.append(r)
    pots = sorted({r["potcar"] for r in rows}, key=lambda p: (min(x["Z"] for x in rows if x["potcar"] == p), p))
    first = {p: next(r for r in rows if r["potcar"] == p) for p in pots}
    npass = sum(r["passed"] for r in rows)
    L = [f"# basisfit summary ({a.conf}, 2026 mode, auto extras)", "",
         f"* POTCARs fitted: {len(pots)}; shells: {len(rows)}; passing all checks: {npass} "
         f"({100 * npass / max(len(rows), 1):.1f} %)",
         f"* SCF setting A: {sum(first[p]['setting'] == 'A' for p in pots)}, "
         f"B: {sum(first[p]['setting'] == 'B' for p in pots)}",
         f"* 24 A box: {', '.join(p for p in pots if first[p]['box'] == '24') or 'none'}",
         "", "## Failing checks", *(["* " + x for x in fails] or ["* none"]),
         "", "## <r> outliers vs Koga (ratio outside 0.8-1.3)", *(["* " + x for x in outl] or ["* none"]),
         "", "## Per POTCAR", "| POTCAR | Z | setting | box | shells (m, spilling, <r> bohr) |", "|---|---|---|---|---|"]
    for p in pots:
        rr = [r for r in rows if r["potcar"] == p]
        L.append(f"| {p} | {rr[0]['Z']} | {rr[0]['setting']} | {rr[0]['box']} | " +
                 ", ".join(f"{r['shell']}{'*' if r['added'] else ''} m{r['m']} {r['spill']:.0e} {r['r']:.2f}"
                           + ("" if r["passed"] else " FAIL") for r in rr) + " |")
    L.append("\n`*` = added (polarization) shell.")
    Path("reports").mkdir(exist_ok=True)
    out = Path(f"reports/summary_{a.conf}.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:12 + len(fails) + len(outl)]))
    if not a.no_plot:
        plot_rmean(rows, pots, Path(f"reports/rmean_vs_Z_{a.conf}.png"), a.conf)
    print("wrote", out)


def plot_rmean(rows, pots, path, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for l, ax in zip(range(4), axes.flat):
        occ_r = [r for r in rows if r["l"] == l and not r["added"]]
        add_r = [r for r in rows if r["l"] == l and r["added"]]
        kg_r = [r for r in occ_r if "r_koga" in r]
        style = dict(zorder=3, edgecolors="white", linewidths=0.8)
        if kg_r:
            ax.scatter([r["Z"] for r in kg_r], [r["r_koga"] for r in kg_r], s=22, color=ORANGE,
                       label="Koga HF (occupied)", **{**style, "zorder": 2})
        if occ_r:
            ax.scatter([r["Z"] for r in occ_r], [r["r"] for r in occ_r], s=22, color=BLUE, label="fit, occupied", **style)
        if add_r:
            ax.scatter([r["Z"] for r in add_r], [r["r"] for r in add_r], s=26, marker="D", color=AQUA,
                       label="fit, added shell", **style)
        bad = [r for r in occ_r + add_r if not r["passed"]]
        if bad:
            ax.scatter([r["Z"] for r in bad], [r["r"] for r in bad], s=90, facecolors="none", edgecolors="#222222",
                       linewidths=1.2, label="fails a check", zorder=4)
        ax.set_title(f"l = {l} ({'spdf'[l]} shells)", fontsize=11, loc="left")
        ax.set_ylabel("<r> (bohr)", fontsize=9)
        ax.grid(color="#e5e5e5", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors="#555555")
        if occ_r or add_r:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Z", fontsize=9)
    fig.suptitle(f"Fitted shells: <r> vs Z ({conf}, {len(pots)} POTCARs)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------------------------------------------- aecheck
def cmd_aecheck(a):
    import math
    import numpy as np
    from pbefit.aeatom import Grid, solve_atom
    from pbefit.fit import _nrm
    from pbefit.library import auto_m
    from pbefit.shells import koga

    def on_grid(prims, r):
        R = np.zeros_like(r)
        for p in prims:
            R += p["coef"] * _nrm(p["n"], p["zeta"]) * r ** (p["n"] - 1) * np.exp(-p["zeta"] * r)
        return R

    print("| POTCAR | shell | occ | eps AE / VASP (eV) | overlap AE-fit | <r> AE / fit (bohr) | dev | overlap AE-Koga |")
    print("|---|---|---|---|---|---|---|---|")
    res = {}
    for pot in a.potcars:
        rep = json.loads(Path(f"fits/{pot}_{a.conf}_2026.json").read_text())
        kg = koga()[rep["element"]]
        cfg = {s["label"]: float(s["occupation"]) for s in kg["shells"]}
        for t, f in rep["fits"].items():       # fitted shells: the VASP occupations
            cfg[t] = (2.0 if rep["nspin"] == 1 else 1.0) * sum(f[next(k for k in f if k.isdigit())]["occ"])
        for lab in list(cfg):                  # every shell from n = l+1 must be present
            n, l = int(lab[:-1]), "spdf".index(lab[-1])
            for k in range(l + 1, n):
                cfg.setdefault(f"{k}{lab[-1]}", 0.0)
        grid = Grid()
        at = solve_atom(kg["Z"], cfg, "pbe", rel=not a.nonrel, grid=grid, maxiter=400)
        r = grid.r
        rows = []
        for t, f in rep["fits"].items():
            e = f[auto_m(f)]
            Rae = at.orbitals[t]
            Rf = on_grid(e["primitives"], r)
            Rf /= math.sqrt(grid.integrate(Rf * Rf * r * r))
            r_ae = grid.integrate(Rae * Rae * r ** 3)
            row = {"shell": t, "occ": cfg[t], "eps_ae_ev": at.energies[t] * 27.211386, "eps_vasp_ev": e["eps_ev"][0],
                   "overlap": abs(grid.integrate(Rae * Rf * r * r)), "r_ae": r_ae, "r_fit": e["r_mean_bohr"],
                   "r_dev_pct": 100 * (e["r_mean_bohr"] / r_ae - 1)}
            ks = next((s for s in kg["shells"] if s["label"] == t), None)
            if ks:
                Rk = on_grid(ks["primitives"], r)
                Rk /= math.sqrt(grid.integrate(Rk * Rk * r * r))
                row["overlap_koga"] = abs(grid.integrate(Rae * Rk * r * r))
            rows.append(row)
            print(f"| {pot} | {t} | {row['occ']:.2f} | {row['eps_ae_ev']:.2f} / {row['eps_vasp_ev']:.2f} | "
                  f"{row['overlap']:.5f} | {r_ae:.3f} / {row['r_fit']:.3f} | {row['r_dev_pct']:+.2f} % | "
                  f"{row.get('overlap_koga', float('nan')):.5f} |", flush=True)
        res[pot] = {"converged": at.converged, "rel": not a.nonrel, "rows": rows}
    Path("reports").mkdir(exist_ok=True)
    out = Path(f"reports/aecheck{'_nonrel' if a.nonrel else ''}.json")
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **res}, indent=1))


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default=os.environ.get("PBEFIT_WORK", "."), help="work directory (default: $PBEFIT_WORK or .)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def run_opts(p):
        p.add_argument("--conf", default="nonspin", choices=["nonspin", "spin", "s1"])
        p.add_argument("--box", type=float, default=20.0)

    p = sub.add_parser("atoms"); p.add_argument("potcars", nargs="+"); run_opts(p)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--jobs", type=int, default=1, help="atoms run at the same time")
    p.add_argument("--fit", action="store_true", help="fit each converged atom right away")
    p.add_argument("--modes", default="2026,2015")
    p.set_defaults(func=cmd_atoms)

    p = sub.add_parser("fit"); p.add_argument("potcars", nargs="+"); run_opts(p)
    p.add_argument("--modes", default="2026,2015"); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--set", nargs="*", default=[], help="fitter settings key=value (see pbefit.fit.DEFAULTS)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("fit-one", help="internal: fit one run directory (used by fit)")
    p.add_argument("run"); p.add_argument("potcar"); p.add_argument("out"); p.add_argument("settings", nargs="*")
    p.set_defaults(func=cmd_fit_one)

    p = sub.add_parser("queue"); p.add_argument("jobs")
    p.add_argument("--slots", required=True, help="e.g. 0-3:2x2,4-7:2x2 (physical cores : ranks x threads)")
    p.add_argument("--fit", action="store_true"); p.add_argument("--log", default="reports/queue.log")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("boxrule"); p.add_argument("jobs"); p.set_defaults(func=cmd_boxrule)

    p = sub.add_parser("library"); p.add_argument("name")
    p.add_argument("--mode", default="2026", choices=["2026", "2015"])
    p.add_argument("--conf", default="boxrule", help="'boxrule' (frozen method) or a run tag such as nonspin_box20")
    p.add_argument("--rule", default="2026", choices=["2026", "all"],
                   help="2026: frozen selection of refitted shells; all: every fitted shell replaces Koga")
    p.add_argument("--potcars", default="", help="comma-separated POTCARs (one per element unless --all)")
    p.add_argument("--all", action="store_true",
                   help="one entry per element from all its fitted variants (library.merge_variants)")
    p.add_argument("--m", default="auto", help="extra diffuse primitives: auto or 0-3")
    p.add_argument("--scale", type=float, default=None, help="scale the diffuse extras (sensitivity check)")
    p.add_argument("--scale-shells", default="", help="restrict --scale, e.g. Fe:4p,Na:3p")
    p.add_argument("--sto", default="", help="also write LOBSTER-style .sto files to this directory")
    p.set_defaults(func=cmd_library)

    p = sub.add_parser("report"); p.add_argument("--conf", default="boxrule")
    p.add_argument("--no-plot", action="store_true"); p.set_defaults(func=cmd_report)

    p = sub.add_parser("aecheck"); p.add_argument("potcars", nargs="+")
    p.add_argument("--conf", default="nonspin_box20"); p.add_argument("--nonrel", action="store_true")
    p.set_defaults(func=cmd_aecheck)

    a = ap.parse_args(argv)
    if a.cmd in ("queue", "boxrule"):          # given relative to the caller's directory
        a.jobs = str(Path(a.jobs).resolve())
    os.environ["PBEFIT_WORK"] = str(Path(a.work).resolve())
    os.chdir(a.work)
    a.func(a)


if __name__ == "__main__":
    main()
