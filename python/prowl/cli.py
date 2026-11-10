"""Command-line interface: ``prowl inspect|validate|run|export``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cfg(path: str | None, profile: str | None = None):
    from . import _core
    from .config import BondingConfig

    if not path:
        return BondingConfig.for_profile(profile or "prowl")
    cfg = BondingConfig.from_file(path)
    if profile and profile != cfg.profile:
        raise _core.ProwlError(f"InvalidArgument: --profile {profile} conflicts with profile '{cfg.profile}' in "
                                f"{path}; set it in the configuration file")
    return cfg


def cmd_inspect(a) -> int:
    from .calculation import Calculation

    rep = Calculation.open(a.run_dir).inspect()
    print(json.dumps(rep.to_dict(), indent=1, default=str) if a.json else rep.summary())
    return 0 if rep.ok else 2


def cmd_plan(a) -> int:
    from .calculation import Calculation

    plan = Calculation.open(a.run_dir).plan(_cfg(a.config, a.profile))
    if a.json:
        print(json.dumps(plan, indent=1, default=str))
        return 0 if plan.get("inspection_ok") and "error" not in plan else 2
    for i in plan.get("issues", []):
        print(f"[{i['level'].upper()}] {i['code']}: {i['message']}")
    if "sizes" in plan:
        print("sizes     :", ", ".join(f"{k}={v}" for k, v in plan["sizes"].items()))
        print("basis     :", plan["basis"])
        rq = plan["required_nbands"]
        print(f"bands     : NBANDS={plan['sizes']['N_band']}, minimum {rq['minimum']} (= N_orb, as LOBSTER), "
              f"recommended >= {rq['recommended']}")
        st = plan["streaming"]
        print(f"streaming : {st['band_block']} bands/block ({st['band_blocks_per_k']} per k), "
              f"{st['factor_columns_per_block']} factor columns/block, budget {st['memory_budget_gb']}")
        print("memory    :", ", ".join(f"{k} {v:.3f}" for k, v in plan["memory_estimate"].items()),
              f"| peak ~{plan['estimated_peak_GiB']:.2f} GiB")
        print(f"GEMM work : {plan['gemm_flops_per_k'] / 1e9:.2f} GFLOP per k-point")
        for r in plan["recommendations"]:
            print("recommend :", r)
    if "error" in plan:
        print("error     :", plan["error"])
    return 0 if plan.get("inspection_ok") and "error" not in plan else 2


def cmd_validate(a) -> int:
    from .calculation import Calculation

    cfg = _cfg(a.config)
    rep = Calculation.open(a.run_dir).inspect()
    print(rep.summary())
    print("config OK:", json.dumps(cfg.to_dict(), default=str))
    return 0 if rep.ok else 2


def cmd_run(a) -> int:
    import numpy as np

    from . import _core
    from .calculation import Calculation

    try:
        cfg = _cfg(a.config, a.profile)
        backend = a.backend or cfg.backend
        prog = None
        if not a.quiet:
            from .calculation import _omp_threads
            from .progress import ProgressReporter

            prog = ProgressReporter(backend=backend, threads=_omp_threads())
            print(f"prowl run {a.run_dir}: backend {backend}, {_omp_threads()} threads", file=sys.stderr)
        res = Calculation.open(a.run_dir).analyze(cfg, backend=backend, reuse=a.reuse, progress=prog)
    except _core.ProwlError as e:
        print(f"prowl: error: {e}", file=sys.stderr)
        return 2
    out = Path(a.output or Path(a.run_dir) / cfg.output_path)
    res.save(out)
    print(res.quality_report())
    r = res.result
    ws = r.diagnostics.get("window_sensitivity") or {}
    alt = np.asarray(ws["icohp"]).sum(axis=0) if ws.get("available") else None
    print("\nICOHP at E_F (eV, COHP sign: negative = bonding):")
    for i, b in enumerate(r.bonds[: a.max_print]):
        print(f"  {b.label(res.input.elements):28s} d = {b.distance:7.4f} A   ICOHP = {r.icohp[:, i].sum():9.5f}   "
              f"ICOOP = {r.icoop[:, i].sum():8.5f}   ICOBI = {r.icobi[:, i].sum():7.4f}" +
              (f"   [{ws['window']} window: {alt[i]:9.5f}]" if alt is not None else ""))
    if len(r.bonds) > a.max_print:
        print(f"  ... {len(r.bonds) - a.max_print} more bonds in {out}")
    if r.bonds and a.max_print > 0:
        import h5py

        from .io import orbital_resolved_icohp

        with h5py.File(out, "r") as f:
            parts = sorted(orbital_resolved_icohp(f, 0), key=lambda e: sum(e[2]))
        print(f"\nShell-pair ICOHP of {r.bonds[0].label(res.input.elements)} (eV):")
        for sa, sb, v in parts[:6]:
            print(f"  {sa:>4s} - {sb:<4s} {sum(v):9.4f}")
        if getattr(r, "icohp_channel", None) is not None:
            ch = r.icohp_channel.sum(axis=0)  # (nbond, 5)
            used = [c for c in range(5) if np.abs(ch[:, c]).max() > 5e-5]
            names = ["sigma", "pi", "delta", "phi", "mixed"]
            print("\nICOHP by bond-frame channel (eV; z' along the bond):")
            print("  " + " " * 28 + "".join(f"{names[c]:>10s}" for c in used))
            for i, bd in enumerate(r.bonds[: a.max_print]):
                print(f"  {bd.label(res.input.elements):28s}" + "".join(f"{ch[i, c]:10.4f}" for c in used))
    print(f"\nwrote {out}")
    return 0


def cmd_export(a) -> int:
    if a.format == "lobster":
        from .lobster_export import write_lobster_format

        for p in write_lobster_format(a.h5, a.output):
            print(p)
        return 0
    from .io import export

    for p in export(a.h5, a.output, a.format):
        print(p)
    return 0


def cmd_plot(a) -> int:
    from .plotting import plot_bonding

    bonds = [int(x) for x in a.bonds.split(",")] if a.bonds else None
    print(plot_bonding(a.h5, a.output, bonds=bonds, kind=a.kind, emin=a.emin, emax=a.emax))
    return 0


def main(argv: list[str] | None = None) -> int:
    from . import __version__

    ap = argparse.ArgumentParser(prog="prowl", description="Open VASP bonding analysis (COHP, COOP, pDOS, charges)")
    ap.add_argument("--version", action="version", version=f"prowl {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="check a VASP run directory and report what can be analysed")
    p.add_argument("run_dir")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("plan", help="dry run: sizes, required NBANDS, memory estimate and streaming plan (no projection)")
    p.add_argument("run_dir")
    p.add_argument("--config")
    p.add_argument("--profile", choices=["prowl", "lobster"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("validate", help="inspect inputs and validate a configuration file")
    p.add_argument("run_dir")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("run", help="run the bonding analysis and write bonding.h5")
    p.add_argument("run_dir")
    p.add_argument("--config")
    p.add_argument("--backend", choices=["cpu", "cuda", "auto"])
    p.add_argument("--output")
    p.add_argument("--profile", choices=["prowl", "lobster"],
                   help="conventions without a config file: prowl (default; LOBSTER's window where well "
                        "conditioned, else frozen) or lobster (always the lowest N_orb bands, as LOBSTER)")
    p.add_argument("--max-print", type=int, default=20)
    p.add_argument("--quiet", action="store_true", help="no progress and time estimates on stderr")
    p.add_argument("--reuse", help="bonding.h5 with stored projections (output.write_intermediate_projection: true)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("export", help="export bonding.h5 to CSV/JSON")
    p.add_argument("h5")
    p.add_argument("--format", choices=["csv", "json", "lobster"], default="csv")
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("plot", help="plot -COHP / COOP / pDOS from bonding.h5 (needs matplotlib)")
    p.add_argument("h5")
    p.add_argument("--kind", choices=["cohp", "coop", "pdos"], default="cohp")
    p.add_argument("--bonds", help="comma-separated 0-based bond (or orbital) indices")
    p.add_argument("--emin", type=float)
    p.add_argument("--emax", type=float)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_plot)
    a = ap.parse_args(argv)
    from . import _core

    try:
        return a.fn(a)
    except _core.ProwlError as e:
        print(f"prowl: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
