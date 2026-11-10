#!/usr/bin/env python3
"""Generate the documentation gallery: real PROWL results for CO, CH4 and diamond.

For each system the script
  1. runs a PROWL analysis with the configuration in docs/examples/gallery/,
  2. (molecules) computes Kohn-Sham orbitals from the WAVECAR, tessellates them
     with PyTessel and renders them with QASR,
  3. draws pDOS / -COHP figures in the website's light and dark colour themes,
  4. records provenance in docs/_static/gallery/manifest.json.

The VASP runs come from the recipes in tests/private_vasp_manifest/recipes
(c_diamond_k7) or equivalent static molecule-in-a-box runs. They are licensed
inputs and are not distributed; pass their location with --runs.

    PYTHONPATH=python python tools/docs_gallery/make_gallery.py \
        --runs /path/to/runs --qasr /path/to/qasr/build/qasr
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from orbitals import GammaOrbitals, check_tessellation_convention, tessellate  # noqa: E402

CONFIGS = ROOT / "docs" / "examples" / "gallery"
OUT = ROOT / "docs" / "_static" / "gallery"

# --------------------------------------------------------------------- themes
# Mirrors the colour tokens in docs/_static/prowl.css.
THEMES = {
    "light": dict(text="#24323d", muted="#53636d", border="#dce7e8", occupied="#eef3f4",
                  bond="#126b78", antibond="#b4472c", sigma="#3d4f9f", pi="#b07a12",
                  el={"C": "#3d4f9f", "O": "#b07a12", "H": "#53636d"}),
    "dark": dict(text="#e4ecef", muted="#b0bfc7", border="#354851", occupied="#1f2e36",
                 bond="#76ced5", antibond="#f0977c", sigma="#a3b1f5", pi="#e8bd62",
                 el={"C": "#a3b1f5", "O": "#e8bd62", "H": "#b0bfc7"}),
}
ORBITAL_COLORS = {"positive": "#3b9aa6", "negative": "#e48a6b"}
ATOM_STYLE = {"C": {"color": "#56636b", "radius_A": 0.2}, "O": {"color": "#c9503b", "radius_A": 0.2},
              "H": {"color": "#f2f5f5", "radius_A": 0.14}}

# ------------------------------------------------------------------- systems
# Band indices are zero-based. Degenerate sets are listed together; the first
# real member (aligned with a Cartesian axis) is drawn.
SYSTEMS = {
    "co": dict(run="co_molecule", title="Carbon monoxide (CO)", ylim=(-24.0, 9.0),
               view=[0.1, 1.0, 0.05], up=[1, 0, 0], iso=0.08, member_axis=0, slot=0.22,
               pdos_legend=dict(loc="center right", bbox_to_anchor=(1.0, 0.4)), summary_y=0.42,
               mos=[("3σ", [0]), ("4σ", [1]), ("1π", [2, 3]), ("5σ", [4], "HOMO"), ("2π*", [5, 6], "LUMO")]),
    "ch4": dict(run="ch4_molecule", title="Methane (CH₄)", ylim=(-10.0, 12.0),
                view=[1.0, 0.55, 0.32], up=[0, 0, 1], iso=0.08, member_axis=2, slot=0.27,
                pdos_legend=dict(loc="center right", bbox_to_anchor=(1.0, 0.25)), summary_y=0.62,
                note="Empty states above E$_\\mathrm{F}$ mix with\nthe vacuum states of the 12 Å box,\nso they are not drawn.",
                mos=[("2a₁", [0]), ("1t₂", [1, 2, 3], "HOMO")]),
    "diamond": dict(run="c_diamond_k7", title="Diamond (C)", ylim=(-23.0, 12.0),
                    pdos_legend=dict(loc="center right", bbox_to_anchor=(1.0, 0.76)), summary_y=0.755),
}


def run_analysis(name: str, run_dir: Path, work: Path) -> Path:
    from prowl import BondingConfig, Calculation

    cfg = BondingConfig.from_file(CONFIGS / f"{name}.yaml")
    res = Calculation.open(run_dir).analyze(cfg, backend="cpu")
    out = work / f"{name}.h5"
    res.save(out)
    print(res.quality_report())
    return out


# ------------------------------------------------------------------ orbitals
def render_orbitals(name: str, spec: dict, run_dir: Path, qasr: Path, work: Path) -> list[dict]:
    from pytessel import PyTessel

    tess = PyTessel()
    check_tessellation_convention(tess)
    orb = GammaOrbitals(str(run_dir / "WAVECAR"), n_grid=128)
    struct = _read_poscar(run_dir / "CONTCAR")
    pos = struct["frac"] @ struct["lattice"]
    orb.centre = pos.mean(axis=0)
    xyz = work / f"{name}.xyz"
    xyz.write_text(f"{len(pos)}\n{name}; Angstrom, centred on the box\n" + "".join(
        f"{e} {p[0] - orb.L[0] / 2:.6f} {p[1] - orb.L[1] / 2:.6f} {p[2] - orb.L[2] / 2:.6f}\n"
        for e, p in zip(struct["elements"], pos)))
    (OUT / "mo").mkdir(parents=True, exist_ok=True)
    records = []
    for entry in spec["mos"]:
        label, bands = entry[0], entry[1]
        note = entry[2] if len(entry) > 2 else ""
        members = orb.real_orbitals(bands)
        axis = spec.get("member_axis")
        field = members[orb.last_axes.index(axis)] if axis in orb.last_axes else members[0]
        x = np.arange(orb.n) * orb.L[0] / orb.n
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt((X - orb.centre[0]) ** 2 + (Y - orb.centre[1]) ** 2 + (Z - orb.centre[2]) ** 2)
        local = float(np.sum(field[r < 2.5] ** 2) / np.sum(field ** 2))
        if local < 0.6:
            raise RuntimeError(f"{name} {label}: only {local:.0%} within 2.5 A; not a molecular orbital")
        edge = max(np.abs(np.take(field, [0, -1], axis=d)).max() for d in range(3))
        if edge >= spec["iso"]:
            raise RuntimeError(f"{name} {label}: isosurface reaches the box boundary")
        layers = []
        for sign, phase in ((1, "positive"), (-1, "negative")):
            mesh = tessellate(tess, orb, field, spec["iso"], sign)
            if mesh is None:
                continue
            ply = work / f"{name}-{bands[0]}-{phase}.ply"
            tess.write_ply(str(ply), *mesh)
            layers.append({"file": str(ply), "units": "angstrom", "color": ORBITAL_COLORS[phase]})
        cfg = {"schema_version": 1,
               "image": {"width": 560, "height": 400, "style": "illustrated", "aa": "4x",
                         "outline_width": 1.6, "background": "#ffffff00"},
               "camera": {"view": spec["view"], "up": spec["up"], "margin": 0.06},
               "overlays": {"cell": False, "scale_bar": False},
               "geometry": {"bond_radius_A": 0.065, "bond_colors": "atoms"},
               "elements": {e: ATOM_STYLE[e] for e in set(struct["elements"])},
               "orbitals": layers}
        cfg_path = work / f"{name}-{bands[0]}.json"
        cfg_path.write_text(json.dumps(cfg))
        png = OUT / "mo" / f"{name}-{bands[0] + 1:02d}.png"
        subprocess.run([str(qasr), str(xyz), "--no-user-config", "--config", str(cfg_path), "-o", str(png)],
                       check=True, stdout=subprocess.DEVNULL)
        e = float(np.mean(orb.eigenvalues[bands]) - orb.efermi)
        records.append(dict(label=label, bands=[b + 1 for b in bands], note=note, energy_ev=e,
                            weight_within_2p5_A=round(local, 3),
                            occupied=bool(orb.occupations[bands[0]] > 0.5), image=str(png.relative_to(OUT))))
    return records


def render_structure(run_dir: Path, qasr: Path) -> str:
    png = OUT / "diamond-structure.png"
    cfg = {"schema_version": 1,
           "image": {"width": 600, "height": 520, "style": "illustrated", "aa": "4x",
                     "outline_width": 1.4, "background": "#ffffff00"},
           "camera": {"view": [1.0, 0.38, 0.2], "up": [0, 0, 1], "margin": 0.04},
           "overlays": {"cell": False, "scale_bar": False},
           "geometry": {"repeat": [2, 2, 2], "bond_radius_A": 0.13, "bond_colors": "atoms"},
           "elements": {"C": {"color": "#6f8f96", "radius_A": 0.36}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
    subprocess.run([str(qasr), str(run_dir / "CONTCAR"), "--no-user-config", "--config", fh.name, "-o", str(png)],
                   check=True, stdout=subprocess.DEVNULL)
    os.unlink(fh.name)
    return str(png.relative_to(OUT))


def _read_poscar(path: Path) -> dict:
    lines = path.read_text().split("\n")
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    species, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    n = sum(counts)
    start = 8 if lines[7].strip()[0] in "DdCc" else 9
    frac = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in range(n)])
    if lines[start - 1].strip()[0] in "Cc":
        frac = frac @ np.linalg.inv(lattice)
    return dict(lattice=lattice, frac=frac, elements=[s for s, c in zip(species, counts) for _ in range(c)])


# --------------------------------------------------------------------- plots
def _style(t: dict):
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 12, "svg.fonttype": "path",
        "axes.edgecolor": t["muted"], "axes.labelcolor": t["text"], "axes.linewidth": 0.9,
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["text"],
        "ytick.labelcolor": t["text"], "text.color": t["text"], "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none",
    })


def load(h5: Path) -> dict:
    with h5py.File(h5, "r") as f:
        d = dict(E=f["analyses/cohp/energies"][()], cohp=f["analyses/cohp/cohp"][()].sum(axis=0),
                 chan=f["analyses/cohp/cohp_channel"][()].sum(axis=0),
                 chan_names=[x.decode() for x in f["analyses/cohp/channel_names"][()]],
                 icohp=f["analyses/cohp/icohp"][()].sum(axis=0), icobi=f["analyses/cobi/icobi"][()].sum(axis=0),
                 icoop=f["analyses/coop/icoop"][()].sum(axis=0),
                 icohp_chan=f["analyses/cohp/icohp_channel"][()].sum(axis=0),
                 labels=[x.decode() for x in f["analyses/cohp/bond_label"][()]],
                 dist=f["analyses/cohp/bond_distance"][()],
                 pdos=f["analyses/pdos/pdos"][()].sum(axis=0),
                 orb=[x.decode() for x in f["basis/orbital_labels"][()]],
                 diag=json.loads(f["reconstruction/diagnostics_json"][()]),
                 prov=json.loads(f["meta/provenance_json"][()]),
                 lowdin=f["analyses/charges/lowdin_charge"][()],
                 elements=[x.decode() for x in f["structure/elements"][()]])
    return d


def _pdos_panel(ax, d, t, ylim, legend_loc):
    E = d["E"]
    groups: dict[str, np.ndarray] = {}
    for i, lab in enumerate(d["orb"]):
        el = "".join(c for c in lab.split("_")[0] if c.isalpha())
        groups[el] = groups.get(el, 0) + d["pdos"][i]
    if "C" in groups and len(groups) == 1:  # diamond: split by shell instead of element
        groups = {"C 2s": sum(d["pdos"][i] for i, l in enumerate(d["orb"]) if l.endswith("2s")),
                  "C 2p": sum(d["pdos"][i] for i, l in enumerate(d["orb"]) if "_2p" in l)}
        colors = {"C 2s": t["pi"], "C 2p": t["sigma"]}
    else:
        colors = {k: t["el"][k] for k in groups}
    for k, v in groups.items():
        ax.fill_betweenx(E, 0, v, color=colors[k], alpha=0.18, lw=0)
        ax.plot(v, E, color=colors[k], lw=1.3, label=k)
    ax.set_xlim(0, None)
    ax.xaxis.set_major_locator(MaxNLocator(4, prune="upper"))  # keep clear of the next panel's ticks
    ax.set_ylim(*ylim)
    ax.set_xlabel("pDOS (states/eV)")
    ax.set_ylabel("E − E$_\\mathrm{F}$ (eV)")
    ax.legend(fontsize=11, handlelength=1.4, **legend_loc)
    ax.set_title("Projected DOS", loc="left", fontsize=13, fontweight="semibold", color=t["text"])


def _cohp_panel(ax, d, t, ylim, bond=0, summary_y=0.42):
    E, y = d["E"], -d["cohp"][bond]
    ax.fill_betweenx(E, 0, y, where=y > 0, color=t["bond"], alpha=0.38, lw=0, interpolate=True)
    ax.fill_betweenx(E, 0, y, where=y < 0, color=t["antibond"], alpha=0.38, lw=0, interpolate=True)
    ax.plot(y, E, color=t["text"], lw=1.1)
    m = 1.2 * np.abs(y[(E >= ylim[0]) & (E <= ylim[1])]).max()
    ax.set_xlim(-m, m)
    ax.set_ylim(*ylim)
    ax.axvline(0, color=t["muted"], lw=0.8)
    ax.set_xlabel("−COHP (1/eV)")
    ax.tick_params(labelleft=False)
    ax.text(0.97, 0.03, "bonding →", transform=ax.transAxes, ha="right", va="bottom", color=t["bond"],
            fontsize=11, fontweight="semibold")
    ax.text(0.03, 0.03, "← antibonding", transform=ax.transAxes, ha="left", va="bottom", color=t["antibond"],
            fontsize=11, fontweight="semibold")
    lab = "".join(c for c in d["labels"][bond].split("[")[0] if not c.isdigit()).replace("-", "–")
    ax.set_title(f"−COHP of {lab}", loc="left", fontsize=13, fontweight="semibold", color=t["text"])
    summary = (f"ICOHP = {d['icohp'][bond]:.2f} eV\nICOBI = {d['icobi'][bond]:.2f}\n"
               f"d = {d['dist'][bond]:.3f} Å")
    ax.text(0.97, summary_y, summary, transform=ax.transAxes, ha="right", va="center", fontsize=11,
            color=t["text"], linespacing=1.5)


def _fermi(axes, t, ylim):
    for ax in axes:
        ax.axhspan(ylim[0], 0, color=t["occupied"], zorder=-5, lw=0)
        ax.axhline(0, color=t["muted"], lw=0.9, ls=(0, (5, 4)), zorder=-4)
    axes[0].text(0.97, 0.0, "E$_\\mathrm{F}$", transform=axes[0].get_yaxis_transform(), va="bottom",
                 ha="right", color=t["muted"], fontsize=11)


def figure_molecule(name: str, d: dict, mos: list[dict], theme: str) -> Path:
    t = THEMES[theme]
    _style(t)
    spec = SYSTEMS[name]
    ylim = spec["ylim"]
    fig = plt.figure(figsize=(8.8, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.6], wspace=0.08, left=0.085, right=0.995,
                          bottom=0.09, top=0.93)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2])
    _pdos_panel(ax1, d, t, ylim, spec["pdos_legend"])
    _cohp_panel(ax2, d, t, ylim, summary_y=spec["summary_y"])
    _fermi([ax1, ax2], t, ylim)
    ax3.set_axis_off()
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("Kohn–Sham orbitals", loc="left", fontsize=13, fontweight="semibold", color=t["text"],
                  x=0.06)
    order = sorted(mos, key=lambda m: -m["energy_ev"])
    n = len(order)
    slots = np.linspace(1 - 0.5 / n, 0.5 / n, n) if n > 2 else np.array([0.72, 0.28])
    for m, ys in zip(order, slots):
        img = plt.imread(OUT / m["image"])
        # Displayed height in points: the slot height, capped for systems with few orbitals.
        h_pt = spec["slot"] * fig.get_figheight() * 0.84 * 72
        ab = AnnotationBbox(OffsetImage(img, zoom=h_pt / img.shape[0]), (0.71, ys), xycoords=ax3.transAxes,
                            frameon=False, box_alignment=(0.5, 0.5))
        ax3.add_artist(ab)
        deg = len(m["bands"])
        name_txt = m["label"] + (f"  (×{deg})" if deg > 1 else "")
        ax3.text(0.07, ys + 0.022, name_txt, transform=ax3.transAxes, fontsize=12.5, fontweight="semibold",
                 va="bottom", color=t["text"])
        ax3.text(0.07, ys - 0.006, f"{m['energy_ev']:+.2f} eV", transform=ax3.transAxes, fontsize=11, va="top",
                 color=t["muted"])
        if m["note"]:
            ax3.text(0.07, ys - 0.052, m["note"], transform=ax3.transAxes, fontsize=11, va="top", color=t["muted"])
        con = ConnectionPatch(xyA=(ax2.get_xlim()[1], m["energy_ev"]), coordsA=ax2.transData,
                              xyB=(0.05, ys), coordsB=ax3.transAxes, color=t["muted"], lw=0.8,
                              ls=(0, (2, 2)), zorder=-1)
        fig.add_artist(con)
        ax2.plot([ax2.get_xlim()[1] * 0.93, ax2.get_xlim()[1]], [m["energy_ev"]] * 2, color=t["muted"], lw=1.4,
                 solid_capstyle="butt")
    if spec.get("note"):
        ax3.text(0.07, 0.97, spec["note"], transform=ax3.transAxes, va="top", fontsize=11, color=t["muted"],
                 linespacing=1.45)
    path = OUT / f"{name}-{theme}.svg"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_diamond(d: dict, structure: str, theme: str) -> Path:
    t = THEMES[theme]
    _style(t)
    ylim = SYSTEMS["diamond"]["ylim"]
    fig = plt.figure(figsize=(8.8, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.1], wspace=0.08, left=0.085, right=0.995,
                          bottom=0.1, top=0.92)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2])
    spec = SYSTEMS["diamond"]
    _pdos_panel(ax1, d, t, ylim, spec["pdos_legend"])
    _cohp_panel(ax2, d, t, ylim, summary_y=spec["summary_y"])
    _fermi([ax1, ax2], t, ylim)
    ax3.set_axis_off()
    img = plt.imread(OUT / structure)
    ax3.imshow(img)
    ax3.set_title("Crystal structure", loc="left", fontsize=13, fontweight="semibold",
                  color=t["text"])
    path = OUT / f"diamond-{theme}.svg"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_bonds(data: dict, theme: str) -> Path:
    t = THEMES[theme]
    _style(t)
    rows = [("C–O in CO", data["co"], 0), ("C–C in diamond", data["diamond"], 0), ("C–H in CH₄", data["ch4"], 0)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.1), gridspec_kw=dict(wspace=0.55, left=0.2, right=0.97,
                                                                             bottom=0.2, top=0.84))
    y = np.arange(len(rows))[::-1]
    labels = [r[0] for r in rows]
    icohp = np.array([r[1]["icohp"][r[2]] for r in rows])
    icobi = np.array([r[1]["icobi"][r[2]] for r in rows])
    sig = np.array([r[1]["icohp_chan"][r[2], 0] for r in rows])
    pi = np.array([r[1]["icohp_chan"][r[2], 1] for r in rows])
    a1.barh(y, -sig, color=t["sigma"], height=0.56, label="σ")
    a1.barh(y, -pi, left=-sig, color=t["pi"], height=0.56, label="π")
    for yi, v in zip(y, icohp):
        a1.text(-v + 0.3, yi, f"{v:.2f} eV", va="center", fontsize=11, color=t["text"])
    a1.set_yticks(y, labels)
    a1.set_xlabel("−ICOHP (eV)")
    a1.set_xlim(0, 1.28 * (-icohp).max())
    for x0, w, txt in ((0, -sig[0], "σ"), (-sig[0], -pi[0], "π")):  # label the segments of the CO bar
        a1.text(x0 + w / 2, y[0], txt, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    a1.set_title("−ICOHP, split into σ and π", loc="left", fontsize=13, fontweight="semibold")
    a2.barh(y, icobi, color=t["bond"], height=0.56)
    for yi, v in zip(y, icobi):
        a2.text(v + 0.04, yi, f"{v:.2f}", va="center", fontsize=11, color=t["text"])
    for x in (1, 2, 3):
        a2.axvline(x, color=t["border"], lw=0.9, zorder=-3)
    a2.set_yticks(y, labels)
    a2.set_xlim(0, 3.2)
    a2.set_xticks([0, 1, 2, 3])
    a2.set_xlabel("ICOBI")
    a2.set_title("Bond index ICOBI", loc="left", fontsize=13, fontweight="semibold")
    for a in (a1, a2):
        a.tick_params(axis="y", length=0)
    path = OUT / f"bonds-{theme}.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ manifest
def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path(os.environ.get("PROWL_TEST_RUNS", "runs")))
    p.add_argument("--qasr", type=Path, default=Path("qasr"))
    p.add_argument("--work", type=Path, default=None, help="scratch directory (default: temporary)")
    a = p.parse_args()
    import prowl
    import pytessel

    OUT.mkdir(parents=True, exist_ok=True)
    work = a.work or Path(tempfile.mkdtemp(prefix="prowl-gallery-"))
    work.mkdir(parents=True, exist_ok=True)
    data, mos, manifest = {}, {}, {"systems": {}}
    for name, spec in SYSTEMS.items():
        run = a.runs / spec["run"]
        print(f"=== {name}: {run}")
        data[name] = load(run_analysis(name, run, work))
        if "mos" in spec:
            mos[name] = render_orbitals(name, spec, run, a.qasr, work)
    structure = render_structure(a.runs / SYSTEMS["diamond"]["run"], a.qasr)
    figures = []
    for theme in THEMES:
        for name in ("co", "ch4"):
            figures.append(figure_molecule(name, data[name], mos[name], theme))
        figures.append(figure_diamond(data["diamond"], structure, theme))
        figures.append(figure_bonds(data, theme))
    for name, spec in SYSTEMS.items():
        d = data[name]
        run = a.runs / spec["run"]
        manifest["systems"][name] = {
            "vasp_run": spec["run"],
            "incar": (run / "INCAR").read_text(),
            "kpoints": (run / "KPOINTS").read_text(),
            "potcar": d["prov"]["potcar_titel"],
            "config": f"docs/examples/gallery/{name}.yaml",
            "input_sha256": {k: v["sha256"] for k, v in d["prov"]["inputs"].items()},
            "diagnostics": {k: d["diag"].get(k) for k in (
                "band_window", "band_window_note", "abs_charge_spilling", "abs_total_spilling",
                "max_paw_norm_deviation", "min_overlap_eigenvalue", "electrons_dft")},
            "bonds": [{"label": l, "distance_A": float(r), "icohp_eV": float(h), "icoop": float(o),
                       "icobi": float(b), "icohp_sigma_eV": float(c[0]), "icohp_pi_eV": float(c[1])}
                      for l, r, h, o, b, c in zip(d["labels"], d["dist"], d["icohp"], d["icoop"], d["icobi"],
                                                  d["icohp_chan"])],
            "lowdin_charges": dict(zip([f"{e}{i + 1}" for i, e in enumerate(d["elements"])],
                                       map(float, d["lowdin"]))),
        }
        if name in mos:
            manifest["systems"][name]["orbitals"] = {
                "method": "VASP pseudo-wavefunction at Gamma, FFT on a 128^3 grid; real gauge; degenerate sets "
                          "rotated onto Cartesian axes; PyTessel marching cubes; QASR illustrated style",
                "isovalue_A^-3/2": spec["iso"], "levels": mos[name]}
    manifest["generated"] = datetime.date.today().isoformat()
    manifest["software"] = {"prowl": data["co"]["prov"]["prowl_version"],
                            "prowl_commit": _git("rev-parse", "--short", "HEAD"),
                            "qasr": subprocess.run([str(a.qasr), "--version"], capture_output=True,
                                                   text=True).stdout.strip(),
                            "pytessel": getattr(pytessel, "__version__", "unknown"),
                            "matplotlib": matplotlib.__version__}
    manifest["figures"] = [str(f.relative_to(OUT)) for f in figures]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
