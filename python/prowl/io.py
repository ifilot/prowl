"""Versioned HDF5 output (``bonding.h5``) and CSV/JSON export.

Layout (schema ``prowl-bonding/1``)::

    /meta              version, provenance JSON, conventions, units
    /structure         lattice, frac, elements, zval
    /kpoints           k (fractional), weights, mesh, spin factor
    /basis             orbital labels, atom offsets
    /reconstruction    diagnostics JSON, band window
    /analyses/pdos     energies, pdos (spin, orbital, E), occupied integrals
    /analyses/cohp     bond table, cohp (spin, bond, E), icohp, orbital-resolved icohp
    /analyses/coop     coop, icoop
    /analyses/charges  Mulliken / Löwdin gross populations and charges
    /projection        optional (output.write_intermediate_projection): A_AE per (spin, k) + input-hash key
    /performance       timings JSON, backend
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "prowl-bonding/1"


def _json(o) -> str:
    def default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        return str(x)
    return json.dumps(o, default=default, indent=1)


def write_hdf5(br, path: str | Path) -> Path:
    import h5py

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    try:
        _write(br, tmp, h5py)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave partial output behind
        raise
    os.replace(tmp, path)  # never leave a half-written file under the final name
    return path


def _write(br, tmp: Path, h5py) -> None:
    r, inp = br.result, br.input
    with h5py.File(tmp, "w") as f:
        f.attrs["schema"] = SCHEMA
        m = f.create_group("meta")
        m.attrs["prowl_version"] = br.provenance["prowl_version"]
        m.attrs["energy_unit"] = "eV"
        m.attrs["length_unit"] = "Angstrom"
        m.attrs["cohp_sign"] = "COHP (negative = bonding)"
        m.attrs["efermi_ev"] = float(inp.efermi)
        m["provenance_json"] = _json(br.provenance)
        m["inspection_json"] = _json(br.inspect.to_dict())
        s = f.create_group("structure")
        s["lattice"] = inp.lattice
        s["frac"] = inp.frac
        s["elements"] = np.array(inp.elements, dtype="S")
        s["zval"] = inp.zval
        k = f.create_group("kpoints")
        k["k_frac"] = inp.kfrac
        k["weights"] = inp.weights
        k.attrs["mesh"] = list(inp.mesh)
        k.attrs["spin_factor"] = inp.spin_factor
        b = f.create_group("basis")
        b["orbital_labels"] = np.array(inp.orbital_labels, dtype="S")
        b["atom_orbital_offsets"] = np.array(inp.orb_offsets)
        rc = f.create_group("reconstruction")
        rc["diagnostics_json"] = _json(r.diagnostics)
        a = f.create_group("analyses")
        pd = a.create_group("pdos")
        pd["energies"] = r.energies
        pd["pdos"] = r.pdos
        pd["occupied_integral"] = r.pdos_integral
        pd.attrs["basis"] = "Löwdin-orthonormalised orbitals"
        ch = a.create_group("cohp")
        ch["energies"] = r.energies
        ch["bond_atoms"] = np.array([[bd.atom_a, bd.atom_b] for bd in r.bonds], dtype=int).reshape(-1, 2)
        ch["bond_translation"] = np.array([bd.R for bd in r.bonds], dtype=int).reshape(-1, 3)
        ch["bond_distance"] = np.array([bd.distance for bd in r.bonds])
        ch["bond_label"] = np.array([bd.label(inp.elements) for bd in r.bonds], dtype="S")
        ch["cohp"] = r.cohp
        ch["icohp"] = r.icohp
        og = ch.create_group("icohp_orbital")
        for i, arr in enumerate(r.icohp_orbital):
            og[f"bond_{i}"] = arr
        frame = getattr(r, "orbital_frame", "global")
        og.attrs["frame"] = frame
        ch.attrs["orbital_frame"] = frame
        if frame == "bond":
            og.attrs["definition"] = ("orbital labels refer to the bond frame: z' along A->B, rows of bond_axes; "
                                      "both atoms' orbitals in the same frame")
            ch["bond_axes"] = r.bond_axes
            ch["channel_names"] = np.array(["sigma", "pi", "delta", "phi", "mixed"], dtype="S")
            ch["cohp_channel"] = r.cohp_channel
            ch["icohp_channel"] = r.icohp_channel
            ch.attrs["channels"] = ("orbital pairs with |m_A| = |m_B| = 0, 1, 2, 3 in the bond frame, and all pairs "
                                    "with |m_A| != |m_B|; channels sum to the total COHP")
        ch.attrs["atom_index_base"] = 0
        co = a.create_group("coop")
        co["energies"] = r.energies
        co["coop"] = r.coop
        co["icoop"] = r.icoop
        cb = a.create_group("cobi")
        cb["energies"] = r.energies
        cb["cobi"] = r.cobi
        cb["icobi"] = r.icobi
        cb.attrs["definition"] = "COBI: COHP with H(R) -> P(R), x2; ICOBI = 2 g sum |P_AB(R)|^2 (Wiberg-type)"
        q = a.create_group("charges")
        q["mulliken_gross_population"] = r.mulliken_gp
        q["lowdin_gross_population"] = r.lowdin_gp
        q["mulliken_charge"] = r.mulliken_charge
        q["lowdin_charge"] = r.lowdin_charge
        if getattr(br, "projection", None):
            key, T, paw_dev = br.projection
            g = f.create_group("projection")
            g.attrs["key_json"] = json.dumps(key)
            g.attrs["max_paw_norm_deviation"] = paw_dev
            g.attrs["definition"] = "A_AE = <chi_mu k|psi_nk> (SCIENCE.md 4.1), norb x nbands per (spin, k)"
            for s_, Ts in enumerate(T):
                for k_, Tk in enumerate(Ts):
                    g.create_dataset(f"T_s{s_}_k{k_}", data=Tk, compression="gzip", compression_opts=1)
        p = f.create_group("performance")
        p["timings_json"] = _json(r.diagnostics.get("timings_s", {}))
        p.attrs["backend"] = r.diagnostics.get("backend", "")


def orbital_resolved_icohp(f, bond: int) -> list[tuple[str, str, list[float]]]:
    """Shell-pair resolved ICOHP of one bond from an open bonding.h5: [(shell_a, shell_b, [per spin])].

    Orbital labels look like 'Fe1_3d_xy'; m-components of a shell are summed (rotation-invariant)."""
    labels = [x.decode() for x in f["basis/orbital_labels"][()]]
    off = f["basis/atom_orbital_offsets"][()]
    a, b = f["analyses/cohp/bond_atoms"][bond]
    M = f[f"analyses/cohp/icohp_orbital/bond_{bond}"][()]  # (spin, na, nb)
    shell = lambda lab: lab.split("_")[1]
    la = [shell(x) for x in labels[off[a]:off[a + 1]]]
    lb = [shell(x) for x in labels[off[b]:off[b + 1]]]
    out = {}
    for i, sa in enumerate(la):
        for j, sb in enumerate(lb):
            out.setdefault((sa, sb), np.zeros(M.shape[0]))
            out[(sa, sb)] += M[:, i, j]
    return [(sa, sb, v.tolist()) for (sa, sb), v in out.items()]


def export(h5path: str | Path, outdir: str | Path, fmt: str = "csv") -> list[Path]:
    import h5py

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    with h5py.File(h5path, "r") as f:
        if f.attrs.get("schema") != SCHEMA:
            raise ValueError(f"{h5path} is not a {SCHEMA} file")
        labels = [x.decode() for x in f["analyses/cohp/bond_label"][()]]
        E = f["analyses/cohp/energies"][()]
        cohp = f["analyses/cohp/cohp"][()]
        coop = f["analyses/coop/coop"][()]
        pdos = f["analyses/pdos/pdos"][()]
        orbl = [x.decode() for x in f["basis/orbital_labels"][()]]
        nspin = cohp.shape[0]
        summary = {
            "bonds": [{"label": lab, "distance": float(d), "icohp": f["analyses/cohp/icohp"][:, i].tolist(),
                       "icoop": f["analyses/coop/icoop"][:, i].tolist(),
                       "icobi": f["analyses/cobi/icobi"][:, i].tolist() if "analyses/cobi" in f else [],
                       "icohp_by_shell_pair": [{"a": sa, "b": sb, "icohp": v}
                                               for sa, sb, v in orbital_resolved_icohp(f, i)]}
                      for i, (lab, d) in enumerate(zip(labels, f["analyses/cohp/bond_distance"][()]))],
            "mulliken_charge": f["analyses/charges/mulliken_charge"][()].tolist(),
            "lowdin_charge": f["analyses/charges/lowdin_charge"][()].tolist(),
            "diagnostics": json.loads(f["reconstruction/diagnostics_json"][()]),
            "provenance": json.loads(f["meta/provenance_json"][()]),
        }
        chnames = ["sigma", "pi", "delta", "phi", "mixed"]
        cohp_ch = f["analyses/cohp/cohp_channel"][()] if "analyses/cohp/cohp_channel" in f else None
        if cohp_ch is not None:
            ich = f["analyses/cohp/icohp_channel"][()]  # (nspin, nbond, 5)
            summary["orbital_frame"] = "bond"
            for i, bd in enumerate(summary["bonds"]):
                bd["icohp_by_channel"] = {c: ich[:, i, j].tolist() for j, c in enumerate(chnames)}
                bd["frame_axes"] = f["analyses/cohp/bond_axes"][i].tolist()
        ws = summary["diagnostics"].get("window_sensitivity") or {}
        alt = np.asarray(ws["icohp"]) if ws.get("available") else None
        if alt is not None:
            for i, bd in enumerate(summary["bonds"]):
                bd[f"icohp_{ws['window']}_window"] = alt[:, i].tolist()
    if fmt == "json":
        p = outdir / "summary.json"
        p.write_text(_json(summary))
        return [p]
    for name, data, cols in (("cohp", cohp, labels), ("coop", coop, labels), ("pdos", pdos, orbl)):
        p = outdir / f"{name}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["energy_eV"] + [f"{c}_spin{s + 1}" for s in range(nspin) for c in cols])
            for i, e in enumerate(E):
                w.writerow([f"{e:.6f}"] + [f"{data[s, j, i]:.8e}" for s in range(nspin) for j in range(len(cols))])
        written.append(p)
    p = outdir / "icohp.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        altkey = f"icohp_{ws['window']}_window" if alt is not None else None
        w.writerow(["bond", "distance_A"] + [f"ICOHP_spin{s + 1}_eV" for s in range(nspin)] +
                   [f"ICOOP_spin{s + 1}" for s in range(nspin)] + [f"ICOBI_spin{s + 1}" for s in range(nspin)] +
                   ([f"ICOHP_{ws['window']}_window_spin{s + 1}_eV" for s in range(nspin)] if altkey else []))
        for bd in summary["bonds"]:
            w.writerow([bd["label"], f"{bd['distance']:.5f}"] + [f"{x:.6f}" for x in bd["icohp"]] +
                       [f"{x:.6f}" for x in bd["icoop"]] + [f"{x:.6f}" for x in bd["icobi"]] +
                       ([f"{x:.6f}" for x in bd[altkey]] if altkey else []))
    written.append(p)
    p = outdir / "icohp_orbital.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bond", "shell_a", "shell_b"] + [f"ICOHP_spin{s + 1}_eV" for s in range(nspin)])
        for bd in summary["bonds"]:
            for e in bd["icohp_by_shell_pair"]:
                w.writerow([bd["label"], e["a"], e["b"]] + [f"{x:.6f}" for x in e["icohp"]])
    written.append(p)
    if cohp_ch is not None:
        p = outdir / "icohp_channel.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bond"] + [f"ICOHP_{c}_spin{s + 1}_eV" for s in range(nspin) for c in chnames])
            for bd in summary["bonds"]:
                w.writerow([bd["label"]] + [f"{bd['icohp_by_channel'][c][s]:.6f}" for s in range(nspin) for c in chnames])
        written.append(p)
        p = outdir / "cohp_channel.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["energy_eV"] + [f"{lab}_{c}_spin{s + 1}" for s in range(nspin) for lab in labels for c in chnames])
            for i, e in enumerate(E):
                w.writerow([f"{e:.6f}"] + [f"{cohp_ch[s, j, c, i]:.8e}" for s in range(nspin) for j in range(len(labels))
                                           for c in range(len(chnames))])
        written.append(p)
    p = outdir / "summary.json"
    p.write_text(_json(summary))
    written.append(p)
    return written
