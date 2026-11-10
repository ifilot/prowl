"""Plotting utilities for bonding.h5 (matplotlib, optional dependency).

Curves are always labelled with their sign convention: PROWL stores COHP (negative =
bonding) and plots -COHP so that bonding contributions point to the right.
"""
from __future__ import annotations

from pathlib import Path



def plot_bonding(h5path: str | Path, out: str | Path, bonds: list[int] | None = None, kind: str = "cohp",
                 emin: float | None = None, emax: float | None = None) -> Path:
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(h5path, "r") as f:
        E = f["analyses/cohp/energies"][()]
        labels = [x.decode() for x in f["analyses/cohp/bond_label"][()]]
        if kind == "cohp":
            data = -f["analyses/cohp/cohp"][()]
            integ = -f["analyses/cohp/icohp"][()]
            xlabel = "−COHP  (bonding → right)"
            ilabel = "−ICOHP"
        elif kind == "coop":
            data = f["analyses/coop/coop"][()]
            integ = f["analyses/coop/icoop"][()]
            xlabel = "COOP (1/eV)  (bonding → right)"
            ilabel = "ICOOP"
        elif kind == "pdos":
            data = f["analyses/pdos/pdos"][()]
            labels = [x.decode() for x in f["basis/orbital_labels"][()]]
            integ = f["analyses/pdos/occupied_integral"][()]
            xlabel = "pDOS (states/eV, Löwdin basis)"
            ilabel = "occ."
        else:
            raise ValueError("kind must be cohp|coop|pdos")
        ref = "E − E_F" if "fermi" in f["meta/provenance_json"][()].decode() else "E"
    nspin = data.shape[0]
    idx = bonds if bonds is not None else list(range(min(len(labels), 6)))
    fig, ax = plt.subplots(figsize=(4.2, 6))
    styles = ["-", "--"]
    for s in range(nspin):
        for i in idx:
            name = labels[i] + (f" spin {s + 1}" if nspin == 2 else "")
            sgn = -1 if (kind == "pdos" and nspin == 2 and s == 1) else 1  # mirror minority pDOS
            ax.plot(sgn * data[s, i], E, styles[s % 2], lw=1.1, label=f"{name}  {ilabel} = {integ[s, i]:.3f}")
    ax.axhline(0.0, color="0.5", lw=0.7, ls=":")
    ax.axvline(0.0, color="0.5", lw=0.7)
    ax.set_ylim(emin if emin is not None else E[0], emax if emax is not None else E[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ref} (eV)")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
