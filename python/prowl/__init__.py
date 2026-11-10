"""PROWL: open VASP -> localized-orbital bonding analysis (COHP, COOP, pDOS, charges)."""
from __future__ import annotations

__version__ = "0.1.0"

import os
from pathlib import Path

from . import _core  # noqa: F401  (native extension)
from ._core import ProwlError  # noqa: F401


# Built-in basis sets: basis.source name -> (file under data/basis/, contains its own polarization shells).
# Self-contained sets are never merged with the generated polarization library (their occupation-0 shells are the
# polarization shells).
BUILTIN_BASES = {
    "koga_hf_sto": ("koga/koga_hf_sto.json", False),
    "paw-pbe-sto-1": ("paw-pbe-sto-1/paw-pbe-sto-1.json", True),
}


def basis_path(source: str = "koga_hf_sto") -> Path:
    """File of a built-in basis set: the copy shipped in the wheel, else the source tree."""
    if source not in BUILTIN_BASES:
        raise ProwlError(f"InvalidArgument: unknown basis source '{source}' (built-in: {', '.join(BUILTIN_BASES)}; "
                          "other sets need basis.library)")
    rel = Path("data") / "basis" / BUILTIN_BASES[source][0]
    here = Path(__file__).resolve().parent
    for cand in (here / rel, here.parents[1] / rel):
        if cand.is_file():
            return cand
    raise ProwlError(f"FileNotFound: basis data for '{source}' not found")


def default_basis_path() -> Path:
    """Koga basis JSON: $PROWL_BASIS_FILE, else the copy shipped in the wheel, else the source tree."""
    env = os.environ.get("PROWL_BASIS_FILE")
    if env:
        return Path(env)
    try:
        return basis_path("koga_hf_sto")
    except ProwlError:
        raise ProwlError("FileNotFound: Koga basis data not found; set PROWL_BASIS_FILE") from None


def default_polarization_path() -> Path:
    """PROWL's generated polarization functions: $PROWL_POLARIZATION_FILE, else the packaged copy."""
    env = os.environ.get("PROWL_POLARIZATION_FILE")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    rel = Path("data") / "basis" / "polarization" / "prowl_lda_polarization.json"
    for cand in (here / rel, here.parents[1] / rel):
        if cand.is_file():
            return cand
    raise ProwlError("FileNotFound: polarization basis data not found; set PROWL_POLARIZATION_FILE")


def load_basis_library(path: str | os.PathLike | None = None, polarization: str | os.PathLike | bool | None = None):
    """Koga library (or ``path``), optionally merged with a polarization library (True = default file)."""
    lib = _core.StoLibrary.load(str(path or default_basis_path()))
    if polarization:
        pol = default_polarization_path() if polarization is True else Path(polarization)
        lib.merge(_core.StoLibrary.load(str(pol)))
    return lib


def __getattr__(name):
    # lazy top-level API (design: ``from prowl import Calculation, BondingConfig``)
    if name == "Calculation":
        from .calculation import Calculation
        return Calculation
    if name == "BondingConfig":
        from .config import BondingConfig
        return BondingConfig
    raise AttributeError(name)
