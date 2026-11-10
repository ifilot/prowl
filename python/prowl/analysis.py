"""Reconstruction and chemical-bonding analysis (FP64 NumPy reference implementation).

All equations are documented in docs/SCIENCE.md, sections 5-7. Notation:

* ``T[k]``  (norb x nb)  PAW all-electron projections <chi_mu k | psi_n k>
* ``S[k]``  (norb x norb) overlap of Bloch-summed localized orbitals
* ``X[k] = S^{-1/2}``     Löwdin transformation on the retained subspace
* ``Tp = X T``            projections onto Löwdin-orthonormalized orbitals
* ``C  = (Tp Tp^H)^{-1/2} Tp``  orthonormal-basis coefficients of the re-orthonormalized
  projected bands (polar form; equals Tp (Tp^H Tp)^{-1/2} when nb == norb, LOBSTER's "LSO2")
* ``Cn = X C``            coefficients in the nonorthogonal STO basis ("LSO1")
* ``H[k] = C diag(eps) C^H``  Hamiltonian in the Löwdin basis
* ``H(R) = sum_k w_k exp(-2 pi i k.R) H[k]`` on a full uniform k mesh

Sign convention: COHP is reported as in the literature (negative = bonding);
use ``-cohp`` for plots with bonding to the right. Energies in eV.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- linear algebra
def hermitian_power(A: np.ndarray, p: float, threshold: float = 1e-10):
    """A^p for Hermitian positive (semi)definite A on the subspace with eigenvalues > threshold*max.

    Returns (A^p, eigenvalues, retained_rank)."""
    w, V = np.linalg.eigh(A)
    if w[-1] <= 0:
        raise ValueError("matrix is not positive definite")
    keep = w > threshold * w[-1]
    wp = np.zeros_like(w)
    wp[keep] = w[keep] ** p
    return (V * wp) @ V.conj().T, w, int(keep.sum())


@dataclass
class KReconstruction:
    C: np.ndarray        # norb x nb, orthonormal-basis coefficients
    Cn: np.ndarray       # norb x nb, nonorthogonal-basis coefficients
    H: np.ndarray        # norb x norb, Löwdin-basis Hamiltonian
    spill: np.ndarray    # nb, 1 - <psi|P|psi> per band (P = projector on local subspace)
    s_eig: np.ndarray    # overlap eigenvalues
    rank: int
    residual: object     # nb == norb: max |C^H H C - diag(eps)|; otherwise per-band residual array
    completeness: np.ndarray  # norb, diag(Tp Tp^H): orbital weight carried by the band window


def cut_at_gap(eps: np.ndarray, n: int, tol: float = 1e-4) -> int:
    """Largest n' <= n such that the window [0, n') does not split a degenerate multiplet."""
    n = min(n, len(eps))
    while 0 < n < len(eps) and eps[n] - eps[n - 1] < tol:
        n -= 1
    return n


def window_conditioning(T: np.ndarray, S: np.ndarray, nwin: int, overlap_threshold: float = 1e-8) -> float:
    """Smallest eigenvalue of Tp Tp^H for the lowest ``nwin`` bands (1 = local space fully spanned)."""
    X, _, _ = hermitian_power(S, -0.5, overlap_threshold)
    Tp = X @ T[:, :nwin]
    return float(np.linalg.eigvalsh(Tp @ Tp.conj().T)[0])


def frozen_window_subspace(T: np.ndarray, S: np.ndarray, eps: np.ndarray, occ: np.ndarray, norb: int,
                           occ_tol: float = 1e-8, overlap_threshold: float = 1e-8, ncand: int | None = None):
    """Projection-based disentanglement with a frozen occupied window.

    Keeps all bands with occupation > occ_tol (frozen set F) and adds the
    m = norb - |F| orthonormal combinations of the remaining bands U that carry
    the largest projected local-orbital weight (top eigenvectors of
    Tp_U^H Tp_U), followed by Rayleigh-Ritz diagonalisation within that block.
    Returns square (T_B, eps_B, occ_B) of size norb.
    """
    if ncand is not None:  # only reliably converged bands, never splitting a degenerate multiplet
        ncand = cut_at_gap(eps, ncand)
        T, eps, occ = T[:, :ncand], eps[:ncand], occ[:ncand]
    nb = T.shape[1]
    frozen = np.where(np.abs(occ) > occ_tol)[0]  # includes negative Methfessel-Paxton occupations
    if len(frozen) and frozen.max() != len(frozen) - 1:
        raise ValueError("occupied bands are not the lowest bands")
    nf = len(frozen)
    if nf > norb:
        raise ValueError(f"{nf} occupied bands exceed the {norb} local basis functions; enlarge the basis")
    m = norb - nf
    if nb - nf < m:
        raise ValueError("not enough bands to complete the subspace; increase NBANDS")
    X, _, _ = hermitian_power(S, -0.5, overlap_threshold)
    TpU = X @ T[:, nf:]
    w, V = np.linalg.eigh(TpU.conj().T @ TpU)
    V = V[:, ::-1][:, :m]  # largest projected weight first
    Hm = (V.conj().T * eps[nf:]) @ V
    e_m, Wm = np.linalg.eigh(0.5 * (Hm + Hm.conj().T))
    V = V @ Wm
    T_B = np.concatenate([T[:, :nf], T[:, nf:] @ V], axis=1)
    eps_B = np.concatenate([eps[:nf], e_m])
    occ_B = np.concatenate([occ[:nf], np.zeros(m)])
    wd = w[::-1]
    weakest = float(wd[m - 1]) if m > 0 else 1.0
    # relative gap at the selection cut: ~0 means the complement is not uniquely defined
    gap = float((wd[m - 1] - wd[m]) / max(wd[m - 1], 1e-300)) if 0 < m < len(wd) else 1.0
    return T_B, eps_B, occ_B, weakest, gap


def reconstruct_k(T: np.ndarray, S: np.ndarray, eps: np.ndarray, overlap_threshold: float = 1e-8) -> KReconstruction:
    norb, nb = T.shape
    if nb < norb:
        raise ValueError(f"band window ({nb}) smaller than basis ({norb}); increase NBANDS")
    X, s_eig, rank = hermitian_power(S, -0.5, overlap_threshold)
    if rank < norb:
        raise ValueError(f"overlap matrix rank-deficient ({rank} < {norb}); basis nearly linearly dependent")
    Tp = X @ T
    M = Tp.conj().T @ Tp
    spill = 1.0 - np.real(np.diag(M))
    O = Tp @ Tp.conj().T
    Oi, o_eig, orank = hermitian_power(O, -0.5, 1e-12)
    if orank < norb:
        raise ValueError(f"the {nb} retained bands do not span the {norb}-dimensional local orbital space "
                         f"(smallest eigenvalue {o_eig[0]:.1e}); rerun VASP with more bands (NBANDS; polarization functions typically need NBANDS >= 2 x N_orb); with the lowest-N_orb window, band_window frozen or all may also help")
    C = Oi @ Tp
    H = (C * eps) @ C.conj().T
    H = 0.5 * (H + H.conj().T)
    Cn = X @ C
    if nb == norb:
        resid = float(np.abs(C.conj().T @ H @ C - np.diag(eps)).max())
    else:
        # band-energy reconstruction error <psi_n|P H P|psi_n> - eps_n <psi_n|P|psi_n> per band
        CHC = np.real(np.einsum("in,ij,jn->n", np.conj(C), H, C))
        resid_band = CHC - eps * np.real(np.einsum("in,in->n", np.conj(C), C))
        resid = resid_band  # array; summarised by the caller over occupied bands
    return KReconstruction(C=C, Cn=Cn, H=H, spill=spill, s_eig=s_eig, rank=rank, residual=resid,
                           completeness=np.real(np.diag(O)))


# --------------------------------------------------------------------------- smearing
def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


@dataclass
class EnergyGrid:
    emin: float
    emax: float
    npts: int
    sigma: float
    reference: float  # absolute energy subtracted (E_F)
    kind: str = "gaussian"   # gaussian | tetrahedron (C++ engine only)
    bin_offset: float = -0.5  # tetrahedron bins [E + offset dE, E + (offset + 1) dE]; -1 = LOBSTER convention

    @property
    def energies(self) -> np.ndarray:
        return np.linspace(self.emin, self.emax, self.npts)

    def spread(self, eps_abs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """sum_i weights[i, ...] g(E - (eps_i - ref)) on the grid; weights shape (n,) or (n, m)."""
        E = self.energies
        g = gaussian(E[:, None] - (eps_abs[None, :] - self.reference), self.sigma)  # (npts, n)
        return g @ weights


# --------------------------------------------------------------------------- bonds
@dataclass(frozen=True)
class Bond:
    atom_a: int
    atom_b: int
    R: tuple[int, int, int]
    distance: float

    def label(self, elements: list[str]) -> str:
        return f"{elements[self.atom_a]}{self.atom_a + 1}-{elements[self.atom_b]}{self.atom_b + 1}[{self.R[0]} {self.R[1]} {self.R[2]}]"


def find_bonds(lattice: np.ndarray, frac: np.ndarray, dmin: float, dmax: float) -> list[Bond]:
    """Unique pairs (A, B, R) with dmin < |tau_B + R - tau_A| <= dmax; (B, A, -R) is not repeated."""
    frac = np.asarray(frac, float)
    nat = len(frac)
    span = np.ptp(frac, axis=0) if len(frac) else np.zeros(3)  # unwrapped coordinates widen the search
    nmax = [int(np.ceil(dmax * np.linalg.norm(np.linalg.inv(lattice)[:, i]) + span[i])) + 1 for i in range(3)]
    Rs = np.array([(i, j, k) for i in range(-nmax[0], nmax[0] + 1) for j in range(-nmax[1], nmax[1] + 1)
                   for k in range(-nmax[2], nmax[2] + 1)], dtype=int)
    Rc = Rs @ lattice
    cart = frac @ lattice
    out = []
    for a in range(nat):
        d = (cart[a:, None, :] + Rc[None, :, :]) - cart[a]          # (nat - a, nR, 3)
        dist = np.linalg.norm(d, axis=2)
        bi, ri = np.nonzero((dist > dmin) & (dist <= dmax))
        for b_off, r in zip(bi, ri):
            b = a + int(b_off)
            R = tuple(int(x) for x in Rs[r])
            if a == b and R <= (0, 0, 0):
                continue  # keep one of (A,A,R)/(A,A,-R)
            out.append(Bond(a, b, R, float(dist[b_off, r])))
    out.sort(key=lambda x: (round(x.distance, 6), x.atom_a, x.atom_b, x.R))
    return out


def bvk_alias_ok(lattice: np.ndarray, mesh: tuple[int, int, int], frac_a, frac_b, R, tol: float = 1e-6) -> bool:
    """True if tau_B + R - tau_A is the unique shortest image within the Born-von Karman supercell."""
    d0 = (np.asarray(frac_b) + R - np.asarray(frac_a)) @ lattice
    n0 = np.linalg.norm(d0)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if (i, j, k) == (0, 0, 0):
                    continue
                L = np.array([i * mesh[0], j * mesh[1], k * mesh[2]]) @ lattice
                if np.linalg.norm(d0 + L) <= n0 + tol:
                    return False
    return True


# --------------------------------------------------------------------------- full analysis
@dataclass
class AnalysisInput:
    kfrac: np.ndarray            # (nk, 3) full mesh
    weights: np.ndarray          # (nk,)
    T: list                      # [spin][k] norb x nb
    S: list                      # [k] norb x norb
    eps: np.ndarray              # (nspin, nk, nb)
    occ: np.ndarray              # (nspin, nk, nb) in [0, 1]
    lattice: np.ndarray
    frac: np.ndarray
    elements: list[str]
    zval: np.ndarray             # per atom
    orb_offsets: list[int]
    orbital_labels: list[str]
    overlap_terms: list          # (A, B, R, block) real-space S blocks
    efermi: float
    mesh: tuple[int, int, int]
    spin_factor: float           # 2 for ISPIN=1, 1 otherwise
    orbital_lm: list | None = None  # (l, m) per orbital; needed for orbital_frame = bond


@dataclass
class AnalysisResult:
    energies: np.ndarray
    bonds: list[Bond]
    cohp: np.ndarray        # (nspin, nbond, npts)
    icohp: np.ndarray       # (nspin, nbond) occupation-weighted
    icohp_orbital: list     # [bond] -> (nspin, na, nb) orbital-resolved ICOHP
    coop: np.ndarray
    icoop: np.ndarray
    pdos: np.ndarray        # (nspin, norb, npts), Löwdin basis
    pdos_integral: np.ndarray  # (nspin, norb), occupied
    mulliken_gp: np.ndarray    # (nspin, norb)
    lowdin_gp: np.ndarray
    mulliken_charge: np.ndarray  # per atom
    lowdin_charge: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    cobi: np.ndarray | None = None   # (nspin, nbond, npts) crystal orbital bond index
    icobi: np.ndarray | None = None  # (nspin, nbond) = 2 g sum |P_AB(R)|^2 (Wiberg-type bond index)
    orbital_frame: str = "global"    # frame of icohp_orbital: global | bond
    bond_axes: np.ndarray | None = None      # (nbond, 3, 3) rows x', y', z' of each bond frame (bond frame only)
    cohp_channel: np.ndarray | None = None   # (nspin, nbond, 5, npts) sigma, pi, delta, phi, mixed (bond frame only)
    icohp_channel: np.ndarray | None = None  # (nspin, nbond, 5)


BOND_CHANNELS = ("sigma", "pi", "delta", "phi", "mixed")


def bond_frame_axes(lattice, frac, bond: Bond) -> np.ndarray:
    """Rows x', y', z' of the bond frame: z' along A -> B (B translated by R); x' is the Cartesian axis least parallel
    to z' (ties: x, y, z), orthogonalized; y' = z' x x'. The sigma/pi/delta channel sums do not depend on x'."""
    lattice = np.asarray(lattice, float)
    d = (np.asarray(frac[bond.atom_b], float) + np.asarray(bond.R) - np.asarray(frac[bond.atom_a], float)) @ lattice
    z = d / np.linalg.norm(d)
    e = np.eye(3)[int(np.argmin(np.abs(z) + 1e-12 * np.arange(3)))]
    x = e - (e @ z) * z
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def orbital_rotation(lm: list, axes: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Block-diagonal rotation of one atom's orbitals (shells of 2l+1 consecutive m = -l..l) into the frame `axes`,
    and |m| of every orbital in that frame."""
    from . import _core

    n = len(lm)
    D = np.zeros((n, n))
    i = 0
    while i < n:
        l = lm[i][0]
        if [m for _, m in lm[i:i + 2 * l + 1]] != list(range(-l, l + 1)) or any(x[0] != l for x in lm[i:i + 2 * l + 1]):
            raise ValueError(f"orbital_frame=bond: orbitals of a shell must be ordered m = -l..l (at {i}: {lm[i:i + 2 * l + 1]})")
        D[i:i + 2 * l + 1, i:i + 2 * l + 1] = _core.real_harmonic_rotation(l, axes)
        i += 2 * l + 1
    return D, [abs(m) for _, m in lm]


def _block(off, a):
    return slice(off[a], off[a + 1])


def analyze(inp: AnalysisInput, bonds: list[Bond], grid: EnergyGrid, overlap_threshold: float = 1e-8,
            engine: str = "cpp", eig_backend=None, orbital_frame: str = "global") -> AnalysisResult:
    """Reconstruction + analyses. ``engine="cpp"`` (default) runs the C++/OpenMP core;
    ``engine="numpy"`` is the independent FP64 reference implementation below. ``orbital_frame="bond"``
    (C++ engine): orbital-resolved ICOHP in each bond's frame plus sigma/pi/delta/phi channel COHP."""
    if orbital_frame not in ("global", "bond"):
        raise ValueError("orbital_frame must be global|bond")
    if engine == "cpp":
        return _analyze_cpp(inp, bonds, grid, overlap_threshold, eig_backend, orbital_frame)
    if engine != "numpy":
        raise ValueError("engine must be cpp|numpy")
    if orbital_frame != "global":
        raise ValueError("orbital_frame=bond is implemented in the C++ engine only")
    if grid.kind != "gaussian":
        raise ValueError("the NumPy reference engine implements Gaussian broadening only")
    return _analyze_numpy(inp, bonds, grid, overlap_threshold)


def _finalize(inp: AnalysisInput, bonds, E, cohp, icohp, icohp_orb, coop, icoop, pdos, pdos_int, mull, lowd, diag,
              cobi=None, icobi=None):
    off = inp.orb_offsets
    nat = len(inp.frac)
    mull_atom = np.array([mull[:, off[a]:off[a + 1]].sum() for a in range(nat)])
    lowd_atom = np.array([lowd[:, off[a]:off[a + 1]].sum() for a in range(nat)])
    diag["bonds_alias_free"] = [bvk_alias_ok(inp.lattice, inp.mesh, inp.frac[b.atom_a], inp.frac[b.atom_b], b.R)
                                for b in bonds]
    diag["electrons_projected_lowdin"] = float(lowd.sum())
    diag["electrons_projected_mulliken"] = float(mull.sum())
    return AnalysisResult(energies=E, bonds=bonds, cohp=cohp, icohp=icohp, icohp_orbital=icohp_orb, coop=coop,
                          icoop=icoop, pdos=pdos, pdos_integral=pdos_int, mulliken_gp=mull, lowdin_gp=lowd,
                          mulliken_charge=inp.zval - mull_atom, lowdin_charge=inp.zval - lowd_atom, diagnostics=diag,
                          cobi=cobi, icobi=icobi)


def _analyze_cpp(inp: AnalysisInput, bonds: list[Bond], grid: EnergyGrid, overlap_threshold: float,
                 eig_backend=None, orbital_frame: str = "global") -> AnalysisResult:
    from . import _core

    nspin = len(inp.T)
    nk = len(inp.kfrac)
    Sblocks = {(A, B, tuple(R)): blk for A, B, R, blk in (inp.overlap_terms or [])}
    blist = [(b.atom_a, b.atom_b, list(b.R), Sblocks.get((b.atom_a, b.atom_b, b.R))) for b in bonds]
    axes = None
    if orbital_frame == "bond":
        if inp.orbital_lm is None:
            raise ValueError("orbital_frame=bond needs AnalysisInput.orbital_lm")
        off = inp.orb_offsets
        axes = np.array([bond_frame_axes(inp.lattice, inp.frac, b) for b in bonds]).reshape(-1, 3, 3)
        ext = []
        for t, b, ax in zip(blist, bonds, axes):
            Da, ma = orbital_rotation(inp.orbital_lm[off[b.atom_a]:off[b.atom_a + 1]], ax)
            Db, mb = orbital_rotation(inp.orbital_lm[off[b.atom_b]:off[b.atom_b + 1]], ax)
            ext.append((*t, Da, Db, ma, mb))
        blist = ext
    o = _core.analyze(np.asarray(inp.kfrac, float), list(map(float, inp.weights)),
                      [[np.asarray(inp.T[s][k], complex) for k in range(nk)] for s in range(nspin)],
                      [np.asarray(x, complex) for x in inp.S],
                      [[np.asarray(inp.eps[s][k], float) for k in range(nk)] for s in range(nspin)],
                      [[np.asarray(inp.occ[s][k], float) for k in range(nk)] for s in range(nspin)],
                      list(inp.orb_offsets), float(inp.spin_factor), blist,
                      [grid.emin, grid.emax, float(grid.npts), grid.sigma, grid.reference,
                       {"gaussian": 0.0, "tetrahedron": 1.0}[grid.kind], grid.bin_offset], overlap_threshold,
                      eig_backend, np.rint(np.asarray(inp.kfrac) * np.array(inp.mesh)).astype(np.int32),
                      list(inp.mesh), np.linalg.inv(np.asarray(inp.lattice)).T)
    nb = len(bonds)
    icohp_orb = [np.stack([o["icohp_orbital"][s * nb + i] for s in range(nspin)]) for i in range(nb)]
    diag = {
        "abs_charge_spilling": o["abs_charge_spilling"],
        "abs_total_spilling": o["abs_total_spilling"],
        "max_reconstruction_residual_ev": o["max_reconstruction_residual"],
        "min_overlap_eigenvalue": o["min_overlap_eigenvalue"],
        "band_energy_ev": list(o["band_energy"]),
        "electrons_dft": o["electrons_dft"],
        "analysis_engine": "cpp",
        "device_eigensolver": bool(o["used_device_eigensolver"]),
    }
    res = _finalize(inp, bonds, grid.energies, o["cohp"], o["icohp"], icohp_orb, o["coop"], o["icoop"], o["pdos"],
                    o["pdos_integral"], o["mulliken_gp"], o["lowdin_gp"], diag, o["cobi"], o["icobi"])
    if orbital_frame == "bond":
        res.orbital_frame, res.bond_axes = "bond", axes
        res.cohp_channel, res.icohp_channel = np.asarray(o["cohp_channel"]), np.asarray(o["icohp_channel"])
    return res


def _analyze_numpy(inp: AnalysisInput, bonds: list[Bond], grid: EnergyGrid, overlap_threshold: float = 1e-8) -> AnalysisResult:
    nspin = len(inp.T)
    nk = len(inp.kfrac)
    norb = inp.T[0][0].shape[0]
    off = inp.orb_offsets
    sf = inp.spin_factor
    w = inp.weights

    recon = [[reconstruct_k(inp.T[s][k], inp.S[k], inp.eps[s, k], overlap_threshold) for k in range(nk)] for s in range(nspin)]

    # Real-space Hamiltonian blocks for the requested bonds: H_AB(R) = sum_k w_k e^{-2 pi i k.R} H_AB(k)
    phases = {b.R: np.exp(2j * np.pi * inp.kfrac @ np.array(b.R)) for b in bonds}
    HR = [[sum(w[k] * np.conj(phases[b.R][k]) * recon[s][k].H[_block(off, b.atom_a), _block(off, b.atom_b)] for k in range(nk))
           for b in bonds] for s in range(nspin)]
    # density-matrix blocks P_AB(R) = sum_k w_k e^{-ik.R} sum_n f_n C_An C_Bn^*  (per spin, Löwdin basis)
    PR = [[sum(w[k] * np.conj(phases[b.R][k]) *
               ((recon[s][k].C[_block(off, b.atom_a)] * inp.occ[s, k]) @ recon[s][k].C[_block(off, b.atom_b)].conj().T)
               for k in range(nk)) for b in bonds] for s in range(nspin)]
    Sblocks = {(A, B, tuple(R)): blk for A, B, R, blk in inp.overlap_terms}

    E = grid.energies
    npts = len(E)
    nbond = len(bonds)
    cohp = np.zeros((nspin, nbond, npts))
    icohp = np.zeros((nspin, nbond))
    coop = np.zeros((nspin, nbond, npts))
    icoop = np.zeros((nspin, nbond))
    cobi = np.zeros((nspin, nbond, npts))
    icobi = np.zeros((nspin, nbond))
    icohp_orb = [np.zeros((nspin, off[b.atom_a + 1] - off[b.atom_a], off[b.atom_b + 1] - off[b.atom_b])) for b in bonds]
    pdos = np.zeros((nspin, norb, npts))
    pdos_int = np.zeros((nspin, norb))
    mull = np.zeros((nspin, norb))
    lowd = np.zeros((nspin, norb))
    band_energy = np.zeros(nspin)
    spill_occ_abs = spill_occ_num = spill_all_abs = spill_all_num = 0.0
    max_resid = 0.0
    min_seig = np.inf

    for s in range(nspin):
        for k in range(nk):
            r = recon[s][k]
            eps = inp.eps[s, k]
            f = inp.occ[s, k]
            wk = w[k]
            C, Cn = r.C, r.Cn
            # broadening kernel (npts x nb), built once per (spin, k) and shared by all spectra
            G = sf * wk * gaussian(E[:, None] - (eps[None, :] - grid.reference), grid.sigma)
            # pDOS (Löwdin basis) and populations
            pw = np.abs(C) ** 2  # (norb, nb)
            pdos[s] += (G @ pw.T).T
            pdos_int[s] += sf * wk * (pw @ f)
            lowd[s] += sf * wk * (pw @ f)
            SCn = inp.S[k] @ Cn
            mull[s] += sf * wk * (np.real(np.conj(Cn) * SCn) @ f)
            band_energy[s] += sf * wk * np.sum(f * eps)
            spill_occ_abs += wk * np.sum(f * np.abs(r.spill))
            spill_occ_num += wk * np.sum(f)
            spill_all_abs += wk * np.sum(np.abs(r.spill))
            spill_all_num += wk * len(r.spill)
            if np.ndim(r.residual) == 0:
                max_resid = max(max_resid, float(r.residual))
            else:
                occ_mask = f > 1e-8
                if occ_mask.any():
                    max_resid = max(max_resid, float(np.abs(r.residual[occ_mask]).max()))
            min_seig = min(min_seig, r.s_eig.min())
            cohp_bands = np.zeros((nbond, len(eps)))
            coop_bands = np.zeros((nbond, len(eps)))
            cobi_bands = np.zeros((nbond, len(eps)))
            Cc, Cnc = np.conj(C), np.conj(Cn)
            for ib, b in enumerate(bonds):
                A, B = _block(off, b.atom_a), _block(off, b.atom_b)
                ph = phases[b.R][k]
                # COHP: Re[ C*_mu n H_mu nu(R) e^{ik.R} C_nu n ]
                HC = (HR[s][ib] * ph) @ C[B]  # (na, nband)
                orb = np.real(Cc[A][:, None, :] * (HR[s][ib] * ph)[:, :, None] * C[B][None, :, :])
                cohp_bands[ib] = np.real(np.sum(Cc[A] * HC, axis=0))
                cobi_bands[ib] = 2.0 * np.real(np.sum(Cc[A] * ((PR[s][ib] * ph) @ C[B]), axis=0))
                icohp_orb[ib][s] += sf * wk * np.einsum("ijn,n->ij", orb, f)
                # COOP in the nonorthogonal basis with exact real-space overlap S_AB(R)
                Sb = Sblocks.get((b.atom_a, b.atom_b, b.R))
                if Sb is not None:
                    coop_bands[ib] = np.real(np.sum(Cnc[A] * ((Sb * ph) @ Cn[B]), axis=0))
            cohp[s] += cohp_bands @ G.T
            coop[s] += coop_bands @ G.T
            icohp[s] += sf * wk * (cohp_bands @ f)
            icoop[s] += sf * wk * (coop_bands @ f)
            cobi[s] += cobi_bands @ G.T
            icobi[s] += sf * wk * (cobi_bands @ f)

    diag = {
        "abs_charge_spilling": spill_occ_abs / spill_occ_num,
        "abs_total_spilling": spill_all_abs / spill_all_num,
        "max_reconstruction_residual_ev": max_resid,
        "min_overlap_eigenvalue": float(min_seig),
        "band_energy_ev": band_energy.tolist(),
        "electrons_dft": float(sf * np.sum(w[None, :, None] * inp.occ)),
        "analysis_engine": "numpy",
    }
    return _finalize(inp, bonds, E, cohp, icohp, icohp_orb, coop, icoop, pdos, pdos_int, mull, lowd, diag, cobi, icobi)
