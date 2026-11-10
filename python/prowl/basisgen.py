"""Generation of polarization functions (unoccupied valence shells, e.g. Fe 4p) for the STO basis.

Koga's Hartree-Fock tabulations contain occupied shells only. Polarization functions are generated
here, independently of any third-party basis data, in three steps (docs/BASIS_PROVENANCE.md):

1. A non-relativistic, spherical, spin-unpolarized Kohn-Sham LDA atom (Slater exchange + VWN5
   correlation, the functional of the NIST atomic reference data) is solved for an *excited*
   configuration in which the polarization shell is singly occupied, e.g. Fe [Ar] 3d6 4s1 4p1.
   Occupying the shell makes it a bound, self-consistent orbital of the neutral atom with a
   compact, physically sensible radial extent.
2. Its all-electron radial function R(r) (with the correct nodal structure) is fitted by linear
   least squares to an even-tempered set of normalized Slater primitives r^(n-1) exp(-zeta r).
3. The result is written in PROWL's basis schema (prowl-sto-basis/1) with occupation 0, so the
   default ZVAL-based selection never picks it; it is used only when requested.

Numerics: radial equation in x = ln r with u(r) = r^(1/2) w(x), second-order finite differences on
a uniform x grid, solved as a symmetric tridiagonal eigenproblem (scipy eigh_tridiagonal). All
quantities in Hartree atomic units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh_tridiagonal

L_LETTER = "spdf"


# ----------------------------------------------------------------------------- LDA (VWN5)
def lda_xc(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spin-unpolarized LDA: Slater exchange + VWN5 correlation. Returns (eps_xc, v_xc) per particle."""
    rho = np.maximum(rho, 1e-30)
    kf = (3.0 * rho / math.pi) ** (1.0 / 3.0)
    ex, vx = -0.75 * kf, -kf
    rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    A, x0, b, c = 0.0310907, -0.10498, 3.72744, 12.9352
    Q = math.sqrt(4 * c - b * b)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    at = np.arctan(Q / (2 * x + b))
    ec = A * (np.log(x * x / X) + 2 * b / Q * at
              - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2 * (b + 2 * x0) / Q * at))
    den = (2 * x + b) ** 2 + Q * Q
    dec = A * (2 / x - (2 * x + b) / X - 4 * b / den
               - b * x0 / X0 * (2 / (x - x0) - (2 * x + b) / X - 4 * (b + 2 * x0) / den))
    vc = ec - x / 6.0 * dec
    return ex + ec, vx + vc


# ----------------------------------------------------------------------------- radial grid / solver
@dataclass
class RadialGrid:
    rmin: float = 1e-9   # w ~ r^(l+1/2) at the inner Dirichlet boundary: error ~ Z rmin
    rmax: float = 50.0
    n: int = 20000

    def __post_init__(self):
        self.x = np.linspace(math.log(self.rmin), math.log(self.rmax), self.n)
        self.h = self.x[1] - self.x[0]
        self.r = np.exp(self.x)

    def integrate(self, f_r: np.ndarray) -> float:
        """int_0^inf f(r) dr = int f r dx (Simpson in x)."""
        from scipy.integrate import simpson

        return float(simpson(f_r * self.r, x=self.x))

    def cumulative(self, f_r: np.ndarray) -> np.ndarray:
        from scipy.integrate import cumulative_trapezoid

        return cumulative_trapezoid(f_r * self.r, self.x, initial=0.0)


def solve_radial(grid: RadialGrid, V: np.ndarray, l: int, nstates: int, guess=None):
    """Lowest `nstates` eigenpairs of -1/2 u'' + (V + l(l+1)/2r^2) u = E u, u(0) = u(rmax) = 0.
    Returns energies, radial functions R(r) = u/r normalized to int R^2 r^2 dr = 1, and the raw
    solver vectors (pass ``guess=(E, W)`` from a nearby potential to warm-start; the node count of
    every state is checked, so a warm start cannot silently converge to the wrong state)."""
    from scipy.linalg import solve_banded

    r, h = grid.r, grid.h
    q = 2 * r * r * V + (l + 0.5) ** 2
    b = 2 * r * r
    a_diag = 2.0 / h**2 + q          # A = -D2/h^2 + diag(q) (unscaled, well conditioned)
    a_off = -1.0 / h**2
    # 1) eigenvalue estimates of the symmetric standard form B^-1/2 A B^-1/2 by bisection (stebz;
    #    the graded matrix spans many orders of magnitude, other drivers lose accuracy here)
    if guess is None:
        d = a_diag / b
        e = a_off / np.sqrt(b[:-1] * b[1:])
        E0 = eigh_tridiagonal(d, e, eigvals_only=True, select="i", select_range=(0, nstates - 1),
                              lapack_driver="stebz", tol=1e-300)
    else:
        E0 = guess[0]
    # 2) Rayleigh-quotient inverse iteration on the generalized pencil (A4, B) with banded LU, where A4
    #    uses the fourth-order five-point second derivative (the estimates from step 1 are within the
    #    O(h^2) discretization difference, far below the level spacing)
    c0, c1, c2 = 30.0 / (12 * h * h), -16.0 / (12 * h * h), 1.0 / (12 * h * h)
    n = len(r)
    A_diag = c0 + q
    E = np.empty(nstates)
    W = np.empty((n, nstates))
    for i in range(nstates):
        lam = E0[i]
        w = guess[1][:, i].copy() if guess is not None else np.exp(-((np.arange(n) - n / 2) / (n / 4)) ** 2)
        for _ in range(8):
            ab = np.zeros((5, n))
            ab[0, 2:], ab[1, 1:], ab[2], ab[3, :-1], ab[4, :-2] = c2, c1, A_diag - lam * b, c1, c2
            w = solve_banded((2, 2), ab, b * w)
            w /= math.sqrt(np.sum(b * w * w))
            Aw = A_diag * w
            Aw[1:] += c1 * w[:-1]
            Aw[:-1] += c1 * w[1:]
            Aw[2:] += c2 * w[:-2]
            Aw[:-2] += c2 * w[2:]
            lam_new = float(w @ Aw)  # w^T B w = 1
            done = abs(lam_new - lam) < 1e-13 * max(1.0, abs(lam))
            lam = lam_new
            if done:
                break
        nodes = int(np.sum(np.abs(np.diff(np.sign(w[np.abs(w) > 1e-10 * np.abs(w).max()]))) > 0))
        if nodes != i and guess is not None:
            return solve_radial(grid, V, l, nstates)  # warm start failed: cold start
        if nodes != i:
            raise RuntimeError(f"radial solver: state {i} of l={l} has {nodes} nodes (eigenvalue {lam:.6f})")
        E[i], W[:, i] = lam, w
    u = np.sqrt(r)[:, None] * W
    R = u / r[:, None]
    for i in range(nstates):
        nrm = grid.integrate(R[:, i] ** 2 * r * r)
        R[:, i] /= math.sqrt(nrm)
        # sign convention: positive near the last extremum (outer lobe), as Koga's tables at large r
        idx = np.argmax(np.abs(R[:, i] * r))
        if R[idx, i] < 0:
            R[:, i] *= -1
    return E, R, W


def thomas_fermi_potential(Z: int, r: np.ndarray) -> np.ndarray:
    """Analytic fit to the Thomas-Fermi screening function (starting guess)."""
    alpha, beta, gamma = 0.7280642371, -0.5430794693, 0.3612163121
    x = r * (128.0 * Z / (9.0 * math.pi**2)) ** (1.0 / 3.0)
    sx = np.sqrt(x)
    Zeff = Z * (1 + alpha * sx + beta * x * np.exp(-gamma * sx)) ** 2 * np.exp(-2 * alpha * sx)
    return -np.maximum(Zeff, 1.0) / r


@dataclass
class AtomResult:
    Z: int
    occupations: dict            # "4p" -> occupation
    energies: dict               # "4p" -> eigenvalue (Ha)
    orbitals: dict               # "4p" -> R(r)
    total_energy: float
    grid: RadialGrid
    iterations: int
    converged: bool
    history: list = field(default_factory=list)


def parse_shells(config: dict) -> dict[int, list[tuple[int, str, float]]]:
    by_l: dict[int, list] = {}
    for lab, f in config.items():
        n, l = int(lab[:-1]), L_LETTER.index(lab[-1])
        by_l.setdefault(l, []).append((n, lab, float(f)))
    for l in by_l:
        by_l[l].sort()
        ns = [n for n, _, _ in by_l[l]]
        if ns != list(range(l + 1, l + 1 + len(ns))):
            raise ValueError(f"configuration must list every shell of l={l} from n={l + 1}: {ns}")
    return by_l


def solve_atom(Z: int, config: dict, grid: RadialGrid | None = None, mix: float = 0.3, tol: float = 1e-9,
               maxiter: int = 400) -> AtomResult:
    """Self-consistent LDA atom. `config`: shell label -> occupation, every shell listed (e.g. Ne:
    {'1s': 2, '2s': 2, '2p': 6}); occupations may be zero (the orbital is still computed)."""
    grid = grid or RadialGrid()
    r = grid.r
    by_l = parse_shells(config)
    if abs(sum(config.values()) - Z) > 1e-9:
        raise ValueError(f"configuration has {sum(config.values())} electrons, Z = {Z} (neutral atoms only)")
    V = thomas_fermi_potential(Z, r)
    hist, E_old, conv, it = [], 0.0, False, 0
    # Anderson (two-step) mixing of the potential
    Vin_prev = Vout_prev = None
    guesses: dict = {}
    for it in range(1, maxiter + 1):
        eig, orb, rho = {}, {}, np.zeros_like(r)
        for l, shells in by_l.items():
            E, R, W = solve_radial(grid, V, l, len(shells), guesses.get(l))
            guesses[l] = (E, W)
            for i, (n, lab, f) in enumerate(shells):
                eig[lab], orb[lab] = float(E[i]), R[:, i]
                rho += f * R[:, i] ** 2 / (4 * math.pi)
        # Hartree potential
        q_in = grid.cumulative(4 * math.pi * rho * r * r)
        outer = grid.cumulative(4 * math.pi * rho * r)
        VH = q_in / r + (outer[-1] - outer)
        exc, vxc = lda_xc(rho)
        Vout = -Z / r + VH + vxc
        Etot = (sum(config[k] * eig[k] for k in config)
                - 0.5 * grid.integrate(VH * rho * 4 * math.pi * r * r)
                + grid.integrate((exc - vxc) * rho * 4 * math.pi * r * r))
        dV = Vout - V
        err = math.sqrt(grid.integrate(dV**2 * rho * 4 * math.pi * r * r) / Z)
        hist.append((Etot, err))
        if abs(Etot - E_old) < tol and err < 1e-6:
            conv = True
            break
        E_old = Etot
        if Vin_prev is None:
            Vn = V + mix * dV
        else:
            dV_prev = Vout_prev - Vin_prev
            ddv = dV - dV_prev
            w = 4 * math.pi * rho * r * r
            den = grid.integrate(ddv * ddv * w)
            beta = grid.integrate(dV * ddv * w) / den if den > 0 else 0.0
            beta = min(max(beta, -1.0), 1.0)
            Vbar_in = V - beta * (V - Vin_prev)
            Vbar_out = Vout - beta * (Vout - Vout_prev)
            Vn = Vbar_in + mix * (Vbar_out - Vbar_in)
        Vin_prev, Vout_prev = V, Vout
        V = Vn
    return AtomResult(Z, dict(config), eig, orb, Etot, grid, it, conv, hist)


# ----------------------------------------------------------------------------- STO fit
def sto_normalization(n: int, zeta: float) -> float:
    return (2 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def sto_value(n: int, zeta: float, r: np.ndarray) -> np.ndarray:
    return sto_normalization(n, zeta) * r ** (n - 1) * np.exp(-zeta * r)


def even_tempered(l: int, zmin: float, zmax: float, ratio: float = 1.45) -> list[tuple[int, float]]:
    nprim = int(math.ceil(math.log(zmax / zmin) / math.log(ratio))) + 1
    return [(l + 1, zmin * ratio**k) for k in range(nprim)]


def fit_sto(grid: RadialGrid, R: np.ndarray, prims: list[tuple[int, float]], rcond: float = 1e-11):
    """Least-squares fit R(r) ~ sum_p c_p phi_p(r) in the metric int f g r^2 dr (analytic Gram matrix).
    Returns coefficients (normalized fit) and the residual norm^2 of the normalized functions."""
    r = grid.r
    m = len(prims)
    G = np.empty((m, m))
    for i, (ni, zi) in enumerate(prims):
        for j, (nj, zj) in enumerate(prims):
            N = ni + nj
            # log form avoids overflow for large n
            G[i, j] = math.exp(math.log(sto_normalization(ni, zi)) + math.log(sto_normalization(nj, zj))
                               + math.lgamma(N + 1) - (N + 1) * math.log(zi + zj))
    b = np.array([grid.integrate(sto_value(n, z, r) * R * r * r) for n, z in prims])
    w, U = np.linalg.eigh(G)
    keep = w > rcond * w.max()
    c = U[:, keep] @ ((U[:, keep].T @ b) / w[keep])
    norm2 = float(c @ G @ c)
    c /= math.sqrt(norm2)
    overlap = float(c @ b)  # <R|fit> with both normalized
    return c, 2.0 * (1.0 - overlap), int(keep.sum())


# ----------------------------------------------------------------------------- configurations
def excited_configuration(ground: dict, pol_label: str) -> tuple[dict, str]:
    """Move one electron into the polarization shell: from the ns shell of the same n (or n of the
    outermost s shell) if occupied, otherwise from the outermost d shell. Returns (config, description)."""
    cfg = dict(ground)
    n = int(pol_label[:-1])
    s_lab = f"{n}s"
    if cfg.get(s_lab, 0) >= 1:
        donor = s_lab
    else:
        ds = sorted((k for k in cfg if k.endswith("d") and cfg[k] > 0), key=lambda k: int(k[:-1]))
        if not ds:
            raise ValueError(f"no donor shell for {pol_label} in {ground}")
        donor = ds[-1]
    cfg[donor] -= 1
    # every lower shell of the same l must be present (possibly empty) for the solver
    l = L_LETTER.index(pol_label[-1])
    for nn in range(l + 1, n + 1):
        cfg.setdefault(f"{nn}{pol_label[-1]}", 0.0)
    cfg[pol_label] = cfg.get(pol_label, 0.0) + 1
    return cfg, f"{donor} -> {pol_label}"


def koga_ground_configuration(lib_json: dict, element: str) -> dict:
    return {s["label"]: float(s["occupation"]) for s in lib_json["elements"][element]["shells"]}
