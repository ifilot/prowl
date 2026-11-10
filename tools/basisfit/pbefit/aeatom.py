"""All-electron spherical Kohn-Sham atom: PBE (or LDA), non-relativistic or scalar-relativistic (M4 cross-check).

Radial problem for u = r R on a logarithmic grid, discretized as a symmetric Sturm-Liouville problem
    -1/2 (p u')' + q u = E u,
non-relativistic: p = 1, q = V + l(l+1)/(2 r^2);
scalar-relativistic (Koelling-Harmon, no spin-orbit), with M(r) = 1 + (E - V)/(2 c^2):
    p = 1/M,  q = V + l(l+1)/(2 M r^2) - M'/(2 M^2 r).
M depends on the eigenvalue: each state is iterated to self-consistency in E. Second-order finite
differences on the non-uniform r grid (symmetric tridiagonal pencil), eigenvalues by bisection (stebz),
eigenvectors by inverse iteration. The scalar-relativistic radial function is the large component g = u/r,
normalized as int g^2 r^2 dr = 1 (small-component norm neglected, as in VASP's partial waves).

PBE: Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996); PW92 correlation. Spin-unpolarized, spherical density.
v_xc = df/drho - (1/r^2) d/dr (r^2 2 df/dsigma rho'),  sigma = rho'^2; partial derivatives of the energy density
f(rho, sigma) by central differences (relative step 1e-4, error ~1e-9).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded

C_LIGHT = 137.035999084
LETTER = "spdf"


# ------------------------------------------------------------------------------------------------- functionals
def _pw92(rs):
    A, a1, b1, b2, b3, b4 = 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    sr = np.sqrt(rs)
    return -2 * A * (1 + a1 * rs) * np.log1p(1.0 / (2 * A * (b1 * sr + b2 * rs + b3 * rs * sr + b4 * rs * rs)))


def f_pbe(rho, sigma):
    """PBE exchange-correlation energy density per volume, spin-unpolarized."""
    rho = np.maximum(rho, 1e-30)
    kf = (3 * math.pi**2 * rho) ** (1 / 3)
    ex_lda = -3 / (4 * math.pi) * kf
    s2 = sigma / (2 * kf * rho) ** 2
    kappa, mu = 0.804, 0.2195149727645171
    Fx = 1 + kappa - kappa / (1 + mu * s2 / kappa)
    rs = (3 / (4 * math.pi * rho)) ** (1 / 3)
    ec = _pw92(rs)
    beta, gamma = 0.06672455060314922, (1 - math.log(2)) / math.pi**2
    ks = np.sqrt(4 * kf / math.pi)
    t2 = sigma / (2 * ks * rho) ** 2
    Ac = beta / gamma / np.expm1(-ec / gamma)
    At2 = Ac * t2
    H = gamma * np.log1p(beta / gamma * t2 * (1 + At2) / (1 + At2 + At2 * At2))
    return rho * (ex_lda * Fx + ec + H)


def fs_pbe(rho, sigma):
    """Analytic d f_pbe / d sigma."""
    rho = np.maximum(rho, 1e-30)
    kf = (3 * math.pi**2 * rho) ** (1 / 3)
    ex_lda = -3 / (4 * math.pi) * kf
    s2 = sigma / (2 * kf * rho) ** 2
    kappa, mu = 0.804, 0.2195149727645171
    fx_s = rho * ex_lda * mu / (1 + mu * s2 / kappa) ** 2 / (2 * kf * rho) ** 2
    rs = (3 / (4 * math.pi * rho)) ** (1 / 3)
    ec = _pw92(rs)
    beta, gamma = 0.06672455060314922, (1 - math.log(2)) / math.pi**2
    ks = np.sqrt(4 * kf / math.pi)
    t2 = sigma / (2 * ks * rho) ** 2
    Ac = beta / gamma / np.expm1(-ec / gamma)
    y = Ac * t2
    X = beta / gamma * t2 * (1 + y) / (1 + y + y * y)
    dX = beta / gamma * (1 + 2 * y) / (1 + y + y * y) ** 2
    fc_s = rho * gamma * dX / (1 + X) / (2 * ks * rho) ** 2
    return fx_s + fc_s


def f_lda(rho, sigma):
    rho = np.maximum(rho, 1e-30)
    kf = (3 * math.pi**2 * rho) ** (1 / 3)
    rs = (3 / (4 * math.pi * rho)) ** (1 / 3)
    return rho * (-3 / (4 * math.pi) * kf + _pw92(rs))


# ------------------------------------------------------------------------------------------------------- grid
@dataclass
class Grid:
    rmin: float = 1e-7
    rmax: float = 60.0
    n: int = 16000

    def __post_init__(self):
        self.x = np.linspace(math.log(self.rmin), math.log(self.rmax), self.n)
        self.h = self.x[1] - self.x[0]
        self.r = np.exp(self.x)
        w = np.full(self.n, self.h)        # trapezoid in x: int f dr = int f r dx
        w[0] = w[-1] = self.h / 2
        self.wx = w * self.r

    def integrate(self, f):
        return float(np.sum(f * self.wx))

    def cumulative(self, f):
        g = f * self.r
        out = np.zeros_like(g)
        out[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * self.h)
        return out

    def deriv(self, f):
        """df/dr by 4th-order differences in x."""
        d = np.gradient(f, self.h, edge_order=2)
        d[2:-2] = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * self.h)
        return d / self.r


def xc_potential(grid, rho, func="pbe", r0=0.0):
    """(energy density f, potential v_xc) of a spherical density. r0 > 0: the density gradient is switched off
    smoothly inside r ~ r0 (LDA in the deep core; used for scalar-relativistic densities, which diverge at r=0)."""
    f = f_pbe if func == "pbe" else f_lda
    dr = grid.deriv(rho)
    if r0 > 0:
        dr = dr * (1 - np.exp(-(grid.r / r0) ** 4))
    sig = dr * dr
    e = f(rho, sig)
    hr = 1e-4 * np.maximum(rho, 1e-30)
    f_r = (f(rho + hr, sig) - f(np.maximum(rho - hr, 1e-30), sig)) / (2 * hr)
    if func == "lda":
        v = f_r
    else:
        f_s = fs_pbe(rho, sig)
        flux = grid.r ** 2 * 2 * f_s * dr
        v = f_r - grid.deriv(flux) / grid.r ** 2
    small = rho < 1e-12
    v[small] = 0.0
    e[small] = 0.0
    return e, v


# --------------------------------------------------------------------------------------------- radial solver
def _pencil(grid, p, q):
    """Symmetric tridiagonal pencil (A, B) for -1/2 (p u')' + q u = E u with u = r^(1/2) w(x), x = ln r:
    -(p w_x)_x + (p/4 - p_x/2 + 2 r^2 q) w = E 2 r^2 w,  w = 0 at both ends (w ~ r^(l+1/2) near 0)."""
    r, h = grid.r, grid.h
    px = np.gradient(p, h)
    ph = 0.5 * (p[1:] + p[:-1]) / h ** 2
    diag = p / 4 - px / 2 + 2 * r * r * q
    diag[:-1] += ph
    diag[1:] += ph
    off = -ph
    B = 2 * r * r
    return diag, off, B


def _solve_states(grid, p, q, idx, guess_vecs=None):
    """Eigenpairs number idx (0-based) of the pencil."""
    d, e, B = _pencil(grid, p, q)
    sb = np.sqrt(B)
    ds, es = d / B, e / (sb[:-1] * sb[1:])
    E = eigh_tridiagonal(ds, es, eigvals_only=True, select="i", select_range=(idx, idx), lapack_driver="stebz", tol=1e-300)[0]
    n = len(d)
    ab = np.zeros((3, n))
    ab[0, 1:] = e
    ab[1] = d - E * B
    ab[2, :-1] = e
    v = np.ones(n) if guess_vecs is None else guess_vecs.copy()
    for _ in range(4):                       # fixed-shift inverse iteration: the eigenvector nearest E
        v = solve_banded((1, 1), ab, B * v)
        v /= math.sqrt(np.sum(B * v * v))
    Av = d * v
    Av[1:] += e * v[:-1]
    Av[:-1] += e * v[1:]
    lam = float(v @ Av)
    return lam, v


def solve_radial(grid, V, l, k, rel=False, E0=None, dV=None):
    """k-th (0-based) state of angular momentum l. Returns (E, R) with R = u/r, int R^2 r^2 dr = 1."""
    r = grid.r
    cent = l * (l + 1) / (2 * r * r)
    if not rel:
        E, u = _solve_states(grid, np.ones_like(r), V + cent, k)
    else:
        E = E0 if E0 is not None else _solve_states(grid, np.ones_like(r), V + cent, k)[0]
        if dV is None:
            dV = grid.deriv(V)
        for _ in range(30):
            M = 1 + (E - V) / (2 * C_LIGHT ** 2)
            dM = -dV / (2 * C_LIGHT ** 2)
            q = V + cent / M - dM / (2 * M * M * r)
            En, u = _solve_states(grid, 1 / M, q, k)
            if abs(En - E) < 1e-11 * max(1, abs(E)):
                E = En
                break
            E = En
    u = np.sqrt(r) * u                      # solver vector is w(x); u = r^(1/2) w
    R = u / r
    R /= math.sqrt(grid.integrate(R * R * r * r))
    big = np.abs(u) > 1e-6 * np.abs(u).max()
    nodes = int(np.sum(np.diff(np.sign(u[big])) != 0))
    if nodes != k:
        raise RuntimeError(f"l={l} state {k}: {nodes} nodes")
    if R[np.argmax(np.abs(R * r))] < 0:
        R = -R
    return E, R


@dataclass
class AtomResult:
    Z: int
    config: dict
    energies: dict
    orbitals: dict
    total_energy: float
    grid: Grid
    converged: bool
    iterations: int
    history: list = field(default_factory=list)

    def r_mean(self, lab):
        R = self.orbitals[lab]
        return self.grid.integrate(R * R * self.grid.r ** 3)


def solve_atom(Z, config, func="pbe", rel=False, grid=None, mix=0.3, tol=1e-8, maxiter=300, r_gga=1e-2):
    """config: {'1s': 2, ..., '4p': 0.0} (every shell from n = l+1 must be listed); neutral or charged."""
    grid = grid or Grid()
    r = grid.r
    by_l = {}
    for lab, f in config.items():
        by_l.setdefault(LETTER.index(lab[-1]), []).append((int(lab[:-1]), lab, float(f)))
    for l in by_l:
        by_l[l].sort()
        if [n for n, _, _ in by_l[l]] != list(range(l + 1, l + 1 + len(by_l[l]))):
            raise ValueError(f"shells of l={l} must be consecutive from n={l + 1}")
    # Thomas-Fermi start
    a, b, g = 0.7280642371, -0.5430794693, 0.3612163121
    xx = r * (128.0 * Z / (9.0 * math.pi ** 2)) ** (1 / 3)
    sx = np.sqrt(xx)
    V = -np.maximum(Z * (1 + a * sx + b * xx * np.exp(-g * sx)) ** 2 * np.exp(-2 * a * sx), 1.0) / r
    hist, Eold, conv, prevE = [], 0.0, False, {}
    Vin_prev = Vout_prev = None
    # dV/dr for the scalar-relativistic term: nuclear + Hartree part only (analytic, smooth); the GGA xc potential
    # is noisy near the nucleus and its derivative would create spurious deep states
    dVrel = grid.deriv(V)
    for it in range(1, maxiter + 1):
        eig, orb, rho = {}, {}, np.zeros_like(r)
        for l, shells in by_l.items():
            for k, (n, lab, f) in enumerate(shells):
                E, R = solve_radial(grid, V, l, k, rel, prevE.get(lab), dVrel)
                eig[lab], orb[lab] = E, R
                rho += f * R * R / (4 * math.pi)
        prevE = dict(eig)
        q_in = grid.cumulative(4 * math.pi * rho * r * r)
        outer = grid.cumulative(4 * math.pi * rho * r)
        VH = q_in / r + (outer[-1] - outer)
        dVrel = (Z - q_in) / (r * r)
        # scalar-relativistic s densities diverge weakly at r -> 0: LDA inside ~1e-3 bohr
        exc, vxc = xc_potential(grid, rho, func, r0=r_gga if rel else 0.0)
        Vout = -Z / r + VH + vxc
        Etot = (sum(config[k] * eig[k] for k in config) - 0.5 * grid.integrate(VH * rho * 4 * math.pi * r * r)
                + grid.integrate((exc - vxc * rho) * 4 * math.pi * r * r))
        dV = Vout - V
        w = 4 * math.pi * rho * r * r
        err = math.sqrt(grid.integrate(dV ** 2 * w) / Z)
        hist.append((Etot, err))
        if abs(Etot - Eold) < tol and err < 1e-6:
            conv = True
            break
        Eold = Etot
        if Vin_prev is None:
            Vn = V + mix * dV
        else:
            ddv = dV - (Vout_prev - Vin_prev)
            den = grid.integrate(ddv * ddv * w)
            beta = min(max(grid.integrate(dV * ddv * w) / den if den > 0 else 0.0, -1.0), 1.0)
            Vbi = V - beta * (V - Vin_prev)
            Vbo = Vout - beta * (Vout - Vout_prev)
            Vn = Vbi + mix * (Vbo - Vbi)
        Vin_prev, Vout_prev = V, Vout
        V = Vn
    return AtomResult(Z, dict(config), eig, orb, Etot, grid, conv, it, hist)
