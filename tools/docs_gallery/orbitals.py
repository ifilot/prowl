"""Real-space Kohn-Sham orbitals from a Gamma-point VASP WAVECAR.

The pseudo-wavefunction of band n is psi(r) = Omega^(-1/2) sum_G c_G exp(i G.r).
It is evaluated on a uniform grid by FFT, made real (a molecule at Gamma has
real orbitals up to a global phase), and degenerate sets are rotated so that
their members align with the Cartesian axes. Only the smooth plane-wave part is
used: inside the PAW spheres the true orbital has extra core nodes, which do not
change the shape of a valence orbital at a typical isovalue.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from prowl import _core


class GammaOrbitals:
    def __init__(self, wavecar: str, n_grid: int = 128):
        self.reader = _core.WavecarReader(wavecar)
        h = self.reader.header
        self.lattice = np.asarray(h["lattice"], float)
        if not np.allclose(self.lattice, np.diag(np.diag(self.lattice))):
            raise ValueError("orthorhombic boxes only")
        if h["nkpts"] != 1 or h["nspin"] != 1:
            raise ValueError("expects a single Gamma point and ISPIN = 1")
        self.L = np.diag(self.lattice)
        self.volume = float(h["volume"])
        self.n = n_grid
        k = self.reader.kpoint(0, 0)
        self.eigenvalues = np.asarray(k["eigenvalues"])
        self.occupations = np.asarray(k["occupations"])
        self.efermi = float(h["efermi_ev"])

    def complex_field(self, band: int) -> np.ndarray:
        """psi[ix, iy, iz] on r = (i/N) * L, in Angstrom^-3/2."""
        b = self.reader.read_block(0, 0, band, 1)
        G = np.asarray(b["gvectors"], int)
        c = np.asarray(b["coefficients"])[0]
        n = self.n
        if np.abs(G).max() >= n // 2:
            raise ValueError("grid too coarse for the plane-wave cutoff")
        grid = np.zeros((n, n, n), complex)
        grid[G[:, 0] % n, G[:, 1] % n, G[:, 2] % n] = c
        return np.fft.ifftn(grid) * n**3 / np.sqrt(self.volume)

    def real_orbitals(self, bands: list[int]) -> list[np.ndarray]:
        """Real, orthonormal orbitals spanning the given (degenerate) bands."""
        fields = [self.complex_field(b) for b in bands]
        parts = []
        for f in fields:
            parts += [f.real.ravel(), f.imag.ravel()]
        M = np.array(parts)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        real = Vt[: len(bands)] * s[: len(bands), None] / np.sqrt(2.0)
        # Rotate the real set so each member points along one Cartesian axis.
        if len(bands) > 1:
            r = self._coords()
            centre = self.centre
            D = np.array([[np.sum(v * (r[a] - centre[a]).ravel()) for a in range(3)] for v in real])
            # Keep the axes in which the set has the largest first moments.
            axes = np.argsort(-np.linalg.norm(D, axis=0))[: len(bands)]
            Q, _, Pt = np.linalg.svd(D[:, axes])
            R = (Q @ Pt).T
            real = R @ real
            self.last_axes = [int(x) for x in axes]  # member i points along Cartesian axis axes[i]
        else:
            self.last_axes = [None]
        out = []
        dv = self.volume / self.n**3
        for v in real:
            v = v / np.sqrt(np.sum(v * v) * dv)
            out.append(v.reshape(self.n, self.n, self.n))
        return out

    def _coords(self):
        axes = [np.arange(self.n) * self.L[a] / self.n for a in range(3)]
        return np.meshgrid(*axes, indexing="ij")

    centre = None  # set by the caller: molecular centre in Angstrom

    def gradient(self, field: np.ndarray) -> list[np.ndarray]:
        """Spectral gradient of a periodic real field."""
        n = self.n
        F = np.fft.fftn(field)
        freqs = [2j * np.pi * np.fft.fftfreq(n, d=self.L[a] / n) for a in range(3)]
        out = []
        for a in range(3):
            shape = [1, 1, 1]
            shape[a] = n
            out.append(np.real(np.fft.ifftn(F * freqs[a].reshape(shape))))
        return out


def tessellate(tess, orb: GammaOrbitals, field: np.ndarray, iso: float, sign: int):
    """Marching cubes on sign*field at +iso; vertices centred on the box centre."""
    n = orb.n
    cell = np.diag(orb.L).ravel()
    # PyTessel wants x fastest and z slowest.
    v, normals, idx = tess.marching_cubes((sign * field).transpose(2, 1, 0).ravel(), (n, n, n), cell, iso)
    if not len(idx):
        return None
    grads = orb.gradient(field)
    # Fractional grid coordinates of the vertices (vertex = r - L/2).
    pos = (v + orb.L / 2) / orb.L * n
    g = np.stack([map_coordinates(gc, pos.T, order=1, mode="wrap") for gc in grads], axis=1)
    norm = np.linalg.norm(g, axis=1)
    good = norm > 1e-12
    normals = np.where(good[:, None], -sign * g / np.where(good, norm, 1)[:, None], normals)
    return v, normals, idx


def check_tessellation_convention(tess, n: int = 64, L: float = 12.0):
    """An asymmetric plane catches axis swaps, shifts and N vs N-1 errors."""
    x = np.arange(n) * L / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    for a, A in enumerate((X, Y, Z)):
        field = A - 7.31
        v, _, _ = tess.marching_cubes(field.transpose(2, 1, 0).ravel(), (n, n, n), np.diag([L] * 3).ravel(), 0.0)
        if not np.allclose(v[:, a], 7.31 - L / 2, atol=1e-5):
            raise RuntimeError(f"PyTessel convention changed along axis {a}")
