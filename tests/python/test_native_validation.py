"""Malformed native API/file inputs must raise typed errors before indexing or allocation."""
import numpy as np
import pytest

from prowl import _core


def problem():
    return dict(kfrac=np.zeros((1, 3)), weights=[1.0], T=[[np.eye(2, dtype=complex)]],
                S=[np.eye(2, dtype=complex)], eps=[[np.array([-1.0, 1.0])]], occ=[[np.array([1.0, 0.0])]],
                orb_offsets=[0, 1, 2], spin_factor=2.0, bonds=[(0, 1, (0, 0, 0), None)],
                grid=[-2.0, 2.0, 101, 0.05, 0, 0, -0.5])


@pytest.mark.parametrize("updates", [
    {"T": []}, {"kfrac": np.zeros((0, 3)), "weights": [], "S": [], "T": [[]]},
    {"T": [[]]}, {"eps": []}, {"occ": [[]]}, {"S": [np.eye(3, dtype=complex)]},
    {"eps": [[np.array([1.0])]]}, {"occ": [[np.array([1.0])]]},
    {"orb_offsets": []}, {"orb_offsets": [0, 3, 2]}, {"weights": [np.nan]},
    {"kfrac": np.zeros((1, 2))}, {"bonds": [(2, 0, (0, 0, 0), None)]},
    {"bonds": [(0, 1)]}, {"bonds": [(0, 1, (0, 0), None)]},
    {"bonds": [(0, 1, (0, 0, 0), np.ones((2, 2)))]},
    {"grid": [-2, 2, 0, .05, 0, 0, -.5]}, {"grid": [-2, 2, np.nan, .05, 0, 0, -.5]},
    {"grid": [-2, 2, 2.5, .05, 0, 0, -.5]}, {"grid": [-2, 2, 10, 0, 0, 0, -.5]},
    {"grid": [-2, 2, 10, .05, 0, 3, -.5]}, {"mesh": [1, 1]},
    {"grid": [-2, 2, 10, .05, 0, 1, -.5], "mesh": [0, 1, 1], "kgrid": np.zeros((1, 3), int)},
])
def test_analysis_rejects_invalid_dimensions(updates):
    args = problem()
    args.update(updates)
    with pytest.raises(_core.ProwlError, match="InvalidArgument"):
        _core.analyze(**args)


@pytest.mark.parametrize("energies", [[], [1], [1, 2, 3], [1, 2, 3, 4, 5], [3, 2, 1, 0], [0, 1, 2, np.nan]])
def test_tetrahedron_requires_four_sorted_finite_corners(energies):
    with pytest.raises(_core.ProwlError, match="InvalidArgument"):
        _core.tetra_weights(energies, 0.5, 1.0)


@pytest.mark.parametrize("args", [([], [], [], []), ([[]], [np.eye(2)], [[]], [[]]),
                                   ([[np.eye(2)]], [np.eye(3)], [[np.zeros(2)]], [[np.zeros(2)]])])
def test_repair_rejects_malformed_batches(args):
    with pytest.raises(_core.ProwlError, match="InvalidArgument"):
        _core.repaired_windows(*args, 2, 2, 1e-4, 1e-3)


@pytest.mark.parametrize("T,S,eps", [(np.zeros((0, 2)), np.eye(2), np.zeros(2)),
                                    (np.eye(2), np.eye(3), np.zeros(2)),
                                    (np.eye(2), np.eye(2), np.zeros(3)),
                                    (np.eye(2) * np.nan, np.eye(2), np.zeros(2))])
def test_reconstruction_rejects_malformed_arrays(T, S, eps):
    with pytest.raises(_core.ProwlError, match="InvalidArgument"):
        _core.reconstruct_k(T.astype(complex), S.astype(complex), eps)


@pytest.mark.parametrize("mode", ["", "   ", "Selective dynamics\n", "Selective dynamics\n   ", "Unknown"])
def test_poscar_requires_coordinate_mode(tmp_path, mode):
    path = tmp_path / "POSCAR"
    path.write_text("toy\n1\n1 0 0\n0 1 0\n0 0 1\nH\n1\n" + mode + "\n")
    with pytest.raises(_core.ProwlError, match="CorruptFile"):
        _core.read_poscar(str(path))


@pytest.mark.parametrize("count", ["-1", "nan", "1.5", "2147483648", "1x"])
def test_poscar_rejects_invalid_atom_counts(tmp_path, count):
    path = tmp_path / "POSCAR"
    path.write_text(f"toy\n1\n1 0 0\n0 1 0\n0 0 1\nH\n{count}\nDirect\n0 0 0\n")
    with pytest.raises(_core.ProwlError, match="CorruptFile"):
        _core.read_poscar(str(path))


@pytest.mark.parametrize("call", [lambda: _core.real_ylm(-1, [1, 0, 0]),
                                  lambda: _core.real_ylm(7, [1, 0, 0]),
                                  lambda: _core.sph_bessel(13, 1),
                                  lambda: _core.sph_bessel(1, -1),
                                  lambda: _core.gaunt(-1, 0, 0, 0, 0, 0),
                                  lambda: _core.gaunt(1, 2, 0, 0, 0, 0),
                                  lambda: _core.sto_bessel_integral(4, 16, 1., 1.)])
def test_scalar_kernel_bounds_are_checked(call):
    with pytest.raises(_core.ProwlError, match="InvalidArgument"):
        call()


def synthetic_potcar(path, ngrid, grid):
    block = " ".join(["0.1"] * 100) + "\n"
    path.write_text("PAW_PBE H\nVRHFIN =H: s1\nTITEL = PAW_PBE H\nZVAL = 1\n25.0 T\n"
                    "Non local Part\n0 1 1.0\n1.0\nReciprocal Space Part\n" + block +
                    "Real Space Part\n" + block + f"PAW radial sets\n{ngrid} 1.0\ngrid\n" +
                    " ".join(map(str, grid)) + "\npseudo wavefunction\n0.1 0.1 0.1 0.1\n"
                    "ae wavefunction\n0.2 0.2 0.2 0.2\nEnd of Dataset\n")


@pytest.mark.parametrize("count,grid", [(0, []), (1, [.1]), (2, [.1, .2]), (3, [.1, .2, .4]),
                                       (-1, [.1, .2, .4, .8]), ("nan", [.1, .2, .4, .8]),
                                       (4, [0, .2, .4, .8]), (4, [.1, .2, .1, .8]),
                                       (4, [.001, .002, .01, .02])])
def test_potcar_rejects_invalid_radial_grids(tmp_path, count, grid):
    path = tmp_path / "POTCAR"
    synthetic_potcar(path, count, grid)
    with pytest.raises(_core.ProwlError, match="CorruptFile|UnsupportedFeature"):
        _core.read_potcar(str(path))


def test_minimal_valid_potcar(tmp_path):
    path = tmp_path / "POTCAR"
    synthetic_potcar(path, 4, [.1, .2, .4, .8])
    assert len(_core.read_potcar(str(path))) == 1
