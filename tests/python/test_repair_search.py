"""The fast band-window repair search returns the window of the original linear scan (SCIENCE.md 5.3).

Synthetic (spin, k) problems whose lowest-N_orb window lacks weight on a few local directions, so the repair has
to add bands until they are spanned. Covers: no repair needed, short and long repairs (both the A^H A and the
A A^H branch), degenerate band groups, failure within the reliable bands, and the CUDA repair stepper when present.
"""
import numpy as np
import pytest

from prowl import _core


def _problem(seed, norb=40, nb=160, nocc=24, weak=3, damp=1e-3, recover_at=None, degenerate=False):
    rng = np.random.default_rng(seed)
    T = (rng.standard_normal((norb, nb)) + 1j * rng.standard_normal((norb, nb))) / np.sqrt(nb)
    # local directions with almost no weight in the low bands; their weight returns from band `recover_at` on
    V, _ = np.linalg.qr(rng.standard_normal((norb, weak)) + 1j * rng.standard_normal((norb, weak)))
    P = V @ V.conj().T
    stop = recover_at if recover_at is not None else nb
    T[:, :stop] -= (1.0 - damp) * (P @ T[:, :stop])
    A = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    S = np.eye(norb) + 0.05 * (A + A.conj().T) / np.sqrt(norb)
    eps = np.sort(rng.uniform(-10.0, 25.0, nb))
    if degenerate:  # pairs of degenerate bands above the occupied ones
        for i in range(nocc + 2, nb - 1, 7):
            eps[i + 1] = eps[i]
        eps = np.sort(eps)
    occ = np.zeros(nb)
    occ[:nocc] = 1.0
    return T, S, eps, occ, norb


def _run(T, S, eps, occ, norb, ncand, reference, device=None):
    res = _core.repaired_windows([[T]], [S], [[eps]], [[occ]], norb, ncand, 1e-4, 1e-3, reference=reference,
                                 device=device)
    tw, ew, ow, c, nc, failed = res[0]
    return tw, ew, ow, c, nc, failed


CASES = [
    dict(seed=1, recover_at=None, damp=1.0),          # nothing damped: lowest-N_orb window is fine
    dict(seed=2, recover_at=60),                      # short repair, u <= norb
    dict(seed=3, recover_at=110),                     # long scan in the A A^H branch that fails
    dict(seed=4, recover_at=90, degenerate=True),     # degenerate groups at the cuts
    dict(seed=5, recover_at=10_000),                  # never recovers: repair fails
    dict(seed=6, recover_at=130, nocc=8, weak=6),     # few occupied bands, many weak directions
    dict(seed=10, recover_at=95, damp=3e-2),          # long successful scan (>= 16 steps: CUDA stepper)
]


def _devices():
    out = [None]
    if "cuda" in _core.available_backends():
        out.append(_core.make_backend("cuda"))
    return out


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("device", _devices(), ids=lambda d: "cpu" if d is None else "cuda")
def test_fast_search_matches_linear_scan(case, device):
    kw = {k: v for k, v in case.items()}
    T, S, eps, occ, norb = _problem(**kw)
    ncand = T.shape[1] - 10
    ref = _run(T, S, eps, occ, norb, ncand, reference=True)
    new = _run(T, S, eps, occ, norb, ncand, reference=False, device=device)
    assert new[4] == ref[4], "repaired band count differs"
    assert new[5] == ref[5], "failure point differs"
    if not ref[5]:
        np.testing.assert_allclose(new[1], ref[1], atol=1e-10)
        np.testing.assert_allclose(new[2], ref[2], atol=0)
        # the window is defined up to rotations among degenerate states: compare projectors
        np.testing.assert_allclose(new[0] @ new[0].conj().T, ref[0] @ ref[0].conj().T, atol=1e-10)
        assert abs(new[3] - ref[3]) < 1e-10


def test_cases_exercise_repairs():
    """The synthetic set contains an unrepaired window, short and long repairs, and failures."""
    seen = set()
    for case in CASES:
        T, S, eps, occ, norb = _problem(**case)
        r = _core.repaired_windows([[T]], [S], [[eps]], [[occ]], norb, T.shape[1] - 10, 1e-4, 1e-3,
                                   diagnostics=True)[0]
        nc, failed, steps = r[4], r[5], r[6]
        seen.add("failed" if failed else ("long repair" if steps >= 16 else "repaired") if nc else "norb")
    assert seen == {"norb", "repaired", "long repair", "failed"}
