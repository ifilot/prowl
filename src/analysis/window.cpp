// SPDX-License-Identifier: GPL-3.0-or-later
#include "analysis/numerics.hpp"
#include "prowl/analysis.hpp"
#include "prowl/backend.hpp"
#include "prowl/errors.hpp"
#include "prowl/threading.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace prowl {
using detail::exceeds;
using detail::heev;
using detail::heevr_top;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
using Guard = detail::ExceptionGuard;
namespace {
// Band energies in ascending order up to round-off: VASP writes degenerate states with inversions of
// ~1e-8 eV (si216_gamma_gam: four pairs), which the window searches treat as degenerate anyway.
bool sorted_band_energies(const VectorXd& eps) {
    constexpr double inversion_tol = 1e-6; // eV
    for (Eigen::Index i = 1; i < eps.size(); ++i)
        if (eps(i) < eps(i - 1) - inversion_tol)
            return false;
    return true;
}
double window_conditioning_x(const MatrixXcd& T, const MatrixXcd& X, int nwin) {
    require(nwin >= 0 && nwin <= T.cols(), ErrorCode::InvalidArgument,
            "band window of " + std::to_string(nwin) + " exceeds the " + std::to_string(T.cols()) + " available bands");
    const MatrixXcd Tp = X * T.leftCols(nwin);
    MatrixXcd O = Tp * Tp.adjoint();
    VectorXd w;
    heev(O, w, false);
    return w(0);
}
FrozenWindow frozen_window_x(const MatrixXcd& T0, const MatrixXcd& X, const VectorXd& eps0, const VectorXd& occ0,
                             int norb, int ncand, double occ_tol);
} // namespace

double window_conditioning(const MatrixXcd& T, const MatrixXcd& S, int nwin, double overlap_threshold) {
    BlasThreads policy;
    detail::validate_projection(T, S);
    return window_conditioning_x(T, detail::factor_overlap(S, overlap_threshold).inverse_sqrt, nwin);
}

int cut_at_gap(const VectorXd& eps, int n, double tol) {
    require(n >= 0 && std::isfinite(tol) && tol >= 0 && eps.allFinite(), ErrorCode::InvalidArgument,
            "invalid band cut or degeneracy tolerance");
    n = std::min<int>(n, static_cast<int>(eps.size()));
    while (n > 0 && n < eps.size() && eps(n) - eps(n - 1) < tol)
        --n;
    return n;
}

FrozenWindow frozen_window_subspace(const MatrixXcd& T0, const MatrixXcd& S, const VectorXd& eps0, const VectorXd& occ0,
                                    int norb, int ncand, double occ_tol, double overlap_threshold) {
    BlasThreads policy;
    detail::validate_projection(T0, S, &eps0, &occ0);
    require(norb == T0.rows() && ncand >= -1 && ncand <= T0.cols() && std::isfinite(occ_tol) && occ_tol >= 0,
            ErrorCode::InvalidArgument, "invalid frozen-window dimensions or occupation tolerance");
    require(sorted_band_energies(eps0), ErrorCode::InvalidArgument,
            "window energies must be sorted");
    return frozen_window_x(T0, detail::factor_overlap(S, overlap_threshold).inverse_sqrt, eps0, occ0, norb, ncand,
                           occ_tol);
}

namespace {
FrozenWindow frozen_window_x(const MatrixXcd& T0, const MatrixXcd& X, const VectorXd& eps0, const VectorXd& occ0,
                             int norb, int ncand, double occ_tol) {
    Eigen::Index nb = T0.cols();
    if (ncand >= 0) nb = cut_at_gap(eps0, ncand);
    const MatrixXcd T = T0.leftCols(nb);
    const VectorXd eps = eps0.head(nb), occ = occ0.head(nb);
    int nf = 0;
    for (Eigen::Index n = 0; n < nb; ++n)
        if (std::abs(occ(n)) > occ_tol) { // negative (Methfessel-Paxton) occupations are frozen too
            require(n == nf, ErrorCode::NumericalFailure, "occupied bands are not the lowest bands");
            ++nf;
        }
    for (Eigen::Index n = nf; n < occ0.size(); ++n)
        require(std::abs(occ0(n)) <= occ_tol || n < nb, ErrorCode::InvalidArgument,
                "occupied bands beyond the candidate window; increase NBANDS");
    require(nf <= norb, ErrorCode::InvalidArgument,
            std::to_string(nf) + " occupied bands exceed the " + std::to_string(norb) +
                " local basis functions; enlarge the basis");
    const int m = norb - nf;
    require(nb - nf >= m, ErrorCode::InvalidArgument, "not enough bands to complete the subspace; increase NBANDS");
    FrozenWindow out;
    const MatrixXcd TpU = X * T.rightCols(nb - nf);
    MatrixXcd Mu = TpU.adjoint() * TpU;
    VectorXd w; // ascending
    heev(Mu, w, true);
    const Eigen::Index nu = w.size();
    MatrixXcd V(nu, m);
    for (int j = 0; j < m; ++j)
        V.col(j) = Mu.col(nu - 1 - j); // largest weight first
    const VectorXd epsU = eps.tail(nb - nf);
    MatrixXcd Hm = V.adjoint() * epsU.asDiagonal() * V;
    Hm = 0.5 * (Hm + Hm.adjoint()).eval();
    VectorXd e_m;
    heev(Hm, e_m, true);
    V = V * Hm;
    out.T.resize(T.rows(), norb);
    out.T.leftCols(nf) = T.leftCols(nf);
    out.T.rightCols(m) = T.rightCols(nb - nf) * V;
    out.eps.resize(norb);
    out.eps.head(nf) = eps.head(nf);
    out.eps.tail(m) = e_m;
    out.occ = VectorXd::Zero(norb);
    out.occ.head(nf) = occ.head(nf);
    if (m > 0) {
        out.weakest = w(nu - m);
        out.gap = m < nu ? (w(nu - m) - w(nu - m - 1)) / std::max(w(nu - m), 1e-300) : 1.0;
    }
    return out;
}

int next_gap(const VectorXd& e, int n, double tol) {
    while (n > 0 && n < e.size() && e(n) - e(n - 1) < tol)
        ++n;
    return n;
}
} // namespace

namespace {
RepairedWindow repaired_window_reference_x(const MatrixXcd& T, const MatrixXcd& X, const VectorXd& eps,
                                           const VectorXd& occ, int norb, int ncand, double cond_threshold,
                                           double gap_threshold, double gap_tol) {
    // Same logic as the Python loop it replaces (calculation._repaired_window), with S^-1/2 computed once.
    RepairedWindow r;
    if (cut_at_gap(eps, norb, gap_tol) == norb) {
        const double c = window_conditioning_x(T, X, norb);
        if (c >= cond_threshold) {
            r.T = T.leftCols(norb);
            r.eps = eps.head(norb);
            r.occ = occ.head(norb);
            r.cond = c;
            return r;
        }
    }
    int nc = next_gap(eps, norb + 1, gap_tol);
    while (true) {
        if (nc > ncand) {
            r.failed_at = nc;
            return r;
        }
        auto fw = frozen_window_x(T, X, eps, occ, norb, nc, 1e-8);
        const double c = window_conditioning_x(fw.T, X, norb);
        if (c >= cond_threshold && fw.gap >= gap_threshold) {
            r.T = std::move(fw.T);
            r.eps = std::move(fw.eps);
            r.occ = std::move(fw.occ);
            r.cond = c;
            r.repaired_nc = nc;
            return r;
        }
        nc = next_gap(eps, nc + 1, gap_tol);
    }
}

RepairedWindow repaired_window_x(const MatrixXcd& T, const MatrixXcd& X, const VectorXd& eps, const VectorXd& occ,
                                 int norb, int ncand, double cond_threshold, double gap_threshold, double gap_tol,
                                 Backend* device) {
    // Margins: a step is rejected without the reference predicate only if it fails a threshold by more than any
    // rounding difference between the two formulations (eigenvalues of matrices with norm ~1).
    constexpr double cond_margin = 1e-10, gap_margin = 1e-8, occ_tol = 1e-8;
    const auto t_setup = std::chrono::steady_clock::now();
    if (cut_at_gap(eps, norb, gap_tol) == norb) {
        const double c = window_conditioning_x(T, X, norb);
        if (c >= cond_threshold) { // LOBSTER's lowest-norb window, unchanged
            RepairedWindow r;
            r.T = T.leftCols(norb);
            r.eps = eps.head(norb);
            r.occ = occ.head(norb);
            r.cond = c;
            return r;
        }
    }
    std::vector<int> ncs; // candidate band counts in scan order (cuts at gaps)
    int nc = next_gap(eps, norb + 1, gap_tol);
    while (nc <= ncand) {
        ncs.push_back(nc);
        nc = next_gap(eps, nc + 1, gap_tol);
    }
    const int failed_at = nc;
    // Frozen (occupied) set as frozen_window_x determines it; unusual inputs go to the reference scan, which reports
    // them with its own errors.
    int nf = 0;
    bool regular = !ncs.empty() && ncs.back() <= T.cols();
    for (Eigen::Index n = 0; regular && n < occ.size(); ++n)
        if (std::abs(occ(n)) > occ_tol) {
            if (n != nf || n >= ncs.front()) regular = false;
            ++nf;
        }
    const int m = norb - nf;
    if (!regular || m <= 0)
        return repaired_window_reference_x(T, X, eps, occ, norb, ncand, cond_threshold, gap_threshold, gap_tol);

    RepairedWindow r;
    using clock = std::chrono::steady_clock;
    auto since = [](clock::time_point t0) { return std::chrono::duration<double>(clock::now() - t0).count(); };
    auto reference_step = [&](int ncur) {
        const auto t0 = clock::now();
        struct Add {
            double& acc;
            clock::time_point t0;
            ~Add() { acc += std::chrono::duration<double>(clock::now() - t0).count(); }
        } add{r.seconds_reference, t0};
        ++r.steps_reference;
        auto fw = frozen_window_x(T, X, eps, occ, norb, ncur, occ_tol);
        const double c = window_conditioning_x(fw.T, X, norb);
        if (c < cond_threshold || fw.gap < gap_threshold) return false;
        r.T = std::move(fw.T);
        r.eps = std::move(fw.eps);
        r.occ = std::move(fw.occ);
        r.cond = c;
        r.repaired_nc = ncur;
        return true;
    };
    // Most repairs succeed at the first candidate: try it as the reference does before projecting further bands.
    if (reference_step(ncs.front())) return r;
    const int nlast = static_cast<int>(ncs.size()) - 1;
    MatrixXcd Tp; // X T for the lowest bands, extended on demand (doubling)
    auto need = [&](int ncols) {
        if (Tp.cols() >= ncols) return;
        const int target = std::min<int>(ncs.back(), std::max<int>(ncols, 2 * static_cast<int>(Tp.cols())));
        MatrixXcd Tn(T.rows(), target);
        if (Tp.cols()) Tn.leftCols(Tp.cols()) = Tp;
        Tn.rightCols(target - Tp.cols()).noalias() = X * T.middleCols(Tp.cols(), target - Tp.cols());
        Tp = std::move(Tn);
    };
    // bound(i): min eig(Tp Tp^H) over the lowest ncs[i] bands exceeds the threshold; no window built from those bands
    // is better conditioned than all of them, so a false bound rules out every step up to i (Cholesky test).
    // Tp Tp^H only grows with the band count: it is kept for the highest failing index (base) and each probe adds
    // the bands between (rank-k update of the lower triangle), so the whole search costs about two full products.
    MatrixXcd base = MatrixXcd::Zero(norb, norb);
    int base_nc = 0;
    auto bound = [&](int i) {
        need(ncs[i]);
        MatrixXcd C = base;
        if (ncs[i] > base_nc) C.selfadjointView<Eigen::Lower>().rankUpdate(Tp.middleCols(base_nc, ncs[i] - base_nc));
        const bool ok = exceeds(C, cond_threshold - cond_margin);
        if (!ok && ncs[i] > base_nc) {
            base = std::move(C);
            base_nc = ncs[i];
        }
        return ok;
    };
    // First index with a true bound: exponential search from index 1, then bisection (the bound is monotone).
    const auto t_bound = clock::now();
    r.seconds_setup = std::chrono::duration<double>(t_bound - t_setup).count() - r.seconds_reference;
    int lo = 1, hi = 1;
    while (true) {
        if (hi >= nlast) {
            hi = nlast;
            if (nlast < 1 || !bound(nlast)) { // even all reliable bands fail: no repair possible
                r.failed_at = failed_at;
                r.seconds_bound = since(t_bound);
                return r;
            }
            break;
        }
        if (bound(hi)) break;
        lo = hi + 1;
        hi *= 2;
    }
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (bound(mid))
            hi = mid;
        else
            lo = mid + 1;
    }
    need(ncs[lo]);
    r.seconds_bound = since(t_bound);
    const auto t_scan = clock::now();
    const double ref_before_scan = r.seconds_reference;
    const MatrixXcd PF = Tp.leftCols(nf) * Tp.leftCols(nf).adjoint();
    // One scan step: false if the step fails a threshold by more than the margin (rejected without the reference).
    // Mu = A^H A (u <= norb) or G = A A^H (u > norb) of the unoccupied candidates A = Tp[:, nf:nc], grown in place
    // across consecutive steps (st_nc = band count they hold, nf = empty).
    struct State {
        MatrixXcd Mu, G;
        int mu_nc = 0, g_nc = 0;
    };
    auto passes = [&](int ncur, State& st) {
        const int u = ncur - nf;
        double gap = 1.0;
        VectorXd w;
        MatrixXcd Z, B;
        if (u <= norb) { // eigenproblem of A^H A (u x u), as frozen_window_x; top m + 1 eigenpairs only
            const auto A = Tp.middleCols(nf, u);
            if (st.mu_nc <= nf || st.mu_nc > ncur) {
                st.Mu = A.adjoint() * A;
            } else if (st.mu_nc < ncur) {
                const int uo = st.mu_nc - nf, du = ncur - st.mu_nc;
                const auto An = Tp.middleCols(st.mu_nc, du);
                st.Mu.conservativeResize(u, u);
                st.Mu.block(0, uo, uo, du).noalias() = Tp.middleCols(nf, uo).adjoint() * An;
                st.Mu.block(uo, 0, du, uo) = st.Mu.block(0, uo, uo, du).adjoint();
                st.Mu.block(uo, uo, du, du).noalias() = An.adjoint() * An;
            }
            st.mu_nc = ncur;
            MatrixXcd H = st.Mu;
            heevr_top(H, m < u ? m + 1 : m, w, Z);
            if (m < u) gap = (w(1) - w(0)) / std::max(w(1), 1e-300);
            if (gap < gap_threshold - gap_margin) return false;
            B.noalias() = A * Z.rightCols(m); // projected top-m completion
        } else { // same nonzero spectrum from A A^H (norb x norb), updated by the new band groups
            if (st.g_nc <= nf || st.g_nc > ncur) {
                const auto A = Tp.middleCols(nf, u);
                st.G = A * A.adjoint();
            } else if (st.g_nc < ncur) {
                const auto A = Tp.middleCols(st.g_nc, ncur - st.g_nc);
                st.G.noalias() += A * A.adjoint();
            }
            st.g_nc = ncur;
            MatrixXcd H = st.G;
            if (m < norb) { // the (m+1)-th largest eigenvalue of A A^H is that of A^H A (u > norb > m)
                heevr_top(H, m + 1, w, Z);
                gap = (w(1) - w(0)) / std::max(w(1), 1e-300);
            } else { // nothing occupied: the (m+1)-th eigenvalue of A^H A is zero, gap 1 as in frozen_window_x
                heevr_top(H, m, w, Z);
                gap = 1.0;
            }
            if (gap < gap_threshold - gap_margin) return false;
            B = Z.rightCols(m) * w.tail(m).cwiseMax(0.0).cwiseSqrt().asDiagonal();
        }
        MatrixXcd M = PF;
        M.noalias() += B * B.adjoint();
        return exceeds(M, cond_threshold - cond_margin); // false: condition fails by more than the margin
    };
    // Clear accept (u > norb): both tests pass by more than the margin, so the reference predicate cannot reject;
    // the window is then built from the norb x norb problem: with G = A A^H = Q S Q^H, W = A^H Q_m S_m^(-1/2) are the
    // top-m eigenvectors of A^H A (the selection of frozen_window_x), followed by the same Rayleigh-Ritz step.
    // Values agree with frozen_window_x to rounding; near a threshold (or u <= norb) the reference path is used.
    auto fast_accept = [&](int ncur) {
        const int u = ncur - nf;
        if (u <= norb || m >= norb) return false;
        const auto A = Tp.middleCols(nf, u);
        MatrixXcd G = MatrixXcd::Zero(norb, norb);
        G.selfadjointView<Eigen::Lower>().rankUpdate(A);
        VectorXd w;
        MatrixXcd Z;
        heevr_top(G, m + 1, w, Z);
        if ((w(1) - w(0)) / std::max(w(1), 1e-300) < gap_threshold + gap_margin || w(1) <= 0.0) return false;
        const VectorXd s = w.tail(m);
        const MatrixXcd Q = Z.rightCols(m);
        MatrixXcd M = PF;
        const MatrixXcd B = Q * s.cwiseSqrt().asDiagonal();
        M.noalias() += B * B.adjoint();
        if (!exceeds(M, cond_threshold + cond_margin)) return false;
        const MatrixXcd W = A.adjoint() * (Q * s.cwiseSqrt().cwiseInverse().asDiagonal()); // u x m, orthonormal
        const VectorXd epsU = eps.segment(nf, u);
        MatrixXcd Hm = W.adjoint() * epsU.asDiagonal() * W;
        Hm = 0.5 * (Hm + Hm.adjoint()).eval();
        VectorXd e_m;
        heev(Hm, e_m, true);
        MatrixXcd TB(T.rows(), norb);
        TB.leftCols(nf) = T.leftCols(nf);
        TB.rightCols(m).noalias() = T.middleCols(nf, u) * (W * Hm);
        const double c = window_conditioning_x(TB, X, norb);
        if (c < cond_threshold) return false;
        r.T = std::move(TB);
        r.eps.resize(norb);
        r.eps.head(nf) = eps.head(nf);
        r.eps.tail(m) = e_m;
        r.occ = VectorXd::Zero(norb);
        r.occ.head(nf) = occ.head(nf);
        r.cond = c;
        r.repaired_nc = ncur;
        return true;
    };
    // Steps are independent. Without an enclosing parallel region (one (spin, k) at a time) all threads take chunks
    // of consecutive steps from a queue in index order; a thread stops at its first passing step, and nobody starts
    // a chunk beyond the lowest passing step found so far, so every lower step has been evaluated when the queue is
    // empty. The lowest passing step goes to the reference predicate; if it rejects, the scan resumes after it.
    need(ncs.back()); // workers read Tp concurrently: no growth inside the parallel region
    std::unique_ptr<RepairStepper> stepper;
    // A device stepper (own stream and handles, Tp upload) pays off only for long scans; short ones (k meshes with
    // a few extra bands per (spin, k)) stay on the CPU.
    constexpr int min_device_steps = 16;
    if (device && nlast - lo + 1 >= min_device_steps)
        stepper = device->make_repair_stepper(Tp.data(), norb, static_cast<int>(Tp.cols()), nf, m,
                                              gap_threshold - gap_margin, cond_threshold - cond_margin);
        // With a device stepper the scan runs on it alone (CPU threads are slower per step and would hold up the end
        // of a pass); the caller runs several (spin, k) concurrently, each with its own stepper and stream.
#ifdef _OPENMP
    const int nthreads = (omp_in_parallel() || stepper) ? 1 : std::max(1, omp_get_max_threads());
#else
    const int nthreads = 1;
#endif
    constexpr int cpu_chunk = 4, gpu_chunk = 16;
    int i = lo;
    while (i <= nlast) {
        std::atomic<int> next{i}, best{std::numeric_limits<int>::max()}, scanned{0};
        Guard guard;
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
        guard.run([&] {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            const bool on_device = stepper && tid == 0;
            const int chunk = on_device ? gpu_chunk : cpu_chunk;
            State st;
            while (true) {
                const int s0 = next.fetch_add(chunk);
                if (s0 > nlast || s0 > best.load()) break;
                const int s1 = std::min(nlast, s0 + chunk - 1);
                for (int j = s0; j <= s1 && j < best.load(); ++j) {
                    scanned.fetch_add(1);
                    if (on_device ? stepper->passes(ncs[j]) : passes(ncs[j], st)) {
                        int b = best.load();
                        while (j < b && !best.compare_exchange_weak(b, j)) {
                        }
                        break;
                    }
                }
            }
        });
        guard.rethrow();
        r.steps_scanned += scanned.load();
        const int first = best.load();
        if (first == std::numeric_limits<int>::max()) break;
        if (fast_accept(ncs[first]) || reference_step(ncs[first])) {
            r.seconds_scan = since(t_scan) - (r.seconds_reference - ref_before_scan);
            return r;
        }
        i = first + 1;
    }
    r.seconds_scan = since(t_scan) - (r.seconds_reference - ref_before_scan);
    r.failed_at = failed_at;
    return r;
}

// Validate once at the boundary, before either search touches band slices or Eigen expressions.
void validate_window(const MatrixXcd& T, const MatrixXcd& S, const VectorXd& eps, const VectorXd& occ, int norb,
                     int ncand, double cond, double gap, double tol) {
    detail::validate_projection(T, S, &eps, &occ);
    require(norb == T.rows() && T.cols() >= norb && ncand >= 0 && ncand <= T.cols(), ErrorCode::InvalidArgument,
            "invalid repair-window dimensions");
    require(std::isfinite(cond) && cond >= 0 && std::isfinite(gap) && gap >= 0 && std::isfinite(tol) && tol >= 0,
            ErrorCode::InvalidArgument, "invalid repair thresholds");
    require(sorted_band_energies(eps), ErrorCode::InvalidArgument,
            "window energies must be sorted");
}
} // namespace

RepairedWindow repaired_window_reference(const MatrixXcd& T, const MatrixXcd& S, const VectorXd& eps,
                                         const VectorXd& occ, int norb, int ncand, double cond_threshold,
                                         double gap_threshold, double gap_tol) {
    BlasThreads policy;
    validate_window(T, S, eps, occ, norb, ncand, cond_threshold, gap_threshold, gap_tol);
    const auto overlap = detail::factor_overlap(S, 1e-8);
    return repaired_window_reference_x(T, overlap.inverse_sqrt, eps, occ, norb, ncand, cond_threshold, gap_threshold,
                                       gap_tol);
}

RepairedWindow repaired_window(const MatrixXcd& T, const MatrixXcd& S, const VectorXd& eps, const VectorXd& occ,
                               int norb, int ncand, double cond_threshold, double gap_threshold, double gap_tol,
                               Backend* device) {
    BlasThreads policy;
    validate_window(T, S, eps, occ, norb, ncand, cond_threshold, gap_threshold, gap_tol);
    const auto overlap = detail::factor_overlap(S, 1e-8);
    return repaired_window_x(T, overlap.inverse_sqrt, eps, occ, norb, ncand, cond_threshold, gap_threshold, gap_tol,
                             device);
}

std::vector<RepairedWindow>
repaired_windows(const std::vector<std::vector<MatrixXcd>>& T, const std::vector<MatrixXcd>& S,
                 const std::vector<std::vector<VectorXd>>& eps, const std::vector<std::vector<VectorXd>>& occ, int norb,
                 int ncand, double cond_threshold, double gap_threshold, bool reference, Backend* device) {
    BlasThreads policy;
    const std::size_t ns = T.size(), nk = S.size();
    require(ns > 0 && nk > 0 && eps.size() == ns && occ.size() == ns, ErrorCode::InvalidArgument,
            "repair arrays must contain matching spin and k dimensions");
    require(ns <= static_cast<std::size_t>(std::numeric_limits<int>::max()) / nk, ErrorCode::InvalidArgument,
            "too many spin/k entries");
    for (std::size_t spin = 0; spin < ns; ++spin) {
        require(T[spin].size() == nk && eps[spin].size() == nk && occ[spin].size() == nk, ErrorCode::InvalidArgument,
                "repair arrays have inconsistent k dimensions");
        for (std::size_t k = 0; k < nk; ++k)
            validate_window(T[spin][k], S[k], eps[spin][k], occ[spin][k], norb, ncand, cond_threshold, gap_threshold,
                            1e-4);
    }
    std::vector<detail::OverlapFactorization> overlap(nk);
    Guard guard;
#pragma omp parallel for schedule(dynamic) if (nk > 1)
    for (std::size_t k = 0; k < nk; ++k)
        guard.run([&] { overlap[k] = detail::factor_overlap(S[k], 1e-8); });
    guard.rethrow();
    std::vector<RepairedWindow> result(ns * nk);
    const int nsk = static_cast<int>(ns * nk);
#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
#else
    const int nthr = 1;
#endif
    const bool outer = device ? nsk > 1 : (nthr > 1 && nsk >= nthr);
#pragma omp parallel for schedule(dynamic) num_threads(std::max(1, std::min(nsk, nthr))) if (outer)
    for (std::size_t i = 0; i < ns * nk; ++i)
        guard.run([&] {
            const auto spin = i / nk, k = i % nk;
            result[i] =
                reference ? repaired_window_reference_x(T[spin][k], overlap[k].inverse_sqrt, eps[spin][k], occ[spin][k],
                                                        norb, ncand, cond_threshold, gap_threshold, 1e-4)
                          : repaired_window_x(T[spin][k], overlap[k].inverse_sqrt, eps[spin][k], occ[spin][k], norb,
                                              ncand, cond_threshold, gap_threshold, 1e-4, device);
        });
    guard.rethrow();
    return result;
}

} // namespace prowl
