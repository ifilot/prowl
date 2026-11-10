// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/analysis.hpp"
#include "analysis/numerics.hpp"
#include "prowl/backend.hpp"
#include "prowl/constants.hpp"
#include "prowl/errors.hpp"
#include "prowl/threading.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace prowl {
using cd = std::complex<double>;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using detail::heev;
using detail::heevr_top;
using detail::exceeds;
using Guard = detail::ExceptionGuard;
void tetra_weights(const double e[4], double E, double vt, double w[4]) {
    // Blöchl, Jepsen, Andersen, PRB 49, 16223 (1994), integration weights without correction term.
    const double e1 = e[0], e2 = e[1], e3 = e[2], e4 = e[3];
    const double q = vt / 4.0;
    if (E <= e1) { w[0] = w[1] = w[2] = w[3] = 0.0; return; }
    if (E >= e4) { w[0] = w[1] = w[2] = w[3] = q; return; }
    const double e21 = e2 - e1, e31 = e3 - e1, e41 = e4 - e1, e32 = e3 - e2, e42 = e4 - e2, e43 = e4 - e3;
    if (E < e2) {
        const double x = E - e1, C = q * x * x * x / (e21 * e31 * e41);
        w[0] = C * (4.0 - x * (1.0 / e21 + 1.0 / e31 + 1.0 / e41));
        w[1] = C * x / e21;
        w[2] = C * x / e31;
        w[3] = C * x / e41;
    } else if (E < e3) {
        const double C1 = q * (E - e1) * (E - e1) / (e41 * e31);
        const double C2 = q * (E - e1) * (E - e2) * (e3 - E) / (e41 * e32 * e31);
        const double C3 = q * (E - e2) * (E - e2) * (e4 - E) / (e42 * e32 * e41);
        w[0] = C1 + (C1 + C2) * (e3 - E) / e31 + (C1 + C2 + C3) * (e4 - E) / e41;
        w[1] = C1 + C2 + C3 + (C2 + C3) * (e3 - E) / e32 + C3 * (e4 - E) / e42;
        w[2] = (C1 + C2) * (E - e1) / e31 + (C2 + C3) * (E - e2) / e32;
        w[3] = (C1 + C2 + C3) * (E - e1) / e41 + C3 * (E - e2) / e42;
    } else {
        const double x = e4 - E, C = q * x * x * x / (e41 * e42 * e43);
        w[0] = q - C * x / e41;
        w[1] = q - C * x / e42;
        w[2] = q - C * x / e43;
        w[3] = q - C * (4.0 - x * (1.0 / e41 + 1.0 / e42 + 1.0 / e43));
    }
}

namespace {
/// The 6 tetrahedra of a mesh subcell split along its shortest main diagonal (corner offsets).
std::vector<std::array<IVec3, 4>> subcell_tetrahedra(const Mat3& b, const IVec3& mesh) {
    // corners c0..c7 with bits (x, y, z); diagonals connect c with 7 - c
    auto vec = [&](int c) {
        Vec3 v{};
        const int o[3] = {c & 1, (c >> 1) & 1, (c >> 2) & 1};
        for (int i = 0; i < 3; ++i)
            for (int d = 0; d < 3; ++d) v[d] += o[i] * b[i][d] / mesh[i];
        return v;
    };
    int best = 0;
    double bl = 1e300;
    for (int c = 0; c < 4; ++c) {
        Vec3 a = vec(c), z = vec(7 - c);
        const double l = norm(Vec3{z[0] - a[0], z[1] - a[1], z[2] - a[2]});
        if (l < bl - 1e-12) { bl = l; best = c; }
    }
    // the 6 paths from corner `best` to `7 - best` flipping one differing bit at a time
    const int start = best, diff = start ^ 7;
    int bits[3], nb = 0;
    for (int i = 0; i < 3; ++i)
        if (diff & (1 << i)) bits[nb++] = 1 << i;
    std::vector<std::array<IVec3, 4>> out;
    const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    auto offs = [](int c) { return IVec3{c & 1, (c >> 1) & 1, (c >> 2) & 1}; };
    for (const auto& p : perm) {
        int c1 = start ^ bits[p[0]], c2 = c1 ^ bits[p[1]], c3 = c2 ^ bits[p[2]];
        out.push_back({offs(start), offs(c1), offs(c2), offs(c3)});
    }
    return out;
}
} // namespace

AnalysisOutputs analyze(const AnalysisInputs& in, const std::vector<BondSpec>& bonds, const SpectrumGrid& grid) {
    BlasThreads policy;
    const int nspin = static_cast<int>(in.T.size());
    require(nspin >= 1, ErrorCode::InvalidArgument, "no spin channels");
    const int nk = static_cast<int>(in.kfrac.size());
    require(nk > 0 && static_cast<int>(in.S.size()) == nk && static_cast<int>(in.weights.size()) == nk, ErrorCode::InvalidArgument,
            "k-point arrays inconsistent");
    require(in.eps.size() == in.T.size() && in.occ.size() == in.T.size(), ErrorCode::InvalidArgument,
            "spin dimensions inconsistent");
    require(in.T[0].size() == in.kfrac.size() && !in.T[0].empty(), ErrorCode::InvalidArgument,
            "projection k dimensions inconsistent");
    const int norb = static_cast<int>(in.T[0][0].rows());
    const int nbond = static_cast<int>(bonds.size());
    const auto& off = in.orb_offsets;
    const double sf = in.spin_factor;
    require(norb > 0 && off.size() >= 2 && off.front() == 0 && off.back() == norb &&
                std::is_sorted(off.begin(), off.end()), ErrorCode::InvalidArgument, "invalid orbital offsets");
    require(std::isfinite(sf) && sf > 0 && std::isfinite(in.overlap_threshold) &&
                in.overlap_threshold >= 0 && in.overlap_threshold < 1 && in.eig_min_n > 0,
            ErrorCode::InvalidArgument, "invalid spin factor or overlap threshold");
    require(grid.npts > 0 && std::isfinite(grid.emin) && std::isfinite(grid.emax) && grid.emax >= grid.emin &&
                std::isfinite(grid.reference) && std::isfinite(grid.bin_offset) && (grid.kind == 0 || grid.kind == 1),
            ErrorCode::InvalidArgument, "invalid spectrum grid");
    if (grid.kind == 0)
        require(std::isfinite(grid.sigma) && grid.sigma > 0, ErrorCode::InvalidArgument, "Gaussian sigma must be positive");
    else {
        require(grid.npts > 1 && grid.emax > grid.emin && in.kgrid.size() == in.kfrac.size(),
                ErrorCode::InvalidArgument, "invalid tetrahedron grid");
        std::size_t size = 1;
        for (int axis : in.mesh) {
            require(axis > 0 && static_cast<std::size_t>(axis) <= in.kfrac.size() / size,
                    ErrorCode::InvalidArgument, "invalid tetrahedron mesh dimensions");
            size *= static_cast<std::size_t>(axis);
        }
        require(size == in.kfrac.size(), ErrorCode::InvalidArgument, "tetrahedron mesh size differs from k count");
    }
    for (int k = 0; k < nk; ++k) {
        require(std::isfinite(in.weights[k]) && in.weights[k] >= 0, ErrorCode::InvalidArgument, "invalid k weight");
        for (double x : in.kfrac[k]) require(std::isfinite(x), ErrorCode::InvalidArgument, "non-finite k coordinate");
    }
    for (int spin = 0; spin < nspin; ++spin) {
        require(in.T[spin].size() == in.kfrac.size() && in.eps[spin].size() == in.kfrac.size() &&
                    in.occ[spin].size() == in.kfrac.size(), ErrorCode::InvalidArgument, "inconsistent per-spin k arrays");
        for (int k = 0; k < nk; ++k) {
            detail::validate_projection(in.T[spin][k], in.S[k], &in.eps[spin][k], &in.occ[spin][k]);
            require(in.T[spin][k].rows() == norb && in.T[spin][k].cols() >= norb, ErrorCode::InvalidArgument,
                    "projection basis or band dimensions inconsistent");
            if (grid.kind == 1)
                require(in.eps[spin][k].size() == in.eps[spin][0].size(), ErrorCode::InvalidArgument,
                        "tetrahedron spectra require equal band counts at every k");
        }
    }
    for (const auto& bond : bonds) {
        require(bond.atom_a >= 0 && bond.atom_b >= 0 && static_cast<std::size_t>(bond.atom_a) < off.size() - 1 &&
                    static_cast<std::size_t>(bond.atom_b) < off.size() - 1, ErrorCode::InvalidArgument, "bond atom out of range");
        const int na = off[bond.atom_a + 1] - off[bond.atom_a], nb = off[bond.atom_b + 1] - off[bond.atom_b];
        require(bond.S_R.size() == 0 || (bond.S_R.rows() == na && bond.S_R.cols() == nb && bond.S_R.allFinite()),
                ErrorCode::InvalidArgument, "bond overlap block has invalid dimensions or values");
        if (bond.has_frame()) {
            require(bond.rot_a.rows() == na && bond.rot_a.cols() == na && bond.rot_b.rows() == nb && bond.rot_b.cols() == nb &&
                        bond.rot_a.allFinite() && bond.rot_b.allFinite() && bond.absm_a.size() == static_cast<std::size_t>(na) &&
                        bond.absm_b.size() == static_cast<std::size_t>(nb), ErrorCode::InvalidArgument, "invalid bond frame");
            for (const auto* ms : {&bond.absm_a, &bond.absm_b})
                for (int m : *ms) require(m >= 0 && m < n_bond_channels - 1, ErrorCode::InvalidArgument, "invalid bond |m|");
        }
    }
    AnalysisOutputs out;
    out.nspin = nspin;
    out.nbond = nbond;
    out.norb = norb;
    out.npts = grid.npts;
    std::vector<double> E(grid.npts);
    for (int i = 0; i < grid.npts; ++i)
        E[i] = grid.npts > 1 ? grid.emin + (grid.emax - grid.emin) * i / (grid.npts - 1) : grid.emin;

    // Parallelise over (spin, k) only when that fills the threads; otherwise run the k loop
    // serially so that Eigen's own multithreaded matrix products are used (Gamma supercells).
#ifdef _OPENMP
    const bool par_k = nspin * nk >= omp_get_max_threads();
#else
    const bool par_k = false;
#endif
    // 1. reconstruction for every (spin, k)
    std::vector<detail::OverlapFactorization> overlap(nk);
    Guard g1;
#pragma omp parallel for schedule(dynamic) if (par_k)
    for (int k = 0; k < nk; ++k) g1.run([&] {
        overlap[k] = detail::factor_overlap(in.S[k], in.overlap_threshold, par_k ? nullptr : in.eig_backend, in.eig_min_n);
    });
    g1.rethrow();
    std::vector<ReconstructionK> rec(static_cast<std::size_t>(nspin) * nk);
#pragma omp parallel for schedule(dynamic) if (par_k)
    for (int sk = 0; sk < nspin * nk; ++sk) g1.run([&] {
        const int s = sk / nk, k = sk % nk;
        rec[sk] = detail::reconstruct_with_overlap(in.T[s][k], in.eps[s][k], overlap[k], par_k ? nullptr : in.eig_backend,
                                                  in.eig_min_n);
    });
    g1.rethrow();
    overlap.clear();
    out.used_device_eigensolver = !par_k && in.eig_backend && in.eig_backend->has_hermitian_eigensolver() &&
                                  norb >= in.eig_min_n;

    // 2. real-space Hamiltonian blocks of the requested bonds
    auto phase = [&](int k, const IVec3& R) {
        const double a = constants::two_pi * (in.kfrac[k][0] * R[0] + in.kfrac[k][1] * R[1] + in.kfrac[k][2] * R[2]);
        return cd(std::cos(a), std::sin(a));
    };
    std::vector<MatrixXcd> HR(static_cast<std::size_t>(nspin) * nbond), PR(HR.size());
#pragma omp parallel for schedule(static)
    for (int sb = 0; sb < nspin * nbond; ++sb) {
        const int s = sb / nbond, b = sb % nbond;
        const auto& bd = bonds[b];
        const int A0 = off[bd.atom_a], na = off[bd.atom_a + 1] - A0, B0 = off[bd.atom_b], nbo = off[bd.atom_b + 1] - B0;
        MatrixXcd acc = MatrixXcd::Zero(na, nbo), accp = MatrixXcd::Zero(na, nbo);
        for (int k = 0; k < nk; ++k) {
            const auto& r = rec[s * nk + k];
            const cd ph = std::conj(phase(k, bd.R));
            acc += in.weights[k] * ph * r.H.block(A0, B0, na, nbo);
            // density matrix block P_AB(k) = sum_n f_n C_An C_Bn^*  (per spin channel, Löwdin basis)
            accp += in.weights[k] * ph * (r.C.middleRows(A0, na) * in.occ[s][k].asDiagonal() * r.C.middleRows(B0, nbo).adjoint());
        }
        HR[sb] = acc;
        PR[sb] = accp;
    }

    // 3. band-resolved weights per (spin, k)
    struct KBands {
        MatrixXd cohp, coop, cobi, pw; // nbond x nb (x3), norb x nb
        MatrixXd cohp_ch;              // (nbond * n_bond_channels) x nb, empty without bond frames
    };
    std::vector<KBands> kb(static_cast<std::size_t>(nspin) * nk);
    out.icohp_orbital.assign(static_cast<std::size_t>(nspin) * nbond, MatrixXd());
    for (int s = 0; s < nspin; ++s)
        for (int b = 0; b < nbond; ++b)
            out.icohp_orbital[s * nbond + b] = MatrixXd::Zero(off[bonds[b].atom_a + 1] - off[bonds[b].atom_a],
                                                              off[bonds[b].atom_b + 1] - off[bonds[b].atom_b]);
    out.icohp.assign(static_cast<std::size_t>(nspin) * nbond, 0.0);
    constexpr int NC = n_bond_channels;
    const bool any_frame = std::any_of(bonds.begin(), bonds.end(), [](const BondSpec& b) { return b.has_frame(); });
    const int nch_rows = any_frame ? NC * nbond : 0;
    out.icohp_channel.assign(static_cast<std::size_t>(nspin) * nbond * NC, 0.0);
    out.icoop.assign(static_cast<std::size_t>(nspin) * nbond, 0.0);
    out.icobi.assign(static_cast<std::size_t>(nspin) * nbond, 0.0);
    out.pdos_integral.assign(static_cast<std::size_t>(nspin) * norb, 0.0);
    out.mulliken_gp.assign(static_cast<std::size_t>(nspin) * norb, 0.0);
    out.band_energy.assign(nspin, 0.0);
    double spill_occ_abs = 0, spill_occ_num = 0, spill_all_abs = 0, spill_all_num = 0, max_resid = 0, electrons = 0;
    double min_seig = std::numeric_limits<double>::infinity();
    Guard g3;
#pragma omp parallel if (par_k)
    {
        std::vector<double> pint_l(out.pdos_integral.size(), 0.0), mull_l(out.mulliken_gp.size(), 0.0), be_l(nspin, 0.0);
        double so_a = 0, so_n = 0, sa_a = 0, sa_n = 0, mr = 0, el = 0, ms = std::numeric_limits<double>::infinity();
#pragma omp for schedule(dynamic)
        for (int sk = 0; sk < nspin * nk; ++sk) g3.run([&] {
            const int s = sk / nk, k = sk % nk;
            const auto& r = rec[sk];
            const VectorXd& eps = in.eps[s][k];
            const VectorXd& f = in.occ[s][k];
            const double wk = in.weights[k];
            const auto nb = r.C.cols();
            KBands& B = kb[sk];
            B.pw = r.C.cwiseAbs2();
            const VectorXd pwf = B.pw * f;
            const MatrixXcd SCn = in.S[k] * r.Cn;
            const VectorXd mul = (r.Cn.conjugate().cwiseProduct(SCn)).real() * f;
            for (int i = 0; i < norb; ++i) {
                pint_l[s * norb + i] += sf * wk * pwf(i);
                mull_l[s * norb + i] += sf * wk * mul(i);
            }
            be_l[s] += sf * wk * f.dot(eps);
            el += sf * wk * f.sum();
            so_a += wk * f.dot(r.spill.cwiseAbs());
            so_n += wk * f.sum();
            sa_a += wk * r.spill.cwiseAbs().sum();
            sa_n += wk * static_cast<double>(nb);
            if (nb == norb) {
                mr = std::max(mr, r.band_residual.maxCoeff());
            } else {
                for (Eigen::Index n = 0; n < nb; ++n)
                    if (f(n) > 1e-8) mr = std::max(mr, std::abs(r.band_residual(n)));
            }
            ms = std::min(ms, r.s_eig.minCoeff());
        });
#pragma omp critical(prowl_analysis_reduce)
        {
            for (std::size_t i = 0; i < pint_l.size(); ++i) { out.pdos_integral[i] += pint_l[i]; out.mulliken_gp[i] += mull_l[i]; }
            for (int s = 0; s < nspin; ++s) out.band_energy[s] += be_l[s];
            spill_occ_abs += so_a; spill_occ_num += so_n; spill_all_abs += sa_a; spill_all_num += sa_n;
            max_resid = std::max(max_resid, mr);
            min_seig = std::min(min_seig, ms);
            electrons += el;
        }
    }
    g3.rethrow();

    // bond-resolved band weights: parallel over (spin, bond), sequential over k (conflict-free)
    for (int sk = 0; sk < nspin * nk; ++sk) {
        kb[sk].cohp = MatrixXd::Zero(nbond, rec[sk].C.cols());
        kb[sk].coop = MatrixXd::Zero(nbond, rec[sk].C.cols());
        kb[sk].cobi = MatrixXd::Zero(nbond, rec[sk].C.cols());
        kb[sk].cohp_ch = MatrixXd::Zero(nch_rows, rec[sk].C.cols());
    }
    Guard g5;
#pragma omp parallel for schedule(dynamic)
    for (int sb = 0; sb < nspin * nbond; ++sb) g5.run([&] {
        const int s = sb / nbond, b = sb % nbond;
        const auto& bd = bonds[b];
        const int A0 = off[bd.atom_a], na = off[bd.atom_a + 1] - A0, B0 = off[bd.atom_b], nbo = off[bd.atom_b + 1] - B0;
        const MatrixXcd SR = bd.S_R.cast<cd>();
        // bond frame: orbital rotations and channel masks (pairs with |m_A| = |m_B| = c; c = NC-1: |m_A| != |m_B|)
        const bool fr = bd.has_frame();
        MatrixXcd DA, DB;
        std::array<MatrixXcd, NC> mask;
        if (fr) {
            require(bd.rot_a.rows() == na && bd.rot_a.cols() == na && bd.rot_b.rows() == nbo && bd.rot_b.cols() == nbo &&
                        static_cast<int>(bd.absm_a.size()) == na && static_cast<int>(bd.absm_b.size()) == nbo,
                    ErrorCode::InvalidArgument, "bond frame: rotation / |m| sizes do not match the orbital blocks");
            DA = bd.rot_a.cast<cd>();
            DB = bd.rot_b.cast<cd>();
            for (int c = 0; c < NC; ++c) {
                mask[c] = MatrixXcd::Zero(na, nbo);
                for (int i = 0; i < na; ++i)
                    for (int j = 0; j < nbo; ++j) {
                        const int ma = bd.absm_a[i], mb = bd.absm_b[j];
                        if (c < NC - 1 ? (ma == c && mb == c) : (ma != mb)) mask[c](i, j) = 1.0;
                    }
            }
        }
        for (int k = 0; k < nk; ++k) {
            const auto& r = rec[s * nk + k];
            const VectorXd& f = in.occ[s][k];
            const double wk = in.weights[k];
            const cd ph = phase(k, bd.R);
            const MatrixXcd Hb = HR[sb] * ph;
            const auto CA = r.C.middleRows(A0, na);
            const auto CB = r.C.middleRows(B0, nbo);
            auto& B = kb[s * nk + k];
            B.cohp.row(b) = (CA.conjugate().cwiseProduct(Hb * CB)).colwise().sum().real();
            out.icohp[sb] += sf * wk * B.cohp.row(b).dot(f);
            // COBI: COHP with H(R) -> P(R), prefactor 2 so that ICOBI = 2 g sum |P_AB(R)|^2 (Wiberg)
            B.cobi.row(b) = 2.0 * (CA.conjugate().cwiseProduct((PR[sb] * ph) * CB)).colwise().sum().real();
            out.icobi[sb] += sf * wk * B.cobi.row(b).dot(f);
            if (fr) {
                // the same bond in the rotated (bond) frame: C' = D C, H' = D_A H D_B^T (totals unchanged)
                const MatrixXcd CAr = DA * CA, CBr = DB * CB, Hr = DA * Hb * DB.transpose();
                out.icohp_orbital[sb] += sf * wk * (Hr.cwiseProduct(CAr.conjugate() * f.asDiagonal() * CBr.transpose())).real();
                for (int c = 0; c < NC; ++c) {
                    B.cohp_ch.row(b * NC + c) = (CAr.conjugate().cwiseProduct(Hr.cwiseProduct(mask[c]) * CBr)).colwise().sum().real();
                    out.icohp_channel[static_cast<std::size_t>(sb) * NC + c] += sf * wk * B.cohp_ch.row(b * NC + c).dot(f);
                }
            } else {
                out.icohp_orbital[sb] += sf * wk * (Hb.cwiseProduct(CA.conjugate() * f.asDiagonal() * CB.transpose())).real();
            }
            if (bd.S_R.size() > 0) {
                const auto CnA = r.Cn.middleRows(A0, na);
                const auto CnB = r.Cn.middleRows(B0, nbo);
                B.coop.row(b) = (CnA.conjugate().cwiseProduct((SR * ph) * CnB)).colwise().sum().real();
                out.icoop[sb] += sf * wk * B.coop.row(b).dot(f);
            }
        }
    });
    g5.rethrow();
    rec.clear();
    out.lowdin_gp = out.pdos_integral;
    out.abs_charge_spilling = spill_occ_num > 0 ? spill_occ_abs / spill_occ_num : 0.0;
    out.abs_total_spilling = spill_all_num > 0 ? spill_all_abs / spill_all_num : 0.0;
    out.max_reconstruction_residual = max_resid;
    out.min_overlap_eigenvalue = min_seig;
    out.electrons_dft = electrons;

    out.cohp.assign(static_cast<std::size_t>(nspin) * nbond * grid.npts, 0.0);
    out.coop.assign(out.cohp.size(), 0.0);
    out.cobi.assign(out.cohp.size(), 0.0);
    out.cohp_channel.assign(out.cohp.size() * NC, 0.0);
    out.pdos.assign(static_cast<std::size_t>(nspin) * norb * grid.npts, 0.0);
    if (grid.kind == 1) {
        // 4'. linear tetrahedron spectra (bin averages of the integrated quantity)
        const IVec3 M = in.mesh;
        const int ntot = M[0] * M[1] * M[2];
        require(ntot == nk && static_cast<int>(in.kgrid.size()) == nk && ntot > 1, ErrorCode::InvalidArgument,
                "tetrahedron method needs a full uniform k mesh with more than one point");
        std::vector<int> idx(ntot, -1);
        for (int k = 0; k < nk; ++k) {
            const auto& g = in.kgrid[k];
            const int i = ((g[0] % M[0]) + M[0]) % M[0], j = ((g[1] % M[1]) + M[1]) % M[1], l = ((g[2] % M[2]) + M[2]) % M[2];
            idx[(i * M[1] + j) * M[2] + l] = k;
        }
        for (int v : idx) require(v >= 0, ErrorCode::InvalidArgument, "k mesh incomplete for the tetrahedron method");
        const auto tets = subcell_tetrahedra(in.recip, M);
        const double vt = 1.0 / (6.0 * ntot);
        const double dE = grid.npts > 1 ? (grid.emax - grid.emin) / (grid.npts - 1) : 1.0;
        const int nq = 3 * nbond + norb + nch_rows;
        Guard gt;
        for (int s = 0; s < nspin; ++s) {
            const auto nbands = in.eps[s][0].size();
#pragma omp parallel
            {
                MatrixXd acc = MatrixXd::Zero(nq, grid.npts);
                std::vector<double> edges(grid.npts + 1);
                for (int i = 0; i <= grid.npts; ++i) edges[i] = E[0] + (i + grid.bin_offset) * dE;
#pragma omp for schedule(dynamic) collapse(2)
                for (int cell = 0; cell < ntot; ++cell)
                    for (int t = 0; t < 6; ++t) gt.run([&] {
                        const int ci = cell / (M[1] * M[2]), cj = (cell / M[2]) % M[1], cl = cell % M[2];
                        int kc[4];
                        for (int c = 0; c < 4; ++c) {
                            const auto& o = tets[t][c];
                            kc[c] = idx[(((ci + o[0]) % M[0]) * M[1] + (cj + o[1]) % M[1]) * M[2] + (cl + o[2]) % M[2]];
                        }
                        for (Eigen::Index n = 0; n < static_cast<Eigen::Index>(nbands); ++n) {
                            int ord[4] = {0, 1, 2, 3};
                            double e[4];
                            for (int c = 0; c < 4; ++c) e[c] = in.eps[s][kc[c]](n) - grid.reference;
                            std::sort(ord, ord + 4, [&](int a, int b) { return e[a] < e[b]; });
                            double es[4];
                            for (int c = 0; c < 4; ++c) es[c] = e[ord[c]];
                            for (int c = 1; c < 4; ++c) // split exact degeneracies (limits are continuous)
                                if (es[c] - es[c - 1] < 1e-9) es[c] = es[c - 1] + 1e-9;
                            const int b0 = std::max(0, static_cast<int>(std::floor((es[0] - edges[0]) / dE)) - 1);
                            const int b1 = std::min(grid.npts - 1, static_cast<int>(std::ceil((es[3] - edges[0]) / dE)) + 1);
                            if (b1 < 0 || b0 > grid.npts - 1) continue;
                            double wlo[4], whi[4];
                            tetra_weights(es, edges[b0], vt, wlo);
                            for (int b = b0; b <= b1; ++b) {
                                tetra_weights(es, edges[b + 1], vt, whi);
                                for (int c = 0; c < 4; ++c) {
                                    const double dw = sf * (whi[c] - wlo[c]) / dE;
                                    wlo[c] = whi[c];
                                    if (dw == 0.0) continue;
                                    const auto& B = kb[s * nk + kc[ord[c]]];
                                    for (int q = 0; q < nbond; ++q) {
                                        acc(q, b) += dw * B.cohp(q, n);
                                        acc(nbond + q, b) += dw * B.coop(q, n);
                                        acc(2 * nbond + q, b) += dw * B.cobi(q, n);
                                    }
                                    for (int o = 0; o < norb; ++o) acc(3 * nbond + o, b) += dw * B.pw(o, n);
                                    for (int q = 0; q < nch_rows; ++q) acc(3 * nbond + norb + q, b) += dw * B.cohp_ch(q, n);
                                }
                            }
                        }
                    });
#pragma omp critical(prowl_analysis_tetra)
                for (int b = 0; b < grid.npts; ++b) {
                    for (int q = 0; q < nbond; ++q) {
                        out.cohp[(static_cast<std::size_t>(s) * nbond + q) * grid.npts + b] += acc(q, b);
                        out.coop[(static_cast<std::size_t>(s) * nbond + q) * grid.npts + b] += acc(nbond + q, b);
                        out.cobi[(static_cast<std::size_t>(s) * nbond + q) * grid.npts + b] += acc(2 * nbond + q, b);
                    }
                    for (int o = 0; o < norb; ++o)
                        out.pdos[(static_cast<std::size_t>(s) * norb + o) * grid.npts + b] += acc(3 * nbond + o, b);
                    for (int q = 0; q < nch_rows; ++q)
                        out.cohp_channel[(static_cast<std::size_t>(s) * nch_rows + q) * grid.npts + b] +=
                            acc(3 * nbond + norb + q, b);
                }
            }
            gt.rethrow();
        }
        return out;
    }
    // 4. broadened spectra: per spin, sum over k of (weights x nb) * G_k(nb x npts), GEMM per k
    const double norm = 1.0 / (grid.sigma * std::sqrt(constants::two_pi));
    Guard g4;
#pragma omp parallel if (par_k)
    {
        MatrixXd cohp_l = MatrixXd::Zero(nbond, grid.npts), coop_l = MatrixXd::Zero(nbond, grid.npts);
        MatrixXd cobi_l = MatrixXd::Zero(nbond, grid.npts);
        MatrixXd pdos_l = MatrixXd::Zero(norb, grid.npts);
        MatrixXd ch_l = MatrixXd::Zero(nch_rows, grid.npts);
        for (int s = 0; s < nspin; ++s) {
            cohp_l.setZero(); coop_l.setZero(); cobi_l.setZero(); pdos_l.setZero(); ch_l.setZero();
#pragma omp for schedule(dynamic)
            for (int k = 0; k < nk; ++k) g4.run([&] {
                const auto& B = kb[s * nk + k];
                const VectorXd& eps = in.eps[s][k];
                const auto nb = eps.size();
                MatrixXd G(nb, grid.npts);
                for (Eigen::Index n = 0; n < nb; ++n) {
                    const double e0 = eps(n) - grid.reference;
                    for (int i = 0; i < grid.npts; ++i) {
                        const double x = (E[i] - e0) / grid.sigma;
                        G(n, i) = x * x < 1400.0 ? sf * in.weights[k] * norm * std::exp(-0.5 * x * x) : 0.0;
                    }
                }
                cohp_l.noalias() += B.cohp * G;
                coop_l.noalias() += B.coop * G;
                cobi_l.noalias() += B.cobi * G;
                pdos_l.noalias() += B.pw * G;
                if (nch_rows > 0) ch_l.noalias() += B.cohp_ch * G;
            });
#pragma omp critical(prowl_analysis_spectra)
            {
                for (int b = 0; b < nbond; ++b)
                    for (int i = 0; i < grid.npts; ++i) {
                        out.cohp[(static_cast<std::size_t>(s) * nbond + b) * grid.npts + i] += cohp_l(b, i);
                        out.coop[(static_cast<std::size_t>(s) * nbond + b) * grid.npts + i] += coop_l(b, i);
                        out.cobi[(static_cast<std::size_t>(s) * nbond + b) * grid.npts + i] += cobi_l(b, i);
                    }
                for (int o = 0; o < norb; ++o)
                    for (int i = 0; i < grid.npts; ++i)
                        out.pdos[(static_cast<std::size_t>(s) * norb + o) * grid.npts + i] += pdos_l(o, i);
                for (int q = 0; q < nch_rows; ++q)
                    for (int i = 0; i < grid.npts; ++i)
                        out.cohp_channel[(static_cast<std::size_t>(s) * nch_rows + q) * grid.npts + i] += ch_l(q, i);
            }
#pragma omp barrier
        }
    }
    g4.rethrow();
    return out;
}

} // namespace prowl
