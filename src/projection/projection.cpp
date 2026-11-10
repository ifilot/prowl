// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/projection.hpp"
#include "prowl/constants.hpp"
#include "prowl/errors.hpp"
#include "detail/omp_exception.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace prowl {
using cd = std::complex<double>;
using OmpExceptionGuard = detail::ExceptionGuard;
double ProjectionModel::projector_qmax() const noexcept {
    double q = 1e300;
    for (const auto& d : datasets_) q = std::min(q, d.psmaxn * (PawDataset::nq() - 1) / PawDataset::nq());
    return q;
}

Eigen::MatrixXcd ProjectionModel::lattice_sum(const std::vector<LatticeTerm>& terms, const Vec3& k, int nrows,
                                              int ncols, const std::vector<int>& row_off,
                                              const std::vector<int>& col_off) const {
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(nrows, ncols);
    for (const auto& t : terms) {
        const double arg = constants::two_pi * (k[0] * t.R[0] + k[1] * t.R[1] + k[2] * t.R[2]);
        const cd ph(std::cos(arg), std::sin(arg));
        M.block(row_off[t.row_atom], col_off[t.col_atom], t.block.rows(), t.block.cols()) += ph * t.block.cast<cd>();
    }
    return M;
}

Eigen::MatrixXcd ProjectionModel::overlap(const Vec3& k) const {
    return lattice_sum(s_terms_, k, norb(), norb(), orb_offset_, orb_offset_);
}

namespace {
/// Smallest N (<= 128) such that k N has the same fractional part for every k (a regular, possibly shifted mesh
/// along this axis); 0 if there is none.
int mesh_divisions(const std::vector<Vec3>& ks, int axis, double& shift) {
    for (int N = 1; N <= 128; ++N) {
        const double x0 = ks[0][axis] * N, f0 = x0 - std::round(x0);
        bool ok = true;
        for (const auto& k : ks) {
            const double x = k[axis] * N;
            double df = (x - std::round(x)) - f0;
            df -= std::round(df);
            if (std::abs(df) > 1e-7) { ok = false; break; }
        }
        if (ok) { shift = f0 / N; return N; }
    }
    return 0;
}
} // namespace

std::vector<Eigen::MatrixXcd> ProjectionModel::overlaps(const std::vector<Vec3>& ks) const {
    // Regular (shifted) k mesh: fold the lattice terms exactly into N1 x N2 x N3 bins per atom pair,
    //   S(k) = sum_rbar e^{2 pi i k.rbar} sum_{R = rbar mod N} e^{2 pi i sigma.(R - rbar)} B_R,
    // with k = (m + s)/N and sigma = s/N (k.(R - rbar) = (m + s).Q, Q integer). Exact up to rounding.
    std::vector<LatticeTerm> folded;
    const std::vector<LatticeTerm>* terms = &s_terms_;
    std::vector<Eigen::MatrixXcd> fblocks;
    if (ks.size() > 1) {
        int N[3];
        double sig[3];
        bool mesh = true;
        for (int c = 0; c < 3 && mesh; ++c) mesh = (N[c] = mesh_divisions(ks, c, sig[c])) > 0;
        if (mesh && static_cast<double>(N[0]) * N[1] * N[2] * s_.natoms() * s_.natoms() < 0.5 * static_cast<double>(s_terms_.size())) {
            std::map<std::array<int, 5>, int> bin;
            for (const auto& t : s_terms_) {
                IVec3 rb;
                for (int c = 0; c < 3; ++c) rb[c] = ((t.R[c] % N[c]) + N[c]) % N[c];
                const double arg = constants::two_pi * (sig[0] * (t.R[0] - rb[0]) + sig[1] * (t.R[1] - rb[1]) +
                                                        sig[2] * (t.R[2] - rb[2]));
                auto [it, ins] = bin.try_emplace({t.row_atom, t.col_atom, rb[0], rb[1], rb[2]},
                                                 static_cast<int>(folded.size()));
                if (ins) {
                    folded.push_back({t.row_atom, t.col_atom, rb, {}});
                    fblocks.push_back(Eigen::MatrixXcd::Zero(t.block.rows(), t.block.cols()));
                }
                fblocks[it->second] += cd(std::cos(arg), std::sin(arg)) * t.block.cast<cd>();
            }
            terms = &folded;
        }
    }
    const bool cplx = !fblocks.empty();
    // distinct lattice vectors of the terms
    std::map<std::array<int, 3>, int> rid;
    std::vector<int> term_r(terms->size());
    std::vector<IVec3> Rs;
    for (std::size_t t = 0; t < terms->size(); ++t) {
        const auto& R = (*terms)[t].R;
        auto [it, inserted] = rid.try_emplace({R[0], R[1], R[2]}, static_cast<int>(Rs.size()));
        if (inserted) Rs.push_back(R);
        term_r[t] = it->second;
    }
    std::vector<Eigen::MatrixXcd> out(ks.size());
    OmpExceptionGuard guard;
#pragma omp parallel
    {
        std::vector<cd> ph(Rs.size());
#pragma omp for schedule(dynamic)
        for (std::size_t ik = 0; ik < ks.size(); ++ik) guard.run([&] {
            const auto& k = ks[ik];
            for (std::size_t r = 0; r < Rs.size(); ++r) {
                const double arg = constants::two_pi * (k[0] * Rs[r][0] + k[1] * Rs[r][1] + k[2] * Rs[r][2]);
                ph[r] = cd(std::cos(arg), std::sin(arg));
            }
            Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(norb(), norb());
            if (cplx) {
                for (std::size_t t = 0; t < terms->size(); ++t) {
                    const auto& term = (*terms)[t];
                    M.block(orb_offset_[term.row_atom], orb_offset_[term.col_atom], fblocks[t].rows(),
                            fblocks[t].cols()) += ph[term_r[t]] * fblocks[t];
                }
            } else {
                // real blocks: accumulate Re and Im parts separately (no complex temporaries)
                Eigen::MatrixXd Mr = Eigen::MatrixXd::Zero(norb(), norb()), Mi = Eigen::MatrixXd::Zero(norb(), norb());
                for (std::size_t t = 0; t < terms->size(); ++t) {
                    const auto& term = (*terms)[t];
                    const Eigen::Index r0 = orb_offset_[term.row_atom], c0 = orb_offset_[term.col_atom];
                    Mr.block(r0, c0, term.block.rows(), term.block.cols()) += ph[term_r[t]].real() * term.block;
                    Mi.block(r0, c0, term.block.rows(), term.block.cols()) += ph[term_r[t]].imag() * term.block;
                }
                M.real() = Mr;
                M.imag() = Mi;
            }
            out[ik] = std::move(M);
        });
    }
    guard.rethrow();
    return out;
}

Eigen::VectorXd ProjectionModel::paw_q_norms(const Eigen::MatrixXcd& b) const {
    // Q couples projectors of the same atom only
    Eigen::VectorXd extra = Eigen::VectorXd::Zero(b.cols());
    for (int a = 0; a < s_.natoms(); ++a) {
        const int off = proj_offset_[a], na = proj_offset_[a + 1] - proj_offset_[a];
        if (na == 0) continue;
        const auto ba = b.middleRows(off, na);
        const Eigen::MatrixXcd qb = Q_.block(off, off, na, na).cast<cd>() * ba;
        extra += ba.conjugate().cwiseProduct(qb).colwise().sum().real().transpose();
    }
    return extra;
}

void ProjectionModel::finish_paw(const Vec3& k, KProjection& kp) const {
    const auto nb = kp.b.cols();
    const Eigen::VectorXd extra = paw_q_norms(kp.b);
    for (Eigen::Index n = 0; n < nb; ++n) kp.paw_norm[n] = kp.pw_norm[n] + extra(n);
    // all-electron amplitudes: D(k) = sum_t e^{2 pi i k.R_t} B_t; block by block unless the terms are dense
    double sparse = 0.0;
    for (const auto& t : d_terms_) sparse += static_cast<double>(t.block.size());
    if (sparse > 0.5 * static_cast<double>(norb()) * nproj()) {
        kp.a_ae = kp.a_soft + paw_coupling(k) * kp.b;
        return;
    }
    kp.a_ae = kp.a_soft;
    for (const auto& t : d_terms_) {
        const double arg = constants::two_pi * (k[0] * t.R[0] + k[1] * t.R[1] + k[2] * t.R[2]);
        const cd ph(std::cos(arg), std::sin(arg));
        kp.a_ae.middleRows(orb_offset_[t.row_atom], t.block.rows()).noalias() +=
            (ph * t.block.cast<cd>()) * kp.b.middleRows(proj_offset_[t.col_atom], t.block.cols());
    }
}

Eigen::MatrixXcd ProjectionModel::paw_coupling(const Vec3& k) const {
    return lattice_sum(d_terms_, k, norb(), nproj(), orb_offset_, proj_offset_);
}

void ProjectionModel::check_q_range(const Vec3& k, const std::vector<IVec3>& G, bool projectors) const {
    const double qlim = projectors ? projector_qmax() : orbital_qmax_;
    for (const auto& g : G) {
        const double qn = norm(s_.lattice.reciprocal_cartesian({k[0] + g[0], k[1] + g[1], k[2] + g[2]}));
        if (qn > qlim)
            fail(ErrorCode::UnsupportedFeature,
                 projectors ? "plane wave |k+G| exceeds POTCAR projector table (PSMAXN); ENCUT too high"
                            : "plane wave beyond orbital transform table");
    }
}

Eigen::MatrixXcd ProjectionModel::orbital_factors(const Vec3& k, const std::vector<IVec3>& G, int first, int n) const {
    if (n < 0) n = norb() - first;
    require(first >= 0 && first + n <= norb(), ErrorCode::InvalidArgument, "orbital column range out of bounds");
    check_q_range(k, G, false);
    std::lock_guard<std::mutex> lock(spec_mutex_);
    orb_spec_.k = k;
    orb_spec_.G = G.data();
    orb_spec_.ng = static_cast<int>(G.size());
    Eigen::MatrixXcd W(static_cast<Eigen::Index>(G.size()), n);
    evaluate_separable_host(orb_spec_, first, n, W.data());
    return W;
}

Eigen::MatrixXcd ProjectionModel::projector_factors(const Vec3& k, const std::vector<IVec3>& G, int first, int n) const {
    if (n < 0) n = nproj() - first;
    require(first >= 0 && first + n <= nproj(), ErrorCode::InvalidArgument, "projector column range out of bounds");
    check_q_range(k, G, true);
    std::lock_guard<std::mutex> lock(spec_mutex_);
    proj_spec_.k = k;
    proj_spec_.G = G.data();
    proj_spec_.ng = static_cast<int>(G.size());
    Eigen::MatrixXcd W(static_cast<Eigen::Index>(G.size()), n);
    evaluate_separable_host(proj_spec_, first, n, W.data());
    return W;
}

Eigen::MatrixXcd ProjectionModel::factors_on_backend(const Vec3& k, const std::vector<IVec3>& G, bool projectors,
                                                     Backend& backend) const {
    std::lock_guard<std::mutex> lock(spec_mutex_);
    auto& sp = projectors ? proj_spec_ : orb_spec_;
    sp.k = k;
    sp.G = G.data();
    sp.ng = static_cast<int>(G.size());
    Eigen::MatrixXcd W(static_cast<Eigen::Index>(G.size()), sp.nf());
    backend.evaluate_factors(sp, 0, sp.nf(), W.data());
    return W;
}


} // namespace prowl
