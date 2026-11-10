// SPDX-License-Identifier: GPL-3.0-or-later
#include "analysis/numerics.hpp"
#include "prowl/analysis.hpp"
#include "prowl/backend.hpp"
#include "prowl/errors.hpp"
#include "prowl/threading.hpp"
#include <algorithm>
#include <cmath>

namespace prowl {
using detail::heev;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
MatrixXcd hermitian_power(const MatrixXcd& A, double p, double threshold, int* rank, VectorXd* eigenvalues,
                          Backend* eig, int eig_min_n) {
    BlasThreads policy;
    return detail::matrix_power(A, p, threshold, rank, eigenvalues, eig, eig_min_n);
}

MatrixXcd detail::matrix_power(const MatrixXcd& A, double p, double threshold, int* rank, VectorXd* eigenvalues,
                               Backend* eig, int eig_min_n) {
    require(A.rows() > 0 && A.rows() == A.cols() && A.allFinite(), ErrorCode::InvalidArgument,
            "Hermitian matrix must be nonempty, square and finite");
    require(std::isfinite(p) && std::isfinite(threshold) && threshold >= 0 && threshold < 1 && eig_min_n > 0,
            ErrorCode::InvalidArgument, "invalid matrix power or eigenvalue threshold");
    MatrixXcd V = 0.5 * (A + A.adjoint());
    VectorXd w;
    if (eig && eig->has_hermitian_eigensolver() && V.rows() >= eig_min_n) {
        w.resize(V.rows());
        eig->hermitian_eig(V.data(), static_cast<int>(V.rows()), w.data());
    } else {
        heev(V, w, true);
    }
    require(w.size() > 0 && w(w.size() - 1) > 0, ErrorCode::NumericalFailure, "matrix is not positive definite");
    const double cut = threshold * w(w.size() - 1);
    VectorXd wp = VectorXd::Zero(w.size());
    int r = 0;
    for (Eigen::Index i = 0; i < w.size(); ++i)
        if (w(i) > cut) {
            wp(i) = std::pow(w(i), p);
            ++r;
        }
    if (rank) *rank = r;
    if (eigenvalues) *eigenvalues = w;
    return V * wp.asDiagonal() * V.adjoint();
}

ReconstructionK reconstruct_k(const MatrixXcd& T, const MatrixXcd& S, const VectorXd& eps, double overlap_threshold,
                              Backend* eig, int eig_min_n) {
    BlasThreads policy;
    detail::validate_projection(T, S, &eps);
    return detail::reconstruct_with_overlap(T, eps, detail::factor_overlap(S, overlap_threshold, eig, eig_min_n), eig,
                                            eig_min_n);
}

detail::OverlapFactorization detail::factor_overlap(const MatrixXcd& S, double threshold, Backend* eig, int eig_min_n) {
    OverlapFactorization overlap;
    overlap.inverse_sqrt = matrix_power(S, -0.5, threshold, &overlap.rank, &overlap.eigenvalues, eig, eig_min_n);
    return overlap;
}

void detail::validate_projection(const MatrixXcd& T, const MatrixXcd& S, const VectorXd* eps, const VectorXd* occ) {
    require(T.rows() > 0 && T.cols() > 0 && T.allFinite(), ErrorCode::InvalidArgument,
            "projections must be nonempty and finite");
    require(S.rows() == T.rows() && S.cols() == T.rows() && S.allFinite(), ErrorCode::InvalidArgument,
            "overlap matrix dimensions must match the local basis and contain finite values");
    if (eps)
        require(eps->size() == T.cols() && eps->allFinite(), ErrorCode::InvalidArgument,
                "eigenvalue count does not match band window or contains non-finite values");
    if (occ)
        require(occ->size() == T.cols() && occ->allFinite(), ErrorCode::InvalidArgument,
                "occupation count does not match band window or contains non-finite values");
}

ReconstructionK detail::reconstruct_with_overlap(const MatrixXcd& T, const VectorXd& eps,
                                                 const OverlapFactorization& overlap, Backend* eig, int eig_min_n) {
    const auto norb = T.rows(), nb = T.cols();
    require(nb >= norb, ErrorCode::InvalidArgument,
            "band window (" + std::to_string(nb) + ") smaller than basis (" + std::to_string(norb) +
                "); increase NBANDS");
    require(eps.size() == nb, ErrorCode::InvalidArgument, "eigenvalue count does not match band window");
    ReconstructionK r;
    const auto& X = overlap.inverse_sqrt;
    r.rank = overlap.rank;
    r.s_eig = overlap.eigenvalues;
    require(r.rank == norb, ErrorCode::NumericalFailure,
            "overlap matrix rank-deficient (" + std::to_string(r.rank) + " < " + std::to_string(norb) +
                "); basis nearly linearly dependent");
    const MatrixXcd Tp = X * T;
    r.spill = VectorXd::Ones(nb) - Tp.colwise().squaredNorm().transpose();
    const MatrixXcd O = Tp * Tp.adjoint();
    r.completeness = O.diagonal().real();
    int orank = 0;
    VectorXd oeig;
    const MatrixXcd Oi = matrix_power(O, -0.5, 1e-12, &orank, &oeig, eig, eig_min_n);
    if (orank < norb) {
        fail(ErrorCode::NumericalFailure, "the " + std::to_string(nb) + " retained bands do not span the " +
                                              std::to_string(norb) +
                                              "-dimensional local orbital space; rerun VASP with more bands (NBANDS; "
                                              "polarization functions typically need NBANDS >= 2 x N_orb); with the "
                                              "lowest-N_orb window, band_window frozen or all may also help");
    }
    r.C = Oi * Tp;
    MatrixXcd H = r.C * eps.asDiagonal() * r.C.adjoint();
    r.H = 0.5 * (H + H.adjoint());
    r.Cn = X * r.C;
    r.band_residual.resize(nb);
    if (nb == norb) {
        const MatrixXcd CHC = r.C.adjoint() * r.H * r.C;
        for (Eigen::Index n = 0; n < nb; ++n) {
            double m = 0.0;
            for (Eigen::Index i = 0; i < nb; ++i)
                m = std::max(m, std::abs(CHC(i, n) - (i == n ? eps(n) : 0.0)));
            r.band_residual(n) = m;
        }
    } else {
        // Rectangular windows need only diag(C^H H C); forming the nb x nb matrix adds quadratic band storage.
        const MatrixXcd HC = r.H * r.C;
        const VectorXd cc = r.C.colwise().squaredNorm().transpose();
        for (Eigen::Index n = 0; n < nb; ++n)
            r.band_residual(n) = r.C.col(n).dot(HC.col(n)).real() - eps(n) * cc(n);
    }
    return r;
}

} // namespace prowl
