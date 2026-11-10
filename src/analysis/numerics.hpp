// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "prowl/analysis.hpp"
#include "detail/omp_exception.hpp"

namespace prowl::detail {
// Private numerical boundary shared by reconstruction and band-window selection.
void heev(Eigen::MatrixXcd& A, Eigen::VectorXd& w, bool vectors);
void heevr_top(Eigen::MatrixXcd& A, int k, Eigen::VectorXd& w, Eigen::MatrixXcd& Z);
bool exceeds(const Eigen::MatrixXcd& A, double shift);
Eigen::MatrixXcd matrix_power(const Eigen::MatrixXcd& A, double p, double threshold, int* rank,
                              Eigen::VectorXd* eigenvalues, Backend* eig, int eig_min_n);

struct OverlapFactorization {
    Eigen::MatrixXcd inverse_sqrt;
    Eigen::VectorXd eigenvalues;
    int rank{};
};
OverlapFactorization factor_overlap(const Eigen::MatrixXcd& S, double threshold, Backend* eig = nullptr,
                                    int eig_min_n = 256);
ReconstructionK reconstruct_with_overlap(const Eigen::MatrixXcd& T, const Eigen::VectorXd& eps,
                                         const OverlapFactorization& overlap, Backend* eig, int eig_min_n);
void validate_projection(const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, const Eigen::VectorXd* eps = nullptr,
                         const Eigen::VectorXd* occ = nullptr);

} // namespace prowl::detail
