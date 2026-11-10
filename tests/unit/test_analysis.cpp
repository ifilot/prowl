// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/analysis.hpp"
#include "prowl/backend.hpp"
#include <Eigen/Eigenvalues>
#include <catch2/catch_test_macros.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
class CountingEigensolver final : public prowl::Backend {
public:
    int calls = 0;
    std::string name() const override { return "counting"; }
    std::string device_description() const override { return "test eigensolver"; }
    void set_factor_memory_limit(std::size_t) override {}
    prowl::FactorMemoryStats factor_memory_stats() const override { return {}; }
    void project_full(const std::complex<double>*, const std::complex<double>*, int, int, int,
                      std::complex<double>*) override {}
    void project_gamma(const std::complex<double>*, const std::complex<double>*, const double*, int, int, int,
                       double*) override {}
    void load_coefficients(const std::complex<double>*, int, int, const double*) override {}
    void project_separable(const prowl::SeparableFactorSpec&, int, int, std::complex<double>*, double*) override {}
    void evaluate_factors(const prowl::SeparableFactorSpec&, int, int, std::complex<double>*) override {}
    bool has_hermitian_eigensolver() const override { return true; }
    void hermitian_eig(std::complex<double>* A, int n, double* w) override {
        ++calls;
        Eigen::Map<Eigen::MatrixXcd> matrix(A, n, n);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(matrix);
        matrix = solver.eigenvectors();
        Eigen::Map<Eigen::VectorXd>(w, n) = solver.eigenvalues();
    }
};
} // namespace

TEST_CASE("spin channels share a k-point overlap factorization", "[analysis]") {
    prowl::AnalysisInputs in;
    in.kfrac = {{0, 0, 0}};
    in.weights = {1};
    const Eigen::MatrixXcd T = Eigen::MatrixXcd::Identity(2, 2);
    in.T = {{T}, {T}};
    Eigen::MatrixXcd S = T;
    S(0, 0) = 2;
    S(1, 1) = 3;
    in.S = {S};
    Eigen::VectorXd e(2), f(2);
    e << -1, 1;
    f << 1, 0;
    in.eps = {{e}, {e}};
    in.occ = {{f}, {f}};
    in.orb_offsets = {0, 1, 2};
    CountingEigensolver eig;
    in.eig_backend = &eig;
    in.eig_min_n = 1;
    // Force sequential spin processing, so the optional eigensolver observes every factorization.
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    const auto out = prowl::analyze(in, {}, {-2, 2, 31, .05, 0});
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
    REQUIRE(eig.calls == 3); // S once, polar overlap once per spin
    REQUIRE(out.electrons_dft == 4.0);
    REQUIRE(out.max_reconstruction_residual < 1e-12);
}
