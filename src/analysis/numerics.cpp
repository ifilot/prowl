// SPDX-License-Identifier: GPL-3.0-or-later
#include "analysis/numerics.hpp"
#include "prowl/errors.hpp"
#include <algorithm>
#include <complex>
#include <vector>

extern "C" void zheevd_(const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda,
                        double* w, std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork,
                        int* iwork, const int* liwork, int* info, std::size_t, std::size_t);
extern "C" void zheevr_(const char* jobz, const char* range, const char* uplo, const int* n, std::complex<double>* a,
                        const int* lda, const double* vl, const double* vu, const int* il, const int* iu,
                        const double* abstol, int* m, double* w, std::complex<double>* z, const int* ldz, int* isuppz,
                        std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork, int* iwork,
                        const int* liwork, int* info, std::size_t, std::size_t, std::size_t);

namespace prowl::detail {
/// Hermitian eigendecomposition with LAPACK zheevd (divide and conquer); eigenvalues ascending.
void heev(Eigen::MatrixXcd& A, Eigen::VectorXd& w, bool vectors) {
    const int n = static_cast<int>(A.rows());
    w.resize(n);
    if (n == 0) return;
    const char jobz = vectors ? 'V' : 'N', uplo = 'L';
    int info = 0, lwork = -1, lrwork = -1, liwork = -1, iw = 0;
    std::complex<double> wq;
    double rq = 0;
    zheevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), &wq, &lwork, &rq, &lrwork, &iw, &liwork, &info, 1, 1);
    lwork = std::max(1, static_cast<int>(wq.real()));
    lrwork = std::max(1, static_cast<int>(rq));
    liwork = std::max(1, iw);
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);
    zheevd_(&jobz, &uplo, &n, A.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
            &info, 1, 1);
    require(info == 0, ErrorCode::NumericalFailure, "LAPACK zheevd failed (info = " + std::to_string(info) + ")");
}

/// The k largest eigenpairs of a Hermitian matrix (LAPACK zheevr, MRRR; lower triangle of A is read and destroyed).
/// w: k eigenvalues ascending; Z: n x k eigenvectors.
void heevr_top(Eigen::MatrixXcd& A, int k, Eigen::VectorXd& w, Eigen::MatrixXcd& Z) {
    const int n = static_cast<int>(A.rows());
    k = std::min(k, n);
    w.resize(n);
    Z.resize(n, std::max(k, 1));
    if (n == 0 || k <= 0) {
        w.resize(0);
        Z.resize(n, 0);
        return;
    }
    const char jobz = 'V', range = 'I', uplo = 'L';
    const int il = n - k + 1, iu = n, ldz = n;
    const double vl = 0.0, vu = 0.0, abstol = 0.0;
    int m = 0, info = 0, lwork = -1, lrwork = -1, liwork = -1, iw = 0;
    std::vector<int> isuppz(2 * static_cast<std::size_t>(std::max(k, 1)));
    std::complex<double> wq;
    double rq = 0;
    zheevr_(&jobz, &range, &uplo, &n, A.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, w.data(), Z.data(), &ldz,
            isuppz.data(), &wq, &lwork, &rq, &lrwork, &iw, &liwork, &info, 1, 1, 1);
    lwork = std::max(1, static_cast<int>(wq.real()));
    lrwork = std::max(1, static_cast<int>(rq));
    liwork = std::max(1, iw);
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);
    zheevr_(&jobz, &range, &uplo, &n, A.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, w.data(), Z.data(), &ldz,
            isuppz.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info, 1, 1, 1);
    require(info == 0 && m == k, ErrorCode::NumericalFailure,
            "LAPACK zheevr failed (info = " + std::to_string(info) + ")");
    w.conservativeResize(k);
}

/// True when the Hermitian matrix A - shift I is positive definite (Cholesky succeeds), i.e. min eig(A) > shift.
bool exceeds(const Eigen::MatrixXcd& A, double shift) {
    Eigen::MatrixXcd B = A;
    B.diagonal().array() -= shift;
    Eigen::LLT<Eigen::MatrixXcd, Eigen::Lower> llt(B);
    return llt.info() == Eigen::Success;
}

} // namespace prowl::detail
