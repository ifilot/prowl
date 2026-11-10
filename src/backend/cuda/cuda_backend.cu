// SPDX-License-Identifier: GPL-3.0-or-later
/// Native NVIDIA CUDA backend (FP64 "accurate" mode only).
///
/// Implements exactly the operations of the CPU reference backend:
///   project_full : out = W^T C                (cublasZgemm, op(W) = T)
///   project_gamma: out = Re(sum_g w_g W C)    (device packing kernel + cublasDgemm on K = 2 ng)
/// Chemistry conventions are not known here. Device buffers persist between
/// calls and grow on demand. Compute-operation statuses are checked and mapped to
/// prowl::Error; resource cleanup is best effort and does not throw. No TF32 or reduced precision is used.
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <complex>
#include <vector>
#include <sstream>
#include <string>

#include "prowl/backend.hpp"
#include "prowl/errors.hpp"
#include "prowl/separable.hpp"

namespace prowl {
namespace {

void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    std::ostringstream os;
    os << what << ": " << cudaGetErrorName(e) << " (" << cudaGetErrorString(e) << ")";
    fail(e == cudaErrorMemoryAllocation ? ErrorCode::OutOfMemory : ErrorCode::DeviceError, os.str());
}

void check(cusolverStatus_t s, const char* what) {
    if (s == CUSOLVER_STATUS_SUCCESS) return;
    std::ostringstream os;
    os << what << ": cuSOLVER status " << static_cast<int>(s);
    fail(s == CUSOLVER_STATUS_ALLOC_FAILED ? ErrorCode::OutOfMemory : ErrorCode::DeviceError, os.str());
}

void check(cublasStatus_t s, const char* what) {
    if (s == CUBLAS_STATUS_SUCCESS) return;
    std::ostringstream os;
    os << what << ": cuBLAS status " << static_cast<int>(s);
    fail(s == CUBLAS_STATUS_ALLOC_FAILED ? ErrorCode::OutOfMemory : ErrorCode::DeviceError, os.str());
}

/// Growable device buffer.
struct DeviceBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    void reset() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
    }
    void ensure(std::size_t n) {
        if (n <= bytes) return;
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
        check(cudaMalloc(&ptr, n), "cudaMalloc");
        bytes = n;
    }
    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
    template <class T>
    T* as() const { return static_cast<T*>(ptr); }
};

/// Wr[f*K + g] = w_g Re W_gf ; Wr[f*K + ng + g] = -w_g Im W_gf   (K = 2 ng)
__global__ void pack_gamma_factors(const double2* W, const double* w, int ng, int nf, double* Wr) {
    const long long K = 2LL * ng;
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<long long>(ng) * nf) return;
    const int f = static_cast<int>(i / ng);
    const int g = static_cast<int>(i % ng);
    const double2 v = W[i];
    Wr[f * K + g] = w[g] * v.x;
    Wr[f * K + ng + g] = -w[g] * v.y;
}

/// Cr[b*K + g] = Re C_gb ; Cr[b*K + ng + g] = Im C_gb
__global__ void pack_gamma_coeffs(const double2* C, int ng, int nb, double* Cr) {
    const long long K = 2LL * ng;
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<long long>(ng) * nb) return;
    const int b = static_cast<int>(i / ng);
    const int g = static_cast<int>(i % ng);
    const double2 v = C[i];
    Cr[b * K + g] = v.x;
    Cr[b * K + ng + g] = v.y;
}

struct DevSpec {
    const int* G; // 3 * ng
    int ng;
    double k0, k1, k2;
    double b[9];
    double pref;
    const double* tau;
    const int* lm;
    const int* table;
    const double2* coef;
    const double* tx0;
    const double* th;
    const double* txmax;
    const int* tn;
    const int* toff;
    const double* ty;
    const double* tm;
    const int* xoff;  // nf + 1 or nullptr (no PAW-fold extra radial terms)
    const int* xtab;
    const double* xw;
};

/// W[j * ng + g] for columns first..first+n-1: one thread per plane wave (coalesced writes).
__global__ void eval_separable(DevSpec s, int first, int n, double2* W) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= s.ng) return;
    const double two_pi = 6.283185307179586476925286766559;
    const double f0 = s.k0 + s.G[3 * g], f1 = s.k1 + s.G[3 * g + 1], f2 = s.k2 + s.G[3 * g + 2];
    double q[3];
    for (int c = 0; c < 3; ++c) q[c] = (f0 * s.b[c] + f1 * s.b[3 + c] + f2 * s.b[6 + c]) * two_pi;
    const double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    double Y[16];
    detail::real_ylm3(q[0], q[1], q[2], Y);
    for (int j = 0; j < n; ++j) {
        const int f = first + j;
        const int t = s.table[f];
        double R = detail::table_value(s.ty, s.tm, s.tn, s.toff, s.tx0, s.th, s.txmax, t, qn);
        if (s.xoff)
            for (int x = s.xoff[f]; x < s.xoff[f + 1]; ++x)
                R += s.xw[x] * detail::table_value(s.ty, s.tm, s.tn, s.toff, s.tx0, s.th, s.txmax, s.xtab[x], qn);
        const double arg = two_pi * (f0 * s.tau[3 * f] + f1 * s.tau[3 * f + 1] + f2 * s.tau[3 * f + 2]);
        double sn, cs;
        sincos(arg, &sn, &cs);
        const double a = s.pref * Y[s.lm[f]] * R;
        const double2 c = s.coef[f];
        // a * c * (cs + i sn)
        W[static_cast<long long>(j) * s.ng + g] = make_double2(a * (c.x * cs - c.y * sn), a * (c.x * sn + c.y * cs));
    }
}

/// G <- G - I (n x n, column-major)
__global__ void subtract_identity(double2* G, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) G[static_cast<long long>(i) * n + i].x -= 1.0;
}

/// A[i, i] -= s (n x n, leading dimension ld)
__global__ void shift_diagonal(double2* A, int n, int ld, double s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[static_cast<long long>(i) * ld + i].x -= s;
}

/// B[:, j] = V[:, off + j] * sqrt(max(w[off + j], 0)), j < m (n rows, leading dimension ld for V and B)
__global__ void scale_columns_sqrt(const double2* V, const double* w, int n, int ld, int off, int m, double2* B) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<long long>(n) * m) return;
    const int j = static_cast<int>(i / n), r = static_cast<int>(i % n);
    const double s = sqrt(fmax(w[off + j], 0.0));
    const double2 v = V[static_cast<long long>(off + j) * ld + r];
    B[static_cast<long long>(j) * ld + r] = make_double2(s * v.x, s * v.y);
}

/// Band-window repair steps on the device; mirrors passes() in analysis.cpp (u <= norb: eigenproblem of A^H A,
/// grown by rows; u > norb: of A A^H, grown by rank-k updates). Own stream and handles; FP64 only.
class CudaRepairStepper final : public RepairStepper {
public:
    CudaRepairStepper(int device, const std::complex<double>* Tp, int norb, int ncols, int nf, int m, double gap_min,
                      double cond_min)
        : device_(device), norb_(norb), ncols_(ncols), nf_(nf), m_(m), gap_min_(gap_min) {
        check(cudaSetDevice(device_), "cudaSetDevice");
        check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        check(cublasCreate(&blas_), "cublasCreate");
        check(cublasSetStream(blas_, stream_), "cublasSetStream");
        check(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        check(cusolverDnCreate(&solver_), "cusolverDnCreate");
        check(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
        const std::size_t nn = static_cast<std::size_t>(norb) * norb;
        dTp_.ensure(sizeof(double2) * static_cast<std::size_t>(norb) * ncols);
        for (DeviceBuffer* b : {&dPFs_, &dMu_, &dG_, &dH_, &dM_, &dB_}) b->ensure(sizeof(double2) * std::max<std::size_t>(nn, 1));
        dw_.ensure(sizeof(double) * std::max(norb, 1));
        dinfo_.ensure(sizeof(int));
        check(cudaMemcpyAsync(dTp_.ptr, Tp, sizeof(double2) * static_cast<std::size_t>(norb) * ncols,
                              cudaMemcpyHostToDevice, stream_), "H2D Tp");
        // PFs = Tp_F Tp_F^H - cond_min I (lower triangle)
        const double one = 1.0, zero = 0.0;
        check(cublasZherk(blas_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, norb, nf, &one, dTp_.as<cuDoubleComplex>(), norb,
                          &zero, dPFs_.as<cuDoubleComplex>(), norb), "zherk PF");
        shift_diagonal<<<(norb + 255) / 256, 256, 0, stream_>>>(dPFs_.as<double2>(), norb, norb, cond_min);
        check(cudaGetLastError(), "shift_diagonal launch");
        int lw1 = 0, lw2 = 0;
        check(cusolverDnZheevd_bufferSize(solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, norb,
                                          dH_.as<cuDoubleComplex>(), norb, dw_.as<double>(), &lw1), "zheevd_bufferSize");
        check(cusolverDnZpotrf_bufferSize(solver_, CUBLAS_FILL_MODE_LOWER, norb, dM_.as<cuDoubleComplex>(), norb, &lw2),
              "zpotrf_bufferSize");
        lwork_ = std::max({lw1, lw2, 1});
        dwork_.ensure(sizeof(double2) * static_cast<std::size_t>(lwork_));
        check(cudaStreamSynchronize(stream_), "stream sync");
    }
    ~CudaRepairStepper() override {
        if (solver_) cusolverDnDestroy(solver_);
        if (blas_) cublasDestroy(blas_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    CudaRepairStepper(const CudaRepairStepper&) = delete;
    CudaRepairStepper& operator=(const CudaRepairStepper&) = delete;

    bool passes(int nc) override {
        check(cudaSetDevice(device_), "cudaSetDevice");
        require(nc > nf_ && nc <= ncols_, ErrorCode::InvalidArgument, "repair stepper: band count out of range");
        const int n = norb_, u = nc - nf_, m = m_;
        const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0};
        const double rone = 1.0;
        auto* Tp = dTp_.as<cuDoubleComplex>();
        auto* A = Tp + static_cast<std::size_t>(nf_) * n; // unoccupied candidates, ld n
        auto* H = dH_.as<cuDoubleComplex>();
        auto* B = dB_.as<cuDoubleComplex>();
        int dim = 0, lo_index = 0; // eigenproblem size; index of the (m+1)-th largest eigenvalue
        if (u <= n) {
            auto* Mu = dMu_.as<cuDoubleComplex>(); // ld n, lower triangle valid
            if (mu_nc_ <= nf_ || mu_nc_ > nc) {
                check(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, u, u, n, &one, A, n, A, n, &zero, Mu, n), "zgemm Mu");
            } else if (mu_nc_ < nc) { // new rows uo..u-1: Mu[uo:u, 0:u] = An^H A
                const int uo = mu_nc_ - nf_, du = nc - mu_nc_;
                check(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, du, u, n, &one, A + static_cast<std::size_t>(uo) * n, n,
                                  A, n, &zero, Mu + uo, n), "zgemm Mu rows");
            }
            mu_nc_ = nc;
            check(cudaMemcpy2DAsync(H, sizeof(double2) * n, Mu, sizeof(double2) * n, sizeof(double2) * u, u,
                                    cudaMemcpyDeviceToDevice, stream_), "copy Mu");
            dim = u;
        } else {
            auto* G = dG_.as<cuDoubleComplex>();
            if (g_nc_ <= nf_ || g_nc_ > nc) {
                const double rzero = 0.0;
                check(cublasZherk(blas_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, u, &rone, A, n, &rzero, G, n), "zherk G");
            } else if (g_nc_ < nc) {
                check(cublasZherk(blas_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, nc - g_nc_, &rone,
                                  Tp + static_cast<std::size_t>(g_nc_) * n, n, &rone, G, n), "zherk G update");
            }
            g_nc_ = nc;
            check(cudaMemcpyAsync(H, G, sizeof(double2) * static_cast<std::size_t>(n) * n, cudaMemcpyDeviceToDevice,
                                  stream_), "copy G");
            dim = n;
        }
        check(cusolverDnZheevd(solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, dim, H, n, dw_.as<double>(),
                               dwork_.as<cuDoubleComplex>(), lwork_, dinfo_.as<int>()), "cusolverDnZheevd");
        lo_index = dim - m - 1;
        double wtop[2] = {0.0, 0.0};
        int info = 0;
        check(cudaMemcpyAsync(&info, dinfo_.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream_), "D2H info");
        if (lo_index >= 0)
            check(cudaMemcpyAsync(wtop, dw_.as<double>() + lo_index, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream_),
                  "D2H w");
        check(cudaStreamSynchronize(stream_), "stream sync");
        require(info == 0, ErrorCode::NumericalFailure, "cusolverDnZheevd failed (info = " + std::to_string(info) + ")");
        const double gap = lo_index >= 0 ? (wtop[1] - wtop[0]) / std::max(wtop[1], 1e-300) : 1.0;
        if (gap < gap_min_) return false;
        if (u <= n) { // B = A V_top (n x m)
            check(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, u, &one, A, n, H + static_cast<std::size_t>(u - m) * n,
                              n, &zero, B, n), "zgemm B");
        } else { // B = V_top sqrt(w_top)
            const long long tot = static_cast<long long>(n) * m;
            scale_columns_sqrt<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, stream_>>>(
                dH_.as<double2>(), dw_.as<double>(), n, n, n - m, m, dB_.as<double2>());
            check(cudaGetLastError(), "scale_columns_sqrt launch");
        }
        auto* M = dM_.as<cuDoubleComplex>();
        check(cudaMemcpyAsync(M, dPFs_.ptr, sizeof(double2) * static_cast<std::size_t>(n) * n, cudaMemcpyDeviceToDevice,
                              stream_), "copy PF");
        check(cublasZherk(blas_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m, &rone, B, n, &rone, M, n), "zherk M");
        check(cusolverDnZpotrf(solver_, CUBLAS_FILL_MODE_LOWER, n, M, n, dwork_.as<cuDoubleComplex>(), lwork_,
                               dinfo_.as<int>()), "cusolverDnZpotrf");
        check(cudaMemcpyAsync(&info, dinfo_.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream_), "D2H info");
        check(cudaStreamSynchronize(stream_), "stream sync");
        require(info >= 0, ErrorCode::NumericalFailure, "cusolverDnZpotrf failed (info = " + std::to_string(info) + ")");
        return info == 0; // > 0: not positive definite, the condition fails by more than the margin
    }

private:
    int device_, norb_, ncols_, nf_, m_;
    double gap_min_;
    int mu_nc_ = 0, g_nc_ = 0, lwork_ = 1;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    DeviceBuffer dTp_, dPFs_, dMu_, dG_, dH_, dM_, dB_, dw_, dinfo_, dwork_;
};

class CudaBackend final : public Backend {
public:
    explicit CudaBackend(int device) : device_(device) {
        check(cudaSetDevice(device_), "cudaSetDevice");
        check(cudaGetDeviceProperties(&prop_, device_), "cudaGetDeviceProperties");
        check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        check(cublasCreate(&handle_), "cublasCreate");
        check(cublasSetStream(handle_, stream_), "cublasSetStream");
        // FP64 everywhere; forbid tensor-op down-conversion explicitly.
        check(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    }
    ~CudaBackend() override {
        if (solver_) cusolverDnDestroy(solver_);
        if (handle_) cublasDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    std::string name() const override { return "cuda"; }
    void set_factor_memory_limit(std::size_t limit) override {
        check(cudaSetDevice(device_), "cudaSetDevice");
        check(cudaStreamSynchronize(stream_), "stream sync");
        for (DeviceBuffer* b : {&dSW_, &dWr_, &dW_, &dC_, &dCr_, &dLC_, &dLCr_, &dLw_, &dO_}) b->reset();
        factor_limit_ = limit;
        factor_peak_ = 0;
        ng_ = nb_ = 0;
    }
    FactorMemoryStats factor_memory_stats() const override {
        return {factor_limit_, dSW_.bytes + dWr_.bytes, factor_peak_};
    }
    std::string device_description() const override {
        std::ostringstream os;
        int rt = 0, drv = 0;
        cudaRuntimeGetVersion(&rt);
        cudaDriverGetVersion(&drv);
        os << prop_.name << " (sm_" << prop_.major << prop_.minor << ", " << (prop_.totalGlobalMem >> 20)
           << " MiB, CUDA runtime " << rt / 1000 << "." << (rt % 1000) / 10 << ", driver " << drv / 1000 << "."
           << (drv % 1000) / 10 << ", FP64 pedantic math)";
        return os.str();
    }

    void project_full(const std::complex<double>* W, const std::complex<double>* C, int ng, int nf, int nb,
                      std::complex<double>* out) override {
        const std::size_t wb = sizeof(double2) * static_cast<std::size_t>(ng) * nf;
        const std::size_t cb = sizeof(double2) * static_cast<std::size_t>(ng) * nb;
        const std::size_t ob = sizeof(double2) * static_cast<std::size_t>(nf) * nb;
        dW_.ensure(wb);
        dC_.ensure(cb);
        dO_.ensure(ob);
        check(cudaMemcpyAsync(dW_.ptr, W, wb, cudaMemcpyHostToDevice, stream_), "H2D W");
        check(cudaMemcpyAsync(dC_.ptr, C, cb, cudaMemcpyHostToDevice, stream_), "H2D C");
        const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0};
        check(cublasZgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, nf, nb, ng, &one, dW_.as<cuDoubleComplex>(), ng,
                          dC_.as<cuDoubleComplex>(), ng, &zero, dO_.as<cuDoubleComplex>(), nf),
              "cublasZgemm");
        check(cudaMemcpyAsync(out, dO_.ptr, ob, cudaMemcpyDeviceToHost, stream_), "D2H out");
        check(cudaStreamSynchronize(stream_), "stream sync");
    }

    void project_gamma(const std::complex<double>* W, const std::complex<double>* C, const double* weights, int ng,
                       int nf, int nb, double* out) override {
        const std::size_t K = 2 * static_cast<std::size_t>(ng);
        dW_.ensure(sizeof(double2) * static_cast<std::size_t>(ng) * nf);
        dC_.ensure(sizeof(double2) * static_cast<std::size_t>(ng) * nb);
        dWr_.ensure(sizeof(double) * K * nf);
        dCr_.ensure(sizeof(double) * K * nb);
        dw_.ensure(sizeof(double) * ng);
        dO_.ensure(sizeof(double) * static_cast<std::size_t>(nf) * nb);
        check(cudaMemcpyAsync(dW_.ptr, W, sizeof(double2) * static_cast<std::size_t>(ng) * nf, cudaMemcpyHostToDevice, stream_), "H2D W");
        check(cudaMemcpyAsync(dC_.ptr, C, sizeof(double2) * static_cast<std::size_t>(ng) * nb, cudaMemcpyHostToDevice, stream_), "H2D C");
        check(cudaMemcpyAsync(dw_.ptr, weights, sizeof(double) * ng, cudaMemcpyHostToDevice, stream_), "H2D w");
        const int threads = 256;
        const long long nW = static_cast<long long>(ng) * nf, nC = static_cast<long long>(ng) * nb;
        pack_gamma_factors<<<static_cast<unsigned>((nW + threads - 1) / threads), threads, 0, stream_>>>(
            dW_.as<double2>(), dw_.as<double>(), ng, nf, dWr_.as<double>());
        check(cudaGetLastError(), "pack_gamma_factors launch");
        pack_gamma_coeffs<<<static_cast<unsigned>((nC + threads - 1) / threads), threads, 0, stream_>>>(
            dC_.as<double2>(), ng, nb, dCr_.as<double>());
        check(cudaGetLastError(), "pack_gamma_coeffs launch");
        const double one = 1.0, zero = 0.0;
        check(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, nf, nb, static_cast<int>(K), &one, dWr_.as<double>(),
                          static_cast<int>(K), dCr_.as<double>(), static_cast<int>(K), &zero, dO_.as<double>(), nf),
              "cublasDgemm");
        check(cudaMemcpyAsync(out, dO_.ptr, sizeof(double) * static_cast<std::size_t>(nf) * nb, cudaMemcpyDeviceToHost, stream_), "D2H out");
        check(cudaStreamSynchronize(stream_), "stream sync");
    }

    bool has_hermitian_eigensolver() const override { return true; }

    std::unique_ptr<RepairStepper> make_repair_stepper(const std::complex<double>* Tp, int norb, int ncols, int nf, int m,
                                                       double gap_min, double cond_min) override {
        return std::make_unique<CudaRepairStepper>(device_, Tp, norb, ncols, nf, m, gap_min, cond_min);
    }

    /// cusolverDnZheevd + device residual check ||A V - V diag(w)||_F / ||A||_F.
    void hermitian_eig(std::complex<double>* A, int n, double* w) override {
        if (!solver_) {
            check(cusolverDnCreate(&solver_), "cusolverDnCreate");
            check(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
        }
        const std::size_t bytes = sizeof(double2) * static_cast<std::size_t>(n) * n;
        dEA_.ensure(bytes);
        dEA0_.ensure(bytes);
        dER_.ensure(bytes);
        dEw_.ensure(sizeof(double) * std::max(n, 1));
        dEwc_.ensure(sizeof(double2) * std::max(n, 1));
        dEinfo_.ensure(sizeof(int));
        check(cudaMemcpyAsync(dEA_.ptr, A, bytes, cudaMemcpyHostToDevice, stream_), "H2D A");
        check(cudaMemcpyAsync(dEA0_.ptr, dEA_.ptr, bytes, cudaMemcpyDeviceToDevice, stream_), "D2D A");
        int lwork = 0;
        check(cusolverDnZheevd_bufferSize(solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
                                          dEA_.as<cuDoubleComplex>(), n, dEw_.as<double>(), &lwork),
              "zheevd_bufferSize");
        dEwork_.ensure(sizeof(double2) * std::max(lwork, 1));
        check(cusolverDnZheevd(solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dEA_.as<cuDoubleComplex>(), n,
                               dEw_.as<double>(), dEwork_.as<cuDoubleComplex>(), lwork, dEinfo_.as<int>()),
              "cusolverDnZheevd");
        int info = 0;
        check(cudaMemcpyAsync(&info, dEinfo_.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream_), "D2H info");
        check(cudaMemcpyAsync(w, dEw_.ptr, sizeof(double) * n, cudaMemcpyDeviceToHost, stream_), "D2H w");
        check(cudaStreamSynchronize(stream_), "stream sync");
        require(info == 0, ErrorCode::NumericalFailure, "cusolverDnZheevd failed (info = " + std::to_string(info) + ")");
        // residual: R = A0 V - V diag(w)
        std::vector<double2> wc(n);
        for (int i = 0; i < n; ++i) wc[i] = make_double2(w[i], 0.0);
        check(cudaMemcpyAsync(dEwc_.ptr, wc.data(), sizeof(double2) * n, cudaMemcpyHostToDevice, stream_), "H2D wc");
        const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0}, mone{-1.0, 0.0};
        check(cublasZgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dEA0_.as<cuDoubleComplex>(), n,
                          dEA_.as<cuDoubleComplex>(), n, &zero, dER_.as<cuDoubleComplex>(), n), "residual gemm");
        double anorm = 0.0;
        check(cublasDznrm2(handle_, n * n, dEA0_.as<cuDoubleComplex>(), 1, &anorm), "norm A");
        // dEA0 <- V diag(w); R <- R - dEA0
        check(cublasZdgmm(handle_, CUBLAS_SIDE_RIGHT, n, n, dEA_.as<cuDoubleComplex>(), n, dEwc_.as<cuDoubleComplex>(), 1,
                          dEA0_.as<cuDoubleComplex>(), n), "zdgmm");
        check(cublasZaxpy(handle_, n * n, &mone, dEA0_.as<cuDoubleComplex>(), 1, dER_.as<cuDoubleComplex>(), 1), "zaxpy");
        double rnorm = 0.0;
        check(cublasDznrm2(handle_, n * n, dER_.as<cuDoubleComplex>(), 1, &rnorm), "norm R");
        // orthonormality: ||V^H V - I||_F (S^{-1/2} = V diag(w^p) V^H requires orthonormal V)
        check(cublasZgemm(handle_, CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, &one, dEA_.as<cuDoubleComplex>(), n,
                          dEA_.as<cuDoubleComplex>(), n, &zero, dER_.as<cuDoubleComplex>(), n), "orthonormality gemm");
        subtract_identity<<<(n + 255) / 256, 256, 0, stream_>>>(dER_.as<double2>(), n);
        check(cudaGetLastError(), "subtract_identity launch");
        double onorm = 0.0;
        check(cublasDznrm2(handle_, n * n, dER_.as<cuDoubleComplex>(), 1, &onorm), "norm V^H V - I");
        check(cudaMemcpyAsync(A, dEA_.ptr, bytes, cudaMemcpyDeviceToHost, stream_), "D2H V");
        check(cudaStreamSynchronize(stream_), "stream sync");
        require(onorm < 1e-10 * std::sqrt(static_cast<double>(std::max(n, 1))), ErrorCode::NumericalFailure,
                "GPU eigenvectors not orthonormal (||V^H V - I||_F = " + std::to_string(onorm) + ")");
        last_eig_residual_ = anorm > 0 ? rnorm / anorm : rnorm;
        require(last_eig_residual_ < 1e-10 * std::max(1, n / 100), ErrorCode::NumericalFailure,
                "GPU eigensolver residual " + std::to_string(last_eig_residual_) + " exceeds FP64 tolerance");
    }

    void load_coefficients(const std::complex<double>* C, int ng, int nb, const double* gamma_weights) override {
        ng_ = ng;
        nb_ = nb;
        gamma_ = gamma_weights != nullptr;
        const std::size_t cb = sizeof(double2) * static_cast<std::size_t>(ng) * nb;
        dLC_.ensure(cb);
        check(cudaMemcpyAsync(dLC_.ptr, C, cb, cudaMemcpyHostToDevice, stream_), "H2D C");
        if (gamma_) {
            const std::size_t K = 2 * static_cast<std::size_t>(ng);
            dLCr_.ensure(sizeof(double) * K * nb);
            dLw_.ensure(sizeof(double) * ng);
            check(cudaMemcpyAsync(dLw_.ptr, gamma_weights, sizeof(double) * ng, cudaMemcpyHostToDevice, stream_), "H2D w");
            const long long nC = static_cast<long long>(ng) * nb;
            pack_gamma_coeffs<<<static_cast<unsigned>((nC + 255) / 256), 256, 0, stream_>>>(dLC_.as<double2>(), ng, nb,
                                                                                         dLCr_.as<double>());
            check(cudaGetLastError(), "pack_gamma_coeffs launch");
        }
    }

    void evaluate_factors(const SeparableFactorSpec& spec, int first, int n, std::complex<double>* W) override {
        const std::size_t wb = sizeof(double2) * static_cast<std::size_t>(spec.ng) * n;
        ensure_factor_buffers(wb, false);
        launch_eval(spec, first, n);
        check(cudaMemcpyAsync(W, dSW_.ptr, wb, cudaMemcpyDeviceToHost, stream_), "D2H W");
        check(cudaStreamSynchronize(stream_), "stream sync");
    }

    void project_separable(const SeparableFactorSpec& spec, int first, int n, std::complex<double>* out_full,
                           double* out_gamma) override {
        require(spec.ng == ng_, ErrorCode::InvalidArgument, "factor spec and resident coefficients differ in N_G");
        require(first >= 0 && n > 0 && first <= spec.nf() && n <= spec.nf() - first,
                ErrorCode::InvalidArgument, "factor column range out of bounds");
        ensure_factor_buffers(sizeof(double2) * static_cast<std::size_t>(ng_) * n, gamma_);
        launch_eval(spec, first, n);
        if (!gamma_) {
            dO_.ensure(sizeof(double2) * static_cast<std::size_t>(n) * nb_);
            const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0};
            check(cublasZgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, nb_, ng_, &one, dSW_.as<cuDoubleComplex>(), ng_,
                              dLC_.as<cuDoubleComplex>(), ng_, &zero, dO_.as<cuDoubleComplex>(), n),
                  "cublasZgemm");
            check(cudaMemcpyAsync(out_full, dO_.ptr, sizeof(double2) * static_cast<std::size_t>(n) * nb_,
                                  cudaMemcpyDeviceToHost, stream_), "D2H out");
        } else {
            const std::size_t K = 2 * static_cast<std::size_t>(ng_);
            dWr_.ensure(sizeof(double) * K * n);
            dO_.ensure(sizeof(double) * static_cast<std::size_t>(n) * nb_);
            const long long nW = static_cast<long long>(ng_) * n;
            pack_gamma_factors<<<static_cast<unsigned>((nW + 255) / 256), 256, 0, stream_>>>(
                dSW_.as<double2>(), dLw_.as<double>(), ng_, n, dWr_.as<double>());
            check(cudaGetLastError(), "pack_gamma_factors launch");
            const double one = 1.0, zero = 0.0;
            check(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, nb_, static_cast<int>(K), &one, dWr_.as<double>(),
                              static_cast<int>(K), dLCr_.as<double>(), static_cast<int>(K), &zero, dO_.as<double>(), n),
                  "cublasDgemm");
            check(cudaMemcpyAsync(out_gamma, dO_.ptr, sizeof(double) * static_cast<std::size_t>(n) * nb_,
                                  cudaMemcpyDeviceToHost, stream_), "D2H out");
        }
        check(cudaStreamSynchronize(stream_), "stream sync");
    }

private:
    int device_;
    cudaDeviceProp prop_{};
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    DeviceBuffer dEA_, dEA0_, dER_, dEw_, dEwc_, dEinfo_, dEwork_;
    double last_eig_residual_ = 0.0;
    DeviceBuffer dW_, dC_, dO_, dWr_, dCr_, dw_;
    // resident coefficient block and separable-factor state
    DeviceBuffer dLC_, dLCr_, dLw_, dSW_;
    DeviceBuffer dG_, dTau_, dLm_, dTab_, dCoef_, dTx0_, dTh_, dTxmax_, dTn_, dToff_, dTy_, dTm_, dXoff_, dXtab_, dXw_;
    int ng_ = 0, nb_ = 0;
    bool gamma_ = false;
    std::size_t factor_limit_ = std::size_t{1} << 31, factor_peak_ = 0;
    unsigned long long cachedGhash_ = 0;
    int cachedNg_ = -1;
    const void* cachedFun_ = nullptr;
    int cachedNf_ = -1;
    unsigned long long cachedTables_ = 0;

    void ensure_factor_buffers(std::size_t bytes, bool packed) {
        require(bytes <= factor_limit_ / (packed ? 2u : 1u), ErrorCode::OutOfMemory,
                "factor generation exceeds the aggregate factor memory limit");
        if (!packed) dWr_.reset();
        const std::size_t wr = packed ? std::max(dWr_.bytes, bytes) : 0;
        if (wr > factor_limit_ || std::max(dSW_.bytes, bytes) > factor_limit_ - wr) {
            dSW_.reset();
            dWr_.reset();
        }
        dSW_.ensure(bytes);
        if (packed) dWr_.ensure(bytes);
        factor_peak_ = std::max(factor_peak_, dSW_.bytes + dWr_.bytes);
    }

    template <class T>
    void upload(DeviceBuffer& d, const T* src, std::size_t count) {
        d.ensure(sizeof(T) * std::max<std::size_t>(count, 1));
        check(cudaMemcpyAsync(d.ptr, src, sizeof(T) * count, cudaMemcpyHostToDevice, stream_), "H2D spec");
    }

    void launch_eval(const SeparableFactorSpec& s, int first, int n) {
        require(first >= 0 && first + n <= s.nf(), ErrorCode::InvalidArgument, "factor column range out of bounds");
        const unsigned long long gh = hash_gvectors(s.G, s.ng);
        if (gh != cachedGhash_ || s.ng != cachedNg_) {
            upload(dG_, reinterpret_cast<const int*>(s.G), 3 * static_cast<std::size_t>(s.ng));
            cachedGhash_ = gh;
            cachedNg_ = s.ng;
        }
        const bool new_model = s.table_id != cachedTables_ || s.table_id == 0;
        if (new_model || s.tau.data() != cachedFun_ || s.nf() != cachedNf_) {
            upload(dTau_, s.tau.data(), s.tau.size());
            upload(dLm_, s.lm.data(), s.lm.size());
            upload(dTab_, s.table.data(), s.table.size());
            std::vector<double2> c(s.coef.size());
            for (std::size_t i = 0; i < c.size(); ++i) c[i] = make_double2(s.coef[i].real(), s.coef[i].imag());
            upload(dCoef_, c.data(), c.size());
            if (s.has_extra()) {
                upload(dXoff_, s.x_off.data(), s.x_off.size());
                upload(dXtab_, s.x_table.data(), s.x_table.size());
                upload(dXw_, s.x_w.data(), s.x_w.size());
            }
            check(cudaStreamSynchronize(stream_), "stream sync"); // c is a temporary
            cachedFun_ = s.tau.data();
            cachedNf_ = s.nf();
        }
        if (new_model) {
            upload(dTx0_, s.t_x0.data(), s.t_x0.size());
            upload(dTh_, s.t_h.data(), s.t_h.size());
            upload(dTxmax_, s.t_xmax.data(), s.t_xmax.size());
            upload(dTn_, s.t_n.data(), s.t_n.size());
            upload(dToff_, s.t_offset.data(), s.t_offset.size());
            upload(dTy_, s.t_y.data(), s.t_y.size());
            upload(dTm_, s.t_m.data(), s.t_m.size());
            cachedTables_ = s.table_id;
        }
        DevSpec d{dG_.as<int>(), s.ng, s.k[0], s.k[1], s.k[2], {}, s.pref, dTau_.as<double>(), dLm_.as<int>(),
                  dTab_.as<int>(), dCoef_.as<double2>(), dTx0_.as<double>(), dTh_.as<double>(), dTxmax_.as<double>(),
                  dTn_.as<int>(), dToff_.as<int>(), dTy_.as<double>(), dTm_.as<double>(),
                  s.has_extra() ? dXoff_.as<int>() : nullptr, s.has_extra() ? dXtab_.as<int>() : nullptr,
                  s.has_extra() ? dXw_.as<double>() : nullptr};
        for (int i = 0; i < 3; ++i)
            for (int c = 0; c < 3; ++c) d.b[3 * i + c] = s.b[i][c];
        eval_separable<<<static_cast<unsigned>((s.ng + 127) / 128), 128, 0, stream_>>>(d, first, n, dSW_.as<double2>());
        check(cudaGetLastError(), "eval_separable launch");
    }
};

} // namespace

bool cuda_device_available() {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        cudaGetLastError(); // clear sticky "no device/driver" error
        return false;
    }
    return n > 0;
}

std::unique_ptr<Backend> make_cuda_backend(int device) { return std::make_unique<CudaBackend>(device); }

} // namespace prowl
