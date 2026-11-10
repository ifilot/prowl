// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file backend.hpp
/// Compute backends. Backends evaluate *mathematically identical* operations;
/// chemistry conventions (factor matrices, phases, PAW) are owned by the core
/// and never by a backend. No CUDA headers appear in this interface.

#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace prowl {

struct SeparableFactorSpec;

enum class Precision { Accurate }; ///< FP64 throughout (the only mode accepted for results)

/// Actual owned factor-buffer capacities; library/allocator overhead and coefficients are excluded from the limit.
struct FactorMemoryStats {
    std::size_t limit_bytes{};
    std::size_t retained_bytes{};
    std::size_t peak_bytes{};
};

/// Device evaluation of band-window repair scan steps (analysis.cpp, repaired_window). Holds Tp = X T (norb x ncols)
/// of the lowest bands and PF = Tp_F Tp_F^H of the nf occupied ones. passes(nc) is the fast test of one candidate
/// band count: the top-m selection of the unoccupied candidates A = Tp[:, nf:nc] has relative gap >= gap_min and
/// PF + B B^H (B its projected completion) is positive definite after subtracting cond_min. The caller confirms a
/// passing step through the clear-accept or reference path. Not thread-safe; one stepper per thread.
class RepairStepper {
public:
    /**
     * @brief Releases the stepper and its private stream/workspace resources.
     */
    virtual ~RepairStepper() = default;
    /**
     * @brief Screens one candidate band count with the fast repair predicate.
     * @param[in] nc Candidate prefix length within the uploaded projection array.
     * @return True when screening passes; the caller still confirms with the clear-accept/reference path.
     * Not thread-safe; each concurrent scan owns its stepper.
     */
    virtual bool passes(int nc) = 0;
};

/// Stateful compute context. Calls on the same instance require external serialization.
class Backend {
public:
    /**
     * @brief Releases owned compute resources after all synchronous operations have finished.
     */
    virtual ~Backend() = default;
    /**
     * @brief Identifies the implementation selected for this context.
     * @return Stable backend kind, cpu or cuda for production implementations.
     */
    [[nodiscard]] virtual std::string name() const = 0;
    /**
     * @brief Describes the compute device and numerical mode.
     * @return Owning human-readable hardware/library description for provenance.
     */
    [[nodiscard]] virtual std::string device_description() const = 0;

    /**
     * @brief Sets the aggregate limit for cached and transient separable-factor buffers and resets peak accounting.
     * @param[in] bytes Maximum owned factor-buffer bytes; zero evicts all factors and prevents nonempty generation.
     * Coefficients, projection outputs and numerical-library workspace have separate lifetimes and are excluded.
     */
    virtual void set_factor_memory_limit(std::size_t bytes) = 0;
    /** @brief Reports owned factor capacities and their peak since the most recent limit change.
     * @return Byte counts for the aggregate factor budget.
     */
    [[nodiscard]] virtual FactorMemoryStats factor_memory_stats() const = 0;

    /**
     * @brief Projects full complex plane-wave coefficients without conjugating W.
     * @param[in] W Column-major ng x nf factors; caller retains ownership.
     * @param[in] C Column-major ng x nb coefficients; caller retains ownership.
     * @param[in] ng Positive plane-wave count.
     * @param[in] nf Positive function count.
     * @param[in] nb Positive band count.
     * @param[out] out Caller-owned column-major nf x nb complex result, W^T*C.
     * Buffers must have the stated capacities and not alias the output. Calls are synchronous.
     */
    virtual void project_full(const std::complex<double>* W, const std::complex<double>* C, int ng, int nf, int nb,
                              std::complex<double>* out) = 0;

    /**
     * @brief Projects Gamma-half coefficients through the conjugate-pair identity.
     * @param[in] W Column-major ng x nf half-sphere factors for real local functions.
     * @param[in] C Column-major ng x nb logical half-sphere coefficients in a real gauge.
     * @param[in] weights ng multipliers; physical values are one at G=0 and two otherwise.
     * @param[in] ng Positive stored plane-wave count.
     * @param[in] nf Positive function count.
     * @param[in] nb Positive band count.
     * @param[out] out Column-major nf x nb real result, Re(sum_g weights[g]*W_gf*C_gb).
     * Caller owns all buffers and provides valid nonaliasing capacities. The full sphere is never expanded.
     */
    virtual void project_gamma(const std::complex<double>* W, const std::complex<double>* C, const double* weights,
                               int ng, int nf, int nb, double* out) = 0;

    /**
     * @brief Copies a coefficient block into backend-owned resident storage.
     * @param[in] C Borrowed column-major ng x nb logical coefficients.
     * @param[in] ng Positive plane-wave count.
     * @param[in] nb Positive band count.
     * @param[in] gamma_weights Borrowed ng Gamma multipliers, or nullptr for full complex arithmetic.
     * CPU copies finish on return. CUDA uploads may be queued: keep both input buffers alive and unchanged until
     * the next synchronous projection/evaluation completes. Resident storage survives until the next load,
     * factor-limit reset or backend destruction.
     */
    virtual void load_coefficients(const std::complex<double>* C, int ng, int nb, const double* gamma_weights) = 0;

    /**
     * @brief Generates factor columns and projects the resident coefficient block.
     * @param[in] spec Borrowed valid separable specification; ng must match the most recent coefficient load.
     * @param[in] first Zero-based first factor column.
     * @param[in] n Positive factor-column count within spec.nf().
     * @param[out] out_full Caller-owned column-major n x resident-nb output for full encoding; nullptr for Gamma.
     * @param[out] out_gamma Caller-owned column-major n x resident-nb output for Gamma encoding; nullptr for full.
     * The active output must have valid capacity and not alias inputs. Model data identified by a nonzero table_id stay immutable.
     * @throws Error For invalid ranges, mismatched ng or aggregate factor-budget exhaustion.
     */
    virtual void project_separable(const SeparableFactorSpec& spec, int first, int n, std::complex<double>* out_full,
                                   double* out_gamma) = 0;

    /**
     * @brief Evaluates separable factors for kernel validation without projecting bands.
     * @param[in] spec Borrowed valid specification with model/cache identity rules from separable.hpp.
     * @param[in] first First requested column.
     * @param[in] n Number of columns in the valid range.
     * @param[out] W Caller-owned column-major spec.ng x n complex factors.
     * Any backend-owned temporary factors participate in the factor limit; the caller-owned output does not.
     */
    virtual void evaluate_factors(const SeparableFactorSpec& spec, int first, int n, std::complex<double>* W) = 0;

    /**
     * @brief Reports support for the optional dense Hermitian solve.
     * @return True when hermitian_eig is implemented; false for the CPU reference backend.
     */
    [[nodiscard]] virtual bool has_hermitian_eigensolver() const { return false; }
    /**
     * @brief Diagonalizes a dense Hermitian matrix and checks its numerical residual.
     * @param[in,out] A Borrowed column-major n x n matrix, overwritten by orthonormal eigenvectors.
     * @param[in] n Positive square matrix dimension.
     * @param[out] w Caller-owned n-entry ascending eigenvalue array.
     * Calls are synchronous; the same backend cannot be used concurrently.
     * @throws Error For unavailable support, device/library failure or an excessive FP64 residual.
     */
    virtual void hermitian_eig(std::complex<double>* A, int n, double* w);

    /**
     * @brief Creates an independent device context for a long band-window repair scan.
     * @param[in] Tp Borrowed column-major norb x ncols orthonormalized projections; copied by supported implementations.
     * @param[in] norb Positive local basis size.
     * @param[in] ncols Uploaded candidate-band count.
     * @param[in] nf Occupied prefix length.
     * @param[in] m Selected unoccupied-complement count.
     * @param[in] gap_min Relaxed relative selection-gap threshold.
     * @param[in] cond_min Relaxed minimum window-eigenvalue threshold.
     * @return Owning private stepper, or nullptr when unsupported.
     * Each concurrent scan owns its stepper; the parent backend is borrowed only during construction.
     */
    [[nodiscard]] virtual std::unique_ptr<RepairStepper> make_repair_stepper(const std::complex<double>* Tp,
        int norb, int ncols, int nf, int m, double gap_min, double cond_min);

};

/**
 * @brief Selects an available production compute backend.
 * @param[in] kind cpu, cuda, or auto; auto chooses CUDA when compiled and available.
 * @return Owning stateful context; no external buffers are retained by this factory.
 * @throws Error For unknown kinds or explicitly requested unavailable CUDA.
 */
[[nodiscard]] std::unique_ptr<Backend> make_backend(const std::string& kind);
/**
 * @brief Queries usable production backends in this process.
 * @return Owning list beginning with cpu, with cuda appended only when compiled and a device is usable.
 */
[[nodiscard]] std::vector<std::string> available_backends();

/**
 * @brief Creates the OpenBLAS FP64 reference context.
 * @return Owning CPU backend, independent of other contexts.
 */
std::unique_ptr<Backend> make_cpu_backend();
#ifdef PROWL_HAVE_CUDA
/**
 * @brief Creates a native FP64 CUDA context on a selected device.
 * @param[in] device Zero-based CUDA device index.
 * @return Owning context with private stream, handles and device buffers.
 * @throws Error If device selection or CUDA/library initialization fails.
 */
std::unique_ptr<Backend> make_cuda_backend(int device = 0);
/**
 * @brief Checks whether the CUDA runtime exposes a usable device.
 * @return True when at least one device is visible; false on runtime/driver errors.
 */
bool cuda_device_available();
#endif

} // namespace prowl
