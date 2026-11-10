// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file separable.hpp
/// Backend-neutral description of plane-wave factor matrices of separable localized
/// functions, and the scalar kernels used to evaluate them on host and device:
///
///   W_{g f} = pref * coef_f * Y_{lm_f}(q^) * R_{table_f}(|q|) * exp(2 pi i (k + G_g) . tau_f),
///   q = 2 pi sum_i (k + G_g)_i b_i  (Cartesian, 1/Angstrom).
///
/// The chemistry (which functions, their radial tables, i^l factors, Omega^{-1/2}) is decided
/// by ProjectionModel; backends only evaluate this formula. Host and CUDA share the inline
/// kernels below (PROWL_HD), so both evaluate exactly the same expressions.

#include <cmath>
#include <complex>
#include <vector>

#include "prowl/lattice.hpp"

#ifdef __CUDACC__
#define PROWL_HD __host__ __device__ inline
#else
#define PROWL_HD inline
#endif

namespace prowl {

struct SeparableFactorSpec {
    Vec3 k{};          ///< fractional k
    Mat3 b{};          ///< reciprocal rows (no 2 pi)
    double pref{1.0};  ///< global prefactor (Omega^{-1/2})
    const IVec3* G{};  ///< plane-wave Miller indices
    int ng{};
    // per function (column)
    std::vector<double> tau;   ///< 3 * nf, fractional positions
    std::vector<int> lm;       ///< nf, flat index l*l + l + m (l <= 3)
    std::vector<int> table;    ///< nf, radial table index
    std::vector<std::complex<double>> coef; ///< nf, complex constant (i^l, ...)
    // uniform natural cubic-spline tables, concatenated
    std::vector<double> t_x0, t_h, t_xmax; ///< per table
    std::vector<int> t_n, t_offset;        ///< per table: point count and offset into t_y/t_m
    std::vector<double> t_y, t_m;          ///< values and second derivatives
    /// Cache identity for all model data (including b, pref, functions and tables). A nonzero id must change
    /// whenever these data change; k, G and ng may vary independently. Zero disables model-data cache reuse.
    unsigned long long table_id{};
    // Optional extra radial terms (PAW fold): R_f(q) = R_{table_f}(q) + sum_{j in [x_off[f], x_off[f+1])} x_w[j]
    // R_{x_table[j]}(q). Empty x_off: none.
    std::vector<int> x_off;     ///< nf + 1, or empty
    std::vector<int> x_table;   ///< table index of each extra term
    std::vector<double> x_w;    ///< weight of each extra term

    /**
     * @brief Counts factor columns in the model specification.
     * @return lm.size() as an int; valid specifications keep the function count representable.
     */
    [[nodiscard]] int nf() const noexcept { return static_cast<int>(lm.size()); }
    /**
     * @brief Reports whether folded PAW radial terms are described.
     * @return True when x_off is populated, including an all-empty per-function term list.
     */
    [[nodiscard]] bool has_extra() const noexcept { return !x_off.empty(); }
};

namespace detail {

/**
 * @brief Evaluates real harmonics through f degree with the shared host/device convention.
 * @param[in] x Finite Cartesian direction component.
 * @param[in] y Finite Cartesian direction component.
 * @param[in] z Finite Cartesian direction component.
 * @param[out] out Caller-owned array of 16 values in l*l+l+m ordering.
 * Zero direction is treated as +z; buffers must have the stated capacity.
 */
PROWL_HD void real_ylm3(double x, double y, double z, double* out) {
    const double r = sqrt(x * x + y * y + z * z);
    double ct = 1.0, st = 0.0, cp = 1.0, sp = 0.0;
    if (r > 1e-300) {
        ct = z / r;
        ct = ct > 1.0 ? 1.0 : (ct < -1.0 ? -1.0 : ct);
        const double rho = sqrt(x * x + y * y);
        st = rho / r;
        if (rho > 1e-300 * r) { cp = x / rho; sp = y / rho; }
    }
    double P[4][4] = {{0}};
    P[0][0] = 1.0;
    for (int m = 1; m <= 3; ++m) P[m][m] = P[m - 1][m - 1] * (2.0 * m - 1.0) * st;
    for (int m = 0; m < 3; ++m) P[m + 1][m] = (2.0 * m + 1.0) * ct * P[m][m];
    for (int m = 0; m <= 3; ++m)
        for (int l = m + 2; l <= 3; ++l)
            P[l][m] = ((2.0 * l - 1.0) * ct * P[l - 1][m] - (l + m - 1.0) * P[l - 2][m]) / (l - m);
    double cm[4], sm[4];
    cm[0] = 1.0; sm[0] = 0.0;
    for (int m = 1; m <= 3; ++m) {
        cm[m] = cm[m - 1] * cp - sm[m - 1] * sp;
        sm[m] = sm[m - 1] * cp + cm[m - 1] * sp;
    }
    const double four_pi = 12.566370614359172953850573533118;
    for (int l = 0; l <= 3; ++l) {
        double fact_ratio = 1.0;
        out[l * l + l] = sqrt((2.0 * l + 1.0) / four_pi) * P[l][0];
        for (int m = 1; m <= l; ++m) {
            fact_ratio /= static_cast<double>((l + m) * (l - m + 1));
            const double N = sqrt((2.0 * l + 1.0) / four_pi * fact_ratio) * 1.4142135623730950488016887242097;
            out[l * l + l + m] = N * P[l][m] * cm[m];
            out[l * l + l - m] = N * P[l][m] * sm[m];
        }
    }
}

/**
 * @brief Evaluates a natural cubic segment on a uniform grid.
 * @param[in] y Borrowed n function samples.
 * @param[in] m Borrowed n second derivatives.
 * @param[in] n Sample count, at least two.
 * @param[in] x0 First abscissa.
 * @param[in] h Finite positive spacing.
 * @param[in] x Finite query abscissa.
 * @return Interpolated value, or endpoint-segment extrapolation outside the grid.
 * Scalar host/device kernel: grid validity and input/output capacities are caller preconditions.
 */
PROWL_HD double spline_eval(const double* y, const double* m, int n, double x0, double h, double x) {
    const double t = (x - x0) / h;
    int lo = t <= 0 ? 0 : static_cast<int>(t);
    if (lo > n - 2) lo = n - 2;
    const double xlo = x0 + lo * h, xhi = xlo + h;
    const double a = (xhi - x) / h, b = (x - xlo) / h;
    return a * y[lo] + b * y[lo + 1] + ((a * a * a - a) * m[lo] + (b * b * b - b) * m[lo + 1]) * (h * h) / 6.0;
}

/**
 * @brief Evaluates a radial table from the packed specification arrays.
 * @param[in] ty Concatenated sample values.
 * @param[in] tm Concatenated second derivatives.
 * @param[in] tn Point count for each table.
 * @param[in] toff Offset of each table in ty/tm.
 * @param[in] tx0 First abscissa of each table.
 * @param[in] th Positive grid spacing for each table.
 * @param[in] txmax Upper supported abscissa of each table.
 * @param[in] t Valid table index shared by all metadata arrays.
 * @param[in] qn Nonnegative finite wave-number magnitude.
 * @return Spline value, or zero when qn exceeds txmax[t]; valid metadata/array capacities are required.
 */
PROWL_HD double table_value(const double* ty, const double* tm, const int* tn, const int* toff, const double* tx0,
                             const double* th, const double* txmax, int t, double qn) {
    return qn > txmax[t] ? 0.0 : spline_eval(ty + toff[t], tm + toff[t], tn[t], tx0[t], th[t], qn);
}

} // namespace detail

/**
 * @brief Hashes reciprocal-index content for factor-cache lookup.
 * @param[in] G Borrowed ng Miller-index triplets; may be null only when ng is zero.
 * @param[in] ng Nonnegative number of index triplets.
 * @return FNV-style component hash including ng; a content-based cache key independent of pointer reuse.
 */
inline unsigned long long hash_gvectors(const IVec3* G, int ng) {
    unsigned long long h = 1469598103934665603ULL;
    for (int g = 0; g < ng; ++g)
        for (int c = 0; c < 3; ++c) {
            h ^= static_cast<unsigned long long>(static_cast<unsigned int>(G[g][c]));
            h *= 1099511628211ULL;
        }
    return h ^ static_cast<unsigned long long>(ng);
}

/**
 * @brief Generates a column range using shared radial/angular/phase kernels.
 * @param[in] s Valid borrowed specification; G and all table/function arrays must outlive this synchronous call.
 * @param[in] first First column in [0,s.nf()].
 * @param[in] n Nonnegative column count within s.nf().
 * @param[out] W Caller-owned column-major s.ng x n complex output.
 * Uses OpenMP over plane waves. Concurrent immutable specifications with distinct outputs are supported.
 * @throws Error For invalid column bounds or a missing G list.
 */
void evaluate_separable_host(const SeparableFactorSpec& s, int first, int n, std::complex<double>* W);

} // namespace prowl
