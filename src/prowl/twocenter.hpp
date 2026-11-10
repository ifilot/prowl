// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file twocenter.hpp
/// Two-centre overlap integrals of STO-type functions,
///   O_{m1 m2}(d) = int R1(|r|) Y_{l1 m1}(r^) R2(|r-d|) Y_{l2 m2}((r-d)^) d^3r,
/// evaluated exactly in angle via real Gaunt coefficients and numerically in
/// |q| (Simpson, analytic integrands):
///   O = 8 sum_{L,M} i^{l1-l2-L} G(l1 m1, l2 m2, L M) Y_LM(d^) int_0^qmax q^2 Rt1 Rt2 j_L(q d) dq.
/// For d = 0 a log-grid radial quadrature of the analytic functions is used.

#include <vector>

#include "prowl/basis.hpp"
#include "prowl/radial.hpp"

namespace prowl {

/// Per-shell data shared by all overlap tables of that shell: Rt(q) on the q grid, R(r) on the
/// same-centre log grid, and the extent. Computing them once per shell (not once per pair) is exact.
struct TwoCenterShell {
    int l{};
    double qmax{};
    int nq{};
    double extent{};
    std::vector<double> ft;   ///< Rt(q_i), q_i = i qmax / (nq - 1)
    std::vector<double> rlog; ///< R(r_i) on the 4001-point log grid 1e-7..80
    /**
     * @brief Precomputes one radial shell for reuse across two-centre shell pairs.
     * @param[in] r Borrowed valid STO radial shell; samples are copied into the result.
     * @param[in] qmax Positive wave-number cutoff in inverse Angstrom.
     * @param[in] nq Odd number of q samples, at least five.
     * @param[in] extent_threshold Positive threshold for radial extent screening.
     * @return Owning Fourier/log-grid samples and a threshold-based support estimate.
     * @throws Error For an invalid sample count.
     */
    static TwoCenterShell prepare(const RadialSto& r, double qmax, int nq, double extent_threshold);
};

class TwoCenterOverlap {
public:
    /**
     * @brief Builds an overlap evaluator from reusable prepared-shell samples.
     * @param[in] a Borrowed first shell, on the same q grid as b.
     * @param[in] b Borrowed second shell, copied during construction.
     * @param[in] dd Nonnegative distance spacing in Angstrom; zero evaluates radial integrals on demand.
     * The evaluator owns its derived tables; input samples need only outlive construction.
     * @throws Error For incompatible q grids or unsupported combined degree.
     */
    TwoCenterOverlap(const TwoCenterShell& a, const TwoCenterShell& b, double dd);
    /**
     * @brief Prepares STO shells and builds their angular and radial overlap tables.
     * @param[in] a Borrowed first radial shell.
     * @param[in] b Borrowed second radial shell.
     * @param[in] qmax Positive radial Fourier cutoff in inverse Angstrom.
     * @param[in] nq Odd q sample count of at least five.
     * @param[in] dd Nonnegative distance-table spacing in Angstrom; zero integrates on demand.
     * @param[in] extent_threshold Positive support-screening threshold.
     * Construction copies derived data; original shells may be destroyed afterwards.
     * @throws Error For invalid sampling or unsupported combined angular degree.
     */
    TwoCenterOverlap(const RadialSto& a, const RadialSto& b, double qmax = 200.0, int nq = 8001, double dd = 0.0,
                     double extent_threshold = 1e-9);
    /**
     * @brief Evaluates the real overlap block for a Cartesian separation.
     * @param[in] d Displacement of the second centre from the first, in Angstrom.
     * @param[out] out Caller-owned (2*l1+1)*(2*l2+1) real array, row-major with first-shell m slow.
     * Zero separation uses log-grid radial quadrature; other separations use splines or direct integration.
     * cutoff() is a screening hint, not an enforced absolute overlap bound. Concurrent calls with separate outputs are supported.
     */
    void evaluate(const Vec3& d, double* out) const;
    /**
     * @brief Exposes the threshold-based sum of radial support extents.
     * @return Screening distance in Angstrom; evaluate does not enforce it or guarantee a rigorous tail bound.
     */
    [[nodiscard]] double cutoff() const noexcept { return cutoff_; }
    /**
     * @brief Reports the angular degree of the first shell.
     * @return First shell l, fixing the output row count to 2*l+1.
     */
    [[nodiscard]] int l1() const noexcept { return l1_; }
    /**
     * @brief Reports the angular degree of the second shell.
     * @return Second shell l, fixing the output column count to 2*l+1.
     */
    [[nodiscard]] int l2() const noexcept { return l2_; }

private:
    int l1_, l2_;
    double same_center_{}; ///< log-grid quadrature of int R1 R2 r^2 dr
    double cutoff_{};
    std::vector<double> q_, g_; ///< q grid and Simpson-weighted q^2 Rt1 Rt2
    std::size_t nq_eff_{};      ///< q points up to the last one whose tail contributes > 1e-15 of sum |g|
    std::vector<radial::CubicSpline> jtab_; ///< per L (index L), empty when not tabulated
    double tab_dmin_{};
    /// Nonzero angular couplings in evaluation order (L, M, m1, m2): out[out] += c * Y_LM * J_L, c = 8 i^p G.
    struct Coupling {
        int out, y, L;
        double c;
    };
    std::vector<Coupling> couplings_;
    void radial_integrals(double dist, double* J) const;
    /// J_L for n <= 8 distances at once (independent sin/cos recurrences per lane; same formulas as above).
    /// J: n x (l1 + l2 + 1), row-major.
    void radial_integrals_block(const double* dist, int n, double* J) const;
};

} // namespace prowl
