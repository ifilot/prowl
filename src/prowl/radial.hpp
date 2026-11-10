// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file radial.hpp
/// Radial numerics: spherical Bessel functions, cubic splines, quadrature on
/// logarithmic grids.

#include <vector>

namespace prowl::radial {

/**
 * @brief Evaluates a spherical Bessel function with series or stable recurrences.
 * @param[in] l Angular degree in [0,12].
 * @param[in] x Finite nonnegative argument.
 * @return j_l(x), with relative accuracy around 1e-14 away from zeros.
 * The noexcept scalar kernel requires valid arguments; Python validates them at its boundary.
 */
[[nodiscard]] double sph_bessel(int l, double x) noexcept;

/**
 * @brief Evaluates all spherical Bessel functions through a requested degree.
 * @param[in] lmax Maximum degree in [0,12].
 * @param[in] x Finite nonnegative argument.
 * @param[out] out Caller-owned array with at least lmax+1 elements.
 * Uses the same convention and recurrences as sph_bessel; arguments and output capacity are caller preconditions.
 */
void sph_bessel_all(int lmax, double x, double* out) noexcept;

/// Natural cubic spline on strictly increasing abscissae (zero endpoint second derivatives).
/// Evaluation outside the grid extrapolates the first/last cubic segment; a default-constructed spline
/// must be assigned a populated spline before evaluation or endpoint access.
class CubicSpline {
public:
    /**
     * @brief Creates an empty spline that must be populated before evaluation or endpoint access.
     */
    CubicSpline() = default;
    /**
     * @brief Fits a natural cubic spline with zero endpoint second derivatives.
     * @param[in] x Owned strictly increasing abscissae, at least three entries.
     * @param[in] y Owned function values with the same length as x.
     * Input vectors are moved into the spline. Uniform grids use constant-time interval lookup.
     * @throws Error For mismatched lengths or non-increasing abscissae.
     */
    CubicSpline(std::vector<double> x, std::vector<double> y);
    /**
     * @brief Evaluates the interpolating or extrapolating cubic segment.
     * @param[in] x Finite query abscissa; the spline must be populated.
     * @return Spline value; outside the grid, the first or last segment is extrapolated.
     */
    [[nodiscard]] double operator()(double x) const noexcept;
    /**
     * @brief Returns the lower endpoint of a populated spline.
     * @return First abscissa; calling on an empty spline violates the precondition.
     */
    [[nodiscard]] double xmin() const noexcept { return x_.front(); }
    /**
     * @brief Returns the upper endpoint of a populated spline.
     * @return Last abscissa; calling on an empty spline violates the precondition.
     */
    [[nodiscard]] double xmax() const noexcept { return x_.back(); }
    /**
     * @brief Reports whether the fitted abscissae have uniform spacing.
     * @return True when spacings agree within the constructor tolerance; false for an empty spline.
     */
    [[nodiscard]] bool uniform() const noexcept { return uniform_; }
    /**
     * @brief Exposes the average spacing used for uniform-grid evaluation.
     * @return Mean endpoint spacing; meaningful after fitting, and used directly only when uniform() is true.
     */
    [[nodiscard]] double h() const noexcept { return h_; }
    /**
     * @brief Exposes fitted sample values.
     * @return Borrowed values, valid until the spline is assigned or destroyed.
     */
    [[nodiscard]] const std::vector<double>& y() const noexcept { return y_; }
    /**
     * @brief Exposes natural-spline curvature coefficients.
     * @return Borrowed second derivatives in sample order, valid until assignment/destruction.
     */
    [[nodiscard]] const std::vector<double>& second_derivatives() const noexcept { return m_; }

private:
    std::vector<double> x_, y_, m_; // m_ = second derivatives
    bool uniform_{false};
    double h_{0};
};

/**
 * @brief Integrates values over a logarithmically spaced radial grid.
 * @param[in] r Positive logarithmic abscissae, at least four points.
 * @param[in] f Integrand values matching r; the radial Jacobian r is applied internally.
 * @return Integral of f(r) dr using Simpson weights in log radius and a closing 3/8 rule.
 * @throws Error For inconsistent lengths or fewer than four points; logarithmic spacing is a caller precondition.
 */
[[nodiscard]] double integrate_log_grid(const std::vector<double>& r, const std::vector<double>& f);

/**
 * @brief Builds composite Simpson weights on a uniform grid.
 * @param[in] n At least two grid points.
 * @param[in] h Grid spacing in the integration variable.
 * @return Owning weights; two points use trapezoidal weights and even counts close with a 3/8 rule.
 * @throws Error For fewer than two points.
 */
[[nodiscard]] std::vector<double> simpson_weights(int n, double h);

} // namespace prowl::radial
