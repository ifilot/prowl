// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file lattice.hpp
/// Real-space lattice and reciprocal lattice in Angstrom units.
///
/// Conventions:
///  - `a[i]` is the i-th direct lattice vector (Cartesian, Angstrom).
///  - `b[i]` is the i-th reciprocal vector WITHOUT the 2*pi factor
///    (a[i] . b[j] = delta_ij), as in VASP. Cartesian wave vectors are
///    q = 2*pi * sum_i (k_i + G_i) b[i].

#include <array>
#include <cmath>

#include "prowl/errors.hpp"

namespace prowl {

using Vec3 = std::array<double, 3>;
using IVec3 = std::array<int, 3>;
using Mat3 = std::array<Vec3, 3>; ///< row-major: m[i] is row i

/**
 * @brief Forms a Cartesian dot product.
 * @param[in] x First three-component vector.
 * @param[in] y Second vector in the same coordinate frame.
 * @return Scalar sum of component products, with the product of the input units.
 */
[[nodiscard]] inline double dot(const Vec3& x, const Vec3& y) noexcept {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}
/**
 * @brief Forms the right-handed Cartesian cross product.
 * @param[in] x First three-component vector.
 * @param[in] y Second vector in the same frame.
 * @return Value vector x cross y, in the product of the input units.
 */
[[nodiscard]] inline Vec3 cross(const Vec3& x, const Vec3& y) noexcept {
    return {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
}
/**
 * @brief Computes the Euclidean length of a Cartesian vector.
 * @param[in] x Three-component vector.
 * @return Nonnegative square-root dot product, in the input units.
 */
[[nodiscard]] inline double norm(const Vec3& x) noexcept { return std::sqrt(dot(x, x)); }

struct Lattice {
    Mat3 a{};      ///< direct lattice vectors (rows), Angstrom
    Mat3 b{};      ///< reciprocal vectors (rows), 1/Angstrom, no 2*pi
    double volume{}; ///< cell volume, Angstrom^3 (positive)

    /**
     * @brief Creates an uninitialized zero lattice for later assignment; geometric operations require initialization.
     */
    Lattice() = default;
    /**
     * @brief Builds reciprocal vectors and a positive cell volume from direct rows.
     * @param[in] direct Finite independent Cartesian lattice rows, in Angstrom.
     * Reciprocal rows exclude 2*pi and retain the determinant orientation.
     * @throws Error For non-finite components, singular volume or determinant overflow.
     */
    explicit Lattice(const Mat3& direct) : a(direct) {
        for (const auto& row : a)
            for (double x : row) require(std::isfinite(x), ErrorCode::InvalidArgument, "lattice components must be finite");
        const double det = dot(a[0], cross(a[1], a[2]));
        require(std::isfinite(det) && std::abs(det) > 1e-12, ErrorCode::InvalidArgument, "lattice vectors are linearly dependent or overflow");
        volume = std::abs(det);
        b[0] = cross(a[1], a[2]);
        b[1] = cross(a[2], a[0]);
        b[2] = cross(a[0], a[1]);
        for (auto& row : b)
            for (auto& x : row) x /= det;
    }

    /**
     * @brief Converts fractional direct coordinates to Cartesian coordinates.
     * @param[in] f Fractional components along direct lattice rows.
     * @return Cartesian vector in Angstrom; the lattice must be initialized.
     */
    [[nodiscard]] Vec3 to_cartesian(const Vec3& f) const noexcept {
        Vec3 r{};
        for (int i = 0; i < 3; ++i)
            for (int c = 0; c < 3; ++c) r[c] += f[i] * a[i][c];
        return r;
    }
    /**
     * @brief Converts reciprocal fractional components into a Cartesian wave vector.
     * @param[in] f Fractional components, such as k+G, along reciprocal rows.
     * @return Cartesian 2*pi*(f dot b) in inverse Angstrom; the lattice must be initialized.
     */
    [[nodiscard]] Vec3 reciprocal_cartesian(const Vec3& f) const noexcept {
        Vec3 q{};
        for (int i = 0; i < 3; ++i)
            for (int c = 0; c < 3; ++c) q[c] += f[i] * b[i][c];
        for (auto& x : q) x *= 6.283185307179586476925286766559;
        return q;
    }
};

} // namespace prowl
