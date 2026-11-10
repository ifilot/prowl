// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file ylm.hpp
/// Real spherical harmonics, PROWL convention (docs/SCIENCE.md):
///
///   Y_{l,0}  = N_l0 P_l(cos t)
///   Y_{l,m}  = sqrt(2) N_lm P_l^m(cos t) cos(m p),      m > 0
///   Y_{l,-m} = sqrt(2) N_lm P_l^m(cos t) sin(m p),      m > 0
///   N_lm = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!), P_l^m WITHOUT Condon-Shortley phase.
///
/// Hence p: (m=-1,0,1) = (y, z, x) and d: (xy, yz, 3z^2-r^2, xz, x^2-y^2), all
/// with positive prefactors; this is also the orbital order printed by LOBSTER.
/// Flat index: lm = l*l + l + m.

#include <string>

#include <Eigen/Dense>

#include "prowl/lattice.hpp"

namespace prowl::ylm {

inline constexpr int lmax_supported = 6;

/**
 * @brief Maps an angular index pair to the packed harmonic layout.
 * @param[in] l Nonnegative degree.
 * @param[in] m Magnetic index in [-l,l].
 * @return Index l*l+l+m; this constexpr helper assumes valid indices.
 */
[[nodiscard]] constexpr int index(int l, int m) noexcept { return l * l + l + m; }
/**
 * @brief Counts all packed harmonics through a maximum degree.
 * @param[in] lmax Nonnegative maximum degree.
 * @return (lmax+1)^2 entries; this constexpr helper assumes a representable integer result.
 */
[[nodiscard]] constexpr int count(int lmax) noexcept { return (lmax + 1) * (lmax + 1); }

/**
 * @brief Evaluates PROWL real harmonics without the Condon-Shortley phase.
 * @param[in] lmax Maximum degree in [0,lmax_supported].
 * @param[in] v Finite direction, not necessarily normalized; zero is treated as +z.
 * @param[out] out Caller-owned array of at least count(lmax) elements, indexed with index(l,m).
 * This noexcept kernel assumes valid lmax and output capacity.
 */
void real_ylm(int lmax, const Vec3& v, double* out) noexcept;

/**
 * @brief Integrates a product of three real spherical harmonics over the sphere.
 * @param[in] l1 First degree in [0,lmax_supported].
 * @param[in] m1 First magnetic index in [-l1,l1].
 * @param[in] l2 Second supported degree.
 * @param[in] m2 Second valid magnetic index.
 * @param[in] l3 Third supported degree.
 * @param[in] m3 Third valid magnetic index.
 * @return Real Gaunt coefficient, including exact zero values after numerical thresholding.
 * @throws Error For out-of-range angular indices; a shared immutable table is initialized on first use.
 */
[[nodiscard]] double gaunt(int l1, int m1, int l2, int m2, int l3, int m3);

/**
 * @brief Expresses real harmonics in a rotated orthonormal frame.
 * @param[in] l Angular degree in [0,lmax_supported].
 * @param[in] axes Rows are the new right-handed Cartesian x/y/z axes in the old coordinates.
 * @return Owning (2l+1)-square D with Y_l(axes*v)=D*Y_l(v); coefficients transform as D*c and blocks as D_A*H*D_B^T.
 * @throws Error For invalid degree, non-orthonormal axes or a left-handed frame.
 */
[[nodiscard]] Eigen::MatrixXd rotation_matrix(int l, const Mat3& axes);

/**
 * @brief Names an orbital using the real-harmonic ordering.
 * @param[in] l Nonnegative angular degree.
 * @param[in] m Magnetic index in [-l,l].
 * @return LOBSTER-style suffix for s/p/d/f, or l/m notation for higher degrees.
 * Valid magnetic indices are required for the s/p/d/f lookup tables.
 */
[[nodiscard]] std::string orbital_name(int l, int m);

} // namespace prowl::ylm
