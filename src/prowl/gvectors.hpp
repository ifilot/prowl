// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file gvectors.hpp
/// Deterministic reconstruction of the plane-wave basis used by VASP.
///
/// The set for k-point k is { G : HSQDTM * |2*pi (k+G) . b|^2 < ENCUT }.
/// Ordering (VASP 5.4+/6.x): z outermost, then y, then x fastest; along each
/// axis the integer frequencies run 0, 1, ..., +n, -n, ..., -1.
///
/// Gamma-half (vasp_gam, VASP >= 5.4 canonical "x-half" written via
/// GEN_INDEX_GAMMA): only G with gx > 0, or gx == 0 && gy > 0, or
/// gx == gy == 0 && gz >= 0 are stored; the omitted -G are implied by
/// c(-G) = conj(c(G)). Older binaries compiled with -DwNGZhalf used the
/// analogous "z-half"; it is supported only by explicit override.

#include <cstddef>
#include <vector>

#include "prowl/lattice.hpp"

namespace prowl {

enum class GammaHalf { X, Z };

/**
 * @brief Enumerates the full VASP plane-wave sphere in file ordering.
 * @param[in] lat Nonsingular finite direct/reciprocal lattice.
 * @param[in] k_frac Finite fractional k-vector.
 * @param[in] encut_ev Finite positive kinetic-energy cutoff in eV.
 * @return Owning Miller-index list with z outermost and x innermost.
 * @throws Error For invalid lattice, k or cutoff/bounds.
 */
[[nodiscard]] std::vector<IVec3> generate_gvectors_full(const Lattice& lat, const Vec3& k_frac, double encut_ev);

/**
 * @brief Enumerates the stored half of a Gamma plane-wave sphere.
 * @param[in] lat Nonsingular finite lattice.
 * @param[in] encut_ev Finite positive energy cutoff in eV.
 * @param[in] half X-half canonical layout or explicitly selected Z-half.
 * @return Owning half-sphere indices, including G=0 once; k is fixed to Gamma.
 */
[[nodiscard]] std::vector<IVec3> generate_gvectors_gamma_half(const Lattice& lat, double encut_ev,
                                                              GammaHalf half = GammaHalf::X);

/**
 * @brief Evaluates VASP-ordered plane-wave kinetic energy.
 * @param[in] lat Initialized reciprocal lattice.
 * @param[in] k_frac Fractional k-vector.
 * @param[in] g Integer plane-wave Miller indices.
 * @return Kinetic energy in eV, using VASP constants and arithmetic order.
 */
[[nodiscard]] double kinetic_energy_ev(const Lattice& lat, const Vec3& k_frac, const IVec3& g);

/// Smallest energy distance of any included / excluded vector
/// from the cutoff sphere, in eV. Used to report when sphere-surface rounding
/// could make the reconstruction ambiguous.
struct SphereMargin {
    double inside_closest_ev;  ///< ENCUT - max(E_kin) over included vectors (>= 0)
    double outside_closest_ev; ///< min(E_kin) - ENCUT over excluded vectors (>= 0)
};
/**
 * @brief Finds the closest included and excluded vectors to the cutoff sphere.
 * @param[in] lat Nonsingular finite lattice.
 * @param[in] k_frac Finite fractional k-vector.
 * @param[in] encut_ev Finite positive cutoff in eV.
 * @return Nonnegative energy distances in eV; an absent side retains infinity.
 */
[[nodiscard]] SphereMargin cutoff_sphere_margin(const Lattice& lat, const Vec3& k_frac, double encut_ev);

} // namespace prowl
