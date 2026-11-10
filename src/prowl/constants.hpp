// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file constants.hpp
/// Physical constants. Internal length unit is the Angstrom and energy unit the
/// eV, matching VASP files. The plane-wave cutoff test uses exactly the
/// constants VASP uses (constant.F in VASP 6.x) so that G-sphere membership is
/// reproduced bit-for-bit away from the sphere surface.

namespace prowl::constants {

inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double two_pi = 2.0 * pi;
inline constexpr double four_pi = 4.0 * pi;

/// Bohr radius in Angstrom as used by VASP (AUTOA).
inline constexpr double vasp_autoa = 0.529177249;
/// Rydberg in eV as used by VASP (RYTOEV).
inline constexpr double vasp_rytoev = 13.605826;
/// hbar^2 / (2 m_e) in eV*Angstrom^2 as used by VASP (HSQDTM).
inline constexpr double vasp_hsqdtm = vasp_rytoev * vasp_autoa * vasp_autoa;

/// Hartree in eV (CODATA 2018), used for Koga orbital energies.
inline constexpr double hartree_ev = 27.211386245988;
/// Bohr radius in Angstrom (CODATA 2018), used to convert atomic-unit basis data.
inline constexpr double bohr_angstrom = 0.529177210903;

} // namespace prowl::constants
