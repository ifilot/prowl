// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file structure.hpp
/// Crystal structure from POSCAR/CONTCAR (VASP 5+ format with species line).

#include <filesystem>
#include <string>
#include <vector>

#include "prowl/lattice.hpp"

namespace prowl {

struct Structure {
    std::string comment;
    Lattice lattice;
    std::vector<std::string> species;  ///< per type, POSCAR order
    std::vector<int> counts;           ///< atoms per type
    std::vector<int> atom_type;        ///< per atom -> type index
    std::vector<Vec3> frac;            ///< fractional coordinates per atom

    /**
     * @brief Counts atoms in the fractional-coordinate array.
     * @return Atom count; consistency with the species/type arrays is a caller invariant for manually built structures.
     */
    [[nodiscard]] int natoms() const noexcept { return static_cast<int>(frac.size()); }
    /**
     * @brief Converts one atom position to Cartesian coordinates.
     * @param[in] atom Zero-based atom index.
     * @return Cartesian position in Angstrom.
     * @throws std::out_of_range For an invalid atom index.
     */
    [[nodiscard]] Vec3 cartesian(int atom) const { return lattice.to_cartesian(frac.at(atom)); }
    /**
     * @brief Looks up the chemical element for an atom.
     * @param[in] atom Zero-based atom index.
     * @return Borrowed species string, valid until species storage changes or the Structure is destroyed.
     * @throws std::out_of_range For invalid atom/type indices.
     */
    [[nodiscard]] const std::string& element(int atom) const { return species.at(atom_type.at(atom)); }
};

/**
 * @brief Reads a VASP 5+ POSCAR/CONTCAR with an explicit species line.
 * @param[in] path Input file; both direct and Cartesian coordinates and a scalar scale are supported.
 * @return Owning structure in Angstrom and fractional coordinates.
 * @throws Error For missing, malformed or unsupported input; VASP 4 species-free files are rejected.
 */
[[nodiscard]] Structure read_poscar(const std::filesystem::path& path);

/**
 * @brief Checks component-wise agreement of two direct lattices.
 * @param[in] a First lattice in Angstrom.
 * @param[in] b Second lattice in Angstrom.
 * @param[in] tol Allowed absolute component error in Angstrom.
 * @param[in] what Description of the compared sources for diagnostics.
 * @throws Error With InconsistentInput when any component differs by more than tol.
 */
void check_same_lattice(const Lattice& a, const Lattice& b, double tol, const std::string& what);

} // namespace prowl
