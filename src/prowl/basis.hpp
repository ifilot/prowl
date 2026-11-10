// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file basis.hpp
/// Slater-type atomic orbital basis (Koga Hartree-Fock tabulations).
///
/// Units: r in Angstrom, R(r) in Angstrom^{-3/2}; exponents converted from
/// bohr^{-1} with the CODATA Bohr radius. chi_mu(r) = R_mu(|r|) Y_{l m}(r^)
/// with PROWL real harmonics (ylm.hpp).
///
/// Fourier-Bessel transform convention:
///   Rt(q) = int_0^inf r^2 j_l(q r) R(r) dr,
///   f~(q) = int f(r) exp(-i q.r) d^3r = 4 pi (-i)^l Y_lm(q^) Rt(|q|).

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "prowl/structure.hpp"

namespace prowl {

struct StoPrimitive {
    int n{};       ///< principal quantum number of the primitive
    double zeta{}; ///< exponent, 1/Angstrom
    double coef{}; ///< coefficient multiplying the *normalised* primitive
};

struct RadialSto {
    std::string element;
    std::string label; ///< e.g. "3s"
    int n{};
    int l{};
    double occupation{};
    double energy_ev{};
    std::vector<StoPrimitive> primitives;

    /**
     * @brief Evaluates the contracted normalized-primitive radial function.
     * @param[in] r Nonnegative radius in Angstrom.
     * @return R(r) in Angstrom^(-3/2); no spherical harmonic is included.
     */
    [[nodiscard]] double value(double r) const noexcept;
    /**
     * @brief Evaluates the analytic Fourier-Bessel radial transform.
     * @param[in] q Nonnegative Cartesian wave-number magnitude in inverse Angstrom.
     * @return Integral of r^2 j_l(qr) R(r) dr; the 4*pi and angular factors are not included.
     */
    [[nodiscard]] double fourier(double q) const noexcept;
    /**
     * @brief Computes the analytic radial normalization integral.
     * @return Integral of R(r)^2 r^2 dr, dimensionless; cross terms of all primitives are included.
     */
    [[nodiscard]] double norm2() const noexcept;
    /**
     * @brief Estimates radial support by scanning inward from 60 Angstrom.
     * @param[in] threshold Positive cutoff for |r R(r)| in Angstrom^(-1/2).
     * @return Last above-threshold sample plus 0.01 Angstrom, or 0.01 if none is found; not a rigorous tail bound.
     */
    [[nodiscard]] double extent(double threshold = 1e-8) const noexcept;
};

/**
 * @brief Integrates an exponential radial primitive against a spherical Bessel function.
 * @param[in] N Integer radial power, at least l+2.
 * @param[in] l Angular degree in [0,15].
 * @param[in] zeta Finite positive exponential decay, inverse Angstrom.
 * @param[in] q Finite nonnegative wave-number magnitude, inverse Angstrom.
 * @return Integral of r^N exp(-zeta*r) j_l(qr) dr, evaluated analytically up to rounding.
 * Arguments must satisfy the stated ranges; this noexcept scalar kernel does not validate them.
 */
[[nodiscard]] double sto_bessel_integral(int N, int l, double zeta, double q) noexcept;

struct StoLibraryInfo {
    std::string name;
    std::string schema;
    std::string provenance_json; ///< verbatim provenance block for output metadata
    std::string file_sha256;
};

class StoLibrary {
public:
    /**
     * @brief Loads and converts a prowl-sto-basis/1 JSON library from atomic units.
     * @param[in] json_path Basis JSON file with bohr lengths and hartree energies.
     * @return Owning library with internal Angstrom/eV values and provenance.
     * @throws Error For missing files, invalid JSON, unsupported schema/units or invalid primitives.
     * Malformed required JSON fields may additionally raise nlohmann::json exceptions.
     */
    static StoLibrary load(const std::filesystem::path& json_path);
    /**
     * @brief Loads the configured Koga basis data.
     * @return Owning library from PROWL_DATA_DIR/basis/koga/koga_hf_sto.json.
     * @throws Error If the configured data cannot be loaded.
     */
    static StoLibrary load_default();

    /**
     * @brief Exposes the current library identity and provenance.
     * @return Borrowed metadata, valid while the library lives; merge updates its contents.
     */
    [[nodiscard]] const StoLibraryInfo& info() const noexcept { return info_; }
    /**
     * @brief Checks whether the library has an element record.
     * @param[in] el Case-sensitive chemical symbol.
     * @return True if an element record exists, even when its shell list is empty.
     */
    [[nodiscard]] bool has_element(const std::string& el) const { return shells_.count(el) > 0; }
    /**
     * @brief Returns all radial shells for one element.
     * @param[in] element Case-sensitive element symbol.
     * @return Borrowed shell vector, valid until library mutation/destruction.
     * @throws Error With UnsupportedFeature when the element is absent.
     */
    [[nodiscard]] const std::vector<RadialSto>& shells(const std::string& element) const;
    /**
     * @brief Finds a labeled radial shell for an element.
     * @param[in] element Case-sensitive element symbol.
     * @param[in] label Shell label, such as 3s or 2p.
     * @return Borrowed shell; merge can invalidate this reference.
     * @throws Error If the element or shell is absent.
     */
    [[nodiscard]] const RadialSto& shell(const std::string& element, const std::string& label) const;
    /**
     * @brief Appends extension shells to existing elements and records their provenance.
     * @param[in] extension Borrowed library; elements must already exist and labels must not duplicate existing shells.
     * May invalidate borrowed shell references. On validation failure earlier insertions may remain; the operation is not transactional.
     * @throws Error For a missing base element or duplicate shell label.
     */
    void merge(const StoLibrary& extension);

private:
    StoLibraryInfo info_;
    std::map<std::string, std::vector<RadialSto>> shells_;
};

/// One localised basis function chi_mu centred on an atom in the home cell.
struct OrbitalSpec {
    int atom{};
    int radial{};  ///< index into AtomicBasis::radials()
    int l{};
    int m{};
    std::string label; ///< e.g. "C1_2p_y" (LOBSTER-style atom numbering from 1)
};

class AtomicBasis {
public:
    /**
     * @brief Builds an owning atom-ordered local basis from selected library shells.
     * @param[in] s Structure whose atom/species order determines orbital order.
     * @param[in] lib Borrowed radial library; selected shells are copied.
     * @param[in] selection Element-to-shell-label map; labels are unique within each element.
     * Arguments need only outlive construction. Shell m values run from -l through l.
     * @throws Error For absent selections, missing shells or duplicate labels.
     */
    AtomicBasis(const Structure& s, const StoLibrary& lib, const std::map<std::string, std::vector<std::string>>& selection);

    /**
     * @brief Exposes the unique radial shells copied into this basis.
     * @return Borrowed radial array, valid until this basis is assigned or destroyed.
     */
    [[nodiscard]] const std::vector<RadialSto>& radials() const noexcept { return radials_; }
    /**
     * @brief Exposes atom-ordered orbital descriptors.
     * @return Borrowed orbital array; indices refer to radials() and the construction-time atom order.
     */
    [[nodiscard]] const std::vector<OrbitalSpec>& orbitals() const noexcept { return orbitals_; }
    /**
     * @brief Counts angular basis functions over all selected shells and atoms.
     * @return Number of orbitals, not the number of radial shells.
     */
    [[nodiscard]] int size() const noexcept { return static_cast<int>(orbitals_.size()); }
    /**
     * @brief Finds the largest angular degree in this basis.
     * @return Maximum selected l, or zero for an empty radial list.
     */
    [[nodiscard]] int lmax() const noexcept;

private:
    std::vector<RadialSto> radials_;
    std::vector<OrbitalSpec> orbitals_;
};

} // namespace prowl
