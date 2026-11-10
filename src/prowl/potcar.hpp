// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file potcar.hpp
/// Reader for the PAW data needed by PROWL from a user-supplied, locally
/// licensed POTCAR. Nothing read here is ever written to outputs except the
/// TITEL/SHA256 identifiers for provenance.
///
/// Conventions established and tested (docs/SCIENCE.md, section PAW):
///  - radial grid r_i (Angstrom), logarithmic;
///  - partial waves stored as u(r) = r * phi(r) (AE and pseudo);
///  - reciprocal projectors PSPNL(q) = 4 pi int r^2 j_l(q r) p(r) dr on the
///    linear grid q_j = j * PSMAXN / 100, j = 0..99 (q in 1/Angstrom);
///  - projectors are used exactly as VASP does for LREAL = .FALSE.

#include <filesystem>
#include <string>
#include <vector>

namespace prowl {

struct PawChannel {
    int l{};
    std::vector<double> projector_q; ///< PSPNL values on the q grid
    std::vector<double> u_ae;        ///< r * phi_AE on the radial grid
    std::vector<double> u_ps;        ///< r * phi~ on the radial grid
};

struct PawDataset {
    std::string titel;   ///< e.g. "PAW_PBE C 08Apr2002"
    std::string symbol;  ///< POTCAR symbol, e.g. "Ga_d"
    std::string element; ///< chemical element, e.g. "Ga"
    std::string valence_config; ///< VRHFIN configuration after the colon, e.g. "d10s2p1"
    std::string sha256;  ///< SHA256 line from header if present
    double zval{};
    double psmaxn{};     ///< max q of projector table (1/Angstrom)
    double rmax_proj{};  ///< real-space projector range PSRMAX (Angstrom)
    std::vector<double> grid; ///< radial log grid (Angstrom)
    std::vector<PawChannel> channels;
    std::vector<double> qpaw; ///< "augmentation charges" matrix from file (nch x nch, row-major)

    /**
     * @brief Counts radial PAW channels without magnetic degeneracy.
     * @return Number of channel records in this dataset.
     */
    [[nodiscard]] int nchannels() const noexcept { return static_cast<int>(channels.size()); }
    /**
     * @brief Counts magnetic projector functions over all channels.
     * @return Sum of 2*l+1 over the validated channels; manually constructed datasets require valid degrees.
     */
    [[nodiscard]] int nprojectors() const noexcept;
    /**
     * @brief Reports the supported VASP reciprocal-projector table length.
     * @return 100 q samples per channel.
     */
    [[nodiscard]] static constexpr int nq() noexcept { return 100; }
};

/**
 * @brief Parses all complete PAW datasets in a local concatenated POTCAR.
 * @param[in] path Licensed input file; its contents remain local and are not redistributed.
 * @return Owning datasets in file/species order with Angstrom radial grids and partial-wave/projector tables.
 * @throws Error For missing files, malformed numeric blocks/grids or unsupported datasets.
 */
[[nodiscard]] std::vector<PawDataset> read_potcar(const std::filesystem::path& path);

} // namespace prowl
