// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file wavecar.hpp
/// Streaming, validating reader for VASP WAVECAR files.
///
/// Physical layout (Fortran direct access, record length RECL bytes):
///   rec 1   : RECL, NSPIN, RTAG                               (float64)
///   rec 2   : NKPTS, NBANDS, ENCUT, A(3,3), EFERMI             (float64)
///   per (spin, k):
///     header: NPLW, k(3), {Re eps, Im eps, f} x NBANDS         (float64,
///             spanning several records for RTAG 53300/53310)
///     NBANDS records of NPLW complex coefficients (complex64 for
///             RTAG 45200/53300, complex128 for 45210/53310)
///
/// Logical contract (see docs/SCIENCE.md, section "Wavefunction convention"):
///   psi~_nk(r) = Omega^{-1/2} sum_G c_G exp(i (k+G).r),
/// with c_G the coefficients of the PAW pseudo wavefunction normalised such
/// that <psi~|S_PAW|psi~> = 1. The plane-wave norm alone need not equal one;
/// its PAW augmentation is included by the projection model.
/// For gamma-half files the stored value for G != 0 is sqrt(2) c_G; the reader
/// returns the *logical* c_G on the half set (stored / sqrt(2)) and flags the
/// encoding so that callers weight G != 0 terms by 2 in real inner products.
/// No full complex sphere is ever materialised implicitly.

#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "prowl/gvectors.hpp"
#include "prowl/lattice.hpp"

namespace prowl {

enum class WavefunctionEncoding { FullComplex, GammaHalf, Spinor };
enum class CoefficientPrecision { Single, Double };

/**
 * @brief Names a wavefunction storage encoding.
 * @param[in] e Detected or explicitly requested encoding.
 * @return Borrowed process-lifetime name, or Unknown for an unrecognized value.
 */
[[nodiscard]] const char* to_string(WavefunctionEncoding e) noexcept;

struct WavecarHeader {
    std::uint64_t record_length{}; ///< bytes
    int nspin{};
    int rtag{};
    CoefficientPrecision precision{};
    bool header_spans_records{}; ///< RTAG 53300/53310 layout
    int nkpts{};
    int nbands{};
    double encut_ev{};
    Lattice lattice;
    double efermi_ev{}; ///< informational only; authoritative value from vasprun.xml
    std::uint64_t file_size{};
    int records_per_kheader{};
};

struct KPointInfo {
    int spin{};
    int k{};
    int nplw{}; ///< stored coefficients per band
    Vec3 k_frac{};
    std::vector<double> eigenvalues_ev;
    std::vector<double> eigenvalues_imag;
    std::vector<double> occupations; ///< as stored (0..1 per spin channel)
};

struct ReaderOptions {
    /// Force the encoding instead of auto-detecting it from plane-wave counts.
    std::optional<WavefunctionEncoding> encoding_override;
    /// Gamma half-space convention. Auto-detection cannot distinguish X and Z
    /// (both have identical counts); X is the VASP >= 5.4 canonical layout.
    GammaHalf gamma_half = GammaHalf::X;
    bool gamma_half_overridden = false;
};

/// Per-encoding diagnostics recorded when the layout is detected.
struct LayoutReport {
    WavefunctionEncoding encoding{};
    GammaHalf gamma_half{GammaHalf::X};
    bool gamma_half_assumed{}; ///< true when X-half is the documented default, not an override
    std::vector<int> expected_full;   ///< per k: full-sphere count
    std::vector<int> stored;          ///< per k: stored NPLW
    double min_sphere_margin_ev{};    ///< smallest |E_kin - ENCUT| over all k (rounding risk)
    std::vector<std::string> notes;
};

struct WavefunctionBlock {
    int spin{};
    int k{};
    int first_band{};
    int band_count{};
    Vec3 k_frac{};
    WavefunctionEncoding encoding{};
    std::shared_ptr<const std::vector<IVec3>> gvectors; ///< Miller indices (half set for GammaHalf)
    /// Logical coefficients, column-major (band b, plane wave g) -> [b * ng + g].
    std::vector<std::complex<double>> coefficients;
    std::vector<double> eigenvalues_ev;
    std::vector<double> occupations;
    /// Multiplier applied to stored G != 0 values to obtain logical coefficients.
    double gamma_nonzero_scale{1.0};
    /// Plane-wave norms computed while reading (same arithmetic as plane_wave_norms()); empty if not computed.
    std::vector<double> pw_norms;

    /**
     * @brief Counts stored plane-wave vectors in this block.
     * @return Shared G-list length, or zero when no list is assigned.
     */
    [[nodiscard]] std::size_t ng() const noexcept { return gvectors ? gvectors->size() : 0; }
    /**
     * @brief Borrows the logical coefficients of one band.
     * @param[in] b Zero-based index within this block, in [0,band_count); the coefficient storage must be populated.
     * @return Pointer to ng() contiguous values; invalidated by coefficient reallocation, block replacement or destruction.
     */
    [[nodiscard]] const std::complex<double>* band(int b) const { return coefficients.data() + static_cast<std::size_t>(b) * ng(); }
    /**
     * @brief Computes full-sphere pseudo-wavefunction norms for all block bands.
     * @return Owning norm vector; cached read-time values are copied when present, otherwise Gamma weights are applied.
     * The block arrays and any cached pw_norms must match band_count and ng(); the block must not be mutated concurrently.
     */
    [[nodiscard]] std::vector<double> plane_wave_norms() const;
};

class WavecarReader {
public:
    /**
     * @brief Opens and validates a VASP WAVECAR and reconstructs its reciprocal indices.
     * @param[in] path Input WAVECAR file, retained as an owning path.
     * @param[in] options Encoding/half-space overrides; contradictory count-based encodings are rejected.
     * The reader owns file resources and immutable metadata. Concurrent reads into different blocks are supported.
     * @throws Error For missing/truncated/unsupported input or inconsistent layout.
     */
    explicit WavecarReader(const std::filesystem::path& path, ReaderOptions options = {});
    /**
     * @brief Closes owned file resources; callers must finish concurrent reads before destruction.
     */
    ~WavecarReader();
    WavecarReader(const WavecarReader&) = delete;
    WavecarReader& operator=(const WavecarReader&) = delete;

    /**
     * @brief Borrows validated file-level metadata.
     * @return Immutable header reference valid until reader destruction.
     */
    [[nodiscard]] const WavecarHeader& header() const noexcept { return header_; }
    /**
     * @brief Borrows the detected storage layout and its assumptions.
     * @return Immutable report reference valid until reader destruction.
     */
    [[nodiscard]] const LayoutReport& layout() const noexcept { return layout_; }
    /**
     * @brief Borrows band metadata for one spin and k-point.
     * @param[in] spin Zero-based spin-channel index.
     * @param[in] k Zero-based k-point index.
     * @return Immutable eigenvalue/occupation metadata valid until reader destruction.
     * @throws Error For out-of-range indices.
     */
    [[nodiscard]] const KPointInfo& kpoint(int spin, int k) const;
    /**
     * @brief Reports the validated coefficient encoding.
     * @return FullComplex, GammaHalf or Spinor; spinor data remain inspection-only.
     */
    [[nodiscard]] WavefunctionEncoding encoding() const noexcept { return layout_.encoding; }
    /**
     * @brief Borrows the input path used to construct the reader.
     * @return Immutable path reference valid until reader destruction.
     */
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    /**
     * @brief Shares the immutable Miller-index list for a k-point.
     * @param[in] k Zero-based k-point index.
     * @return Shared full- or half-sphere list; ownership may outlive the reader.
     * @throws Error For an invalid k index.
     */
    [[nodiscard]] std::shared_ptr<const std::vector<IVec3>> gvectors(int k) const;

    /**
     * @brief Reads an owning logical coefficient block without expanding Gamma storage.
     * @param[in] spin Zero-based spin channel.
     * @param[in] k Zero-based k-point.
     * @param[in] first_band Zero-based first band.
     * @param[in] count Positive band count within the file range.
     * @return Owning block with logical complex128 coefficients and shared immutable G indices.
     * @throws Error For invalid indices/ranges, spinors, truncated reads or non-finite coefficients.
     */
    [[nodiscard]] WavefunctionBlock read_block(int spin, int k, int first_band, int count) const;
    /**
     * @brief Reads a logical coefficient block while reusing caller-owned buffers.
     * @param[in] spin Zero-based spin channel.
     * @param[in] k Zero-based k-point.
     * @param[in] first_band Zero-based first band.
     * @param[in] count Positive number of bands within the file range.
     * @param[in,out] blk Exclusive mutable destination; its old content and borrowed coefficient pointers are invalidated.
     * Retains capacity when useful, so allocated capacity can exceed the final partial block size. Concurrent calls need different destinations.
     * @throws Error For invalid ranges, unsupported spinors or malformed reads; destination content may be partially replaced.
     */
    void read_block_into(int spin, int k, int first_band, int count, WavefunctionBlock& blk) const;

    /**
     * @brief Estimates logical coefficient storage for one band block.
     * @param[in] k Zero-based k-point.
     * @param[in] count Nonnegative count no larger than the file band count.
     * @return NPLW*count*sizeof(complex<double>) bytes, excluding metadata, read scratch and backend copies.
     * @throws Error For invalid k/count.
     */
    [[nodiscard]] std::size_t block_bytes(int k, int count) const;

private:
    std::filesystem::path path_;
    ReaderOptions options_;
    WavecarHeader header_;
    LayoutReport layout_;
    std::vector<KPointInfo> kinfo_; ///< index spin * nkpts + k
    mutable std::vector<std::shared_ptr<const std::vector<IVec3>>> gcache_;
    mutable std::ifstream file_;
    mutable std::mutex io_mutex_; ///< serialises seek+read (bindings release the GIL)
    int fd_ = -1;                 ///< POSIX descriptor for concurrent positional reads (-1: not available)

    void read_headers();
    void detect_layout();
    [[nodiscard]] std::uint64_t kheader_offset(int spin, int k) const noexcept;
    void read_bytes(std::uint64_t offset, void* dst, std::size_t n) const;
    /// As read_bytes, split into concurrent positional reads (several requests in flight help network and 9p
    /// file systems, e.g. WSL's /mnt drives: about 2x for 4 requests).
    void read_bytes_parallel(std::uint64_t offset, void* dst, std::size_t n) const;
};

} // namespace prowl
