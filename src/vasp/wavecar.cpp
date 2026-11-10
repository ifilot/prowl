// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/wavecar.hpp"

#include <algorithm>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <cmath>
#include <limits>
#include <sstream>

namespace prowl {

const char* to_string(WavefunctionEncoding e) noexcept {
    switch (e) {
    case WavefunctionEncoding::FullComplex: return "FullComplex";
    case WavefunctionEncoding::GammaHalf: return "GammaHalf";
    case WavefunctionEncoding::Spinor: return "Spinor";
    }
    return "Unknown";
}

namespace {

bool is_gamma(const Vec3& k) { return std::abs(k[0]) < 1e-10 && std::abs(k[1]) < 1e-10 && std::abs(k[2]) < 1e-10; }

int checked_int(double v, const char* what, double lo, double hi) {
    if (!std::isfinite(v) || v < lo || v > hi || std::abs(v - std::nearbyint(v)) > 1e-6) {
        std::ostringstream os;
        os << "WAVECAR field " << what << " has invalid value " << v;
        fail(ErrorCode::CorruptFile, os.str());
    }
    return static_cast<int>(std::nearbyint(v));
}

} // namespace

std::vector<double> WavefunctionBlock::plane_wave_norms() const {
    if (!pw_norms.empty()) return pw_norms;
    std::vector<double> out(band_count, 0.0);
    const std::size_t n = ng();
    for (int b = 0; b < band_count; ++b) {
        const auto* c = band(b);
        double s = 0.0;
        if (encoding == WavefunctionEncoding::GammaHalf) {
            for (std::size_t g = 0; g < n; ++g) {
                const auto& G = (*gvectors)[g];
                const double w = (G[0] == 0 && G[1] == 0 && G[2] == 0) ? 1.0 : 2.0;
                s += w * std::norm(c[g]);
            }
        } else {
            for (std::size_t g = 0; g < n; ++g) s += std::norm(c[g]);
        }
        out[b] = s;
    }
    return out;
}

WavecarReader::WavecarReader(const std::filesystem::path& path, ReaderOptions options)
    : path_(path), options_(options) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path_, ec))
        fail(ErrorCode::FileNotFound, "WAVECAR not found: " + path_.string());
    header_.file_size = std::filesystem::file_size(path_);
    file_.open(path_, std::ios::binary);
    if (!file_) fail(ErrorCode::FileNotFound, "cannot open " + path_.string());
    read_headers();
    detect_layout();
#if defined(__unix__) || defined(__APPLE__)
    // Acquire the positional-read descriptor only after validation: a throwing constructor never runs our destructor.
    fd_ = ::open(path_.c_str(), O_RDONLY);
#endif
}

WavecarReader::~WavecarReader() {
#if defined(__unix__) || defined(__APPLE__)
    if (fd_ >= 0) ::close(fd_);
#endif
}

void WavecarReader::read_bytes_parallel(std::uint64_t offset, void* dst, std::size_t n) const {
#if defined(__unix__) || defined(__APPLE__)
    constexpr std::size_t min_piece = std::size_t{8} << 20;
    const int pieces = static_cast<int>(std::min<std::size_t>(4, std::max<std::size_t>(1, n / min_piece)));
    if (fd_ < 0 || pieces < 2) return read_bytes(offset, dst, n);
    if (offset + n > header_.file_size) return read_bytes(offset, dst, n); // reports the error
    const std::size_t piece = (n + pieces - 1) / pieces;
    bool ok = true;
#pragma omp parallel for num_threads(pieces) schedule(static, 1) reduction(&& : ok)
    for (int i = 0; i < pieces; ++i) {
        std::size_t done = static_cast<std::size_t>(i) * piece;
        const std::size_t end = std::min(n, done + piece);
        while (ok && done < end) {
            const ssize_t got = ::pread(fd_, static_cast<char*>(dst) + done, end - done,
                                        static_cast<off_t>(offset + done));
            if (got <= 0) ok = false;
            else done += static_cast<std::size_t>(got);
        }
    }
    if (!ok) fail(ErrorCode::CorruptFile, "I/O error reading " + path_.string());
#else
    read_bytes(offset, dst, n);
#endif
}

void WavecarReader::read_bytes(std::uint64_t offset, void* dst, std::size_t n) const {
    if (offset + n > header_.file_size) {
        std::ostringstream os;
        os << "attempt to read " << n << " bytes at offset " << offset << " beyond end of file (" << header_.file_size
           << " bytes); WAVECAR is truncated or its header is inconsistent";
        fail(ErrorCode::CorruptFile, os.str());
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(offset));
    file_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!file_) fail(ErrorCode::CorruptFile, "I/O error reading " + path_.string());
}

std::uint64_t WavecarReader::kheader_offset(int spin, int k) const noexcept {
    const std::uint64_t per_k = static_cast<std::uint64_t>(header_.records_per_kheader + header_.nbands);
    const std::uint64_t rec = 2 + (static_cast<std::uint64_t>(spin) * header_.nkpts + k) * per_k;
    return rec * header_.record_length;
}

void WavecarReader::read_headers() {
    auto& h = header_;
    require(h.file_size >= 24, ErrorCode::CorruptFile, "file too small to be a WAVECAR");
    double r1[3];
    read_bytes(0, r1, sizeof r1);
    const double recl = r1[0];
    if (!std::isfinite(recl) || recl < 24 || recl > static_cast<double>(h.file_size) ||
        std::abs(recl - std::nearbyint(recl)) > 0 || static_cast<std::uint64_t>(recl) % 8 != 0)
        fail(ErrorCode::CorruptFile, "invalid record length in WAVECAR record 1 (byte order or format not supported)");
    h.record_length = static_cast<std::uint64_t>(recl);
    h.nspin = checked_int(r1[1], "NSPIN", 1, 2);
    h.rtag = checked_int(r1[2], "RTAG", 0, 1e6);
    switch (h.rtag) {
    case 45200: h.precision = CoefficientPrecision::Single; h.header_spans_records = false; break;
    case 45210: h.precision = CoefficientPrecision::Double; h.header_spans_records = false; break;
    case 53300: h.precision = CoefficientPrecision::Single; h.header_spans_records = true; break;
    case 53310: h.precision = CoefficientPrecision::Double; h.header_spans_records = true; break;
    default:
        fail(ErrorCode::UnsupportedFeature,
             "unknown WAVECAR RTAG " + std::to_string(h.rtag) + " (supported: 45200, 45210, 53300, 53310)");
    }

    const std::size_t nrec2 = std::min<std::uint64_t>(13, h.record_length / 8);
    require(nrec2 >= 12, ErrorCode::CorruptFile, "record 2 too short");
    double r2[13] = {};
    read_bytes(h.record_length, r2, nrec2 * 8);
    h.nkpts = checked_int(r2[0], "NKPTS", 1, 1e7);
    h.nbands = checked_int(r2[1], "NBANDS", 1, 1e7);
    h.encut_ev = r2[2];
    require(std::isfinite(h.encut_ev) && h.encut_ev > 0 && h.encut_ev < 1e5, ErrorCode::CorruptFile, "invalid ENCUT");
    Mat3 a{};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) a[j][i] = r2[3 + 3 * j + i]; // Fortran A(I,J): vector J, component I
    h.lattice = Lattice(a);
    h.efermi_ev = nrec2 >= 13 ? r2[12] : std::numeric_limits<double>::quiet_NaN();

    const std::uint64_t doubles_per_rec = h.record_length / 8;
    const std::uint64_t kheader_doubles = 4 + 3 * static_cast<std::uint64_t>(h.nbands);
    if (h.header_spans_records) {
        h.records_per_kheader = static_cast<int>((kheader_doubles + doubles_per_rec - 1) / doubles_per_rec);
    } else {
        require(kheader_doubles <= doubles_per_rec, ErrorCode::CorruptFile,
                "k-point header does not fit into one record for a legacy (45200/45210) WAVECAR");
        h.records_per_kheader = 1;
    }

    const std::uint64_t nrec = 2 + static_cast<std::uint64_t>(h.nspin) * h.nkpts * (h.records_per_kheader + h.nbands);
    const std::uint64_t expected = nrec * h.record_length;
    if (h.file_size < expected) {
        std::ostringstream os;
        os << "WAVECAR truncated: header implies " << expected << " bytes (" << nrec << " records of "
           << h.record_length << "), file has " << h.file_size;
        fail(ErrorCode::CorruptFile, os.str());
    }

    const std::size_t coef_bytes = h.precision == CoefficientPrecision::Single ? 8 : 16;
    kinfo_.resize(static_cast<std::size_t>(h.nspin) * h.nkpts);
    std::vector<double> buf(kheader_doubles);
    for (int s = 0; s < h.nspin; ++s) {
        for (int k = 0; k < h.nkpts; ++k) {
            const std::uint64_t off = kheader_offset(s, k);
            std::uint64_t done = 0;
            int r = 0;
            while (done < kheader_doubles) {
                const std::uint64_t n = std::min(doubles_per_rec, kheader_doubles - done);
                read_bytes(off + static_cast<std::uint64_t>(r) * h.record_length, buf.data() + done, n * 8);
                done += n;
                ++r;
            }
            KPointInfo& ki = kinfo_[static_cast<std::size_t>(s) * h.nkpts + k];
            ki.spin = s;
            ki.k = k;
            ki.nplw = checked_int(buf[0], "NPLW", 1, 1e9);
            if (static_cast<std::uint64_t>(ki.nplw) * coef_bytes > h.record_length)
                fail(ErrorCode::CorruptFile, "NPLW=" + std::to_string(ki.nplw) + " at k=" + std::to_string(k + 1) +
                                                 " does not fit into the record length");
            ki.k_frac = {buf[1], buf[2], buf[3]};
            ki.eigenvalues_ev.resize(h.nbands);
            ki.eigenvalues_imag.resize(h.nbands);
            ki.occupations.resize(h.nbands);
            for (int b = 0; b < h.nbands; ++b) {
                ki.eigenvalues_ev[b] = buf[4 + 3 * b];
                ki.eigenvalues_imag[b] = buf[5 + 3 * b];
                ki.occupations[b] = buf[6 + 3 * b];
                if (!std::isfinite(ki.eigenvalues_ev[b]) || !std::isfinite(ki.occupations[b]))
                    fail(ErrorCode::CorruptFile, "non-finite eigenvalue/occupation in k-point header");
            }
            if (s == 1) {
                const auto& k0 = kinfo_[k].k_frac;
                for (int c = 0; c < 3; ++c)
                    require(std::abs(k0[c] - ki.k_frac[c]) < 1e-8, ErrorCode::CorruptFile,
                            "spin channels list different k-points");
            }
        }
    }
}

void WavecarReader::detect_layout() {
    const auto& h = header_;
    auto& L = layout_;
    L.expected_full.resize(h.nkpts);
    L.stored.resize(h.nkpts);
    L.min_sphere_margin_ev = std::numeric_limits<double>::infinity();
    gcache_.assign(h.nkpts, nullptr);

    std::optional<WavefunctionEncoding> enc;
    for (int k = 0; k < h.nkpts; ++k) {
        const auto& ki = kinfo_[k];
        auto full = std::make_shared<std::vector<IVec3>>(generate_gvectors_full(h.lattice, ki.k_frac, h.encut_ev));
        const int nfull = static_cast<int>(full->size());
        L.expected_full[k] = nfull;
        L.stored[k] = ki.nplw;
        const auto m = cutoff_sphere_margin(h.lattice, ki.k_frac, h.encut_ev);
        L.min_sphere_margin_ev = std::min({L.min_sphere_margin_ev, m.inside_closest_ev, m.outside_closest_ev});

        std::optional<WavefunctionEncoding> this_enc;
        if (ki.nplw == nfull) {
            this_enc = WavefunctionEncoding::FullComplex;
            gcache_[k] = full;
        } else if (ki.nplw == 2 * nfull) {
            this_enc = WavefunctionEncoding::Spinor;
            gcache_[k] = full;
        } else if (is_gamma(ki.k_frac) && ki.nplw == (nfull + 1) / 2) {
            this_enc = WavefunctionEncoding::GammaHalf;
            gcache_[k] = std::make_shared<std::vector<IVec3>>(
                generate_gvectors_gamma_half(h.lattice, h.encut_ev, options_.gamma_half));
        }
        if (!this_enc) {
            std::ostringstream os;
            os << "cannot reconstruct plane-wave set at k=" << k + 1 << ": stored NPLW=" << ki.nplw
               << ", full sphere=" << nfull << ", spinor=" << 2 * nfull;
            if (is_gamma(ki.k_frac)) os << ", gamma-half=" << (nfull + 1) / 2;
            os << ". Possible causes: unsupported VASP version, modified lattice, or rounding at the cutoff sphere"
               << " (closest |E_kin-ENCUT| = " << std::min(m.inside_closest_ev, m.outside_closest_ev) << " eV).";
            fail(ErrorCode::CorruptFile, os.str());
        }
        if (enc && *enc != *this_enc)
            fail(ErrorCode::CorruptFile, "inconsistent wavefunction encodings across k-points");
        enc = this_enc;
    }

    if (options_.encoding_override && *options_.encoding_override != *enc) {
        fail(ErrorCode::InconsistentInput, std::string("encoding override ") + to_string(*options_.encoding_override) +
                                               " contradicts plane-wave counts, which imply " + to_string(*enc));
    }
    L.encoding = *enc;
    L.gamma_half = options_.gamma_half;
    if (L.encoding == WavefunctionEncoding::GammaHalf) {
        require(h.nkpts == 1, ErrorCode::CorruptFile, "gamma-half WAVECAR with more than one k-point");
        L.gamma_half_assumed = !options_.gamma_half_overridden;
        if (L.gamma_half_assumed)
            L.notes.emplace_back(
                "gamma-half layout: X-half (VASP >= 5.4 canonical, GEN_INDEX_GAMMA) assumed; counts cannot "
                "distinguish X/Z halves. Verify with the PAW norm check or override for -DwNGZhalf builds.");
        L.notes.emplace_back("gamma-half coefficients for G != 0 are stored as sqrt(2) c_G; reader returns c_G.");
    }
    if (L.encoding == WavefunctionEncoding::Spinor)
        L.notes.emplace_back("noncollinear/spinor WAVECAR (vasp_ncl): inspection only; projection is unsupported.");
    if (L.min_sphere_margin_ev < 1e-6)
        L.notes.emplace_back("a plane wave lies within 1e-6 eV of the cutoff sphere; count validation passed but "
                             "ordering relies on VASP-identical rounding.");
}

const KPointInfo& WavecarReader::kpoint(int spin, int k) const {
    require(spin >= 0 && spin < header_.nspin && k >= 0 && k < header_.nkpts, ErrorCode::InvalidArgument,
            "spin/k index out of range");
    return kinfo_[static_cast<std::size_t>(spin) * header_.nkpts + k];
}

std::shared_ptr<const std::vector<IVec3>> WavecarReader::gvectors(int k) const {
    require(k >= 0 && k < header_.nkpts, ErrorCode::InvalidArgument, "k index out of range");
    return gcache_[k];
}

std::size_t WavecarReader::block_bytes(int k, int count) const {
    require(count >= 0 && count <= header_.nbands, ErrorCode::InvalidArgument, "band count out of range");
    return static_cast<std::size_t>(kpoint(0, k).nplw) * count * sizeof(std::complex<double>);
}

WavefunctionBlock WavecarReader::read_block(int spin, int k, int first_band, int count) const {
    WavefunctionBlock blk;
    read_block_into(spin, k, first_band, count, blk);
    return blk;
}

void WavecarReader::read_block_into(int spin, int k, int first_band, int count, WavefunctionBlock& blk) const {
    const auto& ki = kpoint(spin, k);
    require(first_band >= 0 && count > 0 && first_band <= header_.nbands && count <= header_.nbands - first_band, ErrorCode::InvalidArgument,
            "band range out of bounds");
    if (layout_.encoding == WavefunctionEncoding::Spinor)
        fail(ErrorCode::UnsupportedFeature,
             "noncollinear (spinor) wavefunctions are detected but not supported for projection in this version");

    blk.spin = spin;
    blk.k = k;
    blk.first_band = first_band;
    blk.band_count = count;
    blk.k_frac = ki.k_frac;
    blk.encoding = layout_.encoding;
    blk.gamma_nonzero_scale = 1.0;
    blk.pw_norms.clear();
    blk.gvectors = gcache_[k];
    const std::size_t ng = blk.gvectors->size();
    require(ng == static_cast<std::size_t>(ki.nplw), ErrorCode::Internal, "G-vector cache size mismatch");
    blk.coefficients.resize(ng * count);
    blk.eigenvalues_ev.assign(ki.eigenvalues_ev.begin() + first_band, ki.eigenvalues_ev.begin() + first_band + count);
    blk.occupations.assign(ki.occupations.begin() + first_band, ki.occupations.begin() + first_band + count);

    const std::uint64_t base = kheader_offset(spin, k) + static_cast<std::uint64_t>(header_.records_per_kheader) * header_.record_length;
    bool finite = true;
    if (header_.precision == CoefficientPrecision::Single) {
        // Consecutive band records in one read per chunk (the records are contiguous), converted in parallel with
        // the finiteness check fused into the same pass.
        const std::size_t rec = header_.record_length;
        require(rec >= ng * 8 && rec % alignof(std::complex<float>) == 0, ErrorCode::CorruptFile,
                "WAVECAR record length inconsistent with the plane-wave count");
        constexpr int chunk = 64;
        const bool fused_norms = layout_.encoding == WavefunctionEncoding::FullComplex;
        // Gamma-half: stored G != 0 values are scaled to logical coefficients in the same pass
        const bool half = layout_.encoding == WavefunctionEncoding::GammaHalf;
        const double scale = 1.0 / std::sqrt(2.0);
        std::size_t g0 = ng;
        if (half) {
            const auto& G = *blk.gvectors;
            for (std::size_t g = 0; g < ng && g0 == ng; ++g)
                if (G[g][0] == 0 && G[g][1] == 0 && G[g][2] == 0) g0 = g;
        }
        if (fused_norms) blk.pw_norms.assign(static_cast<std::size_t>(count), 0.0);
        std::vector<std::complex<float>> raw((rec / sizeof(std::complex<float>)) * std::min(count, chunk));
        for (int b0 = 0; b0 < count; b0 += chunk) {
            const int nb = std::min(chunk, count - b0);
            read_bytes_parallel(base + static_cast<std::uint64_t>(first_band + b0) * rec, raw.data(),
                                static_cast<std::size_t>(nb - 1) * rec + ng * 8);
            bool ok = true;
#pragma omp parallel for schedule(static) reduction(&& : ok) if (static_cast<std::size_t>(nb) * ng > 100000)
            for (int b = 0; b < nb; ++b) {
                const auto* src = raw.data() + static_cast<std::size_t>(b) * (rec / sizeof(std::complex<float>));
                auto* dst = blk.coefficients.data() + static_cast<std::size_t>(b0 + b) * ng;
                double norm = 0.0; // plane_wave_norms() of the full-complex encoding, same summation order
                bool fin = true;
                for (std::size_t g = 0; g < ng; ++g) {
                    const float re = src[g].real(), im = src[g].imag();
                    fin &= std::isfinite(re) & std::isfinite(im);
                    dst[g] = std::complex<double>(re, im);
                    norm += std::norm(dst[g]);
                }
                ok = ok && fin;
                if (fused_norms) blk.pw_norms[static_cast<std::size_t>(b0 + b)] = norm;
                if (half)
                    for (std::size_t g = 0; g < ng; ++g)
                        if (g != g0) dst[g] *= scale;
            }
            finite = finite && ok;
        }
    } else {
        for (int b = 0; b < count; ++b)
            read_bytes(base + static_cast<std::uint64_t>(first_band + b) * header_.record_length,
                       blk.coefficients.data() + static_cast<std::size_t>(b) * ng, ng * 16);
        for (const auto& c : blk.coefficients) finite = finite && std::isfinite(c.real()) && std::isfinite(c.imag());
    }
    if (!finite)
        fail(ErrorCode::CorruptFile, "non-finite plane-wave coefficient at spin " + std::to_string(spin + 1) +
                                         ", k " + std::to_string(k + 1));

    if (blk.encoding == WavefunctionEncoding::GammaHalf) {
        blk.gamma_nonzero_scale = 1.0 / std::sqrt(2.0);
        if (header_.precision == CoefficientPrecision::Single) return; // scaled during conversion
        const auto& G = *blk.gvectors;
        for (int b = 0; b < count; ++b) {
            auto* c = blk.coefficients.data() + static_cast<std::size_t>(b) * ng;
            for (std::size_t g = 0; g < ng; ++g)
                if (!(G[g][0] == 0 && G[g][1] == 0 && G[g][2] == 0)) c[g] *= blk.gamma_nonzero_scale;
        }
    }
}

} // namespace prowl
