// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// Writer for small synthetic WAVECAR files with known content (test-only).
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "prowl/gvectors.hpp"

namespace prowl::testing {

struct SyntheticK {
    Vec3 k;
    std::vector<std::vector<std::complex<double>>> bands; // stored values per band
    std::vector<double> eig, occ;
};

inline void write_wavecar(const std::filesystem::path& p, int rtag, int nspin, double encut, const Mat3& a,
                          const std::vector<std::vector<SyntheticK>>& spins, std::uint64_t recl_bytes) {
    const bool single = (rtag == 45200 || rtag == 53300);
    const int nk = static_cast<int>(spins[0].size());
    const int nb = static_cast<int>(spins[0][0].bands.size());
    const std::uint64_t dpr = recl_bytes / 8;
    const std::uint64_t hdr = 4 + 3 * static_cast<std::uint64_t>(nb);
    const int rph = (rtag == 53300 || rtag == 53310) ? static_cast<int>((hdr + dpr - 1) / dpr) : 1;
    const std::uint64_t nrec = 2 + static_cast<std::uint64_t>(nspin) * nk * (rph + nb);
    std::vector<char> buf(nrec * recl_bytes, 0);
    auto put = [&](std::uint64_t rec, std::uint64_t idx, double v) {
        std::memcpy(&buf[rec * recl_bytes + idx * 8], &v, 8);
    };
    put(0, 0, static_cast<double>(recl_bytes));
    put(0, 1, nspin);
    put(0, 2, rtag);
    put(1, 0, nk);
    put(1, 1, nb);
    put(1, 2, encut);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) put(1, 3 + 3 * j + i, a[j][i]);
    put(1, 12, 1.234);
    std::uint64_t rec = 2;
    for (int s = 0; s < nspin; ++s)
        for (int k = 0; k < nk; ++k) {
            const auto& K = spins[s][k];
            std::vector<double> h(hdr);
            h[0] = static_cast<double>(K.bands[0].size());
            h[1] = K.k[0]; h[2] = K.k[1]; h[3] = K.k[2];
            for (int b = 0; b < nb; ++b) { h[4 + 3 * b] = K.eig[b]; h[5 + 3 * b] = 0.0; h[6 + 3 * b] = K.occ[b]; }
            for (std::uint64_t i = 0; i < hdr; ++i) put(rec + i / dpr, i % dpr, h[i]);
            rec += rph;
            for (int b = 0; b < nb; ++b, ++rec) {
                if (K.bands[b].size() * (single ? 8u : 16u) > recl_bytes)
                    throw std::runtime_error("synthetic WAVECAR: record too short for coefficients");
                char* dst = &buf[rec * recl_bytes];
                for (std::size_t g = 0; g < K.bands[b].size(); ++g) {
                    if (single) {
                        const std::complex<float> c(static_cast<float>(K.bands[b][g].real()), static_cast<float>(K.bands[b][g].imag()));
                        std::memcpy(dst + g * 8, &c, 8);
                    } else {
                        std::memcpy(dst + g * 16, &K.bands[b][g], 16);
                    }
                }
            }
        }
    std::ofstream(p, std::ios::binary).write(buf.data(), static_cast<std::streamsize>(buf.size()));
}

} // namespace prowl::testing
