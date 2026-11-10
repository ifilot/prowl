// SPDX-License-Identifier: GPL-3.0-or-later
/// Minimal native entry point: `prowl-native wavecar-info <WAVECAR>`.
/// The user-facing CLI is the Python `prowl` command; this binary exists for
/// environments without Python and as a smoke test of the C++ reader.
#include <cstdio>
#include <exception>
#include <string>

#include "prowl/wavecar.hpp"

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "wavecar-info") {
        std::fprintf(stderr, "usage: %s wavecar-info <WAVECAR> [--bands]\n", argv[0]);
        return 2;
    }
    try {
        prowl::WavecarReader r(argv[2]);
        const auto& h = r.header();
        std::printf("file           : %s (%llu bytes)\n", argv[2], static_cast<unsigned long long>(h.file_size));
        std::printf("RTAG           : %d (%s precision%s)\n", h.rtag,
                    h.precision == prowl::CoefficientPrecision::Single ? "single" : "double",
                    h.header_spans_records ? ", spanning k headers" : "");
        std::printf("record length  : %llu bytes\n", static_cast<unsigned long long>(h.record_length));
        std::printf("spins/k/bands  : %d / %d / %d\n", h.nspin, h.nkpts, h.nbands);
        std::printf("ENCUT          : %.4f eV\n", h.encut_ev);
        std::printf("volume         : %.6f A^3\n", h.lattice.volume);
        std::printf("encoding       : %s\n", prowl::to_string(r.encoding()));
        std::printf("sphere margin  : %.3e eV\n", r.layout().min_sphere_margin_ev);
        for (const auto& n : r.layout().notes) std::printf("note           : %s\n", n.c_str());
        const bool bands = argc > 3 && std::string(argv[3]) == "--bands";
        if (bands && r.encoding() != prowl::WavefunctionEncoding::Spinor) {
            for (int s = 0; s < h.nspin; ++s)
                for (int k = 0; k < std::min(h.nkpts, 3); ++k) {
                    auto blk = r.read_block(s, k, 0, h.nbands);
                    auto nrm = blk.plane_wave_norms();
                    std::printf("spin %d k %d nplw %zu:", s + 1, k + 1, blk.ng());
                    for (int b = 0; b < std::min(h.nbands, 8); ++b) std::printf(" %.6f", nrm[b]);
                    std::printf("\n");
                }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
