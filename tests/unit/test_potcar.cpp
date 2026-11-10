// SPDX-License-Identifier: GPL-3.0-or-later
/// POTCAR parser tests on a *synthetic* dataset with fabricated numbers (no
/// licensed data is used or committed). Real-POTCAR checks run in pytest only
/// when PROWL_POTCAR_DIR is set.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "prowl/errors.hpp"
#include "prowl/potcar.hpp"

using namespace prowl;

namespace {
void block(std::ofstream& o, int n, double scale) {
    for (int i = 0; i < n; ++i) {
        o << "  " << scale * std::exp(-0.01 * i);
        if (i % 5 == 4 || i == n - 1) o << "\n";
    }
}
} // namespace

TEST_CASE("synthetic POTCAR structure is parsed", "[potcar]") {
    const auto p = std::filesystem::temp_directory_path() / "prowl_test_POTCAR";
    {
        std::ofstream o(p);
        o.precision(15);
        o << "  PAW_PBE Xx 01Jan2000\n   4.0\n parameters from PSCTR are:\n   VRHFIN =Xx: s2p2\n"
          << "   TITEL  = PAW_PBE Xx 01Jan2000\n   POMASS =   12.011; ZVAL   =    4.000    mass and valenz\n"
          << " local part\n  1.0\n";
        block(o, 10, 1.0);
        o << "   25.0      T\n";
        for (int l = 0; l <= 1; ++l) {
            o << " Non local Part\n           " << l << "           1  0.8\n  1.5\n";
            o << " Reciprocal Space Part\n";
            block(o, 100, 2.0);
            o << " Real Space Part\n";
            block(o, 100, 3.0);
        }
        o << " PAW radial sets\n         20  0.7\n(5E20.12)\n augmentation charges (non sperical)\n";
        // Fortran D exponents and the exponent-without-E form occur in real POTCARs (e.g. W DION)
        o << "  1.0D-001  -1.83810562657937D-002  2.5d+000  1.0-100\n";
        o << " grid\n";
        for (int i = 0; i < 20; ++i) o << "  " << 1e-3 * std::exp(0.3 * i) << (i % 5 == 4 ? "\n" : "");
        for (int c = 0; c < 2; ++c) {
            o << " pseudo wavefunction\n";
            block(o, 20, 0.5);
            o << " ae wavefunction\n";
            block(o, 20, 0.6);
        }
        o << " End of Dataset\n";
    }
    auto ds = read_potcar(p);
    REQUIRE(ds.size() == 1);
    REQUIRE(ds[0].element == "Xx");
    REQUIRE(ds[0].zval == Catch::Approx(4.0));
    REQUIRE(ds[0].psmaxn == Catch::Approx(25.0));
    REQUIRE(ds[0].nchannels() == 2);
    REQUIRE(ds[0].nprojectors() == 4);
    REQUIRE(ds[0].channels[1].l == 1);
    REQUIRE(ds[0].grid.size() == 20);
    REQUIRE(ds[0].channels[0].u_ae.size() == 20);
    REQUIRE(ds[0].qpaw.size() == 4);
    REQUIRE(ds[0].qpaw[1] == Catch::Approx(-1.83810562657937e-2));
    REQUIRE(ds[0].qpaw[2] == Catch::Approx(2.5));
    std::filesystem::remove(p);
}
