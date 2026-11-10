// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "prowl/basis.hpp"
#include "prowl/constants.hpp"
#include "prowl/radial.hpp"

using namespace prowl;
using Catch::Approx;

namespace {
double numeric_bessel(int N, int l, double z, double q) {
    // fine log grid reference
    const int n = 40001;
    std::vector<double> r(n), f(n);
    for (int i = 0; i < n; ++i) {
        r[i] = 1e-8 * std::exp(i * std::log(80.0 / z / 1e-8) / (n - 1));
        f[i] = std::pow(r[i], N) * std::exp(-z * r[i]) * radial::sph_bessel(l, q * r[i]);
    }
    return radial::integrate_log_grid(r, f);
}
} // namespace

TEST_CASE("analytic STO Bessel integral vs quadrature", "[basis]") {
    for (int l = 0; l <= 3; ++l)
        for (int N = l + 2; N <= l + 5; ++N)
            for (double q : {0.0, 0.1, 0.29, 0.31, 0.9, 2.5, 9.0, 40.0}) {
                const double z = 1.3;
                const double ref = numeric_bessel(N, l, z, q * z);
                const double v = sto_bessel_integral(N, l, z, q * z);
                REQUIRE(v == Approx(ref).epsilon(1e-7).margin(1e-12));
            }
}

TEST_CASE("Koga library: normalisation, Parseval, selection labels", "[basis]") {
    const auto lib = StoLibrary::load_default();
    REQUIRE(lib.info().name == "koga_hf_sto");
    for (const char* el : {"H", "C", "Si", "Ga", "As", "Fe", "Au"})
        for (const auto& s : lib.shells(el)) REQUIRE(s.norm2() == Approx(1.0).epsilon(1e-7));
    const auto& c2p = lib.shell("C", "2p");
    REQUIRE(c2p.l == 1);
    // Parseval: (2/pi) int q^2 Rt(q)^2 dq = int r^2 R^2 dr = 1
    const int nq = 20001;
    const double qmax = 400.0, h = qmax / (nq - 1);
    const auto w = radial::simpson_weights(nq, h);
    double s = 0;
    for (int i = 0; i < nq; ++i) { const double q = i * h, t = c2p.fourier(q); s += w[i] * q * q * t * t; }
    REQUIRE(2.0 / constants::pi * s == Approx(1.0).epsilon(1e-6));

    Structure st;
    st.lattice = Lattice(Mat3{Vec3{3, 0, 0}, Vec3{0, 3, 0}, Vec3{0, 0, 3}});
    st.species = {"C"};
    st.counts = {2};
    st.atom_type = {0, 0};
    st.frac = {Vec3{0, 0, 0}, Vec3{0.5, 0.5, 0.5}};
    AtomicBasis b(st, lib, {{"C", {"2s", "2p"}}});
    REQUIRE(b.size() == 8);
    REQUIRE(b.orbitals()[1].label == "C1_2p_y");
    REQUIRE(b.orbitals()[7].label == "C2_2p_x");
}
