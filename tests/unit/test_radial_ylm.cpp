// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "prowl/constants.hpp"
#include "prowl/radial.hpp"
#include "prowl/ylm.hpp"

using namespace prowl;
using Catch::Approx;

TEST_CASE("spherical Bessel functions match closed forms", "[radial]") {
    for (double x : {1e-6, 0.01, 0.3, 0.7, 1.5, 2.9, 5.0, 12.0, 47.0}) {
        const double s = std::sin(x), c = std::cos(x);
        const double j0 = s / x, j1 = s / (x * x) - c / x;
        const double j2 = (3 / (x * x) - 1) * s / x - 3 * c / (x * x);
        const double j3 = (15 / (x * x * x) - 6 / x) * s / x - (15 / (x * x) - 1) * c / x;
        const double tol = x < 0.1 ? 1e-9 : 1e-12;
        REQUIRE(radial::sph_bessel(0, x) == Approx(j0).margin(tol));
        REQUIRE(radial::sph_bessel(1, x) == Approx(j1).margin(tol));
        REQUIRE(radial::sph_bessel(2, x) == Approx(j2).margin(tol));
        REQUIRE(radial::sph_bessel(3, x) == Approx(j3).margin(tol));
        // high order against std::sph_bessel reference
        for (int l = 4; l <= 6; ++l) REQUIRE(radial::sph_bessel(l, x) == Approx(std::sph_bessel(l, x)).margin(1e-13));
    }
}

TEST_CASE("cubic spline and log-grid quadrature", "[radial]") {
    std::vector<double> x(201), y(201);
    for (int i = 0; i <= 200; ++i) { x[i] = i * 0.05; y[i] = std::sin(x[i]); }
    radial::CubicSpline sp(x, y);
    for (double t : {0.33, 1.234, 7.77}) REQUIRE(sp(t) == Approx(std::sin(t)).margin(2e-6));
    std::vector<double> r(1001), f(1001);
    for (int i = 0; i <= 1000; ++i) { r[i] = 1e-5 * std::exp(i * 0.02); f[i] = r[i] * r[i] * std::exp(-r[i]); }
    REQUIRE(radial::integrate_log_grid(r, f) == Approx(2.0).epsilon(1e-9));
}

TEST_CASE("real spherical harmonics: convention and orthonormality", "[ylm]") {
    double Y[16];
    ylm::real_ylm(3, {0.0, 1.0, 0.0}, Y);
    const double c1 = std::sqrt(3 / constants::four_pi);
    REQUIRE(Y[ylm::index(1, -1)] == Approx(c1)); // p_y
    REQUIRE(Y[ylm::index(1, 0)] == Approx(0).margin(1e-15));
    ylm::real_ylm(3, {1.0, 1.0, 0.0}, Y);
    REQUIRE(Y[ylm::index(2, -2)] == Approx(std::sqrt(15 / constants::four_pi) * 0.5)); // xy/r^2
    REQUIRE(ylm::orbital_name(1, -1) == "p_y");
    REQUIRE(ylm::orbital_name(2, 0) == "d_z^2");
    // orthonormality through Gaunt with Y00
    const double y00 = 1 / std::sqrt(constants::four_pi);
    for (int l1 = 0; l1 <= 3; ++l1)
        for (int m1 = -l1; m1 <= l1; ++m1)
            for (int l2 = 0; l2 <= 3; ++l2)
                for (int m2 = -l2; m2 <= l2; ++m2)
                    REQUIRE(ylm::gaunt(l1, m1, l2, m2, 0, 0) == Approx((l1 == l2 && m1 == m2) ? y00 : 0.0).margin(1e-13));
    // a known nonzero: <p_z p_z Y20> = 2/sqrt(5)*sqrt(5/(4pi))/... check selection rule & symmetry instead
    REQUIRE(ylm::gaunt(1, 0, 1, 0, 2, 0) == Approx(ylm::gaunt(2, 0, 1, 0, 1, 0)));
    REQUIRE(ylm::gaunt(1, 0, 1, 0, 1, 0) == Approx(0).margin(1e-14));
    REQUIRE(ylm::gaunt(1, 0, 1, 0, 2, 0) == Approx(1.0 / std::sqrt(5.0 * constants::pi)));
}
