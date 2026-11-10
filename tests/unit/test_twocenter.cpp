// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "prowl/twocenter.hpp"

using namespace prowl;
using Catch::Approx;

namespace {
RadialSto sto1s(double z) {
    RadialSto r;
    r.l = 0; r.n = 1; r.label = "1s";
    r.primitives = {{1, z, 1.0}};
    return r;
}
RadialSto sto2p(double z) {
    RadialSto r;
    r.l = 1; r.n = 2; r.label = "2p";
    r.primitives = {{2, z, 1.0}};
    return r;
}
} // namespace

TEST_CASE("1s-1s overlap vs analytic formula", "[twocenter]") {
    const double z = 1.7;
    TwoCenterOverlap t(sto1s(z), sto1s(z));
    double o;
    for (double d : {0.0, 0.5, 1.2, 2.5, 4.0}) {
        t.evaluate({0.3 * d, -0.4 * d, std::sqrt(0.75) * d}, &o);
        const double x = z * d;
        REQUIRE(o == Approx(std::exp(-x) * (1 + x + x * x / 3)).margin(2e-9));
    }
}

TEST_CASE("overlap symmetry and rotation behaviour", "[twocenter]") {
    TwoCenterOverlap ab(sto1s(1.3), sto2p(1.1)), ba(sto2p(1.1), sto1s(1.3));
    const Vec3 d{0.4, -1.1, 0.7};
    double o1[3], o2[3];
    ab.evaluate(d, o1);
    ba.evaluate({-d[0], -d[1], -d[2]}, o2);
    for (int m = 0; m < 3; ++m) REQUIRE(o1[m] == Approx(o2[m]).margin(1e-10));
    // <s(0)|p(r-d)>: p lobe pointing back towards s has negative overlap: O_m proportional to -d_m
    const double n = norm(d);
    REQUIRE(o1[0] / o1[2] == Approx(d[1] / d[0]).epsilon(1e-8)); // (p_y, p_z, p_x)
    REQUIRE(o1[1] / o1[2] == Approx(d[2] / d[0]).epsilon(1e-8));
    REQUIRE(o1[2] * d[0] < 0);
    (void)n;
}
