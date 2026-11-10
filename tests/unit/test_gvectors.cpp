// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <set>

#include "prowl/gvectors.hpp"

using namespace prowl;

namespace {
Lattice fcc(double a) { return Lattice(Mat3{Vec3{0, a / 2, a / 2}, Vec3{a / 2, 0, a / 2}, Vec3{a / 2, a / 2, 0}}); }
}

TEST_CASE("full G sphere matches brute force and VASP ordering", "[gvectors]") {
    const Lattice L = fcc(5.43);
    const Vec3 k{0.25, -0.125, 0.5};
    const double encut = 250.0;
    auto G = generate_gvectors_full(L, k, encut);
    std::size_t brute = 0;
    for (int x = -20; x <= 20; ++x)
        for (int y = -20; y <= 20; ++y)
            for (int z = -20; z <= 20; ++z)
                if (kinetic_energy_ev(L, k, {x, y, z}) < encut) ++brute;
    REQUIRE(G.size() == brute);
    // z outermost: z frequency order 0,1,..,-1 must be non-decreasing in "order rank"
    auto rank = [](int f) { return f >= 0 ? f : 1000 + (f + 500); };
    for (std::size_t i = 1; i < G.size(); ++i) REQUIRE(rank(G[i][2]) >= rank(G[i - 1][2]));
    std::set<IVec3> uniq(G.begin(), G.end());
    REQUIRE(uniq.size() == G.size());
}

TEST_CASE("gamma half set plus mirror reproduces full sphere", "[gvectors]") {
    const Lattice L = fcc(5.43);
    for (auto half : {GammaHalf::X, GammaHalf::Z}) {
        auto full = generate_gvectors_full(L, {0, 0, 0}, 300.0);
        auto h = generate_gvectors_gamma_half(L, 300.0, half);
        REQUIRE(h.size() == (full.size() + 1) / 2);
        REQUIRE(h.front() == IVec3{0, 0, 0});
        std::set<IVec3> s;
        for (auto g : h) { s.insert(g); s.insert({-g[0], -g[1], -g[2]}); }
        REQUIRE(s == std::set<IVec3>(full.begin(), full.end()));
    }
}
