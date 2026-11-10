// SPDX-License-Identifier: GPL-3.0-or-later
/// Gate C toy models for the PAW coupling matrix D (no licensed data).
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "prowl/projection.hpp"
#include "prowl/twocenter.hpp"

using namespace prowl;

namespace {
/// Fake dataset: one l=1 channel with u_ae - u_ps = r * R_compact(r).
PawDataset toy_dataset(const std::string& el, const RadialSto& compact, bool identity) {
    PawDataset d;
    d.element = el;
    d.symbol = el;
    d.psmaxn = 30.0;
    for (int i = 0; i < 900; ++i) d.grid.push_back(1e-5 * std::exp(i * std::log(4.0 / 1e-5) / 899));
    PawChannel c;
    c.l = 1;
    c.projector_q.assign(100, 0.0);
    for (double r : d.grid) {
        c.u_ps.push_back(0.3 * r * std::exp(-r));
        c.u_ae.push_back(c.u_ps.back() + (identity ? 0.0 : r * compact.value(r)));
    }
    d.channels.push_back(c);
    return d;
}
} // namespace

TEST_CASE("off-site PAW coupling equals analytic two-centre overlap (toy)", "[paw][toy]") {
    const auto lib = StoLibrary::load_default();
    RadialSto compact;
    compact.l = 1; compact.n = 2; compact.label = "2p";
    compact.primitives = {{2, 9.0, 1.0}}; // ~ vanishes beyond 4 A, like a PAW sphere
    Structure s;
    s.lattice = Lattice(Mat3{Vec3{14, 0, 0}, Vec3{0, 14, 0}, Vec3{0, 0, 14}});
    s.species = {"C", "Si"};
    s.counts = {1, 1};
    s.atom_type = {0, 1};
    s.frac = {Vec3{0.1, 0.2, 0.3}, Vec3{0.1 + 1.1 / 14, 0.2 - 0.9 / 14, 0.3 + 0.7 / 14}};
    AtomicBasis basis(s, lib, {{"C", {"2p"}}, {"Si", {"3s"}}});
    for (bool identity : {false, true}) {
        ProjectionOptions o;
        o.offsite_threshold = 1e-12;
        // toy PAW sphere (~3 A) exceeds the bond length, so the neighbour cusp lies inside it:
        // use a fine angular grid (real PAW spheres converge with the 16x32 default)
        o.angular_ntheta = 32; o.angular_nphi = 64;

        ProjectionModel m(s, basis, {toy_dataset("C", compact, identity), toy_dataset("Si", compact, identity)}, o);
        const auto D = m.paw_coupling({0, 0, 0});
        // rows 0..2: C1_2p; cols 3..5: Si projectors (l=1)
        // reference with a high q cutoff: the compact (zeta = 9/A) function needs q >> 200/A
        TwoCenterOverlap tc(lib.shell("C", "2p"), compact, 600.0, 40001);
        const Vec3 d = s.lattice.to_cartesian({s.frac[1][0] - s.frac[0][0], s.frac[1][1] - s.frac[0][1], s.frac[1][2] - s.frac[0][2]});
        double ref[9];
        tc.evaluate(d, ref);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const double expect = identity ? 0.0 : ref[i * 3 + j];
                INFO("identity=" << identity << " i=" << i << " j=" << j << " D=" << D(i, 3 + j).real() << " ref=" << expect);
                REQUIRE(std::abs(D(i, 3 + j).real() - expect) < 2e-7);
            }
        // on-site block (C 2p with C l=1 channel) = radial overlap, diagonal in m
        if (identity) REQUIRE(D.cwiseAbs().maxCoeff() < 1e-14);
    }
}
