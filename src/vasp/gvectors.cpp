// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/gvectors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "prowl/constants.hpp"

namespace prowl {
namespace {

/// Frequency order along one axis: 0, 1, ..., n, -n, ..., -1.
std::vector<int> axis_order(int n) {
    std::vector<int> f;
    f.reserve(2 * n + 1);
    for (int i = 0; i <= n; ++i) f.push_back(i);
    for (int i = -n; i <= -1; ++i) f.push_back(i);
    return f;
}

IVec3 axis_bounds(const Lattice& lat, const Vec3& k, double encut_ev) {
    require(std::isfinite(encut_ev) && encut_ev > 0 && std::isfinite(lat.volume) && lat.volume > 0,
            ErrorCode::InvalidArgument, "finite positive ENCUT and initialized lattice required");
    // |k+G| (in cycles/Angstrom) <= gmax; |(k+G)_i| <= gmax*|a_i|.
    const double gmax = std::sqrt(encut_ev / constants::vasp_hsqdtm) / constants::two_pi;
    IVec3 n{};
    for (int i = 0; i < 3; ++i) {
        require(std::isfinite(k[i]), ErrorCode::InvalidArgument, "fractional k coordinates must be finite");
        const double bound = std::ceil(gmax * norm(lat.a[i]) + std::abs(k[i]));
        require(std::isfinite(bound) && bound < (std::numeric_limits<int>::max() - 3) / 2,
                ErrorCode::InvalidArgument, "plane-wave enumeration bounds exceed supported integer range");
        n[i] = static_cast<int>(bound) + 1;
    }
    return n;
}

template <class Filter>
std::vector<IVec3> generate(const Lattice& lat, const Vec3& k, double encut_ev, Filter keep) {
    require(encut_ev > 0, ErrorCode::InvalidArgument, "ENCUT must be positive");
    const IVec3 n = axis_bounds(lat, k, encut_ev);
    const auto fx = axis_order(n[0]);
    const auto fy = axis_order(n[1]);
    const auto fz = axis_order(n[2]);
    std::vector<IVec3> out;
    for (int gz : fz)
        for (int gy : fy)
            for (int gx : fx) {
                const IVec3 g{gx, gy, gz};
                if (!keep(g)) continue;
                if (kinetic_energy_ev(lat, k, g) < encut_ev) out.push_back(g);
            }
    return out;
}

} // namespace

double kinetic_energy_ev(const Lattice& lat, const Vec3& k, const IVec3& g) {
    // Same arithmetic order as VASP: q_c = (sum_i (G_i + k_i) B(c,i)) * 2pi.
    double e = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double qc = ((g[0] + k[0]) * lat.b[0][c] + (g[1] + k[1]) * lat.b[1][c] + (g[2] + k[2]) * lat.b[2][c]) *
                          constants::two_pi;
        e += qc * qc;
    }
    return constants::vasp_hsqdtm * e;
}

std::vector<IVec3> generate_gvectors_full(const Lattice& lat, const Vec3& k, double encut_ev) {
    return generate(lat, k, encut_ev, [](const IVec3&) { return true; });
}

std::vector<IVec3> generate_gvectors_gamma_half(const Lattice& lat, double encut_ev, GammaHalf half) {
    const Vec3 k{0.0, 0.0, 0.0};
    if (half == GammaHalf::X) {
        return generate(lat, k, encut_ev, [](const IVec3& g) {
            return g[0] > 0 || (g[0] == 0 && g[1] > 0) || (g[0] == 0 && g[1] == 0 && g[2] >= 0);
        });
    }
    return generate(lat, k, encut_ev, [](const IVec3& g) {
        return g[2] > 0 || (g[2] == 0 && g[1] > 0) || (g[2] == 0 && g[1] == 0 && g[0] >= 0);
    });
}

SphereMargin cutoff_sphere_margin(const Lattice& lat, const Vec3& k, double encut_ev) {
    const IVec3 n = axis_bounds(lat, k, encut_ev);
    SphereMargin m{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    for (int gz = -n[2]; gz <= n[2]; ++gz)
        for (int gy = -n[1]; gy <= n[1]; ++gy)
            for (int gx = -n[0]; gx <= n[0]; ++gx) {
                const double e = kinetic_energy_ev(lat, k, {gx, gy, gz});
                if (e < encut_ev)
                    m.inside_closest_ev = std::min(m.inside_closest_ev, encut_ev - e);
                else
                    m.outside_closest_ev = std::min(m.outside_closest_ev, e - encut_ev);
            }
    return m;
}

} // namespace prowl
