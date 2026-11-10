// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/ylm.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <cmath>
#include <mutex>
#include <vector>

#include "prowl/constants.hpp"
#include "prowl/errors.hpp"

namespace prowl::ylm {

void real_ylm(int lmax, const Vec3& v, double* out) noexcept {
    const double r = norm(v);
    double ct = 1.0, st = 0.0, cp = 1.0, sp = 0.0;
    if (r > 1e-300) {
        ct = std::clamp(v[2] / r, -1.0, 1.0);
        const double rho = std::sqrt(v[0] * v[0] + v[1] * v[1]);
        st = rho / r;
        if (rho > 1e-300 * r) { cp = v[0] / rho; sp = v[1] / rho; }
    }
    // Associated Legendre P_l^m(ct) without Condon-Shortley phase.
    double P[lmax_supported + 1][lmax_supported + 1] = {};
    P[0][0] = 1.0;
    for (int m = 1; m <= lmax; ++m) P[m][m] = P[m - 1][m - 1] * (2.0 * m - 1.0) * st;
    for (int m = 0; m < lmax; ++m) P[m + 1][m] = (2.0 * m + 1.0) * ct * P[m][m];
    for (int m = 0; m <= lmax; ++m)
        for (int l = m + 2; l <= lmax; ++l)
            P[l][m] = ((2.0 * l - 1.0) * ct * P[l - 1][m] - (l + m - 1.0) * P[l - 2][m]) / (l - m);
    // cos(m p), sin(m p) by recursion
    double cm[lmax_supported + 1], sm[lmax_supported + 1];
    cm[0] = 1.0; sm[0] = 0.0;
    for (int m = 1; m <= lmax; ++m) {
        cm[m] = cm[m - 1] * cp - sm[m - 1] * sp;
        sm[m] = sm[m - 1] * cp + cm[m - 1] * sp;
    }
    for (int l = 0; l <= lmax; ++l) {
        double fact_ratio = 1.0; // (l-m)!/(l+m)!
        out[index(l, 0)] = std::sqrt((2.0 * l + 1.0) / constants::four_pi) * P[l][0];
        for (int m = 1; m <= l; ++m) {
            fact_ratio /= static_cast<double>((l + m) * (l - m + 1));
            const double N = std::sqrt((2.0 * l + 1.0) / constants::four_pi * fact_ratio) * std::sqrt(2.0);
            out[index(l, m)] = N * P[l][m] * cm[m];
            out[index(l, -m)] = N * P[l][m] * sm[m];
        }
    }
}

namespace {
struct GauntTable {
    static constexpr int L = lmax_supported;
    static constexpr int N = count(L);
    std::vector<double> t; // N^3
    GauntTable() : t(static_cast<std::size_t>(N) * N * N, 0.0) {
        // Gauss-Legendre in cos(theta) x uniform phi: exact for degree <= 3L.
        const int nt = 3 * L / 2 + 4, np = 3 * L + 4;
        std::vector<double> x(nt), w(nt);
        for (int i = 0; i < nt; ++i) { // Newton on Legendre polynomial
            double z = std::cos(constants::pi * (i + 0.75) / (nt + 0.5)), pp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < nt; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1); }
                pp = nt * (z * p1 - p2) / (z * z - 1.0);
                const double dz = p1 / pp;
                z -= dz;
                if (std::abs(dz) < 1e-15) break;
            }
            x[i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        }
        std::vector<double> y(N);
        for (int i = 0; i < nt; ++i)
            for (int j = 0; j < np; ++j) {
                const double phi = constants::two_pi * j / np, st = std::sqrt(1.0 - x[i] * x[i]);
                real_ylm(L, {st * std::cos(phi), st * std::sin(phi), x[i]}, y.data());
                const double wt = w[i] * constants::two_pi / np;
                for (int a = 0; a < N; ++a)
                    for (int b = a; b < N; ++b) {
                        const double yab = wt * y[a] * y[b];
                        for (int c = b; c < N; ++c) t[(static_cast<std::size_t>(a) * N + b) * N + c] += yab * y[c];
                    }
            }
        for (auto& v : t) if (std::abs(v) < 1e-13) v = 0.0;
    }
    double get(int a, int b, int c) const {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return t[(static_cast<std::size_t>(a) * N + b) * N + c];
    }
};
} // namespace

Eigen::MatrixXd rotation_matrix(int l, const Mat3& axes) {
    require(l >= 0 && l <= lmax_supported, ErrorCode::InvalidArgument, "rotation_matrix: l out of range");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double d = axes[i][0] * axes[j][0] + axes[i][1] * axes[j][1] + axes[i][2] * axes[j][2];
            require(std::abs(d - (i == j ? 1.0 : 0.0)) < 1e-10, ErrorCode::InvalidArgument,
                    "rotation_matrix: axes are not orthonormal");
        }
    const Vec3 zc = cross(axes[0], axes[1]);
    require(zc[0] * axes[2][0] + zc[1] * axes[2][1] + zc[2] * axes[2][2] > 0.0, ErrorCode::InvalidArgument,
            "rotation_matrix: axes are not right-handed");
    // Y(axes v) = D Y(v) holds exactly on the (2l+1)-dimensional space; fit it on a fixed spiral of points
    // (many more than 2l+1, so the least-squares solution is exact to rounding).
    const int n = 2 * l + 1, npts = 8 * n + 16, off = index(l, -l);
    Eigen::MatrixXd A(npts, n), B(npts, n);
    std::vector<double> y(static_cast<std::size_t>(count(l)));
    for (int i = 0; i < npts; ++i) {
        const double z = 1.0 - (2.0 * i + 1.0) / npts, rr = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = i * 2.399963229728653; // golden angle
        const Vec3 v{rr * std::cos(phi), rr * std::sin(phi), z};
        const Vec3 w{axes[0][0] * v[0] + axes[0][1] * v[1] + axes[0][2] * v[2],
                     axes[1][0] * v[0] + axes[1][1] * v[1] + axes[1][2] * v[2],
                     axes[2][0] * v[0] + axes[2][1] * v[1] + axes[2][2] * v[2]};
        real_ylm(l, v, y.data());
        for (int m = 0; m < n; ++m) A(i, m) = y[static_cast<std::size_t>(off + m)];
        real_ylm(l, w, y.data());
        for (int m = 0; m < n; ++m) B(i, m) = y[static_cast<std::size_t>(off + m)];
    }
    // B = A D^T
    Eigen::MatrixXd D = A.colPivHouseholderQr().solve(B).transpose();
    for (Eigen::Index i = 0; i < D.size(); ++i)
        if (std::abs(D.data()[i]) < 1e-14) D.data()[i] = 0.0;
    return D;
}

double gaunt(int l1, int m1, int l2, int m2, int l3, int m3) {
    require(l1 >= 0 && l1 <= lmax_supported && l2 >= 0 && l2 <= lmax_supported && l3 >= 0 && l3 <= lmax_supported &&
                m1 >= -l1 && m1 <= l1 && m2 >= -l2 && m2 <= l2 && m3 >= -l3 && m3 <= l3,
            ErrorCode::InvalidArgument, "gaunt: angular indices out of range");
    static const GauntTable table;
    return table.get(index(l1, m1), index(l2, m2), index(l3, m3));
}

std::string orbital_name(int l, int m) {
    static const char* p[] = {"p_y", "p_z", "p_x"};
    static const char* d[] = {"d_xy", "d_yz", "d_z^2", "d_xz", "d_x^2-y^2"};
    static const char* f[] = {"f_y(3x^2-y^2)", "f_xyz", "f_yz^2", "f_z^3", "f_xz^2", "f_z(x^2-y^2)", "f_x(x^2-3y^2)"};
    switch (l) {
    case 0: return "s";
    case 1: return p[m + 1];
    case 2: return d[m + 2];
    case 3: return f[m + 3];
    default: return "l" + std::to_string(l) + "m" + std::to_string(m);
    }
}

} // namespace prowl::ylm
