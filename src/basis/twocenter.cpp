// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/twocenter.hpp"

#include <algorithm>
#include <cmath>

#include "prowl/errors.hpp"
#include "prowl/radial.hpp"
#include "prowl/ylm.hpp"

namespace prowl {

namespace {
double log_grid_r(int i) { return 1e-7 * std::exp(i * std::log(80.0 / 1e-7) / 4000.0); }
} // namespace

TwoCenterShell TwoCenterShell::prepare(const RadialSto& r, double qmax, int nq, double extent_threshold) {
    require(nq >= 5 && nq % 2 == 1, ErrorCode::InvalidArgument, "nq must be odd and >= 5");
    require(std::isfinite(qmax) && qmax > 0 && std::isfinite(extent_threshold) && extent_threshold > 0 &&
                r.l >= 0 && r.l <= ylm::lmax_supported, ErrorCode::InvalidArgument, "invalid overlap shell or radial range");
    TwoCenterShell s;
    s.l = r.l;
    s.qmax = qmax;
    s.nq = nq;
    s.extent = r.extent(extent_threshold);
    s.ft.resize(nq);
    s.rlog.resize(4001);
    const double h = qmax / (nq - 1);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nq; ++i) s.ft[i] = r.fourier(i * h);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 4001; ++i) s.rlog[i] = r.value(log_grid_r(i));
    return s;
}

TwoCenterOverlap::TwoCenterOverlap(const RadialSto& a, const RadialSto& b, double qmax, int nq, double dd,
                                   double extent_threshold)
    : TwoCenterOverlap(TwoCenterShell::prepare(a, qmax, nq, extent_threshold),
                       TwoCenterShell::prepare(b, qmax, nq, extent_threshold), dd) {}

TwoCenterOverlap::TwoCenterOverlap(const TwoCenterShell& a, const TwoCenterShell& b, double dd) : l1_(a.l), l2_(b.l) {
    require(std::isfinite(dd) && dd >= 0 && a.nq >= 5 && a.nq % 2 == 1 && std::isfinite(a.qmax) && a.qmax > 0 &&
                a.ft.size() == static_cast<std::size_t>(a.nq) && b.ft.size() == static_cast<std::size_t>(b.nq) &&
                a.rlog.size() == 4001 && b.rlog.size() == 4001 && l1_ >= 0 && l2_ >= 0,
            ErrorCode::InvalidArgument, "invalid prepared overlap shells or distance spacing");
    require(a.nq == b.nq && a.qmax == b.qmax, ErrorCode::InvalidArgument, "shells prepared on different q grids");
    require(l1_ + l2_ <= ylm::lmax_supported, ErrorCode::UnsupportedFeature, "angular momentum too high for overlap");
    for (int L = std::abs(l1_ - l2_); L <= l1_ + l2_; L += 2) {
        const int p = ((l1_ - l2_ - L) % 4 + 4) % 4; // i^p, p even
        const double phase = (p == 0) ? 1.0 : -1.0;
        for (int M = -L; M <= L; ++M)
            for (int m1 = -l1_; m1 <= l1_; ++m1)
                for (int m2 = -l2_; m2 <= l2_; ++m2) {
                    const double G = ylm::gaunt(l1_, m1, l2_, m2, L, M);
                    if (G != 0.0)
                        couplings_.push_back({(m1 + l1_) * (2 * l2_ + 1) + (m2 + l2_), ylm::index(L, M), L, 8.0 * phase * G});
                }
    }
    const int nq = a.nq;
    if (l1_ == l2_) {
        // Same-centre radial overlap via a fine log-grid quadrature of the analytic functions.
        std::vector<double> r(4001), f(4001);
        for (int i = 0; i < 4001; ++i) {
            r[i] = log_grid_r(i);
            f[i] = a.rlog[i] * b.rlog[i] * r[i] * r[i];
        }
        same_center_ = radial::integrate_log_grid(r, f);
    }
    cutoff_ = a.extent + b.extent;
    const double h = a.qmax / (nq - 1);
    const auto w = radial::simpson_weights(nq, h);
    q_.resize(nq);
    g_.resize(nq);
    for (int i = 0; i < nq; ++i) {
        q_[i] = i * h;
        g_[i] = w[i] * q_[i] * q_[i] * a.ft[i] * b.ft[i];
    }
    // Truncate the q integral where the remaining tail cannot matter: |j_L| <= 1, so dropping the points
    // above nq_eff_ changes every J_L by at most sum_{i >= nq_eff_} |g_i| <= 1e-15 sum |g| (diffuse shells:
    // the integrand dies far below qmax, which is set by the most compact primitive in the library).
    double gsum = 0.0;
    for (double v : g_) gsum += std::abs(v);
    nq_eff_ = g_.size();
    double tail = 0.0;
    while (nq_eff_ > 1 && tail + std::abs(g_[nq_eff_ - 1]) <= 1e-15 * gsum) tail += std::abs(g_[--nq_eff_]);
    if (dd > 0.0) {
        // Tabulate from a small positive distance (d = 0 is handled analytically).
        tab_dmin_ = 0.5 * dd;
        const int nd = static_cast<int>((cutoff_ - tab_dmin_) / dd) + 4;
        const int Lmax = l1_ + l2_;
        std::vector<double> x(nd), vals(static_cast<std::size_t>(nd) * (Lmax + 1));
        for (int i = 0; i < nd; ++i) x[i] = tab_dmin_ + i * dd;
        const int nblk = (nd + 7) / 8;
#pragma omp parallel for schedule(dynamic, 4)
        for (int ib = 0; ib < nblk; ++ib) {
            const int i0 = 8 * ib, n = std::min(8, nd - i0);
            radial_integrals_block(&x[i0], n, &vals[static_cast<std::size_t>(i0) * (Lmax + 1)]);
        }
        jtab_.resize(Lmax + 1);
        for (int L = std::abs(l1_ - l2_); L <= Lmax; L += 2) {
            std::vector<double> y(nd);
            for (int i = 0; i < nd; ++i) y[i] = vals[static_cast<std::size_t>(i) * (Lmax + 1) + L];
            jtab_[L] = radial::CubicSpline(x, y);
        }
    }
}

void TwoCenterOverlap::radial_integrals(double dist, double* J) const {
    const int Lmax = l1_ + l2_, L0 = std::abs(l1_ - l2_);
    for (int L = 0; L <= Lmax; ++L) J[L] = 0.0;
    double jb[16];
    // Small arguments (series / Miller recursion, as radial::sph_bessel_all) point by point.
    const double xsw = std::max(0.5, static_cast<double>(Lmax));
    std::size_t i = 0;
    for (; i < nq_eff_ && q_[i] * dist <= xsw; ++i) {
        if (g_[i] == 0.0) continue;
        radial::sph_bessel_all(Lmax, q_[i] * dist, jb);
        for (int L = L0; L <= Lmax; L += 2) J[L] += g_[i] * jb[L];
    }
    if (i >= nq_eff_) return;
    // x > max(0.5, Lmax): j_0, j_1 from sin/cos and upward recursion (the sph_bessel_all branch for x > lmax).
    // q is uniform, so sin(q_i d), cos(q_i d) follow by rotation; resynchronised every 32 points (error ~1e-15).
    const double h = q_[1] - q_[0], th = h * dist, cr = std::cos(th), sr = std::sin(th);
    double s = 0.0, c = 1.0;
    for (std::size_t i0 = i; i < nq_eff_; ++i) {
        if (((i - i0) & 31u) == 0) { s = std::sin(q_[i] * dist); c = std::cos(q_[i] * dist); }
        if (g_[i] != 0.0) {
            const double inv = 1.0 / (q_[i] * dist);
            jb[0] = s * inv;
            if (Lmax >= 1) {
                jb[1] = (jb[0] - c) * inv;
                for (int l = 1; l < Lmax; ++l) jb[l + 1] = (2.0 * l + 1.0) * inv * jb[l] - jb[l - 1];
            }
            for (int L = L0; L <= Lmax; L += 2) J[L] += g_[i] * jb[L];
        }
        const double s2 = s * cr + c * sr;
        c = c * cr - s * sr;
        s = s2;
    }
}

void TwoCenterOverlap::radial_integrals_block(const double* dist, int n, double* J) const {
    constexpr int W = 8;
    const int Lmax = l1_ + l2_, L0 = std::abs(l1_ - l2_), nl = Lmax + 1;
    for (int k = 0; k < n * nl; ++k) J[k] = 0.0;
    double jb[16];
    const double xsw = std::max(0.5, static_cast<double>(Lmax));
    // point-by-point region (small q d) per lane, up to a common start index of the vectorised loop
    std::size_t istart = 0;
    for (int k = 0; k < n; ++k) {
        std::size_t i = 0;
        while (i < nq_eff_ && q_[i] * dist[k] <= xsw) ++i;
        istart = std::max(istart, i);
    }
    for (int k = 0; k < n; ++k)
        for (std::size_t i = 0; i < istart && i < nq_eff_; ++i) {
            if (g_[i] == 0.0) continue;
            radial::sph_bessel_all(Lmax, q_[i] * dist[k], jb);
            for (int L = L0; L <= Lmax; L += 2) J[k * nl + L] += g_[i] * jb[L];
        }
    if (istart >= nq_eff_) return;
    // vectorised region: W independent sin/cos rotations (lanes beyond n are dummies)
    const double h = q_[1] - q_[0];
    double s[W], c[W], cr[W], sr[W], invd[W], acc[W * 8];
    for (int k = 0; k < W; ++k) {
        const double d = dist[std::min(k, n - 1)];
        cr[k] = std::cos(h * d);
        sr[k] = std::sin(h * d);
        invd[k] = 1.0 / d;
    }
    for (int k = 0; k < W * 8; ++k) acc[k] = 0.0;
    for (std::size_t i = istart; i < nq_eff_; ++i) {
        if (((i - istart) & 31u) == 0)
            for (int k = 0; k < W; ++k) {
                const double x = q_[i] * dist[std::min(k, n - 1)];
                s[k] = std::sin(x);
                c[k] = std::cos(x);
            }
        const double gi = g_[i], invq = 1.0 / q_[i];
        double j0[W], j1[W], jl[W], jm[W];
#pragma omp simd
        for (int k = 0; k < W; ++k) {
            const double inv = invq * invd[k];
            j0[k] = s[k] * inv;
            j1[k] = (j0[k] - c[k]) * inv;
        }
        // accumulate the needed L (parity of L0) with the upward recursion
        for (int k = 0; k < W; ++k) { jm[k] = j0[k]; jl[k] = j1[k]; }
        if (L0 == 0)
            for (int k = 0; k < W; ++k) acc[k * 8 + 0] += gi * j0[k];
        if (L0 <= 1 && Lmax >= 1 && (1 - L0) % 2 == 0)
            for (int k = 0; k < W; ++k) acc[k * 8 + 1] += gi * j1[k];
        for (int l = 1; l < Lmax; ++l) {
#pragma omp simd
            for (int k = 0; k < W; ++k) {
                const double nxt = (2.0 * l + 1.0) * (invq * invd[k]) * jl[k] - jm[k];
                jm[k] = jl[k];
                jl[k] = nxt;
            }
            const int L = l + 1;
            if (L >= L0 && (L - L0) % 2 == 0)
                for (int k = 0; k < W; ++k) acc[k * 8 + L] += gi * jl[k];
        }
#pragma omp simd
        for (int k = 0; k < W; ++k) {
            const double s2 = s[k] * cr[k] + c[k] * sr[k];
            c[k] = c[k] * cr[k] - s[k] * sr[k];
            s[k] = s2;
        }
    }
    for (int k = 0; k < n; ++k)
        for (int L = L0; L <= Lmax; L += 2) J[k * nl + L] += acc[k * 8 + L];
}

void TwoCenterOverlap::evaluate(const Vec3& d, double* out) const {
    const int n1 = 2 * l1_ + 1, n2 = 2 * l2_ + 1;
    const double dist = norm(d);
    for (int i = 0; i < n1 * n2; ++i) out[i] = 0.0;
    if (dist < 1e-10) {
        if (l1_ == l2_)
            for (int m = 0; m < n1; ++m) out[m * n2 + m] = same_center_;
        return;
    }
    const int Lmax = l1_ + l2_;
    double Y[ylm::count(ylm::lmax_supported)];
    ylm::real_ylm(Lmax, d, Y);
    double J[ylm::lmax_supported + 1] = {};
    if (!jtab_.empty() && dist >= tab_dmin_) {
        for (int L = std::abs(l1_ - l2_); L <= Lmax; L += 2) J[L] = jtab_[L](dist);
    } else {
        radial_integrals(dist, J);
    }
    // same products and summation order as the full (L, M, m1, m2) loop; zero Gaunt coefficients precomputed away
    for (const auto& g : couplings_) {
        const double yl = Y[g.y];
        if (yl == 0.0) continue;
        out[g.out] += g.c * yl * J[g.L];
    }
}

} // namespace prowl
