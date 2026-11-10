// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/radial.hpp"

#include <algorithm>
#include <cmath>

#include "prowl/errors.hpp"

namespace prowl::radial {

namespace {
double series(int l, double x) noexcept {
    // j_l(x) = x^l/(2l+1)!! * sum_s (-x^2/2)^s / (s! (2l+3)(2l+5)...(2l+2s+1))
    double df = 1.0;
    for (int k = 3; k <= 2 * l + 1; k += 2) df *= k;
    const double pre = std::pow(x, l) / df;
    const double x2 = 0.5 * x * x;
    double term = 1.0, sum = 1.0;
    for (int s = 1; s < 40; ++s) {
        term *= -x2 / (s * (2.0 * l + 2.0 * s + 1.0));
        sum += term;
        if (std::abs(term) < 1e-17 * std::abs(sum)) break;
    }
    return pre * sum;
}
} // namespace

void sph_bessel_all(int lmax, double x, double* out) noexcept {
    if (x < 0.5) {
        for (int l = 0; l <= lmax; ++l) out[l] = series(l, x);
        return;
    }
    const double s = std::sin(x), c = std::cos(x);
    out[0] = s / x;
    if (lmax == 0) return;
    if (x > lmax) {
        out[1] = s / (x * x) - c / x;
        for (int l = 1; l < lmax; ++l) out[l + 1] = (2.0 * l + 1.0) / x * out[l] - out[l - 1];
        return;
    }
    // Miller downward recursion, normalised to j_0.
    const int start = lmax + 20 + static_cast<int>(x);
    double jp1 = 0.0, j = 1e-300;
    double tmp[64];
    for (int l = start; l >= 1; --l) {
        const double jm1 = (2.0 * l + 1.0) / x * j - jp1;
        jp1 = j;
        j = jm1;
        if (l - 1 <= lmax) tmp[l - 1] = j;
        if (std::abs(j) > 1e250) { // rescale
            j *= 1e-250;
            jp1 *= 1e-250;
            for (int k = l - 1; k <= lmax && k < 64; ++k) tmp[k] *= 1e-250;
        }
    }
    const double scale = out[0] / tmp[0];
    for (int l = 1; l <= lmax; ++l) out[l] = tmp[l] * scale;
}

double sph_bessel(int l, double x) noexcept {
    double buf[16];
    sph_bessel_all(l, x, buf);
    return buf[l];
}

CubicSpline::CubicSpline(std::vector<double> x, std::vector<double> y) : x_(std::move(x)), y_(std::move(y)) {
    const std::size_t n = x_.size();
    require(n >= 3 && y_.size() == n, ErrorCode::InvalidArgument, "spline needs >= 3 points of equal length");
    require(std::all_of(x_.begin(), x_.end(), [](double v) { return std::isfinite(v); }) &&
                std::all_of(y_.begin(), y_.end(), [](double v) { return std::isfinite(v); }),
            ErrorCode::InvalidArgument, "spline samples must be finite");
    for (std::size_t i = 1; i < n; ++i)
        require(x_[i] > x_[i - 1], ErrorCode::InvalidArgument, "spline abscissae not increasing");
    m_.assign(n, 0.0);
    std::vector<double> u(n, 0.0);
    for (std::size_t i = 1; i + 1 < n; ++i) { // natural spline, tridiagonal solve
        const double sig = (x_[i] - x_[i - 1]) / (x_[i + 1] - x_[i - 1]);
        const double p = sig * m_[i - 1] + 2.0;
        m_[i] = (sig - 1.0) / p;
        const double d = (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]) - (y_[i] - y_[i - 1]) / (x_[i] - x_[i - 1]);
        u[i] = (6.0 * d / (x_[i + 1] - x_[i - 1]) - sig * u[i - 1]) / p;
    }
    m_[n - 1] = 0.0;
    for (std::size_t k = n - 1; k-- > 0;) m_[k] = m_[k] * m_[k + 1] + u[k];
    h_ = (x_.back() - x_.front()) / static_cast<double>(n - 1);
    uniform_ = true;
    for (std::size_t i = 1; i < n; ++i)
        if (std::abs((x_[i] - x_[i - 1]) - h_) > 1e-10 * h_) { uniform_ = false; break; }
}

double CubicSpline::operator()(double x) const noexcept {
    const std::size_t n = x_.size();
    std::size_t lo;
    if (uniform_) {
        const double t = (x - x_[0]) / h_;
        lo = t <= 0 ? 0 : std::min(static_cast<std::size_t>(t), n - 2);
    } else {
        auto it = std::upper_bound(x_.begin(), x_.end(), x);
        lo = it == x_.begin() ? 0 : std::min(static_cast<std::size_t>(it - x_.begin()) - 1, n - 2);
    }
    const std::size_t hi = lo + 1;
    const double h = x_[hi] - x_[lo];
    const double a = (x_[hi] - x) / h, b = (x - x_[lo]) / h;
    return a * y_[lo] + b * y_[hi] + ((a * a * a - a) * m_[lo] + (b * b * b - b) * m_[hi]) * (h * h) / 6.0;
}

std::vector<double> simpson_weights(int n, double h) {
    require(n >= 2, ErrorCode::InvalidArgument, "need >= 2 points");
    std::vector<double> w(n, 0.0);
    if (n == 2) { w[0] = w[1] = 0.5 * h; return w; }
    if (n == 3) { w[0] = w[2] = h / 3; w[1] = 4 * h / 3; return w; }
    const int ns = (n % 2 == 1) ? n : n - 3; // points covered by Simpson
    for (int i = 0; i < ns; ++i) w[i] += h / 3.0 * ((i == 0 || i == ns - 1) ? 1.0 : (i % 2 ? 4.0 : 2.0));
    if (ns != n) { // 3/8 rule on the last 4 points
        const int s = n - 4;
        w[s] += 3.0 * h / 8.0; w[s + 1] += 9.0 * h / 8.0; w[s + 2] += 9.0 * h / 8.0; w[s + 3] += 3.0 * h / 8.0;
    }
    return w;
}

double integrate_log_grid(const std::vector<double>& r, const std::vector<double>& f) {
    const std::size_t n = r.size();
    require(f.size() == n && n >= 4, ErrorCode::InvalidArgument, "integrate_log_grid size mismatch");
    const double h = std::log(r[1] / r[0]);
    const auto w = simpson_weights(static_cast<int>(n), h);
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += w[i] * f[i] * r[i];
    return s;
}

} // namespace prowl::radial
