// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <random>
#include <vector>

#include "prowl/backend.hpp"
#include "prowl/separable.hpp"
#include "prowl/errors.hpp"

using namespace prowl;
using cd = std::complex<double>;

namespace {
void check_backend(Backend& be) {
    std::mt19937 rng(11);
    std::normal_distribution<double> d;
    const int ng = 137, nf = 9, nb = 13;
    std::vector<cd> W(ng * nf), C(ng * nb), out(nf * nb);
    for (auto& x : W) x = {d(rng), d(rng)};
    for (auto& x : C) x = {d(rng), d(rng)};
    be.project_full(W.data(), C.data(), ng, nf, nb, out.data());
    for (int f = 0; f < nf; ++f)
        for (int b = 0; b < nb; ++b) {
            cd s = 0;
            for (int g = 0; g < ng; ++g) s += W[f * ng + g] * C[b * ng + g];
            REQUIRE(std::abs(s - out[b * nf + f]) < 1e-12 * (1 + std::abs(s)));
        }
    std::vector<double> w(ng, 2.0), outg(nf * nb);
    w[0] = 1.0;
    be.project_gamma(W.data(), C.data(), w.data(), ng, nf, nb, outg.data());
    for (int f = 0; f < nf; ++f)
        for (int b = 0; b < nb; ++b) {
            double s = 0;
            for (int g = 0; g < ng; ++g) s += w[g] * (W[f * ng + g] * C[b * ng + g]).real();
            REQUIRE(std::abs(s - outg[b * nf + f]) < 1e-12 * (1 + std::abs(s)));
        }
}
} // namespace

TEST_CASE("CPU backend GEMM matches naive sums", "[backend]") {
    auto be = make_backend("cpu");
    check_backend(*be);
}

TEST_CASE("separable factor cache follows encoding, weights and G content", "[backend][cache]") {
    constexpr int ng = 7, nf = 3, nb = 2;
    std::vector<IVec3> G{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
    SeparableFactorSpec spec;
    spec.G = G.data();
    spec.ng = ng;
    spec.b = Mat3{Vec3{0.1, 0, 0}, Vec3{0, 0.1, 0}, Vec3{0, 0, 0.1}};
    spec.tau = {0, 0, 0, 0.13, 0.21, 0.32, 0.4, 0.3, 0.2};
    spec.lm = {0, 1, 2};
    spec.table = {0, 0, 0};
    spec.coef = {cd{1, 0}, cd{0, 1}, cd{0, 1}};
    spec.t_x0 = {0}; spec.t_h = {2}; spec.t_xmax = {4};
    spec.t_n = {3}; spec.t_offset = {0}; spec.t_y = {1, 0.5, 0}; spec.t_m = {0, 0, 0};
    spec.table_id = 1;
    std::vector<cd> C(ng * nb), W(ng * nf), ref(nf * nb), got(nf * nb);
    for (std::size_t i = 0; i < C.size(); ++i) C[i] = cd(0.1 * static_cast<double>(i), -0.2);
    std::vector<double> weights(ng, 2.0), refr(nf * nb), gotr(nf * nb);
    weights[0] = 1.0;
    auto be = make_cpu_backend();
    auto check = [&](bool gamma) {
        be->evaluate_factors(spec, 0, nf, W.data());
        be->load_coefficients(C.data(), ng, nb, gamma ? weights.data() : nullptr);
        if (gamma) {
            be->project_gamma(W.data(), C.data(), weights.data(), ng, nf, nb, refr.data());
            for (int repeat = 0; repeat < 2; ++repeat) {
                be->project_separable(spec, 0, nf, nullptr, gotr.data());
                for (std::size_t i = 0; i < gotr.size(); ++i) REQUIRE(std::abs(gotr[i] - refr[i]) < 1e-12);
            }
            // A different column range must not reuse the full-range packing.
            be->project_separable(spec, 1, 1, nullptr, gotr.data());
            for (int b = 0; b < nb; ++b) REQUIRE(std::abs(gotr[b] - refr[b * nf + 1]) < 1e-12);
        } else {
            be->project_full(W.data(), C.data(), ng, nf, nb, ref.data());
            be->project_separable(spec, 0, nf, got.data(), nullptr);
            for (std::size_t i = 0; i < got.size(); ++i) REQUIRE(std::abs(got[i] - ref[i]) < 1e-12);
        }
    };
    check(false);
    check(true);
    C[3] *= cd{0.7, 0.2}; // new coefficients reuse the same factors
    check(true);
    weights[2] = 0.5; // the low-level backend accepts caller-provided weights
    check(true);
    check(false);
    check(true);
    G[2] = {2, 0, 0}; // same pointer and length, different content
    check(true);
    spec.k = {0.1, 0.2, 0.3};
    check(true);
    spec.table_id = 0; // uncached specs may change between calls
    check(true);
    spec.t_y[1] = 0.7;
    check(true);
    // A small limit has room for one cached packed column plus the next column's cold W/Wr allocation.
    const std::size_t column = ng * sizeof(cd);
    const std::size_t limit = 3 * column;
    be->evaluate_factors(spec, 0, nf, W.data());
    be->project_gamma(W.data(), C.data(), weights.data(), ng, nf, nb, refr.data());
    for (const std::string& backend : available_backends()) {
        auto limited = make_backend(backend);
        limited->set_factor_memory_limit(limit);
        limited->load_coefficients(C.data(), ng, nb, weights.data());
        for (int repeat = 0; repeat < 3; ++repeat)
            for (int f = 0; f < nf; ++f) {
                limited->project_separable(spec, f, 1, nullptr, gotr.data());
                for (int b = 0; b < nb; ++b) REQUIRE(std::abs(gotr[b] - refr[b * nf + f]) < 1e-12);
                const auto stats = limited->factor_memory_stats();
                REQUIRE(stats.retained_bytes <= limit);
                REQUIRE(stats.peak_bytes <= limit);
                REQUIRE(stats.peak_bytes > 0);
            }
        REQUIRE_THROWS_AS(limited->project_separable(spec, 0, nf, nullptr, gotr.data()), Error);
        limited->set_factor_memory_limit(0);
        REQUIRE(limited->factor_memory_stats().retained_bytes == 0);
    }
}

#ifdef PROWL_HAVE_CUDA
TEST_CASE("CUDA backend matches naive sums", "[backend][cuda]") {
    if (!cuda_device_available()) SKIP("no CUDA device");
    auto be = make_backend("cuda");
    check_backend(*be);
}
#endif
