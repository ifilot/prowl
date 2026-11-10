// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "prowl/errors.hpp"
#include "prowl/wavecar.hpp"
#include "synthetic_wavecar.hpp"

using namespace prowl;
using Catch::Approx;
namespace fs = std::filesystem;

namespace {
const Mat3 A{Vec3{0, 2.715, 2.715}, Vec3{2.715, 0, 2.715}, Vec3{2.715, 2.715, 0}};
fs::path tmp(const std::string& name) { return fs::temp_directory_path() / ("prowl_test_" + name); }

std::vector<std::complex<double>> random_coeffs(std::size_t n, std::mt19937& rng) {
    std::normal_distribution<double> d;
    std::vector<std::complex<double>> v(n);
    double s = 0;
    for (auto& c : v) { c = {d(rng), d(rng)}; s += std::norm(c); }
    for (auto& c : v) c /= std::sqrt(s);
    return v;
}
} // namespace

TEST_CASE("standard double-precision WAVECAR, 2 spins, 2 k-points", "[wavecar]") {
    const Lattice L(A);
    const double encut = 150.0;
    std::mt19937 rng(7);
    std::vector<std::vector<testing::SyntheticK>> spins(2);
    const std::vector<Vec3> ks{{0, 0, 0}, {0.25, 0.25, 0.0}};
    std::size_t maxpw = 0;
    for (int s = 0; s < 2; ++s)
        for (const auto& k : ks) {
            const auto n = generate_gvectors_full(L, k, encut).size();
            maxpw = std::max(maxpw, n);
            testing::SyntheticK K{k, {}, {-5.0 + s, 1.0}, {1.0, 0.0}};
            for (int b = 0; b < 2; ++b) K.bands.push_back(random_coeffs(n, rng));
            spins[s].push_back(K);
        }
    const auto p = tmp("std.wavecar");
    testing::write_wavecar(p, 45210, 2, encut, A, spins, std::max<std::uint64_t>(maxpw * 16, 14 * 8));
    WavecarReader r(p);
    REQUIRE(r.header().nspin == 2);
    REQUIRE(r.header().nkpts == 2);
    REQUIRE(r.header().precision == CoefficientPrecision::Double);
    REQUIRE(r.encoding() == WavefunctionEncoding::FullComplex);
    auto blk = r.read_block(1, 1, 1, 1);
    REQUIRE(blk.ng() == spins[1][1].bands[1].size());
    for (std::size_t g = 0; g < blk.ng(); ++g) REQUIRE(blk.coefficients[g] == spins[1][1].bands[1][g]);
    REQUIRE(blk.plane_wave_norms()[0] == Approx(1.0).epsilon(1e-14));
    REQUIRE(r.kpoint(1, 0).eigenvalues_ev[0] == Approx(-4.0));
    fs::remove(p);
}

TEST_CASE("gamma-half single precision with spanning headers: sqrt(2) convention", "[wavecar][gamma]") {
    const Lattice L(A);
    const double encut = 200.0;
    auto half = generate_gvectors_gamma_half(L, encut);
    std::mt19937 rng(3);
    // logical c(G): real at G=0, arbitrary elsewhere, normalised over the full sphere
    const int nb = 80; // many bands -> k header longer than one record
    testing::SyntheticK K{{0, 0, 0}, {}, std::vector<double>(nb, 0.0), std::vector<double>(nb, 1.0)};
    std::vector<std::vector<std::complex<double>>> logical;
    for (int b = 0; b < nb; ++b) {
        auto c = random_coeffs(half.size(), rng);
        c[0] = std::abs(c[0]);
        double s = std::norm(c[0]);
        for (std::size_t g = 1; g < c.size(); ++g) s += 2 * std::norm(c[g]);
        for (auto& x : c) x /= std::sqrt(s);
        logical.push_back(c);
        auto stored = c;
        for (std::size_t g = 1; g < c.size(); ++g) stored[g] *= std::sqrt(2.0);
        K.bands.push_back(stored);
    }
    const auto p = tmp("gam.wavecar");
    const std::uint64_t recl = std::max<std::uint64_t>(((half.size() + 1) / 2) * 16, 7 * 16);
    REQUIRE(4 + 3 * nb > static_cast<int>(recl / 8)); // header really spans records
    testing::write_wavecar(p, 53300, 1, encut, A, {{K}}, recl);
    WavecarReader r(p);
    REQUIRE(r.header().header_spans_records);
    REQUIRE(r.header().records_per_kheader > 1);
    REQUIRE(r.encoding() == WavefunctionEncoding::GammaHalf);
    REQUIRE(r.layout().gamma_half_assumed);
    auto blk = r.read_block(0, 0, 0, nb);
    REQUIRE(blk.gamma_nonzero_scale == Approx(1 / std::sqrt(2.0)));
    for (int b = 0; b < nb; ++b) {
        REQUIRE(blk.plane_wave_norms()[b] == Approx(1.0).epsilon(1e-6));
        for (std::size_t g = 0; g < half.size(); ++g)
            REQUIRE(std::abs(blk.band(b)[g] - logical[b][g]) < 1e-6);
    }
    fs::remove(p);
}

TEST_CASE("spinor detection and typed rejection", "[wavecar]") {
    const Lattice L(A);
    const double encut = 120.0;
    const auto n = generate_gvectors_full(L, {0, 0, 0}, encut).size();
    std::mt19937 rng(1);
    testing::SyntheticK K{{0, 0, 0}, {random_coeffs(2 * n, rng)}, {0.0}, {1.0}};
    const auto p = tmp("ncl.wavecar");
    testing::write_wavecar(p, 45200, 1, encut, A, {{K}}, std::max<std::uint64_t>(2 * n * 8, 7 * 16));
    WavecarReader r(p);
    REQUIRE(r.encoding() == WavefunctionEncoding::Spinor);
    try {
        (void)r.read_block(0, 0, 0, 1);
        FAIL("spinor read must throw");
    } catch (const Error& e) {
        REQUIRE(e.code() == ErrorCode::UnsupportedFeature);
    }
    fs::remove(p);
}

TEST_CASE("truncated, mislabelled and inconsistent files fail with typed errors", "[wavecar]") {
    const Lattice L(A);
    const double encut = 120.0;
    const auto n = generate_gvectors_full(L, {0, 0, 0}, encut).size();
    std::mt19937 rng(2);
    testing::SyntheticK K{{0, 0, 0}, {random_coeffs(n, rng), random_coeffs(n, rng)}, {0.0, 1.0}, {1.0, 0.0}};
    const auto p = tmp("bad.wavecar");
    const std::uint64_t recl = std::max<std::uint64_t>(n * 16, 7 * 16);
    auto code_of = [&](auto&& fn) -> ErrorCode {
        try { fn(); } catch (const Error& e) { return e.code(); }
        return ErrorCode::Internal;
    };

    testing::write_wavecar(p, 45200, 1, encut, A, {{K}}, recl);
    fs::resize_file(p, fs::file_size(p) - recl);
    REQUIRE(code_of([&] { WavecarReader r(p); }) == ErrorCode::CorruptFile);

    testing::write_wavecar(p, 12345, 1, encut, A, {{K}}, recl);
    REQUIRE(code_of([&] { WavecarReader r(p); }) == ErrorCode::UnsupportedFeature);

    // wrong ENCUT in header -> plane-wave count cannot be reconstructed
    testing::write_wavecar(p, 45200, 1, encut * 1.5, A, {{K}}, recl);
    REQUIRE(code_of([&] { WavecarReader r(p); }) == ErrorCode::CorruptFile);

    // override that contradicts the counts
    testing::write_wavecar(p, 45200, 1, encut, A, {{K}}, recl);
    ReaderOptions o;
    o.encoding_override = WavefunctionEncoding::GammaHalf;
    REQUIRE(code_of([&] { WavecarReader r(p, o); }) == ErrorCode::InconsistentInput);
    REQUIRE(code_of([&] { WavecarReader r(tmp("does_not_exist")); }) == ErrorCode::FileNotFound);
    fs::remove(p);
}
