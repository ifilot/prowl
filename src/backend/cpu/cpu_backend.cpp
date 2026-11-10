// SPDX-License-Identifier: GPL-3.0-or-later
/// CPU FP64 reference backend (OpenBLAS CBLAS).
#include <cblas.h>

#include <algorithm>
#include <vector>

#include "prowl/backend.hpp"
#include "prowl/errors.hpp"
#include "prowl/separable.hpp"

namespace prowl {
namespace {

class CpuBackend final : public Backend {
    std::vector<std::complex<double>> C_;
    std::vector<double> Cr_; // packed [Re C; Im C] for Gamma
    std::vector<double> w_;
    int ng_ = 0, nb_ = 0;
    bool gamma_ = false;
    // Cache of recently generated factor blocks (same k, G list, columns): spins and band
    // blocks of one k-point reuse them. Bounded to a few blocks.
    struct CachedW {
        unsigned long long id;
        unsigned long long ghash;
        int ng, first, n;
        Vec3 k;
        bool gamma;
        std::vector<std::complex<double>> W;
        std::vector<double> Wr;
    };
    std::vector<CachedW> cache_;
    static constexpr std::size_t max_cache_ = 4;
    std::size_t limit_bytes_ = std::size_t{1} << 31, peak_bytes_ = 0;

    static std::size_t bytes(const CachedW& c) {
        return c.W.capacity() * sizeof(std::complex<double>) + c.Wr.capacity() * sizeof(double);
    }
    std::size_t retained_bytes() const {
        std::size_t total = 0;
        for (const auto& c : cache_) total += bytes(c);
        return total;
    }

    const CachedW& factors(const SeparableFactorSpec& s, int first, int n) {
        require(s.ng > 0 && first >= 0 && n > 0 && first <= s.nf() && n <= s.nf() - first,
                ErrorCode::InvalidArgument, "invalid factor dimensions or column range");
        const unsigned long long gh = hash_gvectors(s.G, s.ng);
        for (const auto& c : cache_)
            if (c.id == s.table_id && c.id != 0 && c.ghash == gh && c.ng == s.ng && c.first == first && c.n == n &&
                c.k == s.k && c.gamma == gamma_)
                return c;
        const std::size_t count = static_cast<std::size_t>(s.ng) * static_cast<std::size_t>(n);
        const std::size_t multiplier = sizeof(std::complex<double>) * (gamma_ ? 2u : 1u);
        require(count <= limit_bytes_ / multiplier, ErrorCode::OutOfMemory,
                "factor generation exceeds the aggregate factor memory limit");
        const std::size_t needed = count * multiplier;
        while (!cache_.empty() && (cache_.size() >= max_cache_ || retained_bytes() > limit_bytes_ - needed))
            cache_.erase(cache_.begin());
        CachedW c{s.table_id, gh, s.ng, first, n, s.k, gamma_,
                  std::vector<std::complex<double>>(static_cast<std::size_t>(s.ng) * n), {}};
        evaluate_separable_host(s, first, n, c.W.data());
        peak_bytes_ = std::max(peak_bytes_, retained_bytes() + bytes(c));
        if (gamma_) {
            // Cache the representation consumed by DGEMM so every band block reuses the packing too.
            const std::size_t K = 2 * static_cast<std::size_t>(s.ng);
            c.Wr.resize(K * n);
            peak_bytes_ = std::max(peak_bytes_, retained_bytes() + bytes(c));
            for (int f = 0; f < n; ++f)
                for (int g = 0; g < s.ng; ++g) {
                    const auto w = c.W[static_cast<std::size_t>(f) * s.ng + g];
                    c.Wr[f * K + g] = w_[g] * w.real();
                    c.Wr[f * K + s.ng + g] = -w_[g] * w.imag();
                }
            // Packed and complex factors have equal size; retain only the representation used by this entry.
            std::vector<std::complex<double>>().swap(c.W);
        }
        cache_.push_back(std::move(c));
        return cache_.back();
    }

public:
    std::string name() const override { return "cpu"; }
    std::string device_description() const override { return "CPU FP64 reference (OpenBLAS CBLAS)"; }

    void set_factor_memory_limit(std::size_t limit) override {
        limit_bytes_ = limit;
        while (!cache_.empty() && retained_bytes() > limit_bytes_) cache_.erase(cache_.begin());
        peak_bytes_ = retained_bytes();
        // A reused backend must not carry a previous projection's larger coefficient allocation into a new plan.
        std::vector<std::complex<double>>().swap(C_);
        std::vector<double>().swap(Cr_);
        std::vector<double>().swap(w_);
        ng_ = nb_ = 0;
        // Packed factors include the old weights; the first Gamma load invalidates them when weights are reset.
    }
    FactorMemoryStats factor_memory_stats() const override { return {limit_bytes_, retained_bytes(), peak_bytes_}; }

    void project_full(const std::complex<double>* W, const std::complex<double>* C, int ng, int nf, int nb,
                      std::complex<double>* out) override {
        const std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nf, nb, ng, &one, W, ng, C, ng, &zero, out, nf);
    }

    void project_gamma(const std::complex<double>* W, const std::complex<double>* C, const double* weights, int ng,
                       int nf, int nb, double* out) override {
        // Re(sum w W C) = sum_g [w ReW] ReC + [-w ImW] ImC  -> one real GEMM with K = 2 ng
        const std::size_t K = 2 * static_cast<std::size_t>(ng);
        std::vector<double> Wr(K * nf), Cr(K * nb);
        for (int f = 0; f < nf; ++f)
            for (int g = 0; g < ng; ++g) {
                const auto w = W[static_cast<std::size_t>(f) * ng + g];
                Wr[f * K + g] = weights[g] * w.real();
                Wr[f * K + ng + g] = -weights[g] * w.imag();
            }
        for (int b = 0; b < nb; ++b)
            for (int g = 0; g < ng; ++g) {
                const auto c = C[static_cast<std::size_t>(b) * ng + g];
                Cr[b * K + g] = c.real();
                Cr[b * K + ng + g] = c.imag();
            }
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nf, nb, static_cast<int>(K), 1.0, Wr.data(),
                    static_cast<int>(K), Cr.data(), static_cast<int>(K), 0.0, out, nf);
    }

    void load_coefficients(const std::complex<double>* C, int ng, int nb, const double* gamma_weights) override {
        ng_ = ng;
        nb_ = nb;
        gamma_ = gamma_weights != nullptr;
        if (gamma_) {
            // Weights belong to the resident coefficients; a changed set invalidates the packed factors.
            if (w_.size() != static_cast<std::size_t>(ng) || !std::equal(w_.begin(), w_.end(), gamma_weights))
                cache_.clear();
            const std::size_t K = 2 * static_cast<std::size_t>(ng);
            Cr_.resize(K * nb);
            for (int b = 0; b < nb; ++b)
                for (int g = 0; g < ng; ++g) {
                    const auto c = C[static_cast<std::size_t>(b) * ng + g];
                    Cr_[b * K + g] = c.real();
                    Cr_[b * K + ng + g] = c.imag();
                }
            w_.assign(gamma_weights, gamma_weights + ng);
            std::vector<std::complex<double>>().swap(C_);
        } else {
            C_.assign(C, C + static_cast<std::size_t>(ng) * nb);
            std::vector<double>().swap(Cr_);
        }
    }

    void evaluate_factors(const SeparableFactorSpec& spec, int first, int n, std::complex<double>* W) override {
        evaluate_separable_host(spec, first, n, W);
    }

    void project_separable(const SeparableFactorSpec& spec, int first, int n, std::complex<double>* out_full,
                           double* out_gamma) override {
        require(spec.ng == ng_, ErrorCode::InvalidArgument, "factor spec and resident coefficients differ in N_G");
        const auto& cached = factors(spec, first, n);
        if (!gamma_) {
            const std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, nb_, ng_, &one, cached.W.data(), ng_, C_.data(), ng_,
                        &zero, out_full, n);
            return;
        }
        const std::size_t K = 2 * static_cast<std::size_t>(ng_);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, nb_, static_cast<int>(K), 1.0, cached.Wr.data(),
                    static_cast<int>(K), Cr_.data(), static_cast<int>(K), 0.0, out_gamma, n);
    }
};

} // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

void evaluate_separable_host(const SeparableFactorSpec& s, int first, int n, std::complex<double>* W) {
    require(first >= 0 && n >= 0 && first <= s.nf() && n <= s.nf() - first && s.ng >= 0 &&
                (s.ng == 0 || s.G), ErrorCode::InvalidArgument, "factor column range or G list out of bounds");
    const double two_pi = 6.283185307179586476925286766559;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < s.ng; ++g) {
        const double f0 = s.k[0] + s.G[g][0], f1 = s.k[1] + s.G[g][1], f2 = s.k[2] + s.G[g][2];
        double q[3];
        for (int c = 0; c < 3; ++c) q[c] = (f0 * s.b[0][c] + f1 * s.b[1][c] + f2 * s.b[2][c]) * two_pi;
        const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        double Y[16];
        detail::real_ylm3(q[0], q[1], q[2], Y);
        // Columns come grouped by atom and radial shell (2l+1 m's): the phase depends only on tau and the radial
        // value only on the table, so both are reused while they repeat (identical values, far fewer sin/cos).
        int last_t = -1, last_f = -1;
        double R = 0.0, ltau[3] = {1e300, 1e300, 1e300};
        std::complex<double> phase;
        const bool extra = s.has_extra();
        auto tv = [&](int t) {
            return detail::table_value(s.t_y.data(), s.t_m.data(), s.t_n.data(), s.t_offset.data(), s.t_x0.data(),
                                       s.t_h.data(), s.t_xmax.data(), t, qn);
        };
        // same radial factor as the previous column: same table and (PAW fold) the same extra terms
        auto same_extra = [&](int f, int p) {
            const int nf_ = s.x_off[f + 1] - s.x_off[f];
            if (nf_ != s.x_off[p + 1] - s.x_off[p]) return false;
            for (int j = 0; j < nf_; ++j)
                if (s.x_table[s.x_off[f] + j] != s.x_table[s.x_off[p] + j] || s.x_w[s.x_off[f] + j] != s.x_w[s.x_off[p] + j])
                    return false;
            return true;
        };
        for (int j = 0; j < n; ++j) {
            const int f = first + j;
            const int t = s.table[f];
            if (t != last_t || (extra && !same_extra(f, last_f))) {
                R = tv(t);
                if (extra)
                    for (int x = s.x_off[f]; x < s.x_off[f + 1]; ++x) R += s.x_w[x] * tv(s.x_table[x]);
                last_t = t;
                last_f = f;
            }
            const double* tau = &s.tau[3 * f];
            if (tau[0] != ltau[0] || tau[1] != ltau[1] || tau[2] != ltau[2]) {
                const double arg = two_pi * (f0 * tau[0] + f1 * tau[1] + f2 * tau[2]);
                phase = std::complex<double>(std::cos(arg), std::sin(arg));
                ltau[0] = tau[0]; ltau[1] = tau[1]; ltau[2] = tau[2];
            }
            W[static_cast<std::size_t>(j) * s.ng + g] = s.pref * s.coef[f] * (Y[s.lm[f]] * R) * phase;
        }
    }
}

} // namespace prowl
