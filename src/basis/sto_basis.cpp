// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include "prowl/basis.hpp"
#include "prowl/constants.hpp"
#include "prowl/errors.hpp"
#include "prowl/ylm.hpp"

namespace prowl {

namespace {
double factorial(int n) noexcept {
    static const auto table = [] {
        std::array<double, 171> t{};
        t[0] = 1.0;
        for (int i = 1; i < 171; ++i) t[i] = std::tgamma(i + 1.0);
        return t;
    }();
    return (n >= 0 && n < 171) ? table[static_cast<std::size_t>(n)] : std::tgamma(n + 1.0);
}
double double_factorial_odd(int n) noexcept { // n!! for odd n >= -1
    double r = 1.0;
    for (int k = n; k > 1; k -= 2) r *= k;
    return r;
}
double sto_norm(int n, double zeta) noexcept { return std::pow(2.0 * zeta, n + 0.5) / std::sqrt(factorial(2 * n)); }
} // namespace

double sto_bessel_integral(int N, int l, double zeta, double q) noexcept {
    if (q <= 0.0) return l == 0 ? factorial(N) / std::pow(zeta, N + 1) : 0.0;
    if (q < 0.3 * zeta) {
        // Taylor series of j_l: sum_s (-1)^s q^{l+2s} (N+l+2s)! / (2^s s! (2l+2s+1)!! zeta^{N+l+2s+1})
        double term = std::pow(q, l) * factorial(N + l) / (double_factorial_odd(2 * l + 1) * std::pow(zeta, N + l + 1));
        double sum = term;
        const double x = q * q / (zeta * zeta);
        for (int s = 0; s < 400; ++s) {
            term *= -x * (N + l + 2.0 * s + 1.0) * (N + l + 2.0 * s + 2.0) / (2.0 * (s + 1.0) * (2.0 * l + 2.0 * s + 3.0));
            sum += term;
            if (std::abs(term) < 1e-18 * std::abs(sum)) break;
        }
        return sum;
    }
    // Exact closed form from the finite expansion of the spherical Hankel function:
    // j_l(x) = Re[(-i)^{l+1} e^{ix}/x sum_k i^k (l+k)!/(k!(l-k)!(2x)^k)].
    using C = std::complex<double>;
    const C zinv = 1.0 / C(zeta, -q);
    // z^-(N-k) for k = 0..l by integer powers (std::pow(complex, int) goes through exp/log)
    C zp[16];
    zp[l] = C(1.0, 0.0);
    for (int i = 0; i < N - l; ++i) zp[l] *= zinv;
    for (int k = l - 1; k >= 0; --k) zp[k] = zp[k + 1] * zinv;
    C sum(0.0, 0.0);
    C ik(1.0, 0.0);
    const double qinv = 1.0 / q;
    double qp = qinv, two_k = 1.0;
    for (int k = 0; k <= l; ++k) {
        const double ak = factorial(l + k) / (factorial(k) * factorial(l - k) * two_k);
        sum += ik * ak * qp * factorial(N - k - 1) * zp[k];
        ik *= C(0.0, 1.0);
        qp *= qinv;
        two_k *= 2.0;
    }
    C pre(1.0, 0.0);
    for (int k = 0; k <= l; ++k) pre *= C(0.0, -1.0);
    return (pre * sum).real();
}

double RadialSto::value(double r) const noexcept {
    double v = 0.0;
    for (const auto& p : primitives) v += p.coef * sto_norm(p.n, p.zeta) * std::pow(r, p.n - 1) * std::exp(-p.zeta * r);
    return v;
}

double RadialSto::fourier(double q) const noexcept {
    double v = 0.0;
    for (const auto& p : primitives) v += p.coef * sto_norm(p.n, p.zeta) * sto_bessel_integral(p.n + 1, l, p.zeta, q);
    return v;
}

double RadialSto::norm2() const noexcept {
    double s = 0.0;
    for (const auto& a : primitives)
        for (const auto& b : primitives)
            s += a.coef * b.coef * sto_norm(a.n, a.zeta) * sto_norm(b.n, b.zeta) * factorial(a.n + b.n) /
                 std::pow(a.zeta + b.zeta, a.n + b.n + 1);
    return s;
}

double RadialSto::extent(double threshold) const noexcept {
    for (double r = 60.0; r > 0.0; r -= 0.01)
        if (std::abs(r * value(r)) > threshold) return r + 0.01;
    return 0.01;
}

StoLibrary StoLibrary::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::FileNotFound, "basis library not found: " + path.string());
    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        fail(ErrorCode::CorruptFile, "basis JSON parse error: " + std::string(e.what()));
    }
    StoLibrary lib;
    lib.info_.schema = j.value("schema", "");
    require(lib.info_.schema == "prowl-sto-basis/1", ErrorCode::UnsupportedFeature, "unknown basis schema " + lib.info_.schema);
    lib.info_.name = j.value("name", "");
    lib.info_.provenance_json = j.at("provenance").dump();
    require(j.at("units").at("length") == "bohr", ErrorCode::UnsupportedFeature, "basis length unit must be bohr");
    const double a0 = constants::bohr_angstrom;
    for (auto& [el, rec] : j.at("elements").items()) {
        std::vector<RadialSto> v;
        for (const auto& sh : rec.at("shells")) {
            RadialSto r;
            r.element = el;
            r.label = sh.at("label").get<std::string>();
            r.n = sh.at("n").get<int>();
            r.l = sh.at("l").get<int>();
            r.occupation = sh.at("occupation").get<double>();
            r.energy_ev = sh.at("energy").get<double>() * constants::hartree_ev;
            for (const auto& p : sh.at("primitives")) {
                StoPrimitive sp{p.at("n").get<int>(), p.at("zeta").get<double>() / a0, p.at("coef").get<double>()};
                require(sp.n >= r.l + 1 && sp.zeta > 0, ErrorCode::CorruptFile, "invalid STO primitive in " + el);
                r.primitives.push_back(sp);
            }
            v.push_back(std::move(r));
        }
        lib.shells_[el] = std::move(v);
    }
    return lib;
}

void StoLibrary::merge(const StoLibrary& ext) {
    for (const auto& [el, shells] : ext.shells_) {
        auto it = shells_.find(el);
        require(it != shells_.end(), ErrorCode::InvalidArgument,
                "basis extension " + ext.info_.name + " has element " + el + " missing from " + info_.name);
        for (const auto& sh : shells) {
            for (const auto& have : it->second)
                require(have.label != sh.label, ErrorCode::InvalidArgument,
                        "basis extension " + ext.info_.name + " duplicates shell " + el + " " + sh.label);
            it->second.push_back(sh);
        }
    }
    nlohmann::json prov = nlohmann::json::parse(info_.provenance_json);
    if (!prov.is_object() || !prov.contains("extensions")) prov = {{"base", prov}, {"extensions", nlohmann::json::array()}};
    prov["extensions"].push_back({{"name", ext.info_.name}, {"provenance", nlohmann::json::parse(ext.info_.provenance_json)}});
    info_.provenance_json = prov.dump();
    info_.name += "+" + ext.info_.name;
}

StoLibrary StoLibrary::load_default() {
    return load(std::filesystem::path(PROWL_DATA_DIR) / "basis" / "koga" / "koga_hf_sto.json");
}

const std::vector<RadialSto>& StoLibrary::shells(const std::string& element) const {
    auto it = shells_.find(element);
    if (it == shells_.end()) fail(ErrorCode::UnsupportedFeature, "no basis data for element " + element);
    return it->second;
}

const RadialSto& StoLibrary::shell(const std::string& element, const std::string& label) const {
    for (const auto& s : shells(element))
        if (s.label == label) return s;
    fail(ErrorCode::InvalidArgument, "basis library has no shell " + label + " for " + element);
}

AtomicBasis::AtomicBasis(const Structure& s, const StoLibrary& lib,
                         const std::map<std::string, std::vector<std::string>>& selection) {
    std::map<std::pair<std::string, std::string>, int> radial_index;
    for (int a = 0; a < s.natoms(); ++a) {
        const auto& el = s.element(a);
        auto it = selection.find(el);
        if (it == selection.end())
            fail(ErrorCode::InvalidArgument, "no basis functions selected for element " + el);
        std::set<std::string> seen;
        for (const auto& lab : it->second) {
            require(seen.insert(lab).second, ErrorCode::InvalidArgument, "duplicate basis function " + lab + " for " + el);
            auto key = std::make_pair(el, lab);
            if (!radial_index.count(key)) {
                radial_index[key] = static_cast<int>(radials_.size());
                radials_.push_back(lib.shell(el, lab));
            }
            const int ri = radial_index[key];
            const int l = radials_[ri].l;
            for (int m = -l; m <= l; ++m) {
                OrbitalSpec o{a, ri, l, m, el + std::to_string(a + 1) + "_" + lab};
                if (l > 0) o.label += ylm::orbital_name(l, m).substr(1);
                orbitals_.push_back(o);
            }
        }
    }
}

int AtomicBasis::lmax() const noexcept {
    int l = 0;
    for (const auto& r : radials_) l = std::max(l, r.l);
    return l;
}

} // namespace prowl
