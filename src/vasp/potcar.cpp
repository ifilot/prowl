// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/potcar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#include "prowl/errors.hpp"

namespace prowl {
namespace {

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

/// Parse a Fortran real, accepting D exponents (1.0D-002) and the exponent-without-E form (e.g. 1.0-100).
bool parse_real(const std::string& tok_in, double& out) {
    std::string tok = tok_in;
    for (auto& ch : tok)
        if (ch == 'D' || ch == 'd') ch = 'E';
    char* end = nullptr;
    out = std::strtod(tok.c_str(), &end);
    if (end != tok.c_str() && *end == '\0') return true;
    const auto p = tok.find_last_of("+-");
    if (p != std::string::npos && p > 0 && tok[p - 1] != 'E' && tok[p - 1] != 'e') {
        const std::string fixed = tok.substr(0, p) + "E" + tok.substr(p);
        out = std::strtod(fixed.c_str(), &end);
        return end != fixed.c_str() && *end == '\0';
    }
    return false;
}

bool numeric_line(const std::string& s, std::vector<double>* vals = nullptr) {
    std::istringstream is(s);
    std::string w;
    bool any = false;
    while (is >> w) {
        double v;
        if (!parse_real(w, v) || !std::isfinite(v)) return false;
        if (vals) vals->push_back(v);
        any = true;
    }
    return any;
}

struct Cursor {
    const std::vector<std::string>& lines;
    std::size_t i;
    std::vector<double> read_numbers(std::size_t count, const std::string& what) {
        std::vector<double> v;
        while (v.size() < count) {
            require(i < lines.size(), ErrorCode::CorruptFile, "POTCAR ended while reading " + what);
            require(numeric_line(lines[i], &v), ErrorCode::CorruptFile,
                    "unexpected line while reading " + what + ": '" + trim(lines[i]) + "'");
            ++i;
        }
        require(v.size() == count, ErrorCode::CorruptFile, "wrong number of values in " + what);
        return v;
    }
    std::vector<double> read_numeric_block() {
        std::vector<double> v;
        while (i < lines.size() && numeric_line(lines[i], &v)) ++i;
        return v;
    }
};

PawDataset parse_dataset(const std::vector<std::string>& L) {
    PawDataset d;
    {
        std::istringstream is(L.at(0));
        std::string kind;
        is >> kind >> d.symbol;
    }
    std::size_t first_nl = 0;
    for (std::size_t i = 0; i < L.size(); ++i) {
        const std::string t = trim(L[i]);
        if (t.rfind("TITEL", 0) == 0) d.titel = trim(t.substr(t.find('=') + 1));
        else if (t.rfind("VRHFIN", 0) == 0) {
            d.element = trim(t.substr(t.find('=') + 1, t.find(':') - t.find('=') - 1));
            if (t.find(':') != std::string::npos) d.valence_config = trim(t.substr(t.find(':') + 1));
        }
        else if (t.rfind("SHA256", 0) == 0) d.sha256 = trim(t.substr(t.find('=') + 1));
        else if (t.find("ZVAL") != std::string::npos && d.zval == 0.0) {
            const auto p = t.find("ZVAL");
            std::istringstream is(t.substr(t.find('=', p) + 1));
            is >> d.zval;
        } else if (t == "Non local Part" && first_nl == 0) first_nl = i;
    }
    require(first_nl > 0, ErrorCode::UnsupportedFeature,
            "POTCAR dataset " + d.symbol + " has no nonlocal part (not a PAW/US dataset)");
    require(!d.element.empty(), ErrorCode::CorruptFile, "POTCAR dataset lacks VRHFIN element line");

    // PSMAXN: "<value> T|F" line preceding the first nonlocal block.
    for (std::size_t j = first_nl; j-- > 0;) {
        std::istringstream is(L[j]);
        std::string a, b;
        if ((is >> a >> b) && (b == "T" || b == "F") && parse_real(a, d.psmaxn)) break;
    }
    require(std::isfinite(d.psmaxn) && d.psmaxn > 0, ErrorCode::CorruptFile, "cannot locate PSMAXN in POTCAR " + d.symbol);

    Cursor c{L, first_nl};
    while (c.i < L.size() && trim(L[c.i]) == "Non local Part") {
        ++c.i;
        auto hdr = c.read_numbers(3, "nonlocal header");
        require(hdr[0] >= 0 && hdr[0] <= 3 && std::floor(hdr[0]) == hdr[0] &&
                    hdr[1] >= 1 && hdr[1] <= 4 && std::floor(hdr[1]) == hdr[1],
                ErrorCode::CorruptFile, "bad nonlocal header");
        const int l = static_cast<int>(hdr[0]);
        const int npro = static_cast<int>(hdr[1]);
        d.rmax_proj = hdr[2];
        require(l >= 0 && l <= 3 && npro >= 1 && npro <= 4, ErrorCode::CorruptFile, "bad nonlocal header");
        (void)c.read_numbers(static_cast<std::size_t>(npro * npro), "DION");
        for (int p = 0; p < npro; ++p) {
            require(c.i < L.size() && trim(L[c.i]) == "Reciprocal Space Part", ErrorCode::CorruptFile, "expected Reciprocal Space Part");
            ++c.i;
            PawChannel ch;
            ch.l = l;
            ch.projector_q = c.read_numbers(PawDataset::nq(), "reciprocal projector");
            require(c.i < L.size() && trim(L[c.i]) == "Real Space Part", ErrorCode::CorruptFile, "expected Real Space Part");
            ++c.i;
            (void)c.read_numbers(100, "real-space projector");
            d.channels.push_back(std::move(ch));
        }
        while (c.i < L.size() && trim(L[c.i]).empty()) ++c.i;
    }

    std::size_t paw = 0;
    for (std::size_t i = c.i; i < L.size(); ++i)
        if (trim(L[i]) == "PAW radial sets") { paw = i; break; }
    require(paw > 0, ErrorCode::UnsupportedFeature, "POTCAR dataset " + d.symbol + " is not PAW (no radial sets)");
    std::vector<double> nm;
    require(paw + 1 < L.size() && numeric_line(L[paw + 1], &nm) && !nm.empty() &&
                nm[0] >= 4 && nm[0] < static_cast<double>(std::numeric_limits<std::size_t>::max()) && std::floor(nm[0]) == nm[0],
            ErrorCode::CorruptFile, "bad PAW radial header (need at least four grid points)");
    const auto nmax = static_cast<std::size_t>(nm[0]);
    c.i = paw + 2;
    std::vector<std::vector<double>> ps, ae;
    while (c.i < L.size()) {
        const std::string t = trim(L[c.i]);
        ++c.i;
        if (t.rfind("End of Dataset", 0) == 0) break;
        if (t.empty() || t[0] == '(') continue; // format line
        auto v = c.read_numeric_block();
        if (t == "grid") d.grid = v;
        else if (t == "pseudo wavefunction") ps.push_back(v);
        else if (t == "ae wavefunction") ae.push_back(v);
        else if (t.rfind("augmentation charges", 0) == 0) d.qpaw = v;
    }
    require(d.grid.size() == nmax, ErrorCode::CorruptFile, "PAW grid size mismatch");
    require(ps.size() == d.channels.size() && ae.size() == d.channels.size(), ErrorCode::CorruptFile,
            "number of partial waves does not match number of projectors in " + d.symbol);
    for (std::size_t k = 0; k < d.channels.size(); ++k) {
        require(ps[k].size() == nmax && ae[k].size() == nmax, ErrorCode::CorruptFile, "partial-wave length mismatch");
        d.channels[k].u_ps = std::move(ps[k]);
        d.channels[k].u_ae = std::move(ae[k]);
    }
    const double h0 = std::log(d.grid[1] / d.grid[0]);
    require(d.grid[0] > 0 && std::isfinite(h0) && h0 > 0, ErrorCode::CorruptFile, "invalid PAW radial grid");
    for (std::size_t i = 1; i < nmax; ++i) {
        require(d.grid[i] > d.grid[i - 1] && std::isfinite(d.grid[i]), ErrorCode::CorruptFile,
                "PAW radial grid must be finite, positive and increasing");
        const double h = std::log(d.grid[i] / d.grid[i - 1]);
        require(std::abs(h - h0) < 1e-8 * std::max(1.0, std::abs(h0)) + 1e-10, ErrorCode::UnsupportedFeature,
                "PAW radial grid is not logarithmic");
    }
    const double h1 = std::log(d.grid[nmax - 1] / d.grid[nmax - 2]);
    require(std::abs(h0 - h1) < 1e-8 * std::max(1.0, std::abs(h0)) + 1e-10, ErrorCode::UnsupportedFeature,
            "PAW radial grid is not logarithmic");
    return d;
}

} // namespace

int PawDataset::nprojectors() const noexcept {
    int n = 0;
    for (const auto& c : channels) n += 2 * c.l + 1;
    return n;
}

std::vector<PawDataset> read_potcar(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::FileNotFound, "cannot open POTCAR " + path.string());
    std::vector<PawDataset> out;
    std::vector<std::string> cur;
    for (std::string l; std::getline(in, l);) {
        cur.push_back(l);
        if (trim(l).rfind("End of Dataset", 0) == 0) {
            out.push_back(parse_dataset(cur));
            cur.clear();
        }
    }
    require(!out.empty(), ErrorCode::CorruptFile, "no complete dataset found in POTCAR " + path.string());
    return out;
}

} // namespace prowl
