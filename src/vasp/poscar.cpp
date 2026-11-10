// SPDX-License-Identifier: GPL-3.0-or-later
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#include "prowl/structure.hpp"

namespace prowl {
namespace {
std::vector<std::string> tokens(const std::string& s) {
    std::istringstream is(s);
    std::vector<std::string> t;
    for (std::string w; is >> w;) t.push_back(w);
    return t;
}
bool is_number(const std::string& s) {
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}
double real(const std::string& token, const char* field) {
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    require(end != token.c_str() && *end == '\0' && std::isfinite(value), ErrorCode::CorruptFile,
            std::string("invalid POSCAR ") + field);
    return value;
}
} // namespace

Structure read_poscar(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::FileNotFound, "cannot open structure file " + path.string());
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);
    require(lines.size() >= 8, ErrorCode::CorruptFile, "POSCAR too short: " + path.string());

    Structure s;
    s.comment = lines[0];
    auto sc = tokens(lines[1]);
    require(!sc.empty() && is_number(sc[0]), ErrorCode::CorruptFile, "bad POSCAR scale line");
    require(sc.size() == 1, ErrorCode::UnsupportedFeature, "POSCAR requires one isotropic scale value");
    double scale = real(sc[0], "scale");
    require(scale != 0.0, ErrorCode::CorruptFile, "POSCAR scale cannot be zero");
    Mat3 a{};
    for (int i = 0; i < 3; ++i) {
        auto t = tokens(lines[2 + i]);
        require(t.size() >= 3, ErrorCode::CorruptFile, "bad POSCAR lattice line");
        for (int c = 0; c < 3; ++c) a[i][c] = real(t[c], "lattice component");
    }
    if (scale < 0) { // negative scale = target volume
        const double vol = std::abs(dot(a[0], cross(a[1], a[2])));
        require(std::isfinite(vol) && vol > 1e-12, ErrorCode::CorruptFile, "POSCAR lattice is singular");
        scale = std::cbrt(-scale / vol);
    }
    for (auto& row : a)
        for (auto& x : row) x *= scale;
    s.lattice = Lattice(a);

    auto sp = tokens(lines[5]);
    require(!sp.empty() && !is_number(sp[0]), ErrorCode::UnsupportedFeature,
            "POSCAR without species line (VASP 4 format) is not supported; add the element line");
    for (auto& w : sp) {
        // strip POTCAR suffixes like "Ga_d" or hashes "Si/abcd"
        auto cut = w.find_first_of("_/");
        s.species.push_back(cut == std::string::npos ? w : w.substr(0, cut));
    }
    auto ct = tokens(lines[6]);
    require(ct.size() == s.species.size(), ErrorCode::CorruptFile, "species and count lines differ in length");
    int nat = 0;
    for (std::size_t t = 0; t < ct.size(); ++t) {
        const double count = real(ct[t], "atom count");
        require(count >= 0 && std::floor(count) == count && count <= std::numeric_limits<int>::max() - nat,
                ErrorCode::CorruptFile, "POSCAR atom count must be a nonnegative integer");
        s.counts.push_back(static_cast<int>(count));
        for (int i = 0; i < s.counts.back(); ++i) s.atom_type.push_back(static_cast<int>(t));
        nat += s.counts.back();
    }
    require(nat > 0, ErrorCode::CorruptFile, "POSCAR has no atoms");
    std::size_t li = 7;
    auto mode_at = [&] {
        require(li < lines.size(), ErrorCode::CorruptFile, "POSCAR coordinate mode is missing");
        const auto pos = lines[li].find_first_not_of(" \t\r");
        require(pos != std::string::npos, ErrorCode::CorruptFile, "POSCAR coordinate mode is blank");
        return static_cast<char>(std::toupper(static_cast<unsigned char>(lines[li][pos])));
    };
    if (mode_at() == 'S') ++li; // selective dynamics
    const char mode = mode_at();
    require(mode == 'D' || mode == 'C' || mode == 'K', ErrorCode::CorruptFile, "invalid POSCAR coordinate mode");
    const bool cart = mode == 'C' || mode == 'K';
    ++li;
    require(lines.size() >= li + static_cast<std::size_t>(nat), ErrorCode::CorruptFile, "POSCAR truncated");
    for (int i = 0; i < nat; ++i) {
        auto t = tokens(lines[li + i]);
        require(t.size() >= 3, ErrorCode::CorruptFile, "bad POSCAR coordinate line");
        Vec3 v{real(t[0], "coordinate"), real(t[1], "coordinate"), real(t[2], "coordinate")};
        if (cart) {
            for (auto& x : v) x *= scale;
            Vec3 f{};
            for (int j = 0; j < 3; ++j) f[j] = dot(v, s.lattice.b[j]);
            v = f;
        }
        s.frac.push_back(v);
    }
    return s;
}

void check_same_lattice(const Lattice& a, const Lattice& b, double tol, const std::string& what) {
    require(std::isfinite(tol) && tol >= 0, ErrorCode::InvalidArgument, "lattice tolerance must be finite and nonnegative");
    for (int i = 0; i < 3; ++i)
        for (int c = 0; c < 3; ++c)
            if (std::abs(a.a[i][c] - b.a[i][c]) > tol)
                fail(ErrorCode::InconsistentInput, "lattice mismatch between " + what);
}

} // namespace prowl
