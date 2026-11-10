// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/constants.hpp"
#include "prowl/errors.hpp"
#include "prowl/projection.hpp"
#include "prowl/twocenter.hpp"
#include "prowl/ylm.hpp"
#include "detail/omp_exception.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace prowl {
using cd = std::complex<double>;
using OmpExceptionGuard = detail::ExceptionGuard;
namespace {
using cd = std::complex<double>;

cd ipow(int l) {
    switch (l & 3) {
    case 0:
        return {1.0, 0.0};
    case 1:
        return {0.0, 1.0};
    case 2:
        return {-1.0, 0.0};
    default:
        return {0.0, -1.0};
    }
}

/// Largest radius at which any channel has a non-negligible AE - PS difference.
double paw_radius(const PawDataset& d) {
    double mx = 0.0;
    for (const auto& c : d.channels)
        for (std::size_t i = 0; i < d.grid.size(); ++i)
            mx = std::max(mx, std::abs(c.u_ae[i] - c.u_ps[i]));
    double r = d.grid.front();
    for (const auto& c : d.channels)
        for (std::size_t i = 0; i < d.grid.size(); ++i)
            if (std::abs(c.u_ae[i] - c.u_ps[i]) > 1e-10 * mx) r = std::max(r, d.grid[i]);
    return r;
}

/// Enumerate lattice vectors R with |tau_b + R - tau_a| < rcut.
std::vector<IVec3> lattice_vectors_within(const Lattice& lat, double rcut, const std::vector<Vec3>& frac) {
    // |tau_b + R - tau_a| < rcut requires |R_i| < rcut |b_i| + |frac_b - frac_a|_i
    IVec3 n{};
    for (int i = 0; i < 3; ++i) {
        double lo = 0.0, hi = 0.0, span = 0.0;
        if (!frac.empty()) {
            lo = hi = frac[0][i];
            for (const auto& f : frac) {
                lo = std::min(lo, f[i]);
                hi = std::max(hi, f[i]);
            }
            span = hi - lo;
        }
        n[i] = static_cast<int>(std::ceil(rcut * norm(lat.b[i]) + span)) + 1;
    }
    std::vector<IVec3> out;
    for (int x = -n[0]; x <= n[0]; ++x)
        for (int y = -n[1]; y <= n[1]; ++y)
            for (int z = -n[2]; z <= n[2]; ++z)
                out.push_back({x, y, z});
    return out;
}

struct AngularGrid {
    std::vector<Vec3> dir;
    std::vector<double> w;
    AngularGrid(int nt, int np) {
        std::vector<double> x(nt), wt(nt);
        for (int i = 0; i < nt; ++i) {
            double z = std::cos(constants::pi * (i + 0.75) / (nt + 0.5)), pp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < nt; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
                }
                pp = nt * (z * p1 - p2) / (z * z - 1.0);
                const double dz = p1 / pp;
                z -= dz;
                if (std::abs(dz) < 1e-15) break;
            }
            x[i] = z;
            wt[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        }
        for (int i = 0; i < nt; ++i)
            for (int j = 0; j < np; ++j) {
                const double phi = constants::two_pi * (j + 0.5) / np, st = std::sqrt(1.0 - x[i] * x[i]);
                dir.push_back({st * std::cos(phi), st * std::sin(phi), x[i]});
                w.push_back(wt[i] * constants::two_pi / np);
            }
    }
};

/// Unique id of a factor table set (device caching).
unsigned long long new_table_id() {
    static std::atomic<unsigned long long> next_id{1};
    return next_id++;
}

} // namespace

ProjectionModel::ProjectionModel(const Structure& s, const AtomicBasis& basis, std::vector<PawDataset> datasets,
                                 ProjectionOptions opt)
    : s_(s), basis_(basis), datasets_(std::move(datasets)), opt_(opt) {
    require(s_.natoms() > 0 && s_.atom_type.size() == s_.frac.size() && basis_.size() > 0, ErrorCode::InvalidArgument,
            "projection requires atoms, consistent atom types and a nonempty basis");
    for (int a = 0; a < s_.natoms(); ++a) {
        require(s_.atom_type[a] >= 0 && static_cast<std::size_t>(s_.atom_type[a]) < s_.species.size(),
                ErrorCode::InvalidArgument, "projection atom type out of range");
        for (double x : s_.frac[a])
            require(std::isfinite(x), ErrorCode::InvalidArgument, "non-finite atom coordinate");
    }
    require(opt_.overlap_nq >= 5 && opt_.overlap_nq % 2 == 1 && std::isfinite(opt_.overlap_qmax) &&
                opt_.overlap_qmax > 0 && std::isfinite(opt_.overlap_table_dd) && opt_.overlap_table_dd >= 0 &&
                std::isfinite(opt_.overlap_extent_threshold) && opt_.overlap_extent_threshold > 0 &&
                std::isfinite(opt_.offsite_threshold) && opt_.offsite_threshold > 0 &&
                std::isfinite(opt_.coupling_table_dd) && opt_.coupling_table_dd > 0 &&
                std::isfinite(opt_.coupling_table_rmin_factor) && opt_.coupling_table_rmin_factor >= 0 &&
                opt_.angular_ntheta > 0 && opt_.angular_nphi > 0,
            ErrorCode::InvalidArgument, "invalid projection table/quadrature options");
    require(datasets_.size() == s_.species.size(), ErrorCode::InconsistentInput,
            "POTCAR has " + std::to_string(datasets_.size()) + " datasets but POSCAR lists " +
                std::to_string(s_.species.size()) + " species");
    for (std::size_t t = 0; t < datasets_.size(); ++t)
        require(datasets_[t].element == s_.species[t], ErrorCode::InconsistentInput,
                "POTCAR dataset " + std::to_string(t + 1) + " is for element '" + datasets_[t].element +
                    "' but POSCAR species is '" + s_.species[t] + "'");
    for (const auto& d : datasets_) {
        require(d.grid.size() >= 4 && !d.channels.empty() && std::isfinite(d.psmaxn) && d.psmaxn > 0,
                ErrorCode::InvalidArgument, "projection PAW dataset has missing grid/channels or invalid q range");
        for (std::size_t i = 0; i < d.grid.size(); ++i)
            require(std::isfinite(d.grid[i]) && d.grid[i] > 0 && (i == 0 || d.grid[i] > d.grid[i - 1]),
                    ErrorCode::InvalidArgument, "projection PAW grid must be positive and increasing");
        for (const auto& channel : d.channels)
            require(channel.l >= 0 && channel.l <= 3 && channel.projector_q.size() == PawDataset::nq() &&
                        channel.u_ae.size() == d.grid.size() && channel.u_ps.size() == d.grid.size(),
                    ErrorCode::InvalidArgument, "projection PAW channel dimensions are inconsistent");
    }
    // orbitals must be grouped by atom in POSCAR order
    orb_offset_.assign(s_.natoms() + 1, 0);
    int prev = -1;
    for (int i = 0; i < basis_.size(); ++i) {
        const int a = basis_.orbitals()[i].atom;
        require(a >= 0 && a < s_.natoms() && a >= prev, ErrorCode::InvalidArgument,
                "basis atom order or index is invalid");
        prev = a;
        orb_offset_[a + 1] = i + 1;
    }
    for (int a = 1; a <= s_.natoms(); ++a)
        orb_offset_[a] = std::max(orb_offset_[a], orb_offset_[a - 1]);
    auto timed = [this](const char* name, auto&& f) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        setup_timings_.emplace_back(name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    };
    timed("projectors", [&] { build_projectors(); });
    timed("fourier_tables", [&] { build_tables(); });
    timed("factor_specs", [&] { build_specs(); });
    timed("overlap_tables", [&] { build_overlap_terms(); });
    timed("paw_coupling", [&] { build_coupling_terms(); });
    timed("paw_fold", [&] { build_fold_spec(); });
}

void ProjectionModel::build_projectors() {
    proj_offset_.assign(s_.natoms() + 1, 0);
    for (int a = 0; a < s_.natoms(); ++a) {
        const auto& d = datasets_[s_.atom_type[a]];
        for (int c = 0; c < d.nchannels(); ++c) {
            const int l = d.channels[c].l;
            for (int m = -l; m <= l; ++m)
                projectors_.push_back({a, c, l, m});
        }
        proj_offset_[a + 1] = static_cast<int>(projectors_.size());
    }
    const int np = nproj();
    Q_ = Eigen::MatrixXd::Zero(np, np);
    for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j) {
            const auto& pi = projectors_[i];
            const auto& pj = projectors_[j];
            if (pi.atom != pj.atom || pi.l != pj.l || pi.m != pj.m) continue;
            const auto& d = datasets_[s_.atom_type[pi.atom]];
            const auto& ci = d.channels[pi.channel];
            const auto& cj = d.channels[pj.channel];
            std::vector<double> f(d.grid.size());
            for (std::size_t r = 0; r < f.size(); ++r)
                f[r] = ci.u_ae[r] * cj.u_ae[r] - ci.u_ps[r] * cj.u_ps[r];
            Q_(i, j) = radial::integrate_log_grid(d.grid, f);
        }
}

void ProjectionModel::build_tables() {
    orbital_qmax_ = 40.0;
    const int nq = 10001;
    std::vector<double> q(nq);
    for (int i = 0; i < nq; ++i)
        q[i] = orbital_qmax_ * i / (nq - 1);
    for (const auto& r : basis_.radials()) {
        std::vector<double> v(nq);
        for (int i = 0; i < nq; ++i)
            v[i] = constants::four_pi * r.fourier(q[i]);
        orbital_ft_.emplace_back(q, v);
    }
    for (const auto& d : datasets_) {
        std::vector<radial::CubicSpline> per;
        const double dq = d.psmaxn / PawDataset::nq();
        for (const auto& c : d.channels) {
            // ghost points at negative q with even/odd parity, as the table is a function of |q| times q^l
            std::vector<double> x, y;
            const double sgn = (c.l % 2 == 0) ? 1.0 : -1.0;
            for (int i = 3; i >= 1; --i) {
                x.push_back(-i * dq);
                y.push_back(sgn * c.projector_q[i]);
            }
            for (int i = 0; i < PawDataset::nq(); ++i) {
                x.push_back(i * dq);
                y.push_back(c.projector_q[i]);
            }
            per.emplace_back(x, y);
        }
        projector_ft_.push_back(std::move(per));
    }
}

void ProjectionModel::build_overlap_terms() {
    const auto& R = basis_.radials();
    std::map<std::pair<int, int>, TwoCenterOverlap> tables;
    double rcut = 0.0;
    // Per-shell transforms once; one table per unordered pair: S_ba(d) = S_ab(-d)^T (J_L symmetric in a, b).
    std::vector<TwoCenterShell> sh;
    for (const auto& r : R)
        sh.push_back(TwoCenterShell::prepare(r, opt_.overlap_qmax, opt_.overlap_nq, opt_.overlap_extent_threshold));
    for (std::size_t i = 0; i < R.size(); ++i)
        for (std::size_t j = i; j < R.size(); ++j) {
            auto [it, inserted] = tables.try_emplace(std::make_pair(static_cast<int>(i), static_cast<int>(j)), sh[i],
                                                     sh[j], opt_.overlap_table_dd);
            rcut = std::max(rcut, it->second.cutoff());
        }
    const auto t_tab = std::chrono::steady_clock::now();
    const auto Rs = lattice_vectors_within(s_.lattice, rcut, s_.frac);
    const auto& orbs = basis_.orbitals();
    // candidate (a, b, R) list
    std::vector<LatticeTerm> cand;
    for (int a = 0; a < s_.natoms(); ++a)
        for (int b = 0; b < s_.natoms(); ++b)
            for (const auto& Rv : Rs) {
                const Vec3 fd{s_.frac[b][0] + Rv[0] - s_.frac[a][0], s_.frac[b][1] + Rv[1] - s_.frac[a][1],
                              s_.frac[b][2] + Rv[2] - s_.frac[a][2]};
                if (norm(s_.lattice.to_cartesian(fd)) < rcut) cand.push_back({a, b, Rv, {}});
            }
    std::vector<char> keep(cand.size(), 0);
    OmpExceptionGuard guard;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t t = 0; t < cand.size(); ++t)
        guard.run([&] {
            auto& term = cand[t];
            const int a = term.row_atom, b = term.col_atom;
            const Vec3 fd{s_.frac[b][0] + term.R[0] - s_.frac[a][0], s_.frac[b][1] + term.R[1] - s_.frac[a][1],
                          s_.frac[b][2] + term.R[2] - s_.frac[a][2]};
            const Vec3 d = s_.lattice.to_cartesian(fd);
            const double dist = norm(d);
            const int na = orb_offset_[a + 1] - orb_offset_[a], nb = orb_offset_[b + 1] - orb_offset_[b];
            term.block = Eigen::MatrixXd::Zero(na, nb);
            double buf[49];
            bool any = false;
            // iterate over radial shells (contiguous m blocks)
            for (int i = orb_offset_[a]; i < orb_offset_[a + 1];) {
                const int ri = orbs[i].radial, li = orbs[i].l;
                for (int j = orb_offset_[b]; j < orb_offset_[b + 1];) {
                    const int rj = orbs[j].radial, lj = orbs[j].l;
                    const bool swap = ri > rj;
                    const auto& tab = swap ? tables.at({rj, ri}) : tables.at({ri, rj});
                    if (dist < tab.cutoff()) {
                        if (!swap) {
                            tab.evaluate(d, buf);
                            for (int m1 = 0; m1 < 2 * li + 1; ++m1)
                                for (int m2 = 0; m2 < 2 * lj + 1; ++m2)
                                    term.block(i - orb_offset_[a] + m1, j - orb_offset_[b] + m2) =
                                        buf[m1 * (2 * lj + 1) + m2];
                        } else { // table (rj, ri) at -d, transposed: buf is (2 lj + 1) x (2 li + 1)
                            tab.evaluate({-d[0], -d[1], -d[2]}, buf);
                            for (int m1 = 0; m1 < 2 * li + 1; ++m1)
                                for (int m2 = 0; m2 < 2 * lj + 1; ++m2)
                                    term.block(i - orb_offset_[a] + m1, j - orb_offset_[b] + m2) =
                                        buf[m2 * (2 * li + 1) + m1];
                        }
                        any = true;
                    }
                    j += 2 * lj + 1;
                }
                i += 2 * li + 1;
            }
            keep[t] = any ? 1 : 0;
        });
    guard.rethrow();
    for (std::size_t t = 0; t < cand.size(); ++t)
        if (keep[t]) s_terms_.push_back(std::move(cand[t]));
    setup_timings_.emplace_back("overlap_terms",
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tab).count());
    setup_timings_.emplace_back("overlap_terms_count", static_cast<double>(s_terms_.size()));
}

void ProjectionModel::build_coupling_terms() {
    const auto& orbs = basis_.orbitals();
    const auto& rad = basis_.radials();
    const AngularGrid ang(opt_.angular_ntheta, opt_.angular_nphi);
    double ext = 0.0;
    for (const auto& r : rad)
        ext = std::max(ext, r.extent(opt_.offsite_threshold));
    std::vector<double> rpaw(datasets_.size());
    double rpaw_max = 0.0;
    for (std::size_t t = 0; t < datasets_.size(); ++t)
        rpaw_max = std::max(rpaw_max, rpaw[t] = paw_radius(datasets_[t]));
    const auto Rs = lattice_vectors_within(s_.lattice, ext + rpaw_max, s_.frac);
    // Splined radial functions (exact below r_spline_min where s-type cusps make splines inaccurate).
    const double r_spline_min = 0.2, dr = 0.0005;
    std::vector<radial::CubicSpline> rspl;
    for (const auto& r : rad) {
        const int n = static_cast<int>((ext + rpaw_max + 1.0 - r_spline_min) / dr) + 2;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = r_spline_min + i * dr;
            y[i] = r.value(x[i]);
        }
        rspl.emplace_back(x, y);
    }
    auto radial_value = [&](int ri, double r) {
        return r < r_spline_min ? rad[ri].value(r) : (r > rspl[ri].xmax() ? 0.0 : rspl[ri](r));
    };
    // radial grid inside each PAW sphere
    std::vector<std::vector<double>> sgrid(datasets_.size());
    for (std::size_t t = 0; t < datasets_.size(); ++t) {
        const auto& ds = datasets_[t];
        std::size_t nr = 0;
        while (nr < ds.grid.size() && ds.grid[nr] <= rpaw[t] * 1.0001)
            ++nr;
        nr = std::min(ds.grid.size(), std::max<std::size_t>(nr + 2, 8));
        sgrid[t].assign(ds.grid.begin(), ds.grid.begin() + static_cast<long>(nr));
    }
    constexpr int lmax_p = 3, nlm_p = ylm::count(lmax_p);
    std::vector<double> Yp(ang.dir.size() * nlm_p);
    for (std::size_t g = 0; g < ang.dir.size(); ++g)
        ylm::real_ylm(lmax_p, ang.dir[g], &Yp[g * nlm_p]);

    // Direct single-centre expansion for one radial shell (all m) against ALL PAW channels of
    // species t: block (2 la + 1) x nproj_t with columns ordered (channel, m') as the projectors.
    auto direct_shell = [&](int ra, int t, const Vec3& d, Eigen::MatrixXd& out) {
        const int la = rad[ra].l;
        const auto& ds = datasets_[t];
        const auto& grid = sgrid[t];
        const std::size_t nr = grid.size();
        int lcmax = 0;
        for (const auto& ch : ds.channels)
            lcmax = std::max(lcmax, ch.l);
        const int nlm = ylm::count(lcmax);
        std::vector<double> F(static_cast<std::size_t>(2 * la + 1) * nlm * nr, 0.0);
        double Yo[ylm::count(ylm::lmax_supported)];
        for (std::size_t r = 0; r < nr; ++r)
            for (std::size_t g = 0; g < ang.dir.size(); ++g) {
                const Vec3 x{d[0] + grid[r] * ang.dir[g][0], d[1] + grid[r] * ang.dir[g][1],
                             d[2] + grid[r] * ang.dir[g][2]};
                const double R = radial_value(ra, norm(x)) * ang.w[g];
                if (R == 0.0) continue;
                ylm::real_ylm(la, x, Yo);
                const double* yp = &Yp[g * nlm_p];
                for (int m = 0; m < 2 * la + 1; ++m) {
                    const double chi = R * Yo[la * la + m];
                    double* Fm = &F[static_cast<std::size_t>(m) * nlm * nr + r];
                    for (int lm = 0; lm < nlm; ++lm)
                        Fm[lm * nr] += chi * yp[lm];
                }
            }
        out.resize(2 * la + 1, ds.nprojectors());
        std::vector<double> f(nr);
        int col = 0;
        for (const auto& ch : ds.channels)
            for (int mc = -ch.l; mc <= ch.l; ++mc, ++col)
                for (int m = 0; m < 2 * la + 1; ++m) {
                    const double* Fi = &F[(static_cast<std::size_t>(m) * nlm + ylm::index(ch.l, mc)) * nr];
                    for (std::size_t r = 0; r < nr; ++r)
                        f[r] = Fi[r] * (ch.u_ae[r] - ch.u_ps[r]) * grid[r];
                    out(m, col) = radial::integrate_log_grid(grid, f);
                }
    };

    // z-frame couplings for tables: with the neighbour on +z the integrand is axially symmetric, so
    // the phi integral is analytic (2 pi for m = 0, pi for m > 0) and only theta is integrated.
    // Fills the same block layout as direct_shell, but only the (m, m) entries with m >= 0.
    const int nth = 2 * opt_.angular_ntheta;
    std::vector<double> gl_x(nth), gl_w(nth);
    for (int i = 0; i < nth; ++i) {
        double z = std::cos(constants::pi * (i + 0.75) / (nth + 0.5)), pp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < nth; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
            }
            pp = nth * (z * p1 - p2) / (z * z - 1.0);
            const double dz = p1 / pp;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        gl_x[i] = z;
        gl_w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
    auto zframe_shell = [&](int ra, int t, double dist, Eigen::MatrixXd& out) {
        const int la = rad[ra].l;
        const auto& ds = datasets_[t];
        const auto& grid = sgrid[t];
        const std::size_t nr = grid.size();
        int lcmax = 0;
        for (const auto& ch : ds.channels)
            lcmax = std::max(lcmax, ch.l);
        const int mm_max = std::min(la, lcmax);
        // F[m][lc][r] = int dOmega R(|x|) Y_{la m}(x^) Y_{lc m}(y^), x = d z^ + r y^
        std::vector<double> F(static_cast<std::size_t>(mm_max + 1) * (lcmax + 1) * nr, 0.0);
        double Yx[ylm::count(ylm::lmax_supported)], Yy[ylm::count(ylm::lmax_supported)];
        for (std::size_t r = 0; r < nr; ++r)
            for (int g = 0; g < nth; ++g) {
                const double ct = gl_x[g], st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
                const Vec3 y{st, 0.0, ct};
                const Vec3 xv{grid[r] * st, 0.0, dist + grid[r] * ct};
                const double R = radial_value(ra, norm(xv)) * gl_w[g];
                if (R == 0.0) continue;
                ylm::real_ylm(la, xv, Yx);
                ylm::real_ylm(lcmax, y, Yy);
                for (int m = 0; m <= mm_max; ++m) {
                    const double phi = m == 0 ? constants::two_pi : constants::pi;
                    const double a = R * Yx[ylm::index(la, m)] * phi;
                    for (int lc = m; lc <= lcmax; ++lc)
                        F[(static_cast<std::size_t>(m) * (lcmax + 1) + lc) * nr + r] += a * Yy[ylm::index(lc, m)];
                }
            }
        out = Eigen::MatrixXd::Zero(2 * la + 1, ds.nprojectors());
        std::vector<double> f(nr);
        int col0 = 0;
        for (const auto& ch : ds.channels) {
            for (int m = 0; m <= std::min(la, ch.l); ++m) {
                const double* Fi = &F[(static_cast<std::size_t>(m) * (lcmax + 1) + ch.l) * nr];
                for (std::size_t r = 0; r < nr; ++r)
                    f[r] = Fi[r] * (ch.u_ae[r] - ch.u_ps[r]) * grid[r];
                out(m + la, col0 + m + ch.l) = radial::integrate_log_grid(grid, f);
            }
            col0 += 2 * ch.l + 1;
        }
    };

    // Slater-Koster style tables: in the frame with the neighbour along +z only m-diagonal
    // couplings V_|m|(d) survive; tabulate them once per (radial, species, channel).
    struct SkTable {
        double dmin{}, dmax{};
        std::vector<radial::CubicSpline> v; // index |m|
    };
    std::map<std::tuple<int, int, int>, SkTable> tables;
    std::set<int> used_types;
    for (int a = 0; a < s_.natoms(); ++a)
        used_types.insert(s_.atom_type[a]);
    if (opt_.include_offsite_paw) {
        const double dd = opt_.coupling_table_dd;
        for (std::size_t ra = 0; ra < rad.size(); ++ra)
            for (int t : used_types) {
                const auto& ds = datasets_[t];
                const double dmin = opt_.coupling_table_rmin_factor * rpaw[t];
                // per-shell range: beyond its own extent the shell is screened out anyway
                const double dmax = rad[ra].extent(opt_.offsite_threshold) + rpaw[t] + 2 * dd;
                if (!(dmin + 4 * dd < dmax)) continue; // empty range: direct quadrature everywhere
                const int nd = static_cast<int>((dmax - dmin) / dd) + 2;
                const int la = rad[ra].l;
                std::vector<double> x(nd);
                std::vector<Eigen::MatrixXd> blocks(nd);
                OmpExceptionGuard guard;
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < nd; ++i)
                    guard.run([&] {
                        x[i] = dmin + i * dd;
                        zframe_shell(static_cast<int>(ra), t, x[i], blocks[i]);
                    });
                guard.rethrow();
                int col0 = 0;
                for (int c = 0; c < ds.nchannels(); ++c) {
                    const int lc = ds.channels[c].l, mm = std::min(la, lc);
                    SkTable tab;
                    tab.dmin = dmin;
                    tab.dmax = x.back();
                    for (int m = 0; m <= mm; ++m) {
                        std::vector<double> y(nd);
                        for (int i = 0; i < nd; ++i)
                            y[i] = blocks[i](m + la, col0 + m + lc);
                        tab.v.emplace_back(x, y);
                    }
                    tables.emplace(std::make_tuple(static_cast<int>(ra), t, c), std::move(tab));
                    col0 += 2 * lc + 1;
                }
            }
    }
    // Rotation matrices D^l_{m m''}(Q) = int Y_lm(Q y) Y_lm''(y) dOmega for Q z = d^ (exact quadrature).
    auto rotation_matrices = [&](const Vec3& dhat, std::array<Eigen::MatrixXd, 4>& Dl) {
        // Rodrigues rotation taking z to dhat
        const Vec3 z{0, 0, 1};
        Vec3 axis = cross(z, dhat);
        const double sn = norm(axis), cs = dhat[2];
        double Q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        if (sn > 1e-14) {
            for (auto& v : axis)
                v /= sn;
            const double K[3][3] = {{0, -axis[2], axis[1]}, {axis[2], 0, -axis[0]}, {-axis[1], axis[0], 0}};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double kk = 0;
                    for (int l = 0; l < 3; ++l)
                        kk += K[i][l] * K[l][j];
                    Q[i][j] = (i == j ? 1.0 : 0.0) + sn * K[i][j] + (1 - cs) * kk;
                }
        } else if (cs < 0) {
            Q[1][1] = -1;
            Q[2][2] = -1; // rotation by pi about x
        }
        for (int l = 0; l <= 3; ++l)
            Dl[l] = Eigen::MatrixXd::Zero(2 * l + 1, 2 * l + 1);
        double Yq[16];
        for (std::size_t g = 0; g < ang.dir.size(); ++g) {
            const auto& y = ang.dir[g];
            const Vec3 qy{Q[0][0] * y[0] + Q[0][1] * y[1] + Q[0][2] * y[2],
                          Q[1][0] * y[0] + Q[1][1] * y[1] + Q[1][2] * y[2],
                          Q[2][0] * y[0] + Q[2][1] * y[1] + Q[2][2] * y[2]};
            ylm::real_ylm(3, qy, Yq);
            for (int l = 0; l <= 3; ++l)
                for (int m = -l; m <= l; ++m)
                    for (int m2 = -l; m2 <= l; ++m2)
                        Dl[l](m + l, m2 + l) += ang.w[g] * Yq[ylm::index(l, m)] * Yp[g * nlm_p + ylm::index(l, m2)];
        }
    };

    std::vector<LatticeTerm> cand;
    for (int a = 0; a < s_.natoms(); ++a)
        for (int b = 0; b < s_.natoms(); ++b)
            for (const auto& Rv : Rs) {
                const bool onsite = (a == b && Rv[0] == 0 && Rv[1] == 0 && Rv[2] == 0);
                const Vec3 fd{s_.frac[b][0] + Rv[0] - s_.frac[a][0], s_.frac[b][1] + Rv[1] - s_.frac[a][1],
                              s_.frac[b][2] + Rv[2] - s_.frac[a][2]};
                const double dist = norm(s_.lattice.to_cartesian(fd));
                if (onsite || (opt_.include_offsite_paw && dist < ext + rpaw[s_.atom_type[b]]))
                    cand.push_back({a, b, Rv, {}});
            }

    OmpExceptionGuard guard;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t t = 0; t < cand.size(); ++t)
        guard.run([&] {
            auto& term = cand[t];
            const int a = term.row_atom, b = term.col_atom, tb = s_.atom_type[b];
            const auto& ds = datasets_[tb];
            const int na = orb_offset_[a + 1] - orb_offset_[a];
            const int nb = proj_offset_[b + 1] - proj_offset_[b];
            term.block = Eigen::MatrixXd::Zero(na, nb);
            const auto& grid = sgrid[tb];
            const std::size_t nr = grid.size();
            const bool onsite = (a == b && term.R[0] == 0 && term.R[1] == 0 && term.R[2] == 0);
            const Vec3 fd{s_.frac[b][0] + term.R[0] - s_.frac[a][0], s_.frac[b][1] + term.R[1] - s_.frac[a][1],
                          s_.frac[b][2] + term.R[2] - s_.frac[a][2]};
            const Vec3 d = s_.lattice.to_cartesian(fd);
            const double dist = norm(d);

            if (onsite) {
                for (int i = orb_offset_[a]; i < orb_offset_[a + 1]; ++i)
                    for (int p = proj_offset_[b]; p < proj_offset_[b + 1]; ++p) {
                        const auto& pj = projectors_[p];
                        if (pj.l != orbs[i].l || pj.m != orbs[i].m) continue;
                        const auto& ch = ds.channels[pj.channel];
                        std::vector<double> f(nr);
                        for (std::size_t r = 0; r < nr; ++r)
                            f[r] = rad[orbs[i].radial].value(grid[r]) * (ch.u_ae[r] - ch.u_ps[r]) * grid[r];
                        term.block(i - orb_offset_[a], p - proj_offset_[b]) = radial::integrate_log_grid(grid, f);
                    }
                return;
            }
            // Screening: skip if every orbital of atom a is negligible on the PAW sphere of b.
            {
                const double rmin = std::max(dist - rpaw[tb], 0.0);
                double mx = 0.0;
                for (int i = orb_offset_[a]; i < orb_offset_[a + 1]; ++i)
                    mx = std::max(mx, std::abs(rmin * rad[orbs[i].radial].value(rmin)));
                if (rmin > 0.0 && mx < opt_.offsite_threshold) {
                    term.block.resize(0, 0);
                    return;
                }
            }
            std::array<Eigen::MatrixXd, 4> Dl;
            bool have_rot = false;
            Eigen::MatrixXd blk;
            for (int i = orb_offset_[a]; i < orb_offset_[a + 1]; i += 2 * orbs[i].l + 1) {
                const int ra = orbs[i].radial, la = orbs[i].l;
                {
                    const double rmin = std::max(dist - rpaw[tb], 0.0);
                    if (rmin > 0.0 && std::abs(rmin * rad[ra].value(rmin)) < opt_.offsite_threshold)
                        continue; // stays zero
                }
                bool tabulated = ds.nchannels() > 0;
                for (int c = 0; c < ds.nchannels() && tabulated; ++c) {
                    auto it = tables.find({ra, tb, c});
                    tabulated = it != tables.end() && dist >= it->second.dmin && dist <= it->second.dmax;
                }
                if (!tabulated) {
                    direct_shell(ra, tb, d, blk);
                    term.block.block(i - orb_offset_[a], 0, 2 * la + 1, nb) = blk;
                    continue;
                }
                if (!have_rot) {
                    rotation_matrices({d[0] / dist, d[1] / dist, d[2] / dist}, Dl);
                    have_rot = true;
                }
                int pcol = 0;
                for (int c = 0; c < ds.nchannels(); ++c) {
                    const int lc = ds.channels[c].l, mm = std::min(la, lc);
                    const auto& tab = tables.at({ra, tb, c});
                    blk = Eigen::MatrixXd::Zero(2 * la + 1, 2 * lc + 1);
                    for (int m2 = -mm; m2 <= mm; ++m2)
                        blk += tab.v[std::abs(m2)](dist) * Dl[la].col(m2 + la) * Dl[lc].col(m2 + lc).transpose();
                    term.block.block(i - orb_offset_[a], pcol, 2 * la + 1, 2 * lc + 1) = blk;
                    pcol += 2 * lc + 1;
                }
            }
            if (term.block.cwiseAbs().maxCoeff() < 1e-14) term.block.resize(0, 0);
        });
    guard.rethrow();
    for (auto& t : cand)
        if (t.block.size() > 0) d_terms_.push_back(std::move(t));
}

void ProjectionModel::build_specs() {
    auto add_table = [](SeparableFactorSpec& sp, const radial::CubicSpline& t, double xmax) {
        require(t.uniform(), ErrorCode::Internal, "factor tables must be uniform splines");
        sp.t_x0.push_back(t.xmin());
        sp.t_h.push_back(t.h());
        sp.t_xmax.push_back(xmax);
        sp.t_n.push_back(static_cast<int>(t.y().size()));
        sp.t_offset.push_back(static_cast<int>(sp.t_y.size()));
        sp.t_y.insert(sp.t_y.end(), t.y().begin(), t.y().end());
        sp.t_m.insert(sp.t_m.end(), t.second_derivatives().begin(), t.second_derivatives().end());
    };
    const double pref = 1.0 / std::sqrt(s_.lattice.volume);
    // orbitals: Omega^{-1/2} i^l Y_lm 4 pi Rt(q) e^{i q.tau}
    orb_spec_ = SeparableFactorSpec{};
    orb_spec_.b = s_.lattice.b;
    orb_spec_.pref = pref;
    for (const auto& t : orbital_ft_)
        add_table(orb_spec_, t, t.xmax());
    for (const auto& o : basis_.orbitals()) {
        require(o.l <= 3, ErrorCode::UnsupportedFeature, "basis functions beyond f are not supported");
        for (int c = 0; c < 3; ++c)
            orb_spec_.tau.push_back(s_.frac[o.atom][c]);
        orb_spec_.lm.push_back(ylm::index(o.l, o.m));
        orb_spec_.table.push_back(o.radial);
        orb_spec_.coef.push_back(ipow(o.l));
    }
    orb_spec_.table_id = new_table_id();
    // PAW projectors: Omega^{-1/2} i^l Y_lm PSPNL(q) e^{i q.tau}, zero beyond the table
    proj_spec_ = SeparableFactorSpec{};
    proj_spec_.b = s_.lattice.b;
    proj_spec_.pref = pref;
    std::vector<std::vector<int>> tid(datasets_.size());
    for (std::size_t t = 0; t < datasets_.size(); ++t) {
        const double qlim = datasets_[t].psmaxn * (PawDataset::nq() - 1) / PawDataset::nq();
        for (const auto& sp : projector_ft_[t]) {
            tid[t].push_back(static_cast<int>(proj_spec_.t_x0.size()));
            add_table(proj_spec_, sp, qlim);
        }
    }
    for (const auto& pj : projectors_) {
        for (int c = 0; c < 3; ++c)
            proj_spec_.tau.push_back(s_.frac[pj.atom][c]);
        proj_spec_.lm.push_back(ylm::index(pj.l, pj.m));
        proj_spec_.table.push_back(tid[s_.atom_type[pj.atom]][pj.channel]);
        proj_spec_.coef.push_back(ipow(pj.l));
    }
    proj_spec_.table_id = new_table_id();
}

void ProjectionModel::build_fold_spec() {
    fold_spec_ = SeparableFactorSpec{};
    // Foldable if every coupling is one-centre and (l, m)-diagonal: the orbital and projector factors then share
    // Y_lm, i^l and the phase of the atom, and differ only in the radial factor.
    const auto& orbs = basis_.orbitals();
    std::vector<std::vector<std::pair<int, double>>> extra(norb());
    for (const auto& t : d_terms_) {
        const bool onsite = t.row_atom == t.col_atom && t.R == IVec3{0, 0, 0};
        for (Eigen::Index i = 0; i < t.block.rows(); ++i)
            for (Eigen::Index p = 0; p < t.block.cols(); ++p) {
                const double v = t.block(i, p);
                if (v == 0.0) continue;
                const int mu = orb_offset_[t.row_atom] + static_cast<int>(i);
                const int pj = proj_offset_[t.col_atom] + static_cast<int>(p);
                if (!onsite || orbs[mu].l != projectors_[pj].l || orbs[mu].m != projectors_[pj].m) return;
                extra[mu].emplace_back(pj, v);
            }
    }
    auto& f = fold_spec_;
    f.b = orb_spec_.b;
    f.pref = orb_spec_.pref;
    auto append_tables = [&f](const SeparableFactorSpec& src) {
        const int t0 = static_cast<int>(f.t_x0.size()), y0 = static_cast<int>(f.t_y.size());
        f.t_x0.insert(f.t_x0.end(), src.t_x0.begin(), src.t_x0.end());
        f.t_h.insert(f.t_h.end(), src.t_h.begin(), src.t_h.end());
        f.t_xmax.insert(f.t_xmax.end(), src.t_xmax.begin(), src.t_xmax.end());
        f.t_n.insert(f.t_n.end(), src.t_n.begin(), src.t_n.end());
        for (const int o : src.t_offset)
            f.t_offset.push_back(o + y0);
        f.t_y.insert(f.t_y.end(), src.t_y.begin(), src.t_y.end());
        f.t_m.insert(f.t_m.end(), src.t_m.begin(), src.t_m.end());
        return t0;
    };
    const int to = append_tables(orb_spec_), tp = append_tables(proj_spec_);
    auto add_function = [&f](const SeparableFactorSpec& src, int i, int table) {
        f.tau.insert(f.tau.end(), src.tau.begin() + 3 * i, src.tau.begin() + 3 * i + 3);
        f.lm.push_back(src.lm[i]);
        f.table.push_back(table);
        f.coef.push_back(src.coef[i]);
    };
    f.x_off.push_back(0);
    for (int mu = 0; mu < norb(); ++mu) {
        add_function(orb_spec_, mu, orb_spec_.table[mu] + to);
        for (const auto& [pj, v] : extra[mu]) {
            f.x_table.push_back(proj_spec_.table[pj] + tp);
            f.x_w.push_back(v);
        }
        f.x_off.push_back(static_cast<int>(f.x_table.size()));
    }
    for (int pj = 0; pj < nproj(); ++pj) {
        add_function(proj_spec_, pj, proj_spec_.table[pj] + tp);
        f.x_off.push_back(static_cast<int>(f.x_table.size()));
    }
    f.table_id = new_table_id();
}

} // namespace prowl
