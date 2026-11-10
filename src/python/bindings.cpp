// SPDX-License-Identifier: GPL-3.0-or-later
/// nanobind bindings for the PROWL C++ core (module prowl._core).
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "prowl/analysis.hpp"
#include "prowl/backend.hpp"
#include "prowl/basis.hpp"
#include "prowl/errors.hpp"
#include "prowl/potcar.hpp"
#include "prowl/projection.hpp"
#include "prowl/structure.hpp"
#include "prowl/twocenter.hpp"
#include "prowl/wavecar.hpp"
#include "prowl/ylm.hpp"
#include "prowl/threading.hpp"

#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
/// Bands covered by b and paw_norm: the sample of a folded projection, otherwise all bands.
std::vector<int> b_bands(const prowl::KProjection& kp) {
    if (!kp.paw_bands.empty()) return kp.paw_bands;
    std::vector<int> all(static_cast<std::size_t>(kp.paw_norm.size()));
    std::iota(all.begin(), all.end(), 0);
    return all;
}
} // namespace

namespace nb = nanobind;
using namespace nb::literals;
using prowl::Vec3;

namespace {

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T> v, std::vector<std::size_t> shape) {
    auto* p = new std::vector<T>(std::move(v));
    nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
    return nb::ndarray<nb::numpy, T>(p->data(), shape.size(), shape.data(), owner);
}

Eigen::Matrix3d lattice_matrix(const prowl::Lattice& L) {
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i)
        for (int c = 0; c < 3; ++c) m(i, c) = L.a[i][c];
    return m;
}

Vec3 to_vec3(const std::vector<double>& v) {
    if (v.size() != 3) throw prowl::Error(prowl::ErrorCode::InvalidArgument, "expected a 3-vector");
    prowl::require(std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); }),
                    prowl::ErrorCode::InvalidArgument, "vector components must be finite");
    return {v[0], v[1], v[2]};
}

std::size_t to_byte_count(double value) {
    prowl::require(std::isfinite(value) && value >= 0 &&
                        value < static_cast<double>(std::numeric_limits<std::size_t>::max()),
                    prowl::ErrorCode::InvalidArgument, "factor memory budget must be finite and nonnegative");
    return static_cast<std::size_t>(value);
}

nb::dict block_to_dict(const prowl::WavefunctionBlock& b) {
    nb::dict d;
    std::vector<int> g;
    g.reserve(b.ng() * 3);
    for (const auto& x : *b.gvectors) g.insert(g.end(), x.begin(), x.end());
    d["gvectors"] = to_numpy(std::move(g), {b.ng(), 3});
    d["coefficients"] = to_numpy(b.coefficients, {static_cast<std::size_t>(b.band_count), b.ng()});
    d["eigenvalues"] = b.eigenvalues_ev;
    d["occupations"] = b.occupations;
    d["encoding"] = std::string(prowl::to_string(b.encoding));
    d["plane_wave_norms"] = b.plane_wave_norms();
    d["gamma_nonzero_scale"] = b.gamma_nonzero_scale;
    d["k_frac"] = std::vector<double>(b.k_frac.begin(), b.k_frac.end());
    return d;
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "PROWL native core: VASP readers, STO basis, PAW projection, compute backends";
    static nb::exception<prowl::Error> exc(m, "ProwlError");
    m.def("omp_max_threads", [] { return omp_get_max_threads(); });
    nb::class_<prowl::BlasThreads>(m, "BlasThreads")
        .def("__init__", [](prowl::BlasThreads* self, int threads) {
            nb::gil_scoped_release release;
            new (self) prowl::BlasThreads(threads);
        }, "threads"_a = 0)
        .def("__enter__", [](prowl::BlasThreads& self) -> prowl::BlasThreads& { return self; }, nb::rv_policy::reference)
        .def("__exit__", [](prowl::BlasThreads& self, nb::handle, nb::handle, nb::handle) { self.release(); },
             "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none());

    // ---------------------------------------------------------------- WAVECAR
    nb::class_<prowl::WavecarReader>(m, "WavecarReader")
        .def(
            "__init__",
            [](prowl::WavecarReader* self, const std::string& path, const std::string& encoding,
               const std::string& gamma_half) {
                prowl::ReaderOptions o;
                if (encoding == "full") o.encoding_override = prowl::WavefunctionEncoding::FullComplex;
                else if (encoding == "gamma") o.encoding_override = prowl::WavefunctionEncoding::GammaHalf;
                else if (encoding == "spinor") o.encoding_override = prowl::WavefunctionEncoding::Spinor;
                else if (encoding != "auto")
                    throw prowl::Error(prowl::ErrorCode::InvalidArgument, "encoding must be auto|full|gamma|spinor");
                if (gamma_half == "z") { o.gamma_half = prowl::GammaHalf::Z; o.gamma_half_overridden = true; }
                else if (gamma_half == "x-explicit") { o.gamma_half_overridden = true; }
                else if (gamma_half != "x")
                    throw prowl::Error(prowl::ErrorCode::InvalidArgument, "gamma_half must be x|x-explicit|z");
                new (self) prowl::WavecarReader(path, o);
            },
            "path"_a, "encoding"_a = "auto", "gamma_half"_a = "x")
        .def_prop_ro("header",
                     [](const prowl::WavecarReader& r) {
                         const auto& h = r.header();
                         nb::dict d;
                         d["record_length"] = h.record_length;
                         d["nspin"] = h.nspin;
                         d["rtag"] = h.rtag;
                         d["precision"] = h.precision == prowl::CoefficientPrecision::Single ? "single" : "double";
                         d["header_spans_records"] = h.header_spans_records;
                         d["nkpts"] = h.nkpts;
                         d["nbands"] = h.nbands;
                         d["encut_ev"] = h.encut_ev;
                         d["lattice"] = lattice_matrix(h.lattice);
                         d["volume"] = h.lattice.volume;
                         d["efermi_ev"] = h.efermi_ev;
                         d["file_size"] = h.file_size;
                         return d;
                     })
        .def_prop_ro("layout",
                     [](const prowl::WavecarReader& r) {
                         const auto& L = r.layout();
                         nb::dict d;
                         d["encoding"] = std::string(prowl::to_string(L.encoding));
                         d["gamma_half"] = L.gamma_half == prowl::GammaHalf::X ? "x" : "z";
                         d["gamma_half_assumed"] = L.gamma_half_assumed;
                         d["expected_full"] = L.expected_full;
                         d["stored"] = L.stored;
                         d["min_sphere_margin_ev"] = L.min_sphere_margin_ev;
                         d["notes"] = L.notes;
                         return d;
                     })
        .def("kpoint",
             [](const prowl::WavecarReader& r, int spin, int k) {
                 const auto& ki = r.kpoint(spin, k);
                 nb::dict d;
                 d["nplw"] = ki.nplw;
                 d["k_frac"] = std::vector<double>(ki.k_frac.begin(), ki.k_frac.end());
                 d["eigenvalues"] = ki.eigenvalues_ev;
                 d["eigenvalues_imag"] = ki.eigenvalues_imag;
                 d["occupations"] = ki.occupations;
                 return d;
             })
        .def("read_block", [](const prowl::WavecarReader& r, int spin, int k, int first, int count) {
            return block_to_dict(r.read_block(spin, k, first, count));
        });

    // -------------------------------------------------------------- structure
    nb::class_<prowl::Structure>(m, "Structure")
        .def_ro("comment", &prowl::Structure::comment)
        .def_ro("species", &prowl::Structure::species)
        .def_ro("counts", &prowl::Structure::counts)
        .def_ro("atom_type", &prowl::Structure::atom_type)
        .def_prop_ro("lattice", [](const prowl::Structure& s) { return lattice_matrix(s.lattice); })
        .def_prop_ro("volume", [](const prowl::Structure& s) { return s.lattice.volume; })
        .def_prop_ro("frac",
                     [](const prowl::Structure& s) {
                         Eigen::MatrixXd f(s.natoms(), 3);
                         for (int i = 0; i < s.natoms(); ++i)
                             for (int c = 0; c < 3; ++c) f(i, c) = s.frac[i][c];
                         return f;
                     })
        .def_prop_ro("natoms", &prowl::Structure::natoms)
        .def("element", &prowl::Structure::element);
    m.def("read_poscar", [](const std::string& p) { return prowl::read_poscar(p); });

    // ----------------------------------------------------------------- POTCAR
    nb::class_<prowl::PawChannel>(m, "PawChannel")
        .def_ro("l", &prowl::PawChannel::l)
        .def_ro("projector_q", &prowl::PawChannel::projector_q)
        .def_ro("u_ae", &prowl::PawChannel::u_ae)
        .def_ro("u_ps", &prowl::PawChannel::u_ps);
    nb::class_<prowl::PawDataset>(m, "PawDataset")
        .def_ro("titel", &prowl::PawDataset::titel)
        .def_ro("symbol", &prowl::PawDataset::symbol)
        .def_ro("element", &prowl::PawDataset::element)
        .def_ro("valence_config", &prowl::PawDataset::valence_config)
        .def_ro("sha256", &prowl::PawDataset::sha256)
        .def_ro("zval", &prowl::PawDataset::zval)
        .def_ro("psmaxn", &prowl::PawDataset::psmaxn)
        .def_ro("grid", &prowl::PawDataset::grid)
        .def_ro("channels", &prowl::PawDataset::channels)
        .def_ro("qpaw", &prowl::PawDataset::qpaw)
        .def_prop_ro("nprojectors", &prowl::PawDataset::nprojectors);
    m.def("read_potcar", [](const std::string& p) { return prowl::read_potcar(p); });

    // ------------------------------------------------------------------ basis
    nb::class_<prowl::RadialSto>(m, "RadialSto")
        .def_ro("element", &prowl::RadialSto::element)
        .def_ro("label", &prowl::RadialSto::label)
        .def_ro("n", &prowl::RadialSto::n)
        .def_ro("l", &prowl::RadialSto::l)
        .def_ro("occupation", &prowl::RadialSto::occupation)
        .def_ro("energy_ev", &prowl::RadialSto::energy_ev)
        .def("value", &prowl::RadialSto::value)
        .def("fourier", &prowl::RadialSto::fourier)
        .def("norm2", &prowl::RadialSto::norm2)
        .def("extent", &prowl::RadialSto::extent, "threshold"_a = 1e-8)
        .def("values", [](const prowl::RadialSto& r, const std::vector<double>& x) {
            std::vector<double> v(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) v[i] = r.value(x[i]);
            return v;
        })
        .def("fourier_values", [](const prowl::RadialSto& r, const std::vector<double>& q) {
            std::vector<double> v(q.size());
            for (std::size_t i = 0; i < q.size(); ++i) v[i] = r.fourier(q[i]);
            return v;
        });
    m.def("sto_bessel_integral", [](int N, int l, double zeta, double q) {
        prowl::require(l >= 0 && l <= 15 && N >= l + 2 && std::isfinite(zeta) && zeta > 0 &&
                            std::isfinite(q) && q >= 0, prowl::ErrorCode::InvalidArgument, "invalid STO integral arguments");
        return prowl::sto_bessel_integral(N, l, zeta, q);
    }, "N"_a, "l"_a, "zeta"_a, "q"_a);

    nb::class_<prowl::StoLibrary>(m, "StoLibrary")
        .def_static("load", [](const std::string& p) { return prowl::StoLibrary::load(p); })
        .def_static("load_default", &prowl::StoLibrary::load_default)
        .def_prop_ro("name", [](const prowl::StoLibrary& l) { return l.info().name; })
        .def_prop_ro("provenance_json", [](const prowl::StoLibrary& l) { return l.info().provenance_json; })
        .def("has_element", &prowl::StoLibrary::has_element)
        .def("shells", &prowl::StoLibrary::shells)
        .def("shell", &prowl::StoLibrary::shell)
        .def("merge", &prowl::StoLibrary::merge, "extension"_a);

    nb::class_<prowl::OrbitalSpec>(m, "OrbitalSpec")
        .def_ro("atom", &prowl::OrbitalSpec::atom)
        .def_ro("radial", &prowl::OrbitalSpec::radial)
        .def_ro("l", &prowl::OrbitalSpec::l)
        .def_ro("m", &prowl::OrbitalSpec::m)
        .def_ro("label", &prowl::OrbitalSpec::label);
    nb::class_<prowl::AtomicBasis>(m, "AtomicBasis")
        .def(nb::init<const prowl::Structure&, const prowl::StoLibrary&,
                      const std::map<std::string, std::vector<std::string>>&>(),
             "structure"_a, "library"_a, "selection"_a)
        .def_prop_ro("orbitals", &prowl::AtomicBasis::orbitals)
        .def_prop_ro("radials", &prowl::AtomicBasis::radials)
        .def_prop_ro("size", &prowl::AtomicBasis::size);

    nb::class_<prowl::TwoCenterOverlap>(m, "TwoCenterOverlap")
        .def(nb::init<const prowl::RadialSto&, const prowl::RadialSto&, double, int, double, double>(), "a"_a, "b"_a,
             "qmax"_a = 200.0, "nq"_a = 8001, "dd"_a = 0.0, "extent_threshold"_a = 1e-9)
        .def("evaluate",
             [](const prowl::TwoCenterOverlap& t, const std::vector<double>& d) {
                 Eigen::MatrixXd o(2 * t.l1() + 1, 2 * t.l2() + 1);
                 std::vector<double> buf(o.size());
                 t.evaluate(to_vec3(d), buf.data());
                 for (int i = 0; i < o.rows(); ++i)
                     for (int j = 0; j < o.cols(); ++j) o(i, j) = buf[i * o.cols() + j];
                 return o;
             })
        .def_prop_ro("cutoff", &prowl::TwoCenterOverlap::cutoff);

    m.def("gvectors_full", [](const Eigen::Matrix3d& A, const std::vector<double>& k, double encut) {
        prowl::Mat3 a{};
        for (int i = 0; i < 3; ++i) for (int c = 0; c < 3; ++c) a[i][c] = A(i, c);
        auto G = prowl::generate_gvectors_full(prowl::Lattice(a), to_vec3(k), encut);
        Eigen::MatrixXi out(static_cast<Eigen::Index>(G.size()), 3);
        for (std::size_t i = 0; i < G.size(); ++i) for (int c = 0; c < 3; ++c) out(static_cast<Eigen::Index>(i), c) = G[i][c];
        return out;
    }, "lattice"_a, "k_frac"_a, "encut_ev"_a);
    m.def("gvectors_gamma_half", [](const Eigen::Matrix3d& A, double encut) {
        prowl::Mat3 a{};
        for (int i = 0; i < 3; ++i) for (int c = 0; c < 3; ++c) a[i][c] = A(i, c);
        auto G = prowl::generate_gvectors_gamma_half(prowl::Lattice(a), encut);
        Eigen::MatrixXi out(static_cast<Eigen::Index>(G.size()), 3);
        for (std::size_t i = 0; i < G.size(); ++i) for (int c = 0; c < 3; ++c) out(static_cast<Eigen::Index>(i), c) = G[i][c];
        return out;
    }, "lattice"_a, "encut_ev"_a);
    m.def("real_ylm", [](int lmax, const std::vector<double>& v) {
        prowl::require(lmax >= 0 && lmax <= prowl::ylm::lmax_supported, prowl::ErrorCode::InvalidArgument,
                        "harmonic degree out of range");
        std::vector<double> out(prowl::ylm::count(lmax));
        prowl::ylm::real_ylm(lmax, to_vec3(v), out.data());
        return out;
    });
    m.def("gaunt", &prowl::ylm::gaunt);
    m.def("sph_bessel", [](int l, double x) {
        prowl::require(l >= 0 && l <= 12 && std::isfinite(x) && x >= 0, prowl::ErrorCode::InvalidArgument,
                        "invalid spherical Bessel arguments");
        return prowl::radial::sph_bessel(l, x);
    });

    // --------------------------------------------------------------- backends
    nb::class_<prowl::Backend>(m, "Backend")
        .def_prop_ro("name", &prowl::Backend::name)
        .def_prop_ro("device_description", &prowl::Backend::device_description)
        .def("set_factor_memory_limit", [](prowl::Backend& b, double bytes) { b.set_factor_memory_limit(to_byte_count(bytes)); })
        .def_prop_ro("factor_memory_stats", [](const prowl::Backend& b) {
            const auto s = b.factor_memory_stats();
            nb::dict d;
            d["limit_bytes"] = s.limit_bytes;
            d["retained_bytes"] = s.retained_bytes;
            d["peak_bytes"] = s.peak_bytes;
            return d;
        })
        .def("project_full",
             [](prowl::Backend& b, const Eigen::MatrixXcd& W, const Eigen::MatrixXcd& C) {
                 if (W.rows() != C.rows()) throw prowl::Error(prowl::ErrorCode::InvalidArgument, "ng mismatch");
                 Eigen::MatrixXcd out(W.cols(), C.cols());
                 b.project_full(W.data(), C.data(), static_cast<int>(W.rows()), static_cast<int>(W.cols()),
                                static_cast<int>(C.cols()), out.data());
                 return out;
             })
        .def_prop_ro("has_hermitian_eigensolver", &prowl::Backend::has_hermitian_eigensolver)
        .def("hermitian_eig", [](prowl::Backend& b, const Eigen::MatrixXcd& A) {
            prowl::require(A.rows() > 0 && A.rows() == A.cols() && A.allFinite(), prowl::ErrorCode::InvalidArgument,
                            "eigensolver matrix must be nonempty, square and finite");
            Eigen::MatrixXcd V = A;
            Eigen::VectorXd w(A.rows());
            b.hermitian_eig(V.data(), static_cast<int>(A.rows()), w.data());
            return nb::make_tuple(w, V);
        })
        .def("project_gamma", [](prowl::Backend& b, const Eigen::MatrixXcd& W, const Eigen::MatrixXcd& C,
                                 const std::vector<double>& w) {
            if (W.rows() != C.rows() || static_cast<std::size_t>(W.rows()) != w.size())
                throw prowl::Error(prowl::ErrorCode::InvalidArgument, "ng mismatch");
            Eigen::MatrixXd out(W.cols(), C.cols());
            b.project_gamma(W.data(), C.data(), w.data(), static_cast<int>(W.rows()), static_cast<int>(W.cols()),
                            static_cast<int>(C.cols()), out.data());
            return out;
        });
    m.def("make_backend", &prowl::make_backend, "kind"_a = "cpu");
    m.def("available_backends", &prowl::available_backends);

    // ------------------------------------------------------------- projection
    nb::class_<prowl::ProjectionOptions>(m, "ProjectionOptions")
        .def(nb::init<>())
        .def_rw("include_offsite_paw", &prowl::ProjectionOptions::include_offsite_paw)
        .def_rw("overlap_qmax", &prowl::ProjectionOptions::overlap_qmax)
        .def_rw("overlap_nq", &prowl::ProjectionOptions::overlap_nq)
        .def_rw("overlap_table_dd", &prowl::ProjectionOptions::overlap_table_dd)
        .def_rw("overlap_extent_threshold", &prowl::ProjectionOptions::overlap_extent_threshold)
        .def_rw("offsite_threshold", &prowl::ProjectionOptions::offsite_threshold)
        .def_rw("angular_ntheta", &prowl::ProjectionOptions::angular_ntheta)
        .def_rw("angular_nphi", &prowl::ProjectionOptions::angular_nphi)
        .def_rw("coupling_table_dd", &prowl::ProjectionOptions::coupling_table_dd)
        .def_rw("coupling_table_rmin_factor", &prowl::ProjectionOptions::coupling_table_rmin_factor);

    nb::class_<prowl::ProjectionModel>(m, "ProjectionModel")
        .def(nb::init<const prowl::Structure&, const prowl::AtomicBasis&, std::vector<prowl::PawDataset>,
                      prowl::ProjectionOptions>(),
             "structure"_a, "basis"_a, "datasets"_a, "options"_a = prowl::ProjectionOptions{})
        .def_prop_ro("norb", &prowl::ProjectionModel::norb)
        .def_prop_ro("nproj", &prowl::ProjectionModel::nproj)
        .def_prop_ro("foldable", &prowl::ProjectionModel::foldable,
                     "PAW coupling foldable into the orbital factors (one-centre PAW terms only)")
        .def_prop_ro("paw_q", &prowl::ProjectionModel::paw_q)
        .def_prop_ro("n_overlap_terms", [](const prowl::ProjectionModel& p) { return p.overlap_terms().size(); })
        .def_prop_ro("n_coupling_terms", [](const prowl::ProjectionModel& p) { return p.coupling_terms().size(); })
        .def_prop_ro("projector_labels",
                     [](const prowl::ProjectionModel& p) {
                         std::vector<std::string> v;
                         for (const auto& s : p.projectors())
                             v.push_back(std::to_string(s.atom + 1) + ":ch" + std::to_string(s.channel) + ":" +
                                         prowl::ylm::orbital_name(s.l, s.m));
                         return v;
                     })
        .def("overlap", [](const prowl::ProjectionModel& p, const std::vector<double>& k) { return p.overlap(to_vec3(k)); })
        .def_prop_ro("setup_timings", &prowl::ProjectionModel::setup_timings)
        .def("overlaps",
             [](const prowl::ProjectionModel& p, const std::vector<std::vector<double>>& ks) {
                 std::vector<prowl::Vec3> kv;
                 for (const auto& k : ks) kv.push_back(to_vec3(k));
                 std::vector<Eigen::MatrixXcd> out;
                 {
                     nb::gil_scoped_release release;
                     out = p.overlaps(kv);
                 }
                 return out;
             },
             "k_frac"_a, "S(k) for a list of k-points (OpenMP over k)")
        .def("overlap_terms",
             [](const prowl::ProjectionModel& p) {
                 // real-space blocks <chi_A(r - tau_A)|chi_B(r - tau_B - R)> as (A, B, (R1,R2,R3), block)
                 nb::list out;
                 for (const auto& t : p.overlap_terms())
                     out.append(nb::make_tuple(t.row_atom, t.col_atom, nb::make_tuple(t.R[0], t.R[1], t.R[2]), t.block));
                 return out;
             })
        .def("overlap_blocks",
             [](const prowl::ProjectionModel& p, const std::vector<std::tuple<int, int, int, int, int>>& keys) {
                 // only the requested (A, B, R1, R2, R3) real-space overlap blocks
                 std::map<std::tuple<int, int, int, int, int>, const prowl::LatticeTerm*> idx;
                 for (const auto& t : p.overlap_terms())
                     idx[{t.row_atom, t.col_atom, t.R[0], t.R[1], t.R[2]}] = &t;
                 nb::list out;
                 for (const auto& k : keys) {
                     auto it = idx.find(k);
                     if (it == idx.end()) continue;
                     const auto& t = *it->second;
                     out.append(nb::make_tuple(t.row_atom, t.col_atom, nb::make_tuple(t.R[0], t.R[1], t.R[2]), t.block));
                 }
                 return out;
             })
        .def_prop_ro("orbital_offsets",
                     [](const prowl::ProjectionModel& p) {
                         std::vector<int> off(p.structure().natoms() + 1, 0);
                         for (const auto& o : p.basis().orbitals()) off[o.atom + 1]++;
                         for (std::size_t i = 1; i < off.size(); ++i) off[i] += off[i - 1];
                         return off;
                     })
        .def("paw_coupling",
             [](const prowl::ProjectionModel& p, const std::vector<double>& k) { return p.paw_coupling(to_vec3(k)); })
        .def("orbital_factors",
             [](const prowl::ProjectionModel& p, const prowl::WavecarReader& r, int k) {
                 return p.orbital_factors(r.kpoint(0, k).k_frac, *r.gvectors(k));
             })
        .def("factors_on_backend",
             [](const prowl::ProjectionModel& p, const prowl::WavecarReader& r, int k, bool projectors,
                prowl::Backend& backend) {
                 return p.factors_on_backend(r.kpoint(0, k).k_frac, *r.gvectors(k), projectors, backend);
             })
        .def(
            "project",
            [](const prowl::ProjectionModel& p, const prowl::WavecarReader& r, int spin, int k, int first, int count,
               prowl::Backend& backend) {
                prowl::KProjection kp;
                {
                    nb::gil_scoped_release rel;
                    auto blk = r.read_block(spin, k, first, count);
                    kp = p.project(blk, backend);
                }
                nb::dict d;
                d["a_soft"] = kp.a_soft;
                d["b"] = kp.b;
                d["a_ae"] = kp.a_ae;
                d["pw_norm"] = kp.pw_norm;
                d["paw_norm"] = kp.paw_norm;
                d["real_gauge"] = kp.real_gauge;
                return d;
            },
            "reader"_a, "spin"_a, "k"_a, "first_band"_a, "count"_a, "backend"_a)
        .def(
            "project_k",
            [](const prowl::ProjectionModel& p, const prowl::WavecarReader& r, int k, int first, int count,
               prowl::Backend& backend, int block_bands, double factor_budget_bytes, int paw_check_stride) {
                std::vector<prowl::KProjection> kps;
                {
                    nb::gil_scoped_release rel;
                    kps = p.project_k(r, k, first, count, backend, block_bands,
                                      to_byte_count(factor_budget_bytes), paw_check_stride);
                }
                nb::list out;
                for (auto& kp : kps) {
                    nb::dict d;
                    // folded projection: no soft amplitudes; b only for b_bands
                    if (kp.paw_bands.empty())
                        d["a_soft"] = kp.a_soft;
                    else
                        d["a_soft"] = nb::none();
                    d["b"] = kp.b;
                    d["b_bands"] = b_bands(kp);
                    d["a_ae"] = kp.a_ae;
                    d["pw_norm"] = kp.pw_norm;
                    d["paw_norm"] = kp.paw_norm;
                    d["real_gauge"] = kp.real_gauge;
                    d["factor_peak_bytes"] = kp.factor_peak_bytes;
                    out.append(d);
                }
                return out;
            },
            "reader"_a, "k"_a, "first_band"_a, "count"_a, "backend"_a, "block_bands"_a = 0,
            "factor_budget_bytes"_a = 2147483648.0, "paw_check_stride"_a = 1,
            "Per spin {a_soft, b, b_bands, a_ae, pw_norm, paw_norm, real_gauge}; b and paw_norm hold the bands "
            "b_bands (all bands for paw_check_stride 1). paw_check_stride > 1 with a foldable model: folded "
            "projection, a_soft None and b_bands the sample 0, s, 2s, ... and the last band (projection.hpp)")
        .def(
            "project_many",
            [](const prowl::ProjectionModel& p, const prowl::WavecarReader& r, const std::vector<int>& ks, int first,
               int count, int block_bands, double factor_budget_bytes, int paw_check_stride) {
                std::vector<std::vector<prowl::KProjection>> all;
                {
                    nb::gil_scoped_release rel;
                    all = p.project_many(r, ks, first, count, block_bands, to_byte_count(factor_budget_bytes),
                                         paw_check_stride);
                }
                nb::list out;
                for (auto& kps : all) {
                    nb::list per;
                    for (auto& kp : kps) {
                        nb::dict d;
                        d["a_ae"] = kp.a_ae;
                        d["paw_norm"] = kp.paw_norm;
                        d["b_bands"] = b_bands(kp);
                        d["factor_peak_bytes"] = kp.factor_peak_bytes;
                        per.append(d);
                    }
                    out.append(per);
                }
                return out;
            },
            "reader"_a, "ks"_a, "first_band"_a, "count"_a, "block_bands"_a = 0, "factor_budget_bytes"_a = 2147483648.0,
            "paw_check_stride"_a = 1,
            "project_k for many k-points (CPU, OpenMP over k); per k a list of {a_ae, paw_norm, b_bands} per spin (paw_norm of the bands b_bands)");

    // ------------------------------------------------------- reconstruction / analysis (C++)
    m.def("reconstruct_k", [](const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, const Eigen::VectorXd& eps,
                              double thr) {
        prowl::ReconstructionK r;
        {
            nb::gil_scoped_release release;
            r = prowl::reconstruct_k(T, S, eps, thr);
        }
        nb::dict d;
        d["C"] = r.C; d["Cn"] = r.Cn; d["H"] = r.H; d["spill"] = r.spill; d["s_eig"] = r.s_eig;
        d["completeness"] = r.completeness; d["band_residual"] = r.band_residual; d["rank"] = r.rank;
        return d;
    }, "T"_a, "S"_a, "eps"_a, "overlap_threshold"_a = 1e-8);
    m.def("window_conditioning", &prowl::window_conditioning, "T"_a, "S"_a, "nwin"_a, "overlap_threshold"_a = 1e-8,
          nb::call_guard<nb::gil_scoped_release>());
    m.def("cut_at_gap", &prowl::cut_at_gap, "eps"_a, "n"_a, "tol"_a = 1e-4);
    m.def("frozen_window_subspace", [](const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, const Eigen::VectorXd& eps,
                                       const Eigen::VectorXd& occ, int norb, int ncand) {
        prowl::FrozenWindow r;
        {
            nb::gil_scoped_release release;
            r = prowl::frozen_window_subspace(T, S, eps, occ, norb, ncand);
        }
        return nb::make_tuple(r.T, r.eps, r.occ, r.weakest, r.gap);
    }, "T"_a, "S"_a, "eps"_a, "occ"_a, "norb"_a, "ncand"_a = -1);
    m.def("repaired_windows",
          [](const std::vector<std::vector<Eigen::MatrixXcd>>& T, const std::vector<Eigen::MatrixXcd>& S,
             const std::vector<std::vector<Eigen::VectorXd>>& eps, const std::vector<std::vector<Eigen::VectorXd>>& occ,
             int norb, int ncand, double cond_threshold, double gap_threshold, bool reference, bool diagnostics,
             prowl::Backend* device) {
              std::vector<prowl::RepairedWindow> r;
              {
                  nb::gil_scoped_release release;
                  r = prowl::repaired_windows(T, S, eps, occ, norb, ncand, cond_threshold, gap_threshold, reference, device);
              }
              nb::list out;
              for (auto& w : r) {
                  if (diagnostics)
                      out.append(nb::make_tuple(w.T, w.eps, w.occ, w.cond, w.repaired_nc, w.failed_at, w.steps_scanned,
                                                w.steps_reference, w.seconds_setup, w.seconds_bound,
                                                w.seconds_scan, w.seconds_reference));
                  else
                      out.append(nb::make_tuple(w.T, w.eps, w.occ, w.cond, w.repaired_nc, w.failed_at));
              }
              return out;
          },
          "T"_a, "S"_a, "eps"_a, "occ"_a, "norb"_a, "ncand"_a, "cond_threshold"_a, "gap_threshold"_a,
          "reference"_a = false, "diagnostics"_a = false, "device"_a.none() = nb::none(),
          "repaired_window per (spin, k) in parallel; list over spin-major (s, k) of (T, eps, occ, cond, nc, failed_at)"
          " [+ (steps_scanned, steps_reference, seconds_setup, seconds_bound, seconds_scan, seconds_reference)"
          " with diagnostics]. reference: original linear scan; device: backend whose repair stepper (GPU)"
          " replaces the CPU scan for long searches");
    m.def("analyze", [](const Eigen::MatrixXd& kfrac, const std::vector<double>& weights,
                        const std::vector<std::vector<Eigen::MatrixXcd>>& T, const std::vector<Eigen::MatrixXcd>& S,
                        const std::vector<std::vector<Eigen::VectorXd>>& eps,
                        const std::vector<std::vector<Eigen::VectorXd>>& occ, const std::vector<int>& orb_offsets,
                        double spin_factor, nb::list bond_list, std::vector<double> grid, double overlap_threshold,
                        prowl::Backend* eig_backend, const Eigen::MatrixXi& kgrid, std::vector<int> mesh,
                        const Eigen::Matrix3d& recip) {
        prowl::require(kfrac.cols() == 3 && kgrid.cols() == 3 && mesh.size() == 3,
                        prowl::ErrorCode::InvalidArgument, "k coordinates and mesh must have three components");
        prowl::AnalysisInputs in;
        for (Eigen::Index i = 0; i < kfrac.rows(); ++i) in.kfrac.push_back({kfrac(i, 0), kfrac(i, 1), kfrac(i, 2)});
        in.weights = weights; in.T = T; in.S = S; in.eps = eps; in.occ = occ; in.orb_offsets = orb_offsets;
        in.spin_factor = spin_factor; in.overlap_threshold = overlap_threshold; in.eig_backend = eig_backend;
        std::vector<prowl::BondSpec> bonds;
        for (nb::handle h : bond_list) {
            nb::tuple t = nb::cast<nb::tuple>(h);
            prowl::require(t.size() == 4 || t.size() == 8, prowl::ErrorCode::InvalidArgument,
                            "bond tuple must have four or eight entries");
            prowl::BondSpec b;
            b.atom_a = nb::cast<int>(t[0]);
            b.atom_b = nb::cast<int>(t[1]);
            auto R = nb::cast<std::vector<int>>(t[2]);
            prowl::require(R.size() == 3, prowl::ErrorCode::InvalidArgument, "bond translation must have three components");
            b.R = {R.at(0), R.at(1), R.at(2)};
            if (!t[3].is_none()) b.S_R = nb::cast<Eigen::MatrixXd>(t[3]);
            if (t.size() >= 8 && !t[4].is_none()) { // bond frame: rot_a, rot_b, |m|_a, |m|_b
                b.rot_a = nb::cast<Eigen::MatrixXd>(t[4]);
                b.rot_b = nb::cast<Eigen::MatrixXd>(t[5]);
                b.absm_a = nb::cast<std::vector<int>>(t[6]);
                b.absm_b = nb::cast<std::vector<int>>(t[7]);
            }
            bonds.push_back(std::move(b));
        }
        if (grid.size() != 7)
            throw prowl::Error(prowl::ErrorCode::InvalidArgument, "grid = (emin, emax, npts, sigma, reference, kind, bin_offset)");
        prowl::require(std::isfinite(grid[2]) && grid[2] >= 1 && grid[2] <= std::numeric_limits<int>::max() &&
                            std::floor(grid[2]) == grid[2] && (grid[5] == 0 || grid[5] == 1),
                        prowl::ErrorCode::InvalidArgument, "invalid spectrum point count or kind");
        prowl::SpectrumGrid gr{grid[0], grid[1], static_cast<int>(grid[2]), grid[3], grid[4], static_cast<int>(grid[5]), grid[6]};
        for (Eigen::Index i = 0; i < kgrid.rows(); ++i) in.kgrid.push_back({kgrid(i, 0), kgrid(i, 1), kgrid(i, 2)});
        if (mesh.size() == 3) in.mesh = {mesh[0], mesh[1], mesh[2]};
        for (int i = 0; i < 3; ++i) for (int c = 0; c < 3; ++c) in.recip[i][c] = recip(i, c);
        prowl::AnalysisOutputs o;
        {
            nb::gil_scoped_release rel;
            o = prowl::analyze(in, bonds, gr);
        }
        const std::size_t ns = o.nspin, nbd = o.nbond, no = o.norb, np = o.npts;
        nb::dict d;
        d["cohp"] = to_numpy(o.cohp, {ns, nbd, np});
        d["coop"] = to_numpy(o.coop, {ns, nbd, np});
        d["icohp"] = to_numpy(o.icohp, {ns, nbd});
        d["icoop"] = to_numpy(o.icoop, {ns, nbd});
        d["cobi"] = to_numpy(o.cobi, {ns, nbd, np});
        d["icobi"] = to_numpy(o.icobi, {ns, nbd});
        d["pdos"] = to_numpy(o.pdos, {ns, no, np});
        d["pdos_integral"] = to_numpy(o.pdos_integral, {ns, no});
        d["mulliken_gp"] = to_numpy(o.mulliken_gp, {ns, no});
        d["lowdin_gp"] = to_numpy(o.lowdin_gp, {ns, no});
        d["band_energy"] = o.band_energy;
        d["icohp_orbital"] = o.icohp_orbital;
        const std::size_t nch = prowl::n_bond_channels;
        d["cohp_channel"] = to_numpy(o.cohp_channel, {ns, nbd, nch, np});
        d["icohp_channel"] = to_numpy(o.icohp_channel, {ns, nbd, nch});
        d["abs_charge_spilling"] = o.abs_charge_spilling;
        d["abs_total_spilling"] = o.abs_total_spilling;
        d["max_reconstruction_residual"] = o.max_reconstruction_residual;
        d["min_overlap_eigenvalue"] = o.min_overlap_eigenvalue;
        d["electrons_dft"] = o.electrons_dft;
        d["used_device_eigensolver"] = o.used_device_eigensolver;
        return d;
    }, "kfrac"_a, "weights"_a, "T"_a, "S"_a, "eps"_a, "occ"_a, "orb_offsets"_a, "spin_factor"_a, "bonds"_a, "grid"_a,
       "overlap_threshold"_a = 1e-8, "eig_backend"_a.none() = nb::none(), "kgrid"_a = Eigen::MatrixXi(0, 3),
       "mesh"_a = std::vector<int>{1, 1, 1}, "recip"_a = Eigen::Matrix3d::Identity());
    m.def("real_harmonic_rotation", [](int l, const Eigen::Matrix3d& axes) {
        prowl::Mat3 a;
        for (int i = 0; i < 3; ++i) for (int c = 0; c < 3; ++c) a[i][c] = axes(i, c);
        return prowl::ylm::rotation_matrix(l, a);
    }, "l"_a, "axes"_a,
       "D with Y_lm(axes v) = D Y_l(v): orbitals of the frame whose x', y', z' are the rows of axes, in the old ones");
    m.def("tetra_weights", [](std::vector<double> e, double E, double vt) {
        prowl::require(e.size() == 4 && std::all_of(e.begin(), e.end(), [](double x) { return std::isfinite(x); }) &&
                            std::is_sorted(e.begin(), e.end()) && std::isfinite(E) && std::isfinite(vt) && vt >= 0,
                        prowl::ErrorCode::InvalidArgument, "tetrahedron needs four sorted finite energies and a nonnegative volume");
        double w[4];
        prowl::tetra_weights(e.data(), E, vt, w);
        return std::vector<double>(w, w + 4);
    });
}
