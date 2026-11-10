// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file projection.hpp
/// PAW-consistent projection of VASP pseudo Bloch states onto localised STO
/// orbitals (docs/SCIENCE.md, sections 3-4).
///
/// Bloch convention (periodic gauge): chi_{mu k}(r) = sum_R e^{i k.R} chi_mu(r - tau_mu - R).
/// All "cell" inner products below integrate one localised function against a
/// Bloch function over all space, equivalently the Bloch sum over one cell.
///
///   A_soft_{mu n}(k) = <chi_mu k | psi~_nk>  = sum_G W_{G mu} c_{G n},
///       W_{G mu} = Omega^{-1/2} 4 pi i^l Y_lm(q^) Rt_mu(|q|) e^{i q.tau_mu},  q = k + G (Cartesian, 2 pi included)
///   B_{i n}(k)       = <p~_i k | psi~_nk>   (same form with 4 pi Rt -> PSPNL)
///   D_{mu i}(k)      = sum_R e^{i k.R} < chi_mu(r - tau_mu) | phi_i - phi~_i (r - tau_a - R) >
///   A_AE             = A_soft + D B
///   S_{mu nu}(k)     = sum_R e^{i k.R} < chi_mu(r - tau_mu) | chi_nu(r - tau_nu - R) >
///   Q_{ij}           = <phi_i|phi_j> - <phi~_i|phi~_j>   (same atom, same l,m)
/// PAW norm check: sum_G |c_G|^2 + B^H Q B = 1 for every band (or a sample of bands, below).
///
/// PAW fold (one-centre PAW, the Python application's default; C++ ProjectionOptions defaults to two-centre):
/// D couples an orbital only to projectors of its own atom with the same
/// (l, m), so the columns of W_orb + W_proj D^T are separable functions with the radial factor
/// 4 pi Rt_mu(q) + sum_i D_{mu i} PSPNL_i(q), and A_AE = (W_orb + W_proj D^T)^T c is one projection onto N_orb
/// columns instead of N_orb + N_proj. B is then computed for a deterministic sample of bands only (PAW norm check).

#include <complex>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "prowl/backend.hpp"
#include "prowl/basis.hpp"
#include "prowl/potcar.hpp"
#include "prowl/radial.hpp"
#include "prowl/separable.hpp"
#include "prowl/structure.hpp"
#include "prowl/wavecar.hpp"

namespace prowl {

struct ProjectionOptions {
    bool include_offsite_paw = true; ///< two-centre <chi_b|phi_i - phi~_i>_a terms
    double overlap_qmax = 200.0;     ///< 1/Angstrom, two-centre q-integration limit
    int overlap_nq = 8001;
    double overlap_table_dd = 0.01;       ///< distance spacing of splined J_L(d) tables (Angstrom; 4e-11 vs 0.002)
    double overlap_extent_threshold = 1e-6; ///< |r R(r)| threshold defining orbital extent for S(k) sums
    double offsite_threshold = 1e-6; ///< skip off-site PAW pairs where |r R(r)| on the sphere is below this
    int angular_ntheta = 16;         ///< single-centre expansion quadrature (Gauss-Legendre x uniform phi)
    int angular_nphi = 32;
    double coupling_table_dd = 0.02;         ///< distance spacing of off-site PAW coupling tables (Angstrom)
    double coupling_table_rmin_factor = 1.2; ///< tables used for d >= factor * r_PAW; direct quadrature below
};

struct ProjectorSpec {
    int atom{};
    int channel{}; ///< index into dataset channels
    int l{};
    int m{};
};

/// Result of projecting one wavefunction block.
struct KProjection {
    Eigen::MatrixXcd a_soft; ///< norb x nb, empty for a folded projection
    Eigen::MatrixXcd b;      ///< nproj x nb, or nproj x paw_bands.size() for a folded projection
    Eigen::MatrixXcd a_ae;   ///< norb x nb
    std::vector<double> pw_norm;
    std::vector<double> paw_norm;
    /// Bands of b and paw_norm (indices into the projected bands) when the PAW norm was checked on a sample
    /// (folded projection; a_soft is then empty). Empty: all bands, b and paw_norm indexed by band.
    std::vector<int> paw_bands;
    bool real_gauge{}; ///< gamma-half input: all matrices real
    /// Streamed project_k/project_many factor peak including cold packing; zero for project().
    std::size_t factor_peak_bytes{};
};

/// One lattice-summed block term: value(k) += exp(2 pi i k.R) * block.
struct LatticeTerm {
    int row_atom{};
    int col_atom{};
    IVec3 R{};
    Eigen::MatrixXd block; ///< rows: orbitals of row_atom; cols: functions of col_atom
};

class ProjectionModel {
public:
    /**
     * @brief Builds owning basis/PAW tables and real-space lattice terms for a structure.
     * @param[in] s Finite structure with a valid atom/species order; copied.
     * @param[in] basis Nonempty atom-ordered basis matching s; copied.
     * @param[in] datasets One valid PAW dataset per species, in species order; transferred into the model.
     * @param[in] opt Validated quadrature, screening and PAW options; copied.
     * Input arguments need only outlive construction. Cached tables are immutable except per-call k/G descriptors protected by the model mutex.
     * @throws Error For inconsistent structure/basis/datasets, invalid options or unsupported numerical ranges.
     */
    ProjectionModel(const Structure& s, const AtomicBasis& basis, std::vector<PawDataset> datasets,
                    ProjectionOptions opt = {});

    /**
     * @brief Counts local basis functions.
     * @return Number of angular orbitals in the cell.
     */
    [[nodiscard]] int norb() const noexcept { return basis_.size(); }
    /**
     * @brief Counts PAW magnetic projector functions.
     * @return Number of (atom,channel,m) projectors.
     */
    [[nodiscard]] int nproj() const noexcept { return static_cast<int>(projectors_.size()); }
    /**
     * @brief Borrows projector descriptors in projection-column order.
     * @return Immutable descriptor array, valid until model destruction.
     */
    [[nodiscard]] const std::vector<ProjectorSpec>& projectors() const noexcept { return projectors_; }
    /**
     * @brief Borrows the model-owned local basis.
     * @return Immutable basis reference, valid until model destruction.
     */
    [[nodiscard]] const AtomicBasis& basis() const noexcept { return basis_; }
    /**
     * @brief Borrows the model-owned construction-time structure.
     * @return Immutable structure reference, valid until model destruction.
     */
    [[nodiscard]] const Structure& structure() const noexcept { return s_; }
    /**
     * @brief Borrows the numerical and PAW choices used to build this model.
     * @return Immutable option reference, valid until model destruction.
     */
    [[nodiscard]] const ProjectionOptions& options() const noexcept { return opt_; }

    /**
     * @brief Sums real-space basis overlaps at one reciprocal point.
     * @param[in] k_frac Finite fractional k coordinates.
     * @return Owning norb x norb complex S(k), in the model orbital order.
     */
    [[nodiscard]] Eigen::MatrixXcd overlap(const Vec3& k_frac) const;      ///< S(k)
    /**
     * @brief Evaluates S(k) for a list of reciprocal points.
     * @param[in] k_frac Finite fractional k list; empty input produces an empty result.
     * @return Owning overlap matrices in input order; regular meshes may use an equivalent folded lattice sum.
     */
    [[nodiscard]] std::vector<Eigen::MatrixXcd> overlaps(const std::vector<Vec3>& k_frac) const;
    /**
     * @brief Sums real-space all-electron PAW couplings at a reciprocal point.
     * @param[in] k_frac Finite fractional k coordinates.
     * @return Owning norb x nproj complex D(k), with atom/image Bloch phases.
     */
    [[nodiscard]] Eigen::MatrixXcd paw_coupling(const Vec3& k_frac) const; ///< D(k)
    /**
     * @brief Borrows the block-diagonal PAW norm correction matrix.
     * @return Immutable nproj x nproj Q, valid until model destruction.
     */
    [[nodiscard]] const Eigen::MatrixXd& paw_q() const noexcept { return Q_; }
    /**
     * @brief Borrows real-space local-basis overlap blocks.
     * @return Immutable atom/image terms, valid until model destruction.
     */
    [[nodiscard]] const std::vector<LatticeTerm>& overlap_terms() const noexcept { return s_terms_; }
    /**
     * @brief Exposes constructor timings and table-size diagnostics.
     * @return Borrowed named values: seconds except overlap_terms_count; overlap_terms time is included in overlap_tables.
     */
    [[nodiscard]] const std::vector<std::pair<std::string, double>>& setup_timings() const noexcept {
        return setup_timings_;
    }
    /**
     * @brief Borrows the all-electron PAW correction blocks.
     * @return Immutable atom/image D terms, valid until model destruction.
     */
    [[nodiscard]] const std::vector<LatticeTerm>& coupling_terms() const noexcept { return d_terms_; }

    /**
     * @brief Generates host plane-wave factors for a local-orbital column range.
     * @param[in] k_frac Finite fractional k coordinates.
     * @param[in] G Borrowed Miller indices in WAVECAR ordering.
     * @param[in] first First orbital column in [0,norb()].
     * @param[in] n Column count, or a negative value for all remaining columns.
     * @return Owning G.size() x n column-major complex factors.
     * @throws Error For invalid columns or a plane wave outside the tabulated transform range.
     */
    [[nodiscard]] Eigen::MatrixXcd orbital_factors(const Vec3& k_frac, const std::vector<IVec3>& G, int first = 0,
                                                   int n = -1) const;
    /**
     * @brief Generates host plane-wave factors for a PAW-projector range.
     * @param[in] k_frac Finite fractional k coordinates.
     * @param[in] G Borrowed ordered Miller indices.
     * @param[in] first First projector column in [0,nproj()].
     * @param[in] n Column count, or a negative value for all remaining projectors.
     * @return Owning G.size() x n column-major complex factors.
     * @throws Error For invalid columns or wave numbers beyond the POTCAR table.
     */
    [[nodiscard]] Eigen::MatrixXcd projector_factors(const Vec3& k_frac, const std::vector<IVec3>& G, int first = 0,
                                                     int n = -1) const;

    /**
     * @brief Projects one caller-owned wavefunction block with full PAW diagnostics.
     * @param[in] blk Consistent logical coefficient block with shared G indices; spinors are unsupported.
     * @param[in,out] backend Borrowed compute context, used exclusively during this call.
     * @return Owning soft/all-electron overlaps, projector amplitudes and per-band norms.
     * @throws Error For invalid blocks, unsupported storage or an exceeded transform range.
     */
    [[nodiscard]] KProjection project(const WavefunctionBlock& blk, Backend& backend) const;

    /**
     * @brief Checks whether on-site PAW corrections can be absorbed into radial factors.
     * @return True for a populated fold spec with only same-atom, angular-diagonal couplings.
     */
    [[nodiscard]] bool foldable() const noexcept { return !fold_spec_.lm.empty(); }

    /**
     * @brief Evaluates factors through a selected backend for kernel parity checks.
     * @param[in] k_frac Finite fractional k coordinates.
     * @param[in] G Borrowed ordered Miller indices.
     * @param[in] projectors True selects PAW projectors; false selects orbitals.
     * @param[in,out] backend Borrowed exclusive backend used for evaluation.
     * @return Owning G.size() x selected-function-count complex factor matrix.
     */
    [[nodiscard]] Eigen::MatrixXcd factors_on_backend(const Vec3& k_frac, const std::vector<IVec3>& G, bool projectors,
                                                      Backend& backend) const;

    /**
     * @brief Projects a band range at one k-point for all spins, streaming coefficient and factor blocks.
     * @param[in] reader Borrowed WAVECAR reader; must outlive this call.
     * @param[in] k Zero-based k-point index.
     * @param[in] first Zero-based first band.
     * @param[in] count Number of consecutive bands.
     * @param[in,out] backend Borrowed compute context, used exclusively for this call.
     * @param[in] block_bands Maximum bands per coefficient block; 0 selects count (large blocks may be halved for prefetch).
     * @param[in] factor_budget_bytes Aggregate backend limit for cached and transient factor buffers, including
     * complex and packed-real copies. Insufficient space for one column raises OutOfMemory before generation.
     * Coefficients, results and numerical-library workspaces are accounted for separately by the planner.
     * @param[in] paw_check_stride 1 computes full PAW diagnostics; larger values sample relative band indices
     * 0, stride, ... and count-1 when foldable(), otherwise full diagnostics are computed.
     * @return One owning KProjection per spin; sampled B and PAW norms use paw_bands as their column map.
     *
     * The CPU may reuse cached factors across band blocks and spins; CUDA generates each requested block.
     * Calls using the model's mutable factor specs are serialized. Process-wide OpenBLAS policy leases permit
     * concurrent models using the same thread count; a different count waits until those computations finish.
     */
    [[nodiscard]] std::vector<KProjection> project_k(const WavecarReader& reader, int k, int first, int count,
                                                     Backend& backend, int block_bands = 0,
                                                     std::size_t factor_budget_bytes = std::size_t(1) << 31,
                                                     int paw_check_stride = 1) const;
    /**
     * @brief Projects a k batch on CPU with an aggregate factor-memory limit.
     * @param[in] reader Borrowed validated reader, alive for the entire call.
     * @param[in] ks Zero-based k indices in result order; repeats and empty lists are allowed.
     * @param[in] first Zero-based first band.
     * @param[in] count Positive band count within the file range.
     * @param[in] block_bands Maximum coefficient-block width; zero selects the requested count.
     * @param[in] factor_budget_bytes Aggregate owned-factor limit divided among active k workers; cold packing and retained caches count together.
     * @param[in] paw_check_stride Positive sample stride; one computes every PAW diagnostic, larger values sample 0,stride,...,last when foldable.
     * @return Owning [requested-k][spin] projections; sampled B/norm columns are identified by paw_bands.
     * Workers are reduced when the limit cannot fit one column per worker. Coefficients/results/library overhead are separate planner entries.
     * @throws Error For invalid indices/ranges/stride, a factor limit below one column, or numerical/I/O failures.
     */
    [[nodiscard]] std::vector<std::vector<KProjection>> project_many(const WavecarReader& reader,
                                                                     const std::vector<int>& ks, int first, int count,
                                                                     int block_bands = 0,
                                                                     std::size_t factor_budget_bytes = std::size_t(1)
                                                                                                       << 31,
                                                                     int paw_check_stride = 1) const;

    /**
     * @brief Finds the wave-number range common to every projector dataset.
     * @return Largest supported |q| in inverse Angstrom, limited by the final stored projector sample.
     */
    [[nodiscard]] double projector_qmax() const noexcept;

private:
    std::vector<KProjection> project_k_with(const WavecarReader& reader, int k, int first, int count, Backend& backend,
                                            int block_bands, std::size_t factor_budget_bytes,
                                            SeparableFactorSpec& orb, SeparableFactorSpec& proj,
                                            SeparableFactorSpec& fold, int paw_check_stride) const;
    Structure s_;
    AtomicBasis basis_;
    std::vector<PawDataset> datasets_; ///< per species
    ProjectionOptions opt_;
    std::vector<ProjectorSpec> projectors_;
    std::vector<int> orb_offset_, proj_offset_; ///< per atom, plus sentinel
    std::vector<radial::CubicSpline> orbital_ft_;   ///< per radial: 4 pi Rt(q) table
    std::vector<std::vector<radial::CubicSpline>> projector_ft_; ///< per species, per channel: PSPNL(q)
    double orbital_qmax_{};
    Eigen::MatrixXd Q_;
    std::vector<LatticeTerm> s_terms_, d_terms_;
    std::vector<std::pair<std::string, double>> setup_timings_; ///< seconds per constructor step

    void build_projectors();
    void build_tables();
    void build_overlap_terms();
    /// a_ae = a_soft + D(k) b and paw_norm = pw_norm + diag(b^H Q b), using the block structure of D (lattice terms)
    /// and Q (block-diagonal by atom) instead of dense products.
    void finish_paw(const Vec3& k, KProjection& kp) const;
    /// diag(b^H Q b) for the columns of b (Q block-diagonal by atom).
    [[nodiscard]] Eigen::VectorXd paw_q_norms(const Eigen::MatrixXcd& b) const;
    /// Folded factors: columns [0, norb) = orbitals with the one-centre PAW coupling in the radial factor, columns
    /// [norb, norb + nproj) = the projectors. Left empty if D has two-centre terms.
    void build_fold_spec();
    void build_coupling_terms();
    [[nodiscard]] Eigen::MatrixXcd lattice_sum(const std::vector<LatticeTerm>& terms, const Vec3& k, int nrows,
                                               int ncols, const std::vector<int>& row_off,
                                               const std::vector<int>& col_off) const;
    // Persistent separable-factor descriptions (functions + radial tables); k/G are set per call
    // under spec_mutex_.
    mutable SeparableFactorSpec orb_spec_, proj_spec_, fold_spec_;
    mutable std::mutex spec_mutex_;
    void build_specs();
    void check_q_range(const Vec3& k, const std::vector<IVec3>& G, bool projectors) const;
};

} // namespace prowl
