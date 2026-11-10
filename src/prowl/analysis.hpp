// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file analysis.hpp
/// Reconstruction (Löwdin orthonormalisation, re-orthonormalised projected bands, local
/// Hamiltonian) and bonding analyses (COHP/COOP/pDOS/populations). C++ implementation of
/// docs/SCIENCE.md sections 5-6; the NumPy implementation in python/prowl/analysis.py is the
/// independent reference and both are required to agree (tests/python/test_engines.py).

#include <complex>
#include <vector>

#include <Eigen/Dense>

#include "prowl/lattice.hpp"

namespace prowl {
class Backend;
}

namespace prowl {

struct ReconstructionK {
    Eigen::MatrixXcd C;      ///< norb x nb, orthonormal-basis coefficients (polar form)
    Eigen::MatrixXcd Cn;     ///< norb x nb, nonorthogonal-basis coefficients X C
    Eigen::MatrixXcd H;      ///< norb x norb, Löwdin-basis Hamiltonian C diag(eps) C^H
    Eigen::VectorXd spill;   ///< 1 - (Tp^H Tp)_nn
    Eigen::VectorXd s_eig;   ///< eigenvalues of S
    Eigen::VectorXd completeness; ///< diag(Tp Tp^H)
    /// Rectangular window: (C^H H C)_nn - eps_n (C^H C)_nn. Square window: per-column maximum absolute
    /// error of C^H H C - diag(eps), including off-diagonal elements (zero only up to rounding).
    Eigen::VectorXd band_residual;
    int rank{};
};

/**
 * @brief Computes a power on the supported eigenspace of a Hermitian matrix.
 * @param[in] A Nonempty finite square matrix; its Hermitian part is used.
 * @param[in] p Finite exponent.
 * @param[in] threshold Relative eigenvalue cutoff in [0,1); nonpositive/dropped modes contribute zero.
 * @param[out] rank Optional supported-mode count; nullptr omits it.
 * @param[out] eigenvalues Optional ascending eigenvalues; nullptr omits them.
 * @param[in,out] eig Optional borrowed backend, exclusively used during eligible solves.
 * @param[in] eig_min_n Positive minimum dimension for the optional backend.
 * @return Owning Hermitian matrix power; negative exponents give a truncated inverse.
 * @throws Error For invalid dimensions/parameters, no positive eigenvalue or solver failure.
 */
Eigen::MatrixXcd hermitian_power(const Eigen::MatrixXcd& A, double p, double threshold, int* rank = nullptr,
                                 Eigen::VectorXd* eigenvalues = nullptr, Backend* eig = nullptr, int eig_min_n = 256);

/**
 * @brief Reconstructs the local Hamiltonian and coefficients by Löwdin and polar orthonormalization.
 * @param[in] T Orbital overlaps with the retained bands, norb x nb; nb must be at least norb.
 * @param[in] S Local-basis overlap matrix, norb x norb.
 * @param[in] eps Retained band energies in eV, nb entries.
 * @param[in] overlap_threshold Relative eigenvalue cutoff for S; all norb directions must survive.
 * @param[in,out] eig Optional borrowed backend for dense eigensolves; must outlive this call and be used exclusively.
 * @param[in] eig_min_n Minimum matrix dimension for using the backend eigensolver.
 * @return Owning matrices and per-band diagnostics; band_residual uses the square/rectangular contracts above.
 * @throws Error For an insufficient band count, mismatched energies, deficient overlap or unspanned local space.
 */
ReconstructionK reconstruct_k(const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, const Eigen::VectorXd& eps,
                              double overlap_threshold = 1e-8, Backend* eig = nullptr, int eig_min_n = 256);

/**
 * @brief Measures whether the lowest band window spans the local basis.
 * @param[in] T Finite orbital projections, norb x available bands.
 * @param[in] S Finite norb x norb local overlap.
 * @param[in] nwin Band count in [0,T.cols()].
 * @param[in] overlap_threshold Relative cutoff in [0,1) for S inverse square root.
 * @return Smallest eigenvalue of the orthonormalized projected-window overlap.
 * @throws Error For inconsistent dimensions or failed factorization.
 */
double window_conditioning(const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, int nwin,
                           double overlap_threshold = 1e-8);

/**
 * @brief Moves a band cut down so it does not split an energy multiplet.
 * @param[in] eps Finite ascending band energies in eV.
 * @param[in] n Nonnegative requested band count, clamped to eps.size().
 * @param[in] tol Nonnegative degeneracy gap in eV.
 * @return Largest permitted count at or below the requested cut; zero if no group fits.
 */
int cut_at_gap(const Eigen::VectorXd& eps, int n, double tol = 1e-4);

struct FrozenWindow {
    Eigen::MatrixXcd T;
    Eigen::VectorXd eps, occ;
    double weakest{1.0}; ///< smallest projected weight among selected complement states
    double gap{1.0};     ///< relative gap at the selection cut
};
/**
 * @brief Keeps occupied bands and selects a projected unoccupied completion.
 * @param[in] T Finite orbital projections, norb x available bands.
 * @param[in] S Finite norb x norb overlap.
 * @param[in] eps Ascending energies in eV, one per band.
 * @param[in] occ Finite occupations, one per band; nonzero occupations must form a prefix.
 * @param[in] norb Basis size, matching T.rows().
 * @param[in] ncand Candidate count, or -1 for all bands; degeneracies are kept whole.
 * @param[in] occ_tol Nonnegative magnitude below which an occupation is treated as empty.
 * @param[in] overlap_threshold Relative eigenvalue cutoff in [0,1).
 * @return Owning selected projections, energies and occupations, with complement-weight/gap diagnostics.
 * @throws Error For invalid inputs, too many occupied bands or insufficient candidate bands.
 */
FrozenWindow frozen_window_subspace(const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, const Eigen::VectorXd& eps,
                                    const Eigen::VectorXd& occ, int norb, int ncand = -1, double occ_tol = 1e-8,
                                    double overlap_threshold = 1e-8);

/// LOBSTER's lowest-N_orb window with minimal repair at one (spin, k) (SCIENCE.md 5.3): the lowest norb bands if
/// they do not split a multiplet and are well conditioned, else the frozen-window completion over the fewest
/// candidate bands (cut at gaps) that is well conditioned with a unique selection. failed_at > 0: no repair within
/// ncand bands (the band count that would have been needed next).
struct RepairedWindow {
    Eigen::MatrixXcd T;
    Eigen::VectorXd eps, occ;
    double cond{0.0};
    int repaired_nc{0}; ///< 0 = unrepaired lowest-norb window
    int failed_at{0};
    int steps_scanned{0};   ///< candidate band counts evaluated by the scan (diagnostics)
    int steps_reference{0}; ///< reference-predicate evaluations (including the first candidate, before the scan)
    double seconds_setup{0.0}, seconds_bound{0.0}, seconds_scan{0.0}, seconds_reference{0.0}; ///< diagnostics
};
/**
 * @brief Finds the earliest acceptable band window using the policy of repaired_window_reference.
 * @param[in] T Orbital overlaps, norb x available bands.
 * @param[in] S Local-basis overlap matrix, norb x norb.
 * @param[in] eps Sorted band energies in eV, one per column of T.
 * @param[in] occ Band occupations; the nonzero occupied set must be a contiguous prefix.
 * @param[in] norb Number of local orbitals and selected window dimension.
 * @param[in] ncand Maximum candidate band count, no greater than the available bands.
 * @param[in] cond_threshold Minimum eigenvalue of the orthonormalized window overlap.
 * @param[in] gap_threshold Minimum relative projected-weight gap at the complement selection cut.
 * @param[in] gap_tol Energy gap in eV below which bands belong to the same multiplet.
 * @param[in,out] device Optional borrowed backend that creates a separate stepper for long scans.
 * @return Owning selected window, or failed_at > 0 if no window passes within ncand bands.
 *
 * Candidate overlaps are orthonormalized on demand. A monotone upper bound skips impossible windows; scan
 * steps use the smaller of the candidate-complement and norb eigenproblems. Clear accepts in the latter
 * formulation build the window directly; near-threshold steps and smaller candidate problems use the reference
 * predicate. An available device stepper replaces the CPU scan for long scans; it does not run alongside it.
 */
RepairedWindow repaired_window(const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S, const Eigen::VectorXd& eps,
                               const Eigen::VectorXd& occ, int norb, int ncand, double cond_threshold,
                               double gap_threshold, double gap_tol = 1e-4, Backend* device = nullptr);
/**
 * @brief Evaluates the repair policy by its original linear candidate scan.
 * @param[in] T Finite norb x available-band projections.
 * @param[in] S Finite norb x norb overlap.
 * @param[in] eps Ascending finite band energies in eV.
 * @param[in] occ Finite band occupations; nonzero occupied states must be a prefix.
 * @param[in] norb Local basis and selected-window size.
 * @param[in] ncand Largest candidate band count, at most the available bands.
 * @param[in] cond_threshold Minimum accepted overlap eigenvalue.
 * @param[in] gap_threshold Minimum accepted relative selection gap.
 * @param[in] gap_tol Nonnegative multiplet gap in eV.
 * @return Owning selected window, or failed_at > 0 when no candidate passes.
 * @throws Error For invalid dimensions, occupied-band ordering or numerical failure.
 */
RepairedWindow repaired_window_reference(const Eigen::MatrixXcd& T, const Eigen::MatrixXcd& S,
                                         const Eigen::VectorXd& eps, const Eigen::VectorXd& occ, int norb, int ncand,
                                         double cond_threshold, double gap_threshold, double gap_tol = 1e-4);

/**
 * @brief Repairs all spin/k windows, sharing one overlap factorization per k-point.
 * @param[in] T Projections indexed by spin then k.
 * @param[in] S Overlap matrices indexed by k and shared by all spins.
 * @param[in] eps Sorted band energies indexed by spin then k.
 * @param[in] occ Occupations indexed by spin then k.
 * @param[in] norb Local basis size, matching every matrix's row count.
 * @param[in] ncand Maximum candidate band count.
 * @param[in] cond_threshold Minimum accepted window eigenvalue.
 * @param[in] gap_threshold Minimum relative selection gap.
 * @param[in] reference Selects the original linear scan for verification.
 * @param[in,out] device Optional borrowed backend for independent device scan steppers.
 * @return Owning results in spin-major order; failed_at records unsuccessful searches.
 * @throws Error For inconsistent array dimensions or invalid numerical inputs.
 */
std::vector<RepairedWindow> repaired_windows(const std::vector<std::vector<Eigen::MatrixXcd>>& T,
    const std::vector<Eigen::MatrixXcd>& S, const std::vector<std::vector<Eigen::VectorXd>>& eps,
    const std::vector<std::vector<Eigen::VectorXd>>& occ, int norb, int ncand, double cond_threshold,
    double gap_threshold, bool reference = false, Backend* device = nullptr);

struct BondSpec {
    int atom_a{};
    int atom_b{};
    IVec3 R{};
    Eigen::MatrixXd S_R; ///< exact real-space overlap block (empty: no COOP)
    /// Optional bond frame (analysis.orbital_frame = bond): orthogonal rotations of the orbitals of A and B into a
    /// common frame with z' along the bond (na x na, nb x nb; ylm::rotation_matrix per shell), and |m| of every
    /// orbital in that frame. Empty: global frame, no channel COHP. Totals do not depend on the frame.
    Eigen::MatrixXd rot_a, rot_b;
    std::vector<int> absm_a, absm_b;
    /**
     * @brief Reports whether the bond carries a rotation into bond coordinates.
     * @return True when rot_a is populated; analyze validates the complete frame before use.
     */
    [[nodiscard]] bool has_frame() const noexcept { return rot_a.size() > 0; }
};

/// Channels of the bond-frame COHP: pairs with |m_A| = |m_B| = 0 (sigma), 1 (pi), 2 (delta), 3 (phi), and all
/// pairs with |m_A| != |m_B| ("mixed", zero by symmetry for linear molecules). Each channel sum is invariant under
/// rotations about the bond axis, so it does not depend on the choice of x'.
inline constexpr int n_bond_channels = 5;

struct SpectrumGrid {
    double emin{}, emax{};
    int npts{};
    double sigma{};
    double reference{};
    /// 0 = Gaussian (sigma = standard deviation, values at grid points);
    /// 1 = linear tetrahedron (Blöchl integration weights; bin averages over
    ///     [E_i + bin_offset * dE, E_i + (bin_offset + 1) * dE], dE = grid spacing).
    int kind{0};
    double bin_offset{-0.5};
};

struct AnalysisInputs {
    std::vector<Vec3> kfrac;
    std::vector<double> weights;
    std::vector<std::vector<Eigen::MatrixXcd>> T; ///< [spin][k] norb x nb
    std::vector<Eigen::MatrixXcd> S;              ///< [k]
    std::vector<std::vector<Eigen::VectorXd>> eps, occ; ///< [spin][k]
    std::vector<int> orb_offsets;                 ///< per atom, plus sentinel
    double spin_factor{2.0};
    double overlap_threshold{1e-8};
    Backend* eig_backend{nullptr}; ///< optional GPU eigensolver (used when (spin,k) are processed sequentially)
    std::vector<IVec3> kgrid;      ///< tetrahedron method: integer mesh coordinates of each k (mod mesh)
    IVec3 mesh{1, 1, 1};           ///< tetrahedron method: full Gamma-centred mesh
    Mat3 recip{};                  ///< reciprocal rows (no 2 pi), for the shortest-diagonal choice
    int eig_min_n{256};
};

struct AnalysisOutputs {
    int nspin{}, nbond{}, norb{}, npts{};
    std::vector<double> cohp, coop;        ///< [spin][bond][E]
    std::vector<double> icohp, icoop;      ///< [spin][bond]
    std::vector<double> cobi, icobi;       ///< COBI [spin][bond][E], ICOBI [spin][bond] (Wiberg-type bond index)
    std::vector<Eigen::MatrixXd> icohp_orbital; ///< [spin * nbond + bond] (na x nb), in the bond frame if given
    std::vector<double> cohp_channel;      ///< [spin][bond][channel][E] (zeros for bonds without frame)
    std::vector<double> icohp_channel;     ///< [spin][bond][channel]
    std::vector<double> pdos;              ///< [spin][orb][E]
    std::vector<double> pdos_integral, mulliken_gp, lowdin_gp; ///< [spin][orb]
    std::vector<double> band_energy;       ///< [spin]
    double abs_charge_spilling{}, abs_total_spilling{};
    double max_reconstruction_residual{}, min_overlap_eigenvalue{};
    double electrons_dft{};
    bool used_device_eigensolver{};
};

/**
 * @brief Integrates the step function over a linear tetrahedron.
 * @param[in] e Four finite ascending corner energies; array must have four elements.
 * @param[in] E Finite integration energy in the same units as e.
 * @param[in] vt Nonnegative tetrahedron volume fraction.
 * @param[out] w Four corner weights; their sum is the occupied volume fraction.
 * The caller supplies valid four-element arrays. Constant-energy and endpoint limits are handled explicitly.
 */
void tetra_weights(const double e[4], double E, double vt, double w[4]);

/**
 * @brief Reconstructs all spin/k states and evaluates bonding spectra and populations.
 * @param[in] in Finite, dimensionally consistent projections, overlaps, energies, occupations and orbital offsets.
 * @param[in] bonds Validated atom/image pairs and optional exact overlap blocks/bond frames.
 * @param[in] grid Finite energy grid; positive sigma for Gaussian spectra, complete mesh for tetrahedra.
 * @return Owning arrays with layouts documented on AnalysisOutputs, plus quality diagnostics.
 * Shares one S factorization per k across spin channels. A borrowed eig_backend is used exclusively during sequential solves.
 * @throws Error For invalid input dimensions, malformed grids/bonds or failed numerical invariants.
 */
AnalysisOutputs analyze(const AnalysisInputs& in, const std::vector<BondSpec>& bonds, const SpectrumGrid& grid);

} // namespace prowl
