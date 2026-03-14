// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit_cpu.hpp
 * @brief THE single NMF iteration loop for all backends (unified architecture)
 *
 * This file contains ONE generic NMF implementation that works identically
 * on CPU and GPU. The Resource template parameter selects which primitive
 * implementations (gram, rhs, nnls_batch) are called.
 *
 * Features are applied via runtime if-checks on small k×k matrices —
 * the branch cost is negligible compared to O(nnz·k) primitives.
 *
 * Loss computation uses the Gram trick by default:
 *   ||A - WH||² = tr(AᵀA) - 2·tr(BᵀH) + tr(G · HHᵀ)
 * Cost: O(k²) using already-computed G and B — essentially free.
 *
 * Convergence uses relative tolerance with patience:
 *   |Δloss| / |loss| < tol  must hold for `patience` consecutive checks.
 *
 * Internally, W is stored as k × m (transposed) for cache-efficient
 * column-major iteration. It is transposed to m × k in the result.
 *
 * This is the NEW unified implementation, coexisting with the old
 * nmf/fit.hpp during the migration period.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <chrono>
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/core/constants.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/core/memory.hpp>

// Primitives (template-specialized per Resource)
#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>
#include <FactorNet/primitives/cpu/rhs.hpp>
#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/primitives/cpu/loss.hpp>
#include <FactorNet/primitives/cpu/fused_nnls.hpp>
#include <FactorNet/primitives/cpu/data_accessor.hpp>

// Features (backend-agnostic)
#include <FactorNet/features/sparsity.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>

// Shared algorithm variant dispatch and helpers
#include <FactorNet/nmf/variant_helpers.hpp>

// RNG
#include <FactorNet/rng/rng.hpp>

// SVD (for Lanczos and IRLBA initialization)
#include <FactorNet/svd/lanczos.hpp>
#include <FactorNet/svd/irlba.hpp>

#include <cmath>
#include <limits>
#include <algorithm>

// NMF initialization (Lanczos, IRLBA, random)
#include <FactorNet/nmf/nmf_init.hpp>

// CPU section profiling
#include <FactorNet/profiling/cpu_timer.hpp>

// Gram-trick loss and explicit loss helpers
#include <FactorNet/nmf/explicit_loss.hpp>

// Masked NNLS helpers (user-supplied mask)
#include <FactorNet/nmf/masked_nnls.hpp>

namespace FactorNet {
namespace nmf {

// ========================================================================

namespace detail {

/**
 * @brief Scale rows of F by d: F.row(i) *= d(i)
 */
template<typename Scalar>
inline void apply_scaling(DenseMatrix<Scalar>& F,
                          const DenseVector<Scalar>& d)
{
    for (int i = 0; i < d.size(); ++i) {
        F.row(i) *= d(i);
    }
}

// extract_scaling removed — use variant::extract_scaling from variant_helpers.hpp

/**
 * @brief Compute transposed RHS for W update: B = H · Aᵗ (k × m)
 *
 * For sparse A: uses gather pattern on pre-computed CSC(Aᵀ)
 *      - No write contention, embarrassingly parallel
 *      - Same access pattern as H-update
 *
 * For dense A: B = H * A.transpose() (always BLAS).
 */
template<typename Scalar, typename MatrixType>
inline void rhs_transpose(const MatrixType& A,
                           const DenseMatrix<Scalar>& H,
                           DenseMatrix<Scalar>& B,
                           int m, int n,
                           const SparseMatrix<Scalar>* At = nullptr,
                           int num_threads = 1)
{
    const int k = static_cast<int>(H.rows());
    B.resize(k, m);
    B.setZero();

    if constexpr (is_sparse_v<MatrixType>) {
        // Gather path: iterate columns of Aᵀ (rows of A)
        // B(:, j) = Σ over nnz in Aᵀ(:, j): val * H(:, row)
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64) if(num_threads > 1)
        for (int j = 0; j < m; ++j) {
            for (typename SparseMatrix<Scalar>::InnerIterator it(*At, j); it; ++it) {
                B.col(j).noalias() += it.value() * H.col(it.row());
            }
        }
    } else {
        B.noalias() = H * A.transpose();
    }
}

}  // namespace detail

// ========================================================================

// ========================================================================
// Main NMF fitting function
// ========================================================================

/**
 * @brief NMF fitting — THE single implementation for all backends
 *
 * Template parameter Resource selects CPU or GPU primitives.
 * Template parameter MatrixType selects sparse or dense A.
 * Features are applied via runtime checks (negligible cost on k×k ops).
 *
 * @tparam Resource    primitives::CPU or primitives::GPU
 * @tparam Scalar      float or double
 * @tparam MatrixType  SparseMatrix<Scalar>, MappedSparseMatrix<Scalar>, or DenseMatrix<Scalar>
 *
 * @param A       Data matrix (m × n)
 * @param config  NMF configuration
 * @param W_init  Optional initialization for W (m × k). nullptr = random.
 * @param H_init  Optional initialization for H (k × n). nullptr = random.
 * @return        NMFResult with W (m×k), d (k), H (k×n), convergence info
 */
template<typename Resource, typename Scalar, typename MatrixType>
::FactorNet::NMFResult<Scalar> nmf_fit(
    const MatrixType& A,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.rank;

    // Validate configuration (guards against unsupported combos)
    config.validate();

    // ------------------------------------------------------------------
    // 1. Initialize factors
    // ------------------------------------------------------------------

    // W_T: k × m (transposed storage for cache-efficient row iteration)
    // H:   k × n
    // d:   k (diagonal scaling)
    DenseMatrix<Scalar> W_T, H;
    DenseVector<Scalar> d;

    if (W_init) {
        // User-provided W initialization — transpose from m×k to k×m
        W_T = W_init->transpose();
        d = DenseVector<Scalar>::Ones(k);

        if (H_init) {
            H = *H_init;
        } else {
            // Initialize H randomly using seed
            H.resize(k, n);
            rng::SplitMix64 hrng(config.seed == 0 ? 12345U : config.seed);
            hrng.fill_uniform(H.data(), H.rows(), H.cols());
        }
    } else if (config.init_mode == 1) {
        // Lanczos SVD initialization: W = |U|·√Σ, H = √Σ·|V|^T
        detail::initialize_lanczos(W_T, H, d, A, k, m, n,
                                   config.seed, config.threads, config.verbose);
    } else if (config.init_mode == 2) {
        // IRLBA SVD initialization: W = |U|·√Σ, H = √Σ·|V|^T
        detail::initialize_irlba(W_T, H, d, A, k, m, n,
                                 config.seed, config.threads, config.verbose);
    } else {
        detail::initialize_factors(W_T, H, d, k, m, n, config.seed);
    }

    // ------------------------------------------------------------------
    // 2. Precompute tr(AᵗA) for Gram-trick loss
    // ------------------------------------------------------------------

    const Scalar trAtA = primitives::trace_AtA<Resource, Scalar, MatrixType>(A);

    // ------------------------------------------------------------------
    // 2b. Pre-compute CSC(Aᵀ) for gather-based W update
    // ------------------------------------------------------------------
    // For sparse input, compute the transpose once.
    // This eliminates write contention in the W-update RHS computation
    // (embarrassingly parallel, no atomics).
    // Cost: O(nnz + m + n) time, O(nnz) memory. Amortized over all iterations.

    const SparseMatrix<Scalar>* At_ptr = nullptr;
    SparseMatrix<Scalar> At_storage;

    if constexpr (is_sparse_v<MatrixType>) {
        // Memory safety check: ensure we have enough RAM for the transpose
        const size_t nnz = static_cast<size_t>(A.nonZeros());
        auto mem_check = FactorNet::memory::check_transpose_memory<Scalar>(nnz, m, n);

        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, config.verbose,
            "Transpose: %s\n", mem_check.message.c_str());

        if (!mem_check.fits) {
            throw std::runtime_error(
                std::string("Cannot compute in-memory transpose of sparse matrix.\n") +
                mem_check.message);
        }

        At_storage = A.transpose();
        At_storage.makeCompressed();
        At_ptr = &At_storage;
    }

    int eff_threads = config.threads;
    if (eff_threads <= 0) {
        #ifdef _OPENMP
        eff_threads = omp_get_max_threads();
        #else
        eff_threads = 1;
        #endif
    }

    // ------------------------------------------------------------------
    // 3. Allocate working buffers
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> G(k, k);      // Gram matrix — always host-resident
    DenseMatrix<Scalar> B;             // RHS buffer

    // Pre-compute mask transpose for W-update (needed for per-row masking)
    SparseMatrix<Scalar> mask_T_storage;
    const bool use_mask = config.has_mask();
    if (use_mask) {
        mask_T_storage = config.mask->transpose();
        mask_T_storage.makeCompressed();
    }

    ::FactorNet::NMFResult<Scalar> result;
    Scalar prev_loss = std::numeric_limits<Scalar>::max();
    int patience_counter = 0;
    auto fit_start = std::chrono::high_resolution_clock::now();
    FactorNet::profiling::SectionTimer prof_timer(config.enable_profiling);
    if (config.track_loss_history)
        result.history.reserve(config.max_iter);

    // ------------------------------------------------------------------
    // 3a-gp. Initialize theta for Generalized Poisson loss
    // ------------------------------------------------------------------
    DenseVector<Scalar> theta_vec;
    const bool is_gp = (config.loss.type == LossType::GP);
    const bool is_nb = (config.loss.type == LossType::NB);
    const bool is_gamma = (config.loss.type == LossType::GAMMA);
    const bool is_invgauss = (config.loss.type == LossType::INVGAUSS);
    const bool is_tweedie = (config.loss.type == LossType::TWEEDIE);
    if (is_gp) {
        if (config.gp_dispersion == DispersionMode::PER_ROW) {
            theta_vec = DenseVector<Scalar>::Constant(m, config.gp_theta_init);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            theta_vec = DenseVector<Scalar>::Constant(n, config.gp_theta_init);
        } else if (config.gp_dispersion == DispersionMode::GLOBAL) {
            theta_vec = DenseVector<Scalar>::Constant(m, config.gp_theta_init);
        } else {
            // NONE: theta fixed at 0 (reduces to KL / Poisson)
            theta_vec = DenseVector<Scalar>::Zero(m);
        }
    }

    // ------------------------------------------------------------------
    // 3a-nb. Initialize NB size (inverse-dispersion r) vector
    // ------------------------------------------------------------------
    // NB reuses theta_vec for per-row storage: theta_vec(i) = r_i
    // The compute_irls_weight dispatcher interprets it as NB size when
    // loss.type == NB, and as GP theta when loss.type == GP.
    DenseVector<Scalar> nb_size_vec;
    if (is_nb) {
        if (config.gp_dispersion == DispersionMode::PER_ROW) {
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            nb_size_vec = DenseVector<Scalar>::Constant(n, config.nb_size_init);
        } else if (config.gp_dispersion == DispersionMode::GLOBAL) {
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        } else {
            // NONE: r → ∞ (Poisson limit) — use large fixed value
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_max);
        }
    }

    // ------------------------------------------------------------------
    // 3a-gam. Initialize dispersion φ for Gamma / Inverse Gaussian
    // ------------------------------------------------------------------
    // Gamma:    V(μ) = φ·μ², IRLS weight = 1/μ² (does NOT depend on φ)
    // InvGauss: V(μ) = φ·μ³, IRLS weight = 1/μ³ (does NOT depend on φ)
    // φ is estimated via MoM Pearson estimator each iteration (diagnostic).
    DenseVector<Scalar> phi_vec;
    if (is_gamma || is_invgauss || is_tweedie) {
        if (config.gp_dispersion == DispersionMode::PER_ROW) {
            phi_vec = DenseVector<Scalar>::Constant(m, config.gamma_phi_init);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            phi_vec = DenseVector<Scalar>::Constant(n, config.gamma_phi_init);
        } else if (config.gp_dispersion == DispersionMode::GLOBAL) {
            phi_vec = DenseVector<Scalar>::Constant(m, config.gamma_phi_init);
        } else {
            // NONE: fixed at 1 (unit dispersion)
            phi_vec = DenseVector<Scalar>::Constant(m, static_cast<Scalar>(1));
        }
    }

    // ------------------------------------------------------------------
    // 3a-zi. Initialize pi for zero-inflated GP (ZIGP)
    // ------------------------------------------------------------------
    DenseVector<Scalar> pi_row_vec, pi_col_vec;
    const bool is_zi = (is_gp || is_nb) && config.has_zi();
    if (is_zi) {
        // Data-driven init: pi = min(zero_rate * 0.5, 0.3)
        // Gives ~0 for dense matrices (ZIGP becomes a no-op) and
        // conservative values proportional to actual sparsity.
        if (config.zi_mode == ZIMode::ZI_ROW || config.zi_mode == ZIMode::ZI_TWOWAY) {
            pi_row_vec.resize(m);
            if constexpr (is_sparse_v<MatrixType>) {
                Eigen::VectorXi nnz_per_row = Eigen::VectorXi::Zero(m);
                for (int j = 0; j < n; ++j)
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                        nnz_per_row(it.row())++;
                for (int i = 0; i < m; ++i) {
                    double zero_rate = 1.0 - static_cast<double>(nnz_per_row(i)) / n;
                    pi_row_vec(i) = static_cast<Scalar>(std::min(zero_rate * 0.5, 0.3));
                }
            } else {
                for (int i = 0; i < m; ++i) {
                    int nz = 0;
                    for (int j = 0; j < n; ++j)
                        if (static_cast<double>(A(i, j)) == 0.0) nz++;
                    double zero_rate = static_cast<double>(nz) / n;
                    pi_row_vec(i) = static_cast<Scalar>(std::min(zero_rate * 0.5, 0.3));
                }
            }
        }
        if (config.zi_mode == ZIMode::ZI_COL || config.zi_mode == ZIMode::ZI_TWOWAY) {
            pi_col_vec.resize(n);
            if constexpr (is_sparse_v<MatrixType>) {
                for (int j = 0; j < n; ++j) {
                    int nnz_col = 0;
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                        nnz_col++;
                    double zero_rate = 1.0 - static_cast<double>(nnz_col) / m;
                    pi_col_vec(j) = static_cast<Scalar>(std::min(zero_rate * 0.5, 0.3));
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    int nz = 0;
                    for (int i = 0; i < m; ++i)
                        if (static_cast<double>(A(i, j)) == 0.0) nz++;
                    double zero_rate = static_cast<double>(nz) / m;
                    pi_col_vec(j) = static_cast<Scalar>(std::min(zero_rate * 0.5, 0.3));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 3a-zi-impute. Dense imputation matrix for ZIGP soft imputation
    // ------------------------------------------------------------------
    // When ZIGP is active, we create a dense copy of A. After the E-step,
    // zero entries get imputed with z_ij * mu_ij (soft imputation).
    // The IRLS solver then operates on this imputed dense matrix instead
    // of the original sparse/dense A for subsequent iterations.
    DenseMatrix<Scalar> A_imputed;
    if (is_zi) {
        A_imputed.resize(m, n);
        if constexpr (is_sparse_v<MatrixType>) {
            // Initialize from sparse: copy nonzeros, rest stays 0
            A_imputed.setZero();
            for (int j = 0; j < n; ++j)
                for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                    A_imputed(it.row(), j) = static_cast<Scalar>(it.value());
        } else {
            A_imputed = A;
        }
    }

    // ------------------------------------------------------------------
    // 3b. Fused RHS+NNLS eligibility
    // ------------------------------------------------------------------
    // Use fused path when: sparse data, no mask, no IRLS.
    // Works for all solver modes (CD, Cholesky+clip, hierarchical Cholesky).
    // The fused path computes RHS + warm start + NNLS in a single parallel
    // loop, eliminating the global B matrix and BLAS-3 warm start GEMM.
    const bool can_fuse = is_sparse_v<MatrixType>
                       && !use_mask
                       && !config.requires_irls()
                       && !config.has_target();  // target reg needs standard path for B modification

    if (config.verbose && can_fuse) {
        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, true,
            "Using fused RHS+NNLS path (sparse CD)\n");
    }

    // ------------------------------------------------------------------
    // 4. Iteration loop
    // ------------------------------------------------------------------

    for (int iter = 0; iter < config.max_iter; ++iter) {

        // --- Timing instrumentation ---
        auto _t0 = std::chrono::high_resolution_clock::now();
        auto _t_gram_h = _t0, _t_rhs_h = _t0, _t_nnls_h = _t0,
             _t_norm_h = _t0, _t_gram_w = _t0, _t_rhs_w = _t0,
             _t_nnls_w = _t0, _t_norm_w = _t0, _t_loss = _t0, _t_end = _t0;

        // Loss is always computed for loss-based convergence
        const bool compute_loss_this_iter = true;
        const bool need_loss = true;

        // ==============================================================
        // Update H — dispatch on algorithm variant
        // ==============================================================

        const auto h_mode = nmf::variant::h_update_mode(config);

        if (h_mode == nmf::variant::HUpdateMode::PROJECTIVE) {
            // --------------------------------------------------------
            // PROJECTIVE NMF: H = diag(d) · W_T · A  (tied-weight)
            // No NNLS solve — H is deterministic given W and A.
            // --------------------------------------------------------
            nmf::variant::projective_h_update(
                W_T, H, d,
                [&](const DenseMatrix<Scalar>& factor, DenseMatrix<Scalar>& result) {
                    primitives::rhs<Resource>(A, factor, result);
                },
                config.norm_type);

        } else if (h_mode == nmf::variant::HUpdateMode::SYMMETRIC_SKIP) {
            // --------------------------------------------------------
            // SYMMETRIC NMF: skip H update; H = W_T set after W update
            // --------------------------------------------------------

        } else {
            // --------------------------------------------------------
            // STANDARD NMF: Gram → RHS → features → NNLS → normalize
            // --------------------------------------------------------

            // Use normalized W_T directly (unit-norm rows) for well-conditioned
            // Gram matrix, matching old backend behavior.
            // d is NOT applied before Gram/RHS — this preserves conditioning.

            // 1. Compute Gram — always needed
            _t_gram_h = std::chrono::high_resolution_clock::now();
            prof_timer.begin("gram_H");
            primitives::gram<Resource>(W_T, G);
            prof_timer.end();

            // The fused path is sparse-only: guard with if constexpr so that
            // fused_rhs_nnls_sparse (which calls sparse-only Eigen API) is never
            // instantiated when MatrixType is a dense matrix.
            bool h_used_fused = false;
            if constexpr (is_sparse_v<MatrixType>) {
            if (can_fuse) {
                // ====================================================
                // FUSED PATH: RHS + warm start + NNLS in one parallel loop
                // No global B matrix needed.
                // ====================================================
                _t_rhs_h = std::chrono::high_resolution_clock::now();
                prof_timer.begin("features_H");
                if (config.H.L2 > 0) G.diagonal().array() += config.H.L2;
                // Angular: applied post-NNLS via apply_angular_posthoc
                if (config.H.graph)
                    features::apply_graph_reg(G, *config.H.graph, H, config.H.graph_lambda);
                if (config.H.L21 > 0)
                    features::apply_L21(G, H, config.H.L21);
                prof_timer.end();

                // Fused RHS+NNLS (L1 applied per-column inside the loop)
                prof_timer.begin("fused_rhs_nnls_H");
                if (config.solver_mode == 0) {
                    primitives::fused_rhs_nnls_sparse(
                        A, W_T, G, H,
                        config.cd_max_iter, config.cd_tol,
                        config.H.L1,
                        config.H.nonneg,
                        eff_threads,
                        iter > 0,
                        static_cast<Scalar>(0));
                } else {
                    // Modes 1 and 2: pre-factorized Cholesky fused path
                    primitives::fused_rhs_cholesky_sparse(
                        A, W_T, G, H,
                        config.solver_mode,
                        config.H.L1,
                        config.H.nonneg,
                        eff_threads,
                        iter > 0,
                        static_cast<Scalar>(0));
                }
                prof_timer.end();

                _t_nnls_h = std::chrono::high_resolution_clock::now();
                h_used_fused = true;
            }
            } // end if constexpr sparse
            if (!h_used_fused) {
                // ====================================================
                // STANDARD PATH: separate RHS → features → NNLS
                // Used for dense data, non-CD solvers, mask, IRLS
                // ====================================================
                _t_rhs_h = std::chrono::high_resolution_clock::now();
                prof_timer.begin("rhs_H");
                primitives::rhs<Resource>(A, W_T, B);
                prof_timer.end();

                // 2. Apply all H-side features (shared sequence)
                prof_timer.begin("features_H");
                nmf::variant::apply_h_features(G, B, H, config);
                prof_timer.end();

                _t_nnls_h = std::chrono::high_resolution_clock::now();
                prof_timer.begin("nnls_H");
                // 4. NNLS solve — branch on mask / loss type
                if (use_mask) {
                    // Masked path: per-column NNLS with delta-G correction
                    primitives::gram<Resource>(W_T, G);  // Rebuild unmodified G
                    mask_detail::masked_nnls_h(
                        A, W_T, G, H, *config.mask, config, eff_threads, iter > 0);
                } else if (config.requires_irls()) {
                    // IRLS path: per-column weighted NNLS
                    primitives::gram<Resource>(W_T, G);  // Rebuild unmodified G for IRLS base
                    // GP strategy: use KL weights for W/H updates (stable),
                    // estimate theta separately via closed-form MM update.
                    // KL weight (1/s) and GP share the same fixed point for W/H.
                    // NB strategy: use NB weights directly (well-behaved, r/(μ(r+μ))).
                    FactorNet::LossConfig<Scalar> active_loss = config.loss;
                    if (is_gp) {
                        active_loss.type = LossType::KL;
                    }
                    // NB: pass nb_size_vec as theta for NB weight computation
                    // PER_ROW  → theta_per_row (indexed by row of A)
                    // PER_COL  → theta_per_col (indexed by column of A)
                    // GLOBAL   → theta_per_row (all same value)
                    const Scalar* irls_theta_ptr = nullptr;
                    const Scalar* irls_theta_col_ptr = nullptr;
                    if (is_nb) {
                        if (config.gp_dispersion == DispersionMode::PER_COL) {
                            irls_theta_col_ptr = nb_size_vec.data();
                        } else {
                            irls_theta_ptr = nb_size_vec.data();
                        }
                    }
                    if (is_zi && iter > 0) {
                        // ZIGP/ZINB: use imputed dense matrix (zeros replaced with z*mu)
                        primitives::nnls_batch_irls_dense(
                            A_imputed, W_T, G, H, active_loss,
                            config.H.L1, config.H.L2,
                            config.H.nonneg,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads, irls_theta_ptr, irls_theta_col_ptr);
                    } else if constexpr (std::is_base_of_v<Eigen::SparseMatrixBase<std::decay_t<decltype(A)>>,
                                                    std::decay_t<decltype(A)>>) {
                        primitives::nnls_batch_irls_sparse(
                            A, W_T, G, H, active_loss,
                            config.H.L1, config.H.L2,
                            config.H.nonneg,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads, irls_theta_ptr, irls_theta_col_ptr);
                    } else {
                        primitives::nnls_batch_irls_dense(
                            A, W_T, G, H, active_loss,
                            config.H.L1, config.H.L2,
                            config.H.nonneg,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads, irls_theta_ptr, irls_theta_col_ptr);
                    }
                } else if (config.solver_mode == 1) {
                    // Cholesky + clip solver: unconstrained LS then max(0,x)
                    primitives::cholesky_clip_batch(G, B, H,
                                                   config.H.nonneg,
                                                   config.threads,
                                                   iter > 0);
                } else {
                    primitives::nnls_batch<Resource>(G, B, H,
                                                     config.cd_max_iter, config.cd_tol,
                                                     static_cast<Scalar>(0),
                                                     static_cast<Scalar>(0),
                                                     config.H.nonneg,
                                                     config.threads,
                                                     static_cast<Scalar>(0),  // upper_bound
                                                     iter > 0);               // warm_start
                }
            }  // end fused vs standard H-update NNLS
            if (!h_used_fused) prof_timer.end();  // end nnls_H for standard path

            // 5. Post-NNLS bounds and angular decorrelation
            if (config.H.upper_bound > 0)
                features::apply_upper_bound(H, config.H.upper_bound);
            if (config.H.angular > 0)
                features::apply_angular_posthoc(H, config.H.angular);

            _t_norm_h = std::chrono::high_resolution_clock::now();
            // 6. Normalize H → d
            prof_timer.begin("scaling");
            variant::extract_scaling(H, d, config.norm_type);
            prof_timer.end();
        }  // end standard H-update

        // ==============================================================
        // Update W
        // ==============================================================

        // Declare loss-saving variables before the branch so they remain in scope
        DenseMatrix<Scalar> G_w_saved, B_w_saved;
        bool saved_for_loss = false;
        bool used_fused_w = false;  // Track if fused path was used (no B_w_saved)

        const auto w_mode = nmf::variant::w_update_mode(config);

        if (w_mode == nmf::variant::WUpdateMode::SYMMETRIC) {
            // --------------------------------------------------------
            // SYMMETRIC NMF: A ≈ W^T·diag(d)·W, only one update needed.
            // Gram = W_T·W_T^T, RHS = W_T·A (forward, since A = A^T).
            // After solving for W_T, enforce H = W_T.
            // --------------------------------------------------------
            primitives::gram<Resource>(W_T, G);
            primitives::rhs<Resource>(A, W_T, B);

            // Save pre-feature Gram and RHS for loss computation
            if (need_loss && compute_loss_this_iter && !use_mask && !config.requires_irls()) {
                G_w_saved = G;
                B_w_saved = B;
                saved_for_loss = true;
            }

            // Apply W-side features (shared sequence)
            nmf::variant::apply_w_features(G, B, W_T, config);

            // NNLS solve for W_T
            if (config.solver_mode == 1) {
                primitives::cholesky_clip_batch(G, B, W_T,
                                               config.W.nonneg,
                                               config.threads,
                                               iter > 0);
            } else {
                primitives::nnls_batch<Resource>(G, B, W_T,
                                                 config.cd_max_iter, config.cd_tol,
                                                 static_cast<Scalar>(0),
                                                 static_cast<Scalar>(0),
                                                 config.W.nonneg,
                                                 config.threads,
                                                 static_cast<Scalar>(0),
                                                 iter > 0);
            }

            if (config.W.upper_bound > 0)
                features::apply_upper_bound(W_T, config.W.upper_bound);
            if (config.W.angular > 0)
                features::apply_angular_posthoc(W_T, config.W.angular);

            // Normalize W_T → d
            variant::extract_scaling(W_T, d, config.norm_type);

            // Enforce symmetry: H = W_T (shared helper)
            nmf::variant::symmetric_enforce_h(H, W_T);

        } else {
            // --------------------------------------------------------
            // STANDARD or PROJECTIVE W update
            // --------------------------------------------------------

            // Use normalized H directly (unit-norm rows) for well-conditioned
            // Gram matrix, matching old backend behavior.
            _t_gram_w = std::chrono::high_resolution_clock::now();
            prof_timer.begin("gram_W");
            primitives::gram<Resource>(H, G);
            prof_timer.end();

            // Save G_w before features modify it (needed for loss in both paths)
            if (need_loss && compute_loss_this_iter && !use_mask && !config.requires_irls()) {
                G_w_saved = G;  // k × k (cheap)
                saved_for_loss = true;
            }

            // The fused path is sparse-only: guard with if constexpr so that
            // fused_rhs_nnls_sparse (which calls sparse-only Eigen API) is never
            // instantiated when MatrixType is a dense matrix.
            if constexpr (is_sparse_v<MatrixType>) {
            if (can_fuse && At_ptr) {
                // ====================================================
                // FUSED PATH (W-update): RHS + warm start + NNLS in one loop
                // Iterates columns of A^T (= rows of A).
                // No global B matrix needed.
                // ====================================================
                _t_rhs_w = std::chrono::high_resolution_clock::now();
                prof_timer.begin("features_W");

                // Apply G-modifying features before the fused loop
                if (config.W.L2 > 0) G.diagonal().array() += config.W.L2;
                // Angular: applied post-NNLS via apply_angular_posthoc
                if (config.W.graph)
                    features::apply_graph_reg(G, *config.W.graph, W_T, config.W.graph_lambda);
                if (config.W.L21 > 0)
                    features::apply_L21(G, W_T, config.W.L21);
                prof_timer.end();

                // Fused RHS+NNLS on columns of A^T
                prof_timer.begin("fused_rhs_nnls_W");
                if (config.solver_mode == 0) {
                    primitives::fused_rhs_nnls_sparse(
                        *At_ptr, H, G, W_T,
                        config.cd_max_iter, config.cd_tol,
                        config.W.L1,
                        config.W.nonneg,
                        eff_threads,
                        iter > 0,
                        static_cast<Scalar>(0));
                } else {
                    // Modes 1 and 2: pre-factorized Cholesky fused path
                    primitives::fused_rhs_cholesky_sparse(
                        *At_ptr, H, G, W_T,
                        config.solver_mode,
                        config.W.L1,
                        config.W.nonneg,
                        eff_threads,
                        iter > 0,
                        static_cast<Scalar>(0));
                }
                prof_timer.end();

                _t_nnls_w = std::chrono::high_resolution_clock::now();
                used_fused_w = true;
            }
            } // end if constexpr sparse
            if (!used_fused_w) {
                // ====================================================
                // STANDARD PATH (W-update): separate RHS → features → NNLS
                // ====================================================

                // Transposed RHS: B = H · Aᵗ  (k × m)
                // Uses gather pattern when transpose is available (no atomics)
                _t_rhs_w = std::chrono::high_resolution_clock::now();
                prof_timer.begin("rhs_W");
                detail::rhs_transpose(A, H, B, m, n, At_ptr, eff_threads);
                prof_timer.end();

                // Save B_w = H·A^T BEFORE features modify it.
                if (saved_for_loss) {
                    B_w_saved = B;  // k × m
                }

                // Apply W-side features (shared sequence)
                prof_timer.begin("features_W");
                nmf::variant::apply_w_features(G, B, W_T, config);
                prof_timer.end();

                _t_nnls_w = std::chrono::high_resolution_clock::now();
                prof_timer.begin("nnls_W");
                // NNLS solve — branch on mask / loss type
                if (use_mask) {
                    // Masked path: per-column NNLS with delta-G correction
                    primitives::gram<Resource>(H, G);  // Rebuild unmodified G
                    if constexpr (is_sparse_v<MatrixType>) {
                        // Pass A^T so that DA::for_col(At, j) iterates row j of A
                        mask_detail::masked_nnls_w(
                            *At_ptr, H, G, W_T, mask_T_storage, config, eff_threads, iter > 0);
                    } else {
                        // Pass A; DA::for_row(A, j) iterates row j of A
                        mask_detail::masked_nnls_w(
                            A, H, G, W_T, mask_T_storage, config, eff_threads, iter > 0);
                    }
                } else if (config.requires_irls()) {
                    // IRLS W-update: uses A^T as the "data" with H as the "factor"
                    primitives::gram<Resource>(H, G);  // Rebuild unmodified G
                    // GP: use KL weights (stable), theta estimated separately
                    // NB: use NB weights directly (well-behaved)
                    FactorNet::LossConfig<Scalar> active_loss_w = config.loss;
                    if (is_gp) {
                        active_loss_w.type = LossType::KL;
                    }
                    // For W-update, A^T columns correspond to rows of A.
                    // PER_ROW: nb_size indexed by row of A → theta_per_col (col of A^T = row of A)
                    // PER_COL: nb_size indexed by col of A → theta_per_row (row of A^T = col of A)
                    // GLOBAL:  all same value → theta_per_col
                    const Scalar* irls_theta_ptr_w = nullptr;      // theta_per_row of A^T
                    const Scalar* irls_theta_col_ptr_w = nullptr;  // theta_per_col of A^T
                    if (is_nb) {
                        if (config.gp_dispersion == DispersionMode::PER_COL) {
                            irls_theta_ptr_w = nb_size_vec.data();
                        } else {
                            irls_theta_col_ptr_w = nb_size_vec.data();
                        }
                    }
                    if (is_zi && iter > 0) {
                        // ZIGP/ZINB: use imputed dense matrix transposed
                        DenseMatrix<Scalar> A_imp_T = A_imputed.transpose();
                        primitives::nnls_batch_irls_dense(
                            A_imp_T, H, G, W_T, active_loss_w,
                            config.W.L1, config.W.L2,
                            config.W.nonneg,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads, irls_theta_ptr_w, irls_theta_col_ptr_w);
                    } else if constexpr (is_sparse_v<MatrixType>) {
                        // Use pre-computed transpose
                        if (At_ptr) {
                            primitives::nnls_batch_irls_sparse(
                                *At_ptr, H, G, W_T, active_loss_w,
                                config.W.L1, config.W.L2,
                                config.W.nonneg,
                                config.cd_max_iter, config.cd_tol,
                                config.irls_max_iter, config.irls_tol,
                                config.threads, irls_theta_ptr_w, irls_theta_col_ptr_w);
                        }
                    } else {
                        // Dense case: use A^T directly
                        DenseMatrix<Scalar> A_T = A.transpose();
                        primitives::nnls_batch_irls_dense(
                            A_T, H, G, W_T, active_loss_w,
                            config.W.L1, config.W.L2,
                            config.W.nonneg,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads, irls_theta_ptr_w, irls_theta_col_ptr_w);
                    }
                } else if (config.solver_mode == 1) {
                    // Cholesky + clip solver
                    primitives::cholesky_clip_batch(G, B, W_T,
                                                   config.W.nonneg,
                                                   config.threads,
                                                   iter > 0);
                } else {
                    primitives::nnls_batch<Resource>(G, B, W_T,
                                                     config.cd_max_iter, config.cd_tol,
                                                     static_cast<Scalar>(0),
                                                     static_cast<Scalar>(0),
                                                     config.W.nonneg,
                                                     config.threads,
                                                     static_cast<Scalar>(0),  // upper_bound
                                                     iter > 0);               // warm_start
                }
            }  // end fused vs standard W-update NNLS
            if (!used_fused_w) prof_timer.end();  // end nnls_W for standard path

            if (config.W.upper_bound > 0)
                features::apply_upper_bound(W_T, config.W.upper_bound);
            if (config.W.angular > 0)
                features::apply_angular_posthoc(W_T, config.W.angular);

            _t_norm_w = std::chrono::high_resolution_clock::now();
            // Normalize W_T → d
            prof_timer.begin("scaling");
            variant::extract_scaling(W_T, d, config.norm_type);
            prof_timer.end();
        }  // end W update

        // ==============================================================
        // GP theta update (closed-form, per-row dispersion)
        // ==============================================================
        // Auxiliary-function theta update from Ohashi et al. (2025) Eq. 24.
        // Uses the MM (minorization-maximization) approach where α, β, γ
        // depend on the CURRENT theta estimate (θ^old) via:
        //   η₁ = s_ij / (s_ij + θ^old_i · y_ij)
        // Then: α_i θ² + β_i θ - γ_i = 0
        //   → θ_i = (-β_i + √(β²_i + 4α_i γ_i)) / (2α_i)
        //
        // All accumulation done in fp64 to avoid overflow/precision issues.
        // Theta update runs every iteration (no warmup needed since
        // W/H are always updated with KL weights, which are stable).
        //
        // Inner MM iterations: the MM auxiliary function tightens with
        // each θ update, so multiple inner iterations improve accuracy
        // without changing W/H. We cache per-nonzero (row, y, s) to
        // avoid recomputing dot products.
        if (is_gp && config.gp_dispersion != DispersionMode::NONE) {
            prof_timer.begin("dispersion");
            DenseMatrix<Scalar> W_Td = W_T;
            detail::apply_scaling(W_Td, d);

            constexpr int THETA_INNER_ITERS = 5;

            if (config.gp_dispersion == DispersionMode::PER_ROW ||
                config.gp_dispersion == DispersionMode::GLOBAL) {

                // Theta-independent per-row sums (computed once)
                Eigen::VectorXd sum_y_d = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd sum_s_d = Eigen::VectorXd::Zero(m);
                Eigen::VectorXi n_nz = Eigen::VectorXi::Zero(m);

                // Cache nonzero entries: (row, y, s) for inner iterations
                struct NzEntry { int row; double y; double s; };
                std::vector<NzEntry> nz_cache;

                if constexpr (is_sparse_v<MatrixType>) {
                    // Pre-compute sum_s via Gram trick
                    DenseVector<Scalar> h_rowsum = H.rowwise().sum();
                    for (int i = 0; i < m; ++i) {
                        sum_s_d(i) = static_cast<double>(W_Td.col(i).dot(h_rowsum));
                    }

                    // Single pass: cache (row, y, s) and accumulate theta-independent sums
                    nz_cache.reserve(A.nonZeros());
                    for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            int i = it.row();
                            double y = static_cast<double>(it.value());
                            double s = std::max(static_cast<double>(
                                W_Td.col(i).dot(H.col(j))), 1e-10);
                            sum_y_d(i) += y;
                            if (y >= 1.0) n_nz(i)++;
                            nz_cache.push_back({i, y, s});
                        }
                    }
                } else {
                    // Dense path: cache all entries with y > 0
                    DenseMatrix<Scalar> S = W_Td.transpose() * H;
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                            double y = static_cast<double>(A(i, j));
                            double s = std::max(static_cast<double>(S(i, j)), 1e-10);
                            sum_y_d(i) += y;
                            sum_s_d(i) += s;
                            if (y >= 1.0) n_nz(i)++;
                            if (y >= 1.0) {
                                nz_cache.push_back({i, y, s});
                            }
                        }
                    }
                }

                // Inner MM iterations: re-accumulate alpha/gamma with updated theta
                double cap = static_cast<double>(config.gp_theta_max);
                for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS; ++mm_iter) {
                    Eigen::VectorXd alpha_d = Eigen::VectorXd::Zero(m);
                    Eigen::VectorXd gamma_d = Eigen::VectorXd::Zero(m);

                    // Accumulate theta-dependent statistics
                    for (const auto& nz : nz_cache) {
                        if (nz.y >= 1.0) {
                            double th = static_cast<double>(theta_vec(nz.row));
                            double denom = std::max(nz.s + th * nz.y, 1e-10);
                            double eta1 = nz.s / denom;
                            alpha_d(nz.row) += (nz.y - 1.0) * eta1;
                            gamma_d(nz.row) += (nz.y - 1.0) * (1.0 - eta1);
                        }
                    }

                    // Solve quadratic per row
                    for (int i = 0; i < m; ++i) {
                        double alpha_i = alpha_d(i) + static_cast<double>(n_nz(i));
                        double beta_i = (sum_y_d(i) - sum_s_d(i)) - gamma_d(i) + alpha_i;

                        if (alpha_i > 1e-15) {
                            double disc = beta_i * beta_i + 4.0 * alpha_i * gamma_d(i);
                            if (disc > 0.0 && std::isfinite(disc)) {
                                double new_theta = (-beta_i + std::sqrt(disc)) / (2.0 * alpha_i);
                                if (std::isfinite(new_theta) && new_theta >= 0.0) {
                                    theta_vec(i) = static_cast<Scalar>(std::min(new_theta, cap));
                                }
                            }
                        }
                    }
                }

                // GLOBAL mode: average per-row thetas
                if (config.gp_dispersion == DispersionMode::GLOBAL) {
                    Scalar mean_theta = theta_vec.mean();
                    theta_vec.setConstant(mean_theta);
                }
            } else if (config.gp_dispersion == DispersionMode::PER_COL) {
                // PER_COL: same MM algorithm but accumulate per column (j)
                struct NzEntryCol { int col; double y; double s; int row; };
                std::vector<NzEntryCol> nz_cache_col;

                Eigen::VectorXd sum_y_col = Eigen::VectorXd::Zero(n);
                Eigen::VectorXd sum_s_col = Eigen::VectorXd::Zero(n);
                Eigen::VectorXi n_nz_col = Eigen::VectorXi::Zero(n);

                if constexpr (is_sparse_v<MatrixType>) {
                    DenseVector<Scalar> w_colsum = W_Td.colwise().sum().transpose();
                    for (int j = 0; j < n; ++j) {
                        sum_s_col(j) = static_cast<double>(W_Td.transpose().row(0).dot(H.col(j)));
                        // Approximate total sum_s per column via all-rows dot product
                        double total_s = 0;
                        for (int i = 0; i < m; ++i)
                            total_s += static_cast<double>(W_Td.col(i).dot(H.col(j)));
                        sum_s_col(j) = total_s;
                    }

                    nz_cache_col.reserve(A.nonZeros());
                    for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            double y = static_cast<double>(it.value());
                            double s = std::max(static_cast<double>(
                                W_Td.col(it.row()).dot(H.col(j))), 1e-10);
                            sum_y_col(j) += y;
                            if (y >= 1.0) n_nz_col(j)++;
                            nz_cache_col.push_back({j, y, s, static_cast<int>(it.row())});
                        }
                    }
                } else {
                    DenseMatrix<Scalar> S = W_Td.transpose() * H;
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < m; ++i) {
                            double y = static_cast<double>(A(i, j));
                            double s = std::max(static_cast<double>(S(i, j)), 1e-10);
                            sum_y_col(j) += y;
                            sum_s_col(j) += s;
                            if (y >= 1.0) n_nz_col(j)++;
                            if (y >= 1.0)
                                nz_cache_col.push_back({j, y, s, i});
                        }
                    }
                }

                double cap = static_cast<double>(config.gp_theta_max);
                constexpr int THETA_INNER_ITERS_COL = 5;
                for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS_COL; ++mm_iter) {
                    Eigen::VectorXd alpha_col = Eigen::VectorXd::Zero(n);
                    Eigen::VectorXd gamma_col = Eigen::VectorXd::Zero(n);

                    for (const auto& nz : nz_cache_col) {
                        if (nz.y >= 1.0) {
                            double th = static_cast<double>(theta_vec(nz.col));
                            double denom = std::max(nz.s + th * nz.y, 1e-10);
                            double eta1 = nz.s / denom;
                            alpha_col(nz.col) += (nz.y - 1.0) * eta1;
                            gamma_col(nz.col) += (nz.y - 1.0) * (1.0 - eta1);
                        }
                    }

                    for (int j = 0; j < n; ++j) {
                        double alpha_j = alpha_col(j) + static_cast<double>(n_nz_col(j));
                        double beta_j = (sum_y_col(j) - sum_s_col(j)) - gamma_col(j) + alpha_j;
                        if (alpha_j > 1e-15) {
                            double disc = beta_j * beta_j + 4.0 * alpha_j * gamma_col(j);
                            if (disc > 0.0 && std::isfinite(disc)) {
                                double new_theta = (-beta_j + std::sqrt(disc)) / (2.0 * alpha_j);
                                if (std::isfinite(new_theta) && new_theta >= 0.0)
                                    theta_vec(j) = static_cast<Scalar>(std::min(new_theta, cap));
                            }
                        }
                    }
                }
            }
            prof_timer.end();  // end GP dispersion
        }

        // ==============================================================
        // NB size (r) update — method of moments per row
        // ==============================================================
        // For NB: Var(Y) = μ + μ²/r,  so r_i = Σⱼ μᵢⱼ² / max(Σⱼ [(yᵢⱼ-μᵢⱼ)²-μᵢⱼ], ε)
        // If denominator ≤ 0, data is underdispersed → set r = nb_size_max.
        // Cost: O(nnz·k) for sparse (dot products), O(m) for reduction.
        if (is_nb && config.gp_dispersion != DispersionMode::NONE) {
            prof_timer.begin("dispersion");
            DenseMatrix<Scalar> W_Td_nb = W_T;
            detail::apply_scaling(W_Td_nb, d);

            double r_min = static_cast<double>(config.nb_size_min);
            double r_max = static_cast<double>(config.nb_size_max);

          if (config.gp_dispersion == DispersionMode::PER_COL) {
            // PER_COL: accumulate per column j instead of per row i
            Eigen::VectorXd sum_mu_sq_col = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd sum_excess_var_col = Eigen::VectorXd::Zero(n);

            if constexpr (is_sparse_v<MatrixType>) {
                for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                        double y = static_cast<double>(it.value());
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(it.row()).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        sum_mu_sq_col(j) += mu * mu;
                        sum_excess_var_col(j) += resid * resid - mu;
                    }
                    // Zero-entry contributions for column j
                    // For zero entries: (0-μ)²-μ = μ²-μ
                    // Need total Σᵢ μᵢⱼ² and Σᵢ μᵢⱼ for column j
                    double total_mu_col = 0, total_mu_sq_col = 0;
                    for (int i = 0; i < m; ++i) {
                        double mu = static_cast<double>(W_Td_nb.col(i).dot(H.col(j)));
                        total_mu_col += mu;
                        total_mu_sq_col += mu * mu;
                    }
                    double nz_mu_sum = 0, nz_mu_sq_sum = 0;
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(it.row()).dot(H.col(j))), 1e-10);
                        nz_mu_sum += mu;
                        nz_mu_sq_sum += mu * mu;
                    }
                    // Add zero-entry contributions
                    sum_mu_sq_col(j) += (total_mu_sq_col - nz_mu_sq_sum);
                    sum_excess_var_col(j) += (total_mu_sq_col - nz_mu_sq_sum) - (total_mu_col - nz_mu_sum);
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        double y = static_cast<double>(A(i, j));
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(i).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        sum_mu_sq_col(j) += mu * mu;
                        sum_excess_var_col(j) += resid * resid - mu;
                    }
                }
            }

            for (int j = 0; j < n; ++j) {
                if (sum_excess_var_col(j) > 1e-10 && sum_mu_sq_col(j) > 1e-10) {
                    double r_new = sum_mu_sq_col(j) / sum_excess_var_col(j);
                    r_new = std::max(r_min, std::min(r_new, r_max));
                    if (std::isfinite(r_new))
                        nb_size_vec(j) = static_cast<Scalar>(r_new);
                } else {
                    nb_size_vec(j) = static_cast<Scalar>(r_max);
                }
            }
          } else {
            // PER_ROW or GLOBAL: accumulate per row i

            // Per-row accumulators (fp64 for stability)
            Eigen::VectorXd sum_mu_sq = Eigen::VectorXd::Zero(m);
            Eigen::VectorXd sum_excess_var = Eigen::VectorXd::Zero(m);  // Σ[(y-μ)²-μ]

            if constexpr (is_sparse_v<MatrixType>) {
                // Sparse path: iterate nonzeros only
                // For zero entries: y=0, (0-μ)²-μ = μ²-μ = μ(μ-1)
                // Adding all-entry contributions: Σᵢ μ² = sum via Gram trick,
                // Σᵢ μ = via h_rowsum. Then subtract nonzero corrections.
                DenseVector<Scalar> h_rowsum = H.rowwise().sum();
                for (int i = 0; i < m; ++i) {
                    // Total Σⱼ μᵢⱼ via Gram trick
                    double total_mu = static_cast<double>(W_Td_nb.col(i).dot(h_rowsum));
                    // Total Σⱼ μᵢⱼ² via row of HHᵀ and row of W_Td
                    // μᵢⱼ = w_i · h_j, so Σⱼ μᵢⱼ² = w_i^T (Σⱼ hⱼhⱼ^T) w_i = w_i^T (H·Hᵀ) w_i
                    // But computing H·Hᵀ is O(k²n), which we'd need anyway.
                    // For large sparse data, store and reuse.
                    // For simplicity, use the Gram: Σⱼ μᵢⱼ² = w_i^T (HHᵀ) w_i
                    // where HHᵀ is the base Gram — but we want HHᵀ, not the one available.
                    // Actually: Σⱼ (w_i·h_j)² = Σⱼ (w_i^T h_j)² = w_i^T (Σⱼ h_j h_j^T) w_i = w_i^T G_H w_i
                    // This will be computed per-row below if needed. For now, sparse loop is more practical.
                    (void)total_mu;  // will accumulate via loop below
                }

                // Accumulate from nonzero entries  
                // Also need zero-entry contributions via Gram trick.
                // Practical approach: iterate nonzeros, use Gram for totals.
                Eigen::VectorXd sum_mu_nz = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd sum_mu_sq_nz = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd sum_resid_sq_nz = Eigen::VectorXd::Zero(m);

                for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                        int i = it.row();
                        double y = static_cast<double>(it.value());
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(i).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        sum_mu_nz(i) += mu;
                        sum_mu_sq_nz(i) += mu * mu;
                        sum_resid_sq_nz(i) += resid * resid;
                    }
                }

                // Gram trick for total sums: Σⱼ μᵢⱼ and Σⱼ μᵢⱼ²
                DenseVector<Scalar> h_rs = H.rowwise().sum();
                DenseMatrix<Scalar> G_H;
                primitives::gram<Resource>(H, G_H);  // k×k

                for (int i = 0; i < m; ++i) {
                    double total_mu = static_cast<double>(W_Td_nb.col(i).dot(h_rs));
                    double total_mu_sq = 0.0;
                    for (int a = 0; a < k; ++a)
                        for (int b = 0; b < k; ++b)
                            total_mu_sq += static_cast<double>(W_Td_nb(a, i)) *
                                           static_cast<double>(G_H(a, b)) *
                                           static_cast<double>(W_Td_nb(b, i));

                    // Zero-entry contributions: y=0, so (0-μ)²=μ², total for zeros = total_mu_sq - sum_mu_sq_nz
                    // Total Σⱼ(yᵢⱼ-μᵢⱼ)² = sum_resid_sq_nz + (total_mu_sq - sum_mu_sq_nz)
                    // Total Σⱼ μᵢⱼ = total_mu
                    double total_resid_sq = sum_resid_sq_nz(i) + (total_mu_sq - sum_mu_sq_nz(i));
                    sum_mu_sq(i) = total_mu_sq;
                    sum_excess_var(i) = total_resid_sq - total_mu;
                }
            } else {
                // Dense path: iterate all entries
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        double y = static_cast<double>(A(i, j));
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(i).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        sum_mu_sq(i) += mu * mu;
                        sum_excess_var(i) += resid * resid - mu;
                    }
                }
            }

            // Solve per row: r_i = Σ μ² / max(excess_var, ε)
            for (int i = 0; i < m; ++i) {
                if (sum_excess_var(i) > 1e-10 && sum_mu_sq(i) > 1e-10) {
                    double r_new = sum_mu_sq(i) / sum_excess_var(i);
                    r_new = std::max(r_min, std::min(r_new, r_max));
                    if (std::isfinite(r_new)) {
                        nb_size_vec(i) = static_cast<Scalar>(r_new);
                    }
                } else {
                    // Underdispersed or degenerate → Poisson limit
                    nb_size_vec(i) = static_cast<Scalar>(r_max);
                }
            }

            // GLOBAL mode: use median per-row r (median is robust to outliers)
            if (config.gp_dispersion == DispersionMode::GLOBAL) {
                std::vector<Scalar> r_vals(nb_size_vec.data(), nb_size_vec.data() + m);
                std::nth_element(r_vals.begin(), r_vals.begin() + m / 2, r_vals.end());
                Scalar median_r = r_vals[m / 2];
                nb_size_vec.setConstant(median_r);
            }
          }  // end PER_ROW/GLOBAL vs PER_COL
            prof_timer.end();  // end NB dispersion
        }

        // ==============================================================
        // ZIGP/ZINB E-step + M-step (zero-inflation estimation)
        // ==============================================================
        // After theta update, estimate dropout probabilities pi and
        // posterior z_ij for zero entries. The z_ij are used implicitly:
        // pi absorbs the structural-zero signal, leaving theta to model
        // only non-zero overdispersion. The theta update above uses the
        // full data; on the next iteration, the IRLS weights will
        // naturally downweight entries where the GP predicts zero well.
        //
        // For now, pi is estimated but not fed back into IRLS weights.
        // Future: multiply zero-entry IRLS weights by (1-z_ij).
        //
        // E-step: z_ij = pi_ij / (pi_ij + (1-pi_ij)*GP(0|s_ij, theta_i))
        // M-step: pi_i = (1/n) * sum_j z_ij * I(y_ij=0)
        //
        // Cost: O(m*n*k) for dense scan, O((m*n - nnz)*k) for sparse.
        // For 1K×500 k=5: ~2.5M FLOPs. For 33K×3K k=10: ~1B FLOPs.
        if (is_zi) {
            prof_timer.begin("zi_estep");
            // Reuse W_Td if already computed for theta, otherwise make it
            DenseMatrix<Scalar> W_Td_zi = W_T;
            detail::apply_scaling(W_Td_zi, d);

            for (int zi_iter = 0; zi_iter < config.zi_em_iters; ++zi_iter) {

                // Accumulate per-row and per-col sums of z_ij
                Eigen::VectorXd z_row_sum = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd z_col_sum = Eigen::VectorXd::Zero(n);
                Eigen::VectorXi zero_count_row = Eigen::VectorXi::Zero(m);
                Eigen::VectorXi zero_count_col = Eigen::VectorXi::Zero(n);

                // Thread-local accumulators to avoid omp atomic contention
                // on row-indexed vectors (threads process different columns
                // but accumulate into the same row indices)
                const int num_thr = eff_threads > 1 ? eff_threads : 1;
                std::vector<Eigen::VectorXd> z_row_local(num_thr, Eigen::VectorXd::Zero(m));
                std::vector<Eigen::VectorXi> zc_row_local(num_thr, Eigen::VectorXi::Zero(m));

                if constexpr (is_sparse_v<MatrixType>) {
                    // Sparse: iterate columns, track which rows have nonzeros
                    // For each column j, compute z_ij for rows NOT in A(:,j)
                    #pragma omp parallel for num_threads(eff_threads) schedule(dynamic, 16) if(eff_threads > 1)
                    for (int j = 0; j < n; ++j) {
                        #ifdef _OPENMP
                        const int tid = omp_get_thread_num();
                        #else
                        const int tid = 0;
                        #endif
                        // Build set of nonzero rows in this column
                        std::vector<bool> is_nz(m, false);
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            is_nz[it.row()] = true;
                        }

                        double z_col_local = 0.0;
                        int zc_local = 0;

                        for (int i = 0; i < m; ++i) {
                            if (!is_nz[i]) {
                                // This is a zero entry
                                double s_ij = std::max(static_cast<double>(
                                    W_Td_zi.col(i).dot(H.col(j))), 1e-10);
                                // P(Y=0|μ) depends on distribution
                                double p0;
                                if (is_nb) {
                                    double r_i = std::max(static_cast<double>(nb_size_vec(i)), 1e-10);
                                    // NB(0|μ,r) = (r/(r+μ))^r
                                    p0 = std::pow(r_i / (r_i + s_ij), r_i);
                                } else {
                                    double th_i = static_cast<double>(theta_vec(i));
                                    // GP(0|s,theta) = exp(-s/(1+theta))
                                    double otp = 1.0 + th_i;
                                    p0 = std::exp(-s_ij / otp);
                                }

                                // Get pi for this entry
                                double pi_ij, pr_val = 0.0, pc_val = 0.0;
                                if (config.zi_mode == ZIMode::ZI_ROW) {
                                    pi_ij = static_cast<double>(pi_row_vec(i));
                                } else if (config.zi_mode == ZIMode::ZI_COL) {
                                    pi_ij = static_cast<double>(pi_col_vec(j));
                                } else { // TWOWAY
                                    pr_val = static_cast<double>(pi_row_vec(i));
                                    pc_val = static_cast<double>(pi_col_vec(j));
                                    pi_ij = 1.0 - (1.0 - pr_val) * (1.0 - pc_val);
                                }

                                // z_ij = pi / (pi + (1-pi)*p0)
                                double z_ij = pi_ij / (pi_ij + (1.0 - pi_ij) * p0 + 1e-300);

                                // Thread-local accumulation
                                // Twoway: correct EM attribution — accumulate
                                // z_ij * pr/pi_ij for row, z_ij * pc/pi_ij for col
                                // This prevents double-counting in the compound model
                                if (config.zi_mode == ZIMode::ZI_TWOWAY) {
                                    double inv_pi = 1.0 / std::max(pi_ij, 1e-10);
                                    z_row_local[tid](i) += z_ij * pr_val * inv_pi;
                                    z_col_local += z_ij * pc_val * inv_pi;
                                } else {
                                    z_row_local[tid](i) += z_ij;
                                    z_col_local += z_ij;
                                }
                                zc_row_local[tid](i) += 1;
                                zc_local++;
                            }
                        }
                        // Write column sums (j is loop-private)
                        z_col_sum(j) = z_col_local;
                        zero_count_col(j) = zc_local;
                    }
                } else {
                    // Dense path: iterate all entries
                    #pragma omp parallel for num_threads(eff_threads) schedule(dynamic, 16) if(eff_threads > 1)
                    for (int j = 0; j < n; ++j) {
                        #ifdef _OPENMP
                        const int tid = omp_get_thread_num();
                        #else
                        const int tid = 0;
                        #endif
                        double z_col_local = 0.0;
                        int zc_local = 0;
                        for (int i = 0; i < m; ++i) {
                            if (static_cast<double>(A.coeff(i, j)) == 0.0) {
                                double s_ij = std::max(static_cast<double>(
                                    W_Td_zi.col(i).dot(H.col(j))), 1e-10);
                                double p0;
                                if (is_nb) {
                                    double r_i = std::max(static_cast<double>(nb_size_vec(i)), 1e-10);
                                    p0 = std::pow(r_i / (r_i + s_ij), r_i);
                                } else {
                                    double th_i = static_cast<double>(theta_vec(i));
                                    double otp = 1.0 + th_i;
                                    p0 = std::exp(-s_ij / otp);
                                }

                                double pi_ij, pr_val = 0.0, pc_val = 0.0;
                                if (config.zi_mode == ZIMode::ZI_ROW) {
                                    pi_ij = static_cast<double>(pi_row_vec(i));
                                } else if (config.zi_mode == ZIMode::ZI_COL) {
                                    pi_ij = static_cast<double>(pi_col_vec(j));
                                } else {
                                    pr_val = static_cast<double>(pi_row_vec(i));
                                    pc_val = static_cast<double>(pi_col_vec(j));
                                    pi_ij = 1.0 - (1.0 - pr_val) * (1.0 - pc_val);
                                }

                                double z_ij = pi_ij / (pi_ij + (1.0 - pi_ij) * p0 + 1e-300);

                                if (config.zi_mode == ZIMode::ZI_TWOWAY) {
                                    double inv_pi = 1.0 / std::max(pi_ij, 1e-10);
                                    z_row_local[tid](i) += z_ij * pr_val * inv_pi;
                                    z_col_local += z_ij * pc_val * inv_pi;
                                } else {
                                    z_row_local[tid](i) += z_ij;
                                    z_col_local += z_ij;
                                }
                                zc_row_local[tid](i) += 1;
                                zc_local++;
                            }
                        }
                        z_col_sum(j) = z_col_local;
                        zero_count_col(j) = zc_local;
                    }
                }

                // Reduce thread-local row accumulators
                for (int t = 0; t < num_thr; ++t) {
                    z_row_sum += z_row_local[t];
                    zero_count_row += zc_row_local[t];
                }

                // M-step: update pi
                if (config.zi_mode == ZIMode::ZI_ROW || config.zi_mode == ZIMode::ZI_TWOWAY) {
                    for (int i = 0; i < m; ++i) {
                        if (zero_count_row(i) > 0) {
                            double new_pi = z_row_sum(i) / static_cast<double>(n);
                            if (config.zi_mode == ZIMode::ZI_TWOWAY) {
                                // EMA damping for smooth convergence on high-sparsity data
                                double old_pi = static_cast<double>(pi_row_vec(i));
                                new_pi = 0.9 * new_pi + 0.1 * old_pi;
                                new_pi = std::clamp(new_pi, 0.001, 0.95);
                            } else {
                                new_pi = std::clamp(new_pi, 0.001, 0.999);
                            }
                            pi_row_vec(i) = static_cast<Scalar>(new_pi);
                        }
                    }
                }
                if (config.zi_mode == ZIMode::ZI_COL || config.zi_mode == ZIMode::ZI_TWOWAY) {
                    for (int j = 0; j < n; ++j) {
                        if (zero_count_col(j) > 0) {
                            double new_pi = z_col_sum(j) / static_cast<double>(m);
                            if (config.zi_mode == ZIMode::ZI_TWOWAY) {
                                double old_pi = static_cast<double>(pi_col_vec(j));
                                new_pi = 0.9 * new_pi + 0.1 * old_pi;
                                new_pi = std::clamp(new_pi, 0.001, 0.95);
                            } else {
                                new_pi = std::clamp(new_pi, 0.001, 0.999);
                            }
                            pi_col_vec(j) = static_cast<Scalar>(new_pi);
                        }
                    }
                }

                // Apply theta floor to prevent collapse
                if (config.gp_theta_min > 0) {
                    for (int i = 0; i < m; ++i) {
                        if (theta_vec(i) < config.gp_theta_min)
                            theta_vec(i) = config.gp_theta_min;
                    }
                }
            }  // end zi_em_iters

            // ----------------------------------------------------------
            // Soft imputation: update A_imputed for next ALS iteration
            // ----------------------------------------------------------
            // For zero entries in the original A, set:
            //   A_imputed[i,j] = z_ij * mu_ij
            // where z_ij is the posterior dropout probability and
            // mu_ij = W_Td.col(i).dot(H.col(j)) is the predicted value.
            // This makes structural zeros "invisible" to the loss
            // (loss(z*mu, mu) ≈ 0 for z ≈ 1), while real zeros stay at 0.
            {
                // Reset A_imputed from original data
                if constexpr (is_sparse_v<MatrixType>) {
                    A_imputed.setZero();
                    for (int j = 0; j < n; ++j)
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                            A_imputed(it.row(), j) = static_cast<Scalar>(it.value());
                } else {
                    A_imputed = A;
                }

                // Impute zero entries with z_ij * mu_ij
                #pragma omp parallel for num_threads(eff_threads) schedule(dynamic, 16) if(eff_threads > 1)
                for (int j = 0; j < n; ++j) {
                    // Build nonzero set for this column (sparse path)
                    std::vector<bool> is_nz_imp;
                    if constexpr (is_sparse_v<MatrixType>) {
                        is_nz_imp.assign(m, false);
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                            is_nz_imp[it.row()] = true;
                    }

                    for (int i = 0; i < m; ++i) {
                        bool is_zero;
                        if constexpr (is_sparse_v<MatrixType>) {
                            is_zero = !is_nz_imp[i];
                        } else {
                            is_zero = (static_cast<double>(A.coeff(i, j)) == 0.0);
                        }

                        if (is_zero) {
                            double s_ij = std::max(static_cast<double>(
                                W_Td_zi.col(i).dot(H.col(j))), 1e-10);
                            double p0;
                            if (is_nb) {
                                double r_i = std::max(static_cast<double>(nb_size_vec(i)), 1e-10);
                                p0 = std::pow(r_i / (r_i + s_ij), r_i);
                            } else {
                                double th_i = static_cast<double>(theta_vec(i));
                                double otp = 1.0 + th_i;
                                p0 = std::exp(-s_ij / otp);
                            }

                            double pi_ij;
                            if (config.zi_mode == ZIMode::ZI_ROW) {
                                pi_ij = static_cast<double>(pi_row_vec(i));
                            } else if (config.zi_mode == ZIMode::ZI_COL) {
                                pi_ij = static_cast<double>(pi_col_vec(j));
                            } else { // TWOWAY
                                double pr = static_cast<double>(pi_row_vec(i));
                                double pc = static_cast<double>(pi_col_vec(j));
                                pi_ij = 1.0 - (1.0 - pr) * (1.0 - pc);
                            }

                            double z_ij = pi_ij / (pi_ij + (1.0 - pi_ij) * p0 + 1e-300);
                            // Soft impute: structural zeros → predicted value
                            A_imputed(i, j) = static_cast<Scalar>(z_ij * s_ij);
                        }
                    }
                }
            }
            prof_timer.end();  // end zi_estep
        }  // end ZIGP

        // ==============================================================
        // Gamma / InvGauss dispersion (φ) update — MoM Pearson estimator
        // ==============================================================
        // For Gamma:    V(μ) = φ·μ², Pearson: φ̂_i = (1/n_i) Σ_j (y_ij-μ_ij)²/μ_ij²
        // For InvGauss: V(μ) = φ·μ³, Pearson: φ̂_i = (1/n_i) Σ_j (y_ij-μ_ij)²/μ_ij³
        // Note: IRLS weights (1/μ^p) don't depend on φ, so this is purely
        // diagnostic — it doesn't affect the W/H updates.
        if ((is_gamma || is_invgauss || is_tweedie) && config.gp_dispersion != DispersionMode::NONE) {
            prof_timer.begin("dispersion");
            DenseMatrix<Scalar> W_Td_phi = W_T;
            detail::apply_scaling(W_Td_phi, d);

            const double var_power = is_tweedie ? static_cast<double>(config.loss.power_param)
                                   : (is_gamma ? 2.0 : 3.0);
            double phi_min = static_cast<double>(config.gamma_phi_min);
            double phi_max = static_cast<double>(config.gamma_phi_max);

          if (config.gp_dispersion == DispersionMode::PER_COL) {
            // PER_COL: accumulate per column j
            Eigen::VectorXd sum_pearson_sq_col = Eigen::VectorXd::Zero(n);
            Eigen::VectorXi count_col = Eigen::VectorXi::Zero(n);

            if constexpr (is_sparse_v<MatrixType>) {
                for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                        double y = static_cast<double>(it.value());
                        if (y <= 0.0) continue;
                        double mu = std::max(static_cast<double>(
                            W_Td_phi.col(it.row()).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        double v_mu = std::pow(mu, var_power);
                        sum_pearson_sq_col(j) += (resid * resid) / std::max(v_mu, 1e-20);
                        count_col(j)++;
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        double y = static_cast<double>(A(i, j));
                        if (y <= 0.0) continue;
                        double mu = std::max(static_cast<double>(
                            W_Td_phi.col(i).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        double v_mu = std::pow(mu, var_power);
                        sum_pearson_sq_col(j) += (resid * resid) / std::max(v_mu, 1e-20);
                        count_col(j)++;
                    }
                }
            }

            for (int j = 0; j < n; ++j) {
                if (count_col(j) > 0) {
                    double phi_new = sum_pearson_sq_col(j) / static_cast<double>(count_col(j));
                    phi_new = std::max(phi_min, std::min(phi_new, phi_max));
                    if (std::isfinite(phi_new))
                        phi_vec(j) = static_cast<Scalar>(phi_new);
                }
            }
          } else {
            // PER_ROW or GLOBAL: accumulate per row i

            Eigen::VectorXd sum_pearson_sq = Eigen::VectorXd::Zero(m);
            Eigen::VectorXi count_vec = Eigen::VectorXi::Zero(m);

            if constexpr (is_sparse_v<MatrixType>) {
                // Sparse: iterate nonzero entries only
                // For Gamma/IG, zero entries are degenerate (y=0 is outside
                // the support), so we only use positive entries.
                for (int j = 0; j < static_cast<int>(A.cols()); ++j) {
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                        int i = it.row();
                        double y = static_cast<double>(it.value());
                        if (y <= 0.0) continue;  // Gamma/IG require y > 0
                        double mu = std::max(static_cast<double>(
                            W_Td_phi.col(i).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        // Pearson residual squared / V(μ) = (y-μ)² / μ^p
                        double v_mu = std::pow(mu, var_power);
                        sum_pearson_sq(i) += (resid * resid) / std::max(v_mu, 1e-20);
                        count_vec(i)++;
                    }
                }
            } else {
                // Dense: iterate all entries, skip non-positive
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        double y = static_cast<double>(A(i, j));
                        if (y <= 0.0) continue;
                        double mu = std::max(static_cast<double>(
                            W_Td_phi.col(i).dot(H.col(j))), 1e-10);
                        double resid = y - mu;
                        double v_mu = std::pow(mu, var_power);
                        sum_pearson_sq(i) += (resid * resid) / std::max(v_mu, 1e-20);
                        count_vec(i)++;
                    }
                }
            }

            // Solve per row: φ_i = Σ pearson² / count
            for (int i = 0; i < m; ++i) {
                if (count_vec(i) > 0) {
                    double phi_new = sum_pearson_sq(i) / static_cast<double>(count_vec(i));
                    phi_new = std::max(phi_min, std::min(phi_new, phi_max));
                    if (std::isfinite(phi_new)) {
                        phi_vec(i) = static_cast<Scalar>(phi_new);
                    }
                }
            }

            // GLOBAL mode: use median per-row φ (robust to outlier rows)
            if (config.gp_dispersion == DispersionMode::GLOBAL) {
                std::vector<Scalar> phi_vals(phi_vec.data(), phi_vec.data() + m);
                std::nth_element(phi_vals.begin(), phi_vals.begin() + m / 2, phi_vals.end());
                Scalar median_phi = phi_vals[m / 2];
                phi_vec.setConstant(median_phi);
            }
          }  // end PER_ROW/GLOBAL vs PER_COL
            prof_timer.end();  // end Gamma/IG/Tweedie dispersion
        }

        // ==============================================================
        // Convergence check
        // ==============================================================

        _t_loss = std::chrono::high_resolution_clock::now();
        prof_timer.begin("loss");

        bool loss_converged = false;

        // --- Compute loss when needed (modes 0/2, or tracking) ---
        if (need_loss && compute_loss_this_iter) {
            Scalar loss_val;
            if (use_mask) {
                // Masked loss: explicit computation skipping masked entries
                DenseMatrix<Scalar> W_Td_loss = W_T;
                detail::apply_scaling(W_Td_loss, d);
                loss_val = mask_detail::masked_loss(
                    A, W_Td_loss, H, *config.mask, config.loss, eff_threads);
            } else if (config.requires_irls()) {
                // Non-MSE loss: explicit per-element computation
                DenseMatrix<Scalar> W_Td_loss = W_T;
                detail::apply_scaling(W_Td_loss, d);
                const Scalar* dispersion_ptr = is_gp ? theta_vec.data()
                                             : (is_nb ? nb_size_vec.data()
                                             : ((is_gamma || is_invgauss || is_tweedie) ? phi_vec.data()
                                             : nullptr));
                const bool disp_is_per_col = (config.gp_dispersion == DispersionMode::PER_COL);
                if constexpr (is_sparse_v<MatrixType>) {
                    loss_val = detail::explicit_loss_sparse(
                        A, W_Td_loss, H, config.loss, config.threads,
                        dispersion_ptr, disp_is_per_col);
                } else {
                    loss_val = detail::explicit_loss_dense(
                        A, W_Td_loss, H, config.loss,
                        dispersion_ptr, disp_is_per_col);
                }
            } else if (saved_for_loss && !used_fused_w) {
                // MSE: Optimized Gram trick reusing W-update matrices.
                // B_w_saved was materialized in the standard path.
                // Gram of new W_T: O(k²·m)
                DenseMatrix<Scalar> G_wt;
                primitives::gram<Resource>(W_T, G_wt);

                // Cross term: Σᵢ dᵢ · dot(W_T.row(i), B_w_saved.row(i)): O(k·m)
                Scalar cross_term = 0;
                for (int i = 0; i < k; ++i)
                    cross_term += d(i) * W_T.row(i).dot(B_w_saved.row(i));

                // Recon norm: Σᵢⱼ dᵢ·dⱼ · G_wt(i,j) · G_w_saved(i,j): O(k²)
                Scalar recon_norm = 0;
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)
                        recon_norm += d(i) * d(j) * G_wt(i, j) * G_w_saved(i, j);

                loss_val = trAtA - static_cast<Scalar>(2) * cross_term + recon_norm;
            } else if (saved_for_loss && used_fused_w) {
                // MSE: Fused path — B_w was never materialized.
                // Use sparse cross-term computation: O(nnz·k) + O(k²·m).

                // Gram of new W_T: O(k²·m)
                DenseMatrix<Scalar> G_wt;
                primitives::gram<Resource>(W_T, G_wt);

                // Cross term via sparse traversal of A^T: O(nnz·k)
                Scalar cross_term;
                if constexpr (is_sparse_v<MatrixType>) {
                    cross_term = primitives::loss_cross_term_sparse_via_At(
                        *At_ptr, W_T, H, d, eff_threads);
                } else {
                    // Dense fallback (shouldn't reach here since can_fuse requires sparse)
                    cross_term = 0;
                }

                // Recon norm: Σᵢⱼ dᵢ·dⱼ · G_wt(i,j) · G_w_saved(i,j): O(k²)
                Scalar recon_norm = 0;
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)
                        recon_norm += d(i) * d(j) * G_wt(i, j) * G_w_saved(i, j);

                loss_val = trAtA - static_cast<Scalar>(2) * cross_term + recon_norm;
            } else {
                // MSE fallback: full recomputation
                DenseMatrix<Scalar> W_Td_loss = W_T;
                detail::apply_scaling(W_Td_loss, d);
                DenseMatrix<Scalar> G_loss;
                primitives::gram<Resource>(W_Td_loss, G_loss);
                DenseMatrix<Scalar> B_loss;
                primitives::rhs<Resource>(A, W_Td_loss, B_loss);
                loss_val = detail::gram_trick_loss(
                    trAtA, G_loss, B_loss, H);
            }

            if (config.track_loss_history)
                result.loss_history.push_back(loss_val);

            Scalar rel_change = 0;
            if (iter > 0) {
                rel_change = std::abs(prev_loss - loss_val)
                                  / (std::abs(prev_loss) + static_cast<Scalar>(1e-15));
                result.final_tol = rel_change;
                if (rel_change < config.tol) loss_converged = true;
            }
            prev_loss = loss_val;

            // Record FitHistory snapshot
            if (config.track_loss_history) {
                double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - fit_start).count();
                Scalar nan_val = std::numeric_limits<Scalar>::quiet_NaN();
                result.history.push(iter + 1, loss_val, nan_val, rel_change, elapsed_ms);
            }

            // Fire per-iteration callback
            if (config.on_iteration) {
                Scalar nan_val = std::numeric_limits<Scalar>::quiet_NaN();
                config.on_iteration(iter + 1, loss_val, nan_val);
            }
        }
        prof_timer.end();  // end loss

        // --- Apply patience ---
        bool iter_converged = loss_converged;

        if (iter > 0) {
            if (iter_converged) {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    result.converged = true;
                    result.train_loss = prev_loss;
                    result.iterations = iter + 1;
                    break;
                }
            } else {
                patience_counter = 0;
            }
        }

        result.iterations = iter + 1;

        // --- Timer report ---
        _t_end = std::chrono::high_resolution_clock::now();
        if (config.verbose) {
            auto us = [](auto a, auto b) { return static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count()); };
            Rprintf("[iter %d] gram_h=%dus rhs_h=%dus nnls_h=%dus norm_h=%dus | "
                    "gram_w=%dus rhs_w=%dus nnls_w=%dus norm_w=%dus | loss=%dus | total=%dus\n",
                    iter, us(_t_gram_h, _t_rhs_h), us(_t_rhs_h, _t_nnls_h),
                    us(_t_nnls_h, _t_norm_h), us(_t_norm_h, _t_gram_w),
                    us(_t_gram_w, _t_rhs_w), us(_t_rhs_w, _t_nnls_w),
                    us(_t_nnls_w, _t_norm_w), us(_t_norm_w, _t_loss),
                    us(_t_loss, _t_end), us(_t0, _t_end));
        }
    }

    // ------------------------------------------------------------------
    // 5. Package result
    // ------------------------------------------------------------------

    result.W = W_T.transpose();  // k×m → m×k
    result.d = d;
    result.H = H;
    if (is_gp) result.theta = theta_vec;
    if (is_nb) result.theta = nb_size_vec;
    if (is_gamma || is_invgauss || is_tweedie) result.dispersion = phi_vec;
    if (is_zi) {
        if (pi_row_vec.size() > 0) result.pi_row = pi_row_vec;
        if (pi_col_vec.size() > 0) result.pi_col = pi_col_vec;
    }

    if (result.train_loss == 0 && !result.loss_history.empty())
        result.train_loss = result.loss_history.back();
    else if (result.train_loss == 0)
        result.train_loss = prev_loss;

    if (config.sort_model)
        result.sort();

    // Store profiling data if active
    if (config.enable_profiling)
        result.profile = prof_timer.results();

    return result;
}

}  // namespace nmf
}  // namespace FactorNet

