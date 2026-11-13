// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit_cv.hpp
 * @brief Cross-validation NMF using the unified architecture
 *
 * CV NMF differs from standard NMF in one key way: each column j has a
 * per-column set of held-out rows, requiring a Gram correction:
 *
 *   G_local = G_full - W_test · W_test^T
 *
 * where W_test are the rows of W corresponding to held-out entries in col j.
 * This means we CANNOT use the batch nnls_batch primitive — we must solve
 * each column individually with its corrected Gram matrix.
 *
 * The mask is a LazySpeckledMask: O(1) per-entry queries via deterministic
 * PRNG hash. No mask matrix is ever materialized.
 *
 * Loss computation:
 *   - Test loss: explicit reconstruction at held-out entries only
 *   - Train loss: only when requested (track_train_loss or verbose), NaN
 *     otherwise. Gram trick (total minus test) for squared error on all
 *     entries; entry-wise for every other case.
 *
 * Conventions:
 *   - Returned factors follow the non-CV fit: unit-norm W columns and H rows
 *     with the scale in d, so that A ≈ W · diag(d) · H.
 *   - mask_zeros = true means zeros are missing, for every loss.
 *   - With cv_patience > 0 the factors of the best test iteration are
 *     returned; test_loss, best_test_loss, train_loss and best_iter all
 *     describe that returned model.
 *   - Convergence (tol) is judged on 1 - cosine similarity of successive W,
 *     not on the test loss.
 *   - Penalties (L2, graph, L21, positive target) are built by
 *     variant::build_column_penalty(), the routine the non-CV fit uses, from
 *     the previous iterate before the parallel solves start. The upper bound
 *     is enforced inside the per-column solver.
 *   - The IRLS solves warm-start and use the same step control as the
 *     non-CV IRLS solver.
 *
 * IRLS support: For non-MSE losses, per-column/row IRLS loops build weighted
 * Gram and RHS from train entries only, with weights evaluated at the
 * observed value and the fitted dispersion.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/core/constants.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/core/memory.hpp>

// Primitives (for full Gram computation + tr(A^T A))
#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>
#include <FactorNet/primitives/cpu/loss.hpp>

// Features (same as standard NMF — applied to G before correction)
#include <FactorNet/features/sparsity.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>

// Shared algorithm variant dispatch and helpers
#include <FactorNet/nmf/variant_helpers.hpp>

// CD solver for per-column NNLS
#include <FactorNet/primitives/cpu/nnls_batch.hpp>

// Shared IRLS weight computation (replaces local cv_irls_weight)
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>

// CV helper functions (Gram correction, IRLS, RHS, user mask)
#include <FactorNet/nmf/cv_detail.hpp>

// Cholesky solvers for per-column NNLS (solver_mode 1 & 2)
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>

// SVD (for Lanczos and IRLBA initialization)
#include <FactorNet/svd/lanczos.hpp>
#include <FactorNet/svd/irlba.hpp>

// Lazy mask
#include <FactorNet/nmf/speckled_cv.hpp>

// RNG for initialization
#include <FactorNet/rng/rng.hpp>

#include <cmath>
#include <cstdint>
#include <chrono>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace nmf {


// ========================================================================
// Main CV NMF fitting function
// ========================================================================

/**
 * @brief CV NMF fitting — per-column Gram correction with lazy mask
 *
 * Template parameter Resource selects CPU or GPU primitives (for full Gram).
 * Template parameter MatrixType selects sparse or dense A.
 *
 * Key difference from nmf_fit: each column has per-column Gram correction
 * so batch NNLS cannot be used. Instead, individual CD solves are run
 * with corrected G_local per column.
 *
 * @tparam Resource    primitives::CPU or primitives::GPU
 * @tparam Scalar      float or double
 * @tparam MatrixType  SparseMatrix<Scalar>, MappedSparseMatrix<Scalar>, or DenseMatrix<Scalar>
 *
 * @param A       Data matrix (m × n)
 * @param config  NMF configuration (with CV fields set)
 * @param W_init  Optional initialization for W (m × k). nullptr = random.
 * @param H_init  Optional initialization for H (k × n). nullptr = random.
 * @return        NMFResult with W, d, H, train+test loss, early stopping info
 */
template<typename Resource, typename Scalar, typename MatrixType>
::FactorNet::NMFResult<Scalar> nmf_fit_cv(
    const MatrixType& A,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.rank;

    // ------------------------------------------------------------------
    // 0. Reject configurations the CV path cannot honor
    // ------------------------------------------------------------------

    if (config.projective) {
        throw std::invalid_argument(
            "Projective NMF is not supported with cross-validation "
            "(test_fraction > 0): the projective H update H = W^T A uses every "
            "entry of A, including the held-out ones. Fit without "
            "test_fraction, or use standard NMF for cross-validation.");
    }
    if ((config.H.target != nullptr && config.H.target_lambda < 0) ||
        (config.W.target != nullptr && config.W.target_lambda < 0)) {
        throw std::invalid_argument(
            "A negative target_lambda (target removal) is not supported with "
            "cross-validation (test_fraction > 0). Use a positive "
            "target_lambda or fit without test_fraction.");
    }
    // Positive target: ridge toward the target, G += lambda*I, b += lambda*T,
    // as in the standard fit.
    const bool h_target = (config.H.target != nullptr && config.H.target_lambda > 0);
    const bool w_target = (config.W.target != nullptr && config.W.target_lambda > 0);
    if (h_target && (config.H.target->rows() != k || config.H.target->cols() != n)) {
        throw std::invalid_argument(
            "Cross-validation: the H target must be a k x ncol(A) matrix.");
    }
    if (w_target && (config.W.target->rows() != k || config.W.target->cols() != m)) {
        throw std::invalid_argument(
            "Cross-validation: the W target must be a k x nrow(A) matrix.");
    }
    // Same restriction as the non-CV fit, so that both penalize one objective
    if (config.requires_irls() && (config.W.L21 > 0 || config.H.L21 > 0)) {
        throw std::invalid_argument(
            "L21 is not supported together with a non-Gaussian or robust loss "
            "(with or without cross-validation). Use loss = \"mse\" without robust.");
    }
    if (config.H.graph && config.H.graph_lambda > 0 && !config.symmetric)
        features::validate_graph(*config.H.graph, n, "graph_H");
    if (config.W.graph && config.W.graph_lambda > 0)
        features::validate_graph(*config.W.graph, m, "graph_W");

    // ------------------------------------------------------------------
    // 1. Initialize factors
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> W_T, H;
    DenseVector<Scalar> d;

    if (W_init) {
        W_T = W_init->transpose();  // m×k → k×m
        d = DenseVector<Scalar>::Ones(k);

        if (H_init) {
            H = *H_init;
        } else {
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
        rng::SplitMix64 rng(config.seed);
        W_T.resize(k, m);
        rng.fill_uniform(W_T.data(), W_T.rows(), W_T.cols());
        H.resize(k, n);
        rng.fill_uniform(H.data(), H.rows(), H.cols());
        d = DenseVector<Scalar>::Ones(k);
    }

    // ------------------------------------------------------------------
    // 2. Build lazy mask
    // ------------------------------------------------------------------

    // Compute nnz: sparse uses .nonZeros(), dense counts non-zero entries
    int64_t nnz;
    if constexpr (is_sparse_v<MatrixType>) {
        nnz = A.nonZeros();
    } else {
        nnz = (A.array() != Scalar(0)).count();
    }

    FactorNet::nmf::LazySpeckledMask<Scalar> mask(
        m, n, nnz,
        static_cast<double>(config.holdout_fraction),
        config.effective_cv_seed(),
        config.mask_zeros);
    // Data outside the support of the loss is rejected, as in the non-CV fit.
    // Zeros only count when they are data (no mask of any kind).
    if (config.requires_irls()) {
        bool any_negative = false;
        bool any_zero = false;
        if constexpr (is_sparse_v<MatrixType>) {
            for (int j = 0; j < n; ++j) {
                for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                    if (it.value() < 0) any_negative = true;
                    else if (it.value() == 0) any_zero = true;
                }
            }
            if (nnz < static_cast<int64_t>(m) * static_cast<int64_t>(n)) any_zero = true;
        } else {
            any_negative = (A.array() < Scalar(0)).any();
            any_zero = (nnz < static_cast<int64_t>(m) * static_cast<int64_t>(n));
        }
        if (config.mask_zeros || config.has_mask()) any_zero = false;
        primitives::detail::irls_check_support(config.loss, any_negative, any_zero);
    }

    // Symmetric NMF fits a symmetric matrix: (i,j) and (j,i) hold the same
    // value, so they must be held out together or the test entry leaks.
    mask.set_symmetric(config.symmetric);

    // ------------------------------------------------------------------
    // 2b. Initialize theta for GP dispersion (CV path)
    // ------------------------------------------------------------------
    const bool is_gp = (config.loss.type == LossType::GP);
    const bool is_nb = (config.loss.type == LossType::NB);
    const bool is_gamma = (config.loss.type == LossType::GAMMA);
    const bool is_invgauss = (config.loss.type == LossType::INVGAUSS);
    const bool is_tweedie = (config.loss.type == LossType::TWEEDIE);
    DenseVector<Scalar> theta_vec;
    if (is_gp) {
        if (config.gp_dispersion == DispersionMode::PER_ROW ||
            config.gp_dispersion == DispersionMode::GLOBAL) {
            theta_vec = DenseVector<Scalar>::Constant(m, config.gp_theta_init);
        } else {
            theta_vec = DenseVector<Scalar>::Zero(m);
        }
    }

    // ------------------------------------------------------------------
    // 2c. Initialize NB size (inverse-dispersion r) vector
    // ------------------------------------------------------------------
    DenseVector<Scalar> nb_size_vec;
    if (is_nb) {
        if (config.gp_dispersion == DispersionMode::PER_ROW) {
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            nb_size_vec = DenseVector<Scalar>::Constant(n, config.nb_size_init);
        } else if (config.gp_dispersion == DispersionMode::GLOBAL) {
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        } else {
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_max);
        }
    }

    // ------------------------------------------------------------------
    // 2d. Initialize dispersion φ for Gamma / InvGauss / Tweedie
    // ------------------------------------------------------------------
    DenseVector<Scalar> phi_vec;
    if (is_gamma || is_invgauss || is_tweedie) {
        if (config.gp_dispersion == DispersionMode::PER_ROW) {
            phi_vec = DenseVector<Scalar>::Constant(m, config.gamma_phi_init);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            phi_vec = DenseVector<Scalar>::Constant(n, config.gamma_phi_init);
        } else if (config.gp_dispersion == DispersionMode::GLOBAL) {
            phi_vec = DenseVector<Scalar>::Constant(m, config.gamma_phi_init);
        } else {
            phi_vec = DenseVector<Scalar>::Constant(m, static_cast<Scalar>(1));
        }
    }

    // ------------------------------------------------------------------
    // 2e. Initialize ZI (zero-inflation) state
    // ------------------------------------------------------------------
    const bool is_zi = (is_gp || is_nb) && config.has_zi();
    DenseVector<Scalar> pi_row_vec, pi_col_vec;
    DenseMatrix<Scalar> A_imputed;
    if (is_zi) {
        // Data-driven pi init: min(zero_rate * 0.5, 0.3)
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
                    for (int j2 = 0; j2 < n; ++j2)
                        if (static_cast<double>(A(i, j2)) == 0.0) nz++;
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
                    for (int i2 = 0; i2 < m; ++i2)
                        if (static_cast<double>(A(i2, j)) == 0.0) nz++;
                    double zero_rate = static_cast<double>(nz) / m;
                    pi_col_vec(j) = static_cast<Scalar>(std::min(zero_rate * 0.5, 0.3));
                }
            }
        }

        // Dense imputation matrix: starts as copy of A
        A_imputed.resize(m, n);
        if constexpr (is_sparse_v<MatrixType>) {
            A_imputed.setZero();
            for (int j = 0; j < n; ++j)
                for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                    A_imputed(it.row(), j) = static_cast<Scalar>(it.value());
        } else {
            A_imputed = A;
        }
    }

    // ------------------------------------------------------------------
    // 3. Transpose A for W update (sparse only)
    // ------------------------------------------------------------------

    // For sparse matrices, transpose into CSC for efficient column iteration.
    // Dense matrices don't need a transpose — we access A(i, col) directly.
    [[maybe_unused]] SparseMatrix<Scalar> A_T;
    if constexpr (is_sparse_v<MatrixType>) {
        // Memory safety check: ensure we have enough RAM for the transpose
        const size_t nnz = static_cast<size_t>(A.nonZeros());
        auto mem_check = FactorNet::memory::check_transpose_memory<Scalar>(nnz, m, n);

        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, config.verbose,
            "CV Transpose: %s\n", mem_check.message.c_str());

        if (!mem_check.fits) {
            throw std::runtime_error(
                std::string("Cannot compute in-memory transpose of sparse matrix.\n") +
                mem_check.message);
        }

        A_T = SparseMatrix<Scalar>(A.template cast<Scalar>()).transpose();
    }

    // ------------------------------------------------------------------
    // 3b. User-supplied mask precomputation
    // ------------------------------------------------------------------

    const bool use_mask = config.has_mask();
    [[maybe_unused]] SparseMatrix<Scalar> mask_T;
    if (use_mask) {
        mask_T = config.mask->transpose();
    }

    // ------------------------------------------------------------------
    // 4. Working state
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> G(k, k);

    // Initial width of the per-thread index buffers. They grow on demand, so
    // this is a starting size only, never a limit.
    const int est_max_test = static_cast<int>(std::min<double>(
        static_cast<double>(std::max(m, n)),
        static_cast<double>(std::max(m, n)) *
            static_cast<double>(config.holdout_fraction) * 3.0 + 10.0));

    // mask_zeros = true: zeros are MISSING. They enter neither the Gram, the
    // right-hand side nor any loss, on the Gaussian and the IRLS path alike.
    const bool missing_zeros = config.mask_zeros;
    const bool use_irls = config.requires_irls();

    // Train loss is computed only on request.
    const bool want_train_loss = config.track_train_loss || config.verbose;
    // Entry-wise loss is needed whenever the train set is not "all entries
    // minus the test set" or the loss is not squared error.
    const bool explicit_loss = use_irls || use_mask || missing_zeros;
    // Otherwise the train loss comes from the Gram trick (total minus test).
    const bool gram_train_loss = want_train_loss && !explicit_loss;

    // IRLS weights follow the standard (non-CV) fit: the Generalized Poisson
    // mean update uses Poisson (KL) weights 1/mu, every other distribution its
    // own variance-function weight evaluated at the observed value and the
    // fitted dispersion.
    FactorNet::LossConfig<Scalar> irls_loss = config.loss;
    if (is_gp) irls_loss.type = LossType::KL;

    // Precompute tr(A^T A) — constant across iterations
    [[maybe_unused]] const Scalar trAtA =
        primitives::trace_AtA<Resource, Scalar, MatrixType>(A);

    ::FactorNet::NMFResult<Scalar> result;
    const Scalar nan_val = std::numeric_limits<Scalar>::quiet_NaN();

    // Early stopping: with cv_patience > 0 the factors of the iteration with
    // the lowest test loss are kept and returned. With cv_patience == 0 the
    // last iterate is returned together with its own test loss.
    const bool keep_best = config.cv_patience > 0;
    Scalar best_test_loss = std::numeric_limits<Scalar>::max();
    Scalar best_train_loss = nan_val;
    int best_iter = -1;
    int patience_count = 0;
    DenseMatrix<Scalar> best_W_T, best_H;
    DenseVector<Scalar> best_d, best_theta, best_nb, best_phi, best_pi_row, best_pi_col;

    Scalar last_train_loss = nan_val;
    Scalar last_test_loss = nan_val;

    // Previous W for the convergence measure
    DenseMatrix<Scalar> W_prev;

    // Penalty terms, built once per half-update by the same routine as the
    // non-CV fit: shared k×k terms (L2, target ridge) and per-column terms
    // (target and graph right-hand side, graph ridge, L21 group norms).
    DenseMatrix<Scalar> G_reg_H, G_reg_W;
    nmf::variant::ColumnPenalty<Scalar> cp_H, cp_W;

    auto fit_start = std::chrono::high_resolution_clock::now();
    if (config.track_loss_history)
        result.history.reserve(config.max_iter);

#ifdef _OPENMP
    const int actual_threads = config.threads > 0
        ? config.threads
        : omp_get_max_threads();
#endif

    // ------------------------------------------------------------------
    // 4b. Holdout lists, built once when they fit in memory
    // ------------------------------------------------------------------
    // The held-out rows of every column and the held-out columns of every row
    // are listed once with the same functions the per-iteration (hash) route
    // uses, so both routes hold out identical entries. Memory is about
    // 8 bytes × holdout_fraction × (entries that can be held out); above
    // detail::cv_mask_cache_limit_bytes() (1 GB by default) the lists are
    // recomputed from the hash on every pass instead.
    detail::CVMaskCache mask_cache;
    {
        const double eligible = config.mask_zeros
            ? static_cast<double>(nnz)
            : static_cast<double>(m) * static_cast<double>(n);
        const double est_bytes = 8.0 * eligible * mask.holdout_fraction();
        if (est_bytes <= detail::cv_mask_cache_limit_bytes()) {
            mask_cache.col.resize(static_cast<size_t>(n));
            mask_cache.row.resize(static_cast<size_t>(m));
#ifdef _OPENMP
            #pragma omp parallel num_threads(actual_threads)
#endif
            {
                std::vector<int> tmp;
#ifdef _OPENMP
                #pragma omp for schedule(dynamic, 64)
#endif
                for (int j = 0; j < n; ++j) {
                    detail::list_holdout_col(A, j, mask, config.mask_zeros, m, tmp);
                    mask_cache.col[static_cast<size_t>(j)].assign(tmp.begin(), tmp.end());
                }
#ifdef _OPENMP
                #pragma omp for schedule(dynamic, 64)
#endif
                for (int i = 0; i < m; ++i) {
                    if constexpr (is_sparse_v<MatrixType>) {
                        detail::list_holdout_row(A_T, i, mask, config.mask_zeros, n, tmp);
                    } else {
                        detail::list_holdout_row(A, i, mask, config.mask_zeros, n, tmp);
                    }
                    mask_cache.row[static_cast<size_t>(i)].assign(tmp.begin(), tmp.end());
                }
            }
            mask_cache.active = true;
        }
    }

    // ------------------------------------------------------------------
    // 5. Iteration loop
    // ------------------------------------------------------------------

    // Timing accumulators (microseconds) for profiling
    int64_t t_h_update_us = 0, t_w_update_us = 0, t_loss_us = 0;
    auto tick = []() { return std::chrono::high_resolution_clock::now(); };
    auto elapsed_us = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };

    for (int iter = 0; iter < config.max_iter; ++iter) {

        auto t_iter_start = tick();

        // ZI: after the first iteration the solves read the imputed matrix
        const bool use_imputed = is_zi && iter > 0;

        // NB size r, indexed by row of A unless dispersion is per column
        const Scalar* nb_by_row =
            (is_nb && config.gp_dispersion != DispersionMode::PER_COL)
                ? nb_size_vec.data() : nullptr;
        const Scalar* nb_by_col =
            (is_nb && config.gp_dispersion == DispersionMode::PER_COL)
                ? nb_size_vec.data() : nullptr;

        // ==============================================================
        // H UPDATE
        // ==============================================================

        const auto h_mode = nmf::variant::h_update_mode(config);

        if (h_mode == nmf::variant::HUpdateMode::SYMMETRIC_SKIP) {
            // Symmetric: set H = W_T for use as factor in W-update
            nmf::variant::symmetric_enforce_h(H, W_T);

        } else {
            // Standard CV H-update: one solve per column on its train entries

            // Put H on the scale of the variable being solved (A ≈ W_T^T H):
            // it is both the warm start and the reference point of the graph
            // and L21 penalties.
            nmf::variant::apply_scaling(H, d);

            // Penalties from the previous iterate, before any column is
            // overwritten (no reads of H inside the parallel loop other than
            // each column's own warm start).
            G_reg_H.setZero(k, k);
            nmf::variant::build_column_penalty(G_reg_H, H, config.H, cp_H, true, false);

            if (!use_irls && !missing_zeros) {
                // Full Gram: G = W_T · W_T^T, corrected per column below
                primitives::gram<Resource>(W_T, G);
                G.diagonal().array() += static_cast<Scalar>(1e-15);
                G += G_reg_H;
            }

            auto solve_h_col = [&](const auto& M, int j,
                                   detail::CVWorkspace<Scalar>& ws) {
                // Held-out rows of this column: cached list, or from the hash.
                // (The imputed matrix has its own set of observed entries, so
                // it always takes the hash route.)
                const std::vector<int>* held = &ws.test_idx;
                if (mask_cache.active && !use_imputed) {
                    held = &mask_cache.col[static_cast<size_t>(j)];
                } else {
                    detail::list_holdout_col(M, j, mask, config.mask_zeros, m, ws.test_idx);
                }
                detail::rhs_excluding_col(M, W_T, j, *held, ws.b);

                // User mask: exclude masked entries from b and from the Gram
                const std::vector<int>* excl = held;
                if (use_mask) {
                    detail::adjust_rhs_for_user_mask_h(
                        M, W_T, *config.mask, j, ws.b, *held, ws.excluded_idx);
                    excl = &ws.excluded_idx;
                }

                ws.x = H.col(j);

                if (use_irls) {
                    // Target and graph terms enter the weighted system itself
                    const Scalar* b_reg =
                        cp_H.has_B() ? cp_H.B_add.col(j).data() : nullptr;
                    const Scalar ridge_col = cp_H.has_ridge() ? cp_H.ridge(j) : Scalar(0);
                    const Scalar theta_col = nb_by_col ? nb_by_col[j] : Scalar(0);
                    detail::irls_solve_col_cv(
                        M, W_T, j, ws.x, *excl, config.mask_zeros,
                        irls_loss, config, G_reg_H, b_reg, ridge_col,
                        theta_col, nb_by_row, m, ws);
                } else {
                    bool allow_cholesky = true;
                    bool has_data = true;
                    if (missing_zeros) {
                        const int cnt = detail::gram_from_observed_col(
                            M, W_T, j, *excl, ws.G_local, ws.buf);
                        ws.G_local += G_reg_H;
                        ws.G_local.diagonal().array() += static_cast<Scalar>(1e-15);
                        allow_cholesky = (cnt >= k);
                        has_data = (cnt > 0);
                    } else {
                        detail::apply_gram_correction(G, W_T, *excl, ws.G_local, ws.buf);
                    }
                    // Column-specific penalty terms
                    if (cp_H.has_B())
                        ws.b.noalias() += cp_H.B_add.col(j);
                    Scalar ridge_col = cp_H.has_ridge() ? cp_H.ridge(j) : Scalar(0);
                    bool is_zero = false;
                    if (cp_H.has_group()) {
                        Scalar g_ref = ws.G_local.diagonal().maxCoeff();
                        if (!(g_ref > Scalar(0))) g_ref = Scalar(1);
                        ridge_col += features::l21_column_ridge(
                            ws.b.data(), k, config.H.L1, config.H.nonneg,
                            config.H.L21, cp_H.group_norm(j), g_ref, is_zero);
                    }

                    if (is_zero || (!has_data && !cp_H.has_B())) {
                        ws.x.setZero();
                    } else {
                        if (ridge_col > Scalar(0))
                            ws.G_local.diagonal().array() += ridge_col;
                        detail::cv_solve_col(ws.G_local, ws.b, ws.x,
                                             config.H.L1, config.H.nonneg,
                                             config.H.upper_bound,
                                             config, ws.scratch, allow_cholesky);
                    }
                }
                H.col(j) = ws.x;
            };

#ifdef _OPENMP
            #pragma omp parallel num_threads(actual_threads)
#endif
            {
                detail::CVWorkspace<Scalar> ws(k, est_max_test);
#ifdef _OPENMP
                #pragma omp for schedule(dynamic, 16)
#endif
                for (int j = 0; j < n; ++j) {
                    if (use_imputed) {
                        solve_h_col(A_imputed, j, ws);
                    } else {
                        solve_h_col(A, j, ws);
                    }
                }
            }

            // Angular on H (the upper bound is enforced inside the solver)
            if (config.H.angular > 0)
                features::apply_angular_posthoc(H, config.H.angular);

            // Normalize H → d
            if (config.norm_type == NormType::None) {
                d.setOnes();
            } else {
                for (int i = 0; i < k; ++i) {
                    Scalar norm = (config.norm_type == NormType::L1)
                        ? H.row(i).template lpNorm<1>()
                        : H.row(i).norm();
                    d(i) = norm + static_cast<Scalar>(1e-15);
                    H.row(i) /= d(i);
                }
            }
        }  // end standard H update

        auto t_after_h = tick();
        t_h_update_us += elapsed_us(t_iter_start, t_after_h);

        // ==============================================================
        // W UPDATE — one solve per row on its train entries (using A^T)
        // ==============================================================

        // Put W on the scale of the variable being solved (A ≈ W^T H with H
        // normalized): warm start and penalty reference point.
        nmf::variant::apply_scaling(W_T, d);

        // Penalties from the previous iterate of W
        G_reg_W.setZero(k, k);
        nmf::variant::build_column_penalty(G_reg_W, W_T, config.W, cp_W, true, false);

        // Unpenalized Gram for the Gram-trick train loss
        DenseMatrix<Scalar> G_H_saved;
        if (!use_irls && !missing_zeros) {
            // Full Gram of the (normalized) H: G = H · H^T, corrected per row below
            primitives::gram<Resource>(H, G);
            if (gram_train_loss) {
                G_H_saved = G;  // k×k copy — cheap
            }
            G.diagonal().array() += static_cast<Scalar>(1e-15);
            G += G_reg_W;
        }

        // B_W_full.col(i) = H · A(i,:)^T over ALL entries of row i, reused by
        // the Gram-trick train loss
        DenseMatrix<Scalar> B_W_full;
        if (gram_train_loss) {
            B_W_full.resize(k, m);
            B_W_full.setZero();
        }

        auto solve_w_row = [&](const auto& M, int i,
                               detail::CVWorkspace<Scalar>& ws) {
            // Held-out columns of this row: cached list, or from the hash
            const std::vector<int>* held = &ws.test_idx;
            if (mask_cache.active && !use_imputed) {
                held = &mask_cache.row[static_cast<size_t>(i)];
            } else {
                detail::list_holdout_row(M, i, mask, config.mask_zeros, n, ws.test_idx);
            }
            detail::rhs_excluding_row(M, H, i, *held, ws.b,
                                      gram_train_loss ? &ws.b_test : nullptr);

            if (gram_train_loss)
                B_W_full.col(i) = ws.b + ws.b_test;

            const std::vector<int>* excl = held;
            if (use_mask) {
                detail::adjust_rhs_for_user_mask_w(
                    M, H, mask_T, i, ws.b, *held, ws.excluded_idx);
                excl = &ws.excluded_idx;
            }

            ws.x = W_T.col(i);

            if (use_irls) {
                const Scalar* b_reg =
                    cp_W.has_B() ? cp_W.B_add.col(i).data() : nullptr;
                const Scalar ridge_col = cp_W.has_ridge() ? cp_W.ridge(i) : Scalar(0);
                const Scalar theta_row = nb_by_row ? nb_by_row[i] : Scalar(0);
                detail::irls_solve_row_cv(
                    M, H, i, ws.x, *excl, config.mask_zeros,
                    irls_loss, config, G_reg_W, b_reg, ridge_col,
                    theta_row, nb_by_col, n, ws);
            } else {
                bool allow_cholesky = true;
                bool has_data = true;
                if (missing_zeros) {
                    const int cnt = detail::gram_from_observed_row(
                        M, H, i, *excl, ws.G_local, ws.buf);
                    ws.G_local += G_reg_W;
                    ws.G_local.diagonal().array() += static_cast<Scalar>(1e-15);
                    allow_cholesky = (cnt >= k);
                    has_data = (cnt > 0);
                } else {
                    detail::apply_gram_correction(G, H, *excl, ws.G_local, ws.buf);
                }
                // Row-specific penalty terms
                if (cp_W.has_B())
                    ws.b.noalias() += cp_W.B_add.col(i);
                Scalar ridge_col = cp_W.has_ridge() ? cp_W.ridge(i) : Scalar(0);
                bool is_zero = false;
                if (cp_W.has_group()) {
                    Scalar g_ref = ws.G_local.diagonal().maxCoeff();
                    if (!(g_ref > Scalar(0))) g_ref = Scalar(1);
                    ridge_col += features::l21_column_ridge(
                        ws.b.data(), k, config.W.L1, config.W.nonneg,
                        config.W.L21, cp_W.group_norm(i), g_ref, is_zero);
                }

                if (is_zero || (!has_data && !cp_W.has_B())) {
                    ws.x.setZero();
                } else {
                    if (ridge_col > Scalar(0))
                        ws.G_local.diagonal().array() += ridge_col;
                    detail::cv_solve_col(ws.G_local, ws.b, ws.x,
                                         config.W.L1, config.W.nonneg,
                                         config.W.upper_bound,
                                         config, ws.scratch, allow_cholesky);
                }
            }
            W_T.col(i) = ws.x;
        };

#ifdef _OPENMP
        #pragma omp parallel num_threads(actual_threads)
#endif
        {
            detail::CVWorkspace<Scalar> ws(k, est_max_test);
#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 16)
#endif
            for (int i = 0; i < m; ++i) {
                if (use_imputed) {
                    solve_w_row(A_imputed, i, ws);
                } else {
                    if constexpr (is_sparse_v<MatrixType>) {
                        solve_w_row(A_T, i, ws);
                    } else {
                        solve_w_row(A, i, ws);
                    }
                }
            }
        }

        // Angular on W (the upper bound is enforced inside the solver)
        if (config.W.angular > 0)
            features::apply_angular_posthoc(W_T, config.W.angular);

        // Normalize W → d (with epsilon to prevent component death)
        if (config.norm_type == NormType::None) {
            d.setOnes();
        } else {
            for (int i = 0; i < k; ++i) {
                Scalar norm = (config.norm_type == NormType::L1)
                    ? W_T.row(i).template lpNorm<1>()
                    : W_T.row(i).norm();
                d(i) = norm + static_cast<Scalar>(1e-15);
                W_T.row(i) /= d(i);
            }
        }

        // Enforce H = W_T after W-update (symmetric only)
        if (nmf::variant::w_update_mode(config) == nmf::variant::WUpdateMode::SYMMETRIC) {
            nmf::variant::symmetric_enforce_h(H, W_T);
        }

        // ==============================================================
        // GP theta MM update (training entries only, excluding held-out)
        // ==============================================================
        if (is_gp && config.gp_dispersion != DispersionMode::NONE) {
            DenseMatrix<Scalar> W_Td_th = W_T;
            detail::apply_scaling(W_Td_th, d);

            // Build theta-independent sums (train entries only)
            Eigen::VectorXd sum_y_d = Eigen::VectorXd::Zero(m);
            Eigen::VectorXd sum_s_d = Eigen::VectorXd::Zero(m);
            Eigen::VectorXi n_nz = Eigen::VectorXi::Zero(m);

            struct NzEntry { int row; double y; double s; };
            std::vector<NzEntry> nz_cache;

            // sum_s: via Gram trick over training columns only
            // Actually we need to sum s_ij = Wd.col(i) . H.col(j) for
            // training entries. For simplicity, sum over ALL columns then
            // subtract held-out contributions.
            DenseVector<Scalar> h_rowsum = H.rowwise().sum();
            for (int i = 0; i < m; ++i) {
                sum_s_d(i) = static_cast<double>(W_Td_th.col(i).dot(h_rowsum));
            }
            // Subtract held-out column contributions
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    if (mask.is_holdout(i, j)) {
                        sum_s_d(i) -= static_cast<double>(W_Td_th.col(i).dot(H.col(j)));
                    }
                }
            }

            // Single pass through nonzeros: accumulate sums (training only)
            if constexpr (is_sparse_v<MatrixType>) {
                for (int j = 0; j < n; ++j) {
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                        if (mask.is_holdout(it.row(), j)) continue;  // skip test
                        double y = static_cast<double>(it.value());
                        double s_val = static_cast<double>(W_Td_th.col(it.row()).dot(H.col(j)));
                        sum_y_d(it.row()) += y;
                        if (y >= 1.0) n_nz(it.row())++;
                        nz_cache.push_back(NzEntry{static_cast<int>(it.row()), y, std::max(s_val, 1e-10)});
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        if (mask.is_holdout(i, j)) continue;
                        double y = static_cast<double>(A(i, j));
                        if (y > 0.0) {
                            double s_val = static_cast<double>(W_Td_th.col(i).dot(H.col(j)));
                            sum_y_d(i) += y;
                            if (y >= 1.0) n_nz(i)++;
                            nz_cache.push_back({i, y, std::max(s_val, 1e-10)});
                        }
                    }
                }
            }

            // Inner MM iterations
            constexpr int THETA_INNER_ITERS = 5;
            double cap = static_cast<double>(config.gp_theta_max);
            for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS; ++mm_iter) {
                Eigen::VectorXd alpha_d = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd gamma_d = Eigen::VectorXd::Zero(m);

                for (const auto& nz : nz_cache) {
                    if (nz.y >= 1.0) {
                        double th = static_cast<double>(theta_vec(nz.row));
                        double denom = std::max(nz.s + th * nz.y, 1e-10);
                        double eta1 = nz.s / denom;
                        alpha_d(nz.row) += (nz.y - 1.0) * eta1;
                        gamma_d(nz.row) += (nz.y - 1.0) * (1.0 - eta1);
                    }
                }

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

            // Global mode: average per-row thetas
            if (config.gp_dispersion == DispersionMode::GLOBAL) {
                Scalar mean_theta = theta_vec.mean();
                theta_vec.setConstant(mean_theta);
            }
        }

        // ==============================================================
        // NB size (r) update — method of moments (training entries only)
        // ==============================================================
        if (is_nb && config.gp_dispersion != DispersionMode::NONE) {
            DenseMatrix<Scalar> W_Td_nb = W_T;
            detail::apply_scaling(W_Td_nb, d);

            double r_min = static_cast<double>(config.nb_size_min);
            double r_max = static_cast<double>(config.nb_size_max);

            const int nb_dim = (config.gp_dispersion == DispersionMode::PER_COL) ? n : m;
            Eigen::VectorXd sum_mu_sq = Eigen::VectorXd::Zero(nb_dim);
            Eigen::VectorXd sum_excess_var = Eigen::VectorXd::Zero(nb_dim);

            if (config.gp_dispersion == DispersionMode::PER_COL) {
                // PER_COL: accumulate per column j
                if constexpr (is_sparse_v<MatrixType>) {
                    for (int j = 0; j < n; ++j) {
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            if (mask.is_holdout(it.row(), j)) continue;
                            double y = static_cast<double>(it.value());
                            double mu = std::max(static_cast<double>(
                                W_Td_nb.col(it.row()).dot(H.col(j))), 1e-10);
                            double resid = y - mu;
                            sum_mu_sq(j) += mu * mu;
                            sum_excess_var(j) += resid * resid - mu;
                        }
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < m; ++i) {
                            if (mask.is_holdout(i, j)) continue;
                            double y = static_cast<double>(A(i, j));
                            double mu = std::max(static_cast<double>(
                                W_Td_nb.col(i).dot(H.col(j))), 1e-10);
                            double resid = y - mu;
                            sum_mu_sq(j) += mu * mu;
                            sum_excess_var(j) += resid * resid - mu;
                        }
                    }
                }
            } else {
                // PER_ROW or GLOBAL: accumulate per row i
                if constexpr (is_sparse_v<MatrixType>) {
                    for (int j = 0; j < n; ++j) {
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            if (mask.is_holdout(it.row(), j)) continue;
                            double y = static_cast<double>(it.value());
                            double mu = std::max(static_cast<double>(
                                W_Td_nb.col(it.row()).dot(H.col(j))), 1e-10);
                            double resid = y - mu;
                            sum_mu_sq(it.row()) += mu * mu;
                            sum_excess_var(it.row()) += resid * resid - mu;
                        }
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < m; ++i) {
                            if (mask.is_holdout(i, j)) continue;
                            double y = static_cast<double>(A(i, j));
                            double mu = std::max(static_cast<double>(
                                W_Td_nb.col(i).dot(H.col(j))), 1e-10);
                            double resid = y - mu;
                            sum_mu_sq(i) += mu * mu;
                            sum_excess_var(i) += resid * resid - mu;
                        }
                    }
                }
            }

            for (int idx = 0; idx < nb_dim; ++idx) {
                if (sum_excess_var(idx) > 1e-10 && sum_mu_sq(idx) > 1e-10) {
                    double r_new = sum_mu_sq(idx) / sum_excess_var(idx);
                    r_new = std::max(r_min, std::min(r_new, r_max));
                    if (std::isfinite(r_new))
                        nb_size_vec(idx) = static_cast<Scalar>(r_new);
                } else {
                    nb_size_vec(idx) = static_cast<Scalar>(r_max);
                }
            }

            if (config.gp_dispersion == DispersionMode::GLOBAL) {
                Scalar mean_r = nb_size_vec.mean();
                nb_size_vec.setConstant(mean_r);
            }
        }

        // ==============================================================
        // Gamma/IG/Tweedie φ update — MoM Pearson (training only)
        // ==============================================================
        if ((is_gamma || is_invgauss || is_tweedie) &&
            config.gp_dispersion != DispersionMode::NONE) {
            DenseMatrix<Scalar> W_Td_phi = W_T;
            detail::apply_scaling(W_Td_phi, d);

            const double var_power = is_tweedie ? static_cast<double>(config.loss.power_param)
                                   : (is_gamma ? 2.0 : 3.0);
            double phi_min = static_cast<double>(config.gamma_phi_min);
            double phi_max = static_cast<double>(config.gamma_phi_max);

            if (config.gp_dispersion == DispersionMode::PER_COL) {
                Eigen::VectorXd sum_pearson_sq_col = Eigen::VectorXd::Zero(n);
                Eigen::VectorXi count_col = Eigen::VectorXi::Zero(n);

                if constexpr (is_sparse_v<MatrixType>) {
                    for (int j = 0; j < n; ++j) {
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            if (mask.is_holdout(it.row(), j)) continue;
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
                            if (mask.is_holdout(i, j)) continue;
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
                // PER_ROW or GLOBAL
                Eigen::VectorXd sum_pearson_sq = Eigen::VectorXd::Zero(m);
                Eigen::VectorXi count_vec = Eigen::VectorXi::Zero(m);

                if constexpr (is_sparse_v<MatrixType>) {
                    for (int j = 0; j < n; ++j) {
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it) {
                            if (mask.is_holdout(it.row(), j)) continue;
                            double y = static_cast<double>(it.value());
                            if (y <= 0.0) continue;
                            double mu = std::max(static_cast<double>(
                                W_Td_phi.col(it.row()).dot(H.col(j))), 1e-10);
                            double resid = y - mu;
                            double v_mu = std::pow(mu, var_power);
                            sum_pearson_sq(it.row()) += (resid * resid) / std::max(v_mu, 1e-20);
                            count_vec(it.row())++;
                        }
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < m; ++i) {
                            if (mask.is_holdout(i, j)) continue;
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
                for (int i = 0; i < m; ++i) {
                    if (count_vec(i) > 0) {
                        double phi_new = sum_pearson_sq(i) / static_cast<double>(count_vec(i));
                        phi_new = std::max(phi_min, std::min(phi_new, phi_max));
                        if (std::isfinite(phi_new))
                            phi_vec(i) = static_cast<Scalar>(phi_new);
                    }
                }
                if (config.gp_dispersion == DispersionMode::GLOBAL) {
                    std::vector<Scalar> phi_vals(phi_vec.data(), phi_vec.data() + m);
                    std::nth_element(phi_vals.begin(), phi_vals.begin() + m / 2, phi_vals.end());
                    Scalar median_phi = phi_vals[m / 2];
                    phi_vec.setConstant(median_phi);
                }
            }
        }

        // ==============================================================
        // ZI EM block: E-step + M-step + soft imputation
        // (training entries only, excluding held-out)
        // ==============================================================
        if (is_zi) {
            DenseMatrix<Scalar> W_Td_zi = W_T;
            detail::apply_scaling(W_Td_zi, d);

            for (int zi_iter = 0; zi_iter < config.zi_em_iters; ++zi_iter) {
                Eigen::VectorXd z_row_sum = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd z_col_sum = Eigen::VectorXd::Zero(n);
                Eigen::VectorXi zero_count_row = Eigen::VectorXi::Zero(m);
                Eigen::VectorXi zero_count_col = Eigen::VectorXi::Zero(n);

                // E-step: compute z_ij posteriors for TRAIN zero entries
                if constexpr (is_sparse_v<MatrixType>) {
                    for (int j = 0; j < n; ++j) {
                        std::vector<bool> is_nz(m, false);
                        for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                            is_nz[it.row()] = true;

                        for (int i = 0; i < m; ++i) {
                            if (is_nz[i]) continue;            // nonzero in original
                            if (mask.is_holdout(i, j)) continue; // skip held-out

                            double s_ij = std::max(static_cast<double>(
                                W_Td_zi.col(i).dot(H.col(j))), 1e-10);
                            double p0;
                            if (is_nb) {
                                int nb_idx = (config.gp_dispersion == DispersionMode::PER_COL) ? j : i;
                                double r_i = std::max(static_cast<double>(nb_size_vec(nb_idx)), 1e-10);
                                p0 = std::pow(r_i / (r_i + s_ij), r_i);
                            } else {
                                double th_i = static_cast<double>(theta_vec(i));
                                double otp = 1.0 + th_i;
                                p0 = std::exp(-s_ij / otp);
                            }

                            double pi_ij;
                            if (config.zi_mode == ZIMode::ZI_ROW)
                                pi_ij = static_cast<double>(pi_row_vec(i));
                            else if (config.zi_mode == ZIMode::ZI_COL)
                                pi_ij = static_cast<double>(pi_col_vec(j));
                            else {
                                double pr = static_cast<double>(pi_row_vec(i));
                                double pc = static_cast<double>(pi_col_vec(j));
                                pi_ij = 1.0 - (1.0 - pr) * (1.0 - pc);
                            }

                            double z_ij = pi_ij / (pi_ij + (1.0 - pi_ij) * p0 + 1e-300);
                            z_row_sum(i) += z_ij;
                            z_col_sum(j) += z_ij;
                            zero_count_row(i)++;
                            zero_count_col(j)++;
                        }
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < m; ++i) {
                            if (static_cast<double>(A.coeff(i, j)) != 0.0) continue;
                            if (mask.is_holdout(i, j)) continue;

                            double s_ij = std::max(static_cast<double>(
                                W_Td_zi.col(i).dot(H.col(j))), 1e-10);
                            double p0;
                            if (is_nb) {
                                int nb_idx = (config.gp_dispersion == DispersionMode::PER_COL) ? j : i;
                                double r_i = std::max(static_cast<double>(nb_size_vec(nb_idx)), 1e-10);
                                p0 = std::pow(r_i / (r_i + s_ij), r_i);
                            } else {
                                double th_i = static_cast<double>(theta_vec(i));
                                double otp = 1.0 + th_i;
                                p0 = std::exp(-s_ij / otp);
                            }

                            double pi_ij;
                            if (config.zi_mode == ZIMode::ZI_ROW)
                                pi_ij = static_cast<double>(pi_row_vec(i));
                            else if (config.zi_mode == ZIMode::ZI_COL)
                                pi_ij = static_cast<double>(pi_col_vec(j));
                            else {
                                double pr = static_cast<double>(pi_row_vec(i));
                                double pc = static_cast<double>(pi_col_vec(j));
                                pi_ij = 1.0 - (1.0 - pr) * (1.0 - pc);
                            }

                            double z_ij = pi_ij / (pi_ij + (1.0 - pi_ij) * p0 + 1e-300);
                            z_row_sum(i) += z_ij;
                            z_col_sum(j) += z_ij;
                            zero_count_row(i)++;
                            zero_count_col(j)++;
                        }
                    }
                }

                // M-step: update pi
                if (config.zi_mode == ZIMode::ZI_ROW || config.zi_mode == ZIMode::ZI_TWOWAY) {
                    for (int i = 0; i < m; ++i) {
                        if (zero_count_row(i) > 0) {
                            double new_pi = z_row_sum(i) / static_cast<double>(n);
                            if (config.zi_mode == ZIMode::ZI_TWOWAY) {
                                double old_pi = static_cast<double>(pi_row_vec(i));
                                double step = std::clamp(new_pi - old_pi, -0.05, 0.05);
                                new_pi = std::clamp(old_pi + step, 0.001, 0.5);
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
                                double step = std::clamp(new_pi - old_pi, -0.05, 0.05);
                                new_pi = std::clamp(old_pi + step, 0.001, 0.5);
                            } else {
                                new_pi = std::clamp(new_pi, 0.001, 0.999);
                            }
                            pi_col_vec(j) = static_cast<Scalar>(new_pi);
                        }
                    }
                }
            }  // end zi_em_iters

            // Soft imputation: update A_imputed for next iteration
            if constexpr (is_sparse_v<MatrixType>) {
                A_imputed.setZero();
                for (int j = 0; j < n; ++j)
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                        A_imputed(it.row(), j) = static_cast<Scalar>(it.value());
            } else {
                A_imputed = A;
            }

            for (int j = 0; j < n; ++j) {
                std::vector<bool> is_nz_imp;
                if constexpr (is_sparse_v<MatrixType>) {
                    is_nz_imp.assign(m, false);
                    for (typename std::decay_t<decltype(A)>::InnerIterator it(A, j); it; ++it)
                        is_nz_imp[it.row()] = true;
                }

                for (int i = 0; i < m; ++i) {
                    bool is_zero;
                    if constexpr (is_sparse_v<MatrixType>)
                        is_zero = !is_nz_imp[i];
                    else
                        is_zero = (static_cast<double>(A.coeff(i, j)) == 0.0);

                    if (is_zero && !mask.is_holdout(i, j)) {
                        double s_ij = std::max(static_cast<double>(
                            W_Td_zi.col(i).dot(H.col(j))), 1e-10);
                        double p0;
                        if (is_nb) {
                            int nb_idx = (config.gp_dispersion == DispersionMode::PER_COL) ? j : i;
                            double r_i = std::max(static_cast<double>(nb_size_vec(nb_idx)), 1e-10);
                            p0 = std::pow(r_i / (r_i + s_ij), r_i);
                        } else {
                            double th_i = static_cast<double>(theta_vec(i));
                            double otp = 1.0 + th_i;
                            p0 = std::exp(-s_ij / otp);
                        }

                        double pi_ij;
                        if (config.zi_mode == ZIMode::ZI_ROW)
                            pi_ij = static_cast<double>(pi_row_vec(i));
                        else if (config.zi_mode == ZIMode::ZI_COL)
                            pi_ij = static_cast<double>(pi_col_vec(j));
                        else {
                            double pr = static_cast<double>(pi_row_vec(i));
                            double pc = static_cast<double>(pi_col_vec(j));
                            pi_ij = 1.0 - (1.0 - pr) * (1.0 - pc);
                        }

                        double z_ij = pi_ij / (pi_ij + (1.0 - pi_ij) * p0 + 1e-300);
                        A_imputed(i, j) = static_cast<Scalar>(z_ij * s_ij);
                    }
                }
            }
        }  // end ZI EM block

        auto t_after_w = tick();
        t_w_update_us += elapsed_us(t_after_h, t_after_w);

        // ==============================================================
        // Convergence measure: 1 - cosine similarity of successive W
        // ==============================================================
        // Convergence is judged on the change of the fitted factor, which is
        // a property of the training fit alone. The test loss only drives
        // early stopping (cv_patience), never the tolerance check.
        double w_change = std::numeric_limits<double>::quiet_NaN();
        bool tol_converged = false;
        if (iter > 0 && W_prev.size() == W_T.size()) {
            double num = 0.0, n_new = 0.0, n_old = 0.0;
            const Scalar* p_new = W_T.data();
            const Scalar* p_old = W_prev.data();
            const Eigen::Index sz = W_T.size();
            for (Eigen::Index q = 0; q < sz; ++q) {
                const double a_new = static_cast<double>(p_new[q]);
                const double a_old = static_cast<double>(p_old[q]);
                num += a_new * a_old;
                n_new += a_new * a_new;
                n_old += a_old * a_old;
            }
            const double den = std::sqrt(n_new) * std::sqrt(n_old);
            w_change = (den > 0.0) ? std::max(0.0, 1.0 - num / den) : 0.0;
            tol_converged = w_change < static_cast<double>(config.tol);
        }
        W_prev = W_T;

        // ==============================================================
        // Compute train and test loss
        // ==============================================================

        // Loss is needed for: (1) early stopping, (2) loss history, (3) verbose,
        // (4) the first iteration and the last one (max_iter or convergence),
        // so that the returned model always has a test loss.
        const bool need_loss_this_iter =
            (config.cv_patience > 0) ||
            config.track_loss_history ||
            config.verbose ||
            (iter == 0) ||
            (iter == config.max_iter - 1) ||
            tol_converged;

        // Train loss only when it was asked for (track_train_loss or verbose).
        // When it is not computed it is reported as NaN, never as 0.
        const bool need_train_loss = need_loss_this_iter && want_train_loss;

        Scalar train_loss = nan_val;
        Scalar test_loss = nan_val;
        int64_t n_test_total = 0;
        int64_t n_train_total = 0;

        if (need_loss_this_iter) {

            // Model used for the loss: W_T^T · diag(d) · H
            DenseMatrix<Scalar> W_Td_loss = W_T;
            for (int r = 0; r < k; ++r) W_Td_loss.row(r) *= d(r);

            const FactorNet::LossConfig<Scalar> loss_cfg = config.loss;
            const bool disp_per_col = (config.gp_dispersion == DispersionMode::PER_COL);
            // Entry-wise train loss (the Gram trick covers the other case)
            const bool accumulate_train = need_train_loss && explicit_loss;

            // Per-column partial sums in double, added up serially afterwards
            // so that the result does not depend on thread scheduling.
            std::vector<double> col_test(static_cast<size_t>(n), 0.0);
            std::vector<double> col_train(static_cast<size_t>(n), 0.0);
            std::vector<int64_t> col_ntest(static_cast<size_t>(n), 0);
            std::vector<int64_t> col_ntrain(static_cast<size_t>(n), 0);

#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64) num_threads(actual_threads)
#endif
            for (int j = 0; j < n; ++j) {
                double s_test = 0.0, s_train = 0.0;
                int64_t c_test = 0, c_train = 0;

                // Is row i of this column held out? Cached list (rows are
                // visited in ascending order) or the hash.
                const std::vector<int>* held_rows =
                    mask_cache.active ? &mask_cache.col[static_cast<size_t>(j)] : nullptr;
                size_t held_pos = 0;
                auto is_held = [&](int i) -> bool {
                    if (held_rows == nullptr) return mask.is_holdout(i, j);
                    const std::vector<int>& hl = *held_rows;
                    while (held_pos < hl.size() && hl[held_pos] < i) ++held_pos;
                    return held_pos < hl.size() && hl[held_pos] == i;
                };

                auto visit = [&](int i, Scalar val, bool is_test) {
                    if (use_mask && detail::is_user_masked_h(i, j, *config.mask))
                        return;
                    if (!is_test && !accumulate_train) return;

                    const Scalar pred = W_Td_loss.col(i).dot(H.col(j));
                    // Fitted dispersion of this entry: GP theta (per row),
                    // NB size r (per row or per column)
                    Scalar disp = Scalar(0);
                    if (is_gp) {
                        disp = theta_vec(i);
                    } else if (is_nb) {
                        disp = nb_size_vec(disp_per_col ? j : i);
                    }
                    const double loss_val = static_cast<double>(
                        compute_loss(val, pred, loss_cfg, disp));
                    if (is_test) {
                        s_test += loss_val;
                        ++c_test;
                    } else {
                        s_train += loss_val;
                        ++c_train;
                    }
                };

                if (held_rows != nullptr && !accumulate_train) {
                    // Only the held-out entries are needed: walk the list
                    if constexpr (is_sparse_v<MatrixType>) {
                        typename MatrixType::InnerIterator it(A, j);
                        for (int i : *held_rows) {
                            while (it && it.row() < i) ++it;
                            Scalar val = 0;
                            if (it && it.row() == i)
                                val = static_cast<Scalar>(it.value());
                            visit(i, val, true);
                        }
                    } else {
                        for (int i : *held_rows) visit(i, A(i, j), true);
                    }
                } else if constexpr (is_sparse_v<MatrixType>) {
                    if (missing_zeros) {
                        // Zeros are missing: only stored entries are data
                        for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                            const int i = static_cast<int>(it.row());
                            visit(i, static_cast<Scalar>(it.value()), is_held(i));
                        }
                    } else {
                        typename MatrixType::InnerIterator it(A, j);
                        for (int i = 0; i < m; ++i) {
                            Scalar val = 0;
                            if (it && it.row() == i) {
                                val = static_cast<Scalar>(it.value());
                                ++it;
                            }
                            visit(i, val, is_held(i));
                        }
                    }
                } else {
                    for (int i = 0; i < m; ++i) {
                        const Scalar val = A(i, j);
                        if (missing_zeros && val == Scalar(0)) continue;
                        visit(i, val, is_held(i));
                    }
                }

                col_test[static_cast<size_t>(j)] = s_test;
                col_train[static_cast<size_t>(j)] = s_train;
                col_ntest[static_cast<size_t>(j)] = c_test;
                col_ntrain[static_cast<size_t>(j)] = c_train;
            }

            double test_sum = 0.0, train_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                test_sum += col_test[static_cast<size_t>(j)];
                train_sum += col_train[static_cast<size_t>(j)];
                n_test_total += col_ntest[static_cast<size_t>(j)];
                n_train_total += col_ntrain[static_cast<size_t>(j)];
            }

            if (need_train_loss && gram_train_loss) {
                // Squared error over ALL entries by the Gram trick, reusing
                // B_W_full from the W update:
                //   total = tr(A^T A) - 2·cross + recon
                //   cross = Σ_r d(r) · W_T.row(r) · B_W_full.row(r)^T
                //   recon = Σ_{r,s} d(r)·d(s) · (W_T W_T^T)(r,s) · (H H^T)(r,s)
                // The train error is the total minus the test error.
                double cross_term = 0.0;
                for (int r = 0; r < k; ++r) {
                    cross_term += static_cast<double>(d(r)) *
                        static_cast<double>(W_T.row(r).dot(B_W_full.row(r)));
                }

                DenseMatrix<Scalar> G_W_new;
                primitives::gram<Resource>(W_T, G_W_new);

                double recon_norm = 0.0;
                for (int r = 0; r < k; ++r) {
                    for (int s = 0; s < k; ++s) {
                        recon_norm += static_cast<double>(d(r)) * static_cast<double>(d(s)) *
                                      static_cast<double>(G_W_new(r, s)) *
                                      static_cast<double>(G_H_saved(r, s));
                    }
                }

                const double total_sq_error = std::max(
                    static_cast<double>(trAtA) - 2.0 * cross_term + recon_norm, 0.0);
                train_sum = std::max(total_sq_error - test_sum, 0.0);
                n_train_total = static_cast<int64_t>(m) * n - n_test_total;
            }

            if (need_train_loss && n_train_total > 0)
                train_loss = static_cast<Scalar>(train_sum / static_cast<double>(n_train_total));
            if (n_test_total > 0)
                test_loss = static_cast<Scalar>(test_sum / static_cast<double>(n_test_total));

            last_train_loss = train_loss;
            last_test_loss = test_loss;

        } // end need_loss_this_iter

        auto t_after_loss = tick();
        t_loss_us += elapsed_us(t_after_w, t_after_loss);

        // Track loss history
        if (config.track_loss_history) {
            result.loss_history.push_back(train_loss);
            result.test_loss_history.push_back(test_loss);
        }

        const Scalar cv_rel_change =
            (iter > 0) ? static_cast<Scalar>(w_change) : Scalar(0);

        // Record FitHistory snapshot
        if (config.track_loss_history) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - fit_start).count();
            result.history.push(iter + 1, train_loss, test_loss, cv_rel_change, elapsed_ms);
        }

        // Fire per-iteration callback
        if (config.on_iteration) {
            config.on_iteration(iter + 1, train_loss, test_loss);
        }

        // Early stopping bookkeeping (only iterations with a valid test loss count)
        const bool have_test = need_loss_this_iter && n_test_total > 0 &&
                               std::isfinite(static_cast<double>(test_loss));
        if (have_test) {
            if (test_loss < best_test_loss) {
                best_test_loss = test_loss;
                best_train_loss = train_loss;
                best_iter = iter;
                patience_count = 0;
                if (keep_best) {
                    best_W_T = W_T;
                    best_H = H;
                    best_d = d;
                    best_theta = theta_vec;
                    best_nb = nb_size_vec;
                    best_phi = phi_vec;
                    best_pi_row = pi_row_vec;
                    best_pi_col = pi_col_vec;
                }
            } else {
                patience_count++;
            }
        }

        if (config.verbose) {
            FACTORNET_PRINT_IMPL("Iter %d: train=%g, test=%g, best=%g  [H=%ldus W=%ldus loss=%ldus]\n",
                              iter + 1, static_cast<double>(train_loss),
                              static_cast<double>(test_loss),
                              static_cast<double>(best_test_loss),
                              (long)elapsed_us(t_iter_start, t_after_h),
                              (long)elapsed_us(t_after_h, t_after_w),
                              (long)elapsed_us(t_after_w, t_after_loss));
        }

        result.iterations = iter + 1;
        if (iter > 0) result.final_tol = cv_rel_change;

        // Early stopping: test loss has not improved for cv_patience iterations
        if (keep_best && patience_count >= config.cv_patience) {
            result.converged = false;
            break;
        }

        // Convergence: the factor has stopped changing
        if (tol_converged) {
            result.converged = true;
            break;
        }
    }

    if (config.verbose) {
        int64_t total_us = t_h_update_us + t_w_update_us + t_loss_us;
        FACTORNET_PRINT_IMPL("[CV timing] H=%ldms W=%ldms loss=%ldms total=%ldms (loss=%.1f%%)\n",
                          (long)(t_h_update_us / 1000), (long)(t_w_update_us / 1000),
                          (long)(t_loss_us / 1000), (long)(total_us / 1000),
                          100.0 * t_loss_us / std::max(total_us, (int64_t)1));
    }

    // ------------------------------------------------------------------
    // 6. Package result
    // ------------------------------------------------------------------

    // The returned factors, losses and best_iter always describe the same
    // model: the best test iteration when early stopping is on, the last
    // iteration otherwise.
    if (keep_best && best_iter >= 0) {
        if (best_iter != result.iterations - 1) {
            W_T = best_W_T;
            H = best_H;
            d = best_d;
            theta_vec = best_theta;
            nb_size_vec = best_nb;
            phi_vec = best_phi;
            pi_row_vec = best_pi_row;
            pi_col_vec = best_pi_col;
        }
        result.best_test_loss = best_test_loss;
        result.best_iter = best_iter;
        result.test_loss = best_test_loss;
        result.train_loss = best_train_loss;
    } else {
        result.test_loss = last_test_loss;
        result.train_loss = last_train_loss;
        result.best_test_loss = last_test_loss;
        result.best_iter = std::max(result.iterations - 1, 0);
    }

    // Same convention as the non-CV fit: W (columns) and H (rows) carry unit
    // norm and the scale lives in d only, so that A ≈ W · diag(d) · H.
    result.W = W_T.transpose();  // k×m → m×k
    result.d = d;
    result.H = H;
    if (is_gp) result.theta = theta_vec;
    if (is_nb && nb_size_vec.size() > 0) result.theta = nb_size_vec;
    if ((is_gamma || is_invgauss || is_tweedie) && phi_vec.size() > 0)
        result.dispersion = phi_vec;
    if (pi_row_vec.size() > 0) result.pi_row = pi_row_vec;
    if (pi_col_vec.size() > 0) result.pi_col = pi_col_vec;

    if (config.sort_model)
        result.sort();

    return result;
}

}  // namespace nmf
}  // namespace FactorNet
