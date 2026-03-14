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
 *   - Train loss: Gram trick using per-column accumulators
 *   - Test loss: explicit reconstruction at held-out entries only
 *
 * This implementation uses the unified config/result types from core/ and
 * reuses CD solvers from solvers/cd_fast.hpp. Features (L1, L2, ortho, L21,
 * graph_reg, bounds) are applied identically to the standard NMF loop.
 *
 * IRLS support: For non-MSE losses (MAE, Huber, KL), per-column/row IRLS
 * loops build weighted Gram and RHS from train entries only. Loss computation
 * uses explicit per-element loss instead of the Gram trick.
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
#include <limits>
#include <algorithm>
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
    // 4. Allocate working buffers
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> G(k, k);
    DenseMatrix<Scalar> G_local(k, k);
    DenseVector<Scalar> b(k);
    DenseVector<Scalar> x(k);

    // Estimate max test entries per column for buffer sizing
    const int est_max_test = static_cast<int>(
        std::max(m, n) * static_cast<double>(config.holdout_fraction) * 3 + 10);
    DenseMatrix<Scalar> W_test_buf(k, est_max_test);

    // Previous loss for convergence check (replaces factor-based W_prev)
    Scalar prev_conv_loss = std::numeric_limits<Scalar>::max();

    // Precompute tr(A^T A) — constant across iterations
    const Scalar trAtA = primitives::trace_AtA<Resource, Scalar, MatrixType>(A);

    ::FactorNet::NMFResult<Scalar> result;
    Scalar best_test_loss = std::numeric_limits<Scalar>::max();
    int best_iter = 0;
    int patience_count = 0;
    auto fit_start = std::chrono::high_resolution_clock::now();
    if (config.track_loss_history)
        result.history.reserve(config.max_iter);

#ifdef _OPENMP
    const int actual_threads = config.threads > 0
        ? config.threads
        : omp_get_max_threads();
#endif

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

        // ==============================================================
        // H UPDATE — variant dispatch
        // ==============================================================

        const auto h_mode = nmf::variant::h_update_mode(config);

        if (h_mode == nmf::variant::HUpdateMode::PROJECTIVE) {
            // Projective: H = diag(d) · W_T · A (full matrix, no CV holdout for H)
            nmf::variant::projective_h_update(
                W_T, H, d,
                [&](const DenseMatrix<Scalar>& factor, DenseMatrix<Scalar>& result) {
                    if constexpr (is_sparse_v<MatrixType>) {
                        result.resize(factor.rows(), n);
                        result.setZero();
                        for (int jj = 0; jj < n; ++jj)
                            for (typename MatrixType::InnerIterator it(A, jj); it; ++it)
                                result.col(jj).noalias() += it.value() * factor.col(it.row());
                    } else {
                        result.noalias() = factor * A;
                    }
                },
                config.norm_type);

        } else if (h_mode == nmf::variant::HUpdateMode::SYMMETRIC_SKIP) {
            // Symmetric: set H = W_T for use as factor in W-update
            nmf::variant::symmetric_enforce_h(H, W_T);

        } else {
            // Standard CV H-update: per-column with Gram correction

        // Full Gram: G = W_T · W_T^T
        primitives::gram<Resource>(W_T, G);
        G.diagonal().array() += static_cast<Scalar>(1e-15);

        // Apply CV features (L2 + Tier 2) to the full Gram
        nmf::variant::apply_h_cv_features(G, H, config);

#ifdef _OPENMP
        #pragma omp parallel num_threads(actual_threads)
        {
            DenseVector<Scalar> b_local(k);
            DenseVector<Scalar> x_local(k);
            DenseMatrix<Scalar> G_local_thr(k, k);
            DenseMatrix<Scalar> W_test_thr(k, est_max_test);
            std::vector<int> test_rows_local;
            test_rows_local.reserve(static_cast<size_t>(est_max_test));
            std::vector<int> excluded_rows_local;

            #pragma omp for schedule(dynamic, 64)
            for (int j = 0; j < n; ++j) {
                // ZI path (iter > 0): compute RHS from dense A_imputed
                if (is_zi && iter > 0) {
                    detail::compute_train_rhs(A_imputed, W_T, j, mask, b_local, test_rows_local, config.mask_zeros, m);
                } else {
                    detail::compute_train_rhs(A, W_T, j, mask, b_local, test_rows_local, config.mask_zeros, m);
                }

                // User mask adjustment: exclude masked entries from b and extend Gram correction
                const auto& gram_rows = use_mask
                    ? (detail::adjust_rhs_for_user_mask_h(
                           A, W_T, *config.mask, j, b_local, test_rows_local, excluded_rows_local),
                       excluded_rows_local)
                    : test_rows_local;

                if (config.requires_irls()) {
                    // IRLS path: builds weighted Gram per column
                    x_local = H.col(j);
                    if (is_zi && iter > 0) {
                        detail::irls_solve_col_cv(A_imputed, W_T, j, x_local, gram_rows,
                            config.mask_zeros, config.loss, config, H, m);
                    } else {
                        detail::irls_solve_col_cv(A, W_T, j, x_local, gram_rows,
                            config.mask_zeros, config.loss, config, H, m);
                    }
                    H.col(j) = x_local;
                } else {
                    // MSE path: Gram correction + solver dispatch
                    detail::apply_gram_correction(
                        G, W_T, gram_rows, G_local_thr, W_test_thr);

                    x_local = H.col(j);
                    if (config.solver_mode == 1) {
                        primitives::detail::cholesky_clip_col(
                            G_local_thr, b_local.data(), x_local.data(), k,
                            config.H.L1, Scalar(0), config.H.nonneg,
                            config.cd_max_iter, config.cd_tol);
                    } else {
                        primitives::detail::cd_nnls_col_fixed(
                            G_local_thr, b_local.data(), x_local.data(), k,
                            config.H.L1, Scalar(0), config.H.nonneg,
                            config.cd_max_iter);
                    }
                    H.col(j) = x_local;
                }
            }
        }
#else
        std::vector<int> test_rows;
        test_rows.reserve(static_cast<size_t>(est_max_test));
        std::vector<int> excluded_rows;

        for (int j = 0; j < n; ++j) {
            if (is_zi && iter > 0) {
                detail::compute_train_rhs(A_imputed, W_T, j, mask, b, test_rows, config.mask_zeros, m);
            } else {
                detail::compute_train_rhs(A, W_T, j, mask, b, test_rows, config.mask_zeros, m);
            }

            // User mask adjustment
            const auto& gram_rows = use_mask
                ? (detail::adjust_rhs_for_user_mask_h(
                       A, W_T, *config.mask, j, b, test_rows, excluded_rows),
                   excluded_rows)
                : test_rows;

            if (config.requires_irls()) {
                x = H.col(j);
                if (is_zi && iter > 0) {
                    detail::irls_solve_col_cv(A_imputed, W_T, j, x, gram_rows,
                        config.mask_zeros, config.loss, config, H, m);
                } else {
                    detail::irls_solve_col_cv(A, W_T, j, x, gram_rows,
                        config.mask_zeros, config.loss, config, H, m);
                }
                H.col(j) = x;
            } else {
                detail::apply_gram_correction(G, W_T, gram_rows, G_local, W_test_buf);

                x = H.col(j);
                if (config.solver_mode == 1) {
                    primitives::detail::cholesky_clip_col(
                        G_local, b.data(), x.data(), k,
                        config.H.L1, Scalar(0), config.H.nonneg,
                        config.cd_max_iter, config.cd_tol);
                } else {
                    primitives::detail::cd_nnls_col_fixed(
                        G_local, b.data(), x.data(), k,
                        config.H.L1, Scalar(0), config.H.nonneg,
                        config.cd_max_iter);
                }
                H.col(j) = x;
            }
        }
#endif

        // Post-NNLS bounds and angular on H
        if (config.H.upper_bound > 0)
            features::apply_upper_bound(H, config.H.upper_bound);
        if (config.H.angular > 0)
            features::apply_angular_posthoc(H, config.H.angular);

        }  // end standard H update

        // Normalize H → d (skip for projective [done in helper] and symmetric [d from W-norm])
        if (h_mode == nmf::variant::HUpdateMode::STANDARD) {
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
        }

        auto t_after_h = tick();
        t_h_update_us += elapsed_us(t_iter_start, t_after_h);

        // ==============================================================
        // W UPDATE — per-row with Gram correction (using A^T)
        // ==============================================================

        // Pre-compute whether we need loss and train loss this iteration
        // (must know BEFORE W-update to decide whether to accumulate B_W_full)
        const bool need_loss_this_iter_pre =
            (config.cv_patience > 0) || config.track_loss_history ||
            config.verbose || (iter == 0) || (iter == config.max_iter - 1);
        const bool need_train_loss_pre = need_loss_this_iter_pre &&
            !config.requires_irls() && !use_mask &&
            (config.track_train_loss || config.verbose ||
             iter == 0 || iter == config.max_iter - 1);

        // Use normalized H directly for well-conditioned Gram

        // Full Gram: G = H · H^T
        primitives::gram<Resource>(H, G);

        // Save unmodified G_H for Gram trick loss computation
        DenseMatrix<Scalar> G_H_saved;
        if (need_train_loss_pre) {
            G_H_saved = G;  // k×k copy — cheap
        }

        G.diagonal().array() += static_cast<Scalar>(1e-15);

        // Apply CV features (L2 + Tier 2) to the full Gram
        nmf::variant::apply_w_cv_features(G, W_T, config);

        // Allocate B_W_full for Gram trick loss reuse
        // B_W_full.col(i) = H · A^T.col(i) = sum_j A(i,j) * H.col(j) (ALL entries)
        DenseMatrix<Scalar> B_W_full;
        if (need_train_loss_pre) {
            B_W_full.resize(k, m);
            B_W_full.setZero();
        }

#ifdef _OPENMP
        #pragma omp parallel num_threads(actual_threads)
        {
            DenseVector<Scalar> b_local_W(k);
            DenseVector<Scalar> x_local_W(k);
            DenseMatrix<Scalar> G_local_W_thr(k, k);
            DenseMatrix<Scalar> H_test_thr(k, est_max_test);
            std::vector<int> test_cols_local;
            test_cols_local.reserve(static_cast<size_t>(est_max_test));
            std::vector<int> excluded_cols_local;
            DenseVector<Scalar> b_full_W(k);  // full RHS (train + test) for B_W_full

            #pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < m; ++i) {
                if (is_zi && iter > 0) {
                    detail::compute_train_rhs_W(
                        A_imputed, H, i, mask, b_local_W, test_cols_local,
                        config.mask_zeros, n);
                } else if constexpr (is_sparse_v<MatrixType>) {
                    detail::compute_train_rhs_W(
                        A_T, H, i, mask, b_local_W, test_cols_local,
                        config.mask_zeros, n);
                } else {
                    detail::compute_train_rhs_W(
                        A, H, i, mask, b_local_W, test_cols_local,
                        config.mask_zeros, n);
                }

                // Accumulate full RHS for Gram trick loss (train + test entries)
                if (need_train_loss_pre && !test_cols_local.empty()) {
                    b_full_W = b_local_W;
                    if constexpr (is_sparse_v<MatrixType>) {
                        // Add back held-out contributions from A_T column i
                        for (int idx : test_cols_local) {
                            if (config.mask_zeros) {
                                for (typename SparseMatrix<Scalar>::InnerIterator it(A_T, i); it; ++it) {
                                    if (it.row() == idx) {
                                        b_full_W.noalias() += it.value() * H.col(idx);
                                        break;
                                    }
                                }
                            } else {
                                Scalar val = 0;
                                for (typename SparseMatrix<Scalar>::InnerIterator it(A_T, i); it; ++it) {
                                    if (it.row() == idx) { val = it.value(); break; }
                                    if (it.row() > idx) break;
                                }
                                if (val != Scalar(0)) {
                                    b_full_W.noalias() += val * H.col(idx);
                                }
                            }
                        }
                    } else {
                        for (int idx : test_cols_local) {
                            Scalar val = A(i, idx);
                            if (val != Scalar(0)) {
                                b_full_W.noalias() += val * H.col(idx);
                            }
                        }
                    }
                    B_W_full.col(i) = b_full_W;
                } else if (need_train_loss_pre) {
                    B_W_full.col(i) = b_local_W;  // no test entries for this row
                }

                // User mask adjustment for W-update
                const auto& gram_cols = use_mask
                    ? (([&]() {
                           if constexpr (is_sparse_v<MatrixType>) {
                               detail::adjust_rhs_for_user_mask_w(
                                   A_T, H, mask_T, i, b_local_W, test_cols_local, excluded_cols_local);
                           } else {
                               detail::adjust_rhs_for_user_mask_w(
                                   A, H, mask_T, i, b_local_W, test_cols_local, excluded_cols_local);
                           }
                       }()),
                       excluded_cols_local)
                    : test_cols_local;

                if (config.requires_irls()) {
                    // IRLS path: builds weighted Gram per row
                    x_local_W = W_T.col(i);
                    if (is_zi && iter > 0) {
                        detail::irls_solve_row_cv(
                            A_imputed, H, i, x_local_W, gram_cols,
                            config.mask_zeros, config.loss,
                            config, W_T, n);
                    } else if constexpr (is_sparse_v<MatrixType>) {
                        detail::irls_solve_row_cv(
                            A_T, H, i, x_local_W, gram_cols,
                            config.mask_zeros, config.loss,
                            config, W_T, n);
                    } else {
                        detail::irls_solve_row_cv(
                            A, H, i, x_local_W, gram_cols,
                            config.mask_zeros, config.loss,
                            config, W_T, n);
                    }
                    W_T.col(i) = x_local_W;
                } else {
                    // MSE path: Gram correction + solver dispatch
                    G_local_W_thr = G;
                    if (!gram_cols.empty()) {
                        const int n_test = static_cast<int>(gram_cols.size());
                        for (int idx = 0; idx < n_test; ++idx) {
                            H_test_thr.col(idx) = H.col(gram_cols[idx]);
                        }
                        auto H_sub = H_test_thr.leftCols(n_test);
                        G_local_W_thr.template selfadjointView<Eigen::Lower>()
                            .rankUpdate(H_sub, Scalar(-1));
                        G_local_W_thr.template triangularView<Eigen::Upper>() =
                            G_local_W_thr.transpose();
                    }

                    x_local_W = W_T.col(i);
                    if (config.solver_mode == 1) {
                        primitives::detail::cholesky_clip_col(
                            G_local_W_thr, b_local_W.data(), x_local_W.data(), k,
                            config.W.L1, Scalar(0), config.W.nonneg,
                            config.cd_max_iter, config.cd_tol);
                    } else {
                        primitives::detail::cd_nnls_col_fixed(
                            G_local_W_thr, b_local_W.data(), x_local_W.data(), k,
                            config.W.L1, Scalar(0), config.W.nonneg,
                            config.cd_max_iter);
                    }
                    W_T.col(i) = x_local_W;
                }
            }
        }
#else
        std::vector<int> test_cols;
        test_cols.reserve(static_cast<size_t>(est_max_test));
        std::vector<int> excluded_cols_serial;
        DenseVector<Scalar> b_full_serial(k);  // full RHS for B_W_full

        for (int i = 0; i < m; ++i) {
            if (is_zi && iter > 0) {
                detail::compute_train_rhs_W(
                    A_imputed, H, i, mask, b, test_cols,
                    config.mask_zeros, n);
            } else if constexpr (is_sparse_v<MatrixType>) {
                detail::compute_train_rhs_W(
                    A_T, H, i, mask, b, test_cols,
                    config.mask_zeros, n);
            } else {
                detail::compute_train_rhs_W(
                    A, H, i, mask, b, test_cols,
                    config.mask_zeros, n);
            }

            // Accumulate full RHS for Gram trick loss (train + test entries)
            if (need_train_loss_pre && !test_cols.empty()) {
                b_full_serial = b;
                if constexpr (is_sparse_v<MatrixType>) {
                    for (int jj : test_cols) {
                        if (config.mask_zeros) {
                            for (typename SparseMatrix<Scalar>::InnerIterator it(A_T, i); it; ++it) {
                                if (it.row() == jj) {
                                    b_full_serial.noalias() += it.value() * H.col(jj);
                                    break;
                                }
                            }
                        } else {
                            Scalar val = 0;
                            for (typename SparseMatrix<Scalar>::InnerIterator it(A_T, i); it; ++it) {
                                if (it.row() == jj) { val = it.value(); break; }
                                if (it.row() > jj) break;
                            }
                            if (val != Scalar(0)) {
                                b_full_serial.noalias() += val * H.col(jj);
                            }
                        }
                    }
                } else {
                    for (int jj : test_cols) {
                        Scalar val = A(i, jj);
                        if (val != Scalar(0)) {
                            b_full_serial.noalias() += val * H.col(jj);
                        }
                    }
                }
                B_W_full.col(i) = b_full_serial;
            } else if (need_train_loss_pre) {
                B_W_full.col(i) = b;
            }

            // User mask adjustment for W-update
            const auto& gram_cols = use_mask
                ? (([&]() {
                       if constexpr (is_sparse_v<MatrixType>) {
                           detail::adjust_rhs_for_user_mask_w(
                               A_T, H, mask_T, i, b, test_cols, excluded_cols_serial);
                       } else {
                           detail::adjust_rhs_for_user_mask_w(
                               A, H, mask_T, i, b, test_cols, excluded_cols_serial);
                       }
                   }()),
                   excluded_cols_serial)
                : test_cols;

            if (config.requires_irls()) {
                x = W_T.col(i);
                if (is_zi && iter > 0) {
                    detail::irls_solve_row_cv(
                        A_imputed, H, i, x, gram_cols,
                        config.mask_zeros, config.loss,
                        config, W_T, n);
                } else if constexpr (is_sparse_v<MatrixType>) {
                    detail::irls_solve_row_cv(
                        A_T, H, i, x, gram_cols,
                        config.mask_zeros, config.loss,
                        config, W_T, n);
                } else {
                    detail::irls_solve_row_cv(
                        A, H, i, x, gram_cols,
                        config.mask_zeros, config.loss,
                        config, W_T, n);
                }
                W_T.col(i) = x;
            } else {
                G_local = G;
                if (!gram_cols.empty()) {
                    const int n_tc = static_cast<int>(gram_cols.size());
                    for (int idx = 0; idx < n_tc; ++idx) {
                        W_test_buf.col(idx) = H.col(gram_cols[idx]);
                    }
                    auto H_sub = W_test_buf.leftCols(n_tc);
                    G_local.template selfadjointView<Eigen::Lower>()
                        .rankUpdate(H_sub, Scalar(-1));
                    G_local.template triangularView<Eigen::Upper>() =
                        G_local.transpose();
                }

                x = W_T.col(i);
                if (config.solver_mode == 1) {
                    primitives::detail::cholesky_clip_col(
                        G_local, b.data(), x.data(), k,
                        config.W.L1, Scalar(0), config.W.nonneg,
                        config.cd_max_iter, config.cd_tol);
                } else {
                    primitives::detail::cd_nnls_col_fixed(
                        G_local, b.data(), x.data(), k,
                        config.W.L1, Scalar(0), config.W.nonneg,
                        config.cd_max_iter);
                }
                W_T.col(i) = x;
            }
        }
#endif

        // Post-NNLS bounds and angular on W
        if (config.W.upper_bound > 0)
            features::apply_upper_bound(W_T, config.W.upper_bound);
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
        // Compute train and test loss
        // ==============================================================

        // Determine if we need loss this iteration
        // Loss is needed for: (1) patience-based early stopping, (2) loss history, (3) verbose
        const bool need_loss_this_iter =
            (config.cv_patience > 0) ||               // patience-based early stopping needs test loss
            config.track_loss_history ||               // loss history requested
            config.verbose ||                          // verbose prints loss
            (iter == 0) ||                             // always compute first iteration
            (iter == config.max_iter - 1);             // always compute last iteration

        // We always need test_loss for patience. We only need train_loss for
        // loss_history/verbose/result reporting. The Gram trick for train loss
        // is expensive, so skip it when track_train_loss is false.
        const bool need_train_loss = need_loss_this_iter &&
            (config.track_train_loss || config.verbose ||
             iter == 0 || iter == config.max_iter - 1);

        Scalar test_sq_error = 0;
        Scalar train_sq_error = 0;
        int64_t n_test_total = 0;
        int64_t n_train_total = 0;
        Scalar train_loss = 0;
        Scalar test_loss = 0;

        if (need_loss_this_iter) {

        // Apply d scaling to W_T for loss: recon = W_T^T · diag(d) · H = W_Td^T · H
        DenseMatrix<Scalar> W_Td_loss = W_T;
        for (int i = 0; i < k; ++i) W_Td_loss.row(i) *= d(i);

        if (config.requires_irls() || use_mask) {
            // Non-MSE or masked: compute train and test loss explicitly per element
            // When use_mask, we skip the Gram trick because it includes masked entries
            FactorNet::LossConfig<Scalar> loss_cfg = config.loss;
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64) num_threads(actual_threads) \
                reduction(+:test_sq_error, train_sq_error, n_test_total, n_train_total)
#endif
            for (int j = 0; j < n; ++j) {
                DenseVector<Scalar> wdh_j = W_Td_loss.transpose() * H.col(j);

                if constexpr (is_sparse_v<MatrixType>) {
                    if (config.mask_zeros) {
                        for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                            if (use_mask && detail::is_user_masked_h(it.row(), j, *config.mask))
                                continue;
                            Scalar pred = wdh_j(it.row());
                            Scalar th = is_gp ? theta_vec(it.row()) : Scalar(0);
                            Scalar loss_val = compute_loss(it.value(), pred, loss_cfg, th);
                            if (mask.is_holdout(it.row(), j)) {
                                test_sq_error += loss_val;
                                n_test_total++;
                            } else {
                                train_sq_error += loss_val;
                                n_train_total++;
                            }
                        }
                    } else {
                        typename MatrixType::InnerIterator it(A, j);
                        for (int i = 0; i < m; ++i) {
                            if (use_mask && detail::is_user_masked_h(i, j, *config.mask)) {
                                if (it && it.row() == i) ++it;
                                continue;
                            }
                            Scalar actual = 0;
                            if (it && it.row() == i) { actual = it.value(); ++it; }
                            Scalar pred = wdh_j(i);
                            Scalar th = is_gp ? theta_vec(i) : Scalar(0);
                            Scalar loss_val = compute_loss(actual, pred, loss_cfg, th);
                            if (mask.is_holdout(i, j)) {
                                test_sq_error += loss_val;
                                n_test_total++;
                            } else {
                                train_sq_error += loss_val;
                                n_train_total++;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < m; ++i) {
                        if (use_mask && detail::is_user_masked_h(i, j, *config.mask))
                            continue;
                        Scalar val = A(i, j);
                        if (config.mask_zeros && val == Scalar(0)) continue;
                        Scalar pred = wdh_j(i);
                        Scalar th = is_gp ? theta_vec(i) : Scalar(0);
                        Scalar loss_val = compute_loss(val, pred, loss_cfg, th);
                        if (mask.is_holdout(i, j)) {
                            test_sq_error += loss_val;
                            n_test_total++;
                        } else {
                            train_sq_error += loss_val;
                            n_train_total++;
                        }
                    }
                }
            }
        } else {
            // MSE path: optimized test loss + optional train loss via Gram trick

            // Compute test loss using per-entry dot products (avoiding full prediction vectors)
            // Cost: O(m*n) mask checks + O(n_test * k) dot products
            // vs old approach: O(m*n*k) for full prediction vectors
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64) num_threads(actual_threads) \
                reduction(+:test_sq_error, n_test_total)
#endif
            for (int j = 0; j < n; ++j) {
                if constexpr (is_sparse_v<MatrixType>) {
                    if (config.mask_zeros) {
                        // Only check non-zero entries; use dot product for predictions
                        for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                            if (mask.is_holdout(it.row(), j)) {
                                Scalar pred = W_Td_loss.col(it.row()).dot(H.col(j));
                                Scalar diff = it.value() - pred;
                                test_sq_error += diff * diff;
                                n_test_total++;
                            }
                        }
                    } else {
                        // Non-mask_zeros: test set includes zeros. Iterate all rows
                        // but only compute dot product at held-out entries.
                        typename MatrixType::InnerIterator it(A, j);
                        for (int i = 0; i < m; ++i) {
                            if (mask.is_holdout(i, j)) {
                                Scalar actual = 0;
                                if (it && it.row() == i) { actual = it.value(); ++it; }
                                Scalar pred = W_Td_loss.col(i).dot(H.col(j));
                                Scalar diff = actual - pred;
                                test_sq_error += diff * diff;
                                n_test_total++;
                            } else {
                                if (it && it.row() == i) ++it;
                            }
                        }
                    }
                } else {
                    // Dense: iterate all entries, dot product only at held-out
                    for (int i = 0; i < m; ++i) {
                        Scalar val = A(i, j);
                        if (config.mask_zeros && val == Scalar(0)) continue;
                        if (mask.is_holdout(i, j)) {
                            Scalar pred = W_Td_loss.col(i).dot(H.col(j));
                            Scalar diff = val - pred;
                            test_sq_error += diff * diff;
                            n_test_total++;
                        }
                    }
                }
            }

            // Train loss via Gram trick — REUSES B_W_full from W-update
            // B_W_full.col(i) = H · A^T.col(i) was accumulated during W-update
            //
            // Total loss = trAtA - 2·cross + recon where:
            //   cross = Σ_r d(r) · W_T.row(r) · B_W_full.row(r)^T   — O(k·m)
            //   recon = Σ_{r,s} d(r)·d(s) · G_W_new(r,s) · G_H_saved(r,s) — O(k²)
            //   G_W_new = W_T · W_T^T  — O(k²·m)
            //   G_H_saved was saved before W-update regularization  — already O(k²)
            //
            // This avoids the previous O(nnz·k) B_full recomputation.
            if (need_train_loss) {
                // Cross-term: Σ_r d(r) * W_T_new.row(r).dot(B_W_full.row(r))
                // O(k·m)
                Scalar cross_term = 0;
                for (int r = 0; r < k; ++r) {
                    cross_term += d(r) * W_T.row(r).dot(B_W_full.row(r));
                }

                // Gram of post-W-update W_T: G_W_new = W_T · W_T^T, O(k²·m)
                DenseMatrix<Scalar> G_W_new;
                primitives::gram<Resource>(W_T, G_W_new);

                // Recon norm: Σ_{r,s} d(r)·d(s) · G_W_new(r,s) · G_H_saved(r,s)
                // O(k²)
                Scalar recon_norm = 0;
                for (int r = 0; r < k; ++r) {
                    for (int s = 0; s < k; ++s) {
                        recon_norm += d(r) * d(s) * G_W_new(r, s) * G_H_saved(r, s);
                    }
                }

                Scalar total_sq_error = std::max(
                    trAtA - static_cast<Scalar>(2) * cross_term + recon_norm,
                    static_cast<Scalar>(0));

                train_sq_error = std::max(total_sq_error - test_sq_error, Scalar(0));

                int64_t total_entries;
                if (config.mask_zeros) {
                    total_entries = nnz;
                } else {
                    total_entries = static_cast<int64_t>(m) * n;
                }
                n_train_total = total_entries - n_test_total;
            }
        }

        train_loss = (n_train_total > 0)
            ? train_sq_error / static_cast<Scalar>(n_train_total) : 0;
        test_loss = (n_test_total > 0)
            ? test_sq_error / static_cast<Scalar>(n_test_total) : 0;

        } // end need_loss_this_iter

        auto t_after_loss = tick();
        t_loss_us += elapsed_us(t_after_w, t_after_loss);

        // Track loss history
        if (config.track_loss_history) {
            result.loss_history.push_back(train_loss);
            result.test_loss_history.push_back(test_loss);
        }

        result.train_loss = train_loss;
        result.test_loss = test_loss;

        // Compute relative change for history/callback (before early stopping)
        Scalar cv_rel_change = 0;
        if (need_loss_this_iter && iter > 0) {
            cv_rel_change = std::abs(prev_conv_loss - test_loss)
                          / (std::abs(prev_conv_loss) + static_cast<Scalar>(1e-15));
        }

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

        // Early stopping
        if (test_loss < best_test_loss) {
            best_test_loss = test_loss;
            best_iter = iter;
            patience_count = 0;
        } else {
            patience_count++;
        }

        if (config.verbose) {
            FACTORNET_PRINT_IMPL("Iter %d: train=%g, test=%g, best=%g  [H=%ldus W=%ldus loss=%ldus]\n",
                              iter + 1, train_loss, test_loss, best_test_loss,
                              (long)elapsed_us(t_iter_start, t_after_h),
                              (long)elapsed_us(t_after_h, t_after_w),
                              (long)elapsed_us(t_after_w, t_after_loss));
        }

        if (config.cv_patience > 0 && patience_count >= config.cv_patience) {
            result.iterations = iter + 1;
            result.converged = false;
            result.best_test_loss = best_test_loss;
            result.best_iter = best_iter;
            break;
        }

        // Loss-based convergence check (using test loss relative change)
        if (need_loss_this_iter && iter > 0) {
            result.final_tol = cv_rel_change;

            if (cv_rel_change < config.tol) {
                result.iterations = iter + 1;
                result.converged = true;
                result.best_test_loss = best_test_loss;
                result.best_iter = best_iter;
                break;
            }
        }
        if (need_loss_this_iter) {
            prev_conv_loss = test_loss;
        }

        result.iterations = iter + 1;
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

    // Absorb d into H for final output
    for (int i = 0; i < k; ++i) {
        H.row(i) *= d(i);
    }

    result.W = W_T.transpose();  // k×m → m×k
    result.d = d;
    result.H = H;
    if (is_gp) result.theta = theta_vec;
    if (is_nb && nb_size_vec.size() > 0) result.theta = nb_size_vec;
    if ((is_gamma || is_invgauss || is_tweedie) && phi_vec.size() > 0)
        result.dispersion = phi_vec;
    if (pi_row_vec.size() > 0) result.pi_row = pi_row_vec;
    if (pi_col_vec.size() > 0) result.pi_col = pi_col_vec;

    if (result.best_test_loss == 0)
        result.best_test_loss = best_test_loss;
    if (result.best_iter == 0)
        result.best_iter = best_iter;

    if (config.sort_model)
        result.sort();

    return result;
}

}  // namespace nmf
}  // namespace FactorNet

