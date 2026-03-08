// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file deflation.hpp
 * @brief Deflation-based truncated SVD/PCA with auto-rank
 *
 * Solves one factor at a time via rank-1 ALS on the deflated residual.
 * Each factor needs only SpMV and scalar-per-element regularization.
 * Auto-rank is built in: test MSE on a speckled holdout is evaluated
 * after each factor, with patience-based stopping.
 *
 * Supports: SVD, PCA (centering), NNSVD, NNPCA, Sparse PCA, Semi-NMF SVD.
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/svd/spmv.hpp>
#include <FactorNet/svd/test_entries.hpp>  // Shared TestEntries
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/rng/rng.hpp>
#include <FactorNet/features/angular.hpp>       // angular penalty (Tier 2)
#include <FactorNet/features/graph_reg.hpp>   // graph Laplacian (Tier 2)

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// Print function: use Rcpp::Rcout when available
#if defined(RCPP_VERSION) || __has_include(<Rcpp.h>)
#include <Rcpp.h>
#define SVD_PRINT(...) { char buf[256]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); Rcpp::Rcout << buf; }
#define SVD_CHECK_INTERRUPT() Rcpp::checkUserInterrupt()
#else
#include <cstdio>
#define SVD_PRINT(...) std::printf(__VA_ARGS__)
#define SVD_CHECK_INTERRUPT() ((void)0)
#endif

namespace FactorNet {
namespace svd {

// ============================================================================
// Huber IRLS weight computation for robust SVD
// ============================================================================

/**
 * @brief Compute Huber weight for a residual value.
 *
 * w(r) = 1                   if |r| <= delta
 *       = delta / |r|        if |r| > delta
 *
 * This downweights large residuals (outliers) while leaving small residuals
 * (inliers) unaffected. The result is bounded in [0, 1].
 */
template<typename Scalar>
inline Scalar huber_weight(Scalar residual, Scalar delta) {
    Scalar abs_r = std::abs(residual);
    if (abs_r <= delta) return static_cast<Scalar>(1);
    return delta / abs_r;
}

/**
 * @brief Compute row-level IRLS weights for robust SVD v-update.
 *
 * For fixed u, the per-row residual norm measures how well the current
 * rank-1 approximation fits each row. Rows with large residuals (outliers)
 * are downweighted via Huber weights.
 *
 * residual_i = ||R_k(i,:) * v - sigma * u_i|| where R_k = A - sum prior factors
 * But we approximate: residual_i = |(A' u)_j - sigma * v_j| summed appropriately.
 *
 * In practice, we compute: for each row i, r_i = (Av)_i - sigma * u_i
 * (the scalar residual for row i in the current factor).
 * weight_i = huber_weight(r_i / scale, delta)
 *
 * @param[in]  Av_result  Forward SpMV result A*v (m x 1), deflation-corrected
 * @param[in]  u          Current left singular vector (m x 1)
 * @param[in]  sigma      Current singular value estimate
 * @param[in]  delta      Huber delta parameter
 * @param[out] weights    Output weight vector (m x 1)
 * @param[in]  threads    Number of OMP threads
 */
template<typename Scalar>
void compute_row_weights(const DenseVector<Scalar>& Av_result,
                         const DenseVector<Scalar>& u,
                         Scalar sigma,
                         Scalar delta,
                         DenseVector<Scalar>& weights,
                         int threads)
{
    const int m = static_cast<int>(u.size());
    weights.resize(m);

    // Compute residuals and MAD-based robust scale estimate
    DenseVector<Scalar> residuals(m);
    for (int i = 0; i < m; ++i) {
        residuals(i) = Av_result(i) - sigma * u(i);
    }

    // Robust scale: MAD / 0.6745
    DenseVector<Scalar> abs_resid = residuals.cwiseAbs();
    std::nth_element(abs_resid.data(), abs_resid.data() + m / 2, abs_resid.data() + m);
    Scalar mad = abs_resid(m / 2);
    Scalar scale = mad / static_cast<Scalar>(0.6745);
    if (scale < std::numeric_limits<Scalar>::epsilon() * 100) {
        scale = static_cast<Scalar>(1);  // fallback if nearly all residuals are zero
    }

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int i = 0; i < m; ++i) {
        weights(i) = huber_weight(residuals(i) / scale, delta);
    }
}

/**
 * @brief Compute column-level IRLS weights for robust SVD u-update.
 *
 * For fixed v, the per-column residual measures fit quality per column.
 * r_j = (A'u)_j - sigma * v_j
 */
template<typename Scalar>
void compute_col_weights(const DenseVector<Scalar>& Atu_result,
                         const DenseVector<Scalar>& v,
                         Scalar sigma,
                         Scalar delta,
                         DenseVector<Scalar>& weights,
                         int threads)
{
    const int n = static_cast<int>(v.size());
    weights.resize(n);

    DenseVector<Scalar> residuals(n);
    for (int j = 0; j < n; ++j) {
        residuals(j) = Atu_result(j) - sigma * v(j);
    }

    // Robust scale: MAD / 0.6745
    DenseVector<Scalar> abs_resid = residuals.cwiseAbs();
    std::nth_element(abs_resid.data(), abs_resid.data() + n / 2, abs_resid.data() + n);
    Scalar mad = abs_resid(n / 2);
    Scalar scale = mad / static_cast<Scalar>(0.6745);
    if (scale < std::numeric_limits<Scalar>::epsilon() * 100) {
        scale = static_cast<Scalar>(1);
    }

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int j = 0; j < n; ++j) {
        weights(j) = huber_weight(residuals(j) / scale, delta);
    }
}

// ============================================================================
// Per-element regularization
// ============================================================================

/**
 * @brief Apply L1/L2/L21/nonneg/bounds regularization element-wise
 *
 * For the v-update with fixed u, the least-squares solution is:
 *   v_j = (u' r_j) / ||u||^2
 *
 * The regularized update modifies this:
 *   L1: v_j = sign(v_j) * max(0, |v_j| - lambda1 / (2 * ||u||^2))
 *   L2: v_j = v_j / (1 + lambda2 / ||u||^2)
 *   L21: v_j = v_j / (1 + L21 / (||v||₂ · ||u||^2))   [adaptive L2 — drives small-norm factors to zero]
 *   nonneg: v_j = max(0, v_j)
 *   bounds: v_j = clip(v_j, 0, upper_bound)
 *
 * L21 for rank-1: the factor is a single vector, so "group sparsity"
 * degenerates to adaptive L2. Rows with small norm get higher penalty.
 * Effectively: L2_eff = L2 + L21/||v||₂
 */
template<typename Scalar>
void apply_regularization(DenseVector<Scalar>& x,
                          Scalar L1, Scalar L2,
                          bool nonneg, Scalar upper_bound,
                          Scalar norm_sq,
                          Scalar L21 = 0)
{
    const int n = static_cast<int>(x.size());

    // L21 adaptive shrinkage (adds to L2 effect, scaled by 1/||x||₂)
    Scalar L2_eff = L2;
    if (L21 > 0) {
        Scalar x_norm = x.norm();
        if (x_norm > static_cast<Scalar>(1e-10)) {
            L2_eff += L21 / x_norm;
        }
    }

    // L2 shrinkage (applied to denominator)
    if (L2_eff > 0) {
        Scalar scale = static_cast<Scalar>(1) / (static_cast<Scalar>(1) + L2_eff / norm_sq);
        x *= scale;
    }

    // L1 soft-thresholding
    if (L1 > 0) {
        Scalar threshold = L1 / (static_cast<Scalar>(2) * norm_sq);
        for (int i = 0; i < n; ++i) {
            Scalar val = x(i);
            if (val > threshold) {
                x(i) = val - threshold;
            } else if (val < -threshold) {
                x(i) = val + threshold;
            } else {
                x(i) = 0;
            }
        }
    }

    // Non-negativity
    if (nonneg) {
        x = x.cwiseMax(static_cast<Scalar>(0));
    }

    // Upper bound
    if (upper_bound > 0) {
        x = x.cwiseMin(upper_bound);
    }
}

// ============================================================================
// Angular penalty for deflation (decorrelation with prior factors)
// ============================================================================

/**
 * @brief Apply angular (orthogonality) penalty against prior factors
 *
 * For factor k's v-update, penalizes overlap with prior v-factors:
 *   v -= angular * sum_{r<k} (v_r' v) * v_r / norm_sq
 *
 * This matches krylov's angular penalty: G += angular * (V_prev * V_prev')
 * projected onto the rank-1 residual. Applied AFTER standard regularization
 * but BEFORE normalization.
 */
template<typename Scalar>
void apply_angular_penalty(DenseVector<Scalar>& x,
                          const DenseMatrix<Scalar>& PriorFactors,
                          int n_prior,
                          Scalar angular_lambda,
                          Scalar norm_sq)
{
    if (angular_lambda <= 0 || n_prior <= 0) return;

    auto F = PriorFactors.leftCols(n_prior);  // (dim x n_prior)
    DenseVector<Scalar> dots = F.transpose() * x;  // overlaps
    x.noalias() -= (angular_lambda / norm_sq) * F * dots;
}

// ============================================================================
// Graph Laplacian penalty for deflation
// ============================================================================

/**
 * @brief Apply graph Laplacian smoothness penalty
 *
 * For v-update: v -= lambda * L * v / norm_sq
 * This is the gradient-step approximation of solving
 * (norm_sq * I + lambda * L) * v = rhs.
 *
 * Applied AFTER standard regularization, BEFORE normalization.
 */
template<typename Scalar>
void apply_graph_penalty(DenseVector<Scalar>& x,
                        const SparseMatrix<Scalar>* laplacian,
                        Scalar graph_lambda,
                        Scalar norm_sq)
{
    if (!laplacian || graph_lambda <= 0) return;

    DenseVector<Scalar> Lx = (*laplacian) * x;
    x.noalias() -= (graph_lambda / norm_sq) * Lx;
}

// ============================================================================
// Deflation correction
// ============================================================================

/**
 * @brief Subtract contribution of prior factors from SpMV result
 *
 * For V-update: raw = A'u, we need R_k'u = A'u - sum_{r<k} sigma_r (u'u_r) v_r
 * For U-update: raw = Av, we need R_k v = Av - sum_{r<k} sigma_r (v'v_r) u_r
 *
 * @param raw  In/out: SpMV result to correct
 * @param x    The fixed vector (u for v-update, v for u-update)
 * @param Factors  Previous factors matrix (m×k_prev or n×k_prev)
 * @param sigma  Previous singular values (k_prev)
 * @param PrevX  Previous x-vectors (m×k_prev or n×k_prev) to compute dot products
 */
template<typename Scalar>
void deflation_correct(DenseVector<Scalar>& raw,
                       const DenseVector<Scalar>& x,
                       const DenseMatrix<Scalar>& Factors,
                       const DenseVector<Scalar>& sigma,
                       const DenseMatrix<Scalar>& PrevX,
                       int n_prev)
{
    if (n_prev <= 0) return;

    // Batched correction via matrix ops (single BLAS call):
    //   dots = PrevX(:, 0:n_prev-1)' * x          → (n_prev × 1)
    //   coeffs = sigma(0:n_prev-1) .* dots         → (n_prev × 1)
    //   raw -= Factors(:, 0:n_prev-1) * coeffs     → GEMV
    auto PX = PrevX.leftCols(n_prev);          // view, no copy
    auto F  = Factors.leftCols(n_prev);        // view, no copy
    DenseVector<Scalar> dots = PX.transpose() * x;
    dots.array() *= sigma.head(n_prev).array();
    raw.noalias() -= F * dots;
}

// ============================================================================
// Train loss computation (Gram trick for deflation)
// ============================================================================

/**
 * @brief Compute train MSE after k factors using the algebraic identity
 *
 * ||A - sum sigma_r u_r v_r'||^2_F = ||A||^2_F - 2 sum sigma_r (u_r' A v_r)
 *                                     + sum_{r,s} sigma_r sigma_s (u_r'u_s)(v_r'v_s)
 *
 * For orthogonal factors (unconstrained SVD), the cross-terms vanish and
 * this simplifies to ||A||^2_F - sum sigma_r^2. But with regularization
 * or non-negativity, factors may not be orthogonal, so we compute the full
 * expression.
 *
 * For CV: train_loss = (total_loss * m * n - test_loss * n_test) / n_train
 * But since we track test entries separately, we just compute the full loss.
 */

// ============================================================================
// Main algorithm: deflation_svd
// ============================================================================

/**
 * @brief Deflation-based truncated SVD/PCA with optional auto-rank
 *
 * @tparam MatrixType  Eigen sparse or dense matrix type
 * @tparam Scalar      Element precision (float or double)
 * @param A            Input matrix (m × n)
 * @param config       SVD configuration
 * @return SVDResult with factors, loss trajectories, and selected rank
 */
template<typename MatrixType, typename Scalar>
SVDResult<Scalar> deflation_svd(const MatrixType& A,
                                const SVDConfig<Scalar>& config)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k_max = config.k_max;

    SVDResult<Scalar> result;
    result.centered = config.center;

    // Compute row means if PCA centering is requested
    DenseVector<Scalar> row_means;
    const DenseVector<Scalar>* row_means_ptr = nullptr;
    if (config.center) {
        row_means = compute_row_means<MatrixType, Scalar>(A, config.threads);
        row_means_ptr = &row_means;
        result.row_means = row_means;  // Store for reconstruction
    }

    // Compute row SDs for correlation PCA scaling
    DenseVector<Scalar> row_inv_sds;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    if (config.scale) {
        DenseVector<Scalar> row_sds = compute_row_sds<MatrixType, Scalar>(A, row_means, config.threads);
        row_inv_sds = row_sds.cwiseInverse();
        result.row_sds = row_sds;
        result.scaled = true;
        sd_ptr = &row_inv_sds;
    }

    // Compute ||A||^2_F (or ||A_centered||^2_F for PCA)
    Scalar A_norm_sq = 0;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
#ifdef _OPENMP
        const int nthreads_norm = config.threads > 0 ? config.threads : omp_get_max_threads();
        #pragma omp parallel for reduction(+:A_norm_sq) num_threads(nthreads_norm) schedule(static)
#endif
        for (int j = 0; j < A.outerSize(); ++j) {
            for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                A_norm_sq += static_cast<Scalar>(it.value()) * static_cast<Scalar>(it.value());
            }
        }
    } else {
        A_norm_sq = A.squaredNorm();
    }
    if (config.center) {
        A_norm_sq -= static_cast<Scalar>(n) * row_means.squaredNorm();
    }
    if (config.scale) {
        A_norm_sq = static_cast<Scalar>(m) * static_cast<Scalar>(n);
    }
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // Pre-allocate forward SpMV workspace (avoids per-call malloc)
    int eff_threads = config.threads;
#ifdef _OPENMP
    if (eff_threads <= 0) eff_threads = omp_get_max_threads();
#else
    eff_threads = 1;
#endif
    SpmvWorkspace<Scalar> ws;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        ws.init(m, eff_threads);
    }

    // Setup speckled mask for auto-rank
    const bool do_cv = config.is_cv();
    Scalar cv_denom_correction = Scalar(1);
    TestEntries<Scalar> test_entries;
    Scalar best_test_loss = std::numeric_limits<Scalar>::max();
    int best_k = 0;
    int patience_counter = 0;

    // Training matrix: when CV or obs_mask is active, we zero masked entries
    // so the model never sees them.  For non-CV/non-masked, these remain empty (unused).
    // OwnedSparse uses the INPUT matrix's scalar (e.g. double) even when
    // Scalar=float.  SpMV/SpMM handle the cast via static_cast internally.
    using MatScalar = typename MatrixType::Scalar;
    using OwnedSparse = Eigen::SparseMatrix<MatScalar, Eigen::ColMajor, typename MatrixType::StorageIndex>;
    OwnedSparse A_train_sparse;
    DenseMatrix<Scalar> A_train_dense;
    bool use_train_matrix = false;

    // ================================================================
    // obs_mask: zero user-specified unobserved entries BEFORE CV
    // ================================================================
    if (config.obs_mask != nullptr) {
        int n_masked = 0;
        if constexpr (is_sparse_matrix<MatrixType>::value) {
            // Copy A, then zero entries where mask is nonzero
            A_train_sparse = A.template cast<MatScalar>();
            A_train_sparse.makeCompressed();
            const auto& mask = *config.obs_mask;
            for (int j = 0; j < mask.outerSize(); ++j) {
                for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit) {
                    // Zero this entry in the training matrix
                    // Find (mit.row(), j) in A_train_sparse
                    for (typename OwnedSparse::InnerIterator ait(A_train_sparse, j); ait; ++ait) {
                        if (ait.row() == mit.row()) {
                            ait.valueRef() = MatScalar(0);
                            ++n_masked;
                            break;
                        }
                    }
                }
            }
        } else {
            // Dense: copy and zero
            A_train_dense = A.template cast<Scalar>();
            const auto& mask = *config.obs_mask;
            for (int j = 0; j < mask.outerSize(); ++j) {
                for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit) {
                    A_train_dense(mit.row(), j) = Scalar(0);
                    ++n_masked;
                }
            }
        }
        use_train_matrix = true;
        if (config.verbose) {
            SVD_PRINT("  obs_mask: %d entries masked as unobserved\n", n_masked);
        }
    }

    if (do_cv) {
        int64_t nnz_est = 0;
        if constexpr (is_sparse_matrix<MatrixType>::value) {
            nnz_est = A.nonZeros();
        } else {
            nnz_est = static_cast<int64_t>(m) * n;
        }

        FactorNet::nmf::LazySpeckledMask<Scalar> mask(m, n, nnz_est,
                              static_cast<double>(config.test_fraction),
                              config.effective_cv_seed(),
                              config.mask_zeros);

        // Collect test entries from ORIGINAL matrix (true held-out values)
        // Note: if obs_mask is active, entries already masked are zero in A_train
        // but may still appear in test_entries from original A. The test loss
        // for those entries would be evaluated against a zero baseline, which is
        // reasonable (we expect the model to predict zero for unobserved entries).
        if constexpr (is_sparse_matrix<MatrixType>::value) {
            test_entries = init_test_entries_sparse<Scalar>(A, mask, row_means_ptr);
        } else {
            test_entries = init_test_entries_dense<Scalar>(A, mask, row_means_ptr);
        }

        // Create training matrix with holdout entries zeroed.
        // If obs_mask was already applied, start from the masked copy;
        // otherwise create fresh from original A.
        int n_zeroed = 0;
        if (use_train_matrix) {
            // obs_mask already applied → additionally zero CV holdout entries
            if constexpr (is_sparse_matrix<MatrixType>::value) {
                for (int j = 0; j < A_train_sparse.outerSize(); ++j) {
                    for (typename OwnedSparse::InnerIterator it(A_train_sparse, j); it; ++it) {
                        if (mask.is_holdout(it.row(), j)) {
                            it.valueRef() = MatScalar(0);
                            ++n_zeroed;
                        }
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        if (mask.is_holdout(i, j)) {
                            A_train_dense(i, j) = Scalar(0);
                            ++n_zeroed;
                        }
                    }
                }
            }
        } else {
            if constexpr (is_sparse_matrix<MatrixType>::value) {
                A_train_sparse = create_training_matrix_sparse<Scalar>(A, mask, n_zeroed);
            } else {
                A_train_dense = create_training_matrix_dense<Scalar>(A, mask, n_zeroed);
            }
        }
        use_train_matrix = true;

        // CV denominator correction: when holdout entries are zeroed, the ALS
        // denominator (||u||^2 for v-update, ||v||^2 for u-update) overcounts by
        // including contributions from held-out rows/columns. This causes
        // systematic shrinkage of singular values and underestimates optimal rank.
        // Correction: scale denominator by (1 - effective_holdout_fraction).
        if (config.mask_zeros) {
            // Only non-zero entries can be held out — correction is small for sparse data
            Scalar density = static_cast<Scalar>(nnz_est) /
                            (static_cast<Scalar>(m) * static_cast<Scalar>(n));
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction) * density;
        } else {
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        }

        if (config.verbose) {
            SVD_PRINT("  CV: %d test entries held out (%d zeroed), denom_correction=%.4f\n",
                      test_entries.count, n_zeroed, static_cast<double>(cv_denom_correction));
        }
    }

    const bool do_robust = config.is_robust();

    if (config.verbose && do_robust) {
        SVD_PRINT("  Robust SVD: Huber delta=%.4f, IRLS reweighting active\n",
                  static_cast<double>(config.robust_delta));
    }

    // SpMV helpers that dispatch to training matrix when CV is active
    auto spmv_t = [&](const DenseVector<Scalar>& u_vec, DenseVector<Scalar>& out) {
        if (use_train_matrix) {
            if constexpr (is_sparse_matrix<MatrixType>::value) {
                spmv_transpose(A_train_sparse, u_vec, out, row_means_ptr, eff_threads, sd_ptr);
            } else {
                spmv_transpose(A_train_dense, u_vec, out, row_means_ptr, eff_threads, sd_ptr);
            }
        } else {
            spmv_transpose(A, u_vec, out, row_means_ptr, eff_threads, sd_ptr);
        }
    };

    auto spmv_f = [&](const DenseVector<Scalar>& v_vec, DenseVector<Scalar>& out) {
        if (use_train_matrix) {
            if constexpr (is_sparse_matrix<MatrixType>::value) {
                spmv_forward(A_train_sparse, v_vec, out, row_means_ptr, eff_threads, &ws, sd_ptr);
            } else {
                spmv_forward(A_train_dense, v_vec, out, row_means_ptr, eff_threads, static_cast<SpmvWorkspace<Scalar>*>(nullptr), sd_ptr);
            }
        } else {
            spmv_forward(A, v_vec, out, row_means_ptr, eff_threads, &ws, sd_ptr);
        }
    };

    // Pre-allocate factor storage for k_max factors
    DenseMatrix<Scalar> U_all(m, k_max);
    DenseVector<Scalar> d_all(k_max);
    DenseMatrix<Scalar> V_all(n, k_max);
    d_all.setZero();

    // RNG for initialization
    rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));

    // Workspace vectors
    DenseVector<Scalar> u(m), v(n), u_old(m), u_prev(m);
    DenseVector<Scalar> spmv_result;

    // IRLS workspace (only allocated when robust)
    DenseVector<Scalar> row_weights, col_weights;
    DenseVector<Scalar> spmv_result_unweighted;  // for residual computation
    if (do_robust) {
        row_weights.resize(m);
        col_weights.resize(n);
        row_weights.setOnes();
        col_weights.setOnes();
    }

    int k_computed = 0;

    for (int k = 0; k < k_max; ++k) {
        SVD_CHECK_INTERRUPT();

        // ============================================================
        // Warm start + Nesterov acceleration for all factors
        // Power-step warm start gives good initial direction
        // Nesterov momentum halves iteration count for clustered SVs
        // ============================================================
        
        if (k == 0) {
            // Random initialization for first factor
            for (int i = 0; i < m; ++i) {
                u(i) = static_cast<Scalar>(rng.uniform<double>());
            }
        } else {
            // Power-step warm start: orthogonalize prior u, one A'A step
            u = U_all.col(k - 1);
            for (int r = 0; r < k; ++r) {
                u -= u.dot(U_all.col(r)) * U_all.col(r);
            }
            Scalar u_norm_init = u.norm();
            if (u_norm_init > std::numeric_limits<Scalar>::epsilon() * 100) {
                u /= u_norm_init;
                spmv_t(u, spmv_result);
                deflation_correct(spmv_result, u, V_all, d_all, U_all, k);
                v = spmv_result;
                Scalar v_norm_tmp = v.norm();
                if (v_norm_tmp > 0) v /= v_norm_tmp;
                spmv_f(v, spmv_result);
                deflation_correct(spmv_result, v, U_all, d_all, V_all, k);
                u = spmv_result;
                for (int r = 0; r < k; ++r) {
                    u -= u.dot(U_all.col(r)) * U_all.col(r);
                }
            } else {
                for (int i = 0; i < m; ++i) {
                    u(i) = static_cast<Scalar>(rng.uniform<double>());
                }
            }
        }
        Scalar u_norm = u.norm();
        if (u_norm > 0) u /= u_norm;

        // ============================================================
        // Adaptive tolerance: tol * sigma_1/sigma_{k-1}, capped at 100×
        // ============================================================
        Scalar tol_k = config.tol;
        if (k > 0 && d_all(0) > 0 && d_all(k - 1) > 0) {
            tol_k = config.tol * d_all(0) / d_all(k - 1);
            tol_k = std::min(tol_k, config.tol * static_cast<Scalar>(100));
        }

        Scalar sigma = 0;
        Scalar prev_sigma = 0;  // for loss-based convergence
        int iter = 0;
        u_prev = u;

        for (iter = 0; iter < config.max_iter; ++iter) {
            u_old = u;

            // Nesterov momentum extrapolation (disabled under IRLS — incompatible)
            Scalar beta = (!do_robust && iter > 1)
                          ? static_cast<Scalar>(iter - 1) / static_cast<Scalar>(iter + 2)
                          : static_cast<Scalar>(0);
            DenseVector<Scalar> u_hat = u + beta * (u - u_prev);
            u_prev = u;

            if (do_robust && iter > 0) {
                // ============================================================
                // IRLS: compute row weights from current u, v, sigma
                // residual_i = (Av)_i - sigma * u_i
                // ============================================================
                spmv_f(v, spmv_result_unweighted);
                deflation_correct(spmv_result_unweighted, v, U_all, d_all, V_all, k);
                compute_row_weights(spmv_result_unweighted, u, sigma,
                                    config.robust_delta, row_weights, eff_threads);

                // Column weights from A'u residuals
                spmv_t(u, spmv_result_unweighted);
                deflation_correct(spmv_result_unweighted, u, V_all, d_all, U_all, k);
                compute_col_weights(spmv_result_unweighted, v, sigma,
                                    config.robust_delta, col_weights, eff_threads);
            }

            // === V-update ===
            if (do_robust && iter > 0) {
                // Weighted v-update: v = A' diag(w_u) u_hat / (u_hat' diag(w_u) u_hat)
                DenseVector<Scalar> wu = u_hat.cwiseProduct(row_weights);
                spmv_t(wu, spmv_result);
                deflation_correct(spmv_result, wu, V_all, d_all, U_all, k);
                Scalar u_sq = wu.dot(u_hat);  // u_hat' diag(w) u_hat
                if (do_cv) u_sq *= cv_denom_correction;
                if (u_sq > 0) {
                    v = spmv_result / u_sq;
                } else {
                    v.setZero();
                    break;
                }
            } else {
                // Standard v-update
                spmv_t(u_hat, spmv_result);
                deflation_correct(spmv_result, u_hat, V_all, d_all, U_all, k);
                Scalar u_sq = u_hat.squaredNorm();
                if (do_cv) u_sq *= cv_denom_correction;
                if (u_sq > 0) {
                    v = spmv_result / u_sq;
                } else {
                    v.setZero();
                    break;
                }
            }

            {
                Scalar u_sq = u_hat.squaredNorm();
                if (do_cv) u_sq *= cv_denom_correction;
                apply_regularization(v, config.L1_v, config.L2_v,
                                     config.nonneg_v, config.upper_bound_v, u_sq,
                                     config.L21_v);
                apply_angular_penalty(v, V_all, k, config.angular_v, u_sq);
                apply_graph_penalty(v, config.graph_v, config.graph_v_lambda, u_sq);
            }

            sigma = v.norm();
            if (sigma > 0) {
                v /= sigma;
            } else {
                break;
            }

            // === U-update ===
            if (do_robust && iter > 0) {
                // Weighted u-update: u = A diag(w_v) v / (v' diag(w_v) v)
                DenseVector<Scalar> wv = v.cwiseProduct(col_weights);
                spmv_f(wv, spmv_result);
                deflation_correct(spmv_result, wv, U_all, d_all, V_all, k);
                Scalar v_sq = wv.dot(v);  // v' diag(w) v
                if (do_cv) v_sq *= cv_denom_correction;
                if (v_sq > 0) {
                    u = spmv_result / v_sq;
                } else {
                    u.setZero();
                    break;
                }
            } else {
                // Standard u-update
                spmv_f(v, spmv_result);
                deflation_correct(spmv_result, v, U_all, d_all, V_all, k);
                Scalar v_sq = v.squaredNorm();
                if (do_cv) v_sq *= cv_denom_correction;
                if (v_sq > 0) {
                    u = spmv_result / v_sq;
                } else {
                    u.setZero();
                    break;
                }
            }

            {
                Scalar v_sq = v.squaredNorm();
                if (do_cv) v_sq *= cv_denom_correction;
                apply_regularization(u, config.L1_u, config.L2_u,
                                     config.nonneg_u, config.upper_bound_u, v_sq,
                                     config.L21_u);
                apply_angular_penalty(u, U_all, k, config.angular_u, v_sq);
                apply_graph_penalty(u, config.graph_u, config.graph_u_lambda, v_sq);
            }

            sigma = u.norm();
            if (sigma > 0) {
                u /= sigma;
            } else {
                break;
            }

            // Convergence check — supports FACTOR, LOSS, BOTH modes
            Scalar cos_sim = std::abs(u.dot(u_old));
            Scalar cos_dist = static_cast<Scalar>(1) - cos_sim;

            bool factor_converged = (cos_dist < tol_k);
            bool loss_converged = false;
            if (config.convergence != SVDConvergence::FACTOR && iter > 0) {
                Scalar d_sigma = std::abs(sigma - prev_sigma) /
                    (prev_sigma + std::numeric_limits<Scalar>::epsilon());
                loss_converged = (d_sigma < tol_k);
            }
            prev_sigma = sigma;

            bool converged = false;
            switch (config.convergence) {
                case SVDConvergence::FACTOR: converged = factor_converged; break;
                case SVDConvergence::LOSS:   converged = loss_converged;   break;
                case SVDConvergence::BOTH:   converged = factor_converged || loss_converged; break;
            }

            if (converged) {
                ++iter;
                break;
            }
        }

        // ============================================================
        // Post-factor Gram-Schmidt reorthogonalization
        // Two-pass GS ensures orthogonality drift stays below eps
        // at high k. Cost: O(k*(m+n)) — negligible vs O(nnz) SpMV.
        // ============================================================
        if (k > 0) {
            auto U_prev = U_all.leftCols(k);
            auto V_prev = V_all.leftCols(k);

            // Reorthogonalize u against U_all[:, 0:k-1] (two passes)
            for (int pass = 0; pass < 2; ++pass) {
                DenseVector<Scalar> dots_u = U_prev.transpose() * u;
                u.noalias() -= U_prev * dots_u;
            }
            Scalar u_norm_gs = u.norm();
            if (u_norm_gs > std::numeric_limits<Scalar>::epsilon() * 100) {
                u /= u_norm_gs;
            }

            // Reorthogonalize v against V_all[:, 0:k-1] (two passes)
            for (int pass = 0; pass < 2; ++pass) {
                DenseVector<Scalar> dots_v = V_prev.transpose() * v;
                v.noalias() -= V_prev * dots_v;
            }
            Scalar v_norm_gs = v.norm();
            if (v_norm_gs > std::numeric_limits<Scalar>::epsilon() * 100) {
                v /= v_norm_gs;
            }
        }

        // Recompute singular value after GS reorthogonalization.
        // The ALS-loop sigma was computed before GS modified u and v,
        // so it's stale.  Rayleigh quotient: sigma = u' * (A*v corrected).
        {
            spmv_f(v, spmv_result);
            deflation_correct(spmv_result, v, U_all, d_all, V_all, k);
            sigma = u.dot(spmv_result);
            if (sigma < 0) sigma = -sigma;  // singular values are non-negative
        }

        // Store factor
        U_all.col(k) = u;
        d_all(k) = sigma;
        V_all.col(k) = v;
        k_computed = k + 1;

        result.iters_per_factor.push_back(iter);

        // === Test loss evaluation ===
        if (do_cv) {
            test_entries.update(u, sigma, v, config.threads);
            Scalar test_mse = test_entries.compute_mse(config.threads);
            result.test_loss_trajectory.push_back(test_mse);

            if (config.verbose) {
                SVD_PRINT("  Factor %d: sigma=%.4e  iters=%d  test_mse=%.6e\n",
                          k + 1, static_cast<double>(sigma), iter,
                          static_cast<double>(test_mse));
            }

            // Auto-rank patience
            if (test_mse < best_test_loss) {
                best_test_loss = test_mse;
                best_k = k + 1;  // 1-indexed rank
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    if (config.verbose) {
                        SVD_PRINT("  Auto-rank: patience exhausted at factor %d, "
                                  "best rank = %d (test_mse=%.6e)\n",
                                  k + 1, best_k, static_cast<double>(best_test_loss));
                    }
                    break;
                }
            }
        } else {
            if (config.verbose) {
                SVD_PRINT("  Factor %d: sigma=%.4e  iters=%d\n",
                          k + 1, static_cast<double>(sigma), iter);
            }
            best_k = k + 1;
        }

        // If sigma is negligible, no more signal to extract
        if (sigma < std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(100)) {
            if (config.verbose) {
                SVD_PRINT("  Factor %d: sigma ~ 0, stopping early\n", k + 1);
            }
            break;
        }
    }

    // Trim to best_k (auto-rank may have computed extra factors)
    result.k_selected = best_k;
    result.U = U_all.leftCols(best_k);
    result.d = d_all.head(best_k);
    result.V = V_all.leftCols(best_k);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

}  // namespace svd
}  // namespace FactorNet

