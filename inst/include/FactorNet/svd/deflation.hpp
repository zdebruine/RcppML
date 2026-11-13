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
#include <string>
#include <stdexcept>
#include <cstdio>

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
 * Missing data. Entries in config.obs_mask and the CV holdout are treated as
 * missing: each ALS sweep replaces them by the current model prediction
 * (see MaskedEntries), so the fixed point is the least-squares fit over the
 * observed entries and singular values are not shrunk. Row means and SDs are
 * taken over observed entries.
 *
 * Constrained factors (nonneg, L1) use projection deflation and are not
 * orthogonal. For them `d` is refit by least squares,
 *   d = argmin || X_obs - U diag(d) V' ||_F,
 * so that U diag(d) V' is the model that reconstruct() returns, and
 * train_loss_trajectory[k] holds the observed mean squared error of the
 * nested rank-(k+1) refit. The per-component drop in that error is the
 * variance explained; it reduces to d^2 for orthogonal factors.
 *
 * L2 (ridge on both factors of a rank-one term a b') is equivalent to
 * soft-thresholding the singular value by sqrt(L2_u * L2_v); a penalty on one
 * side only has no effect on a bilinear model.
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
    const int k_max = std::max(1, std::min(config.k_max, std::min(m, n)));

    if (config.L21_u > 0 || config.L21_v > 0) {
        throw std::invalid_argument(
            "svd(method = \"deflation\"): L21 (group sparsity across factors) is not "
            "defined when factors are fit one at a time; use method = \"krylov\"");
    }
    if (config.upper_bound_u > 0 || config.upper_bound_v > 0) {
        throw std::invalid_argument(
            "svd: 'upper_bound' is not supported: factors are returned with unit "
            "norm, so an element-wise bound has no scale-free meaning");
    }

    SVDResult<Scalar> result;
    result.centered = config.center;

    const bool do_cv = config.is_cv();

    // ------------------------------------------------------------------
    // Missing entries: user mask and CV holdout
    // ------------------------------------------------------------------
    MaskedEntries<Scalar> masked;
    if (config.obs_mask != nullptr || do_cv) {
        masked = build_masked_entries<MatrixType, Scalar>(A, config);
    }
    const bool use_mask = masked.count > 0;

    // Row means / sample SDs over observed entries, in double
    DenseVector<Scalar> row_means, row_sds, row_inv_sds;
    double total_ss = 0.0, n_observed = 0.0;
    {
        int stat_threads = config.threads;
#ifdef _OPENMP
        if (stat_threads <= 0) stat_threads = omp_get_max_threads();
#else
        stat_threads = 1;
#endif
        masked_row_stats<MatrixType, Scalar>(A, masked, config.center, config.scale,
                                             row_means, row_sds, total_ss, n_observed,
                                             stat_threads);
    }

    const DenseVector<Scalar>* row_means_ptr = nullptr;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    if (config.center) {
        row_means_ptr = &row_means;
        result.row_means = row_means;
    }
    if (config.scale) {
        row_inv_sds = row_sds.cwiseInverse();
        sd_ptr = &row_inv_sds;
        result.row_sds = row_sds;
        result.scaled = true;
    }

    // ||X||^2_F of the transformed matrix; with missing entries, the observed
    // sum of squares scaled up to the full matrix size.
    {
        const double n_cells = static_cast<double>(m) * static_cast<double>(n);
        result.frobenius_norm_sq =
            (n_observed > 0.0) ? total_ss * (n_cells / n_observed) : 0.0;
    }

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

    if (config.verbose && use_mask) {
        SVD_PRINT("  %d entries treated as missing (%d held out for CV)\n",
                  masked.count, masked.n_test);
    }

    TestEntries<Scalar> test_entries;
    if (do_cv) test_entries = masked.test_entries();
    Scalar best_test_loss = std::numeric_limits<Scalar>::max();
    int best_k = 0;
    int patience_counter = 0;

    const bool do_robust = config.is_robust();

    if (config.verbose && do_robust) {
        SVD_PRINT("  Robust SVD: Huber delta=%.4f, IRLS reweighting active\n",
                  static_cast<double>(config.robust_delta));
    }

    // Model value of the factors fitted so far at each missing entry
    std::vector<Scalar> prior(static_cast<size_t>(masked.count), static_cast<Scalar>(0));
    auto set_fill_prior = [&]() {
        for (int e = 0; e < masked.count; ++e) masked.fill[e] = prior[e];
    };
    auto set_fill_current = [&](Scalar sig, const DenseVector<Scalar>& uu,
                                const DenseVector<Scalar>& vv) {
        for (int e = 0; e < masked.count; ++e) {
            masked.fill[e] = prior[e] + sig * uu(masked.rows[e]) * vv(masked.cols[e]);
        }
    };

    // Products with the transformed matrix, missing entries imputed
    auto spmv_t = [&](const DenseVector<Scalar>& u_vec, DenseVector<Scalar>& out) {
        spmv_transpose(A, u_vec, out, row_means_ptr, eff_threads, sd_ptr);
        if (use_mask) masked.add_transpose(u_vec, out);
    };

    auto spmv_f = [&](const DenseVector<Scalar>& v_vec, DenseVector<Scalar>& out) {
        spmv_forward(A, v_vec, out, row_means_ptr, eff_threads, &ws, sd_ptr);
        if (use_mask) masked.add_forward(v_vec, out);
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

    // Element-wise constraints on a side make that side's factors
    // incompatible with Gram-Schmidt (see post-factor reorthogonalization).
    const bool u_constrained = config.nonneg_u || config.L1_u > 0;
    const bool v_constrained = config.nonneg_v || config.L1_v > 0;

    // Two-sided ridge on a rank-one term = soft threshold on its scale
    const Scalar l2_tau = (config.L2_u > 0 && config.L2_v > 0)
        ? std::sqrt(config.L2_u * config.L2_v) : static_cast<Scalar>(0);

    // ------------------------------------------------------------------
    // Projection deflation for constrained factors.
    //
    // Hotelling deflation (R_k = A - sum_r sigma_r u_r v_r') only removes a
    // direction exactly when u_r, v_r are singular vectors. Constrained factors
    // are not, so the residual keeps energy along u_r and later factors can
    // re-find (near-)copies of earlier ones. When either side is constrained we
    // instead use generalized (projection) deflation, as in nsprcomp:
    //   R_k = P_u A P_v,   P_u = I - Qu Qu',  P_v = I - Qv Qv',
    // where Qu (Qv) is an orthonormal basis for the earlier constrained u (v)
    // factors. Each new factor can then only explain variance not already
    // captured. Unconstrained SVD/PCA keeps the Hotelling path unchanged.
    // ------------------------------------------------------------------
    const bool use_projection = u_constrained || v_constrained;
    DenseMatrix<Scalar> Qu, Qv;
    if (u_constrained) { Qu.resize(m, k_max); Qu.setZero(); }
    if (v_constrained) { Qv.resize(n, k_max); Qv.setZero(); }
    int n_proj = 0;
    auto project_out = [&](DenseVector<Scalar>& x, const DenseMatrix<Scalar>& Q) {
        if (n_proj == 0) return;
        auto Qk = Q.leftCols(n_proj);
        for (int pass = 0; pass < 2; ++pass) {
            DenseVector<Scalar> c = Qk.transpose() * x;
            x.noalias() -= Qk * c;
        }
    };
    // Deflated products: out = R_k v  and  out = R_k' u
    auto apply_R = [&](const DenseVector<Scalar>& v_in, DenseVector<Scalar>& out, int k_cur) {
        if (!use_projection) {
            spmv_f(v_in, out);
            deflation_correct(out, v_in, U_all, d_all, V_all, k_cur);
            return;
        }
        if (v_constrained && n_proj > 0) {
            DenseVector<Scalar> vp = v_in; project_out(vp, Qv); spmv_f(vp, out);
        } else {
            spmv_f(v_in, out);
        }
        if (u_constrained) project_out(out, Qu);
    };
    auto apply_Rt = [&](const DenseVector<Scalar>& u_in, DenseVector<Scalar>& out, int k_cur) {
        if (!use_projection) {
            spmv_t(u_in, out);
            deflation_correct(out, u_in, V_all, d_all, U_all, k_cur);
            return;
        }
        if (u_constrained && n_proj > 0) {
            DenseVector<Scalar> up = u_in; project_out(up, Qu); spmv_t(up, out);
        } else {
            spmv_t(u_in, out);
        }
        if (v_constrained) project_out(out, Qv);
    };

    // Least-squares refit of d for non-orthogonal (constrained) factors:
    //   (U'U o V'V - masked terms) d = diag(U' X_obs V)
    Eigen::MatrixXd Gls = Eigen::MatrixXd::Zero(k_max, k_max);
    Eigen::VectorXd cls = Eigen::VectorXd::Zero(k_max);
    std::vector<Eigen::VectorXd> dls_hist;

    int n_unconverged = 0;
    int k_done = 0;

    for (int k = 0; k < k_max; ++k) {
        SVD_CHECK_INTERRUPT();

        // Missing entries start each factor at the model of the previous ones
        if (use_mask) set_fill_prior();

        // ============================================================
        // Warm start + Nesterov acceleration for all factors
        // Power-step warm start gives good initial direction
        // Nesterov momentum halves iteration count for clustered SVs
        // ============================================================

        // Constrained u: the orthogonalized power-step warm start can point
        // into the negative orthant and collapse to zero after projection,
        // so use a fresh non-negative random start instead.
        if (k == 0 || u_constrained) {
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
                apply_Rt(u, spmv_result, k);
                v = spmv_result;
                Scalar v_norm_tmp = v.norm();
                if (v_norm_tmp > 0) v /= v_norm_tmp;
                apply_R(v, spmv_result, k);
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
        if (u_norm > 0) {
            u /= u_norm;
        } else {
            for (int i = 0; i < m; ++i) {
                u(i) = static_cast<Scalar>(rng.uniform<double>());
            }
            u /= u.norm();
        }

        // ============================================================
        // Adaptive tolerance: tol * sigma_1/sigma_{k-1}, capped at 100×
        // ============================================================
        // (a tolerance below what 1 - |cos| can resolve in this precision is
        // raised to that resolution)
        const Scalar tol_base = std::max(config.tol,
            static_cast<Scalar>(4) * std::numeric_limits<Scalar>::epsilon());
        Scalar tol_k = tol_base;
        if (k > 0 && d_all(0) > 0 && d_all(k - 1) > 0) {
            tol_k = tol_base * d_all(0) / d_all(k - 1);
            tol_k = std::min(tol_k, tol_base * static_cast<Scalar>(100));
        }

        Scalar sigma = 0;
        Scalar prev_sigma = 0;  // for loss-based convergence
        int iter = 0;
        bool conv_k = false;
        u_prev = u;
        v.setZero();

        // The L1 threshold is absolute on the unnormalized update, whose size
        // depends on how well the current iterate is aligned with the factor.
        // From a random start it can be far below the threshold and the whole
        // vector would be zeroed, so the first sweeps run without L1.
        const int l1_warm = (config.L1_u > 0 || config.L1_v > 0) ? 2 : 0;

        for (iter = 0; iter < config.max_iter; ++iter) {
            u_old = u;

            // Nesterov momentum extrapolation (disabled under IRLS and under
            // imputation of missing entries, where the operator changes each sweep)
            Scalar beta = (!do_robust && !use_mask && iter > 1)
                          ? static_cast<Scalar>(iter - 1) / static_cast<Scalar>(iter + 2)
                          : static_cast<Scalar>(0);
            DenseVector<Scalar> u_hat = u + beta * (u - u_prev);
            u_prev = u;

            if (do_robust && iter > 0) {
                // ============================================================
                // IRLS: compute row weights from current u, v, sigma
                // residual_i = (Av)_i - sigma * u_i
                // ============================================================
                apply_R(v, spmv_result_unweighted, k);
                compute_row_weights(spmv_result_unweighted, u, sigma,
                                    config.robust_delta, row_weights, eff_threads);

                // Column weights from A'u residuals
                apply_Rt(u, spmv_result_unweighted, k);
                compute_col_weights(spmv_result_unweighted, v, sigma,
                                    config.robust_delta, col_weights, eff_threads);
            }

            // === V-update ===
            if (do_robust && iter > 0) {
                // Weighted v-update: v = A' diag(w_u) u_hat / (u_hat' diag(w_u) u_hat)
                DenseVector<Scalar> wu = u_hat.cwiseProduct(row_weights);
                apply_Rt(wu, spmv_result, k);
                Scalar u_sq = wu.dot(u_hat);  // u_hat' diag(w) u_hat
                if (u_sq > 0) {
                    v = spmv_result / u_sq;
                } else {
                    v.setZero();
                    conv_k = true;
                    break;
                }
            } else {
                // Standard v-update
                apply_Rt(u_hat, spmv_result, k);
                Scalar u_sq = u_hat.squaredNorm();
                if (u_sq > 0) {
                    v = spmv_result / u_sq;
                } else {
                    v.setZero();
                    conv_k = true;
                    break;
                }
            }

            {
                // L2/L21/upper_bound are handled outside this projection
                // (see l2_tau and the argument checks at the top).
                Scalar u_sq = u_hat.squaredNorm();
                apply_regularization(v,
                                     (iter >= l1_warm) ? config.L1_v : static_cast<Scalar>(0),
                                     static_cast<Scalar>(0),
                                     config.nonneg_v, static_cast<Scalar>(0), u_sq,
                                     static_cast<Scalar>(0));
                apply_angular_penalty(v, V_all, k, config.angular_v, u_sq);
                apply_graph_penalty(v, config.graph_v, config.graph_v_lambda, u_sq);
            }

            sigma = v.norm();
            if (sigma > 0) {
                v /= sigma;
            } else {
                conv_k = true;
                break;
            }

            // === U-update ===
            if (do_robust && iter > 0) {
                // Weighted u-update: u = A diag(w_v) v / (v' diag(w_v) v)
                DenseVector<Scalar> wv = v.cwiseProduct(col_weights);
                apply_R(wv, spmv_result, k);
                Scalar v_sq = wv.dot(v);  // v' diag(w) v
                if (v_sq > 0) {
                    u = spmv_result / v_sq;
                } else {
                    u.setZero();
                    conv_k = true;
                    break;
                }
            } else {
                // Standard u-update
                apply_R(v, spmv_result, k);
                Scalar v_sq = v.squaredNorm();
                if (v_sq > 0) {
                    u = spmv_result / v_sq;
                } else {
                    u.setZero();
                    conv_k = true;
                    break;
                }
            }

            {
                Scalar v_sq = v.squaredNorm();
                apply_regularization(u,
                                     (iter >= l1_warm) ? config.L1_u : static_cast<Scalar>(0),
                                     static_cast<Scalar>(0),
                                     config.nonneg_u, static_cast<Scalar>(0), v_sq,
                                     static_cast<Scalar>(0));
                apply_angular_penalty(u, U_all, k, config.angular_u, v_sq);
                apply_graph_penalty(u, config.graph_u, config.graph_u_lambda, v_sq);
            }

            sigma = u.norm();
            if (sigma > 0) {
                u /= sigma;
            } else {
                conv_k = true;
                break;
            }

            // E-step: impute missing entries with the current rank-one estimate
            if (use_mask) set_fill_current(sigma, u, v);

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

            if (converged && iter >= l1_warm) {
                conv_k = true;
                ++iter;
                break;
            }
        }

        // (b) The constraints left nothing for this factor (for example an L1
        // penalty above every entry): stop and return the factors found so far
        // instead of storing a zero vector with a scale.
        if (!(u.norm() > 0) || !(v.norm() > 0)) {
            char zbuf[320];
            std::snprintf(zbuf, sizeof(zbuf),
                "svd(method = \"deflation\"): the constraints (L1/nonneg) removed all "
                "entries of factor %d; returning %d factor(s). Reduce L1 or k.",
                k + 1, k);
            add_warning(std::string(zbuf));
            break;
        }
        if (!conv_k) ++n_unconverged;

        // ============================================================
        // Post-factor Gram-Schmidt reorthogonalization
        // Two-pass GS ensures orthogonality drift stays below eps
        // at high k. Cost: O(k*(m+n)) — negligible vs O(nnz) SpMV.
        // ============================================================
        // Gram-Schmidt is skipped on any side with projection constraints
        // (nonneg, L1): subtracting projections onto prior factors
        // re-introduces negative entries and destroys sparsity, so
        // constrained factors are returned as fitted (non-orthogonal), as in
        // other constrained/sparse PCA methods.
        if (k > 0 && !use_projection) {
            auto U_prev = U_all.leftCols(k);
            auto V_prev = V_all.leftCols(k);

            for (int pass = 0; pass < 2; ++pass) {
                DenseVector<Scalar> dots_u = U_prev.transpose() * u;
                u.noalias() -= U_prev * dots_u;
            }
            Scalar u_norm_gs = u.norm();
            if (u_norm_gs > std::numeric_limits<Scalar>::epsilon() * 100) {
                u /= u_norm_gs;
            }

            for (int pass = 0; pass < 2; ++pass) {
                DenseVector<Scalar> dots_v = V_prev.transpose() * v;
                v.noalias() -= V_prev * dots_v;
            }
            Scalar v_norm_gs = v.norm();
            if (v_norm_gs > std::numeric_limits<Scalar>::epsilon() * 100) {
                v /= v_norm_gs;
            }
        }

        // Recompute the scale after reorthogonalization. With the missing
        // entries set to the model of the previous factors, R_k is zero there
        // (Hotelling path), so u' R_k v sums over observed entries only and
        // dividing by sum_obs u_i^2 v_j^2 gives the least-squares scale.
        {
            if (use_mask) set_fill_prior();
            apply_R(v, spmv_result, k);
            sigma = u.dot(spmv_result);
            if (use_mask && !use_projection) {
                Scalar msum = 0;
                for (int e = 0; e < masked.count; ++e) {
                    const Scalar t = u(masked.rows[e]) * v(masked.cols[e]);
                    msum += t * t;
                }
                const Scalar denom = u.squaredNorm() * v.squaredNorm() - msum;
                if (denom > std::numeric_limits<Scalar>::epsilon() * 100) sigma /= denom;
            }
            if (sigma < 0) sigma = -sigma;  // singular values are non-negative
        }

        // Store factor
        U_all.col(k) = u;
        d_all(k) = sigma;
        V_all.col(k) = v;
        k_done = k + 1;

        if (use_projection) {
            const Scalar eps_q = std::numeric_limits<Scalar>::epsilon() * 100;
            if (u_constrained) {
                DenseVector<Scalar> q = u; project_out(q, Qu);
                Scalar qn = q.norm();
                if (qn > eps_q) Qu.col(n_proj) = q / qn; else Qu.col(n_proj).setZero();
            }
            if (v_constrained) {
                DenseVector<Scalar> q = v; project_out(q, Qv);
                Scalar qn = q.norm();
                if (qn > eps_q) Qv.col(n_proj) = q / qn; else Qv.col(n_proj).setZero();
            }
            ++n_proj;
        }

        result.iters_per_factor.push_back(iter);

        // ============================================================
        // Model bookkeeping: values at missing entries and test error
        // ============================================================
        Scalar test_mse = 0;
        if (use_projection) {
            // c_k = u_k' X_obs v_k  (missing entries contribute zero)
            if (use_mask) masked.set_fill_zero();
            spmv_f(v, spmv_result);
            cls(k) = static_cast<double>(u.dot(spmv_result));

            // G(k, r) = sum over observed (i,j) of u_ik v_jk u_ir v_jr
            Eigen::VectorXd grow(k + 1);
            for (int r = 0; r <= k; ++r) {
                grow(r) = static_cast<double>(U_all.col(r).dot(u)) *
                          static_cast<double>(V_all.col(r).dot(v));
            }
            for (int e = 0; e < masked.count; ++e) {
                const int i = masked.rows[e], j = masked.cols[e];
                const double t = static_cast<double>(u(i)) * static_cast<double>(v(j));
                if (t == 0.0) continue;
                for (int r = 0; r <= k; ++r) {
                    grow(r) -= t * static_cast<double>(U_all(i, r)) *
                                   static_cast<double>(V_all(j, r));
                }
            }
            for (int r = 0; r <= k; ++r) {
                Gls(k, r) = grow(r);
                Gls(r, k) = grow(r);
            }

            Eigen::MatrixXd Gk = Gls.topLeftCorner(k + 1, k + 1);
            Eigen::VectorXd ck = cls.head(k + 1);
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(Gk);
            Eigen::VectorXd dk = cod.solve(ck);
            dls_hist.push_back(dk);

            double rss = total_ss - dk.dot(ck);
            if (rss < 0.0) rss = 0.0;
            result.train_loss_trajectory.push_back(
                static_cast<Scalar>(n_observed > 0.0 ? rss / n_observed : 0.0));

            // Model of the first k+1 factors at the missing entries
            double test_ss = 0.0;
            int test_cnt = 0;
            for (int e = 0; e < masked.count; ++e) {
                const int i = masked.rows[e], j = masked.cols[e];
                double pred = 0.0;
                for (int r = 0; r <= k; ++r) {
                    pred += dk(r) * static_cast<double>(U_all(i, r)) *
                                    static_cast<double>(V_all(j, r));
                }
                prior[static_cast<size_t>(e)] = static_cast<Scalar>(pred);
                if (masked.is_test[e]) {
                    const double diff = static_cast<double>(masked.base[e]) - pred;
                    test_ss += diff * diff;
                    ++test_cnt;
                }
            }
            if (test_cnt > 0) test_mse = static_cast<Scalar>(test_ss / test_cnt);
        } else {
            for (int e = 0; e < masked.count; ++e) {
                prior[static_cast<size_t>(e)] +=
                    sigma * u(masked.rows[e]) * v(masked.cols[e]);
            }

            // With missing entries, rank-one fits made one after another are
            // not the joint least-squares fit: each factor was estimated while
            // the later ones were still unknown, which leaves structured
            // residual (and makes rank selection overshoot). Refine all factors
            // found so far by block coordinate sweeps on the matrix whose
            // missing entries hold the current model; the fixed point is the
            // rank-(k+1) SVD of that completed matrix.
            if (use_mask && k > 0) {
                const Scalar eps_s = std::numeric_limits<Scalar>::epsilon() * 100;
                DenseVector<Scalar> ur(m), vr(n), u_new, v_new;
                for (int sweep = 0; sweep < 2; ++sweep) {
                    for (int r = 0; r <= k; ++r) {
                        ur = U_all.col(r);
                        vr = V_all.col(r);
                        const Scalar d_old = d_all(r);
                        set_fill_prior();

                        // u <- (X - sum_{s != r} d_s u_s v_s') v_r
                        spmv_f(vr, u_new);
                        for (int s2 = 0; s2 <= k; ++s2) {
                            if (s2 == r) continue;
                            const Scalar c = d_all(s2) * V_all.col(s2).dot(vr);
                            u_new.noalias() -= c * U_all.col(s2);
                        }
                        for (int pass = 0; pass < 2; ++pass) {
                            for (int s2 = 0; s2 <= k; ++s2) {
                                if (s2 == r) continue;
                                const Scalar c = U_all.col(s2).dot(u_new);
                                u_new.noalias() -= c * U_all.col(s2);
                            }
                        }
                        const Scalar un = u_new.norm();
                        if (!(un > 0)) continue;
                        u_new /= un;

                        // v <- (X - sum_{s != r} d_s u_s v_s')' u
                        spmv_t(u_new, v_new);
                        for (int s2 = 0; s2 <= k; ++s2) {
                            if (s2 == r) continue;
                            const Scalar c = d_all(s2) * U_all.col(s2).dot(u_new);
                            v_new.noalias() -= c * V_all.col(s2);
                        }
                        for (int pass = 0; pass < 2; ++pass) {
                            for (int s2 = 0; s2 <= k; ++s2) {
                                if (s2 == r) continue;
                                const Scalar c = V_all.col(s2).dot(v_new);
                                v_new.noalias() -= c * V_all.col(s2);
                            }
                        }
                        const Scalar vn = v_new.norm();
                        if (!(vn > 0)) continue;
                        v_new /= vn;

                        // Least-squares scale over observed entries: the missing
                        // entries of the residual hold the old term d_old u_r v_r'.
                        Scalar cross = 0, msum = 0;
                        for (int e = 0; e < masked.count; ++e) {
                            const int i = masked.rows[e], j = masked.cols[e];
                            const Scalar t = u_new(i) * v_new(j);
                            cross += ur(i) * vr(j) * t;
                            msum += t * t;
                        }
                        Scalar s_new = vn - d_old * cross;
                        const Scalar denom = static_cast<Scalar>(1) - msum;
                        if (denom > eps_s) s_new /= denom;
                        if (s_new < 0) s_new = -s_new;

                        for (int e = 0; e < masked.count; ++e) {
                            const int i = masked.rows[e], j = masked.cols[e];
                            prior[static_cast<size_t>(e)] +=
                                s_new * u_new(i) * v_new(j) - d_old * ur(i) * vr(j);
                        }
                        U_all.col(r) = u_new;
                        V_all.col(r) = v_new;
                        d_all(r) = s_new;
                    }
                }
                u = U_all.col(k);
                v = V_all.col(k);
                sigma = d_all(k);
            }

            if (do_cv) {
                if (use_mask) {
                    // held-out error of the current model
                    double test_ss = 0.0;
                    int test_cnt = 0;
                    for (int e = 0; e < masked.count; ++e) {
                        if (masked.is_test[e]) {
                            const double diff = static_cast<double>(masked.base[e]) -
                                static_cast<double>(prior[static_cast<size_t>(e)]);
                            test_ss += diff * diff;
                            ++test_cnt;
                        }
                    }
                    if (test_cnt > 0) test_mse = static_cast<Scalar>(test_ss / test_cnt);
                } else {
                    const Scalar sig_model = (l2_tau > 0)
                        ? std::max(static_cast<Scalar>(0), sigma - l2_tau) : sigma;
                    test_entries.update(u, sig_model, v, config.threads);
                    test_mse = test_entries.compute_mse(config.threads);
                }
            }
        }

        // === Test loss evaluation ===
        if (do_cv) {
            result.test_loss_trajectory.push_back(test_mse);

            if (config.verbose) {
                SVD_PRINT("  Factor %d: sigma=%.4e  iters=%d  test_mse=%.6e\n",
                          k + 1, static_cast<double>(sigma), iter,
                          static_cast<double>(test_mse));
            }

            // Auto-rank patience. A new best must lower the held-out error by
            // at least 0.1%, so a factor that fits noise cannot win by a
            // chance fluctuation.
            if (test_mse < best_test_loss * static_cast<Scalar>(0.999)) {
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

        // If sigma is negligible relative to the first factor, no signal is left
        {
            const Scalar sig_ref = (k > 0) ? d_all(0) : sigma;
            const Scalar rel = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(100);
            if (!(sigma > 0) || (k > 0 && sigma <= rel * sig_ref)) {
                if (config.verbose) {
                    SVD_PRINT("  Factor %d: sigma ~ 0, stopping early\n", k + 1);
                }
                break;
            }
        }
    }

    // Trim to best_k (auto-rank may have computed extra factors)
    if (best_k < 1) best_k = std::min(1, k_done);
    result.k_selected = best_k;
    result.U = U_all.leftCols(best_k);
    result.V = V_all.leftCols(best_k);
    if (use_projection && best_k > 0 && static_cast<int>(dls_hist.size()) >= best_k) {
        result.d = dls_hist[static_cast<size_t>(best_k - 1)].cast<Scalar>();
        result.train_loss_trajectory.resize(static_cast<size_t>(best_k));
    } else {
        result.d = d_all.head(best_k);
    }
    if (l2_tau > 0) {
        for (int i = 0; i < best_k; ++i) {
            result.d(i) = std::max(static_cast<Scalar>(0), result.d(i) - l2_tau);
        }
    }

    if (n_unconverged > 0) {
        char wbuf[320];
        std::snprintf(wbuf, sizeof(wbuf),
            "svd(method = \"deflation\"): %d of %d factors did not reach tol = %.1e "
            "within maxit = %d iterations; increase 'maxit' or loosen 'tol'",
            n_unconverged, k_done, static_cast<double>(config.tol), config.max_iter);
        add_warning(std::string(wbuf));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

}  // namespace svd
}  // namespace FactorNet
