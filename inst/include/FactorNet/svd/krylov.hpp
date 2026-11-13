// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file krylov.hpp
 * @brief Krylov-Seeded Projected Refinement (KSPR) for constrained truncated SVD
 *
 * Two-phase algorithm combining fast subspace identification with cheap
 * projected constraint enforcement:
 *
 * Phase 1: Unconstrained Lanczos seed to identify the optimal k-dimensional
 *          subspace. Cost: O((k+p) × nnz) — ~670ms at k=50 on pbmc3k.
 *
 * Phase 2: Projected refinement using block SpMM + Cholesky solve + element-wise
 *          constraint projection. Each pass costs O(2 × nnz × k) for two SpMMs
 *          plus O(k³ + k²(m+n)) for Gram solve and projection.
 *
 *          Key design: W and V are always unit-norm; d tracks singular values.
 *          SpMM and Gram use V/W directly (unit-norm). After Cholesky solve,
 *          values are at the correct scale for constraint projection.
 *
 *          Constraint application matches deflation's math:
 *          - L2: embedded in Gram diagonal (G + λI) — single application
 *          - L1: soft_threshold(x, L1/(2·norm_sq)) on post-solve values
 *          - Nonneg: x ← max(0, x) on post-solve values
 *          After projection, columns are normalized (scale absorbed into d).
 *
 * This approach mirrors deflation's per-element constraint strategy but
 * operates in batch (all k components simultaneously), avoiding deflation
 * correction overhead.
 *
 * Falls back to pure Lanczos when no constraints are active.
 *
 * Supports: PCA centering, L1/L2 regularization, non-negativity, bounds.
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/svd/spmv.hpp>      // SpMM, compute_row_means, is_sparse_matrix
#include <FactorNet/svd/lanczos.hpp>
#include <FactorNet/svd/test_entries.hpp> // TestEntries for CV auto-rank
#include <FactorNet/nmf/speckled_cv.hpp>           // LazySpeckledMask for CV
#include <FactorNet/rng/rng.hpp>
#include <FactorNet/features/L21.hpp>              // apply_L21 (group sparsity)
#include <FactorNet/features/angular.hpp>            // apply_ortho (angular/decorrelation)
#include <FactorNet/features/graph_reg.hpp>        // apply_graph_reg (graph Laplacian)

#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Print/interrupt macros
#if defined(RCPP_VERSION) || __has_include(<Rcpp.h>)
#include <Rcpp.h>
#define KSVD_PRINT(...) { char buf[512]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); Rcpp::Rcout << buf; }
#define KSVD_CHECK_INTERRUPT() Rcpp::checkUserInterrupt()
#else
#include <cstdio>
#define KSVD_PRINT(...) std::printf(__VA_ARGS__)
#define KSVD_CHECK_INTERRUPT() ((void)0)
#endif

namespace FactorNet {
namespace svd {

namespace detail {

// ============================================================================
// Element-wise constraint projection (matches deflation's apply_regularization)
// ============================================================================

/**
 * @brief Apply L1/L2/nonneg/bounds column-wise to post-Cholesky-solve values
 *
 * After Cholesky solve W = B·G⁻¹, the values are already at the correct scale
 * (G⁻¹ accounts for squared norms). We only need norm_sq for L1/L2 threshold
 * scaling, NOT for dividing the values.
 *
 * Matches deflation's math:
 *   deflation: v = dot/(norm_sq), then L1_thr = L1/(2*norm_sq), L2_scale = 1/(1+L2/norm_sq)
 *   krylov:    W = B·G⁻¹ (already divided), then same L1_thr and L2_scale
 *
 * @param X      Matrix after Cholesky solve (values already at correct scale)
 * @param L1     L1 penalty strength  
 * @param L2     L2 penalty strength
 * @param nonneg Whether to enforce non-negativity
 * @param upper_bound Upper bound (0 = no bound)
 * @param norm_sq_vec Per-column squared norms of the fixed factor (for threshold scaling)
 */
template<typename Scalar>
void project_constraints(DenseMatrix<Scalar>& X,
                         Scalar L1, Scalar L2,
                         bool nonneg, Scalar upper_bound,
                         const DenseVector<Scalar>& norm_sq_vec)
{
    const int rows = static_cast<int>(X.rows());
    const int cols = static_cast<int>(X.cols());

    for (int c = 0; c < cols; ++c) {
        Scalar nsq = norm_sq_vec(c);
        if (nsq <= 0) { X.col(c).setZero(); continue; }

        // L2 shrinkage (same formula as deflation)
        if (L2 > 0) {
            Scalar scale = static_cast<Scalar>(1) / (static_cast<Scalar>(1) + L2 / nsq);
            X.col(c) *= scale;
        }

        // L1 soft-thresholding (same formula as deflation)
        if (L1 > 0) {
            Scalar threshold = L1 / (static_cast<Scalar>(2) * nsq);
            for (int i = 0; i < rows; ++i) {
                Scalar val = X(i, c);
                if (val > threshold) X(i, c) = val - threshold;
                else if (val < -threshold) X(i, c) = val + threshold;
                else X(i, c) = 0;
            }
        }

        // Non-negativity
        if (nonneg) {
            X.col(c) = X.col(c).cwiseMax(static_cast<Scalar>(0));
        }

        // Upper bound
        if (upper_bound > 0) {
            X.col(c) = X.col(c).cwiseMin(upper_bound);
        }
    }
}

/**
 * @brief Fused: apply L1/nonneg/bounds + normalize columns in one pass.
 *
 * Combines the constraint projection (project_constraints) and column
 * normalization into a single loop over columns, halving memory traffic
 * vs calling them separately. L2 must have been handled in the Gram solve
 * already (pass L2=0 here, same as the separate-call version).
 *
 * @param X          Matrix to project and normalize in-place (m × k)
 * @param L1         L1 penalty (soft-threshold scale)
 * @param L2         L2 penalty (should be 0; already in Gram)
 * @param nonneg     Enforce non-negativity
 * @param upper_bound Upper bound (0 = disabled)
 * @param norm_sq_vec Per-column squared norms of the OTHER factor (for threshold scaling)
 * @param d          Output: column norms before normalization (the singular value proxy)
 * @param eps100     Small epsilon × 100 — columns with norm ≤ this are zeroed
 */
template<typename Scalar>
void project_and_normalize(DenseMatrix<Scalar>& X,
                           Scalar L1, Scalar L2,
                           bool nonneg, Scalar upper_bound,
                           const DenseVector<Scalar>& norm_sq_vec,
                           DenseVector<Scalar>& d,
                           Scalar eps100)
{
    const int rows = static_cast<int>(X.rows());
    const int cols = static_cast<int>(X.cols());

    for (int c = 0; c < cols; ++c) {
        Scalar nsq = norm_sq_vec(c);
        if (nsq <= static_cast<Scalar>(0)) {
            X.col(c).setZero();
            d(c) = static_cast<Scalar>(0);
            continue;
        }

        // L2 shrinkage (same formula as deflation; usually L2=0 here since it's in Gram)
        if (L2 > static_cast<Scalar>(0)) {
            Scalar scale = static_cast<Scalar>(1) / (static_cast<Scalar>(1) + L2 / nsq);
            X.col(c) *= scale;
        }

        // L1 soft-thresholding
        if (L1 > static_cast<Scalar>(0)) {
            Scalar thr = L1 / (static_cast<Scalar>(2) * nsq);
            for (int i = 0; i < rows; ++i) {
                Scalar val = X(i, c);
                if      (val >  thr) X(i, c) = val - thr;
                else if (val < -thr) X(i, c) = val + thr;
                else                 X(i, c) = static_cast<Scalar>(0);
            }
        }

        // Non-negativity
        if (nonneg) X.col(c) = X.col(c).cwiseMax(static_cast<Scalar>(0));

        // Upper bound
        if (upper_bound > static_cast<Scalar>(0))
            X.col(c) = X.col(c).cwiseMin(upper_bound);

        // Extract d(c) and normalize — fused with project to save one column pass
        Scalar nrm = X.col(c).norm();
        d(c) = (nrm > eps100) ? nrm : static_cast<Scalar>(0);
        if (d(c) > static_cast<Scalar>(0)) X.col(c) /= d(c);
    }
}

// ============================================================================
// Extract proper U, Σ, V from factor matrices via QR + small SVD
// ============================================================================

template<typename Scalar>
void extract_svd_from_factors(
    const DenseMatrix<Scalar>& W,    // m × k
    const DenseMatrix<Scalar>& H,    // n × k
    DenseMatrix<Scalar>& U,          // m × k (out)
    DenseVector<Scalar>& sigma,      // k (out)
    DenseMatrix<Scalar>& V)          // n × k (out)
{
    const int k = static_cast<int>(W.cols());

    Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_w(W);
    Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_h(H);

    DenseMatrix<Scalar> Q_w = qr_w.householderQ() *
        DenseMatrix<Scalar>::Identity(W.rows(), k);
    DenseMatrix<Scalar> R_w = qr_w.matrixQR().topRows(k)
        .template triangularView<Eigen::Upper>();

    DenseMatrix<Scalar> Q_h = qr_h.householderQ() *
        DenseMatrix<Scalar>::Identity(H.rows(), k);
    DenseMatrix<Scalar> R_h = qr_h.matrixQR().topRows(k)
        .template triangularView<Eigen::Upper>();

    // Small SVD of R_w · R_h'  (k × k)
    DenseMatrix<Scalar> S = R_w * R_h.transpose();
    Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_small(
        S, Eigen::ComputeFullU | Eigen::ComputeFullV);

    sigma = svd_small.singularValues();
    U.noalias() = Q_w * svd_small.matrixU();
    V.noalias() = Q_h * svd_small.matrixV();
}

}  // namespace detail

// ============================================================================
// Main algorithm: krylov_constrained_svd
// ============================================================================

/**
 * @brief Krylov-seeded projected refinement (KSPR).
 *
 * Missing data (config.obs_mask and the CV holdout) is handled by iterative
 * imputation: before each block update the missing entries of the transformed
 * matrix are set to the current model W diag(d) V' (see MaskedEntries), so the
 * refinement converges to the least-squares fit over observed entries. The
 * Lanczos seed runs with missing entries at the row mean. Row means and SDs
 * are computed from observed entries only.
 *
 * Without constraints and without missing data this is plain Lanczos.
 */
template<typename MatrixType, typename Scalar>
SVDResult<Scalar> krylov_constrained_svd(const MatrixType& A,
                                         const SVDConfig<Scalar>& config)
{
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = std::min(config.k_max, std::min(m, n));

    if (config.upper_bound_u > 0 || config.upper_bound_v > 0) {
        throw std::invalid_argument(
            "svd: 'upper_bound' is not supported: factors are returned with unit "
            "norm, so an element-wise bound has no scale-free meaning");
    }

    // --- Detect whether constraints are active ---
    const bool has_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.L2_u > 0 || config.L2_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        config.angular_u > 0 || config.angular_v > 0 ||
        config.has_gram_level_reg();

    const bool do_cv = config.is_cv();
    const bool need_mask = do_cv || (config.obs_mask != nullptr);

    // No constraints and nothing missing: plain Lanczos is optimal.
    if (!has_constraints && !need_mask) {
        if (config.verbose) {
            KSVD_PRINT("  Krylov SVD: no constraints — direct Lanczos\n");
        }
        return lanczos_svd<MatrixType, Scalar>(A, config, true);
    }

    int eff_threads = config.threads;
#ifdef _OPENMP
    if (eff_threads <= 0) eff_threads = omp_get_max_threads();
#else
    eff_threads = 1;
#endif

    // ====================================================================
    // Missing entries and row statistics over observed entries
    // ====================================================================
    MaskedEntries<Scalar> masked;
    if (need_mask) {
        masked = build_masked_entries<MatrixType, Scalar>(A, config);
    }
    const bool use_mask = masked.count > 0;

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

    const DenseVector<Scalar>* rm_ptr = nullptr;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    if (config.center) rm_ptr = &row_means;
    if (config.scale) {
        row_inv_sds = row_sds.cwiseInverse();
        sd_ptr = &row_inv_sds;
    }

    if (config.verbose && use_mask) {
        KSVD_PRINT("  Krylov SVD: %d entries treated as missing (%d held out for CV)\n",
                   masked.count, masked.n_test);
    }

    // ====================================================================
    // PHASE 1: Unconstrained Lanczos seed (missing entries at the row mean)
    // ====================================================================

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD: Phase 1 — Lanczos seed (k=%d)\n", k);
    }

    auto t_phase1_start = Clock::now();

    DenseMatrix<Scalar> W;   // m × k (unit-norm columns)
    DenseMatrix<Scalar> V;   // n × k (unit-norm columns)
    DenseVector<Scalar> d;   // k
    {
        SpmvWorkspace<Scalar> ws;
        if constexpr (is_sparse_matrix<MatrixType>::value) {
            ws.init(m, eff_threads);
        }
        auto fwd = [&](const DenseVector<Scalar>& x, DenseVector<Scalar>& out) {
            spmv_forward(A, x, out, rm_ptr, eff_threads, &ws, sd_ptr);
            if (use_mask) masked.add_forward(x, out);
        };
        auto trn = [&](const DenseVector<Scalar>& x, DenseVector<Scalar>& out) {
            spmv_transpose(A, x, out, rm_ptr, eff_threads, sd_ptr);
            if (use_mask) masked.add_transpose(x, out);
        };
        int seed_steps = 0, seed_unconv = 0;
        detail::lanczos_bidiag_core<Scalar>(
            m, n, k, config.tol, 0, config.seed, false, false,
            fwd, trn, W, d, V, seed_steps, seed_unconv);
    }

    auto t_phase1_end = Clock::now();

    const int k_actual = static_cast<int>(d.size());

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD: Phase 1 done: %.1f ms, k_actual=%d\n",
                   Ms(t_phase1_end - t_phase1_start).count(), k_actual);
    }

    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = row_means;
    if (config.scale) {
        result.scaled = true;
        result.row_sds = row_sds;
    }
    {
        const double n_cells = static_cast<double>(m) * static_cast<double>(n);
        result.frobenius_norm_sq =
            (n_observed > 0.0) ? total_ss * (n_cells / n_observed) : 0.0;
    }

    if (k_actual == 0) {
        result.U.resize(m, 0);
        result.V.resize(n, 0);
        result.d.resize(0);
        result.k_selected = 0;
        result.wall_time_ms = Ms(Clock::now() - t_start).count();
        return result;
    }

    // ====================================================================
    // PHASE 2: Gram-solve-then-project refinement
    //
    // Key insight: The Gram matrix G=V'V (or W'W) provides component
    // decoupling — preventing all vectors from collapsing to the dominant
    // direction. After the Cholesky solve, element-wise constraint
    // projection (nonneg, L1) is applied. This "solve-then-project"
    // approach is suboptimal vs iterative CD but works well starting
    // from a near-optimal Lanczos seed.
    //
    // W and V are ALWAYS unit-norm columns; d tracks the scale, so the
    // current model is W diag(d) V'.
    // ====================================================================

    auto t_phase2_start = Clock::now();

    // Imputation of missing entries with the current model takes more passes
    // than the constraint projection alone.
    const int max_passes = config.max_iter > 0
        ? config.max_iter
        : (use_mask ? 100
                    : std::max(10, static_cast<int>(std::ceil(std::log2(
                          static_cast<double>(k_actual)))) * 2 + 3));
    // A tolerance below single-precision resolution is raised to it
    const Scalar refine_tol = std::max(config.tol,
        static_cast<Scalar>(16) * std::numeric_limits<Scalar>::epsilon());
    const Scalar eps100 = std::numeric_limits<Scalar>::epsilon() * 100;

    DenseMatrix<Scalar> prev_W;
    DenseVector<Scalar> prev_d;  // for loss-based convergence
    DenseMatrix<Scalar> SpMM_result;
    DenseMatrix<Scalar> G(k_actual, k_actual);  // Gram matrix

    double t_spmm_ms = 0, t_solve_ms = 0;

    // norm_sq_vec used for deflation-compatible constraint scaling
    DenseVector<Scalar> norm_sq_vec(k_actual);

    // Pre-allocate SpMM workspace to avoid ~115MB/call thread-local allocs
    SpmmWorkspace<Scalar> spmm_ws;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        spmm_ws.init(m, k_actual, eff_threads);
    }

    // Pre-allocate Cholesky solve buffers (k × m and k × n)
    DenseMatrix<Scalar> Bt_u(k_actual, m);  // k × m — W-update Cholesky buffer
    DenseMatrix<Scalar> Bt_v(k_actual, n);  // k × n — V-update Cholesky buffer

    // E-step: set every missing entry to the current model W diag(d) V'
    auto update_fill = [&]() {
        for (int e = 0; e < masked.count; ++e) {
            const int i = masked.rows[e], j = masked.cols[e];
            Scalar pred = 0;
            for (int c = 0; c < k_actual; ++c) pred += W(i, c) * d(c) * V(j, c);
            masked.fill[e] = pred;
        }
    };

    bool refine_converged = false;
    int passes_done = 0;

    for (int pass = 0; pass < max_passes; ++pass) {
        KSVD_CHECK_INTERRUPT();
        passes_done = pass + 1;

        // ==============================================================
        // W-update: W_new = A · V · (V'V)⁻¹  then project constraints
        // ==============================================================
        auto t_w_start = Clock::now();

        if (use_mask) update_fill();

        // Gram: G = V' · V  (k × k) — unscaled, natural squared norms
        G.noalias() = V.transpose() * V;
        G.diagonal().array() += static_cast<Scalar>(1e-12);
        // L2 goes into the Gram to regularize the solve — this IS the L2 effect
        if (config.L2_u > 0) G.diagonal().array() += config.L2_u;

        // Tier 2: Gram-level regularization using current W (k × m when transposed)
        const bool need_Wt = (config.L21_u > 0) || (config.angular_u > 0) ||
                             (config.graph_u && config.graph_u_lambda > 0);
        DenseMatrix<Scalar> Wt;
        if (need_Wt) Wt = W.transpose();  // k × m — computed once

        if (config.L21_u > 0) {
            features::apply_L21(G, Wt, config.L21_u);
        }
        if (config.angular_u > 0) {
            features::apply_angular(G, Wt, config.angular_u);
        }
        if (config.graph_u && config.graph_u_lambda > 0) {
            features::apply_graph_reg(G, *config.graph_u, Wt, config.graph_u_lambda);
        }

        // SpMM forward: B (m × k) = X · V, missing entries imputed
        spmm_forward<MatrixType, Scalar>(A, V, SpMM_result, rm_ptr, eff_threads, &spmm_ws, sd_ptr);
        if (use_mask) masked.add_forward_block(V, SpMM_result);

        auto t_w_spmm = Clock::now();
        t_spmm_ms += Ms(t_w_spmm - t_w_start).count();

        // Cholesky solve: W_raw = B · G⁻¹
        {
            Eigen::LLT<DenseMatrix<Scalar>> llt(G);
            Bt_u.noalias() = SpMM_result.transpose();  // k × m
            llt.solveInPlace(Bt_u);
            W = Bt_u.transpose();  // m × k
        }

        // Project constraints + normalize W — fused single pass per column.
        // L2 is ALREADY in the Gram solve — pass 0 to avoid double-counting.
        for (int c = 0; c < k_actual; ++c) {
            norm_sq_vec(c) = V.col(c).squaredNorm();
        }
        detail::project_and_normalize(W,
            config.L1_u, static_cast<Scalar>(0), config.nonneg_u, static_cast<Scalar>(0),
            norm_sq_vec, d, eps100);

        auto t_w_solve = Clock::now();
        t_solve_ms += Ms(t_w_solve - t_w_spmm).count();

        // ==============================================================
        // V-update: V_new = A' · W · (W'W)⁻¹  then project constraints
        // ==============================================================
        auto t_v_start = Clock::now();

        if (use_mask) update_fill();

        // Gram: G = W' · W
        G.noalias() = W.transpose() * W;
        G.diagonal().array() += static_cast<Scalar>(1e-12);
        if (config.L2_v > 0) G.diagonal().array() += config.L2_v;

        const bool need_Vt = (config.L21_v > 0) || (config.angular_v > 0) ||
                             (config.graph_v && config.graph_v_lambda > 0);
        DenseMatrix<Scalar> Vt;
        if (need_Vt) Vt = V.transpose();  // k × n — computed once

        if (config.L21_v > 0) {
            features::apply_L21(G, Vt, config.L21_v);
        }
        if (config.angular_v > 0) {
            features::apply_angular(G, Vt, config.angular_v);
        }
        if (config.graph_v && config.graph_v_lambda > 0) {
            features::apply_graph_reg(G, *config.graph_v, Vt, config.graph_v_lambda);
        }

        // SpMM transpose: B (n × k) = X' · W, missing entries imputed
        spmm_transpose<MatrixType, Scalar>(A, W, SpMM_result, rm_ptr, eff_threads, sd_ptr);
        if (use_mask) masked.add_transpose_block(W, SpMM_result);

        auto t_v_spmm = Clock::now();
        t_spmm_ms += Ms(t_v_spmm - t_v_start).count();

        // Cholesky solve: V_raw = B · G⁻¹
        {
            Eigen::LLT<DenseMatrix<Scalar>> llt(G);
            Bt_v.noalias() = SpMM_result.transpose();  // k × n
            llt.solveInPlace(Bt_v);
            V = Bt_v.transpose();  // n × k
        }

        for (int c = 0; c < k_actual; ++c) {
            norm_sq_vec(c) = W.col(c).squaredNorm();
        }
        detail::project_and_normalize(V,
            config.L1_v, static_cast<Scalar>(0), config.nonneg_v, static_cast<Scalar>(0),
            norm_sq_vec, d, eps100);

        auto t_v_solve = Clock::now();
        t_solve_ms += Ms(t_v_solve - t_v_spmm).count();

        // ==============================================================
        // Convergence check: supports FACTOR, LOSS, and BOTH modes
        // ==============================================================
        if (pass > 0) {
            Scalar eps = std::numeric_limits<Scalar>::epsilon();
            bool factor_converged = false;
            bool loss_converged = false;

            // Factor-based: relative change in W
            const bool need_dW = (config.convergence != SVDConvergence::LOSS) || config.verbose;
            Scalar dW = static_cast<Scalar>(0);
            if (need_dW && prev_W.size() > 0) {
                dW = (W - prev_W).norm() / (prev_W.norm() + eps);
                if (config.convergence != SVDConvergence::LOSS && dW < refine_tol)
                    factor_converged = true;
            }

            // Loss-based: relative change in sum(d²) as variance proxy
            if (config.convergence != SVDConvergence::FACTOR && prev_d.size() > 0) {
                Scalar var_new = d.squaredNorm();
                Scalar var_old = prev_d.squaredNorm();
                Scalar dL = std::abs(var_new - var_old) / (var_old + eps);
                if (dL < refine_tol) loss_converged = true;
            }

            if (config.verbose) {
                KSVD_PRINT("  Krylov SVD: pass %d: dW=%.2e  d[0]=%.4e  d[k-1]=%.4e\n",
                           pass + 1, static_cast<double>(dW),
                           static_cast<double>(d(0)),
                           static_cast<double>(d(k_actual - 1)));
            }

            bool converged = false;
            switch (config.convergence) {
                case SVDConvergence::FACTOR: converged = factor_converged; break;
                case SVDConvergence::LOSS:   converged = loss_converged;   break;
                case SVDConvergence::BOTH:   converged = factor_converged || loss_converged; break;
            }

            if (converged) {
                refine_converged = true;
                if (config.verbose) {
                    KSVD_PRINT("  Krylov SVD: converged at pass %d (dW=%.2e, tol=%.2e)\n",
                               pass + 1, static_cast<double>(dW),
                               static_cast<double>(refine_tol));
                }
                break;
            }
        } else if (config.verbose) {
            KSVD_PRINT("  Krylov SVD: pass %d: d[0]=%.4e  d[k-1]=%.4e\n",
                       pass + 1, static_cast<double>(d(0)),
                       static_cast<double>(d(k_actual - 1)));
        }

        if ((config.convergence != SVDConvergence::LOSS) || config.verbose) prev_W = W;
        prev_d = d;
    }

    auto t_phase2_end = Clock::now();

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD: Phase 2 done: %.1f ms (spmm=%.1f, solve+proj=%.1f)\n",
                   Ms(t_phase2_end - t_phase2_start).count(),
                   t_spmm_ms, t_solve_ms);
    }

    if (!refine_converged) {
        char wbuf[320];
        std::snprintf(wbuf, sizeof(wbuf),
            "svd(method = \"krylov\"): refinement did not reach tol = %.1e in %d "
            "passes; increase 'maxit' (number of passes) or loosen 'tol'",
            static_cast<double>(refine_tol), passes_done);
        add_warning(std::string(wbuf));
    }

    // ====================================================================
    // PHASE 2.5: CV auto-rank selection (post-convergence)
    //
    // After the block has converged, evaluate test MSE at each rank
    // k'=1,...,k_actual and find the best rank via patience-based stopping.
    // ====================================================================

    int best_k = k_actual;  // default: use all factors

    // If hard constraints are active (nonneg, L1, L21, graph), QR extraction
    // would DESTROY the constraint projections, so those factors are returned
    // as fitted. Otherwise rotate to proper singular triplets now, so that the
    // rank-k' truncations evaluated below are the true nested SVD models.
    const bool has_hard_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        (config.graph_u != nullptr && config.graph_u_lambda > 0) ||
        (config.graph_v != nullptr && config.graph_v_lambda > 0);

    if (!has_hard_constraints) {
        DenseMatrix<Scalar> W_scaled = W;
        for (int c = 0; c < k_actual; ++c) {
            W_scaled.col(c) *= d(c);
        }
        DenseMatrix<Scalar> U_x(m, k_actual);
        DenseMatrix<Scalar> V_x(n, k_actual);
        DenseVector<Scalar> d_x(k_actual);
        detail::extract_svd_from_factors(W_scaled, V, U_x, d_x, V_x);
        W = U_x;
        V = V_x;
        d = d_x;
    }

    if (do_cv) {
        // Sort factors by descending d for proper rank truncation
        std::vector<int> cv_order(k_actual);
        std::iota(cv_order.begin(), cv_order.end(), 0);
        std::sort(cv_order.begin(), cv_order.end(),
                  [&](int a, int b) { return d(a) > d(b); });

        // Held-out entries in the transformed space (true values)
        TestEntries<Scalar> test_entries = masked.test_entries();

        Scalar best_test_mse = std::numeric_limits<Scalar>::max();
        int patience_counter = 0;

        for (int kk = 0; kk < k_actual; ++kk) {
            int idx = cv_order[kk];
            DenseVector<Scalar> w_col = W.col(idx);
            DenseVector<Scalar> v_col = V.col(idx);
            test_entries.update(w_col, d(idx), v_col, config.threads);
            Scalar test_mse = test_entries.compute_mse(config.threads);

            if (config.verbose) {
                KSVD_PRINT("  Krylov CV: rank %d: test_mse=%.6e\n",
                           kk + 1, static_cast<double>(test_mse));
            }

            result.test_loss_trajectory.push_back(test_mse);

            if (test_mse < best_test_mse) {
                best_test_mse = test_mse;
                best_k = kk + 1;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    if (config.verbose) {
                        KSVD_PRINT("  Krylov CV: patience exhausted at rank %d, "
                                   "best rank = %d (test_mse=%.6e)\n",
                                   kk + 1, best_k, static_cast<double>(best_test_mse));
                    }
                    break;
                }
            }
        }
    }

    // ====================================================================
    // PHASE 3: Extract proper U, Σ, V from constrained factors
    // ====================================================================

    // Return the best_k leading factors in descending-d order. Constrained
    // factors are returned as fitted; the others were rotated to singular
    // triplets above.
    {
        std::vector<int> order(k_actual);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return d(a) > d(b); });

        result.d.resize(best_k);
        result.U.resize(W.rows(), best_k);
        result.V.resize(V.rows(), best_k);
        for (int i = 0; i < best_k; ++i) {
            result.d(i) = d(order[i]);
            result.U.col(i) = W.col(order[i]);
            result.V.col(i) = V.col(order[i]);
        }
    }

    result.k_selected = best_k;
    result.iters_per_factor.push_back(passes_done);

    auto t_end = Clock::now();
    result.wall_time_ms = Ms(t_end - t_start).count();

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD total: %.1f ms (Phase1=%.1f, Phase2=%.1f)\n",
                   result.wall_time_ms,
                   Ms(t_phase1_end - t_phase1_start).count(),
                   Ms(t_phase2_end - t_phase2_start).count());
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet
