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
    DenseMatrix<Scalar> R_w = qr_w.matrixQR().template topRows(k)
        .template triangularView<Eigen::Upper>();

    DenseMatrix<Scalar> Q_h = qr_h.householderQ() *
        DenseMatrix<Scalar>::Identity(H.rows(), k);
    DenseMatrix<Scalar> R_h = qr_h.matrixQR().template topRows(k)
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
 * @brief Core krylov SVD logic — works on any matrix type (original or training copy)
 *
 * @tparam TrainMatrix  Matrix used for training SpMMs (could be Map or owned SparseMatrix)
 * @tparam OrigMatrix   Original matrix used for test entry initialization
 * @tparam Scalar       Precision type
 */
template<typename TrainMatrix, typename OrigMatrix, typename Scalar>
SVDResult<Scalar> krylov_constrained_svd_impl(
    const TrainMatrix& A_train,
    const OrigMatrix& A_orig,
    const SVDConfig<Scalar>& config,
    std::chrono::high_resolution_clock::time_point t_start)
{
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    const int m = static_cast<int>(A_train.rows());
    const int n = static_cast<int>(A_train.cols());
    const int k = config.k_max;

    // --- Detect whether constraints are active ---
    const bool has_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.L2_u > 0 || config.L2_v > 0 ||
        config.upper_bound_u > 0 || config.upper_bound_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        config.angular_u > 0 || config.angular_v > 0 ||
        config.has_gram_level_reg();

    // If no constraints AND no CV, just run Lanczos directly (optimal).
    // When CV is active, we need Phase 2.5 for test evaluation, so we
    // go through the full refinement path (will converge in ~1 pass).
    if (!has_constraints && !config.is_cv()) {
        if (config.verbose) {
            KSVD_PRINT("  Krylov SVD: no constraints — direct Lanczos\n");
        }
        return lanczos_svd<TrainMatrix, Scalar>(A_train, config);
    }

    int eff_threads = config.threads;
#ifdef _OPENMP
    if (eff_threads <= 0) eff_threads = omp_get_max_threads();
#else
    eff_threads = 1;
#endif

    // ====================================================================
    // PHASE 1: Unconstrained Lanczos seed
    // ====================================================================

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD: Phase 1 — Lanczos seed (k=%d)\n", k);
    }

    SVDConfig<Scalar> lanczos_config;
    lanczos_config.k_max = k;
    lanczos_config.tol = config.tol;
    lanczos_config.max_iter = 0;
    lanczos_config.center = config.center;
    lanczos_config.scale = config.scale;
    lanczos_config.verbose = false;  // suppress Lanczos verbosity
    lanczos_config.seed = config.seed;
    lanczos_config.threads = config.threads;

    auto t_phase1_start = Clock::now();
    SVDResult<Scalar> lanczos_result = lanczos_svd<TrainMatrix, Scalar>(A_train, lanczos_config);
    auto t_phase1_end = Clock::now();

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD: Phase 1 done: %.1f ms, k_actual=%d\n",
                   Ms(t_phase1_end - t_phase1_start).count(),
                   lanczos_result.k_selected);
    }

    const int k_actual = lanczos_result.k_selected;
    if (k_actual == 0) {
        lanczos_result.wall_time_ms = Ms(Clock::now() - t_start).count();
        return lanczos_result;
    }

    // ====================================================================
    // PHASE 2: Gram-solve-then-project refinement
    //
    // Key insight: The Gram matrix G=V'V (or W'W) provides component
    // decoupling — preventing all vectors from collapsing to the dominant
    // direction. The Cholesky direct solve is O(k³ + k²·dim) per update,
    // versus CD solver at O(cd_iters · k² · dim). For cd_iters=20, this
    // is 20× cheaper while producing identical unconstrained solutions.
    //
    // After the Cholesky solve, element-wise constraint projection
    // (nonneg, L1, L2, bounds) is applied. This "solve-then-project"
    // approach is suboptimal vs iterative CD but works well starting
    // from a near-optimal Lanczos seed.
    //
    // Per pass: 2 SpMMs + 2 Grams + 2 Cholesky solves + 2 projections
    //   SpMM: O(nnz · k) each
    //   Gram: O(k² · dim) each
    //   Cholesky: O(k³) factorize + O(k² · dim) backsolve each
    //   Project: O(k · dim) each
    // Total: O(2·nnz·k + 4·k²·dim + 2·k³) — much cheaper than CD
    // ====================================================================

    auto t_phase2_start = Clock::now();

    // Row means for PCA centering
    DenseVector<Scalar> row_means;
    const DenseVector<Scalar>* rm_ptr = nullptr;
    if (config.center) {
        if (lanczos_result.row_means.size() > 0) {
            row_means = lanczos_result.row_means;
        } else {
            row_means = compute_row_means<TrainMatrix, Scalar>(A_train, eff_threads);
        }
        rm_ptr = &row_means;
    }

    // Row inv-SDs for correlation PCA scaling
    DenseVector<Scalar> row_inv_sds;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    if (config.scale) {
        if (lanczos_result.row_sds.size() > 0) {
            row_inv_sds = lanczos_result.row_sds.cwiseInverse();
        } else {
            DenseVector<Scalar> row_sds = compute_row_sds<TrainMatrix, Scalar>(A_train, row_means, eff_threads);
            row_inv_sds = row_sds.cwiseInverse();
        }
        sd_ptr = &row_inv_sds;
    }

    // --- Initialize W (m × k) and V (n × k) from Lanczos ---
    // W and V are unit-norm; d carries the singular value scaling.
    // The first ALS pass will naturally incorporate constraints.
    DenseMatrix<Scalar> W = lanczos_result.U.leftCols(k_actual);   // m × k (unit-norm)
    DenseMatrix<Scalar> V = lanczos_result.V.leftCols(k_actual);   // n × k (unit-norm)
    DenseVector<Scalar> d = lanczos_result.d.head(k_actual);       // k (singular values)

    // CV denominator correction: when holdout entries are zeroed in A_train,
    // the Gram matrix (V'V or W'W) overcounts by including contributions from
    // rows/columns whose data was held out. Scale the data-term of the Gram
    // by (1 - effective_holdout_fraction) to correct.
    Scalar cv_denom_correction = Scalar(1);
    if (config.is_cv()) {
        if (config.mask_zeros) {
            int64_t nnz_est = 0;
            if constexpr (is_sparse_matrix<OrigMatrix>::value) {
                nnz_est = A_orig.nonZeros();
            } else {
                nnz_est = static_cast<int64_t>(m) * n;
            }
            Scalar density = static_cast<Scalar>(nnz_est) /
                            (static_cast<Scalar>(m) * static_cast<Scalar>(n));
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction) * density;
        } else {
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        }
        if (config.verbose) {
            KSVD_PRINT("  Krylov CV: denom_correction=%.4f\n",
                       static_cast<double>(cv_denom_correction));
        }
    }
    const bool do_cv = config.is_cv();

    // --- Gram-solve-then-project refinement passes ---
    //
    // Key design:
    //   - W and V are ALWAYS unit-norm columns; d tracks the scale
    //   - Before SpMM/Gram: scale V (or W) by d to include singular values
    //   - After Cholesky solve: normalize → project constraints → re-normalize
    //   - L1/L2/nonneg applied to UNIT-NORM values (same scale as deflation)
    //   - d-scaled Gram prevents L2 compounding through normalization
    //
    const int max_passes = config.max_iter > 0
        ? config.max_iter
        : std::max(10, static_cast<int>(std::ceil(std::log2(
              static_cast<double>(k_actual)))) * 2 + 3);
    const Scalar refine_tol = config.tol;
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
    if constexpr (is_sparse_matrix<TrainMatrix>::value) {
        spmm_ws.init(m, k_actual, eff_threads);
    }

    // Pre-allocate Cholesky solve buffers (k × m and k × n) to avoid per-pass
    // heap allocation.  Reused each W-update and V-update respectively.
    DenseMatrix<Scalar> Bt_u(k_actual, m);  // k × m — W-update Cholesky buffer
    DenseMatrix<Scalar> Bt_v(k_actual, n);  // k × n — V-update Cholesky buffer

    for (int pass = 0; pass < max_passes; ++pass) {
        KSVD_CHECK_INTERRUPT();

        // ==============================================================
        // W-update: W_new = A · V · (V'V)⁻¹  then project constraints
        //
        // This matches deflation's math:
        //   deflation:  w_j = (A·v) / (v'v)  then  apply_reg(w, L1, L2, nn, v'v)
        //   krylov:     W   = A·V · (V'V)⁻¹   then  project_constraints(W, L1, L2, nn, diag(V'V))
        //
        // No d-scaling of V. The Gram V'V directly encodes the squared
        // norms of each component. After Cholesky solve, SpMM_result/G has
        // values at the SAME scale as deflation's (spmv_result / norm_sq).
        // ==============================================================
        auto t_w_start = Clock::now();

        // Gram: G = V' · V  (k × k) — unscaled, natural squared norms
        G.noalias() = V.transpose() * V;
        if (do_cv) G *= cv_denom_correction;
        G.diagonal().array() += static_cast<Scalar>(1e-12);
        // L2 goes into the Gram to regularize the solve — this IS the L2 effect
        if (config.L2_u > 0) G.diagonal().array() += config.L2_u;

        // Tier 2: Gram-level regularization using current W (k × m when transposed)
        // Compute Wt once and reuse for all Gram-level features
        const bool need_Wt = (config.L21_u > 0) || (config.angular_u > 0) ||
                             (config.graph_u && config.graph_u_lambda > 0);
        DenseMatrix<Scalar> Wt;
        if (need_Wt) Wt = W.transpose();  // k × m — computed once

        // L21: group sparsity — drives entire components of W to zero
        if (config.L21_u > 0) {
            features::apply_L21(G, Wt, config.L21_u);
        }
        // Angular: decorrelates W components — penalizes overlap
        if (config.angular_u > 0) {
            features::apply_angular(G, Wt, config.angular_u);
        }
        // Graph Laplacian: smoothness of W along feature graph
        if (config.graph_u && config.graph_u_lambda > 0) {
            features::apply_graph_reg(G, *config.graph_u, Wt, config.graph_u_lambda);
        }

        // SpMM forward: B (m × k) = A · V
        spmm_forward<TrainMatrix, Scalar>(A_train, V, SpMM_result, rm_ptr, eff_threads, &spmm_ws, sd_ptr);

        auto t_w_spmm = Clock::now();
        t_spmm_ms += Ms(t_w_spmm - t_w_start).count();

        // Cholesky solve: W_raw = B · G⁻¹
        // After solve, W_raw(:,c) ≈ deflation's (A·v_c) / (v_c'v_c)
        // but accounting for inter-component coupling via off-diagonal G
        {
            Eigen::LLT<DenseMatrix<Scalar>> llt(G);
            Bt_u.noalias() = SpMM_result.transpose();  // k × m — no alloc (pre-allocated)
            llt.solveInPlace(Bt_u);
            W = Bt_u.transpose();  // m × k
        }

        // Project constraints + normalize W — fused single pass per column.
        // norm_sq_vec(c) = ||V[:,c]||² for L1 threshold scaling.
        // L2 is ALREADY in the Gram solve — pass 0 to avoid double-counting.
        for (int c = 0; c < k_actual; ++c) {
            norm_sq_vec(c) = V.col(c).squaredNorm();
        }
        if (do_cv) norm_sq_vec *= cv_denom_correction;
        detail::project_and_normalize(W,
            config.L1_u, static_cast<Scalar>(0), config.nonneg_u, config.upper_bound_u,
            norm_sq_vec, d, eps100);

        auto t_w_solve = Clock::now();
        t_solve_ms += Ms(t_w_solve - t_w_spmm).count();

        // ==============================================================
        // V-update: V_new = A' · W · (W'W)⁻¹  then project constraints
        // ==============================================================
        auto t_v_start = Clock::now();

        // Gram: G = W' · W
        G.noalias() = W.transpose() * W;
        if (do_cv) G *= cv_denom_correction;
        G.diagonal().array() += static_cast<Scalar>(1e-12);
        if (config.L2_v > 0) G.diagonal().array() += config.L2_v;

        // Tier 2: Gram-level regularization using current V (k × n when transposed)
        // Compute Vt once and reuse for all Gram-level features
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

        // SpMM transpose: B (n × k) = A' · W
        spmm_transpose<TrainMatrix, Scalar>(A_train, W, SpMM_result, rm_ptr, eff_threads, sd_ptr);

        auto t_v_spmm = Clock::now();
        t_spmm_ms += Ms(t_v_spmm - t_v_start).count();

        // Cholesky solve: V_raw = B · G⁻¹
        {
            Eigen::LLT<DenseMatrix<Scalar>> llt(G);
            Bt_v.noalias() = SpMM_result.transpose();  // k × n — no alloc (pre-allocated)
            llt.solveInPlace(Bt_v);
            V = Bt_v.transpose();  // n × k
        }

        // Project constraints + normalize V — fused single pass per column.
        // norm_sq_vec(c) = ||W[:,c]||² for L1 threshold scaling.
        // L2 already in Gram solve, pass 0 to avoid double-counting.
        for (int c = 0; c < k_actual; ++c) {
            norm_sq_vec(c) = W.col(c).squaredNorm();
        }
        if (do_cv) norm_sq_vec *= cv_denom_correction;
        detail::project_and_normalize(V,
            config.L1_v, static_cast<Scalar>(0), config.nonneg_v, config.upper_bound_v,
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

            // Factor-based: relative change in W (O(m×k) — skip in LOSS-only mode)
            // Still computed in verbose mode so we can print meaningful diagnostics.
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

        // Only store prev_W when it will be used next pass (FACTOR or BOTH mode, or verbose).
        // Avoids an O(m×k) copy in LOSS-only convergence mode.
        if ((config.convergence != SVDConvergence::LOSS) || config.verbose) prev_W = W;
        prev_d = d;
    }

    auto t_phase2_end = Clock::now();

    if (config.verbose) {
        KSVD_PRINT("  Krylov SVD: Phase 2 done: %.1f ms (spmm=%.1f, solve+proj=%.1f)\n",
                   Ms(t_phase2_end - t_phase2_start).count(),
                   t_spmm_ms, t_solve_ms);
    }

    // ====================================================================
    // PHASE 2.5: CV auto-rank selection (post-convergence)
    //
    // After the block has converged, evaluate test MSE at each rank
    // k'=1,...,k_actual and find the best rank via patience-based stopping.
    // This is cheaper than per-pass evaluation since we only do it once.
    // ====================================================================

    SVDResult<Scalar> result;
    result.centered = config.center;
    result.frobenius_norm_sq = static_cast<double>(lanczos_result.frobenius_norm_sq);
    if (config.center) {
        result.row_means = row_means;
    }

    int best_k = k_actual;  // default: use all factors

    if (config.is_cv()) {
        // Sort factors by descending d for proper rank truncation
        std::vector<int> cv_order(k_actual);
        std::iota(cv_order.begin(), cv_order.end(), 0);
        std::sort(cv_order.begin(), cv_order.end(),
                  [&](int a, int b) { return d(a) > d(b); });

        // Initialize test entries via speckled mask from ORIGINAL matrix
        // (not the training copy — test values must reflect true held-out A(i,j))
        int64_t nnz_est = 0;
        if constexpr (is_sparse_matrix<OrigMatrix>::value) {
            nnz_est = A_orig.nonZeros();
        } else {
            nnz_est = static_cast<int64_t>(m) * n;
        }

        FactorNet::nmf::LazySpeckledMask<Scalar> mask(m, n, nnz_est,
                              static_cast<double>(config.test_fraction),
                              config.effective_cv_seed(),
                              config.mask_zeros);

        TestEntries<Scalar> test_entries;
        if constexpr (is_sparse_matrix<OrigMatrix>::value) {
            test_entries = init_test_entries_sparse<Scalar>(A_orig, mask, rm_ptr);
        } else {
            test_entries = init_test_entries_dense<Scalar>(A_orig, mask, rm_ptr);
        }

        // Incrementally add factors and evaluate test MSE
        Scalar best_test_mse = std::numeric_limits<Scalar>::max();
        int patience_counter = 0;

        for (int kk = 0; kk < k_actual; ++kk) {
            int idx = cv_order[kk];
            // Update test residuals: residual -= sigma_k * u_k(i) * v_k(j)
            test_entries.update(W.col(idx), d(idx), V.col(idx), config.threads);
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

    // Determine if any hard constraints are active (nonneg, L1, upper_bound, L21, graph).
    // If so, Phase 3 QR extraction would DESTROY the constraint projections.
    // Instead, return Phase 2 factors directly with d-ordering.
    // L2-only and angular-only are fine with QR (L2 only affects scale,
    // angular encourages decorrelation which QR perfects).
    const bool has_hard_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.upper_bound_u > 0 || config.upper_bound_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        (config.graph_u != nullptr && config.graph_u_lambda > 0) ||
        (config.graph_v != nullptr && config.graph_v_lambda > 0);

    if (has_hard_constraints) {
        // --- Constrained path: return Phase 2 factors directly ---
        // Sort by descending d to match SVD convention, truncate to best_k
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
    } else {
        // --- Unconstrained/L2-only path: QR extraction for perfect orthogonality ---
        // Scale W by d for extraction: W_scaled (m × k) = W · diag(d)
        DenseMatrix<Scalar> W_scaled = W;
        for (int c = 0; c < k_actual; ++c) {
            W_scaled.col(c) *= d(c);
        }
        detail::extract_svd_from_factors(W_scaled, V, result.U, result.d, result.V);

        // Truncate to best_k if CV selected fewer factors
        if (best_k < k_actual) {
            result.U = result.U.leftCols(best_k).eval();
            result.d = result.d.head(best_k).eval();
            result.V = result.V.leftCols(best_k).eval();
        }
    }

    result.k_selected = best_k;

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

// ============================================================================
// Public entry point: krylov_constrained_svd
//
// When CV is active, creates a training matrix with holdout entries zeroed
// so the model never sees the test set during training.
// ============================================================================

template<typename MatrixType, typename Scalar>
SVDResult<Scalar> krylov_constrained_svd(const MatrixType& A,
                                         const SVDConfig<Scalar>& config)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();

    if (config.is_cv()) {
        // Create training matrix with holdout entries zeroed
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
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

        int n_zeroed = 0;
        if constexpr (is_sparse_matrix<MatrixType>::value) {
            // OwnedSparse uses the INPUT matrix's scalar type (e.g. double)
            // even when Scalar=float, because Eigen forbids mixed-type sparse copy.
            // SpMV/SpMM handle float↔double via static_cast internally.
            using MatScalar = typename MatrixType::Scalar;
            using OwnedSparse = Eigen::SparseMatrix<MatScalar, Eigen::ColMajor,
                                                     typename MatrixType::StorageIndex>;
            OwnedSparse A_train = create_training_matrix_sparse<Scalar>(A, mask, n_zeroed);
            if (config.verbose) {
                KSVD_PRINT("  Krylov CV: %d entries zeroed in training matrix\n", n_zeroed);
            }
            return krylov_constrained_svd_impl<OwnedSparse, MatrixType, Scalar>(
                A_train, A, config, t_start);
        } else {
            DenseMatrix<Scalar> A_train = create_training_matrix_dense<Scalar>(A, mask, n_zeroed);
            if (config.verbose) {
                KSVD_PRINT("  Krylov CV: %d entries zeroed in training matrix\n", n_zeroed);
            }
            return krylov_constrained_svd_impl<DenseMatrix<Scalar>, MatrixType, Scalar>(
                A_train, A, config, t_start);
        }
    } else {
        // No CV — use original matrix directly (no copy needed)
        return krylov_constrained_svd_impl<MatrixType, MatrixType, Scalar>(
            A, A, config, t_start);
    }
}

}  // namespace svd
}  // namespace FactorNet

