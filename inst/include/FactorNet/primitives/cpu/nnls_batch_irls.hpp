// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/nnls_batch_irls.hpp
 * @brief IRLS-wrapped batch NNLS for non-MSE losses (MAE, Huber, KL)
 *
 * Wraps the standard CD NNLS solver in an Iteratively Reweighted Least
 * Squares loop. For each column j, the outer IRLS loop:
 *   1. Computes the current residual r_j = a_j - W * h_j
 *   2. Computes per-element IRLS weights from the residual (loss-specific)
 *   3. Builds the weighted Gram: G_w = W^T diag(w) W
 *   4. Builds the weighted RHS:  b_w = W^T diag(w) a_j
 *   5. Applies L1/L2/Tier2 regularization to G_w, b_w
 *   6. Solves via CD NNLS
 *   7. Checks convergence (relative max change < irls_tol)
 *
 * The columns are independent — OpenMP parallelizes over columns.
 *
 * @author Zach DeBruine
 * @date 2026-02-10
 */

#pragma once

#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/math/loss.hpp>
#include <FactorNet/core/types.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

// ========================================================================
// Single-column IRLS NNLS solver
// ========================================================================

namespace detail {

/**
 * @brief Compute distribution-derived IRLS weight (variance function inverse)
 */
template<typename Scalar>
inline Scalar distribution_weight(Scalar predicted,
                                   const LossConfig<Scalar>& loss,
                                   Scalar observed = static_cast<Scalar>(0),
                                   Scalar theta = static_cast<Scalar>(0)) {
    switch (loss.type) {
        case LossType::MAE:
            // Legacy path: standalone MAE (not composed with distribution)
            return static_cast<Scalar>(1);
        case LossType::HUBER:
            // Legacy path: standalone Huber (not composed with distribution)
            return static_cast<Scalar>(1);
        case LossType::KL:
            return irls_weight_kl(predicted);
        case LossType::GP:
            return irls_weight_gp(observed, predicted, theta, loss.gp_blend);
        case LossType::NB:
            return irls_weight_nb(predicted, theta);
        case LossType::GAMMA:
            return irls_weight_power(predicted, static_cast<Scalar>(2));
        case LossType::INVGAUSS:
            return irls_weight_power(predicted, static_cast<Scalar>(3));
        case LossType::TWEEDIE:
            return irls_weight_power(predicted, loss.power_param);
        case LossType::MSE:
            // Gaussian: V(μ)=1, weight=1. Only reaches here if robust_delta>0.
            return static_cast<Scalar>(1);
        default:
            return static_cast<Scalar>(1);
    }
}

/**
 * @brief Compute IRLS weight: distribution weight × optional robust modifier
 *
 * For the new API (distribution + robust), the weight is decomposed:
 *   w = w_distribution(μ, θ) × w_robust(r_P)
 * where r_P = (y − μ) / √(1/w_dist) is the Pearson residual.
 *
 * For legacy MAE/HUBER loss types, falls back to the original behavior.
 */
template<typename Scalar>
inline Scalar compute_irls_weight(Scalar residual, Scalar predicted,
                                   const LossConfig<Scalar>& loss,
                                   Scalar observed = static_cast<Scalar>(0),
                                   Scalar theta = static_cast<Scalar>(0)) {
    // Legacy standalone MAE/HUBER (old API: loss="mae"/"huber")
    if (loss.type == LossType::MAE) {
        return irls_weight_mae(residual);
    }
    if (loss.type == LossType::HUBER) {
        return irls_weight_huber(residual, loss.huber_delta);
    }

    // Distribution-derived weight
    Scalar w_dist = distribution_weight(predicted, loss, observed, theta);

    // Apply robust modifier if active
    if (loss.robust_delta > static_cast<Scalar>(0)) {
        // Pearson residual: r_P = residual / √V(μ) = residual × √w_dist
        Scalar sd_inv = std::sqrt(std::max(w_dist, tiny_num<Scalar>()));
        Scalar pearson_r = residual * sd_inv;
        Scalar w_robust = robust_huber_modifier(pearson_r, loss.robust_delta);
        return w_dist * w_robust;
    }

    return w_dist;
}

/**
 * @brief Pre-allocated workspace for IRLS per-column solve.
 *
 * Avoids per-column heap allocation in the OpenMP parallel loop.
 * The batch function creates one workspace per thread.
 */
template<typename Scalar>
struct IRLSWorkspace {
    DenseMatrix<Scalar> G_w;           // k × k
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_w;    // k
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_old;  // k
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_copy; // k

    // Sparse-Gram path (weight_zeros=false): only nnz_j columns
    DenseMatrix<Scalar> W_nnz;         // k × max_nnz
    DenseMatrix<Scalar> W_nnz_scaled;  // k × max_nnz
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> delta_w; // max_nnz
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> wvals;   // max_nnz
    std::vector<int> nnz_rows;         // max_nnz

    // Full-Gram path (weight_zeros=true): all m rows
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> w_vec;   // m
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> recon;   // m
    DenseMatrix<Scalar> W_scratch;     // k × m

    int allocated_nnz = 0;

    void init_sparse(int k, int max_nnz) {
        G_w.resize(k, k);
        b_w.resize(k);
        x_old.resize(k);
        b_copy.resize(k);
        W_nnz.resize(k, max_nnz);
        W_nnz_scaled.resize(k, max_nnz);
        delta_w.resize(max_nnz);
        wvals.resize(max_nnz);
        nnz_rows.resize(max_nnz);
        allocated_nnz = max_nnz;
    }

    void init_full(int k, int m) {
        G_w.resize(k, k);
        b_w.resize(k);
        x_old.resize(k);
        b_copy.resize(k);
        w_vec.resize(m);
        recon.resize(m);
        W_scratch.resize(k, m);
    }

    void ensure_nnz(int k, int nnz) {
        if (nnz > allocated_nnz) {
            W_nnz.resize(k, nnz);
            W_nnz_scaled.resize(k, nnz);
            delta_w.resize(nnz);
            wvals.resize(nnz);
            nnz_rows.resize(nnz);
            allocated_nnz = nnz;
        }
    }
};

/**
 * @brief Solve one column via IRLS-wrapped CD NNLS (sparse A)
 *
 * Two Gram strategies depending on weight_zeros:
 *
 * weight_zeros=false (batch NMF, default):
 *   Sparse-Gram trick: G_w = G_base + W_nnz * diag(w-1) * W_nnz^T
 *   where W_nnz gathers only the nnz_j columns with non-unit weights.
 *   Cost: O(nnz_j · k²) instead of O(m · k²).  For 10% density → 10× faster.
 *   Reconstruction is also sparse: O(nnz_j · k) instead of O(m · k).
 *
 * weight_zeros=true (streaming NMF):
 *   Full Gram: G_w = W_T * diag(w) * W_T^T.  All rows get distribution weights.
 *
 * @param ws  Pre-allocated workspace (from IRLSWorkspace::init_sparse or init_full)
 */
template<typename Scalar, typename SparseMatType>
void irls_nnls_col_sparse_ws(
    const SparseMatType& A,
    const DenseMatrix<Scalar>& W_T,
    int col_j,
    Scalar* __restrict__ x,
    const DenseMatrix<Scalar>& G_base,
    const LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    bool weight_zeros,
    const Scalar* theta_per_row,
    const Scalar* theta_per_col,
    IRLSWorkspace<Scalar>& ws)
{
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(W_T.cols());

    for (int irls_iter = 0; irls_iter < irls_max_iter; ++irls_iter) {

        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_map(x, k);

        if (!weight_zeros) {
            // ============================================================
            // SPARSE-GRAM PATH: O(nnz_j · k²) Gram, O(nnz_j · k) recon
            //
            // Since w[i]=1 for zero entries, the weighted Gram decomposes:
            //   G_w = G_base + sum_{i in nnz} (w_i - 1) · W_T.col(i) · W_T.col(i)^T
            // We gather W columns at nonzero positions into a k × nnz_j block
            // and compute the delta Gram via a single small GEMM.
            // ============================================================

            int nnz_j = 0;
            for (typename SparseMatType::InnerIterator it(A, col_j); it; ++it) {
                const int row = it.row();
                // Sparse reconstruction: dot product at this row only, O(k)
                const Scalar recon_i = W_T.col(row).dot(x_map);
                const Scalar theta_i = theta_per_col ? theta_per_col[col_j]
                                     : (theta_per_row ? theta_per_row[row]
                                     : static_cast<Scalar>(0));
                const Scalar w = compute_irls_weight(
                    it.value() - recon_i, recon_i, loss, it.value(), theta_i);

                ws.nnz_rows[nnz_j] = row;
                ws.delta_w(nnz_j) = w - Scalar(1);
                ws.wvals(nnz_j) = w * it.value();
                ++nnz_j;
            }

            // Gather W columns at nonzero rows → W_nnz (k × nnz_j)
            for (int idx = 0; idx < nnz_j; ++idx) {
                ws.W_nnz.col(idx) = W_T.col(ws.nnz_rows[idx]);
            }
            auto W_block = ws.W_nnz.leftCols(nnz_j);
            auto dw = ws.delta_w.head(nnz_j);

            // Sparse weighted Gram: G_w = G_base + W_nnz * diag(dw) * W_nnz^T
            ws.G_w = G_base;
            // Scale each column of W_nnz by corresponding delta weight
            for (int idx = 0; idx < nnz_j; ++idx) {
                ws.W_nnz_scaled.col(idx) = W_block.col(idx) * dw(idx);
            }
            ws.G_w.noalias() += ws.W_nnz_scaled.leftCols(nnz_j) * W_block.transpose();

            // Weighted RHS: b_w = W_nnz * wvals  (GEMV, k × nnz_j)
            ws.b_w.noalias() = W_block * ws.wvals.head(nnz_j);

        } else {
            // ============================================================
            // FULL-GRAM PATH: O(m · k²) — used by streaming NMF
            // All rows get distribution-derived weights.
            // ============================================================

            ws.recon.noalias() = W_T.transpose() * x_map;

            for (int i = 0; i < m; ++i) {
                Scalar theta_i = theta_per_col ? theta_per_col[col_j]
                               : (theta_per_row ? theta_per_row[i]
                               : static_cast<Scalar>(0));
                ws.w_vec(i) = compute_irls_weight(
                    -ws.recon(i), ws.recon(i), loss, static_cast<Scalar>(0), theta_i);
            }
            for (typename SparseMatType::InnerIterator it(A, col_j); it; ++it) {
                int row = it.row();
                Scalar theta_i = theta_per_col ? theta_per_col[col_j]
                               : (theta_per_row ? theta_per_row[row]
                               : static_cast<Scalar>(0));
                ws.w_vec(row) = compute_irls_weight(
                    it.value() - ws.recon(row), ws.recon(row), loss, it.value(), theta_i);
            }

            ws.W_scratch.array() = W_T.array().rowwise() * ws.w_vec.transpose().array();
            ws.G_w.noalias() = ws.W_scratch * W_T.transpose();

            ws.b_w.setZero();
            for (typename SparseMatType::InnerIterator it(A, col_j); it; ++it) {
                Scalar wv = ws.w_vec(it.row()) * it.value();
                for (int fi = 0; fi < k; ++fi)
                    ws.b_w(fi) += W_T(fi, it.row()) * wv;
            }
        }

        // Apply L2 to weighted Gram
        if (L2 > 0) {
            for (int i = 0; i < k; ++i)
                ws.G_w(i, i) += L2;
        }

        // Save old x, compute warm-start residual, solve
        ws.x_old = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(x, k);
        ws.b_copy = ws.b_w;
        ws.b_copy.noalias() -= ws.G_w * ws.x_old;

        cd_nnls_col_fixed(ws.G_w, ws.b_copy.data(), x, k, L1, static_cast<Scalar>(0),
                     nonneg, cd_maxit);

        // IRLS convergence check
        Scalar max_change = 0;
        for (int i = 0; i < k; ++i) {
            Scalar rel = std::abs(x[i] - ws.x_old(i)) /
                         (std::abs(ws.x_old(i)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < irls_tol) break;
    }
}

/**
 * @brief Backward-compatible wrapper: allocates workspace locally per call.
 *
 * Used by streaming/CV callers that invoke this directly (not via the batch).
 * The batch function uses irls_nnls_col_sparse_ws with pre-allocated workspaces.
 */
template<typename Scalar, typename SparseMatType>
void irls_nnls_col_sparse(
    const SparseMatType& A,
    const DenseMatrix<Scalar>& W_T,
    int col_j,
    Scalar* __restrict__ x,
    const DenseMatrix<Scalar>& G_base,
    const LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    bool weight_zeros = false,
    const Scalar* theta_per_row = nullptr,
    const Scalar* theta_per_col = nullptr)
{
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(W_T.cols());

    IRLSWorkspace<Scalar> ws;
    if (weight_zeros) {
        ws.init_full(k, m);
    } else {
        // Count nnz for this column
        int col_nnz = 0;
        for (typename SparseMatType::InnerIterator it(A, col_j); it; ++it) ++col_nnz;
        ws.init_sparse(k, std::max(col_nnz, 1));
    }

    irls_nnls_col_sparse_ws(
        A, W_T, col_j, x, G_base, loss,
        L1, L2, nonneg, cd_maxit, cd_tol,
        irls_max_iter, irls_tol, weight_zeros,
        theta_per_row, theta_per_col, ws);
}

/**
 * @brief Solve one column via IRLS-wrapped CD NNLS (dense A)
 */
template<typename Scalar, typename DenseMatType>
void irls_nnls_col_dense(
    const DenseMatType& A,
    const DenseMatrix<Scalar>& W_T,
    int col_j,
    Scalar* __restrict__ x,
    const DenseMatrix<Scalar>& G_base,
    const LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    const Scalar* theta_per_row = nullptr,
    const Scalar* theta_per_col = nullptr)
{
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(W_T.cols());

    DenseMatrix<Scalar> G_w(k, k);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_w(k);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> w_vec(m);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> recon(m);

    for (int irls_iter = 0; irls_iter < irls_max_iter; ++irls_iter) {

        // 1. Reconstruction
        recon.noalias() = W_T.transpose() * Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(x, k);

        // 2. Compute IRLS weights for all entries
        for (int i = 0; i < m; ++i) {
            Scalar observed = A(i, col_j);
            Scalar residual = observed - recon(i);
            Scalar predicted = recon(i);
            Scalar theta = theta_per_col ? theta_per_col[col_j]
                         : (theta_per_row ? theta_per_row[i]
                         : static_cast<Scalar>(0));
            w_vec(i) = compute_irls_weight(residual, predicted, loss, observed, theta);
        }

        // 3. Weighted Gram: G_w = W_T * diag(w) * W_T^T
        // Use Eigen operations for dense case (more cache-friendly)
        auto W_weighted = W_T.array().rowwise() * w_vec.transpose().array();
        G_w.noalias() = DenseMatrix<Scalar>(W_weighted.matrix()) * W_T.transpose();

        // 4. Weighted RHS: b_w = W_T * diag(w) * a_col
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> wa = A.col(col_j).array() * w_vec.array();
        b_w.noalias() = W_T * wa;

        // 5. Apply L2
        if (L2 > 0) {
            for (int i = 0; i < k; ++i)
                G_w(i, i) += L2;
        }

        // 6. Save old x
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_old =
            Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(x, k);

        // Compute initial CD residual: b_w - G_w * x
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_copy = b_w;
        b_copy.noalias() -= G_w * x_old;

        cd_nnls_col_fixed(G_w, b_copy.data(), x, k, L1, static_cast<Scalar>(0),
                     nonneg, cd_maxit);

        // 7. Convergence
        Scalar max_change = 0;
        for (int i = 0; i < k; ++i) {
            Scalar rel = std::abs(x[i] - x_old(i)) /
                         (std::abs(x_old(i)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < irls_tol) break;
    }
}

}  // namespace detail

// ========================================================================
// Batch IRLS NNLS — CPU specializations
// ========================================================================

/**
 * @brief CPU batch IRLS NNLS for sparse A
 *
 * Parallelized over columns with OpenMP. Pre-allocates per-thread
 * workspaces to avoid heap allocation churn in the inner loop.
 * Uses sparse-Gram trick: O(nnz_j · k²) per column instead of O(m · k²).
 */
template<typename Scalar, typename SparseMatType>
void nnls_batch_irls_sparse(
    const SparseMatType& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_base,
    DenseMatrix<Scalar>& H,
    const LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    int threads,
    const Scalar* theta_per_row = nullptr,
    const Scalar* theta_per_col = nullptr)
{
    const int k = static_cast<int>(W_T.rows());
    const int n = static_cast<int>(A.cols());
    H.resize(k, n);
    H.setZero();

    // Find max nnz per column for workspace sizing
    int max_nnz = 0;
    for (int j = 0; j < n; ++j) {
        int col_nnz = static_cast<int>(A.outerIndexPtr()[j + 1] - A.outerIndexPtr()[j]);
        if (col_nnz > max_nnz) max_nnz = col_nnz;
    }
    if (max_nnz == 0) max_nnz = 1;

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    // Pre-allocate per-thread workspaces (avoids per-column heap allocation)
    std::vector<detail::IRLSWorkspace<Scalar>> workspaces(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        workspaces[t].init_sparse(k, max_nnz);
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        detail::irls_nnls_col_sparse_ws(
            A, W_T, j, H.col(j).data(), G_base, loss,
            L1, L2, nonneg, cd_maxit, cd_tol,
            irls_max_iter, irls_tol, false, theta_per_row, theta_per_col,
            workspaces[tid]);
    }
}

/**
 * @brief CPU batch IRLS NNLS for dense A
 */
template<typename Scalar, typename DenseMatType>
void nnls_batch_irls_dense(
    const DenseMatType& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_base,
    DenseMatrix<Scalar>& H,
    const LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    int threads,
    const Scalar* theta_per_row = nullptr,
    const Scalar* theta_per_col = nullptr)
{
    const int k = static_cast<int>(W_T.rows());
    const int n = static_cast<int>(A.cols());
    H.resize(k, n);
    H.setZero();

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        detail::irls_nnls_col_dense(
            A, W_T, j, H.col(j).data(), G_base, loss,
            L1, L2, nonneg, cd_maxit, cd_tol,
            irls_max_iter, irls_tol, theta_per_row, theta_per_col);
    }
}

}  // namespace primitives
}  // namespace FactorNet

