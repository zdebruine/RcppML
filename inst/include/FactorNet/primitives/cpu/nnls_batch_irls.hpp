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
 * The batch entry points (nnls_batch_irls_sparse / nnls_batch_irls_dense) use
 * the unified column solver detail::irls_solve_col_core: warm start from the
 * current factor, the distribution's own treatment of zero entries (identical
 * for sparse and dense input), and step control so that a column update never
 * increases the penalised deviance. The older per-column functions
 * (irls_nnls_col_sparse / irls_nnls_col_dense) are kept for the streaming
 * callers.
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
#include <stdexcept>
#include <string>

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
 *   w = w_distribution(μ, θ) × w_robust(r_D)
 * where r_D = sqrt(d(y, μ)) is the deviance residual of the working model.
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
        // Deviance residual of the working model: sqrt(d(y, μ)), with
        // y = residual + predicted. Equal to |y − μ| for MSE. The product
        // w_dist × min(1, δ / sqrt(d)) reproduces the gradient of the
        // Huberised deviance that compute_robust_loss() reports.
        (void)observed;
        const Scalar y = residual + predicted;
        const Scalar dev_r = static_cast<Scalar>(
            std::sqrt(unit_deviance(y, predicted, loss, theta)));
        Scalar w_robust = robust_huber_modifier(dev_r, loss.robust_delta);
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

// ========================================================================
// Unified IRLS column solver (sparse and dense input share this code)
// ========================================================================
//
// For one column of A with model value mu_i = W_T.col(i) . x, the solver
// minimises
//
//   F(x) = sum_i f(y_i, mu_i) + L1 sum(x) + (L2/2) |x|^2
//          + (1/2) x' G_penalty x - B_penalty.col(j)' x
//
// with f = unit_deviance / 2 (or the Huber function of the deviance residual
// when robust_delta > 0). Each IRLS iteration builds a quadratic model whose
// gradient at the current x equals the gradient of F, solves it by
// coordinate descent, and accepts the result only if F did not increase;
// otherwise it re-solves the model with a proximal term
// (lam/2) (x - x_cur)' diag(G) (x - x_cur), doubling 1 + lam each time (at
// most IRLS_MAX_HALVINGS times) and, failing that, keeps the current x.
// A damped re-solve shortens the step like step-halving does, but the result
// is still a coordinate-descent solution, so L1 and the non-negativity bound
// keep producing exact zeros; a straight line back to x_cur would not.
// L1 is handled inside coordinate descent (shift for x >= 0, soft threshold
// otherwise) and is part of F.
//
// What L1 does here. The penalty is absolute: L1 * sum(x). When the fixed
// factor W_T is non-negative with unit row sums (the default L1
// normalisation), sum_f x_f = sum_i mu_i, so the penalty is L1 * sum(mu), a
// charge on the total fitted mean. For least squares that subtracts L1 from
// every target and thresholds small entries to exact zeros. For a Poisson
// likelihood, sum(mu - y log mu) + L1 sum(mu) is minimised by the unpenalised
// solution divided by (1 + L1): the fit is rescaled and its zero pattern does
// not depend on L1. Negative binomial and Tweedie (1 < p < 2) lie in between
// and have no thresholding at y > 0 either. With an estimated NB size the
// shrunken mean under-predicts, the moment estimator reads the bias as extra
// variance and lowers the size, the weights move toward 1 / mu^2, and the
// factors become denser as L1 grows. The KKT conditions of F hold in every
// case; a rising count of exact zeros is a property of least squares only.
//
// Entries are split into P (y != 0) and Z (y == 0). The same split is used
// for sparse and dense input, so the two give the same fit. P always
// contributes w_i (y_i - mu_i)^2 / 2 with the Fisher weight w_i = 1 / V(mu_i).
// Z is handled according to IRLSZeroMode:
//
//   NONE          the support excludes zero (gamma, inverse Gaussian,
//                 Tweedie p >= 2); the batch entry point rejects data with
//                 zeros, so Z is empty.
//   LINEAR_CONST  Poisson-type losses (KL, GP, NB without a size, Tweedie
//                 p = 1). f(0, mu) = q mu is linear in x, so Z contributes
//                 the exact linear term c' x with c = sum_Z q_i W_T.col(i),
//                 obtained from the row sums of W_T minus the P rows. No
//                 work proportional to m.
//   LINEAR_MU     NB with a size and Tweedie 1 < p < 2. f(0, mu) is concave
//                 in mu, so its tangent at the current mu is a majoriser: Z
//                 contributes c' x with c = sum_Z f'(0, mu_i) W_T.col(i).
//                 Needs mu at every row: two matrix-vector products, O(m k).
//   FULL          everything else (robust weights, or factors that are not
//                 constrained non-negative): every row gets its own weight
//                 and the full weighted Gram is formed, O(m k^2).
//
// The linear modes need x >= 0 and W_T >= 0, which keeps c >= 0 and the
// subproblem bounded below.

enum class IRLSZeroMode { NONE = 0, LINEAR_CONST = 1, LINEAR_MU = 2, FULL = 3 };

/// Maximum number of damped re-solves per IRLS iteration (step factor 1/2 ... 1/64).
constexpr int IRLS_MAX_HALVINGS = 6;

/// Data support required by a loss: 0 = any real, 1 = non-negative, 2 = strictly positive.
template<typename Scalar>
inline int irls_support_class(const LossConfig<Scalar>& loss) {
    switch (loss.type) {
        case LossType::KL:
        case LossType::GP:
        case LossType::NB:
            return 1;
        case LossType::GAMMA:
        case LossType::INVGAUSS:
            return 2;
        case LossType::TWEEDIE:
            return (static_cast<double>(loss.power_param) < 2.0 - 1e-6) ? 1 : 2;
        default:
            return 0;
    }
}

/// Name of a loss type for error messages.
template<typename Scalar>
inline const char* irls_loss_name(const LossConfig<Scalar>& loss) {
    switch (loss.type) {
        case LossType::KL:       return "kl";
        case LossType::GP:       return "gp";
        case LossType::NB:       return "nb";
        case LossType::GAMMA:    return "gamma";
        case LossType::INVGAUSS: return "inverse_gaussian";
        case LossType::TWEEDIE:  return "tweedie";
        default:                 return "mse";
    }
}

/// Throw if the data lie outside the support of the loss (serial code only).
template<typename Scalar>
inline void irls_check_support(const LossConfig<Scalar>& loss,
                               bool any_negative, bool any_zero) {
    const int support = irls_support_class(loss);
    if (support >= 1 && any_negative)
        throw std::invalid_argument(
            std::string("loss = \"") + irls_loss_name(loss) +
            "\" requires non-negative data, but negative values were found.");
    if (support == 2 && any_zero)
        throw std::invalid_argument(
            std::string("loss = \"") + irls_loss_name(loss) +
            "\" requires strictly positive data, but zeros were found "
            "(for a sparse matrix every structural zero counts). Zero is "
            "outside the support of this distribution; use loss = \"tweedie\" "
            "with 1 < tweedie_power < 2 for non-negative data with zeros, or "
            "mask the zeros.");
}

/// Choose how zero entries are treated (see the block comment above).
template<typename Scalar>
inline IRLSZeroMode irls_select_zero_mode(const LossConfig<Scalar>& loss,
                                          bool nonneg,
                                          const DenseMatrix<Scalar>& W_T,
                                          bool has_theta) {
    if (loss.type == LossType::MAE || loss.type == LossType::HUBER)
        return IRLSZeroMode::FULL;
    const int support = irls_support_class(loss);
    if (support == 2) return IRLSZeroMode::NONE;
    if (support == 0) return IRLSZeroMode::FULL;
    if (loss.robust_delta > static_cast<Scalar>(0)) return IRLSZeroMode::FULL;
    if (!nonneg) return IRLSZeroMode::FULL;
    if (W_T.size() > 0 && W_T.minCoeff() < static_cast<Scalar>(0))
        return IRLSZeroMode::FULL;
    if (loss.type == LossType::NB)
        return has_theta ? IRLSZeroMode::LINEAR_MU : IRLSZeroMode::LINEAR_CONST;
    if (loss.type == LossType::TWEEDIE)
        return (std::abs(static_cast<double>(loss.power_param) - 1.0) < 1e-6)
            ? IRLSZeroMode::LINEAR_CONST : IRLSZeroMode::LINEAR_MU;
    return IRLSZeroMode::LINEAR_CONST;  // KL, GP
}

/// d/dmu of f(0, mu) for the LINEAR_CONST losses (independent of mu).
template<typename Scalar>
inline Scalar irls_zero_slope_const(const LossConfig<Scalar>& loss, Scalar theta) {
    if (loss.type == LossType::GP) {
        const double c = 1.0 + std::max(static_cast<double>(theta), 0.0);
        return static_cast<Scalar>(1.0 / (c * c));
    }
    return static_cast<Scalar>(1);
}

/// d/dmu of f(0, mu) at the current mu for the LINEAR_MU losses.
template<typename Scalar>
inline Scalar irls_zero_slope_mu(Scalar predicted, const LossConfig<Scalar>& loss,
                                 Scalar theta) {
    const double mu = std::max(static_cast<double>(predicted), loss_detail::mu_floor());
    if (loss.type == LossType::NB) {
        const double r = static_cast<double>(theta);
        if (!(r > 0.0)) return static_cast<Scalar>(1);
        return static_cast<Scalar>(r / (r + mu));
    }
    if (loss.type == LossType::TWEEDIE) {
        const double p = static_cast<double>(loss.power_param);
        return static_cast<Scalar>(std::min(std::pow(mu, 1.0 - p), 1e6));
    }
    return static_cast<Scalar>(1);
}

/// Per-entry value of the function the IRLS step control monitors.
template<typename Scalar>
inline double irls_entry_merit(Scalar observed, Scalar predicted,
                               const LossConfig<Scalar>& loss, Scalar theta) {
    const double d = unit_deviance(observed, predicted, loss, theta);
    if (loss.robust_delta > static_cast<Scalar>(0))
        return loss_detail::huber_rho(std::sqrt(d),
                                      static_cast<double>(loss.robust_delta));
    return 0.5 * d;
}

/// Read-only settings shared by every column of one batch solve.
template<typename Scalar>
struct IRLSColContext {
    const DenseMatrix<Scalar>* W_T = nullptr;        // k x m
    const DenseMatrix<Scalar>* G_base = nullptr;     // k x k, unweighted Gram
    const LossConfig<Scalar>* loss = nullptr;
    Scalar L1 = 0;
    Scalar L2 = 0;
    bool nonneg = true;
    int cd_maxit = 100;
    Scalar cd_tol = 0;
    int irls_max_iter = 5;
    Scalar irls_tol = 0;
    IRLSZeroMode zero_mode = IRLSZeroMode::FULL;
    bool step_control = true;
    const Scalar* theta_per_row = nullptr;
    const Scalar* theta_per_col = nullptr;
    const DenseMatrix<Scalar>* G_penalty = nullptr;  // k x k or null
    const DenseMatrix<Scalar>* B_penalty = nullptr;  // k x ncols or null
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* s_plain = nullptr;  // row sums of W_T (k)
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* s_zero = nullptr;   // sum_i q_i W_T.col(i), or null

    Scalar theta_at(int row, int col) const {
        return theta_per_col ? theta_per_col[col]
             : (theta_per_row ? theta_per_row[row] : static_cast<Scalar>(0));
    }
};

/// Per-thread buffers for irls_solve_col_core.
template<typename Scalar>
struct IRLSColWork {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vec;

    DenseMatrix<Scalar> G;      // k x k
    Vec b, b_res, x_old, x_try, c_lin, tmp_k, diag0, r0;   // k

    std::vector<int> rows;      // cap: rows of the entries in P
    Vec y, mu_p, w_p, wy;       // cap
    DenseMatrix<Scalar> Wp;     // k x cap: W_T columns at the P rows
    DenseMatrix<Scalar> Wp_s;   // k x cap: the same, scaled by the weights

    Vec recon, w_full;          // m (LINEAR_MU and FULL)
    double last_zero = 0.0;     // zero-entry part of the last irls_col_merit value
    DenseMatrix<Scalar> W_s;    // k x m (FULL)

    void init(int k, int cap, int m, IRLSZeroMode mode) {
        if (cap < 1) cap = 1;
        G.resize(k, k);
        b.resize(k);
        b_res.resize(k);
        x_old.resize(k);
        x_try.resize(k);
        c_lin.resize(k);
        tmp_k.resize(k);
        diag0.resize(k);
        r0.resize(k);
        rows.resize(static_cast<size_t>(cap));
        y.resize(cap);
        mu_p.resize(cap);
        w_p.resize(cap);
        wy.resize(cap);
        Wp.resize(k, cap);
        Wp_s.resize(k, cap);
        if (mode == IRLSZeroMode::LINEAR_MU || mode == IRLSZeroMode::FULL) {
            recon.resize(m);
            w_full.resize(m);
        }
        if (mode == IRLSZeroMode::FULL) W_s.resize(k, m);
    }
};

/**
 * @brief Tangent slopes d/dmu f(0, mu) at every row (LINEAR_MU), single
 *        precision, with the loss and dispersion layout resolved outside
 *        the loop.
 */
template<typename Scalar>
inline void irls_fill_zero_slopes(
    const IRLSColContext<Scalar>& ctx, int col_j,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& recon,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& out)
{
    const LossConfig<Scalar>& loss = *ctx.loss;
    const int m = static_cast<int>(recon.size());
    const Scalar floor_mu = static_cast<Scalar>(loss_detail::mu_floor());
    const Scalar one = static_cast<Scalar>(1);
    if (loss.type == LossType::NB) {
        if (ctx.theta_per_col) {
            const Scalar r = ctx.theta_per_col[col_j];
            if (!(r > static_cast<Scalar>(0))) {
                out.setOnes();
            } else {
                for (int i = 0; i < m; ++i)
                    out(i) = r / (r + std::max(recon(i), floor_mu));
            }
        } else if (ctx.theta_per_row) {
            const Scalar* th = ctx.theta_per_row;
            for (int i = 0; i < m; ++i) {
                const Scalar r = th[i];
                out(i) = (r > static_cast<Scalar>(0))
                    ? r / (r + std::max(recon(i), floor_mu)) : one;
            }
        } else {
            out.setOnes();
        }
    } else if (loss.type == LossType::TWEEDIE) {
        const Scalar e = one - loss.power_param;
        out = recon.array().max(floor_mu).pow(e).min(static_cast<Scalar>(1e6)).matrix();
    } else {
        out.setOnes();
    }
}

/**
 * @brief Value of the penalised column objective F(x) (double precision).
 *
 * Uses ws.rows / ws.y / ws.Wp (the P entries of the column). Overwrites
 * ws.mu_p (model value at the P rows for this x), ws.tmp_k and, for
 * zero_kind 2, ws.recon (model value at every row). Stores the zero-entry
 * part of the returned value in ws.last_zero.
 *
 * @param zero_kind how the zero entries are counted:
 *   0  not at all (returns the P part plus penalties);
 *   1  by the linear term ws.c_lin' x. Exact for LINEAR_CONST. For LINEAR_MU
 *      it is the tangent majoriser built at the current iterate: the value
 *      is an upper bound of F that is tight at that iterate, so a step that
 *      does not increase it does not increase F. O(k), no pass over m.
 *   2  exactly, from the model value at every row. O(m k) plus one deviance
 *      evaluation per row.
 */
template<typename Scalar>
inline double irls_col_merit(const IRLSColContext<Scalar>& ctx,
                             IRLSColWork<Scalar>& ws,
                             int nnz, int col_j, const Scalar* x,
                             int zero_kind)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vec;
    const DenseMatrix<Scalar>& W_T = *ctx.W_T;
    const LossConfig<Scalar>& loss = *ctx.loss;
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(W_T.cols());
    Eigen::Map<const Vec> xm(x, k);

    double total = 0.0;

    if (nnz > 0) {
        ws.mu_p.head(nnz).noalias() = ws.Wp.leftCols(nnz).transpose() * xm;
        for (int idx = 0; idx < nnz; ++idx) {
            total += irls_entry_merit(ws.y(idx), ws.mu_p(idx), loss,
                                      ctx.theta_at(ws.rows[idx], col_j));
        }
    }

    double zero_part = 0.0;
    if (zero_kind == 1) {
        zero_part = static_cast<double>(ws.c_lin.dot(xm));
    } else if (zero_kind == 2) {
        ws.recon.noalias() = W_T.transpose() * xm;
        double z = 0.0;
        for (int i = 0; i < m; ++i) {
            z += irls_entry_merit(static_cast<Scalar>(0), ws.recon(i), loss,
                                  ctx.theta_at(i, col_j));
        }
        for (int idx = 0; idx < nnz; ++idx) {
            const int row = ws.rows[idx];
            z -= irls_entry_merit(static_cast<Scalar>(0), ws.recon(row), loss,
                                  ctx.theta_at(row, col_j));
        }
        zero_part = z;
    }
    ws.last_zero = zero_part;
    total += zero_part;

    if (ctx.L1 != static_cast<Scalar>(0)) {
        const Scalar l1norm = ctx.nonneg ? xm.sum() : xm.cwiseAbs().sum();
        total += static_cast<double>(ctx.L1) * static_cast<double>(l1norm);
    }
    if (ctx.L2 != static_cast<Scalar>(0)) {
        total += 0.5 * static_cast<double>(ctx.L2) *
                 static_cast<double>(xm.squaredNorm());
    }
    if (ctx.G_penalty) {
        ws.tmp_k.noalias() = (*ctx.G_penalty) * xm;
        total += 0.5 * static_cast<double>(xm.dot(ws.tmp_k));
    }
    if (ctx.B_penalty) {
        total -= static_cast<double>(ctx.B_penalty->col(col_j).dot(xm));
    }
    return total;
}

/**
 * @brief Solve one column by IRLS with warm start and step control.
 *
 * On entry ws.rows[0..nnz), ws.y[0..nnz) hold the rows and values of the
 * non-zero entries of the column and ws.Wp.leftCols(nnz) the matching
 * columns of W_T; x holds the starting point (all zeros = cold start).
 */
template<typename Scalar>
inline void irls_solve_col_core(const IRLSColContext<Scalar>& ctx,
                                IRLSColWork<Scalar>& ws,
                                int nnz, int col_j, Scalar* x)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vec;
    const DenseMatrix<Scalar>& W_T = *ctx.W_T;
    const LossConfig<Scalar>& loss = *ctx.loss;
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(W_T.cols());
    const IRLSZeroMode mode = ctx.zero_mode;
    Eigen::Map<Vec> xm(x, k);

    // How irls_col_merit counts the zero entries:
    //   kind_true  the exact value of F (used to compare starting points);
    //   kind_step  the value tested by the step control. For LINEAR_MU this
    //              is the tangent majoriser, which needs no pass over m;
    //   kind_base  what F_old carries between iterations (for LINEAR_MU the
    //              P part and penalties only; the tangent term is added once
    //              the tangent at the current iterate is known).
    const int kind_true = (mode == IRLSZeroMode::NONE) ? 0
                        : (mode == IRLSZeroMode::LINEAR_CONST) ? 1 : 2;
    const int kind_step = (mode == IRLSZeroMode::LINEAR_MU) ? 1 : kind_true;
    const int kind_base = (mode == IRLSZeroMode::LINEAR_MU) ? 0 : kind_true;

    // ---- Exact linear term of the zero entries (LINEAR_CONST) ----
    if (mode == IRLSZeroMode::LINEAR_CONST) {
        if (ctx.s_zero) {
            ws.c_lin = *ctx.s_zero;
        } else {
            const Scalar q_col = irls_zero_slope_const(
                loss, ctx.theta_per_col ? ctx.theta_per_col[col_j]
                                        : static_cast<Scalar>(0));
            ws.c_lin = (*ctx.s_plain) * q_col;
        }
        if (nnz > 0) {
            for (int idx = 0; idx < nnz; ++idx) {
                ws.w_p(idx) = irls_zero_slope_const(
                    loss, ctx.theta_at(ws.rows[idx], col_j));
            }
            ws.c_lin.noalias() -= ws.Wp.leftCols(nnz) * ws.w_p.head(nnz);
        }
        // c >= 0 in exact arithmetic; remove rounding noise.
        ws.c_lin = ws.c_lin.cwiseMax(static_cast<Scalar>(0));
    }

    // ---- Starting point ----
    bool cold = true;
    for (int f = 0; f < k; ++f) {
        if (x[f] != static_cast<Scalar>(0)) { cold = false; break; }
    }
    if (cold) {
        // Unweighted non-negative least squares gives the first iterate.
        ws.G = *ctx.G_base;
        if (ctx.L2 > static_cast<Scalar>(0)) ws.G.diagonal().array() += ctx.L2;
        if (ctx.G_penalty) ws.G += *ctx.G_penalty;
        if (nnz > 0) {
            ws.b.noalias() = ws.Wp.leftCols(nnz) * ws.y.head(nnz);
        } else {
            ws.b.setZero();
        }
        if (ctx.B_penalty) ws.b += ctx.B_penalty->col(col_j);
        ws.b_res = ws.b;
        cd_nnls_col_fixed(ws.G, ws.b_res.data(), x, k,
                          ctx.L1, static_cast<Scalar>(0),
                          ctx.nonneg, ctx.cd_maxit,
                          static_cast<Scalar>(0), ctx.cd_tol);
    } else if (ctx.step_control && nnz > 0) {
        // The caller may hand over a factor that is missing the diagonal
        // scaling. Match the column total if that lowers the objective.
        const double sum_y = static_cast<double>(ws.y.head(nnz).sum());
        const double sum_mu = static_cast<double>(ctx.s_plain->dot(xm));
        if (sum_y > 0.0 && sum_mu > 0.0) {
            const double scale = sum_y / sum_mu;
            if (std::isfinite(scale) && std::abs(scale - 1.0) > 0.05) {
                const double F_a = irls_col_merit(ctx, ws, nnz, col_j, x, kind_true);
                ws.x_try = xm * static_cast<Scalar>(scale);
                const double F_b = irls_col_merit(ctx, ws, nnz, col_j,
                                                  ws.x_try.data(), kind_true);
                if (F_b < F_a) xm = ws.x_try;
            }
        }
    }

    double F_old = ctx.step_control
        ? irls_col_merit(ctx, ws, nnz, col_j, x, kind_base) : 0.0;

    // With step control the last irls_col_merit call is always made at the
    // accepted iterate, so ws.mu_p (and ws.recon for kind 2) already hold the
    // model value at the current x and are not recomputed.
    const bool mu_cached = ctx.step_control;
    const bool recon_cached = ctx.step_control && (kind_base == 2);

    for (int irls_iter = 0; irls_iter < ctx.irls_max_iter; ++irls_iter) {

        // ---- Weights at the non-zero entries ----
        if (nnz > 0) {
            if (!mu_cached)
                ws.mu_p.head(nnz).noalias() = ws.Wp.leftCols(nnz).transpose() * xm;
            for (int idx = 0; idx < nnz; ++idx) {
                const Scalar mu = ws.mu_p(idx);
                const Scalar yv = ws.y(idx);
                const Scalar w = compute_irls_weight(
                    yv - mu, mu, loss, yv, ctx.theta_at(ws.rows[idx], col_j));
                ws.w_p(idx) = w;
                ws.wy(idx) = w * yv;
            }
            ws.b.noalias() = ws.Wp.leftCols(nnz) * ws.wy.head(nnz);
        } else {
            ws.b.setZero();
        }

        // ---- Weighted Gram ----
        if (mode == IRLSZeroMode::FULL) {
            if (!recon_cached) ws.recon.noalias() = W_T.transpose() * xm;
            for (int i = 0; i < m; ++i) {
                ws.w_full(i) = compute_irls_weight(
                    -ws.recon(i), ws.recon(i), loss,
                    static_cast<Scalar>(0), ctx.theta_at(i, col_j));
            }
            for (int idx = 0; idx < nnz; ++idx)
                ws.w_full(ws.rows[idx]) = ws.w_p(idx);
            ws.W_s.array() = W_T.array().rowwise() * ws.w_full.transpose().array();
            ws.G.noalias() = ws.W_s * W_T.transpose();
        } else if (nnz > 0) {
            // G = Wp diag(w) Wp' as a symmetric rank update: one triangle
            // is computed from Wp scaled by sqrt(w), then mirrored.
            for (int idx = 0; idx < nnz; ++idx)
                ws.Wp_s.col(idx) = ws.Wp.col(idx) * std::sqrt(ws.w_p(idx));
            ws.G.setZero();
            ws.G.template selfadjointView<Eigen::Lower>().rankUpdate(
                ws.Wp_s.leftCols(nnz));
            ws.G.template triangularView<Eigen::StrictlyUpper>() = ws.G.transpose();
        } else {
            ws.G.setZero();
        }

        // ---- Linear term of the zero entries ----
        if (mode == IRLSZeroMode::LINEAR_MU) {
            ws.recon.noalias() = W_T.transpose() * xm;
            irls_fill_zero_slopes(ctx, col_j, ws.recon, ws.w_full);
            for (int idx = 0; idx < nnz; ++idx)
                ws.w_full(ws.rows[idx]) = static_cast<Scalar>(0);
            ws.c_lin.noalias() = W_T * ws.w_full;
            ws.b -= ws.c_lin;
        } else if (mode == IRLSZeroMode::LINEAR_CONST) {
            ws.b -= ws.c_lin;
        }

        // ---- Penalties ----
        if (ctx.L2 > static_cast<Scalar>(0)) ws.G.diagonal().array() += ctx.L2;
        if (ctx.G_penalty) ws.G += *ctx.G_penalty;
        if (ctx.B_penalty) ws.b += ctx.B_penalty->col(col_j);

        ws.x_old = xm;
        // Residual of the unpenalised model at the current x (L1 is applied
        // by coordinate descent).
        ws.r0 = ws.b;
        ws.r0.noalias() -= ws.G * ws.x_old;

        // A factor with no curvature (it does not load on any weighted row,
        // so its row and column of G are zero) and a negative gradient term
        // is optimal at zero; coordinate descent skips zero-diagonal
        // coordinates, so set it here.
        if (ctx.nonneg) {
            for (int f = 0; f < k; ++f) {
                if (ws.G(f, f) <= static_cast<Scalar>(0) &&
                    ws.b(f) - ctx.L1 < static_cast<Scalar>(0)) {
                    x[f] = static_cast<Scalar>(0);
                }
            }
        }

        // ---- Solve the quadratic model, warm-started at x ----
        ws.b_res = ws.r0;
        cd_nnls_col_fixed(ws.G, ws.b_res.data(), x, k,
                          ctx.L1, static_cast<Scalar>(0),
                          ctx.nonneg, ctx.cd_maxit,
                          static_cast<Scalar>(0), ctx.cd_tol);

        // ---- Step control: never accept an increase of F ----
        if (ctx.step_control) {
            // Reference value at x_old. For LINEAR_MU add the tangent term
            // at x_old, where the majoriser equals F.
            double F_ref = F_old;
            if (mode == IRLSZeroMode::LINEAR_MU)
                F_ref += static_cast<double>(ws.c_lin.dot(ws.x_old));
            double F_new = irls_col_merit(ctx, ws, nnz, col_j, x, kind_step);
            const double slack = 1e-6 * (std::abs(F_ref) + 1e-12);
            if (!(F_new <= F_ref + slack)) {
                // Damped re-solves from x_old. With D = diag(G) the damped
                // model has Gram G + lam D and right-hand side b + lam D x_old,
                // so its residual at x_old is still r0.
                ws.diag0 = ws.G.diagonal();
                const Scalar dmax = ws.diag0.maxCoeff();
                const Scalar dfloor = (dmax > static_cast<Scalar>(0))
                    ? dmax * static_cast<Scalar>(1e-6) : static_cast<Scalar>(1);
                bool accepted = false;
                Scalar mult = static_cast<Scalar>(1);
                for (int h = 0; h < IRLS_MAX_HALVINGS; ++h) {
                    mult *= static_cast<Scalar>(2);
                    for (int f = 0; f < k; ++f) {
                        ws.G(f, f) = ws.diag0(f) + (mult - static_cast<Scalar>(1)) *
                                     std::max(ws.diag0(f), dfloor);
                    }
                    xm = ws.x_old;
                    ws.b_res = ws.r0;
                    cd_nnls_col_fixed(ws.G, ws.b_res.data(), x, k,
                                      ctx.L1, static_cast<Scalar>(0),
                                      ctx.nonneg, ctx.cd_maxit,
                                      static_cast<Scalar>(0), ctx.cd_tol);
                    F_new = irls_col_merit(ctx, ws, nnz, col_j, x, kind_step);
                    if (F_new <= F_ref + slack) { accepted = true; break; }
                }
                if (!accepted) {
                    xm = ws.x_old;
                    break;
                }
            }
            // Carry the part that does not depend on the tangent.
            F_old = (mode == IRLSZeroMode::LINEAR_MU)
                ? F_new - ws.last_zero : F_new;
        }

        // ---- IRLS convergence check ----
        Scalar max_change = 0;
        for (int i = 0; i < k; ++i) {
            Scalar rel = std::abs(x[i] - ws.x_old(i)) /
                         (std::abs(ws.x_old(i)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < ctx.irls_tol) break;
    }
}

/**
 * @brief Shared driver: validates, prepares the context and runs all columns.
 *
 * @param gather  callable (int j, IRLSColWork<Scalar>& ws) -> int: writes the
 *                rows and values of the non-zero entries of column j into
 *                ws.rows / ws.y and returns their count (at most `cap`).
 */
template<typename Scalar, typename GatherFn>
inline void irls_batch_run(
    int n, int cap,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_base,
    DenseMatrix<Scalar>& H,
    const LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    int threads,
    const Scalar* theta_per_row,
    const Scalar* theta_per_col,
    const DenseMatrix<Scalar>* G_penalty,
    const DenseMatrix<Scalar>* B_penalty,
    const GatherFn& gather)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vec;
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(W_T.cols());

    if (G_penalty && (G_penalty->rows() != k || G_penalty->cols() != k))
        throw std::invalid_argument("IRLS: G_penalty must be k x k.");
    if (B_penalty && (B_penalty->rows() != k || B_penalty->cols() != n))
        throw std::invalid_argument("IRLS: B_penalty must be k x ncol.");

    // Warm start: keep the current factor when it has the right shape and
    // is finite; otherwise start cold from zero.
    if (H.rows() != k || H.cols() != n || !H.allFinite()) {
        H.resize(k, n);
        H.setZero();
    }

    const bool has_theta = (theta_per_row != nullptr) || (theta_per_col != nullptr);

    IRLSColContext<Scalar> ctx;
    ctx.W_T = &W_T;
    ctx.G_base = &G_base;
    ctx.loss = &loss;
    ctx.L1 = L1;
    ctx.L2 = L2;
    ctx.nonneg = nonneg;
    ctx.cd_maxit = cd_maxit;
    ctx.cd_tol = cd_tol;
    ctx.irls_max_iter = irls_max_iter;
    ctx.irls_tol = irls_tol;
    ctx.zero_mode = irls_select_zero_mode(loss, nonneg, W_T, has_theta);
    ctx.step_control = !(loss.type == LossType::MAE || loss.type == LossType::HUBER);
    ctx.theta_per_row = theta_per_row;
    ctx.theta_per_col = theta_per_col;
    ctx.G_penalty = G_penalty;
    ctx.B_penalty = B_penalty;

    Vec s_plain = W_T.rowwise().sum();
    ctx.s_plain = &s_plain;

    Vec s_zero;
    if (ctx.zero_mode == IRLSZeroMode::LINEAR_CONST &&
        loss.type == LossType::GP && theta_per_row && !theta_per_col) {
        Vec q(m);
        for (int i = 0; i < m; ++i)
            q(i) = irls_zero_slope_const(loss, theta_per_row[i]);
        s_zero = W_T * q;
        ctx.s_zero = &s_zero;
    }

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
#else
    int n_threads = 1;
    (void)threads;
#endif
    if (n_threads < 1) n_threads = 1;

    std::vector<IRLSColWork<Scalar>> workspaces(static_cast<size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t)
        workspaces[static_cast<size_t>(t)].init(k, cap, m, ctx.zero_mode);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        IRLSColWork<Scalar>& ws = workspaces[static_cast<size_t>(tid)];
        const int nnz = gather(j, ws);
        for (int idx = 0; idx < nnz; ++idx)
            ws.Wp.col(idx) = W_T.col(ws.rows[static_cast<size_t>(idx)]);
        irls_solve_col_core(ctx, ws, nnz, j, H.col(j).data());
    }
}

}  // namespace detail

// ========================================================================
// Batch IRLS NNLS — CPU specializations
// ========================================================================

/**
 * @brief CPU batch IRLS NNLS for sparse A
 *
 * Parallelized over columns with OpenMP. Stored zeros and structural zeros
 * are treated alike and get the distribution's own treatment (see the block
 * comment above detail::IRLSZeroMode), so a matrix gives the same fit as
 * sparse or dense input.
 *
 * H is the starting point when it is k x n and finite (warm start);
 * otherwise it is resized and the solve starts cold.
 *
 * Cost per IRLS iteration, per column with nnz_j non-zeros:
 *   Poisson-type losses (KL/GP, NB without size, Tweedie p = 1): O(nnz_j k^2)
 *   NB with size, Tweedie 1 < p < 2:                           O(nnz_j k^2 + m k)
 *   robust weights or unconstrained factors:                   O(m k^2)
 *   gamma, inverse Gaussian, Tweedie p >= 2 (no zeros allowed): O(m k^2)
 *
 * @param G_penalty optional k x k matrix added to every weighted Gram
 * @param B_penalty optional k x ncol matrix; column j is added to the
 *                  right-hand side of column j
 *
 * @throws std::invalid_argument for an unsupported Tweedie power, data
 *         outside the support of the loss, or mis-sized penalties
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
    const Scalar* theta_per_col = nullptr,
    const DenseMatrix<Scalar>* G_penalty = nullptr,
    const DenseMatrix<Scalar>* B_penalty = nullptr)
{
    const int n = static_cast<int>(A.cols());
    const int m = static_cast<int>(A.rows());

    validate_loss_config(loss);
    if (static_cast<int>(W_T.cols()) != m)
        throw std::invalid_argument("IRLS: factor and data dimensions do not match.");

    // Support check and workspace sizing in one pass over the stored entries.
    int max_nnz = 0;
    bool any_negative = false;
    bool any_zero = false;
    for (int j = 0; j < n; ++j) {
        int col_nnz = 0;
        for (typename SparseMatType::InnerIterator it(A, j); it; ++it) {
            const Scalar v = static_cast<Scalar>(it.value());
            if (v < static_cast<Scalar>(0)) any_negative = true;
            if (v != static_cast<Scalar>(0)) ++col_nnz;
        }
        if (col_nnz < m) any_zero = true;
        if (col_nnz > max_nnz) max_nnz = col_nnz;
    }
    detail::irls_check_support(loss, any_negative, any_zero);

    auto gather = [&A](int j, detail::IRLSColWork<Scalar>& ws) -> int {
        int nnz = 0;
        for (typename SparseMatType::InnerIterator it(A, j); it; ++it) {
            const Scalar v = static_cast<Scalar>(it.value());
            if (v == static_cast<Scalar>(0)) continue;
            ws.rows[static_cast<size_t>(nnz)] = static_cast<int>(it.row());
            ws.y(nnz) = v;
            ++nnz;
        }
        return nnz;
    };

    detail::irls_batch_run(
        n, max_nnz, W_T, G_base, H, loss, L1, L2, nonneg,
        cd_maxit, cd_tol, irls_max_iter, irls_tol, threads,
        theta_per_row, theta_per_col, G_penalty, B_penalty, gather);
}

/**
 * @brief CPU batch IRLS NNLS for dense A
 *
 * Same algorithm and options as nnls_batch_irls_sparse: entries equal to
 * zero are handled by the zero rule of the loss, all others by their
 * Fisher weight.
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
    const Scalar* theta_per_col = nullptr,
    const DenseMatrix<Scalar>* G_penalty = nullptr,
    const DenseMatrix<Scalar>* B_penalty = nullptr)
{
    const int n = static_cast<int>(A.cols());
    const int m = static_cast<int>(A.rows());

    validate_loss_config(loss);
    if (static_cast<int>(W_T.cols()) != m)
        throw std::invalid_argument("IRLS: factor and data dimensions do not match.");

    bool any_negative = false;
    bool any_zero = false;
    if (detail::irls_support_class(loss) > 0) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const Scalar v = static_cast<Scalar>(A(i, j));
                if (v < static_cast<Scalar>(0)) any_negative = true;
                if (v == static_cast<Scalar>(0)) any_zero = true;
            }
        }
    }
    detail::irls_check_support(loss, any_negative, any_zero);

    auto gather = [&A, m](int j, detail::IRLSColWork<Scalar>& ws) -> int {
        int nnz = 0;
        for (int i = 0; i < m; ++i) {
            const Scalar v = static_cast<Scalar>(A(i, j));
            if (v == static_cast<Scalar>(0)) continue;
            ws.rows[static_cast<size_t>(nnz)] = i;
            ws.y(nnz) = v;
            ++nnz;
        }
        return nnz;
    };

    detail::irls_batch_run(
        n, m, W_T, G_base, H, loss, L1, L2, nonneg,
        cd_maxit, cd_tol, irls_max_iter, irls_tol, threads,
        theta_per_row, theta_per_col, G_penalty, B_penalty, gather);
}

}  // namespace primitives
}  // namespace FactorNet
