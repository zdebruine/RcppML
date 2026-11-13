// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/masked_nnls.hpp
 * @brief Per-column masked NNLS helpers for NMF with user-supplied mask
 *
 * Contains masked_solve_col, masked_nnls_h, masked_nnls_w, and masked_loss.
 * Uses DataAccessor for unified sparse/dense access.
 *
 * Extracted from fit_cpu.hpp namespace mask_detail block.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/primitives/cpu/data_accessor.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/nmf/variant_helpers.hpp>

#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace nmf {

namespace mask_detail {

/**
 * @brief Dispatch per-column solve based on solver_mode.
 *
 * Solves  min_x ½ xᵀG x − bᵀx + L1‖x‖₁ + ½·L2_col·‖x‖²  over the box.
 *
 * @param G_local   Per-column Gram (shared penalties included, L2_col NOT)
 * @param b         ORIGINAL right-hand side (not residual form); modified
 * @param x         Warm start on entry when warm_start, solution on exit
 * @param L1        Absolute L1 penalty
 * @param L2_col    Ridge for this column that is not in G_local
 */
template<typename Scalar>
inline void masked_solve_col(
    const DenseMatrix<Scalar>& G_local,
    Scalar* b,
    Scalar* x,
    int k,
    Scalar L1,
    Scalar L2_col,
    bool nonneg,
    Scalar upper_bound,
    int cd_max_iter,
    Scalar cd_tol,
    int solver_mode,
    bool warm_start)
{
    if (solver_mode == 1) {
        // Direct solve + KKT polish; the warm start is not needed.
        FactorNet::primitives::detail::cholesky_clip_col(
            G_local, b, x, k,
            L1, L2_col,
            nonneg, cd_max_iter, cd_tol, upper_bound);
    } else {
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> b_vec(b, k);
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_vec(x, k);
        if (warm_start) {
            // CD expects the residual of the unpenalized system: b − G·x
            DenseVector<Scalar> Gx = G_local * x_vec;
            b_vec -= Gx;
        } else {
            x_vec.setZero();
        }
        FactorNet::primitives::detail::cd_nnls_col_fixed(
            G_local, b, x, k,
            L1, L2_col,
            nonneg, cd_max_iter, upper_bound, cd_tol);
    }
}

/**
 * @brief Per-column masked solve for the H-update (sparse or dense A).
 *
 * Two modes of missingness, which may be combined:
 *   - mask:        entries listed in `mask` (nonzero = missing) are excluded.
 *                  The per-column Gram is G_shared − Σ_{i masked} w_i w_iᵀ.
 *   - mask_zeros:  zero entries of A are missing too.  The per-column Gram is
 *                  built from the observed (nonzero, unmasked) rows only and
 *                  G_shared must then hold ONLY the shared penalty terms
 *                  (no data Gram).
 *
 * Penalties: L1 / upper bound / nonneg come from `fc` and are enforced inside
 * the solver; the shared ridge terms (L2, target) are expected in G_shared;
 * column-specific terms (graph, target RHS, L21) come from `cp`.
 *
 * DataAccessor<MatrixType> provides the column iteration strategy:
 *   - Sparse: inner iterator over non-zeros only.
 *   - Dense:  fast BLAS mat-vec for unmasked; element-wise for masked columns.
 *
 * @param A         Data matrix (m × n)
 * @param W_T       Fixed factor (k × m)
 * @param G_shared  k × k: W_T·W_Tᵀ + shared penalties (mask mode), or shared
 *                  penalties only (mask_zeros mode)
 * @param H         Solution matrix (k × n), updated in-place
 * @param mask      Sparse mask matrix (m × n), nonzeros = masked
 * @param config    NMF configuration (cd parameters, solver mode)
 * @param fc        Factor configuration of the factor being solved
 * @param cp        Column penalty terms (may be nullptr)
 * @param mask_zeros  Treat zero entries of A as missing
 */
template<typename Scalar, typename MatrixType>
void masked_nnls_h(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_shared,
    DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const ::FactorNet::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start,
    const ::FactorNet::FactorConfig<Scalar>& fc,
    const variant::ColumnPenalty<Scalar>* cp = nullptr,
    bool mask_zeros = false)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int k = static_cast<int>(W_T.rows());
    const int n = static_cast<int>(A.cols());
    const int m = static_cast<int>(A.rows());

    if (H.rows() != k || H.cols() != n) {
        H.resize(k, n);
        H.setZero();
        warm_start = false;
    }

    const DenseMatrix<Scalar>* B_add = cp ? cp->B_ptr() : nullptr;
    const Scalar* col_ridge  = cp ? cp->ridge_ptr() : nullptr;
    const Scalar* group_norm = cp ? cp->group_ptr() : nullptr;
    const bool has_group = (fc.L21 > Scalar(0)) && (group_norm != nullptr);

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < n; ++j) {
        // Collect masked rows for column j
        std::vector<int> masked_rows;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit)
            if (mit.value() != 0)
                masked_rows.push_back(static_cast<int>(mit.row()));

        DenseVector<Scalar> b(k);
        b.setZero();
        DenseMatrix<Scalar> G_local = G_shared;

        if (mask_zeros) {
            // Observed set = nonzero AND not masked.  Gram and RHS over it.
            const bool any_masked = !masked_rows.empty();
            std::vector<bool> is_masked;
            if (any_masked) {
                is_masked.assign(m, false);
                for (int r : masked_rows) is_masked[r] = true;
            }
            DA::for_col(A, j, [&](int r, Scalar v) {
                if (v == Scalar(0)) return;
                if (any_masked && is_masked[r]) return;
                b.noalias() += v * W_T.col(r);
                G_local.noalias() += W_T.col(r) * W_T.col(r).transpose();
            });
        } else {
            if (masked_rows.empty()) {
                // Fast path: no masking — BLAS for dense, element-wise for sparse
                DA::fast_rhs_col(A, W_T, j, b);
            } else {
                std::vector<bool> is_masked(m, false);
                for (int r : masked_rows) is_masked[r] = true;
                DA::for_col(A, j, [&](int r, Scalar v) {
                    if (!is_masked[r])
                        b.noalias() += v * W_T.col(r);
                });
            }
            // Delta-G correction (no-op when masked_rows is empty)
            for (int r : masked_rows)
                G_local.noalias() -= W_T.col(r) * W_T.col(r).transpose();
        }

        // Column-specific penalty terms
        if (B_add) b.noalias() += B_add->col(j);
        Scalar L2_col = col_ridge ? col_ridge[j] : Scalar(0);
        if (has_group) {
            Scalar g_ref = G_local.diagonal().maxCoeff();
            if (!(g_ref > Scalar(0))) g_ref = Scalar(1);
            bool is_zero = false;
            L2_col += features::l21_column_ridge(
                b.data(), k, fc.L1, fc.nonneg, fc.L21, group_norm[j], g_ref, is_zero);
            if (is_zero) {
                H.col(j).setZero();
                continue;
            }
        }

        DenseVector<Scalar> x = warm_start
            ? DenseVector<Scalar>(H.col(j))
            : DenseVector<Scalar>::Zero(k);
        masked_solve_col(G_local, b.data(), x.data(), k,
            fc.L1, L2_col, fc.nonneg, fc.upper_bound,
            config.cd_max_iter, config.cd_tol,
            config.solver_mode, warm_start);
        H.col(j) = x;
    }
}

/**
 * @brief Per-column masked solve for the W-update (sparse or dense A).
 *
 * Same contract as masked_nnls_h (see there for the mask / mask_zeros modes
 * and the penalty conventions).
 *
 * For sparse: pass A^T (n×m CSC); DataAccessor::for_col iterates nonzeros
 * in column j of A^T, which are the nonzero entries in row j of A.
 * For dense:  pass A (m×n);       DataAccessor::for_row iterates all columns
 * of row j, with a BLAS fast-path when no masking is needed.
 *
 * @tparam Scalar     float or double
 * @tparam MatrixType SparseMatrix<Scalar> (pass A^T) or DenseMatrix<Scalar> (pass A)
 * @param data      A^T for sparse (n × m), or A for dense (m × n)
 * @param H         Fixed factor (k × n)
 * @param G_shared  k × k: H·H^T + shared penalties (mask mode), or shared
 *                  penalties only (mask_zeros mode)
 * @param W_T       Solution matrix (k × m), updated in-place
 * @param mask_T    Transposed mask (n × m): col j = masked columns for row j of A
 */
template<typename Scalar, typename MatrixType>
void masked_nnls_w(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    const DenseMatrix<Scalar>& G_shared,
    DenseMatrix<Scalar>& W_T,
    const SparseMatrix<Scalar>& mask_T,
    const ::FactorNet::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start,
    const ::FactorNet::FactorConfig<Scalar>& fc,
    const variant::ColumnPenalty<Scalar>* cp = nullptr,
    bool mask_zeros = false)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int k = static_cast<int>(H.rows());
    const int n_h = static_cast<int>(H.cols());
    // m = number of rows of A: cols of A^T (sparse) or rows of A (dense)
    int m;
    if constexpr (DA::is_sparse)
        m = static_cast<int>(data.cols());
    else
        m = static_cast<int>(data.rows());

    if (W_T.rows() != k || W_T.cols() != m) {
        W_T.resize(k, m);
        W_T.setZero();
        warm_start = false;
    }

    const DenseMatrix<Scalar>* B_add = cp ? cp->B_ptr() : nullptr;
    const Scalar* col_ridge  = cp ? cp->ridge_ptr() : nullptr;
    const Scalar* group_norm = cp ? cp->group_ptr() : nullptr;
    const bool has_group = (fc.L21 > Scalar(0)) && (group_norm != nullptr);

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < m; ++j) {
        // Masked columns for row j of A = nonzeros in column j of mask_T
        std::vector<int> masked_cols;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask_T, j); mit; ++mit)
            if (mit.value() != 0)
                masked_cols.push_back(static_cast<int>(mit.row()));

        DenseVector<Scalar> b(k);
        b.setZero();
        DenseMatrix<Scalar> G_local = G_shared;

        if (mask_zeros) {
            const bool any_masked = !masked_cols.empty();
            std::vector<bool> is_masked;
            if (any_masked) {
                is_masked.assign(n_h, false);
                for (int c : masked_cols) is_masked[c] = true;
            }
            auto accumulate = [&](int c, Scalar v) {
                if (v == Scalar(0)) return;
                if (any_masked && is_masked[c]) return;
                b.noalias() += v * H.col(c);
                G_local.noalias() += H.col(c) * H.col(c).transpose();
            };
            if constexpr (DA::is_sparse) {
                DA::for_col(data, j, accumulate);
            } else {
                DA::for_row(data, j, accumulate);
            }
        } else {
            if (masked_cols.empty()) {
                if constexpr (DA::is_sparse) {
                    DA::fast_rhs_col(data, H, j, b);           // b += H * At.col(j)
                } else {
                    b.noalias() = H * data.row(j).transpose(); // BLAS fast path
                }
            } else {
                std::vector<bool> is_masked(n_h, false);
                for (int c : masked_cols) is_masked[c] = true;
                if constexpr (DA::is_sparse) {
                    DA::for_col(data, j, [&](int c, Scalar v) {
                        if (!is_masked[c]) b.noalias() += v * H.col(c);
                    });
                } else {
                    DA::for_row(data, j, [&](int c, Scalar v) {
                        if (!is_masked[c]) b.noalias() += v * H.col(c);
                    });
                }
            }
            for (int c : masked_cols)
                G_local.noalias() -= H.col(c) * H.col(c).transpose();
        }

        // Column-specific penalty terms
        if (B_add) b.noalias() += B_add->col(j);
        Scalar L2_col = col_ridge ? col_ridge[j] : Scalar(0);
        if (has_group) {
            Scalar g_ref = G_local.diagonal().maxCoeff();
            if (!(g_ref > Scalar(0))) g_ref = Scalar(1);
            bool is_zero = false;
            L2_col += features::l21_column_ridge(
                b.data(), k, fc.L1, fc.nonneg, fc.L21, group_norm[j], g_ref, is_zero);
            if (is_zero) {
                W_T.col(j).setZero();
                continue;
            }
        }

        DenseVector<Scalar> x = warm_start
            ? DenseVector<Scalar>(W_T.col(j))
            : DenseVector<Scalar>::Zero(k);
        masked_solve_col(G_local, b.data(), x.data(), k,
            fc.L1, L2_col, fc.nonneg, fc.upper_bound,
            config.cd_max_iter, config.cd_tol,
            config.solver_mode, warm_start);
        W_T.col(j) = x;
    }
}

/**
 * @brief Compute loss excluding masked entries (sparse or dense A).
 *
 * DataAccessor::for_col iterates non-zeros for sparse (skipping structural zeros)
 * and all entries for dense, matching the original per-type semantics.
 * With skip_zeros = true, zero entries are excluded for dense input as well
 * (mask = "zeros": zeros are missing).
 */
template<typename Scalar, typename MatrixType>
Scalar masked_loss(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_Td,
    const DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const LossConfig<Scalar>& loss_config,
    int threads = 1,
    bool skip_zeros = false)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int n = static_cast<int>(A.cols());
    const int k = static_cast<int>(W_Td.rows());
    Scalar total_loss = 0;

    #pragma omp parallel for reduction(+:total_loss) num_threads(threads) \
        schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < n; ++j) {
        std::vector<bool> is_masked(A.rows(), false);
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit)
            if (mit.value() != 0)
                is_masked[mit.row()] = true;

        DA::for_col(A, j, [&](int i, Scalar a_ij) {
            if (skip_zeros && a_ij == Scalar(0)) return;
            if (!is_masked[i]) {
                Scalar pred = 0;
                for (int f = 0; f < k; ++f)
                    pred += W_Td(f, i) * H(f, j);
                total_loss += compute_loss(a_ij, pred, loss_config);
            }
        });
    }
    return total_loss;
}

}  // namespace mask_detail

}  // namespace nmf
}  // namespace FactorNet
