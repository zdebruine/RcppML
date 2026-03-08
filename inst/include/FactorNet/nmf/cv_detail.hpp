// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/cv_detail.hpp
 * @brief Internal helper functions for cross-validation NMF
 *
 * Extracted from fit_cv.hpp to keep the main fitting function readable.
 * Contains:
 *   - apply_gram_correction  : MSE per-column Gram down-rank update
 *   - irls_solve_col_cv      : IRLS-wrapped H-column solve (unified sparse/dense)
 *   - irls_solve_row_cv      : IRLS-wrapped W-row solve (unified sparse/dense)
 *   - compute_train_rhs      : build train-only RHS for H update
 *   - compute_train_rhs_W    : build train-only RHS for W update
 *   - User mask helpers      : adjust_rhs_for_user_mask_h/w, is_user_masked_h
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/primitives/cpu/data_accessor.hpp>
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/nmf/variant_helpers.hpp>

#include <vector>
#include <algorithm>
#include <cmath>

namespace FactorNet {
namespace nmf {

// ========================================================================
// CV-specific internal helpers
// ========================================================================

namespace detail {

// ========================================================================
// Gram correction helper (MSE path only)
// ========================================================================

/**
 * @brief Apply Gram correction for CV: G_local = G - W_test * W_test^T
 *
 * Copies the full Gram G into G_local, then subtracts the outer products of
 * the held-out W columns (one per test row) to give the per-column corrected
 * Gram used in the MSE NNLS solve.
 *
 * @param G           Full Gram matrix W_T * W_T^T (k×k, symmetric)
 * @param W_T         Transposed W factor (k×m)
 * @param test_rows   Indices of held-out rows for this column
 * @param G_local     Output corrected Gram (k×k), modified in-place
 * @param W_test_buf  Workspace matrix (k × max_test), overwritten
 */
template<typename Scalar>
inline void apply_gram_correction(
    const DenseMatrix<Scalar>& G,
    const DenseMatrix<Scalar>& W_T,
    const std::vector<int>& test_rows,
    DenseMatrix<Scalar>& G_local,
    DenseMatrix<Scalar>& W_test_buf)
{
    G_local = G;
    if (!test_rows.empty()) {
        const int n_test = static_cast<int>(test_rows.size());
        for (int idx = 0; idx < n_test; ++idx)
            W_test_buf.col(idx) = W_T.col(test_rows[idx]);
        auto W_sub = W_test_buf.leftCols(n_test);
        G_local.template selfadjointView<Eigen::Lower>()
            .rankUpdate(W_sub, Scalar(-1));
        G_local.template triangularView<Eigen::Upper>() =
            G_local.transpose();
    }
}

// ========================================================================
// IRLS helpers for per-column/row CV NNLS
// ========================================================================

/**
 * @brief IRLS-wrapped per-column solve for H update in CV.
 *
 * Unified template replacing `irls_solve_col_cv_sparse` and
 * `irls_solve_col_cv_dense`.
 *
 * For sparse A, `DataAccessor::for_col` iterates nonzeros; for mask_zeros=false
 * a sequential InnerIterator scan enumerates all m entries including zeros.
 * For dense A, all m rows are scanned with an optional zero skip.
 */
template<typename Scalar, typename MatrixType>
void irls_solve_col_cv(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    int j,
    DenseVector<Scalar>& x,
    const std::vector<int>& test_rows,
    bool mask_zeros,
    const LossConfig<Scalar>& loss,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>& full_H,
    int m)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int k = static_cast<int>(W_T.rows());

    std::vector<bool> is_test(m, false);
    for (int r : test_rows) is_test[r] = true;

    struct TrainEntry { int row; Scalar value; };
    std::vector<TrainEntry> train_entries;

    if (mask_zeros) {
        // Only nonzeros contribute — DA::for_col skips structural zeros for sparse
        DA::for_col(A, j, [&](int row, Scalar val) {
            if (!is_test[row]) train_entries.push_back({row, val});
        });
    } else {
        // ALL m entries (including zeros) can be train
        if constexpr (DA::is_sparse) {
            typename MatrixType::InnerIterator it(A, j);
            for (int row = 0; row < m; ++row) {
                Scalar val = 0;
                if (it && it.row() == row) { val = it.value(); ++it; }
                if (!is_test[row]) train_entries.push_back({row, val});
            }
        } else {
            for (int row = 0; row < m; ++row) {
                if (!is_test[row]) train_entries.push_back({row, A(row, j)});
            }
        }
    }

    DenseMatrix<Scalar> G_w(k, k);
    DenseVector<Scalar> b_w(k);
    DenseVector<Scalar> x_old(k);

    for (int irls_iter = 0; irls_iter < config.irls_max_iter; ++irls_iter) {
        G_w.setZero();
        b_w.setZero();

        for (const auto& te : train_entries) {
            Scalar predicted = W_T.col(te.row).dot(x);
            Scalar residual = te.value - predicted;
            Scalar w = primitives::detail::compute_irls_weight(residual, predicted, loss);

            auto w_col = W_T.col(te.row);
            Scalar wv = w * te.value;
            for (int a = 0; a < k; ++a) {
                b_w(a) += w_col(a) * wv;
                for (int bb = a; bb < k; ++bb)
                    G_w(a, bb) += w * w_col(a) * w_col(bb);
            }
        }
        for (int a = 0; a < k; ++a)
            for (int bb = a + 1; bb < k; ++bb)
                G_w(bb, a) = G_w(a, bb);
        variant::apply_h_cv_features(G_w, full_H, config);
        G_w.diagonal().array() += static_cast<Scalar>(1e-15);

        x_old = x;
        if (config.solver_mode == 1) {
            primitives::detail::cholesky_clip_col(G_w, b_w.data(), x.data(), k,
                config.H.L1, Scalar(0), config.H.nonneg,
                config.cd_max_iter, config.cd_tol);
        } else {
            primitives::detail::cd_nnls_col_fixed(G_w, b_w.data(), x.data(), k,
                config.H.L1, Scalar(0), config.H.nonneg, config.cd_max_iter);
        }

        Scalar max_change = 0;
        for (int i = 0; i < k; ++i) {
            Scalar rel = std::abs(x(i) - x_old(i)) /
                         (std::abs(x_old(i)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < config.irls_tol) break;
    }
}

/**
 * @brief IRLS-wrapped per-row solve for W update in CV.
 *
 * Unified template replacing `irls_solve_row_cv_sparse` and
 * `irls_solve_row_cv_dense`.
 *
 * For sparse: pass A^T (n×m CSC); DA::for_col(A_T, i) iterates row i of A.
 * For dense:  pass A (m×n);       DA::for_row(A, i) iterates row i of A.
 */
template<typename Scalar, typename MatrixType>
void irls_solve_row_cv(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    int i,
    DenseVector<Scalar>& x,
    const std::vector<int>& test_cols,
    bool mask_zeros,
    const LossConfig<Scalar>& loss,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>& full_W_T,
    int n)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int k = static_cast<int>(H.rows());

    std::vector<bool> is_test(n, false);
    for (int c : test_cols) is_test[c] = true;

    struct TrainEntry { int col; Scalar value; };
    std::vector<TrainEntry> train_entries;

    if (mask_zeros) {
        if constexpr (DA::is_sparse) {
            DA::for_col(data, i, [&](int col, Scalar val) {
                if (!is_test[col]) train_entries.push_back({col, val});
            });
        } else {
            DA::for_row(data, i, [&](int col, Scalar val) {
                if (val != Scalar(0) && !is_test[col])
                    train_entries.push_back({col, val});
            });
        }
    } else {
        if constexpr (DA::is_sparse) {
            typename MatrixType::InnerIterator it(data, i);
            for (int col = 0; col < n; ++col) {
                Scalar val = 0;
                if (it && it.row() == col) { val = it.value(); ++it; }
                if (!is_test[col]) train_entries.push_back({col, val});
            }
        } else {
            DA::for_row(data, i, [&](int col, Scalar val) {
                if (!is_test[col]) train_entries.push_back({col, val});
            });
        }
    }

    DenseMatrix<Scalar> G_w(k, k);
    DenseVector<Scalar> b_w(k);
    DenseVector<Scalar> x_old(k);

    for (int irls_iter = 0; irls_iter < config.irls_max_iter; ++irls_iter) {
        G_w.setZero();
        b_w.setZero();

        for (const auto& te : train_entries) {
            Scalar predicted = H.col(te.col).dot(x);
            Scalar residual = te.value - predicted;
            Scalar w = primitives::detail::compute_irls_weight(residual, predicted, loss);

            auto h_col = H.col(te.col);
            Scalar wv = w * te.value;
            for (int a = 0; a < k; ++a) {
                b_w(a) += h_col(a) * wv;
                for (int bb = a; bb < k; ++bb)
                    G_w(a, bb) += w * h_col(a) * h_col(bb);
            }
        }
        for (int a = 0; a < k; ++a)
            for (int bb = a + 1; bb < k; ++bb)
                G_w(bb, a) = G_w(a, bb);
        variant::apply_w_cv_features(G_w, full_W_T, config);
        G_w.diagonal().array() += static_cast<Scalar>(1e-15);

        x_old = x;
        if (config.solver_mode == 1) {
            primitives::detail::cholesky_clip_col(G_w, b_w.data(), x.data(), k,
                config.W.L1, Scalar(0), config.W.nonneg,
                config.cd_max_iter, config.cd_tol);
        } else {
            primitives::detail::cd_nnls_col_fixed(G_w, b_w.data(), x.data(), k,
                config.W.L1, Scalar(0), config.W.nonneg, config.cd_max_iter);
        }

        Scalar max_change = 0;
        for (int idx = 0; idx < k; ++idx) {
            Scalar rel = std::abs(x(idx) - x_old(idx)) /
                         (std::abs(x_old(idx)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < config.irls_tol) break;
    }
}

/**
 * @brief Compute RHS for column j using only TRAIN entries (H update).
 *
 * Unified template replacing `compute_train_rhs_sparse` and
 * `compute_train_rhs_dense`.
 *
 * For sparse A: pass A directly; DA::for_col iterates nonzeros for mask_zeros=true.
 * For dense A:  pass A directly; all rows iterated.
 */
template<typename Scalar, typename MatrixType>
inline void compute_train_rhs(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    int j,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    DenseVector<Scalar>& b,
    std::vector<int>& test_rows,
    bool mask_zeros,
    int m)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    b.setZero();
    test_rows.clear();

    if (mask_zeros) {
        DA::for_col(A, j, [&](int i, Scalar val) {
            if (mask.is_holdout(i, j)) test_rows.push_back(i);
            else b.noalias() += val * W_T.col(i);
        });
    } else {
        if constexpr (DA::is_sparse) {
            typename MatrixType::InnerIterator it(A, j);
            for (int i = 0; i < m; ++i) {
                Scalar val = 0;
                if (it && it.row() == i) { val = it.value(); ++it; }
                if (mask.is_holdout(i, j)) {
                    test_rows.push_back(i);
                } else if (val != Scalar(0)) {
                    b.noalias() += val * W_T.col(i);
                }
            }
        } else {
            for (int i = 0; i < m; ++i) {
                Scalar val = A(i, j);
                if (mask.is_holdout(i, j)) test_rows.push_back(i);
                else b.noalias() += val * W_T.col(i);
            }
        }
    }
}

/**
 * @brief Compute RHS for row i using only TRAIN entries (W update).
 *
 * Unified template replacing `compute_train_rhs_W_sparse` and
 * `compute_train_rhs_W_dense`.
 *
 * For sparse: pass A^T (n×m CSC); column i of A_T = row i of A.
 * For dense:  pass A (m×n);       row i accessed via DA::for_row.
 */
template<typename Scalar, typename MatrixType>
inline void compute_train_rhs_W(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    int i,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    DenseVector<Scalar>& b,
    std::vector<int>& test_cols,
    bool mask_zeros,
    int n)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    b.setZero();
    test_cols.clear();

    if (mask_zeros) {
        if constexpr (DA::is_sparse) {
            DA::for_col(data, i, [&](int col, Scalar val) {
                if (mask.is_holdout(i, col)) test_cols.push_back(col);
                else b.noalias() += val * H.col(col);
            });
        } else {
            DA::for_row(data, i, [&](int col, Scalar val) {
                if (val == Scalar(0)) return;
                if (mask.is_holdout(i, col)) test_cols.push_back(col);
                else b.noalias() += val * H.col(col);
            });
        }
    } else {
        if constexpr (DA::is_sparse) {
            typename MatrixType::InnerIterator it(data, i);
            for (int col = 0; col < n; ++col) {
                Scalar val = 0;
                if (it && it.row() == col) { val = it.value(); ++it; }
                if (mask.is_holdout(i, col)) test_cols.push_back(col);
                else if (val != Scalar(0)) b.noalias() += val * H.col(col);
            }
        } else {
            DA::for_row(data, i, [&](int col, Scalar val) {
                if (mask.is_holdout(i, col)) test_cols.push_back(col);
                else b.noalias() += val * H.col(col);
            });
        }
    }
}

// ========================================================================
// User mask helpers for CV with user-supplied mask
// ========================================================================

/**
 * @brief O(log n) check: is entry (row, col) in the user mask?
 */
template<typename Scalar>
inline bool is_user_masked_h(int row, int col,
                             const SparseMatrix<Scalar>& mask) {
    const int start = mask.outerIndexPtr()[col];
    const int end   = mask.outerIndexPtr()[col + 1];
    return std::binary_search(mask.innerIndexPtr() + start,
                              mask.innerIndexPtr() + end, row);
}

/**
 * @brief Adjust b and build excluded_rows after compute_train_rhs for H-update
 *
 * After compute_train_rhs fills b and test_rows, this function:
 * 1. Subtracts user-masked entries' contributions from b (they were incorrectly included as train)
 * 2. Builds excluded_rows = test_rows ∪ masked_rows (for Gram correction)
 *
 * @param A           Data matrix
 * @param W_T         Factor W in transposed storage (k × m)
 * @param user_mask   User-supplied sparse mask (nonzero = masked)
 * @param j           Column index
 * @param b           RHS vector to adjust (in/out)
 * @param test_rows   Holdout rows (from compute_train_rhs)
 * @param excluded_rows Output: holdout ∪ masked rows (for Gram correction)
 */
template<typename Scalar, typename MatrixType>
inline void adjust_rhs_for_user_mask_h(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    const SparseMatrix<Scalar>& user_mask,
    int j,
    DenseVector<Scalar>& b,
    const std::vector<int>& test_rows,
    std::vector<int>& excluded_rows)
{
    excluded_rows = test_rows;
    const int* inner = user_mask.innerIndexPtr();
    const int start = user_mask.outerIndexPtr()[j];
    const int end   = user_mask.outerIndexPtr()[j + 1];

    for (int p = start; p < end; ++p) {
        int row = inner[p];
        // If already in test_rows, it was already excluded from b
        if (std::binary_search(test_rows.begin(), test_rows.end(), row))
            continue;

        // Subtract from b (was wrongly included as train entry)
        Scalar a_val;
        if constexpr (is_sparse_v<MatrixType>) {
            a_val = A.coeff(row, j);
        } else {
            a_val = A(row, j);
        }
        if (a_val != Scalar(0))
            b.noalias() -= a_val * W_T.col(row);

        excluded_rows.push_back(row);
    }
    std::sort(excluded_rows.begin(), excluded_rows.end());
}

/**
 * @brief Adjust b and build excluded_cols for W-update with user mask
 *
 * Same as adjust_rhs_for_user_mask_h but for the W-update which uses
 * A^T (or rows of A), H instead of W_T, and mask^T instead of mask.
 */
template<typename Scalar, typename MatrixType>
inline void adjust_rhs_for_user_mask_w(
    const MatrixType& A_or_At,
    const DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask_T,
    int i,
    DenseVector<Scalar>& b,
    const std::vector<int>& test_cols,
    std::vector<int>& excluded_cols)
{
    excluded_cols = test_cols;
    const int* inner = mask_T.innerIndexPtr();
    const int start = mask_T.outerIndexPtr()[i];
    const int end   = mask_T.outerIndexPtr()[i + 1];

    for (int p = start; p < end; ++p) {
        int col = inner[p];
        if (std::binary_search(test_cols.begin(), test_cols.end(), col))
            continue;

        // For sparse A^T: A_or_At.coeff(col, i) = A_T(col, i) = A(i, col)
        // For dense A:    A_or_At(i, col) = A(i, col)
        Scalar a_val;
        if constexpr (is_sparse_v<MatrixType>) {
            a_val = A_or_At.coeff(col, i);
        } else {
            a_val = A_or_At(i, col);
        }
        if (a_val != Scalar(0))
            b.noalias() -= a_val * H.col(col);

        excluded_cols.push_back(col);
    }
    std::sort(excluded_cols.begin(), excluded_cols.end());
}

}  // namespace detail
}  // namespace nmf
}  // namespace FactorNet
