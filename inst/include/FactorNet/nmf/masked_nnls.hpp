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

#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace nmf {

namespace mask_detail {

/**
 * @brief Dispatch per-column NNLS solve based on solver_mode.
 * G_local is the per-column corrected Gram matrix. L1/L2 already applied.
 */
template<typename Scalar>
inline void masked_solve_col(
    const DenseMatrix<Scalar>& G_local,
    Scalar* b,
    Scalar* x,
    int k,
    bool nonneg,
    int cd_max_iter,
    Scalar cd_tol,
    int solver_mode)
{
    if (solver_mode == 1) {
        FactorNet::primitives::detail::cholesky_clip_col(
            G_local, b, x, k,
            static_cast<Scalar>(0), static_cast<Scalar>(0),
            nonneg, 0, static_cast<Scalar>(0));
    } else {
        FactorNet::primitives::detail::cd_nnls_col_fixed(
            G_local, b, x, k,
            static_cast<Scalar>(0), static_cast<Scalar>(0),
            nonneg, cd_max_iter, static_cast<Scalar>(0), cd_tol);
    }
}

/**
 * @brief Per-column masked NNLS for H-update (sparse A)
 *
 * For each column j, identifies masked rows from the mask matrix,
 * builds delta_G = Σ_{i ∈ masked(j)} w_i * w_i^T, and solves:
 *   (G_full - delta_G) * h_j = b_j
 * where b_j only includes non-masked entries.
 *
 * @param A       Sparse data matrix (m × n)
 * @param W_T     Factor matrix (k × m), transposed W
 * @param G_full  Pre-computed Gram matrix W_T * W_T^T (k × k)
 * @param H       Solution matrix (k × n), updated in-place
 * @param mask    Sparse mask matrix (m × n), nonzeros = masked
 * @param config  NMF configuration (L1, L2, nonneg, cd params)
 * @param threads Number of OpenMP threads
 * @param warm_start Whether to warm-start from current H
 */
/**
 * @brief Per-column masked NNLS for H-update (sparse or dense A).
 *
 * For each column j, identifies masked rows, builds delta_G correction, and solves:
 *   (G_full - delta_G) * h_j = b_j  with b_j summing only non-masked entries.
 *
 * DataAccessor<MatrixType> provides the column iteration strategy:
 *   - Sparse: inner iterator over non-zeros only.
 *   - Dense:  fast BLAS mat-vec for unmasked; element-wise for masked columns.
 */
template<typename Scalar, typename MatrixType>
void masked_nnls_h(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_full,
    DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const ::FactorNet::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int k = static_cast<int>(W_T.rows());
    const int n = static_cast<int>(A.cols());
    const int m = static_cast<int>(A.rows());

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < n; ++j) {
        // Collect masked rows for column j
        std::vector<int> masked_rows;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit)
            if (mit.value() != 0)
                masked_rows.push_back(static_cast<int>(mit.row()));

        // Build b: RHS for this column, skipping masked rows
        DenseVector<Scalar> b(k);
        b.setZero();
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

        // Apply delta_G correction (no-op when masked_rows is empty)
        DenseMatrix<Scalar> G_local = G_full;
        for (int r : masked_rows)
            G_local.noalias() -= W_T.col(r) * W_T.col(r).transpose();

        // Apply L1/L2 regularization
        for (int i = 0; i < k; ++i) {
            b(i)           -= config.H.L1;
            G_local(i, i)  += config.H.L2;
        }

        DenseVector<Scalar> x = warm_start
            ? DenseVector<Scalar>(H.col(j))
            : DenseVector<Scalar>::Zero(k);
        masked_solve_col(G_local, b.data(), x.data(), k,
            config.H.nonneg, config.cd_max_iter, config.cd_tol,
            config.solver_mode);
        H.col(j) = x;
    }
}

/**
 * @brief Per-column masked NNLS for W-update (sparse or dense A).
 *
 * Unified template replacing the separate sparse/dense overloads.
 *
 * For sparse: pass A^T (n×m CSC); DataAccessor::for_col iterates nonzeros
 * in column j of A^T, which are the nonzero entries in row j of A.
 * For dense:  pass A (m×n);       DataAccessor::for_row iterates all columns
 * of row j, with a BLAS fast-path when no masking is needed.
 *
 * @tparam Scalar     float or double
 * @tparam MatrixType SparseMatrix<Scalar> (pass A^T) or DenseMatrix<Scalar> (pass A)
 * @param data     A^T for sparse (n × m), or A for dense (m × n)
 * @param H        Factor matrix (k × n)
 * @param G_full   Pre-computed Gram matrix H·H^T (k × k)
 * @param W_T      Solution matrix (k × m), updated in-place
 * @param mask_T   Transposed mask (n × m): col j = masked columns for row j of A
 * @param config   NMF configuration
 * @param threads  Number of threads
 * @param warm_start Whether to warm-start the NNLS solver
 */
template<typename Scalar, typename MatrixType>
void masked_nnls_w(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    const DenseMatrix<Scalar>& G_full,
    DenseMatrix<Scalar>& W_T,
    const SparseMatrix<Scalar>& mask_T,
    const ::FactorNet::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    const int k = static_cast<int>(H.rows());
    // m = number of rows of A: cols of A^T (sparse) or rows of A (dense)
    int m;
    if constexpr (DA::is_sparse)
        m = static_cast<int>(data.cols());
    else
        m = static_cast<int>(data.rows());

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < m; ++j) {
        // Masked columns for row j of A = nonzeros in column j of mask_T
        std::vector<int> masked_cols;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask_T, j); mit; ++mit)
            if (mit.value() != 0)
                masked_cols.push_back(static_cast<int>(mit.row()));

        DenseVector<Scalar> b(k);
        b.setZero();
        if (masked_cols.empty()) {
            if constexpr (DA::is_sparse) {
                DA::fast_rhs_col(data, H, j, b);           // b += H * At.col(j)
            } else {
                b.noalias() = H * data.row(j).transpose(); // BLAS fast path
            }
        } else {
            std::vector<bool> is_masked(static_cast<int>(H.cols()), false);
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

        DenseMatrix<Scalar> G_local = G_full;
        for (int c : masked_cols)
            G_local.noalias() -= H.col(c) * H.col(c).transpose();

        for (int i = 0; i < k; ++i) {
            b(i) -= config.W.L1;
            G_local(i, i) += config.W.L2;
        }

        DenseVector<Scalar> x = warm_start ? DenseVector<Scalar>(W_T.col(j)) : DenseVector<Scalar>::Zero(k);
        masked_solve_col(G_local, b.data(), x.data(), k,
            config.W.nonneg, config.cd_max_iter, config.cd_tol,
            config.solver_mode);
        W_T.col(j) = x;
    }
}

/**
 * @brief Compute loss excluding masked entries (sparse or dense A).
 *
 * DataAccessor::for_col iterates non-zeros for sparse (skipping structural zeros)
 * and all entries for dense, matching the original per-type semantics.
 */
template<typename Scalar, typename MatrixType>
Scalar masked_loss(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_Td,
    const DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const LossConfig<Scalar>& loss_config,
    int threads = 1)
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
