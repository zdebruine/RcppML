// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file test_entries.hpp
 * @brief Holdout test entries for speckled cross-validation in SVD
 *
 * Shared infrastructure for test loss computation used by both
 * deflation and krylov SVD algorithms.
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/svd/spmv.hpp>  // is_sparse_matrix

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace svd {

/**
 * @brief Store and evaluate test entries for speckled CV
 *
 * For each holdout entry (i,j), stores the residual A(i,j) - sum_k sigma_k*u_k(i)*v_k(j).
 * After each factor, update residuals and compute MSE.
 */
template<typename Scalar>
struct TestEntries {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<Scalar> residuals;
    int count = 0;

    Scalar compute_mse(int threads = 0) const {
        if (count == 0) return 0;
        Scalar sum_sq = 0;
#ifdef _OPENMP
        const int nthreads = threads > 0 ? threads : omp_get_max_threads();
        #pragma omp parallel for reduction(+:sum_sq) num_threads(nthreads) schedule(static)
#endif
        for (int i = 0; i < count; ++i) {
            sum_sq += residuals[i] * residuals[i];
        }
        return sum_sq / static_cast<Scalar>(count);
    }

    void update(const DenseVector<Scalar>& u, Scalar sigma,
                const DenseVector<Scalar>& v, int threads = 0) {
#ifdef _OPENMP
        const int nthreads = threads > 0 ? threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (int i = 0; i < count; ++i) {
            residuals[i] -= sigma * u(rows[i]) * v(cols[i]);
        }
    }
};

/**
 * @brief Initialize test entries from sparse matrix using LazySpeckledMask
 */
template<typename Scalar, typename SparseMat>
TestEntries<Scalar> init_test_entries_sparse(
    const SparseMat& A,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    const int n = static_cast<int>(A.cols());

    te.rows.reserve(static_cast<size_t>(A.nonZeros() * 0.15));
    te.cols.reserve(static_cast<size_t>(A.nonZeros() * 0.15));
    te.residuals.reserve(static_cast<size_t>(A.nonZeros() * 0.15));

    if (mask.mask_zeros()) {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                if (mask.is_holdout(it.row(), j)) {
                    Scalar val = static_cast<Scalar>(it.value());
                    if (row_means) val -= (*row_means)(it.row());
                    te.rows.push_back(it.row());
                    te.cols.push_back(j);
                    te.residuals.push_back(val);
                }
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                if (mask.is_holdout(it.row(), j)) {
                    Scalar val = static_cast<Scalar>(it.value());
                    if (row_means) val -= (*row_means)(it.row());
                    te.rows.push_back(it.row());
                    te.cols.push_back(j);
                    te.residuals.push_back(val);
                }
            }
        }
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

/**
 * @brief Initialize test entries from dense matrix using LazySpeckledMask
 */
template<typename Scalar>
TestEntries<Scalar> init_test_entries_dense(
    const DenseMatrix<Scalar>& A,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    te.rows.reserve(static_cast<size_t>(m) * n / 20);
    te.cols.reserve(static_cast<size_t>(m) * n / 20);
    te.residuals.reserve(static_cast<size_t>(m) * n / 20);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (mask.is_holdout(i, j)) {
                Scalar val = A(i, j);
                if (row_means) val -= (*row_means)(i);
                te.rows.push_back(i);
                te.cols.push_back(j);
                te.residuals.push_back(val);
            }
        }
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

/**
 * @brief Initialize test entries from streaming cached panels
 *
 * Iterates over all cached Eigen sparse panels in a StreamingContext
 * to collect holdout entries via LazySpeckledMask.
 * Only supports mask_zeros=true mode (sparse data — only nonzeros are candidates).
 *
 * @tparam Scalar  float or double
 * @tparam CtxType StreamingContext<Scalar> (templated to avoid circular include)
 */
template<typename Scalar, typename CtxType>
TestEntries<Scalar> init_test_entries_streaming(
    const CtxType& ctx,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    ctx.ensure_cache();

    // Estimate reservation: ~test_fraction of total nnz
    int64_t total_nnz = 0;
    for (uint32_t c = 0; c < ctx.num_chunks; ++c)
        total_nnz += ctx.cached_panels[c].nonZeros();
    te.rows.reserve(static_cast<size_t>(total_nnz * 0.15));
    te.cols.reserve(static_cast<size_t>(total_nnz * 0.15));
    te.residuals.reserve(static_cast<size_t>(total_nnz * 0.15));

    uint32_t col_offset = 0;
    for (uint32_t c = 0; c < ctx.num_chunks; ++c) {
        const auto& panel = ctx.cached_panels[c];
        uint32_t nc = ctx.chunks[c].num_cols;
        for (int lj = 0; lj < static_cast<int>(nc); ++lj) {
            int j = static_cast<int>(col_offset) + lj;  // global column
            for (typename std::remove_const<
                    typename std::remove_reference<decltype(panel)>::type
                 >::type::InnerIterator it(panel, lj); it; ++it)
            {
                int i = static_cast<int>(it.row());
                if (mask.is_holdout(i, j)) {
                    Scalar val = static_cast<Scalar>(it.value());
                    if (row_means) val -= (*row_means)(i);
                    te.rows.push_back(i);
                    te.cols.push_back(j);
                    te.residuals.push_back(val);
                }
            }
        }
        col_offset += nc;
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

/**
 * @brief Initialize test entries from raw CSC arrays
 *
 * Used when the matrix is available as raw CSC pointers (e.g., from GPU path)
 * without needing to construct an Eigen sparse matrix.
 * Only supports mask_zeros=true (only nonzeros are holdout candidates).
 */
template<typename Scalar>
TestEntries<Scalar> init_test_entries_csc(
    const int* col_ptr, const int* row_idx, const Scalar* values,
    int m, int n, int64_t nnz,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    te.rows.reserve(static_cast<size_t>(nnz * 0.15));
    te.cols.reserve(static_cast<size_t>(nnz * 0.15));
    te.residuals.reserve(static_cast<size_t>(nnz * 0.15));

    for (int j = 0; j < n; ++j) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int i = row_idx[p];
            if (mask.is_holdout(i, j)) {
                Scalar val = values[p];
                if (row_means) val -= (*row_means)(i);
                te.rows.push_back(i);
                te.cols.push_back(j);
                te.residuals.push_back(val);
            }
        }
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

}  // namespace svd
}  // namespace FactorNet

// ============================================================================
// Training matrix construction — zero out holdout entries
// ============================================================================
namespace FactorNet {
namespace svd {

/**
 * @brief Create a training copy of a sparse matrix with holdout entries zeroed
 *
 * The sparsity pattern is preserved (structural zeros remain), but their
 * values are set to 0 so they contribute nothing to SpMV during training.
 * This ensures the model never sees the held-out entries.
 *
 * The output uses the input matrix's own scalar type (e.g. double) even when
 * the working Scalar is float, because Eigen forbids mixed-type sparse
 * construction.  The SpMV/SpMM helpers use static_cast internally to handle
 * the double→float conversion at element level.
 *
 * @tparam MaskScalar  Scalar type used for the mask/config (float or double)
 * @param A          Original sparse matrix (CSC); may be Map<const ...>
 * @param mask       Speckled mask identifying holdout entries
 * @param n_zeroed   [out] Number of entries zeroed (for diagnostics)
 * @return Deep copy of A with holdout values set to 0 (same scalar as A)
 */
template<typename MaskScalar, typename InputMat>
Eigen::SparseMatrix<typename InputMat::Scalar, Eigen::ColMajor, typename InputMat::StorageIndex>
create_training_matrix_sparse(
    const InputMat& A,
    const FactorNet::nmf::LazySpeckledMask<MaskScalar>& mask,
    int& n_zeroed)
{
    // Always return an owned SparseMatrix matching A's scalar type
    using MatScalar = typename InputMat::Scalar;
    using OwnedSp = Eigen::SparseMatrix<MatScalar, Eigen::ColMajor,
                                         typename InputMat::StorageIndex>;
    OwnedSp A_train(A);  // deep copy from Map or owned
    n_zeroed = 0;

    const int n = static_cast<int>(A_train.cols());
    for (int j = 0; j < n; ++j) {
        for (typename OwnedSp::InnerIterator it(A_train, j); it; ++it) {
            if (mask.is_holdout(it.row(), j)) {
                it.valueRef() = 0;
                ++n_zeroed;
            }
        }
    }

    return A_train;
}

/**
 * @brief Create a training copy of a dense matrix with holdout entries zeroed
 */
template<typename Scalar>
DenseMatrix<Scalar> create_training_matrix_dense(
    const DenseMatrix<Scalar>& A,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    int& n_zeroed)
{
    DenseMatrix<Scalar> A_train(A);
    n_zeroed = 0;

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (mask.is_holdout(i, j)) {
                A_train(i, j) = 0;
                ++n_zeroed;
            }
        }
    }

    return A_train;
}

}  // namespace svd
}  // namespace FactorNet

