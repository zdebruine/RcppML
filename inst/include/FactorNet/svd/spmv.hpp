// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file spmv.hpp
 * @brief Sparse/dense matrix–vector and matrix–matrix products for SVD
 *
 * Provides:
 *   SpmvWorkspace      — pre-allocated thread-local buffers (reused across calls)
 *   spmv_transpose     — A'u  (n × 1), with optional PCA centering
 *   spmv_forward       — Av   (m × 1), with optional PCA centering
 *   spmm_forward       — A*X  (m × l), batched SpMM with cache reuse
 *   spmm_transpose     — A'*X (n × l), batched SpMM
 *   compute_row_means  — row means (m × 1)
 *
 * All SpMV functions accept Eigen expressions (column refs, blocks) via
 * DenseVector overloads. SpMV forward supports pre-allocated workspace.
 */

#pragma once

#include <FactorNet/core/types.hpp>

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace svd {

// ============================================================================
// Pre-allocated workspace for thread-local SpMV forward buffers
// ============================================================================

/**
 * @brief Workspace holding pre-allocated thread-local buffers for spmv_forward.
 *
 * The forward SpMV (y = A*v for CSC sparse A) requires thread-local accumulation
 * buffers of size m. Allocating these per-call is expensive (malloc + zero-fill).
 * This workspace allocates them once and reuses across all SpMV calls.
 */
template<typename Scalar>
struct SpmvWorkspace {
    std::vector<DenseVector<Scalar>> locals;
    int nthreads = 0;
    int m = 0;

    SpmvWorkspace() = default;

    void init(int rows, int threads) {
        if (rows == m && threads == nthreads && !locals.empty()) return;
        m = rows;
        nthreads = threads;
        locals.resize(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            locals[t].resize(m);
        }
    }

    void zero() {
        for (int t = 0; t < nthreads; ++t) {
            locals[t].setZero();
        }
    }
};

/**
 * @brief Workspace holding pre-allocated thread-local buffers for spmm_forward.
 *
 * The forward SpMM (Y = A*X for CSC sparse A with X being n×l) requires
 * thread-local accumulation buffers of size m×l. Without this workspace,
 * each call allocates nt dense m×l matrices (e.g., 8 × 30000 × 60 × 8 bytes
 * = ~115 MB for typical scRNA-seq). This workspace allocates once and reuses.
 */
template<typename Scalar>
struct SpmmWorkspace {
    std::vector<DenseMatrix<Scalar>> locals;
    int nthreads = 0;
    int m = 0;
    int l = 0;

    SpmmWorkspace() = default;

    void init(int rows, int cols, int threads) {
        if (rows == m && cols >= l && threads == nthreads && !locals.empty()) return;
        m = rows;
        l = cols;
        nthreads = threads;
        locals.resize(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            locals[t].resize(m, l);
        }
    }

    void zero() {
        for (int t = 0; t < nthreads; ++t) {
            locals[t].setZero();
        }
    }
};

// ============================================================================
// Row mean computation
// ============================================================================

template<typename Scalar, typename SparseMat>
DenseVector<Scalar> compute_row_means_sparse(const SparseMat& A, int threads = 0) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    DenseVector<Scalar> means = DenseVector<Scalar>::Zero(m);

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    if (nthreads > 1 && n > 64) {
        const int nt = std::min(nthreads, n);
        std::vector<DenseVector<Scalar>> locals(nt, DenseVector<Scalar>::Zero(m));
        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int j = 0; j < n; ++j) {
                for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                    locals[tid](it.row()) += static_cast<Scalar>(it.value());
                }
            }
        }
        for (int t = 0; t < nt; ++t) means += locals[t];
    } else
#endif
    {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                means(it.row()) += static_cast<Scalar>(it.value());
            }
        }
    }
    if (n > 0) means /= static_cast<Scalar>(n);
    return means;
}

template<typename Scalar>
DenseVector<Scalar> compute_row_means_dense(const DenseMatrix<Scalar>& A) {
    return A.rowwise().mean();
}

// ============================================================================
// Row standard deviation computation (for correlation PCA)
// ============================================================================

/**
 * @brief Compute row standard deviations (population SD) from a sparse matrix
 *
 * Uses the identity: Var(x) = E[x^2] - (E[x])^2
 * Requires pre-computed row means.
 */
template<typename Scalar, typename SparseMat>
DenseVector<Scalar> compute_row_sds_sparse(const SparseMat& A,
                                           const DenseVector<Scalar>& row_means,
                                           int threads = 0) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    DenseVector<Scalar> sum_sq = DenseVector<Scalar>::Zero(m);

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    if (nthreads > 1 && n > 64) {
        const int nt = std::min(nthreads, n);
        std::vector<DenseVector<Scalar>> locals(nt, DenseVector<Scalar>::Zero(m));
        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int j = 0; j < n; ++j) {
                for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                    Scalar val = static_cast<Scalar>(it.value());
                    locals[tid](it.row()) += val * val;
                }
            }
        }
        for (int t = 0; t < nt; ++t) sum_sq += locals[t];
    } else
#endif
    {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                Scalar val = static_cast<Scalar>(it.value());
                sum_sq(it.row()) += val * val;
            }
        }
    }

    // Var = E[x^2] - (E[x])^2 = sum_sq/n - mean^2
    // SD = sqrt(Var), clamped away from zero for numerical safety
    const Scalar inv_n = n > 0 ? static_cast<Scalar>(1) / static_cast<Scalar>(n)
                               : static_cast<Scalar>(0);
    const Scalar eps = std::numeric_limits<Scalar>::epsilon() * 100;
    DenseVector<Scalar> sds(m);
    for (int i = 0; i < m; ++i) {
        Scalar var = sum_sq(i) * inv_n - row_means(i) * row_means(i);
        if (var < eps) var = eps;  // prevent division by zero
        sds(i) = std::sqrt(var);
    }
    return sds;
}

template<typename Scalar>
DenseVector<Scalar> compute_row_sds_dense(const DenseMatrix<Scalar>& A,
                                          const DenseVector<Scalar>& row_means) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const Scalar inv_n = n > 0 ? static_cast<Scalar>(1) / static_cast<Scalar>(n)
                               : static_cast<Scalar>(0);
    const Scalar eps = std::numeric_limits<Scalar>::epsilon() * 100;
    DenseVector<Scalar> sds(m);
    for (int i = 0; i < m; ++i) {
        Scalar sum_sq = A.row(i).squaredNorm();
        Scalar var = sum_sq * inv_n - row_means(i) * row_means(i);
        if (var < eps) var = eps;
        sds(i) = std::sqrt(var);
    }
    return sds;
}

// ============================================================================
// SpMV transpose: result = A' * u  (n × 1)
//   With centering: result = (A - mu*1')' * u = A'u - mu'u * 1
//   With scaling:   result = (A - mu*1')' * D^{-1} * u
//                   = A'(D^{-1}u) - mu'(D^{-1}u) * 1
//   where D = diag(row_sds), D^{-1} = diag(1/row_sds) = diag(row_inv_sds)
// ============================================================================

template<typename Scalar, typename SparseMat>
void spmv_transpose_sparse(const SparseMat& A,
                           const DenseVector<Scalar>& u,
                           DenseVector<Scalar>& result,
                           const DenseVector<Scalar>* row_means = nullptr,
                           int threads = 0,
                           const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    const int n = static_cast<int>(A.cols());
    result.resize(n);

    // If scaling, pre-compute u_scaled = D^{-1} * u
    const DenseVector<Scalar>* u_eff = &u;
    DenseVector<Scalar> u_scaled;
    if (row_inv_sds) {
        u_scaled = u.cwiseProduct(*row_inv_sds);
        u_eff = &u_scaled;
    }

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
#endif
    for (int j = 0; j < n; ++j) {
        Scalar dot = 0;
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            dot += static_cast<Scalar>(it.value()) * (*u_eff)(it.row());
        }
        result(j) = dot;
    }

    if (row_means) {
        Scalar correction = row_means->dot(*u_eff);
        result.array() -= correction;
    }
}

template<typename DenseMatrixType, typename Scalar>
void spmv_transpose_dense(const DenseMatrixType& A,
                          const DenseVector<Scalar>& u,
                          DenseVector<Scalar>& result,
                          const DenseVector<Scalar>* row_means = nullptr,
                          const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    if (row_inv_sds) {
        DenseVector<Scalar> u_scaled = u.cwiseProduct(*row_inv_sds);
        result.noalias() = A.transpose() * u_scaled;
        if (row_means) {
            Scalar correction = row_means->dot(u_scaled);
            result.array() -= correction;
        }
    } else {
        result.noalias() = A.transpose() * u;
        if (row_means) {
            Scalar correction = row_means->dot(u);
            result.array() -= correction;
        }
    }
}

// ============================================================================
// SpMV forward: result = A * v  (m × 1)
//   With centering: result = (A - mu*1') * v = Av - mu*sum(v)
//   With scaling:   result = D^{-1} * (Av - mu*sum(v))
// ============================================================================

/**
 * @brief Sparse Av with optional pre-allocated workspace.
 *
 * If ws is non-null and initialized, reuses thread-local buffers instead of
 * allocating per-call. This eliminates ~880KB malloc+zero per call (8 threads).
 */
template<typename Scalar, typename SparseMat>
void spmv_forward_sparse(const SparseMat& A,
                         const DenseVector<Scalar>& v,
                         DenseVector<Scalar>& result,
                         const DenseVector<Scalar>* row_means = nullptr,
                         int threads = 0,
                         SpmvWorkspace<Scalar>* ws = nullptr,
                         const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    result.resize(m);
    result.setZero();

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    if (nthreads > 1) {
        const int nt = std::min(nthreads, n);

        // Use workspace if available, otherwise allocate
        std::vector<DenseVector<Scalar>> tmp_locals;
        std::vector<DenseVector<Scalar>>* locals_ptr;

        if (ws && ws->nthreads >= nt && ws->m == m) {
            ws->zero();
            locals_ptr = &ws->locals;
        } else {
            tmp_locals.resize(nt, DenseVector<Scalar>::Zero(m));
            locals_ptr = &tmp_locals;
        }

        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            auto& local = (*locals_ptr)[tid];
            #pragma omp for schedule(static)
            for (int j = 0; j < n; ++j) {
                Scalar vj = v(j);
                if (vj == 0) continue;
                for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                    local(it.row()) += static_cast<Scalar>(it.value()) * vj;
                }
            }
        }
        for (int t = 0; t < nt; ++t) result += (*locals_ptr)[t];
    } else
#endif
    {
        for (int j = 0; j < n; ++j) {
            Scalar vj = v(j);
            if (vj == 0) continue;
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                result(it.row()) += static_cast<Scalar>(it.value()) * vj;
            }
        }
    }

    if (row_means) {
        Scalar sum_v = v.sum();
        result -= (*row_means) * sum_v;
    }
    if (row_inv_sds) {
        result = result.cwiseProduct(*row_inv_sds);
    }
}

template<typename DenseMatrixType, typename Scalar>
void spmv_forward_dense(const DenseMatrixType& A,
                        const DenseVector<Scalar>& v,
                        DenseVector<Scalar>& result,
                        const DenseVector<Scalar>* row_means = nullptr,
                        const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    result.noalias() = A * v;

    if (row_means) {
        Scalar sum_v = v.sum();
        result -= (*row_means) * sum_v;
    }
    if (row_inv_sds) {
        result = result.cwiseProduct(*row_inv_sds);
    }
}

// ============================================================================
// SpMM forward: Y = A * X  (m × l) — cache-efficient batched multiply
// ============================================================================

/**
 * @brief Sparse A*X where X is (n × l) and Y is (m × l).
 *
 * Each nonzero of A is read ONCE and scattered into l columns of Y,
 * amortizing the CSC indirect indexing cost over the block width l.
 *
 * If ws is non-null and initialized, reuses thread-local buffers instead of
 * allocating per-call. This eliminates ~115MB malloc+zero per call (8 threads, m=30k, l=60).
 */
template<typename Scalar, typename SparseMat>
void spmm_forward_sparse(const SparseMat& A,
                         const DenseMatrix<Scalar>& X,
                         DenseMatrix<Scalar>& Y,
                         const DenseVector<Scalar>* row_means = nullptr,
                         int threads = 0,
                         SpmmWorkspace<Scalar>* ws = nullptr,
                         const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int l = static_cast<int>(X.cols());
    Y.resize(m, l);
    Y.setZero();

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    if (nthreads > 1) {
        const int nt = std::min(nthreads, n);

        // Use workspace if available, otherwise allocate
        std::vector<DenseMatrix<Scalar>> tmp_locals;
        std::vector<DenseMatrix<Scalar>>* locals_ptr;

        if (ws && ws->nthreads >= nt && ws->m == m && ws->l >= l) {
            ws->zero();
            locals_ptr = &ws->locals;
        } else {
            tmp_locals.resize(nt, DenseMatrix<Scalar>::Zero(m, l));
            locals_ptr = &tmp_locals;
        }

        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            auto& local = (*locals_ptr)[tid];
            #pragma omp for schedule(static)
            for (int j = 0; j < n; ++j) {
                for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                    const Scalar aij = static_cast<Scalar>(it.value());
                    const int row = it.row();
                    for (int c = 0; c < l; ++c) {
                        local(row, c) += aij * X(j, c);
                    }
                }
            }
        }
        for (int t = 0; t < nt; ++t) Y += (*locals_ptr)[t];
    } else
#endif
    {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                const Scalar aij = static_cast<Scalar>(it.value());
                const int row = it.row();
                for (int c = 0; c < l; ++c) {
                    Y(row, c) += aij * X(j, c);
                }
            }
        }
    }

    if (row_means) {
        DenseVector<Scalar> col_sums = X.colwise().sum();
        Y.noalias() -= (*row_means) * col_sums.transpose();
    }
    if (row_inv_sds) {
        Y.array().colwise() *= row_inv_sds->array();
    }
}

template<typename DenseMatrixType, typename Scalar>
void spmm_forward_dense(const DenseMatrixType& A,
                        const DenseMatrix<Scalar>& X,
                        DenseMatrix<Scalar>& Y,
                        const DenseVector<Scalar>* row_means = nullptr,
                        const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    Y.noalias() = A * X;
    if (row_means) {
        DenseVector<Scalar> col_sums = X.colwise().sum();
        Y.noalias() -= (*row_means) * col_sums.transpose();
    }
    if (row_inv_sds) {
        Y.array().colwise() *= row_inv_sds->array();
    }
}

// ============================================================================
// SpMM transpose: Z = A' * X  (n × l)
// ============================================================================

template<typename Scalar, typename SparseMat>
void spmm_transpose_sparse(const SparseMat& A,
                           const DenseMatrix<Scalar>& X,
                           DenseMatrix<Scalar>& Z,
                           const DenseVector<Scalar>* row_means = nullptr,
                           int threads = 0,
                           const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    const int n = static_cast<int>(A.cols());
    const int l = static_cast<int>(X.cols());
    Z.resize(n, l);

    // Effective X: if scaling, pre-multiply X by D^{-1}
    // Ã^T X = (A - μ1^T)^T D^{-1} X = A^T(D^{-1}X) - 1·μ^T(D^{-1}X)
    DenseMatrix<Scalar> X_scaled_storage;
    const DenseMatrix<Scalar>* X_eff = &X;
    if (row_inv_sds) {
        X_scaled_storage = X.array().colwise() * row_inv_sds->array();
        X_eff = &X_scaled_storage;
    }

#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
#endif
    for (int j = 0; j < n; ++j) {
        for (int c = 0; c < l; ++c) {
            Scalar dot = 0;
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                dot += static_cast<Scalar>(it.value()) * (*X_eff)(it.row(), c);
            }
            Z(j, c) = dot;
        }
    }

    if (row_means) {
        DenseMatrix<Scalar> correction = row_means->transpose() * (*X_eff);  // 1 × l
        Z.rowwise() -= correction.row(0);
    }
}

template<typename DenseMatrixType, typename Scalar>
void spmm_transpose_dense(const DenseMatrixType& A,
                          const DenseMatrix<Scalar>& X,
                          DenseMatrix<Scalar>& Z,
                          const DenseVector<Scalar>* row_means = nullptr,
                          const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    if (row_inv_sds) {
        DenseMatrix<Scalar> X_scaled = X.array().colwise() * row_inv_sds->array();
        Z.noalias() = A.transpose() * X_scaled;
        if (row_means) {
            DenseMatrix<Scalar> correction = row_means->transpose() * X_scaled;
            Z.rowwise() -= correction.row(0);
        }
    } else {
        Z.noalias() = A.transpose() * X;
        if (row_means) {
            DenseMatrix<Scalar> correction = row_means->transpose() * X;
            Z.rowwise() -= correction.row(0);
        }
    }
}

// ============================================================================
// Type traits
// ============================================================================

template<typename T>
struct is_sparse_matrix : std::false_type {};

template<typename Scalar, int Options, typename StorageIndex>
struct is_sparse_matrix<Eigen::SparseMatrix<Scalar, Options, StorageIndex>> : std::true_type {};

template<typename Scalar>
struct is_sparse_matrix<Eigen::Map<const Eigen::SparseMatrix<Scalar>>> : std::true_type {};

// Match Map with explicit SparseMatrix template parameters
template<typename Scalar, int Options, typename StorageIndex>
struct is_sparse_matrix<Eigen::Map<const Eigen::SparseMatrix<Scalar, Options, StorageIndex>>> : std::true_type {};

// ============================================================================
// Type-dispatched wrappers — SpMV
// ============================================================================

template<typename MatrixType, typename Scalar>
void spmv_transpose(const MatrixType& A,
                    const DenseVector<Scalar>& u,
                    DenseVector<Scalar>& result,
                    const DenseVector<Scalar>* row_means = nullptr,
                    int threads = 0,
                    const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        spmv_transpose_sparse(A, u, result, row_means, threads, row_inv_sds);
    } else {
        spmv_transpose_dense(A, u, result, row_means, row_inv_sds);
    }
}

template<typename MatrixType, typename Scalar>
void spmv_forward(const MatrixType& A,
                  const DenseVector<Scalar>& v,
                  DenseVector<Scalar>& result,
                  const DenseVector<Scalar>* row_means = nullptr,
                  int threads = 0,
                  SpmvWorkspace<Scalar>* ws = nullptr,
                  const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        spmv_forward_sparse(A, v, result, row_means, threads, ws, row_inv_sds);
    } else {
        spmv_forward_dense(A, v, result, row_means, row_inv_sds);
    }
}

// ============================================================================
// Type-dispatched wrappers — SpMM
// ============================================================================

template<typename MatrixType, typename Scalar>
void spmm_forward(const MatrixType& A,
                  const DenseMatrix<Scalar>& X,
                  DenseMatrix<Scalar>& Y,
                  const DenseVector<Scalar>* row_means = nullptr,
                  int threads = 0,
                  SpmmWorkspace<Scalar>* ws = nullptr,
                  const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        spmm_forward_sparse(A, X, Y, row_means, threads, ws, row_inv_sds);
    } else {
        spmm_forward_dense(A, X, Y, row_means, row_inv_sds);
    }
}

template<typename MatrixType, typename Scalar>
void spmm_transpose(const MatrixType& A,
                    const DenseMatrix<Scalar>& X,
                    DenseMatrix<Scalar>& Z,
                    const DenseVector<Scalar>* row_means = nullptr,
                    int threads = 0,
                    const DenseVector<Scalar>* row_inv_sds = nullptr)
{
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        spmm_transpose_sparse(A, X, Z, row_means, threads, row_inv_sds);
    } else {
        spmm_transpose_dense(A, X, Z, row_means, row_inv_sds);
    }
}

// ============================================================================
// Row means dispatch
// ============================================================================

template<typename MatrixType, typename Scalar>
DenseVector<Scalar> compute_row_means(const MatrixType& A, int threads = 0) {
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        return compute_row_means_sparse<Scalar>(A, threads);
    } else {
        return compute_row_means_dense<Scalar>(A);
    }
}

// ============================================================================
// Row SDs dispatch
// ============================================================================

template<typename MatrixType, typename Scalar>
DenseVector<Scalar> compute_row_sds(const MatrixType& A,
                                    const DenseVector<Scalar>& row_means,
                                    int threads = 0) {
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        return compute_row_sds_sparse<Scalar>(A, row_means, threads);
    } else {
        return compute_row_sds_dense<Scalar>(A, row_means);
    }
}

}  // namespace svd
}  // namespace FactorNet

