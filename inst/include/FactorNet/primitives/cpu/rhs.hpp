// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/rhs.hpp
 * @brief CPU specialization of the right-hand side primitive
 *
 * B = H · A  (k × m)  for sparse and dense A.
 *
 * For sparse A (CSC format), the computation iterates over columns of A
 * and accumulates outer products. For dense A, a single BLAS GEMM call.
 *
 * Also provides trace_AtA (||A||²_F) for Gram-trick loss.
 *
 * Extracted from nmf/rhs.hpp.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/core/traits.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

// ========================================================================
// rhs — sparse A overload
// ========================================================================

/**
 * @brief CPU RHS (sparse): B = H · A  (k × n)
 *
 * Iterates over CSC columns of A.  Each column j of A has nnz(j) nonzeros;
 * for each nonzero A(row, j) we accumulate A(row,j) * H.col(row) into B.col(j).
 *
 * Note: H is k × m (row-factor matrix in transposed storage).
 * The operation computes  B = H * A  where A is m × n.
 *
 * @tparam SparseMat  Eigen SparseMatrix or Map<const SparseMatrix>
 */
template<typename SparseMat, typename Scalar>
inline void rhs_sparse_impl(const SparseMat& A,
                             const DenseMatrix<Scalar>& H,
                             DenseMatrix<Scalar>& B,
                             int threads = 0)
{
    const int k = static_cast<int>(H.rows());
    const int n = static_cast<int>(A.cols());
    B.resize(k, n);
    B.setZero();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(threads > 0 ? threads : omp_get_max_threads())
#endif
    for (int j = 0; j < n; ++j) {
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            B.col(j).noalias() += it.value() * H.col(it.row());
        }
    }
}

/**
 * @brief CPU RHS (sparse double) — template specialization
 */
template<>
inline void rhs<CPU, double, SparseMatrix<double>>(
    const SparseMatrix<double>& A,
    const DenseMatrix<double>& H,
    DenseMatrix<double>& B)
{
    rhs_sparse_impl(A, H, B);
}

/**
 * @brief CPU RHS (sparse float) — template specialization
 */
template<>
inline void rhs<CPU, float, SparseMatrix<float>>(
    const SparseMatrix<float>& A,
    const DenseMatrix<float>& H,
    DenseMatrix<float>& B)
{
    rhs_sparse_impl(A, H, B);
}

/**
 * @brief CPU RHS (MappedSparseMatrix<double>) — template specialization
 */
template<>
inline void rhs<CPU, double, MappedSparseMatrix<double>>(
    const MappedSparseMatrix<double>& A,
    const DenseMatrix<double>& H,
    DenseMatrix<double>& B)
{
    rhs_sparse_impl(A, H, B);
}

// ========================================================================
// rhs — dense A overload
// ========================================================================

/**
 * @brief CPU RHS (dense): B = H · A
 *
 * Single BLAS GEMM call via Eigen.
 */
template<>
inline void rhs<CPU, double, DenseMatrix<double>>(
    const DenseMatrix<double>& A,
    const DenseMatrix<double>& H,
    DenseMatrix<double>& B)
{
    B.noalias() = H * A;
}

template<>
inline void rhs<CPU, float, DenseMatrix<float>>(
    const DenseMatrix<float>& A,
    const DenseMatrix<float>& H,
    DenseMatrix<float>& B)
{
    B.noalias() = H * A;
}

// trace_AtA is now defined in primitives.hpp (not here) to ensure
// visibility across all TUs (NMF and SVD).

}  // namespace primitives
}  // namespace FactorNet

