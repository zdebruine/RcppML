// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file blas.hpp
 * @brief BLAS-optimized matrix operations
 * 
 * Provides high-performance implementations of:
 * - Gram matrix computation (A×A^T, A^T×A)
 * - Sparse matrix-matrix multiplication
 * - Column/row extraction and norms
 * 
 * Uses Eigen's optimized BLAS backends for maximum performance.
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

#ifndef FactorNet_MATH_BLAS_HPP
#define FactorNet_MATH_BLAS_HPP

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>

namespace FactorNet {
namespace math {

/**
 * @brief Compute Gram matrix G = A×A^T using SYRK
 * 
 * For column-major A (k×n), computes symmetric k×k Gram matrix.
 * Uses Eigen's selfadjointView with rankUpdate (~1.7x faster than naive).
 * 
 * @tparam Scalar Computation precision
 * @param A Input matrix (k×n)
 * @param G Output Gram matrix (k×k, only lower triangle computed)
 * 
 * @par Complexity
 * Time: O(k²n) flops
 * Space: O(1) additional
 * 
 * @par Thread Safety
 * Reentrant if A and G do not alias
 */
template<typename Scalar>
inline void gram_aat(const DenseMatrix<Scalar>& A, DenseMatrix<Scalar>& G) {
    G.template triangularView<Eigen::Lower>().setZero();
    G.template selfadjointView<Eigen::Lower>().rankUpdate(A, static_cast<Scalar>(1.0));
}

/**
 * @brief Compute Gram matrix G = A^T×A using SYRK
 * 
 * @tparam Scalar Computation precision
 * @param A Input matrix (m×k)
 * @param G Output Gram matrix (k×k, only lower triangle computed)
 * 
 * @par Complexity
 * Time: O(mk²) flops
 */
template<typename Scalar>
inline void gram_ata(const DenseMatrix<Scalar>& A, DenseMatrix<Scalar>& G) {
    // Compute symmetric Gram matrix G = A^T * A using efficient rank update
    // selfadjointView<Lower>().rankUpdate(B) computes B * B^T
    // So for A^T * A, we need B = A^T, which means B * B^T = A^T * A
    G.setZero();
    // Note: rankUpdate(A.transpose()) computes A^T * (A^T)^T = A^T * A
    G.template selfadjointView<Eigen::Lower>().rankUpdate(A.transpose());
    G.template triangularView<Eigen::Upper>() = G.transpose();
}

/**
 * @brief Add diagonal regularization for numerical stability
 * 
 * Ensures all diagonal elements are at least eps (prevents singular matrices).
 * 
 * @tparam Scalar Precision type
 * @param G Gram matrix (modified in place)
 * @param eps Minimum diagonal value
 */
template<typename Scalar>
inline void gram_regularize_diagonal(DenseMatrix<Scalar>& G, Scalar eps) {
    const int k = G.rows();
    for (int i = 0; i < k; ++i) {
        G(i, i) = std::max(G(i, i), eps);
    }
}

/**
 * @brief Add L2 regularization to diagonal
 * 
 * @tparam Scalar Precision type
 * @param G Gram matrix (modified in place)
 * @param L2 L2 penalty strength
 * @param tiny Numerical stability constant
 */
template<typename Scalar>
inline void gram_add_diagonal(DenseMatrix<Scalar>& G, Scalar L2, Scalar tiny) {
    G.diagonal().array() += L2 + tiny;
}

/**
 * @brief Subtract Gram correction for masked columns
 * 
 * Given base Gram G_full = W×W^T, compute G_masked by subtracting
 * contribution from masked columns: G = G_full - W_masked×W_masked^T
 * 
 * @tparam Scalar Precision type
 * @param G Gram matrix (modified in place)
 * @param W Full matrix (k×n)
 * @param masked_indices Column indices to exclude
 * 
 * @par Complexity
 * Time: O(k²×n_masked)
 */
template<typename Scalar>
inline void gram_subtract_masked(DenseMatrix<Scalar>& G, 
                                 const DenseMatrix<Scalar>& W,
                                 const std::vector<int>& masked_indices) {
    if (masked_indices.empty()) return;
    
    const int k = W.rows();
    const int n_masked = static_cast<int>(masked_indices.size());
    
    DenseMatrix<Scalar> W_masked(k, n_masked);
    for (int i = 0; i < n_masked; ++i) {
        W_masked.col(i) = W.col(masked_indices[i]);
    }
    
    G.noalias() -= W_masked * W_masked.transpose();
}

/**
 * @brief Compute squared Frobenius norm of dense matrix
 * 
 * @tparam Scalar Precision type
 * @param A Input matrix
 * @return ||A||²_F
 * 
 * @par Complexity
 * Time: O(mn) where A is m×n
 */
template<typename Scalar>
inline Scalar squared_frobenius_norm(const DenseMatrix<Scalar>& A) {
    return A.squaredNorm();
}

/**
 * @brief Compute squared Frobenius norm of sparse matrix
 * 
 * @tparam Scalar Precision type
 * @param A Sparse input matrix
 * @return ||A||²_F (sum of squared non-zero elements)
 * 
 * @par Complexity
 * Time: O(nnz) where nnz is number of non-zeros
 */
template<typename Scalar>
inline Scalar squared_frobenius_norm(const SparseMatrix<Scalar>& A) {
    Scalar sum = 0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename SparseMatrix<Scalar>::InnerIterator it(A, k); it; ++it) {
            Scalar val = it.value();
            sum += val * val;
        }
    }
    return sum;
}

/**
 * @brief Extract column from sparse matrix as dense vector
 * 
 * @tparam SpMat Sparse matrix type
 * @tparam Scalar Output precision
 * @param A Sparse matrix
 * @param col Column index
 * @return Dense column vector
 * 
 * @par Complexity
 * Time: O(m) where m is number of rows
 */
template<typename SpMat, typename Scalar>
inline DenseVector<Scalar> get_column(const SpMat& A, int col) {
    DenseVector<Scalar> c = DenseVector<Scalar>::Zero(A.rows());
    for (typename SpMat::InnerIterator it(A, col); it; ++it) {
        c(it.row()) = static_cast<Scalar>(it.value());
    }
    return c;
}

/**
 * @brief Get inner indices (row indices) of non-zeros in sparse column
 * 
 * @tparam SpMat Sparse matrix type
 * @param A Sparse matrix
 * @param col Column index
 * @return Vector of row indices with non-zero values
 * 
 * @par Complexity
 * Time: O(nnz_col) where nnz_col is non-zeros in column
 */
template<typename SpMat>
inline std::vector<int> get_inner_indices(const SpMat& A, int col) {
    std::vector<int> indices;
    indices.reserve(A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col]);
    for (typename SpMat::InnerIterator it(A, col); it; ++it) {
        indices.push_back(it.row());
    }
    return indices;
}

/**
 * @brief Check if sparse matrix is approximately symmetric
 * 
 * @tparam SpMat Sparse matrix type
 * @param A Sparse matrix
 * @param tol Tolerance for element-wise comparison
 * @return true if A ≈ A^T
 * 
 * @par Complexity
 * Time: O(min(10, n) × avg_nnz_per_col) where n is matrix size
 */
template<typename SpMat>
inline bool is_approximately_symmetric(const SpMat& A, double tol = 1e-10) {
    if (A.rows() != A.cols()) return false;
    
    int checks = std::min(10, static_cast<int>(A.cols()));
    for (int k = 0; k < checks; ++k) {
        for (typename SpMat::InnerIterator it(A, k); it; ++it) {
            if (it.row() != it.col()) {
                double transpose_val = A.coeff(it.col(), it.row());
                if (std::abs(it.value() - transpose_val) > tol) return false;
            }
        }
    }
    return true;
}

}  // namespace math
}  // namespace FactorNet

#endif // FactorNet_MATH_BLAS_HPP
