// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file types.hpp
 * @brief Core type definitions for RcppML
 * 
 * Provides type aliases for dense and sparse matrices, vectors, and small
 * fixed-size types used throughout RcppML. All types are templated on scalar
 * precision (float or double) for flexibility.
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

#ifndef FactorNet_CORE_TYPES_HPP
#define FactorNet_CORE_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace FactorNet {

// ============================================================================
// Dense Matrix Types
// ============================================================================

/**
 * @brief Dynamic-size dense matrix with column-major storage
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Dynamic-size dense column vector
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/**
 * @brief Dynamic-size dense row vector
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using DenseRowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

// ============================================================================
// Fixed-Size Matrix Types
// ============================================================================

/**
 * @brief 2x2 matrix (used in bipartition for k=2 direct solve)
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;

/**
 * @brief 2-element vector
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

// ============================================================================
// Sparse Matrix Types
// ============================================================================

/**
 * @brief Sparse matrix in compressed sparse column (CSC) format
 * 
 * Uses column-major storage consistent with Eigen's default and R's dgCMatrix.
 * 
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;

/**
 * @brief Mapped (non-owning) sparse matrix view
 * 
 * Provides zero-copy access to externally owned sparse data (e.g., from R).
 * 
 * @tparam Scalar Element type (float or double)
 */
template<typename Scalar>
using MappedSparseMatrix = Eigen::Map<const SparseMatrix<Scalar>>;

// ============================================================================
// ============================================================================
// Normalization Type
// ============================================================================

/**
 * @brief Normalization type for the scaling diagonal in NMF
 *
 * Controls how factor norms are extracted into the diagonal d:
 *   - L1:   d[i] = sum(|F[i,:]|),    F[i,:] /= d[i]   (L1 / Manhattan norm)
 *   - L2:   d[i] = sqrt(sum(F[i,:]²)), F[i,:] /= d[i]   (L2 / Euclidean norm)
 *   - None: d[i] = 1, factors are not normalized
 */
enum class NormType { L1, L2, None };

// ============================================================================
// Backward Compatibility Aliases (Double Precision)
// ============================================================================

using MatrixXd = DenseMatrix<double>;
using VectorXd = DenseVector<double>;
using Matrix2d = Matrix2<double>;
using Vector2d = Vector2<double>;
using SparseMatrixXd = SparseMatrix<double>;

}  // namespace FactorNet

#endif // FactorNet_CORE_TYPES_HPP
