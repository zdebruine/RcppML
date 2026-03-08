// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file traits.hpp
 * @brief Type traits and compile-time type detection
 * 
 * Provides template metaprogramming utilities for:
 * - Sparse vs dense matrix detection
 * - Scalar type extraction
 * - Constraint type tags (NoBounds, NonNegative, BoundedNonNegative)
 * - Transpose storage type selection
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

#ifndef FactorNet_CORE_TRAITS_HPP
#define FactorNet_CORE_TRAITS_HPP

#include <FactorNet/core/types.hpp>
#include <type_traits>

namespace FactorNet {

// ============================================================================
// Constraint Type Tags
// ============================================================================

/**
 * @brief Tag for unconstrained optimization (semi-NMF W)
 * 
 * Used as template parameter to enable compile-time specialization
 * of coordinate descent solvers.
 */
struct NoBounds {};

/**
 * @brief Tag for non-negativity constraint (standard NMF)
 * 
 * Enforces x >= 0 constraint via projection in coordinate descent.
 */
struct NonNegative {};

/**
 * @brief Tag for bounded non-negativity (0 <= x <= upper_bound)
 * 
 * Enforces box constraints with runtime upper bound parameter.
 */
struct BoundedNonNegative {};

// ============================================================================
// Sparse Matrix Detection
// ============================================================================

/**
 * @brief Type trait to detect Eigen sparse matrix types
 * 
 * Provides compile-time boolean `value` indicating whether T is a
 * sparse matrix type (SparseMatrix, MappedSparseMatrix, or Map thereof).
 * 
 * @tparam T Type to check
 * 
 * @par Example
 * @code
 * static_assert(is_sparse_matrix<Eigen::SparseMatrix<double>>::value);
 * static_assert(!is_sparse_matrix<Eigen::MatrixXd>::value);
 * @endcode
 */
template<typename T>
struct is_sparse_matrix : std::false_type {};

// Specialize for Eigen::SparseMatrix
template<typename Scalar, int Options, typename StorageIndex>
struct is_sparse_matrix<Eigen::SparseMatrix<Scalar, Options, StorageIndex>> 
    : std::true_type {};

// Specialize for Eigen::Map<Eigen::SparseMatrix>
template<typename Scalar, int Options, typename StorageIndex, int MapOptions, typename StrideType>
struct is_sparse_matrix<Eigen::Map<Eigen::SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>>
    : std::true_type {};

// Specialize for Eigen::Map<const Eigen::SparseMatrix>
template<typename Scalar, int Options, typename StorageIndex, int MapOptions, typename StrideType>
struct is_sparse_matrix<Eigen::Map<const Eigen::SparseMatrix<Scalar, Options, StorageIndex>, MapOptions, StrideType>>
    : std::true_type {};

/**
 * @brief Convenience variable template for sparse detection
 * 
 * @tparam T Type to check
 * @return true if T is a sparse matrix type, false otherwise
 * 
 * @par Example
 * @code
 * if constexpr (is_sparse_v<Mat>) {
 *     // Sparse-specific implementation
 * } else {
 *     // Dense implementation
 * }
 * @endcode
 */
template<typename T>
inline constexpr bool is_sparse_v = is_sparse_matrix<std::decay_t<T>>::value;

// ============================================================================
// Scalar Type Extraction
// ============================================================================

/**
 * @brief Extract scalar type from matrix type
 * 
 * Provides member typedef `type` equal to the element type of the matrix.
 * 
 * @tparam Mat Matrix type (Eigen dense or sparse)
 */
template<typename Mat>
struct matrix_scalar {
    using type = typename std::decay_t<Mat>::Scalar;
};

/**
 * @brief Convenience alias for matrix_scalar<Mat>::type
 * 
 * @tparam Mat Matrix type
 * 
 * @par Example
 * @code
 * using Scalar = matrix_scalar_t<Eigen::MatrixXd>;  // double
 * @endcode
 */
template<typename Mat>
using matrix_scalar_t = typename matrix_scalar<Mat>::type;

// ============================================================================
// Transpose Storage Selection
// ============================================================================

/**
 * @brief Select appropriate transpose storage type
 * 
 * For sparse matrices, transpose is stored as SparseMatrix<Scalar>.
 * For dense matrices, transpose is stored as DenseMatrix<Scalar>.
 * 
 * This enables compile-time selection of storage strategy for transposed data.
 * 
 * @tparam Mat Input matrix type
 * @tparam Scalar Computation precision (may differ from Mat::Scalar)
 * @tparam IsSparse Detected sparsity (default: automatic)
 */
template<typename Mat, typename Scalar, bool IsSparse = is_sparse_v<Mat>>
struct transpose_storage {
    using type = DenseMatrix<Scalar>;
};

// Specialization for sparse matrices
template<typename Mat, typename Scalar>
struct transpose_storage<Mat, Scalar, true> {
    using type = SparseMatrix<Scalar>;
};

/**
 * @brief Convenience alias for transpose_storage<Mat, Scalar>::type
 * 
 * @tparam Mat Input matrix type
 * @tparam Scalar Computation precision
 */
template<typename Mat, typename Scalar>
using transpose_storage_t = typename transpose_storage<Mat, Scalar>::type;

// ============================================================================
// Reference Type Selection
// ============================================================================

/**
 * @brief Select appropriate reference type for input matrices
 * 
 * Sparse matrices should use const references (never modified).
 * Dense matrices may use non-const references (can be modified in place).
 * 
 * @tparam Mat Matrix type
 * @tparam IsSparse Detected sparsity (default: automatic)
 */
template<typename Mat, bool IsSparse = is_sparse_v<Mat>>
struct input_reference {
    using type = Mat&;  // Dense: non-const reference
};

// Specialization for sparse matrices
template<typename Mat>
struct input_reference<Mat, true> {
    using type = const Mat&;  // Sparse: const reference
};

/**
 * @brief Convenience alias for input_reference<Mat>::type
 * 
 * @tparam Mat Matrix type
 */
template<typename Mat>
using input_reference_t = typename input_reference<Mat>::type;

}  // namespace FactorNet

#endif // FactorNet_CORE_TRAITS_HPP
