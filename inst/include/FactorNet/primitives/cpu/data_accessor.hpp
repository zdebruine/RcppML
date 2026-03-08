#pragma once
/**
 * @file data_accessor.hpp
 * @brief Template traits for sparse/dense matrix element access.
 *
 * The DataAccessor<MatrixType> struct provides a unified iteration interface
 * over matrix columns/rows so that algorithmic code can be written once and
 * instantiated for both sparse and dense inputs.
 *
 * Usage example (replacing a sparse/dense pair with a single template):
 * @code
 * template<typename Scalar, typename MatrixType>
 * void some_h_update(const MatrixType& A, ...) {
 *     using DA = DataAccessor<MatrixType>;
 *     // Iterate over non-zero (sparse) or all (dense) entries in column j:
 *     DA::for_col(A, j, [&](int row, Scalar val) {
 *         b.noalias() += val * W_T.col(row);
 *     });
 * }
 * @endcode
 *
 * **Performance note**: For dense matrices, iteration covers ALL rows (including
 * structural zeros), so element-wise loops are O(m) vs. O(nnz) for sparse.
 * The fast BLAS path (e.g., `b.noalias() = W_T * A.col(j)`) is preserved in
 * `fast_rhs_col()` when no masking is needed.
 */

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <Eigen/Sparse>

namespace FactorNet {
namespace primitives {

// ============================================================================
// Primary template — no default; must use a specialization
// ============================================================================

template<typename MatrixType>
struct DataAccessor;  // Forward declaration; define specializations below.

// ============================================================================
// Sparse specialization (CSC format, Eigen::SparseMatrix)
// ============================================================================

template<typename Scalar, int Options, typename StorageIndex>
struct DataAccessor<Eigen::SparseMatrix<Scalar, Options, StorageIndex>> {
    using Matrix = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;

    /// Iterate over non-zero entries in column j: fn(row, value)
    template<typename Fn>
    static void for_col(const Matrix& A, int j, Fn&& fn) {
        for (typename Matrix::InnerIterator it(A, j); it; ++it)
            fn(static_cast<int>(it.row()), static_cast<Scalar>(it.value()));
    }

    /// Iterate over non-zero entries in row i (requires row-major or CSR).
    /// For CSC SparseMatrix this is O(nnz) — use sparingly.
    template<typename Fn>
    static void for_row(const Matrix& A, int i, Fn&& fn) {
        // For CSC sparse, row iteration requires scanning columns — provide
        // a generic fallback. Callers that need efficient row access should
        // store A^T and use for_col on the transpose.
        const int n = static_cast<int>(A.cols());
        for (int j = 0; j < n; ++j) {
            for (typename Matrix::InnerIterator it(A, j); it; ++it) {
                if (static_cast<int>(it.row()) == i)
                    fn(j, static_cast<Scalar>(it.value()));
            }
        }
    }

    /// Fast RHS: b += A.col(j) .* W_T (non-masked, BLAS version not available
    /// for sparse — falls back to element-wise accumulation)
    static void fast_rhs_col(const Matrix& A, const DenseMatrix<Scalar>& W_T,
                              int j, DenseVector<Scalar>& b) {
        for_col(A, j, [&](int r, Scalar v) {
            b.noalias() += v * W_T.col(r);
        });
    }

    static constexpr bool is_sparse = true;
};

// ============================================================================
// MappedSparseMatrix specialization (Eigen::Map<const SparseMatrix<...>>)
// Provides the same interface for the mapped (read-only) variant used as
// MappedSparseMatrix<Scalar> = Eigen::Map<const SparseMatrix<Scalar>>.
// ============================================================================

template<typename Scalar, int Options, typename StorageIndex>
struct DataAccessor<Eigen::Map<const Eigen::SparseMatrix<Scalar, Options, StorageIndex>>> {
    using Matrix = Eigen::Map<const Eigen::SparseMatrix<Scalar, Options, StorageIndex>>;

    template<typename Fn>
    static void for_col(const Matrix& A, int j, Fn&& fn) {
        for (typename Matrix::InnerIterator it(A, j); it; ++it)
            fn(static_cast<int>(it.row()), static_cast<Scalar>(it.value()));
    }

    template<typename Fn>
    static void for_row(const Matrix& A, int i, Fn&& fn) {
        const int n = static_cast<int>(A.cols());
        for (int j = 0; j < n; ++j) {
            for (typename Matrix::InnerIterator it(A, j); it; ++it) {
                if (static_cast<int>(it.row()) == i)
                    fn(j, static_cast<Scalar>(it.value()));
            }
        }
    }

    static void fast_rhs_col(const Matrix& A, const DenseMatrix<Scalar>& W_T,
                              int j, DenseVector<Scalar>& b) {
        for_col(A, j, [&](int r, Scalar v) {
            b.noalias() += v * W_T.col(r);
        });
    }

    static constexpr bool is_sparse = true;
};

// ============================================================================
// Dense specialization (ColMajor DenseMatrix)
// ============================================================================

template<typename Scalar>
struct DataAccessor<DenseMatrix<Scalar>> {
    using Matrix = DenseMatrix<Scalar>;

    /// Iterate over ALL entries in column j (structural zeros included): fn(row, value)
    template<typename Fn>
    static void for_col(const Matrix& A, int j, Fn&& fn) {
        const int m = static_cast<int>(A.rows());
        for (int r = 0; r < m; ++r)
            fn(r, A(r, j));
    }

    /// Iterate over ALL entries in row i: fn(col, value)
    template<typename Fn>
    static void for_row(const Matrix& A, int i, Fn&& fn) {
        const int n = static_cast<int>(A.cols());
        for (int j = 0; j < n; ++j)
            fn(j, A(i, j));
    }

    /// Fast RHS using BLAS matrix-vector product: b = W_T * A.col(j)
    static void fast_rhs_col(const Matrix& A, const DenseMatrix<Scalar>& W_T,
                              int j, DenseVector<Scalar>& b) {
        b.noalias() = W_T * A.col(j);
    }

    static constexpr bool is_sparse = false;
};

// ============================================================================
// Mapped Dense specialization (Eigen::Map<const DenseMatrix<Scalar>>)
// Same interface as the DenseMatrix specialization for read-only mapped data.
// ============================================================================

template<typename Scalar>
struct DataAccessor<Eigen::Map<const DenseMatrix<Scalar>>> {
    using Matrix = Eigen::Map<const DenseMatrix<Scalar>>;

    template<typename Fn>
    static void for_col(const Matrix& A, int j, Fn&& fn) {
        const int m = static_cast<int>(A.rows());
        for (int r = 0; r < m; ++r)
            fn(r, A(r, j));
    }

    template<typename Fn>
    static void for_row(const Matrix& A, int i, Fn&& fn) {
        const int n = static_cast<int>(A.cols());
        for (int j = 0; j < n; ++j)
            fn(j, A(i, j));
    }

    static void fast_rhs_col(const Matrix& A, const DenseMatrix<Scalar>& W_T,
                              int j, DenseVector<Scalar>& b) {
        b.noalias() = W_T * A.col(j);
    }

    static constexpr bool is_sparse = false;
};

// ============================================================================
// Convenience: include is_sparse_v from core traits (canonical check)
// ============================================================================

} // namespace primitives
} // namespace FactorNet
