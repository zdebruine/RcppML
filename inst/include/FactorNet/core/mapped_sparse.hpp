// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file mapped_sparse.hpp
 * @brief Zero-copy sparse matrix wrapper for direct CSC access
 * 
 * Provides a lightweight POD struct for zero-copy access to R's dgCMatrix
 * sparse matrices. This matches singlet's Rcpp::SparseMatrix approach
 * for optimal performance.
 * 
 * The struct stores raw pointers to the CSC components:
 * - x: Non-zero values (nnz elements)
 * - i: Row indices (nnz elements)  
 * - p: Column pointers (ncol+1 elements)
 * 
 * @author Zach DeBruine
 * @date 2026-01-28
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace FactorNet {

/**
 * @brief Zero-copy CSC sparse matrix wrapper
 * 
 * This is a POD struct with no methods that modify state, no virtual functions,
 * and all inline accessors for maximum performance.
 * 
 * @tparam Scalar  Value type (double or float)
 * @tparam Index   Index type (int for R compatibility)
 * 
 * @par Memory Layout (CSC Format)
 * For a matrix with m rows, n columns, and nnz non-zeros:
 * - x: array of nnz values (column-major order)
 * - i: array of nnz row indices (0-based)
 * - p: array of n+1 column pointers (p[j] = start of column j, p[n] = nnz)
 * 
 * @par Thread Safety
 * Read-only access is thread-safe. This struct does not own the underlying
 * memory and should not be used after the source data is deallocated.
 * 
 * @par Example
 * @code
 * // From R's dgCMatrix SEXP
 * CSCMatrix<double, int> A = CSCMatrix<double, int>::from_sexp(A_sexp);
 * 
 * // Iterate over column j
 * for (auto it = A.begin_col(j); it != A.end_col(j); ++it) {
 *     double val = *it.value();
 *     int row = it.row();
 * }
 * @endcode
 */
template<typename Scalar = double, typename Index = int>
struct CSCMatrix {
    // Raw CSC component pointers (non-owning)
    const Scalar* x;     ///< Non-zero values array
    const Index* i;      ///< Row indices array
    const Index* p;      ///< Column pointers array
    Index rows_;         ///< Number of rows
    Index cols_;         ///< Number of columns
    
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /// Default constructor (creates empty/invalid matrix)
    CSCMatrix() : x(nullptr), i(nullptr), p(nullptr), rows_(0), cols_(0) {}
    
    /**
     * @brief Construct from raw pointers
     * 
     * @param x_ptr   Pointer to values array
     * @param i_ptr   Pointer to row indices array
     * @param p_ptr   Pointer to column pointers array
     * @param nrow    Number of rows
     * @param ncol    Number of columns
     */
    CSCMatrix(const Scalar* x_ptr, const Index* i_ptr, const Index* p_ptr,
              Index nrow, Index ncol)
        : x(x_ptr), i(i_ptr), p(p_ptr), rows_(nrow), cols_(ncol) {}
    
    // ========================================================================
    // Dimensions
    // ========================================================================
    
    /// Number of rows
    Index rows() const { return rows_; }
    
    /// Number of columns
    Index cols() const { return cols_; }
    
    /// Total number of non-zeros
    Index nnz() const { return p[cols_]; }
    
    /// Number of non-zeros in column j
    Index col_nnz(Index j) const { return p[j + 1] - p[j]; }
    
    /// Check if column j is empty (no non-zeros)
    bool col_empty(Index j) const { return p[j] == p[j + 1]; }
    
    // ========================================================================
    // Direct Column Access
    // ========================================================================
    
    /// Start index into x/i arrays for column j
    Index col_start(Index j) const { return p[j]; }
    
    /// End index (exclusive) into x/i arrays for column j
    Index col_end(Index j) const { return p[j + 1]; }
    
    /// Pointer to first value in column j
    const Scalar* col_values(Index j) const { return x + p[j]; }
    
    /// Pointer to first row index in column j
    const Index* col_indices(Index j) const { return i + p[j]; }
    
    // ========================================================================
    // Inner Iterator (for column-wise traversal)
    // ========================================================================
    
    /**
     * @brief Iterator for traversing non-zeros within a column
     * 
     * Provides efficient iteration over a single column's entries.
     * Matches Eigen::SparseMatrix::InnerIterator interface.
     */
    class InnerIterator {
    private:
        const Scalar* x_ptr_;
        const Index* i_ptr_;
        const Index* end_ptr_;
        
    public:
        /// Construct iterator for column j
        InnerIterator(const CSCMatrix& mat, Index j)
            : x_ptr_(mat.x + mat.p[j])
            , i_ptr_(mat.i + mat.p[j])
            , end_ptr_(mat.i + mat.p[j + 1])
        {}
        
        /// Check if more elements remain
        operator bool() const { return i_ptr_ < end_ptr_; }
        
        /// Advance to next element
        InnerIterator& operator++() {
            ++x_ptr_;
            ++i_ptr_;
            return *this;
        }
        
        /// Current value
        Scalar value() const { return *x_ptr_; }
        
        /// Current row index
        Index row() const { return *i_ptr_; }
        
        /// Current index into sparse arrays
        Index index() const { return static_cast<Index>(i_ptr_ - end_ptr_); }
    };
    
    // ========================================================================
    // Factory Methods
    // ========================================================================
    
#ifdef RCPP_HPP  // Only available when Rcpp is included
    
    /**
     * @brief Create from R dgCMatrix SEXP (zero-copy)
     * 
     * @param s  R dgCMatrix object
     * @return   CSCMatrix wrapping the R object's data
     * 
     * @note The returned matrix is only valid while the R object exists.
     */
    static CSCMatrix from_sexp(SEXP s) {
        Rcpp::S4 mat(s);
        Rcpp::NumericVector x_vec = mat.slot("x");
        Rcpp::IntegerVector i_vec = mat.slot("i");
        Rcpp::IntegerVector p_vec = mat.slot("p");
        Rcpp::IntegerVector dim_vec = mat.slot("Dim");
        
        return CSCMatrix(
            x_vec.begin(),
            i_vec.begin(),
            p_vec.begin(),
            dim_vec[0],
            dim_vec[1]
        );
    }
    
#endif  // RCPP_HPP
    
    // ========================================================================
    // Eigen Interoperability
    // ========================================================================
    
    /**
     * @brief Create from Eigen sparse matrix
     * 
     * @param mat  Eigen sparse matrix (must be compressed)
     * @return     CSCMatrix wrapping Eigen's internal arrays
     * 
     * @note The Eigen matrix must outlive the CSCMatrix.
     * @note Only works with column-major (default) Eigen sparse matrices.
     */
    template<typename EigenScalar>
    static CSCMatrix from_eigen(const Eigen::SparseMatrix<EigenScalar, Eigen::ColMajor>& mat) {
        return CSCMatrix(
            mat.valuePtr(),
            mat.innerIndexPtr(),
            mat.outerIndexPtr(),
            static_cast<Index>(mat.rows()),
            static_cast<Index>(mat.cols())
        );
    }
    
    /**
     * @brief Convert to Eigen sparse matrix (creates a copy)
     * 
     * @return Eigen::SparseMatrix with copy of data
     */
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> to_eigen() const {
        // Map as Eigen sparse matrix then make a copy
        Eigen::Map<const Eigen::SparseMatrix<Scalar, Eigen::ColMajor>> mapped(
            rows_, cols_, nnz(), p, i, x
        );
        return Eigen::SparseMatrix<Scalar, Eigen::ColMajor>(mapped);
    }
};

// Convenience aliases
using SpMat = CSCMatrix<double, int>;
using SpMatF = CSCMatrix<float, int>;

}  // namespace FactorNet

