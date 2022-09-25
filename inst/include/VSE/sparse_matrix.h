// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_SPARSE_MATRIX_H
#define VSE_SPARSE_MATRIX_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// SPARSE MATRIX
// this class wraps all above sparse encodings and confers matrix-like functionality
//   using std::deque for non-contiguous storage of column data
template <typename encoding_class, typename idx_t, typename val_t>
class sparse_matrix {
   public:
    std::deque<encoding_class> col_data;
    size_t nrow, ncol;

    const size_t cols() { return ncol; }
    const size_t rows() { return nrow; }

    sparse_matrix(const Eigen::SparseMatrix<val_t>& ptr) {
        ncol = ptr.cols();
        nrow = ptr.rows();
        for (size_t col = 0; col < ncol; ++col) {
            col_data.push_back(encoding_class(ptr, col, nrow));
        }
    }

    // compute column sums with range-based for loop
    //   (uses iterator to sum up unique values without accessing indices)
    Eigen::Matrix<val_t, 1, -1> col_sums() {
        Eigen::Matrix<val_t, 1, -1> res = Eigen::Matrix<val_t, 1, -1>::Zero(ncol);
        for (size_t col = 0; col < ncol; ++col)
            for (auto& x : col_data[col])
                res(col) += x;
        return res;
    }

    // sparse-dense multiplication
    //   (uses iterator to traverse all non-zero values, not necessarily in order, pulling
    //    corresponding indices from dense vector)
    template <typename T>
    Eigen::Matrix<T, -1, -1> operator*(const Eigen::Matrix<T, -1, -1>& m) {
        if (m.cols() != nrow) throw std::out_of_range("number of columns in dense matrix is not equal to number of rows in sparse matrix");
        Eigen::Matrix<T, -1, -1> res = Eigen::Matrix<T, -1, -1>::Zero(m.rows(), ncol);
        for (size_t col = 0; col < ncol; ++col)
            for (size_t row = 0; row < m.rows(); ++row)
                for (auto it = col_data[col].begin(); it != col_data[col].end(); ++it)
                    res(row, col) += (T)*it * m(row, it.row());
        return res;
    }

    Rcpp::List view() {
        Rcpp::List l = Rcpp::List::create(0);
        for (size_t i = 0; i < col_data.size(); ++i)
            l.push_back(col_data[i].view());
        return l;
    }

    size_t mem_usage() {
        size_t mem_used = 0;
        for (size_t i = 0; i < col_data.size(); ++i)
            mem_used += col_data[i].mem_usage();
        return mem_used;
    }
};

#endif