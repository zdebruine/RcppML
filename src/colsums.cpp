// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_H
#include <VSE.h>
#endif

//' Compute column sums of sparse-encoded data
//'
//' @param A object of class \code{dgCMatrix}
//' @param encoding sparse encoding to be used
//' @export
//' @return vector of column sums
//' @examples
//' library(Matrix)
//' A <- rsparsematrix(100, 1000, density = 0.5,
//'   rand.x = function(n) { sample(1:10, n, replace = TRUE)})
//' v <- colsums(A, encoding = "TRCSC_NP")
//' plot(v, Matrix::colSums(A))
//[[Rcpp::export]]
Eigen::Matrix<int, -1, -1> colsums(const Eigen::SparseMatrix<int>& A, std::string encoding = "CSC") {
    if (encoding == "CSC") {
        sparse_matrix<CSC<size_t, int>, size_t, int> mat(A);
        return mat.col_sums();
    } else if (encoding == "CSC_P") {
        sparse_matrix<CSC_P<size_t, int>, size_t, int> mat(A);
        return mat.col_sums();
    } else if (encoding == "CSC_NP") {
        sparse_matrix<CSC_NP<int, int>, int, int> mat(A);
        return mat.col_sums();
    } else if (encoding == "TCSC_P") {
        sparse_matrix<TCSC_P<size_t, int>, size_t, int> mat(A);
        return mat.col_sums();
    } else if (encoding == "TCSC_NP") {
        sparse_matrix<TCSC_NP<int, int>, int, int> mat(A);
        return mat.col_sums();
    } else if (encoding == "TRCSC_NP") {
        sparse_matrix<TRCSC_NP<int, int>, int, int> mat(A);
        return mat.col_sums();
    } else {
        Rcpp::stop("'encoding' must be one of 'CSC', 'CSC_P', 'CSC_NP', 'TCSC_P', 'TCSC_NP', or 'TRCSC_NP'");
    }
}
