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

//' Cross-product of transpose
//'
//' @param x sparse matrix object of class \code{dgCMatrix} of dimensions \code{m x n} with integral non-zero values
//' @param y dense matrix object of class \code{matrix} of dimensions \code{n x k}
//' @param encoding sparse encoding to be used
//' @export
//' @return dense matrix giving the transpose cross-product
//' @examples
//' library(Matrix)
//' x <- rsparsematrix(100, 1000, density = 0.5,
//'        rand.x = function(n){sample(1 : 10, n, replace = TRUE)})
//' y <- matrix(runif(10 * 1000), 10, 1000)
//' plot(VSE::tcrossprod(x, y), t(Matrix::tcrossprod(x, y)))
//[[Rcpp::export]]
Eigen::Matrix<double, -1, -1> tcrossprod(const Eigen::SparseMatrix<int>& x, const Eigen::Matrix<double, -1, -1>& y, std::string encoding = "CSC") {
    Eigen::SparseMatrix<int> A = x.transpose();
    if (encoding == "CSC") {
        sparse_matrix<CSC<size_t, int>, size_t, int> mat(A);
        return mat * y;
    } else if (encoding == "CSC_P") {
        sparse_matrix<CSC_P<size_t, int>, size_t, int> mat(A);
        return mat * y;
    } else if (encoding == "CSC_NP") {
        sparse_matrix<CSC_NP<int, int>, int, int> mat(A);
        return mat * y;
    } else if (encoding == "TCSC_P") {
        sparse_matrix<TCSC_P<size_t, int>, size_t, int> mat(A);
        return mat * y;
    } else if (encoding == "TCSC_NP") {
        sparse_matrix<TCSC_NP<int, int>, int, int> mat(A);
        return mat * y;
    } else if (encoding == "TRCSC_NP") {
        sparse_matrix<TRCSC_NP<int, int>, int, int> mat(A);
        return mat * y;
    } else {
        Rcpp::stop("'encoding' must be one of 'CSC', 'CSC_P', 'CSC_NP', 'TCSC_P', 'TCSC_NP', or 'TRCSC_NP'");
    }
}
