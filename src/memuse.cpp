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

//' Get memory used by C++ object
//'
//' Calculates size of vector container, all sub-containers, and all index and value types within the container.  This function stores all values and indices as \code{short int}.
//'
//' @param A object of class \code{dgCMatrix}
//' @param encoding sparse encoding to be used
//' @export
//' @return size of object in bytes
//' @examples
//' library(Matrix)
//' A <- rsparsematrix(100, 1000, density = 0.5,
//'   rand.x = function(n) { sample(1:10, n, replace = TRUE)})
//' memuse(A, "TRCSC_NP")
//[[Rcpp::export]]
size_t memuse(const Eigen::SparseMatrix<short int>& A, std::string encoding = "CSC") {
    if (encoding == "CSC") {
        sparse_matrix<CSC<short int, short int>, short int, short int> mat(A);
        return mat.mem_usage();
    } else if (encoding == "CSC_P") {
        sparse_matrix<CSC_P<short int, short int>, short int, short int> mat(A);
        return mat.mem_usage();
    } else if (encoding == "CSC_NP") {
        sparse_matrix<CSC_NP<short int, short int>, short int, short int> mat(A);
        return mat.mem_usage();
    } else if (encoding == "TCSC_P") {
        sparse_matrix<TCSC_P<short int, short int>, short int, short int> mat(A);
        return mat.mem_usage();
    } else if (encoding == "TCSC_NP") {
        sparse_matrix<TCSC_NP<short int, short int>, short int, short int> mat(A);
        return mat.mem_usage();
    } else if (encoding == "TRCSC_NP") {
        sparse_matrix<TRCSC_NP<short int, short int>, short int, short int> mat(A);
        return mat.mem_usage();
    } else {
        Rcpp::stop("'encoding' must be one of 'CSC', 'CSC_P', 'CSC_NP', 'TCSC_P', 'TCSC_NP', or 'TRCSC_NP'");
    }
}
