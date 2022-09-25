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

//[[Rcpp::export]]
Rcpp::List c_inspect(const Eigen::SparseMatrix<int>& A, std::string encoding = "CSC") {
    if (encoding == "CSC") {
        sparse_matrix<CSC<size_t, int>, size_t, int> mat(A);
        return mat.view();
    } else if (encoding == "CSC_P") {
        sparse_matrix<CSC_P<size_t, int>, size_t, int> mat(A);
        return mat.view();
    } else if (encoding == "CSC_NP") {
        sparse_matrix<CSC_NP<int, int>, int, int> mat(A);
        return mat.view();
    } else if (encoding == "TCSC_P") {
        sparse_matrix<TCSC_P<size_t, int>, size_t, int> mat(A);
        return mat.view();
    } else if (encoding == "TCSC_NP") {
        sparse_matrix<TCSC_NP<int, int>, int, int> mat(A);
        return mat.view();
    } else if (encoding == "TRCSC_NP") {
        sparse_matrix<TRCSC_NP<int, int>, int, int> mat(A);
        return mat.view();
    }
}
