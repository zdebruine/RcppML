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

//[[Rcpp::depends(RcppClock)]]
#include <RcppClock.h>

//' Benchmark structure performance
//'
//' Measure runtime for computing column sums and sparse-dense matrix multiplication
//'
//' @param x sparse matrix object of class \code{dgCMatrix} of dimensions \code{m x n} with integral non-zero values
//' @param y dense matrix object of class \code{matrix} of dimensions \code{n x k}
//' @param n_reps number of reps
//' @return global variables \code{bench_colsums} and \code{bench_tcrossprod} in the global environment. Use \code{RcppClock::plot} method to visualize results, or coerce to a data.frame.
//' @export
//' @seealso \code{\link{tcrossprod}}
//' @examples
//' library(Matrix)
//' x <- rsparsematrix(50, 100, density = 0.5,
//'   rand.x = function(n) { sample(1:10, n, replace = TRUE)})
//' y <- matrix(runif(10 * 100), 10, 100)
//' VSE::benchmark(x, y)
//' plot(bench_tcrossprod)
//' plot(bench_colsums)
//[[Rcpp::export]]
void benchmark(const Eigen::SparseMatrix<int>& x, const Eigen::Matrix<double, -1, -1>& y, size_t n_reps = 100) {
    // construct different matrices (although constructors are certainly not optimized)

    const Eigen::SparseMatrix<int> A = x.transpose();

    Rprintf("constructing CSC\n");
    Rcpp::Clock construct;
    construct.tick("CSC");
    sparse_matrix<CSC<size_t, int>, size_t, int> mat_CSC(A);
    construct.tock("CSC");

    Rprintf("constructing CSC_P\n");
    construct.tick("CSC_P");
    sparse_matrix<CSC_P<size_t, int>, size_t, int> mat_CSC_P(A);
    construct.tock("CSC_P");

    Rprintf("constructing CSC_NP\n");
    construct.tick("CSC_NP");
    sparse_matrix<CSC_NP<int, int>, int, int> mat_CSC_NP(A);
    construct.tock("CSC_NP");

    Rprintf("constructing TCSC_P\n");
    construct.tick("TCSC_P");
    sparse_matrix<TCSC_P<size_t, int>, size_t, int> mat_TCSC_P(A);
    construct.tock("TCSC_P");

    Rprintf("constructing TCSC_NP\n");
    construct.tick("TCSC_NP");
    sparse_matrix<CSC<int, int>, int, int> mat_TCSC_NP(A);
    construct.tock("TCSC_NP");

    Rprintf("constructing RTCSC_NP\n");
    construct.tick("RTCSC_NP");
    sparse_matrix<CSC<int, int>, int, int> mat_RTCSC_NP(A);
    construct.tock("RTCSC_NP");

    construct.stop("bench_construct");

    // run column sums using range-based "for" loop
    Rcpp::Clock sums;
    for (size_t rep = 0; rep < n_reps; ++rep) {
        sums.tick("CSC");
        Eigen::Matrix<int, 1, -1> dummy = mat_CSC.col_sums();
        sums.tock("CSC");

        sums.tick("CSC_P");
        dummy = mat_CSC_P.col_sums();
        sums.tock("CSC_P");

        sums.tick("CSC_NP");
        dummy = mat_CSC_NP.col_sums();
        sums.tock("CSC_NP");

        sums.tick("TCSC_P");
        dummy = mat_TCSC_P.col_sums();
        sums.tock("TCSC_P");

        sums.tick("TCSC_NP");
        dummy = mat_TCSC_NP.col_sums();
        sums.tock("TCSC_NP");

        sums.tick("RTCSC_NP");
        dummy = mat_RTCSC_NP.col_sums();
        sums.tock("RTCSC_NP");
    }
    sums.stop("bench_colsums");

    // run sparse-dense matrix multiplication
    Rcpp::Clock mult;
    for (size_t rep = 0; rep < n_reps; ++rep) {
        mult.tick("CSC");
        Eigen::Matrix<double, -1, -1> dummy = mat_CSC * y;
        mult.tock("CSC");
        mult.tick("CSC_P");
        dummy = mat_CSC_P * y;
        mult.tock("CSC_P");

        mult.tick("CSC_NP");
        dummy = mat_CSC_NP * y;
        mult.tock("CSC_NP");

        mult.tick("TCSC_P");
        dummy = mat_TCSC_P * y;
        mult.tock("TCSC_P");
        mult.tick("TCSC_NP");
        dummy = mat_TCSC_NP * y;
        mult.tock("TCSC_NP");

        mult.tick("RTCSC_NP");
        dummy = mat_RTCSC_NP * y;
        mult.tock("RTCSC_NP");
    }
    mult.stop("bench_tcrossprod");
}