// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_nmf
#define RcppML_nmf

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

#ifndef RcppML_project
#include <RcppML/project.hpp>
#endif

wdhmodel c_nmf_sparse(
  Rcpp::dgCMatrix& A,
  Rcpp::dgCMatrix& At,
  const bool symmetric,
  Eigen::MatrixXd& w,
  const double tol,
  const bool nonneg,
  const double L1_w,
  const double L1_h,
  const unsigned int maxit,
  const bool diag,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const bool verbose,
  const unsigned int threads) {

  const unsigned int k = w.rows();  
  Eigen::MatrixXd h(k, A.cols());
  Eigen::VectorXd d(k);
  d = d.setOnes();
  double tol_ = 1;
  unsigned int it;

  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");

  for (it = 0; it < maxit; ++it) {
    // update h
    h = project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1_h, threads);
    Rcpp::checkUserInterrupt();

    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = h.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
    }

    // update w
    Eigen::MatrixXd w_it = w;
    if (symmetric) w = project(A, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    else w = project(At, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    Rcpp::checkUserInterrupt();

    // reset diagonal and scale "w"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = w.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
    }

    // calculate tolerance
    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);

    // check for convergence
    if (tol_ < tol) break;
  }

  // reorder factors by diagonal
  if (diag) {
    std::vector<int> indx = sort_index(d);
    w = reorder_rows(w, indx);
    d = reorder(d, indx);
    h = reorder_rows(h, indx);
  }

  return wdhmodel{w.transpose(), d, h, tol_, it};
}

wdhmodel c_nmf_dense(
  const Rcpp::NumericMatrix& A,
  const Rcpp::NumericMatrix& At,
  const bool symmetric,
  Eigen::MatrixXd& w,
  const double tol,
  const bool nonneg,
  const double L1_w,
  const double L1_h,
  const unsigned int maxit,
  const bool diag,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const bool verbose,
  const unsigned int threads) {

  const unsigned int k = w.rows();

  Eigen::MatrixXd h(k, A.cols());
  Eigen::VectorXd d(k);
  d = d.setOnes();
  double tol_ = 1;
  unsigned int it;

  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");

  for (it = 0; it < maxit; ++it) {
    // update h
    h = project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1_h, threads);
    Rcpp::checkUserInterrupt();

    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = h.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
    }

    // update w
    Eigen::MatrixXd w_it = w;
    if (symmetric) w = project(A, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    else w = project(At, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    Rcpp::checkUserInterrupt();

    // reset diagonal and scale "w"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = w.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
    }

    // calculate tolerance
    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);

    // check for convergence
    if (tol_ < tol) break;
  }

  // reorder factors by diagonal
  if (diag) {
    std::vector<int> indx = sort_index(d);
    w = reorder_rows(w, indx);
    d = reorder(d, indx);
    h = reorder_rows(h, indx);
  }

  return wdhmodel{w.transpose(), d, h, tol_, it};
}

#endif