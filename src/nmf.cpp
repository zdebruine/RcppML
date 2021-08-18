// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

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
    h = c_project_sparse(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1_h, threads);
    Rcpp::checkUserInterrupt();

    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = h.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
    }

    // update w
    Eigen::MatrixXd w_it = w;
    if (symmetric) w = c_project_sparse(A, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    else w = c_project_sparse(At, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
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
    h = Rcpp_project_dense(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1_h, threads);
    Rcpp::checkUserInterrupt();

    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = h.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
    }

    // update w
    Eigen::MatrixXd w_it = w;
    if (symmetric) w = Rcpp_project_dense(A, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    else w = Rcpp_project_dense(At, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
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

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_sparse(
  const Rcpp::S4& A_S4,
  const Rcpp::S4& At_S4,
  const bool symmetric,
  Eigen::MatrixXd& w_init,
  const double tol = 1e-3,
  const bool nonneg = true,
  const double L1_w = 0,
  const double L1_h = 0,
  const unsigned int maxit = 100,
  const bool diag = true,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool verbose = false,
  const unsigned int threads = 0) {

  Rcpp::dgCMatrix A(A_S4), At(At_S4);

  wdhmodel m = c_nmf_sparse(A, At, symmetric, w_init, tol, nonneg, L1_w, L1_h, maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads);
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = m.it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_dense(
  const Rcpp::NumericMatrix& A,
  const bool symmetric,
  Eigen::MatrixXd& w_init,
  const double tol = 1e-3,
  const bool nonneg = true,
  const double L1_w = 0,
  const double L1_h = 0,
  const unsigned int maxit = 100,
  const bool diag = true,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool verbose = false,
  const unsigned int threads = 0) {

  Rcpp::NumericMatrix At;
  if(!symmetric) At = Rcpp::transpose(A);

  wdhmodel m = c_nmf_dense(A, At, symmetric, w_init, tol, nonneg, L1_w, L1_h, maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads);
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = m.it);
}