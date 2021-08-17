// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

// solve the least squares equation A = wh for h given A and w
Eigen::MatrixXd c_project_sparse(
  Rcpp::dgCMatrix& A,
  Eigen::MatrixXd& w,
  const bool nonneg,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const double L1,
  const unsigned int threads) {

  // "w" must be in "wide" format, where w.cols() == A.rows()

  // compute left-hand side of linear system, "a"
  Eigen::MatrixXd a = w * w.transpose();
  for(unsigned int i = 0; i < a.cols(); ++i) a(i, i) += 1e-15;
  Eigen::LLT<Eigen::MatrixXd, 1> a_llt = a.llt();
  unsigned int k = a.rows();
  Eigen::MatrixXd h(k, A.cols());

  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    // compute right-hand side of linear system, "b", in-place in h
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k);
    for (Rcpp::dgCMatrix::InnerIterator it(A, i); it; ++it)
      for (unsigned int j = 0; j < k; ++j) b(j) += it.value() * w(j, it.row());

    if (L1 != 0) for (unsigned int j = 0; j < k; ++j) b(j) -= L1;

    // solve least squares
    h.col(i) = c_nnls(a, b, a_llt, fast_maxit, cd_maxit, cd_tol, nonneg);
  }
  return h;
}

// Non-Negative Least Squares
//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w,
  const bool nonneg,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const double L1,
  const unsigned int threads) {

  Rcpp::dgCMatrix A(A_S4);
  return c_project_sparse(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads);
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_dense(
    const Rcpp::NumericMatrix& A,
    Eigen::MatrixXd& w,
    const bool nonneg,
    const unsigned int fast_maxit,
    const unsigned int cd_maxit,
    const double cd_tol,
    const double L1,
    const unsigned int threads) {
  
  Eigen::MatrixXd a = w * w.transpose();
  for(unsigned int i = 0; i < a.cols(); ++i) a(i, i) += 1e-15;
  Eigen::LLT<Eigen::MatrixXd, 1> a_llt = a.llt();
  unsigned int k = a.rows();
  Eigen::MatrixXd h(k, A.cols());
  
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    // compute right-hand side of linear system, "b", in-place in h
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k);
    for (unsigned int it = 0; it < A.rows(); ++it)
      b += A(it, i) * w.col(it);
    if (L1 != 0) for (unsigned int j = 0; j < k; ++j) b(j) -= L1;
    h.col(i) = c_nnls(a, b, a_llt, fast_maxit, cd_maxit, cd_tol, nonneg);
  }
  return h;
}