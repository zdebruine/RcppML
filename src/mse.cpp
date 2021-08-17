// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

//[[Rcpp::export]]
double Rcpp_mse_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  Rcpp::dgCMatrix A(A_S4);
  if (w.rows() == h.rows()) w.transposeInPlace();

  // multiply w by diagonal
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < w.cols(); ++i)
    for (unsigned int j = 0; j < w.rows(); ++j)
      w(j, i) *= d(i);

  // calculate total loss with parallelization across samples
  Eigen::VectorXd losses(A.cols());
  losses.setZero();
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    Eigen::VectorXd wh_i = w * h.col(i);
    for (Rcpp::dgCMatrix::InnerIterator iter(A, i); iter; ++iter)
      wh_i(iter.row()) -= iter.value();
    for (unsigned int j = 0; j < wh_i.size(); ++j)
      losses(i) += std::pow(wh_i(j), 2);
  }
  return losses.sum() / (A.cols() * A.rows());
}

//[[Rcpp::export]]
double Rcpp_mse_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  if (w.rows() == h.rows()) w.transposeInPlace();

  // multiply w by diagonal
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < w.cols(); ++i)
    for (unsigned int j = 0; j < w.rows(); ++j)
      w(j, i) *= d(i);

  // calculate total loss with parallelization across samples
  Eigen::VectorXd losses(A.cols());
  losses.setZero();
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    Eigen::VectorXd wh_i = w * h.col(i);
    for(unsigned int iter = 0; iter < A.rows(); ++iter)
      wh_i(iter) -= A(iter, i);      
    for (unsigned int j = 0; j < wh_i.size(); ++j)
      losses(i) += std::pow(wh_i(j), 2);
  }
  return losses.sum() / (A.cols() * A.rows());
}