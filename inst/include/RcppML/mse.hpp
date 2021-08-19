// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RcppML_mse
#define RcppML_mse

#ifndef RcppML_common
#include <RcppMLCommon.h>
#endif

//[[Rcpp::export]]
double mse(
  Rcpp::dgCMatrix& A,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  if (w.rows() == h.rows()) w.transposeInPlace();

  // multiply w by diagonal
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  #endif
  for (unsigned int i = 0; i < w.cols(); ++i)
    for (unsigned int j = 0; j < w.rows(); ++j)
      w(j, i) *= d(i);

  // calculate total loss with parallelization across samples
  Eigen::VectorXd losses(A.cols());
  losses.setZero();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  #endif
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
double mse(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  if (w.rows() == h.rows()) w.transposeInPlace();

  // multiply w by diagonal
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  #endif
  for (unsigned int i = 0; i < w.cols(); ++i)
    for (unsigned int j = 0; j < w.rows(); ++j)
      w(j, i) *= d(i);

  // calculate total loss with parallelization across samples
  Eigen::VectorXd losses(A.cols());
  losses.setZero();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  #endif
  for (unsigned int i = 0; i < A.cols(); ++i) {
    Eigen::VectorXd wh_i = w * h.col(i);
    for(unsigned int iter = 0; iter < A.rows(); ++iter)
      wh_i(iter) -= A(iter, i);      
    for (unsigned int j = 0; j < wh_i.size(); ++j)
      losses(i) += std::pow(wh_i(j), 2);
  }
  return losses.sum() / (A.cols() * A.rows());
}

#endif