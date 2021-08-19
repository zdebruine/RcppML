// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RcppML_nmf1
#define RcppML_nmf1

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

wdhmodel c_nmf1_sparse(
  Rcpp::dgCMatrix& A,
  Eigen::MatrixXd& w,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag) {
  
  Eigen::MatrixXd h(1, A.cols());
  double d = 1, tol_ = 1;
  unsigned int it;
  
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  
  for (it = 0; it < maxit; ++it) {
    Eigen::MatrixXd w_it = w;
    // update h
    h.setZero();
    double a = 0;
    for(unsigned int i = 0; i < A.rows(); ++i) a += w(0, i) * w(0, i);
    for (unsigned int i = 0; i < A.cols(); ++i){
      for (Rcpp::dgCMatrix::InnerIterator it(A, i); it; ++it){
        h(0, i) += it.value() * w(0, it.row());
      }
      h(0, i) /= a;
    }

    // scale h
    if(diag) {
      d = h.array().sum();
      for (unsigned int i = 0; i < A.cols(); ++i) h(i) /= d;
    }

    // update w
    w.setZero();
    a = 0;
    for(unsigned int i = 0; i < A.cols(); ++i) a += h(0, i) * h(0, i);
    for (unsigned int i = 0; i < A.cols(); ++i)
      for (Rcpp::dgCMatrix::InnerIterator it(A, i); it; ++it)
        w(0, it.row()) += it.value() * h(0, i);
    for (unsigned int i = 0; i < A.rows(); ++i) w(0, i) /= a;

    // scale w
    if(diag) {
      d = w.array().sum();
      for (unsigned int i = 0; i < A.rows(); ++i) w(0, i) /= d;
    }
    
    // calculate tolerance and check for convergence
    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);
    if (tol_ < tol) break;
  }
  
  Eigen::VectorXd d_(1);
  d_(0) = d;
  
  return wdhmodel{w.transpose(), d_, h, tol_, it};
}

wdhmodel c_nmf1_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd& w,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag) {
  
  Eigen::MatrixXd h(1, A.cols());
  double d = 1, tol_ = 1;
  unsigned int it;
  
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  
  for (it = 0; it < maxit; ++it) {
    Eigen::MatrixXd w_it = w;
    // update h
    h.setZero();
    double a = 0;
    for(unsigned int i = 0; i < A.rows(); ++i) a += w(0, i) * w(0, i);
    for (unsigned int i = 0; i < A.cols(); ++i){
      for (unsigned int it = 0; it < A.rows(); ++it){
        h(0, i) += A(it, i) * w(0, it);
      }
      h(0, i) /= a;
    }

    // scale h
    if(diag) {
      d = h.array().sum();
      for (unsigned int i = 0; i < A.cols(); ++i) h(i) /= d;
    }
    
    // update w
    w.setZero();
    a = 0;
    for(unsigned int i = 0; i < A.cols(); ++i) a += h(0, i) * h(0, i);
    for (unsigned int i = 0; i < A.cols(); ++i)
      for (unsigned int it = 0; it < A.rows(); ++it)
        w(0, it) += A(it, i) * h(0, i);
    for (unsigned int i = 0; i < A.rows(); ++i) w(0, i) /= a;

    // scale w
    if(diag) {
      d = w.array().sum();
      for (unsigned int i = 0; i < A.rows(); ++i) w(0, i) /= d;
    }

    // calculate tolerance and check for convergence
    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);
    if (tol_ < tol) break;
  }

  Eigen::VectorXd d_(1);
  d_(0) = d;
  
  return wdhmodel{w.transpose(), d_, h, tol_, it};
}

#endif