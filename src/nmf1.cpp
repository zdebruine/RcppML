// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

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

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf1_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag){

  Rcpp::dgCMatrix A(A_S4);
  wdhmodel m = c_nmf1_sparse(A, w_init, tol, nonneg, maxit, verbose, diag);
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = m.it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf1_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag){

  wdhmodel m = c_nmf1_dense(A, w_init, tol, nonneg, maxit, verbose, diag);
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = m.it);
}