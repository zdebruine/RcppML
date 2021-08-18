// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

wdhmodel c_nmf2_sparse(
    Rcpp::dgCMatrix& A,
    Eigen::MatrixXd& w,
    const double tol,
    const bool nonneg,
    const unsigned int maxit,
    const bool verbose,
    const bool diag,
    const std::vector<unsigned int> samples) {
  
  Eigen::MatrixXd h(2, A.cols()), wb(2, A.rows());
  Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
  double tol_ = 1;
  unsigned int it;
  
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  
  for (it = 0; it < maxit; ++it) {
    // update h
    Eigen::MatrixXd w_it = w;
    Eigen::Matrix2d a = w * w.transpose();
    double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    for (unsigned int i = 0; i < samples.size(); ++i) {
      // compute right-hand side of linear system, "b", in-place in h
      double b0 = 0, b1 = 0;
      for (Rcpp::dgCMatrix::InnerIterator it(A, samples[i]); it; ++it){
        b0 += it.value() * w(0, it.row());
        b1 += it.value() * w(1, it.row());
      }
      
      // solve least squares
      if (nonneg) {
        const double a01b1 = a(0, 1) * b1;
        const double a11b0 = a(1, 1) * b0;
        if (a11b0 < a01b1) {
          h(0, i) = 0;
          h(1, i) = b1 / a(1, 1);
        } else {
          const double a01b0 = a(0, 1) * b0;
          const double a00b1 = a(0, 0) * b1;
          if (a00b1 < a01b0) {
            h(0, i) = b0 / a(0, 0);
            h(1, i) = 0;
          } else {
            h(0, i) = (a11b0 - a01b1) / denom;
            h(1, i) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        h(0, i) = (a(1, 1) * b0 - a(0, 1) * b1) / denom;
        h(1, i) = (a(0, 0) * b1 - a(0, 1) * b0) / denom;
      }
    }
    
    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < 2; ++i) {
      d[i] = h.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < h.cols(); ++j) h(i, j) /= d(i);
    }
    
    // update w
    a = h * h.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    wb.setZero();
    for (unsigned int i = 0; i < samples.size(); ++i) {
      // compute right-hand side of linear system, "b", in-place in h
      for (Rcpp::dgCMatrix::InnerIterator it(A, samples[i]); it; ++it){
        wb(0, it.row()) += it.value() * h(0, i);
        wb(1, it.row()) += it.value() * h(1, i);
      }
    }
    
    for (unsigned int i = 0; i < A.rows(); ++i) {
      // solve least squares
      if (nonneg) {
        const double a01b1 = a(0, 1) * wb(1, i);
        const double a11b0 = a(1, 1) * wb(0, i);
        if (a11b0 < a01b1) {
          w(0, i) = 0;
          w(1, i) = wb(1, i) / a(1, 1);
        } else {
          const double a01b0 = a(0, 1) * wb(0, i);
          const double a00b1 = a(0, 0) * wb(1, i);
          if (a00b1 < a01b0) {
            w(0, i) = wb(0, i) / a(0, 0);
            w(1, i) = 0;
          } else {
            w(0, i) = (a11b0 - a01b1) / denom;
            w(1, i) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        w(0, i) = (a(1, 1) * wb(0, i) - a(0, 1) * wb(1, i)) / denom;
        w(1, i) = (a(0, 0) * wb(1, i) - a(0, 1) * wb(0, i)) / denom;
      }
    }
    
    // reset diagonal and scale "w"
    for (unsigned int i = 0; diag && i < 2; ++i) {
      d[i] = w.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
    }
   
    // calculate tolerance and check for convergence
    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);
    if (tol_ < tol) break;
  }
  
  // sort factors by diagonal value
  if (diag && d(0) < d(1)) {
    w.row(1).swap(w.row(0));
    h.row(1).swap(h.row(0));
    const double d1 = d(1); 
    d(1) = d(0); 
    d(0) = d1;
  }
  
  return wdhmodel{w.transpose(), d, h, tol_, it};
}

wdhmodel c_nmf2_dense(
  const Rcpp::NumericMatrix& A,
    Eigen::MatrixXd& w,
    const double tol,
    const bool nonneg,
    const unsigned int maxit,
    const bool verbose,
    const bool diag,
    const std::vector<unsigned int> samples) {
  
  Eigen::MatrixXd h(2, A.cols()), wb(2, A.rows());
  Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
  double tol_ = 1;
  unsigned int it;
  
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  
  for (it = 0; it < maxit; ++it) {
    // update h
    Eigen::MatrixXd w_it = w;
    Eigen::Matrix2d a = w * w.transpose();
    double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    for (unsigned int i = 0; i < samples.size(); ++i) {
      // compute right-hand side of linear system, "b", in-place in h
      double b0 = 0, b1 = 0;
      for (unsigned int it = 0; it < A.rows(); ++it){
        b0 += A(it, samples[i]) * w(0, it);
        b1 += A(it, samples[i]) * w(1, it);
      }
      
      // solve least squares
      if (nonneg) {
        const double a01b1 = a(0, 1) * b1;
        const double a11b0 = a(1, 1) * b0;
        if (a11b0 < a01b1) {
          h(0, i) = 0;
          h(1, i) = b1 / a(1, 1);
        } else {
          const double a01b0 = a(0, 1) * b0;
          const double a00b1 = a(0, 0) * b1;
          if (a00b1 < a01b0) {
            h(0, i) = b0 / a(0, 0);
            h(1, i) = 0;
          } else {
            h(0, i) = (a11b0 - a01b1) / denom;
            h(1, i) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        h(0, i) = (a(1, 1) * b0 - a(0, 1) * b1) / denom;
        h(1, i) = (a(0, 0) * b1 - a(0, 1) * b0) / denom;
      }
    }
    
    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < 2; ++i) {
      d[i] = h.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < h.cols(); ++j) h(i, j) /= d(i);
    }

    // update w
    a = h * h.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    wb.setZero();
    for (unsigned int i = 0; i < samples.size(); ++i) {
      // compute right-hand side of linear system, "b", in-place in h
      for (unsigned int it = 0; it < A.rows(); ++it){
        wb(0, it) += A(it, samples[i]) * h(0, i);
        wb(1, it) += A(it, samples[i]) * h(1, i);
      }
    }
    
    for (unsigned int i = 0; i < A.rows(); ++i) {
      // solve least squares
      if (nonneg) {
        const double a01b1 = a(0, 1) * wb(1, i);
        const double a11b0 = a(1, 1) * wb(0, i);
        if (a11b0 < a01b1) {
          w(0, i) = 0;
          w(1, i) = wb(1, i) / a(1, 1);
        } else {
          const double a01b0 = a(0, 1) * wb(0, i);
          const double a00b1 = a(0, 0) * wb(1, i);
          if (a00b1 < a01b0) {
            w(0, i) = wb(0, i) / a(0, 0);
            w(1, i) = 0;
          } else {
            w(0, i) = (a11b0 - a01b1) / denom;
            w(1, i) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        w(0, i) = (a(1, 1) * wb(0, i) - a(0, 1) * wb(1, i)) / denom;
        w(1, i) = (a(0, 0) * wb(1, i) - a(0, 1) * wb(0, i)) / denom;
      }
    }
    
    // reset diagonal and scale "w"
    for (unsigned int i = 0; diag && i < 2; ++i) {
      d[i] = w.row(i).sum() + 1e-15;
      for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
    }

    // calculate tolerance and check for convergence
    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);
    if (tol_ < tol) break;
  }
  
  // sort factors by diagonal value
  if (diag && d(0) < d(1)) {
    w.row(1).swap(w.row(0));
    h.row(1).swap(h.row(0));
    const double d1 = d(1); 
    d(1) = d(0); 
    d(0) = d1;
  }
  
  return wdhmodel{w.transpose(), d, h, tol_, it};
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf2_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag,
  const std::vector<unsigned int> samples){

  Rcpp::dgCMatrix A(A_S4);
  wdhmodel m = c_nmf2_sparse(A, w_init, tol, nonneg, maxit, verbose, diag, samples);
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = m.it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf2_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag,
  const std::vector<unsigned int> samples){

  wdhmodel m = c_nmf2_dense(A, w_init, tol, nonneg, maxit, verbose, diag, samples);
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = m.it);
}