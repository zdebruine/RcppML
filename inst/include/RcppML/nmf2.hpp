// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RcppML_nmf2
#define RcppML_nmf2

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

wdhmodel c_nmf2_sparse(
    Rcpp::dgCMatrix& A,
    Eigen::MatrixXd& w,
    const double tol,
    const bool nonneg,
    const unsigned int maxit,
    const bool verbose,
    const bool diag,
    const std::vector<unsigned int> samples) {
  
  Eigen::MatrixXd h(2, A.cols());
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
        const double val = it.value();
        const unsigned int r = it.row();
        b0 += val * w(0, r);
        b1 += val * w(1, r);
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
    if(diag){
      #pragma omp simd
      for (unsigned int i = 0; i < 2; ++i) {
        d[i] = h.row(i).sum() + 1e-15;
        for (unsigned int j = 0; j < h.cols(); ++j) h(i, j) /= d(i);
      }
    }
    
    // update w
    a = h * h.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    w.setZero();
    for (unsigned int i = 0; i < samples.size(); ++i) {
      // compute right-hand side of linear system, "b", in-place in h
      for (Rcpp::dgCMatrix::InnerIterator it(A, samples[i]); it; ++it)
      #pragma omp simd
        for(unsigned int j = 0; j < 2; ++j)
          w(j, it.row()) += it.value() * h(j, i);
    }
    
    for (unsigned int i = 0; i < A.rows(); ++i) {
      // solve least squares
      if (nonneg) {
        const double a01b1 = a(0, 1) * w(1, i);
        const double a11b0 = a(1, 1) * w(0, i);
        if (a11b0 < a01b1) {
          w(0, i) = 0;
          w(1, i) /= a(1, 1);
        } else {
          const double a01b0 = a(0, 1) * w(0, i);
          const double a00b1 = a(0, 0) * w(1, i);
          if (a00b1 < a01b0) {
            w(0, i) /= a(0, 0);
            w(1, i) = 0;
          } else {
            w(0, i) = (a11b0 - a01b1) / denom;
            w(1, i) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        double b0 = w(0, i);
        w(0, i) = (a(1, 1) * b0 - a(0, 1) * w(1, i)) / denom;
        w(1, i) = (a(0, 0) * w(1, i) - a(0, 1) * b0) / denom;
      }
    }
    
    // reset diagonal and scale "w"
    if(diag){
      #pragma omp simd
      for (unsigned int i = 0; i < 2; ++i) {
        d[i] = w.row(i).sum() + 1e-15;
        for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
      }
    }
    
    // calculate tolerance and check for convergence
    tol_ = cor(w, w_it);
    Rcpp::checkUserInterrupt();
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

   Eigen::MatrixXd h(2, A.cols());
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
        const double val = A(it, samples[i]);
        b0 += val * w(0, it);
        b1 += val * w(1, it);
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
    if(diag){
      #pragma omp simd
      for (unsigned int i = 0; i < 2; ++i) {
        d[i] = h.row(i).sum() + 1e-15;
        for (unsigned int j = 0; j < h.cols(); ++j) h(i, j) /= d(i);
      }
    }
    
    // update w
    a = h * h.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    w.setZero();
    for (unsigned int i = 0; i < samples.size(); ++i) {
      // compute right-hand side of linear system, "b", in-place in h
      for (unsigned int it = 0; it < A.rows(); ++it)
        #pragma omp simd
        for(unsigned int j = 0; j < 2; ++j)
          w(j, it) += A(it, samples[i]) * h(j, i);
    }
    
    for (unsigned int i = 0; i < A.rows(); ++i) {
      // solve least squares
      if (nonneg) {
        const double a01b1 = a(0, 1) * w(1, i);
        const double a11b0 = a(1, 1) * w(0, i);
        if (a11b0 < a01b1) {
          w(0, i) = 0;
          w(1, i) /= a(1, 1);
        } else {
          const double a01b0 = a(0, 1) * w(0, i);
          const double a00b1 = a(0, 0) * w(1, i);
          if (a00b1 < a01b0) {
            w(0, i) /= a(0, 0);
            w(1, i) = 0;
          } else {
            w(0, i) = (a11b0 - a01b1) / denom;
            w(1, i) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        double b0 = w(0, i);
        w(0, i) = (a(1, 1) * b0 - a(0, 1) * w(1, i)) / denom;
        w(1, i) = (a(0, 0) * w(1, i) - a(0, 1) * b0) / denom;
      }
    }
    
    // reset diagonal and scale "w"
    if(diag){
      #pragma omp simd
      for (unsigned int i = 0; i < 2; ++i) {
        d[i] = w.row(i).sum() + 1e-15;
        for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
      }
    }
    
    // calculate tolerance and check for convergence
    tol_ = cor(w, w_it);
    Rcpp::checkUserInterrupt();
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

#endif