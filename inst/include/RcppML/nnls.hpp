// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_nnls
#define RcppML_nnls

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

inline Eigen::VectorXd c_nnls(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, const Eigen::LLT<Eigen::MatrixXd, 1>& a_llt,
                              const bool nonneg, unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol) {

  // initialize with unconstrained least squares solution
  Eigen::VectorXd x = a_llt.solve(b);
  if (!nonneg) return x;

  // iterative feasible set reduction while unconstrained least squares solutions at feasible indices contain negative values
  for (unsigned int it = 0; it < fast_maxit && (x.array() < 0).any(); ++it) {
    // get indices in "x" greater than zero (the "feasible set")
    Eigen::VectorXi gtz_ind = find_gtz(x); // find indices in "x" > 0
    // subset "a" and "b" to those indices in the feasible set
    Eigen::VectorXd bsub = subvec(b, gtz_ind);
    Eigen::MatrixXd asub = submat(a, gtz_ind, gtz_ind);
    // solve for those indices in "x"
    Eigen::VectorXd xsub = asub.llt().solve(bsub);
    x.setZero();
    for (unsigned int i = 0; i < gtz_ind.size(); ++i) x(gtz_ind(i)) = xsub(i);
  }

  // refine solution by coordinate descent
  if (cd_maxit > 0) {
    Eigen::VectorXd b0 = a * x;
    b0 -= b;
    double tol = 1, xi, tol_, diff;
    if (nonneg) {
      while (cd_maxit-- > 0 && (2 * tol) > cd_tol) {
        tol = 0;
        for (unsigned int i = 0; i < x.size(); ++i) {
          xi = x(i) - b0(i) / a(i, i);
          if (xi < 0) xi = 0; // impose non-negativity constraints
          diff = xi - x(i);
          if (diff != 0) {
            b0 += a.col(i) * diff;
            tol_ = std::abs(diff) / (xi + x(i) + TINY_NUM);
            if (tol_ > tol) tol = tol_;
            x(i) = xi;
          }
        }
      }
    }
  }
  return x;
}

inline Eigen::VectorXd c_nnls(const Eigen::MatrixXd& a, Eigen::VectorXd& b, unsigned int cd_maxit, const double cd_tol) {

  // refine solution by coordinate descent
  Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
  double tol = 1, xi, tol_, diff;
  while (cd_maxit-- > 0 && (2 * tol) > cd_tol) {
    tol = 0;
    for (unsigned int i = 0; i < x.size(); ++i) {
      xi = x(i) + b(i) / a(i, i);
      if (xi < 0) xi = 0; // impose non-negativity constraints
      diff = xi - x(i);
      if (diff != 0) {
        b -= a.col(i) * diff;
        tol_ = std::abs(diff) / (xi + x(i) + TINY_NUM);
        if (tol_ > tol) tol = tol_;
        x(i) = xi;
      }
    }
  }
  return x;
}

#endif