// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_project
#define RcppML_project

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

#ifndef RcppML_nnls
#include <RcppML/nnls.hpp>
#endif

// solves ax_i = b for a two-variable system of equations, where the "i"th column in "x" contains the solution.
// "denom" is a pre-conditioned denominator for solving by direct substition.
inline void nnls2(const Eigen::Matrix2d& a, const double b0, const double b1, const double denom, Eigen::MatrixXd& x, const unsigned int i, const bool nonneg) {
  // solve least squares
  if (nonneg) {
    const double a01b1 = a(0, 1) * b1;
    const double a11b0 = a(1, 1) * b0;
    if (a11b0 < a01b1) {
      x(0, i) = 0;
      x(1, i) = b1 / a(1, 1);
    } else {
      const double a01b0 = a(0, 1) * b0;
      const double a00b1 = a(0, 0) * b1;
      if (a00b1 < a01b0) {
        x(0, i) = b0 / a(0, 0);
        x(1, i) = 0;
      } else {
        x(0, i) = (a11b0 - a01b1) / denom;
        x(1, i) = (a00b1 - a01b0) / denom;
      }
    }
  } else {
    x(0, i) = (a(1, 1) * b0 - a(0, 1) * b1) / denom;
    x(1, i) = (a(0, 0) * b1 - a(0, 1) * b0) / denom;
  }
}

// solves ax = b for a two-variable system of equations, where "b" is given in the "i"th column of "w", and replaced with the solution "x".
// "denom" is a pre-conditioned denominator for solving by direct substitution.
inline void nnls2InPlace(const Eigen::Matrix2d& a, const double denom, Eigen::MatrixXd& w, const bool nonneg) {
  for (unsigned int i = 0; i < w.cols(); ++i) {
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
}

void project(RcppML::SparseMatrix& A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const bool nonneg, const double L1,
             const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  if (w.rows() == 1) {
    h.setZero();
    double a = 0;
    for (unsigned int i = 0; i < w.cols(); ++i) a += w(0, i) * w(0, i);
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
        h(0, i) += it.value() * w(0, it.row());
      h(0, i) /= a;
    }
  } else if (w.rows() == 2) {
    Eigen::Matrix2d a = w * w.transpose();
    const double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    for (unsigned int i = 0; i < h.cols(); ++i) {
      double b0 = 0, b1 = 0;
      for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it) {
        const double val = it.value();
        const unsigned int r = it.row();
        b0 += val * w(0, r);
        b1 += val * w(1, r);
      }
      nnls2(a, b0, b1, denom, h, i, nonneg);
    }
  } else {
    Eigen::MatrixXd a = w * w.transpose();
    a.diagonal().array() += TINY_NUM;
    unsigned int k = a.rows();
    Eigen::LLT<Eigen::MatrixXd, 1> a_llt;
    if (fast_maxit > 0) a_llt = a.llt();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      Eigen::VectorXd b = Eigen::VectorXd::Zero(k);
      for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
        b += it.value() * w.col(it.row());
      if (L1 != 0) b.array() -= L1;
      // solve least squares
      if (fast_maxit > 0) h.col(i) = c_nnls(a, b, a_llt, nonneg, cd_maxit, fast_maxit, cd_tol);
      else h.col(i) = c_nnls(a, b, cd_maxit, cd_tol);
    }
  }
}

void project(const Rcpp::NumericMatrix& A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const bool nonneg, const double L1,
             const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  if (w.rows() == 1) {
    h.setZero();
    double a = 0;
    for (unsigned int i = 0; i < w.cols(); ++i) a += w(0, i) * w(0, i);
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (int j = 0; j < A.rows(); ++j)
        h(0, i) += A(j, i) * w(0, j);
      h(0, i) /= a;
    }
  } else if (w.rows() == 2) {
    Eigen::Matrix2d a = w * w.transpose();
    const double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    for (unsigned int i = 0; i < h.cols(); ++i) {
      double b0 = 0, b1 = 0;
      for (int j = 0; j < A.rows(); ++j) {
        const double val = A(j, i);
        b0 += val * w(0, j);
        b1 += val * w(1, j);
      }
      nnls2(a, b0, b1, denom, h, i, nonneg);
    }
  } else {
    Eigen::MatrixXd a = w * w.transpose();
    a.diagonal().array() += TINY_NUM;
    unsigned int k = a.rows();
    Eigen::LLT<Eigen::MatrixXd, 1> a_llt;
    if (fast_maxit > 0) a_llt = a.llt();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      Eigen::VectorXd b = Eigen::VectorXd::Zero(k);
      for (int j = 0; j < A.rows(); ++j)
        b += A(j, i) * w.col(j);
      if (L1 != 0) b.array() -= L1;
      // solve least squares
      if (fast_maxit > 0) h.col(i) = c_nnls(a, b, a_llt, nonneg, cd_maxit, fast_maxit, cd_tol);
      else h.col(i) = c_nnls(a, b, cd_maxit, cd_tol);
    }
  }
}

void projectInPlace(RcppML::SparseMatrix& A, const Eigen::MatrixXd& h, Eigen::MatrixXd& w, const bool nonneg, const double L1,
                    const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  const unsigned int k = w.rows();
  if (k == 1) {
    w.setZero();
    double a = 0;
    for (unsigned int i = 0; i < h.cols(); ++i) a += h(0, i) * h(0, i);
    for (unsigned int i = 0; i < h.cols(); ++i)
      for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
        w(0, it.row()) += it.value() * h(0, i);
    for (unsigned int i = 0; i < w.cols(); ++i) w(0, i) /= a;
  } else if (k == 2) {
    Eigen::Matrix2d a = h * h.transpose();
    const double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    w.setZero();
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
        for (unsigned int j = 0; j < 2; ++j)
          w(j, it.row()) += it.value() * h(j, i);
    }
    nnls2InPlace(a, denom, w, nonneg);
  } else {
    Eigen::MatrixXd a = h * h.transpose();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
        for (unsigned int j = 0; j < k; ++j)
          w(j, it.row()) += it.value() * h(j, i);
    }
    Eigen::LLT<Eigen::MatrixXd, 1> a_llt;
    if (fast_maxit > 0) a_llt = a.llt();
    if (L1 != 0) w.array() -= L1;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < w.cols(); ++i) {
      Eigen::VectorXd b = w.col(i);
      // solve least squares
      if (fast_maxit > 0) w.col(i) = c_nnls(a, b, a_llt, nonneg, cd_maxit, fast_maxit, cd_tol);
      else w.col(i) = c_nnls(a, b, cd_maxit, cd_tol);
    }
  }
}

void projectInPlace(const Rcpp::NumericMatrix& A, const Eigen::MatrixXd& h, Eigen::MatrixXd& w, const bool nonneg, const double L1,
                    const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  const unsigned int k = w.rows();
  if (k == 1) {
    w.setZero();
    double a = 0;
    for (unsigned int i = 0; i < h.cols(); ++i) a += h(0, i) * h(0, i);
    for (unsigned int i = 0; i < h.cols(); ++i)
      for (int j = 0; j < A.rows(); ++j)
        w(0, j) += A(j, i) * h(0, i);
    for (unsigned int i = 0; i < w.cols(); ++i) w(0, i) /= a;
  } else if (k == 2) {
    Eigen::Matrix2d a = h * h.transpose();
    const double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    w.setZero();
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (int j = 0; j < A.rows(); ++j)
        for (unsigned int l = 0; l < 2; ++l)
          w(l, j) += A(j, i) * h(l, i);
    }
    nnls2InPlace(a, denom, w, nonneg);
  } else {
    Eigen::MatrixXd a = h * h.transpose();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (int j = 0; j < A.rows(); ++j)
        for (unsigned int l = 0; l < k; ++l)
          w(l, j) += A(j, i) * h(l, i);
    }
    Eigen::LLT<Eigen::MatrixXd, 1> a_llt;
    if (fast_maxit > 0) a_llt = a.llt();
    if (L1 != 0) w.array() -= L1;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < w.cols(); ++i) {
      Eigen::VectorXd b = w.col(i);
      // solve least squares
      if (fast_maxit > 0) w.col(i) = c_nnls(a, b, a_llt, nonneg, cd_maxit, fast_maxit, cd_tol);
      else w.col(i) = c_nnls(a, b, cd_maxit, cd_tol);
    }
  }
}

#endif