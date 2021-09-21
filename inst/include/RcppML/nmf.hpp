// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_nmf
#define RcppML_nmf

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

#ifndef RcppML_project
#include <RcppML/project.hpp>
#endif

inline bool is_appx_symmetric(RcppML::SparseMatrix& A) {
  if (A.rows() == A.cols()) {
    RcppML::SparseMatrix::InnerIterator col_it(A, 0);
    RcppML::SparseMatrix::InnerRowIterator row_it(A, 0);
    while (++col_it && ++row_it)
      if (col_it.value() != row_it.value())
        return false;
    return true;
  } else return false;
}

inline bool is_appx_symmetric(Eigen::MatrixXd& A) {
  if (A.rows() == A.cols()) {
    for (int i = 0; i < A.cols(); ++i)
      if (A(i, 0) != A(0, i))
        return false;
    return true;
  } else return false;
}

namespace RcppML {
  class MatrixFactorization {
    public:
    Eigen::MatrixXd w;
    Eigen::VectorXd d;
    Eigen::MatrixXd h;
    double tol_ = -1;
    unsigned int iter_ = 0;
    bool nonneg = true, updateInPlace = false, diag = true, verbose = true, mask_zeros = true;
    double L1_w = 0, L1_h = 0, tol = 1e-4;
    unsigned int maxit = 100, threads = 0;

    MatrixFactorization(const unsigned int k, const unsigned int nrow, const unsigned int ncol, const unsigned int seed = 0) {
      w = randomMatrix(k, nrow, seed);
      h = Eigen::MatrixXd(k, ncol);
      d = Eigen::VectorXd::Ones(k);
    }

    MatrixFactorization(Eigen::MatrixXd& w, Eigen::VectorXd& d, Eigen::MatrixXd& h) : w(w), d(d), h(h) {
      if (w.rows() != h.rows()) Rcpp::stop("number of rows in 'w' and 'h' are not equal!");
      if (d.size() != w.rows()) Rcpp::stop("length of 'd' is not equal to number of rows in 'w' and 'h'");
    }

    Eigen::MatrixXd matrixW() { return w; }
    Eigen::VectorXd vectorD() { return d; }
    Eigen::MatrixXd matrixH() { return h; }
    double fit_tol() { return tol_; }
    unsigned int fit_iter() { return iter_; }

    // update "h"
    void projectH(RcppML::SparseMatrix& A) { project(A, w, h, nonneg, L1_h, threads, mask_zeros); }
    void projectH(Eigen::MatrixXd& A) { project(A, w, h, nonneg, L1_h, threads, mask_zeros); }

    // update "w"
    void projectW(RcppML::SparseMatrix& A) {
      if (is_appx_symmetric(A)) project(A, h, w, nonneg, L1_w, threads, mask_zeros);
      else projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros);
    }
    void projectW(Eigen::MatrixXd& A) {
      if (is_appx_symmetric(A)) project(A, h, w, nonneg, L1_w, threads, mask_zeros);
      else projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros);
    }

    double mse(RcppML::SparseMatrix& A) {
      Eigen::MatrixXd w0 = w.transpose();
      // multiply w by diagonal
      for (unsigned int i = 0; i < w0.cols(); ++i)
        for (unsigned int j = 0; j < w0.rows(); ++j)
          w0(j, i) *= d(i);

      // calculate total loss with parallelization across samples
      Eigen::VectorXd losses(h.cols());
      losses.setZero();
      double sq_loss = 0;
      if (!mask_zeros) {
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
          Eigen::VectorXd wh_i = w0 * h.col(i);
          for (RcppML::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
            wh_i(iter.row()) -= iter.value();
          sq_loss += wh_i.array().square().sum();
        }
      } else {
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
          Eigen::VectorXd wh_i = w0 * h.col(i);
          for (RcppML::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
            sq_loss += std::pow(wh_i(iter.row()) - iter.value(), 2);
        }
      }
      return sq_loss / (h.cols() * w.cols());
    }

    double mse(Eigen::MatrixXd& A) {
      if(mask_zeros) Rcpp::stop("mask_zeros = TRUE is not supported for mse(Eigen::MatrixXd)");
      Eigen::MatrixXd w0 = w.transpose();
      for (unsigned int i = 0; i < w0.cols(); ++i)
        for (unsigned int j = 0; j < w0.rows(); ++j)
          w0(j, i) *= d(i);
      Eigen::VectorXd losses(h.cols());
      losses.setZero();
      #ifdef _OPENMP
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      #endif
      for (unsigned int i = 0; i < h.cols(); ++i) {
        Eigen::VectorXd wh_i = w0 * h.col(i);
        for (int j = 0; j < A.rows(); ++j)
          wh_i(j) -= A(j, i);
        for (unsigned int j = 0; j < wh_i.size(); ++j)
          losses(i) += std::pow(wh_i(j), 2);
      }
      return losses.sum() / (h.cols() * w.cols());
    }

    void sortByDiagonal() {
      if (w.rows() == 2 && d(0) < d(1)) {
        w.row(1).swap(w.row(0));
        h.row(1).swap(h.row(0));
        const double d1 = d(1);
        d(1) = d(0);
        d(0) = d1;
      } else if (w.rows() > 2) {
        std::vector<int> indx = sort_index(d);
        w = reorder_rows(w, indx);
        d = reorder(d, indx);
        h = reorder_rows(h, indx);
      }
    }

    // scale rows in "w" to sum to 1, where "d" is rowsums of "w"
    void scaleW() {
      d = w.rowwise().sum();
      d.array() += TINY_NUM;
      for (unsigned int i = 0; i < w.rows(); ++i)
        for (unsigned int j = 0; j < w.cols(); ++j)
          w(i, j) /= d(i);
    }
    void scaleH() {
      d = h.rowwise().sum();
      d.array() += TINY_NUM;
      for (unsigned int i = 0; i < h.rows(); ++i)
        for (unsigned int j = 0; j < h.cols(); ++j)
          h(i, j) /= d(i);
    }

    // fit the model by alternating least squares projections
    void fit(RcppML::SparseMatrix& A) {
      if(mask_zeros && updateInPlace) {
        Rcpp::warning("'mask_zeros = TRUE' is not supported when 'updateInPlace = true'. Setting 'updateInPlace = false'");
        updateInPlace = false;
      } else if(mask_zeros && w.rows() < 3) Rcpp::stop("'mask_zeros = TRUE' is not supported when k = 1 or 2");
      if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
      RcppML::SparseMatrix At;
      bool is_A_symmetric = is_appx_symmetric(A);
      if (!is_A_symmetric && !updateInPlace) At = A.t();
      for (; iter_ < maxit; ++iter_) { // alternating least squares updates
        Eigen::MatrixXd w_it = w;

        // update "h"
        project(A, w, h, nonneg, L1_h, threads, mask_zeros);
        if (diag) scaleH(); // reset diagonal and scale "h"

        // update "w"
        if (is_A_symmetric) project(A, h, w, nonneg, L1_w, threads, mask_zeros);
        else if (updateInPlace) projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros);
        else project(At, h, w, nonneg, L1_w, threads, mask_zeros);
        if (diag) scaleW(); // reset diagonal and scale "w"

        tol_ = cor(w, w_it); // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        if (tol_ < tol) break;
        Rcpp::checkUserInterrupt();
      }
      if (tol_ > tol && iter_ == maxit && verbose)
        Rprintf("\n convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)", iter_, tol_, tol);
      if (diag) sortByDiagonal();
    }

    // fit the model by alternating least squares projections
    void fit(Eigen::MatrixXd& A) {
      if(mask_zeros) Rcpp::stop("'mask_zeros = TRUE' is not supported for fit(Eigen::MatrixXd)");
      if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
      Eigen::MatrixXd At;
      bool is_A_symmetric = is_appx_symmetric(A);
      if (!is_A_symmetric && !updateInPlace) At = A.transpose();
      for (; iter_ < maxit; ++iter_) { // alternating least squares updates
        Eigen::MatrixXd w_it = w;

        // update "h"
        project(A, w, h, nonneg, L1_h, threads, mask_zeros);
        if (diag) scaleH(); // reset diagonal and scale "h"

        // update "w"
        if (is_A_symmetric) project(A, h, w, nonneg, L1_w, threads, mask_zeros);
        else if (updateInPlace) projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros);
        else project(At, h, w, nonneg, L1_w, threads, mask_zeros);
        if (diag) scaleW(); // reset diagonal and scale "w"

        tol_ = cor(w, w_it); // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        if (tol_ < tol) break;
        Rcpp::checkUserInterrupt();
      }
      if (tol_ > tol && iter_ == maxit && verbose)
        Rprintf("\n convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)", iter_, tol_, tol);
      if (diag) sortByDiagonal();
    }
  };
}

#endif