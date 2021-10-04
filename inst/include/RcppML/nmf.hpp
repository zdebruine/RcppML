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
    std::vector<unsigned int> samples, features;
    double tol_ = -1;
    unsigned int iter_ = 0;
    bool nonneg = true, updateInPlace = false, diag = true, verbose = true, mask_zeros = false;
    double L1_w = 0, L1_h = 0, tol = 1e-4;
    unsigned int maxit = 100, threads = 0;

    // constructor for random initialization of "w"
    MatrixFactorization(const unsigned int k, const unsigned int nrow, const unsigned int ncol, const unsigned int seed = 0) {
      w = randomMatrix(k, nrow, seed);
      h = Eigen::MatrixXd(k, ncol);
      d = Eigen::VectorXd::Ones(k);
      samples = std::vector<unsigned int>(ncol);
      features = std::vector<unsigned int>(nrow);
      std::iota(samples.begin(), samples.end(), 0);
      std::iota(features.begin(), features.end(), 0);
    }

    // constructor for random initialization of "w"
    MatrixFactorization(const unsigned int k, const std::vector<unsigned int>& samples_, const std::vector<unsigned int>& features_, const unsigned int seed = 0) : samples(samples_), features(features_) {
      w = randomMatrix(k, features.size(), seed);
      h = Eigen::MatrixXd(k, samples.size());
      d = Eigen::VectorXd::Ones(k);
    }

    // constructor for complete initialization
    MatrixFactorization(Eigen::MatrixXd& w, Eigen::VectorXd& d, Eigen::MatrixXd& h) : w(w), d(d), h(h) {
      if (w.rows() != h.rows()) Rcpp::stop("number of rows in 'w' and 'h' are not equal!");
      if (d.size() != w.rows()) Rcpp::stop("length of 'd' is not equal to number of rows in 'w' and 'h'");
      samples = std::vector<unsigned int>(h.cols());
      features = std::vector<unsigned int>(w.cols());
      std::iota(samples.begin(), samples.end(), 0);
      std::iota(features.begin(), features.end(), 0);
    }

    // constructor for detailed initialization for some samples
    MatrixFactorization(Eigen::MatrixXd& w, const std::vector<unsigned int>& samples_, const std::vector<unsigned int>& features_) :
      w(w), samples(samples_), features(features_) {
      d = Eigen::VectorXd::Ones(w.rows());
      h = Eigen::MatrixXd(w.rows(), samples_.size());
    }

    Eigen::MatrixXd matrixW() { return w; }
    Eigen::VectorXd vectorD() { return d; }
    Eigen::MatrixXd matrixH() { return h; }
    double fit_tol() { return tol_; }
    unsigned int fit_iter() { return iter_; }

    // update "h"
    void projectH(RcppML::SparseMatrix& A) { project(A, w, h, nonneg, L1_h, threads, mask_zeros, samples, features); }
    void projectH(Eigen::MatrixXd& A) { project(A, w, h, nonneg, L1_h, threads, mask_zeros, samples, features); }

    // update "w"
    void projectW(RcppML::SparseMatrix& A) {
      if (is_appx_symmetric(A)) project(A, h, w, nonneg, L1_w, threads, mask_zeros, samples, features);
      else projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros, samples, features);
    }
    void projectW(Eigen::MatrixXd& A) {
      if (is_appx_symmetric(A)) project(A, h, w, nonneg, L1_w, threads, mask_zeros, samples, features);
      else projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros, samples, features);
    }

    double mse(RcppML::SparseMatrix& A) {
      bool all_features = (features.size() == A.rows());
      Eigen::MatrixXd w0 = w.transpose();
      // multiply w by diagonal
      for (unsigned int i = 0; i < w0.cols(); ++i)
        for (unsigned int j = 0; j < w0.rows(); ++j)
          w0(j, i) *= d(i);

      // calculate total loss with parallelization across samples
      double sq_loss = 0;
      if (!mask_zeros) {
        if(threads != 1){
          Eigen::ArrayXd losses(h.cols());
          #ifdef _OPENMP
          #pragma omp parallel for num_threads(threads) schedule(dynamic)
          #endif
          for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            if (all_features)
              for (RcppML::SparseMatrix::InnerIterator iter(A, samples[i]); iter; ++iter)
                wh_i(iter.row()) -= iter.value();
            else
              for (RcppML::SparseMatrix::InnerRangedIterator iter(A, samples[i], features); iter; ++iter)
                wh_i(iter.rangedRow()) -= iter.value();
            losses(i) += wh_i.array().square().sum();
          }
          sq_loss = losses.sum();
        } else {
          for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            if (all_features)
              for (RcppML::SparseMatrix::InnerIterator iter(A, samples[i]); iter; ++iter)
                wh_i(iter.row()) -= iter.value();
            else
              for (RcppML::SparseMatrix::InnerRangedIterator iter(A, samples[i], features); iter; ++iter)
                wh_i(iter.rangedRow()) -= iter.value();
            sq_loss += wh_i.array().square().sum();
          }          
        }
      } else {
        if(threads != 1){
          Eigen::ArrayXd losses(h.cols());
          #ifdef _OPENMP
          #pragma omp parallel for num_threads(threads) schedule(dynamic)
          #endif
          for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            if (all_features)
              for (RcppML::SparseMatrix::InnerIterator iter(A, samples[i]); iter; ++iter)
                losses(i) += std::pow(wh_i(iter.row()) - iter.value(), 2);
            else
              for (RcppML::SparseMatrix::InnerRangedIterator iter(A, samples[i], features); iter; ++iter)
                losses(i) += std::pow(wh_i(iter.rangedRow()) - iter.value(), 2);
          }
          sq_loss = losses.sum();
        } else {
            for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            if (all_features)
              for (RcppML::SparseMatrix::InnerIterator iter(A, samples[i]); iter; ++iter)
                sq_loss += std::pow(wh_i(iter.row()) - iter.value(), 2);
            else
              for (RcppML::SparseMatrix::InnerRangedIterator iter(A, samples[i], features); iter; ++iter)
                sq_loss += std::pow(wh_i(iter.rangedRow()) - iter.value(), 2);
          }
        }
      }
      return sq_loss / (h.cols() * w.cols());
    }

    double mse(Eigen::MatrixXd& A) {
      bool all_features = (features.size() == (unsigned int)A.rows());
      if (mask_zeros) Rcpp::stop("mask_zeros = TRUE is not supported for mse(Eigen::MatrixXd)");
      Eigen::MatrixXd w0 = w.transpose();
      for (unsigned int i = 0; i < w0.cols(); ++i)
        for (unsigned int j = 0; j < w0.rows(); ++j)
          w0(j, i) *= d(i);
      double sq_loss = 0;
      if(threads != 1){
        Eigen::ArrayXd losses(h.cols());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
          Eigen::VectorXd wh_i = w0 * h.col(i);
          if (all_features)
            wh_i -= A.col(samples[i]);
          else
            for (unsigned int j = 0; j < w.cols(); ++j)
              wh_i(j) -= A(features[j], samples[i]);
          losses(i) += wh_i.array().square().sum();
        }
        return losses.sum() / (h.cols() * w.cols());
      } else {
        for (unsigned int i = 0; i < h.cols(); ++i) {
          Eigen::VectorXd wh_i = w0 * h.col(i);
          if (all_features)
            wh_i -= A.col(samples[i]);
          else
            for (unsigned int j = 0; j < w.cols(); ++j)
              wh_i(j) -= A(features[j], samples[i]);
          sq_loss += wh_i.array().square().sum();
        }
        return sq_loss / (h.cols() * w.cols());
      }
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
      if (mask_zeros && w.rows() < 3) Rcpp::stop("'mask_zeros = TRUE' is not supported when k = 1 or 2");
      if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
      RcppML::SparseMatrix At;
      bool is_A_symmetric = is_appx_symmetric(A);
      if (!is_A_symmetric && !updateInPlace) At = A.t();
      for (; iter_ < maxit; ++iter_) { // alternating least squares updates
        Eigen::MatrixXd w_it = w;

        // update "h"
        project(A, w, h, nonneg, L1_h, threads, mask_zeros, samples, features);
        if (diag) scaleH(); // reset diagonal and scale "h"

        // update "w"
        if (is_A_symmetric) project(A, h, w, nonneg, L1_w, threads, mask_zeros, features, samples);
        else if (updateInPlace) projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros, samples, features);
        else project(At, h, w, nonneg, L1_w, threads, mask_zeros, features, samples);
        if (diag) scaleW(); // reset diagonal and scale "w"

        tol_ = cor(w, w_it); // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        if (tol_ < tol) break;
        Rcpp::checkUserInterrupt();
      }

      if (tol_ > tol && iter_ == maxit && verbose)
        Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
      if (diag) sortByDiagonal();
    }

    // fit the model by alternating least squares projections
    void fit(Eigen::MatrixXd& A) {
      if (mask_zeros) Rcpp::stop("'mask_zeros = TRUE' is not supported for fit(Eigen::MatrixXd)");
      if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
      Eigen::MatrixXd At;
      bool is_A_symmetric = is_appx_symmetric(A);
      if (!is_A_symmetric && !updateInPlace) At = A.transpose();
      for (; iter_ < maxit; ++iter_) { // alternating least squares updates
        Eigen::MatrixXd w_it = w;

        // update "h"
        project(A, w, h, nonneg, L1_h, threads, mask_zeros, samples, features);
        if (diag) scaleH(); // reset diagonal and scale "h"

        // update "w"
        if (is_A_symmetric) project(A, h, w, nonneg, L1_w, threads, mask_zeros, features, samples);
        else if (updateInPlace) projectInPlace(A, h, w, nonneg, L1_w, threads, mask_zeros, samples, features);
        else project(At, h, w, nonneg, L1_w, threads, mask_zeros, features, samples);
        if (diag) scaleW(); // reset diagonal and scale "w"

        tol_ = cor(w, w_it); // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        if (tol_ < tol) break;
        Rcpp::checkUserInterrupt();
      }
      if (tol_ > tol && iter_ == maxit && verbose)
        Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)", iter_, tol_, tol);
      if (diag) sortByDiagonal();
    }
  };
}

#endif