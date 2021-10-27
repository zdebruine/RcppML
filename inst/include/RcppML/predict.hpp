// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_predict
#define RcppML_predict

#ifndef RcppML_nnls
#include <RcppML/nnls.hpp>
#endif

// solve for 'h' given sparse 'A' in 'A = wh'
void predict(RcppML::SparseMatrix& A, RcppML::SparsePatternMatrix& m, RcppML::SparsePatternMatrix& l, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const bool nonneg, const double L1, const double L2,
             const unsigned int threads, const bool mask_zeros, const bool mask, const bool link) {

  if (!mask_zeros && !mask) {
    int rank = w.rows();
    // RANK-1 IMPLEMENTATION
    if (rank == 1 && !link) {
      h.setZero();
      // calculate "a" in "ax = b" as the inner product of the "w" vector
      double a = 0;
      for (unsigned int i = 0; i < w.cols(); ++i) a += w(0, i) * w(0, i);
      for (unsigned int i = 0; i < h.cols(); ++i) {
        // calculate "b" in "ax = b" as "wA_j" for all columns "j" in "A"
        for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
          h(0, i) += it.value() * w(0, it.row());
        // solve for "x" in "ax = b" directly
        h(0, i) /= a;
      }
    }

    // RANK-2 IMPLEMENTATION
    else if (rank == 2 && !link) {
      // calculate "a" in "ax = b" as w^Tw
      Eigen::Matrix2d a = w * w.transpose();
      a.diagonal().array() += TINY_NUM + L2;
      const double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
      for (unsigned int i = 0; i < h.cols(); ++i) {
        // calculate "b" in "ax = b" as w^Tw
        Eigen::Vector2d b(0, 0);
        for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it) {
          const double val = it.value();
          const unsigned int r = it.row();
          b(0) += val * w(0, r);
          b(1) += val * w(1, r);
        }
        // solve "ax = b" by direct substitution
        nnls2(a, b, denom, h, i, nonneg);
      }
    }

    // GENERAL RANK IMPLEMENTATION
    else if (rank > 2) {
      Eigen::MatrixXd a = w * w.transpose();
      a.diagonal().array() += TINY_NUM + L2;
      Eigen::LLT<Eigen::MatrixXd, 1> a_llt = a.llt();
      #ifdef _OPENMP
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      #endif
      for (unsigned int i = 0; i < h.cols(); ++i) {
        // calculate right-hand side of system of equations, "b"
        Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

        // linked_update considers information corresponding to non-zero values in "h"
        if (link) {
          for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) += it.value() * w(j.row(), it.row());

          if (L1 != 0)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) -= L1;
        } else {
          for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
            b += it.value() * w.col(it.row());

          // subtract L1 penalty from "b"
          if (L1 != 0) b.array() -= L1;
        }

        // solve least squares equation, ax = b where x is h.col(i)
        h.col(i) = a_llt.solve(b);
        // if unconstrained solution contains negative values, refine by NNLS coordinate descent
        if (nonneg && (h.col(i).array() < 0).any()) {
          b -= a * h.col(i);
          c_nnls(a, b, h, i);
        }
      }
    }
  } else if (mask_zeros) {
    h.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      std::vector<unsigned int> nz = A.nonzeroRowsInCol(i);
      if (nz.size() > 0) {
        Eigen::VectorXi nnz(nz.size());
        for (unsigned int j = 0; j < nz.size(); ++j)
          nnz(j) = (int)nz[j];
        Eigen::MatrixXd w_ = submat(w, nnz);
        Eigen::MatrixXd a = w_ * w_.transpose();
        a.diagonal().array() += TINY_NUM + L2;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
        if (link) {
          for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) += it.value() * w(j.row(), it.row());
          if (L1 != 0)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) -= L1;
        } else {
          for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
            b += it.value() * w.col(it.row());
          if (L1 != 0) b.array() -= L1;
        }
        if (!nonneg) h.col(i) = a.llt().solve(b);
        else c_nnls(a, b, h, i);
      }
    }
  } else if (mask) {
    h.setZero();
    Eigen::MatrixXd a = w * w.transpose();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      // subtract contribution of masked rows from "a"
      std::vector<unsigned int> masked_rows_ = m.nonzeroRowsInCol(i);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
      if (masked_rows_.size() > 0) {
        Eigen::VectorXi masked_rows(masked_rows_.size());
        for (unsigned int j = 0; j < masked_rows.size(); ++j)
          masked_rows(j) = (int)masked_rows_[j];
        Eigen::MatrixXd w_ = submat(w, masked_rows);
        Eigen::MatrixXd a_ = w_ * w_.transpose();
        a_ = a - a_;
        a_.diagonal().array() += TINY_NUM + L2;

        // calculate "b" for all non-masked rows
        if (link) {
          for (RcppML::SparseMatrix::InnerIteratorNotInRange it(A, i, masked_rows_); it; ++it)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) += it.value() * w(j.row(), it.row());
          if (L1 != 0)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) -= L1;
        } else {
          for (RcppML::SparseMatrix::InnerIteratorNotInRange it(A, i, masked_rows_); it; ++it)
            b += it.value() * w.col(it.row());
          if (L1 != 0) b.array() -= L1;
        }

        if (!nonneg) h.col(i) = a_.llt().solve(b);
        else c_nnls(a_, b, h, i);
      } else {
        for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
          b += it.value() * w.col(it.row());
        if (L1 != 0) b.array() -= L1;
        if (!nonneg) h.col(i) = a.llt().solve(b);
        else c_nnls(a, b, h, i);
      }
    }
  }
}

// solve for 'h' given sparse 'A' in 'A = wh'
void predict(Eigen::MatrixXd& A, RcppML::SparsePatternMatrix& m, RcppML::SparsePatternMatrix& l, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const bool nonneg, const double L1, const double L2,
             const unsigned int threads, const bool mask_zeros, const bool mask, const bool link) {

  if (!mask_zeros && !mask) {
    int rank = w.rows();
    // RANK-1 IMPLEMENTATION
    if (rank == 1 && !link) {
      h.setZero();
      // calculate "a" in "ax = b" as the inner product of the "w" vector
      double a = 0;
      for (unsigned int i = 0; i < w.cols(); ++i) a += w(0, i) * w(0, i);
      for (unsigned int i = 0; i < h.cols(); ++i) {
        // calculate "b" in "ax = b" as "wA_j" for all columns "j" in "A"
        for (unsigned int it = 0; it < A.rows(); ++it)
          h(0, i) += A(it, i) * w(0, it);
        // solve for "x" in "ax = b" directly
        h(0, i) /= a;
      }
    }

    // RANK-2 IMPLEMENTATION
    else if (rank == 2 && !link) {
      // calculate "a" in "ax = b" as w^Tw
      Eigen::Matrix2d a = w * w.transpose();
      a.diagonal().array() += TINY_NUM + L2;
      const double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
      for (unsigned int i = 0; i < h.cols(); ++i) {
        // calculate "b" in "ax = b" as w^Tw
        Eigen::Vector2d b(0, 0);
        for (unsigned int it = 0; it < A.rows(); ++it) {
          const double val = A(it, i);
          b(0) += val * w(0, it);
          b(1) += val * w(1, it);
        }
        // solve "ax = b" by direct substitution
        nnls2(a, b, denom, h, i, nonneg);
      }
    }

    // GENERAL RANK IMPLEMENTATION
    else if (rank > 2) {
      Eigen::MatrixXd a = w * w.transpose();
      a.diagonal().array() += TINY_NUM + L2;
      Eigen::LLT<Eigen::MatrixXd, 1> a_llt = a.llt();
      #ifdef _OPENMP
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      #endif
      for (unsigned int i = 0; i < h.cols(); ++i) {
        // calculate right-hand side of system of equations, "b"
        Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
        if (link) {
          for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
            for (unsigned int it = 0; it < A.rows(); ++it)
              b(j.row()) += A(it, i) * w(j.row(), it);
          if (L1 != 0)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) -= L1;
        } else {
          b += w * A.col(i);
          // subtract L1 penalty from "b"
          if (L1 != 0) b.array() -= L1;
        }

        // solve least squares equation, ax = b where x is h.col(i)
        h.col(i) = a_llt.solve(b);
        // if unconstrained solution contains negative values, refine by NNLS coordinate descent
        if (nonneg && (h.col(i).array() < 0).any()) {
          b -= a * h.col(i);
          c_nnls(a, b, h, i);
        }
      }
    }
  } else if (mask_zeros) {
    h.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      std::vector<unsigned int> nz = nonzeroRowsInCol(A, i);
      if (nz.size() > 0) {
        Eigen::VectorXi nnz(nz.size());
        for (unsigned int j = 0; j < nz.size(); ++j)
          nnz(j) = (int)nz[j];
        Eigen::MatrixXd w_ = submat(w, nnz);
        Eigen::MatrixXd a = w_ * w_.transpose();
        a.diagonal().array() += TINY_NUM + L2;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

        if (link) {
          for (unsigned int it = 0; it < A.rows(); ++it) {
            const double val = A(it, i);
            if (val != 0)
              for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
                b(j.row()) += A(it, i) * w(j.row(), it);
          }
          if (L1 != 0)
            for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
              b(j.row()) -= L1;
        } else {
          for (unsigned int it = 0; it < A.rows(); ++it) {
            const double val = A(it, i);
            if (val != 0) {
              b += val * w.col(it);
            }
          }
          if (L1 != 0) b.array() -= L1;
        }

        if (!nonneg) h.col(i) = a.llt().solve(b);
        else c_nnls(a, b, h, i);
      }
    }
  } else if (mask) {
    Eigen::MatrixXd a = w * w.transpose();
    h.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      // subtract contribution of masked rows from "a"
      std::vector<unsigned int> masked_rows_ = m.nonzeroRowsInCol(i);
      Eigen::VectorXi masked_rows(masked_rows_.size());
      for (unsigned int j = 0; j < masked_rows.size(); ++j)
        masked_rows(j) = (int)masked_rows_[j];
      Eigen::MatrixXd w_ = submat(w, masked_rows);
      Eigen::MatrixXd a_ = w_ * w_.transpose();
      a_ = a - a_;
      a_.diagonal().array() += TINY_NUM + L2;

      // calculate "b" for all non-masked rows
      Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
      if (link) {
        for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
          for (unsigned int it = 0; it < A.rows(); ++it)
            b(j.row()) += A(it, i) * w(j.row(), it);
        // subtract contributions of masked rows from "b"
        for (unsigned int it = 0; it < masked_rows_.size(); ++it)
          for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
            b(j.row()) -= A(masked_rows_[it], i) * w(j.row(), masked_rows_[it]);

        if (L1 != 0) {
          for (RcppML::SparsePatternMatrix::InnerIterator j(l, i); j; ++j)
            b(j.row()) -= L1;
        }
      } else {
        b += w * A.col(i);
        // subtract contributions of masked rows from "b"
        for (unsigned int it = 0; it < masked_rows_.size(); ++it)
          b -= A(masked_rows_[it], i) * w.col(masked_rows_[it]);
        if (L1 != 0) b.array() -= L1;
      }

      // solve system with least squares
      if (!nonneg) h.col(i) = a_.llt().solve(b);
      else c_nnls(a_, b, h, i);
    }
  }
}

#endif