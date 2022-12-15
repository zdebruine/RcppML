// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

/* 5/10/2022
 * This approach to calculating w * w.transpose() significantly reduces compute time
 *
 * template <typename T>
 * Eigen::Matrix<T, -1, -1> AAt(Eigen::Matrix<T, -1, -1> A){
 *   Eigen::Matrix<T, -1, -1> a = Eigen::Matrix<T, -1, -1>::Zero(A.rows(), A.rows());
 *   a.selfadjointView<Eigen::Lower>().rankUpdate(A);
 *   a.triangularView<Eigen::Upper>() = a.transpose();
 *   return a;
 * }
 *
 */

#ifndef RcppML_predict
#define RcppML_predict

#ifndef RcppML_nnls
#include <RcppML/nnls.hpp>
#endif

// solve for 'h' given sparse 'A' in 'A = wh'
void predict(Rcpp::SparseMatrix A, Rcpp::SparseMatrix mask_A, Rcpp::SparseMatrix& mask_h, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const double L1, const double L2, const int threads, const bool mask_zeros,
             const bool masking_A, const bool masking_h, const double upper_bound) {
    // set upper_bound = 0 to not impose an upper bound
    if (!mask_zeros) {
        // calculate "a"
        //  * calculate "a" for updates of all columns of "h"
        //  * if masking is applied to "A", we will subtract away the contributions of masked
        //       values in each column update
        Eigen::MatrixXd a = w * w.transpose();
        a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
        for (int i = 0; i < h.cols(); ++i) {
            // if there are no nonzeros in this column of "A", no need to solve anything
            if (A.p[i] == A.p[i + 1]) continue;

            // find the number of masked values in "A.col(i)"
            int num_masked = 0;
            if (masking_A)
                num_masked = mask_A.p[i + 1] - mask_A.p[i];

            Eigen::MatrixXd a_i;

            // calculate "b"
            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
            if (num_masked == 0) {
                // calculate "b" without masking on "A"
                for (Rcpp::SparseMatrix::InnerIterator it(A, i); it; ++it)
                    b += it.value() * w.col(it.row());
            } else {
                // calculate "b" with weighted masking on "A"
                //  * traverse both A.col(i) and mask_A.col(i) similar to a boost ForwardTraversalIterator
                Rcpp::SparseMatrix::InnerIterator it_A(A, i), it_mask(mask_A, i);
                while (it_A) {
                    if (!it_mask || it_A.row() < it_mask.row()) {
                        b += it_A.value() * w.col(it_A.row());
                        ++it_A;
                    } else if (it_mask && it_A.row() == it_mask.row()) {
                        if (it_mask.value() < 1)
                            b += ((it_A.value() * (1 - it_mask.value())) * w.col(it_A.row()));
                        ++it_mask;
                        ++it_A;
                    } else if (it_mask) {
                        ++it_mask;
                    }
                }
                // if masking values in A.col(i), subtract contributions of masked indices from "a"
                //  * we only need to consider columns in "w" that correspond to non-zero rows in "A" because
                //      this code block does not consider the masked_zeros case
                Eigen::MatrixXd w_(w.rows(), num_masked);
                int j = 0;
                for (Rcpp::SparseMatrix::InnerIterator it(mask_A, i); it; ++it, ++j)
                    w_.col(j) = w.col(it.row()) * it.value();
                Eigen::MatrixXd a_ = w_ * w_.transpose();
                a_i = a - a_;
            }

            // apply L1 penalty on "b"
            if (L1 != 0) b.array() -= L1;

            // apply masking on "h"
            if (masking_h) {
                Rcpp::NumericVector mask_h_i = mask_h.col(i);
                for (int k = 0; k < b.size(); ++k)
                    b[k] *= mask_h_i[k];
            }

            // solve nnls equations
            if (upper_bound > 0) {
                (num_masked == 0) ? c_bnnls(a, b, h, i, upper_bound) : c_bnnls(a_i, b, h, i, upper_bound);
            } else {
                (num_masked == 0) ? c_nnls(a, b, h, i) : c_nnls(a_i, b, h, i);
            }
        }
    } else {  // mask_zeros = true
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
        for (int i = 0; i < h.cols(); ++i) {
            if (A.p[i] == A.p[i + 1]) continue;

            int num_masked = 0;
            if (masking_A)
                num_masked = mask_A.p[i + 1] - mask_A.p[i];

            Eigen::VectorXi nnz(A.p[i + 1] - A.p[i]);
            for (int ind = A.p[i], j = 0; j < nnz.size(); ++ind, ++j)
                nnz(j) = A.i[ind];
            Eigen::MatrixXd w_ = submat(w, nnz);

            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
            if (num_masked == 0) {
                for (Rcpp::SparseMatrix::InnerIterator it(A, i); it; ++it)
                    b += it.value() * w.col(it.row());
            } else {
                // subset "w" at non-masked indices in A.col(i) to calculate "a"
                Rcpp::SparseMatrix::InnerIterator it_mask(mask_A, i), it_A(A, i);
                int j = 0;
                while (it_mask && it_A) {
                    if (it_mask.row() == it_A.row()) {
                        w_.col(j) *= (1 - it_mask.value());
                        ++it_mask;
                        ++it_A;
                        ++j;
                    } else if (it_mask.row() < it_A.row()) {
                        ++it_mask;
                    } else {
                        ++it_A;
                    }
                }

                // calculate "b" with masking on "A"
                Rcpp::SparseMatrix::InnerIterator it_mask2(mask_A, i), it_A2(A, i);
                while (it_A) {
                    if (!it_mask2 || it_A2.row() < it_mask2.row()) {
                        b += it_A2.value() * w.col(it_A2.row());
                        ++it_A2;
                    } else if (it_mask2 && it_A2.row() == it_mask2.row()) {
                        if (it_mask2.value() < 1)
                            b += ((it_A2.value() * (1 - it_mask2.value())) * w.col(it_A2.row()));
                        ++it_mask2;
                        ++it_A2;
                    } else if (it_mask2) {
                        ++it_mask2;
                    }
                }
            }
            Eigen::MatrixXd a = w_ * w_.transpose();

            if (L1 != 0) b.array() -= L1;
            a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;
            if (masking_h) {
                Rcpp::NumericVector mask_h_i = mask_h.col(i);
                for (int k = 0; k < b.size(); ++k)
                    b[k] *= mask_h_i[k];
            }
            if (upper_bound > 0) {
                c_bnnls(a, b, h, i, upper_bound);
            } else {
                c_nnls(a, b, h, i);
            }
        }
    }
}

// solve for 'h' given dense 'A' in 'A = wh'
void predict(Eigen::MatrixXd& A, Rcpp::SparseMatrix& m, Rcpp::SparseMatrix& l, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const double L1, const double L2,
             const unsigned int threads, const bool mask_zeros, const bool mask, const bool link, const double upper_bound = 0) {
    if (!mask_zeros && !mask) {
        // GENERAL RANK IMPLEMENTATION
        Eigen::MatrixXd a = w * w.transpose();
        a.diagonal().array() += TINY_NUM + L2;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            // calculate right-hand side of system of equations, "b"
            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
            if (link) {
                for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
                    for (unsigned int it = 0; it < A.rows(); ++it)
                        b(j.row()) += A(it, i) * w(j.row(), it);
                if (L1 != 0)
                    for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
                        b(j.row()) -= L1;
            } else {
                b += w * A.col(i);
                // subtract L1 penalty from "b"
                if (L1 != 0) b.array() -= L1;
            }

            (upper_bound > 0) ? c_bnnls(a, b, h, i, upper_bound) : c_nnls(a, b, h, i);
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
                            for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
                                b(j.row()) += A(it, i) * w(j.row(), it);
                    }
                    if (L1 != 0)
                        for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
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
                (upper_bound > 0) ? c_bnnls(a, b, h, i, upper_bound) : c_nnls(a, b, h, i);
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
            std::vector<unsigned int> masked_rows_ = m.InnerIndices(i);
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
                for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
                    for (unsigned int it = 0; it < A.rows(); ++it)
                        b(j.row()) += A(it, i) * w(j.row(), it);
                // subtract contributions of masked rows from "b"
                for (unsigned int it = 0; it < masked_rows_.size(); ++it)
                    for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
                        b(j.row()) -= A(masked_rows_[it], i) * w(j.row(), masked_rows_[it]);

                if (L1 != 0) {
                    for (Rcpp::SparseMatrix::InnerIterator j(l, i); j; ++j)
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
            (upper_bound > 0) ? c_bnnls(a_, b, h, i, upper_bound) : c_nnls(a_, b, h, i);
        }
    }
}

#endif
