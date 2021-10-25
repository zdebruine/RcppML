// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_nmfSparse
#define RcppML_nmfSparse

#ifndef RcppML_predict
#include <RcppML/predict.hpp>
#endif

namespace RcppML {
    template <class T> class nmf {
    private:
        T& A;
        T t_A;
        SparsePatternMatrix mask_matrix = SparsePatternMatrix(), t_mask_matrix;
        Eigen::MatrixXd w;
        Eigen::VectorXd d;
        Eigen::MatrixXd h;
        double tol_ = -1, mse_ = 0;
        unsigned int iter_ = 0, best_model_ = 0;
        bool mask = false, mask_zeros = false, symmetric = false, transposed = false;

    public:
        bool nonneg = true, diag = true, verbose = true;
        unsigned int maxit = 100, threads = 0;
        std::vector<double> L1 = std::vector<double>(2), L2 = std::vector<double>(2);
        std::vector<bool> link = { false, false };
        bool sort_model = true;

        double tol = 1e-4;

        // CONSTRUCTORS
        // constructor for initialization with a randomly generated "w" matrix
        nmf(T& A, const unsigned int k, const unsigned int seed = 0) : A(A) {
            w = randomMatrix(k, A.rows(), seed);
            h = Eigen::MatrixXd(k, A.cols());
            d = Eigen::VectorXd::Ones(k);
            isSymmetric();
        }

        // constructor for initialization with an initial "w" matrix
        nmf(T& A, Eigen::MatrixXd w) : A(A), w(w) {
            if (A.rows() != w.cols()) Rcpp::stop("number of rows in 'A' and columns in 'w' are not equal!");
            d = Eigen::VectorXd::Ones(w.rows());
            h = Eigen::MatrixXd(w.rows(), A.cols());
            isSymmetric();
        }

        // constructor for initialization with an initial "w" matrix
        nmf(T& A, Eigen::MatrixXd w, Eigen::MatrixXd h) : A(A), w(w), h(h) {
            if (A.rows() != w.cols()) Rcpp::stop("dimensions of 'w' and 'A' are not compatible");
            if (A.cols() != h.cols()) Rcpp::stop("dimensions of 'h' and 'A' are not compatible");
            if (w.rows() != h.rows()) Rcpp::stop("rank of 'w' and 'h' are not equal!");
            d = Eigen::VectorXd::Ones(w.rows());
            isSymmetric();
        }

        // constructor for initialization with a fully-specified model
        nmf(T& A, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h) : A(A), w(w), d(d), h(h) {
            if (A.rows() != w.cols()) Rcpp::stop("dimensions of 'w' and 'A' are not compatible");
            if (A.cols() != h.cols()) Rcpp::stop("dimensions of 'h' and 'A' are not compatible");
            if (w.rows() != h.rows()) Rcpp::stop("rank of 'w' and 'h' are not equal!");
            if (d.size() != w.rows()) Rcpp::stop("length of 'd' is not equal to rank of 'w' and 'h'");
            isSymmetric();
        }

        // SETTERS
        void isSymmetric();
        void maskZeros() {
            if (mask) Rcpp::stop("a masking function has already been specified");
            mask_zeros = true;
        }

        void maskMatrix(SparsePatternMatrix& m) {
            if (mask) Rcpp::stop("a masking function has already been specified");
            if (m.rows() != A.rows() || m.cols() != A.cols()) Rcpp::stop("dimensions of masking matrix and 'A' are not equivalent");
            if (mask_zeros) Rcpp::stop("you already specified to mask zeros. You cannot also supply a masking matrix.");
            mask = true;
            mask_matrix = m;
            if (symmetric) symmetric = mask_matrix.isAppxSymmetric();
        }

        // GETTERS
        Eigen::MatrixXd matrixW() { return w; }
        Eigen::VectorXd vectorD() { return d; }
        Eigen::MatrixXd matrixH() { return h; }
        double fit_tol() { return tol_; }
        unsigned int fit_iter() { return iter_; }
        double fit_mse() { return mse_; }
        unsigned int best_model() { return best_model_; }

        // FUNCTIONS
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
        };

        // scale rows in "h" to sum to 1, where "d" is rowsums of "h"
        void scaleH() {
            d = h.rowwise().sum();
            d.array() += TINY_NUM;
            for (unsigned int i = 0; i < h.rows(); ++i)
                for (unsigned int j = 0; j < h.cols(); ++j)
                    h(i, j) /= d(i);
        };

        // project "w" onto "A" to solve for "h" in the equation "A = wh"
        void predictH() {
            predict(A, mask_matrix, w, h, nonneg, L1[1], L2[1], threads, mask_zeros, mask, link[1]);
        }

        // project "h" onto "t(A)" to solve for "w"
        void predictW() {
            if (symmetric) predict(A, mask_matrix, h, w, nonneg, L1[0], L2[0], threads, mask_zeros, mask, link[0]);
            else {
                if (!transposed) {
                    t_A = A.transpose();
                    if (mask) t_mask_matrix = mask_matrix.transpose();
                    transposed = true;
                }
                predict(t_A, t_mask_matrix, h, w, nonneg, L1[0], L2[0], threads, mask_zeros, mask, link[0]);
            }
        };

        // requires specialized dense and sparse backends
        double mse();
        double mse_masked();

        // fit the model by alternating least squares projections
        void fit() {
            if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");

            // alternating least squares updates
            for (; iter_ < maxit; ++iter_) {
                Eigen::MatrixXd w_it = w;
                predictH(); // update "h"
                if (diag) scaleH();
                predictW(); // update "w"
                if (diag) scaleW();
                tol_ = cor(w, w_it); // correlation between "w" across consecutive iterations
                if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
                if (tol_ < tol) break;
                Rcpp::checkUserInterrupt();
            }

            if (tol_ > tol && iter_ == maxit && verbose)
                Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);

            if (diag && sort_model) sortByDiagonal();
        }

        // fit the model multiple times and return the best one
        void fit_restarts(Rcpp::List& w_init) {
            Eigen::MatrixXd w_best = w;
            Eigen::MatrixXd h_best = h;
            Eigen::VectorXd d_best = d;
            double tol_best = tol_;
            double mse_best = 0;
            for (unsigned int i = 0; i < w_init.length(); ++i) {
                if (verbose) Rprintf("Fitting model %i/%i:", i + 1, w_init.length());
                w = Rcpp::as<Eigen::MatrixXd>(w_init[i]);
                tol_ = 1; iter_ = 0;
                if (w.rows() != h.rows()) Rcpp::stop("rank of 'w' is not equal to rank of 'h'");
                if (w.cols() != A.rows()) Rcpp::stop("dimensions of 'w' and 'A' are not compatible");
                fit();
                mse_ = mse();
                if (verbose) Rprintf("MSE: %8.4e\n\n", mse_);
                if (i == 0 || mse_ < mse_best) {
                    best_model_ = i;
                    w_best = w;
                    h_best = h;
                    d_best = d;
                    tol_best = tol_;
                    mse_best = mse_;
                }
            }
            if (best_model_ != (w_init.length() - 1)) {
                w = w_best;
                h = h_best;
                d = d_best;
                tol_ = tol_best;
                mse_ = mse_best;
            }
        }

    };

    // nmf class methods with specialized dense/sparse backends
    template <>
    void nmf<SparseMatrix>::isSymmetric() {
        symmetric = A.isAppxSymmetric();
    }

    template <>
    void nmf<Eigen::MatrixXd>::isSymmetric() {
        symmetric = isAppxSymmetric(A);
    }

    template <>
    double nmf<SparseMatrix>::mse() {
        Eigen::MatrixXd w0 = w.transpose();
        // multiply w by diagonal
        for (unsigned int i = 0; i < w0.cols(); ++i)
            for (unsigned int j = 0; j < w0.rows(); ++j)
                w0(j, i) *= d(i);

        // compute losses across all samples in parallel
        Eigen::ArrayXd losses(h.cols());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            if (mask_zeros) {
                for (SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                    losses(i) += std::pow(wh_i(iter.row()) - iter.value(), 2);
            } else {
                for (SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                    wh_i(iter.row()) -= iter.value();
                if (mask) {
                    std::vector<unsigned int> m = mask_matrix.nonzeroRowsInCol(i);
                    for (unsigned int it = 0; it < m.size(); ++it)
                        wh_i(m[it]) = 0;
                }
                losses(i) += wh_i.array().square().sum();
            }
        }

        // divide total loss by number of applicable measurements
        if (mask)
            return losses.sum() / ((h.cols() * w.cols()) - mask_matrix.i.size());
        else if (mask_zeros)
            return losses.sum() / A.x.size();
        return losses.sum() / ((h.cols() * w.cols()));
    };

    template <>
    double nmf<Eigen::MatrixXd>::mse() {
        Eigen::MatrixXd w0 = w.transpose();
        // multiply w by diagonal
        for (unsigned int i = 0; i < w0.cols(); ++i)
            for (unsigned int j = 0; j < w0.rows(); ++j)
                w0(j, i) *= d(i);

        // compute losses across all samples in parallel
        Eigen::ArrayXd losses(h.cols());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            if (mask_zeros) {
                for (unsigned int iter = 0; iter < A.rows(); ++iter)
                    if (A(iter, i) != 0)
                        losses(i) += std::pow(wh_i(iter) - A(iter, i), 2);
            } else {
                for (unsigned int iter = 0; iter < A.rows(); ++iter)
                    wh_i(iter) -= A(iter, i);
                if (mask) {
                    std::vector<unsigned int> m = mask_matrix.nonzeroRowsInCol(i);
                    for (unsigned int it = 0; it < m.size(); ++it)
                        wh_i(m[it]) = 0;
                }
                losses(i) += wh_i.array().square().sum();
            }
        }

        // divide total loss by number of applicable measurements
        if (mask)
            return losses.sum() / ((h.cols() * w.cols()) - mask_matrix.i.size());
        else if (mask_zeros)
            return losses.sum() / n_nonzeros(A);
        return losses.sum() / ((h.cols() * w.cols()));
    };

    template <>
    double nmf<SparseMatrix>::mse_masked() {
        if (!mask) Rcpp::stop("'mse_masked' can only be run when a masking matrix has been specified");

        Eigen::MatrixXd w0 = w.transpose();
        for (unsigned int i = 0; i < w0.cols(); ++i)
            for (unsigned int j = 0; j < w0.rows(); ++j)
                w0(j, i) *= d(i);

        Eigen::ArrayXd losses(h.cols());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            std::vector<unsigned int> masked_rows = mask_matrix.nonzeroRowsInCol(i);
            if (masked_rows.size() > 0) {
                for (SparseMatrix::InnerIteratorInRange iter(A, i, masked_rows); iter; ++iter) {
                    losses(i) += std::pow((w0.row(iter.row()) * h.col(i)) - iter.value(), 2);
                }
                // get masked rows that are also zero in A.col(i)
                std::vector<unsigned int> zero_rows = A.zeroRowsInCol(i);
                std::vector<unsigned int> masked_zero_rows;
                std::set_intersection(zero_rows.begin(), zero_rows.end(),
                                      masked_rows.begin(), masked_rows.end(),
                                      std::back_inserter(masked_zero_rows));
                for (unsigned int it = 0; it < masked_zero_rows.size(); ++it)
                    losses(i) += std::pow(w0.row(masked_zero_rows[it]) * h.col(i), 2);
            }
        }
        return losses.sum() / mask_matrix.i.size();
    };

    template <>
    double nmf<Eigen::MatrixXd>::mse_masked() {
        if (!mask) Rcpp::stop("'mse_masked' can only be run when a masking matrix has been specified");

        Eigen::MatrixXd w0 = w.transpose();
        for (unsigned int i = 0; i < w0.cols(); ++i)
            for (unsigned int j = 0; j < w0.rows(); ++j)
                w0(j, i) *= d(i);

        Eigen::ArrayXd losses(h.cols());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            std::vector<unsigned int> masked_rows = mask_matrix.nonzeroRowsInCol(i);
            for (unsigned int it = 0; it < masked_rows.size(); ++it) {
                const unsigned int row = masked_rows[it];
                losses(i) += std::pow((w0.row(row) * h.col(i)) - A(row, i), 2);
            }
        }
        return losses.sum() / mask_matrix.i.size();
    };
}

#endif