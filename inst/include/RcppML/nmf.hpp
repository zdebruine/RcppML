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
        unsigned int iter_ = 0;
        bool mask = false, mask_zeros = false, symmetric = false, transposed = false;

    public:
        bool nonneg = true, diag = true, verbose = true;
        unsigned int maxit = 100, threads = 0;
        std::vector<unsigned int> L1 = std::vector<unsigned int>(2);
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
        nmf(T& A, Eigen::MatrixXd w_init) : A(A), w(w_init) {
            if (A.rows() != w_init.cols()) Rcpp::stop("number of rows in 'A' and columns in 'w_init' are not equal!");
            d = Eigen::VectorXd::Ones(w.rows());
            h = Eigen::MatrixXd(w.rows(), A.cols());
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
            predict(A, mask_matrix, w, h, nonneg, L1[1], threads, mask_zeros, mask);
        }

        // project "h" onto "t(A)" to solve for "w"
        void predictW() {
            if (symmetric) predict(A, mask_matrix, h, w, nonneg, L1[0], threads, mask_zeros, mask);
            else {
                if (!transposed) {
                    t_A = A.transpose();
                    if (mask) t_mask_matrix = mask_matrix.transpose();
                    transposed = true;
                }
                predict(t_A, t_mask_matrix, h, w, nonneg, L1[0], threads, mask_zeros, mask);
            }
        };

        // requires specialized dense and sparse backends
        double mse();

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

            if (diag) sortByDiagonal();
        }

        // fit the model multiple times and return the best one
        void fit_restarts(Rcpp::List& w_init) {
            Eigen::MatrixXd w_best = w;
            Eigen::MatrixXd h_best = h;
            Eigen::VectorXd d_best = d;
            double tol_best = tol_;
            double iter_best = iter_;
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
                    if (i != (w_init.length() - 1)) {
                        w_best = w;
                        h_best = h;
                        d_best = d;
                        tol_best = tol_;
                        iter_best = iter_;
                        mse_best = mse_;
                    }
                }
            }
            if (mse_ < mse_best) {
                w = w_best; h = h_best; d = d_best; tol_ = tol_best; iter_ = iter_best; mse_ = mse_best;
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
                    for (int it = 0; it < m.size(); ++it)
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
                    losses(i) += std::pow(wh_i(iter) - A(iter, i), 2);
            } else {
                for (unsigned int iter = 0; iter < A.rows(); ++iter)
                    wh_i(iter) -= A(iter, i);
                if (mask) {
                    std::vector<unsigned int> m = mask_matrix.nonzeroRowsInCol(i);
                    for (int it = 0; it < m.size(); ++it)
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
}

/*
// fit the model by alternating least squares projections
void nmfDense::fit() {
    if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
    if (!symmetric && !transposed) { At = A.transpose(); transposed = true; }
    for (; iter_ < maxit; ++iter_) {
        Eigen::MatrixXd w_it = w;
        predictH();
        if (diag) scaleH();
        predictW();
        if (diag) scaleW();
        tol_ = cor(w, w_it);
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        if (tol_ < tol) break;
        Rcpp::checkUserInterrupt();
    }
    if (tol_ > tol && iter_ == maxit && verbose)
        Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
    if (diag) sortByDiagonal();
}
*/
/*
double mse(Eigen::MatrixXd& A) {
    Eigen::MatrixXd w0 = w.transpose();
    for (unsigned int i = 0; i < w0.cols(); ++i)
        for (unsigned int j = 0; j < w0.rows(); ++j)
            w0(j, i) *= d(i);
    double sq_loss = 0;
    if (threads != 1) {
        Eigen::ArrayXd losses(h.cols());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            wh_i -= A.col(i);
            losses(i) += wh_i.array().square().sum();
        }
        return losses.sum() / (h.cols() * w.cols());
    } else {
        for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::VectorXd wh_i = w0 * h.col(i);
            wh_i -= A.col(i);
            sq_loss += wh_i.array().square().sum();
        }
        return sq_loss / (h.cols() * w.cols());
    }
}
*/

#endif