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
template <class T>
class svd {
   private:
    T& A;
    T t_A;
    Rcpp::SparseMatrix mask_matrix = Rcpp::SparseMatrix(), t_mask_matrix;
    Rcpp::SparseMatrix link_matrix_u = Rcpp::SparseMatrix(), link_matrix_v = Rcpp::SparseMatrix();
    Eigen::MatrixXd u;
    Eigen::VectorXd s;
    Eigen::MatrixXd v;
    double tol_ = -1, mse_ = 0;
    unsigned int iter_ = 0, best_model_ = 0;
    bool mask = false, mask_zeros = false, symmetric = false, transposed = false;

   public:
    bool verbose = true;
    unsigned int maxit = 100, threads = 0;
    std::vector<double> L1 = std::vector<double>(2), L2 = std::vector<double>(2);
    std::vector<bool> link = {false, false};
    double upper_bound = 0;  // set to 0 or negative to not impose upper bound limit

    double tol = 1e-4;

    // CONSTRUCTORS
    // constructor for initialization with a randomly generated "w" matrix
    svd(T& A, const unsigned int k, const unsigned int seed = 0) : A(A) {
        u = randomMatrix(k, A.rows(), seed);
        v = Eigen::MatrixXd(k, A.cols());
        s = Eigen::VectorXd::Ones(k);
        isSymmetric();
    }

    // constructor for initialization with an initial "w" matrix
    svd(T& A, Eigen::MatrixXd u) : A(A), u(u) {
        if (A.rows() != u.cols()) Rcpp::stop("number of rows in 'A' and columns in 'u' are not equal!");
        s = Eigen::VectorXd::Ones(u.rows());
        v = Eigen::MatrixXd(u.rows(), A.cols());
        isSymmetric();
    }

    // constructor for initialization with a fully-specified model
    svd(T& A, Eigen::MatrixXd u, Eigen::VectorXd s, Eigen::MatrixXd v) : A(A), u(u), s(s), v(v) {
        if (A.rows() != u.cols()) Rcpp::stop("dimensions of 'u' and 'A' are not compatible");
        if (A.cols() != v.cols()) Rcpp::stop("dimensions of 'v' and 'A' are not compatible");
        if (u.rows() != v.rows()) Rcpp::stop("rank of 'u' and 'v' are not equal!");
        if (s.size() != u.rows()) Rcpp::stop("length of 's' is not equal to rank of 'u' and 'v'");
        isSymmetric();
    }

    // SETTERS
    void isSymmetric();
    void maskZeros() {
        if (mask) Rcpp::stop("a masking function has already been specified");
        mask_zeros = true;
    }

    void maskMatrix(Rcpp::SparseMatrix& m) {
        if (mask) Rcpp::stop("a masking function has already been specified");
        if (m.rows() != A.rows() || m.cols() != A.cols()) Rcpp::stop("dimensions of masking matrix and 'A' are not equivalent");
        if (mask_zeros) Rcpp::stop("you already specified to mask zeros. You cannot also supply a masking matrix.");
        mask = true;
        mask_matrix = m;
        if (symmetric) symmetric = mask_matrix.isAppxSymmetric();
    }

    // TODO: Whats going on here
    void linkH(Rcpp::SparseMatrix& l) {
        if (l.cols() == A.cols())
            link_matrix_h = l;
        else
            Rcpp::stop("dimensions of linking matrix and 'A' are not equivalent");
        link[1] = true;
    }

    void linkW(Rcpp::SparseMatrix& l) {
        if (l.cols() == A.rows())
            link_matrix_w = l;
        else
            Rcpp::stop("dimensions of linking matrix and 'A' are not equivalent");
        link[0] = true;
    }

    // impose upper maximum limit on NNLS solutions
    void upperBound(double upperbound) {
        upper_bound = upperbound;
    }

    // GETTERS
    Eigen::MatrixXd matrixU() { return u; }
    Eigen::VectorXd vectorS() { return s; }
    Eigen::MatrixXd matrixV() { return v; }
    double fit_tol() { return tol_; }
    unsigned int fit_iter() { return iter_; }
    double fit_mse() { return mse_; }
    unsigned int best_model() { return best_model_; }


    // scale rows in "w" to sum to 1, where "d" is rowsums of "w"
    void scaleU(int k) {
        if(k >= u.rows()){
            Rcpp::stop("cannot scale u for k greater than SVD rank");
        }
        s = u.rowwise().sum();
        s.array() += TINY_NUM;
        for (unsigned int i = 0; i < k; ++i)
            for (unsigned int j = 0; j < u.cols(); ++j)
                u(i, j) /= s(i);
    };

    // scale rows in "h" to sum to 1, where "d" is rowsums of "h"
    void scaleV(int k) {
        if(k >= v.rows()){
            Rcpp::stop("cannot scale v for k greater than SVD rank");
        }
        s = v.rowwise().sum();
        s.array() += TINY_NUM;
        for (unsigned int i = 0; i < k; ++i)
            for (unsigned int j = 0; j < v.cols(); ++j)
                v(i, j) /= s(i);
    };

    void predictV(int k) {
        predict(A, mask_matrix, link_matrix_v, u, v, L1[1], L2[1], threads, mask_zeros, mask, link[1], upper_bound, k);
    }

    // project "h" onto "t(A)" to solve for "w"
    void predictW(int k) {
        if (symmetric)
            predict(A, mask_matrix, link_matrix_u, v, u, L1[0], L2[0], threads, mask_zeros, mask, link[0], upper_bound, k);
        else {
            if (!transposed) {
                t_A = A.transpose();
                if (mask) t_mask_matrix = mask_matrix.transpose();
                transposed = true;
            }
            predict(t_A, t_mask_matrix, link_matrix_u, v, u, L1[0], L2[0], threads, mask_zeros, mask, link[0], upper_bound, k);
        }
    };

    // requires specialized dense and sparse backends
    double mse();
    double mse_masked();

    // fit the model by alternating least squares projections
    void fit() {
        if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");

        for(int k = 0; k < w.rows(); ++k){
            // alternating least squares updates
            for (; iter_ < maxit; ++iter_) {
                Eigen::MatrixXd u_it = u;
                predictV(k);  // update "v[:k]"
                scaleV(k);
                predictU(k);  // update "w[:k]"
                scaleU(k);
                tol_ = cor(u, u_it);  // correlation between "w" across consecutive iterations
                if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
                if (tol_ < tol) break;
                Rcpp::checkUserInterrupt();
            }

            if (tol_ > tol && iter_ == maxit && verbose)
                Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
        }
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
            tol_ = 1;
            iter_ = 0;
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
void svd<Rcpp::SparseMatrix>::isSymmetric() {
    symmetric = A.isAppxSymmetric();
}

template <>
void svd<Eigen::MatrixXd>::isSymmetric() {
    symmetric = isAppxSymmetric(A);
}

template <>
double svd<Rcpp::SparseMatrix>::mse() {
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
            for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                losses(i) += std::pow(wh_i(iter.row()) - iter.value(), 2);
        } else {
            for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                wh_i(iter.row()) -= iter.value();
            if (mask) {
                std::vector<unsigned int> m = mask_matrix.InnerIndices(i);
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
double svd<Eigen::MatrixXd>::mse() {
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
                std::vector<unsigned int> m = mask_matrix.InnerIndices(i);
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
double svd<Rcpp::SparseMatrix>::mse_masked() {
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
        std::vector<unsigned int> masked_rows = mask_matrix.InnerIndices(i);
        if (masked_rows.size() > 0) {
            for (Rcpp::SparseMatrix::InnerIteratorInRange iter(A, i, masked_rows); iter; ++iter) {
                losses(i) += std::pow((w0.row(iter.row()) * h.col(i)) - iter.value(), 2);
            }
            // get masked rows that are also zero in A.col(i)
            std::vector<unsigned int> zero_rows = A.emptyInnerIndices(i);
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
double svd<Eigen::MatrixXd>::mse_masked() {
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
        std::vector<unsigned int> masked_rows = mask_matrix.InnerIndices(i);
        for (unsigned int it = 0; it < masked_rows.size(); ++it) {
            const unsigned int row = masked_rows[it];
            losses(i) += std::pow((w0.row(row) * h.col(i)) - A(row, i), 2);
        }
    }
    return losses.sum() / mask_matrix.i.size();
};
}  // namespace RcppML



// solve for 'h' given dense 'A' in 'A = wh'
// Applies no non-negativity constraints and only solves up to the Kth column of h
void predict_svd(Eigen::MatrixXd& A, Rcpp::SparseMatrix& m, Rcpp::SparseMatrix& l, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const double L1, const double L2,
             const unsigned int threads, const bool mask_zeros, const bool mask, const bool link, const double upper_bound = 0,
             const unsigned int k) {
    if(k >= v.rows()){
        Rcpp::stop("cannot make prediction for k greater than SVD rank");
    }
    
    if (!mask_zeros && !mask) {
        // GENERAL RANK IMPLEMENTATION
        Eigen::MatrixXd a = w * w.transpose();
        a.diagonal().array() += TINY_NUM + L2;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
    for (unsigned int i = 0; i < k; ++i) {
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
