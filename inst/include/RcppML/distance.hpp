// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_dist
#define RcppML_dist

#ifndef RcppML_common
#include <RcppMLCommon.h>
#endif

// sparse/dense column-wise distance calculation between two matrices
Eigen::MatrixXd distance(Rcpp::SparseMatrix& A, Eigen::MatrixXd& B, std::string method, const unsigned int threads) {
    Eigen::MatrixXd dists(A.cols(), B.cols());
    if (method == "euclidean") {
#pragma omp parallel for num_threads(threads)
        for (unsigned int i = 0; i < B.cols(); ++i) {
            for (unsigned int j = 0; j < A.cols(); ++j) {
                double dist = B.col(i).array().square().sum();
                for (Rcpp::SparseMatrix::InnerIterator it(A, j); it; ++it)
                    dist += std::pow(B(it.row(), i) - it.value(), 2) - std::pow(B(it.row(), i), 2);
                dists(j, i) = dist;
            }
        }
    }
    return dists;
}

// sparse/sparse column-wise distance calculation between two matrices
Eigen::MatrixXd distance(Rcpp::SparseMatrix& A, Rcpp::SparseMatrix& B, std::string method, const unsigned int threads) {
    Eigen::MatrixXd dists(A.cols(), B.cols());
    if (method == "euclidean") {
#pragma omp parallel for num_threads(threads)
        for (unsigned int i = 0; i < B.cols(); ++i) {
            for (unsigned int j = 0; j < A.cols(); ++j) {
                double dist = 0;
                for (Rcpp::SparseMatrix::InnerIterator it(A, j); it; ++it)
                    dist += it.value() * it.value();
                for (Rcpp::SparseMatrix::InnerIterator it(B, j); it; ++it)
                    dist += it.value() * it.value();
                Rcpp::SparseMatrix::InnerIterator it_A(A, j), it_B(B, i);
                while (it_A && it_B) {
                    if (it_A.row() < it_B.row())
                        ++it_A;
                    else if (it_A.row() > it_B.row())
                        ++it_B;
                    else {
                        dist -= it_A.value() * it_A.value();
                        dist -= it_B.value() * it_B.value();
                        dist += std::pow(it_A.value() - it_B.value(), 2);
                        ++it_A;
                        ++it_B;
                    }
                }
                dists(j, i) = std::sqrt(dist);
            }
        }
    }
    return dists;
}

// dense/dense column-wise distance calculation between two matrices
inline Eigen::MatrixXd distance(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::string method, const unsigned int threads) {
    Eigen::MatrixXd dists(A.cols(), B.cols());
    if (method == "euclidean") {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (unsigned int i = 0; i < B.cols(); ++i) {
            for (unsigned int j = 0; j < A.cols(); ++j) {
                dists(j, i) = (A.col(j) - B.col(i)).array().square().sum();
            }
        }
    }
    return dists;
}

// sparse column-wise distance calculation between two matrices
inline Eigen::MatrixXd distance(Rcpp::SparseMatrix& A, std::string method, const unsigned int threads) {
    Eigen::MatrixXd dists(A.cols(), A.cols());

    if (method == "euclidean") {
        Eigen::VectorXd sq_colsums(A.cols());
        for (int col = 0; col < A.cols(); ++col)
            for (Rcpp::SparseMatrix::InnerIterator it(A, col); it; ++it)
                sq_colsums(col) += it.value() * it.value();

#pragma omp parallel for num_threads(threads)
        for (unsigned int i = 0; i < (A.cols() - 1); ++i) {
            for (unsigned int j = (i + 1); j < A.cols(); ++j) {
                double dist = sq_colsums(i) + sq_colsums(j);
                Rcpp::SparseMatrix::InnerIterator it1(A, i), it2(A, j);
                while (it1 && it2) {
                    if (it1.row() < it2.row())
                        ++it1;
                    else if (it1.row() > it2.row())
                        ++it2;
                    else {
                        dist -= it1.value() * it1.value();
                        dist -= it2.value() * it2.value();
                        dist += std::pow(it1.value() - it2.value(), 2);
                        ++it1;
                        ++it2;
                    }
                }
                dists(i, j) = std::sqrt(dist);
                dists(j, i) = dists(i, j);
            }
        }
    }
    dists.diagonal().array() = 1;
    return dists;
}

#endif