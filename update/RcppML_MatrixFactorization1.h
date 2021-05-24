// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_MATRIXFACTORIZATION1_H
#define RCPPML_MATRIXFACTORIZATION1_H

#ifndef RCPPML_H
#include "RcppML.h"
#endif
#ifndef RCPPML_SPARSEMATRIX_H
#include "RcppML_SparseMatrix.h"
#endif

namespace RcppML {

    template <typename T>
    Eigen::Matrix<T, -1, 1> svd1(const Eigen::SparseMatrix<T>& A, const T tol = 1e-3, const unsigned int maxit = 100,
                                 const bool return_v = true) {

        const unsigned int threads = Eigen::nbThreads();
        const unsigned int n_features = A.rows(), n_samples = A.cols();
        Eigen::Matrix<T, -1, 1> h(n_samples), h_it(n_samples), w(n_features);
        h.setRandom();
        h = h.cwiseAbs();
        for (unsigned int it = 0; it < maxit; ++it) {
            // update w
            w.setZero();
            T a = 0;
            for (unsigned int i = 0; i < n_samples; ++i) a += h(i) * h(i);
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int sample = 0; sample < n_samples; ++sample)
                for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, sample); iter; ++iter)
                    w(iter.row()) += iter.value() * h(sample);
            for (unsigned int i = 0; i < n_features; ++i) w(i) /= a;
            // update h
            h_it = h;
            h.setZero();
            a = 0;
            for (unsigned int i = 0; i < n_features; ++i) a += w(i) * w(i);
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int sample = 0; sample < n_samples; ++sample) {
                for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, sample); iter; ++iter)
                    h(sample) += iter.value() * w(iter.row());
                h(sample) /= a;
            }
            if (it > 0 && cor(h, h_it) < tol) break;
        }
        return return_v ? h : w;
    }
} // namespace RcppML

#endif // RCPPML_MATRIXFACTORIZATION1_H
