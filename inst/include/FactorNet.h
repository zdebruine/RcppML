// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

#ifndef FactorNet_H
#define FactorNet_H

// OpenMP support
//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

// Eigen configuration
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#ifndef EIGEN_INITIALIZE_MATRICES_BY_ZERO
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#endif

// Include Eigen headers
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Core type infrastructure only — lightweight for RcppExports.cpp.
// Algorithm headers (SVD, NMF rank_cv, clustering) are included by
// RcppFunctions.cpp directly, not here, so they are NOT parsed by
// the compileAttributes-generated RcppExports.cpp.
#include "FactorNet/core/types.hpp"
#include "FactorNet/core/traits.hpp"
#include "FactorNet/core/constants.hpp"
#include "FactorNet/core/memory.hpp"
#include "FactorNet/core/logging.hpp"
#include "FactorNet/core/resource_tags.hpp"
#include "FactorNet/math/blas.hpp"
#include "FactorNet/math/loss.hpp"
#include "FactorNet/rng/rng.hpp"

// Helper functions
namespace FactorNet {

// Helper function to get column inner indices (row indices of non-zeros in a column)
template<typename SpMat>
inline std::vector<int> innerIndices(const SpMat& A, int col) {
    std::vector<int> indices;
    indices.reserve(A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col]);
    for (typename SpMat::InnerIterator it(A, col); it; ++it) {
        indices.push_back(it.row());
    }
    return indices;
}

// Helper function to get a dense column from a sparse matrix
template<typename SpMat>
inline Eigen::VectorXd getColumn(const SpMat& A, int col) {
    Eigen::VectorXd c = Eigen::VectorXd::Zero(A.rows());
    for (typename SpMat::InnerIterator it(A, col); it; ++it) {
        c(it.row()) = it.value();
    }
    return c;
}

// Check if sparse matrix is approximately symmetric (for square matrices)
template<typename SpMat>
inline bool isApproxSymmetric(const SpMat& A, double tol = 1e-10) {
    if (A.rows() != A.cols()) return false;
    // Quick check on a subset of elements
    int checks = std::min(10, static_cast<int>(A.cols()));
    for (int k = 0; k < checks; ++k) {
        for (typename SpMat::InnerIterator it(A, k); it; ++it) {
            if (it.row() != it.col()) {
                double transpose_val = A.coeff(it.col(), it.row());
                if (std::abs(it.value() - transpose_val) > tol) return false;
            }
        }
    }
    return true;
}

}  // namespace FactorNet

#endif  // FactorNet_H
