// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file svd_result.hpp
 * @brief Result type for deflation-based SVD/PCA
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <vector>

namespace FactorNet {

template<typename Scalar>
struct SVDResult {

    /// Scores matrix: m × k (left singular vectors, scaled)
    DenseMatrix<Scalar> U;

    /// Singular values: k
    DenseVector<Scalar> d;

    /// Loadings matrix: n × k (right singular vectors, scaled)
    DenseMatrix<Scalar> V;

    /// Rank selected by auto-rank (or user-specified k)
    int k_selected = 0;

    /// Was centering applied?
    bool centered = false;

    /// Row means (only populated if centered, length m)
    DenseVector<Scalar> row_means;

    /// Was scaling applied?
    bool scaled = false;

    /// Row standard deviations (only populated if scaled, length m)
    DenseVector<Scalar> row_sds;

    /// Test loss (MSE) at each rank 1..k_selected (empty if no CV)
    std::vector<Scalar> test_loss_trajectory;

    /// Train loss (MSE) at each rank 1..k_selected
    std::vector<Scalar> train_loss_trajectory;

    /// Number of ALS iterations for each factor
    std::vector<int> iters_per_factor;

    /// Squared Frobenius norm of (centered) input: ||A_c||^2_F
    double frobenius_norm_sq = 0;

    /// Total wall-clock time in milliseconds
    double wall_time_ms = 0;

    /// Reconstruct approximation: U * diag(d) * V'
    DenseMatrix<Scalar> reconstruct() const {
        return U * d.asDiagonal() * V.transpose();
    }
};

}  // namespace FactorNet

