// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file core/fit_history.hpp
 * @brief Structured per-iteration diagnostics for NMF/SVD fits
 *
 * Provides FitSnapshot (one iteration's metrics) and FitHistory
 * (the complete sequence), returned as part of NMFResult.
 * The R side receives this as a tidy data.frame that is
 * immediately plottable.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <vector>
#include <cmath>
#include <limits>

namespace FactorNet {

/// One snapshot of metrics at iteration `i`
template<typename Scalar>
struct FitSnapshot {
    int iteration;             ///< 1-based iteration number
    Scalar train_loss;         ///< Training loss at this iteration
    Scalar test_loss;          ///< Test loss (NaN if no CV)
    Scalar delta_loss;         ///< Relative change from previous iteration
    double wall_ms;            ///< Cumulative wall time in ms since fit start
};

/// Complete fit history — returned as part of NMFResult
template<typename Scalar>
struct FitHistory {
    std::vector<FitSnapshot<Scalar>> snapshots;

    /// Reserve capacity for expected number of iterations
    void reserve(int n) { snapshots.reserve(n); }

    /// Add a snapshot
    void push(int iteration, Scalar train_loss, Scalar test_loss,
              Scalar delta_loss, double wall_ms) {
        snapshots.push_back({iteration, train_loss, test_loss, delta_loss, wall_ms});
    }

    /// Number of recorded snapshots
    int size() const { return static_cast<int>(snapshots.size()); }

    /// Whether any snapshots have been recorded
    bool empty() const { return snapshots.empty(); }

    // ================================================================
    // Convenience column accessors (for R export or analysis)
    // ================================================================

    std::vector<int> iterations() const {
        std::vector<int> v(snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) v[i] = snapshots[i].iteration;
        return v;
    }

    std::vector<double> train_loss() const {
        std::vector<double> v(snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) v[i] = static_cast<double>(snapshots[i].train_loss);
        return v;
    }

    std::vector<double> test_loss() const {
        std::vector<double> v(snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) v[i] = static_cast<double>(snapshots[i].test_loss);
        return v;
    }

    std::vector<double> delta_loss() const {
        std::vector<double> v(snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) v[i] = static_cast<double>(snapshots[i].delta_loss);
        return v;
    }

    std::vector<double> wall_ms() const {
        std::vector<double> v(snapshots.size());
        for (size_t i = 0; i < snapshots.size(); ++i) v[i] = snapshots[i].wall_ms;
        return v;
    }
};

}  // namespace FactorNet

