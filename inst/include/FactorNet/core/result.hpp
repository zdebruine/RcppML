// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file result.hpp
 * @brief Unified NMF result — one struct for all backends
 *
 * Merges the following prior result types:
 *   - nmf::NMFResult<Scalar>         — iterations, tol, loss, converged
 *   - nmf::CVResult<Scalar>          — + train/test loss, early stopping
 *   - algorithms::FitResult<Scalar>  — + loss history, regularization loss
 *   - algorithms::NMFModel<Scalar>   — W_, d_, H_, normalize(), sort()
 *   - nmf::NMFFactors<Scalar>        — W (k×m transposed), H (k×n), d
 *
 * The unified NMFResult stores W in standard orientation (m × k).
 * Internal iteration loop may use k × m transposed storage for cache
 * efficiency, but transposes back into the result at completion.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/fit_history.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace FactorNet {

/**
 * @brief Resource diagnostics — transparent debugging information
 *
 * Exposed in R as attr(result, "diagnostics") so users can inspect
 * why a particular resource plan was chosen. Never affects behavior.
 */
struct ResourceDiagnostics {
    int cpu_cores_detected = 0;
    int gpus_detected = 0;
    std::vector<std::string> gpu_names;
    int64_t gpu_memory_available = 0;   ///< Total VRAM in bytes across all GPUs

    /// e.g. "CPU (48 cores)" or "GPU"
    std::string plan_chosen;

    /// e.g. "GPU selected: 2 devices, 188GB VRAM"
    ///   or "CPU fallback: no GPUs detected"
    ///   or "CPU forced: resource_override='cpu'"
    std::string plan_reason;
};

/**
 * @brief Unified NMF result for all backends
 *
 * Contains the factorization W · diag(d) · H plus convergence
 * information, loss history, and resource diagnostics.
 *
 * W is always in standard orientation (m × k).
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct NMFResult {

    // ====================================================================
    // Factorization: A ≈ W · diag(d) · H
    // ====================================================================

    DenseMatrix<Scalar> W;   ///< m × k
    DenseVector<Scalar> d;   ///< k    (diagonal scaling)
    DenseMatrix<Scalar> H;   ///< k × n

    /// Estimated GP dispersion parameters (empty if loss != GP)
    DenseVector<Scalar> theta;  ///< m (per-row) or 1 (global)

    /// Estimated dispersion φ for Gamma/InvGauss (empty if not applicable)
    /// Gamma: V(μ) = φ·μ², InvGauss: V(μ) = φ·μ³
    DenseVector<Scalar> dispersion;  ///< m (per-row) or 1 (global)

    /// Estimated ZIGP dropout probabilities (empty if zi_mode == NONE)
    DenseVector<Scalar> pi_row;  ///< m (per-row dropout), empty if not used
    DenseVector<Scalar> pi_col;  ///< n (per-col dropout), empty if not used

    // ====================================================================
    // Convergence
    // ====================================================================

    int iterations = 0;          ///< Number of outer iterations performed
    bool converged = false;      ///< Did the algorithm satisfy tolerance?
    Scalar final_tol = 0;        ///< Final relative change at last check

    // ====================================================================
    // Loss
    // ====================================================================

    Scalar train_loss = 0;                         ///< Final training loss
    Scalar test_loss = 0;                          ///< Final test loss (CV only)
    Scalar best_test_loss = 0;                     ///< Best test loss seen (CV only)
    int best_iter = 0;                             ///< Iteration of best test loss
    std::vector<Scalar> loss_history;              ///< Per-check train loss
    std::vector<Scalar> test_loss_history;         ///< Per-check test loss (CV only)

    // ====================================================================
    // Fit history (structured per-iteration diagnostics)
    // ====================================================================

    FitHistory<Scalar> history;                     ///< Per-iteration snapshots

    // ====================================================================
    // Diagnostics
    // ====================================================================

    double wall_time_ms = 0;                       ///< Total wall-clock time
    ResourceDiagnostics diagnostics;

    /// Per-section cumulative profiling data (ms). Empty when profiling disabled.
    std::unordered_map<std::string, double> profile;

    // ====================================================================
    // Model operations
    // ====================================================================

    /**
     * @brief Reconstruct approximation A ≈ W · diag(d) · H
     */
    DenseMatrix<Scalar> reconstruct() const {
        return W * d.asDiagonal() * H;
    }

    /**
     * @brief Normalize factors so that each column of W and row of H has unit norm.
     *
     * Absorbs column norms of W and row norms of H into d:
     *   W_new[:,i] = W[:,i] / norm(W[:,i])
     *   H_new[i,:] = H[i,:] / norm(H[i,:])
     *   d_new[i]   = d[i] * norm(W[:,i]) * norm(H[i,:])
     *
     * @param norm_type  L1, L2, or None normalization
     */
    void normalize(NormType norm_type = NormType::L1) {
        if (norm_type == NormType::None) return;
        const int k = d.size();
        for (int i = 0; i < k; ++i) {
            Scalar w_norm, h_norm;
            if (norm_type == NormType::L1) {
                w_norm = W.col(i).template lpNorm<1>();
                h_norm = H.row(i).template lpNorm<1>();
            } else {
                w_norm = W.col(i).norm();
                h_norm = H.row(i).norm();
            }
            if (w_norm > 0) W.col(i) /= w_norm;
            if (h_norm > 0) H.row(i) /= h_norm;
            d(i) *= w_norm * h_norm;
        }
    }

    /**
     * @brief Sort factors by decreasing d
     */
    void sort() {
        const int k = d.size();
        std::vector<int> idx(k);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [this](int a, int b) { return d(a) > d(b); });

        // Permute W, d, H
        DenseMatrix<Scalar> W_sorted(W.rows(), k);
        DenseVector<Scalar> d_sorted(k);
        DenseMatrix<Scalar> H_sorted(k, H.cols());
        for (int i = 0; i < k; ++i) {
            W_sorted.col(i) = W.col(idx[i]);
            d_sorted(i) = d(idx[i]);
            H_sorted.row(i) = H.row(idx[i]);
        }
        W = std::move(W_sorted);
        d = std::move(d_sorted);
        H = std::move(H_sorted);
    }
};

}  // namespace FactorNet

