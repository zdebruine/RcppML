// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file svd_config.hpp
 * @brief Configuration for all SVD/PCA algorithms
 *
 * Unified config shared by deflation, krylov, lanczos, irlba, and randomized
 * methods. Each method uses the subset of fields it supports; unsupported
 * combinations are validated at the R layer.
 */

#pragma once

#include <cstdint>
#include <string>
#include <FactorNet/core/types.hpp>

namespace FactorNet {

/// Convergence criterion for iterative SVD methods
enum class SVDConvergence {
    FACTOR,   ///< Track relative change in factor vectors (default)
    LOSS,     ///< Track relative change in reconstruction loss
    BOTH      ///< Stop when EITHER criterion is met
};

template<typename Scalar>
struct SVDConfig {

    // ====================================================================
    // Core
    // ====================================================================

    /// Maximum rank (auto-rank stops before this)
    int k_max = 50;

    /// Convergence tolerance per rank-1 subproblem (cosine distance)
    Scalar tol = static_cast<Scalar>(1e-6);

    /// Maximum ALS iterations per rank-1 factor
    int max_iter = 100;

    /// Implicit column-mean centering (PCA mode)
    bool center = false;

    /// Implicit row-SD scaling (correlation PCA mode)
    /// Requires center=true. Applies D^{-1}(A - μ1^T) where D = diag(σ_i)
    bool scale = false;

    /// Print per-factor diagnostics
    bool verbose = false;

    /// Random seed for initialization
    uint32_t seed = 0;

    /// OpenMP thread limit (0 = all available)
    int threads = 0;

    /// Convergence criterion
    SVDConvergence convergence = SVDConvergence::FACTOR;

    // ====================================================================
    // Per-element regularization (u-side and v-side)
    // ====================================================================

    /// L1 (lasso) penalty — element-wise sparsity
    Scalar L1_u = 0, L1_v = 0;

    /// L2 (ridge) penalty — shrinkage
    Scalar L2_u = 0, L2_v = 0;

    /// Non-negativity constraints
    bool nonneg_u = false, nonneg_v = false;

    /// Upper bound constraints (0 = no bound)
    Scalar upper_bound_u = 0, upper_bound_v = 0;

    // ====================================================================
    // Gram-level regularization (Tier 2 — krylov and deflation only)
    // ====================================================================

    /// L2,1 (group sparsity) penalty — drives entire rows/columns to zero
    Scalar L21_u = 0, L21_v = 0;

    /// Angular (orthogonality) penalty — decorrelates components
    Scalar angular_u = 0, angular_v = 0;

    /// Graph Laplacian regularization
    const SparseMatrix<Scalar>* graph_u = nullptr;
    const SparseMatrix<Scalar>* graph_v = nullptr;
    Scalar graph_u_lambda = 0, graph_v_lambda = 0;

    // ====================================================================
    // Robust SVD via IRLS (Huber loss on residuals)
    // ====================================================================

    /// Huber delta for robust SVD. <= 0 means standard (MSE) SVD.
    /// > 0 applies Huber-type IRLS reweighting to downweight outlier entries.
    /// 1.345 = 95% asymptotic efficiency under Gaussian.
    /// Small values (~1e-4) approximate MAE behavior.
    Scalar robust_delta = 0;

    /// Maximum IRLS outer iterations per ALS step (only used when robust_delta > 0)
    int irls_max_iter = 5;

    /// IRLS convergence tolerance (relative change in weights)
    Scalar irls_tol = static_cast<Scalar>(1e-4);

    // ====================================================================
    // Speckled CV for auto-rank
    // ====================================================================

    /// Fraction of entries to hold out for test MSE (0 = no CV, no auto-rank)
    Scalar test_fraction = static_cast<Scalar>(0.05);

    /// Separate seed for CV holdout mask (0 = derive from seed)
    uint32_t cv_seed = 0;

    /// Patience: stop after this many consecutive non-improving factors
    int patience = 3;

    /// If true, only mask non-zero entries (for sparse recommendation data)
    bool mask_zeros = false;

    /// Optional custom observation mask: if non-null, entries where mask(i,j)!=0
    /// are treated as unobserved (zeroed before factorization).
    /// The mask must have the same sparsity structure as the input (or a subset).
    const SparseMatrix<Scalar>* obs_mask = nullptr;

    // ====================================================================
    // Streaming SPZ (for out-of-core SVD)
    // ====================================================================

    /// Path to .spz v2 file (empty = not streaming)
    std::string spz_path;

    // ====================================================================
    // Helpers
    // ====================================================================

    bool is_cv() const noexcept {
        return test_fraction > 0 && test_fraction < 1;
    }

    uint32_t effective_cv_seed() const noexcept {
        return cv_seed != 0 ? cv_seed : (seed != 0 ? seed ^ 0xBEEF : 42u);
    }

    bool has_regularization_u() const noexcept {
        return L1_u > 0 || L2_u > 0 || nonneg_u || upper_bound_u > 0 ||
               L21_u > 0 || angular_u > 0 ||
               (graph_u != nullptr && graph_u_lambda > 0);
    }

    bool has_regularization_v() const noexcept {
        return L1_v > 0 || L2_v > 0 || nonneg_v || upper_bound_v > 0 ||
               L21_v > 0 || angular_v > 0 ||
               (graph_v != nullptr && graph_v_lambda > 0);
    }

    bool has_gram_level_reg() const noexcept {
        return L21_u > 0 || L21_v > 0 ||
               angular_u > 0 || angular_v > 0 ||
               (graph_u != nullptr && graph_u_lambda > 0) ||
               (graph_v != nullptr && graph_v_lambda > 0);
    }

    /// Whether robust IRLS reweighting is active
    bool is_robust() const noexcept {
        return robust_delta > static_cast<Scalar>(0);
    }
};

}  // namespace FactorNet

