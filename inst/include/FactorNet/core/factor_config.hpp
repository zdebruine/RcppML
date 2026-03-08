// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file core/factor_config.hpp
 * @brief Per-factor configuration for regularization, constraints, and guides
 *
 * FactorConfig<Scalar> collects all parameters that are specific to a
 * single factor matrix (e.g., W or H in standard NMF, or any factor in
 * a multi-layer FactorNet).  NMFConfig<Scalar> embeds two of these
 * (W and H) to replace the old flat _W/_H field pairs.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <vector>

namespace FactorNet {

// Forward-declare Guide so factor_config.hpp doesn't pull in full guide headers
namespace guides {
template<typename Scalar> struct Guide;
}  // namespace guides

/**
 * @brief Per-factor configuration: regularization, constraints, and guides
 *
 * All parameters that may differ between W and H (or between any two
 * factors in a multi-layer network).  Algorithm-level parameters
 * (rank, tolerance, solver mode, etc.) stay in NMFConfig.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct FactorConfig {

    // ====================================================================
    // Regularization (Tier 1: cheap, O(k) per column)
    // ====================================================================

    /// L1 (lasso) penalty
    Scalar L1 = 0;

    /// L2 (ridge) penalty
    Scalar L2 = 0;

    // ====================================================================
    // Regularization (Tier 2: require full factor matrix, O(k²·n) or host roundtrip)
    // ====================================================================

    /// L2,1 (group sparsity) penalty
    Scalar L21 = 0;

    /// Angular (orthogonality) penalty
    Scalar angular = 0;

    // ====================================================================
    // Constraints
    // ====================================================================

    /// Upper bound constraint (0 = no bound)
    Scalar upper_bound = 0;

    /// Non-negativity constraint
    bool nonneg = true;

    // ====================================================================
    // Graph Laplacian regularization (Tier 2)
    // ====================================================================

    /// Graph Laplacian matrix (caller owns; nullptr = disabled)
    const SparseMatrix<Scalar>* graph = nullptr;

    /// Graph regularization strength
    Scalar graph_lambda = 0;

    // ====================================================================
    // Guides — soft targets that steer factors toward desired structure
    // ====================================================================

    /// Guides for this factor (caller owns; applied during update)
    std::vector<guides::Guide<Scalar>*> guides;

    // ====================================================================
    // Queries
    // ====================================================================

    /// True if any Tier-2 features are active (angular, graph, L21)
    /// Note: does NOT include upper_bound or nonneg since those are
    /// post-NNLS constraints, not Gram/RHS modifications.
    bool has_tier2_features() const noexcept {
        return angular > 0 || graph != nullptr || L21 > 0;
    }

    /// True if any regularization beyond L1/L2 is active
    bool has_full_reg() const noexcept {
        return L21 > 0 || angular > 0 || graph != nullptr
            || upper_bound > 0 || !nonneg;
    }

    /// True if guides are configured
    bool has_guides_active() const noexcept {
        return !guides.empty();
    }

    /// True if any regularization at all is active on this factor
    bool has_any_reg() const noexcept {
        return L1 > 0 || L2 > 0 || has_full_reg();
    }
};

}  // namespace FactorNet
