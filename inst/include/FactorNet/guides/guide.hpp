// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file guides/guide.hpp
 * @brief Abstract base class for factor guides
 *
 * A Guide modifies the Gram matrix G and RHS matrix B before the NNLS
 * solve, steering a factor toward a target:
 *
 *   G.diagonal() += |λ|           (regularization toward target)
 *   B.col(j)     += λ · t_j       (per-column target direction)
 *
 * This is equivalent to adding a penalty term λ||h_j - t_j||² to the
 * per-column NNLS objective. Positive λ attracts toward the target;
 * negative λ repels.
 *
 * Guides are applied AFTER standard features (L1/L2, angular, graph, L21)
 * and BEFORE the NNLS solve.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

namespace FactorNet {
namespace guides {

/**
 * @brief Abstract base class for factor guides
 *
 * Subclasses implement:
 *   - apply(): modify G and B to steer the factor toward a target
 *   - update(): recompute internal targets given current factor state
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct Guide {
    /// Guide strength: positive = attract, negative = repel
    Scalar lambda = static_cast<Scalar>(1);

    virtual ~Guide() = default;

    /**
     * @brief Apply guide to Gram and RHS before NNLS solve
     *
     * Must modify G.diagonal() and B columns according to the guide's
     * target. Called once per ALS iteration after features, before NNLS.
     *
     * @param G  Gram matrix (k × k), modified in-place
     * @param B  RHS matrix (k × n_cols), modified in-place
     */
    virtual void apply(DenseMatrix<Scalar>& G, DenseMatrix<Scalar>& B) const = 0;

    /**
     * @brief Update internal targets given current factor state
     *
     * Called at the start of each factor update, before Gram/RHS/features.
     * Subclasses use this to recompute targets (e.g., class centroids).
     *
     * @param factor     The factor being guided (k × n)
     * @param iteration  Current outer ALS iteration (0-indexed)
     */
    virtual void update(const DenseMatrix<Scalar>& factor, int iteration) {
        (void)factor; (void)iteration;  // default: no-op
    }

    /// True if this guide is active (|λ| > 0)
    bool is_active() const { return lambda != static_cast<Scalar>(0); }
};

}  // namespace guides
}  // namespace FactorNet
