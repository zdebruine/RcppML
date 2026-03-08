// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/angular.hpp
 * @brief Angular (orthogonality) penalty — Tier 1 (k×k operation)
 *
 * Encourages factor columns/rows to be orthogonal by adding a penalty:
 *   G += λ · (factor · factorᵀ)  with diagonal zeroed
 *
 * This drives the off-diagonal entries of factorᵀ·factor toward zero.
 *
 * Cost: O(k² · m) for the factor product, O(k²) for diagonal zeroing.
 * Since k is small (typically ≤ 64), this is negligible.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

namespace FactorNet {
namespace features {

/**
 * @brief Apply angular (orthogonality) penalty to Gram matrix
 *
 * Adds λ · (factor · factorᵀ - I) to G.
 * The diagonal is NOT penalized (only off-diagonal overlap).
 *
 * For H update: factor = H_current (k × n), penalizes column overlap.
 * For W update: factor = W_current (k × m), penalizes row overlap.
 *
 * @param G       Gram matrix (k × k), modified in-place
 * @param factor  Current factor matrix (k × n_cols)
 * @param lambda  Angular penalty strength
 */
template<typename Scalar>
inline void apply_angular(DenseMatrix<Scalar>& G,
                          const DenseMatrix<Scalar>& factor,
                          Scalar lambda)
{
    if (lambda <= 0) return;

    const int k = static_cast<int>(G.rows());

    // Compute factor overlap: overlap = factor · factorᵀ  (k × k)
    DenseMatrix<Scalar> overlap(k, k);
    overlap.setZero();
    overlap.template selfadjointView<Eigen::Lower>().rankUpdate(factor);
    overlap.template triangularView<Eigen::Upper>() = overlap.transpose();

    // Normalize to cosine similarity for scale-invariance.
    // After NMF diag-normalization, factor rows are typically unit-norm
    // so this is often a no-op, but handles un-normalized cases correctly.
    DenseVector<Scalar> norms = overlap.diagonal().cwiseSqrt();
    for (int i = 0; i < k; ++i) {
        if (norms(i) > 0) {
            overlap.row(i) /= norms(i);
            overlap.col(i) /= norms(i);
        }
    }

    // Add FULL normalized overlap to G (including diagonal).
    // Since overlap is PSD, G + λ*overlap is guaranteed PSD.
    // This penalizes solutions aligned with the shared-variance direction
    // of the factor matrix, encouraging orthogonal factors.
    // The self-overlap on the diagonal acts as uniform L2 regularization,
    // while off-diagonal terms add extra penalty for correlated factors.
    G.noalias() += lambda * overlap;
}

}  // namespace features
}  // namespace FactorNet

