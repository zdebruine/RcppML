// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/angular.hpp
 * @brief Angular (orthogonality) penalty for NMF and SVD
 *
 * Two mechanisms:
 *
 * 1. **Gram-based** (`apply_angular`): Adds λ · cosine_matrix to the Gram.
 *    Used in SVD paths where iterative refinement handles the penalty.
 *
 * 2. **Post-NNLS projection** (`apply_angular_posthoc`): After the NNLS
 *    solve, subtracts correlated components from each factor row. This
 *    directly decorrelates factor *profiles* (directions), which the
 *    Gram-based approach cannot do effectively due to re-normalization.
 *    Used in NMF paths.
 *
 * Cost: O(k² · m) for both approaches. Negligible for k ≤ 64.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

namespace FactorNet {
namespace features {

/**
 * @brief Gram-based angular penalty (used by SVD paths)
 *
 * Adds λ · cosine_similarity_matrix(factor) to G.
 * The full cosine matrix (including diagonal) is added to preserve PSD.
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

    // Normalize to cosine similarity
    DenseVector<Scalar> norms = overlap.diagonal().cwiseSqrt();
    for (int i = 0; i < k; ++i) {
        if (norms(i) > 0) {
            overlap.row(i) /= norms(i);
            overlap.col(i) /= norms(i);
        }
    }

    G.noalias() += lambda * overlap;
}

/**
 * @brief Post-NNLS angular decorrelation (used by NMF paths)
 *
 * Directly decorrelates factor row profiles by subtracting correlated
 * components.  For each row i, subtracts a weighted combination of other
 * rows where weights = cosine similarity.  This gradient descent step on
 * L = Σ_{i<j} cos(w_i, w_j) pushes correlated factors apart.
 *
 * Applied AFTER the NNLS solve, BEFORE extract_scaling.
 * Results are clipped to non-negative only when the factor is constrained
 * to be non-negative; a signed factor keeps its signs.
 *
 * @param Factor  Factor matrix (k × cols), modified in-place
 * @param lambda  Angular penalty strength (typical: 0.01–1.0)
 * @param nonneg  Whether the factor carries a non-negativity constraint
 */
template<typename Scalar>
inline void apply_angular_posthoc(DenseMatrix<Scalar>& Factor,
                                  Scalar lambda,
                                  bool nonneg = true)
{
    if (lambda <= 0) return;

    const int k = static_cast<int>(Factor.rows());

    // 1. Compute row norms and normalize
    DenseVector<Scalar> row_norms(k);
    for (int i = 0; i < k; ++i) {
        row_norms(i) = Factor.row(i).norm();
    }

    DenseMatrix<Scalar> F_hat = Factor;
    for (int i = 0; i < k; ++i) {
        if (row_norms(i) > Scalar(1e-15)) {
            F_hat.row(i) /= row_norms(i);
        }
    }

    // 2. Cosine similarity matrix (k × k) with diagonal zeroed
    DenseMatrix<Scalar> cos_mat(k, k);
    cos_mat.setZero();
    cos_mat.template selfadjointView<Eigen::Lower>().rankUpdate(F_hat);
    cos_mat.template triangularView<Eigen::Upper>() = cos_mat.transpose();
    cos_mat.diagonal().setZero();

    // 3. Gradient: grad_i = ||row_i|| · Σ_{j≠i} cos(i,j) · F_hat.row(j)
    //    Scale by row_norms so perturbation is proportional to factor magnitude.
    //    Equivalent to decorrelating on the unit sphere, then re-scaling.
    DenseMatrix<Scalar> grad = cos_mat * F_hat;
    for (int i = 0; i < k; ++i) {
        grad.row(i) *= row_norms(i);
    }

    // 4. Step size.  On the unit rows the update is F̂ ← (I − λ·C₀)·F̂ with
    //    C₀ the off-diagonal cosine matrix.  A shared component with
    //    eigenvalue μ of C₀ is scaled by (1 − λμ): for λμ > 1 it changes sign
    //    and for λμ > 2 it grows, so a large λ re-correlates the factors
    //    instead of separating them.  For a non-negative factor the clip at
    //    zero below absorbs the overshoot.  A signed factor has no such
    //    guard, so its step is capped at 1/μ_max, bounded by the largest
    //    absolute row sum of C₀ (Gershgorin): the most correlated component
    //    is removed at most completely, never reversed.
    Scalar step = lambda;
    if (!nonneg && k > 0) {
        const Scalar max_row = cos_mat.cwiseAbs().rowwise().sum().maxCoeff();
        if (max_row > Scalar(0) && step * max_row > Scalar(1))
            step = Scalar(1) / max_row;
    }

    // 5. Subtract gradient; clip to non-negative only for a nonneg factor
    Factor -= step * grad;
    if (nonneg) Factor = Factor.cwiseMax(Scalar(0));
}

}  // namespace features
}  // namespace FactorNet

