// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/graph_reg.hpp
 * @brief Graph Laplacian regularization — Tier 2 (touches full factor)
 *
 * Adds a smoothness penalty that encourages neighboring features (or samples)
 * in a graph to have similar factor loadings:
 *
 *   G += λ · factor · L · factorᵀ  (k × k)
 *
 * where L is a sparse graph Laplacian (m × m or n × n).
 *
 * Classification: Tier 2 — accesses the full factor matrix (k × m or k × n)
 * multiplied by the sparse Laplacian. For GPU paths, this maps naturally to
 * a cuSPARSE SpMM as an "extended primitive".
 *
 * Cost: O(k² · nnz(L)) — depends on graph sparsity.
 *
 * Extracted from nmf/gram.hpp::compute_gram_regularized (graph path) and
 * nmf/cd_solver.hpp::cd_nnls_full_reg (graph path).
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

namespace FactorNet {
namespace features {

/**
 * @brief Apply graph Laplacian regularization to Gram matrix
 *
 * Computes:  G += λ · factor · L · factorᵀ
 *
 * This smoothness penalty penalizes factor differences along graph edges.
 * The result is a k×k matrix added to the existing Gram.
 *
 * @param G          Gram matrix (k × k), modified in-place
 * @param laplacian  Sparse graph Laplacian (n × n, or m × m)
 * @param factor     Factor matrix (k × n, or k × m)
 * @param lambda     Graph regularization strength
 */
template<typename Scalar>
inline void apply_graph_reg(DenseMatrix<Scalar>& G,
                            const SparseMatrix<Scalar>& laplacian,
                            const DenseMatrix<Scalar>& factor,
                            Scalar lambda)
{
    if (lambda <= 0) return;

    // FL = factor · L  (k × n)  — sparse matrix multiply
    DenseMatrix<Scalar> FL = factor * laplacian;

    // G += λ · FL · factorᵀ  (k × k)
    G.noalias() += lambda * FL * factor.transpose();
}

}  // namespace features
}  // namespace FactorNet

