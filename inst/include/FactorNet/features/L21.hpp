// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/L21.hpp
 * @brief L2,1 (group sparsity) penalty — Tier 2 (touches full factor)
 *
 * Encourages entire rows of the factor matrix to be zero (row sparsity):
 *   G(i,i) += λ / ||factor_row_i||₂
 *
 * This is an adaptive L2 penalty per row — rows with small norm get
 * larger penalty, driving them to zero.
 *
 * Classification: Tier 2 — accesses full factor matrix (k × m or k × n).
 * On GPU paths, the factor may need to be transferred to host or a
 * thin GPU kernel may compute row norms.
 *
 * Cost: O(k · m) to compute row norms, O(k) to update diagonal.
 *
 * Extracted from nmf/gram.hpp::compute_gram_L21() and
 * nmf/cd_solver.hpp::cd_nnls_full_reg().
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/constants.hpp>

namespace FactorNet {
namespace features {

/**
 * @brief Apply L2,1 group sparsity penalty to Gram matrix
 *
 * Modifies G diagonal:  G(i,i) += λ / ||row_i||₂
 *
 * Rows with small norm get larger penalty → driven to zero.
 * Rows with large norm get smaller penalty → preserved.
 *
 * @param G       Gram matrix (k × k), modified in-place
 * @param factor  Factor matrix (k × n_cols) whose rows are penalized
 * @param lambda  L2,1 penalty strength
 */
template<typename Scalar>
inline void apply_L21(DenseMatrix<Scalar>& G,
                      const DenseMatrix<Scalar>& factor,
                      Scalar lambda)
{
    if (lambda <= 0) return;

    const int k = static_cast<int>(factor.rows());
    for (int i = 0; i < k; ++i) {
        Scalar row_norm = factor.row(i).norm();
        if (row_norm > static_cast<Scalar>(1e-10)) {
            G(i, i) += lambda / row_norm;
        }
    }
}

}  // namespace features
}  // namespace FactorNet

