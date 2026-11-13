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

#include <cmath>

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

/**
 * @brief Per-column group-lasso step for the CPU NMF paths
 *
 * The group is one column of the factor being solved (a feature across all
 * factors for W, a sample across all factors for H).  The penalty on that
 * column is  lambda · ‖x‖₂  on top of
 *
 *   ½ xᵀG x − bᵀx + L1‖x‖₁,   x ≥ 0 when nonneg.
 *
 * x = 0 is optimal exactly when ‖t(b)‖₂ ≤ lambda, where t(b) = max(b − L1, 0)
 * under non-negativity and the soft-threshold S(b, L1) otherwise.  In that
 * case `is_zero` is set and the caller writes an all-zero column.
 *
 * Otherwise the function returns the ridge  lambda / ‖x_prev‖₂  of the
 * quadratic majorizer of lambda·‖x‖₂ at the previous iterate.  `prev_norm`
 * must be on the scale of the variable being solved.  When the previous
 * column is (numerically) zero the norm is replaced by the estimate
 * (‖t(b)‖₂ − lambda) / g_ref, which is exact for G = g_ref·I.
 *
 * @param b          Unpenalized right-hand side of this column (length k)
 * @param g_ref      Reference curvature (largest diagonal entry of G)
 * @return           Ridge to add for this column (0 when is_zero)
 */
template<typename Scalar>
inline Scalar l21_column_ridge(const Scalar* b,
                               int k,
                               Scalar L1,
                               bool nonneg,
                               Scalar lambda,
                               Scalar prev_norm,
                               Scalar g_ref,
                               bool& is_zero)
{
    is_zero = false;
    if (lambda <= Scalar(0)) return Scalar(0);

    Scalar ss = 0;
    for (int i = 0; i < k; ++i) {
        Scalar v = b[i];
        if (nonneg) {
            v -= L1;
            if (v < Scalar(0)) v = Scalar(0);
        } else if (v > L1) {
            v -= L1;
        } else if (v < -L1) {
            v += L1;
        } else {
            v = Scalar(0);
        }
        ss += v * v;
    }
    const Scalar bn = std::sqrt(ss);
    if (!(bn > lambda)) {
        is_zero = true;
        return Scalar(0);
    }

    const Scalar est = (bn - lambda) / ((g_ref > Scalar(0)) ? g_ref : Scalar(1));
    Scalar nrm = prev_norm;
    if (!(nrm > est * static_cast<Scalar>(1e-4))) nrm = est;
    if (!(nrm > Scalar(0))) return Scalar(0);
    return lambda / nrm;
}

}  // namespace features
}  // namespace FactorNet

