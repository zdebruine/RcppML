// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/bounds.hpp
 * @brief Element-wise bounds — Tier 1 (O(km) clamp)
 *
 * Clamps factor elements to [0, upper_bound] after the NNLS solve.
 * Also handles the non-negativity relaxation case (nonneg=false
 * in the NNLS primitive already handles this, but bounds provides
 * the post-solve upper bound enforcement).
 *
 * Cost: O(k · m) element-wise — negligible.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

namespace FactorNet {
namespace features {

/**
 * @brief Clamp factor elements to [0, upper_bound]
 *
 * Applied after NNLS solve to enforce box constraints.
 *
 * @param X            Factor matrix (k × n_cols), modified in-place
 * @param upper_bound  Maximum allowed value (0 = no bound)
 */
template<typename Scalar>
inline void apply_upper_bound(DenseMatrix<Scalar>& X, Scalar upper_bound)
{
    if (upper_bound <= 0) return;
    X = X.cwiseMin(upper_bound);
}

/**
 * @brief Clamp factor elements to be non-negative
 *
 * Safety net after unconstrained solve (nonneg=false in NNLS).
 * Typically not needed since NNLS enforces this, but useful for
 * post-processing or when upper bounds have been applied.
 *
 * @param X  Factor matrix (k × n_cols), modified in-place
 */
template<typename Scalar>
inline void apply_nonneg(DenseMatrix<Scalar>& X)
{
    X = X.cwiseMax(static_cast<Scalar>(0));
}

}  // namespace features
}  // namespace FactorNet

