// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/sparsity.hpp
 * @brief L1/L2 regularization — Tier 1 (small-matrix, zero data movement)
 *
 * Modifies the Gram matrix G diagonal and/or the RHS matrix B:
 *   - L2 (ridge):  G.diagonal() += L2
 *   - L1 (lasso):  B -= L1
 *
 * Cost: O(k² + km) — negligible vs O(nnz·k) primitives.
 * Always runs on CPU. On GPU paths, only the k×k Gram needs to move.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

namespace FactorNet {
namespace features {

/**
 * @brief Apply L1 (lasso) and L2 (ridge) regularization
 *
 * L2 adds to the Gram diagonal (equivalent to ridge penalty).
 * L1 subtracts from the RHS (equivalent to soft-threshold in CD).
 *
 * @param G   Gram matrix (k × k), modified in-place
 * @param B   RHS matrix (k × n_cols), modified in-place
 * @param L1  L1 penalty strength (0 = disabled)
 * @param L2  L2 penalty strength (0 = disabled)
 */
template<typename Scalar>
inline void apply_L1_L2(DenseMatrix<Scalar>& G,
                        DenseMatrix<Scalar>& B,
                        Scalar L1,
                        Scalar L2)
{
    if (L2 > 0) G.diagonal().array() += L2;
    if (L1 > 0) B.array() -= L1;
}

}  // namespace features
}  // namespace FactorNet

