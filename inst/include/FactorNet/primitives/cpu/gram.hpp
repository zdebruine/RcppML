// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/gram.hpp
 * @brief CPU specialization of the Gram matrix primitive
 *
 * G = H · Hᵀ  (k × k)
 *
 * Uses Eigen's selfadjointView::rankUpdate for ~2× speedup over
 * naive H * H.transpose() — computes only the lower triangle and mirrors.
 *
 * Extracted from nmf/gram.hpp::compute_gram().
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/core/constants.hpp>

namespace FactorNet {
namespace primitives {

/**
 * @brief CPU Gram matrix: G = H · Hᵀ  (k × k)
 *
 * Adds tiny diagonal for numerical stability (prevents division by
 * zero in coordinate descent).
 */
template<>
inline void gram<CPU, double>(const DenseMatrix<double>& H,
                               DenseMatrix<double>& G)
{
    const int k = static_cast<int>(H.rows());
    G.resize(k, k);
    G.setZero();

    // Compute lower triangle of H · Hᵀ using rank update (~2× speedup)
    G.selfadjointView<Eigen::Lower>().rankUpdate(H);

    // Mirror to upper triangle
    G.triangularView<Eigen::Upper>() = G.transpose();

    // Numerical stability: prevent division by zero in CD solver
    G.diagonal().array() += tiny_num<double>();
}

/**
 * @brief CPU Gram matrix (float specialization)
 */
template<>
inline void gram<CPU, float>(const DenseMatrix<float>& H,
                              DenseMatrix<float>& G)
{
    const int k = static_cast<int>(H.rows());
    G.resize(k, k);
    G.setZero();
    G.selfadjointView<Eigen::Lower>().rankUpdate(H);
    G.triangularView<Eigen::Upper>() = G.transpose();
    G.diagonal().array() += tiny_num<float>();
}

}  // namespace primitives
}  // namespace FactorNet

