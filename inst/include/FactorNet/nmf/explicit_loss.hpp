// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/explicit_loss.hpp
 * @brief Explicit loss helpers for NMF training
 *
 * Contains gram_trick_loss (MSE fast path) and explicit_loss_{sparse,dense}
 * (element-wise loss for non-MSE objectives).
 *
 * Extracted from fit_cpu.hpp namespace detail block.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace nmf {
namespace detail {

/**
 * @brief Gram-trick loss — delegates to primitives::cpu::gram_trick_loss
 */
template<typename Scalar>
inline Scalar gram_trick_loss(Scalar trAtA,
                              const DenseMatrix<Scalar>& G,
                              const DenseMatrix<Scalar>& B_original,
                              const DenseMatrix<Scalar>& H)
{
    return primitives::cpu::gram_trick_loss(trAtA, G, B_original, H);
}

/**
 * @brief Explicit loss for non-MSE losses over sparse nonzeros
 *
 * For each nonzero A(i,j), computes predicted = W_Td.col(i)^T * H.col(j),
 * then sums loss(A(i,j), predicted) over all nonzeros.
 * Cost: O(nnz·k) — same as one RHS computation.
 */
template<typename Scalar, typename SparseMat>
inline Scalar explicit_loss_sparse(const SparseMat& A,
                                    const DenseMatrix<Scalar>& W_Td,
                                    const DenseMatrix<Scalar>& H,
                                    const LossConfig<Scalar>& loss_config,
                                    int num_threads = 1,
                                    const Scalar* theta_vec = nullptr,
                                    bool theta_is_per_col = false)
{
    const int n = static_cast<int>(A.cols());
    Scalar total_loss = 0;

    #pragma omp parallel for reduction(+:total_loss) num_threads(num_threads) \
                             schedule(dynamic, 64) if(num_threads > 1)
    for (int j = 0; j < n; ++j) {
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            Scalar predicted = W_Td.col(it.row()).dot(H.col(j));
            Scalar theta = theta_vec ? (theta_is_per_col ? theta_vec[j]
                                                          : theta_vec[it.row()])
                                     : static_cast<Scalar>(0);
            total_loss += compute_robust_loss(it.value(), predicted, loss_config, theta);
        }
    }
    return total_loss;
}

/**
 * @brief Explicit loss for non-MSE losses over all elements (dense)
 *
 * Reconstructs WH and sums element-wise loss.
 * Cost: O(m·n·k) for reconstruction + O(m·n) for loss.
 */
template<typename Scalar>
inline Scalar explicit_loss_dense(const DenseMatrix<Scalar>& A,
                                   const DenseMatrix<Scalar>& W_Td,
                                   const DenseMatrix<Scalar>& H,
                                   const LossConfig<Scalar>& loss_config,
                                   const Scalar* theta_vec = nullptr,
                                   bool theta_is_per_col = false)
{
    // Compute reconstruction: R = W_Td^T * H  (m × n)
    DenseMatrix<Scalar> R = W_Td.transpose() * H;
    Scalar total_loss = 0;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Scalar theta = theta_vec ? (theta_is_per_col ? theta_vec[j]
                                                          : theta_vec[i])
                                     : static_cast<Scalar>(0);
            total_loss += compute_robust_loss(A(i, j), R(i, j), loss_config, theta);
        }
    }
    return total_loss;
}

}  // namespace detail
}  // namespace nmf
}  // namespace FactorNet
