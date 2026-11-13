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

#include <algorithm>
#include <cmath>

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
 * @brief Explicit loss for non-MSE losses, sparse A (all m*n entries)
 *
 * Returns the sum over ALL entries of compute_robust_loss(a_ij, mu_ij), the
 * same number explicit_loss_dense returns for the densified matrix. Stored
 * and structural zeros both contribute loss(0, mu).
 *
 * Cost:
 *   GP without robust: the zero term mu / (1 + theta) is linear in mu, so the
 *     zeros are summed in closed form from the row sums of W_Td. O(nnz·k + m·k).
 *   otherwise: the model value is needed at every entry, one matrix-vector
 *     product per column. O(m·n·k).
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
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vec;
    const int n = static_cast<int>(A.cols());
    const int m = static_cast<int>(W_Td.cols());
    const Scalar zero = static_cast<Scalar>(0);

#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)num_threads;
#endif
    if (nt < 1) nt = 1;

    const bool gp_linear = (loss_config.type == LossType::GP) &&
                           !(loss_config.robust_delta > zero);

    double total = 0.0;

    if (gp_linear) {
        // loss(0, mu) = mu / (1 + theta). Sum over the zeros of a column is
        // (sum over all rows) - (sum over its non-zero rows).
        Vec q_row;      // 1 / (1 + theta_i) when theta is per row
        Vec s_all(W_Td.rows());
        if (theta_vec && !theta_is_per_col) {
            q_row.resize(m);
            for (int i = 0; i < m; ++i) {
                q_row(i) = static_cast<Scalar>(
                    1.0 / (1.0 + std::max(static_cast<double>(theta_vec[i]), 0.0)));
            }
            s_all = W_Td * q_row;
        } else {
            s_all = W_Td.rowwise().sum();
        }

        #pragma omp parallel for reduction(+:total) num_threads(nt) \
                                 schedule(dynamic, 64) if(nt > 1)
        for (int j = 0; j < n; ++j) {
            double q_col = 1.0;
            if (theta_vec && theta_is_per_col) {
                q_col = 1.0 / (1.0 + std::max(static_cast<double>(theta_vec[j]), 0.0));
            }
            double acc = q_col * static_cast<double>(s_all.dot(H.col(j)));
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                const Scalar y = static_cast<Scalar>(it.value());
                if (y == zero) continue;
                const int row = static_cast<int>(it.row());
                const Scalar predicted = W_Td.col(row).dot(H.col(j));
                const Scalar theta = theta_vec ? (theta_is_per_col ? theta_vec[j]
                                                                    : theta_vec[row])
                                               : zero;
                const double q = 1.0 / (1.0 + std::max(static_cast<double>(theta), 0.0));
                acc += static_cast<double>(compute_loss(y, predicted, loss_config, theta))
                     - q * static_cast<double>(predicted);
            }
            total += acc;
        }
        return static_cast<Scalar>(total);
    }

    #pragma omp parallel num_threads(nt) reduction(+:total) if(nt > 1)
    {
        Vec recon(m);
        #pragma omp for schedule(dynamic, 16)
        for (int j = 0; j < n; ++j) {
            recon.noalias() = W_Td.transpose() * H.col(j);
            double acc = 0.0;
            for (int i = 0; i < m; ++i) {
                const Scalar theta = theta_vec ? (theta_is_per_col ? theta_vec[j]
                                                                    : theta_vec[i])
                                               : zero;
                acc += static_cast<double>(
                    compute_robust_loss(zero, recon(i), loss_config, theta));
            }
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                const Scalar y = static_cast<Scalar>(it.value());
                if (y == zero) continue;
                const int row = static_cast<int>(it.row());
                const Scalar theta = theta_vec ? (theta_is_per_col ? theta_vec[j]
                                                                    : theta_vec[row])
                                               : zero;
                acc += static_cast<double>(
                           compute_robust_loss(y, recon(row), loss_config, theta))
                     - static_cast<double>(
                           compute_robust_loss(zero, recon(row), loss_config, theta));
            }
            total += acc;
        }
    }
    return static_cast<Scalar>(total);
}

/**
 * @brief Explicit loss for non-MSE losses over all elements (dense)
 *
 * Reconstructs WH and sums compute_robust_loss element-wise (accumulated in
 * double precision).
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
    double total_loss = 0.0;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Scalar theta = theta_vec ? (theta_is_per_col ? theta_vec[j]
                                                          : theta_vec[i])
                                     : static_cast<Scalar>(0);
            total_loss += static_cast<double>(
                compute_robust_loss(A(i, j), R(i, j), loss_config, theta));
        }
    }
    return static_cast<Scalar>(total_loss);
}

}  // namespace detail
}  // namespace nmf
}  // namespace FactorNet
