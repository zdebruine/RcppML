// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/loss.hpp
 * @brief CPU loss computation primitives
 *
 * Provides the explicit loss computation primitive for cases where the
 * Gram trick doesn't apply (e.g. KL divergence, Itakura-Saito).
 *
 * For standard Frobenius loss, the algorithm layer uses the Gram trick:
 *   ||A - WH||² = tr(AᵀA) - 2·tr(BᵀH) + tr(G·HHᵀ)
 * which is O(k²) using already-computed G and B — essentially free.
 *
 * This file provides the O(nnz·k) explicit computation for:
 * - Non-Frobenius losses
 * - Validation/debugging
 * - CV test loss on held-out entries
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/constants.hpp>

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {
namespace cpu {

/**
 * @brief Gram-trick loss — canonical implementation lives in primitives.hpp.
 *        This CPU-namespace alias is kept for backward compatibility.
 */
template<typename Scalar>
inline Scalar gram_trick_loss(Scalar trAtA,
                              const DenseMatrix<Scalar>& G,
                              const DenseMatrix<Scalar>& B,
                              const DenseMatrix<Scalar>& H)
{
    return ::FactorNet::primitives::gram_trick_loss(trAtA, G, B, H);
}

/**
 * @brief Explicit Frobenius loss: ||A - WH||² computed over sparse A
 *
 * Cost: O(nnz · k). Use only when Gram trick doesn't apply.
 *
 * @tparam SparseMat  Sparse matrix type (Eigen SparseMatrix or Map)
 * @param A   Sparse data matrix (m × n)
 * @param W   Factor W (k × m, transposed storage)
 * @param H   Factor H (k × n)
 * @return    ||A - WᵀH||²_F (note W is k×m, so reconstruction is Wᵀ·something)
 */
template<typename SparseMat, typename Scalar>
inline Scalar explicit_frobenius_sparse(const SparseMat& A,
                                        const DenseMatrix<Scalar>& W,
                                        const DenseMatrix<Scalar>& H)
{
    // For each nonzero A(i,j): residual = A(i,j) - dot(W.col(i), H.col(j))
    // Total loss = Σ residual² + sum of (W.col(i)·H.col(j))² for zero entries
    // The second term makes this O(m·n·k) — too expensive.
    // Instead: ||A - WᵀH||² = ||A||² - 2·tr(Aᵀ·Wᵀ·H) + ||WᵀH||²
    //
    // For sparse A the cross term is:
    //   tr(Aᵀ·WᵀH) = Σ_{(i,j)∈nnz} A(i,j) · dot(W.col(i), H.col(j))
    //   → O(nnz · k)
    //
    // The reconstruction norm ||WᵀH||² = tr(HᵀWWᵀH) = tr((WᵀW)(HHᵀ))
    //   → O(k²·m + k²·n + k²) using Gram matrices
    //
    // So explicit loss is really: ||A||² - 2·cross + tr(G_W · G_H)
    // where G_W = WW  and G_H = HHᵀ

    const int k = static_cast<int>(W.rows());
    const int n = static_cast<int>(A.cols());

    // 1. ||A||²_F
    Scalar A_sqnorm = 0;
    for (int j = 0; j < n; ++j) {
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            A_sqnorm += it.value() * it.value();
        }
    }

    // 2. Cross term: Σ A(i,j) · W.col(i)ᵀ · H.col(j)
    Scalar cross = 0;
    for (int j = 0; j < n; ++j) {
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            cross += it.value() * W.col(it.row()).dot(H.col(j));
        }
    }

    // 3. Reconstruction norm: tr(WW ᵀ · HHᵀ)
    DenseMatrix<Scalar> GW(k, k), GH(k, k);
    GW.setZero();
    GW.template selfadjointView<Eigen::Lower>().rankUpdate(W);
    GW.template triangularView<Eigen::Upper>() = GW.transpose();

    GH.setZero();
    GH.template selfadjointView<Eigen::Lower>().rankUpdate(H);
    GH.template triangularView<Eigen::Upper>() = GH.transpose();

    Scalar recon_sq = (GW.array() * GH.array()).sum();

    return std::max(A_sqnorm - static_cast<Scalar>(2) * cross + recon_sq,
                    static_cast<Scalar>(0));
}

}  // namespace cpu
}  // namespace primitives
}  // namespace FactorNet

