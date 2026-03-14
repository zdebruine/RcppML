// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives.hpp
 * @brief Primitive API — the ONLY backend-specific interface
 *
 * Every NMF iteration reduces to four operations on the data matrix.
 * These are the ONLY operations that differ between CPU and GPU:
 *
 *   1. gram(H, G)           — G = H · Hᵀ  (k × k)
 *   2. rhs(A, H, B)         — B = H · A   (k × m)  or  B = H · Aᵀ  (k × n)
 *   3. nnls_batch(G, B, X)  — solve G·x = b  for each column of B, x ≥ 0
 *   4. trace_AtA(A)         — tr(AᵀA) = ||A||²_F  (precomputed once)
 *
 * The algorithm layer calls these with a Resource tag (CPU or GPU).
 * Feature application lives above this layer and never touches A.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/core/traits.hpp>

namespace FactorNet {
namespace primitives {

// ========================================================================
// Primitive function declarations
// ========================================================================
// Each is template-specialized per Resource in cpu/ or gpu/ subdirectories.

/**
 * @brief Gram matrix: G = H · Hᵀ  (k × k)
 *
 * Uses selfadjointView rank update for ~2× speedup on CPU.
 * Uses cuBLAS DGEMM on GPU.
 */
template<typename Resource, typename Scalar>
void gram(const DenseMatrix<Scalar>& H,
          DenseMatrix<Scalar>& G);

/**
 * @brief Right-hand side (sparse A): B = H · A  (k × m or k × n)
 *
 * @tparam MatrixType  Sparse or dense matrix type
 */
template<typename Resource, typename Scalar, typename MatrixType>
void rhs(const MatrixType& A,
         const DenseMatrix<Scalar>& H,
         DenseMatrix<Scalar>& B);

/**
 * @brief NNLS batch solve: solve G·x = b for each column of B, x ≥ 0
 *
 * B is used as the initial RHS and is modified in-place (residual tracking).
 * X receives the solution.
 *
 * @param G          Gram matrix (k × k)
 * @param B          RHS matrix (k × n_cols), modified in-place
 * @param X          Output solution (k × n_cols)
 * @param cd_maxit   CD iterations per column
 * @param cd_tol     CD convergence tolerance
 * @param L1         L1 penalty (applied in CD loop)
 * @param L2         L2 penalty (applied in CD loop)
 * @param nonneg     Enforce non-negativity
 * @param threads    Number of OpenMP threads (CPU only, 0 = all)
 * @param upper_bound  Upper bound for box constraint (0 = no upper bound)
 * @param warm_start   If true and X has correct dimensions, use X as
 *                      initial guess (adjusts B -= G*X internally).
 *                      If false, X is zeroed (cold start).
 */
template<typename Resource, typename Scalar>
void nnls_batch(const DenseMatrix<Scalar>& G,
                DenseMatrix<Scalar>& B,
                DenseMatrix<Scalar>& X,
                int cd_maxit,
                Scalar cd_tol,
                Scalar L1 = 0,
                Scalar L2 = 0,
                bool nonneg = true,
                int threads = 0,
                Scalar upper_bound = 0,
                bool warm_start = false);

/**
 * @brief Precompute tr(AᵀA) = ||A||²_F for Gram-trick loss
 *
 * Called once before the iteration loop.
 * Uses if constexpr to dispatch sparse vs dense—avoids explicit
 * specializations whose inline bodies GCC may fail to emit.
 */
template<typename Resource, typename Scalar, typename MatrixType>
inline Scalar trace_AtA(const MatrixType& A) {
    static_assert(std::is_same_v<Resource, CPU>,
                  "Only CPU resource is supported for trace_AtA");
    if constexpr (is_sparse_v<MatrixType>) {
        Scalar total = 0;
        for (int j = 0; j < A.cols(); ++j) {
            for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                total += it.value() * it.value();
            }
        }
        return total;
    } else {
        return A.squaredNorm();
    }
}

/**
 * @brief Gram-trick loss: ||A - WH||² from precomputed G, B, H
 *
 * ||A - WH||² = tr(AᵀA) - 2·tr(BᵀH) + tr(G · HHᵀ)
 *
 * Cost: O(k² · n) — essentially free using already-computed Gram/RHS.
 * This is resource-independent (operates on dense matrices only).
 */
template<typename Scalar>
inline Scalar gram_trick_loss(Scalar trAtA,
                              const DenseMatrix<Scalar>& G,
                              const DenseMatrix<Scalar>& B,
                              const DenseMatrix<Scalar>& H)
{
    Scalar trBtH = (B.array() * H.array()).sum();
    DenseMatrix<Scalar> GH = G * H;
    Scalar trGHHt = (H.array() * GH.array()).sum();
    return std::max(trAtA - static_cast<Scalar>(2) * trBtH + trGHHt,
                    static_cast<Scalar>(0));
}

/**
 * @brief Scale rows of H by d: H_scaled[i,:] = H[i,:] * d[i]
 *
 * Used to form W*diag(d) or H*diag(d) before gram/rhs.
 * Backend-agnostic (tiny operation) but declared here for consistency.
 */
template<typename Scalar>
inline void scale_rows(DenseMatrix<Scalar>& H,
                       const DenseVector<Scalar>& d) {
    for (int i = 0; i < d.size(); ++i) {
        H.row(i) *= d(i);
    }
}

/**
 * @brief Normalize factor: extract row norms into d, divide rows by norms
 *
 * After NNLS solve: d[i] *= norm(H[i,:]) and H[i,:] /= norm(H[i,:]).
 * Used to maintain W·diag(d)·H factorization.
 *
 * @param norm_type  L1, L2, or None normalization
 */
template<typename Scalar>
inline void normalize_rows(DenseMatrix<Scalar>& H,
                           DenseVector<Scalar>& d,
                           NormType norm_type = NormType::L1) {
    if (norm_type == NormType::None) return;
    for (int i = 0; i < d.size(); ++i) {
        Scalar norm = (norm_type == NormType::L1)
            ? H.row(i).template lpNorm<1>()
            : H.row(i).norm();
        if (norm > 0) {
            d(i) *= norm;
            H.row(i) /= norm;
        }
    }
}

}  // namespace primitives
}  // namespace FactorNet

