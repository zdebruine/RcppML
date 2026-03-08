// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/cholesky_clip.hpp
 * @brief Cholesky-solve-then-clip batch NNLS: unconstrained LS + max(0,x)
 *
 * Alternative to coordinate descent NNLS for non-negative least squares.
 * Instead of interleaving constraints with the solve (CD NNLS), this approach:
 *   1. Solves the unconstrained system G * X = B via Cholesky factorization
 *   2. Clips negative values: X = max(0, X)
 *
 * This is ~20× cheaper per column than CD NNLS (O(k³ + k²n) vs O(cd_iter·k²·n))
 * but produces an approximate solution since clipping ignores cross-coordinate
 * coupling. Works best when starting near the true nonneg solution (warm start
 * from Lanczos initialization or later iterations).
 *
 * The factorization G = L·L^T is computed ONCE, then reused for all n columns.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

// ========================================================================
// Per-column Cholesky solvers (for CV NMF per-column Gram correction)
// ========================================================================
namespace detail {

/**
 * @brief Per-column Cholesky+clip NNLS: solve G·x = b then clip x ≥ 0
 *
 * Drop-in replacement for cd_nnls_col() when solver_mode == 1.
 * Since CV NMF modifies G per column (Gram correction), we must factorize
 * per column. For k ≤ 64, this is still very fast (~k³ flops per column).
 *
 * Note: b is treated as the ORIGINAL RHS (not residual form).
 * The maxit and tol parameters are ignored (direct solve).
 *
 * @param G            Gram matrix (k × k), possibly corrected per column
 * @param b            RHS vector (k), modified in-place (L1 subtraction)
 * @param x            Solution vector (k), updated in-place
 * @param k            Dimension
 * @param L1           L1 penalty (subtracted from b before solve)
 * @param L2           L2 penalty (added to G diagonal if > 0)
 * @param nonneg       Enforce non-negativity via clipping
 * @param maxit        Ignored (direct solve)
 * @param tol          Ignored (direct solve)
 * @param upper_bound  Upper bound for box constraint (0 = no upper bound)
 */
template<typename Scalar>
inline void cholesky_clip_col(const DenseMatrix<Scalar>& G,
                              Scalar* __restrict__ b,
                              Scalar* __restrict__ x,
                              int k,
                              Scalar L1,
                              Scalar L2,
                              bool nonneg,
                              int /*maxit*/,
                              Scalar /*tol*/,
                              Scalar upper_bound = 0)
{
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> b_vec(b, k);
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_vec(x, k);

    // Apply L1 to RHS (soft-thresholding)
    if (L1 > 0) {
        b_vec.array() -= L1;
    }

    // Apply L2 to Gram diagonal if needed (CV path typically has L2=0)
    if (L2 > 0) {
        DenseMatrix<Scalar> G_reg = G;
        G_reg.diagonal().array() += L2;
        Eigen::LLT<DenseMatrix<Scalar>> llt(G_reg);
        x_vec = b_vec;
        llt.solveInPlace(x_vec);
    } else {
        Eigen::LLT<DenseMatrix<Scalar>> llt(G);
        x_vec = b_vec;
        llt.solveInPlace(x_vec);
    }

    // Clip negatives
    if (nonneg) {
        x_vec = x_vec.cwiseMax(static_cast<Scalar>(0));
    }

    // Upper bound
    if (upper_bound > 0) {
        x_vec = x_vec.cwiseMin(upper_bound);
    }
}

}  // namespace detail

/**
 * @brief Batch Cholesky+clip NNLS: solve G·X = B then clip X ≥ 0
 *
 * Solves min_X ||G·X - B||² for each column, then applies max(0,·).
 * G is k×k (Gram matrix), B is k×n (RHS), X is k×n (output).
 *
 * When warm_start=true, B is adjusted to residual form B -= G*X before solving,
 * matching the CD NNLS warm-start convention. However, for Cholesky we actually
 * re-solve from scratch (the warm start just keeps the initial B adjustment
 * consistent with the caller's expectation that B is modified in-place).
 *
 * @param G         Gram matrix (k × k), symmetric positive definite
 * @param B         RHS matrix (k × n), modified in-place
 * @param X         Solution matrix (k × n), updated in-place
 * @param nonneg    If true, clip negatives to zero after solve
 * @param threads   Number of OpenMP threads (0 = all available)
 * @param warm_start If true, subtract G*X from B first (for API compatibility)
 */
template<typename Scalar>
inline void cholesky_clip_batch(
    const DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& X,
    bool nonneg,
    int threads,
    bool warm_start)
{
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(B.cols());

    // Ensure X is properly sized
    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
        X.setZero();
        warm_start = false;  // can't warm start with wrong-sized X
    }

    // NOTE: B is always FRESH (recomputed by rhs<Resource> each NMF iteration).
    // Unlike CD which converts B to residual form (B -= G*X), Cholesky solves
    // G*X = B directly — no warm-start adjustment to B is needed.
    // warm_start is accepted for API compatibility but has no effect here.
    (void)warm_start;

    // Cholesky factorization of G (done once)
    Eigen::LLT<DenseMatrix<Scalar>> llt(G);

    // Solve in-place: X = G^{-1} * B
    X = B;  // copy B into X
    llt.solveInPlace(X);

    // Clip negatives if nonneg constraint is active
    if (nonneg) {
        X = X.cwiseMax(static_cast<Scalar>(0));
    }
}

}  // namespace primitives
}  // namespace FactorNet

