// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/nnls_batch.hpp
 * @brief CPU specialization of the batch NNLS solve primitive
 *
 * Solves  G · x = b  for each column of B, subject to x ≥ 0 (non-negative
 * least squares) using fixed-iteration coordinate descent.  Parallelized
 * over columns with OpenMP.
 *
 * Fixed-iteration CD (no per-element tolerance checking) is used everywhere
 * because benchmarking showed per-element tolerance overhead exactly offsets
 * early-termination savings.  Instead, we use per-SWEEP convergence checking:
 * after each full sweep over k elements, if the max absolute change is below
 * cd_tol, exit early.  This adds only O(1) overhead per sweep while enabling
 * adaptive iteration counts (typically 2-3 sweeps with warm starts).
 *
 * cd_nnls_col_fixed is the ONLY CD solver.  It is used by:
 *   - The standard NMF batch path (via nnls_batch<CPU>)
 *   - The CV/IRLS per-column path (directly)
 *   - The streaming SPZ per-column path (directly)
 *   - The GPU CV host-side IRLS path (directly)
 *
 * This is the main workhorse of NMF — O(k² · n_cols) per call.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/core/constants.hpp>

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

// ========================================================================
// Single-column fixed-iteration CD NNLS solver
// ========================================================================

namespace detail {

/**
 * @brief Fixed-iteration CD NNLS solve for one column
 *
 * Supports L1/L2 regularization, upper-bound box constraints, and optional
 * per-sweep early termination using a **relative convergence criterion**:
 *
 *   sum_i |diff_i / (x[i] + cd_abs_tol)| / k  <  cd_tol
 *
 * This matches singlet's convergence criterion, which naturally adapts to the
 * scale of the solution.  Large x values tolerate small absolute changes;
 * small x values continue iterating until truly converged.
 *
 * With cd_abs_tol = 1e-15 (default) and cd_tol = 1e-8, this exactly matches
 * singlet's `(tol / b.size()) > 1e-8` criterion.
 */
template<typename Scalar>
inline int cd_nnls_col_fixed(const DenseMatrix<Scalar>& G,
                              Scalar* __restrict__ b,
                              Scalar* __restrict__ x,
                              int k,
                              Scalar L1,
                              Scalar L2,
                              bool nonneg,
                              int maxit,
                              Scalar upper_bound = 0,
                              Scalar cd_tol = 0)
{
    const bool has_upper = (upper_bound > 0);
    const bool check_convergence = (cd_tol > 0);
    const Scalar inv_k = Scalar(1) / static_cast<Scalar>(k);

    for (int iter = 0; iter < maxit; ++iter) {
        Scalar tol_sum = 0;
        for (int i = 0; i < k; ++i) {
            const Scalar g_diag = G(i, i);
            if (g_diag <= Scalar(0)) continue;

            Scalar diff = b[i] / g_diag;

            if (L1 != 0) diff -= L1;
            if (L2 != 0) diff += L2 * x[i];

            Scalar new_val = x[i] + diff;
            Scalar actual_diff;

            if (nonneg && new_val < Scalar(0)) {
                actual_diff = -x[i];
                if (actual_diff == Scalar(0)) continue;
                x[i] = Scalar(0);
            } else if (has_upper && new_val > upper_bound) {
                actual_diff = upper_bound - x[i];
                if (actual_diff == Scalar(0)) continue;
                x[i] = upper_bound;
            } else {
                if (diff == Scalar(0)) continue;
                actual_diff = diff;
                x[i] = new_val;
            }

            // Accumulate relative change for convergence check
            if (check_convergence) {
                const Scalar abs_diff = (actual_diff >= 0) ? actual_diff : -actual_diff;
                tol_sum += abs_diff / (std::abs(x[i]) + static_cast<Scalar>(CD_ABS_TOL));
            }

            // Update residual: b -= G(:,i) * actual_diff
            const Scalar* __restrict__ g_col = G.data() + static_cast<std::ptrdiff_t>(i) * k;
            for (int r = 0; r < k; ++r) {
                b[r] -= g_col[r] * actual_diff;
            }
        }
        // Per-sweep early termination: average relative change < threshold
        if (check_convergence && tol_sum * inv_k < cd_tol) {
            return iter + 1;
        }
    }
    return maxit;
}

}  // namespace detail

// ========================================================================
// Batch NNLS — CPU specialization (fixed-iteration CD)
// ========================================================================

/**
 * @brief CPU batch NNLS: solve G·x = b for each column of B, x ≥ 0
 *
 * Uses coordinate descent with optional per-sweep early termination.
 * Parallelized over columns with OpenMP.
 * BLAS-3 warm start: B -= G * X.
 *
 * B is modified in-place (residual tracking).
 * X receives the solution.
 */
template<>
inline void nnls_batch<CPU, double>(
    const DenseMatrix<double>& G,
    DenseMatrix<double>& B,
    DenseMatrix<double>& X,
    int cd_maxit,
    double cd_tol,
    double L1,
    double L2,
    bool nonneg,
    int threads,
    double upper_bound,
    bool warm_start)
{
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(B.cols());

    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
        X.setZero();
    } else if (warm_start) {
        B.noalias() -= G * X;
    } else {
        X.setZero();
    }

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        detail::cd_nnls_col_fixed(G, B.col(j).data(), X.col(j).data(),
                                  k, L1, L2, nonneg, cd_maxit, upper_bound,
                                  cd_tol);
    }
}

/**
 * @brief CPU batch NNLS (float specialization)
 */
template<>
inline void nnls_batch<CPU, float>(
    const DenseMatrix<float>& G,
    DenseMatrix<float>& B,
    DenseMatrix<float>& X,
    int cd_maxit,
    float cd_tol,
    float L1,
    float L2,
    bool nonneg,
    int threads,
    float upper_bound,
    bool warm_start)
{
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(B.cols());

    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
        X.setZero();
    } else if (warm_start) {
        B.noalias() -= G * X;
    } else {
        X.setZero();
    }

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        detail::cd_nnls_col_fixed(G, B.col(j).data(), X.col(j).data(),
                                  k, L1, L2, nonneg, cd_maxit, upper_bound,
                                  cd_tol);
    }
}

}  // namespace primitives
}  // namespace FactorNet

