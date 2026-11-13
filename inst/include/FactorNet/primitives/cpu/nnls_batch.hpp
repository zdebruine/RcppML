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
 * Convergence is checked once per sweep with a KKT test (largest violation
 * of the optimality conditions), not with the size of the last change: on an
 * ill-conditioned Gram the changes become small long before the iterate is
 * optimal.  A column that reaches the sweep cap without passing the test is
 * finished by the exact active-set solve where the problem allows it.
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
#include <FactorNet/features/L21.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

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
 * Supports L1/L2 regularization and upper-bound box constraints.  A sweep
 * ends the solve when the largest KKT violation is at most
 *
 *   max(cd_tol, kkt_rel_tol(k)) · scale,
 *
 * where kkt_rel_tol is the precision floor of the scalar type and scale
 * estimates max|b₀_i|.  `maxit` caps the sweeps; at the cap the column is
 * handed to the exact active-set solve (see the end of the function).
 *
 * Objective (per column):
 *
 *   min_x  ½ xᵀG x − b₀ᵀx + L1·‖x‖₁ + ½·L2·‖x‖²   s.t.  lower ≤ x ≤ upper
 *
 * Conventions:
 *   - `b` is the residual of the UNPENALIZED system on entry: b = b₀ − G·x.
 *     It is kept as that residual throughout (only G enters the update).
 *   - L1 is an ABSOLUTE shift of the right-hand side.  With nonneg = true the
 *     coordinate step is max(0, (c − L1)/(G_ii + L2)); with nonneg = false it
 *     is the soft-threshold S(c, L1)/(G_ii + L2), where c = b_i + G_ii·x_i.
 *   - L2 is a ridge that is NOT already in G.  A caller that has added the
 *     ridge to the diagonal of G must pass L2 = 0.
 *   - upper_bound > 0 enforces x_i ≤ upper_bound inside the sweep (box QP).
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
    const bool has_pen = (L1 != Scalar(0)) || (L2 != Scalar(0));

    // Stopping rule: largest violation of the optimality (KKT) conditions.
    // For coordinate i the exact coordinate step d_i satisfies
    // |d_i|·(G_ii + L2) = |projected gradient_i|, so the largest such value
    // over a sweep is the KKT violation.  The sweep stops when it is below
    // a relative tolerance times the scale of the right-hand side.
    //
    // A change-based rule (mean relative change per sweep) is NOT a
    // convergence test: on an ill-conditioned Gram coordinate descent moves
    // slowly, so the changes are small while the iterate is still far from
    // the optimum.
    //
    // The tolerance is floored at what the scalar type can reach
    // (kkt_rel_tol); cd_tol only loosens it.  The scale max_i(|r_i| +
    // G_ii·|x_i|) is a cheap lower estimate of max|b₀_i| for the
    // non-negative systems solved here.
    Scalar scale = 0;
    for (int i = 0; i < k; ++i) {
        const Scalar g = G(i, i);
        const Scalar a = std::abs(b[i]) + ((g > Scalar(0)) ? g : Scalar(0)) * std::abs(x[i]);
        if (a > scale) scale = a;
    }
    Scalar rel_tol = kkt_rel_tol<Scalar>(k);
    if (cd_tol > rel_tol) rel_tol = cd_tol;
    const Scalar tol = rel_tol * scale;

    for (int iter = 0; iter < maxit; ++iter) {
        Scalar max_viol = 0;
        for (int i = 0; i < k; ++i) {
            const Scalar g_diag = G(i, i);
            const Scalar denom = g_diag + L2;
            Scalar new_val;

            if (!has_pen) {
                if (g_diag <= Scalar(0)) continue;
                new_val = x[i] + b[i] / g_diag;
            } else {
                if (denom <= Scalar(0)) continue;
                // c = b₀_i − Σ_{j≠i} G_ij x_j
                Scalar c = b[i] + g_diag * x[i];
                if (nonneg) {
                    c -= L1;
                } else if (c > L1) {
                    c -= L1;
                } else if (c < -L1) {
                    c += L1;
                } else {
                    c = Scalar(0);
                }
                new_val = c / denom;
            }

            if (nonneg && new_val < Scalar(0)) new_val = Scalar(0);
            if (has_upper && new_val > upper_bound) new_val = upper_bound;

            const Scalar actual_diff = new_val - x[i];
            if (actual_diff == Scalar(0)) continue;
            x[i] = new_val;

            const Scalar viol =
                ((actual_diff >= Scalar(0)) ? actual_diff : -actual_diff) * denom;
            if (viol > max_viol) max_viol = viol;

            // Update residual: b -= G(:,i) * actual_diff
            const Scalar* __restrict__ g_col = G.data() + static_cast<std::ptrdiff_t>(i) * k;
            for (int r = 0; r < k; ++r) {
                b[r] -= g_col[r] * actual_diff;
            }
        }
        if (max_viol <= tol) {
            return iter + 1;
        }
    }

    // Sweep cap reached without meeting the KKT test: the Gram is
    // ill-conditioned (coordinate descent needs on the order of cond(G)
    // sweeps).  Where the problem has the form the direct solver handles —
    // non-negative with L1/L2, or unconstrained without L1; no upper bound —
    // finish the column with the exact active-set solve instead of returning
    // an unconverged iterate.
    if (!has_upper && (nonneg || L1 == Scalar(0)) && maxit >= 8) {
        ActiveSetWorkspace<Scalar> ws(k);
        Scalar* b0 = ws.b0.data();
        // b₀ = r + G·x (unpenalized right-hand side)
        for (int i = 0; i < k; ++i) b0[i] = b[i];
        for (int i = 0; i < k; ++i) {
            const Scalar xi = x[i];
            if (xi == Scalar(0)) continue;
            const Scalar* g_col = G.data() + static_cast<std::ptrdiff_t>(i) * k;
            for (int r = 0; r < k; ++r) b0[r] += g_col[r] * xi;
        }
        Scalar b_scale = 0;
        if (nonneg && L1 != Scalar(0)) {
            for (int i = 0; i < k; ++i) b0[i] -= L1;   // exact for x ≥ 0
        }
        for (int i = 0; i < k; ++i) {
            const Scalar a = std::abs(b0[i]);
            if (a > b_scale) b_scale = a;
        }
        active_set_solve(G, b0, x, k, ws, kkt_rel_tol<Scalar>(k) * b_scale,
                         L2, nonneg);
        // Keep b the residual of the unpenalized system at the new x
        for (int i = 0; i < k; ++i)
            b[i] = b0[i] + ((nonneg && L1 != Scalar(0)) ? L1 : Scalar(0));
        for (int i = 0; i < k; ++i) {
            const Scalar xi = x[i];
            if (xi == Scalar(0)) continue;
            const Scalar* g_col = G.data() + static_cast<std::ptrdiff_t>(i) * k;
            for (int r = 0; r < k; ++r) b[r] -= g_col[r] * xi;
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
    #pragma omp parallel for schedule(dynamic, 16) num_threads(n_threads)
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
    #pragma omp parallel for schedule(dynamic, 16) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        detail::cd_nnls_col_fixed(G, B.col(j).data(), X.col(j).data(),
                                  k, L1, L2, nonneg, cd_maxit, upper_bound,
                                  cd_tol);
    }
}

// ========================================================================
// Batch NNLS with per-column ridge and group (L2,1) penalty
// ========================================================================

/**
 * @brief CD batch solve where each column carries its own ridge
 *
 * Solves, for each column j,
 *
 *   min_x ½ xᵀG x − b_jᵀx + L1‖x‖₁ + ½·ridge_j·‖x‖² + L21·‖x‖₂   (+ box)
 *
 * The group term is handled by one majorization step around the previous
 * iterate (ridge L21/‖x_prev‖), with an exact test for the all-zero solution.
 *
 * A ridge that differs from column to column rules out one shared Cholesky
 * factorization, so the graph and L21 penalties are always solved here by
 * coordinate descent, whatever solver was requested.  Both solvers are exact,
 * so the result is the same.
 *
 * @param B           Unpenalized right-hand side (k × n), modified in place
 * @param col_ridge   Per-column ridge (length n) or nullptr
 * @param L21         Group-lasso strength (0 = disabled)
 * @param group_norm  ‖x_prev‖₂ per column (length n); required when L21 > 0
 */
template<typename Scalar>
inline void nnls_batch_percol(
    const DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& X,
    int cd_maxit,
    Scalar cd_tol,
    Scalar L1,
    bool nonneg,
    int threads,
    Scalar upper_bound,
    bool warm_start,
    const Scalar* col_ridge,
    Scalar L21,
    const Scalar* group_norm)
{
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(B.cols());

    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
        X.setZero();
        warm_start = false;
    }

    const bool has_group = (L21 > Scalar(0)) && (group_norm != nullptr);
    Scalar g_ref = (k > 0) ? G.diagonal().maxCoeff() : Scalar(1);
    if (!(g_ref > Scalar(0))) g_ref = Scalar(1);

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 16) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        Scalar L2_col = col_ridge ? col_ridge[j] : Scalar(0);
        if (has_group) {
            bool is_zero = false;
            L2_col += features::l21_column_ridge(
                B.col(j).data(), k, L1, nonneg, L21, group_norm[j], g_ref, is_zero);
            if (is_zero) {
                X.col(j).setZero();
                continue;
            }
        }
        if (warm_start) {
            B.col(j).noalias() -= G * X.col(j);
        } else {
            X.col(j).setZero();
        }
        detail::cd_nnls_col_fixed(G, B.col(j).data(), X.col(j).data(),
                                  k, L1, L2_col, nonneg, cd_maxit, upper_bound,
                                  cd_tol);
    }
}

}  // namespace primitives
}  // namespace FactorNet

