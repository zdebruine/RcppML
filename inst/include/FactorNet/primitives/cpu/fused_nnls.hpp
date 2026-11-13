// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/fused_nnls.hpp
 * @brief Fused RHS+NNLS batch solver for sparse matrices
 *
 * Fuses the RHS computation (B = Factor · A) with the NNLS solve
 * (G · x = b) into a single OpenMP parallel loop over columns.
 * This eliminates:
 *   - The global B matrix allocation (k × n_cols)
 *   - The BLAS-3 warm start subtraction (B -= G * X)
 *   - A second OpenMP parallel region dispatch
 *
 * Each thread computes a stack-local k-vector RHS from sparse data,
 * applies warm start correction (b -= G * x_old) with G in L1 cache,
 * and immediately solves NNLS while b is still hot in L1.
 *
 * This matches the architecture used by singlet's predict() function
 * and provides 30-50% speedup at low k (k ≤ 16).
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/core/traits.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

// ========================================================================
// Fused RHS + NNLS solve for sparse matrices
// ========================================================================

/**
 * @brief Fused RHS+NNLS: compute B = Factor · A and solve G·x = b in one loop
 *
 * For each column j of A (sparse CSC):
 *   1. Compute b_j = Factor · A[:,j]  (stack-local k-vector), plus the
 *      optional per-column penalty term B_add[:,j]
 *   2. Group (L2,1) test / per-column ridge
 *   3. Warm start: b_j -= G · x_old_j
 *   4. CD solve with absolute L1 (soft-threshold when nonneg = false),
 *      per-column ridge and box constraint applied inside the sweep
 *
 * @tparam SparseMat  Eigen SparseMatrix or Map<const SparseMatrix>
 * @tparam Scalar     float or double
 *
 * @param A           Sparse data matrix (m × n_cols, CSC format)
 * @param Factor      Factor matrix (k × m) — W_T for H-update
 * @param G           Pre-computed Gram matrix (k × k), including the
 *                    penalties shared by all columns (L2, target)
 * @param X           Solution matrix (k × n_cols), updated in-place
 * @param cd_maxit    Max CD iterations per column
 * @param cd_tol      Per-sweep early termination tolerance (0 = disabled)
 * @param L1          Absolute L1 penalty
 * @param nonneg      If true, enforce x ≥ 0
 * @param threads     Number of OpenMP threads
 * @param warm_start  If true, X has previous solution for warm starting
 * @param upper_bound Upper bound on x (0 = disabled)
 * @param B_add       Optional k × n_cols matrix added to the RHS (graph
 *                    neighbor term, target term); nullptr = none
 * @param col_ridge   Optional per-column ridge (length n_cols); nullptr = none
 * @param L21         Group-lasso strength on each column (0 = disabled)
 * @param group_norm  ‖x_prev‖₂ per column (length n_cols), needed if L21 > 0
 * @param cross_out   Optional: receives Σ_j dot(x_j, Factor·A[:,j]), the
 *                    cross term tr(X·(Factor·A)ᵀ) of the Gram-trick loss,
 *                    accumulated in double from the data right-hand side
 *                    (before any penalty term) and the solved column.  This
 *                    replaces a separate O(nnz·k) pass over A for the loss.
 */
template<typename SparseMat, typename Scalar>
inline void fused_rhs_nnls_sparse(
    const SparseMat& A,
    const DenseMatrix<Scalar>& Factor,
    const DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& X,
    int cd_maxit,
    Scalar cd_tol,
    Scalar L1,
    bool nonneg,
    int threads,
    bool warm_start,
    Scalar upper_bound = 0,
    const DenseMatrix<Scalar>* B_add = nullptr,
    const Scalar* col_ridge = nullptr,
    Scalar L21 = 0,
    const Scalar* group_norm = nullptr,
    double* cross_out = nullptr)
{
    const int k = static_cast<int>(Factor.rows());
    const int n = static_cast<int>(A.cols());

    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
        X.setZero();
        warm_start = false;
    }

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
#else
    int n_threads = 1;  // NOLINT(misc-const-correctness)
    (void)threads;
    (void)n_threads;
#endif

    const bool has_group = (L21 > Scalar(0)) && (group_norm != nullptr);
    Scalar g_ref = (k > 0) ? G.diagonal().maxCoeff() : Scalar(1);
    if (!(g_ref > Scalar(0))) g_ref = Scalar(1);

    const bool want_cross = (cross_out != nullptr);
    double cross_acc = 0.0;

    #pragma omp parallel num_threads(n_threads) if(n_threads > 1)
    {
        // Stack-local RHS vector — stays in L1 cache across RHS and NNLS
        DenseVector<Scalar> b(k);
        // Copy of the data RHS for the loss cross term (the solver turns b
        // into a residual)
        DenseVector<Scalar> b_raw(k);

        #pragma omp for schedule(dynamic, 16) reduction(+:cross_acc)
        for (int j = 0; j < n; ++j) {
            // 1. Compute RHS: b = Factor · A[:,j]  (+ penalty term)
            b.setZero();
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                b.noalias() += static_cast<Scalar>(it.value()) * Factor.col(it.row());
            }
            if (want_cross) b_raw = b;
            if (B_add) b.noalias() += B_add->col(j);

            // 2. Per-column ridge and group-lasso step
            Scalar L2_col = col_ridge ? col_ridge[j] : Scalar(0);
            if (has_group) {
                bool is_zero = false;
                L2_col += features::l21_column_ridge(
                    b.data(), k, L1, nonneg, L21, group_norm[j], g_ref, is_zero);
                if (is_zero) {
                    X.col(j).setZero();
                    continue;
                }
            }

            // 3. Warm start: b -= G · x_old  (residual of the unpenalized
            //    system).  Without a warm start x must begin at zero so that
            //    b is a consistent residual.
            if (warm_start) {
                b.noalias() -= G * X.col(j);
            } else {
                X.col(j).setZero();
            }

            // 4. CD solve.  L1 is applied inside CD (absolute shift, or
            //    soft-threshold for signed solutions); the shared ridge is
            //    already in G, the per-column ridge is passed as L2.
            detail::cd_nnls_col_fixed(G, b.data(), X.col(j).data(),
                                      k,
                                      L1,
                                      L2_col,
                                      nonneg, cd_maxit,
                                      upper_bound, cd_tol);

            // 5. Loss cross term: dot(solution, data RHS), in double
            if (want_cross) {
                const Scalar* xj = X.col(j).data();
                double s = 0.0;
                for (int i = 0; i < k; ++i)
                    s += static_cast<double>(xj[i]) * static_cast<double>(b_raw[i]);
                cross_acc += s;
            }
        }
    }  // end parallel

    if (want_cross) *cross_out = cross_acc;
}

// ========================================================================
// Fused RHS + Cholesky NNLS for sparse matrices
// ========================================================================

/**
 * @brief Fused RHS+Cholesky NNLS for sparse matrices
 *
 * Pre-factorizes G = LLᵀ once (O(k³) cost) then for each column j:
 *   1. Compute b_j = Factor · A[:,j]  (O(nnz_j · k)), plus B_add[:,j]
 *   2. Apply the L1 penalty (shift of b_j under non-negativity)
 *   3. Triangular solve via the pre-computed LLT
 *   4. If the unconstrained solution violates a constraint, re-solve on the
 *      positive set (active-set passes), then polish with coordinate-descent
 *      sweeps until the KKT conditions hold (see cholesky_clip.hpp).
 *      Feasible columns need no further work.
 *
 * If the factorization fails, every column is solved by coordinate descent
 * from zero.
 *
 * The Gram may have a wider scalar type than the factors (GScalar = double
 * with Scalar = float): the right-hand side is still accumulated in Scalar,
 * then widened, and the factorization, active-set passes and polish run in
 * GScalar before the solution is rounded back to Scalar.  This removes the
 * cond(G)·eps loss of a single-precision solve on ill-conditioned Grams.
 *
 * @param solver_mode  Unused (kept for API compatibility)
 * @param B_add        Optional k × n_cols matrix added to the RHS
 * @param max_sweeps   Sweep cap for the polish
 * @param cross_out    Optional: receives Σ_j dot(x_j, Factor·A[:,j]) in
 *                     double (see fused_rhs_nnls_sparse)
 */
template<typename SparseMat, typename Scalar, typename GScalar>
inline void fused_rhs_cholesky_sparse(
    const SparseMat& A,
    const DenseMatrix<Scalar>& Factor,
    const DenseMatrix<GScalar>& G,
    DenseMatrix<Scalar>& X,
    int /*solver_mode*/,
    Scalar L1,
    bool nonneg,
    int threads,
    bool /*warm_start*/,
    Scalar upper_bound = 0,
    const DenseMatrix<Scalar>* B_add = nullptr,
    int max_sweeps = static_cast<int>(CD_MAXIT),
    double* cross_out = nullptr)
{
    const int k = static_cast<int>(Factor.rows());
    const int n = static_cast<int>(A.cols());

    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
        X.setZero();
    }

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
#else
    int n_threads = 1;
    (void)threads;
    (void)n_threads;
#endif

    // Pre-factorize G once — O(k³), amortized over n columns
    Eigen::LLT<DenseMatrix<GScalar>> llt(G);
    const bool llt_ok = (llt.info() == Eigen::Success);

    // L1: shift of the RHS under non-negativity, soft-threshold otherwise
    const GScalar L1_shift = (L1 > Scalar(0) && nonneg)
                           ? static_cast<GScalar>(L1) : GScalar(0);
    const GScalar L1_cd    = (L1 > Scalar(0) && !nonneg)
                           ? static_cast<GScalar>(L1) : GScalar(0);
    const GScalar ub_w = static_cast<GScalar>(upper_bound);

    const bool want_cross = (cross_out != nullptr);
    double cross_acc = 0.0;

    #pragma omp parallel num_threads(n_threads) if(n_threads > 1)
    {
        DenseVector<Scalar> b(k);       // data RHS (kept for the loss cross term)
        DenseVector<GScalar> bw(k);     // working RHS / residual
        DenseVector<GScalar> x_j(k);    // working solution
        detail::ActiveSetWorkspace<GScalar> as_ws(k);

        #pragma omp for schedule(dynamic, 16) reduction(+:cross_acc)
        for (int j = 0; j < n; ++j) {
            // 1. Compute RHS: b = Factor · A[:,j]  (+ penalty term)
            b.setZero();
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                b.noalias() += static_cast<Scalar>(it.value()) * Factor.col(it.row());
            }
            // Working copy in the Gram's precision (+ penalty term)
            bw = b.template cast<GScalar>();
            if (B_add) bw.noalias() += B_add->col(j).template cast<GScalar>();

            // 2. L1 penalty
            if (L1_shift > GScalar(0)) bw.array() -= L1_shift;

            // NOTE: No warm start for Cholesky — llt.solve(b) is a direct
            // solve for x = G^{-1}b.

            // 3. Unconstrained solve via pre-factored LLT
            if (llt_ok) {
                x_j = bw;
                llt.solveInPlace(x_j);
            }

            // 4. Exact constrained solution: no-op when x_j is feasible,
            //    active set / clip + KKT polish otherwise (bw becomes the
            //    residual).
            detail::cholesky_finish_col(G, bw.data(), x_j.data(), k, L1_cd,
                                        nonneg, ub_w, max_sweeps,
                                        llt_ok, &as_ws);

            // 5. Store (rounded to Scalar) and accumulate the loss cross
            //    term dot(stored solution, data RHS) in double
            Scalar* xout = X.col(j).data();
            double s = 0.0;
            for (int i = 0; i < k; ++i) {
                const Scalar xf = static_cast<Scalar>(x_j[i]);
                xout[i] = xf;
                s += static_cast<double>(xf) * static_cast<double>(b[i]);
            }
            if (want_cross) cross_acc += s;
        }
    }  // end parallel

    if (want_cross) *cross_out = cross_acc;
}

// ========================================================================
// Loss cross-term for fused path (sparse)
// ========================================================================

/**
 * @brief Compute the Gram-trick loss cross term without materializing B
 *
 * Computes: cross_term = Σᵢ dᵢ · dot(W_T.row(i), [H · A^T].row(i))
 *         = Σⱼ Σ_{ℓ∈nnz(j)} A(ℓ,j) · dot(diag(d)·W_T[:,ℓ], H[:,j])
 *
 * This is O(nnz · k) — same cost as one RHS computation.
 * Used when the fused path skips materializing the global B matrix.
 *
 * @param A     Sparse data matrix (m × n, CSC format)
 * @param W_T   Post-update W factor (k × m), NOT scaled by d
 * @param H     H factor (k × n)
 * @param d     Diagonal scaling vector (k × 1)
 * @param threads Number of OpenMP threads
 * @return      The cross term: trace(diag(d) · W_T · B_w^T) where B_w = H · A^T
 */
template<typename SparseMat, typename Scalar>
inline Scalar loss_cross_term_sparse(
    const SparseMat& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& H,
    const DenseVector<Scalar>& d,
    int threads)
{
    const int k = static_cast<int>(W_T.rows());
    const int n = static_cast<int>(A.cols());

    // Pre-compute U = diag(d) · W_T  (k × m)
    DenseMatrix<Scalar> U = W_T;
    for (int i = 0; i < k; ++i)
        U.row(i) *= d(i);

    Scalar cross_term = 0;

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
    #pragma omp parallel num_threads(n_threads) reduction(+:cross_term)
    {
        DenseVector<Scalar> u_a(k);

        #pragma omp for schedule(dynamic)
        for (int j = 0; j < n; ++j) {
            // u_a = U · A[:,j] = diag(d) · W_T · A[:,j]
            u_a.setZero();
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                u_a.noalias() += static_cast<Scalar>(it.value()) * U.col(it.row());
            }
            cross_term += u_a.dot(H.col(j));
        }
    }
#else
    DenseVector<Scalar> u_a(k);
    for (int j = 0; j < n; ++j) {
        u_a.setZero();
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            u_a.noalias() += static_cast<Scalar>(it.value()) * U.col(it.row());
        }
        cross_term += u_a.dot(H.col(j));
    }
#endif

    return cross_term;
}

/**
 * @brief Transposed variant: cross term using A^T (for W-update loss)
 *
 * Same as loss_cross_term_sparse but iterates columns of A^T.
 * Used after the fused W-update to compute the loss cross term.
 *
 * Computes: cross_term = Σᵢ dᵢ · dot(W_T.row(i), [H · A^T].row(i))
 *
 * @param At    Pre-computed transpose of A (n × m, CSC format)
 * @param W_T   Post-update W factor (k × m), NOT scaled by d
 * @param H     H factor (k × n)
 * @param d     Diagonal scaling vector (k × 1)
 * @param threads Number of OpenMP threads
 */
template<typename SparseMat, typename Scalar>
inline Scalar loss_cross_term_sparse_via_At(
    const SparseMat& At,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& H,
    const DenseVector<Scalar>& d,
    int threads)
{
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(At.cols());

    // We need: Σᵢ dᵢ · dot(W_T.row(i), [H·A^T].row(i))
    // [H·A^T](i, ℓ) = Σⱼ H(i,j) · A(ℓ,j) = Σⱼ H(i,j) · At(j,ℓ)
    //
    // Iterate columns ℓ of At:
    //   For each ℓ: [H·A^T](:,ℓ) = Σ_{j∈nnz(ℓ)} At(j,ℓ) · H(:,j)
    //   contribution = Σᵢ dᵢ · W_T(i,ℓ) · [H·A^T](i,ℓ)
    //               = dot(diag(d)·W_T(:,ℓ), [H·A^T](:,ℓ))
    //
    // This iterates the same nonzeros as the W-update RHS.

    // Pre-compute U = diag(d) · W_T  (k × m)
    DenseMatrix<Scalar> U = W_T;
    for (int i = 0; i < k; ++i)
        U.row(i) *= d(i);

    Scalar cross_term = 0;

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
    #pragma omp parallel num_threads(n_threads) reduction(+:cross_term)
    {
        DenseVector<Scalar> h_at(k);

        #pragma omp for schedule(dynamic)
        for (int ell = 0; ell < m; ++ell) {
            // h_at = H · At[:,ℓ] = Σ_{j∈nnz(ℓ)} At(j,ℓ) · H(:,j)
            h_at.setZero();
            for (typename SparseMat::InnerIterator it(At, ell); it; ++it) {
                h_at.noalias() += static_cast<Scalar>(it.value()) * H.col(it.row());
            }
            // contribution = dot(U(:,ℓ), h_at)
            cross_term += U.col(ell).dot(h_at);
        }
    }
#else
    DenseVector<Scalar> h_at(k);
    for (int ell = 0; ell < m; ++ell) {
        h_at.setZero();
        for (typename SparseMat::InnerIterator it(At, ell); it; ++it) {
            h_at.noalias() += static_cast<Scalar>(it.value()) * H.col(it.row());
        }
        cross_term += U.col(ell).dot(h_at);
    }
#endif

    return cross_term;
}

}  // namespace primitives
}  // namespace FactorNet

