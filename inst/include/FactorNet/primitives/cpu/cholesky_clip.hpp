// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/cholesky_clip.hpp
 * @brief Cholesky-initialized exact NNLS: unconstrained solve, then polish
 *
 * Alternative to pure coordinate descent for (box-constrained) least squares:
 *   1. Solve the unconstrained system G * X = B via Cholesky factorization.
 *   2. Columns whose unconstrained solution is feasible are already optimal
 *      and cost nothing more.
 *   3. Columns that violate a constraint are finished exactly.  Plain NNLS
 *      (x ≥ 0, no bound, no signed L1) uses an active-set solve that checks
 *      both the primal and the dual KKT conditions (active_set_solve).
 *      Columns with an upper bound or a signed L1 penalty are clipped to
 *      the feasible box and polished by coordinate-descent sweeps until the
 *      KKT conditions hold or a sweep cap is reached; on a badly
 *      conditioned Gram that polish can stop at the cap short of the
 *      optimum.
 *
 * The graph and L21 penalties add a ridge that differs per column, which one
 * shared factorization cannot represent.  Callers route those cases to
 * coordinate descent (nnls_batch_percol / the fused CD path) instead; both
 * solvers are exact, so the result does not depend on the choice.
 *
 * Clipping alone is NOT a solution of the constrained problem (the gradient
 * on the retained coordinates is G_PN·|x_N| instead of zero), so step 3 is
 * what makes this solver exact.
 *
 * If the factorization fails (G not numerically positive definite) or the
 * solve returns non-finite values, the column is solved by coordinate descent
 * from zero instead.
 *
 * The factorization G = L·L^T is computed ONCE in the batch entry points and
 * reused for all n columns.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/constants.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

namespace detail {

template<typename Scalar>
inline Scalar kkt_rel_tol(int k);

/**
 * @brief Coordinate-descent polish to KKT for one column
 *
 * Minimizes ½ xᵀG x − b₀ᵀx + L1‖x‖₁ over the box (x ≥ 0 when nonneg,
 * x ≤ upper_bound when upper_bound > 0), starting from a feasible x.
 *
 * @param r           Residual of the unpenalized system, r = b₀ − G·x, on
 *                    entry; maintained in place
 * @param L1          Absolute L1 penalty (0 when already folded into b₀)
 * @param kkt_scale   Scale of the right-hand side (e.g. max |b₀_i|); the
 *                    sweep stops once the largest coordinate-wise KKT
 *                    violation is below a relative tolerance times this scale
 * @return            Number of sweeps performed
 */
template<typename Scalar>
inline int kkt_polish_col(const DenseMatrix<Scalar>& G,
                          Scalar* __restrict__ r,
                          Scalar* __restrict__ x,
                          int k,
                          Scalar L1,
                          bool nonneg,
                          Scalar upper_bound,
                          int max_sweeps,
                          Scalar kkt_scale)
{
    const bool has_upper = (upper_bound > Scalar(0));
    const bool has_L1 = (L1 > Scalar(0));
    const Scalar tol = kkt_rel_tol<Scalar>(k) * kkt_scale;
    // A coordinate whose violation is already far below the tolerance is
    // left alone: updating it would cost a length-k residual update for a
    // change at rounding level.  A column that is already optimal therefore
    // costs k divisions, not k².
    const Scalar skip_tol = tol * static_cast<Scalar>(0.25);

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        Scalar max_viol = 0;
        for (int i = 0; i < k; ++i) {
            const Scalar g = G(i, i);
            if (g <= Scalar(0)) continue;

            Scalar new_val;
            if (!has_L1) {
                new_val = x[i] + r[i] / g;
            } else {
                Scalar c = r[i] + g * x[i];
                if (nonneg) {
                    c -= L1;
                } else if (c > L1) {
                    c -= L1;
                } else if (c < -L1) {
                    c += L1;
                } else {
                    c = Scalar(0);
                }
                new_val = c / g;
            }
            if (nonneg && new_val < Scalar(0)) new_val = Scalar(0);
            if (has_upper && new_val > upper_bound) new_val = upper_bound;

            const Scalar diff = new_val - x[i];
            if (diff == Scalar(0)) continue;

            // |diff|·G_ii is the projected-gradient violation at coordinate i
            const Scalar viol = ((diff >= Scalar(0)) ? diff : -diff) * g;
            if (viol <= skip_tol) continue;
            x[i] = new_val;
            if (viol > max_viol) max_viol = viol;

            const Scalar* __restrict__ g_col =
                G.data() + static_cast<std::ptrdiff_t>(i) * k;
            for (int q = 0; q < k; ++q) {
                r[q] -= g_col[q] * diff;
            }
        }
        if (max_viol <= tol) return sweep + 1;
    }
    return max_sweeps;
}

/**
 * @brief Scratch space for active_set_solve / cholesky_finish_col
 */
template<typename Scalar>
struct ActiveSetWorkspace {
    std::vector<Scalar> L;      ///< k × k Cholesky factor of G[P,P] (lower)
    std::vector<Scalar> y;      ///< k: solution on P
    std::vector<Scalar> b0;     ///< k: pristine copy of the right-hand side
    std::vector<int>    idx;    ///< k: indices in the passive (free) set P
    std::vector<char>   state;  ///< k: 0 = at zero, 1 = in P, 2 = excluded

    explicit ActiveSetWorkspace(int k)
        : L(static_cast<std::size_t>(k) * static_cast<std::size_t>(k)),
          y(static_cast<std::size_t>(k)),
          b0(static_cast<std::size_t>(k)),
          idx(static_cast<std::size_t>(k)),
          state(static_cast<std::size_t>(k)) {}
};

/**
 * @brief Relative KKT tolerance a scalar type can reach for a k-vector solve
 *
 * A residual b − G·x that is zero in exact arithmetic is computed with an
 * error of a few k·eps times the size of its terms, so the threshold grows
 * with k.  Multiply by the scale of the right-hand side (max |b_i|).
 */
template<typename Scalar>
inline Scalar kkt_rel_tol(int k)
{
    const Scalar mult = static_cast<Scalar>((k > 8) ? 4 * k : 32);
    return std::max(mult * std::numeric_limits<Scalar>::epsilon(),
                    static_cast<Scalar>(1e-8));
}

/**
 * @brief Exact active-set solve of  min ½xᵀ(G + δI)x − bᵀx,  x ≥ 0
 *
 * Starts from the support of x (P = {i : x_i > 0}; empty if x = 0) and
 * alternates two steps until the KKT conditions hold:
 *
 *   primal: solve (G + δI)[P,P]·z = b[P] by Cholesky; drop every index with
 *           z_i ≤ 0 and re-solve until z > 0 on P;
 *   dual:   for i ∉ P compute r_i = b_i − (G·x)_i; if no r_i exceeds
 *           dual_tol the point is optimal; otherwise put the violating
 *           indices back into P (all of them in the first three rounds, then
 *           only the largest, which rules out cycling in practice).
 *
 * Dropping the non-positive coordinates alone (the "fast" active set of
 * RcppML 0.3.7) is not enough: a coordinate dropped while others were still
 * wrong often belongs in the solution, and coordinate descent needs on the
 * order of cond(G) sweeps to bring it back, far more than the sweep cap on
 * an ill-conditioned Gram.  The dual step puts it back directly.
 *
 * An index whose pivot vanishes (a column linearly dependent on the ones
 * before it, e.g. duplicated factors) is excluded for the rest of the solve,
 * so a singular Gram is handled without a ridge.
 *
 * @param b          Right-hand side (length k), not modified
 * @param x          Starting point on entry (only its support is used);
 *                   solution on exit
 * @param dual_tol   Absolute tolerance on the dual condition
 * @param diag_add   Ridge δ that is not already in G
 * @param nonneg     false: plain linear solve on all coordinates
 * @return           true if the KKT conditions hold; false if the round cap
 *                   was reached (x is then a feasible iterate)
 */
template<typename Scalar>
inline bool active_set_solve(const DenseMatrix<Scalar>& G,
                             const Scalar* b,
                             Scalar* x,
                             int k,
                             ActiveSetWorkspace<Scalar>& ws,
                             Scalar dual_tol,
                             Scalar diag_add = 0,
                             bool nonneg = true)
{
    Scalar* L = ws.L.data();
    Scalar* y = ws.y.data();
    int* idx = ws.idx.data();
    char* state = ws.state.data();
    const Scalar* Gd = G.data();
    const std::ptrdiff_t ld = static_cast<std::ptrdiff_t>(k);
    const Scalar piv_rel =
        static_cast<Scalar>(16) * std::numeric_limits<Scalar>::epsilon();
    const int max_round = 2 * k + 8;

    int np = 0;
    for (int i = 0; i < k; ++i) {
        if (!nonneg || x[i] > Scalar(0)) {
            idx[np++] = i;
            state[i] = 1;
        } else {
            state[i] = 0;
        }
    }

    for (int round = 0; ; ++round) {
        // ---- primal step: solve on P, drop non-positive, repeat ----------
        while (np > 0) {
            // Cholesky of (G + δI)[P,P], lower triangle, leading dimension k
            int bad = -1;
            for (int j = 0; j < np && bad < 0; ++j) {
                const Scalar* g_col = Gd + static_cast<std::ptrdiff_t>(idx[j]) * ld;
                const Scalar g_jj = g_col[idx[j]] + diag_add;
                Scalar ljj = Scalar(1);
                for (int i = j; i < np; ++i) {
                    Scalar s = (i == j) ? g_jj : g_col[idx[i]];
                    for (int t = 0; t < j; ++t) {
                        s -= L[i + t * ld] * L[j + t * ld];
                    }
                    if (i == j) {
                        if (!(s > piv_rel * g_jj)) {
                            bad = j;
                            break;
                        }
                        ljj = std::sqrt(s);
                        L[j + j * ld] = ljj;
                    } else {
                        L[i + j * ld] = s / ljj;
                    }
                }
            }
            if (bad >= 0) {
                // Linearly dependent on the earlier columns: exclude it
                state[idx[bad]] = 2;
                for (int t = bad; t + 1 < np; ++t) idx[t] = idx[t + 1];
                --np;
                continue;
            }
            // Forward substitution: L·u = b[P]
            for (int i = 0; i < np; ++i) {
                Scalar s = b[idx[i]];
                for (int t = 0; t < i; ++t) s -= L[i + t * ld] * y[t];
                y[i] = s / L[i + i * ld];
            }
            // Back substitution: Lᵀ·z = u
            for (int i = np - 1; i >= 0; --i) {
                Scalar s = y[i];
                for (int t = i + 1; t < np; ++t) s -= L[t + i * ld] * y[t];
                y[i] = s / L[i + i * ld];
            }
            if (!nonneg) break;
            // Keep the strictly positive part
            int nn = 0;
            for (int i = 0; i < np; ++i) {
                if (y[i] > Scalar(0)) {
                    idx[nn] = idx[i];
                    y[nn] = y[i];
                    ++nn;
                } else {
                    state[idx[i]] = 0;
                }
            }
            if (nn == np) break;
            np = nn;
        }

        for (int i = 0; i < k; ++i) x[i] = Scalar(0);
        for (int i = 0; i < np; ++i) x[idx[i]] = y[i];
        if (!nonneg) return true;

        // ---- dual step: r_i = b_i − (G·x)_i on the zero set --------------
        int n_viol = 0;
        int worst = -1;
        Scalar worst_r = dual_tol;
        for (int i = 0; i < k; ++i) {
            if (state[i] != 0) continue;
            Scalar r = b[i];
            for (int t = 0; t < np; ++t) {
                r -= Gd[i + static_cast<std::ptrdiff_t>(idx[t]) * ld] * y[t];
            }
            if (r > dual_tol) {
                ++n_viol;
                state[i] = 3;                 // marked as violating
                if (r > worst_r) {
                    worst_r = r;
                    worst = i;
                }
            }
        }
        if (n_viol == 0) return true;

        const bool last_round = (round + 1 >= max_round);
        const bool add_all = (round < 3);
        for (int i = 0; i < k; ++i) {
            if (state[i] != 3) continue;
            if (!last_round && (add_all || i == worst)) {
                idx[np++] = i;
                state[i] = 1;
            } else {
                state[i] = 0;
            }
        }
        if (last_round) return false;
    }
}

/**
 * @brief Finish one column after the unconstrained solve
 *
 * @param b          Right-hand side that was solved (L1 already subtracted
 *                   when nonneg).  May be overwritten with a residual.
 * @param x          Unconstrained solution on entry (ignored if !solved_ok);
 *                   exact constrained solution on exit
 * @param L1_cd      L1 penalty still to be applied by soft-thresholding
 *                   (non-zero only for nonneg = false)
 * @param solved_ok  False if the factorization failed
 * @param ws         Scratch space.  With it, plain NNLS columns (x ≥ 0, no
 *                   bound, no remaining L1) are finished by the exact
 *                   active-set solve; other columns, and columns for which
 *                   the active set hits its round cap, are clipped to the
 *                   box and polished by coordinate descent.
 */
template<typename Scalar>
inline void cholesky_finish_col(const DenseMatrix<Scalar>& G,
                                Scalar* __restrict__ b,
                                Scalar* __restrict__ x,
                                int k,
                                Scalar L1_cd,
                                bool nonneg,
                                Scalar upper_bound,
                                int max_sweeps,
                                bool solved_ok,
                                ActiveSetWorkspace<Scalar>* ws = nullptr)
{
    const bool has_upper = (upper_bound > Scalar(0));
    const bool has_L1 = (L1_cd > Scalar(0));
    bool need_polish = has_L1;

    if (solved_ok) {
        for (int i = 0; i < k; ++i) {
            const Scalar v = x[i];
            if (!std::isfinite(v)) {
                solved_ok = false;
                break;
            }
            if ((nonneg && v < Scalar(0)) || (has_upper && v > upper_bound))
                need_polish = true;
        }
    }

    // Feasible unconstrained solution: already the exact answer.
    if (solved_ok && !need_polish) return;

    Scalar scale = 0;
    for (int i = 0; i < k; ++i) {
        const Scalar a = (b[i] >= Scalar(0)) ? b[i] : -b[i];
        if (a > scale) scale = a;
    }

    if (!solved_ok) {
        for (int i = 0; i < k; ++i) x[i] = Scalar(0);
    }

    // Plain NNLS: exact active-set solve (b is not modified).
    if (ws != nullptr && nonneg && !has_upper && !has_L1) {
        if (active_set_solve(G, b, x, k, *ws, kkt_rel_tol<Scalar>(k) * scale))
            return;
        // Round cap reached: x is a feasible iterate; polish it below.
    } else if (solved_ok) {
        // Project the unconstrained solution onto the box
        for (int i = 0; i < k; ++i) {
            if (nonneg && x[i] < Scalar(0)) x[i] = Scalar(0);
            if (has_upper && x[i] > upper_bound) x[i] = upper_bound;
        }
    }

    // Keep the right-hand side, then turn b into the residual r = b − G·x
    if (ws != nullptr) {
        Scalar* b0 = ws->b0.data();
        for (int i = 0; i < k; ++i) b0[i] = b[i];
    }
    for (int i = 0; i < k; ++i) {
        const Scalar xi = x[i];
        if (xi == Scalar(0)) continue;
        const Scalar* __restrict__ g_col =
            G.data() + static_cast<std::ptrdiff_t>(i) * k;
        for (int q = 0; q < k; ++q) {
            b[q] -= g_col[q] * xi;
        }
    }

    const int sweeps = (max_sweeps > 0) ? max_sweeps
                                        : static_cast<int>(CD_MAXIT);
    const int used =
        kkt_polish_col(G, b, x, k, L1_cd, nonneg, upper_bound, sweeps, scale);

    // Cap reached.  If the starting point was wild (a nearly singular G gives
    // a huge unconstrained solve), the iterate can still be worse than x = 0:
    // compare the objective f(x) = ½xᵀGx − b₀ᵀx + L1‖x‖₁ = −½·xᵀ(b₀ + r)
    // + L1‖x‖₁ with f(0) = 0 and restart once from zero in that case.  An
    // iterate that is merely converging slowly is kept.
    if (ws != nullptr && solved_ok && used >= sweeps) {
        const Scalar* b0 = ws->b0.data();
        Scalar f = 0;
        for (int i = 0; i < k; ++i) {
            const Scalar ax = (x[i] >= Scalar(0)) ? x[i] : -x[i];
            f += -static_cast<Scalar>(0.5) * x[i] * (b0[i] + b[i]) + L1_cd * ax;
        }
        if (!(f < Scalar(0))) {
            for (int i = 0; i < k; ++i) {
                x[i] = Scalar(0);
                b[i] = b0[i];
            }
            kkt_polish_col(G, b, x, k, L1_cd, nonneg, upper_bound, sweeps, scale);
        }
    }
}

/**
 * @brief Per-column Cholesky NNLS: solve G·x = b, then polish to KKT
 *
 * Used where the Gram matrix differs per column (CV and masked NMF), so the
 * factorization is per column.
 *
 * Note: b is treated as the ORIGINAL RHS (not residual form) and is modified
 * in place.
 *
 * @param G            Gram matrix (k × k), possibly corrected per column
 * @param b            RHS vector (k), modified in-place
 * @param x            Solution vector (k), overwritten
 * @param k            Dimension
 * @param L1           L1 penalty (absolute; subtracted from b when nonneg,
 *                     soft-thresholded in the polish otherwise)
 * @param L2           L2 penalty (added to the Gram diagonal if > 0)
 * @param nonneg       Enforce non-negativity
 * @param maxit        Sweep cap for the polish (<= 0 uses CD_MAXIT)
 * @param tol          Unused (the polish stops on a KKT criterion)
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
                              int maxit,
                              Scalar /*tol*/,
                              Scalar upper_bound = 0)
{
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> b_vec(b, k);
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_vec(x, k);

    // L1: exact as a shift of the RHS under non-negativity; needs
    // soft-thresholding (done in the polish) for a signed solution.
    Scalar L1_cd = 0;
    if (L1 > Scalar(0)) {
        if (nonneg) {
            b_vec.array() -= L1;
        } else {
            L1_cd = L1;
        }
    }

    if (L2 > Scalar(0)) {
        DenseMatrix<Scalar> G_reg = G;
        G_reg.diagonal().array() += L2;
        Eigen::LLT<DenseMatrix<Scalar>> llt(G_reg);
        const bool ok = (llt.info() == Eigen::Success);
        if (ok) {
            x_vec = b_vec;
            llt.solveInPlace(x_vec);
        }
        ActiveSetWorkspace<Scalar> ws(k);
        cholesky_finish_col(G_reg, b, x, k, L1_cd, nonneg, upper_bound,
                            maxit, ok, &ws);
    } else {
        Eigen::LLT<DenseMatrix<Scalar>> llt(G);
        const bool ok = (llt.info() == Eigen::Success);
        if (ok) {
            x_vec = b_vec;
            llt.solveInPlace(x_vec);
        }
        ActiveSetWorkspace<Scalar> ws(k);
        cholesky_finish_col(G, b, x, k, L1_cd, nonneg, upper_bound,
                            maxit, ok, &ws);
    }
}

}  // namespace detail

/**
 * @brief Batch Cholesky NNLS: solve G·X = B, then polish violating columns
 *
 * Solves, for each column, min_x ½ xᵀG x − bᵀx + L1‖x‖₁ subject to x ≥ 0
 * (when nonneg) and x ≤ upper_bound (when upper_bound > 0).  G is k×k
 * (Gram matrix, including any ridge), B is k×n (RHS), X is k×n (output).
 *
 * B is not modified.  The previous contents of X are not used: the direct
 * solve does not need a warm start.
 *
 * @param G           Gram matrix (k × k), symmetric positive (semi)definite
 * @param B           RHS matrix (k × n)
 * @param X           Solution matrix (k × n), overwritten
 * @param nonneg      Enforce non-negativity
 * @param threads     Number of OpenMP threads (0 = all available)
 * @param warm_start  Accepted for API compatibility; has no effect
 * @param L1          Absolute L1 penalty NOT already subtracted from B
 * @param upper_bound Upper bound (0 = none)
 * @param max_sweeps  Sweep cap for the polish
 */
template<typename Scalar>
inline void cholesky_clip_batch(
    const DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& X,
    bool nonneg,
    int threads,
    bool warm_start,
    Scalar L1 = 0,
    Scalar upper_bound = 0,
    int max_sweeps = static_cast<int>(CD_MAXIT))
{
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(B.cols());
    (void)warm_start;

    if (X.rows() != k || X.cols() != n) {
        X.resize(k, n);
    }

    // L1: shift of the RHS under non-negativity, soft-threshold otherwise
    const Scalar L1_shift = (L1 > Scalar(0) && nonneg) ? L1 : Scalar(0);
    const Scalar L1_cd    = (L1 > Scalar(0) && !nonneg) ? L1 : Scalar(0);

    // Cholesky factorization of G (done once)
    Eigen::LLT<DenseMatrix<Scalar>> llt(G);
    const bool llt_ok = (llt.info() == Eigen::Success);

    // Unconstrained solve for all columns at once: X = G^{-1} * (B − L1)
    X = B;
    if (L1_shift > Scalar(0)) X.array() -= L1_shift;
    if (llt_ok) llt.solveInPlace(X);

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
#else
    int n_threads = 1;
    (void)threads;
    (void)n_threads;
#endif

    #pragma omp parallel num_threads(n_threads) if(n_threads > 1)
    {
        // Thread-local RHS / residual buffer and active-set scratch space
        // (only touched for columns that violate a constraint)
        DenseVector<Scalar> r(k);
        detail::ActiveSetWorkspace<Scalar> as_ws(k);

        #pragma omp for schedule(dynamic, 64)
        for (int j = 0; j < n; ++j) {
            Scalar* xj = X.col(j).data();
            bool need = !llt_ok || (L1_cd > Scalar(0));
            if (!need) {
                const bool has_upper = (upper_bound > Scalar(0));
                for (int i = 0; i < k; ++i) {
                    const Scalar v = xj[i];
                    if (!std::isfinite(v)
                        || (nonneg && v < Scalar(0))
                        || (has_upper && v > upper_bound)) {
                        need = true;
                        break;
                    }
                }
            }
            if (!need) continue;

            r = B.col(j);
            if (L1_shift > Scalar(0)) r.array() -= L1_shift;
            detail::cholesky_finish_col(G, r.data(), xj, k, L1_cd, nonneg,
                                        upper_bound, max_sweeps, llt_ok,
                                        &as_ws);
        }
    }  // end parallel
}

// ========================================================================
// Double-precision Gram and solve for single-precision factors
// ========================================================================

/**
 * @brief Gram matrix G = F·Fᵀ accumulated in double precision
 *
 * The factors are stored in single precision, but the k × k normal equations
 * built from them lose about cond(G)·eps when they are formed and solved in
 * single precision.  On nearly exact low-rank data the rows of a factor are
 * close to collinear, cond(G) reaches 1e6 or more, and a single-precision
 * Gram is then accurate to only a digit or two in its small directions.
 * Forming G in double (and solving in double, see
 * cholesky_clip_batch_double) removes that limit at O(k²·ncols) cost, which
 * is small next to the O(nnz·k) work that stays in single precision.
 *
 * The columns are processed in fixed-size chunks whose partial Grams are
 * summed in chunk order, so the result does not depend on the number of
 * threads.
 *
 * @param F        Factor (k × ncols), any scalar type
 * @param G        Output Gram (k × k), double
 * @param threads  OpenMP threads (0 = all available)
 */
template<typename Scalar>
inline void gram_double(const DenseMatrix<Scalar>& F,
                        DenseMatrix<double>& G,
                        int threads)
{
    const int k = static_cast<int>(F.rows());
    const long long n = static_cast<long long>(F.cols());
    G.setZero(k, k);
    if (k == 0 || n == 0) return;

    // At most 256 chunks, each at least 2048 columns: bounded memory for the
    // partial Grams, and a chunking that depends on the size only.
    long long chunk = (n + 255) / 256;
    if (chunk < 2048) chunk = 2048;
    const int n_chunks = static_cast<int>((n + chunk - 1) / chunk);

#ifdef _OPENMP
    int n_threads = (threads > 0) ? threads : omp_get_max_threads();
#else
    int n_threads = 1;
    (void)threads;
    (void)n_threads;
#endif

    std::vector<DenseMatrix<double>> parts(static_cast<std::size_t>(n_chunks));

    #pragma omp parallel num_threads(n_threads) if(n_threads > 1 && n_chunks > 1)
    {
        DenseMatrix<double> buf;
        #pragma omp for schedule(static)
        for (int c = 0; c < n_chunks; ++c) {
            const long long start = static_cast<long long>(c) * chunk;
            const long long len = std::min(chunk, n - start);
            buf = F.middleCols(start, len).template cast<double>();
            DenseMatrix<double>& part = parts[static_cast<std::size_t>(c)];
            part.resize(k, k);
            part.noalias() = buf * buf.transpose();
        }
    }

    for (int c = 0; c < n_chunks; ++c)
        G += parts[static_cast<std::size_t>(c)];
}

/**
 * @brief Batch Cholesky NNLS with a double-precision Gram
 *
 * Same problem as cholesky_clip_batch, with the factorization, the
 * active-set passes and the KKT polish carried out in double: the
 * right-hand side is widened on the way in and the solution rounded to
 * Scalar on the way out.  B is not modified.
 */
template<typename Scalar>
inline void cholesky_clip_batch_double(
    const DenseMatrix<double>& G,
    const DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& X,
    bool nonneg,
    int threads,
    Scalar L1 = 0,
    Scalar upper_bound = 0,
    int max_sweeps = static_cast<int>(CD_MAXIT))
{
    DenseMatrix<double> Bd = B.template cast<double>();
    DenseMatrix<double> Xd;
    cholesky_clip_batch(G, Bd, Xd, nonneg, threads, false,
                        static_cast<double>(L1),
                        static_cast<double>(upper_bound),
                        max_sweeps);
    X = Xd.template cast<Scalar>();
}

}  // namespace primitives
}  // namespace FactorNet
