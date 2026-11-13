// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file lanczos.hpp
 * @brief Golub-Kahan-Lanczos bidiagonalization for truncated SVD/PCA
 *
 * The Lanczos bidiagonalization builds a sequence of bidiagonal matrices:
 *
 *   A · P_j = Q_j · B_j   (approximately)
 *
 * where P_j (n × j) and Q_j (m × j) have orthonormal columns, and
 * B_j (j × j) is upper bidiagonal.
 *
 * The SVD of B_j gives Ritz approximations to A's singular values.
 * With full reorthogonalization, convergence is monotonic.
 *
 * Cost per step: 2 SpMVs O(nnz) + reorthogonalization O((m+n)·j).
 * Total: O(nnz · j_max + (m+n) · j_max²) where j_max = k + extra.
 *
 * Breakdown (an invariant subspace) is handled by continuing from a random
 * vector orthogonal to the basis, so exactly low-rank input and repeated
 * singular values are resolved. All thresholds are relative to a running
 * estimate of ||A||.
 *
 * Supports: centering (PCA), scaling.
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/svd/spmv.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>  // for trace_AtA
#include <FactorNet/rng/rng.hpp>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <stdexcept>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(RCPP_VERSION) || __has_include(<Rcpp.h>)
#include <Rcpp.h>
#define LSVD_PRINT(...) { char buf[512]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); Rcpp::Rcout << buf; }
#define LSVD_CHECK_INTERRUPT() Rcpp::checkUserInterrupt()
#else
#include <cstdio>
#define LSVD_PRINT(...) std::printf(__VA_ARGS__)
#define LSVD_CHECK_INTERRUPT() ((void)0)
#endif

namespace FactorNet {
namespace svd {

namespace detail {

/**
 * @brief Fill x with a unit vector orthogonal to the first ncols columns of Basis.
 *
 * Used to continue after a Lanczos breakdown (an invariant subspace was found).
 * Returns false when no such direction exists (the basis already spans the space).
 */
template<typename Scalar>
inline bool lanczos_random_orth(const DenseMatrix<Scalar>& Basis, int ncols,
                                DenseVector<Scalar>& x, rng::SplitMix64& gen)
{
    const int dim = static_cast<int>(x.size());
    if (dim <= 0 || ncols >= dim) return false;
    DenseVector<Scalar> h;
    for (int attempt = 0; attempt < 3; ++attempt) {
        for (int i = 0; i < dim; ++i)
            x(i) = static_cast<Scalar>(gen.uniform<double>() - 0.5);
        Scalar nrm0 = x.norm();
        if (!(nrm0 > 0)) continue;
        x /= nrm0;
        if (ncols > 0) {
            for (int pass = 0; pass < 2; ++pass) {
                h.resize(ncols);
                h.noalias() = Basis.leftCols(ncols).transpose() * x;
                x.noalias() -= Basis.leftCols(ncols) * h;
            }
        }
        Scalar nrm = x.norm();
        if (nrm > static_cast<Scalar>(1e-4)) {
            x /= nrm;
            if (ncols > 0) {
                h.resize(ncols);
                h.noalias() = Basis.leftCols(ncols).transpose() * x;
                x.noalias() -= Basis.leftCols(ncols) * h;
                Scalar nrm2 = x.norm();
                if (nrm2 > 0) x /= nrm2;
            }
            return true;
        }
    }
    return false;
}

/**
 * @brief Golub-Kahan-Lanczos bidiagonalization with full reorthogonalization,
 *        written against abstract matrix-vector products.
 *
 * Relations maintained (upper bidiagonal B):
 *   A  p_j = beta_j q_{j-1} + alpha_j q_j
 *   A' q_j = alpha_j p_j   + beta_{j+1} p_{j+1}
 *
 * Breakdown (alpha_j or beta_{j+1} negligible relative to the running estimate
 * of ||A||) means an invariant subspace was found. The coupling is set to zero
 * and the recurrence continues from a random vector orthogonal to the basis,
 * so repeated singular values and exactly low-rank input are handled. When a
 * fresh random direction is itself annihilated the remaining spectrum is zero.
 *
 * The final Ritz extraction uses the j x (j+1) matrix [B_j | beta_j e_j] when
 * the last right vector exists, which equals Q_j' A exactly; this is what
 * makes k = min(m, n) and wide matrices come out right.
 *
 * @param forward_vec    callable (const DenseVector&, DenseVector&): out = A x
 * @param transpose_vec  callable (const DenseVector&, DenseVector&): out = A' x
 * @param extended_budget  larger default step budget (used by svd(); the
 *                         NMF initializer and the Krylov seed keep the short one)
 * @param n_unconverged_out  number of the k wanted Ritz values whose residual
 *                           bound exceeds tol * sigma_1 at exit
 */
template<typename Scalar, typename FwdOp, typename TrnOp>
inline void lanczos_bidiag_core(const int m, const int n, const int k_wanted,
                                const Scalar tol_in, const int user_steps,
                                const uint32_t seed, const bool verbose,
                                const bool extended_budget,
                                FwdOp&& forward_vec, TrnOp&& transpose_vec,
                                DenseMatrix<Scalar>& U_out,
                                DenseVector<Scalar>& d_out,
                                DenseMatrix<Scalar>& V_out,
                                int& steps_out, int& n_unconverged_out)
{
    const int mn = std::min(m, n);
    const int kk = std::max(0, std::min(k_wanted, mn));
    steps_out = 0;
    n_unconverged_out = 0;
    if (kk == 0) {
        U_out.resize(m, 0);
        V_out.resize(n, 0);
        d_out.resize(0);
        return;
    }

    const int legacy_steps = std::max(3 * kk, kk + 50);
    const int default_steps = extended_budget ? std::max(8 * kk, kk + 120) : legacy_steps;
    // A positive user value is the step limit (at least k); 0 means default.
    const int jmax = std::min(mn, user_steps > 0 ? std::max(user_steps, kk) : default_steps);

    const Scalar eps_mach = std::numeric_limits<Scalar>::epsilon();
    const Scalar bd_rel = std::pow(eps_mach, static_cast<Scalar>(2.0 / 3.0));
    const Scalar conv_tol = std::max(tol_in, static_cast<Scalar>(16) * eps_mach);

    // Storage grows on demand so easy problems do not pay for the full budget
    int cap = std::min(jmax, legacy_steps);
    DenseMatrix<Scalar> P(n, cap + 1);   // right Lanczos vectors
    DenseMatrix<Scalar> Q(m, cap);       // left Lanczos vectors
    DenseVector<Scalar> alpha = DenseVector<Scalar>::Zero(jmax);
    DenseVector<Scalar> beta = DenseVector<Scalar>::Zero(jmax + 1);
    DenseVector<Scalar> r(m), s(n), xm(m), xn(n), h;

    auto reorth = [&h](const DenseMatrix<Scalar>& Basis, int ncols, DenseVector<Scalar>& x) {
        if (ncols <= 0) return;
        for (int pass = 0; pass < 2; ++pass) {
            h.resize(ncols);
            h.noalias() = Basis.leftCols(ncols).transpose() * x;
            x.noalias() -= Basis.leftCols(ncols) * h;
        }
    };

    rng::SplitMix64 gen(seed == 0 ? 42ULL : static_cast<uint64_t>(seed));
    for (int i = 0; i < n; ++i)
        xn(i) = static_cast<Scalar>(gen.uniform<double>() - 0.5);
    {
        Scalar nrm = xn.norm();
        if (nrm > 0) { xn /= nrm; } else { xn.setZero(); xn(0) = static_cast<Scalar>(1); }
    }
    P.col(0) = xn;

    Scalar anorm = 0;          // running lower bound on ||A||_2
    int j_actual = 0;          // completed steps = rows of B
    bool have_extra = false;   // P.col(j_actual) exists and couples through beta(j_actual)
    bool p_random = false;     // p_j was a random restart vector
    bool exhausted = false;    // remaining spectrum is numerically zero
    bool converged = false;
    int n_restarts = 0;

    const int check_interval = 5;
    const int min_steps_before_check = kk + 2;

    // Number of the kk leading Ritz values of the square B_j (j x j) whose
    // residual bound beta_j |U_B(j-1, i)| exceeds conv_tol * sigma_1.
    // B B' is symmetric tridiagonal with diagonal alpha_i^2 + beta_{i+1}^2 and
    // off-diagonal alpha_{i+1} beta_{i+1}; its eigenvectors are U_B. The
    // tridiagonal QL solve in double is far cheaper than a Jacobi SVD of B,
    // which matters because this runs every few steps.
    auto count_unconverged = [&](int jc) -> int {
        if (jc <= 0) return kk;
        Eigen::VectorXd td(jc);
        Eigen::VectorXd te(jc > 1 ? jc - 1 : 0);
        for (int jj = 0; jj < jc; ++jj) {
            const double a = static_cast<double>(alpha(jj));
            const double b = (jj < jc - 1) ? static_cast<double>(beta(jj + 1)) : 0.0;
            td(jj) = a * a + b * b;
            if (jj < jc - 1) te(jj) = static_cast<double>(alpha(jj + 1)) * b;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.computeFromTridiagonal(td, te, Eigen::ComputeEigenvectors);
        if (es.info() != Eigen::Success) return kk;
        const double lam1 = es.eigenvalues()(jc - 1);     // ascending order
        const double s1 = lam1 > 0.0 ? std::sqrt(lam1) : 0.0;
        const double beta_last = std::abs(static_cast<double>(beta(jc)));
        const double thr = static_cast<double>(conv_tol) * s1;
        const int k_chk = std::min(kk, jc);
        int bad = kk - k_chk;
        for (int i = 0; i < k_chk; ++i) {
            const double err = beta_last * std::abs(es.eigenvectors()(jc - 1, jc - 1 - i));
            if (err > thr) ++bad;
        }
        return bad;
    };

    if (verbose) {
        LSVD_PRINT("  Lanczos bidiag: k=%d, max steps=%d, tol=%.1e\n",
                   kk, jmax, static_cast<double>(conv_tol));
    }

    for (int j = 0; j < jmax; ++j) {
        LSVD_CHECK_INTERRUPT();

        if (j >= cap) {
            cap = std::min(jmax, 2 * cap);
            P.conservativeResize(Eigen::NoChange, cap + 1);
            Q.conservativeResize(Eigen::NoChange, cap);
        }

        // --- r = A p_j - beta_j q_{j-1} ---
        xn = P.col(j);
        forward_vec(xn, r);
        {
            const Scalar ap = r.norm();
            if (!std::isfinite(ap))
                throw std::runtime_error("svd: non-finite values in A * v (does the input contain NA/NaN/Inf?)");
            if (ap > anorm) anorm = ap;
        }
        if (j > 0 && beta(j) != 0) r.noalias() -= beta(j) * Q.col(j - 1);
        reorth(Q, j, r);

        const Scalar a_j = r.norm();
        bool q_random = false;
        if (!(a_j > bd_rel * anorm)) {
            // Breakdown: A p_j lies in span(Q). Continue from a random direction.
            alpha(j) = 0;
            if (p_random) exhausted = true;   // a random p was annihilated
            if (!lanczos_random_orth(Q, j, xm, gen)) break;   // left space exhausted
            Q.col(j) = xm;
            q_random = true;
            ++n_restarts;
        } else {
            alpha(j) = a_j;
            Q.col(j) = r / a_j;
        }

        // --- s = A' q_j - alpha_j p_j ---
        xm = Q.col(j);
        transpose_vec(xm, s);
        {
            const Scalar atq = s.norm();
            if (!std::isfinite(atq))
                throw std::runtime_error("svd: non-finite values in A' * u (does the input contain NA/NaN/Inf?)");
            if (atq > anorm) anorm = atq;
        }
        if (alpha(j) != 0) s.noalias() -= alpha(j) * P.col(j);
        reorth(P, j + 1, s);

        const Scalar b_next = s.norm();
        j_actual = j + 1;

        if (j + 1 >= n) {
            // Right space is complete: B is exact, nothing left to couple to
            beta(j + 1) = 0;
            have_extra = false;
            exhausted = true;
            break;
        } else if (!(b_next > bd_rel * anorm)) {
            beta(j + 1) = 0;
            have_extra = false;
            if (q_random) exhausted = true;   // a random q was annihilated
            if (!lanczos_random_orth(P, j + 1, xn, gen)) { exhausted = true; break; }
            P.col(j + 1) = xn;
            p_random = true;
            ++n_restarts;
        } else {
            beta(j + 1) = b_next;
            P.col(j + 1) = s / b_next;
            p_random = false;
            have_extra = true;
        }

        // Remaining spectrum is zero: stop once k triplets exist
        if (exhausted && j_actual >= kk) { converged = true; break; }

        // Early stop on Ritz residual bounds. After a restart the bounds say
        // nothing about directions not yet explored, so only stop early when
        // no restart happened (or the space is exhausted, handled above).
        if (n_restarts == 0 &&
            j_actual >= min_steps_before_check &&
            j_actual % check_interval == 0 &&
            j_actual < jmax) {
            const bool all_converged = (count_unconverged(j_actual) == 0);
            if (all_converged) {
                converged = true;
                if (verbose) {
                    LSVD_PRINT("    Early stop at step %d: all %d Ritz values converged\n",
                               j_actual, kk);
                }
                break;
            }
        }
    }

    steps_out = j_actual;
    const int jb = j_actual;
    if (jb == 0) {
        U_out = DenseMatrix<Scalar>::Zero(m, kk);
        V_out = DenseMatrix<Scalar>::Zero(n, kk);
        d_out = DenseVector<Scalar>::Zero(kk);
        return;
    }

    // The decomposition is exact when either space has been spanned
    const bool complete = exhausted || jb >= m || jb >= n;

    // Count unconverged wanted values from the square-B residual bounds
    if (!converged && !complete) {
        n_unconverged_out = count_unconverged(jb);
    }

    // Ritz extraction from [B | beta_jb e_jb] (jb x nc)
    const int nc = jb + (have_extra ? 1 : 0);
    DenseMatrix<Scalar> B = DenseMatrix<Scalar>::Zero(jb, nc);
    for (int jj = 0; jj < jb; ++jj) {
        B(jj, jj) = alpha(jj);
        if (jj + 1 < nc) B(jj, jj + 1) = beta(jj + 1);
    }
    Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_b(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const int k_ret = std::min(kk, jb);
    DenseMatrix<Scalar> U_B = svd_b.matrixU().leftCols(k_ret);
    DenseMatrix<Scalar> V_B = svd_b.matrixV().leftCols(k_ret);
    d_out = svd_b.singularValues().head(k_ret);
    U_out.noalias() = Q.leftCols(jb) * U_B;
    V_out.noalias() = P.leftCols(nc) * V_B;

    if (verbose) {
        LSVD_PRINT("    Completed %d Lanczos steps (%d restarts), %d unconverged\n",
                   jb, n_restarts, n_unconverged_out);
    }
}

}  // namespace detail

/**
 * @brief Truncated SVD by Lanczos bidiagonalization.
 *
 * @param strict  true for user-facing svd(): larger default step budget and a
 *                deferred warning when wanted values have not reached tol.
 *                false (default) keeps the short budget and stays silent; the
 *                NMF initializer and the Krylov seed use that.
 */
template<typename MatrixType, typename Scalar>
SVDResult<Scalar> lanczos_svd(const MatrixType& A,
                              const SVDConfig<Scalar>& config,
                              bool strict = false)
{
    using FactorNet::primitives::CPU;
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.k_max;

    SVDResult<Scalar> result;
    result.centered = config.center;

    int eff_threads = config.threads;
#ifdef _OPENMP
    if (eff_threads <= 0) eff_threads = omp_get_max_threads();
#else
    eff_threads = 1;
#endif

    // --- Row means, sample SDs and ||X||^2_F in one parallel pass (double) ---
    DenseVector<Scalar> mu, row_sds_vec, row_inv_sds;
    const DenseVector<Scalar>* mu_ptr = nullptr;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    result.frobenius_norm_sq = row_stats_one_pass<MatrixType, Scalar>(
        A, config.center, config.scale, eff_threads, mu, row_sds_vec);
    if (config.center) {
        result.row_means = mu;
        mu_ptr = &mu;
    }
    if (config.scale) {
        row_inv_sds = row_sds_vec.cwiseInverse();
        result.row_sds = row_sds_vec;
        result.scaled = true;
        sd_ptr = &row_inv_sds;
    }

    // Pre-allocate workspace for forward SpMV (avoids malloc per call)
    SpmvWorkspace<Scalar> ws;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        ws.init(m, eff_threads);
    }

    auto forward_vec = [&](const DenseVector<Scalar>& v, DenseVector<Scalar>& out) {
        spmv_forward(A, v, out, mu_ptr, eff_threads, &ws, sd_ptr);
    };

    auto transpose_vec = [&](const DenseVector<Scalar>& u, DenseVector<Scalar>& out) {
        spmv_transpose(A, u, out, mu_ptr, eff_threads, sd_ptr);
    };

    int steps = 0, n_unconv = 0;
    detail::lanczos_bidiag_core<Scalar>(
        m, n, k, config.tol, config.max_iter > 0 ? config.max_iter : 0,
        config.seed, config.verbose, strict,
        forward_vec, transpose_vec,
        result.U, result.d, result.V, steps, n_unconv);

    result.k_selected = static_cast<int>(result.d.size());
    result.iters_per_factor.push_back(steps);

    if (strict && n_unconv > 0) {
        char wbuf[320];
        std::snprintf(wbuf, sizeof(wbuf),
            "svd(method = \"lanczos\"): %d of %d singular values did not reach "
            "tol = %.1e after %d Lanczos steps; increase 'maxit' (number of steps) "
            "or use method = \"irlba\"",
            n_unconv, result.k_selected, static_cast<double>(config.tol), steps);
        add_warning(std::string(wbuf));
    }

    auto t_end = Clock::now();
    result.wall_time_ms = Ms(t_end - t_start).count();

    if (config.verbose) {
        LSVD_PRINT("  Lanczos SVD done: %d steps, %.1f ms\n", steps, result.wall_time_ms);
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet
