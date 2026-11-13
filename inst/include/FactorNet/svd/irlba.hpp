// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file irlba.hpp
 * @brief Augmented Implicitly Restarted Lanczos Bidiagonalization Algorithm (IRLBA)
 *
 * Implements the algorithm of Baglama & Reichel (2005):
 *   "Augmented implicitly restarted Lanczos bidiagonalization methods"
 *   SIAM J. Sci. Comput. 27(1):19-42
 *
 * Also draws on the reference C implementation by B.W. Lewis (irlba R package).
 *
 * Key idea: instead of one long Lanczos pass, use short passes of `work` steps
 * (typically k+7) with implicit restarts that compress the Krylov subspace to
 * keep only the k best Ritz directions plus the residual vector. This converges
 * faster than basic Lanczos and requires O((m+n) * work) memory instead of
 * O((m+n) * 2k).
 *
 * Uses OpenMP-parallel SpMV from spmv.hpp for multi-threaded matrix-vector
 * products. Supports PCA centering.
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/svd/spmv.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>
#include <FactorNet/rng/rng.hpp>
#include <FactorNet/svd/lanczos.hpp>   // detail::lanczos_random_orth

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
#define IRLBA_PRINT(...) { char buf[512]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); Rcpp::Rcout << buf; }
#define IRLBA_CHECK_INTERRUPT() Rcpp::checkUserInterrupt()
#else
#include <cstdio>
#define IRLBA_PRINT(...) std::printf(__VA_ARGS__)
#define IRLBA_CHECK_INTERRUPT() ((void)0)
#endif

namespace FactorNet {
namespace svd {

namespace detail {

/**
 * @brief Classical Gram-Schmidt orthogonalization (double pass)
 *
 * Two passes of Y = Y - X * (X' * Y) for numerical stability.
 * Matches the approach in our basic Lanczos implementation.
 *
 * Templated on VecType to accept both DenseVector and column expressions.
 */
template<typename Scalar, typename VecType>
inline void orthog(const DenseMatrix<Scalar>& X, Eigen::MatrixBase<VecType>& Y,
                   int ncols) {
    if (ncols <= 0) return;
    auto Xk = X.leftCols(ncols);
    // First pass
    DenseVector<Scalar> h = Xk.transpose() * Y.derived();
    Y.derived().noalias() -= Xk * h;
    // Second pass for numerical stability
    h.noalias() = Xk.transpose() * Y.derived();
    Y.derived().noalias() -= Xk * h;
}

#ifndef FACTORNET_SVD_GPU_DEF_conv_test
#define FACTORNET_SVD_GPU_DEF_conv_test
/**
 * @brief Convergence test (matches irlba's convtests)
 *
 * A singular value is converged when:
 *   1. |residual[j]| < tol * Smax   (residual small relative to largest SV)
 *   2. svratio[j] < svtol            (relative SV change is small)
 *
 * Returns true if n_wanted values have converged.
 * Also adaptively adjusts k_restart upward to include nearly-converged values.
 */
template<typename Scalar>
inline bool conv_test(int work, int n_wanted, Scalar tol, Scalar svtol,
                      Scalar Smax, const DenseVector<Scalar>& svratio,
                      const DenseVector<Scalar>& residuals,
                      int& k_restart, Scalar S_norm) {
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    int n_converged = 0;
    for (int j = 0; j < work; ++j) {
        if (std::abs(residuals(j)) < tol * Smax && svratio(j) < svtol)
            n_converged++;
    }

    if (n_converged >= n_wanted || S_norm < eps) {
        return true;
    }

    // Adaptive k: grow restart dimension to include nearly-converged SVs
    if (k_restart < n_wanted + n_converged)
        k_restart = n_wanted + n_converged;
    if (k_restart > work - 3)
        k_restart = work - 3;
    if (k_restart < 1)
        k_restart = 1;

    return false;
}
#endif // FACTORNET_SVD_GPU_DEF_conv_test

}  // namespace detail

// ============================================================================
// IRLBA core, written against abstract matrix-vector products
// ============================================================================

namespace detail {

/**
 * @brief Augmented implicitly restarted Lanczos bidiagonalization.
 *
 * Conventions (as in Baglama & Reichel and the irlba C code): B is UPPER
 * bidiagonal with
 *   A  V = W B,          A' W = V B' + F e_work'
 * so B(j, j) = S_j and B(j, j+1) = R_j. With B = Ub S Vb', the left Ritz
 * vectors are W Ub, the right Ritz vectors are V Vb, and the residual of
 * triplet i is R_F * Ub(work-1, i).
 *
 * After a restart that keeps k Ritz triplets and appends the residual
 * direction f as V[:, k]:
 *   A v_i = s_i w_i,   A' w_i = s_i v_i + res_i f      (i < k)
 * hence W' A f has entries res_i, i.e. B(i, i) = s_i and B(i, k) = res_i.
 * The same upper convention is used before and after restarts, so B always
 * equals W' A V.
 *
 * Breakdown thresholds are relative to a running estimate of ||A||.
 */
template<typename Scalar, typename FwdOp, typename TrnOp>
inline void irlba_core(const int m, const int n, const int nu_in,
                       const Scalar tol_in, const int maxit_in,
                       const uint32_t seed, const bool verbose,
                       FwdOp&& forward_vec, TrnOp&& transpose_vec,
                       DenseMatrix<Scalar>& U_out,
                       DenseVector<Scalar>& d_out,
                       DenseMatrix<Scalar>& V_out,
                       int& iters_out, int& mprod_out, int& n_unconverged_out)
{
    const int mn_min = std::min(m, n);
    const int nu = std::max(0, std::min(nu_in, mn_min));
    iters_out = 0;
    mprod_out = 0;
    n_unconverged_out = 0;
    if (nu == 0) {
        U_out.resize(m, 0);
        V_out.resize(n, 0);
        d_out.resize(0);
        return;
    }

    const int work = std::min(nu + 15, mn_min);
    const int maxit = maxit_in > 0 ? maxit_in : 1000;

    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const Scalar tol = std::max(tol_in > 0 ? tol_in : static_cast<Scalar>(1e-5),
                                static_cast<Scalar>(16) * eps);
    const Scalar svtol = tol;
    const Scalar eps23 = std::pow(eps, static_cast<Scalar>(2.0 / 3.0));

    DenseMatrix<Scalar> V(n, work);
    DenseMatrix<Scalar> W(m, work);
    DenseMatrix<Scalar> V1(n, work);
    DenseMatrix<Scalar> W1(m, work);
    DenseVector<Scalar> F(n), xm(m), xn(n);
    DenseMatrix<Scalar> B = DenseMatrix<Scalar>::Zero(work, work);

    DenseVector<Scalar> sv_prev = DenseVector<Scalar>::Zero(work);
    DenseVector<Scalar> svratio(work);
    DenseVector<Scalar> residuals(work);
    DenseVector<Scalar> signed_res(work);

    rng::SplitMix64 gen(seed == 0 ? 42ULL : static_cast<uint64_t>(seed));
    for (int i = 0; i < n; ++i)
        xn(i) = static_cast<Scalar>(gen.uniform<double>() - 0.5);
    {
        Scalar nrm = xn.norm();
        if (nrm > 0) { xn /= nrm; } else { xn.setZero(); xn(0) = static_cast<Scalar>(1); }
    }
    V.col(0) = xn;

    int k = nu;
    int iter = 0;
    int mprod = 0;
    Scalar Smax = 0;
    Scalar anorm = 0;
    bool converged = false;
    int n_conv_top = 0;

    DenseVector<Scalar> BS;
    DenseMatrix<Scalar> BU, BV;

    auto check_finite = [](Scalar x) {
        if (!std::isfinite(x))
            throw std::runtime_error("svd: non-finite values in a matrix-vector product (does the input contain NA/NaN/Inf?)");
    };

    if (verbose) {
        IRLBA_PRINT("  IRLBA: k=%d, work=%d, tol=%.1e, maxit=%d\n",
                     nu, work, static_cast<double>(tol), maxit);
    }

    while (iter < maxit) {
        IRLBA_CHECK_INTERRUPT();

        int j = (iter == 0) ? 0 : k;

        // W[:, j] = A V[:, j], orthogonalized against the kept left vectors
        xn = V.col(j);
        forward_vec(xn, xm);
        mprod++;
        {
            const Scalar nr = xm.norm();
            check_finite(nr);
            if (nr > anorm) anorm = nr;
        }
        if (j > 0) orthog(W, xm, j);

        Scalar S = xm.norm();
        if (!(S > eps23 * anorm)) {
            if (!lanczos_random_orth(W, j, xm, gen)) xm.setZero();
            S = 0;
        } else {
            xm /= S;
        }
        W.col(j) = xm;

        while (j < work) {
            // F = A' W[:, j] - S V[:, j]
            xm = W.col(j);
            transpose_vec(xm, F);
            mprod++;
            {
                const Scalar nr = F.norm();
                check_finite(nr);
                if (nr > anorm) anorm = nr;
            }
            if (S != 0) F.noalias() -= S * V.col(j);
            orthog(V, F, j + 1);

            if (j + 1 < work) {
                Scalar R_F = F.norm();
                if (!(R_F > eps23 * anorm)) {
                    if (!lanczos_random_orth(V, j + 1, xn, gen)) xn.setZero();
                    R_F = 0;
                } else {
                    xn = F / R_F;
                }
                V.col(j + 1) = xn;

                B(j, j) = S;
                B(j, j + 1) = R_F;      // upper bidiagonal

                // W[:, j+1] = A V[:, j+1] - R_F W[:, j]
                forward_vec(xn, xm);
                mprod++;
                {
                    const Scalar nr = xm.norm();
                    check_finite(nr);
                    if (nr > anorm) anorm = nr;
                }
                if (R_F != 0) xm.noalias() -= R_F * W.col(j);
                orthog(W, xm, j + 1);

                S = xm.norm();
                if (!(S > eps23 * anorm)) {
                    if (!lanczos_random_orth(W, j + 1, xm, gen)) xm.setZero();
                    S = 0;
                } else {
                    xm /= S;
                }
                W.col(j + 1) = xm;
            } else {
                B(j, j) = S;
            }
            j++;
        }

        // SVD of the projected matrix B = W' A V (upper form)
        Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_b(
            B, Eigen::ComputeFullU | Eigen::ComputeFullV);
        BS = svd_b.singularValues();
        BU = svd_b.matrixU();     // left rotation:  U ~ W * BU
        BV = svd_b.matrixV();     // right rotation: V ~ V * BV

        Scalar R_F = F.norm();
        const bool invariant = !(R_F > eps23 * anorm);
        if (invariant) { R_F = 0; } else { F /= R_F; }

        if (work >= m && !invariant) {
            // Left space is complete (wide matrix, k near m): the factorization
            // A = W [B | R_F e_work] [V f]' is exact, so finish from it.
            DenseMatrix<Scalar> Bext = DenseMatrix<Scalar>::Zero(work, work + 1);
            Bext.leftCols(work) = B;
            Bext(work - 1, work) = R_F;
            Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_e(
                Bext, Eigen::ComputeThinU | Eigen::ComputeThinV);
            DenseMatrix<Scalar> Vext(n, work + 1);
            Vext.leftCols(work) = V;
            Vext.col(work) = F;
            DenseMatrix<Scalar> Ue = svd_e.matrixU().leftCols(nu);
            DenseMatrix<Scalar> Ve = svd_e.matrixV().leftCols(nu);
            U_out.noalias() = W * Ue;
            V_out.noalias() = Vext * Ve;
            d_out = svd_e.singularValues().head(nu);
            iters_out = iter + 1;
            mprod_out = mprod;
            n_unconverged_out = 0;
            return;
        }

        if (BS(0) > Smax) Smax = BS(0);

        for (int i = 0; i < work; ++i) {
            signed_res(i) = R_F * BU(work - 1, i);
            residuals(i) = std::abs(signed_res(i));
        }
        for (int i = 0; i < work; ++i) {
            if (sv_prev(i) > 0 && BS(i) > 0) {
                svratio(i) = std::abs(sv_prev(i) - BS(i)) / BS(i);
            } else {
                svratio(i) = static_cast<Scalar>(1);
            }
        }

        int n_conv = 0;
        n_conv_top = 0;
        for (int i = 0; i < work; ++i) {
            if (residuals(i) < tol * Smax && svratio(i) < svtol) {
                n_conv++;
                if (i < nu) n_conv_top++;
            }
        }

        if (verbose) {
            IRLBA_PRINT("    iter %d: sigma_1=%.6e, |res_1|=%.2e, converged=%d/%d, mprod=%d\n",
                         iter + 1, static_cast<double>(BS(0)),
                         static_cast<double>(residuals(0)), n_conv_top, nu, mprod);
        }

        // Done when the wanted triplets converged, the matrix is zero, or the
        // whole working subspace is invariant (all residuals vanish).
        if (n_conv_top >= nu || !(Smax > 0) || invariant) {
            converged = true;
            n_conv_top = nu;
            iter++;
            break;
        }
        if (iter >= maxit - 1) {
            iter++;
            break;
        }

        // Restart dimension (irlba): wanted + converged, leaving room to expand
        k = nu + n_conv;
        if (k > work - 3) k = work - 3;
        if (k < 1) k = 1;
        if (k > work - 1) k = work - 1;

        sv_prev = BS;

        // Thick restart: keep k Ritz triplets, append the residual direction
        V1.leftCols(k).noalias() = V * BV.leftCols(k);
        V.leftCols(k) = V1.leftCols(k);
        V.col(k) = F;
        W1.leftCols(k).noalias() = W * BU.leftCols(k);
        W.leftCols(k) = W1.leftCols(k);

        B.setZero();
        for (int i = 0; i < k; ++i) {
            B(i, i) = BS(i);
            B(i, k) = signed_res(i);    // W' A f, same upper convention
        }

        iter++;
    }

    DenseMatrix<Scalar> BU_k = BU.leftCols(nu);
    DenseMatrix<Scalar> BV_k = BV.leftCols(nu);
    U_out.noalias() = W * BU_k;
    V_out.noalias() = V * BV_k;
    d_out = BS.head(nu);

    iters_out = iter;
    mprod_out = mprod;
    n_unconverged_out = converged ? 0 : (nu - n_conv_top);
}

}  // namespace detail

// ============================================================================
// Main IRLBA implementation
// ============================================================================

/**
 * @param warn_unconverged  record a deferred warning (see spmv.hpp) when the
 *                          restart limit is reached before all wanted values
 *                          meet tol. Off by default (NMF initialization).
 */
template<typename MatrixType, typename Scalar>
SVDResult<Scalar> irlba_svd(const MatrixType& A,
                            const SVDConfig<Scalar>& config,
                            bool warn_unconverged = false)
{
    using FactorNet::primitives::CPU;
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

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

    int iter = 0, mprod = 0, n_unconv = 0;
    detail::irlba_core<Scalar>(
        m, n, config.k_max, config.tol, config.max_iter,
        config.seed, config.verbose,
        forward_vec, transpose_vec,
        result.U, result.d, result.V, iter, mprod, n_unconv);

    result.k_selected = static_cast<int>(result.d.size());
    result.iters_per_factor.push_back(iter);

    if (warn_unconverged && n_unconv > 0) {
        char wbuf[320];
        std::snprintf(wbuf, sizeof(wbuf),
            "svd(method = \"irlba\"): %d of %d singular values did not reach "
            "tol = %.1e after %d restarts; increase 'maxit' (number of restarts) "
            "or loosen 'tol'",
            n_unconv, result.k_selected, static_cast<double>(config.tol), iter);
        add_warning(std::string(wbuf));
    }

    auto t_end = Clock::now();
    result.wall_time_ms = Ms(t_end - t_start).count();

    if (config.verbose) {
        IRLBA_PRINT("  IRLBA done: %d restarts, %d matvecs, %.1f ms, %s\n",
                     iter, mprod, result.wall_time_ms,
                     n_unconv == 0 ? "converged" : "NOT converged");
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet
