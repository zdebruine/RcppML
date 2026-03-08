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

#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>

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
// Main IRLBA implementation
// ============================================================================

template<typename MatrixType, typename Scalar>
SVDResult<Scalar> irlba_svd(const MatrixType& A,
                            const SVDConfig<Scalar>& config)
{
    using FactorNet::primitives::CPU;
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int nu = config.k_max;  // number of wanted singular triplets
    const int mn_min = std::min(m, n);

    // Working subspace dimension: k + 15, capped at min(m, n)
    // Larger work dimension gives more Ritz vectors per restart,
    // reducing restart count and total matrix-vector products.
    const int work = std::min(nu + 15, mn_min);

    // Max outer restart iterations
    const int maxit = config.max_iter > 0 ? config.max_iter : 1000;

    const Scalar tol = config.tol > 0 ? config.tol : static_cast<Scalar>(1e-5);
    const Scalar svtol = tol;  // relative SV change tolerance (same as tol)
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const Scalar eps23 = std::pow(eps, static_cast<Scalar>(2.0 / 3.0));

    SVDResult<Scalar> result;
    result.centered = config.center;

    int eff_threads = config.threads;
#ifdef _OPENMP
    if (eff_threads <= 0) eff_threads = omp_get_max_threads();
#else
    eff_threads = 1;
#endif

    // --- Row means for PCA centering ---
    DenseVector<Scalar> mu;
    const DenseVector<Scalar>* mu_ptr = nullptr;
    if (config.center) {
        mu = compute_row_means<MatrixType, Scalar>(A, eff_threads);
        result.row_means = mu;
        mu_ptr = &mu;
    }

    // --- Row SDs for correlation PCA scaling ---
    DenseVector<Scalar> row_inv_sds;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    if (config.scale) {
        DenseVector<Scalar> row_sds = compute_row_sds<MatrixType, Scalar>(A, mu, eff_threads);
        row_inv_sds = row_sds.cwiseInverse();
        result.row_sds = row_sds;
        result.scaled = true;
        sd_ptr = &row_inv_sds;
    }

    // --- ||A||²_F ---
    Scalar A_norm_sq;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        SparseMatrix<Scalar> A_cast = A.template cast<Scalar>();
        A_cast.makeCompressed();
        A_norm_sq = primitives::trace_AtA<CPU, Scalar, SparseMatrix<Scalar>>(A_cast);
    } else {
        A_norm_sq = primitives::trace_AtA<CPU, Scalar, MatrixType>(A);
    }
    if (config.center) {
        A_norm_sq -= static_cast<Scalar>(n) * mu.squaredNorm();
    }
    if (config.scale) {
        A_norm_sq = static_cast<Scalar>(m) * static_cast<Scalar>(n);
    }
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // --- SpMV helpers using OpenMP-parallel spmv.hpp routines ---
    // Pre-allocate workspace for forward SpMV (avoids malloc per call)
    SpmvWorkspace<Scalar> ws;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        ws.init(m, eff_threads);
    }
    // We use temporary buffers because spmv takes DenseVector& (not column exprs)
    DenseVector<Scalar> spmv_out_m(m), spmv_in_vec;

    // ========================================================================
    // Allocate working storage
    // ========================================================================
    DenseMatrix<Scalar> V(n, work);           // Right Lanczos vectors (n × work)
    DenseMatrix<Scalar> W(m, work);           // Left Lanczos vectors  (m × work)
    DenseVector<Scalar> F(n);                 // Residual vector (length n)
    DenseMatrix<Scalar> B = DenseMatrix<Scalar>::Zero(work, work);  // Bidiagonal matrix

    // Restart compression workspaces
    DenseMatrix<Scalar> V1(n, work);
    DenseMatrix<Scalar> W1(m, work);

    // Convergence tracking
    DenseVector<Scalar> svratio_prev = DenseVector<Scalar>::Zero(work);  // Previous BS values
    DenseVector<Scalar> svratio_comp(work);                               // Computed ratios
    DenseVector<Scalar> residuals(work);         // Unsigned residuals for convergence test
    DenseVector<Scalar> signed_res(work);        // Signed residuals for restart B matrix

    // ========================================================================
    // Initialize V[:, 0] = random unit vector
    // ========================================================================
    {
        DenseMatrix<Scalar> v0_mat(n, 1);
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        rng.fill_uniform(v0_mat.data(), v0_mat.rows(), v0_mat.cols());
        V.col(0) = v0_mat.col(0);
        V.col(0).array() -= static_cast<Scalar>(0.5);
        V.col(0).normalize();
    }

    int k = nu;       // Adaptive restart dimension (starts at nu)
    int iter = 0;     // Outer restart iteration counter
    int mprod = 0;    // Total matrix-vector products
    Scalar Smax = 0;  // Largest singular value seen
    bool converged = false;

    // SVD of B results (persist across iterations for final extraction)
    DenseVector<Scalar> BS;     // Singular values of B
    DenseMatrix<Scalar> BU;     // Left singular vectors of B
    DenseMatrix<Scalar> BV_mat; // Right singular vectors of B

    if (config.verbose) {
        IRLBA_PRINT("  IRLBA: k=%d, work=%d, tol=%.1e, maxit=%d\n",
                     nu, work, static_cast<double>(tol), maxit);
    }

    // ========================================================================
    // Main IRLBA loop
    // ========================================================================
    while (iter < maxit) {
        IRLBA_CHECK_INTERRUPT();

        // Starting column index for Lanczos expansion
        int j_start = (iter == 0) ? 0 : k;
        int j = j_start;

        // ==========================================================
        // Step 1: First forward multiply  W[:, j] = A * V[:, j]
        // ==========================================================
        spmv_in_vec = V.col(j);  // copy col to DenseVector for spmv
        spmv_forward(A, spmv_in_vec, spmv_out_m, mu_ptr, eff_threads, &ws, sd_ptr);
        W.col(j) = spmv_out_m;
        mprod++;

        // Reorthogonalize W[:, j] against previous W columns
        if (j > 0) {
            auto w_col = W.col(j);
            detail::orthog(W, w_col, j);
        }

        Scalar S = W.col(j).norm();

        // Invariant subspace check
        if (S < eps23 && j == 0) {
            // Starting vector is in null space — fatal at first iteration
            if (config.verbose) {
                IRLBA_PRINT("    Starting vector near null space (S=%.2e)\n",
                             static_cast<double>(S));
            }
            break;
        }
        if (S < eps23) {
            // Replace with random vector, orthogonalize
            DenseMatrix<Scalar> r_mat(m, 1);
            rng::SplitMix64 rng2(config.seed + iter * 1000 + 1);
            rng2.fill_uniform(r_mat.data(), r_mat.rows(), r_mat.cols());
            W.col(j) = r_mat.col(0);
            W.col(j).array() -= static_cast<Scalar>(0.5);
            auto w_col = W.col(j);
            detail::orthog(W, w_col, j);
            W.col(j).normalize();
            S = 0;
        } else {
            W.col(j) /= S;
        }

        // ==========================================================
        // Step 2: Inner Lanczos bidiagonalization loop  (j → work)
        // ==========================================================
        while (j < work) {
            // --- Transpose multiply: F = A' * W[:, j] ---
            spmv_in_vec = W.col(j);
            spmv_transpose(A, spmv_in_vec, F, mu_ptr, eff_threads, sd_ptr);
            mprod++;

            // Three-term recurrence
            F -= S * V.col(j);

            // Reorthogonalize F against V[:, 0:j]
            detail::orthog(V, F, j + 1);

            if (j + 1 < work) {
                Scalar R_F = F.norm();

                // Invariant subspace check
                if (R_F < eps23) {
                    DenseMatrix<Scalar> f_mat(n, 1);
                    rng::SplitMix64 rng3(config.seed + iter * 1000 + j + 2);
                    rng3.fill_uniform(f_mat.data(), f_mat.rows(), f_mat.cols());
                    F = f_mat.col(0);
                    F.array() -= static_cast<Scalar>(0.5);
                    detail::orthog(V, F, j + 1);
                    R_F = F.norm();
                    V.col(j + 1) = F / R_F;
                    R_F = 0;  // Signal invariant subspace
                } else {
                    V.col(j + 1) = F / R_F;
                }

                // Fill bidiagonal entries
                B(j, j) = S;
                B(j + 1, j) = R_F;  // Lower bidiagonal (irlba convention)

                // --- Forward multiply: W[:, j+1] = A * V[:, j+1] ---
                spmv_in_vec = V.col(j + 1);
                spmv_forward(A, spmv_in_vec, spmv_out_m, mu_ptr, eff_threads, &ws, sd_ptr);
                W.col(j + 1) = spmv_out_m;
                mprod++;

                // Classical Gram-Schmidt step (three-term recurrence)
                W.col(j + 1) -= R_F * W.col(j);

                // Full reorthogonalization against W[:, 0:j]
                {
                    auto w_col = W.col(j + 1);
                    detail::orthog(W, w_col, j + 1);
                }

                S = W.col(j + 1).norm();
                if (S < eps23) {
                    DenseMatrix<Scalar> w_mat(m, 1);
                    rng::SplitMix64 rng4(config.seed + iter * 1000 + j + 3);
                    rng4.fill_uniform(w_mat.data(), w_mat.rows(), w_mat.cols());
                    W.col(j + 1) = w_mat.col(0);
                    W.col(j + 1).array() -= static_cast<Scalar>(0.5);
                    auto w_col2 = W.col(j + 1);
                    detail::orthog(W, w_col2, j + 1);
                    W.col(j + 1).normalize();
                    S = 0;
                } else {
                    W.col(j + 1) /= S;
                }
            } else {
                // Last step: only set diagonal, no subdiagonal
                B(j, j) = S;
            }
            j++;
        }

        // ==========================================================
        // Step 3: SVD of bidiagonal B (work × work)
        // ==========================================================
        // Use JacobiSVD for small matrices (work is typically k+7)
        //
        // CRITICAL: B is stored as LOWER bidiagonal, but the actual
        // Lanczos relation is A*V = W*B_upper where B_upper = B_lower^T.
        // Since B_lower = matrixU() * S * matrixV()^T, we have
        //   B_upper = matrixV() * S * matrixU()^T
        // Therefore:
        //   Right SVs of A ≈ V * matrixU()  (left SVs of B_lower)
        //   Left SVs of A  ≈ W * matrixV()  (right SVs of B_lower)
        //   Residuals use last row of matrixV()
        //
        // We assign BU = matrixV() and BV_mat = matrixU() so that the
        // rest of the code (restart V with BV_mat, restart W with BU,
        // residuals from BU) all use the correct components.
        Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_b(
            B.topLeftCorner(work, work),
            Eigen::ComputeFullU | Eigen::ComputeFullV);

        BS = svd_b.singularValues();
        BU = svd_b.matrixV();       // RIGHT SVs of B_lower → for W compression + residuals
        BV_mat = svd_b.matrixU();   // LEFT SVs of B_lower → for V compression

        // Normalize the residual vector
        Scalar R_F = F.norm();
        if (R_F > eps) {
            F /= R_F;
        } else {
            R_F = 0;
        }

        // ==========================================================
        // Step 4: Convergence test
        // ==========================================================
        Smax = std::max(Smax, BS(0));

        // Residual bounds: signed for restart, unsigned for convergence
        for (int i = 0; i < work; ++i) {
            signed_res(i) = R_F * BU(work - 1, i);     // Signed: for B restart
            residuals(i) = std::abs(signed_res(i));      // Unsigned: for convergence
        }

        // SV ratio: relative change from previous iteration
        for (int i = 0; i < work; ++i) {
            if (svratio_prev(i) > 0 && BS(i) > eps) {
                svratio_comp(i) = std::abs(svratio_prev(i) - BS(i)) / BS(i);
            } else {
                svratio_comp(i) = static_cast<Scalar>(1.0);  // Not converged yet
            }
        }

        // CRITICAL: k must be re-initialized to work before conv_test,
        // matching irlba C code's "kk = j" (j=work). conv_test then
        // clamps it to work-3, which keeps more Ritz vectors during
        // restart and prevents spurious singular values.
        k = work;
        converged = detail::conv_test(work, nu, tol, svtol, Smax,
                                       svratio_comp, residuals, k, S);

        if (config.verbose) {
            int n_conv = 0;
            for (int i = 0; i < work; ++i) {
                if (std::abs(residuals(i)) < tol * Smax && svratio_comp(i) < svtol)
                    n_conv++;
            }
            IRLBA_PRINT("    iter %d: sigma_1=%.6e, |res_1|=%.2e, converged=%d/%d, mprod=%d\n",
                         iter + 1, static_cast<double>(BS(0)),
                         static_cast<double>(std::abs(residuals(0))),
                         n_conv, nu, mprod);
        }

        if (converged) {
            iter++;
            break;
        }

        if (iter >= maxit - 1) {
            iter++;
            if (config.verbose) {
                IRLBA_PRINT("    IRLBA: reached maxit=%d without full convergence\n", maxit);
            }
            break;
        }

        // ==========================================================
        // Step 5: Store current BS for svratio computation next iter
        // ==========================================================
        for (int i = 0; i < work; ++i) {
            svratio_prev(i) = BS(i);
        }

        // ==========================================================
        // Step 6: Implicit restart — compress to k directions
        //
        // Matches irlba C code: use TRUNCATED BV/BU blocks (k×k),
        // not the full work×k rectangular blocks. This is the
        // "thick restart" truncation that the algorithm requires.
        // ==========================================================

        // V1[:, 0:k] = V * BV[:, 0:k]    (full right Ritz vectors)
        V1.leftCols(k).noalias() = V.leftCols(work) * BV_mat.topLeftCorner(work, k);
        V.leftCols(k) = V1.leftCols(k);
        V.col(k) = F;  // Append normalized residual as next starting vector

        // W1[:, 0:k] = W * BU[:, 0:k]    (full left Ritz vectors)
        W1.leftCols(k).noalias() = W.leftCols(work) * BU.topLeftCorner(work, k);
        W.leftCols(k) = W1.leftCols(k);

        // Rebuild B: diagonal with converged SVs + last column with residuals
        B.setZero();
        for (int i = 0; i < k; ++i) {
            B(i, i) = BS(i);                  // Diagonal: converged singular values
            B(i, k) = signed_res(i);           // Last column (col k): SIGNED residuals
        }

        iter++;
    }

    // ========================================================================
    // Extract final results
    // ========================================================================
    int k_actual = std::min(nu, work);

    if (BS.size() >= k_actual && BU.rows() >= work && BV_mat.rows() >= work) {
        // Final singular vectors: rotate Lanczos vectors by Ritz vectors
        result.U = W.leftCols(work) * BU.leftCols(k_actual);   // m × k_actual
        result.V = V.leftCols(work) * BV_mat.leftCols(k_actual); // n × k_actual
        result.d = BS.head(k_actual);
    } else {
        // Fallback: shouldn't happen, but be safe
        result.U = DenseMatrix<Scalar>::Zero(m, k_actual);
        result.V = DenseMatrix<Scalar>::Zero(n, k_actual);
        result.d = DenseVector<Scalar>::Zero(k_actual);
    }

    result.k_selected = k_actual;
    result.iters_per_factor.push_back(iter);

    auto t_end = Clock::now();
    result.wall_time_ms = Ms(t_end - t_start).count();

    if (config.verbose) {
        IRLBA_PRINT("  IRLBA done: %d restarts, %d matvecs, %.1f ms, %s\n",
                     iter, mprod, result.wall_time_ms,
                     converged ? "converged" : "NOT converged");
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet

