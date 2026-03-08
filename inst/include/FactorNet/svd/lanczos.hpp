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
 * This is the optimal algorithm for truncated SVD — convergence depends
 * on spectral gaps but via Chebyshev polynomials (much faster than
 * power iteration used by deflation/block ALS).
 *
 * Supports: centering (PCA).
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/svd/spmv.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>  // for trace_AtA
#include <FactorNet/rng/rng.hpp>

#include <cmath>
#include <algorithm>
#include <chrono>

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

template<typename MatrixType, typename Scalar>
SVDResult<Scalar> lanczos_svd(const MatrixType& A,
                              const SVDConfig<Scalar>& config)
{
    using FactorNet::primitives::CPU;
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.k_max;

    // Number of Lanczos steps: k + extra, capped at min(m,n)-1
    // Number of Lanczos steps: if max_iter > 0, use it directly.
    // Otherwise default to max(3k, k+50) — 2k is too tight for bottom SVs
    // at large k. Early stopping avoids waste when convergence is fast.
    const int default_steps = std::max(3 * k, k + 50);
    const int user_steps = config.max_iter > 0 ? config.max_iter : 0;
    const int jmax = std::min(std::min(m, n) - 1,
                              std::max(default_steps, user_steps));

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
        // Cast once for trace computation
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
        // ||D^{-1}(A - μ1')||²_F = m*n (by definition of population SD)
        A_norm_sq = static_cast<Scalar>(m) * static_cast<Scalar>(n);
    }
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // --- SpMV helpers using OpenMP-parallel spmv.hpp routines ---
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

    // ========================================================================
    // Lanczos vectors and bidiagonal entries
    // ========================================================================
    DenseMatrix<Scalar> P(n, jmax + 1);   // right Lanczos vectors p_0...p_jmax
    DenseMatrix<Scalar> Q(m, jmax);        // left  Lanczos vectors q_0...q_{jmax-1}

    DenseVector<Scalar> alpha(jmax);       // diagonal of bidiagonal
    DenseVector<Scalar> beta(jmax + 1);    // superdiagonal + initial norm

    const Scalar eps = std::numeric_limits<Scalar>::epsilon() * 100;

    // --- Initialize p_0 = random unit vector ---
    {
        DenseVector<Scalar> p0(n);
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        // Fill via a small matrix (raw-pointer overload for nvcc compat)
        DenseMatrix<Scalar> p0_mat(n, 1);
        rng.fill_uniform(p0_mat.data(), p0_mat.rows(), p0_mat.cols());
        p0 = p0_mat.col(0);
        p0.array() -= static_cast<Scalar>(0.5);
        p0.normalize();
        P.col(0) = p0;
    }

    beta(0) = 0;

    int j_actual = 0;

    // Adaptive early stopping: check convergence every check_interval steps
    // after we have at least k+2 steps (enough for meaningful Ritz estimates)
    const int check_interval = 5;
    const int min_steps_before_check = k + 2;
    const Scalar conv_tol = config.tol > 0 ? config.tol : static_cast<Scalar>(1e-10);

    if (config.verbose) {
        LSVD_PRINT("  Lanczos bidiag: k=%d, jmax=%d, adaptive_tol=%.1e\n",
                   k, jmax, static_cast<double>(conv_tol));
    }

    // ========================================================================
    // Golub-Kahan-Lanczos bidiagonalization
    // ========================================================================
    DenseVector<Scalar> r(m), s(n);

    // Pre-allocate reorthogonalization coefficient vectors at max size
    // to avoid repeated allocation per step (originally O(jmax) allocs)
    DenseVector<Scalar> h_q(jmax);       // for Q reortho (size up to j)
    DenseVector<Scalar> h_p(jmax + 1);   // for P reortho (size up to j+1)

    for (int j = 0; j < jmax; ++j) {
        LSVD_CHECK_INTERRUPT();

        // --- Step 1: r = A · p_j - beta_j · q_{j-1} ---
        forward_vec(P.col(j), r);
        if (j > 0) {
            r.noalias() -= beta(j) * Q.col(j - 1);
        }

        // --- Full reorthogonalization against Q (two gemv passes for cache efficiency) ---
        if (j > 0) {
            auto Q_j = Q.leftCols(j);
            auto h = h_q.head(j);
            h.noalias() = Q_j.transpose() * r;
            r.noalias() -= Q_j * h;
            // Second pass for numerical stability
            h.noalias() = Q_j.transpose() * r;
            r.noalias() -= Q_j * h;
        }

        alpha(j) = r.norm();
        if (alpha(j) < eps) {
            // Lucky breakdown — found invariant subspace
            if (config.verbose) {
                LSVD_PRINT("    Lucky breakdown at step %d (alpha=%.2e)\n",
                           j, static_cast<double>(alpha(j)));
            }
            j_actual = j;
            break;
        }
        Q.col(j) = r / alpha(j);

        // --- Step 2: s = A' · q_j - alpha_j · p_j ---
        transpose_vec(Q.col(j), s);
        s.noalias() -= alpha(j) * P.col(j);

        // --- Full reorthogonalization against P (two gemv passes for cache efficiency) ---
        {
            auto P_j = P.leftCols(j + 1);
            auto h = h_p.head(j + 1);
            h.noalias() = P_j.transpose() * s;
            s.noalias() -= P_j * h;
            // Second pass for numerical stability
            h.noalias() = P_j.transpose() * s;
            s.noalias() -= P_j * h;
        }

        beta(j + 1) = s.norm();
        if (beta(j + 1) < eps) {
            // Lucky breakdown
            if (config.verbose) {
                LSVD_PRINT("    Lucky breakdown at step %d (beta=%.2e)\n",
                           j + 1, static_cast<double>(beta(j + 1)));
            }
            j_actual = j + 1;
            break;
        }
        P.col(j + 1) = s / beta(j + 1);

        j_actual = j + 1;

        // --- Adaptive early stopping check ---
        // Every check_interval steps (after min_steps), compute SVD of
        // current bidiagonal and check if all k Ritz values have converged.
        // Error bound for sigma_i: beta_{j+1} * |U_B[j, i]|
        if (j_actual >= min_steps_before_check &&
            j_actual % check_interval == 0 &&
            j_actual < jmax) {
            // Build current bidiagonal B_j
            DenseMatrix<Scalar> B_check = DenseMatrix<Scalar>::Zero(j_actual, j_actual);
            for (int jj = 0; jj < j_actual; ++jj) {
                B_check(jj, jj) = alpha(jj);
                if (jj < j_actual - 1)
                    B_check(jj, jj + 1) = beta(jj + 1);
            }
            Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_check(
                B_check, Eigen::ComputeFullU);
            auto sigmas = svd_check.singularValues();
            auto U_B = svd_check.matrixU();

            int k_check = std::min(k, j_actual);
            Scalar beta_last = std::abs(beta(j_actual));
            bool all_converged = true;
            for (int i = 0; i < k_check; ++i) {
                Scalar err = beta_last * std::abs(U_B(j_actual - 1, i));
                if (err > conv_tol * sigmas(0)) {
                    all_converged = false;
                    break;
                }
            }
            if (all_converged) {
                if (config.verbose) {
                    LSVD_PRINT("    Early stop at step %d: all %d Ritz values converged\n",
                               j_actual, k_check);
                }
                break;
            }
        }
    }

    if (config.verbose) {
        LSVD_PRINT("    Completed %d Lanczos steps\n", j_actual);
    }

    // ========================================================================
    // Build bidiagonal B (j_actual × j_actual), upper bidiagonal:
    //   B(j,j) = alpha(j),  B(j, j+1) = beta(j+1)
    //
    // Relation: A · P_{j_actual} ≈ Q_{j_actual} · B
    // ========================================================================
    DenseMatrix<Scalar> B = DenseMatrix<Scalar>::Zero(j_actual, j_actual);
    for (int j = 0; j < j_actual; ++j) {
        B(j, j) = alpha(j);
        if (j < j_actual - 1) {
            B(j, j + 1) = beta(j + 1);
        }
    }

    // ========================================================================
    // SVD of bidiagonal → Ritz values + vectors
    // ========================================================================
    Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_b(
        B, Eigen::ComputeFullU | Eigen::ComputeFullV);

    int k_actual = std::min(k, j_actual);

    result.d = svd_b.singularValues().head(k_actual);

    // U = Q_{j_actual} · U_B(:, 1:k)   (m × k)
    result.U = Q.leftCols(j_actual) * svd_b.matrixU().leftCols(k_actual);

    // V = P_{j_actual} · V_B(:, 1:k)   (n × k)
    result.V = P.leftCols(j_actual) * svd_b.matrixV().leftCols(k_actual);

    result.k_selected = k_actual;
    result.iters_per_factor.push_back(j_actual);

    // --- Convergence diagnostic: residual bounds ---
    if (config.verbose && j_actual > 0) {
        Scalar resid_bound = std::abs(beta(j_actual));
        LSVD_PRINT("    Residual norm (beta_%d) = %.4e\n",
                   j_actual, static_cast<double>(resid_bound));
        // For each wanted Ritz value, error ≤ beta_{j_actual} * |last component of U_B col|
        for (int i = 0; i < std::min(k_actual, 5); ++i) {
            Scalar err_bound = resid_bound *
                std::abs(svd_b.matrixU()(j_actual - 1, i));
            LSVD_PRINT("    sigma_%d = %.4e  err_bound = %.4e\n",
                       i + 1, static_cast<double>(result.d(i)),
                       static_cast<double>(err_bound));
        }
    }

    auto t_end = Clock::now();
    result.wall_time_ms = Ms(t_end - t_start).count();

    if (config.verbose) {
        LSVD_PRINT("  Lanczos SVD done: %.1f ms\n", result.wall_time_ms);
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet

