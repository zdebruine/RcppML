// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file streaming.hpp
 * @brief Unified streaming SVD gateway for all algorithms via .spz v2 files
 *
 * Provides streaming (out-of-core) versions of ALL SVD algorithms:
 *   - deflation:   streaming_deflation_svd()
 *   - krylov:      streaming_krylov_svd()
 *   - lanczos:     streaming_lanczos_svd()
 *   - irlba:       streaming_irlba_svd()
 *   - randomized:  streaming_randomized_svd()
 *
 * All algorithms access A only through streaming SpMV/SpMM products
 * from streaming_matvec.hpp. Memory: O(m*k + n*k + max_chunk_nnz).
 *
 * Constrained methods (deflation, krylov) support all regularization
 * features identical to their in-memory counterparts:
 *   L1, L2, nonneg, bounds, L21, angular, graph Laplacian, CV/auto-rank.
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/svd/streaming_matvec.hpp>
#include <FactorNet/svd/deflation.hpp>
#include <FactorNet/svd/lanczos.hpp>
#include <FactorNet/svd/irlba.hpp>
#include <FactorNet/svd/test_entries.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/rng/rng.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/graph_reg.hpp>

#include <cmath>
#include <algorithm>
#include <numeric>
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
#define SSVD_PRINT(...) { char buf[512]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); Rcpp::Rcout << buf; }
#define SSVD_CHECK_INTERRUPT() Rcpp::checkUserInterrupt()
#else
#include <cstdio>
#define SSVD_PRINT(...) std::printf(__VA_ARGS__)
#define SSVD_CHECK_INTERRUPT() ((void)0)
#endif

namespace FactorNet {
namespace svd {

// ============================================================================
// Streaming Deflation SVD
// ============================================================================

/**
 * @brief Deflation SVD from .spz v2 file (out-of-core)
 *
 * Same math as deflation_svd.hpp but SpMV streams from compressed chunks.
 * Supports all constraints: L1, L2, nonneg, bounds, L21, angular, graph.
 * Supports CV auto-rank.
 *
 * Total file passes: k × maxit × 2.
 */
template<typename Scalar>
SVDResult<Scalar> streaming_deflation_svd(const SVDConfig<Scalar>& config)
{
    using namespace detail;
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    if (config.spz_path.empty()) {
        throw std::runtime_error("streaming_deflation_svd requires config.spz_path");
    }
    if (config.L21_u > 0 || config.L21_v > 0) {
        throw std::invalid_argument(
            "svd(method = \"deflation\"): L21 (group sparsity across factors) is not "
            "defined when factors are fit one at a time; use method = \"krylov\"");
    }
    if (config.upper_bound_u > 0 || config.upper_bound_v > 0) {
        throw std::invalid_argument(
            "svd: 'upper_bound' is not supported: factors are returned with unit "
            "norm, so an element-wise bound has no scale-free meaning");
    }
    // Two-sided ridge on a rank-one term = soft threshold on its scale
    const Scalar l2_tau = (config.L2_u > 0 && config.L2_v > 0)
        ? std::sqrt(config.L2_u * config.L2_v) : static_cast<Scalar>(0);

    StreamingContext<Scalar> ctx;
    ctx.load(config.spz_path, config.center, config.scale);

    const uint32_t m = ctx.m;
    const uint32_t n = ctx.n;
    const int k_max = config.k_max;

    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = ctx.row_means;
    result.scaled = config.scale;
    if (config.scale) result.row_sds = ctx.row_sds;
    result.frobenius_norm_sq = static_cast<double>(ctx.frobenius_norm_sq);

    // Factor storage
    DenseMatrix<Scalar> U_all(m, k_max);
    DenseVector<Scalar> d_all(k_max);
    DenseMatrix<Scalar> V_all(n, k_max);
    d_all.setZero();

    rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));

    DenseVector<Scalar> u(m), v(n), u_old(m), u_prev(m);
    DenseVector<Scalar> spmv_result;
    int best_k = 0;

    // CV setup: build test entries from cached panels
    const bool do_cv = config.is_cv();
    TestEntries<Scalar> test_entries;
    Scalar best_test_mse = std::numeric_limits<Scalar>::max();
    int patience_counter = 0;
    if (do_cv) {
        ctx.ensure_cache();
        int64_t total_nnz = 0;
        for (uint32_t c = 0; c < ctx.num_chunks; ++c)
            total_nnz += ctx.cached_panels[c].nonZeros();

        FactorNet::nmf::LazySpeckledMask<Scalar> mask(
            m, n, total_nnz,
            static_cast<double>(config.test_fraction),
            config.effective_cv_seed(),
            true);  // mask_zeros=true for streaming

        test_entries = init_test_entries_streaming<Scalar>(ctx, mask, ctx.row_means_ptr);

        if (config.verbose) {
            SSVD_PRINT("  Streaming deflation CV: %d test entries\n",
                       test_entries.count);
        }
    }

    for (int k = 0; k < k_max; ++k) {
        SSVD_CHECK_INTERRUPT();

        // Initialization: power-step warm start for k>0
        if (k == 0) {
            for (uint32_t i = 0; i < m; ++i)
                u(i) = static_cast<Scalar>(rng.uniform<double>());
        } else {
            u = U_all.col(k - 1);
            for (int r = 0; r < k; ++r)
                u -= u.dot(U_all.col(r)) * U_all.col(r);
            Scalar u_norm_init = u.norm();
            if (u_norm_init > std::numeric_limits<Scalar>::epsilon() * 100) {
                u /= u_norm_init;
                detail::spmv_transpose(ctx, u, spmv_result);
                deflation_correct(spmv_result, u, V_all, d_all, U_all, k);
                v = spmv_result;
                Scalar v_norm_tmp = v.norm();
                if (v_norm_tmp > 0) v /= v_norm_tmp;
                detail::spmv_forward(ctx, v, spmv_result);
                deflation_correct(spmv_result, v, U_all, d_all, V_all, k);
                u = spmv_result;
                for (int r = 0; r < k; ++r)
                    u -= u.dot(U_all.col(r)) * U_all.col(r);
            } else {
                for (uint32_t i = 0; i < m; ++i)
                    u(i) = static_cast<Scalar>(rng.uniform<double>());
            }
        }
        Scalar u_norm = u.norm();
        if (u_norm > 0) u /= u_norm;

        // Adaptive tolerance
        Scalar tol_k = config.tol;
        if (k > 0 && d_all(0) > 0 && d_all(k - 1) > 0) {
            tol_k = config.tol * d_all(0) / d_all(k - 1);
            tol_k = std::min(tol_k, config.tol * static_cast<Scalar>(100));
        }

        Scalar sigma = 0;
        Scalar prev_sigma = 0;
        int iter = 0;
        u_prev = u;

        for (iter = 0; iter < config.max_iter; ++iter) {
            u_old = u;

            // Nesterov momentum
            Scalar beta = (iter <= 1) ? static_cast<Scalar>(0)
                                      : static_cast<Scalar>(iter - 1) / static_cast<Scalar>(iter + 2);
            DenseVector<Scalar> u_hat = u + beta * (u - u_prev);
            u_prev = u;

            // === V-update ===
            detail::spmv_transpose(ctx, u_hat, spmv_result);
            deflation_correct(spmv_result, u_hat, V_all, d_all, U_all, k);

            Scalar u_sq = u_hat.squaredNorm();
            if (u_sq > 0) {
                v = spmv_result / u_sq;
            } else { v.setZero(); break; }

            apply_regularization(v, config.L1_v, static_cast<Scalar>(0),
                                 config.nonneg_v, static_cast<Scalar>(0), u_sq,
                                 static_cast<Scalar>(0));
            apply_angular_penalty(v, V_all, k, config.angular_v, u_sq);
            apply_graph_penalty(v, config.graph_v, config.graph_v_lambda, u_sq);

            sigma = v.norm();
            if (sigma > 0) v /= sigma; else break;

            // === U-update ===
            detail::spmv_forward(ctx, v, spmv_result);
            deflation_correct(spmv_result, v, U_all, d_all, V_all, k);

            Scalar v_sq = v.squaredNorm();
            if (v_sq > 0) {
                u = spmv_result / v_sq;
            } else { u.setZero(); break; }

            apply_regularization(u, config.L1_u, static_cast<Scalar>(0),
                                 config.nonneg_u, static_cast<Scalar>(0), v_sq,
                                 static_cast<Scalar>(0));
            apply_angular_penalty(u, U_all, k, config.angular_u, v_sq);
            apply_graph_penalty(u, config.graph_u, config.graph_u_lambda, v_sq);

            sigma = u.norm();
            if (sigma > 0) u /= sigma; else break;

            // Convergence
            Scalar cos_dist = static_cast<Scalar>(1) - std::abs(u.dot(u_old));
            bool converged = false;
            switch (config.convergence) {
                case SVDConvergence::FACTOR: converged = (cos_dist < tol_k); break;
                case SVDConvergence::LOSS: {
                    Scalar d_sigma = std::abs(sigma - prev_sigma) /
                        (prev_sigma + std::numeric_limits<Scalar>::epsilon());
                    converged = (iter > 0 && d_sigma < tol_k); break;
                }
                case SVDConvergence::BOTH: {
                    bool fc = (cos_dist < tol_k);
                    Scalar d_sigma = std::abs(sigma - prev_sigma) /
                        (prev_sigma + std::numeric_limits<Scalar>::epsilon());
                    converged = fc || (iter > 0 && d_sigma < tol_k); break;
                }
            }
            prev_sigma = sigma;
            if (converged) { ++iter; break; }
        }

        U_all.col(k) = u;
        d_all(k) = sigma;
        V_all.col(k) = v;
        result.iters_per_factor.push_back(iter);
        best_k = k + 1;

        if (config.verbose) {
            SSVD_PRINT("  Streaming deflation: Factor %d: sigma=%.4e  iters=%d\n",
                       k + 1, static_cast<double>(sigma), iter);
        }

        // Per-rank CV evaluation
        if (do_cv) {
            test_entries.update(u, sigma, v, config.threads);
            Scalar test_mse = test_entries.compute_mse(config.threads);

            if (config.verbose) {
                SSVD_PRINT("  Streaming CV: rank %d: test_mse=%.6e\n",
                           k + 1, static_cast<double>(test_mse));
            }

            result.test_loss_trajectory.push_back(test_mse);

            if (test_mse < best_test_mse) {
                best_test_mse = test_mse;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    if (config.verbose) {
                        SSVD_PRINT("  Streaming CV: patience exhausted at rank %d, "
                                   "best rank = %d (test_mse=%.6e)\n",
                                   k + 1, best_k - patience_counter,
                                   static_cast<double>(best_test_mse));
                    }
                    best_k = best_k - patience_counter;
                    break;
                }
            }
        }

        // Stop when nothing is left relative to the first factor
        if (!(sigma > 0) ||
            (k > 0 && sigma <= std::numeric_limits<Scalar>::epsilon() * 100 * d_all(0))) break;
    }

    result.k_selected = best_k;
    result.U = U_all.leftCols(best_k);
    result.d = d_all.head(best_k);
    result.V = V_all.leftCols(best_k);
    if (l2_tau > 0) {
        for (int i = 0; i < best_k; ++i)
            result.d(i) = std::max(static_cast<Scalar>(0), result.d(i) - l2_tau);
    }
    result.wall_time_ms = Ms(Clock::now() - t_start).count();
    return result;
}

// ============================================================================
// Streaming Lanczos SVD
// ============================================================================

/**
 * @brief Lanczos bidiagonalization from .spz v2 file
 *
 * Same Golub-Kahan-Lanczos with full reorthogonalization,
 * but SpMV/SpMM streams from compressed chunks.
 * Total file passes: O(lanczos_steps).
 */
template<typename Scalar>
SVDResult<Scalar> streaming_lanczos_svd(const SVDConfig<Scalar>& config,
                                        bool strict = true)
{
    using namespace detail;
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    StreamingContext<Scalar> ctx;
    ctx.load(config.spz_path, config.center, config.scale);

    const int m = static_cast<int>(ctx.m);
    const int n = static_cast<int>(ctx.n);
    const int k = config.k_max;

    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = ctx.row_means;
    result.scaled = config.scale;
    if (config.scale) result.row_sds = ctx.row_sds;
    result.frobenius_norm_sq = static_cast<double>(ctx.frobenius_norm_sq);

    auto forward_vec = [&](const DenseVector<Scalar>& x, DenseVector<Scalar>& out) {
        detail::spmv_forward(ctx, x, out);
    };
    auto transpose_vec = [&](const DenseVector<Scalar>& x, DenseVector<Scalar>& out) {
        detail::spmv_transpose(ctx, x, out);
    };

    int j_actual = 0, n_unconv = 0;
    detail::lanczos_bidiag_core<Scalar>(
        m, n, k, config.tol, config.max_iter > 0 ? config.max_iter : 0,
        config.seed, config.verbose, strict,
        forward_vec, transpose_vec,
        result.U, result.d, result.V, j_actual, n_unconv);

    result.k_selected = static_cast<int>(result.d.size());
    result.iters_per_factor.push_back(j_actual);

    if (strict && n_unconv > 0) {
        char wbuf[320];
        std::snprintf(wbuf, sizeof(wbuf),
            "svd(method = \"lanczos\"): %d of %d singular values did not reach "
            "tol = %.1e after %d Lanczos steps; increase 'maxit' (number of steps) "
            "or use method = \"irlba\"",
            n_unconv, result.k_selected, static_cast<double>(config.tol), j_actual);
        add_warning(std::string(wbuf));
    }
    result.wall_time_ms = Ms(Clock::now() - t_start).count();

    if (config.verbose) {
        SSVD_PRINT("  Streaming Lanczos done: %.1f ms, %d steps\n",
                   result.wall_time_ms, j_actual);
    }
    return result;
}

// ============================================================================
// Streaming IRLBA SVD
// ============================================================================

/**
 * @brief Implicitly Restarted Lanczos Bidiagonalization from .spz v2 file
 *
 * Same IRLBA algorithm with augmented implicit restarts,
 * but SpMV/SpMM streams from compressed chunks.
 */
template<typename Scalar>
SVDResult<Scalar> streaming_irlba_svd(const SVDConfig<Scalar>& config)
{
    using namespace detail;
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    StreamingContext<Scalar> ctx;
    ctx.load(config.spz_path, config.center, config.scale);

    const int m = static_cast<int>(ctx.m);
    const int n = static_cast<int>(ctx.n);
    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = ctx.row_means;
    result.scaled = config.scale;
    if (config.scale) result.row_sds = ctx.row_sds;
    result.frobenius_norm_sq = static_cast<double>(ctx.frobenius_norm_sq);

    auto forward_vec = [&](const DenseVector<Scalar>& x, DenseVector<Scalar>& out) {
        detail::spmv_forward(ctx, x, out);
    };
    auto transpose_vec = [&](const DenseVector<Scalar>& x, DenseVector<Scalar>& out) {
        detail::spmv_transpose(ctx, x, out);
    };

    int iter = 0, mprod = 0, n_unconv = 0;
    detail::irlba_core<Scalar>(
        m, n, config.k_max, config.tol, config.max_iter,
        config.seed, config.verbose,
        forward_vec, transpose_vec,
        result.U, result.d, result.V, iter, mprod, n_unconv);

    result.k_selected = static_cast<int>(result.d.size());
    result.iters_per_factor.push_back(iter);

    if (n_unconv > 0) {
        char wbuf[320];
        std::snprintf(wbuf, sizeof(wbuf),
            "svd(method = \"irlba\"): %d of %d singular values did not reach "
            "tol = %.1e after %d restarts; increase 'maxit' (number of restarts) "
            "or loosen 'tol'",
            n_unconv, result.k_selected, static_cast<double>(config.tol), iter);
        add_warning(std::string(wbuf));
    }
    result.wall_time_ms = Ms(Clock::now() - t_start).count();

    if (config.verbose) {
        SSVD_PRINT("  Streaming IRLBA done: %d restarts, %d matvecs, %.1f ms\n",
                   iter, mprod, result.wall_time_ms);
    }
    return result;
}

// ============================================================================
// Streaming Randomized SVD
// ============================================================================

/**
 * @brief Randomized SVD (Halko-Martinsson-Tropp) from .spz v2 file
 *
 * Random sketch + power iterations, all via streaming SpMM.
 * Total file passes: 2*(power_iterations+1) + 1.
 */
template<typename Scalar>
SVDResult<Scalar> streaming_randomized_svd(const SVDConfig<Scalar>& config)
{
    using namespace detail;
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    StreamingContext<Scalar> ctx;
    ctx.load(config.spz_path, config.center, config.scale);

    const int m = static_cast<int>(ctx.m);
    const int n = static_cast<int>(ctx.n);
    const int k = config.k_max;
    const int p = std::min(std::max(10, k / 5), std::min(m, n) - k);
    const int l = k + std::max(p, 0);
    const int q = (config.max_iter <= 0) ? 3 : std::min(config.max_iter, 100);

    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = ctx.row_means;
    result.scaled = config.scale;
    if (config.scale) result.row_sds = ctx.row_sds;
    result.frobenius_norm_sq = static_cast<double>(ctx.frobenius_norm_sq);

    // Random sketch Omega (n × l)
    DenseMatrix<Scalar> Omega(n, l);
    {
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        rng.fill_uniform(Omega);
        Omega.array() -= static_cast<Scalar>(0.5);
    }

    // Y = A * Omega  (m × l)
    DenseMatrix<Scalar> Y(m, l);
    detail::spmm_forward(ctx, Omega, Y);

    if (config.verbose) {
        SSVD_PRINT("  Streaming randomized SVD: k=%d, l=%d, power_iter=%d\n", k, l, q);
    }

    // Power iterations for spectral concentration
    DenseMatrix<Scalar> Z(n, l);
    // Pre-allocate Identity matrices for QR thin-Q extraction
    DenseMatrix<Scalar> I_m = DenseMatrix<Scalar>::Identity(m, l);
    DenseMatrix<Scalar> I_n = DenseMatrix<Scalar>::Identity(n, l);
    for (int i = 0; i < q; ++i) {
        SSVD_CHECK_INTERRUPT();
        Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_y(Y);
        Y = qr_y.householderQ() * I_m;
        detail::spmm_transpose(ctx, Y, Z);
        Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_z(Z);
        Z = qr_z.householderQ() * I_n;
        detail::spmm_forward(ctx, Z, Y);
    }

    // Final QR → orthonormal basis Q for range(A)
    Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_final(Y);
    DenseMatrix<Scalar> Q_mat = qr_final.householderQ() * I_m;

    // Bt = A' * Q → SVD(Bt) with U/V swap (avoids Bt.transpose() copy)
    DenseMatrix<Scalar> Bt(n, l);
    detail::spmm_transpose(ctx, Q_mat, Bt);

    // SVD of Bt (n × l): Bt = Ub · Σ · Vb' ⟹ B = Bt' = Vb · Σ · Ub'
    Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_bt(Bt, Eigen::ComputeThinU | Eigen::ComputeThinV);
    int k_actual = std::min(k, static_cast<int>(svd_bt.singularValues().size()));

    result.d = svd_bt.singularValues().head(k_actual);
    result.U = Q_mat * svd_bt.matrixV().leftCols(k_actual);  // Q * Vb (m × k)
    result.V = svd_bt.matrixU().leftCols(k_actual);            // Ub (n × k)
    result.k_selected = k_actual;
    result.iters_per_factor.push_back(q);
    result.wall_time_ms = Ms(Clock::now() - t_start).count();

    if (config.verbose) {
        SSVD_PRINT("  Streaming randomized SVD done: %.1f ms\n", result.wall_time_ms);
    }
    return result;
}

// ============================================================================
// Streaming Krylov Constrained SVD (KSPR)
// ============================================================================

/**
 * @brief Krylov-seeded projected refinement from .spz v2 file
 *
 * Phase 1: Streaming Lanczos seed (unconstrained warm start)
 * Phase 2: Streaming SpMM-based Gram-solve-then-project refinement
 *
 * Supports all constraints: L1, L2, nonneg, bounds, L21, angular, graph.
 * Supports CV/auto-rank.
 */
template<typename Scalar>
SVDResult<Scalar> streaming_krylov_svd(const SVDConfig<Scalar>& config)
{
    using namespace detail;
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    StreamingContext<Scalar> ctx;
    ctx.load(config.spz_path, config.center, config.scale);

    const int m = static_cast<int>(ctx.m);
    const int n = static_cast<int>(ctx.n);
    const int k = config.k_max;

    // Check for constraints
    const bool has_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.L2_u > 0 || config.L2_v > 0 ||
        config.upper_bound_u > 0 || config.upper_bound_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        config.angular_u > 0 || config.angular_v > 0 ||
        config.has_gram_level_reg();

    if (config.upper_bound_u > 0 || config.upper_bound_v > 0) {
        throw std::invalid_argument(
            "svd: 'upper_bound' is not supported: factors are returned with unit "
            "norm, so an element-wise bound has no scale-free meaning");
    }

    // If no constraints and no CV, just run streaming Lanczos (optimal)
    if (!has_constraints && !config.is_cv()) {
        return streaming_lanczos_svd<Scalar>(config);
    }

    // CV with streaming: data lives on disk but cached panels are in memory.
    // We apply Gram denominator correction during training AND compute
    // per-rank test MSE from cached panels after convergence.
    const bool do_cv = config.is_cv();
    Scalar cv_denom_correction = Scalar(1);
    if (do_cv) {
        // mask_zeros forced true for streaming (only nonzeros are in panel data)
        cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        if (config.verbose) {
            SSVD_PRINT("  Streaming CV: denom_correction=%.4f\n",
                       static_cast<double>(cv_denom_correction));
        }
    }

    // Phase 1: Streaming Lanczos seed
    SVDConfig<Scalar> lanczos_config;
    lanczos_config.k_max = k;
    lanczos_config.tol = config.tol;
    lanczos_config.max_iter = 0;  // use default Lanczos steps
    lanczos_config.center = config.center;
    lanczos_config.scale = config.scale;
    lanczos_config.verbose = false;
    lanczos_config.seed = config.seed;
    lanczos_config.threads = config.threads;
    lanczos_config.spz_path = config.spz_path;

    auto t_p1 = Clock::now();
    // internal seed: short step budget, no convergence warning
    SVDResult<Scalar> lanczos_result = streaming_lanczos_svd<Scalar>(lanczos_config, false);
    auto t_p1_end = Clock::now();

    if (config.verbose) {
        SSVD_PRINT("  Streaming Krylov: Phase 1 (Lanczos seed): %.1f ms, k=%d\n",
                   Ms(t_p1_end - t_p1).count(), lanczos_result.k_selected);
    }

    const int k_actual = lanczos_result.k_selected;
    if (k_actual == 0) {
        lanczos_result.wall_time_ms = Ms(Clock::now() - t_start).count();
        return lanczos_result;
    }

    // Phase 2: Gram-solve-then-project refinement via streaming SpMM
    DenseMatrix<Scalar> W = lanczos_result.U.leftCols(k_actual);   // m × k
    DenseMatrix<Scalar> V = lanczos_result.V.leftCols(k_actual);   // n × k
    DenseVector<Scalar> d = lanczos_result.d.head(k_actual);

    const int max_passes = config.max_iter > 0
        ? config.max_iter
        : std::max(10, static_cast<int>(std::ceil(std::log2(
              static_cast<double>(k_actual)))) * 2 + 3);
    const Scalar refine_tol = config.tol;
    const Scalar eps100 = std::numeric_limits<Scalar>::epsilon() * 100;

    DenseMatrix<Scalar> prev_W;
    DenseMatrix<Scalar> SpMM_result;
    DenseMatrix<Scalar> G(k_actual, k_actual);
    DenseVector<Scalar> norm_sq_vec(k_actual);

    for (int pass = 0; pass < max_passes; ++pass) {
        SSVD_CHECK_INTERRUPT();

        // --- W-update: W = A * V, solve via Gram, then project ---
        G.noalias() = V.transpose() * V;
        if (do_cv) G *= cv_denom_correction;
        G.diagonal().array() += static_cast<Scalar>(1e-12);
        if (config.L2_u > 0) G.diagonal().array() += config.L2_u;

        // Gram-level regularization on W — compute Wt once
        const bool need_Wt = (config.L21_u > 0) || (config.angular_u > 0) ||
                             (config.graph_u && config.graph_u_lambda > 0);
        DenseMatrix<Scalar> Wt;
        if (need_Wt) Wt = W.transpose();

        if (config.L21_u > 0) {
            features::apply_L21(G, Wt, config.L21_u);
        }
        if (config.angular_u > 0) {
            features::apply_angular(G, Wt, config.angular_u);
        }
        if (config.graph_u && config.graph_u_lambda > 0) {
            features::apply_graph_reg(G, *config.graph_u, Wt, config.graph_u_lambda);
        }

        detail::spmm_forward(ctx, V, SpMM_result);   // result = A * V (m × k)

        {
            Eigen::LLT<DenseMatrix<Scalar>> llt(G);
            DenseMatrix<Scalar> Bt = SpMM_result.transpose();
            llt.solveInPlace(Bt);
            W = Bt.transpose();
        }

        for (int c = 0; c < k_actual; ++c) norm_sq_vec(c) = V.col(c).squaredNorm();
        if (do_cv) norm_sq_vec *= cv_denom_correction;

        // Project element-wise constraints
        for (int c = 0; c < k_actual; ++c) {
            Scalar nsq = norm_sq_vec(c);
            if (nsq <= 0) { W.col(c).setZero(); continue; }
            if (config.L1_u > 0) {
                Scalar thr = config.L1_u / (static_cast<Scalar>(2) * nsq);
                for (int i = 0; i < m; ++i) {
                    Scalar val = W(i, c);
                    W(i, c) = (val > thr) ? val - thr : (val < -thr) ? val + thr : 0;
                }
            }
            if (config.nonneg_u) W.col(c) = W.col(c).cwiseMax(static_cast<Scalar>(0));
            if (config.upper_bound_u > 0) W.col(c) = W.col(c).cwiseMin(config.upper_bound_u);
        }

        // Normalize columns → extract d
        for (int c = 0; c < k_actual; ++c) {
            Scalar nrm = W.col(c).norm();
            d(c) = (nrm > eps100) ? nrm : 0;
            if (d(c) > 0) W.col(c) /= d(c);
        }

        // --- V-update: V = A' * W, solve via Gram, then project ---
        G.noalias() = W.transpose() * W;
        if (do_cv) G *= cv_denom_correction;
        G.diagonal().array() += static_cast<Scalar>(1e-12);
        if (config.L2_v > 0) G.diagonal().array() += config.L2_v;

        // Gram-level regularization on V — compute Vt once
        const bool need_Vt = (config.L21_v > 0) || (config.angular_v > 0) ||
                             (config.graph_v && config.graph_v_lambda > 0);
        DenseMatrix<Scalar> Vt_str;
        if (need_Vt) Vt_str = V.transpose();

        if (config.L21_v > 0) {
            features::apply_L21(G, Vt_str, config.L21_v);
        }
        if (config.angular_v > 0) {
            features::apply_angular(G, Vt_str, config.angular_v);
        }
        if (config.graph_v && config.graph_v_lambda > 0) {
            features::apply_graph_reg(G, *config.graph_v, Vt_str, config.graph_v_lambda);
        }

        detail::spmm_transpose(ctx, W, SpMM_result);  // result = A' * W (n × k)

        {
            Eigen::LLT<DenseMatrix<Scalar>> llt(G);
            DenseMatrix<Scalar> Bt = SpMM_result.transpose();
            llt.solveInPlace(Bt);
            V = Bt.transpose();
        }

        for (int c = 0; c < k_actual; ++c) norm_sq_vec(c) = W.col(c).squaredNorm();
        if (do_cv) norm_sq_vec *= cv_denom_correction;

        for (int c = 0; c < k_actual; ++c) {
            Scalar nsq = norm_sq_vec(c);
            if (nsq <= 0) { V.col(c).setZero(); continue; }
            if (config.L1_v > 0) {
                Scalar thr = config.L1_v / (static_cast<Scalar>(2) * nsq);
                for (int i = 0; i < n; ++i) {
                    Scalar val = V(i, c);
                    V(i, c) = (val > thr) ? val - thr : (val < -thr) ? val + thr : 0;
                }
            }
            if (config.nonneg_v) V.col(c) = V.col(c).cwiseMax(static_cast<Scalar>(0));
            if (config.upper_bound_v > 0) V.col(c) = V.col(c).cwiseMin(config.upper_bound_v);
        }

        for (int c = 0; c < k_actual; ++c) {
            Scalar nrm = V.col(c).norm();
            d(c) = (nrm > eps100) ? nrm : 0;
            if (d(c) > 0) V.col(c) /= d(c);
        }

        // Convergence check (factor change)
        if (pass > 0) {
            Scalar dW = (W - prev_W).norm() / (prev_W.norm() + eps100);
            if (config.verbose) {
                SSVD_PRINT("    Streaming Krylov pass %d: dW=%.2e\n",
                           pass + 1, static_cast<double>(dW));
            }
            if (dW < refine_tol) break;
        }
        prev_W = W;
    }

    // ====================================================================
    // Phase 2.5: CV auto-rank selection (post-convergence)
    // ====================================================================

    // Build result — sort by decreasing d
    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = ctx.row_means;
    result.scaled = config.scale;
    if (config.scale) result.row_sds = ctx.row_sds;
    result.frobenius_norm_sq = static_cast<double>(ctx.frobenius_norm_sq);

    std::vector<int> order(k_actual);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return d(a) > d(b); });

    int best_k = k_actual;

    if (do_cv) {
        // Compute total nnz from cached panels for mask construction
        ctx.ensure_cache();
        int64_t total_nnz = 0;
        for (uint32_t c = 0; c < ctx.num_chunks; ++c)
            total_nnz += ctx.cached_panels[c].nonZeros();

        FactorNet::nmf::LazySpeckledMask<Scalar> mask(
            m, n, total_nnz,
            static_cast<double>(config.test_fraction),
            config.effective_cv_seed(),
            true);  // mask_zeros=true for streaming (only nonzeros available)

        TestEntries<Scalar> test_entries =
            init_test_entries_streaming<Scalar>(ctx, mask, ctx.row_means_ptr);

        Scalar best_test_mse = std::numeric_limits<Scalar>::max();
        int patience_counter = 0;

        for (int kk = 0; kk < k_actual; ++kk) {
            int idx = order[kk];
            test_entries.update(W.col(idx), d(idx), V.col(idx), config.threads);
            Scalar test_mse = test_entries.compute_mse(config.threads);

            if (config.verbose) {
                SSVD_PRINT("  Streaming CV: rank %d: test_mse=%.6e\n",
                           kk + 1, static_cast<double>(test_mse));
            }

            result.test_loss_trajectory.push_back(test_mse);

            if (test_mse < best_test_mse) {
                best_test_mse = test_mse;
                best_k = kk + 1;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    if (config.verbose) {
                        SSVD_PRINT("  Streaming CV: patience exhausted at rank %d, "
                                   "best rank = %d (test_mse=%.6e)\n",
                                   kk + 1, best_k, static_cast<double>(best_test_mse));
                    }
                    break;
                }
            }
        }
    }

    result.d.resize(best_k);
    result.U.resize(m, best_k);
    result.V.resize(n, best_k);
    for (int i = 0; i < best_k; ++i) {
        result.d(i) = d(order[i]);
        result.U.col(i) = W.col(order[i]);
        result.V.col(i) = V.col(order[i]);
    }

    result.k_selected = best_k;
    result.wall_time_ms = Ms(Clock::now() - t_start).count();

    if (config.verbose) {
        SSVD_PRINT("  Streaming Krylov done: %.1f ms\n", result.wall_time_ms);
    }
    return result;
}

}  // namespace svd
}  // namespace FactorNet

