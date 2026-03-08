// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/nmf_init.hpp
 * @brief NMF factor initialization helpers (Lanczos, IRLBA, random)
 *
 * Extracted from fit_cpu.hpp namespace detail block.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/svd/lanczos.hpp>
#include <FactorNet/svd/irlba.hpp>
#include <FactorNet/rng/rng.hpp>

#include <cmath>
#include <algorithm>

namespace FactorNet {
namespace nmf {
namespace detail {


/**
 * @brief Initialize factors from Lanczos SVD seed
 *
 * Computes a rank-k Lanczos SVD of A, then sets:
 *   W_T = (|U| * sqrt(Sigma))^T   (k x m)
 *   H   = sqrt(Sigma) * |V|^T     (k x n)
 *   d   = ones (scaling extracted during first iteration)
 *
 * This provides a much better starting point than random init,
 * typically reducing iteration count by 30-50%.
 */
template<typename Scalar, typename MatrixType>
inline void initialize_lanczos(DenseMatrix<Scalar>& W_T,
                               DenseMatrix<Scalar>& H,
                               DenseVector<Scalar>& d,
                               const MatrixType& A,
                               int k, int m, int n,
                               uint32_t seed, int threads,
                               bool verbose)
{
    // Run Lanczos SVD
    FactorNet::SVDConfig<Scalar> svd_config;
    svd_config.k_max = k;
    svd_config.tol = static_cast<Scalar>(1e-10);
    svd_config.max_iter = 0;  // auto
    svd_config.center = false;
    svd_config.verbose = false;
    svd_config.seed = seed;
    svd_config.threads = threads;

    auto svd_result = svd::lanczos_svd<MatrixType, Scalar>(A, svd_config);
    const int k_actual = svd_result.k_selected;

    W_T.resize(k, m);
    H.resize(k, n);
    d = DenseVector<Scalar>::Ones(k);

    // W_T(i,:) = |U(:,i)| * sqrt(sigma_i)
    // H(i,:)  = |V(:,i)| * sqrt(sigma_i)
    for (int i = 0; i < std::min(k, k_actual); ++i) {
        Scalar sqrtd = std::sqrt(svd_result.d(i));
        for (int j = 0; j < m; ++j)
            W_T(i, j) = std::abs(svd_result.U(j, i)) * sqrtd;
        for (int j = 0; j < n; ++j)
            H(i, j) = std::abs(svd_result.V(j, i)) * sqrtd;
    }

    // Fill remaining ranks with random if Lanczos returned fewer
    if (k_actual < k) {
        rng::SplitMix64 rng(seed == 0 ? 54321U : seed + 999);
        DenseMatrix<Scalar> fill_W(k - k_actual, m);
        DenseMatrix<Scalar> fill_H(k - k_actual, n);
        rng.fill_uniform(fill_W.data(), fill_W.rows(), fill_W.cols());
        rng.fill_uniform(fill_H.data(), fill_H.rows(), fill_H.cols());
        W_T.bottomRows(k - k_actual) = fill_W;
        H.bottomRows(k - k_actual) = fill_H;
    }

    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "Lanczos init: k_svd=%d, seed=%u\n", k_actual, seed);
}

/**
 * @brief Initialize NMF factors using IRLBA (Implicitly Restarted Lanczos)
 *
 * IRLBA is more accurate than basic Lanczos for high-rank decompositions
 * (k >= 32) since it uses implicit restarts to maintain orthogonality.
 * For low rank (k <= 16), both methods are similar.
 *
 * Uses same interface as initialize_lanczos: W_T(i,:) = |U(:,i)|·√σ_i,
 * H(i,:) = |V(:,i)|·√σ_i.
 *
 * This provides a high-quality starting point, typically reducing iteration
 * count by 30-50% vs random init, especially for high rank.
 */
template<typename Scalar, typename MatrixType>
inline void initialize_irlba(DenseMatrix<Scalar>& W_T,
                             DenseMatrix<Scalar>& H,
                             DenseVector<Scalar>& d,
                             const MatrixType& A,
                             int k, int m, int n,
                             uint32_t seed, int threads,
                             bool verbose)
{
    // Run IRLBA SVD
    FactorNet::SVDConfig<Scalar> svd_config;
    svd_config.k_max = k;
    svd_config.tol = static_cast<Scalar>(1e-10);
    svd_config.max_iter = 0;  // auto
    svd_config.center = false;
    svd_config.verbose = false;
    svd_config.seed = seed;
    svd_config.threads = threads;

    auto svd_result = svd::irlba_svd<MatrixType, Scalar>(A, svd_config);
    const int k_actual = svd_result.k_selected;

    W_T.resize(k, m);
    H.resize(k, n);
    d = DenseVector<Scalar>::Ones(k);

    // W_T(i,:) = |U(:,i)| * sqrt(sigma_i)
    // H(i,:)  = |V(:,i)| * sqrt(sigma_i)
    for (int i = 0; i < std::min(k, k_actual); ++i) {
        Scalar sqrtd = std::sqrt(svd_result.d(i));
        for (int j = 0; j < m; ++j)
            W_T(i, j) = std::abs(svd_result.U(j, i)) * sqrtd;
        for (int j = 0; j < n; ++j)
            H(i, j) = std::abs(svd_result.V(j, i)) * sqrtd;
    }

    // Fill remaining ranks with random if IRLBA returned fewer
    if (k_actual < k) {
        rng::SplitMix64 rng(seed == 0 ? 54321U : seed + 999);
        DenseMatrix<Scalar> fill_W(k - k_actual, m);
        DenseMatrix<Scalar> fill_H(k - k_actual, n);
        rng.fill_uniform(fill_W.data(), fill_W.rows(), fill_W.cols());
        rng.fill_uniform(fill_H.data(), fill_H.rows(), fill_H.cols());
        W_T.bottomRows(k - k_actual) = fill_W;
        H.bottomRows(k - k_actual) = fill_H;
    }

    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "IRLBA init: k_svd=%d, seed=%u\n", k_actual, seed);
}

/**
 * @brief Initialize factors W (k×m) and H (k×n) with random values
 *
 * Uses SplitMix64 for reproducible initialization.
 * W_T is k × m (transposed storage for cache efficiency).
 */
template<typename Scalar>
inline void initialize_factors(DenseMatrix<Scalar>& W_T,
                               DenseMatrix<Scalar>& H,
                               DenseVector<Scalar>& d,
                               int k, int m, int n,
                               uint32_t seed)
{
    rng::SplitMix64 rng(seed);

    W_T.resize(k, m);
    rng.fill_uniform(W_T.data(), W_T.rows(), W_T.cols());

    H.resize(k, n);
    rng.fill_uniform(H.data(), H.rows(), H.cols());

    d = DenseVector<Scalar>::Ones(k);
}

}  // namespace detail
}  // namespace nmf
}  // namespace FactorNet
