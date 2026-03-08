// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit_cv_gpu.cuh
 * @brief GPU cross-validation NMF with unified config/result types
 *
 * Thin adapter: translates unified NMFConfig<Scalar> → FactorNet::gpu::CVNMFConfig,
 * calls the existing production GPU CV NMF code (nmf_cv_fit_gpu), and
 * translates FactorNet::gpu::CVNMFResult → FactorNet::NMFResult<Scalar>.
 *
 * The existing GPU CV NMF implementation handles all the complexity:
 *   - Fused delta+NNLS kernels for per-column modified Gram
 *   - GPU-side W-update deltas via precomputed Aᵀ
 *   - Gram-trick loss (mask_zeros=false) or direct sparse loss (mask_zeros=true)
 *   - Convergence via factor correlation
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

// Existing GPU CV NMF implementation
#include <FactorNet/gpu/cv_kernels.cuh>

// Unified types
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/types.hpp>

// CPU CV fallback for projective/symmetric (not yet in GPU CV kernels)
#include <FactorNet/nmf/fit_cv.hpp>

// Shared algorithm variant dispatch
#include <FactorNet/nmf/variant_helpers.hpp>

// RNG for initialization
#include <FactorNet/rng/rng.hpp>

// Lanczos SVD initialization (needs fit_unified.hpp for detail::initialize_lanczos)
#include <FactorNet/nmf/fit_cpu.hpp>

// GPU context guard
#include <FactorNet/primitives/gpu/context.cuh>

#include <Eigen/Dense>
#include <vector>

namespace FactorNet {
namespace nmf {


/**
 * @brief GPU cross-validation NMF with unified types
 *
 * Adapts the unified NMFConfig to the existing GPU CV config and runs
 * the production GPU CV NMF code. Returns results in unified NMFResult.
 *
 * @tparam Scalar float or double
 * @param A_csc_ptr  CSC column pointers (n+1, host)
 * @param A_row_idx  CSC row indices (nnz, host)
 * @param A_values   CSC values (nnz, host)
 * @param m          Rows of A
 * @param n          Columns of A
 * @param nnz        Number of nonzeros
 * @param config     Unified NMF configuration (must have is_cv() == true)
 * @param W_init     Optional W initialization (m × k). nullptr = random.
 * @param H_init     Optional H initialization (k × n). nullptr = random.
 * @return           NMFResult with W (m×k), d (k), H (k×n), test_loss
 */
template<typename Scalar>
::FactorNet::NMFResult<Scalar> nmf_cv_fit_gpu(
    const int* A_csc_ptr,
    const int* A_row_idx,
    const Scalar* A_values,
    int m, int n, int nnz,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    using namespace FactorNet::gpu;

    const int k = config.rank;

    // ------------------------------------------------------------------
    // GPU CV uses batched Cholesky (cuSOLVER) internally for all NNLS
    // solves, regardless of solver_mode. No CPU fallback needed.
    // Projective and symmetric are handled natively in cv_kernels.cuh.
    // ------------------------------------------------------------------

    // GPU context and guard
    GPUContext ctx;
    primitives::gpu_detail::GPUGuard guard;

    // ------------------------------------------------------------------
    // Translate unified config → GPU CV config
    // ------------------------------------------------------------------

    CVNMFConfig cv_config;
    cv_config.k = k;
    cv_config.max_iter = config.max_iter;
    cv_config.tol = static_cast<double>(config.tol);
    cv_config.H.L1 = static_cast<double>(config.H.L1);
    cv_config.W.L1 = static_cast<double>(config.W.L1);
    cv_config.H.L2 = static_cast<double>(config.H.L2);
    cv_config.W.L2 = static_cast<double>(config.W.L2);
    cv_config.H.L21 = static_cast<double>(config.H.L21);
    cv_config.W.L21 = static_cast<double>(config.W.L21);
    cv_config.ortho_H = static_cast<double>(config.H.angular);
    cv_config.ortho_W = static_cast<double>(config.W.angular);
    cv_config.H.upper_bound = static_cast<double>(config.H.upper_bound);
    cv_config.W.upper_bound = static_cast<double>(config.W.upper_bound);
    cv_config.W.nonneg = config.W.nonneg;
    cv_config.H.nonneg = config.H.nonneg;
    cv_config.holdout_fraction = static_cast<double>(config.holdout_fraction);
    cv_config.cv_seed = static_cast<uint64_t>(config.effective_cv_seed());
    cv_config.mask_zeros = config.mask_zeros;
    cv_config.verbose = config.verbose;
    cv_config.cd_maxit = config.cd_max_iter;
    cv_config.threads = config.threads;

    // Loss function (IRLS for non-MSE)
    cv_config.loss_type = static_cast<int>(config.loss.type);
    cv_config.huber_delta = static_cast<double>(config.loss.huber_delta);
    cv_config.robust_delta = static_cast<double>(config.loss.robust_delta);
    cv_config.irls_max_iter = config.irls_max_iter;
    cv_config.irls_tol = static_cast<double>(config.irls_tol);
    cv_config.cd_tol = static_cast<double>(config.cd_tol);

    // Graph regularization pointers (void* → type-erased for the CVNMFConfig)
    cv_config.graph_W_data = static_cast<const void*>(config.W.graph);
    cv_config.graph_H_data = static_cast<const void*>(config.H.graph);
    cv_config.W.graph_lambda = config.W.graph ? static_cast<double>(config.W.graph_lambda) : 0.0;
    cv_config.H.graph_lambda = config.H.graph ? static_cast<double>(config.H.graph_lambda) : 0.0;

    // User-supplied mask (type-erased pointer to Eigen::SparseMatrix<Scalar>)
    cv_config.mask_data = static_cast<const void*>(config.mask);

    // Normalization type: L1=0, L2=1, None=2
    cv_config.norm_type_int = (config.norm_type == NormType::L1) ? 0
                            : (config.norm_type == NormType::L2) ? 1 : 2;

    // Algorithm variants
    cv_config.projective = config.projective;
    cv_config.symmetric = config.symmetric;

    // ------------------------------------------------------------------
    // Initialize factors on host
    // ------------------------------------------------------------------

    // W stored as k×m (transposed) for GPU
    std::vector<Scalar> W_host(static_cast<size_t>(k) * m);
    std::vector<Scalar> H_host(static_cast<size_t>(k) * n);
    std::vector<Scalar> d_host(k, Scalar(1));

    if (W_init && H_init) {
        // Transpose user's m×k W to k×m column-major
        for (int i = 0; i < m; ++i)
            for (int f = 0; f < k; ++f)
                W_host[f + static_cast<size_t>(i) * k] = (*W_init)(i, f);

        // H is already k×n, copy column-major
        for (int j = 0; j < n; ++j)
            for (int f = 0; f < k; ++f)
                H_host[f + static_cast<size_t>(j) * k] = (*H_init)(f, j);
    } else if (config.init_mode == 1) {
        // Lanczos SVD initialization on host using CSC arrays
        using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
        Eigen::Map<const SpMat> h_A_map(m, n, nnz, A_csc_ptr, A_row_idx, A_values);
        DenseMatrix<Scalar> h_W_T, h_H;
        DenseVector<Scalar> h_d;
        detail::initialize_lanczos(h_W_T, h_H, h_d, h_A_map, k, m, n,
                                   config.seed, config.threads, config.verbose);
        std::copy(h_W_T.data(), h_W_T.data() + k * m, W_host.begin());
        std::copy(h_H.data(), h_H.data() + k * n, H_host.begin());
        for (int f = 0; f < k; ++f) d_host[f] = h_d(f);
    } else if (config.init_mode == 2) {
        // IRLBA SVD initialization on host using CSC arrays
        using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
        Eigen::Map<const SpMat> h_A_map(m, n, nnz, A_csc_ptr, A_row_idx, A_values);
        DenseMatrix<Scalar> h_W_T, h_H;
        DenseVector<Scalar> h_d;
        detail::initialize_irlba(h_W_T, h_H, h_d, h_A_map, k, m, n,
                                 config.seed, config.threads, config.verbose);
        std::copy(h_W_T.data(), h_W_T.data() + k * m, W_host.begin());
        std::copy(h_H.data(), h_H.data() + k * n, H_host.begin());
        for (int f = 0; f < k; ++f) d_host[f] = h_d(f);
    } else {
        // Random initialization
        rng::SplitMix64 rng(config.seed);
        DenseMatrix<Scalar> h_W_T(k, m), h_H(k, n);
        rng.fill_uniform(h_W_T.data(), k, m);
        rng.fill_uniform(h_H.data(), k, n);
        std::copy(h_W_T.data(), h_W_T.data() + k * m, W_host.begin());
        std::copy(h_H.data(), h_H.data() + k * n, H_host.begin());
    }

    // ------------------------------------------------------------------
    // Convert CSC data to vectors (GPU CV expects std::vector)
    // ------------------------------------------------------------------

    std::vector<int> col_ptr(A_csc_ptr, A_csc_ptr + n + 1);
    std::vector<int> row_idx(A_row_idx, A_row_idx + nnz);
    std::vector<Scalar> values(A_values, A_values + nnz);

    // ------------------------------------------------------------------
    // Call production GPU CV NMF
    // ------------------------------------------------------------------

    CVNMFResult cv_result = FactorNet::gpu::nmf_cv_fit_gpu<Scalar>(
        ctx, col_ptr, row_idx, values, m, n,
        W_host.data(), H_host.data(), d_host.data(),
        cv_config);

    // ------------------------------------------------------------------
    // Translate GPU CV result → unified NMFResult
    // ------------------------------------------------------------------

    ::FactorNet::NMFResult<Scalar> result;

    // W: k×m → m×k
    result.W.resize(m, k);
    for (int i = 0; i < m; ++i)
        for (int f = 0; f < k; ++f)
            result.W(i, f) = W_host[f + static_cast<size_t>(i) * k];

    // H: k×n (same layout)
    result.H.resize(k, n);
    for (int j = 0; j < n; ++j)
        for (int f = 0; f < k; ++f)
            result.H(f, j) = H_host[f + static_cast<size_t>(j) * k];

    // d
    result.d = Eigen::Map<DenseVector<Scalar>>(d_host.data(), k);

    // Convergence info
    result.iterations = cv_result.iterations;
    result.converged = cv_result.converged;
    result.train_loss = static_cast<Scalar>(cv_result.train_loss);
    result.test_loss = static_cast<Scalar>(cv_result.test_loss);
    result.best_test_loss = static_cast<Scalar>(cv_result.best_test_loss);
    result.best_iter = cv_result.best_iter;

    // Loss history
    if (config.track_loss_history) {
        result.loss_history.reserve(cv_result.train_history.size());
        result.test_loss_history.reserve(cv_result.test_history.size());
        for (double v : cv_result.train_history)
            result.loss_history.push_back(static_cast<Scalar>(v));
        for (double v : cv_result.test_history)
            result.test_loss_history.push_back(static_cast<Scalar>(v));
    }

    if (config.sort_model)
        result.sort();

    // Replay per-iteration callback from stored history
    if (config.on_iteration) {
        const size_t nhist = std::min(cv_result.train_history.size(),
                                      cv_result.test_history.size());
        for (size_t i = 0; i < nhist; ++i)
            config.on_iteration(static_cast<int>(i + 1),
                static_cast<Scalar>(cv_result.train_history[i]),
                static_cast<Scalar>(cv_result.test_history[i]));
    }

    return result;
}


}  // namespace nmf
}  // namespace FactorNet
