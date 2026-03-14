// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit.hpp
 * @brief Single entry point for all NMF operations
 *
 * This is the ONLY function that RcppFunctions.cpp needs to call for NMF.
 * It handles:
 *   1. Resource detection (CPU/GPU)
 *   2. Resource override (config or FACTORNET_RESOURCE env var)
 *   3. Standard vs CV dispatch (based on config.is_cv())
 *   4. GPU error handling with CPU fallback
 *   5. Resource diagnostics in result
 *
 * Usage from RcppFunctions.cpp:
 *
 *   auto result = FactorNet::nmf::fit(A, config, &W_init, &H_init);
 *
 * The caller constructs NMFConfig<Scalar> from R parameters, and the
 * gateway does the rest. No sparse/dense/cv/gpu routing needed in R.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/resources.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/core/traits.hpp>

// Algorithm implementations
#include <FactorNet/nmf/fit_cpu.hpp>
#include <FactorNet/nmf/fit_cv.hpp>

// GPU-specific implementations (compiled only with nvcc)
#ifdef FACTORNET_HAS_GPU
#include <FactorNet/nmf/fit_gpu.cuh>
#include <FactorNet/nmf/fit_gpu_dense.cuh>
#include <FactorNet/nmf/fit_cv_gpu.cuh>
#include <FactorNet/nmf/fit_gpu_streaming.cuh>
#include <FactorNet/nmf/fit_chunked_gpu.cuh>
#else
// Runtime GPU dispatch via dlsym bridge (CPU-only build)
#include <FactorNet/gpu/bridge_nmf.hpp>
#endif

#include <string>

namespace FactorNet {
namespace nmf {

// ========================================================================
// Internal dispatch helpers
// ========================================================================

namespace detail {

/**
 * @brief Dispatch to the appropriate NMF implementation based on resource plan
 *
 * For now, only CPU is supported in the unified path. GPU/MPI dispatches
 * will be added in Phase 3/4, falling through to CPU until then.
 */
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> dispatch_standard(
    const MatrixType& A,
    const NMFConfig<Scalar>& config,
    ResourcePlan plan,
    const DenseMatrix<Scalar>* W_init,
    const DenseMatrix<Scalar>* H_init)
{
    // Guard: ZI requires CPU or single-GPU (dense A_imputed matrix, EM iterations)
    if (config.has_zi() && plan != ResourcePlan::CPU && plan != ResourcePlan::GPU) {
        throw std::runtime_error(
            "Zero-inflated NMF (zi_mode != 0) requires resource=\"cpu\" or resource=\"gpu\".");
    }

    switch (plan) {
#ifdef FACTORNET_HAS_GPU
        case ResourcePlan::GPU: {
            try {
                if constexpr (is_sparse_v<MatrixType>) {
                    const int m = A.rows();
                    const int n = A.cols();
                    const int nnz = A.nonZeros();
                    return nmf::nmf_fit_gpu<Scalar>(
                        A.outerIndexPtr(), A.innerIndexPtr(),
                        A.valuePtr(), m, n, nnz,
                        config, W_init, H_init);
                } else {
                    const int m = A.rows();
                    const int n = A.cols();
                    return nmf::nmf_fit_gpu_dense<Scalar>(
                        A.data(), m, n,
                        config, W_init, H_init);
                }
            } catch (const std::exception& e) {
                // GPU failed — fall back to CPU
                auto result = nmf::nmf_fit<primitives::CPU>(
                    A, config, W_init, H_init);
                result.diagnostics.plan_reason =
                    std::string("CPU fallback: GPU error: ") + e.what();
                result.diagnostics.plan_chosen = "CPU (GPU fallback)";
                return result;
            }
        }
#else
        // --- Runtime GPU dispatch via dlsym bridge (CPU-only build) ---
        case ResourcePlan::GPU: {
            try {
                if constexpr (is_sparse_v<MatrixType>) {
                    return gpu::bridge_nmf_sparse<Scalar>(
                        A, config, W_init, H_init);
                } else {
                    return gpu::bridge_nmf_dense<Scalar>(
                        A, config, W_init, H_init);
                }
            } catch (const std::exception& e) {
                // GPU bridge failed — fall back to CPU
                auto result = nmf::nmf_fit<primitives::CPU>(
                    A, config, W_init, H_init);
                result.diagnostics.plan_reason =
                    std::string("CPU fallback: GPU bridge error: ") + e.what();
                result.diagnostics.plan_chosen = "CPU (GPU bridge fallback)";
                return result;
            }
        }
#endif
        case ResourcePlan::CPU:
        default:
            return nmf::nmf_fit<primitives::CPU>(
                A, config, W_init, H_init);
    }
}

/**
 * @brief Dispatch to the CV NMF implementation based on resource plan
 */
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> dispatch_cv(
    const MatrixType& A,
    const NMFConfig<Scalar>& config,
    ResourcePlan plan,
    const DenseMatrix<Scalar>* W_init,
    const DenseMatrix<Scalar>* H_init)
{
    switch (plan) {
#ifdef FACTORNET_HAS_GPU
        case ResourcePlan::GPU: {
            try {
                if constexpr (is_sparse_v<MatrixType>) {
                    // GPU CV path: extract sparse CSC data
                    const int m = A.rows();
                    const int n = A.cols();
                    const int nnz = A.nonZeros();
                    return nmf::nmf_cv_fit_gpu<Scalar>(
                        A.outerIndexPtr(), A.innerIndexPtr(),
                        A.valuePtr(), m, n, nnz,
                        config, W_init, H_init);
                } else {
                    // Dense CV on GPU: convert to sparse CSC and dispatch
                    const int m = A.rows();
                    const int n = A.cols();
                    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int> A_sparse = A.sparseView();
                    A_sparse.makeCompressed();
                    const int nnz = A_sparse.nonZeros();
                    auto result = nmf::nmf_cv_fit_gpu<Scalar>(
                        A_sparse.outerIndexPtr(), A_sparse.innerIndexPtr(),
                        A_sparse.valuePtr(), m, n, nnz,
                        config, W_init, H_init);
                    result.diagnostics.plan_reason = "Dense → sparse conversion for GPU CV";
                    result.diagnostics.plan_chosen = "GPU (dense→sparse CV)";
                    return result;
                }
            } catch (const std::exception& e) {
                auto result = nmf::nmf_fit_cv<primitives::CPU>(
                    A, config, W_init, H_init);
                result.diagnostics.plan_reason =
                    std::string("CPU fallback: GPU error: ") + e.what();
                result.diagnostics.plan_chosen = "CPU (GPU fallback)";
                return result;
            }
        }
#else
        // --- Runtime GPU CV dispatch via dlsym bridge ---
        case ResourcePlan::GPU: {
            try {
                if constexpr (is_sparse_v<MatrixType>) {
                    return gpu::bridge_nmf_cv_sparse<Scalar>(
                        A, config, W_init, H_init);
                } else {
                    // Dense CV: convert to sparse for the bridge
                    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int> A_sparse = A.sparseView();
                    A_sparse.makeCompressed();
                    return gpu::bridge_nmf_cv_sparse<Scalar>(
                        A_sparse, config, W_init, H_init);
                }
            } catch (const std::exception& e) {
                auto result = nmf::nmf_fit_cv<primitives::CPU>(
                    A, config, W_init, H_init);
                result.diagnostics.plan_reason =
                    std::string("CPU fallback: GPU bridge error: ") + e.what();
                result.diagnostics.plan_chosen = "CPU (GPU bridge fallback)";
                return result;
            }
        }
#endif
        case ResourcePlan::CPU:
        default:
            return nmf::nmf_fit_cv<primitives::CPU>(
                A, config, W_init, H_init);
    }
}

}  // namespace detail

// ========================================================================
// Public gateway function
// ========================================================================

/**
 * @brief THE single entry point for all NMF operations
 *
 * Detects resources, selects plan, dispatches to standard or CV NMF.
 * Handles GPU errors with graceful CPU fallback.
 *
 * @tparam Scalar     float or double
 * @tparam MatrixType SparseMatrix<Scalar>, MappedSparseMatrix<Scalar>, or DenseMatrix<Scalar>
 *
 * @param A       Data matrix (m × n)
 * @param config  Complete NMF configuration
 * @param W_init  Optional W initialization (m × k). nullptr = random.
 * @param H_init  Optional H initialization (k × n). nullptr = random.
 * @return        NMFResult with factors, loss, convergence, diagnostics
 */
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> nmf(
    const MatrixType& A,
    const NMFConfig<Scalar>& config_in,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    // ------------------------------------------------------------------
    // 0. Pre-compute target Gram for PROJ_ADV (negative target_lambda)
    // ------------------------------------------------------------------
    // PROJ_ADV subtracts |λ| * T*T^T/n from the Gram matrix each iteration.
    // Pre-computing T*T^T/n once avoids O(k²·n) per iteration.
    // We make a mutable copy of the config only when needed.
    NMFConfig<Scalar> config_mut;
    const bool need_proj_adv_H = config_in.H.target && config_in.H.target_lambda < 0;
    const bool need_proj_adv_W = config_in.W.target && config_in.W.target_lambda < 0;

    if (need_proj_adv_H || need_proj_adv_W) {
        config_mut = config_in;
        if (need_proj_adv_H) {
            const auto& T = *config_mut.H.target;
            config_mut.H.target_gram = (T * T.transpose())
                                     / static_cast<Scalar>(T.cols());
        }
        if (need_proj_adv_W) {
            const auto& T = *config_mut.W.target;
            config_mut.W.target_gram = (T * T.transpose())
                                     / static_cast<Scalar>(T.cols());
        }
    }
    const NMFConfig<Scalar>& config = (need_proj_adv_H || need_proj_adv_W)
                                    ? config_mut : config_in;

    // ------------------------------------------------------------------
    // 1. Detect resources
    // ------------------------------------------------------------------
    Resources resources = Resources::detect();

    // ------------------------------------------------------------------
    // 2. Determine resource plan
    // ------------------------------------------------------------------
    ResourcePlan plan;

    // Check environment variable override first
    std::string env_override = get_env_resource_override();
    if (!env_override.empty() && env_override != "auto") {
        plan = parse_resource_override(env_override, resources);
    }
    // Then check config override
    else if (config.resource_override != "auto") {
        plan = parse_resource_override(config.resource_override, resources);
    }
    // Auto selection
    else {
        plan = select_resources(A, config, resources);
    }

    // ------------------------------------------------------------------
    // 3. Build diagnostics
    // ------------------------------------------------------------------
    ResourceDiagnostics diag = build_diagnostics(resources, plan);

    // ------------------------------------------------------------------
    // 4. Dispatch to standard or CV implementation
    // ------------------------------------------------------------------
    NMFResult<Scalar> result;

    if (config.is_cv()) {
        // CV NMF supports both sparse and dense matrices.
        // GPU CV path requires sparse input; dense falls back to CPU.
        if constexpr (is_sparse_v<MatrixType>) {
            result = detail::dispatch_cv(A, config, plan, W_init, H_init);
        } else {
            // Dense CV: always CPU (GPU CV requires sparse CSC)
            result = nmf::nmf_fit_cv<primitives::CPU>(
                A, config, W_init, H_init);
        }
    } else {
        result = detail::dispatch_standard(A, config, plan, W_init, H_init);
    }

    // Merge diagnostics (algorithm may have overwritten some fields on fallback)
    if (result.diagnostics.plan_chosen.empty()) {
        result.diagnostics = diag;
    } else {
        // Keep algorithm's overridden diagnostics but fill in detection info
        result.diagnostics.cpu_cores_detected = diag.cpu_cores_detected;
        result.diagnostics.gpus_detected = diag.gpus_detected;
        result.diagnostics.gpu_memory_available = diag.gpu_memory_available;

    }

    return result;
}

} // namespace nmf
}  // namespace FactorNet

