// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file resources.hpp
 * @brief Auto-detection of compute resources (CPUs, GPUs)
 *
 * The user never selects a backend. Resources::detect() queries the
 * environment once at entry and select_resources() picks the best plan.
 *
 * GPU detection includes a driver health check (smoke test allocation)
 * to gracefully handle broken drivers or GPUs reserved by other jobs.
 *
 * An internal "resource_override" mechanism exists for debugging and
 * benchmarking (see NMFConfig::resource_override and FACTORNET_RESOURCE).
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>

// Runtime GPU detection via dlsym (CPU-only build)
#ifndef FACTORNET_HAS_GPU
#include <FactorNet/gpu/loader.hpp>
#endif

#include <string>
#include <cstdlib>    // std::getenv

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {

// ========================================================================
// Resource plan selection
// ========================================================================

/// Which compute path to use
enum class ResourcePlan {
    CPU,         ///< Single-node CPU (OpenMP threads)
    GPU          ///< Single GPU
};

/// Convert plan to human-readable string
inline const char* plan_name(ResourcePlan p) {
    switch (p) {
        case ResourcePlan::CPU:       return "CPU";
        case ResourcePlan::GPU:       return "GPU";
    }
    return "UNKNOWN";
}

// ========================================================================
// Resource detection
// ========================================================================

/**
 * @brief Detected compute resources — queried once per gateway call
 */
struct Resources {
    int cpu_cores = 1;          ///< Available CPU threads (OpenMP or hw)
    int gpu_count = 0;          ///< Functional GPUs (post–health-check)
    int64_t gpu_memory = 0;     ///< Total VRAM (bytes) across all GPUs

    /**
     * @brief Detect all available compute resources
     *
     * GPU detection includes a smoke-test allocation to verify the
     * driver is functional. If the driver is broken or the GPU is
     * reserved, gpu_count falls back to 0.
     */
    static Resources detect() {
        Resources r;

        // --- CPU ----------------------------------------------------------
#ifdef _OPENMP
        r.cpu_cores = omp_get_max_threads();
#else
        r.cpu_cores = 1;
#endif

        // --- GPU ----------------------------------------------------------
        // GPU detection is compiled out when FACTORNET_HAS_GPU is not defined.
        // This ensures the R package builds cleanly without CUDA Toolkit.
#ifdef FACTORNET_HAS_GPU
        {
            int count = 0;
            if (cudaGetDeviceCount(&count) == cudaSuccess) {
                // Verify each GPU with a small allocation
                for (int i = 0; i < count; ++i) {
                    cudaDeviceProp prop;
                    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                        // Smoke test
                        cudaSetDevice(i);
                        void* test_ptr = nullptr;
                        if (cudaMalloc(&test_ptr, 1024) == cudaSuccess) {
                            cudaFree(test_ptr);
                            r.gpu_memory += prop.totalGlobalMem;
                            r.gpu_count++;
                        }
                    }
                }
            }
            // Reset to device 0 if any GPUs found
            if (r.gpu_count > 0) cudaSetDevice(0);
        }
#else
        // --- Runtime GPU detection via dlsym ---
        // R calls gpu_available() which does dyn.load("RcppML_gpu.so").
        // We detect the loaded symbols via dlsym(RTLD_DEFAULT, ...).
        {
            int gpu_count_bridge = 0;
            size_t total_mem_bridge = 0;
            if (gpu::detect_gpus_via_bridge(gpu_count_bridge, total_mem_bridge)) {
                r.gpu_count = gpu_count_bridge;
                r.gpu_memory = total_mem_bridge;
            }
        }
#endif

        return r;
    }
};

// ========================================================================
// Resource plan selection
// ========================================================================

/**
 * @brief Select the best resource plan for the given problem
 *
 * @tparam Scalar     Precision type
 * @tparam MatrixType Data matrix type (sparse or dense)
 * @param  A          Data matrix (used for size heuristics)
 * @param  config     NMF configuration
 * @param  r          Detected resources
 * @return            ResourcePlan to use
 */
template<typename Scalar, typename MatrixType>
ResourcePlan select_resources(const MatrixType& A,
                              const NMFConfig<Scalar>& config,
                              const Resources& r)
{
    // Single GPU
    if (r.gpu_count >= 1)
        return ResourcePlan::GPU;

    // CPU (OpenMP handles thread count internally)
    return ResourcePlan::CPU;
}

/**
 * @brief Parse a resource override string into a ResourcePlan
 *
 * @throws InvalidArgument if override is not "auto", "cpu", or "gpu"
 */
inline ResourcePlan parse_resource_override(const std::string& override_str,
                                             const Resources& r)
{
    if (override_str == "cpu")
        return ResourcePlan::CPU;
    if (override_str == "gpu") {
        if (r.gpu_count == 0)
            throw InvalidArgument("resource_override='gpu' but no GPUs available");
        return ResourcePlan::GPU;
    }
    throw InvalidArgument("resource_override must be 'auto', 'cpu', or 'gpu'");
}

/**
 * @brief Check for FACTORNET_RESOURCE environment variable
 * @return  Override string ("auto", "cpu", "gpu") or empty string
 */
inline std::string get_env_resource_override() {
    const char* env = std::getenv("FACTORNET_RESOURCE");
    return env ? std::string(env) : std::string();
}

/**
 * @brief Build ResourceDiagnostics for a given plan
 */
inline ResourceDiagnostics build_diagnostics(const Resources& r,
                                              ResourcePlan plan)
{
    ResourceDiagnostics diag;
    diag.cpu_cores_detected = r.cpu_cores;
    diag.gpus_detected = r.gpu_count;
    diag.gpu_memory_available = r.gpu_memory;
    diag.plan_chosen = plan_name(plan);

    switch (plan) {
        case ResourcePlan::CPU:
            diag.plan_reason = "CPU selected: " + std::to_string(r.cpu_cores) + " cores";
            if (r.gpu_count == 0)
                diag.plan_reason += " (no GPUs detected)";
            break;
        case ResourcePlan::GPU:
            diag.plan_reason = "GPU selected";
            if (r.gpu_count > 1)
                diag.plan_reason += ": using 1 of " + std::to_string(r.gpu_count)
                                  + " devices";
            else
                diag.plan_reason += ": 1 device, "
                                  + std::to_string(r.gpu_memory / (1024*1024)) + " MB VRAM";
            break;
    }
    return diag;
}

}  // namespace FactorNet

