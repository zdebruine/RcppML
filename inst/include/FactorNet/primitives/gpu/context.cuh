// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/context.cuh
 * @brief GPU context management for unified architecture
 *
 * Provides a GPUGuard RAII wrapper for GPU error checking and graceful
 * fallback, plus a managed GPUContext that is shared across all primitives
 * within a single NMF fit call.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/types.cuh>
#include <FactorNet/core/logging.hpp>
#include <stdexcept>
#include <string>

namespace FactorNet {
namespace primitives {
namespace gpu_detail {

/**
 * @brief RAII guard for GPU error checking
 *
 * Checks for asynchronous CUDA errors on scope exit.
 * If errors are detected, they are logged but NOT thrown from
 * the destructor — the gateway try/catch handles error propagation.
 */
struct GPUGuard {
    ~GPUGuard() noexcept {
        // Sync device to surface any in-flight async errors
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            FACTORNET_GPU_WARN("[RcppML] GPU async error on cleanup: %s\n",
                    cudaGetErrorString(err));
        }
    }
};

/**
 * @brief Managed GPU context for unified primitive calls
 *
 * Wraps the existing FactorNet::gpu::GPUContext with ownership semantics.
 * Created once per NMF fit call, shared across all primitive invocations
 * within that call.
 */
class ManagedGPUContext {
private:
    FactorNet::gpu::GPUContext ctx_;

public:
    ManagedGPUContext() = default;

    FactorNet::gpu::GPUContext& get() { return ctx_; }
    const FactorNet::gpu::GPUContext& get() const { return ctx_; }

    void sync() { ctx_.sync(); }
};

/**
 * @brief Estimate GPU memory needed for NMF
 *
 * @param m     Rows of A
 * @param n     Columns of A
 * @param nnz   Non-zeros in A (sparse) or m*n (dense)
 * @param k     Factorization rank
 * @return      Estimated bytes needed
 */
template<typename Scalar>
int64_t estimate_gpu_memory(int m, int n, int nnz, int k) {
    int64_t elem = sizeof(Scalar);

    // Sparse A: col_ptr + row_idx + values
    int64_t A_bytes = (n + 1) * sizeof(int) + nnz * (sizeof(int) + elem);

    // W (k×m), H (k×n), G (k×k), B (k×max(m,n)), B_save (k×n), d (k)
    int64_t factor_bytes = (
        static_cast<int64_t>(k) * m                    // W
        + static_cast<int64_t>(k) * n                  // H
        + static_cast<int64_t>(k) * k                  // G
        + static_cast<int64_t>(k) * std::max(m, n)     // B
        + static_cast<int64_t>(k) * n                  // B_save
        + k                                             // d
        + 4                                             // scratch
    ) * elem;

    return A_bytes + factor_bytes;
}

/**
 * @brief Check if a problem fits in GPU memory
 *
 * @param needed_bytes   Estimated memory requirement
 * @param safety_factor  Fraction of free memory to use (default 0.85)
 * @return               true if problem fits
 */
inline bool fits_in_gpu_memory(int64_t needed_bytes, double safety_factor = 0.85) {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
        return false;
    return needed_bytes < static_cast<int64_t>(free_mem * safety_factor);
}

}  // namespace gpu_detail
}  // namespace primitives
}  // namespace FactorNet
