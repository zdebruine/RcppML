#pragma once
/**
 * @file gpu_bridge_common.cuh
 * @brief Shared includes, types, and helpers for the GPU bridge.
 *
 * Included by gpu_bridge_{cluster,nmf,svd,utils}.cu.
 */


// GPU primitives (alive headers only — dead dispatch layer removed 2026-02)
#include "../inst/include/FactorNet/gpu/bipartition.cuh"
#include "../inst/include/FactorNet/gpu/dclust.cuh"
#include "../inst/include/FactorNet/nmf/fit_gpu.cuh"
#include "../inst/include/FactorNet/nmf/fit_gpu_dense.cuh"
#include "../inst/include/FactorNet/nmf/fit_cv_gpu.cuh"
#include "../inst/include/FactorNet/svd/deflation_gpu.cuh"
#include "../inst/include/FactorNet/svd/randomized_gpu.cuh"
#include "../inst/include/FactorNet/svd/irlba_gpu.cuh"
#include "../inst/include/FactorNet/svd/lanczos_gpu.cuh"
#include "../inst/include/FactorNet/svd/krylov_gpu.cuh"
#include <cstring>
#include <cstdio>
#include <numeric>
#include <FactorNet/core/logging.hpp>

using namespace FactorNet::gpu;

// ============================================================================
// GPU detection (replaces deleted gpu_multi.cuh)
// ============================================================================

struct GPUDeviceInfo {
    int device_id;
    size_t total_memory;
    size_t free_memory;
    int compute_major;
    int compute_minor;
};

inline std::vector<GPUDeviceInfo> detect_gpus(int min_cc_major = 7, int min_cc_minor = 0) {
    std::vector<GPUDeviceInfo> result;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return result;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        // Skip GPUs below minimum compute capability
        if (prop.major < min_cc_major ||
            (prop.major == min_cc_major && prop.minor < min_cc_minor))
            continue;
        // Smoke test: try a small allocation
        cudaSetDevice(i);
        void* test_ptr = nullptr;
        if (cudaMalloc(&test_ptr, 1024) != cudaSuccess) continue;
        cudaFree(test_ptr);
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        result.push_back({i, total_mem, free_mem, prop.major, prop.minor});
    }
    if (!result.empty()) cudaSetDevice(result[0].device_id);
    return result;
}
// ============================================================================
// C-callable interface (no Rcpp dependency — pure C ABI)
// All scalar parameters are pointers per R .C() convention.
// ============================================================================


// ============================================================================
// Helper: reconstruct Eigen::SparseMatrix from raw CSC arrays passed via .C()
// Values arrive as double* (R's native type); cast to Scalar type.
// Returns empty matrix if nnz <= 0.
// ============================================================================

template<typename Scalar>
static Eigen::SparseMatrix<Scalar> sparse_from_csc(
    int rows, int cols, int nnz,
    const int* outer_ptr, const int* inner_idx, const double* values)
{
    if (nnz <= 0) return Eigen::SparseMatrix<Scalar>();
    // Build double-precision matrix from raw CSC arrays
    Eigen::Map<const Eigen::SparseMatrix<double>> map_d(
        rows, cols, nnz, outer_ptr, inner_idx, values);
    Eigen::SparseMatrix<double> mat_d(map_d);
    // Cast to requested precision (no-op for double, converts for float)
    if constexpr (std::is_same_v<Scalar, double>) {
        return mat_d;
    } else {
        return mat_d.template cast<Scalar>();
    }
}

// ==========================================================================
// Unified architecture entry points (Phase 3)
//
// These call the unified GPU NMF which supports ALL features:
//   L1, L2, L21, ortho, upper_bound, graph_reg
// and uses unified NMFConfig<Scalar> / NMFResult<Scalar> types.
//
// Output arrays (W, H, d) are updated in-place. Additional result
// fields are returned via separate output pointers.
//
// Graph regularization and obs_mask sparse matrices are passed as
// raw CSC arrays (p, i, x) through the .C() interface. They are
// reconstructed into Eigen::SparseMatrix objects and set on the config.
// The GPU kernel code then uploads via SparseMatrixGPU<Scalar>.
// ==========================================================================

