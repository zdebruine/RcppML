// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit_gpu_streaming.cuh
 * @brief GPU streaming NMF for matrices exceeding VRAM.
 *
 * Thin wrapper: creates an InMemoryLoader from raw CSC arrays and
 * delegates to nmf_chunked_gpu() which handles GPU-optimised panel
 * processing with minimal PCIe traffic.
 *
 * Also provides the needs_streaming() and compute_panel_cols() helpers
 * for VRAM budget estimation.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#ifdef FACTORNET_HAS_GPU

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/io/in_memory.hpp>
#include <FactorNet/nmf/fit_chunked_gpu.cuh>

#include <cuda_runtime.h>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>

namespace FactorNet {
namespace nmf {

namespace gpu_streaming_detail {

/**
 * @brief Estimate panel size that fits in available VRAM.
 */
inline int compute_panel_cols(
    size_t avail_bytes, int m, int n, int nnz, int k, size_t scalar_size)
{
    size_t fixed = (2 * static_cast<size_t>(k) * m + k * k + k + 64) * scalar_size;
    size_t usable = static_cast<size_t>(avail_bytes * 0.8);
    if (usable <= fixed) return 1;
    size_t panel_budget = usable - fixed;

    double avg_nnz = static_cast<double>(nnz) / n;
    size_t per_col_sparse = static_cast<size_t>(avg_nnz * (sizeof(int) + scalar_size)) + sizeof(int);
    size_t per_col_dense = 2 * k * scalar_size;
    size_t per_col = per_col_sparse + per_col_dense;

    int panel_n = static_cast<int>(panel_budget / per_col);
    return std::max(1, std::min(panel_n, n));
}

/**
 * @brief Check if streaming is needed (matrix doesn't fit in VRAM).
 */
inline bool needs_streaming(int m, int n, int nnz, int k, size_t scalar_size) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t sparse_data = static_cast<size_t>(nnz) * (sizeof(int) + scalar_size)
                       + (n + 1) * sizeof(int);
    size_t factors = (static_cast<size_t>(k) * m + static_cast<size_t>(k) * n
                     + static_cast<size_t>(k) * std::max(m, n)
                     + k * k + k) * scalar_size;
    size_t total_need = sparse_data + factors;

    return total_need > static_cast<size_t>(free_mem * 0.85);
}

}  // namespace gpu_streaming_detail

/**
 * @brief GPU streaming NMF from raw CSC arrays.
 *
 * Creates an InMemoryLoader and delegates to nmf_chunked_gpu().
 * The chunk size is auto-computed from available VRAM.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
NMFResult<Scalar> nmf_fit_gpu_streaming(
    const int* A_csc_ptr,
    const int* A_row_idx,
    const Scalar* A_values,
    int m, int n, int nnz,
    const NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    // Determine chunk size from VRAM budget
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    int panel_cols = gpu_streaming_detail::compute_panel_cols(
        free_mem, m, n, nnz, config.rank, sizeof(Scalar));

    // Map host CSC arrays into an Eigen sparse matrix (zero-copy)
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
    Eigen::Map<const SpMat> A_map(m, n, nnz, A_csc_ptr, A_row_idx, A_values);

    // Create InMemoryLoader with VRAM-appropriate chunk size
    io::InMemoryLoader<Scalar> loader(A_map, panel_cols);

    // Delegate to GPU chunked NMF
    return nmf_chunked_gpu<Scalar>(loader, config);
}

}  // namespace nmf
}  // namespace FactorNet

#endif  // FACTORNET_HAS_GPU
