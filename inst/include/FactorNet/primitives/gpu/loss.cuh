// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/loss.cuh
 * @brief GPU loss computation primitive — Gram trick on device
 *
 * Wraps FactorNet::gpu::compute_total_loss_gpu() which computes:
 *   ||A - WH||² = ||A||² - 2·Σ bⱼᵀhⱼ + Σ hⱼᵀGhⱼ
 *
 * All computation on GPU. Only the 3 scalar terms are transferred to host.
 *
 * Also provides ||A||² precomputation for the Gram trick.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/loss.cuh>
#include <FactorNet/core/resource_tags.hpp>

namespace FactorNet {
namespace primitives {
namespace gpu_ops {

/**
 * @brief GPU ||A||² = sum of squared nonzero values (sparse)
 *
 * Precomputed once before the iteration loop. O(nnz) with block reduction.
 *
 * @param ctx  GPU context
 * @param A    Sparse matrix on device
 * @return     tr(AᵀA) = ||A||²_F
 */
template<typename Scalar>
inline double trace_AtA_gpu(FactorNet::gpu::GPUContext& ctx,
                            const FactorNet::gpu::SparseMatrixGPU<Scalar>& A) {
    // Use double scratch to avoid float overflow for large matrices
    FactorNet::gpu::DeviceMemory<double> scratch(1);
    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, sizeof(double), ctx.stream));

    int threads = 256;
    int blocks = std::min((A.nnz + threads - 1) / threads, 1024);
    FactorNet::gpu::sparse_sq_norm_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        A.values.get(), A.nnz, scratch.get());

    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, scratch.get(), sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);
    return result;
}

/**
 * @brief GPU total loss via Gram trick
 *
 * loss = ||A||² - 2·Σ bⱼᵀhⱼ + Σ hⱼᵀGhⱼ
 *
 * @param ctx     GPU context
 * @param A       Sparse data on device
 * @param W       Factor (k × m) on device
 * @param H       Factor (k × n) on device
 * @param G       Gram matrix (k × k) on device — G = WᵀW
 * @param B_save  Saved RHS (k × n) on device — B = WᵀA (pre-NNLS, pre-L1)
 * @param scratch Device scratch ≥ 3 scalars
 * @param m       Rows of A
 * @param n       Columns of A
 * @return        SSE = ||A - WH||²
 */
template<typename Scalar>
inline double total_loss_gpu(FactorNet::gpu::GPUContext& ctx,
                             const FactorNet::gpu::SparseMatrixGPU<Scalar>& A,
                             const FactorNet::gpu::DenseMatrixGPU<Scalar>& W,
                             const FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                             const FactorNet::gpu::DenseMatrixGPU<Scalar>& G,
                             const FactorNet::gpu::DenseMatrixGPU<Scalar>& B_save,
                             FactorNet::gpu::DeviceMemory<double>& scratch,  // double: avoids float overflow
                             int m, int n) {
    return FactorNet::gpu::compute_total_loss_gpu(ctx, A, W, H, G, B_save, scratch, m, n);
}

/**
 * @brief GPU total loss for dense NMF via Gram trick (precomputed ||A||²)
 */
template<typename Scalar>
inline double total_loss_gpu_dense(FactorNet::gpu::GPUContext& ctx,
                                   double A_sq_norm,
                                   const FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                                   const FactorNet::gpu::DenseMatrixGPU<Scalar>& G,
                                   const FactorNet::gpu::DenseMatrixGPU<Scalar>& B_save,
                                   FactorNet::gpu::DeviceMemory<double>& scratch) {
    return FactorNet::gpu::compute_total_loss_gpu_dense(ctx, A_sq_norm, H, G, B_save, scratch);
}

/**
 * @brief Precompute ||A||²_F for dense matrix on GPU
 */
template<typename Scalar>
inline double dense_sq_norm_gpu(FactorNet::gpu::GPUContext& ctx,
                                const FactorNet::gpu::DenseMatrixGPU<Scalar>& A,
                                FactorNet::gpu::DeviceMemory<double>& scratch) {
    return FactorNet::gpu::compute_dense_sq_norm_gpu(ctx, A, scratch);
}

}  // namespace gpu_ops
}  // namespace primitives
}  // namespace FactorNet
