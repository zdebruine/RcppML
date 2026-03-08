// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/nnls_batch.cuh
 * @brief GPU NNLS batch solve primitive — wraps existing CUDA kernels
 *
 * Delegates to FactorNet::gpu::nnls_batch_gpu() which auto-selects between
 * warp-batched (k≤32), parallel gradient (k=33-63), and shared-memory/global
 * memory paths (k≥64).
 *
 * Also wraps the CV variant (per-column modified Gram) and scaling operations.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/nnls.cuh>
#include <FactorNet/gpu/fused_cv.cuh>
#include <FactorNet/core/resource_tags.hpp>

namespace FactorNet {
namespace primitives {
namespace gpu_ops {

/**
 * @brief GPU batch NNLS: solve G·x = b for each column, x ≥ 0
 *
 * B is overwritten with the solution. Auto-selects kernel variant
 * based on rank k (warp-batched, parallel gradient, or global memory).
 *
 * @param ctx        GPU context
 * @param G          Gram matrix (k × k) on device
 * @param B          RHS / output (k × n) on device — overwritten with solution
 * @param cd_maxit   Coordinate descent iterations per column
 */
template<typename Scalar>
inline void nnls_batch_gpu(FactorNet::gpu::GPUContext& ctx,
                           const FactorNet::gpu::DenseMatrixGPU<Scalar>& G,
                           FactorNet::gpu::DenseMatrixGPU<Scalar>& B,
                           int cd_maxit = 10,
                           bool nonneg = true) {
    FactorNet::gpu::nnls_batch_gpu(ctx, G, B, cd_maxit, nonneg);
}

/**
 * @brief GPU batch CV NNLS: per-column modified Gram
 *
 * Each column j solves with G_j = G_full - delta_G[:,j],
 * b_j = B_full_j - delta_b_j.
 *
 * @param ctx       GPU context
 * @param G_full    Full Gram matrix (k × k) on device
 * @param delta_G   Per-column Gram corrections (k × k × n) on device
 * @param B_full    Full RHS (k × n) on device
 * @param delta_b   Per-column RHS corrections (k × n) on device
 * @param H         Output (k × n) on device
 * @param cd_maxit  CD iterations
 */
template<typename Scalar>
inline void nnls_cv_batch_gpu(FactorNet::gpu::GPUContext& ctx,
                              const FactorNet::gpu::DenseMatrixGPU<Scalar>& G_full,
                              const FactorNet::gpu::DenseMatrixGPU<Scalar>& delta_G,
                              const FactorNet::gpu::DenseMatrixGPU<Scalar>& B_full,
                              const FactorNet::gpu::DenseMatrixGPU<Scalar>& delta_b,
                              FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                              int cd_maxit = 10,
                              bool nonneg = true) {
    FactorNet::gpu::nnls_cv_batch_gpu(ctx, G_full, delta_G, B_full, delta_b, H, cd_maxit, nonneg);
}

/**
 * @brief GPU row scaling: extract row norms, normalize
 *
 * After NNLS: d[i] = ||M[i,:]||, M[i,:] /= d[i]
 *
 * @param ctx  GPU context
 * @param M    Factor matrix (k × n) on device
 * @param d    Output diagonal (k) on device
 */
template<typename Scalar>
inline void scale_rows_gpu(FactorNet::gpu::GPUContext& ctx,
                           FactorNet::gpu::DenseMatrixGPU<Scalar>& M,
                           FactorNet::gpu::DeviceMemory<Scalar>& d) {
    FactorNet::gpu::scale_rows_opt_gpu(ctx, M, d);
}

/**
 * @brief GPU L1 penalty: subtract L1 from all elements of B, clamp to 0
 *
 * @param ctx        GPU context
 * @param B          Matrix on device — modified in-place
 * @param l1_penalty L1 penalty value
 */
template<typename Scalar>
inline void apply_l1_gpu(FactorNet::gpu::GPUContext& ctx,
                         FactorNet::gpu::DenseMatrixGPU<Scalar>& B,
                         Scalar l1_penalty) {
    FactorNet::gpu::apply_l1_gpu(ctx, B, l1_penalty);
}

/**
 * @brief GPU absorb diagonal: M[f,j] *= d[f]
 *
 * @param ctx  GPU context
 * @param M    Factor matrix on device
 * @param d    Diagonal vector on device
 */
template<typename Scalar>
inline void absorb_d_gpu(FactorNet::gpu::GPUContext& ctx,
                         FactorNet::gpu::DenseMatrixGPU<Scalar>& M,
                         const FactorNet::gpu::DeviceMemory<Scalar>& d) {
    FactorNet::gpu::absorb_d_gpu(ctx, M, d);
}

}  // namespace gpu_ops
}  // namespace primitives
}  // namespace FactorNet
