// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/rhs.cuh
 * @brief GPU RHS primitive — wraps existing sparse custom kernels and dense GEMM
 *
 * Template wrappers around FactorNet::gpu RHS functions for unified architecture.
 * Supports both sparse custom kernels and dense TF32 GEMM (adaptive selection).
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/rhs.cuh>
#include <FactorNet/core/resource_tags.hpp>

namespace FactorNet {
namespace primitives {
namespace gpu_ops {

/**
 * @brief GPU RHS forward (sparse): B = W · A  (k × n)
 *
 * Custom CUDA kernel — one thread block per column of A.
 *
 * @param ctx  GPU context
 * @param W    Factor matrix (k × m) on device
 * @param A    Sparse data matrix (m × n, CSC) on device
 * @param B    Output RHS matrix (k × n) on device
 */
template<typename Scalar>
inline void rhs_forward_sparse_gpu(FactorNet::gpu::GPUContext& ctx,
                                   const FactorNet::gpu::DenseMatrixGPU<Scalar>& W,
                                   const FactorNet::gpu::SparseMatrixGPU<Scalar>& A,
                                   FactorNet::gpu::DenseMatrixGPU<Scalar>& B) {
    FactorNet::gpu::compute_rhs_forward_gpu(ctx, W, A, B);
}

/**
 * @brief GPU RHS transpose (sparse): B = H · Aᵀ  (k × m)
 *
 * Custom CUDA kernel using atomicAdd — one block per column of A.
 *
 * @param ctx  GPU context
 * @param H    Factor matrix (k × n) on device
 * @param A    Sparse data matrix (m × n, CSC) on device
 * @param B    Output RHS matrix (k × m) on device — MUST be zeroed before call
 */
template<typename Scalar>
inline void rhs_transpose_sparse_gpu(FactorNet::gpu::GPUContext& ctx,
                                     const FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                                     const FactorNet::gpu::SparseMatrixGPU<Scalar>& A,
                                     FactorNet::gpu::DenseMatrixGPU<Scalar>& B) {
    FactorNet::gpu::compute_rhs_transpose_gpu(ctx, H, A, B);
}

/**
 * @brief GPU RHS forward (dense GEMM): B = W · A_dense  via cuBLAS
 *
 * Uses pre-materialized dense A and Aᵀ. TF32 tensor cores on Ampere+.
 *
 * @param ctx       GPU context
 * @param W         Factor (k × m) on device
 * @param dense_rhs Pre-materialized dense A and Aᵀ on device
 * @param B         Output (k × n) on device
 */
template<typename Scalar>
inline void rhs_forward_dense_gpu(FactorNet::gpu::GPUContext& ctx,
                                  const FactorNet::gpu::DenseMatrixGPU<Scalar>& W,
                                  const FactorNet::gpu::DenseRHS<Scalar>& dense_rhs,
                                  FactorNet::gpu::DenseMatrixGPU<Scalar>& B) {
    FactorNet::gpu::compute_rhs_forward_dense_gpu(ctx, W, dense_rhs, B);
}

/**
 * @brief GPU RHS transpose (dense GEMM): B = H · Aᵀ_dense  via cuBLAS
 */
template<typename Scalar>
inline void rhs_transpose_dense_gpu(FactorNet::gpu::GPUContext& ctx,
                                    const FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                                    const FactorNet::gpu::DenseRHS<Scalar>& dense_rhs,
                                    FactorNet::gpu::DenseMatrixGPU<Scalar>& B) {
    FactorNet::gpu::compute_rhs_transpose_dense_gpu(ctx, H, dense_rhs, B);
}

}  // namespace gpu_ops
}  // namespace primitives
}  // namespace FactorNet
