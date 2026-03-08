// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/gram.cuh
 * @brief GPU Gram matrix primitive — wraps existing cuBLAS GEMM implementation
 *
 * Template specialization of primitives::gram<GPU> that delegates to
 * FactorNet::gpu::compute_gram_gpu() (cuBLAS DGEMM/SGEMM).
 *
 * This operates on DenseMatrixGPU<Scalar> (device-resident).
 * The unified algorithm layer manages host↔device transfers.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/gram.cuh>
#include <FactorNet/core/resource_tags.hpp>

namespace FactorNet {
namespace primitives {
namespace gpu_ops {

/**
 * @brief GPU Gram: G = H · Hᵀ via cuBLAS GEMM
 *
 * @param ctx  GPU context with cuBLAS handle
 * @param H    Factor matrix (k × n) on device
 * @param G    Output Gram matrix (k × k) on device
 */
template<typename Scalar>
inline void gram_gpu(FactorNet::gpu::GPUContext& ctx,
                     const FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                     FactorNet::gpu::DenseMatrixGPU<Scalar>& G) {
    FactorNet::gpu::compute_gram_gpu(ctx, H, G);
}

/**
 * @brief GPU Gram with L2 regularization: G = H · Hᵀ + (eps + L2) · I
 *
 * @param ctx  GPU context
 * @param H    Factor matrix (k × n) on device
 * @param G    Output Gram matrix (k × k) on device
 * @param L2   L2 regularization strength
 * @param eps  Numerical stabilization epsilon
 */
template<typename Scalar>
inline void gram_regularized_gpu(FactorNet::gpu::GPUContext& ctx,
                                 const FactorNet::gpu::DenseMatrixGPU<Scalar>& H,
                                 FactorNet::gpu::DenseMatrixGPU<Scalar>& G,
                                 Scalar L2 = 0,
                                 Scalar eps = 1e-15) {
    FactorNet::gpu::compute_gram_regularized_gpu(ctx, H, G, L2, eps);
}

}  // namespace gpu_ops
}  // namespace primitives
}  // namespace FactorNet
