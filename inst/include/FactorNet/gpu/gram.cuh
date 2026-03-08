/**
 * @file gram.cuh
 * @brief GPU Gram matrix computation via cuBLAS GEMM.
 *
 * Computes G = H * H^T (k×k) where H is k×n.
 *
 * Key insight: cuBLAS SYRK has massive dispatch overhead for small k,
 * taking 0.06-3.6ms on H100 regardless of data size. cuBLAS SGEMM
 * with TF32 tensor cores is 10-200× faster (0.01-0.02ms), because
 * the GEMM code path is far more optimized in cuBLAS.
 *
 * GEMM fills the full k×k output (both triangles), eliminating the
 * need for a mirror kernel. The output is exactly symmetric since
 * G = H * H^T by construction.
 *
 * Also provides L2 regularization (diagonal addition) and epsilon stabilization.
 */

#pragma once

#include "types.cuh"

namespace FactorNet {
namespace gpu {

// ---------------------------------------------------------------------------
// Kernel: add scalar to diagonal (epsilon + L2 regularization)
// ---------------------------------------------------------------------------

inline __global__ void add_to_diagonal_kernel(double* G, int k, double val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < k) {
        G[i * k + i] += val;
    }
}

inline __global__ void add_to_diagonal_kernel_f(float* G, int k, float val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < k) {
        G[i * k + i] += val;
    }
}

// ---------------------------------------------------------------------------
// compute_gram_gpu: G = H * H^T via cuBLAS GEMM
//
//   H: k × n (column-major on device)
//   G: k × k (output, column-major on device)
//
// Uses GEMM: G = H * H^T. This is 10-200× faster than SYRK on H100
// because SYRK has ~0.06-3.6ms dispatch overhead while GEMM uses
// TF32 tensor cores and has a much more optimized dispatch path.
//
// GEMM fills the complete k×k output (no mirror step needed).
// ---------------------------------------------------------------------------

template<typename Scalar>
void compute_gram_gpu(const GPUContext& ctx,
                      const DenseMatrixGPU<Scalar>& H,
                      DenseMatrixGPU<Scalar>& G) {
    const int k = H.rows;
    const int n = H.cols;

    // Guard against degenerate dimensions (cuBLAS error 7)
    if (k == 0 || n == 0) return;

    const Scalar alpha = 1;
    const Scalar beta = 0;

    // G = H * H^T via GEMM: C = α * A * B^T + β * C
    // where A = H (k×n), B = H (k×n), C = G (k×k)
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, H.data.get(), k,
                                         H.data.get(), k,
                                 &beta, G.data.get(), k));
    } else {
        CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, H.data.get(), k,
                                         H.data.get(), k,
                                 &beta, G.data.get(), k));
    }
    // No mirror needed: GEMM fills the complete k×k matrix,
    // and G = H*H^T is exactly symmetric by construction.
}

// ---------------------------------------------------------------------------
// compute_gram_regularized_gpu: G = H * H^T + (eps + L2) * I
// ---------------------------------------------------------------------------

template<typename Scalar>
void compute_gram_regularized_gpu(const GPUContext& ctx,
                                  const DenseMatrixGPU<Scalar>& H,
                                  DenseMatrixGPU<Scalar>& G,
                                  Scalar L2 = 0,
                                  Scalar eps = 1e-15) {
    compute_gram_gpu(ctx, H, G);

    Scalar diag_add = eps + L2;
    if (diag_add > 0) {
        int threads = (H.rows + 255) / 256 * 256;
        if constexpr (std::is_same_v<Scalar, double>) {
            add_to_diagonal_kernel<<<1, threads, 0, ctx.stream>>>(
                G.data.get(), H.rows, diag_add);
        } else {
            add_to_diagonal_kernel_f<<<1, threads, 0, ctx.stream>>>(
                G.data.get(), H.rows, diag_add);
        }
    }
}

} // namespace gpu
} // namespace FactorNet
