#pragma once
/**
 * @file gpu_features.cuh
 * @brief GPU-native L21 and angular regularization kernels for NMF.
 *
 * Key insight: L21 and angular need the FACTOR being regularized (not G).
 * G is the Gram of the OTHER factor. We compute overlap = Factor · Factor^T
 * via cuBLAS GEMM (tiny k×k operation), then:
 *   - L21: G[i,i] += λ / sqrt(overlap[i,i])   (overlap[i,i] = ||row_i||²)
 *   - Angular: G += λ * normalize(overlap)     (cosine similarity penalty)
 *
 * Graph regularization still requires host fallback (needs large Laplacian).
 */

#include <FactorNet/gpu/context.cuh>
#include <FactorNet/gpu/types.cuh>
#include <FactorNet/gpu/gram.cuh>

namespace FactorNet {
namespace nmf {
namespace gpu_features {

// ========================================================================
// L21 group sparsity: G[i,i] += λ / ||factor_row_i||₂
// ========================================================================

template<typename Scalar>
__global__ void apply_L21_from_norms_kernel(
    Scalar* __restrict__ G,                // k×k column-major
    const Scalar* __restrict__ norms,      // k: ||factor_row_i||₂
    int k, Scalar lambda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (norms[i] > Scalar(1e-10)) {
        G[i + static_cast<size_t>(i) * k] += lambda / norms[i];
    }
}

// ========================================================================
// Angular (orthogonality) penalty:
//   overlap = factor · factor^T   (the factor being updated, NOT G)
//   normalize overlap to cosine similarity
//   G += λ * normalized_overlap
// ========================================================================

// Normalize overlap in-place to cosine similarity and scale by lambda.
// Full cosine matrix (including diagonal) is kept to preserve PSD.
template<typename Scalar>
__global__ void normalize_and_scale_overlap_kernel(
    Scalar* __restrict__ overlap,          // k×k column-major
    const Scalar* __restrict__ diag_norms, // k: sqrt(overlap[i,i])
    int k, Scalar lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * k) return;
    int col = idx / k;
    int row = idx % k;
    Scalar denom = diag_norms[row] * diag_norms[col];
    if (denom > Scalar(1e-20)) {
        overlap[idx] = lambda * overlap[idx] / denom;
    } else {
        overlap[idx] = Scalar(0);
    }
}

// Add overlap to G: G[idx] += overlap[idx]
template<typename Scalar>
__global__ void add_matrix_kernel(
    Scalar* __restrict__ G,
    const Scalar* __restrict__ overlap,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    G[idx] += overlap[idx];
}

// Extract diagonal square roots
template<typename Scalar>
__global__ void extract_diag_sqrt_kernel(
    const Scalar* __restrict__ M,  // k×k column-major
    Scalar* __restrict__ out,      // k
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    Scalar diag = M[i + static_cast<size_t>(i) * k];
    out[i] = (diag > Scalar(0)) ? sqrt(diag) : Scalar(0);
}

// ========================================================================
// Dispatchers
// ========================================================================

template<typename Scalar>
inline void apply_L21_gpu(
    ::FactorNet::gpu::GPUContext& ctx,
    ::FactorNet::gpu::DenseMatrixGPU<Scalar>& d_G,
    const ::FactorNet::gpu::DenseMatrixGPU<Scalar>& d_factor,
    Scalar lambda)
{
    if (lambda <= 0) return;
    int k = d_G.rows;

    // Compute factor overlap diagonal to get correct row norms
    // (G is from the OTHER factor, not the one being regularized)
    ::FactorNet::gpu::DenseMatrixGPU<Scalar> d_overlap(k, k);
    ::FactorNet::gpu::compute_gram_gpu(ctx, d_factor, d_overlap);

    // Extract sqrt(diag(overlap)) = factor row norms
    Scalar* d_norms = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_norms, k * sizeof(Scalar), ctx.stream));
    {
        int threads = std::min(k, 256);
        int blocks = (k + threads - 1) / threads;
        extract_diag_sqrt_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_overlap.data.get(), d_norms, k);
    }

    // Apply L21: G[i,i] += lambda / norm[i]
    {
        int threads = std::min(k, 256);
        int blocks = (k + threads - 1) / threads;
        apply_L21_from_norms_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_G.data.get(), d_norms, k, lambda);
    }

    CUDA_CHECK(cudaFreeAsync(d_norms, ctx.stream));
}

template<typename Scalar>
inline void apply_angular_gpu(
    ::FactorNet::gpu::GPUContext& ctx,
    ::FactorNet::gpu::DenseMatrixGPU<Scalar>& d_G,
    const ::FactorNet::gpu::DenseMatrixGPU<Scalar>& d_factor,
    Scalar lambda)
{
    if (lambda <= 0) return;
    int k = d_G.rows;
    int kk = k * k;

    // 1. Compute overlap = factor · factor^T (k×k) via cuBLAS GEMM
    ::FactorNet::gpu::DenseMatrixGPU<Scalar> d_overlap(k, k);
    ::FactorNet::gpu::compute_gram_gpu(ctx, d_factor, d_overlap);

    // 2. Extract sqrt(diag(overlap)) for normalization
    Scalar* d_norms = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_norms, k * sizeof(Scalar), ctx.stream));
    {
        int threads = std::min(k, 256);
        int blocks = (k + threads - 1) / threads;
        extract_diag_sqrt_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_overlap.data.get(), d_norms, k);
    }

    // 3. Normalize overlap to cosine similarity and scale by lambda
    {
        int threads = std::min(kk, 256);
        int blocks = (kk + threads - 1) / threads;
        normalize_and_scale_overlap_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_overlap.data.get(), d_norms, k, lambda);
    }

    // 4. G += lambda * normalized_overlap
    {
        int threads = std::min(kk, 256);
        int blocks = (kk + threads - 1) / threads;
        add_matrix_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_G.data.get(), d_overlap.data.get(), kk);
    }

    CUDA_CHECK(cudaFreeAsync(d_norms, ctx.stream));
    // d_overlap auto-freed by destructor
}

}  // namespace gpu_features
}  // namespace nmf
}  // namespace FactorNet
