#pragma once
/**
 * @file nmf/gpu_alignment_guide.cuh
 * @brief GPU-native alignment guide regularization kernels for NMF.
 *
 * Implements NNLS regularization toward pre-computed targets by modifying
 * the Gram matrix G and RHS B before each NNLS solve:
 *
 *   min_{h>=0} ||Wh - a||^2 + lambda * ||M^{1/2}(h - t)||^2
 *   =>  G_reg = G + lambda*M,  b_reg = b + lambda*M*t
 *
 * Supports:
 *   - Isotropic target (M=I):       G[i,i] += lambda, B[:,j] += lambda*t[:,j]
 *   - Mahalanobis target (M=P):     G += lambda*P, B[:,j] += lambda*(P*t)[:,j]
 *   - Batch nullspace repulsion (M=VV^T): G += lambda*VV^T, no target
 *   - Combined target + nullspace
 *   - Annealed lambda with various schedules
 *
 * Targets (t) can be:
 *   - Per-cell embeddings (OAS-ZCA guided H, one target per column)
 *   - Per-class centroids (looked up by label[j])
 *   - Pre-computed and uploaded at start of NMF
 */

#include <FactorNet/gpu/context.cuh>
#include <FactorNet/gpu/types.cuh>
#include <cmath>

namespace FactorNet {
namespace nmf {
namespace alignment_guide_gpu {

using ::FactorNet::gpu::GPUContext;
using ::FactorNet::gpu::DenseMatrixGPU;
using ::FactorNet::gpu::DeviceMemory;

// ========================================================================
// Kernels
// ========================================================================

/// Add lambda to G diagonal: G[i,i] += lambda  (isotropic M=I)
template<typename Scalar>
__global__ void add_lambda_diag_kernel(
    Scalar* __restrict__ G, Scalar lambda, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) G[i + static_cast<size_t>(i) * k] += lambda;
}

/// Add per-cell target to B: B[f + j*k] += lambda * target[f + j*k]
/// Grid: (k, n_cols)
template<typename Scalar>
__global__ void add_target_B_kernel(
    Scalar* __restrict__ B,
    const Scalar* __restrict__ target,
    Scalar lambda, int k, int n_cols)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= k || j >= n_cols) return;
    size_t idx = f + static_cast<size_t>(j) * k;
    B[idx] += lambda * target[idx];
}

/// Add centroid target to B via labels: B[f + j*k] += lambda * centroids[f + label[j]*k]
/// Grid: (k, n_cols)
template<typename Scalar>
__global__ void add_centroid_target_B_kernel(
    Scalar* __restrict__ B,
    const Scalar* __restrict__ centroids,
    const int* __restrict__ labels,
    Scalar lambda, int k, int n_cols, int n_classes)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= k || j >= n_cols) return;
    int c = labels[j];
    if (c < 0 || c >= n_classes) return;
    B[f + static_cast<size_t>(j) * k] +=
        lambda * centroids[f + static_cast<size_t>(c) * k];
}

/// Add k×k matrix to G: G[idx] += lambda * M[idx]
/// Used for precision matrix (A21) and batch subspace (A22/A23)
template<typename Scalar>
__global__ void add_scaled_matrix_G_kernel(
    Scalar* __restrict__ G,
    const Scalar* __restrict__ M,
    Scalar lambda, int kk)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kk) return;
    G[idx] += lambda * M[idx];
}

/// Add pre-multiplied P*target to B: B[f + j*k] += lambda * Pt[f + j*k]
/// where Pt = P * target has been pre-computed (Mahalanobis, A21)
template<typename Scalar>
__global__ void add_prec_target_B_kernel(
    Scalar* __restrict__ B,
    const Scalar* __restrict__ Pt,
    Scalar lambda, int k, int n_cols)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= k || j >= n_cols) return;
    size_t idx = f + static_cast<size_t>(j) * k;
    B[idx] += lambda * Pt[idx];
}

// ========================================================================
// Annealing schedules
// ========================================================================

/// Compute annealed lambda given base lambda, iter, maxit, schedule type
template<typename Scalar>
inline Scalar anneal_lambda(Scalar base_lambda, int iter, int maxit, int schedule) {
    if (schedule <= 0 || maxit <= 0) return base_lambda;
    Scalar t = static_cast<Scalar>(iter) / static_cast<Scalar>(maxit);
    switch (schedule) {
        case 1:  // Cosine decay
            return base_lambda * Scalar(0.5) * (Scalar(1) + std::cos(t * Scalar(M_PI)));
        case 2:  // Linear decay
            return base_lambda * (Scalar(1) - t);
        case 3:  // Cutoff at 50%
            return (t < Scalar(0.5)) ? base_lambda : Scalar(0);
        case 4:  // Cutoff at 25%
            return (t < Scalar(0.25)) ? base_lambda : Scalar(0);
        default:
            return base_lambda;
    }
}

// ========================================================================
// Pre-computation: compute centroids from H and labels on GPU
// ========================================================================

/// Zero a buffer of given size
template<typename Scalar>
__global__ void zero_buffer_kernel(Scalar* __restrict__ buf, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) buf[idx] = Scalar(0);
}

/// Accumulate H columns into class centroids
template<typename Scalar>
__global__ void accumulate_centroids_kernel(
    const Scalar* __restrict__ H,
    const int* __restrict__ labels,
    Scalar* __restrict__ centroids,
    int k, int n, int C)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= k || j >= n) return;
    int c = labels[j];
    if (c < 0 || c >= C) return;
    atomicAdd(&centroids[f + static_cast<size_t>(c) * k],
              H[f + static_cast<size_t>(j) * k]);
}

/// Divide centroids by counts
template<typename Scalar>
__global__ void average_centroids_kernel(
    Scalar* __restrict__ centroids,
    const int* __restrict__ counts,
    int k, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * C) return;
    int c = idx / k;
    int cnt = counts[c];
    if (cnt > 0) centroids[idx] /= static_cast<Scalar>(cnt);
}

/// Compute class centroids from H and labels (entire pipeline)
template<typename Scalar>
inline void compute_centroids_gpu(
    GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& d_H,
    const DeviceMemory<int>& d_labels,
    const DeviceMemory<int>& d_counts,
    DenseMatrixGPU<Scalar>& d_centroids,
    int k, int n, int C)
{
    // Zero
    {
        int total = k * C;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        zero_buffer_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_centroids.data.get(), total);
    }
    // Accumulate
    {
        dim3 threads(32, 8);
        dim3 blocks((k + 31) / 32, (n + 7) / 8);
        accumulate_centroids_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_H.data.get(), d_labels.get(), d_centroids.data.get(), k, n, C);
    }
    // Average
    {
        int total = k * C;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        average_centroids_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_centroids.data.get(), d_counts.get(), k, C);
    }
}

// ========================================================================
// High-level dispatch: apply alignment guide to G and B
// ========================================================================

/// Device-side buffers for alignment guide (allocated once, reused per iter)
template<typename Scalar>
struct AlignmentGuideBuffers {
    DenseMatrixGPU<Scalar> d_H_target;     // k × n (per-cell) or k × C (centroids)
    DenseMatrixGPU<Scalar> d_W_target;     // k × m (W-side target)
    DenseMatrixGPU<Scalar> d_precision;    // k × k (Mahalanobis precision)
    DenseMatrixGPU<Scalar> d_prec_target;  // k × n (P * H_target, pre-computed)
    DenseMatrixGPU<Scalar> d_batch_sub;    // k × k (batch subspace VV^T)
    DenseMatrixGPU<Scalar> d_centroids;    // k × C (whitened centroids for A27)
    DeviceMemory<int> d_labels;            // n (per-sample labels)
    DeviceMemory<int> d_counts;            // C (per-class counts)
    int n_classes = 0;
    bool has_per_cell_target = false;
    bool has_centroid_target = false;
    bool has_precision = false;
    bool has_prec_target = false;
    bool has_batch_subspace = false;
    bool has_w_target = false;
};

/// Initialize alignment guide buffers from config
/// Call once before the NMF loop starts
template<typename Scalar>
inline void init_alignment_buffers(
    GPUContext& ctx,
    const typename FactorNet::NMFConfig<Scalar>::AlignmentGuideConfig& ag,
    AlignmentGuideBuffers<Scalar>& bufs,
    int k, int n, int m)
{
    int type = ag.type;
    if (type <= 0) return;

    // Upload labels if present
    if (!ag.labels.empty()) {
        bufs.n_classes = ag.n_classes;
        bufs.d_labels = DeviceMemory<int>(n);
        bufs.d_labels.upload(ag.labels.data(), n);
        bufs.d_counts = DeviceMemory<int>(ag.n_classes);
        // Compute counts on host
        std::vector<int> h_counts(ag.n_classes, 0);
        for (int j = 0; j < n; ++j) {
            int c = ag.labels[j];
            if (c >= 0 && c < ag.n_classes) h_counts[c]++;
        }
        bufs.d_counts.upload(h_counts.data(), ag.n_classes);
    }

    // Per-cell H target (A20, A21, A26)
    if (!ag.h_target.empty()) {
        int target_cols = static_cast<int>(ag.h_target.size()) / k;
        if (target_cols == n) {
            bufs.d_H_target = DenseMatrixGPU<Scalar>(k, n);
            bufs.d_H_target.upload_from(ag.h_target.data());
            bufs.has_per_cell_target = true;
        } else if (target_cols == ag.n_classes) {
            // Centroid targets (A27)
            bufs.d_centroids = DenseMatrixGPU<Scalar>(k, ag.n_classes);
            bufs.d_centroids.upload_from(ag.h_target.data());
            bufs.has_centroid_target = true;
        }
    }

    // Precision matrix (A21)
    if (!ag.precision.empty()) {
        bufs.d_precision = DenseMatrixGPU<Scalar>(k, k);
        bufs.d_precision.upload_from(ag.precision.data());
        bufs.has_precision = true;

        // Pre-compute P * H_target if per-cell target exists
        if (bufs.has_per_cell_target) {
            bufs.d_prec_target = DenseMatrixGPU<Scalar>(k, n);
            // P * H_target via cuBLAS GEMM: (k×k) × (k×n) → (k×n)
            Scalar alpha = Scalar(1), beta = Scalar(0);
            cublasStatus_t stat;
            if constexpr (std::is_same_v<Scalar, float>) {
                stat = cublasSgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    k, n, k, &alpha,
                    bufs.d_precision.data.get(), k,
                    bufs.d_H_target.data.get(), k,
                    &beta, bufs.d_prec_target.data.get(), k);
            } else {
                stat = cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    k, n, k, &alpha,
                    bufs.d_precision.data.get(), k,
                    bufs.d_H_target.data.get(), k,
                    &beta, bufs.d_prec_target.data.get(), k);
            }
            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS GEMM failed in alignment guide precision * target");
            }
            bufs.has_prec_target = true;
        }
    }

    // Batch subspace (A22, A23)
    if (!ag.batch_subspace.empty()) {
        bufs.d_batch_sub = DenseMatrixGPU<Scalar>(k, k);
        bufs.d_batch_sub.upload_from(ag.batch_subspace.data());
        bufs.has_batch_subspace = true;
    }

    // W-side target (A24, A25)
    if (!ag.w_target.empty()) {
        bufs.d_W_target = DenseMatrixGPU<Scalar>(k, m);
        bufs.d_W_target.upload_from(ag.w_target.data());
        bufs.has_w_target = true;
    }
}

/// Apply alignment guide regularization to G and B for the H-update
/// Called after classifier guides, before NNLS solve
template<typename Scalar>
inline void apply_H_alignment(
    GPUContext& ctx,
    DenseMatrixGPU<Scalar>& d_G,
    DenseMatrixGPU<Scalar>& d_B,
    const AlignmentGuideBuffers<Scalar>& bufs,
    Scalar lambda_H,
    Scalar lambda_batch,
    int k, int n)
{
    if (lambda_H <= Scalar(0) && lambda_batch <= Scalar(0)) return;

    // --- Target regularization (modifies G diagonal and B) ---
    if (lambda_H > Scalar(0)) {
        if (bufs.has_prec_target) {
            // Mahalanobis (A21): G += lambda * P, B += lambda * P*t
            {
                int kk = k * k;
                int threads = 256;
                int blocks = (kk + threads - 1) / threads;
                add_scaled_matrix_G_kernel<<<blocks, threads, 0, ctx.stream>>>(
                    d_G.data.get(), bufs.d_precision.data.get(), lambda_H, kk);
            }
            {
                dim3 threads(32, 8);
                dim3 blocks((k + 31) / 32, (n + 7) / 8);
                add_prec_target_B_kernel<<<blocks, threads, 0, ctx.stream>>>(
                    d_B.data.get(), bufs.d_prec_target.data.get(), lambda_H, k, n);
            }
        } else if (bufs.has_per_cell_target) {
            // Isotropic (A20): G[i,i] += lambda, B[:,j] += lambda * t[:,j]
            {
                int threads = 256;
                int blocks = (k + threads - 1) / threads;
                add_lambda_diag_kernel<<<blocks, threads, 0, ctx.stream>>>(
                    d_G.data.get(), lambda_H, k);
            }
            {
                dim3 threads(32, 8);
                dim3 blocks((k + 31) / 32, (n + 7) / 8);
                add_target_B_kernel<<<blocks, threads, 0, ctx.stream>>>(
                    d_B.data.get(), bufs.d_H_target.data.get(), lambda_H, k, n);
            }
        } else if (bufs.has_centroid_target) {
            // Centroid target (A27): G[i,i] += lambda, B[:,j] += lambda * centroid[label[j]]
            {
                int threads = 256;
                int blocks = (k + threads - 1) / threads;
                add_lambda_diag_kernel<<<blocks, threads, 0, ctx.stream>>>(
                    d_G.data.get(), lambda_H, k);
            }
            {
                dim3 threads(32, 8);
                dim3 blocks((k + 31) / 32, (n + 7) / 8);
                add_centroid_target_B_kernel<<<blocks, threads, 0, ctx.stream>>>(
                    d_B.data.get(), bufs.d_centroids.data.get(),
                    bufs.d_labels.get(), lambda_H, k, n, bufs.n_classes);
            }
        }
    }

    // --- Batch nullspace repulsion (A22, A23): G += lambda_batch * VV^T ---
    if (lambda_batch > Scalar(0) && bufs.has_batch_subspace) {
        int kk = k * k;
        int threads = 256;
        int blocks = (kk + threads - 1) / threads;
        add_scaled_matrix_G_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_G.data.get(), bufs.d_batch_sub.data.get(), lambda_batch, kk);
    }
}

/// Apply alignment guide regularization to G and B for the W-update
/// Called after graph reg, before NNLS solve
template<typename Scalar>
inline void apply_W_alignment(
    GPUContext& ctx,
    DenseMatrixGPU<Scalar>& d_G,
    DenseMatrixGPU<Scalar>& d_B,
    const AlignmentGuideBuffers<Scalar>& bufs,
    Scalar lambda_W,
    int k, int m)
{
    if (lambda_W <= Scalar(0) || !bufs.has_w_target) return;

    // Isotropic W target: G[i,i] += lambda_W, B[:,i] += lambda_W * W_target[:,i]
    {
        int threads = 256;
        int blocks = (k + threads - 1) / threads;
        add_lambda_diag_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_G.data.get(), lambda_W, k);
    }
    {
        dim3 threads(32, 8);
        dim3 blocks((k + 31) / 32, (m + 7) / 8);
        add_target_B_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_B.data.get(), bufs.d_W_target.data.get(), lambda_W, k, m);
    }
}

}  // namespace alignment_guide_gpu
}  // namespace nmf
}  // namespace FactorNet
