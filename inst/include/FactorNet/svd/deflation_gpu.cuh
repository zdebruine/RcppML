// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file deflation_gpu.cuh
 * @brief GPU-accelerated deflation-based truncated SVD/PCA with auto-rank
 *
 * GPU counterpart to deflation_svd.hpp. Runs the rank-1 ALS deflation
 * loop on GPU using cuSPARSE for SpMV and cuBLAS for vector operations.
 *
 * Data residency:
 *   - A (sparse CSC): uploaded once at start, stays on device
 *   - u, v vectors: device-resident during ALS iteration
 *   - Prior factors U_all, V_all: device-resident (d_U_factors, d_V_factors)
 *     throughout deflation loop; bulk-downloaded to host once at end
 *   - Test entries (CV): device-resident for fast residual updates
 *   - Final U, d, V: downloaded once at end via bulk transfer
 *
 * Performance optimizations:
 *   - cuSPARSE generic SpMV for optimal sparse matvec
 *   - Pre-allocated SpMV buffer (reused across all iterations)
 *   - cuBLAS for nrm2, scal, dot, axpy operations
 *   - Custom CUDA kernels for regularization and CV updates
 *   - CSC treated as CSR of A^T for dual-mode SpMV (forward + transpose)
 *   - PCA centering corrections via lightweight GPU kernels
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/types.cuh>
#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/rng/rng.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <limits>
#include <chrono>
#include <cmath>

namespace FactorNet {
namespace svd {
namespace detail {

// ============================================================================
// CUDA kernels for SVD-specific operations
// ============================================================================

/**
 * @brief Apply per-element regularization on a device vector.
 *
 * Applies L2 shrinkage, L1 soft-thresholding, non-negativity, and
 * upper bound constraints in a single fused kernel.
 */
template<typename Scalar>
__global__ void regularize_kernel(
    Scalar* __restrict__ x, int n,
    Scalar L1, Scalar L2,
    bool nonneg, Scalar upper_bound,
    Scalar norm_sq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Scalar val = x[i];

    // L2 shrinkage
    if (L2 > Scalar(0)) {
        val /= (Scalar(1) + L2 / norm_sq);
    }

    // L1 soft-thresholding
    if (L1 > Scalar(0)) {
        Scalar thresh = L1 / (Scalar(2) * norm_sq);
        if (val > thresh) val -= thresh;
        else if (val < -thresh) val += thresh;
        else val = Scalar(0);
    }

    // Non-negativity
    if (nonneg && val < Scalar(0)) val = Scalar(0);

    // Upper bound
    if (upper_bound > Scalar(0) && val > upper_bound) val = upper_bound;

    x[i] = val;
}

/// Same as regularize_kernel but reads norm_sq from device memory (no host sync)
template<typename Scalar>
__global__ void regularize_device_scalar_kernel(
    Scalar* __restrict__ x, int n,
    Scalar L1, Scalar L2,
    bool nonneg, Scalar upper_bound,
    const Scalar* __restrict__ d_norm_sq)
{
    Scalar norm_sq = d_norm_sq[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Scalar val = x[i];

    if (L2 > Scalar(0)) {
        val /= (Scalar(1) + L2 / norm_sq);
    }
    if (L1 > Scalar(0)) {
        Scalar thresh = L1 / (Scalar(2) * norm_sq);
        if (val > thresh) val -= thresh;
        else if (val < -thresh) val += thresh;
        else val = Scalar(0);
    }
    if (nonneg && val < Scalar(0)) val = Scalar(0);
    if (upper_bound > Scalar(0) && val > upper_bound) val = upper_bound;

    x[i] = val;
}

#ifndef FACTORNET_SVD_GPU_DEF_subtract_scalar_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_scalar_kernel
/**
 * @brief PCA centering correction for transpose SpMV: result[i] -= correction
 *
 * For A_c' u = A'u - (μ·u) * 1_n, subtracts a scalar from every element.
 */
template<typename Scalar>
__global__ void subtract_scalar_kernel(Scalar* vec, int n, Scalar val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec[i] -= val;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_scalar_kernel

#ifndef FACTORNET_SVD_GPU_DEF_subtract_scaled_vector_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_scaled_vector_kernel
/**
 * @brief PCA centering correction for forward SpMV: result[i] -= mu[i] * scale
 *
 * For A_c v = Av - μ * sum(v), subtracts μ scaled by sum(v).
 */
template<typename Scalar>
__global__ void subtract_scaled_vector_kernel(
    Scalar* result, const Scalar* mu, int m, Scalar scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    result[i] -= mu[i] * scale;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_scaled_vector_kernel

/**
 * @brief Update test residuals on GPU after extracting a new factor.
 *
 * residual[i] -= sigma * u[row[i]] * v[col[i]]
 */
template<typename Scalar>
__global__ void update_test_residuals_kernel(
    Scalar* __restrict__ residuals,
    const int* __restrict__ rows,
    const int* __restrict__ cols,
    const Scalar* __restrict__ u,
    Scalar sigma,
    const Scalar* __restrict__ v,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    residuals[i] -= sigma * u[rows[i]] * v[cols[i]];
}

/**
 * @brief Compute sum of squared residuals via parallel reduction.
 *
 * Uses warp-level intrinsics + shared memory reduction for efficiency.
 * Each block reduces a tile, then atomicAdd to a global accumulator.
 */
template<typename Scalar>
__global__ void sum_of_squares_kernel(
    const Scalar* __restrict__ data, int n,
    Scalar* __restrict__ result)
{
    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar val = (i < n) ? data[i] * data[i] : Scalar(0);
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile Scalar* vsmem = sdata;
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

/**
 * @brief Element-wise multiply: out[i] *= scale[i]
 *
 * Used by deflation_correct_gpu to multiply dots by sigma on-device,
 * eliminating a D→H→D roundtrip that was the critical bottleneck
 * (called 2× per ALS iter × max_iter × k factors).
 */
template<typename Scalar>
__global__ void elementwise_mul_kernel(Scalar* __restrict__ out,
                                        const Scalar* __restrict__ scale,
                                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] *= scale[i];
}

/**
 * @brief Deflation correction on GPU.
 *
 * Computes: raw -= Factors * (sigma .* (PrevX^T * x))
 *
 * This is two operations:
 *   1. dots = PrevX^T * x   (GEMV: n_prev × dim)
 *   2. raw -= Factors * (sigma .* dots)   (GEMV: dim × n_prev)
 *
 * For small n_prev (number of prior factors), this is fast.
 * We do it on host with cuBLAS since the factor matrices change size.
 */
template<typename Scalar>
void deflation_correct_gpu(
    const FactorNet::gpu::GPUContext& ctx,
    Scalar* d_raw,             // device: result vector (dim × 1)
    const Scalar* d_x,         // device: fixed vector (x_dim × 1)
    const Scalar* d_Factors,   // device: n_prev columns of factors (dim × n_prev, col-major)
    const Scalar* d_sigma,     // device: n_prev singular values
    const Scalar* d_PrevX,     // device: n_prev columns of prev x (x_dim × n_prev, col-major)
    int dim,                   // dimension of raw/Factors
    int x_dim,                 // dimension of x/PrevX
    int n_prev,                // number of prior factors
    Scalar* d_dots_workspace)  // device workspace: n_prev Scalars
{
    if (n_prev <= 0) return;

    const Scalar alpha = Scalar(1);
    const Scalar beta = Scalar(0);
    const Scalar neg_one = Scalar(-1);

    // Step 1: dots = PrevX^T * x  (n_prev × 1)
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_T,
            x_dim, n_prev,
            &alpha, d_PrevX, x_dim,
            d_x, 1,
            &beta, d_dots_workspace, 1));
    } else {
        CUBLAS_CHECK(cublasSgemv(ctx.cublas, CUBLAS_OP_T,
            x_dim, n_prev,
            &alpha, d_PrevX, x_dim,
            d_x, 1,
            &beta, d_dots_workspace, 1));
    }

    // Step 2: dots *= sigma  (element-wise on device)
    // Single tiny kernel launch — eliminates the previous D→H→D roundtrip
    // that was downloading dots + sigma, multiplying on host, then re-uploading.
    {
        int threads = std::min(n_prev, 256);
        int blocks = (n_prev + threads - 1) / threads;
        elementwise_mul_kernel<<<blocks, threads, 0, ctx.stream>>>(
            d_dots_workspace, d_sigma, n_prev);
    }

    // Step 3: raw -= Factors * dots  (GEMV)
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N,
            dim, n_prev,
            &neg_one, d_Factors, dim,
            d_dots_workspace, 1,
            &alpha, d_raw, 1));
    } else {
        CUBLAS_CHECK(cublasSgemv(ctx.cublas, CUBLAS_OP_N,
            dim, n_prev,
            &neg_one, d_Factors, dim,
            d_dots_workspace, 1,
            &alpha, d_raw, 1));
    }
}

// ============================================================================
// cuBLAS scalar wrappers (host-pointer mode — syncs to host)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_cublas_nrm2
#define FACTORNET_SVD_GPU_DEF_cublas_nrm2
template<typename Scalar>
inline Scalar cublas_nrm2(cublasHandle_t handle, int n, const Scalar* x) {
    Scalar result;
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDnrm2(handle, n, x, 1, &result));
    } else {
        CUBLAS_CHECK(cublasSnrm2(handle, n, x, 1, &result));
    }
    return result;
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_nrm2

#ifndef FACTORNET_SVD_GPU_DEF_cublas_scal
#define FACTORNET_SVD_GPU_DEF_cublas_scal
template<typename Scalar>
inline void cublas_scal(cublasHandle_t handle, int n, Scalar alpha, Scalar* x) {
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDscal(handle, n, &alpha, x, 1));
    } else {
        CUBLAS_CHECK(cublasSscal(handle, n, &alpha, x, 1));
    }
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_scal

#ifndef FACTORNET_SVD_GPU_DEF_cublas_dot
#define FACTORNET_SVD_GPU_DEF_cublas_dot
template<typename Scalar>
inline Scalar cublas_dot(cublasHandle_t handle, int n, const Scalar* x, const Scalar* y) {
    Scalar result;
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDdot(handle, n, x, 1, y, 1, &result));
    } else {
        CUBLAS_CHECK(cublasSdot(handle, n, x, 1, y, 1, &result));
    }
    return result;
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_dot

// ============================================================================
// cuBLAS device-pointer mode helpers (non-blocking, result on device)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_cublas_nrm2_device
#define FACTORNET_SVD_GPU_DEF_cublas_nrm2_device
/// nrm2 with result written to device memory (no host sync)
template<typename Scalar>
inline void cublas_nrm2_device(cublasHandle_t h, int n, const Scalar* x, Scalar* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDnrm2(h, n, x, 1, d_result));
    } else {
        CUBLAS_CHECK(cublasSnrm2(h, n, x, 1, d_result));
    }
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_nrm2_device

#ifndef FACTORNET_SVD_GPU_DEF_cublas_dot_device
#define FACTORNET_SVD_GPU_DEF_cublas_dot_device
/// dot product with result written to device memory (no host sync)
template<typename Scalar>
inline void cublas_dot_device(cublasHandle_t h, int n, const Scalar* x, const Scalar* y, Scalar* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDdot(h, n, x, 1, y, 1, d_result));
    } else {
        CUBLAS_CHECK(cublasSdot(h, n, x, 1, y, 1, d_result));
    }
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_dot_device

// ============================================================================
// Device-scalar fused kernels (fully async, zero host syncs)
// ============================================================================

/// Fused normalize: dst[i] = src[i] / d_norm[0]
/// Reads norm from device memory — no host roundtrip needed
template<typename Scalar>
__global__ void fused_normalize_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src,
    const Scalar* __restrict__ d_norm,
    int n)
{
    Scalar inv_norm = Scalar(1) / d_norm[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * inv_norm;
}

/// Fused scale by reciprocal of device scalar: dst[i] = src[i] / d_divisor[0]
/// Used for v = spmv_result / u_sq where u_sq is computed on device
template<typename Scalar>
__global__ void fused_div_scalar_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src,
    const Scalar* __restrict__ d_divisor,
    int n)
{
    Scalar inv_val = Scalar(1) / d_divisor[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * inv_val;
}

#ifndef FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel
/// PCA centering: result[i] -= mu[i] * d_scale[0]  (reads scale from device)
template<typename Scalar>
__global__ void subtract_scaled_vector_device_scalar_kernel(
    Scalar* __restrict__ result, const Scalar* __restrict__ mu,
    int m, const Scalar* __restrict__ d_scale)
{
    Scalar scale = d_scale[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) result[i] -= mu[i] * scale;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel

#ifndef FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel
/// PCA centering: vec[i] -= d_val[0]  (reads val from device)
template<typename Scalar>
__global__ void subtract_device_scalar_kernel(
    Scalar* __restrict__ vec, int n, const Scalar* __restrict__ d_val)
{
    Scalar val = d_val[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] -= val;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel

// ============================================================================
// CV speckled-mask: GPU inline SplitMix64 hash — zero PCIe traffic for mask
// ============================================================================

/// Pass 1 of 2: count holdout entries on GPU (one thread per column)
template<typename Scalar>
__global__ void count_holdout_csc_kernel(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    int n,
    uint64_t cv_seed,
    uint64_t inv_prob,    ///< round(1 / test_fraction) — holdout if hash % inv_prob == 0
    int* __restrict__     d_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int local_cnt = 0;
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
        int row = row_idx[p];
        // Inline SplitMix64 hash of (seed XOR row XOR col)
        uint64_t h = cv_seed
            ^ (static_cast<uint64_t>(row) * 2654435769ULL)
            ^ (static_cast<uint64_t>(j)   * 1013904223ULL);
        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL;
        h ^= (h >> 33); h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= (h >> 33);
        if ((h % inv_prob) == 0) ++local_cnt;
    }
    atomicAdd(d_count, local_cnt);
}

/// Pass 2 of 2: collect (row,col,val) into compact arrays AND zero d_A.values in-place.
/// After this call, d_A.values holds the training matrix; test entries are in the compact arrays.
template<typename Scalar>
__global__ void collect_and_zero_holdout_kernel(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    Scalar* __restrict__     values,
    int n,
    uint64_t cv_seed,
    uint64_t inv_prob,
    const Scalar* __restrict__ d_row_means,   ///< nullptr if no centering
    int* __restrict__     d_cv_rows,
    int* __restrict__     d_cv_cols,
    Scalar* __restrict__  d_cv_residuals,
    int* __restrict__     d_idx)               ///< atomic fill index, init to 0
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
        int row = row_idx[p];
        uint64_t h = cv_seed
            ^ (static_cast<uint64_t>(row) * 2654435769ULL)
            ^ (static_cast<uint64_t>(j)   * 1013904223ULL);
        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL;
        h ^= (h >> 33); h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= (h >> 33);
        if ((h % inv_prob) == 0) {
            Scalar val = values[p];
            if (d_row_means) val -= d_row_means[row];
            values[p] = Scalar(0);   // zero training matrix in-place
            int idx = atomicAdd(d_idx, 1);
            d_cv_rows[idx]      = row;
            d_cv_cols[idx]      = j;
            d_cv_residuals[idx] = val;
        }
    }
}

// ============================================================================
// L21 and graph Laplacian regularization kernels
// ============================================================================

/// L21-norm adaptive L2: eff_L2 = L2 + L21/||x||; reads ||x|| from device (no sync)
template<typename Scalar>
__global__ void regularize_with_l21_device_kernel(
    Scalar* __restrict__ x, int n,
    Scalar L1, Scalar L2, Scalar L21,
    bool nonneg, Scalar upper_bound,
    const Scalar* __restrict__ d_norm_sq,   ///< d_scalars[0] (= u_sq or 1) for L1/L2 denominator
    const Scalar* __restrict__ d_x_norm)    ///< d_scalars[6 or 7] = ||x|| for L21
{
    Scalar norm_sq = d_norm_sq[0];
    Scalar x_norm  = d_x_norm[0];
    Scalar eff_L2  = L2 + (x_norm > Scalar(1e-12f) ? L21 / x_norm : Scalar(0));
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Scalar val = x[i];
    if (eff_L2 > Scalar(0))  val /= (Scalar(1) + eff_L2 / norm_sq);
    if (L1 > Scalar(0)) {
        Scalar thresh = L1 / (Scalar(2) * norm_sq);
        if (val > thresh)       val -= thresh;
        else if (val < -thresh) val += thresh;
        else                    val = Scalar(0);
    }
    if (nonneg && val < Scalar(0)) val = Scalar(0);
    if (upper_bound > Scalar(0) && val > upper_bound) val = upper_bound;
    x[i] = val;
}

/// Graph Laplacian axpy: x[i] -= (lambda / d_norm_sq[0]) * gx[i]
template<typename Scalar>
__global__ void graph_subtract_device_scalar_kernel(
    Scalar* __restrict__       x,
    const Scalar* __restrict__ gx,
    int n,
    Scalar lambda,
    const Scalar* __restrict__ d_norm_sq)
{
    Scalar scale = lambda / d_norm_sq[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] -= scale * gx[i];
}

}  // namespace detail


// ============================================================================
// Dense-matrix GPU helper — cuBLAS GEMV for forward (A*v) and transpose (A'*u)
// ============================================================================

namespace detail {

/// GEMV: y = alpha * op(A) * x + beta * y  (host-pointer alpha/beta)
template<typename Scalar>
inline void cublas_gemv_host(cublasHandle_t h, cublasOperation_t op,
    int m, int n, Scalar alpha,
    const Scalar* A, int lda,
    const Scalar* x, int incx,
    Scalar beta, Scalar* y, int incy)
{
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDgemv(h, op, m, n, &alpha, A, lda, x, incx, &beta, y, incy));
    } else {
        CUBLAS_CHECK(cublasSgemv(h, op, m, n, &alpha, A, lda, x, incx, &beta, y, incy));
    }
}

/// Two-pass Gram-Schmidt reorthogonalization (same pattern as Lanczos gpu_orthog).
/// Reorthogonalizes d_y (dim×1) against d_X (dim×ncols) using two GEMV passes.
/// d_h is a workspace of size >= ncols.
template<typename Scalar>
void gpu_orthog_against(cublasHandle_t cublas,
                        const Scalar* d_X, Scalar* d_y,
                        int dim, int ncols, Scalar* d_h)
{
    if (ncols <= 0) return;
    Scalar one = Scalar(1), zero = Scalar(0), neg_one = Scalar(-1);

    // Pass 1: h = X^T * y, then y -= X * h
    cublas_gemv_host<Scalar>(cublas, CUBLAS_OP_T, dim, ncols,
        one, d_X, dim, d_y, 1, zero, d_h, 1);
    cublas_gemv_host<Scalar>(cublas, CUBLAS_OP_N, dim, ncols,
        neg_one, d_X, dim, d_h, 1, one, d_y, 1);

    // Pass 2: second GS pass for numerical stability
    cublas_gemv_host<Scalar>(cublas, CUBLAS_OP_T, dim, ncols,
        one, d_X, dim, d_y, 1, zero, d_h, 1);
    cublas_gemv_host<Scalar>(cublas, CUBLAS_OP_N, dim, ncols,
        neg_one, d_X, dim, d_h, 1, one, d_y, 1);
}

}  // namespace detail (second close — ends the GEMV helper addendum)

// ============================================================================
// Main GPU SVD algorithm (sparse path — CSC input)
// ============================================================================

/**
 * @brief GPU deflation-based truncated SVD/PCA with optional auto-rank
 *
 * @tparam Scalar     float or double
 * @param h_col_ptr   Host CSC column pointers (length n+1)
 * @param h_row_idx   Host CSC row indices (length nnz)
 * @param h_values    Host CSC values (length nnz)
 * @param m           Number of rows
 * @param n           Number of columns
 * @param nnz         Number of non-zeros
 * @param config      SVD configuration
 * @return SVDResult with factors, loss trajectories, and selected rank
 */
template<typename Scalar>
SVDResult<Scalar> deflation_svd_gpu(
    const int* h_col_ptr,
    const int* h_row_idx,
    const Scalar* h_values,
    int m, int n, int nnz,
    const SVDConfig<Scalar>& config)
{
    using namespace FactorNet::gpu;
    using DenseMatrixS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVectorS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    auto t_start = std::chrono::high_resolution_clock::now();

    const int k_max = config.k_max;

    SVDResult<Scalar> result;
    result.centered = config.center;

    // ------------------------------------------------------------------
    // 1. GPU context and sparse matrix upload (dual-CSR for all-gather SpMV)
    // ------------------------------------------------------------------

    GPUContext ctx;

    // Upload CSC as "CSR of A^T" to cuSPARSE.
    SparseMatrixGPU<Scalar> d_A(m, n, nnz, h_col_ptr, h_row_idx, h_values);

    // Build dual-CSR: CSR(A) for forward, CSR(A^T) for transpose
    // Both use NON_TRANSPOSE (gather) — eliminates slow TRANSPOSE scatter path
    DualCSR<Scalar> dual_csr;
    dual_csr.init(ctx, d_A, m, n, nnz);

    // ------------------------------------------------------------------
    // 2. Compute row means for PCA centering (on host using Eigen sparse)
    // ------------------------------------------------------------------

    DenseVectorS row_means;
    DeviceMemory<Scalar> d_row_means;
    if (config.center) {
        // Use Eigen map for computing row means
        Eigen::Map<const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>>
            A_map(m, n, nnz, h_col_ptr, h_row_idx, h_values);

        row_means.resize(m);
        row_means.setZero();
        for (int j = 0; j < n; ++j) {
            for (typename decltype(A_map)::InnerIterator it(A_map, j); it; ++it) {
                row_means(it.row()) += static_cast<Scalar>(it.value());
            }
        }
        if (n > 0) row_means /= static_cast<Scalar>(n);

        result.row_means = row_means;

        // Upload to device for PCA correction kernels
        d_row_means = DeviceMemory<Scalar>(m);
        d_row_means.upload(row_means.data(), m);
    }

    // ------------------------------------------------------------------
    // 3. Compute ||A||^2_F (or ||A_centered||^2_F)
    // ------------------------------------------------------------------

    Scalar A_norm_sq = 0;
    for (int i = 0; i < nnz; ++i) {
        A_norm_sq += h_values[i] * h_values[i];
    }
    if (config.center) {
        A_norm_sq -= static_cast<Scalar>(n) * row_means.squaredNorm();
    }
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // ------------------------------------------------------------------
    // 4. Pre-allocate device vectors and workspace
    // ------------------------------------------------------------------

    // Working vectors
    DeviceMemory<Scalar> d_u(m), d_v(n), d_u_old(m);
    DeviceMemory<Scalar> d_spmv_result_n(n);  // for A^T * u (length n)
    DeviceMemory<Scalar> d_spmv_result_m(m);  // for A * v (length m)

    // Device-scalar workspace: stores dot/nrm2 results on device (no host sync)
    // Layout: [0]=u_sq/mu_dot_u, [1]=sigma/nrm2, [2]=sum_v/pca_dot, [3]=cos_num, [4]=norm_new, [5]=norm_old
    DeviceMemory<Scalar> d_scalars(8);

    // Device-side ones vector for PCA sum(v) via dot(ones, v)
    DeviceMemory<Scalar> d_ones_n;
    if (config.center) {
        d_ones_n = DeviceMemory<Scalar>(n);
        std::vector<Scalar> h_ones(n, Scalar(1));
        d_ones_n.upload(h_ones.data(), n);
    }

    // cuSPARSE dense vector descriptors (reusable)
    cusparseDnVecDescr_t vecDescr_u, vecDescr_v;
    cusparseDnVecDescr_t vecDescr_spmv_n, vecDescr_spmv_m;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_u, m, d_u.get(),
        CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_v, n, d_v.get(),
        CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_spmv_n, n, d_spmv_result_n.get(),
        CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_spmv_m, m, d_spmv_result_m.get(),
        CudaDataType<Scalar>::value));

    // Pre-allocate SpMV buffer (dual-CSR: both use NON_TRANSPOSE gather)
    Scalar spmv_alpha = Scalar(1), spmv_beta = Scalar(0);
    size_t bufferSize_fwd = 0, bufferSize_trans = 0;

    // Buffer for A^T * u via NON_TRANSPOSE on CSR(A^T): output is n×1
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_AT, vecDescr_u,
        &spmv_beta, vecDescr_spmv_n,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_trans));

    // Buffer for A * v via NON_TRANSPOSE on CSR(A): output is m×1
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_A, vecDescr_v,
        &spmv_beta, vecDescr_spmv_m,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_fwd));

    // CRITICAL: use SEPARATE preprocess buffers for forward and transpose.
    // Sharing a single buffer causes the second preprocess to overwrite the
    // first's analysis state, corrupting SpMV results for the other direction.
    DeviceMemory<char> d_spmv_buffer_fwd(bufferSize_fwd > 0 ? bufferSize_fwd : 1);
    DeviceMemory<char> d_spmv_buffer_trans(bufferSize_trans > 0 ? bufferSize_trans : 1);

    // Pre-analyze sparsity pattern for faster SpMV (CUDA 12+ only)
#if CUSPARSE_VERSION >= 12000
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_AT, vecDescr_u,
        &spmv_beta, vecDescr_spmv_n,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_trans.get()));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_A, vecDescr_v,
        &spmv_beta, vecDescr_spmv_m,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_fwd.get()));
#endif
    // Factor storage (host-side, accumulated)
    DenseMatrixS U_all(m, k_max);
    DenseVectorS d_all(k_max);
    DenseMatrixS V_all(n, k_max);
    d_all.setZero();

    // Device-side factor storage for deflation correction
    // We accumulate these as we compute factors. Store as column-major:
    //   d_U_all: m × k_max, d_V_all: n × k_max, d_sigma: k_max
    DeviceMemory<Scalar> d_U_factors(static_cast<size_t>(m) * k_max);
    DeviceMemory<Scalar> d_V_factors(static_cast<size_t>(n) * k_max);
    DeviceMemory<Scalar> d_sigma_all(k_max);
    DeviceMemory<Scalar> d_defl_workspace(k_max);    // for deflation dots
    DeviceMemory<Scalar> d_angular_workspace(k_max); // for angular penalty projection dots

    // Device scalar containing 1.0 — used as norm_sq denominator for u-side reg
    // (since v is unit-normalized at u-update time, so v_sq = 1)
    DeviceMemory<Scalar> d_one_scalar(1);
    { Scalar h_one = Scalar(1); d_one_scalar.upload(&h_one, 1); }

    // ------------------------------------------------------------------
    // Optional graph Laplacian regularization — upload L_v and/or L_u
    // ------------------------------------------------------------------

    bool d_has_graph_v = (config.graph_v != nullptr && config.graph_v_lambda > 0 &&
                          config.graph_v->rows() == n && config.graph_v->cols() == n &&
                          config.graph_v->nonZeros() > 0);
    SparseMatrixGPU<Scalar> d_graph_v_mat;
    cusparseSpMatDescr_t    descr_graph_v      = nullptr;
    DeviceMemory<char>      d_graph_v_spmv_buf(1);
    if (d_has_graph_v) {
        const auto* gv = config.graph_v;
        d_graph_v_mat = SparseMatrixGPU<Scalar>(n, n, gv->nonZeros(),
            gv->outerIndexPtr(), gv->innerIndexPtr(), gv->valuePtr());
        // Treat CSC as CSR(L_v^T) — for symmetric L_v, NON_TRANSPOSE gives L_v * x
        CUSPARSE_CHECK(cusparseCreateCsr(
            &descr_graph_v, n, n, gv->nonZeros(),
            (void*)d_graph_v_mat.col_ptr.get(),
            (void*)d_graph_v_mat.row_indices.get(),
            (void*)d_graph_v_mat.values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));
        size_t gv_buf = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, descr_graph_v, vecDescr_v, &spmv_beta, vecDescr_spmv_n,
            CudaDataType<Scalar>::value, CUSPARSE_SPMV_ALG_DEFAULT, &gv_buf));
        d_graph_v_spmv_buf = DeviceMemory<char>(gv_buf > 0 ? gv_buf : 1);
    }

    bool d_has_graph_u = (config.graph_u != nullptr && config.graph_u_lambda > 0 &&
                          config.graph_u->rows() == m && config.graph_u->cols() == m &&
                          config.graph_u->nonZeros() > 0);
    SparseMatrixGPU<Scalar> d_graph_u_mat;
    cusparseSpMatDescr_t    descr_graph_u      = nullptr;
    DeviceMemory<char>      d_graph_u_spmv_buf(1);
    if (d_has_graph_u) {
        const auto* gu = config.graph_u;
        d_graph_u_mat = SparseMatrixGPU<Scalar>(m, m, gu->nonZeros(),
            gu->outerIndexPtr(), gu->innerIndexPtr(), gu->valuePtr());
        CUSPARSE_CHECK(cusparseCreateCsr(
            &descr_graph_u, m, m, gu->nonZeros(),
            (void*)d_graph_u_mat.col_ptr.get(),
            (void*)d_graph_u_mat.row_indices.get(),
            (void*)d_graph_u_mat.values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));
        size_t gu_buf = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, descr_graph_u, vecDescr_u, &spmv_beta, vecDescr_spmv_m,
            CudaDataType<Scalar>::value, CUSPARSE_SPMV_ALG_DEFAULT, &gu_buf));
        d_graph_u_spmv_buf = DeviceMemory<char>(gu_buf > 0 ? gu_buf : 1);
    }

    // ------------------------------------------------------------------
    // 5. Setup speckled mask for auto-rank CV
    // ------------------------------------------------------------------

    const bool do_cv = config.is_cv();
    int cv_count = 0;
    DeviceMemory<int> d_cv_rows, d_cv_cols;
    DeviceMemory<Scalar> d_cv_residuals;
    DeviceMemory<Scalar> d_cv_mse_accum(1);  // for reduction

    Scalar best_test_loss = std::numeric_limits<Scalar>::max();
    int best_k = 0;
    int patience_counter = 0;

    if (do_cv) {
        // GPU-native 2-pass speckled mask — zero PCIe traffic for mask data
        // Pass 1: count holdout entries without downloading any values
        const uint64_t cv_seed_u  = config.effective_cv_seed();
        const double   cv_frac    = static_cast<double>(config.test_fraction);
        const uint64_t cv_inv_prob = (cv_frac > 0.0 && cv_frac < 1.0)
            ? static_cast<uint64_t>(std::round(1.0 / cv_frac)) : 10ULL;

        DeviceMemory<int> d_count(1);
        CUDA_CHECK(cudaMemsetAsync(d_count.get(), 0, sizeof(int), ctx.stream));

        {
            const int bs = 256;
            int blk = (n + bs - 1) / bs;
            detail::count_holdout_csc_kernel<Scalar><<<blk, bs, 0, ctx.stream>>>(
                d_A.col_ptr.get(), d_A.row_indices.get(), n,
                cv_seed_u, cv_inv_prob, d_count.get());
        }
        ctx.sync();

        d_count.download(&cv_count, 1);

        if (cv_count > 0) {
            d_cv_rows      = DeviceMemory<int>(cv_count);
            d_cv_cols      = DeviceMemory<int>(cv_count);
            d_cv_residuals = DeviceMemory<Scalar>(cv_count);

            DeviceMemory<int> d_fill_idx(1);
            CUDA_CHECK(cudaMemsetAsync(d_fill_idx.get(), 0, sizeof(int), ctx.stream));

            // Pass 2: collect (row,col,val) into compact arrays AND zero d_A.values
            {
                const int bs = 256;
                int blk = (n + bs - 1) / bs;
                detail::collect_and_zero_holdout_kernel<Scalar><<<blk, bs, 0, ctx.stream>>>(
                    d_A.col_ptr.get(), d_A.row_indices.get(), d_A.values.get(), n,
                    cv_seed_u, cv_inv_prob,
                    config.center ? d_row_means.get() : nullptr,
                    d_cv_rows.get(), d_cv_cols.get(), d_cv_residuals.get(),
                    d_fill_idx.get());
            }
            ctx.sync();

            // Refresh CSR(A) copy so forward SpMV uses the zeroed training matrix
            // CSR(A^T) (descr_AT) already sees the change since it aliases d_A.values
            dual_csr.refresh_values_from_csc(ctx, d_A);
        }
    }

    // ------------------------------------------------------------------
    // 5b. CV denominator correction
    // ------------------------------------------------------------------
    // When holdout entries are zeroed, the ALS denominator (||u||^2 for
    // v-update, ||v||^2=1 for u-update) overcounts by including
    // contributions from rows/columns whose data was held out.
    // Correction: scale by (1 - effective_holdout_fraction).

    Scalar cv_denom_correction = Scalar(1);
    DeviceMemory<Scalar> d_cv_denom_scalar(1);
    if (do_cv) {
        if (config.mask_zeros) {
            Scalar density = static_cast<Scalar>(nnz) /
                            (static_cast<Scalar>(m) * static_cast<Scalar>(n));
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction) * density;
        } else {
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        }
        d_cv_denom_scalar.upload(&cv_denom_correction, 1);
        if (config.verbose) {
            FACTORNET_PRINT_IMPL("  GPU CV: denom_correction=%.4f\n",
                                 static_cast<double>(cv_denom_correction));
        }
    }

    // ------------------------------------------------------------------
    // 6. RNG for initialization
    // ------------------------------------------------------------------

    rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));

    // Host workspace for init vector
    std::vector<Scalar> h_u_init(m);

    const int BLOCK_SIZE = 256;

    int k_computed = 0;

    // ------------------------------------------------------------------
    // 7. Main deflation loop
    // ------------------------------------------------------------------

    for (int k = 0; k < k_max; ++k) {

        // Random initialization for u (on host, upload)
        for (int i = 0; i < m; ++i) {
            h_u_init[i] = static_cast<Scalar>(rng.uniform<double>());
        }
        // Normalize on host
        Scalar init_norm = 0;
        for (int i = 0; i < m; ++i) init_norm += h_u_init[i] * h_u_init[i];
        init_norm = std::sqrt(init_norm);
        if (init_norm > 0) {
            for (int i = 0; i < m; ++i) h_u_init[i] /= init_norm;
        }
        d_u.upload(h_u_init.data(), m);

        Scalar sigma = 0;
        int iter = 0;

        for (iter = 0; iter < config.max_iter; ++iter) {

            // Save u_old for convergence check
            CUDA_CHECK(cudaMemcpyAsync(d_u_old.get(), d_u.get(),
                m * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));

            // ============================================================
            // V-update: v = (A^T u - deflation_correction) / ||u||^2
            // ============================================================

            // SpMV: spmv_result_n = A^T * u  (n×1)
            // dual_csr.descr_AT is CSR(A^T) [n×m], NON_TRANSPOSE = gather path
            CUSPARSE_CHECK(cusparseSpMV(
                ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &spmv_alpha, dual_csr.descr_AT, vecDescr_u,
                &spmv_beta, vecDescr_spmv_n,
                CudaDataType<Scalar>::value,
                CUSPARSE_SPMV_ALG_DEFAULT,
                d_spmv_buffer_trans.get()));

            // PCA centering: spmv_result_n -= (mu.dot(u)) * 1_n  (fully async)
            if (config.center) {
                detail::cublas_dot_device<Scalar>(
                    ctx.cublas, m, d_row_means.get(), d_u.get(), d_scalars.get());
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::subtract_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_spmv_result_n.get(), n, d_scalars.get());
            }

            // Deflation correction
            if (k > 0) {
                detail::deflation_correct_gpu<Scalar>(
                    ctx,
                    d_spmv_result_n.get(),   // raw = A^T u (n×1)
                    d_u.get(),               // x = u (m×1)
                    d_V_factors.get(),       // Factors = V_all (n × k), col-major
                    d_sigma_all.get(),
                    d_U_factors.get(),       // PrevX = U_all (m × k), col-major
                    n, m, k,
                    d_defl_workspace.get());
            }

            // u_sq = ||u||^2 → device scalar (no host sync)
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, m, d_u.get(), d_u.get(), d_scalars.get());

            // CV correction: scale u_sq by (1 - effective_holdout_fraction)
            // This corrects both the v = spmv/u_sq division AND all v-side
            // regularization kernels (which read d_scalars[0] = u_sq).
            if (do_cv) detail::cublas_scal<Scalar>(ctx.cublas, 1, cv_denom_correction, d_scalars.get());

            // v = spmv_result_n / u_sq (fused: reads u_sq from device memory)
            {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_div_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_v.get(), d_spmv_result_n.get(), d_scalars.get(), n);
            }

            // Apply v-side regularization (reads u_sq from device, no sync)
            if (config.has_regularization_v()) {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if (config.L21_v > 0) {
                    // L21: compute ||v|| → d_scalars[6], then blend into effective L2
                    detail::cublas_nrm2_device<Scalar>(
                        ctx.cublas, n, d_v.get(), d_scalars.get() + 6);
                    detail::regularize_with_l21_device_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_v.get(), n,
                        config.L1_v, config.L2_v, config.L21_v,
                        config.nonneg_v, config.upper_bound_v,
                        d_scalars.get(),      // [0] = u_sq (L1/L2 denominator)
                        d_scalars.get() + 6); // [6] = ||v|| for L21
                } else {
                    detail::regularize_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_v.get(), n,
                        config.L1_v, config.L2_v,
                        config.nonneg_v, config.upper_bound_v,
                        d_scalars.get());
                }

                // Angular penalty: project out prior factor components
                // d_v -= angular_v * V_prev * (V_prev^T * d_v)
                if (config.angular_v > 0 && k > 0) {
                    // dots = V_prev^T * d_v  (k × 1)
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_T,
                        n, k, Scalar(1),
                        d_V_factors.get(), n, d_v.get(), 1,
                        Scalar(0), d_angular_workspace.get(), 1);
                    // d_v -= angular_v * V_prev * dots
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_N,
                        n, k, -config.angular_v,
                        d_V_factors.get(), n, d_angular_workspace.get(), 1,
                        Scalar(1), d_v.get(), 1);
                }

                // Graph Laplacian: d_v -= (graph_v_lambda / u_sq) * (L_v * d_v)
                if (d_has_graph_v) {
                    // SpMV: d_spmv_result_n = L_v * d_v  (reuse spmv_result_n as temp)
                    CUSPARSE_CHECK(cusparseSpMV(ctx.cusparse,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &spmv_alpha, descr_graph_v, vecDescr_v,
                        &spmv_beta,  vecDescr_spmv_n,
                        CudaDataType<Scalar>::value,
                        CUSPARSE_SPMV_ALG_DEFAULT, d_graph_v_spmv_buf.get()));
                    // axpy: d_v -= (graph_v_lambda / u_sq) * d_spmv_result_n
                    {
                        int blk = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        detail::graph_subtract_device_scalar_kernel<<<blk, BLOCK_SIZE, 0, ctx.stream>>>(
                            d_v.get(), d_spmv_result_n.get(), n,
                            config.graph_v_lambda, d_scalars.get()); // [0] = u_sq
                    }
                }
            }

            // sigma = ||v||, then normalize v (fused: reads norm from device)
            detail::cublas_nrm2_device<Scalar>(
                ctx.cublas, n, d_v.get(), d_scalars.get() + 1);
            {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_v.get(), d_v.get(), d_scalars.get() + 1, n);
            }

            // ============================================================
            // U-update: u = (A v - deflation_correction) / ||v||^2
            // ============================================================

            // SpMV: spmv_result_m = A * v  (m×1)
            // dual_csr.descr_A is CSR(A) [m×n], NON_TRANSPOSE = gather path
            CUSPARSE_CHECK(cusparseSpMV(
                ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &spmv_alpha, dual_csr.descr_A, vecDescr_v,
                &spmv_beta, vecDescr_spmv_m,
                CudaDataType<Scalar>::value,
                CUSPARSE_SPMV_ALG_DEFAULT,
                d_spmv_buffer_fwd.get()));

            // PCA centering: spmv_result_m -= mu * sum(v)  (fully async)
            if (config.center) {
                // sum(v) = dot(ones, v) → device scalar
                detail::cublas_dot_device<Scalar>(
                    ctx.cublas, n, d_ones_n.get(), d_v.get(), d_scalars.get() + 2);
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::subtract_scaled_vector_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_spmv_result_m.get(), d_row_means.get(), m, d_scalars.get() + 2);
            }

            // Deflation correction
            if (k > 0) {
                detail::deflation_correct_gpu<Scalar>(
                    ctx,
                    d_spmv_result_m.get(),   // raw = A v (m×1)
                    d_v.get(),               // x = v (n×1)
                    d_U_factors.get(),       // Factors = U_all (m × k), col-major
                    d_sigma_all.get(),
                    d_V_factors.get(),       // PrevX = V_all (n × k), col-major
                    m, n, k,
                    d_defl_workspace.get());
            }

            // v_sq = ||v||^2 = 1 (since we normalized)
            // CV correction: effective v_sq = cv_denom_correction (not 1)
            if (do_cv) {
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_div_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_u.get(), d_spmv_result_m.get(), d_cv_denom_scalar.get(), m);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_u.get(), d_spmv_result_m.get(),
                    m * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
            }

            // Apply u-side regularization
            if (config.has_regularization_u()) {
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if (config.L21_u > 0) {
                    // L21: compute ||u|| → d_scalars[7], then blend into effective L2
                    // norm_sq denominator = 1 (since v was unit-normalized before u-update)
                    detail::cublas_nrm2_device<Scalar>(
                        ctx.cublas, m, d_u.get(), d_scalars.get() + 7);
                    detail::regularize_with_l21_device_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_u.get(), m,
                        config.L1_u, config.L2_u, config.L21_u,
                        config.nonneg_u, config.upper_bound_u,
                        do_cv ? d_cv_denom_scalar.get() : d_one_scalar.get(),
                        d_scalars.get() + 7); // [7] = ||u|| for L21
                } else {
                    detail::regularize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_u.get(), m,
                        config.L1_u, config.L2_u,
                        config.nonneg_u, config.upper_bound_u,
                        do_cv ? cv_denom_correction : Scalar(1));
                }

                // Angular penalty: project out prior factor components
                // d_u -= angular_u * U_prev * (U_prev^T * d_u)
                if (config.angular_u > 0 && k > 0) {
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_T,
                        m, k, Scalar(1),
                        d_U_factors.get(), m, d_u.get(), 1,
                        Scalar(0), d_angular_workspace.get(), 1);
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_N,
                        m, k, -config.angular_u,
                        d_U_factors.get(), m, d_angular_workspace.get(), 1,
                        Scalar(1), d_u.get(), 1);
                }

                // Graph Laplacian: d_u -= graph_u_lambda * (L_u * d_u)
                // (norm_sq = 1, so no division — graph_u_lambda scales directly)
                if (d_has_graph_u) {
                    CUSPARSE_CHECK(cusparseSpMV(ctx.cusparse,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &spmv_alpha, descr_graph_u, vecDescr_u,
                        &spmv_beta,  vecDescr_spmv_m,
                        CudaDataType<Scalar>::value,
                        CUSPARSE_SPMV_ALG_DEFAULT, d_graph_u_spmv_buf.get()));
                    {
                        int blk = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        detail::graph_subtract_device_scalar_kernel<<<blk, BLOCK_SIZE, 0, ctx.stream>>>(
                            d_u.get(), d_spmv_result_m.get(), m,
                            config.graph_u_lambda,
                            do_cv ? d_cv_denom_scalar.get() : d_one_scalar.get());
                    }
                }
            }

            // sigma = ||u||, then normalize u (fused: reads norm from device)
            detail::cublas_nrm2_device<Scalar>(
                ctx.cublas, m, d_u.get(), d_scalars.get() + 1);
            {
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_u.get(), d_u.get(), d_scalars.get() + 1, m);
            }

            // ============================================================
            // Convergence check: 1 - |cos(u, u_old)|
            // Compute dot/nrm2 on device, then single sync to read results
            // ============================================================
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, m, d_u.get(), d_u_old.get(), d_scalars.get() + 3);
            detail::cublas_nrm2_device<Scalar>(
                ctx.cublas, m, d_u.get(), d_scalars.get() + 4);
            detail::cublas_nrm2_device<Scalar>(
                ctx.cublas, m, d_u_old.get(), d_scalars.get() + 5);

            // Single sync to read all convergence values + sigma
            Scalar h_conv[4];  // [0]=sigma, [1]=cos_num, [2]=norm_new, [3]=norm_old
            CUDA_CHECK(cudaMemcpyAsync(&h_conv[0], d_scalars.get() + 1, sizeof(Scalar),
                cudaMemcpyDeviceToHost, ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(&h_conv[1], d_scalars.get() + 3, 3 * sizeof(Scalar),
                cudaMemcpyDeviceToHost, ctx.stream));
            ctx.sync();

            sigma = h_conv[0];
            if (sigma <= Scalar(0)) break;

            Scalar cos_sim_num = std::abs(h_conv[1]);
            Scalar u_norm_new = h_conv[2];
            Scalar u_norm_old = h_conv[3];
            Scalar cos_val = cos_sim_num / (u_norm_new * u_norm_old +
                             std::numeric_limits<Scalar>::epsilon());

            if (Scalar(1) - cos_val < config.tol) {
                ++iter;
                break;
            }
        }

        // ============================================================
        // Post-factor Gram-Schmidt reorthogonalization (on device)
        // Two-pass GS via cuBLAS GEMV: eliminates orthogonality drift
        // at k>=16. Cost: 4 GEMVs of size (m×k) + (n×k) — negligible
        // vs O(nnz) SpMV per ALS iteration.
        // ============================================================
        if (k > 0) {
            // Workspace for GS coefficients (reuse d_defl_workspace, size >= k_max)
            Scalar* d_gs_h = d_defl_workspace.get();

            // Reorthogonalize d_u against d_U_factors[:, 0:k-1]
            detail::gpu_orthog_against<Scalar>(
                ctx.cublas, d_U_factors.get(), d_u.get(), m, k, d_gs_h);
            // Renormalize d_u
            {
                detail::cublas_nrm2_device<Scalar>(
                    ctx.cublas, m, d_u.get(), d_scalars.get());
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_u.get(), d_u.get(), d_scalars.get(), m);
            }

            // Reorthogonalize d_v against d_V_factors[:, 0:k-1]
            detail::gpu_orthog_against<Scalar>(
                ctx.cublas, d_V_factors.get(), d_v.get(), n, k, d_gs_h);
            // Renormalize d_v
            {
                detail::cublas_nrm2_device<Scalar>(
                    ctx.cublas, n, d_v.get(), d_scalars.get());
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_v.get(), d_v.get(), d_scalars.get(), n);
            }
        }

        // ============================================================
        // Store factor on device only (host download deferred to end)
        // D2D copies: d_u → d_U_factors[:,k], d_v → d_V_factors[:,k]
        // sigma D2D: d_scalars[1] → d_sigma_all[k]
        // ============================================================

        k_computed = k + 1;
        result.iters_per_factor.push_back(iter);

        CUDA_CHECK(cudaMemcpyAsync(
            d_U_factors.get() + static_cast<size_t>(k) * m,
            d_u.get(), m * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_V_factors.get() + static_cast<size_t>(k) * n,
            d_v.get(), n * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_sigma_all.get() + k, d_scalars.get() + 1,
            sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));

        // ============================================================
        // Test loss evaluation (on GPU)
        // ============================================================

        if (do_cv && cv_count > 0) {
            // Update test residuals: residual[i] -= sigma * u[row[i]] * v[col[i]]
            int blocks_cv = (cv_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::update_test_residuals_kernel<<<blocks_cv, BLOCK_SIZE, 0, ctx.stream>>>(
                d_cv_residuals.get(),
                d_cv_rows.get(), d_cv_cols.get(),
                d_u.get(), sigma, d_v.get(), cv_count);

            // Compute MSE = sum(residuals^2) / count
            CUDA_CHECK(cudaMemsetAsync(d_cv_mse_accum.get(), 0, sizeof(Scalar), ctx.stream));
            size_t smem_size = BLOCK_SIZE * sizeof(Scalar);
            detail::sum_of_squares_kernel<<<blocks_cv, BLOCK_SIZE, smem_size, ctx.stream>>>(
                d_cv_residuals.get(), cv_count, d_cv_mse_accum.get());

            ctx.sync();
            Scalar sum_sq;
            d_cv_mse_accum.download(&sum_sq, 1);
            Scalar test_mse = sum_sq / static_cast<Scalar>(cv_count);

            result.test_loss_trajectory.push_back(test_mse);

            if (config.verbose) {
                FACTORNET_PRINT_IMPL("  Factor %d: sigma=%.4e  iters=%d  test_mse=%.6e\n",
                    k + 1, static_cast<double>(sigma), iter,
                    static_cast<double>(test_mse));
            }

            // Auto-rank patience
            if (test_mse < best_test_loss) {
                best_test_loss = test_mse;
                best_k = k + 1;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    if (config.verbose) {
                        FACTORNET_PRINT_IMPL(
                            "  Auto-rank: patience exhausted at factor %d, "
                            "best rank = %d (test_mse=%.6e)\n",
                            k + 1, best_k, static_cast<double>(best_test_loss));
                    }
                    break;
                }
            }
        } else {
            if (config.verbose) {
                FACTORNET_PRINT_IMPL("  Factor %d: sigma=%.4e  iters=%d\n",
                    k + 1, static_cast<double>(sigma), iter);
            }
            best_k = k + 1;
        }

        // Early stop on negligible sigma
        if (sigma < std::numeric_limits<Scalar>::epsilon() * Scalar(100)) {
            if (config.verbose) {
                FACTORNET_PRINT_IMPL("  Factor %d: sigma ~ 0, stopping early\n", k + 1);
            }
            break;
        }
    }

    // ------------------------------------------------------------------
    // 8. Cleanup cuSPARSE descriptors
    // ------------------------------------------------------------------

    cusparseDestroyDnVec(vecDescr_u);
    cusparseDestroyDnVec(vecDescr_v);
    cusparseDestroyDnVec(vecDescr_spmv_n);
    cusparseDestroyDnVec(vecDescr_spmv_m);
    // dual_csr destructor handles descr_A and descr_AT cleanup

    // Cleanup optional graph Laplacian descriptors
    if (descr_graph_v) cusparseDestroySpMat(descr_graph_v);
    if (descr_graph_u) cusparseDestroySpMat(descr_graph_u);

    // ------------------------------------------------------------------
    // 9. Bulk-download factors from device → host
    // ------------------------------------------------------------------

    ctx.sync();
    if (k_computed > 0) {
        // U_all and d_U_factors are both column-major (m × k_computed)
        CUDA_CHECK(cudaMemcpy(U_all.data(), d_U_factors.get(),
            static_cast<size_t>(m) * k_computed * sizeof(Scalar),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(V_all.data(), d_V_factors.get(),
            static_cast<size_t>(n) * k_computed * sizeof(Scalar),
            cudaMemcpyDeviceToHost));
        std::vector<Scalar> h_sigmas(k_computed);
        CUDA_CHECK(cudaMemcpy(h_sigmas.data(), d_sigma_all.get(),
            k_computed * sizeof(Scalar), cudaMemcpyDeviceToHost));
        for (int i = 0; i < k_computed; ++i) d_all(i) = h_sigmas[i];
    }

    // ------------------------------------------------------------------
    // 10. Pack result
    // ------------------------------------------------------------------

    result.k_selected = best_k;
    result.U = U_all.leftCols(best_k);
    result.d = d_all.head(best_k);
    result.V = V_all.leftCols(best_k);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

// ============================================================================
// Dense-matrix GPU deflation SVD (uses cuBLAS GEMV instead of cuSPARSE SpMV)
// ============================================================================

/**
 * @brief GPU deflation SVD for dense (column-major) input matrices.
 *
 * Uses cuBLAS DGEMV/SGEMV for forward (A*v) and transpose (A'*u) matvec,
 * which is significantly faster than cuSPARSE SpMV when A is dense.
 * All other logic (ALS loop, CV, regularization, deflation correction) is
 * identical to the sparse variant.
 *
 * @param h_A   Host column-major dense matrix (m × n)
 * @param m     Number of rows
 * @param n     Number of columns
 * @param config SVD configuration (same fields as sparse variant)
 */
template<typename Scalar>
SVDResult<Scalar> deflation_svd_gpu_dense(
    const Scalar* h_A,
    int m, int n,
    const SVDConfig<Scalar>& config)
{
    using namespace FactorNet::gpu;
    using DenseMatrixS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVectorS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    auto t_start = std::chrono::high_resolution_clock::now();

    const int k_max = config.k_max;

    SVDResult<Scalar> result;
    result.centered = config.center;

    // ------------------------------------------------------------------
    // 1. GPU context and dense matrix upload
    // ------------------------------------------------------------------

    GPUContext ctx;

    const size_t mn = static_cast<size_t>(m) * n;
    DeviceMemory<Scalar> d_A_dense(mn);
    d_A_dense.upload(h_A, mn);

    // ------------------------------------------------------------------
    // 2. Compute row means and A_norm_sq for PCA centering (host side)
    // ------------------------------------------------------------------

    DenseVectorS row_means;
    DeviceMemory<Scalar> d_row_means;

    Scalar A_norm_sq = Scalar(0);
    for (size_t i = 0; i < mn; ++i) A_norm_sq += h_A[i] * h_A[i];

    if (config.center) {
        row_means.resize(m);
        row_means.setZero();
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                row_means(i) += h_A[i + static_cast<size_t>(j) * m];
        if (n > 0) row_means /= static_cast<Scalar>(n);

        result.row_means = row_means;

        d_row_means = DeviceMemory<Scalar>(m);
        d_row_means.upload(row_means.data(), m);

        // ||A_centered||^2 = ||A||^2 - n * ||mu||^2
        A_norm_sq -= static_cast<Scalar>(n) * row_means.squaredNorm();
    }
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // ------------------------------------------------------------------
    // 3. Device workspace (same layout as sparse path)
    // ------------------------------------------------------------------

    DeviceMemory<Scalar> d_u(m), d_v(n), d_u_old(m);
    DeviceMemory<Scalar> d_spmv_result_n(n);  // A' * u  (n×1)
    DeviceMemory<Scalar> d_spmv_result_m(m);  // A * v   (m×1)
    DeviceMemory<Scalar> d_scalars(8);

    DeviceMemory<Scalar> d_ones_n;
    if (config.center) {
        d_ones_n = DeviceMemory<Scalar>(n);
        std::vector<Scalar> h_ones(n, Scalar(1));
        d_ones_n.upload(h_ones.data(), n);
    }

    // ------------------------------------------------------------------
    // 4. Factor storage
    // ------------------------------------------------------------------

    DenseMatrixS U_all(m, k_max);
    DenseVectorS d_all(k_max);
    DenseMatrixS V_all(n, k_max);
    d_all.setZero();

    DeviceMemory<Scalar> d_U_factors(static_cast<size_t>(m) * k_max);
    DeviceMemory<Scalar> d_V_factors(static_cast<size_t>(n) * k_max);
    DeviceMemory<Scalar> d_sigma_all(k_max);
    DeviceMemory<Scalar> d_defl_workspace(k_max);

    // ------------------------------------------------------------------
    // 5. Speckled CV setup for dense matrix (iterate all m*n entries)
    // ------------------------------------------------------------------

    const bool do_cv = config.is_cv();
    int cv_count = 0;
    DeviceMemory<int> d_cv_rows, d_cv_cols;
    DeviceMemory<Scalar> d_cv_residuals;
    DeviceMemory<Scalar> d_cv_mse_accum(1);

    Scalar best_test_loss = std::numeric_limits<Scalar>::max();
    int best_k = 0;
    int patience_counter = 0;

    if (do_cv) {
        // For dense matrices: all m*n entries are valid holdout candidates
        FactorNet::nmf::LazySpeckledMask<Scalar> mask(
            m, n, static_cast<int64_t>(m) * n,
            static_cast<double>(config.test_fraction),
            config.effective_cv_seed(),
            false);  // mask_zeros=false: all entries non-zero for dense

        std::vector<int> te_rows, te_cols;
        std::vector<Scalar> te_residuals;
        te_rows.reserve(static_cast<size_t>(m) * n / 10);
        te_cols.reserve(static_cast<size_t>(m) * n / 10);
        te_residuals.reserve(static_cast<size_t>(m) * n / 10);

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if (mask.is_holdout(i, j)) {
                    Scalar val = h_A[i + static_cast<size_t>(j) * m];
                    if (config.center) val -= row_means(i);
                    te_rows.push_back(i);
                    te_cols.push_back(j);
                    te_residuals.push_back(val);
                }
            }
        }

        cv_count = static_cast<int>(te_rows.size());
        if (cv_count > 0) {
            d_cv_rows = DeviceMemory<int>(cv_count);
            d_cv_cols = DeviceMemory<int>(cv_count);
            d_cv_residuals = DeviceMemory<Scalar>(cv_count);
            d_cv_rows.upload(te_rows.data(), cv_count);
            d_cv_cols.upload(te_cols.data(), cv_count);
            d_cv_residuals.upload(te_residuals.data(), cv_count);

            // Zero holdout entries in d_A_dense so ALS trains on observed data only.
            // Build a host copy with holdouts zeroed, then re-upload.
            std::vector<Scalar> h_A_train(h_A, h_A + mn);
            for (int idx = 0; idx < cv_count; ++idx) {
                h_A_train[te_rows[idx] + static_cast<size_t>(te_cols[idx]) * m] = Scalar(0);
            }
            d_A_dense.upload(h_A_train.data(), mn);
        }
    }

    // ------------------------------------------------------------------
    // 5b. CV denominator correction
    // ------------------------------------------------------------------

    Scalar cv_denom_correction = Scalar(1);
    DeviceMemory<Scalar> d_cv_denom_scalar(1);
    if (do_cv) {
        // Dense matrices: all entries valid, so effective holdout = test_fraction
        if (config.mask_zeros) {
            // For dense with mask_zeros, density ≈ 1 so correction ≈ 1 - test_fraction
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        } else {
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        }
        d_cv_denom_scalar.upload(&cv_denom_correction, 1);
        if (config.verbose) {
            FACTORNET_PRINT_IMPL("  GPU dense CV: %d entries zeroed, denom_correction=%.4f\n",
                                 cv_count, static_cast<double>(cv_denom_correction));
        }
    }

    // Device scalar containing 1.0 — used as norm_sq for u-side regularization
    DeviceMemory<Scalar> d_one_scalar(1);
    { Scalar h_one = Scalar(1); d_one_scalar.upload(&h_one, 1); }

    // Angular penalty workspace
    DeviceMemory<Scalar> d_angular_workspace(k_max);

    // ------------------------------------------------------------------
    // 6. RNG for initialization
    // ------------------------------------------------------------------

    rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
    std::vector<Scalar> h_u_init(m);
    const int BLOCK_SIZE = 256;
    int k_computed = 0;

    // ------------------------------------------------------------------
    // 7. Main deflation loop — identical to sparse except GEMV replaces SpMV
    // ------------------------------------------------------------------

    for (int k = 0; k < k_max; ++k) {

        for (int i = 0; i < m; ++i)
            h_u_init[i] = static_cast<Scalar>(rng.uniform<double>());
        Scalar init_norm = 0;
        for (int i = 0; i < m; ++i) init_norm += h_u_init[i] * h_u_init[i];
        init_norm = std::sqrt(init_norm);
        if (init_norm > 0)
            for (int i = 0; i < m; ++i) h_u_init[i] /= init_norm;
        d_u.upload(h_u_init.data(), m);

        Scalar sigma = 0;
        int iter = 0;

        for (iter = 0; iter < config.max_iter; ++iter) {

            CUDA_CHECK(cudaMemcpyAsync(d_u_old.get(), d_u.get(),
                m * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));

            // V-update: spmv_result_n = A^T * u  (GEMV transpose)
            detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_T,
                m, n, Scalar(1), d_A_dense.get(), m,
                d_u.get(), 1, Scalar(0), d_spmv_result_n.get(), 1);

            if (config.center) {
                detail::cublas_dot_device<Scalar>(
                    ctx.cublas, m, d_row_means.get(), d_u.get(), d_scalars.get());
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::subtract_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_spmv_result_n.get(), n, d_scalars.get());
            }

            if (k > 0) {
                detail::deflation_correct_gpu<Scalar>(ctx,
                    d_spmv_result_n.get(), d_u.get(),
                    d_V_factors.get(), d_sigma_all.get(), d_U_factors.get(),
                    n, m, k, d_defl_workspace.get());
            }

            // u_sq = ||u||^2 → device scalar
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, m, d_u.get(), d_u.get(), d_scalars.get());

            // CV correction: scale u_sq by correction factor
            if (do_cv) detail::cublas_scal<Scalar>(ctx.cublas, 1, cv_denom_correction, d_scalars.get());

            // v = spmv_result_n / u_sq (fused: reads u_sq from device memory)
            {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_div_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_v.get(), d_spmv_result_n.get(), d_scalars.get(), n);
            }

            // Apply v-side regularization (full feature set matching sparse path)
            if (config.has_regularization_v()) {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if (config.L21_v > 0) {
                    detail::cublas_nrm2_device<Scalar>(
                        ctx.cublas, n, d_v.get(), d_scalars.get() + 6);
                    detail::regularize_with_l21_device_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_v.get(), n,
                        config.L1_v, config.L2_v, config.L21_v,
                        config.nonneg_v, config.upper_bound_v,
                        d_scalars.get(),       // [0] = u_sq
                        d_scalars.get() + 6);  // [6] = ||v|| for L21
                } else {
                    detail::regularize_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_v.get(), n, config.L1_v, config.L2_v,
                        config.nonneg_v, config.upper_bound_v, d_scalars.get());
                }

                // Angular penalty: project out prior factor components
                if (config.angular_v > 0 && k > 0) {
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_T,
                        n, k, Scalar(1),
                        d_V_factors.get(), n, d_v.get(), 1,
                        Scalar(0), d_angular_workspace.get(), 1);
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_N,
                        n, k, -config.angular_v,
                        d_V_factors.get(), n, d_angular_workspace.get(), 1,
                        Scalar(1), d_v.get(), 1);
                }
            }

            detail::cublas_nrm2_device<Scalar>(ctx.cublas, n, d_v.get(), d_scalars.get() + 1);
            {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_v.get(), d_v.get(), d_scalars.get() + 1, n);
            }

            // U-update: spmv_result_m = A * v  (GEMV non-transpose)
            detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_N,
                m, n, Scalar(1), d_A_dense.get(), m,
                d_v.get(), 1, Scalar(0), d_spmv_result_m.get(), 1);

            if (config.center) {
                detail::cublas_dot_device<Scalar>(
                    ctx.cublas, n, d_ones_n.get(), d_v.get(), d_scalars.get() + 2);
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::subtract_scaled_vector_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_spmv_result_m.get(), d_row_means.get(), m, d_scalars.get() + 2);
            }

            if (k > 0) {
                detail::deflation_correct_gpu<Scalar>(ctx,
                    d_spmv_result_m.get(), d_v.get(),
                    d_U_factors.get(), d_sigma_all.get(), d_V_factors.get(),
                    m, n, k, d_defl_workspace.get());
            }

            // v_sq = ||v||^2 = 1 (since we normalized)
            // CV correction: effective v_sq = cv_denom_correction (not 1)
            if (do_cv) {
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_div_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_u.get(), d_spmv_result_m.get(), d_cv_denom_scalar.get(), m);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_u.get(), d_spmv_result_m.get(),
                    m * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
            }

            // Apply u-side regularization (full feature set matching sparse path)
            if (config.has_regularization_u()) {
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if (config.L21_u > 0) {
                    detail::cublas_nrm2_device<Scalar>(
                        ctx.cublas, m, d_u.get(), d_scalars.get() + 7);
                    detail::regularize_with_l21_device_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_u.get(), m,
                        config.L1_u, config.L2_u, config.L21_u,
                        config.nonneg_u, config.upper_bound_u,
                        do_cv ? d_cv_denom_scalar.get() : d_one_scalar.get(),
                        d_scalars.get() + 7); // [7] = ||u|| for L21
                } else {
                    detail::regularize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_u.get(), m,
                        config.L1_u, config.L2_u,
                        config.nonneg_u, config.upper_bound_u,
                        do_cv ? cv_denom_correction : Scalar(1));
                }

                // Angular penalty: project out prior factor components
                if (config.angular_u > 0 && k > 0) {
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_T,
                        m, k, Scalar(1),
                        d_U_factors.get(), m, d_u.get(), 1,
                        Scalar(0), d_angular_workspace.get(), 1);
                    detail::cublas_gemv_host<Scalar>(ctx.cublas, CUBLAS_OP_N,
                        m, k, -config.angular_u,
                        d_U_factors.get(), m, d_angular_workspace.get(), 1,
                        Scalar(1), d_u.get(), 1);
                }
            }

            detail::cublas_nrm2_device<Scalar>(ctx.cublas, m, d_u.get(), d_scalars.get() + 1);
            {
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_u.get(), d_u.get(), d_scalars.get() + 1, m);
            }

            detail::cublas_dot_device<Scalar>(ctx.cublas, m, d_u.get(), d_u_old.get(), d_scalars.get() + 3);
            detail::cublas_nrm2_device<Scalar>(ctx.cublas, m, d_u.get(), d_scalars.get() + 4);
            detail::cublas_nrm2_device<Scalar>(ctx.cublas, m, d_u_old.get(), d_scalars.get() + 5);

            Scalar h_conv[4];
            CUDA_CHECK(cudaMemcpyAsync(&h_conv[0], d_scalars.get() + 1, sizeof(Scalar),
                cudaMemcpyDeviceToHost, ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(&h_conv[1], d_scalars.get() + 3, 3 * sizeof(Scalar),
                cudaMemcpyDeviceToHost, ctx.stream));
            ctx.sync();

            sigma = h_conv[0];
            if (sigma <= Scalar(0)) break;

            Scalar cos_val = std::abs(h_conv[1]) /
                (h_conv[2] * h_conv[3] + std::numeric_limits<Scalar>::epsilon());
            if (Scalar(1) - cos_val < config.tol) { ++iter; break; }
        }

        // Post-factor Gram-Schmidt reorthogonalization (same as sparse path)
        if (k > 0) {
            Scalar* d_gs_h = d_defl_workspace.get();
            detail::gpu_orthog_against<Scalar>(
                ctx.cublas, d_U_factors.get(), d_u.get(), m, k, d_gs_h);
            {
                detail::cublas_nrm2_device<Scalar>(
                    ctx.cublas, m, d_u.get(), d_scalars.get());
                int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_u.get(), d_u.get(), d_scalars.get(), m);
            }
            detail::gpu_orthog_against<Scalar>(
                ctx.cublas, d_V_factors.get(), d_v.get(), n, k, d_gs_h);
            {
                detail::cublas_nrm2_device<Scalar>(
                    ctx.cublas, n, d_v.get(), d_scalars.get());
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_normalize_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_v.get(), d_v.get(), d_scalars.get(), n);
            }
        }

        // Factor storage (device-only, host download deferred to end)
        k_computed = k + 1;
        result.iters_per_factor.push_back(iter);

        CUDA_CHECK(cudaMemcpyAsync(d_U_factors.get() + static_cast<size_t>(k) * m,
            d_u.get(), m * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_V_factors.get() + static_cast<size_t>(k) * n,
            d_v.get(), n * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_sigma_all.get() + k, d_scalars.get() + 1,
            sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));

        // CV evaluation (identical to sparse path)
        if (do_cv && cv_count > 0) {
            int blocks_cv = (cv_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::update_test_residuals_kernel<<<blocks_cv, BLOCK_SIZE, 0, ctx.stream>>>(
                d_cv_residuals.get(), d_cv_rows.get(), d_cv_cols.get(),
                d_u.get(), sigma, d_v.get(), cv_count);
            CUDA_CHECK(cudaMemsetAsync(d_cv_mse_accum.get(), 0, sizeof(Scalar), ctx.stream));
            size_t smem = BLOCK_SIZE * sizeof(Scalar);
            detail::sum_of_squares_kernel<<<blocks_cv, BLOCK_SIZE, smem, ctx.stream>>>(
                d_cv_residuals.get(), cv_count, d_cv_mse_accum.get());
            ctx.sync();
            Scalar sum_sq;
            d_cv_mse_accum.download(&sum_sq, 1);
            Scalar test_mse = sum_sq / static_cast<Scalar>(cv_count);
            result.test_loss_trajectory.push_back(test_mse);
            if (config.verbose)
                FACTORNET_PRINT_IMPL("  Factor %d: sigma=%.4e  iters=%d  test_mse=%.6e\n",
                    k + 1, static_cast<double>(sigma), iter, static_cast<double>(test_mse));
            if (test_mse < best_test_loss) {
                best_test_loss = test_mse; best_k = k + 1; patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= config.patience) break;
            }
        } else {
            if (config.verbose)
                FACTORNET_PRINT_IMPL("  Factor %d: sigma=%.4e  iters=%d\n",
                    k + 1, static_cast<double>(sigma), iter);
            best_k = k + 1;
        }

        if (sigma < std::numeric_limits<Scalar>::epsilon() * Scalar(100)) break;
    }

    // ------------------------------------------------------------------
    // 8. Bulk-download factors from device → host
    // ------------------------------------------------------------------

    ctx.sync();
    if (k_computed > 0) {
        CUDA_CHECK(cudaMemcpy(U_all.data(), d_U_factors.get(),
            static_cast<size_t>(m) * k_computed * sizeof(Scalar),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(V_all.data(), d_V_factors.get(),
            static_cast<size_t>(n) * k_computed * sizeof(Scalar),
            cudaMemcpyDeviceToHost));
        std::vector<Scalar> h_sigmas(k_computed);
        CUDA_CHECK(cudaMemcpy(h_sigmas.data(), d_sigma_all.get(),
            k_computed * sizeof(Scalar), cudaMemcpyDeviceToHost));
        for (int i = 0; i < k_computed; ++i) d_all(i) = h_sigmas[i];
    }

    // ------------------------------------------------------------------
    // 9. Pack result
    // ------------------------------------------------------------------

    result.k_selected = best_k;
    result.U = U_all.leftCols(best_k);
    result.d = d_all.head(best_k);
    result.V = V_all.leftCols(best_k);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

}  // namespace svd
}  // namespace FactorNet
