/**
 * @file k2.cuh
 * @brief Rank-2 GPU NMF specialization for bipartition/dclust.
 *
 * When k=2, NMF reduces to a much simpler problem:
 *   - Gram matrix is 2×2 (4 scalars) → fits in registers
 *   - NNLS is closed-form (no iteration needed when L1=0)
 *   - No shared memory needed for Gram storage
 *   - Each thread can solve one column independently
 *
 * This specialization eliminates:
 *   - Shared memory allocation for G (register-only)
 *   - Coordinate descent iteration (closed-form)
 *   - Block synchronization (fully independent threads)
 *
 * Performance benefit: Expected 2-5x faster than general k=2 path
 * because the general NNLS kernel launches one block per column with
 * 32+ threads, but only thread 0 does useful work. The k=2 kernel
 * uses one thread per column, achieving much higher occupancy.
 *
 * Also provides a fused RHS+NNLS kernel that computes the RHS and
 * solves NNLS in a single pass over the sparse column, avoiding the
 * global memory round-trip for B.
 */

#pragma once

#include "types.cuh"
#include "gram.cuh"
#include "rhs.cuh"
#include "loss.cuh"
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>

#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>
#include <FactorNet/core/logging.hpp>

namespace FactorNet {
namespace gpu {

// ==========================================================================
// Closed-form 2×2 NNLS (device function)
// ==========================================================================

/**
 * @brief Solve 2×2 NNLS: min_{x≥0} ||Gx - b||² via active-set enumeration.
 *
 * For k=2 there are only 4 possible active sets: {}, {0}, {1}, {0,1}.
 * We check the unconstrained solution first, then the two single-variable
 * solutions, then both-zero. No iteration needed.
 *
 * @param G00, G01, G11  Upper triangle of 2×2 Gram
 * @param inv_det        1 / (G00*G11 - G01²)
 * @param b0, b1         RHS (input), solution (output)
 */
template<typename Scalar>
__device__ __forceinline__ void nnls2_device(
    Scalar G00, Scalar G01, Scalar G11, Scalar inv_det,
    Scalar& b0, Scalar& b1)
{
    // Check if unconstrained solution is non-negative
    Scalar x0 = (G11 * b0 - G01 * b1) * inv_det;
    Scalar x1 = (G00 * b1 - G01 * b0) * inv_det;

    if (x0 >= 0 && x1 >= 0) {
        b0 = x0;
        b1 = x1;
        return;
    }

    // Try single-variable solutions
    Scalar s0 = b0 / G00;  // x1 = 0
    Scalar s1 = b1 / G11;  // x0 = 0

    if (s0 >= 0 && s1 >= 0) {
        // Both single-variable solutions valid; pick the one with lower residual
        // Residual for x=[s0,0]: ||Gs0 - b||² = G00*s0² - 2*b0*s0 + b0² + b1²
        //                      = -b0²/G00 + b1² (since s0=b0/G00)
        // Residual for x=[0,s1]: = b0² - b1²/G11
        // Better = lower residual
        if (b0 * b0 / G00 > b1 * b1 / G11) {
            b0 = s0; b1 = 0;
        } else {
            b0 = 0; b1 = s1;
        }
    } else if (s0 >= 0) {
        b0 = s0; b1 = 0;
    } else if (s1 >= 0) {
        b0 = 0; b1 = s1;
    } else {
        b0 = 0; b1 = 0;
    }
}

/**
 * @brief 2×2 NNLS with L1 penalty via coordinate descent (2 coords only).
 *
 * When L1 > 0, closed-form doesn't directly apply with non-negativity + L1.
 * But with only 2 coordinates, 3-5 CD iterations suffice.
 *
 * Standard CD update: x_k^new = max(0, (r_k + G_{kk}*x_k - L1) / G_{kk})
 * Equivalently: x_k^new = max(0, x_k + (r_k - L1) / G_{kk})
 * where r_k = b_k - Σ_j G_{kj} * x_j is the current residual.
 */
template<typename Scalar>
__device__ __forceinline__ void nnls2_cd_device(
    Scalar G00, Scalar G01, Scalar G11,
    Scalar L1,
    Scalar& b0, Scalar& b1)
{
    Scalar x0 = 0, x1 = 0;
    Scalar r0 = b0, r1 = b1;

    for (int iter = 0; iter < 5; ++iter) {
        // Update x0: x0_new = max(0, x0 + (r0 - L1) / G00)
        {
            Scalar x0_new = x0 + (r0 - L1) / G00;
            if (x0_new < 0) x0_new = 0;
            Scalar delta = x0_new - x0;
            r0 -= G00 * delta;
            r1 -= G01 * delta;
            x0 = x0_new;
        }

        // Update x1: x1_new = max(0, x1 + (r1 - L1) / G11)
        {
            Scalar x1_new = x1 + (r1 - L1) / G11;
            if (x1_new < 0) x1_new = 0;
            Scalar delta = x1_new - x1;
            r0 -= G01 * delta;
            r1 -= G11 * delta;
            x1 = x1_new;
        }
    }

    b0 = x0;
    b1 = x1;
}

// ==========================================================================
// Fused RHS + NNLS kernel for k=2 (H update)
// ==========================================================================

/**
 * @brief Fused sparse RHS + 2×2 NNLS kernel for H update.
 *
 * One thread per column. Each thread:
 *   1. Iterates sparse column to build 2-element RHS
 *   2. Solves 2×2 NNLS (closed-form, no shared memory)
 *   3. Writes solution to H
 *
 * Gram matrix (4 scalars) passed via registers/constant memory.
 *
 * @param W        2 × m factor (column-major)
 * @param col_ptr  CSC column pointers
 * @param row_idx  CSC row indices
 * @param values   CSC values
 * @param H        2 × n output (column-major)
 * @param n        Number of columns
 * @param G00,G01,G11  Gram matrix elements
 * @param inv_det  1/(G00*G11 - G01²)
 * @param L1       L1 penalty (0 for closed-form path)
 */
template<typename Scalar>
__global__ void fused_rhs_nnls_k2_h_kernel(
    const Scalar* __restrict__ W,
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const Scalar* __restrict__ values,
    Scalar* __restrict__ H,
    int n,
    Scalar G00, Scalar G01, Scalar G11, Scalar inv_det,
    Scalar L1)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // Build 2-element RHS from sparse column
    Scalar b0 = 0, b1 = 0;
    int start = col_ptr[j];
    int end = col_ptr[j + 1];

    for (int p = start; p < end; ++p) {
        int i = row_idx[p];
        Scalar v = values[p];
        b0 += v * W[0 + i * 2];  // W(0, i)
        b1 += v * W[1 + i * 2];  // W(1, i)
    }

    // Solve 2×2 NNLS
    if (L1 <= 0) {
        nnls2_device(G00, G01, G11, inv_det, b0, b1);
    } else {
        nnls2_cd_device(G00, G01, G11, L1, b0, b1);
    }

    // Write solution
    H[0 + j * 2] = b0;
    H[1 + j * 2] = b1;
}

// ==========================================================================
// Fused RHS + NNLS kernel for k=2 (W update via A^T)
// ==========================================================================

/**
 * @brief Fused sparse RHS + 2×2 NNLS kernel for W update.
 *
 * Same as H-update kernel but operates on A^T (CSC of transposed A).
 * One thread per row of A (= column of A^T).
 */
template<typename Scalar>
__global__ void fused_rhs_nnls_k2_w_kernel(
    const Scalar* __restrict__ H,       // 2 × n
    const int* __restrict__ row_ptr,    // A^T col_ptr (= A CSR row_ptr)
    const int* __restrict__ col_idx,    // A^T row_idx (= A column indices)
    const Scalar* __restrict__ values,  // A^T values
    Scalar* __restrict__ W,             // 2 × m output
    int m,
    Scalar G00, Scalar G01, Scalar G11, Scalar inv_det,
    Scalar L1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    Scalar b0 = 0, b1 = 0;
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    for (int p = start; p < end; ++p) {
        int j = col_idx[p];
        Scalar v = values[p];
        b0 += v * H[0 + j * 2];
        b1 += v * H[1 + j * 2];
    }

    if (L1 <= 0) {
        nnls2_device(G00, G01, G11, inv_det, b0, b1);
    } else {
        nnls2_cd_device(G00, G01, G11, L1, b0, b1);
    }

    W[0 + i * 2] = b0;
    W[1 + i * 2] = b1;
}

// ==========================================================================
// Scale rows kernel for k=2 (register-based, no shared memory)
// ==========================================================================

template<typename Scalar>
__global__ void scale_rows_k2_kernel(
    Scalar* __restrict__ M,     // 2 × n
    Scalar* __restrict__ d,     // 2
    int n)
{
    // Two rows, process in two passes using all threads
    __shared__ Scalar s_sum[2][256];

    int tid = threadIdx.x;

    // Compute row norms
    for (int row = 0; row < 2; ++row) {
        Scalar local_sum = 0;
        for (int j = tid; j < n; j += blockDim.x) {
            Scalar v = M[row + j * 2];
            local_sum += v * v;
        }
        s_sum[row][tid] = local_sum;
    }
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[0][tid] += s_sum[0][tid + s];
            s_sum[1][tid] += s_sum[1][tid + s];
        }
        __syncthreads();
    }

    // Compute scales
    __shared__ Scalar scales[2];
    if (tid == 0) {
        for (int row = 0; row < 2; ++row) {
            Scalar sq = s_sum[row][0];
            Scalar sc = (sq > 0) ? sqrt(static_cast<double>(sq)) : Scalar(1);
            d[row] = sc;
            scales[row] = Scalar(1) / sc;
        }
    }
    __syncthreads();

    // Apply scaling
    for (int j = tid; j < n; j += blockDim.x) {
        M[0 + j * 2] *= scales[0];
        M[1 + j * 2] *= scales[1];
    }
}

// ==========================================================================
// Fused Gram k=2 kernel (single launch for G00, G01, G11)
// ==========================================================================

/**
 * @brief Compute 2×2 Gram matrix (3 unique elements) in a single kernel launch.
 *
 * Replaces 3 separate cuBLAS dot products + 3 implicit syncs with
 * one kernel + one sync. Each thread accumulates partial G00/G01/G11
 * via grid-stride loop, then block-level reduction + atomicAdd.
 *
 * @param F     2 × N factor matrix (column-major: F[row + col*2])
 * @param N     Number of columns
 * @param G3    Output: 3 scalars [G00, G01, G11] (must be zeroed before launch)
 */
template<typename Scalar>
__global__ void fused_gram_k2_kernel(
    const Scalar* __restrict__ F,
    int N,
    Scalar* __restrict__ G3)
{
    // Shared memory for 3 partial sums per thread
    extern __shared__ char gram_k2_raw[];
    Scalar* sG = reinterpret_cast<Scalar*>(gram_k2_raw);

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    Scalar acc00 = 0, acc01 = 0, acc11 = 0;

    for (int i = gid; i < N; i += stride) {
        Scalar f0 = F[i * 2];
        Scalar f1 = F[i * 2 + 1];
        acc00 += f0 * f0;
        acc01 += f0 * f1;
        acc11 += f1 * f1;
    }

    sG[tid * 3]     = acc00;
    sG[tid * 3 + 1] = acc01;
    sG[tid * 3 + 2] = acc11;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sG[tid * 3]     += sG[(tid + s) * 3];
            sG[tid * 3 + 1] += sG[(tid + s) * 3 + 1];
            sG[tid * 3 + 2] += sG[(tid + s) * 3 + 2];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&G3[0], sG[0]);
        atomicAdd(&G3[1], sG[1]);
        atomicAdd(&G3[2], sG[2]);
    }
}

/**
 * @brief Compute 2×2 Gram from factor matrix using fused kernel.
 *
 * @param F       2 × N factor data on device
 * @param N       Number of columns
 * @param d_G3    Device buffer for 3 scalars (pre-allocated)
 * @param G00,G01,G11  Host output scalars
 */
template<typename Scalar>
void compute_gram_k2(
    const GPUContext& ctx,
    const Scalar* F, int N,
    DeviceMemory<Scalar>& d_G3,
    Scalar& G00, Scalar& G01, Scalar& G11)
{
    CUDA_CHECK(cudaMemsetAsync(d_G3.get(), 0, 3 * sizeof(Scalar), ctx.stream));

    constexpr int THREADS = 256;
    int blocks = std::min((N + THREADS - 1) / THREADS, 128);
    size_t smem = 3 * THREADS * sizeof(Scalar);

    fused_gram_k2_kernel<Scalar><<<blocks, THREADS, smem, ctx.stream>>>(
        F, N, d_G3.get());

    Scalar h_G3[3];
    ctx.sync();
    CUDA_CHECK(cudaMemcpy(h_G3, d_G3.get(), 3 * sizeof(Scalar), cudaMemcpyDeviceToHost));
    G00 = h_G3[0];
    G01 = h_G3[1];
    G11 = h_G3[2];
}

// ==========================================================================
// Host dispatch functions
// ==========================================================================

/**
 * @brief k=2 H update: Gram + fused RHS+NNLS (no scaling).
 *
 * All in one call — no separate Gram/RHS/NNLS kernel launches.
 * Scaling is done separately in the fit function after loss computation.
 */
template<typename Scalar>
void update_H_k2_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W,    // 2 × m
    DenseMatrixGPU<Scalar>& H,          // 2 × n
    DeviceMemory<Scalar>& d,            // 2
    DeviceMemory<Scalar>& d_gram_buf,   // 3-scalar device buffer
    Scalar L1 = 0, Scalar L2 = 0)
{
    const int n = A.cols;
    const int m = A.rows;

    // Fused k=2 Gram: single kernel computes G00, G01, G11
    Scalar G00, G01, G11;
    compute_gram_k2(ctx, W.data.get(), m, d_gram_buf, G00, G01, G11);

    // Add regularization
    G00 += L2 + static_cast<Scalar>(1e-15);
    G11 += L2 + static_cast<Scalar>(1e-15);

    Scalar det = G00 * G11 - G01 * G01;
    Scalar inv_det = Scalar(1) / det;

    // Launch fused RHS+NNLS kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_rhs_nnls_k2_h_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        W.data.get(),
        A.col_ptr.get(), A.row_indices.get(), A.values.get(),
        H.data.get(), n,
        G00, G01, G11, inv_det, L1);
    // Note: scaling deferred to caller for loss computation on unscaled H
}

/**
 * @brief k=2 W update: Gram + fused RHS+NNLS (no scaling).
 *
 * Scaling is done separately in the fit function.
 */
template<typename Scalar>
void update_W_k2_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& At,  // A^T as CSC (n×m)
    const DenseMatrixGPU<Scalar>& H,    // 2 × n
    DenseMatrixGPU<Scalar>& W,          // 2 × m
    DeviceMemory<Scalar>& d,            // 2
    DeviceMemory<Scalar>& d_gram_buf,   // 3-scalar device buffer
    Scalar L1 = 0, Scalar L2 = 0)
{
    const int m = At.cols;  // m rows of original A
    const int n = At.rows;  // n cols of original A

    // Fused k=2 Gram: single kernel computes G00, G01, G11
    Scalar G00, G01, G11;
    compute_gram_k2(ctx, H.data.get(), n, d_gram_buf, G00, G01, G11);

    G00 += L2 + static_cast<Scalar>(1e-15);
    G11 += L2 + static_cast<Scalar>(1e-15);

    Scalar det = G00 * G11 - G01 * G01;
    Scalar inv_det = Scalar(1) / det;

    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    fused_rhs_nnls_k2_w_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        H.data.get(),
        At.col_ptr.get(), At.row_indices.get(), At.values.get(),
        W.data.get(), m,
        G00, G01, G11, inv_det, L1);
    // Note: scaling deferred to caller
}

// ==========================================================================
// Complete k=2 NMF fit
// ==========================================================================

/**
 * @brief Full k=2 NMF fit on GPU using specialized kernels.
 *
 * Suitable for bipartition/dclust inner loop where k is always 2.
 * Much faster than general NMF for k=2 due to:
 *   - Register-only Gram (no shared memory)
 *   - Closed-form NNLS (no CD iteration)
 *   - 1 thread per column (full occupancy)
 *   - Fused RHS + NNLS (no B buffer)
 *
 * Uses loss-based early stopping (MSE = ||A-WH||²/(m*n)),
 * matching the monolithic and streaming GPU kernels and CPU.
 */
template<typename Scalar>
FactorNet::NMFResult<Scalar> nmf_fit_k2_gpu(
    GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const SparseMatrixGPU<Scalar>& At,
    Scalar* W_host,   // 2 × m
    Scalar* H_host,   // 2 × n
    Scalar* d_host,   // 2
    int m, int n,
    const FactorNet::NMFConfig<Scalar>& config)
{
    FactorNet::NMFResult<Scalar> result;

    DenseMatrixGPU<Scalar> W(2, m, W_host);
    DenseMatrixGPU<Scalar> H(2, n, H_host);
    DeviceMemory<Scalar> d(2);
    d.upload(d_host, 2);

    Scalar L1_H = static_cast<Scalar>(config.H.L1);
    Scalar L1_W = static_cast<Scalar>(config.W.L1);
    Scalar L2_H = static_cast<Scalar>(config.H.L2);
    Scalar L2_W = static_cast<Scalar>(config.W.L2);

    // Temporary buffers for loss computation via Gram trick
    // G = W'W (2×2), B = W'A (2×n), scratch (4 scalars)
    DenseMatrixGPU<Scalar> G_loss(2, 2);
    DenseMatrixGPU<Scalar> B_loss(2, n);
    DeviceMemory<Scalar> scratch(4);

    // Fused Gram k=2 buffer (3 scalars: G00, G01, G11)
    DeviceMemory<Scalar> d_gram_buf(3);

    double prev_loss = std::numeric_limits<double>::max();

    for (int iter = 0; iter < config.max_iter; ++iter) {
        // H update
        update_H_k2_gpu(ctx, A, W, H, d, d_gram_buf, L1_H, L2_H);

        // Compute loss BEFORE scaling H (matches monolithic pattern)
        // G = W'W (unregularized, for loss only)
        compute_gram_gpu(ctx, W, G_loss);
        // B = W'A (RHS for cross term)
        compute_rhs_forward_gpu(ctx, W, A, B_loss);
        // MSE = ||A - W*H_unscaled||² / (m*n)
        double loss = compute_total_loss_gpu(
            ctx, A, W, H, G_loss, B_loss, scratch, m, n);
        result.loss_history.push_back(loss);

        // Convergence check
        if (iter > 0) {
            double rel_change = std::abs(prev_loss - loss) / (prev_loss + 1e-10);
            if (rel_change < config.tol) {
                result.converged = true;
                result.iterations = iter + 1;
                FACTORNET_GPU_LOG(config.verbose, "K2 GPU NMF converged at iter %d, loss=%.6e\n",
                           iter + 1, loss);
                break;
            }
        }
        prev_loss = loss;

        FACTORNET_GPU_LOG(config.verbose, "  iter %3d / %d  loss = %.6e\n",
                   iter + 1, config.max_iter, loss);

        // Scale H (after loss computation)
        scale_rows_k2_kernel<Scalar><<<1, 256, 0, ctx.stream>>>(
            H.data.get(), d.get(), n);

        // W update (returns unscaled W)
        update_W_k2_gpu(ctx, At, H, W, d, d_gram_buf, L1_W, L2_W);

        // Scale W rows
        scale_rows_k2_kernel<Scalar><<<1, 256, 0, ctx.stream>>>(
            W.data.get(), d.get(), m);
    }

    if (!result.converged) {
        result.iterations = config.max_iter;
    }

    W.download_to(W_host);
    H.download_to(H_host);
    d.download(d_host, 2);

    return result;
}

} // namespace gpu
} // namespace FactorNet
