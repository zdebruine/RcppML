// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/nnls_batch_zi_irls.cuh
 * @brief GPU-native Zero-Inflated IRLS NNLS + E-step + imputation kernels
 *
 * Implements GPU kernels for ZIGP-NMF and ZINB-NMF:
 *
 *   1. Dense fused IRLS NNLS (G built from scratch — all rows weighted)
 *   2. E-step: compute z_ij posteriors for zero entries
 *   3. M-step: update π from z sums
 *   4. Soft imputation: A_imputed[i,j] = z_ij * μ_ij for zeros
 *   5. Tiled dense transpose kernel
 *
 * Architecture:
 *   - Dense IRLS: 1 block per column, smem = (k²+4k)·sizeof(Scalar)
 *     Same fused approach as sparse IRLS but iterates ALL m rows (not nnz)
 *     G is built from scratch (zero-init) since all m weights differ from 1
 *   - E-step/imputation: 1 block per column, linear scan with nz-skip
 *   - M-step/transpose: simple element-wise kernels
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/types.cuh>
#include <FactorNet/gpu/fused_cv.cuh>      // get_max_smem_per_block, compute_thread_count
#include <FactorNet/primitives/gpu/nnls_batch_irls.cuh>  // compute_irls_weight_device
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>  // host fallback for large-k
#include <FactorNet/math/loss.hpp>                        // LossConfig
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

namespace FactorNet {
namespace primitives {
namespace gpu_zi {

// ============================================================================
// Dense fused IRLS NNLS kernel
//
// Like fused_irls_nnls_kernel but operates on dense A_imputed (m×n).
// G is built from scratch (zero-init) since ALL m rows contribute weighted
// outer products — the base-Gram shortcut doesn't apply.
//
// Shared memory: (k² + 4k) × sizeof(Scalar)
// ============================================================================

template<typename Scalar>
__global__ void fused_zi_irls_nnls_kernel(
    const Scalar* __restrict__ A_imputed,  // m × n col-major dense
    const Scalar* __restrict__ W,          // k × m col-major
    Scalar*       __restrict__ H,          // k × n col-major (read/write)
    int k, int m, int n,
    int loss_type,
    Scalar huber_delta,
    const Scalar* __restrict__ disp_row_ptr,  // per-row dispersion (H-update)
    const Scalar* __restrict__ disp_col_ptr,  // per-col dispersion (W-update)
    Scalar L1, Scalar L2,
    bool nonneg,
    int max_cd_iter)
{
    const int bid = blockIdx.x;
    if (bid >= n) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    extern __shared__ char smem[];
    Scalar* s_G  = reinterpret_cast<Scalar*>(smem);
    Scalar* s_b  = s_G + k * k;
    Scalar* s_h  = s_b + k;
    Scalar* s_db = s_h + k;
    Scalar* s_w  = s_db + k;

    // Phase 0: Zero G and b, warm-start h from H
    for (int i = tid; i < k * k; i += nthreads)
        s_G[i] = Scalar(0);
    for (int i = tid; i < k; i += nthreads) {
        s_b[i] = Scalar(0);
        s_h[i] = H[i + bid * k];
    }
    __syncthreads();

    // Phase 1: Iterate ALL m rows
    const Scalar* a_col = A_imputed + static_cast<size_t>(bid) * m;

    for (int row_i = 0; row_i < m; ++row_i) {
        const Scalar a_val = a_col[row_i];

        __syncthreads();
        for (int f = tid; f < k; f += nthreads)
            s_w[f] = W[f + row_i * k];
        __syncthreads();

        // Prediction
        Scalar mu = Scalar(0);
        for (int f = 0; f < k; ++f)
            mu += s_w[f] * s_h[f];
        mu = max(mu, Scalar(1e-10));

        Scalar disp = disp_col_ptr ? disp_col_ptr[bid]
                    : (disp_row_ptr ? disp_row_ptr[row_i] : Scalar(0));

        Scalar w = gpu_irls::compute_irls_weight_device(
            a_val, mu, loss_type, huber_delta, disp);

        // Accumulate G += w · w_col ⊗ w_col^T (upper triangle)
        for (int idx = tid; idx < k * k; idx += nthreads) {
            int a = idx % k;
            int b = idx / k;
            if (a <= b)
                atomicAdd(&s_G[a + b * k], w * s_w[a] * s_w[b]);
        }

        // Accumulate b += w · a_val · w_col
        Scalar wa = w * a_val;
        for (int f = tid; f < k; f += nthreads)
            atomicAdd(&s_b[f], wa * s_w[f]);
    }
    __syncthreads();

    // Phase 1b: Mirror upper triangle + L2
    for (int idx = tid; idx < k * k; idx += nthreads) {
        int a = idx % k;
        int b = idx / k;
        if (a > b) s_G[a + b * k] = s_G[b + a * k];
    }
    for (int i = tid; i < k; i += nthreads)
        s_G[i + i * k] += L2;
    __syncthreads();

    // L1 on RHS
    if (L1 > Scalar(0)) {
        for (int i = tid; i < k; i += nthreads)
            s_b[i] -= L1;
        __syncthreads();
    }

    // Phase 2: CD NNLS solve
    if (k >= 64) {
        Scalar* s_warp_buf = s_db;
        for (int iter = 0; iter < max_cd_iter; ++iter) {
            for (int coord = 0; coord < k; ++coord) {
                Scalar partial = 0;
                for (int l = tid; l < k; l += nthreads)
                    partial += s_G[coord * k + l] * s_h[l];
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial += __shfl_down_sync(0xffffffff, partial, offset);
                if (lane_id == 0) s_warp_buf[warp_id] = partial;
                __syncthreads();
                if (tid == 0) {
                    Scalar grad = -s_b[coord];
                    for (int ww = 0; ww < nwarps; ++ww) grad += s_warp_buf[ww];
                    Scalar g_diag = s_G[coord * k + coord];
                    if (g_diag > Scalar(0)) {
                        Scalar new_val = s_h[coord] - grad / g_diag;
                        s_h[coord] = nonneg ? max(Scalar(0), new_val) : new_val;
                    }
                }
                __syncthreads();
            }
        }
    } else {
        if (tid == 0) {
            for (int iter = 0; iter < max_cd_iter; ++iter) {
                bool any_changed = false;
                for (int coord = 0; coord < k; ++coord) {
                    Scalar grad = -s_b[coord];
                    for (int l = 0; l < k; ++l)
                        grad += s_G[coord * k + l] * s_h[l];
                    Scalar g_diag = s_G[coord * k + coord];
                    if (g_diag > Scalar(0)) {
                        Scalar old_val = s_h[coord];
                        Scalar new_val = old_val - grad / g_diag;
                        s_h[coord] = nonneg ? max(Scalar(0), new_val) : new_val;
                        if (s_h[coord] != old_val) any_changed = true;
                    }
                }
                if (!any_changed) break;
            }
        }
        __syncthreads();
    }

    // Phase 3: Write back
    for (int i = tid; i < k; i += nthreads)
        H[i + bid * k] = s_h[i];
}


// ============================================================================
// Fused ZI E-step + imputation kernel
//
// Column-parallel: 1 block per column. For each row:
//   - nonzero: copy original value to A_imputed (no z_ij needed)
//   - zero: binary search to identify, compute μ and z_ij, accumulate
//           M-step sums, AND write z_ij * μ to A_imputed
//
// This fuses the formerly separate zi_estep_kernel and zi_impute_kernel,
// eliminating the second binary search + dot product per zero entry.
//
// FUTURE: For very high-nnz columns (>1000 nonzeros), a warp-cooperative
// merge-path approach could replace the per-thread binary search. Each warp
// would collectively merge the sorted row_idx array with its assigned row
// range, amortizing the search cost. Currently not implemented because the
// per-thread binary search (O(log nnz_col) per row) is fast enough for
// typical scRNA-seq sparsity patterns (1-10% density → ~10-15 comparisons).
// ============================================================================

template<typename Scalar>
__global__ void zi_estep_impute_kernel(
    // Sparse A (CSC) — used to identify nonzero positions + values
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    // Factor matrices (d-scaled W for prediction)
    const Scalar* __restrict__ W_Td,    // k × m col-major (W * diag(d))
    const Scalar* __restrict__ H,       // k × n col-major
    // Dispersion
    const Scalar* __restrict__ theta,   // m-length (GP θ or NB r)
    // ZI parameters
    const Scalar* __restrict__ pi_row,  // m-length or nullptr
    const Scalar* __restrict__ pi_col,  // n-length or nullptr
    // Outputs: M-step accumulators
    Scalar*       __restrict__ z_sum_row, // m-length (atomicAdd)
    Scalar*       __restrict__ z_sum_col, // n-length (block reduction)
    int*          __restrict__ zc_row,    // m-length zero counts
    int*          __restrict__ zc_col,    // n-length zero counts
    // Output: imputed matrix
    Scalar*       __restrict__ A_imputed,  // m × n col-major dense
    // Dimensions
    int k, int m, int n,
    int loss_type,   // 4=GP, 5=NB
    int zi_mode)     // 1=ROW, 2=COL, 3=TWOWAY
{
    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int nz_start = col_ptr[j];
    const int nz_end   = col_ptr[j + 1];

    // Load H column into shared memory for dot products
    extern __shared__ char smem[];
    Scalar* s_h = reinterpret_cast<Scalar*>(smem);  // k scalars

    for (int f = tid; f < k; f += nthreads)
        s_h[f] = H[f + j * k];
    __syncthreads();

    // Per-thread accumulators for column j
    Scalar my_z_col_sum = Scalar(0);
    int my_zc_col = 0;

    Scalar* out_col = A_imputed + static_cast<size_t>(j) * m;

    for (int i = tid; i < m; i += nthreads) {
        // Binary search to check if i is a nonzero in this column
        bool is_nonzero = false;
        Scalar orig_val = Scalar(0);
        {
            int lo = nz_start, hi = nz_end;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                int r = row_idx[mid];
                if (r == i) { is_nonzero = true; orig_val = values[mid]; break; }
                if (r < i) lo = mid + 1; else hi = mid;
            }
        }

        if (is_nonzero) {
            // Nonzero: copy original value to imputed matrix
            out_col[i] = orig_val;
        } else {
            // Compute prediction: μ_ij = W_Td(:,i) · H(:,j)
            Scalar mu = Scalar(0);
            for (int f = 0; f < k; ++f)
                mu += W_Td[f + i * k] * s_h[f];
            mu = max(mu, Scalar(1e-10));

            // P(Y=0|μ)
            Scalar p0;
            Scalar disp_i = theta[i];
            if (loss_type == 5) {
                // NB: p0 = (r/(r+μ))^r
                Scalar r = max(disp_i, Scalar(1e-10));
                p0 = pow(r / (r + mu), r);
            } else {
                // GP: p0 = exp(-μ/(1+θ))
                Scalar otp = Scalar(1) + disp_i;
                p0 = exp(-mu / otp);
            }

            // Combined π
            Scalar pi_ij;
            if (zi_mode == 1) {  // ZI_ROW
                pi_ij = pi_row[i];
            } else if (zi_mode == 2) {  // ZI_COL
                pi_ij = pi_col[j];
            } else {  // ZI_TWOWAY
                Scalar pr = pi_row[i];
                Scalar pc = pi_col[j];
                pi_ij = Scalar(1) - (Scalar(1) - pr) * (Scalar(1) - pc);
            }

            // Posterior z_ij = π / (π + (1-π)·p0)
            Scalar z_ij = pi_ij / (pi_ij + (Scalar(1) - pi_ij) * p0 + Scalar(1e-30));

            // Accumulate M-step sums
            atomicAdd(&z_sum_row[i], z_ij);
            atomicAdd(&zc_row[i], 1);
            my_z_col_sum += z_ij;
            my_zc_col++;

            // Write imputed value
            out_col[i] = z_ij * mu;
        }
    }

    // Block reduction for column sum (avoids n atomicAdds per thread)
    __syncthreads();
    Scalar* s_warp_buf = s_h;  // reuse smem (done with H column)

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        my_z_col_sum += __shfl_down_sync(0xffffffff, my_z_col_sum, offset);

    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int nwarps = nthreads >> 5;

    if (lane_id == 0) s_warp_buf[warp_id] = my_z_col_sum;
    __syncthreads();

    if (tid == 0) {
        Scalar total = Scalar(0);
        for (int w = 0; w < nwarps; ++w) total += s_warp_buf[w];
        z_sum_col[j] = total;
    }

    // Same reduction for zero count
    for (int offset = 16; offset > 0; offset >>= 1)
        my_zc_col += __shfl_down_sync(0xffffffff, my_zc_col, offset);

    if (lane_id == 0)
        s_warp_buf[warp_id] = static_cast<Scalar>(my_zc_col);
    __syncthreads();

    if (tid == 0) {
        int total_zc = 0;
        for (int w = 0; w < nwarps; ++w) total_zc += static_cast<int>(s_warp_buf[w]);
        zc_col[j] = total_zc;
    }
}


// ============================================================================
// ZI M-step kernel: update π from E-step z sums
// ============================================================================

template<typename Scalar>
__global__ void zi_mstep_row_kernel(
    Scalar*       __restrict__ pi_row,
    const Scalar* __restrict__ z_sum_row,
    const int*    __restrict__ zc_row,
    int m, int n,
    int zi_mode,      // 1=ROW, 3=TWOWAY
    Scalar pi_min,    // 0.001 for ROW, 0.001 for TWOWAY
    Scalar pi_max,    // 0.999 for ROW, 0.5 for TWOWAY
    Scalar max_step)  // 1.0 for ROW (no damping), 0.05 for TWOWAY
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    if (zc_row[i] > 0) {
        Scalar new_pi = z_sum_row[i] / static_cast<Scalar>(n);
        if (zi_mode == 3) {
            // TWOWAY: EMA damping for smooth convergence (mirrors CPU fix)
            Scalar old_pi = pi_row[i];
            new_pi = Scalar(0.9) * new_pi + Scalar(0.1) * old_pi;
        }
        pi_row[i] = max(pi_min, min(pi_max, new_pi));
    }
}

template<typename Scalar>
__global__ void zi_mstep_col_kernel(
    Scalar*       __restrict__ pi_col,
    const Scalar* __restrict__ z_sum_col,
    const int*    __restrict__ zc_col,
    int m, int n,
    int zi_mode,
    Scalar pi_min,
    Scalar pi_max,
    Scalar max_step)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    if (zc_col[j] > 0) {
        Scalar new_pi = z_sum_col[j] / static_cast<Scalar>(m);
        if (zi_mode == 3) {
            // TWOWAY: EMA damping for smooth convergence (mirrors CPU fix)
            Scalar old_pi = pi_col[j];
            new_pi = Scalar(0.9) * new_pi + Scalar(0.1) * old_pi;
        }
        pi_col[j] = max(pi_min, min(pi_max, new_pi));
    }
}

// Dispersion floor
template<typename Scalar>
__global__ void apply_theta_floor_kernel(
    Scalar* __restrict__ theta, int m, Scalar floor_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && theta[i] < floor_val) theta[i] = floor_val;
}


// (zi_impute_kernel removed — fused into zi_estep_impute_kernel above)


// ============================================================================
// Dense transpose via cublasSgeam (replaces manual tiled kernel)
//
// cublasSgeam: C = α·op(A) + β·op(B).  With α=1, β=0, op(A)=transpose,
// this is a single cuBLAS call that handles coalescing internally.
// ============================================================================

template<typename Scalar>
void transpose_dense_gpu(
    gpu::GPUContext& ctx,
    const gpu::DenseMatrixGPU<Scalar>& in,   // m × n
    gpu::DenseMatrixGPU<Scalar>& out)         // n × m
{
    const int m = in.rows;
    const int n = in.cols;
    const Scalar alpha = Scalar(1);
    const Scalar beta  = Scalar(0);

    // cublasSgeam: C(n×m) = α · A^T(n×m) + β · B(n×m)
    // A is m×n col-major, lda=m; C is n×m col-major, ldc=n
    // B is unused (β=0), pass C as both B and C
    cublasStatus_t stat;
    if constexpr (std::is_same_v<Scalar, float>) {
        stat = cublasSgeam(ctx.cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, m,                       // output dimensions (rows, cols)
            &alpha,
            in.data.get(), m,           // A (m×n), lda=m, transposed
            &beta,
            out.data.get(), n,          // B (unused, β=0), same as C
            out.data.get(), n);         // C (n×m), ldc=n
    } else {
        stat = cublasDgeam(ctx.cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, m,
            &alpha,
            in.data.get(), m,
            &beta,
            out.data.get(), n,
            out.data.get(), n);
    }
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgeam transpose failed");
    }
}

/**
 * @brief GPU-native dense IRLS batch NNLS for ZI-imputed A.
 *
 * Identical API pattern to sparse nnls_batch_irls_gpu, but operates on
 * A_imputed (dense m×n) instead of sparse A (CSC).
 */
template<typename Scalar>
void nnls_batch_zi_irls_gpu(
    gpu::GPUContext& ctx,
    const gpu::DenseMatrixGPU<Scalar>& A_imputed,  // m × n dense
    const gpu::DenseMatrixGPU<Scalar>& W,           // k × m
    gpu::DenseMatrixGPU<Scalar>& H,                 // k × n
    int loss_type,
    Scalar huber_delta,
    const gpu::DeviceMemory<Scalar>* disp_row,
    const gpu::DeviceMemory<Scalar>* disp_col,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit,
    int irls_max,
    Scalar cd_tol = Scalar(1e-4),
    Scalar irls_tol = Scalar(1e-3),
    int threads = 1)
{
    const int k = W.rows;
    const int m = A_imputed.rows;
    const int n = A_imputed.cols;

    const Scalar* disp_row_ptr = disp_row ? disp_row->get() : nullptr;
    const Scalar* disp_col_ptr = disp_col ? disp_col->get() : nullptr;

    const size_t fused_smem = (static_cast<size_t>(k) * k + 4 * k) * sizeof(Scalar);
    const int max_smem = gpu::get_max_smem_per_block();

    if (fused_smem <= static_cast<size_t>(max_smem)) {
        // Fused path
        if (fused_smem > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                fused_zi_irls_nnls_kernel<Scalar>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(fused_smem)));
        }

        // For dense ZI, use more threads since we iterate ALL m rows
        int threads = 256;
        if (k >= 128) threads = 512;

        for (int irls_iter = 0; irls_iter < irls_max; ++irls_iter) {
            fused_zi_irls_nnls_kernel<Scalar>
                <<<n, threads, fused_smem, ctx.stream>>>(
                A_imputed.data.get(), W.data.get(), H.data.get(),
                k, m, n,
                loss_type, huber_delta,
                disp_row_ptr, disp_col_ptr,
                L1, L2, nonneg, cd_maxit);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        // Tiled Cholesky fallback for large k — same as sparse IRLS fallback
        // but with dense delta accumulation
        const int kk = k * k;

        gpu::DeviceMemory<Scalar> d_delta_G(static_cast<size_t>(kk) * n);
        gpu::DeviceMemory<Scalar> d_B_weighted(static_cast<size_t>(k) * n);

        // Use context's cuSOLVER handle
        cusolverDnHandle_t cusolver = ctx.cusolver;

        const int tile_size = std::min(8192, n);
        gpu::DeviceMemory<Scalar> G_batch(static_cast<size_t>(tile_size) * kk);
        gpu::DeviceMemory<Scalar> b_batch(static_cast<size_t>(tile_size) * k);
        gpu::DeviceMemory<Scalar*> G_ptrs_d(tile_size);
        gpu::DeviceMemory<Scalar*> b_ptrs_d(tile_size);
        gpu::DeviceMemory<int> d_info(tile_size);

        std::vector<Scalar*> G_ptrs_h(tile_size), b_ptrs_h(tile_size);
        for (int j = 0; j < tile_size; ++j) {
            G_ptrs_h[j] = G_batch.get() + static_cast<size_t>(j) * kk;
            b_ptrs_h[j] = b_batch.get() + static_cast<size_t>(j) * k;
        }
        CUDA_CHECK(cudaMemcpy(G_ptrs_d.get(), G_ptrs_h.data(),
                              tile_size * sizeof(Scalar*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_ptrs_d.get(), b_ptrs_h.data(),
                              tile_size * sizeof(Scalar*), cudaMemcpyHostToDevice));

        // Build G_base = 0 for dense ZI (all rows weighted, no base)
        // Actually we need per-column weighted Gram — delta kernel computes full G
        gpu::DenseMatrixGPU<Scalar> G_zero(k, k);
        G_zero.zero();

        constexpr int TH = 256;

        for (int irls_iter = 0; irls_iter < irls_max; ++irls_iter) {

            // Phase 1: Compute per-column weighted G and B in global memory
            // We reuse the sparse delta kernel logic but on dense A_imputed
            // For simplicity, use the fused kernel with n=1 block approach
            // But actually we need a dense-specific delta kernel
            // FALLBACK: use the fused path per-column but accumulate to global
            // For very large k (>192), this is rare — just do it simply:
            {
                const size_t delta_smem = 2 * k * sizeof(Scalar);
                int delta_threads = 256;
                if (k >= 128) delta_threads = 512;

                // Dense delta kernel: 1 block per col, iterate all m rows
                // Reuses the fused kernel structure but writes to global instead of solving
                // We need a dedicated kernel for this...

                // ALTERNATIVE: Just download to host and use CPU Cholesky
                // For k>192, this is extremely rare in practice; fall back to host
                ctx.sync();

                // Host-mediated fallback for very large k ZI
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> h_A_imp(m, n);
                A_imputed.download_to(h_A_imp.data());

                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> h_W(k, m);
                W.download_to(h_W.data());

                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> h_H(k, n);
                H.download_to(h_H.data());

                std::vector<Scalar> h_disp(m, Scalar(0));
                if (disp_row) disp_row->download(h_disp.data(), m);

                // CPU dense IRLS (host fallback for very large k)
                FactorNet::LossConfig<Scalar> loss_cfg;
                loss_cfg.type = static_cast<FactorNet::LossType>(loss_type);
                loss_cfg.huber_delta = huber_delta;

                DenseMatrix<Scalar> G_base = h_W * h_W.transpose();
                FactorNet::primitives::nnls_batch_irls_dense(
                    h_A_imp, h_W,
                    G_base,
                    h_H,
                    loss_cfg,
                    L1, L2, nonneg,
                    cd_maxit, cd_tol,
                    irls_max, irls_tol,
                    threads,
                    h_disp.data(), static_cast<const Scalar*>(nullptr));

                H.upload_from(h_H.data());
            }
        }
    }

    ctx.sync();
}


// ============================================================================
// Full ZI EM + imputation step (launched from fit_gpu.cuh)
// ============================================================================

/**
 * @brief Run ZI E-step + M-step + imputation on GPU.
 *
 * Uses the fused zi_estep_impute_kernel which computes z_ij posteriors,
 * accumulates M-step sums, AND writes the imputed matrix in a single pass.
 * Runs zi_em_iters+1 total fused calls:  iterations 0..zi_em_iters-1 are
 * followed by M-step updates; the final call (iteration zi_em_iters) only
 * produces the imputed matrix using the latest π (M-step sums discarded).
 *
 * @param d_W_Td  D-scaled W (k × m), already computed by caller
 */
template<typename Scalar>
void zi_em_step_gpu(
    gpu::GPUContext& ctx,
    const gpu::SparseMatrixGPU<Scalar>& d_A,     // original sparse A (CSC)
    const gpu::DenseMatrixGPU<Scalar>& d_W_Td,   // k × m (d-scaled)
    const gpu::DenseMatrixGPU<Scalar>& d_H,       // k × n
    const gpu::DeviceMemory<Scalar>& d_theta,     // m-length dispersion
    gpu::DeviceMemory<Scalar>& d_pi_row,           // m-length (or size 0)
    gpu::DeviceMemory<Scalar>& d_pi_col,           // n-length (or size 0)
    gpu::DenseMatrixGPU<Scalar>& d_A_imputed,     // m × n dense output
    int k, int m, int n,
    int loss_type,
    int zi_mode,
    int zi_em_iters,
    Scalar gp_theta_min)
{
    constexpr int TH = 256;
    const size_t fused_smem = k * sizeof(Scalar);  // H column in smem

    // Scratch for M-step accumulators
    gpu::DeviceMemory<Scalar> d_z_sum_row(m);
    gpu::DeviceMemory<Scalar> d_z_sum_col(n);
    gpu::DeviceMemory<int> d_zc_row(m);
    gpu::DeviceMemory<int> d_zc_col(n);

    const Scalar* pi_row_ptr = d_pi_row.size() > 0 ? d_pi_row.get() : nullptr;
    const Scalar* pi_col_ptr = d_pi_col.size() > 0 ? d_pi_col.get() : nullptr;

    if (fused_smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            zi_estep_impute_kernel<Scalar>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(fused_smem)));
    }

    // zi_em_iters+1 total passes: iterations 0..zi_em_iters-1 include M-step;
    // the final pass (iter == zi_em_iters) only produces the imputed matrix.
    for (int zi_iter = 0; zi_iter <= zi_em_iters; ++zi_iter) {
        // Zero accumulators
        d_z_sum_row.zero();
        d_z_sum_col.zero();
        CUDA_CHECK(cudaMemsetAsync(d_zc_row.get(), 0, m * sizeof(int), ctx.stream));
        CUDA_CHECK(cudaMemsetAsync(d_zc_col.get(), 0, n * sizeof(int), ctx.stream));

        // Fused E-step + imputation
        zi_estep_impute_kernel<Scalar><<<n, TH, fused_smem, ctx.stream>>>(
            d_A.col_ptr.get(), d_A.row_indices.get(), d_A.values.get(),
            d_W_Td.data.get(), d_H.data.get(),
            d_theta.get(),
            pi_row_ptr, pi_col_ptr,
            d_z_sum_row.get(), d_z_sum_col.get(),
            d_zc_row.get(), d_zc_col.get(),
            d_A_imputed.data.get(),
            k, m, n, loss_type, zi_mode);
        CUDA_CHECK(cudaGetLastError());

        // M-step: update π (skip on the final pass — it only produces imputed A)
        if (zi_iter < zi_em_iters) {
            if (zi_mode == 1 || zi_mode == 3) {  // ROW or TWOWAY
                Scalar pi_max = (zi_mode == 3) ? Scalar(0.95) : Scalar(0.999);
                Scalar max_step = (zi_mode == 3) ? Scalar(0.05) : Scalar(1.0);  // unused for TWOWAY (EMA)
                int blocks = (m + TH - 1) / TH;
                zi_mstep_row_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                    d_pi_row.get(), d_z_sum_row.get(), d_zc_row.get(),
                    m, n, zi_mode,
                    Scalar(0.001), pi_max, max_step);
            }
            if (zi_mode == 2 || zi_mode == 3) {  // COL or TWOWAY
                Scalar pi_max = (zi_mode == 3) ? Scalar(0.95) : Scalar(0.999);
                Scalar max_step = (zi_mode == 3) ? Scalar(0.05) : Scalar(1.0);  // unused for TWOWAY (EMA)
                int blocks = (n + TH - 1) / TH;
                zi_mstep_col_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                    d_pi_col.get(), d_z_sum_col.get(), d_zc_col.get(),
                    m, n, zi_mode,
                    Scalar(0.001), pi_max, max_step);
            }

            // Apply theta floor
            if (gp_theta_min > Scalar(0)) {
                int blocks = (m + TH - 1) / TH;
                apply_theta_floor_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                    const_cast<Scalar*>(d_theta.get()), m, gp_theta_min);
            }
        }
    }
}

// ============================================================================
// Dense ZI E-step + imputation kernel
//
// Like zi_estep_impute_kernel but for dense input matrix A (no CSC lookup).
// For each (i,j): if A[i,j]==0 → compute z_ij posterior and impute.
// ============================================================================

template<typename Scalar>
__global__ void zi_estep_dense_kernel(
    const Scalar* __restrict__ A_dense,    // m × n col-major
    const Scalar* __restrict__ W_Td,       // k × m col-major (W * diag(d))
    const Scalar* __restrict__ H,          // k × n col-major
    const Scalar* __restrict__ theta,      // m-length (GP θ or NB r)
    const Scalar* __restrict__ pi_row,     // m-length or nullptr
    const Scalar* __restrict__ pi_col,     // n-length or nullptr
    Scalar*       __restrict__ z_sum_row,  // m-length (atomicAdd)
    Scalar*       __restrict__ z_sum_col,  // n-length (block reduction)
    int*          __restrict__ zc_row,     // m-length zero counts
    int*          __restrict__ zc_col,     // n-length zero counts
    Scalar*       __restrict__ A_imputed,  // m × n col-major dense output
    int k, int m, int n,
    int loss_type,   // 4=GP, 5=NB
    int zi_mode)     // 1=ROW, 2=COL, 3=TWOWAY
{
    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem[];
    Scalar* s_h = reinterpret_cast<Scalar*>(smem);

    for (int f = tid; f < k; f += nthreads)
        s_h[f] = H[f + j * k];
    __syncthreads();

    Scalar my_z_col_sum = Scalar(0);
    int my_zc_col = 0;

    const Scalar* a_col = A_dense + static_cast<size_t>(j) * m;
    Scalar* out_col = A_imputed + static_cast<size_t>(j) * m;

    for (int i = tid; i < m; i += nthreads) {
        Scalar a_val = a_col[i];

        if (a_val != Scalar(0)) {
            out_col[i] = a_val;
        } else {
            Scalar mu = Scalar(0);
            for (int f = 0; f < k; ++f)
                mu += W_Td[f + i * k] * s_h[f];
            mu = max(mu, Scalar(1e-10));

            Scalar p0;
            Scalar disp_i = theta[i];
            if (loss_type == 5) {
                Scalar r = max(disp_i, Scalar(1e-10));
                p0 = pow(r / (r + mu), r);
            } else {
                Scalar otp = Scalar(1) + disp_i;
                p0 = exp(-mu / otp);
            }

            Scalar pi_ij;
            if (zi_mode == 1)
                pi_ij = pi_row[i];
            else if (zi_mode == 2)
                pi_ij = pi_col[j];
            else {
                Scalar pr = pi_row[i];
                Scalar pc = pi_col[j];
                pi_ij = Scalar(1) - (Scalar(1) - pr) * (Scalar(1) - pc);
            }

            Scalar z_ij = pi_ij / (pi_ij + (Scalar(1) - pi_ij) * p0 + Scalar(1e-30));

            atomicAdd(&z_sum_row[i], z_ij);
            atomicAdd(&zc_row[i], 1);
            my_z_col_sum += z_ij;
            my_zc_col++;

            out_col[i] = z_ij * mu;
        }
    }

    // Block reduction for column sums
    __syncthreads();
    Scalar* s_warp_buf = s_h;

    for (int offset = 16; offset > 0; offset >>= 1)
        my_z_col_sum += __shfl_down_sync(0xffffffff, my_z_col_sum, offset);

    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int nwarps = nthreads >> 5;

    if (lane_id == 0) s_warp_buf[warp_id] = my_z_col_sum;
    __syncthreads();

    if (tid == 0) {
        Scalar total = Scalar(0);
        for (int w = 0; w < nwarps; ++w) total += s_warp_buf[w];
        z_sum_col[j] = total;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        my_zc_col += __shfl_down_sync(0xffffffff, my_zc_col, offset);

    if (lane_id == 0)
        s_warp_buf[warp_id] = static_cast<Scalar>(my_zc_col);
    __syncthreads();

    if (tid == 0) {
        int total_zc = 0;
        for (int w = 0; w < nwarps; ++w) total_zc += static_cast<int>(s_warp_buf[w]);
        zc_col[j] = total_zc;
    }
}

// ============================================================================
// Dense ZI EM wrapper — like zi_em_step_gpu but for dense A input
// ============================================================================

template<typename Scalar>
void zi_em_step_gpu_dense(
    gpu::GPUContext& ctx,
    const gpu::DenseMatrixGPU<Scalar>& d_A,       // original dense A (m×n)
    const gpu::DenseMatrixGPU<Scalar>& d_W_Td,    // k × m (d-scaled)
    const gpu::DenseMatrixGPU<Scalar>& d_H,       // k × n
    const gpu::DeviceMemory<Scalar>& d_theta,     // m-length dispersion
    gpu::DeviceMemory<Scalar>& d_pi_row,
    gpu::DeviceMemory<Scalar>& d_pi_col,
    gpu::DenseMatrixGPU<Scalar>& d_A_imputed,     // m × n dense output
    int k, int m, int n,
    int loss_type, int zi_mode,
    int zi_em_iters, Scalar gp_theta_min)
{
    constexpr int TH = 256;
    const size_t fused_smem = k * sizeof(Scalar);

    gpu::DeviceMemory<Scalar> d_z_sum_row(m);
    gpu::DeviceMemory<Scalar> d_z_sum_col(n);
    gpu::DeviceMemory<int> d_zc_row(m);
    gpu::DeviceMemory<int> d_zc_col(n);

    const Scalar* pi_row_ptr = d_pi_row.size() > 0 ? d_pi_row.get() : nullptr;
    const Scalar* pi_col_ptr = d_pi_col.size() > 0 ? d_pi_col.get() : nullptr;

    if (fused_smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            zi_estep_dense_kernel<Scalar>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(fused_smem)));
    }

    for (int zi_iter = 0; zi_iter <= zi_em_iters; ++zi_iter) {
        d_z_sum_row.zero();
        d_z_sum_col.zero();
        CUDA_CHECK(cudaMemsetAsync(d_zc_row.get(), 0, m * sizeof(int), ctx.stream));
        CUDA_CHECK(cudaMemsetAsync(d_zc_col.get(), 0, n * sizeof(int), ctx.stream));

        zi_estep_dense_kernel<Scalar><<<n, TH, fused_smem, ctx.stream>>>(
            d_A.data.get(), d_W_Td.data.get(), d_H.data.get(),
            d_theta.get(),
            pi_row_ptr, pi_col_ptr,
            d_z_sum_row.get(), d_z_sum_col.get(),
            d_zc_row.get(), d_zc_col.get(),
            d_A_imputed.data.get(),
            k, m, n, loss_type, zi_mode);
        CUDA_CHECK(cudaGetLastError());

        if (zi_iter < zi_em_iters) {
            if (zi_mode == 1 || zi_mode == 3) {
                Scalar pi_max = (zi_mode == 3) ? Scalar(0.95) : Scalar(0.999);
                Scalar max_step = (zi_mode == 3) ? Scalar(0.05) : Scalar(1.0);
                int blocks = (m + TH - 1) / TH;
                zi_mstep_row_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                    d_pi_row.get(), d_z_sum_row.get(), d_zc_row.get(),
                    m, n, zi_mode,
                    Scalar(0.001), pi_max, max_step);
            }
            if (zi_mode == 2 || zi_mode == 3) {
                Scalar pi_max = (zi_mode == 3) ? Scalar(0.95) : Scalar(0.999);
                Scalar max_step = (zi_mode == 3) ? Scalar(0.05) : Scalar(1.0);
                int blocks = (n + TH - 1) / TH;
                zi_mstep_col_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                    d_pi_col.get(), d_z_sum_col.get(), d_zc_col.get(),
                    m, n, zi_mode,
                    Scalar(0.001), pi_max, max_step);
            }

            if (gp_theta_min > Scalar(0)) {
                int blocks = (m + TH - 1) / TH;
                apply_theta_floor_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                    const_cast<Scalar*>(d_theta.get()), m, gp_theta_min);
            }
        }
    }
}


}  // namespace gpu_zi
}  // namespace primitives
}  // namespace FactorNet
