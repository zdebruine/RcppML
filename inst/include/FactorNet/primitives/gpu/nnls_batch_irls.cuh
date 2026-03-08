// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/nnls_batch_irls.cuh
 * @brief GPU-native IRLS NNLS batch solve for non-MSE losses (GP, NB, MAE, etc.)
 *
 * Architecture mirrors fused_cv.cuh: one thread block per column, fused
 * delta accumulation + CD NNLS solve in shared memory.  Eliminates the
 * O(k²×n) global memory allocation for per-column delta_G and the extra
 * global memory round-trip between the delta and solve phases.
 *
 * For sparse A (CSC), the per-column weighted Gram is decomposed as:
 *
 *   G_j = G_base + Σ_{i∈nz(j)} (w_ij - 1) · W(:,i) · W(:,i)^T
 *
 * Kernel hierarchy (fp32):
 *   Fused kernel:  k ≤ K_MAX_FUSED  (smem = (k²+4k)·4 ≤ GPU max smem)
 *   Tiled Cholesky: k > K_MAX_FUSED  (cuSOLVER batched potrf/potrs)
 *
 *   k ≤ 128:   fits all GPUs (≤ 66 KB)   → fused path
 *   k ≤ 192:   fits A100+/H100 (≤ 148 KB) → fused with opt-in smem
 *   k ≤ 228:   fits H100 only (≤ 210 KB)  → fused with opt-in smem
 *   k > 228:   tiled Cholesky fallback     → cuSOLVER batched
 *
 * Shared memory layout (fused kernel):
 *   s_G   [k×k]  Corrected Gram matrix (G_base + delta accumulated in-place)
 *   s_b   [k]    Weighted RHS
 *   s_h   [k]    Current solution vector (warm start from H)
 *   s_db  [k]    Scratch / warp reduction workspace
 *   s_w   [k]    Scratch for current W column
 *   Total: (k² + 4k) × sizeof(Scalar)
 *
 * CD solver (same as fused_cv.cuh):
 *   k < 64:  Serial CD (thread 0), zero __syncthreads during iteration
 *   k ≥ 64:  Parallel warp-reduced gradient computation
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/types.cuh>
#include <FactorNet/gpu/fused_cv.cuh>      // get_max_smem_per_block, compute_thread_count
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

namespace FactorNet {
namespace primitives {
namespace gpu_irls {

// ============================================================================
// Local clip kernel (avoids cross-include dependency on cholesky_nnls.cuh)
// ============================================================================

template<typename Scalar>
__global__ void irls_clip_nonneg_kernel(Scalar* X, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && X[idx] < Scalar(0)) X[idx] = Scalar(0);
}

// ============================================================================
// IRLS weight computation — device helper
// ============================================================================

/**
 * @brief Compute IRLS weight for a single observation.
 *
 * Same weight definitions as the CPU IRLS path in nnls_batch_irls.hpp.
 */
template<typename Scalar>
__device__ __forceinline__ Scalar compute_irls_weight_device(
    Scalar observed,
    Scalar predicted,
    int loss_type,
    Scalar huber_delta,
    Scalar dispersion,
    Scalar robust_delta = Scalar(0),
    Scalar power_param = Scalar(1.5))
{
    const Scalar eps = Scalar(1e-10);
    predicted = max(predicted, eps);

    // Legacy standalone MAE/HUBER (old API: loss="mae"/"huber")
    switch (loss_type) {
    case 1: {  // MAE (legacy)
        Scalar r = observed - predicted;
        return Scalar(1) / (abs(r) + eps);
    }
    case 2: {  // Huber (legacy)
        Scalar r = observed - predicted;
        return (abs(r) <= huber_delta) ? Scalar(1)
               : huber_delta / (abs(r) + eps);
    }
    default:
        break;
    }

    // Distribution-derived weight
    Scalar w_dist;
    switch (loss_type) {
    case 3:  // KL
        w_dist = Scalar(1) / max(predicted, Scalar(1e-4));
        break;
    case 4: {  // GP
        Scalar w = Scalar(1) / (predicted * predicted);
        if (observed >= Scalar(1)) {
            Scalar denom = predicted + dispersion * observed;
            denom = max(denom, eps);
            w += (observed - Scalar(1)) / (denom * denom);
        }
        w_dist = min(w, Scalar(1e6));
        break;
    }
    case 5: {  // NB
        Scalar r = max(dispersion, eps);
        w_dist = min(r / (predicted * (r + predicted)), Scalar(1e6));
        break;
    }
    case 6:  // Gamma: V(μ) = μ², weight = 1/μ²
        w_dist = min(Scalar(1) / (predicted * predicted), Scalar(1e6));
        break;
    case 7:  // Inverse Gaussian: V(μ) = μ³, weight = 1/μ³
        w_dist = min(Scalar(1) / (predicted * predicted * predicted), Scalar(1e6));
        break;
    case 8:  // Tweedie: V(μ) = μ^p, weight = 1/μ^p
        w_dist = min(Scalar(1) / pow(predicted, power_param), Scalar(1e6));
        break;
    default:
        w_dist = Scalar(1);  // MSE
        break;
    }

    // Apply robust Huber modifier if active
    if (robust_delta > Scalar(0)) {
        Scalar residual = observed - predicted;
        Scalar sd_inv = sqrt(max(w_dist, eps));
        Scalar pearson_r = residual * sd_inv;
        Scalar abs_pr = abs(pearson_r);
        Scalar w_robust = (abs_pr <= robust_delta)
                        ? Scalar(1)
                        : robust_delta / (abs_pr + eps);
        return w_dist * w_robust;
    }

    return w_dist;
}

// ============================================================================
// Fused IRLS kernel: delta accumulation + CD solve in one launch
//
// Mirrors fused_baseline_kernel from fused_cv.cuh:
//   - 1 block per column
//   - atomicAdd for delta accumulation in shared memory
//   - CD solve re-using the same shared memory
//   - Works for any k that fits in shared memory
//
// Shared memory: (k² + 4k) × sizeof(Scalar)
// ============================================================================

template<typename Scalar>
__global__ void fused_irls_nnls_kernel(
    // Sparse matrix A (CSC)
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    // Factor matrices
    const Scalar* __restrict__ W,       // k × m col-major (W^T stored transposed)
    Scalar*       __restrict__ H,       // k × n col-major (read/write)
    // Gram matrix
    const Scalar* __restrict__ G_base,  // k × k col-major
    // Dimensions
    int k, int n,
    // Loss
    int loss_type,
    Scalar huber_delta,
    Scalar robust_delta,
    Scalar power_param,
    // Dispersion (per-row for H-update, per-col for W-update)
    const Scalar* __restrict__ disp_row_ptr,  // m-length or nullptr (H-update)
    const Scalar* __restrict__ disp_col_ptr,  // n-length or nullptr (W-update)
    // Regularization
    Scalar L1, Scalar L2,
    // Constraints
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

    // ---- Shared memory layout: same as fused_cv.cuh ----
    extern __shared__ char smem[];
    Scalar* s_G  = reinterpret_cast<Scalar*>(smem);          // k×k
    Scalar* s_b  = s_G + k * k;                              // k  (weighted RHS)
    Scalar* s_h  = s_b + k;                                  // k  (solution)
    Scalar* s_db = s_h + k;                                  // k  (scratch / warp buf)
    Scalar* s_w  = s_db + k;                                 // k  (W column scratch)

    // ---- Phase 0: Load G_base into shared memory, zero accumulators ----
    for (int i = tid; i < k * k; i += nthreads)
        s_G[i] = G_base[i];
    for (int i = tid; i < k; i += nthreads) {
        s_b[i] = Scalar(0);
        s_h[i] = H[i + bid * k];   // warm start
    }
    __syncthreads();

    // ---- Phase 1: Iterate nonzeros, accumulate delta_G in-place + weighted RHS ----
    const int start = col_ptr[bid];
    const int end   = col_ptr[bid + 1];

    for (int p = start; p < end; ++p) {
        const int row_i = row_idx[p];
        const Scalar a_val = values[p];

        // Load W(:, row_i) into shared scratch
        __syncthreads();
        for (int f = tid; f < k; f += nthreads)
            s_w[f] = W[f + row_i * k];
        __syncthreads();

        // Compute prediction μ = W(:,i)·H(:,j) — all threads compute, use lane 0's result
        Scalar mu = Scalar(0);
        for (int f = 0; f < k; ++f)
            mu += s_w[f] * s_h[f];
        mu = max(mu, Scalar(1e-10));

        // Dispersion: W-update uses per-col (col_j = bid), H-update uses per-row
        Scalar disp = disp_col_ptr ? disp_col_ptr[bid]
                    : (disp_row_ptr ? disp_row_ptr[row_i] : Scalar(0));

        // IRLS weight
        Scalar w = compute_irls_weight_device(a_val, mu, loss_type, huber_delta, disp, robust_delta, power_param);

        // Accumulate in-place Gram correction: G += (w-1) · w_col · w_col^T
        // Only update upper triangle, mirror later (saves ~half the atomics)
        Scalar w_minus_1 = w - Scalar(1);
        for (int idx = tid; idx < k * k; idx += nthreads) {
            int a = idx % k;
            int b = idx / k;
            if (a <= b)  // upper triangle only
                atomicAdd(&s_G[a + b * k], w_minus_1 * s_w[a] * s_w[b]);
        }

        // Accumulate weighted RHS: b += w · a · W(:,i)
        Scalar wa = w * a_val;
        for (int f = tid; f < k; f += nthreads)
            atomicAdd(&s_b[f], wa * s_w[f]);
    }
    __syncthreads();

    // ---- Phase 1b: Mirror upper triangle → full symmetric G ----
    for (int idx = tid; idx < k * k; idx += nthreads) {
        int a = idx % k;
        int b = idx / k;
        if (a > b) s_G[a + b * k] = s_G[b + a * k];
    }

    // Add L2 regularization to diagonal
    for (int i = tid; i < k; i += nthreads)
        s_G[i + i * k] += L2;
    __syncthreads();

    // ---- Phase 2: CD NNLS solve (reuse fused_cv.cuh's solve_nnls_cd pattern) ----
    // Apply L1 to RHS before solving
    if (L1 > Scalar(0)) {
        for (int i = tid; i < k; i += nthreads)
            s_b[i] -= L1;
        __syncthreads();
    }

    // Initialize h (warm start, already loaded in Phase 0)
    // s_db is reused as warp reduction workspace for k >= 64

    if (k >= 64) {
        // Parallel gradient via warp-level reduction
        Scalar* s_warp_buf = s_db;
        for (int iter = 0; iter < max_cd_iter; ++iter) {
            for (int coord = 0; coord < k; ++coord) {
                Scalar partial = 0;
                for (int l = tid; l < k; l += nthreads)
                    partial += s_G[coord * k + l] * s_h[l];
                // Warp-level reduction
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial += __shfl_down_sync(0xffffffff, partial, offset);
                if (lane_id == 0) s_warp_buf[warp_id] = partial;
                __syncthreads();
                if (tid == 0) {
                    Scalar grad = -s_b[coord];
                    for (int w = 0; w < nwarps; ++w) grad += s_warp_buf[w];
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
        // Serial CD for k < 64 (single thread, zero synchronization)
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

    // ---- Phase 3: Write solution back ----
    for (int i = tid; i < k; i += nthreads)
        H[i + bid * k] = s_h[i];
}


// ============================================================================
// Tiled Cholesky fallback for large k (exceeds shared memory)
//
// Two-phase: (1) compute per-column delta_G + weighted RHS in global memory,
// then (2) tile-batched cuSOLVER potrf/potrs + clip.
//
// Uses the same IRLS weight logic but stores corrections in device global mem.
// ============================================================================

/**
 * @brief Phase 1 kernel for Cholesky path: compute delta_G and B_weighted
 *        per column, stored in global memory.
 *
 * One block per column, same atomicAdd pattern as fused kernel but writes
 * results to global arrays instead of keeping them in shared memory.
 */
template<typename Scalar>
__global__ void irls_delta_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ W,
    const Scalar* __restrict__ H,
    Scalar*       __restrict__ delta_G,     // k*k × n (output)
    Scalar*       __restrict__ B_weighted,  // k × n (output)
    int k, int n,
    int loss_type,
    Scalar huber_delta,
    Scalar robust_delta,
    Scalar power_param,
    const Scalar* __restrict__ disp_row_ptr,
    const Scalar* __restrict__ disp_col_ptr)
{
    const int bid = blockIdx.x;
    if (bid >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kk = k * k;

    // Shared memory: just k for h_col + k for w_scratch
    extern __shared__ char smem[];
    Scalar* s_h = reinterpret_cast<Scalar*>(smem);
    Scalar* s_w = s_h + k;

    // Load H column
    for (int i = tid; i < k; i += nthreads) s_h[i] = H[i + bid * k];
    __syncthreads();

    // Zero global outputs for this column
    Scalar* my_dG = delta_G + static_cast<size_t>(bid) * kk;
    Scalar* my_b  = B_weighted + static_cast<size_t>(bid) * k;
    for (int i = tid; i < kk; i += nthreads) my_dG[i] = Scalar(0);
    for (int i = tid; i < k;  i += nthreads) my_b[i]  = Scalar(0);
    __threadfence();    // ensure zeros visible before atomics
    __syncthreads();

    const int start = col_ptr[bid];
    const int end   = col_ptr[bid + 1];

    for (int p = start; p < end; ++p) {
        const int row_i = row_idx[p];
        const Scalar a_val = values[p];

        __syncthreads();
        for (int f = tid; f < k; f += nthreads)
            s_w[f] = W[f + row_i * k];
        __syncthreads();

        Scalar mu = Scalar(0);
        for (int f = 0; f < k; ++f) mu += s_w[f] * s_h[f];
        mu = max(mu, Scalar(1e-10));

        Scalar disp = disp_col_ptr ? disp_col_ptr[bid]
                    : (disp_row_ptr ? disp_row_ptr[row_i] : Scalar(0));
        Scalar w = compute_irls_weight_device(a_val, mu, loss_type, huber_delta, disp, robust_delta, power_param);

        Scalar w_minus_1 = w - Scalar(1);
        for (int idx = tid; idx < kk; idx += nthreads) {
            int a = idx % k, b = idx / k;
            if (a <= b)   // upper triangle
                atomicAdd(&my_dG[a + b * k], w_minus_1 * s_w[a] * s_w[b]);
        }
        Scalar wa = w * a_val;
        for (int f = tid; f < k; f += nthreads)
            atomicAdd(&my_b[f], wa * s_w[f]);
    }
}

/**
 * @brief Phase 2 kernel: form G_j = G_base + delta_G_j + L2·I for tiled Cholesky.
 *        Also mirrors upper triangle and applies L1 to RHS.
 */
template<typename Scalar>
__global__ void irls_form_systems_kernel(
    const Scalar* __restrict__ G_base,
    const Scalar* __restrict__ delta_G,     // k*k × tile_n
    Scalar*       __restrict__ G_batch,     // k*k × tile_n (output)
    const Scalar* __restrict__ B_weighted,  // k × tile_n (input)
    Scalar*       __restrict__ b_batch,     // k × tile_n (output)
    int k, int tile_n, int tile_start,
    Scalar L1, Scalar L2)
{
    const int kk = k * k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Form G_batch
    if (idx < tile_n * kk) {
        int j_local = idx / kk;
        int f = idx % kk;
        int j = tile_start + j_local;
        int a = f % k, b = f / k;

        Scalar val = G_base[f] + delta_G[static_cast<size_t>(j) * kk + f];
        // Mirror: if in lower triangle, copy from upper
        if (a > b) {
            int f_upper = b + a * k;
            val = G_base[f_upper] + delta_G[static_cast<size_t>(j) * kk + f_upper];
        }
        // Diagonal: add L2
        if (a == b) val += L2;
        G_batch[static_cast<size_t>(j_local) * kk + f] = val;
    }

    // Form b_batch (separate index range)
    int bidx = idx - tile_n * kk;
    if (bidx >= 0 && bidx < tile_n * k) {
        int j_local = bidx / k;
        int f = bidx % k;
        int j = tile_start + j_local;
        Scalar val = B_weighted[static_cast<size_t>(j) * k + f];
        if (L1 > Scalar(0)) val -= L1;
        b_batch[static_cast<size_t>(j_local) * k + f] = val;
    }
}


// ============================================================================
// High-level dispatch: fused (small k) or tiled Cholesky (large k)
// ============================================================================

/**
 * @brief GPU-native IRLS batch NNLS for sparse A.
 *
 * For each IRLS iteration:
 *   - k fits in smem:  fused kernel (delta + CD solve in one launch)
 *   - k too large:     delta kernel + tiled Cholesky via cuSOLVER
 *
 * Runs entirely on GPU; no PCIe transfers.
 */
template<typename Scalar>
void nnls_batch_irls_gpu(
    gpu::GPUContext& ctx,
    const gpu::SparseMatrixGPU<Scalar>& A,
    const gpu::DenseMatrixGPU<Scalar>& W,       // k × m
    gpu::DenseMatrixGPU<Scalar>& H,             // k × n
    const gpu::DenseMatrixGPU<Scalar>& G_base,  // k × k
    int loss_type,
    Scalar huber_delta,
    Scalar robust_delta,
    Scalar gp_blend,
    Scalar power_param,
    const gpu::DeviceMemory<Scalar>* disp_row,
    const gpu::DeviceMemory<Scalar>* disp_col,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit,
    int irls_max)
{
    const int k = W.rows;
    const int n = A.cols;

    const Scalar* disp_row_ptr = disp_row ? disp_row->get() : nullptr;
    const Scalar* disp_col_ptr = disp_col ? disp_col->get() : nullptr;

    // ---- Check if fused kernel fits in shared memory ----
    const size_t fused_smem = (static_cast<size_t>(k) * k + 4 * k) * sizeof(Scalar);
    const int max_smem = gpu::get_max_smem_per_block();
    const bool use_fused = (fused_smem <= static_cast<size_t>(max_smem));

    if (use_fused) {
        // ============================================================
        // Fused path: 1 block/column, delta + CD solve in shared mem
        // ============================================================

        // Opt-in for > 48KB shared memory
        if (fused_smem > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                fused_irls_nnls_kernel<Scalar>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(fused_smem)));
        }

        int threads = gpu::compute_thread_count(k, A.nnz, n);

        for (int irls_iter = 0; irls_iter < irls_max; ++irls_iter) {
            fused_irls_nnls_kernel<Scalar>
                <<<n, threads, fused_smem, ctx.stream>>>(
                A.col_ptr.get(), A.row_indices.get(), A.values.get(),
                W.data.get(), H.data.get(), G_base.data.get(),
                k, n,
                loss_type, huber_delta, robust_delta, power_param,
                disp_row_ptr, disp_col_ptr,
                L1, L2, nonneg, cd_maxit);
            CUDA_CHECK(cudaGetLastError());
        }

    } else {
        // ============================================================
        // Tiled Cholesky fallback for large k
        // ============================================================

        const int kk = k * k;

        // Allocate persistent buffers for delta_G (k²×n) and B_weighted (k×n)
        gpu::DeviceMemory<Scalar> d_delta_G(static_cast<size_t>(kk) * n);
        gpu::DeviceMemory<Scalar> d_B_weighted(static_cast<size_t>(k) * n);

        // Delta kernel shared memory: just 2k scalars (h + w scratch)
        const size_t delta_smem = 2 * k * sizeof(Scalar);
        int delta_threads = gpu::compute_thread_count(k, A.nnz, n);

        if (delta_smem > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                irls_delta_kernel<Scalar>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(delta_smem)));
        }

        // Tiled Cholesky: tile size bounded by GPU memory
        // Each tile needs: tile_n * (k² + k) * sizeof(Scalar) + overhead
        // Use conservative tile size
        const int tile_size = std::min(8192, n);
        gpu::DeviceMemory<Scalar> G_batch(static_cast<size_t>(tile_size) * kk);
        gpu::DeviceMemory<Scalar> b_batch(static_cast<size_t>(tile_size) * k);
        gpu::DeviceMemory<Scalar*> G_ptrs_d(tile_size);
        gpu::DeviceMemory<Scalar*> b_ptrs_d(tile_size);
        gpu::DeviceMemory<int>  d_info(tile_size);

        // Build pointer arrays (same every tile)
        std::vector<Scalar*> G_ptrs_h(tile_size), b_ptrs_h(tile_size);
        for (int j = 0; j < tile_size; ++j) {
            G_ptrs_h[j] = G_batch.get() + static_cast<size_t>(j) * kk;
            b_ptrs_h[j] = b_batch.get() + static_cast<size_t>(j) * k;
        }
        CUDA_CHECK(cudaMemcpy(G_ptrs_d.get(), G_ptrs_h.data(),
                              tile_size * sizeof(Scalar*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_ptrs_d.get(), b_ptrs_h.data(),
                              tile_size * sizeof(Scalar*), cudaMemcpyHostToDevice));

        // Use context's cuSOLVER handle (already stream-bound)
        cusolverDnHandle_t cusolver = ctx.cusolver;

        constexpr int TH = 256;

        for (int irls_iter = 0; irls_iter < irls_max; ++irls_iter) {

            // Phase 1: Compute delta_G and B_weighted
            irls_delta_kernel<Scalar><<<n, delta_threads, delta_smem, ctx.stream>>>(
                A.col_ptr.get(), A.row_indices.get(), A.values.get(),
                W.data.get(), H.data.get(),
                d_delta_G.get(), d_B_weighted.get(),
                k, n, loss_type, huber_delta, robust_delta, power_param,
                disp_row_ptr, disp_col_ptr);
            CUDA_CHECK(cudaGetLastError());

            // Phase 2: Tiled Cholesky solve
            for (int tile_start = 0; tile_start < n; tile_start += tile_size) {
                const int tile_n = std::min(tile_size, n - tile_start);

                // Form G_batch and b_batch
                {
                    int total_work = tile_n * kk + tile_n * k;
                    int blocks = (total_work + TH - 1) / TH;
                    irls_form_systems_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                        G_base.data.get(), d_delta_G.get(),
                        G_batch.get(), d_B_weighted.get(), b_batch.get(),
                        k, tile_n, tile_start, L1, L2);
                }

                // Batched Cholesky factorization
                if constexpr (std::is_same_v<Scalar, double>) {
                    CUSOLVER_CHECK(cusolverDnDpotrfBatched(cusolver, CUBLAS_FILL_MODE_LOWER,
                        k, reinterpret_cast<double**>(G_ptrs_d.get()), k,
                        d_info.get(), tile_n));
                } else {
                    CUSOLVER_CHECK(cusolverDnSpotrfBatched(cusolver, CUBLAS_FILL_MODE_LOWER,
                        k, reinterpret_cast<float**>(G_ptrs_d.get()), k,
                        d_info.get(), tile_n));
                }

                // Batched triangular solve
                if constexpr (std::is_same_v<Scalar, double>) {
                    CUSOLVER_CHECK(cusolverDnDpotrsBatched(cusolver, CUBLAS_FILL_MODE_LOWER,
                        k, 1, reinterpret_cast<double**>(G_ptrs_d.get()), k,
                        reinterpret_cast<double**>(b_ptrs_d.get()), k,
                        d_info.get(), tile_n));
                } else {
                    CUSOLVER_CHECK(cusolverDnSpotrsBatched(cusolver, CUBLAS_FILL_MODE_LOWER,
                        k, 1, reinterpret_cast<float**>(G_ptrs_d.get()), k,
                        reinterpret_cast<float**>(b_ptrs_d.get()), k,
                        d_info.get(), tile_n));
                }

                // Clip negatives
                if (nonneg) {
                    int blocks = (tile_n * k + TH - 1) / TH;
                    irls_clip_nonneg_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                        b_batch.get(), tile_n * k);
                }

                // Scatter into H
                CUDA_CHECK(cudaMemcpyAsync(
                    H.data.get() + static_cast<size_t>(tile_start) * k,
                    b_batch.get(),
                    static_cast<size_t>(tile_n) * k * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
            }
        }

        // cusolver handle owned by ctx — do NOT destroy
    }

    ctx.sync();
}

}  // namespace gpu_irls
}  // namespace primitives
}  // namespace FactorNet
