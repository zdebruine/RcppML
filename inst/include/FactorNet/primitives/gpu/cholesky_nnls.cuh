// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file primitives/gpu/cholesky_nnls.cuh
 * @brief GPU Cholesky-based NNLS: cuSOLVER potrf + cuBLAS trsm + clip
 *
 * Pure GPU implementation — no host-device transfers.
 * Eliminates CPU fallback overhead for Cholesky solver mode.
 *
 * Provides:
 *   cholesky_clip_batch_gpu: unconstrained Cholesky solve + non-negativity clip
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/context.cuh>
#include <FactorNet/gpu/batch_nnls.cuh>   // batch_cd_nnls_gpu for the KKT polish
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace FactorNet {
namespace gpu {

// ============================================================================
// Clip kernel
// ============================================================================

template<typename Scalar>
__global__ void clip_nonneg_kernel(Scalar* X, int size, bool nonneg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && nonneg) {
        if (X[idx] < static_cast<Scalar>(0)) X[idx] = static_cast<Scalar>(0);
    }
}

// ============================================================================
// Cholesky + clip NNLS (batch)
// ============================================================================

/**
 * @brief Exact (non-negative) least squares for all columns: Cholesky solve,
 *        clip, then coordinate descent until the KKT conditions hold.
 *
 * All operations on GPU:
 *   1. potrf: G = L L^T
 *   2. two trsm: X = G^{-1} B  (exact when no constraint is active)
 *   3. clip X at zero (nonneg only)
 *   4. warm-started coordinate descent on the original G and B until no
 *      coordinate of a column moves by more than tol * max|x| (nonneg only).
 *      Clipping alone is not the non-negative least squares solution.
 * If the factorization fails (G not positive definite, e.g. an empty or
 * duplicated factor), the solve is done by coordinate descent alone.
 *
 * @param ctx      GPU context with cuBLAS handle
 * @param G        Gram matrix (k × k), overwritten with L (lower triangular)
 * @param B        RHS (k × n), not modified
 * @param X        Solution (k × n), output
 * @param nonneg   Enforce non-negativity
 * @param polish_sweeps  Cap on coordinate-descent sweeps (0 disables the
 *                 polish and returns the clipped solution)
 * @param tol      Relative stopping tolerance for the sweeps; <= 0 selects
 *                 16 machine epsilons, the smallest change a float iterate
 *                 can still resolve reliably
 */
template<typename Scalar>
void cholesky_clip_batch_gpu(
    GPUContext& ctx,
    DenseMatrixGPU<Scalar>& G,     // k×k, overwritten
    DenseMatrixGPU<Scalar>& B,     // k×n
    DenseMatrixGPU<Scalar>& X,     // k×n, output
    bool nonneg,
    int polish_sweeps = 200,
    Scalar tol = Scalar(0))
{
    const int k = G.rows;
    const int n = X.cols;
    const Scalar cd_tol = tol > Scalar(0)
        ? tol
        : static_cast<Scalar>(16) * std::numeric_limits<Scalar>::epsilon();
    // The convergence stop exists for k <= 64; larger ranks run every sweep
    const int sweeps = (k <= 64) ? polish_sweeps : std::min(polish_sweeps, 100);

    // potrf overwrites the lower triangle of G with the Cholesky factor, so
    // keep an intact copy for coordinate descent.
    DenseMatrixGPU<Scalar> G_full(k, k);
    CUDA_CHECK(cudaMemcpyAsync(G_full.data.get(), G.data.get(),
                               static_cast<size_t>(k) * k * sizeof(Scalar),
                               cudaMemcpyDeviceToDevice, ctx.stream));

    // 1. Cholesky factorization: G = L * L^T (the context's cuSOLVER handle)
    int lwork = 0;
    if constexpr (std::is_same_v<Scalar, double>) {
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
            ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k, G.data.get(), k, &lwork));
    } else {
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
            ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k, G.data.get(), k, &lwork));
    }

    DeviceMemory<Scalar> d_work(lwork > 0 ? lwork : 1);
    DeviceMemory<int> d_info(1);

    if constexpr (std::is_same_v<Scalar, double>) {
        CUSOLVER_CHECK(cusolverDnDpotrf(
            ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k,
            G.data.get(), k, d_work.get(), lwork, d_info.get()));
    } else {
        CUSOLVER_CHECK(cusolverDnSpotrf(
            ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k,
            G.data.get(), k, d_work.get(), lwork, d_info.get()));
    }

    // Did the factorization succeed? (k × k, so this sync is cheap)
    int h_info = 0;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaMemcpy(&h_info, d_info.get(), sizeof(int),
                          cudaMemcpyDeviceToHost));

    if (h_info != 0) {
        // G is not positive definite: coordinate descent from a cold start
        // (it skips coordinates with a zero diagonal).
        batch_cd_nnls_gpu(ctx, G_full, B, X, std::max(sweeps, 1000),
                          nonneg, /*warm_start=*/false, cd_tol);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        return;
    }

    // Copy B to X (overwritten by the triangular solves)
    CUDA_CHECK(cudaMemcpyAsync(X.data.get(), B.data.get(),
                               static_cast<size_t>(k) * n * sizeof(Scalar),
                               cudaMemcpyDeviceToDevice, ctx.stream));

    // 2. Forward solve L * Y = B, then backward solve L^T * X = Y
    const Scalar one = 1.0;
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDtrsm(ctx.handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            k, n, &one, G.data.get(), k, X.data.get(), k));
        CUBLAS_CHECK(cublasDtrsm(ctx.handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            k, n, &one, G.data.get(), k, X.data.get(), k));
    } else {
        CUBLAS_CHECK(cublasStrsm(ctx.handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            k, n, &one, G.data.get(), k, X.data.get(), k));
        CUBLAS_CHECK(cublasStrsm(ctx.handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            k, n, &one, G.data.get(), k, X.data.get(), k));
    }

    if (nonneg) {
        // 3. Clip negatives
        const size_t total = static_cast<size_t>(k) * n;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        clip_nonneg_kernel<<<blocks, threads, 0, ctx.stream>>>(
            X.data.get(), static_cast<int>(total), nonneg);

        // 4. Polish to the KKT point: warm-started coordinate descent
        if (sweeps > 0) {
            batch_cd_nnls_gpu(ctx, G_full, B, X, sweeps,
                              /*nonneg=*/true, /*warm_start=*/true, cd_tol);
        }
    }
    // G_full, d_work and d_info are released on return; let the work finish
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

// ============================================================================
// Helpers for per-column G/b formation (CV NMF)
// ============================================================================

/**
 * @brief Form G_batch[j_local] = G_full - delta_G[j] for all j in a tile.
 *
 * Layout: delta_G is (k*k) x n column-major, column j holds the flattened
 * k×k perturbation for column j of H.
 */
template<typename Scalar>
__global__ void form_G_batch_kernel(
    const Scalar* __restrict__ G_full,    // k*k elements (shared Gram)
    const Scalar* __restrict__ delta_G,   // k*k × n, column j at j*(k*k)
    Scalar*       __restrict__ G_batch,   // k*k × tile_n, output
    int kk, int tile_n, int tile_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_n * kk) return;
    int j_local = idx / kk;
    int f       = idx % kk;
    int j       = tile_start + j_local;
    G_batch[static_cast<size_t>(j_local) * kk + f] =
        G_full[f] - delta_G[static_cast<size_t>(j) * kk + f];
}

/**
 * @brief Form b_batch[j_local] = B_full[:,j] - delta_b[:,j] for all j in tile.
 *
 * B_full and delta_b are k×n column-major; column j at offset j*k.
 */
template<typename Scalar>
__global__ void form_b_batch_kernel(
    const Scalar* __restrict__ B_full,    // k × n
    const Scalar* __restrict__ delta_b,   // k × n
    Scalar*       __restrict__ b_batch,   // k × tile_n, output
    int k, int tile_n, int tile_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_n * k) return;
    int j_local = idx / k;
    int f       = idx % k;
    int j       = tile_start + j_local;
    b_batch[static_cast<size_t>(j_local) * k + f] =
        B_full[static_cast<size_t>(j) * k + f] -
        delta_b[static_cast<size_t>(j) * k + f];
}

// ============================================================================
// CV Cholesky solve: tile-batched cuSOLVER for per-column Gram systems
// ============================================================================

/**
 * @brief Tiled batched Cholesky NNLS for CV NMF H/W update.
 *
 * For each column j, solves the per-column masked least-squares system
 * exactly (one-shot, no iterations):
 *
 *   (G_full - delta_G[:,j]) * x_j = B_full[:,j] - delta_b[:,j],  x_j >= 0
 *
 * Uses cusolverDnSp/DpotrfBatched + potrsBatched in tiles of tile_size
 * columns.  Memory cost: tile_size * k^2 + tile_size * k (scratch) floats.
 *
 * Default tile_size=8192:  at k=128 this is  8192*(16384+128)*4 ≈ 536 MB.
 *
 * @param ctx        GPU context — provides ctx.cusolver + ctx.stream
 * @param G_full     Shared Gram matrix (k×k, const, not modified)
 * @param delta_G    Per-column delta G  (rows=k*k, cols=n), col j = flattened
 *                   k×k perturbation to subtract from G_full
 * @param B_full     Full RHS matrix (k×n)
 * @param delta_b    Per-column delta b  (k×n)
 * @param H          Output factor (k×n), written in-place
 * @param nonneg     Clip H >= 0 after solve
 * @param tile_size  Columns per tile (default 8192)
 */
template<typename Scalar>
void cholesky_cv_tiled_gpu(
    GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& G_full,
    const DenseMatrixGPU<Scalar>& delta_G,
    const DenseMatrixGPU<Scalar>& B_full,
    const DenseMatrixGPU<Scalar>& delta_b,
    DenseMatrixGPU<Scalar>&       H,
    bool nonneg,
    int  tile_size = 8192)
{
    const int k  = G_full.rows;
    const int n  = H.cols;
    const int kk = k * k;

    tile_size = std::min(tile_size, n);

    // ---- Allocate tile-sized scratch (reused for every tile) ----
    DeviceMemory<Scalar>  G_batch(static_cast<size_t>(tile_size) * kk);
    DeviceMemory<Scalar>  b_batch(static_cast<size_t>(tile_size) * k);
    DeviceMemory<Scalar*> G_ptrs_d(tile_size);
    DeviceMemory<Scalar*> b_ptrs_d(tile_size);
    DeviceMemory<int>     d_info(tile_size);

    // Build host pointer arrays (point into the same scratch blocks every tile)
    std::vector<Scalar*> G_ptrs_h(tile_size);
    std::vector<Scalar*> b_ptrs_h(tile_size);
    for (int j = 0; j < tile_size; ++j) {
        G_ptrs_h[j] = G_batch.get() + static_cast<size_t>(j) * kk;
        b_ptrs_h[j] = b_batch.get() + static_cast<size_t>(j) * k;
    }
    // Upload pointer arrays once — they always point into the same scratch
    CUDA_CHECK(cudaMemcpy(G_ptrs_d.get(), G_ptrs_h.data(),
                          static_cast<size_t>(tile_size) * sizeof(Scalar*),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_ptrs_d.get(), b_ptrs_h.data(),
                          static_cast<size_t>(tile_size) * sizeof(Scalar*),
                          cudaMemcpyHostToDevice));

    constexpr int TH = 256;

    for (int tile_start = 0; tile_start < n; tile_start += tile_size) {
        const int tile_n = std::min(tile_size, n - tile_start);

        // 1. G_batch[j] = G_full - delta_G[:,j]
        {
            int blocks = (tile_n * kk + TH - 1) / TH;
            form_G_batch_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                G_full.data.get(), delta_G.data.get(),
                G_batch.get(), kk, tile_n, tile_start);
        }

        // 2. b_batch[j] = B_full[:,j] - delta_b[:,j]
        {
            int blocks = (tile_n * k + TH - 1) / TH;
            form_b_batch_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                B_full.data.get(), delta_b.data.get(),
                b_batch.get(), k, tile_n, tile_start);
        }

        // 3. Batched Cholesky factorization  G_batch[j] = L_j * L_j^T
        if constexpr (std::is_same_v<Scalar, double>) {
            CUSOLVER_CHECK(cusolverDnDpotrfBatched(
                ctx.cusolver, CUBLAS_FILL_MODE_LOWER,
                k, reinterpret_cast<double**>(G_ptrs_d.get()), k,
                d_info.get(), tile_n));
        } else {
            CUSOLVER_CHECK(cusolverDnSpotrfBatched(
                ctx.cusolver, CUBLAS_FILL_MODE_LOWER,
                k, reinterpret_cast<float**>(G_ptrs_d.get()), k,
                d_info.get(), tile_n));
        }

        // 4. Batched triangular solve  b_batch[j] = G_batch[j]^{-1} b_batch[j]
        if constexpr (std::is_same_v<Scalar, double>) {
            CUSOLVER_CHECK(cusolverDnDpotrsBatched(
                ctx.cusolver, CUBLAS_FILL_MODE_LOWER,
                k, 1,
                reinterpret_cast<double**>(G_ptrs_d.get()), k,
                reinterpret_cast<double**>(b_ptrs_d.get()), k,
                d_info.get(), tile_n));
        } else {
            CUSOLVER_CHECK(cusolverDnSpotrsBatched(
                ctx.cusolver, CUBLAS_FILL_MODE_LOWER,
                k, 1,
                reinterpret_cast<float**>(G_ptrs_d.get()), k,
                reinterpret_cast<float**>(b_ptrs_d.get()), k,
                d_info.get(), tile_n));
        }

        // 5. Clip negatives
        if (nonneg) {
            int blocks = (tile_n * k + TH - 1) / TH;
            clip_nonneg_kernel<Scalar><<<blocks, TH, 0, ctx.stream>>>(
                b_batch.get(), tile_n * k, true);
        }

        // 6. Scatter b_batch into H[:,tile_start:tile_start+tile_n]
        CUDA_CHECK(cudaMemcpyAsync(
            H.data.get() + static_cast<size_t>(tile_start) * k,
            b_batch.get(),
            static_cast<size_t>(tile_n) * k * sizeof(Scalar),
            cudaMemcpyDeviceToDevice, ctx.stream));
    }
}

}  // namespace gpu

// Expose GPU NNLS functions at primitives:: level for callers
namespace primitives {
using gpu::cholesky_clip_batch_gpu;
using gpu::cholesky_cv_tiled_gpu;
}  // namespace primitives

}  // namespace FactorNet
