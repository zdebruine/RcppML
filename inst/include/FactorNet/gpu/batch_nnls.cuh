/**
 * @file batch_nnls.cuh
 * @brief Fixed-iteration batch NNLS solvers for GPU
 *
 * GPU-optimized NNLS solver that exploits fixed iteration count
 * to eliminate branch divergence:
 *
 *   batch_cd_nnls_gpu: Fixed-iteration Gauss-Seidel coordinate descent
 *      - 1 thread per column for k<=32, h[k] in registers, G in shared mem
 *      - 1 warp per column for k=33-64, parallel gradient via warp shuffle
 *      - NO early termination → zero branch divergence
 *      - All threads execute identical (iter, coord) instruction stream
 *
 * These solvers are designed for the NMF inner loop where the iteration
 * budget is known in advance and consistent across all columns.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include "types.cuh"
#include "nnls.cuh"   // nnls_kernel_warmstart_parallel for k > 64 fallback
#include <cublas_v2.h>

namespace FactorNet {
namespace gpu {

// ============================================================================
// Fixed-iteration CD kernel: 1 thread per column (k <= 32)
//
// Each thread processes one column independently. Solution h[k] lives
// entirely in registers. G[k][k] is in shared memory (broadcast reads).
// All threads execute the same (iter, coord) loop — zero divergence.
//
// Register budget: k floats for h + ~8 for locals = ~40 for k=32
// Shared memory: k*k*sizeof(Scalar) = 4KB for k=32, float
// Occupancy: blockDim=256, 40 regs/thread → 10240 regs/block → 6 blocks/SM
// ============================================================================

// Template version for small k (compile-time known)
template<int K, typename Scalar>
__global__ void batch_cd_kernel_tpc(
    const Scalar* __restrict__ G,     // k × k Gram matrix (column-major)
    const Scalar* __restrict__ B,     // k × n RHS (read-only)
    Scalar* __restrict__ H,           // k × n solution (read/write for warm start)
    int n,
    int max_cd_iter,
    bool nonneg,
    bool warm_start)
{
    __shared__ Scalar s_G[K * K];

    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Cooperatively load G into shared memory
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    __syncthreads();

    if (j >= n) return;

    // Load h into registers (warm start from H, or cold start from clip(B))
    Scalar h[K];
    if (warm_start) {
        #pragma unroll
        for (int i = 0; i < K; ++i)
            h[i] = nonneg ? max(H[i + j * K], Scalar(0)) : H[i + j * K];
    } else {
        #pragma unroll
        for (int i = 0; i < K; ++i)
            h[i] = nonneg ? max(B[i + j * K], Scalar(0)) : B[i + j * K];
    }

    // Fixed-iteration coordinate descent (no early termination)
    for (int iter = 0; iter < max_cd_iter; ++iter) {
        #pragma unroll
        for (int coord = 0; coord < K; ++coord) {
            // Compute gradient: grad = G[coord,:] · h - B[coord,j]
            Scalar grad = -B[coord + j * K];
            #pragma unroll
            for (int l = 0; l < K; ++l) {
                grad += s_G[coord * K + l] * h[l];
            }

            // Update h[coord]
            Scalar g_diag = s_G[coord * K + coord];
            if (g_diag > Scalar(0)) {
                Scalar new_val = h[coord] - grad / g_diag;
                h[coord] = nonneg ? max(new_val, Scalar(0)) : new_val;
            }
        }
    }

    // Write back to global memory
    #pragma unroll
    for (int i = 0; i < K; ++i)
        H[i + j * K] = h[i];
}

// Dynamic version for arbitrary k (uses register array + loop)
template<typename Scalar>
__global__ void batch_cd_kernel_tpc_dynamic(
    const Scalar* __restrict__ G,     // k × k Gram matrix
    const Scalar* __restrict__ B,     // k × n RHS
    Scalar* __restrict__ H,           // k × n solution
    int k,
    int n,
    int max_cd_iter,
    bool nonneg,
    bool warm_start)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);

    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Cooperatively load G into shared memory
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    __syncthreads();

    if (j >= n) return;

    // For k <= 64, use stack-allocated array (compiler may use registers)
    // For larger k, this spills to local memory (still fast due to L1 cache)
    Scalar h[64];  // Max supported k for thread-per-column approach

    if (warm_start) {
        for (int i = 0; i < k; ++i)
            h[i] = nonneg ? max(H[i + j * k], Scalar(0)) : H[i + j * k];
    } else {
        for (int i = 0; i < k; ++i)
            h[i] = nonneg ? max(B[i + j * k], Scalar(0)) : B[i + j * k];
    }

    // Fixed-iteration coordinate descent
    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            Scalar grad = -B[coord + j * k];
            for (int l = 0; l < k; ++l) {
                grad += s_G[coord * k + l] * h[l];
            }
            Scalar g_diag = s_G[coord * k + coord];
            if (g_diag > Scalar(0)) {
                Scalar new_val = h[coord] - grad / g_diag;
                h[coord] = nonneg ? max(new_val, Scalar(0)) : new_val;
            }
        }
    }

    for (int i = 0; i < k; ++i)
        H[i + j * k] = h[i];
}

// ============================================================================
// Fixed-iteration CD kernel: 1 warp per column, up to NUM_SLOTS*32 coords
//
// Generalises the 2-slot (k<=64) warp kernel to NUM_SLOTS register slots,
// supporting k up to NUM_SLOTS*32 = 128 (for NUM_SLOTS=4) without any
// __syncthreads() during the CD loop — only __shfl_sync (4 cycles vs 32+).
//
// Memory layout:
//   s_G   k×k Gram in shared, loaded col-major.
//         Access s_G[coord*k + (lane+s*32)] exploits G's symmetry:
//         G[coord, lane+s*32] = G[lane+s*32, coord] (G is symmetric PSD).
//         For k a multiple of 32: bank = (coord*k + l) % 32 = l → zero conflicts.
//   my_h  lane+s*32-th coordinate in slot s register → zero global reads/writes
//         during the entire CD loop.
//
// Occupancy (k=128, NUM_SLOTS=4, 256 threads = 8 warps = 8 cols/block):
//   s_G = 128×128×4 = 64 KB  (H100 allows 228 KB/block)
//   regs ≈ 64/thread (my_h[4] + my_b[4] + scalars)
//   → 3 blocks/SM × 8 cols/block = 24 live cols/SM simultaneously
//
// vs nnls_kernel_warmstart_parallel (k=128, n=40000, maxit=10):
//   1 block/col × 2 __syncthreads/coord × 128 coords × 10 iters = 2560 barriers
//   Only 1 warp active per block → utilisation ≈ 1/nwarps
//
// This kernel: 0 __syncthreads per CD iter, 8 warps/block fully active.
// ============================================================================

template<int NUM_SLOTS, typename Scalar>
__global__ void batch_cd_kernel_warp_general(
    const Scalar* __restrict__ G,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ H,
    int k, int n, int max_cd_iter,
    bool nonneg, bool warm_start)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);

    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    const int nwarps   = blockDim.x >> 5;
    const int col      = blockIdx.x * nwarps + warp_id;

    // One-time cooperative G load (single __syncthreads for the whole kernel)
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x)
        s_G[idx] = G[idx];
    __syncthreads();  // ← the ONLY block barrier in this kernel

    if (col >= n) return;

    // Each lane holds NUM_SLOTS values of h and b:
    //   slot s → row = lane + s*32
    Scalar my_h[NUM_SLOTS];
    Scalar my_b[NUM_SLOTS];

    #pragma unroll
    for (int s = 0; s < NUM_SLOTS; ++s) {
        const int row = lane + s * 32;
        if (row < k) {
            my_b[s] = B[row + col * k];
            Scalar init = warm_start ? H[row + col * k] : my_b[s];
            my_h[s] = nonneg ? max(init, Scalar(0)) : init;
        } else {
            my_b[s] = Scalar(0);
            my_h[s] = Scalar(0);
        }
    }

    // -----------------------------------------------------------------------
    // Fixed-iteration Gauss-Seidel CD — zero __syncthreads, only __shfl_sync
    // -----------------------------------------------------------------------
    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            // owner_lane/slot: which lane/register holds h[coord]
            // coord is loop-uniform across all 32 lanes → no warp divergence
            const int owner_lane = coord & 31;
            const int owner_slot = coord >> 5;  // 0..NUM_SLOTS-1

            // ---- Warp-parallel dot product G[coord,:] · h ----
            // Uses symmetry: s_G[coord*k + (lane+s*32)] = G[lane+s*32, coord]
            // For k % 32 == 0: lane l → bank l → zero shared-mem bank conflicts
            Scalar partial = Scalar(0);
            #pragma unroll
            for (int s = 0; s < NUM_SLOTS; ++s) {
                const int row = lane + s * 32;
                Scalar g = (row < k) ? s_G[coord * k + row] : Scalar(0);
                partial += g * my_h[s];
            }
            // Butterfly warp-reduce → lane 0 holds G[coord,:] · h
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, off);

            // ---- Broadcast b[coord] and h[coord] from owner lane ----
            // owner_slot is warp-uniform → compiler picks exactly one shfl
            Scalar b_coord, h_coord;
            if (owner_slot == 0) {
                b_coord = __shfl_sync(0xffffffff, my_b[0], owner_lane);
                h_coord = __shfl_sync(0xffffffff, my_h[0], owner_lane);
            } else if (NUM_SLOTS > 1 && owner_slot == 1) {
                b_coord = __shfl_sync(0xffffffff, my_b[1], owner_lane);
                h_coord = __shfl_sync(0xffffffff, my_h[1], owner_lane);
            } else if (NUM_SLOTS > 2 && owner_slot == 2) {
                b_coord = __shfl_sync(0xffffffff, my_b[2], owner_lane);
                h_coord = __shfl_sync(0xffffffff, my_h[2], owner_lane);
            } else {
                b_coord = __shfl_sync(0xffffffff, my_b[NUM_SLOTS > 3 ? 3 : 0], owner_lane);
                h_coord = __shfl_sync(0xffffffff, my_h[NUM_SLOTS > 3 ? 3 : 0], owner_lane);
            }

            // ---- Coordinate update (lane 0 computes, broadcast) ----
            const Scalar g_diag = s_G[coord * k + coord];
            Scalar new_h = h_coord;
            if (lane == 0 && g_diag > Scalar(0)) {
                Scalar upd = h_coord - (partial - b_coord) / g_diag;
                new_h = nonneg ? max(upd, Scalar(0)) : upd;
            }
            new_h = __shfl_sync(0xffffffff, new_h, 0);  // broadcast from lane 0

            // ---- Owner lane writes updated h[coord] into its register ----
            // lane == owner_lane is divergent but predicated (1 lane active)
            if (lane == owner_lane) {
                if (owner_slot == 0)                   my_h[0] = new_h;
                else if (NUM_SLOTS > 1 && owner_slot == 1) my_h[1] = new_h;
                else if (NUM_SLOTS > 2 && owner_slot == 2) my_h[2] = new_h;
                else if (NUM_SLOTS > 3)                    my_h[3] = new_h;
            }
        }
    }

    // Write back (coalesced: consecutive lanes → consecutive memory)
    #pragma unroll
    for (int s = 0; s < NUM_SLOTS; ++s) {
        const int row = lane + s * 32;
        if (row < k) H[row + col * k] = my_h[s];
    }
}

// ============================================================================
// Host dispatch: fixed-iteration batch CD
// ============================================================================

template<typename Scalar>
void batch_cd_nnls_gpu(
    const GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& G,
    const DenseMatrixGPU<Scalar>& B,
    DenseMatrixGPU<Scalar>& H,
    int max_cd_iter = 10,
    bool nonneg = true,
    bool warm_start = true)
{
    const int k = G.rows;
    const int n = H.cols;

    if (k <= 64) {
        // Thread-per-column: 1 thread handles all k coordinates.
        // Single dynamic-k kernel for all k<=64 — simple, correct, and within
        // 5% of template-specialized variants (unroll benefit is marginal on
        // modern GPUs with good instruction caches).
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        size_t smem = static_cast<size_t>(k) * k * sizeof(Scalar);
        batch_cd_kernel_tpc_dynamic<Scalar><<<blocks, threads, smem, ctx.stream>>>(
            G.data.get(), B.data.get(), H.data.get(), k, n,
            max_cd_iter, nonneg, warm_start);
    } else if (k <= 128) {
        // k = 33..128: generalised warp-per-column kernel.
        //
        // NUM_SLOTS = ceil(k/32) in {2,3,4}: each warp (32 lanes) handles 1 column;
        // lane l holds h[l], h[l+32], ... in registers (NUM_SLOTS values each).
        // G (k×k) lives in shared memory, shared across all warps in the block.
        // Zero __syncthreads during the CD loop — only __shfl_sync (4 cycles).
        //
        // Block config: 8 warps × 32 = 256 threads → 8 columns per block.
        // Shared mem: k*k*sizeof(Scalar):
        //   k=48 → 9KB   k=64 → 16KB   k=96 → 36KB   k=128 → 64KB
        // All well within H100's 228KB limit.
        const int warps_per_block = 8;   // 8 columns processed per block
        const int threads   = warps_per_block * 32;
        const int blocks    = (n + warps_per_block - 1) / warps_per_block;
        const size_t smem   = static_cast<size_t>(k) * k * sizeof(Scalar);

        // Unlock > 48 KB shared memory if needed (k >= 112)
        auto unlock_smem = [&](auto* fn_ptr) {
            if (smem > 48 * 1024)
                CUDA_CHECK(cudaFuncSetAttribute(fn_ptr,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem)));
        };

        if (k <= 64) {
            // NUM_SLOTS=2 (k≤64)
            unlock_smem(batch_cd_kernel_warp_general<2, Scalar>);
            batch_cd_kernel_warp_general<2, Scalar><<<blocks, threads, smem, ctx.stream>>>(
                G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg, warm_start);
        } else if (k <= 96) {
            // NUM_SLOTS=3 (k=65..96)
            unlock_smem(batch_cd_kernel_warp_general<3, Scalar>);
            batch_cd_kernel_warp_general<3, Scalar><<<blocks, threads, smem, ctx.stream>>>(
                G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg, warm_start);
        } else {
            // NUM_SLOTS=4 (k=97..128)
            unlock_smem(batch_cd_kernel_warp_general<4, Scalar>);
            batch_cd_kernel_warp_general<4, Scalar><<<blocks, threads, smem, ctx.stream>>>(
                G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg, warm_start);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        // k > 128: fall back to the per-block parallel-gradient kernel.
        // (Rare in practice: NMF is seldom run at k>128.)
        int threads_per_block = ((min(max(k, 128), 512) + 31) / 32) * 32;
        int nwarps            = threads_per_block / 32;
        size_t smem = (static_cast<size_t>(k) * k + k + nwarps) * sizeof(Scalar);
        if (smem > 48 * 1024)
            CUDA_CHECK(cudaFuncSetAttribute(
                nnls_kernel_warmstart_parallel<Scalar>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem)));
        nnls_kernel_warmstart_parallel<Scalar><<<n, threads_per_block, smem, ctx.stream>>>(
            G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg);
        CUDA_CHECK(cudaGetLastError());
    }
}


// ============================================================================
// GPU L1 row normalization — coalesced grid-stride
//
// For a k×n column-major matrix, "row i" = {M[i], M[i+k], M[i+2k], ...}.
// Naïve 1-thread-per-row: 40K serial stride-k reads → ~8 ms at k=128.
//
// Coalesced design: blockDim.x = k (one thread per row inside a block).
// At each iteration j, all k threads in a block jointly read M[j*k .. j*k+k-1],
// a contiguous k-element block = 1-4 cache lines → fully coalesced.
//
// Quantitative (k=128, n=40000, H100):
//   Data: 20 MB read (pass1) + 20 MB rw (pass2) = 60 MB total
//   Bandwidth: 3.35 TB/s → ~18 μs ideal
//   Grid: min(1024, ceil(n/k)) blocks × k threads = ~131K active threads
//   Occupancy: 7.8 blocks/SM × 4 warps = 31 warps/SM ≈ 48%  (comfortable)
//
// Atomic contention: ceil(n/k)/blocks ≈ 39 atomicAdds per d[row] per block
// (vs 40000 in the elementwise approach) → negligible at L2 cache speed.
// ============================================================================

template<typename Scalar>
__global__ void l1_accum_gridstride(
    const Scalar* __restrict__ M,  // k×n col-major
    Scalar* __restrict__ d,        // k, must be pre-zeroed
    int k, int n)
{
    // threadIdx.x = row index (0..k-1)
    // blockIdx.x  = starting column tile, stride = gridDim.x
    const int row = threadIdx.x;
    if (row >= k) return;

    Scalar acc = Scalar(0);
    // Each iteration: all k threads access M[row + j*k] for the SAME j
    // → M[j*k], M[j*k+1], ..., M[j*k+k-1]  = contiguous = coalesced
    for (int j = blockIdx.x; j < n; j += gridDim.x) {
        Scalar v = M[row + (size_t)j * k];
        acc += (v < Scalar(0) ? -v : v);
    }
    // ceil(n/k)/gridDim atomics per d[row]: ~39 for grid=1024, n=40000, k=128
    atomicAdd(d + row, acc);
}

template<typename Scalar>
__global__ void l1_normalize_gridstride(
    Scalar* __restrict__ M,        // k×n col-major, in/out
    const Scalar* __restrict__ d,  // k norms (from accum pass)
    int k, int n, Scalar eps)
{
    const int row = threadIdx.x;
    if (row >= k) return;
    const Scalar inv = Scalar(1) / (d[row] + eps);
    for (int j = blockIdx.x; j < n; j += gridDim.x)
        M[row + (size_t)j * k] *= inv;
}

/// 2-pass coalesced L1 row normalisation.
template<typename Scalar>
void l1_norm_rows_gpu(
    const GPUContext& ctx,
    DenseMatrixGPU<Scalar>& M,   // k × n in/out
    DeviceMemory<Scalar>& d)     // k output norms
{
    const int k = M.rows;
    const int n = M.cols;
    const Scalar eps = static_cast<Scalar>(1e-15);

    // Round blockDim up to next warp multiple so all 32 lanes are active
    const int bk = ((k + 31) / 32) * 32;
    // Grid: enough blocks to target ~48% SM occupancy; cap at n (no idle blocks)
    const int blocks = min(1024, (n + bk - 1) / bk);

    CUDA_CHECK(cudaMemsetAsync(d.get(), 0, static_cast<size_t>(k) * sizeof(Scalar), ctx.stream));
    l1_accum_gridstride<Scalar>    <<<blocks, bk, 0, ctx.stream>>>(M.data.get(), d.get(), k, n);
    l1_normalize_gridstride<Scalar><<<blocks, bk, 0, ctx.stream>>>(M.data.get(), d.get(), k, n, eps);
}

// ===========================================================================
// GPU utility: upper-bound clamp
// ===========================================================================
//
// Replaces the host-mediated upper_bound path in fit_gpu_unified.cuh:
//   Old: ctx.sync() + download(k×n) + clamp on CPU + upload(k×n)
//   New: one GPU kernel launch, no PCIe transfer.
// ===========================================================================

template<typename Scalar>
__global__ void clamp_upper_bound_kernel(Scalar* __restrict__ M,
                                          Scalar upper_bound,
                                          int total)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total && M[idx] > upper_bound)
        M[idx] = upper_bound;
}

/**
 * @brief Clamp all elements of a factor matrix to [0, upper_bound] on GPU.
 *
 * @param ctx         GPU context (stream)
 * @param M           Dense factor matrix to clamp in-place
 * @param upper_bound Maximum allowed value
 */
template<typename Scalar>
inline void apply_upper_bound_gpu(const GPUContext& ctx,
                                   DenseMatrixGPU<Scalar>& M,
                                   Scalar upper_bound)
{
    const int total = M.rows * M.cols;
    constexpr int TPB = 256;
    clamp_upper_bound_kernel<Scalar>
        <<<(total + TPB - 1) / TPB, TPB, 0, ctx.stream>>>(
            M.data.get(), upper_bound, total);
}

// ===========================================================================
// GPU utility: fill d with 1.0 (NormType::None)
// ===========================================================================
//
// Replaces: ctx.sync() + h_d.setOnes() + d_d.upload(k)
// With a tiny kernel that avoids the sync and PCIe upload entirely.
// ===========================================================================

template<typename Scalar>
__global__ void fill_ones_kernel(Scalar* __restrict__ d, int k)
{
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < k) d[i] = Scalar(1);
}

/**
 * @brief Set all k elements of d to 1 on the GPU (no sync, no PCIe).
 */
template<typename Scalar>
inline void set_ones_gpu(const GPUContext& ctx,
                          DeviceMemory<Scalar>& d,
                          int k)
{
    constexpr int TPB = 128;
    fill_ones_kernel<Scalar>
        <<<(k + TPB - 1) / TPB, TPB, 0, ctx.stream>>>(d.get(), k);
}

// ===========================================================================
// GPU utility: row-wise diagonal scaling  M[i,:] *= d[i]
// ===========================================================================
// Replaces the host pattern:
//   download d → apply_scaling(M, d) → upload M
// with a single GPU kernel (no sync, no PCIe transfer).
// ===========================================================================

template<typename Scalar>
__global__ void scale_rows_by_d_kernel(Scalar* __restrict__ M,
                                        const Scalar* __restrict__ d,
                                        int k, int n)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= k * n) return;
    const int row = idx % k;  // column-major: consecutive elements are in different rows
    M[idx] *= d[row];
}

/**
 * @brief Scale each row i of M by d[i]: M[i,j] *= d[i] for all j.
 *
 * M is k×n column-major (element [i,j] at offset i + j*k).
 */
template<typename Scalar>
inline void scale_rows_by_d_gpu(const GPUContext& ctx,
                                 DenseMatrixGPU<Scalar>& M,
                                 const DeviceMemory<Scalar>& d)
{
    const int total = M.rows * M.cols;
    constexpr int TPB = 256;
    scale_rows_by_d_kernel<Scalar>
        <<<(total + TPB - 1) / TPB, TPB, 0, ctx.stream>>>(
            M.data.get(), d.get(), M.rows, M.cols);
}

// ---------------------------------------------------------------------------
// GPU factor correlation: mean column-wise cosine similarity
// Used for NMF convergence check without downloading factors to host.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void column_cosine_kernel(
    const Scalar* __restrict__ A,   // k × m, col-major
    const Scalar* __restrict__ B,   // k × m, col-major
    double* __restrict__ sims,      // m similarities (output)
    int k, int m)
{
    int col = blockIdx.x;
    if (col >= m) return;

    const Scalar* a = A + static_cast<size_t>(col) * k;
    const Scalar* b = B + static_cast<size_t>(col) * k;

    extern __shared__ double corr_smem[];
    double* s_dot = corr_smem;
    double* s_n1 = corr_smem + blockDim.x;
    double* s_n2 = corr_smem + 2 * blockDim.x;

    double dot_val = 0, n1_val = 0, n2_val = 0;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        double ai = static_cast<double>(a[i]);
        double bi = static_cast<double>(b[i]);
        dot_val += ai * bi;
        n1_val  += ai * ai;
        n2_val  += bi * bi;
    }
    s_dot[threadIdx.x] = dot_val;
    s_n1[threadIdx.x]  = n1_val;
    s_n2[threadIdx.x]  = n2_val;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            s_n1[threadIdx.x]  += s_n1[threadIdx.x + stride];
            s_n2[threadIdx.x]  += s_n2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double norm1 = sqrt(s_n1[0]);
        double norm2 = sqrt(s_n2[0]);
        sims[col] = (norm1 > 1e-10 && norm2 > 1e-10)
                     ? s_dot[0] / (norm1 * norm2) : 0.0;
    }
}

template<int BLOCK_SIZE_ = 256>
__global__ void mean_reduce_kernel(
    const double* __restrict__ vals,
    double* __restrict__ result,
    int n)
{
    __shared__ double reduce_smem[BLOCK_SIZE_];
    double sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += vals[i];
    reduce_smem[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduce_smem[threadIdx.x] += reduce_smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        *result = reduce_smem[0] / n;
}

/**
 * @brief Compute mean column-wise cosine similarity between two factors on GPU.
 *
 * Returns a single double: mean_i( dot(A[:,i], B[:,i]) / (||A[:,i]|| · ||B[:,i]||) ).
 * Only downloads a single scalar to host.
 */
template<typename Scalar>
double compute_factor_correlation_gpu(
    const GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& curr,
    const DenseMatrixGPU<Scalar>& prev)
{
    int k = curr.rows;
    int m = curr.cols;

    // Per-column cosine similarities
    double* d_sims = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_sims, m * sizeof(double), ctx.stream));

    // Block size: next power of 2 up to 256
    int tpc = 1;
    while (tpc < k && tpc < 256) tpc *= 2;

    size_t smem = 3 * tpc * sizeof(double);
    column_cosine_kernel<<<m, tpc, smem, ctx.stream>>>(
        curr.data.get(), prev.data.get(), d_sims, k, m);

    // Single-block mean reduction
    double* d_mean = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_mean, sizeof(double), ctx.stream));
    mean_reduce_kernel<<<1, 256, 0, ctx.stream>>>(d_sims, d_mean, m);

    double corr;
    CUDA_CHECK(cudaMemcpyAsync(&corr, d_mean, sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    CUDA_CHECK(cudaFreeAsync(d_sims, ctx.stream));
    CUDA_CHECK(cudaFreeAsync(d_mean, ctx.stream));

    return corr;
}

}  // namespace gpu
}  // namespace FactorNet
