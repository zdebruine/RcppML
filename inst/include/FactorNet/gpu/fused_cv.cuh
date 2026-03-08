/**
 * @file fused_cv.cuh
 * @brief Fused delta+NNLS kernel for cross-validation NMF.
 *
 * Replaces the separate compute_cv_deltas + nnls_cv_batch pipeline with a
 * single kernel per column/row that eliminates O(k²×n) global memory for
 * delta_G and removes the global memory round-trip.
 *
 * Single kernel variant (baseline):
 *   - All threads iterate nnz via atomicAdd for B_full, delta_b, delta_G
 *   - Portable: works for any k, any nnz distribution, any GPU architecture
 *   - Thread count: adaptive based on average nnz per block
 *
 * Each kernel is parameterized by BLOCK_IS_COL (template bool):
 *   true  → H-update: block = column, inner index = row
 *   false → W-update: block = row, inner index = column
 * This only affects RNG hash ordering for consistent train/test masking.
 *
 * Shared memory: (k² + 4k) × sizeof(Scalar).
 * Portable: SM 7.0+ (V100), SM 8.0+ (A100), SM 9.0+ (H100).
 */

#pragma once

#include "types.cuh"
#include <FactorNet/rng/rng.hpp>

namespace FactorNet {
namespace gpu {

// ============================================================================
// Utility: optimized row scaling
// ============================================================================

template<typename Scalar>
__global__ void scale_rows_opt_kernel(Scalar* M, Scalar* d, int k, int n) {
    const int row = blockIdx.x;
    if (row >= k) return;

    __shared__ Scalar s_sum[256];
    const int tid = threadIdx.x;

    Scalar local_sum = 0;
    for (int j = tid; j < n; j += blockDim.x) {
        Scalar val = M[row + j * k];
        local_sum += val * val;
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    Scalar scale = (s_sum[0] > 0) ? sqrt(static_cast<double>(s_sum[0])) : 1;
    if (tid == 0) d[row] = scale;
    __syncthreads();

    Scalar inv_scale = Scalar(1) / scale;
    for (int j = tid; j < n; j += blockDim.x) {
        M[row + j * k] *= inv_scale;
    }
}

template<typename Scalar>
void scale_rows_opt_gpu(const GPUContext& ctx,
                        DenseMatrixGPU<Scalar>& M,
                        DeviceMemory<Scalar>& d) {
    scale_rows_opt_kernel<Scalar><<<M.rows, 256, 0, ctx.stream>>>(
        M.data.get(), d.get(), M.rows, M.cols);
}

// ============================================================================
// Hardware query: max shared memory per block (cached)
// ============================================================================

inline int get_max_smem_per_block() {
    static int cached = 0;
    if (cached == 0) {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &cached, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (cached == 0) cached = 49152;  // 48KB fallback
    }
    return cached;
}

// ============================================================================
// Portable thread count selection for baseline kernel
// ============================================================================

inline int compute_thread_count(int k, int total_nnz, int num_blocks) {
    int avg_nnz = (num_blocks > 0) ? (total_nnz / num_blocks) : 1;
    int ideal = avg_nnz / 4;

    int threads = 128;
    if (ideal > 128) threads = 256;
    if (ideal > 256) threads = 512;

    // For large k: ensure enough threads for parallel gradient reduction
    if (k >= 128) threads = max(threads, min(k, 512));
    else if (k >= 64) threads = max(threads, 256);

    threads = ((threads + 31) / 32) * 32;
    return min(threads, 1024);
}

// ============================================================================
// Device helpers — shared across all kernel variants
// ============================================================================

/**
 * @brief Classify a matrix entry as test or train.
 *
 * Maps (block_id, inner_id) to matrix coordinates (row, col) based on
 * BLOCK_IS_COL, then applies the deterministic hash-based RNG.
 * - BLOCK_IS_COL=true  (H-update): row = inner_id, col = block_id
 * - BLOCK_IS_COL=false (W-update): row = block_id, col = inner_id
 *
 * Column subsampling: if col_sub_thresh < UINT64_MAX, only columns whose
 * hash falls below col_sub_thresh are eligible for the test set. Columns
 * not in the subsample have all entries classified as train. Uses a
 * separate hash from the holdout RNG for independence.
 */
template<bool BLOCK_IS_COL>
__device__ __forceinline__ bool is_test_entry(
    int block_id, int inner_id,
    uint64_t rng_state, uint64_t inv_prob,
    uint64_t col_sub_thresh = 0xFFFFFFFFFFFFFFFFULL)
{
    uint64_t row_id, col_id;
    if constexpr (BLOCK_IS_COL) {
        row_id = static_cast<uint64_t>(inner_id);
        col_id = static_cast<uint64_t>(block_id);
    } else {
        row_id = static_cast<uint64_t>(block_id);
        col_id = static_cast<uint64_t>(inner_id);
    }

    // Column subsample fast rejection
    if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
        uint64_t ch = (rng_state ^ 0xa953b1c7217cc550ULL)
                    ^ (col_id * 0x517cc1b727220a95ULL);
        ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
        ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
        ch ^= ch >> 31;
        if (ch >= col_sub_thresh) return false;
    }

    // Standard holdout check  
    uint64_t h = rng_state
               + row_id * 0x9e3779b97f4a7c15ULL
               + col_id * 0x6c62272e07bb0142ULL;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return (inv_prob > 0) && (h < (0xFFFFFFFFFFFFFFFFULL / inv_prob));
}

/**
 * @brief Mirror upper triangle of symmetric G matrix to lower triangle.
 */
template<typename Scalar>
__device__ __forceinline__ void mirror_upper_triangle(
    Scalar* s_G, int k, int tid, int nthreads)
{
    for (int idx = tid; idx < k * k; idx += nthreads) {
        int a = idx % k, b = idx / k;
        if (a > b) s_G[a + b * k] = s_G[b + a * k];
    }
    __syncthreads();
}

/**
 * @brief NNLS coordinate descent solver operating on shared memory.
 *
 * Computes b_train = B_full - delta_b, initializes h (warm-start or from b_train), then runs CD.
 * Serial CD for k < 64; parallel warp-reduced gradient for k >= 64.
 * Reuses s_db as warp reduction workspace for k >= 64.
 *
 * @param nonneg  When true, enforces h >= 0 (standard NMF).
 *                When false, allows negative values (Semi-NMF).
 * @param warm_start When true, reads existing h from H_out instead of initializing from b_train.
 */
template<typename Scalar>
__device__ void solve_nnls_cd(
    Scalar* s_G, Scalar* s_b, Scalar* s_h,
    Scalar* s_db, Scalar* s_bf,
    int k, int max_cd_iter,
    int tid, int nthreads,
    bool nonneg = true,
    const Scalar* H_out = nullptr, int col = 0, bool warm_start = false)
{
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    // b_train = B_full - delta_b; initialize h
    for (int idx = tid; idx < k; idx += nthreads) {
        s_b[idx] = s_bf[idx] - s_db[idx];
        if (warm_start && H_out) {
            s_h[idx] = H_out[col * k + idx];
        } else {
            s_h[idx] = nonneg ? max(Scalar(0), s_b[idx]) : s_b[idx];
        }
    }
    __syncthreads();

    if (k >= 64) {
        // Parallel gradient: warp-level reduction
        Scalar* s_warp_buf = s_db;  // reuse delta_b as workspace
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
                    for (int w = 0; w < nwarps; ++w) grad += s_warp_buf[w];
                    Scalar g_diag = s_G[coord * k + coord];
                    if (g_diag > 0) {
                        Scalar new_val = s_h[coord] - grad / g_diag;
                        s_h[coord] = nonneg ? max(Scalar(0), new_val) : new_val;
                    }
                }
                __syncthreads();
            }
        }
    } else {
        // Serial CD for small k
        if (tid == 0) {
            for (int iter = 0; iter < max_cd_iter; ++iter) {
                bool any_changed = false;
                for (int coord = 0; coord < k; ++coord) {
                    Scalar grad = -s_b[coord];
                    for (int l = 0; l < k; ++l)
                        grad += s_G[coord * k + l] * s_h[l];
                    Scalar g_diag = s_G[coord * k + coord];
                    if (g_diag > 0) {
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
}

// ============================================================================
// Kernel A: Baseline — atomicAdd for all delta accumulation
//
// Suitable for any k and any nnz distribution. Falls back to this when
// specialized kernels aren't beneficial or don't fit in shared memory.
//
// Shared memory: (k² + 4k) × sizeof(Scalar)
// ============================================================================

template<typename Scalar, bool BLOCK_IS_COL>
__global__ void fused_baseline_kernel(
    const Scalar* __restrict__ values,
    const int*    __restrict__ indices,
    const int*    __restrict__ ptr,
    int num_blocks,
    const Scalar* __restrict__ factor_in,
    int k,
    const Scalar* __restrict__ G_full,
    uint64_t rng_state,
    uint64_t inv_prob,
    Scalar* __restrict__ factor_out,
    int max_cd_iter,
    uint64_t col_sub_thresh,
    const __half* __restrict__ factor_half,
    bool nonneg = true,
    bool warm_start = false,
    const int* __restrict__ mask_col_ptr = nullptr,
    const int* __restrict__ mask_row_idx = nullptr)
{
    const int bid = blockIdx.x;
    if (bid >= num_blocks) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem[];
    Scalar* s_G  = reinterpret_cast<Scalar*>(smem);
    Scalar* s_b  = s_G + k * k;
    Scalar* s_h  = s_b + k;
    Scalar* s_db = s_h + k;
    Scalar* s_bf = s_db + k;

    // Load G_full and zero accumulators
    for (int idx = tid; idx < k * k; idx += nthreads) s_G[idx] = G_full[idx];
    for (int idx = tid; idx < k; idx += nthreads) { s_db[idx] = 0; s_bf[idx] = 0; }
    __syncthreads();

    // Phase 1: In-place delta accumulation
    int start = ptr[bid];
    int end   = ptr[bid + 1];
    int block_nnz = end - start;

    for (int p_off = tid; p_off < block_nnz; p_off += nthreads) {
        int p = start + p_off;
        int inner = indices[p];
        Scalar v = values[p];

        // Determine (row, col) for mask lookup
        int mask_row, mask_col;
        if constexpr (BLOCK_IS_COL) {
            mask_row = inner;  // H-update: block=col, inner=row
            mask_col = bid;
        } else {
            mask_row = bid;    // W-update: block=row, inner=col
            mask_col = inner;
        }

        if (is_user_masked_device(mask_row, mask_col, mask_col_ptr, mask_row_idx)) {
            // User-masked: exclude from train/test RHS but add to delta_G
            // so G_train = G_full - delta_G correctly excludes this feature.
            for (int a = 0; a < k; ++a) {
                Scalar neg_fa = -load_factor(factor_in, factor_half, a + inner * k);
                for (int b = a; b < k; ++b)
                    atomicAdd(&s_G[a + b * k], neg_fa * load_factor(factor_in, factor_half, b + inner * k));
            }
            continue;
        }

        // B_full
        for (int f = 0; f < k; ++f)
            atomicAdd(&s_bf[f], v * load_factor(factor_in, factor_half, f + inner * k));

        if (is_test_entry<BLOCK_IS_COL>(bid, inner, rng_state, inv_prob, col_sub_thresh)) {
            // delta_b
            for (int f = 0; f < k; ++f)
                atomicAdd(&s_db[f], v * load_factor(factor_in, factor_half, f + inner * k));
            // delta_G: subtract outer product from G_full (upper triangle)
            for (int a = 0; a < k; ++a) {
                Scalar neg_fa = -load_factor(factor_in, factor_half, a + inner * k);
                for (int b = a; b < k; ++b)
                    atomicAdd(&s_G[a + b * k], neg_fa * load_factor(factor_in, factor_half, b + inner * k));
            }
        }
    }
    __syncthreads();

    mirror_upper_triangle(s_G, k, tid, nthreads);
    solve_nnls_cd(s_G, s_b, s_h, s_db, s_bf, k, max_cd_iter, tid, nthreads, nonneg,
                  factor_out, bid, warm_start);

    for (int idx = tid; idx < k; idx += nthreads)
        factor_out[idx + bid * k] = s_h[idx];
}

// ============================================================================
// ============================================================================
// Dispatch — Single baseline kernel for all k and all GPU architectures
// ============================================================================

/** @brief Check if fused kernel shared memory fits for rank k. */
template<typename Scalar>
bool fused_kernel_fits(int k) {
    size_t smem = (static_cast<size_t>(k) * k + 4 * k) * sizeof(Scalar);
    return smem <= static_cast<size_t>(get_max_smem_per_block());
}

/**
 * @brief Unified fused delta+NNLS dispatch.
 *
 * Uses the baseline kernel for all k and all GPU architectures.
 * No H-vs-W specific logic — the BLOCK_IS_COL template parameter only
 * affects RNG hash ordering inside the kernel.
 *
 * @tparam Scalar       float or double
 * @tparam BLOCK_IS_COL true for H-update (block = column of A),
 *                      false for W-update (block = row of A^T).
 * @param sparse     Sparse matrix (A for H-update, A^T for W-update)
 * @param factor_in  Factor matrix read during delta computation
 *                   (W for H-update, H for W-update)
 * @param G_full     Full Gram matrix (factor_in' × factor_in + regularization)
 * @param k          Factorization rank
 * @param num_blocks Number of blocks to solve (columns or rows)
 * @param rng_state  RNG state for train/test masking
 * @param inv_prob   Inverse holdout probability (e.g., 10 for 10%)
 * @param factor_out Output factor matrix (H for H-update, W for W-update)
 * @param max_cd_iter Maximum coordinate descent iterations
 * @return true if kernel was launched, false if k is too large for smem
 */
template<typename Scalar, bool BLOCK_IS_COL>
bool fused_delta_nnls_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& sparse,
    const DenseMatrixGPU<Scalar>& factor_in,
    const DenseMatrixGPU<Scalar>& G_full,
    int k, int num_blocks,
    uint64_t rng_state,
    uint64_t inv_prob,
    DenseMatrixGPU<Scalar>& factor_out,
    int max_cd_iter = 10,
    uint64_t col_sub_thresh = 0xFFFFFFFFFFFFFFFFULL,
    const __half* factor_half = nullptr,
    bool nonneg = true,
    bool warm_start = false,
    const int* mask_col_ptr = nullptr,
    const int* mask_row_idx = nullptr)
{
    size_t smem_base = (static_cast<size_t>(k) * k + 4 * k) * sizeof(Scalar);
    int max_smem = get_max_smem_per_block();

    if (smem_base > static_cast<size_t>(max_smem))
        return false;

    if (smem_base > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(
            fused_baseline_kernel<Scalar, BLOCK_IS_COL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_base)));

    int threads = compute_thread_count(k, sparse.nnz, num_blocks);
    fused_baseline_kernel<Scalar, BLOCK_IS_COL>
        <<<num_blocks, threads, smem_base, ctx.stream>>>(
        sparse.values.get(), sparse.row_indices.get(), sparse.col_ptr.get(),
        num_blocks, factor_in.data.get(), k, G_full.data.get(),
        rng_state, inv_prob, factor_out.data.get(), max_cd_iter,
        col_sub_thresh, factor_half, nonneg, warm_start,
        mask_col_ptr, mask_row_idx);

    return true;
}

// ============================================================================
// Backward-compatible API wrappers
//
// These preserve the existing call-site signatures used in nmf_cv_gpu.cuh,
// benchmark harnesses, and profiling tools.
// ============================================================================

/** @brief Fused delta+NNLS for H-update (block = column of A). */
template<typename Scalar>
bool fused_delta_nnls_h_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W,
    const DenseMatrixGPU<Scalar>& G_full,
    int k, int n,
    uint64_t rng_state,
    uint64_t inv_prob,
    DenseMatrixGPU<Scalar>& H,
    int max_cd_iter = 10,
    uint64_t col_sub_thresh = 0xFFFFFFFFFFFFFFFFULL,
    const __half* factor_half = nullptr,
    bool nonneg = true,
    bool warm_start = false,
    const int* mask_col_ptr = nullptr,
    const int* mask_row_idx = nullptr)
{
    return fused_delta_nnls_gpu<Scalar, /*BLOCK_IS_COL=*/true>(
        ctx, A, W, G_full, k, n, rng_state, inv_prob, H, max_cd_iter,
        col_sub_thresh, factor_half, nonneg, warm_start,
        mask_col_ptr, mask_row_idx);
}

/** @brief Fused delta+NNLS for W-update (block = row of A^T). */
template<typename Scalar>
bool fused_delta_nnls_w_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& At,
    const DenseMatrixGPU<Scalar>& H,
    const DenseMatrixGPU<Scalar>& G_full,
    int k, int m,
    uint64_t rng_state,
    uint64_t inv_prob,
    DenseMatrixGPU<Scalar>& W,
    int max_cd_iter = 10,
    uint64_t col_sub_thresh = 0xFFFFFFFFFFFFFFFFULL,
    const __half* factor_half = nullptr,
    bool nonneg = true,
    bool warm_start = false,
    const int* mask_col_ptr = nullptr,
    const int* mask_row_idx = nullptr)
{
    return fused_delta_nnls_gpu<Scalar, /*BLOCK_IS_COL=*/false>(
        ctx, At, H, G_full, k, m, rng_state, inv_prob, W, max_cd_iter,
        col_sub_thresh, factor_half, nonneg, warm_start,
        mask_col_ptr, mask_row_idx);
}

} // namespace gpu
} // namespace FactorNet
