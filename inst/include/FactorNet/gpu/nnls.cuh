/**
 * @file nnls.cuh
 * @brief GPU batch NNLS solver via coordinate descent.
 *
 * Solves min_{x≥0} ||Gx - b||² for all columns simultaneously on GPU.
 * Three kernel variants for different rank (k) ranges:
 *   - Fixed-rank template (k ≤ 64): Gram in registers/shared mem
 *   - Dynamic-rank: Gram in shared memory with dynamic allocation
 *   - Large-rank: Parallel gradient computation with warp reduction
 *
 * Also supports per-column Gram modification for CV NMF, where each column
 * solves with G_train = G_full - G_test_col.
 */

#pragma once

#include "types.cuh"

namespace FactorNet {
namespace gpu {

// ---------------------------------------------------------------------------
// Helper: conditional non-negativity clamping for Semi-NMF support
// When nonneg=true, clamps value to max(0, val) (standard NMF).
// When nonneg=false, allows negative values (Semi-NMF).
// ---------------------------------------------------------------------------
template<typename Scalar>
__device__ __forceinline__ Scalar clamp_nonneg(Scalar val, bool nonneg) {
    return nonneg ? max(val, Scalar(0)) : val;
}

// ---------------------------------------------------------------------------
// Standard batch NNLS kernel (one thread block per column)
//
// Each block solves: min_{h≥0} ||G·h - b||² via coordinate descent
// G (k×k) is shared across all columns; b (k) is per-column.
// Thread 0 runs the CD loop; other threads assist with memory loads.
// ---------------------------------------------------------------------------

template<int K, typename Scalar>
__global__ void nnls_kernel_fixed(
    const Scalar* __restrict__ G,   // k × k Gram matrix
    Scalar* __restrict__ B,         // k × n RHS (input) / solution (output)
    int n, int max_cd_iter,
    bool nonneg = true)
{
    __shared__ Scalar s_G[K * K];
    __shared__ Scalar s_h[K];

    const int j = blockIdx.x;
    if (j >= n) return;

    const int tid = threadIdx.x;

    // Cooperative load of G into shared memory
    for (int idx = tid; idx < K * K; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }

    // Initialize h from B
    if (tid < K) {
        s_h[tid] = clamp_nonneg(B[tid + j * K], nonneg);
    }
    __syncthreads();

    // Coordinate descent (serial within block)
    if (tid == 0) {
        for (int iter = 0; iter < max_cd_iter; ++iter) {
            bool any_changed = false;
            for (int coord = 0; coord < K; ++coord) {
                // Compute gradient: g_coord = G[coord,:] * h - b[coord]
                Scalar grad = -B[coord + j * K];
                for (int l = 0; l < K; ++l) {
                    grad += s_G[coord * K + l] * s_h[l];
                }

                // Update: h[coord] = max(0, h[coord] - grad / G[coord,coord])
                Scalar g_diag = s_G[coord * K + coord];
                if (g_diag > 0) {
                    Scalar old_val = s_h[coord];
                    s_h[coord] = clamp_nonneg(old_val - grad / g_diag, nonneg);
                    if (s_h[coord] != old_val) any_changed = true;
                }
            }
            if (!any_changed) break;
        }
    }
    __syncthreads();

    // Write solution back to B
    if (tid < K) {
        B[tid + j * K] = s_h[tid];
    }
}

// ---------------------------------------------------------------------------
// Dynamic-rank NNLS kernel (shared memory size determined at launch)
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_kernel_dynamic(
    const Scalar* __restrict__ G,
    Scalar* __restrict__ B,
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);
    Scalar* s_h = s_G + k * k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    if (tid < k) {
        s_h[tid] = clamp_nonneg(B[tid + j * k], nonneg);
    }
    __syncthreads();

    if (tid == 0) {
        for (int iter = 0; iter < max_cd_iter; ++iter) {
            bool any_changed = false;
            for (int coord = 0; coord < k; ++coord) {
                Scalar grad = -B[coord + j * k];
                for (int l = 0; l < k; ++l) {
                    grad += s_G[coord * k + l] * s_h[l];
                }
                Scalar g_diag = s_G[coord * k + coord];
                if (g_diag > 0) {
                    Scalar old_val = s_h[coord];
                    s_h[coord] = clamp_nonneg(old_val - grad / g_diag, nonneg);
                    if (s_h[coord] != old_val) any_changed = true;
                }
            }
            if (!any_changed) break;
        }
    }
    __syncthreads();

    if (tid < k) {
        B[tid + j * k] = s_h[tid];
    }
}

// ---------------------------------------------------------------------------
// CV batch NNLS kernel: per-column modified Gram
//
// Each column j has a different Gram: G_j = G_full - G_test_j
// G_test_j is the outer product sum of test-row W columns.
//
// delta_G: k*k*n array, column-major blocks of k×k per column
//          delta_G[:,j] = sum_{i in test(j)} W[:,i] * W[:,i]^T
//          G_train_j = G_full - delta_G[:,j]
//
// delta_b: k*n array, per-column RHS correction
//          b_train_j = b_full_j - delta_b[:,j]
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_cv_kernel(
    const Scalar* __restrict__ G_full,    // k × k full Gram
    const Scalar* __restrict__ delta_G,   // k * k * n delta per column
    const Scalar* __restrict__ B_full,    // k × n full RHS
    const Scalar* __restrict__ delta_b,   // k × n RHS correction
    Scalar* __restrict__ H,               // k × n output (also source for warm-start)
    int k, int n, int max_cd_iter,
    bool nonneg = true,
    bool warm_start = false)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);
    Scalar* s_h = s_G + k * k;
    Scalar* s_b = s_h + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;

    // Load G_train_j = G_full - delta_G_j
    const Scalar* dG_j = delta_G + static_cast<size_t>(j) * k * k;
    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G_full[idx] - dG_j[idx];
    }

    // Load b_train_j = B_full_j - delta_b_j
    if (tid < k) {
        s_b[tid] = B_full[tid + j * k] - delta_b[tid + j * k];
        // Warm-start: use existing H; Cold-start: initialize from RHS
        s_h[tid] = warm_start ? H[tid + j * k] : clamp_nonneg(s_b[tid], nonneg);
    }
    __syncthreads();

    // Coordinate descent
    if (tid == 0) {
        for (int iter = 0; iter < max_cd_iter; ++iter) {
            bool any_changed = false;
            for (int coord = 0; coord < k; ++coord) {
                Scalar grad = -s_b[coord];
                for (int l = 0; l < k; ++l) {
                    grad += s_G[coord * k + l] * s_h[l];
                }
                Scalar g_diag = s_G[coord * k + coord];
                if (g_diag > 0) {
                    Scalar old_val = s_h[coord];
                    s_h[coord] = clamp_nonneg(old_val - grad / g_diag, nonneg);
                    if (s_h[coord] != old_val) any_changed = true;
                }
            }
            if (!any_changed) break;
        }
    }
    __syncthreads();

    if (tid < k) {
        H[tid + j * k] = s_h[tid];
    }
}

// ---------------------------------------------------------------------------
// CV batch NNLS kernel with parallel gradient for large k
//
// Same as nnls_cv_kernel but uses all threads to compute the gradient
// via warp shuffle reduction. Mandatory for k >= 64 where serial gradient
// computation is O(k²) per coordinate per iteration.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_cv_kernel_parallel(
    const Scalar* __restrict__ G_full,    // k × k full Gram
    const Scalar* __restrict__ delta_G,   // k * k * n delta per column
    const Scalar* __restrict__ B_full,    // k × n full RHS
    const Scalar* __restrict__ delta_b,   // k × n RHS correction
    Scalar* __restrict__ H,               // k × n output (also source for warm-start)
    int k, int n, int max_cd_iter,
    bool nonneg = true,
    bool warm_start = false)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);
    Scalar* s_h = s_G + k * k;
    Scalar* s_b = s_h + k;
    // Warp reduction buffer follows s_b
    Scalar* s_warp_buf = s_b + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    // Load G_train_j = G_full - delta_G_j
    const Scalar* dG_j = delta_G + static_cast<size_t>(j) * k * k;
    for (int idx = tid; idx < k * k; idx += nthreads) {
        s_G[idx] = G_full[idx] - dG_j[idx];
    }

    // Load b_train_j = B_full_j - delta_b_j
    for (int idx = tid; idx < k; idx += nthreads) {
        s_b[idx] = B_full[idx + j * k] - delta_b[idx + j * k];
        s_h[idx] = warm_start ? H[idx + j * k] : clamp_nonneg(s_b[idx], nonneg);
    }
    __syncthreads();

    // Parallel coordinate descent: all threads compute gradient via reduction
    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            // Each thread computes partial dot product
            Scalar partial = 0;
            for (int l = tid; l < k; l += nthreads) {
                partial += s_G[coord * k + l] * s_h[l];
            }

            // Warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }

            // Lane 0 of each warp writes partial to shared
            if (lane_id == 0) {
                s_warp_buf[warp_id] = partial;
            }
            __syncthreads();

            // Thread 0 reduces across warps and updates
            if (tid == 0) {
                Scalar grad = -s_b[coord];
                for (int w = 0; w < nwarps; ++w) {
                    grad += s_warp_buf[w];
                }
                Scalar g_diag = s_G[coord * k + coord];
                if (g_diag > 0) {
                    s_h[coord] = clamp_nonneg(s_h[coord] - grad / g_diag, nonneg);
                }
            }
            __syncthreads();
        }
    }

    // Write solution
    for (int idx = tid; idx < k; idx += nthreads) {
        H[idx + j * k] = s_h[idx];
    }
}

// ---------------------------------------------------------------------------
// CV batch NNLS kernel with global memory reads for very large k.
// Only h and warp reduction buffer live in shared memory.
// G and delta_G are read from global memory (cached in L1/L2).
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_cv_kernel_global(
    const Scalar* __restrict__ G_full,
    const Scalar* __restrict__ delta_G,
    const Scalar* __restrict__ B_full,
    const Scalar* __restrict__ delta_b,
    Scalar* __restrict__ H,
    int k, int n, int max_cd_iter,
    bool nonneg = true,
    bool warm_start = false)
{
    extern __shared__ char smem[];
    Scalar* s_h = reinterpret_cast<Scalar*>(smem);
    Scalar* s_warp_buf = s_h + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    const Scalar* dG_j = delta_G + static_cast<size_t>(j) * k * k;

    // Initialize h from existing solution (warm-start) or b_train (cold-start)
    for (int idx = tid; idx < k; idx += nthreads) {
        s_h[idx] = warm_start ? H[idx + j * k]
                              : clamp_nonneg(B_full[idx + j * k] - delta_b[idx + j * k], nonneg);
    }
    __syncthreads();

    // Parallel CD: read G from global memory
    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            Scalar partial = 0;
            for (int l = tid; l < k; l += nthreads) {
                Scalar g_val = G_full[coord * k + l] - dG_j[coord * k + l];
                partial += g_val * s_h[l];
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            if (lane_id == 0) {
                s_warp_buf[warp_id] = partial;
            }
            __syncthreads();

            if (tid == 0) {
                Scalar grad = -(B_full[coord + j * k] - delta_b[coord + j * k]);
                for (int w = 0; w < nwarps; ++w) {
                    grad += s_warp_buf[w];
                }
                Scalar g_diag = G_full[coord * k + coord] - dG_j[coord * k + coord];
                if (g_diag > 0) {
                    s_h[coord] = clamp_nonneg(s_h[coord] - grad / g_diag, nonneg);
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < k; idx += nthreads) {
        H[idx + j * k] = s_h[idx];
    }
}

// ---------------------------------------------------------------------------
// Warp-batched NNLS kernel for k ≤ 32
//
// One warp (32 threads) per column, multiple warps per block.
// Gram loaded once into shared memory, shared across all warps.
// h and b stored in per-lane registers (lane i holds h[i] and b[i]).
// Gradient computed in parallel via warp shuffle — NO __syncthreads in CD loop.
// Early termination via __any_sync.
//
// Compared to nnls_kernel_dynamic (serial CD, 1 active thread per block):
//   - 32× more threads active (all lanes participate in gradient)
//   - Multi-column blocking amortizes G load and increases occupancy
//   - Warp-synchronous: no __syncthreads barriers during iteration
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_kernel_warp_batched(
    const Scalar* __restrict__ G,   // k × k Gram matrix
    Scalar* __restrict__ B,         // k × n RHS (in) / solution (out)
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int col = blockIdx.x * nwarps + warp_id;

    // All threads cooperate to load G into shared memory
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    __syncthreads();  // Only sync: after G load

    if (col >= n) return;

    // Each lane holds b[lane] and h[lane] in registers
    Scalar my_b = (lane < k) ? B[lane + col * k] : Scalar(0);
    Scalar my_h = clamp_nonneg(my_b, nonneg);

    // Warp-synchronous coordinate descent
    for (int iter = 0; iter < max_cd_iter; ++iter) {
        bool my_changed = false;

        for (int coord = 0; coord < k; ++coord) {
            // Parallel dot product: G[coord, lane] * h[lane]
            Scalar g_val = (lane < k) ? s_G[coord * k + lane] : Scalar(0);
            Scalar partial = g_val * my_h;

            // Warp reduction (all 32 lanes)
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            // Get b[coord] and h[coord] via shuffle
            Scalar b_coord = __shfl_sync(0xffffffff, my_b, coord);
            Scalar h_old = __shfl_sync(0xffffffff, my_h, coord);
            Scalar g_diag = s_G[coord * k + coord];

            // Lane 0 computes updated h[coord]
            Scalar new_h = h_old;
            if (lane == 0 && g_diag > Scalar(0)) {
                Scalar grad = partial - b_coord;
                new_h = clamp_nonneg(h_old - grad / g_diag, nonneg);
            }
            // Broadcast from lane 0 to all lanes
            new_h = __shfl_sync(0xffffffff, new_h, 0);

            // Only lane coord updates its register
            if (lane == coord) {
                if (my_h != new_h) my_changed = true;
                my_h = new_h;
            }
        }

        // Early termination: no coordinate changed in this sweep
        if (!__any_sync(0xffffffff, my_changed)) break;
    }

    // Write solution back
    if (lane < k) {
        B[lane + col * k] = my_h;
    }
}

template<typename Scalar>
__global__ void scale_rows_kernel(Scalar* M, Scalar* d, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    Scalar row_sq_sum = 0;
    for (int j = 0; j < n; ++j) {
        Scalar val = M[i + j * k];
        row_sq_sum += val * val;
    }

    Scalar scale = (row_sq_sum > 0) ? sqrt(static_cast<double>(row_sq_sum)) : 1;
    d[i] = scale;
    Scalar inv_scale = Scalar(1) / scale;
    for (int j = 0; j < n; ++j) {
        M[i + j * k] *= inv_scale;
    }
}

// ---------------------------------------------------------------------------
// L1 regularization kernel: subtract L1 from RHS, clamp to zero
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void apply_l1_kernel(Scalar* B, int k, int n, Scalar l1_penalty) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * n;
    if (idx < total) {
        B[idx] = max(Scalar(0), B[idx] - l1_penalty);
    }
}

// ---------------------------------------------------------------------------
// Column scaling kernel: multiply each column of M by d[row_index]
// M is k×n, d is k. M[f,j] *= d[f] for all j.
// Used to absorb diagonal scaling: W_scaled = diag(d) * W
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void absorb_d_kernel(Scalar* M, const Scalar* d, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * n;
    if (idx < total) {
        int f = idx % k;  // row (factor) index
        M[idx] *= d[f];
    }
}

// ---------------------------------------------------------------------------
// Dynamic-rank NNLS kernel with parallel gradient for large k
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_kernel_parallel(
    const Scalar* __restrict__ G,
    Scalar* __restrict__ B,
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);
    Scalar* s_h = s_G + k * k;
    Scalar* s_warp_buf = s_h + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    for (int idx = tid; idx < k * k; idx += nthreads) {
        s_G[idx] = G[idx];
    }
    for (int idx = tid; idx < k; idx += nthreads) {
        s_h[idx] = clamp_nonneg(B[idx + j * k], nonneg);
    }
    __syncthreads();

    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            Scalar partial = 0;
            for (int l = tid; l < k; l += nthreads) {
                partial += s_G[coord * k + l] * s_h[l];
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            if (lane_id == 0) {
                s_warp_buf[warp_id] = partial;
            }
            __syncthreads();

            if (tid == 0) {
                Scalar grad = -B[coord + j * k];
                for (int w = 0; w < nwarps; ++w) {
                    grad += s_warp_buf[w];
                }
                Scalar g_diag = s_G[coord * k + coord];
                if (g_diag > 0) {
                    Scalar old_val = s_h[coord];
                    s_h[coord] = clamp_nonneg(old_val - grad / g_diag, nonneg);
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < k; idx += nthreads) {
        B[idx + j * k] = s_h[idx];
    }
}

// ---------------------------------------------------------------------------
// Standard NNLS with global memory reads for very large k
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void nnls_kernel_global(
    const Scalar* __restrict__ G,
    Scalar* __restrict__ B,
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_h = reinterpret_cast<Scalar*>(smem);
    Scalar* s_warp_buf = s_h + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    for (int idx = tid; idx < k; idx += nthreads) {
        s_h[idx] = clamp_nonneg(B[idx + j * k], nonneg);
    }
    __syncthreads();

    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            Scalar partial = 0;
            for (int l = tid; l < k; l += nthreads) {
                partial += G[coord * k + l] * s_h[l];
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            if (lane_id == 0) {
                s_warp_buf[warp_id] = partial;
            }
            __syncthreads();

            if (tid == 0) {
                Scalar grad = -B[coord + j * k];
                for (int w = 0; w < nwarps; ++w) {
                    grad += s_warp_buf[w];
                }
                Scalar g_diag = G[coord * k + coord];
                if (g_diag > 0) {
                    s_h[coord] = clamp_nonneg(s_h[coord] - grad / g_diag, nonneg);
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < k; idx += nthreads) {
        B[idx + j * k] = s_h[idx];
    }
}

// ===========================================================================
// Warm-start NNLS kernels: separate B (RHS) and H (solution) arrays
//
// These kernels initialize h from the current H (warm start) rather than
// from max(0, B). This matches the CPU NNLS behavior and dramatically
// improves convergence when cd_max_iter is small (e.g. 10).
//
// Key difference from cold-start kernels:
//   Cold start: h <- max(0, B[j])         → starts far from solution
//   Warm start: h <- max(0, H_prev[j])    → starts near previous solution
// ===========================================================================

template<typename Scalar>
__global__ void nnls_kernel_warmstart_fixed(
    const Scalar* __restrict__ G,   // k × k Gram
    const Scalar* __restrict__ B,   // k × n RHS (read-only)
    Scalar* __restrict__ H,         // k × n solution (read: init, write: result)
    int n, int k, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);
    Scalar* s_h = s_G + k * k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;

    // Cooperative load of G into shared memory
    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    // Warm start: initialize from previous H (not from B)
    if (tid < k) {
        s_h[tid] = clamp_nonneg(H[tid + j * k], nonneg);
    }
    __syncthreads();

    if (tid == 0) {
        for (int iter = 0; iter < max_cd_iter; ++iter) {
            bool any_changed = false;
            for (int coord = 0; coord < k; ++coord) {
                Scalar grad = -B[coord + j * k];  // RHS from separate array
                for (int l = 0; l < k; ++l) {
                    grad += s_G[coord * k + l] * s_h[l];
                }
                Scalar g_diag = s_G[coord * k + coord];
                if (g_diag > 0) {
                    Scalar old_val = s_h[coord];
                    s_h[coord] = clamp_nonneg(old_val - grad / g_diag, nonneg);
                    if (s_h[coord] != old_val) any_changed = true;
                }
            }
            if (!any_changed) break;
        }
    }
    __syncthreads();

    if (tid < k) {
        H[tid + j * k] = s_h[tid];
    }
}

template<typename Scalar>
__global__ void nnls_kernel_warmstart_warp_batched(
    const Scalar* __restrict__ G,   // k × k Gram
    const Scalar* __restrict__ B,   // k × n RHS (read-only)
    Scalar* __restrict__ H,         // k × n solution (warm start + output)
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int col = blockIdx.x * nwarps + warp_id;

    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    __syncthreads();

    if (col >= n) return;

    // Warm start from H (not B)
    Scalar my_b = (lane < k) ? B[lane + col * k] : Scalar(0);
    Scalar my_h = (lane < k) ? clamp_nonneg(H[lane + col * k], nonneg) : Scalar(0);

    for (int iter = 0; iter < max_cd_iter; ++iter) {
        bool my_changed = false;

        for (int coord = 0; coord < k; ++coord) {
            Scalar g_val = (lane < k) ? s_G[coord * k + lane] : Scalar(0);
            Scalar partial = g_val * my_h;

            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            Scalar b_coord = __shfl_sync(0xffffffff, my_b, coord);
            Scalar h_old = __shfl_sync(0xffffffff, my_h, coord);
            Scalar g_diag = s_G[coord * k + coord];

            Scalar new_h = h_old;
            if (lane == 0 && g_diag > Scalar(0)) {
                Scalar grad = partial - b_coord;
                new_h = clamp_nonneg(h_old - grad / g_diag, nonneg);
            }
            new_h = __shfl_sync(0xffffffff, new_h, 0);

            if (lane == coord) {
                if (my_h != new_h) my_changed = true;
                my_h = new_h;
            }
        }

        if (!__any_sync(0xffffffff, my_changed)) break;
    }

    if (lane < k) {
        H[lane + col * k] = my_h;
    }
}

template<typename Scalar>
__global__ void nnls_kernel_warmstart_parallel(
    const Scalar* __restrict__ G,   // k × k Gram
    const Scalar* __restrict__ B,   // k × n RHS (read-only)
    Scalar* __restrict__ H,         // k × n solution (warm start + output)
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem);
    Scalar* s_h = s_G + k * k;
    Scalar* s_warp_buf = s_h + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    for (int idx = tid; idx < k * k; idx += nthreads) {
        s_G[idx] = G[idx];
    }
    // Warm start from H
    for (int idx = tid; idx < k; idx += nthreads) {
        s_h[idx] = clamp_nonneg(H[idx + j * k], nonneg);
    }
    __syncthreads();

    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            Scalar partial = 0;
            for (int l = tid; l < k; l += nthreads) {
                partial += s_G[coord * k + l] * s_h[l];
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            if (lane_id == 0) {
                s_warp_buf[warp_id] = partial;
            }
            __syncthreads();

            if (tid == 0) {
                Scalar grad = -B[coord + j * k];
                for (int w = 0; w < nwarps; ++w) {
                    grad += s_warp_buf[w];
                }
                Scalar g_diag = s_G[coord * k + coord];
                if (g_diag > 0) {
                    s_h[coord] = clamp_nonneg(s_h[coord] - grad / g_diag, nonneg);
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < k; idx += nthreads) {
        H[idx + j * k] = s_h[idx];
    }
}

template<typename Scalar>
__global__ void nnls_kernel_warmstart_global(
    const Scalar* __restrict__ G,   // k × k Gram
    const Scalar* __restrict__ B,   // k × n RHS (read-only)
    Scalar* __restrict__ H,         // k × n solution (warm start + output)
    int k, int n, int max_cd_iter,
    bool nonneg = true)
{
    extern __shared__ char smem[];
    Scalar* s_h = reinterpret_cast<Scalar*>(smem);
    Scalar* s_warp_buf = s_h + k;

    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;

    for (int idx = tid; idx < k; idx += nthreads) {
        s_h[idx] = clamp_nonneg(H[idx + j * k], nonneg);
    }
    __syncthreads();

    for (int iter = 0; iter < max_cd_iter; ++iter) {
        for (int coord = 0; coord < k; ++coord) {
            Scalar partial = 0;
            for (int l = tid; l < k; l += nthreads) {
                partial += G[coord * k + l] * s_h[l];
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            if (lane_id == 0) {
                s_warp_buf[warp_id] = partial;
            }
            __syncthreads();

            if (tid == 0) {
                Scalar grad = -B[coord + j * k];
                for (int w = 0; w < nwarps; ++w) {
                    grad += s_warp_buf[w];
                }
                Scalar g_diag = G[coord * k + coord];
                if (g_diag > 0) {
                    s_h[coord] = clamp_nonneg(s_h[coord] - grad / g_diag, nonneg);
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < k; idx += nthreads) {
        H[idx + j * k] = s_h[idx];
    }
}

// ---------------------------------------------------------------------------
// Host dispatch functions
// ---------------------------------------------------------------------------

/// Cold-start batch NNLS: B serves as both input RHS and output solution.
/// H is initialized from clamp_nonneg(B). Used by CV paths.
template<typename Scalar>
void nnls_batch_gpu(const GPUContext& ctx,
                    const DenseMatrixGPU<Scalar>& G,
                    DenseMatrixGPU<Scalar>& B,  // in: RHS, out: solution
                    int max_cd_iter = 10,
                    bool nonneg = true) {
    const int k = G.rows;
    const int n = B.cols;

    // Check hardware smem limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    if (k >= 64) {
        int threads = max(k, 128);
        threads = min(threads, 512);
        threads = ((threads + 31) / 32) * 32;
        int nwarps = threads / 32;
        size_t smem = (static_cast<size_t>(k) * k + k + nwarps) * sizeof(Scalar);

        if (smem <= static_cast<size_t>(max_smem) &&
            smem <= static_cast<size_t>(max_smem) / 4) {
            // Use smem path only when occupancy >= 4 blocks/SM
            if (smem > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    nnls_kernel_parallel<Scalar>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem)));
            }
            nnls_kernel_parallel<Scalar><<<n, threads, smem, ctx.stream>>>(
                G.data.get(), B.data.get(), k, n, max_cd_iter, nonneg);
        } else {
            // Global memory path: higher occupancy, G cached in L2
            size_t smem_small = (static_cast<size_t>(k) + nwarps) * sizeof(Scalar);
            nnls_kernel_global<Scalar><<<n, threads, smem_small, ctx.stream>>>(
                G.data.get(), B.data.get(), k, n, max_cd_iter, nonneg);
        }
    } else if (k <= 32) {
        // Warp-batched kernel: 1 warp per column, multi-column per block
        // Each 32-thread warp independently solves one column's NNLS via
        // parallel gradient (warp shuffle) — all threads active.
        int warps_per_block = 16;  // 512 threads = 16 columns per block
        int threads = warps_per_block * 32;
        int cols_per_block = warps_per_block;
        int blocks = (n + cols_per_block - 1) / cols_per_block;
        size_t smem = static_cast<size_t>(k) * k * sizeof(Scalar);

        nnls_kernel_warp_batched<Scalar><<<blocks, threads, smem, ctx.stream>>>(
            G.data.get(), B.data.get(), k, n, max_cd_iter, nonneg);
    } else {
        // k = 33-63: use parallel gradient kernel (was serial CD before)
        int threads = max(k, 64);
        threads = min(threads, 256);
        threads = ((threads + 31) / 32) * 32;
        int nwarps = threads / 32;
        size_t smem = (static_cast<size_t>(k) * k + k + nwarps) * sizeof(Scalar);

        nnls_kernel_parallel<Scalar><<<n, threads, smem, ctx.stream>>>(
            G.data.get(), B.data.get(), k, n, max_cd_iter, nonneg);
    }
}

/// Warm-start batch NNLS: B is the RHS (read-only), H is the solution (read-write).
/// H is warm-started from its current values (previous iteration).
/// This matches CPU NNLS behavior and improves convergence significantly.
template<typename Scalar>
void nnls_batch_warmstart_gpu(const GPUContext& ctx,
                              const DenseMatrixGPU<Scalar>& G,
                              const DenseMatrixGPU<Scalar>& B,  // in: RHS (read-only)
                              DenseMatrixGPU<Scalar>& H,        // in/out: warm-start solution
                              int max_cd_iter = 10,
                              bool nonneg = true) {
    const int k = G.rows;
    const int n = H.cols;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    if (k >= 64) {
        int threads = max(k, 128);
        threads = min(threads, 512);
        threads = ((threads + 31) / 32) * 32;
        int nwarps = threads / 32;
        size_t smem = (static_cast<size_t>(k) * k + k + nwarps) * sizeof(Scalar);

        if (smem <= static_cast<size_t>(max_smem) &&
            smem <= static_cast<size_t>(max_smem) / 4) {
            if (smem > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    nnls_kernel_warmstart_parallel<Scalar>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem)));
            }
            nnls_kernel_warmstart_parallel<Scalar><<<n, threads, smem, ctx.stream>>>(
                G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg);
        } else {
            size_t smem_small = (static_cast<size_t>(k) + nwarps) * sizeof(Scalar);
            nnls_kernel_warmstart_global<Scalar><<<n, threads, smem_small, ctx.stream>>>(
                G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg);
        }
    } else if (k <= 32) {
        int warps_per_block = 16;
        int threads = warps_per_block * 32;
        int cols_per_block = warps_per_block;
        int blocks = (n + cols_per_block - 1) / cols_per_block;
        size_t smem = static_cast<size_t>(k) * k * sizeof(Scalar);

        nnls_kernel_warmstart_warp_batched<Scalar><<<blocks, threads, smem, ctx.stream>>>(
            G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg);
    } else {
        int threads = max(k, 64);
        threads = min(threads, 256);
        threads = ((threads + 31) / 32) * 32;
        int nwarps = threads / 32;
        size_t smem = (static_cast<size_t>(k) * k + k + nwarps) * sizeof(Scalar);

        nnls_kernel_warmstart_parallel<Scalar><<<n, threads, smem, ctx.stream>>>(
            G.data.get(), B.data.get(), H.data.get(), k, n, max_cd_iter, nonneg);
    }
}

/// Batch CV NNLS: each column uses G_full - delta_G[:,j]
/// Uses parallel gradient for k >= 64 to avoid O(k³) serial bottleneck.
/// Falls back to global memory reads for very large k where smem doesn't fit.
template<typename Scalar>
void nnls_cv_batch_gpu(const GPUContext& ctx,
                       const DenseMatrixGPU<Scalar>& G_full,
                       const DenseMatrixGPU<Scalar>& delta_G,  // k*k*n
                       const DenseMatrixGPU<Scalar>& B_full,
                       const DenseMatrixGPU<Scalar>& delta_b,
                       DenseMatrixGPU<Scalar>& H,
                       int max_cd_iter = 10,
                       bool nonneg = true,
                       bool warm_start = false) {
    const int k = G_full.rows;
    const int n = H.cols;

    // Check hardware smem limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    if (k >= 64) {
        int threads = max(k, 128);
        threads = min(threads, 512);
        threads = ((threads + 31) / 32) * 32;
        int nwarps = threads / 32;
        size_t smem = (static_cast<size_t>(k) * k + 2 * k + nwarps) * sizeof(Scalar);

        if (smem <= static_cast<size_t>(max_smem)) {
            if (smem > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    nnls_cv_kernel_parallel<Scalar>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem)));
            }
            nnls_cv_kernel_parallel<Scalar><<<n, threads, smem, ctx.stream>>>(
                G_full.data.get(), delta_G.data.get(),
                B_full.data.get(), delta_b.data.get(),
                H.data.get(), k, n, max_cd_iter, nonneg, warm_start);
        } else {
            // Global memory fallback: only h + warp_buf in smem
            size_t smem_small = (static_cast<size_t>(k) + nwarps) * sizeof(Scalar);
            nnls_cv_kernel_global<Scalar><<<n, threads, smem_small, ctx.stream>>>(
                G_full.data.get(), delta_G.data.get(),
                B_full.data.get(), delta_b.data.get(),
                H.data.get(), k, n, max_cd_iter, nonneg, warm_start);
        }
    } else {
        int threads = max(k, 32);
        size_t smem = (static_cast<size_t>(k) * k + 2 * k) * sizeof(Scalar);

        nnls_cv_kernel<Scalar><<<n, threads, smem, ctx.stream>>>(
            G_full.data.get(), delta_G.data.get(),
            B_full.data.get(), delta_b.data.get(),
            H.data.get(), k, n, max_cd_iter, nonneg, warm_start);
    }
}

template<typename Scalar>
void scale_rows_gpu(const GPUContext& ctx,
                    DenseMatrixGPU<Scalar>& M,
                    DeviceMemory<Scalar>& d) {
    const int k = M.rows;
    scale_rows_kernel<Scalar><<<1, k, 0, ctx.stream>>>(
        M.data.get(), d.get(), k, M.cols);
}

template<typename Scalar>
void apply_l1_gpu(const GPUContext& ctx,
                  DenseMatrixGPU<Scalar>& B,
                  Scalar l1_penalty) {
    if (l1_penalty <= 0) return;
    int total = B.rows * B.cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_l1_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        B.data.get(), B.rows, B.cols, l1_penalty);
}

/// Absorb diagonal scaling d into factor M: M[f,j] *= d[f]
template<typename Scalar>
void absorb_d_gpu(const GPUContext& ctx,
                  DenseMatrixGPU<Scalar>& M,
                  const DeviceMemory<Scalar>& d) {
    int total = M.rows * M.cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    absorb_d_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        M.data.get(), d.get(), M.rows, M.cols);
}

} // namespace gpu
} // namespace FactorNet
