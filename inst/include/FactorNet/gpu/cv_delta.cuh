/**
 * @file cv_delta.cuh
 * @brief GPU kernels for CV Gram/RHS delta computation.
 *
 * For cross-validation NMF, each column j has a modified Gram matrix:
 *   G_train_j = G_full - delta_G_j
 * where delta_G_j = Σ_{i ∈ test(j)} W[:,i] * W[:,i]ᵀ
 *
 * Similarly for the RHS:
 *   b_train_j = b_full_j - delta_b_j
 * where delta_b_j = Σ_{i ∈ test(j)} A_ij * W[:,i]
 *
 * These kernels precompute delta_G and delta_b on GPU so that the
 * CV NNLS batch solver can run entirely on device.
 *
 * Single kernel variant: global-memory atomicAdd, portable across all
 * GPU architectures and all k values.
 *
 * Two modes:
 *   mask_zeros=TRUE:  test entries come from non-zeros only
 *   mask_zeros=FALSE: test entries from all m rows (including zeros)
 */

#pragma once

#include "types.cuh"
#include <FactorNet/rng/rng.hpp>

namespace FactorNet {
namespace gpu {



// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Global-memory delta kernel for very large k where smem doesn't fit.
// Writes directly to per-column global memory via atomicAdd.
// Slower than smem version but works for any k.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void compute_cv_deltas_h_global_kernel(
    const Scalar* __restrict__ values,
    const int*    __restrict__ row_idx,
    const int*    __restrict__ col_ptr,
    int n,
    const Scalar* __restrict__ W,
    int k, int m,
    uint64_t rng_state,
    uint64_t inv_prob,
    Scalar* __restrict__ delta_G,
    Scalar* __restrict__ delta_b,
    Scalar* __restrict__ B_full,
    const __half* __restrict__ W_half,
    const int*    __restrict__ mask_col_ptr  = nullptr,
    const int*    __restrict__ mask_row_idx  = nullptr)
{
    const int j = blockIdx.x;
    if (j >= n) return;
    const int tid = threadIdx.x;

    // Global memory offsets for this column
    Scalar* g_dG = delta_G + static_cast<size_t>(j) * k * k;
    Scalar* g_db = delta_b + static_cast<size_t>(j) * k;
    Scalar* g_bf = B_full ? (B_full + static_cast<size_t>(j) * k) : nullptr;

    int start = col_ptr[j];
    int end   = col_ptr[j + 1];

    for (int p = start + tid; p < end; p += blockDim.x) {
        int i = row_idx[p];
        Scalar v = values[p];

        if (is_user_masked_device(i, j, mask_col_ptr, mask_row_idx)) {
            // User-masked: exclude from both train and test RHS (not in B_full),
            // but add outer product to delta_G so G_train correctly excludes this feature.
            for (int a = 0; a < k; ++a) {
                Scalar wa = load_factor(W, W_half, a + i * k);
                for (int b = a; b < k; ++b) {
                    Scalar contrib = wa * load_factor(W, W_half, b + i * k);
                    atomicAdd(&g_dG[a + b * k], contrib);
                    if (a != b) atomicAdd(&g_dG[b + a * k], contrib);
                }
            }
            continue;
        }

        if (g_bf) {
            for (int f = 0; f < k; ++f) {
                atomicAdd(&g_bf[f], v * load_factor(W, W_half, f + i * k));
            }
        }

        uint64_t h = rng_state
                   + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL
                   + static_cast<uint64_t>(j) * 0x6c62272e07bb0142ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        h ^= h >> 31;
        bool is_test = (inv_prob > 0) && (h < (0xFFFFFFFFFFFFFFFFULL / inv_prob));

        if (is_test) {
            for (int f = 0; f < k; ++f) {
                atomicAdd(&g_db[f], v * load_factor(W, W_half, f + i * k));
            }
            for (int a = 0; a < k; ++a) {
                Scalar wa = load_factor(W, W_half, a + i * k);
                for (int b = a; b < k; ++b) {
                    Scalar contrib = wa * load_factor(W, W_half, b + i * k);
                    atomicAdd(&g_dG[a + b * k], contrib);
                    if (a != b) atomicAdd(&g_dG[b + a * k], contrib);
                }
            }
        }
    }
}

template<typename Scalar>
__global__ void compute_cv_deltas_w_global_kernel(
    const Scalar* __restrict__ values,
    const int*    __restrict__ col_idx,
    const int*    __restrict__ row_ptr,
    int m,
    const Scalar* __restrict__ H,
    int k, int n,
    uint64_t rng_state,
    uint64_t inv_prob,
    Scalar* __restrict__ delta_G,
    Scalar* __restrict__ delta_b,
    Scalar* __restrict__ B_full,
    const __half* __restrict__ H_half,
    const int*    __restrict__ mask_col_ptr  = nullptr,
    const int*    __restrict__ mask_row_idx  = nullptr)
{
    const int i = blockIdx.x;
    if (i >= m) return;
    const int tid = threadIdx.x;

    Scalar* g_dG = delta_G + static_cast<size_t>(i) * k * k;
    Scalar* g_db = delta_b + static_cast<size_t>(i) * k;
    Scalar* g_bf = B_full ? (B_full + static_cast<size_t>(i) * k) : nullptr;

    int start = row_ptr[i];
    int end   = row_ptr[i + 1];

    for (int p = start + tid; p < end; p += blockDim.x) {
        int j = col_idx[p];
        Scalar v = values[p];

        if (is_user_masked_device(i, j, mask_col_ptr, mask_row_idx)) {
            // User-masked: exclude from both train and test RHS (not in B_full),
            // but add outer product to delta_G so G_train correctly excludes this feature.
            for (int a = 0; a < k; ++a) {
                Scalar ha = load_factor(H, H_half, a + j * k);
                for (int b = a; b < k; ++b) {
                    Scalar contrib = ha * load_factor(H, H_half, b + j * k);
                    atomicAdd(&g_dG[a + b * k], contrib);
                    if (a != b) atomicAdd(&g_dG[b + a * k], contrib);
                }
            }
            continue;
        }

        if (g_bf) {
            for (int f = 0; f < k; ++f) {
                atomicAdd(&g_bf[f], v * load_factor(H, H_half, f + j * k));
            }
        }

        uint64_t h = rng_state
                   + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL
                   + static_cast<uint64_t>(j) * 0x6c62272e07bb0142ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        h ^= h >> 31;
        bool is_test = (inv_prob > 0) && (h < (0xFFFFFFFFFFFFFFFFULL / inv_prob));

        if (is_test) {
            for (int f = 0; f < k; ++f) {
                atomicAdd(&g_db[f], v * load_factor(H, H_half, f + j * k));
            }
            for (int a = 0; a < k; ++a) {
                Scalar ha = load_factor(H, H_half, a + j * k);
                for (int b = a; b < k; ++b) {
                    Scalar contrib = ha * load_factor(H, H_half, b + j * k);
                    atomicAdd(&g_dG[a + b * k], contrib);
                    if (a != b) atomicAdd(&g_dG[b + a * k], contrib);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Correction kernel for mask_zeros=FALSE: add zero-entry test contributions
// to delta_G that the sparse kernel missed.
//
// For mask_zeros=FALSE, test entries include zero positions. The sparse kernel
// only iterates over non-zeros, so zero-position test entries are not included
// in delta_G. This kernel iterates over ALL rows per column, skips non-zeros
// (already handled), and adds outer products for zero-position test entries.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void add_zero_test_deltas_h_kernel(
    const int*    __restrict__ row_idx,
    const int*    __restrict__ col_ptr,
    int n, int m,
    const Scalar* __restrict__ W,
    int k,
    uint64_t rng_state,
    uint64_t inv_prob,
    Scalar* __restrict__ delta_G,
    const int*    __restrict__ mask_col_ptr,
    const int*    __restrict__ mask_row_idx)
{
    const int j = blockIdx.x;
    if (j >= n) return;

    Scalar* g_dG = delta_G + static_cast<size_t>(j) * k * k;

    int nz_start = col_ptr[j];
    int nz_end   = col_ptr[j + 1];

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        // Binary search: is row i a non-zero in column j?
        bool is_nz = false;
        {
            int lo = nz_start, hi = nz_end;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                int r = row_idx[mid];
                if (r < i) lo = mid + 1;
                else if (r > i) hi = mid;
                else { is_nz = true; break; }
            }
        }
        if (is_nz) continue; // already handled by sparse kernel

        // Check user mask for this zero entry
        bool add_to_delta = false;
        if (mask_col_ptr) {
            add_to_delta = is_user_masked_device(i, j, mask_col_ptr, mask_row_idx);
        }

        // Check CV holdout
        if (!add_to_delta) {
            uint64_t h = rng_state
                       + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL
                       + static_cast<uint64_t>(j) * 0x6c62272e07bb0142ULL;
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
            h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
            h ^= h >> 31;
            add_to_delta = (inv_prob > 0) && (h < (0xFFFFFFFFFFFFFFFFULL / inv_prob));
        }

        if (add_to_delta) {
            for (int a = 0; a < k; ++a) {
                Scalar wa = W[a + i * k];
                for (int b = a; b < k; ++b) {
                    Scalar contrib = wa * W[b + i * k];
                    atomicAdd(&g_dG[a + b * k], contrib);
                    if (a != b) atomicAdd(&g_dG[b + a * k], contrib);
                }
            }
        }
    }
}

/**
 * @brief Compute H-update CV deltas on GPU.
 *
 * Launches one block per column. Each block iterates over the column's
 * non-zeros, classifies train/test, and accumulates delta_G, delta_b.
 * Uses parallel accumulation with atomicAdd for scalability.
 * Falls back to global-memory atomics for very large k where smem doesn't fit.
 *
 * When mask_zeros=FALSE, launches a second correction kernel that adds
 * Gram delta contributions from zero-valued test entries (which the sparse
 * kernel cannot see).
 *
 * @param delta_G Output: k*k*n flattened array (one k×k block per column)
 * @param delta_b Output: k*n flattened array
 * @param B_full  Output: k*n full RHS (optional, can be nullptr)
 */
template<typename Scalar>
void compute_cv_deltas_h_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W,       // k × m
    int k, int n,
    uint64_t rng_state,
    uint64_t inv_prob,
    DenseMatrixGPU<Scalar>& delta_G,       // k*k × n
    DenseMatrixGPU<Scalar>& delta_b,       // k × n
    DenseMatrixGPU<Scalar>* B_full = nullptr, // k × n (optional)
    const __half* W_half = nullptr,          // mixed precision (optional)
    const int* mask_col_ptr = nullptr,       // user mask CSC col pointer (device, optional)
    const int* mask_row_idx = nullptr,       // user mask CSC row indices (device, optional)
    bool mask_zeros = true)                  // TRUE: test set from non-zeros only
{
    // Zero outputs
    CUDA_CHECK(cudaMemsetAsync(delta_G.data.get(), 0,
                                static_cast<size_t>(k) * k * n * sizeof(Scalar), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(delta_b.data.get(), 0,
                                static_cast<size_t>(k) * n * sizeof(Scalar), ctx.stream));
    if (B_full) {
        CUDA_CHECK(cudaMemsetAsync(B_full->data.get(), 0,
                                    static_cast<size_t>(k) * n * sizeof(Scalar), ctx.stream));
    }

    int threads = max(k, 128);
    threads = min(threads, 512);

    compute_cv_deltas_h_global_kernel<Scalar><<<n, threads, 0, ctx.stream>>>(
        A.values.get(), A.row_indices.get(), A.col_ptr.get(),
        n, W.data.get(), k, W.cols,
        rng_state, inv_prob,
        delta_G.data.get(), delta_b.data.get(),
        B_full ? B_full->data.get() : nullptr,
        W_half, mask_col_ptr, mask_row_idx);

    // For mask_zeros=FALSE, add missing Gram corrections from zero-position
    // test entries. The sparse kernel above only sees non-zeros.
    if (!mask_zeros) {
        int m = W.cols;
        int corr_threads = min(max(m / 4, 128), 512);
        add_zero_test_deltas_h_kernel<Scalar><<<n, corr_threads, 0, ctx.stream>>>(
            A.row_indices.get(), A.col_ptr.get(),
            n, m, W.data.get(), k,
            rng_state, inv_prob,
            delta_G.data.get(),
            mask_col_ptr, mask_row_idx);
    }
}

/**
 * @brief Compute W-update CV deltas on CPU and upload.
 *
 * The W update needs per-ROW deltas from a CSC matrix, which doesn't
 * map well to a simple GPU kernel without atomics. Since k is small,
 * the CPU computation is O(nnz * k²) which is fast enough.
 *
 * @param H_host  H factor on host (k × n, column-major)
 * @param deltas  Output struct with delta_G, delta_b, B_full
 */
// ==========================================================================
// GPU W-update delta computation using transposed matrix A^T
// ==========================================================================

// ---------------------------------------------------------------------------
// Correction kernel for mask_zeros=FALSE (W-update direction): add
// zero-entry test contributions to delta_G that the sparse kernel missed.
//
// Operates on A^T in CSC format. Each block handles row i of A.
// Iterates all n columns, skips non-zeros, adds outer products for
// zero-position test entries.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void add_zero_test_deltas_w_kernel(
    const int*    __restrict__ col_idx,   // At.row_indices (col indices of A's row)
    const int*    __restrict__ row_ptr,   // At.col_ptr (row pointers of A)
    int m, int n,
    const Scalar* __restrict__ H,
    int k,
    uint64_t rng_state,
    uint64_t inv_prob,
    Scalar* __restrict__ delta_G,
    const int*    __restrict__ mask_col_ptr,
    const int*    __restrict__ mask_row_idx)
{
    const int i = blockIdx.x;
    if (i >= m) return;

    Scalar* g_dG = delta_G + static_cast<size_t>(i) * k * k;

    int nz_start = row_ptr[i];
    int nz_end   = row_ptr[i + 1];

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        // Binary search: is column j a non-zero in row i of A?
        bool is_nz = false;
        {
            int lo = nz_start, hi = nz_end;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                int c = col_idx[mid];
                if (c < j) lo = mid + 1;
                else if (c > j) hi = mid;
                else { is_nz = true; break; }
            }
        }
        if (is_nz) continue;

        bool add_to_delta = false;
        if (mask_col_ptr) {
            add_to_delta = is_user_masked_device(i, j, mask_col_ptr, mask_row_idx);
        }

        if (!add_to_delta) {
            uint64_t h = rng_state
                       + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL
                       + static_cast<uint64_t>(j) * 0x6c62272e07bb0142ULL;
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
            h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
            h ^= h >> 31;
            add_to_delta = (inv_prob > 0) && (h < (0xFFFFFFFFFFFFFFFFULL / inv_prob));
        }

        if (add_to_delta) {
            for (int a = 0; a < k; ++a) {
                Scalar ha = H[a + j * k];
                for (int b = a; b < k; ++b) {
                    Scalar contrib = ha * H[b + j * k];
                    atomicAdd(&g_dG[a + b * k], contrib);
                    if (a != b) atomicAdd(&g_dG[b + a * k], contrib);
                }
            }
        }
    }
}



/**
 * @brief Compute CSC transpose: converts A (CSC, m×n) to A^T (CSC, n×m).
 *
 * The CSC representation of A^T is equivalent to CSR of A.
 * Output arrays must be pre-sized: row_ptr[m+1], col_idx[nnz], values_out[nnz].
 */
template<typename Scalar>
void compute_csc_transpose(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    int m, int n, int nnz,
    std::vector<int>& row_ptr_out,
    std::vector<int>& col_idx_out,
    std::vector<Scalar>& values_out)
{
    // Count nnz per row of A
    row_ptr_out.assign(m + 1, 0);
    for (int p = 0; p < nnz; ++p) {
        row_ptr_out[row_idx[p] + 1]++;
    }
    // Prefix sum
    for (int i = 0; i < m; ++i) {
        row_ptr_out[i + 1] += row_ptr_out[i];
    }

    col_idx_out.resize(nnz);
    values_out.resize(nnz);

    // Scatter entries to CSR positions
    std::vector<int> cursor(row_ptr_out.begin(), row_ptr_out.begin() + m);
    for (int j = 0; j < n; ++j) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int i = row_idx[p];
            int pos = cursor[i]++;
            col_idx_out[pos] = j;
            values_out[pos] = values[p];
        }
    }
}

/**
 * @brief Compute W-update CV deltas on GPU using transposed matrix.
 *
 * Uses A^T stored in CSC format on GPU. One block per row of A.
 * Eliminates the need to download H to host, compute on CPU, and upload.
 *
 * When mask_zeros=FALSE, launches a second correction kernel that adds
 * Gram delta contributions from zero-valued test entries.
 *
 * @param At     A^T in CSC format on GPU (n×m, col i has row i's entries)
 * @param H      Current factor H (k × n) on GPU
 * @param delta_G Output: k*k*m per-row Gram deltas on GPU
 * @param delta_b Output: k*m per-row RHS deltas on GPU
 * @param B_full  Output: k*m full RHS on GPU (optional)
 */
template<typename Scalar>
void compute_cv_deltas_w_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& At,    // A^T in CSC (n×m)
    const DenseMatrixGPU<Scalar>& H,       // k × n
    int k, int m,
    uint64_t rng_state,
    uint64_t inv_prob,
    DenseMatrixGPU<Scalar>& delta_G,       // k*k × m
    DenseMatrixGPU<Scalar>& delta_b,       // k × m
    DenseMatrixGPU<Scalar>* B_full = nullptr, // k × m (optional)
    const __half* H_half = nullptr,          // mixed precision (optional)
    const int* mask_col_ptr = nullptr,       // user mask CSC col pointer (device, optional)
    const int* mask_row_idx = nullptr,       // user mask CSC row indices (device, optional)
    bool mask_zeros = true)                  // TRUE: test set from non-zeros only
{
    // Zero outputs
    CUDA_CHECK(cudaMemsetAsync(delta_G.data.get(), 0,
                                static_cast<size_t>(k) * k * m * sizeof(Scalar), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(delta_b.data.get(), 0,
                                static_cast<size_t>(k) * m * sizeof(Scalar), ctx.stream));
    if (B_full) {
        CUDA_CHECK(cudaMemsetAsync(B_full->data.get(), 0,
                                    static_cast<size_t>(k) * m * sizeof(Scalar), ctx.stream));
    }

    int threads = max(k, 128);
    threads = min(threads, 512);

    compute_cv_deltas_w_global_kernel<Scalar><<<m, threads, 0, ctx.stream>>>(
        At.values.get(), At.row_indices.get(), At.col_ptr.get(),
        m, H.data.get(), k, H.cols,
        rng_state, inv_prob,
        delta_G.data.get(), delta_b.data.get(),
        B_full ? B_full->data.get() : nullptr,
        H_half, mask_col_ptr, mask_row_idx);

    // For mask_zeros=FALSE, add missing Gram corrections from zero-position
    // test entries.
    if (!mask_zeros) {
        int n = H.cols;
        int corr_threads = min(max(n / 4, 128), 512);
        add_zero_test_deltas_w_kernel<Scalar><<<m, corr_threads, 0, ctx.stream>>>(
            At.row_indices.get(), At.col_ptr.get(),
            m, n, H.data.get(), k,
            rng_state, inv_prob,
            delta_G.data.get(),
            mask_col_ptr, mask_row_idx);
    }
}

} // namespace gpu
} // namespace FactorNet
