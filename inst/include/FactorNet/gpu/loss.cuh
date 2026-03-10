/**
 * @file loss.cuh
 * @brief GPU loss computation using Gram trick for all NMF modes.
 *
 * Provides O(nnz·k + k²·n) loss computation for standard and CV NMF,
 * avoiding the O(m·n·k) direct reconstruction computation.
 *
 * Gram trick formula:
 *   ||A - WH||² = ||A||² - 2·Σⱼ bⱼᵀhⱼ + Σⱼ hⱼᵀGhⱼ
 *   where G = WᵀW (k×k), bⱼ = WᵀAⱼ (k×1)
 *
 * Three computation modes:
 *   1. Standard NMF total loss: Full Gram trick
 *   2. CV sparse (mask_zeros=TRUE): Direct at non-zeros (O(nnz·k))
 *   3. CV dense (mask_zeros=FALSE): Gram trick with train/test splitting
 *
 * GPU kernels:
 *   - Cross-term: fused with RHS (already computed as bⱼ = WᵀAⱼ)
 *   - Recon-term: h'Gh via BLAS-2 per column, or batched
 *   - ||A||²: single-pass reduction over sparse values
 *
 * For singlet parity, supports per-column-average MSE mode.
 */

#pragma once

#include "types.cuh"
#include "gram.cuh"
#include <cstdint>
#include <cmath>

namespace FactorNet {
namespace gpu {

// ---------------------------------------------------------------------------
// GPU kernel: compute ||A||² = sum of squared non-zero values
// Uses block reduction for efficiency.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void sparse_sq_norm_kernel(
    const Scalar* __restrict__ values,
    int nnz,
    double* __restrict__ result)          // double accumulator: avoids float overflow
{
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    for (int i = gid; i < nnz; i += stride) {
        double v = static_cast<double>(values[i]);
        local_sum += v * v;
    }

    smem[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, smem[0]);
}

// ---------------------------------------------------------------------------
// GPU kernel: cross term = Σⱼ bⱼᵀhⱼ  (dot product per column, then sum)
// B and H are k×n, column-major. One block per column chunk.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void cross_term_kernel(
    const Scalar* __restrict__ B,    // k × n  (RHS vectors)
    const Scalar* __restrict__ H,    // k × n  (factor)
    int k, int n,
    double* __restrict__ result)     // double accumulator: avoids float overflow
{
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    // Each thread handles one or more columns
    for (int j = gid; j < n; j += stride) {
        double dot = 0.0;
        for (int f = 0; f < k; ++f) {
            dot += static_cast<double>(B[f + j * k]) * static_cast<double>(H[f + j * k]);
        }
        local_sum += dot;
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, smem[0]);
}

// ---------------------------------------------------------------------------
// GPU kernel: recon term = Σⱼ hⱼᵀGhⱼ  (quadratic form per column, then sum)
// G is k×k, H is k×n. One thread per column.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void recon_term_kernel(
    const Scalar* __restrict__ G,    // k × k  (Gram matrix)
    const Scalar* __restrict__ H,    // k × n  (factor)
    int k, int n,
    double* __restrict__ result)     // double accumulator: avoids float overflow
{
    // Shared memory layout:
    //   [0, g_bytes)          : s_G  (Scalar, k*k)      — G loaded once per block
    //   [g_bytes_aligned, ...) : s_reduce (double, blockDim.x) — block reduction
    // g_bytes_aligned = (k*k*sizeof(Scalar) + 7) & ~7  to align to 8 bytes.
    extern __shared__ char smem_raw[];
    Scalar* s_G = reinterpret_cast<Scalar*>(smem_raw);
    const size_t g_bytes_aligned = (static_cast<size_t>(k) * k * sizeof(Scalar) + 7u) & ~7u;
    double* s_reduce = reinterpret_cast<double*>(smem_raw + g_bytes_aligned);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Cooperatively load G into shared memory
    for (int idx = tid; idx < k * k; idx += blockDim.x) {
        s_G[idx] = G[idx];
    }
    __syncthreads();

    double local_sum = 0.0;
    for (int j = gid; j < n; j += stride) {
        double quad = 0.0;
        for (int a = 0; a < k; ++a) {
            double Gh_a = 0.0;
            for (int b = 0; b < k; ++b) {
                Gh_a += static_cast<double>(s_G[a + b * k])
                      * static_cast<double>(H[b + j * k]);
            }
            quad += static_cast<double>(H[a + j * k]) * Gh_a;
        }
        local_sum += quad;
    }

    s_reduce[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, s_reduce[0]);
}

// ---------------------------------------------------------------------------
// GPU kernel: direct squared error at sparse non-zeros  (for CV mask_zeros=TRUE)
//
// For each (i,j,v) in A's non-zeros:
//   pred = W[:,i]' · H[:,j]     (W is k×m, stored col-major)
//   err = (v - pred)²
//
// Splits into train/test based on RNG hash.
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void cv_loss_sparse_kernel(
    const Scalar* __restrict__ values,
    const int*    __restrict__ row_idx,
    const int*    __restrict__ col_ptr,
    int n,
    const Scalar* __restrict__ W,     // k × m
    const Scalar* __restrict__ H,     // k × n
    const Scalar* __restrict__ d_vec, // k (scaling)
    int k,
    uint64_t rng_state,
    uint64_t inv_prob,
    uint64_t col_sub_thresh,
    // Outputs: [0]=train_sse, [1]=test_sse, [2]=n_train (as double), [3]=n_test (as double)
    double* __restrict__ out)           // [0]=train_sse, [1]=test_sse,
                                        // [2]=n_train, [3]=n_test,
                                        // [4]=total_nnz_pred_sq
{
    // Use double accumulators for all values including counts
    __shared__ double s_train[256];
    __shared__ double s_test[256];
    __shared__ double s_ntrain[256];
    __shared__ double s_ntest[256];
    __shared__ double s_pred_sq[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double local_train = 0, local_test = 0;
    double local_ntrain = 0, local_ntest = 0;
    double local_pred_sq = 0;

    int total_nnz = col_ptr[n];

    for (int idx = gid; idx < total_nnz; idx += stride) {
        // Binary search for column j: find j such that col_ptr[j] <= idx < col_ptr[j+1]
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (col_ptr[mid + 1] <= idx) lo = mid + 1;
            else hi = mid;
        }
        int j = lo;
        int i = row_idx[idx];

        // Compute prediction: pred = (d .* W[:,i])' · H[:,j]
        double pred = 0;
        for (int f = 0; f < k; ++f) {
            pred += (double)d_vec[f] * (double)W[f + i * k] * (double)H[f + j * k];
        }

        double err = (double)values[idx] - pred;
        double sq = err * err;
        local_pred_sq += pred * pred;

        // SplitMix64 RNG to match CPU exactly
        // Column subsample fast rejection
        bool is_test = false;
        if (inv_prob > 0) {
            bool col_in_sub = true;
            if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
                uint64_t ch = (rng_state ^ 0xa953b1c7217cc550ULL)
                            ^ ((uint64_t)j * 0x517cc1b727220a95ULL);
                ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
                ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
                ch ^= ch >> 31;
                col_in_sub = (ch < col_sub_thresh);
            }
            if (col_in_sub) {
                uint64_t h = rng_state
                           + (uint64_t)i * 0x9e3779b97f4a7c15ULL
                           + (uint64_t)j * 0x6c62272e07bb0142ULL;
                h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
                h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
                h ^= h >> 31;
                is_test = (h < (0xFFFFFFFFFFFFFFFFULL / inv_prob));
            }
        }

        if (is_test) {
            local_test += sq;
            local_ntest += 1.0;
        } else {
            local_train += sq;
            local_ntrain += 1.0;
        }
    }

    s_train[tid]   = local_train;
    s_test[tid]    = local_test;
    s_ntrain[tid]  = local_ntrain;
    s_ntest[tid]   = local_ntest;
    s_pred_sq[tid] = local_pred_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_train[tid]   += s_train[tid + s];
            s_test[tid]    += s_test[tid + s];
            s_ntrain[tid]  += s_ntrain[tid + s];
            s_ntest[tid]   += s_ntest[tid + s];
            s_pred_sq[tid] += s_pred_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out[0], s_train[0]);
        atomicAdd(&out[1], s_test[0]);
        atomicAdd(&out[2], s_ntrain[0]);
        atomicAdd(&out[3], s_ntest[0]);
        atomicAdd(&out[4], s_pred_sq[0]);
    }
}

// ---------------------------------------------------------------------------
// Host dispatch: Standard NMF total loss via Gram trick
//
// Returns MSE = ||A - WH||²_F / (m * n)
// where SSE = ||A||² - 2·cross + recon,
//       cross = Σⱼ bⱼ'hⱼ (B = W'A), recon = Σⱼ hⱼ'Ghⱼ (G = W'W).
//
// MSE matches CPU loss_tracker behavior: total squared error / total entries.
//
// IMPORTANT: Caller must ensure W and H are consistent with the desired
// reconstruction. For standard NMF, call either:
//   (a) After H update NNLS, before scale_rows: H is unscaled, W unchanged
//       → computes ||A - W * H_raw||², the correct reconstruction error
//   (b) With d absorbed into W: W_D = diag(d)*W, H normalized
//       → computes ||A - W_D * H_norm||², also correct
//
// All computation on GPU.
// ---------------------------------------------------------------------------

template<typename Scalar>
double compute_total_loss_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W,      // k × m
    const DenseMatrixGPU<Scalar>& H,      // k × n
    const DenseMatrixGPU<Scalar>& G,      // k × k (pre-computed Gram = W'W)
    const DenseMatrixGPU<Scalar>& B,      // k × n (pre-computed RHS = W'A)
    DeviceMemory<double>& scratch,        // ≥ 3 doubles (double to avoid float overflow)
    int m_rows = 0, int n_cols = 0)       // if >0, returns MSE = SSE/(m*n)
{
    const int k = W.rows;
    const int n = H.cols;
    const int nnz = A.nnz;

    // Zero the 3-element accumulator: [A_sq, cross, recon]
    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, 3 * sizeof(double), ctx.stream));

    // 1. ||A||² (parallel reduction over nnz values)
    int threads = 256;
    int blocks = min((nnz + threads - 1) / threads, 1024);
    sparse_sq_norm_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        A.values.get(), nnz, scratch.get() + 0);

    // 2. Cross term: Σⱼ bⱼ'hⱼ
    //    B already contains W'A (the RHS from most recent H update).
    //    But B may have been overwritten by NNLS. We need the ORIGINAL B.
    //    Caller must pass the saved pre-NNLS RHS.
    blocks = min((n + threads - 1) / threads, 1024);
    cross_term_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        B.data.get(), H.data.get(), k, n, scratch.get() + 1);

    // 3. Recon term: Σⱼ hⱼ'Ghⱼ  (G cached in shared memory)
    // Shared memory: s_G (Scalar k*k, 8-byte aligned) + s_reduce (double, threads)
    size_t g_bytes_aligned = (static_cast<size_t>(k) * k * sizeof(Scalar) + 7u) & ~7u;
    size_t recon_smem = g_bytes_aligned + static_cast<size_t>(threads) * sizeof(double);
    recon_term_kernel<Scalar><<<blocks, threads, recon_smem, ctx.stream>>>(
        G.data.get(), H.data.get(), k, n, scratch.get() + 2);

    // Download the 3 terms (already double on device)
    double terms[3];
    CUDA_CHECK(cudaMemcpyAsync(terms, scratch.get(), 3 * sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);

    double A_sq  = terms[0];
    double cross = terms[1];
    double recon = terms[2];

    double sse = A_sq - 2.0 * cross + recon;

    // Return MSE if dimensions provided, otherwise raw SSE
    if (m_rows > 0 && n_cols > 0) {
        double total_entries = static_cast<double>(m_rows) * static_cast<double>(n_cols);
        return std::max(0.0, sse) / total_entries;
    }
    return sse;
}

// ---------------------------------------------------------------------------
// Host dispatch: Dense NMF total loss via Gram trick (precomputed ||A||²)
//
// Same formula as sparse variant but ||A||² is precomputed once (constant
// across iterations for dense input) instead of recomputed from sparse values.
// This avoids downloading B_save, G, Hd to host for host-side loops.
// ---------------------------------------------------------------------------

template<typename Scalar>
double compute_total_loss_gpu_dense(
    const GPUContext& ctx,
    double A_sq_norm,                     // precomputed ||A||²_F
    const DenseMatrixGPU<Scalar>& H,      // k × n (with d absorbed: Hd)
    const DenseMatrixGPU<Scalar>& G,      // k × k (pre-computed Gram = W'W)
    const DenseMatrixGPU<Scalar>& B,      // k × n (pre-computed RHS = W'A, pre-NNLS)
    DeviceMemory<double>& scratch)        // ≥ 2 doubles
{
    const int k = H.rows;
    const int n = H.cols;

    // Zero the 2-element accumulator: [cross, recon]
    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, 2 * sizeof(double), ctx.stream));

    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);

    // Cross term: Σⱼ bⱼ'hⱼ
    cross_term_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        B.data.get(), H.data.get(), k, n, scratch.get() + 0);

    // Recon term: Σⱼ hⱼ'Ghⱼ
    size_t g_bytes_aligned = (static_cast<size_t>(k) * k * sizeof(Scalar) + 7u) & ~7u;
    size_t recon_smem = g_bytes_aligned + static_cast<size_t>(threads) * sizeof(double);
    recon_term_kernel<Scalar><<<blocks, threads, recon_smem, ctx.stream>>>(
        G.data.get(), H.data.get(), k, n, scratch.get() + 1);

    double terms[2];
    CUDA_CHECK(cudaMemcpyAsync(terms, scratch.get(), 2 * sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);

    return A_sq_norm - 2.0 * terms[0] + terms[1];
}

// ---------------------------------------------------------------------------
// Host dispatch: Precompute ||A||²_F for dense matrix
//
// Reuses sparse_sq_norm_kernel on the flat m*n array.
// Called once before the iteration loop; result is constant.
// ---------------------------------------------------------------------------

template<typename Scalar>
double compute_dense_sq_norm_gpu(
    const GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& A,      // m × n dense on device
    DeviceMemory<double>& scratch)        // ≥ 1 double
{
    const int total = A.rows * A.cols;
    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, sizeof(double), ctx.stream));

    int threads = 256;
    int blocks = min((total + threads - 1) / threads, 1024);
    sparse_sq_norm_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        A.data.get(), total, scratch.get());

    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, scratch.get(), sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);
    return result;
}

// ---------------------------------------------------------------------------
// Host dispatch: Compute recon_term only (for Gram-approx test loss)
// ---------------------------------------------------------------------------

template<typename Scalar>
double compute_recon_term_gpu(
    const GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& G,   // k × k
    const DenseMatrixGPU<Scalar>& H,   // k × n
    DeviceMemory<double>& scratch)     // ≥ 1 double (double to avoid float overflow)
{
    const int k = G.rows;
    const int n = H.cols;

    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, sizeof(double), ctx.stream));

    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);
    size_t g_bytes_aligned = (static_cast<size_t>(k) * k * sizeof(Scalar) + 7u) & ~7u;
    size_t recon_smem = g_bytes_aligned + static_cast<size_t>(threads) * sizeof(double);
    recon_term_kernel<Scalar><<<blocks, threads, recon_smem, ctx.stream>>>(
        G.data.get(), H.data.get(), k, n, scratch.get());

    double val;
    CUDA_CHECK(cudaMemcpyAsync(&val, scratch.get(), sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);
    return val;
}

// ---------------------------------------------------------------------------
// Host dispatch: CV loss for mask_zeros=TRUE (sparse, direct O(nnz·k))
//
// Computes train/test SSE by iterating over all non-zeros on GPU.
// Each non-zero is classified as train or test using RNG hash.
// Returns (train_mse, test_mse).
// ---------------------------------------------------------------------------

template<typename Scalar>
void compute_cv_loss_sparse_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W,      // k × m
    const DenseMatrixGPU<Scalar>& H,      // k × n
    const DeviceMemory<Scalar>& d_vec,    // k
    uint64_t rng_state,
    uint64_t inv_prob,
    double& train_mse,
    double& test_mse,
    uint64_t col_sub_thresh = 0xFFFFFFFFFFFFFFFFULL)
{
    const int k = W.rows;
    const int n = H.cols;

    // Allocate 5-element output: [train_sse, test_sse, n_train, n_test, nnz_pred_sq]
    DeviceMemory<double> d_out(5);
    CUDA_CHECK(cudaMemsetAsync(d_out.get(), 0, 5 * sizeof(double), ctx.stream));

    int threads = 256;
    int blocks = min((A.nnz + threads - 1) / threads, 2048);
    cv_loss_sparse_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        A.values.get(), A.row_indices.get(), A.col_ptr.get(),
        n, W.data.get(), H.data.get(), d_vec.get(), k,
        rng_state, inv_prob, col_sub_thresh, d_out.get());

    double h_out[5];
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.get(), 5 * sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);

    int64_t n_train = static_cast<int64_t>(h_out[2]);
    int64_t n_test  = static_cast<int64_t>(h_out[3]);
    train_mse = (n_train > 0) ? h_out[0] / n_train : 0;
    test_mse  = (n_test  > 0) ? h_out[1] / n_test  : 0;
}

// ---------------------------------------------------------------------------
// Host dispatch: CV loss for mask_zeros=FALSE (Gram trick with splitting)
//
// Uses:
//   Total loss = ||A||² - 2·cross + recon  (Gram trick, all entries)
//   Test cross term: only at test non-zeros
//   Test recon ≈ holdout_fraction × total_recon (statistical approx)
//   Test loss = A_test_sq - 2·test_cross + test_recon
//   Train loss = Total - Test
//
// This avoids iterating over all m×n test entries.
// ---------------------------------------------------------------------------

template<typename Scalar>
void compute_cv_loss_dense_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W,      // k × m
    const DenseMatrixGPU<Scalar>& H,      // k × n
    const DeviceMemory<Scalar>& d_vec,    // k
    const DenseMatrixGPU<Scalar>& G,      // k × k (pre-computed Gram)
    const DenseMatrixGPU<Scalar>& B,      // k × n (pre-computed RHS = W'A)
    double holdout_fraction,
    uint64_t rng_state,
    uint64_t inv_prob,
    int64_t n_train,
    int64_t n_test,
    double A_train_sq,
    double A_test_sq,
    DeviceMemory<double>& scratch,        // ≥ 4 doubles (double for float overflow safety)
    double& train_mse,
    double& test_mse)
{
    const int k = W.rows;
    const int n = H.cols;

    // 1. Total recon = Σⱼ hⱼ'Ghⱼ
    double total_recon = compute_recon_term_gpu(ctx, G, H, scratch);

    // 2. Total cross = Σⱼ bⱼ'hⱼ  (from pre-computed B = W'A)
    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, sizeof(double), ctx.stream));
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);
    cross_term_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        B.data.get(), H.data.get(), k, n, scratch.get());
    double total_cross_val;
    CUDA_CHECK(cudaMemcpyAsync(&total_cross_val, scratch.get(), sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);
    double total_cross = total_cross_val;

// 3. Run sparse CV kernel to get train/test SSE at non-zeros AND total nnz pred²
    DeviceMemory<double> d_cv_out(5);
    CUDA_CHECK(cudaMemsetAsync(d_cv_out.get(), 0, 5 * sizeof(double), ctx.stream));
    blocks = min((A.nnz + threads - 1) / threads, 2048);
    cv_loss_sparse_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        A.values.get(), A.row_indices.get(), A.col_ptr.get(),
        n, W.data.get(), H.data.get(), d_vec.get(), k,
        rng_state, inv_prob, 0xFFFFFFFFFFFFFFFFULL, d_cv_out.get());

    double cv_out[5];
    CUDA_CHECK(cudaMemcpyAsync(cv_out, d_cv_out.get(), 5 * sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);

    double train_sse_nnz  = cv_out[0];   // Σ_{train∩nnz} (A-pred)²
    double test_sse_nnz   = cv_out[1];   // Σ_{test∩nnz} (A-pred)²
    double total_nnz_pred_sq = cv_out[4]; // Σ_{nnz} pred²

    // 4. Compute exact total loss via Gram trick
    double A_sq_total = A_train_sq + A_test_sq;
    double total_loss = A_sq_total - 2.0 * total_cross + total_recon;

    // 5. Split zero-entry pred² between train and test
    //    total_recon = Σ_{all (i,j)} pred² (from Gram trick)
    //    total_zero_pred_sq = total_recon - total_nnz_pred_sq
    //    Approximate: test_zero_pred_sq ≈ holdout_fraction * total_zero_pred_sq
    //    This is the error at zero entries in the test set: (0 - pred)² = pred²
    double total_zero_pred_sq = fmax(0.0, total_recon - total_nnz_pred_sq);
    double test_zero_pred_sq  = holdout_fraction * total_zero_pred_sq;
    double train_zero_pred_sq = total_zero_pred_sq - test_zero_pred_sq;

    // 6. Final train/test SSE
    //    test_sse  = test_sse_nnz  + test_zero_pred_sq
    //    train_sse = train_sse_nnz + train_zero_pred_sq
    //    Verification: test_sse + train_sse ≈ total_loss (approximately)
    double test_sse_total  = test_sse_nnz  + test_zero_pred_sq;
    double train_sse_total = train_sse_nnz + train_zero_pred_sq;

    // Clamp to non-negative (numerical safety)
    train_sse_total = fmax(0.0, train_sse_total);
    test_sse_total  = fmax(0.0, test_sse_total);

    train_mse = (n_train > 0) ? train_sse_total / n_train : 0;
    test_mse  = (n_test  > 0) ? test_sse_total  / n_test  : 0;
}


// ---------------------------------------------------------------------------
// GPU kernel: Frobenius inner product of two k×k matrices  — O(k²)
//   result = Σᵢⱼ G1[i,j] · G2[i,j]
// Used to reduce recon_term from O(k²·n) to O(k²) via:
//   Σⱼ hⱼ'Ghⱼ = tr(G · H·H') = <G, G_H>_F
// ---------------------------------------------------------------------------

template<typename Scalar>
__global__ void gram_frobenius_ip_kernel(
    const Scalar* __restrict__ G1,  // k × k
    const Scalar* __restrict__ G2,  // k × k
    int k2,                          // total elements = k*k
    Scalar* __restrict__ result)
{
    __shared__ Scalar smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    Scalar local_sum = 0;
    for (int i = gid; i < k2; i += stride)
        local_sum += G1[i] * G2[i];

    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, smem[0]);
}

// ---------------------------------------------------------------------------
// Fast total NMF loss via Gram trick with O(k²) recon reduction
//
// Computes:  MSE = (||A||² - 2·cross + recon) / (m·n)
//
// Three terms:
//   ||A||²  — precomputed constant A_sq_host, no kernel needed
//   cross   — cublas dot product on k·n floats/doubles: <B_save, Hd>_F  ← O(k·n), BLAS-1
//   recon   — cuBLAS GEMM for G_Hd = Hd·Hd' (k×k) + k² FIP with G   ← O(k²·n) in GEMM (tensor cores) + O(k²) final sum
//
// The recon GEMM is ≈100× faster than recon_term_kernel because:
//   - Uses TF32/FP32 tensor cores (not custom scalar kernel)
//   - Produces a k×k output vs iterating over n columns
//
// IMPORTANT: Caller provides d_G_tmp as a scratch k×k matrix for G_Hd.
//            Also passes A_sq_host pre-computed once at NMF start.
//
// @param d_G        k×k  Gram of W (already computed for current W)
// @param d_B_save   k×n  RHS = W'A from H update (pre-NNLS)
// @param d_Hd       k×n  H with d absorbed: Hd = H * diag(d)
// @param d_G_tmp    k×k  scratch for G_Hd = Hd·Hd'
// @param d_scratch  ≥ 2 scalars scratch (cross, recon)
// @param A_sq_host  precomputed ||A||² (scalar on host)
// @param m, n       matrix dimensions for MSE denominator (0 = return SSE)
// ---------------------------------------------------------------------------

template<typename Scalar>
double compute_total_loss_fast_gpu(
    const GPUContext& ctx,
    const DenseMatrixGPU<Scalar>& d_G,
    const DenseMatrixGPU<Scalar>& d_B_save,
    const DenseMatrixGPU<Scalar>& d_Hd,
    DenseMatrixGPU<Scalar>& d_G_tmp,         // scratch k×k for G_Hd
    DeviceMemory<Scalar>& d_scratch,          // ≥ 2 scalars
    double A_sq_host,
    int m = 0, int n_cols = 0)
{
    const int k = d_G.rows;
    const int n = d_Hd.cols;
    const int kn = k * n;
    const int k2 = k * k;

    // ---- 1. Cross term: <B_save, Hd>_F via cuBLAS dot ----
    // Treats both k×n matrices as flat 1D vectors of k*n elements.
    // cross = Σ_{f,j} B_save[f,j] * Hd[f,j] = Σⱼ bⱼ'hⱼ
    Scalar cross_val = 0;
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDdot(ctx.cublas, kn,
            d_B_save.data.get(), 1,
            d_Hd.data.get(),     1,
            &cross_val));
    } else {
        CUBLAS_CHECK(cublasSdot(ctx.cublas, kn,
            d_B_save.data.get(), 1,
            d_Hd.data.get(),     1,
            &cross_val));
    }

    // ---- 2. Recon term: <G, G_Hd>_F ----
    // Step 2a: G_Hd = Hd · Hd'  (cuBLAS SGEMM/DGEMM, k×k)
    compute_gram_gpu(ctx, d_Hd, d_G_tmp);

    // Step 2b: recon = <G, G_Hd>_F  (Frobenius inner product, O(k²))
    CUDA_CHECK(cudaMemsetAsync(d_scratch.get(), 0, sizeof(Scalar), ctx.stream));
    int threads = 256;
    int blocks = (k2 + threads - 1) / threads;
    gram_frobenius_ip_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        d_G.data.get(), d_G_tmp.data.get(), k2, d_scratch.get());

    Scalar recon_val;
    CUDA_CHECK(cudaMemcpyAsync(&recon_val, d_scratch.get(), sizeof(Scalar),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);

    // ---- 3. Combine ----
    double sse = A_sq_host
               - 2.0 * static_cast<double>(cross_val)
               + static_cast<double>(recon_val);

    if (m > 0 && n_cols > 0)
        return std::max(0.0, sse) / (static_cast<double>(m) * n_cols);
    return sse;
}

// ---------------------------------------------------------------------------
// Precompute ||A||² on GPU — call once at NMF start
// Returns the scalar on host.
// ---------------------------------------------------------------------------

template<typename Scalar>
double precompute_A_sq_gpu(const GPUContext& ctx,
                           const SparseMatrixGPU<Scalar>& A,
                           DeviceMemory<Scalar>& scratch)
{
    CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0, sizeof(Scalar), ctx.stream));
    int threads = 256;
    int blocks = std::min((A.nnz + threads - 1) / threads, 1024);
    sparse_sq_norm_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
        A.values.get(), A.nnz, scratch.get());
    Scalar val;
    CUDA_CHECK(cudaMemcpyAsync(&val, scratch.get(), sizeof(Scalar),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);
    return static_cast<double>(val);
}

// =========================================================================
// GPU non-MSE (explicit) loss computation over sparse nonzeros
//
// For non-MSE losses (GP, NB, Gamma, InvGauss, Tweedie, KL, MAE, Huber),
// there is no Gram trick — we must iterate over all nonzeros and compute
// the per-element loss contribution.
//
// This replaces the host-mediated explicit_loss_raw_csc() path which
// previously required downloading W, H, d and computing loss on the CPU.
//
// Kernel: 1 thread per nonzero, each computes pred = W_Td[:,row] · H[:,col]
//         and evaluates the distribution-specific loss.
// Reduction: double-precision block reduction + global atomicAdd.
// =========================================================================

/**
 * @brief Device-side per-element loss computation.
 * Mirrors compute_loss() from math/loss.hpp.
 */
template<typename Scalar>
__device__ __forceinline__ double compute_loss_device(
    double observed, double predicted,
    int loss_type,
    double huber_delta,
    double power_param,
    double dispersion)
{
    predicted = fmax(predicted, 1e-10);

    switch (loss_type) {
    case 0: {  // MSE
        double d = observed - predicted;
        return d * d;
    }
    case 1: {  // MAE
        return fabs(observed - predicted);
    }
    case 2: {  // Huber
        double d = fabs(observed - predicted);
        return (d <= huber_delta) ? 0.5 * d * d : huber_delta * d - 0.5 * huber_delta * huber_delta;
    }
    case 3: {  // KL
        double y = fmax(observed, 1e-10);
        return y * log(y / predicted) - (y - predicted);
    }
    case 4: {  // GP
        double s = predicted;
        double y = observed;
        double th = dispersion;
        double one_pt = 1.0 + th;
        double loss = -log(s / one_pt);
        if (y >= 1.0) {
            double inner = fmax((s + th * y) / one_pt, 1e-10);
            loss -= (y - 1.0) * log(inner);
        }
        loss += (s + th * y) / one_pt;
        return loss;
    }
    case 5: {  // NB
        double y = observed;
        double mu = predicted;
        double r = fmax(dispersion, 1e-10);
        return -lgamma(y + r) + lgamma(r) - r * log(r / (r + mu)) - y * log(mu / (r + mu));
    }
    case 6: {  // Gamma
        double y = fmax(observed, 1e-10);
        double mu = predicted;
        return 2.0 * (-log(y / mu) + (y - mu) / mu);
    }
    case 7: {  // InvGauss
        double y = fmax(observed, 1e-10);
        double mu = predicted;
        double diff = y - mu;
        return diff * diff / (mu * mu * y);
    }
    case 8: {  // Tweedie
        double y = fmax(observed, 1e-10);
        double mu = predicted;
        double pp = power_param;
        if (fabs(pp - 1.0) < 1e-6)
            return 2.0 * (y * log(y / mu) - (y - mu));
        if (fabs(pp - 2.0) < 1e-6)
            return 2.0 * (-log(y / mu) + (y - mu) / mu);
        double omp = 1.0 - pp;
        double tmp = 2.0 - pp;
        return 2.0 * (pow(y, tmp) / (omp * tmp) - y * pow(mu, omp) / omp + pow(mu, tmp) / tmp);
    }
    default:
        return 0.0;
    }
}

/**
 * @brief GPU kernel: explicit loss over sparse nonzeros.
 *
 * Each thread handles one or more nonzeros. For each nonzero A(i,j):
 *   1. Compute pred = W_Td[:,i] · H[:,j]  (dot product, length k)
 *   2. Evaluate loss(A(i,j), pred) per the distribution
 *   3. Block-reduce and atomicAdd to global accumulator
 *
 * @param col_ptr   CSC column pointers (n+1)
 * @param row_idx   CSC row indices (nnz)
 * @param values    CSC nonzero values (nnz)
 * @param W_Td      k × m  factor (d-scaling absorbed), column-major
 * @param H         k × n  factor, column-major
 * @param disp      per-row dispersion vector (m), or nullptr
 * @param k         factorization rank
 * @param n         number of columns
 * @param nnz       number of nonzeros
 * @param loss_type 0=MSE,1=MAE,2=Huber,3=KL,4=GP,5=NB,6=Gamma,7=IG,8=Tweedie
 * @param huber_delta     Huber delta parameter
 * @param power_param     Tweedie power parameter
 * @param result    double accumulator (pre-zeroed by caller)
 */
template<typename Scalar>
__global__ void explicit_loss_sparse_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ W_Td,
    const Scalar* __restrict__ H,
    const Scalar* __restrict__ disp,
    int k, int n, int nnz,
    int loss_type,
    Scalar huber_delta,
    Scalar power_param,
    double* __restrict__ result)
{
    __shared__ double smem[256];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (int idx = gid; idx < nnz; idx += stride) {
        // Binary search for column index from CSC col_ptr
        // Since col_ptr is monotonically increasing, use upper_bound logic
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (col_ptr[mid + 1] <= idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        int j = lo;  // column index
        int i = row_idx[idx];  // row index

        // Compute prediction: dot(W_Td[:,i], H[:,j])
        double pred = 0.0;
        for (int f = 0; f < k; ++f)
            pred += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H[f + j * k]);

        double obs = static_cast<double>(values[idx]);
        double theta = disp ? static_cast<double>(disp[i]) : 0.0;

        local_sum += compute_loss_device<Scalar>(obs, pred, loss_type,
            static_cast<double>(huber_delta), static_cast<double>(power_param), theta);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, smem[0]);
}

/**
 * @brief Column-parallel variant: 1 block per column, threads cooperate
 *        on the nonzeros within that column (avoids binary search).
 *
 * For matrices with moderate nnz-per-column this is more efficient
 * than the flat-parallel kernel because column index is free.
 */
template<typename Scalar>
__global__ void explicit_loss_col_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ W_Td,
    const Scalar* __restrict__ H,
    const Scalar* __restrict__ disp,
    int k, int n,
    int loss_type,
    Scalar huber_delta,
    Scalar power_param,
    double* __restrict__ result)
{
    __shared__ double smem[256];
    const int j = blockIdx.x;  // one block per column
    if (j >= n) return;

    const int tid = threadIdx.x;
    const int start = col_ptr[j];
    const int end   = col_ptr[j + 1];

    double local_sum = 0.0;

    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        int i = row_idx[idx];

        // Compute prediction: dot(W_Td[:,i], H[:,j])
        double pred = 0.0;
        for (int f = 0; f < k; ++f)
            pred += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H[f + j * k]);

        double obs = static_cast<double>(values[idx]);
        double theta = disp ? static_cast<double>(disp[i]) : 0.0;

        local_sum += compute_loss_device<Scalar>(obs, pred, loss_type,
            static_cast<double>(huber_delta), static_cast<double>(power_param), theta);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, smem[0]);
}

/**
 * @brief Host dispatch: compute non-MSE loss entirely on GPU.
 *
 * Replaces the host-mediated explicit_loss_raw_csc() path that previously
 * downloaded W, H, d to host and ran OpenMP loops on CPU.
 *
 * W_Td must already have d-scaling absorbed (W_Td = W * diag(d)).
 *
 * @return Total loss (double)
 */
template<typename Scalar>
double compute_explicit_loss_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,     // sparse A (CSC on device)
    const DenseMatrixGPU<Scalar>& W_Td,   // k × m (d-scaled W)
    const DenseMatrixGPU<Scalar>& H,      // k × n
    const DeviceMemory<Scalar>& d_disp,   // m-length dispersion (or size 0)
    int loss_type,
    Scalar huber_delta = Scalar(1),
    Scalar power_param = Scalar(1.5))
{
    const int k = W_Td.rows;
    const int n = H.cols;
    const int nnz = A.nnz;

    DeviceMemory<double> d_result(1);
    CUDA_CHECK(cudaMemsetAsync(d_result.get(), 0, sizeof(double), ctx.stream));

    const Scalar* disp_ptr = (d_disp.size() > 0) ? d_disp.get() : nullptr;

    // Heuristic: use column-parallel kernel when avg nnz/col >= 32
    // (avoids binary search overhead and leverages column-local access)
    int avg_nnz_per_col = nnz / std::max(n, 1);

    if (avg_nnz_per_col >= 32) {
        // Column-parallel: 1 block per column
        int threads = 256;
        explicit_loss_col_kernel<Scalar><<<n, threads, 0, ctx.stream>>>(
            A.col_ptr.get(), A.row_indices.get(), A.values.get(),
            W_Td.data.get(), H.data.get(), disp_ptr,
            k, n, loss_type, huber_delta, power_param,
            d_result.get());
    } else {
        // Flat-parallel: 1 thread per nonzero with binary search
        int threads = 256;
        int blocks = std::min((nnz + threads - 1) / threads, 2048);
        explicit_loss_sparse_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
            A.col_ptr.get(), A.row_indices.get(), A.values.get(),
            W_Td.data.get(), H.data.get(), disp_ptr,
            k, n, nnz, loss_type, huber_delta, power_param,
            d_result.get());
    }

    double total_loss;
    CUDA_CHECK(cudaMemcpyAsync(&total_loss, d_result.get(), sizeof(double),
                                cudaMemcpyDeviceToHost, ctx.stream));
    cudaStreamSynchronize(ctx.stream);
    return total_loss;
}

} // namespace gpu
} // namespace FactorNet
