/**
 * @file bipartition.cuh
 * @brief GPU-accelerated spectral bipartitioning via rank-2 NMF.
 *
 * Port of the CPU bipartition algorithm to GPU using the k=2 specialized
 * kernels from gpu_k2.cuh. The bipartition algorithm:
 *
 *   1. Initialize W (2 × m) randomly
 *   2. ALS iterations (rank-2 NMF on subset of columns):
 *      a. H update: fused RHS+NNLS with closed-form 2×2 solver
 *      b. Scale H rows by L1-norm (sum, not L2!)
 *      c. W update: fused RHS+NNLS via A^T
 *      d. Scale W rows by L1-norm
 *      e. Convergence via correlation distance on W
 *   3. Post-processing:
 *      a. v = h[row0,:] - h[row1,:] (row order by d)
 *      b. Partition by sign(v)
 *      c. Optionally compute centroids and relative cosine distance
 *
 * Key differences from standard NMF gpu_k2:
 *   - Uses L1-norm scaling (row sums), not L2-norm (Euclidean)
 *   - Operates on a SUBSET of columns (indexed by `samples`)
 *   - Includes post-NMF bipartition logic (v-vector, centroids, dist)
 *   - Convergence by correlation distance, not loss-based
 *
 * For dclust, this function is the inner loop called repeatedly.
 * Performance target: <1ms per bipartition for 30K × 2700 matrices.
 */

#pragma once

#include "types.cuh"
#include "k2.cuh"

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <FactorNet/core/logging.hpp>

namespace FactorNet {
namespace gpu {

// ==========================================================================
// GPU bipartition result
// ==========================================================================

struct GPUBipartitionResult {
    std::vector<double> v;                  // Bipartition vector (h[0]-h[1] or h[1]-h[0])
    double dist;                            // Relative cosine distance (-1 if not computed)
    unsigned int size1, size2;              // Cluster sizes
    std::vector<unsigned int> samples1;     // Sample indices in cluster 1
    std::vector<unsigned int> samples2;     // Sample indices in cluster 2
    std::vector<double> center1, center2;   // Cluster centroids (empty if !calc_dist)
};

// ==========================================================================
// L1-norm scaling kernel for k=2 (bipartition uses sum-normalization)
// ==========================================================================

/**
 * @brief Scale rows of 2×n matrix by L1-norm (row sums).
 *
 * d[row] = sum(|M[row, :]|)
 * M[row, :] /= d[row]
 *
 * This matches the CPU bipartition `scale()` function:
 *   d = w.rowwise().sum(); w(i,j) /= d(i);
 *
 * Note: CPU adds tiny_num (~1e-15) to d to avoid division by zero.
 */
template<typename Scalar>
__global__ void scale_rows_l1_k2_kernel(
    Scalar* __restrict__ M,     // 2 × n (column-major)
    Scalar* __restrict__ d,     // 2
    int n)
{
    __shared__ Scalar s_sum[2][256];
    const int tid = threadIdx.x;

    // Compute row sums (L1 norm = sum of elements, NOT absolute values,
    // because NMF factors are non-negative)
    for (int row = 0; row < 2; ++row) {
        Scalar local_sum = 0;
        for (int j = tid; j < n; j += blockDim.x) {
            local_sum += M[row + j * 2];
        }
        s_sum[row][tid] = local_sum;
    }
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[0][tid] += s_sum[0][tid + s];
            s_sum[1][tid] += s_sum[1][tid + s];
        }
        __syncthreads();
    }

    // Compute scales (add tiny_num to match CPU)
    __shared__ Scalar scales[2];
    if (tid == 0) {
        for (int row = 0; row < 2; ++row) {
            Scalar row_sum = s_sum[row][0] + static_cast<Scalar>(1e-15);
            d[row] = row_sum;
            scales[row] = Scalar(1) / row_sum;
        }
    }
    __syncthreads();

    // Apply scaling
    for (int j = tid; j < n; j += blockDim.x) {
        M[0 + j * 2] *= scales[0];
        M[1 + j * 2] *= scales[1];
    }
}

template<typename Scalar>
void scale_rows_l1_k2_gpu(const GPUContext& ctx,
                           DenseMatrixGPU<Scalar>& M,
                           DeviceMemory<Scalar>& d) {
    scale_rows_l1_k2_kernel<Scalar><<<1, 256, 0, ctx.stream>>>(
        M.data.get(), d.get(), M.cols);
}

// ==========================================================================
// Extract subset CSC (samples → contiguous CSC on GPU)
// ==========================================================================

/**
 * @brief Extract columns indexed by `samples` from a host-side CSC matrix
 *        and upload the subset as a contiguous CSC to GPU.
 *
 * This avoids random access on GPU during the NMF inner loop.
 * For dclust with many bipartitions, the dominant cost is the NMF solve,
 * not the extraction (which is O(nnz_subset)).
 */
template<typename Scalar>
SparseMatrixGPU<Scalar> extract_and_upload_columns(
    const int* host_col_ptr,
    const int* host_row_idx,
    const Scalar* host_values,
    int m,                                      // Number of rows
    const std::vector<unsigned int>& samples)   // Column indices to extract
{
    int n_sub = static_cast<int>(samples.size());

    // Build subset CSC on host
    std::vector<int> sub_col_ptr(n_sub + 1);
    sub_col_ptr[0] = 0;

    // First pass: count nnz per column
    for (int s = 0; s < n_sub; ++s) {
        int col = samples[s];
        sub_col_ptr[s + 1] = sub_col_ptr[s] + (host_col_ptr[col + 1] - host_col_ptr[col]);
    }

    int nnz_sub = sub_col_ptr[n_sub];
    std::vector<int> sub_row_idx(nnz_sub);
    std::vector<Scalar> sub_values(nnz_sub);

    // Second pass: copy data
    for (int s = 0; s < n_sub; ++s) {
        int col = samples[s];
        int src_start = host_col_ptr[col];
        int dst_start = sub_col_ptr[s];
        int len = host_col_ptr[col + 1] - src_start;

        std::copy(host_row_idx + src_start, host_row_idx + src_start + len,
                  sub_row_idx.begin() + dst_start);
        std::copy(host_values + src_start, host_values + src_start + len,
                  sub_values.begin() + dst_start);
    }

    // Upload to GPU
    SparseMatrixGPU<Scalar> sub(m, n_sub, nnz_sub);
    sub.upload(sub_col_ptr.data(), sub_row_idx.data(), sub_values.data());

    return sub;
}

// ==========================================================================
// Build CSR (A^T as CSC) from extracted columns for W update
// ==========================================================================

/**
 * @brief Build the transpose of the subset matrix in CSC format.
 *
 * Given subset CSC (m × n_sub), builds the transpose (n_sub × m) in CSC.
 * This is equivalent to building the CSR of the original subset.
 */
template<typename Scalar>
SparseMatrixGPU<Scalar> build_subset_transpose(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    int m, int n_sub)
{
    int nnz = static_cast<int>(values.size());

    // Count nnz per row (= column of transpose)
    std::vector<int> t_col_ptr(m + 1, 0);
    for (int p = 0; p < nnz; ++p) {
        t_col_ptr[row_idx[p] + 1]++;
    }

    // Prefix sum
    for (int i = 0; i < m; ++i) {
        t_col_ptr[i + 1] += t_col_ptr[i];
    }

    std::vector<int> t_row_idx(nnz);
    std::vector<Scalar> t_values(nnz);
    std::vector<int> write_pos(m, 0);

    // Fill transpose
    for (int j = 0; j < n_sub; ++j) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int row = row_idx[p];
            int dst = t_col_ptr[row] + write_pos[row];
            t_row_idx[dst] = j;        // Column j becomes row j in transpose
            t_values[dst] = values[p];
            write_pos[row]++;
        }
    }

    // Upload to GPU
    SparseMatrixGPU<Scalar> At(n_sub, m, nnz);
    At.upload(t_col_ptr.data(), t_row_idx.data(), t_values.data());

    return At;
}

// ==========================================================================
// Correlation-based convergence (matches CPU bipartition)
// ==========================================================================

/**
 * @brief Compute 1 - R² between two factor matrices on host.
 *
 * CPU bipartition uses Pearson correlation distance of all W entries.
 */
template<typename Scalar>
double correlation_distance(const Scalar* w_curr, const Scalar* w_prev, int total) {
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0;
    double n = total;

    for (int i = 0; i < total; ++i) {
        double x = w_prev[i], y = w_curr[i];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_xx - sum_x * sum_x) *
                                   (n * sum_yy - sum_y * sum_y));
    if (denominator < 1e-15) return 0.0;

    return 1.0 - (numerator / denominator);
}

// ==========================================================================
// Centroid computation (host-side, called once per bipartition)
// ==========================================================================

/**
 * @brief Compute cluster centroid from sparse columns.
 *
 * Returns mean of columns indexed by `cluster_samples` from the
 * FULL host CSC matrix (not the subset).
 */
template<typename Scalar>
std::vector<double> compute_centroid_host(
    const int* host_col_ptr,
    const int* host_row_idx,
    const Scalar* host_values,
    int m,
    const std::vector<unsigned int>& cluster_samples)
{
    std::vector<double> center(m, 0.0);
    for (unsigned int s : cluster_samples) {
        for (int p = host_col_ptr[s]; p < host_col_ptr[s + 1]; ++p) {
            center[host_row_idx[p]] += static_cast<double>(host_values[p]);
        }
    }
    double inv_n = 1.0 / cluster_samples.size();
    for (auto& c : center) c *= inv_n;
    return center;
}

/**
 * @brief Relative cosine distance for bipartition quality.
 *
 * Measures how well-separated the two clusters are, using
 * cosine distance of samples to both centroids.
 */
template<typename Scalar>
double relative_cosine_distance(
    const int* host_col_ptr,
    const int* host_row_idx,
    const Scalar* host_values,
    int m,
    const std::vector<unsigned int>& samples1,
    const std::vector<unsigned int>& samples2,
    const std::vector<double>& center1,
    const std::vector<double>& center2)
{
    double c1_norm = std::sqrt(std::inner_product(
        center1.begin(), center1.end(), center1.begin(), 0.0));
    double c2_norm = std::sqrt(std::inner_product(
        center2.begin(), center2.end(), center2.begin(), 0.0));

    if (c1_norm < 1e-15 || c2_norm < 1e-15) return 0.0;

    double dist1 = 0, dist2 = 0;

    // Samples1: distance to assigned (center1) vs unassigned (center2)
    for (unsigned int s : samples1) {
        double dot_c1 = 0, dot_c2 = 0;
        for (int p = host_col_ptr[s]; p < host_col_ptr[s + 1]; ++p) {
            double v = static_cast<double>(host_values[p]);
            int r = host_row_idx[p];
            dot_c1 += center1[r] * v;
            dot_c2 += center2[r] * v;
        }
        dist1 += (std::sqrt(dot_c2) * c1_norm) / (std::sqrt(dot_c1) * c2_norm);
    }

    // Samples2: distance to assigned (center2) vs unassigned (center1)
    for (unsigned int s : samples2) {
        double dot_c1 = 0, dot_c2 = 0;
        for (int p = host_col_ptr[s]; p < host_col_ptr[s + 1]; ++p) {
            double v = static_cast<double>(host_values[p]);
            int r = host_row_idx[p];
            dot_c1 += center1[r] * v;
            dot_c2 += center2[r] * v;
        }
        dist2 += (std::sqrt(dot_c1) * c2_norm) / (std::sqrt(dot_c2) * c1_norm);
    }

    return (dist1 + dist2) / (2.0 * m);
}

// ==========================================================================
// Main GPU bipartition function
// ==========================================================================

/**
 * @brief GPU-accelerated bipartition of a sample set.
 *
 * Uses k=2 specialized GPU kernels for the NMF solve, with L1-norm
 * scaling matching the CPU implementation.
 *
 * @param ctx         GPU context (cuBLAS handle + stream)
 * @param host_col_ptr  Host CSC column pointers for full matrix A
 * @param host_row_idx  Host CSC row indices
 * @param host_values   Host CSC values
 * @param m             Number of rows (features)
 * @param n_total       Total number of columns in A
 * @param samples       Column indices to include in bipartition (0-based)
 * @param tol           Convergence tolerance (correlation distance)
 * @param maxit         Maximum ALS iterations
 * @param nonneg        Enforce non-negativity in NNLS
 * @param seed          Random seed for W initialization
 * @param calc_dist     Compute relative cosine distance
 * @param verbose       Print progress
 *
 * @return GPUBipartitionResult
 */
template<typename Scalar>
GPUBipartitionResult bipartition_gpu(
    GPUContext& ctx,
    const int* host_col_ptr,
    const int* host_row_idx,
    const Scalar* host_values,
    int m,
    int n_total,
    const std::vector<unsigned int>& samples,
    double tol = 1e-5,
    unsigned int maxit = 100,
    bool nonneg = true,
    unsigned int seed = 0,
    bool calc_dist = false,
    bool verbose = false)
{
    int n_sub = static_cast<int>(samples.size());

    // ---- 1. Extract subset CSC and upload to GPU ----
    SparseMatrixGPU<Scalar> A_sub = extract_and_upload_columns<Scalar>(
        host_col_ptr, host_row_idx, host_values, m, samples);

    // Build host-side CSC for transpose construction
    std::vector<int> sub_col_ptr(n_sub + 1);
    sub_col_ptr[0] = 0;
    for (int s = 0; s < n_sub; ++s) {
        int col = samples[s];
        sub_col_ptr[s + 1] = sub_col_ptr[s] +
            (host_col_ptr[col + 1] - host_col_ptr[col]);
    }
    int nnz_sub = sub_col_ptr[n_sub];
    std::vector<int> sub_row_idx(nnz_sub);
    std::vector<Scalar> sub_values(nnz_sub);
    for (int s = 0; s < n_sub; ++s) {
        int col = samples[s];
        int src = host_col_ptr[col];
        int dst = sub_col_ptr[s];
        int len = host_col_ptr[col + 1] - src;
        std::copy(host_row_idx + src, host_row_idx + src + len,
                  sub_row_idx.begin() + dst);
        std::copy(host_values + src, host_values + src + len,
                  sub_values.begin() + dst);
    }

    // Build transpose for W update
    SparseMatrixGPU<Scalar> At_sub = build_subset_transpose<Scalar>(
        sub_col_ptr, sub_row_idx, sub_values, m, n_sub);

    // ---- 2. Initialize W (2 × m) randomly ----
    // Use a simple LCG matching RcppML's RNG pattern
    std::vector<Scalar> W_host(2 * m);
    {
        unsigned int state = seed;
        auto next_rand = [&state]() -> double {
            state = state * 1664525u + 1013904223u;
            return static_cast<double>(state) / 4294967296.0;
        };
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < m; ++j) {
                W_host[i + j * 2] = static_cast<Scalar>(next_rand());
            }
        }
    }

    DenseMatrixGPU<Scalar> W(2, m, W_host.data());
    DenseMatrixGPU<Scalar> H(2, n_sub);
    DeviceMemory<Scalar> d(2);
    Scalar d_init[2] = {Scalar(1), Scalar(1)};
    d.upload(d_init, 2);

    // ---- 3. ALS iteration with L1-norm scaling ----
    std::vector<Scalar> W_prev(2 * m);

    for (unsigned int iter = 0; iter < maxit; ++iter) {
        W.download_to(W_prev.data());

        // --- H update ---
        // Compute Gram G = W·W' (2×2) via cuBLAS dot products
        Scalar G00, G01, G11;
        if constexpr (std::is_same_v<Scalar, double>) {
            CUBLAS_CHECK(cublasDdot(ctx.cublas, m, W.data.get(), 2,
                                    W.data.get(), 2, &G00));
            CUBLAS_CHECK(cublasDdot(ctx.cublas, m, W.data.get(), 2,
                                    W.data.get() + 1, 2, &G01));
            CUBLAS_CHECK(cublasDdot(ctx.cublas, m, W.data.get() + 1, 2,
                                    W.data.get() + 1, 2, &G11));
        } else {
            CUBLAS_CHECK(cublasSdot(ctx.cublas, m, W.data.get(), 2,
                                    W.data.get(), 2, &G00));
            CUBLAS_CHECK(cublasSdot(ctx.cublas, m, W.data.get(), 2,
                                    W.data.get() + 1, 2, &G01));
            CUBLAS_CHECK(cublasSdot(ctx.cublas, m, W.data.get() + 1, 2,
                                    W.data.get() + 1, 2, &G11));
        }
        ctx.sync();

        G00 += static_cast<Scalar>(1e-15);
        G11 += static_cast<Scalar>(1e-15);
        Scalar det = G00 * G11 - G01 * G01;
        Scalar inv_det = Scalar(1) / det;

        // Fused RHS + NNLS for H
        int threads = 256;
        int blocks = (n_sub + threads - 1) / threads;
        Scalar L1_zero = 0;
        fused_rhs_nnls_k2_h_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
            W.data.get(),
            A_sub.col_ptr.get(), A_sub.row_indices.get(), A_sub.values.get(),
            H.data.get(), n_sub,
            G00, G01, G11, inv_det,
            nonneg ? L1_zero : static_cast<Scalar>(-1));
        // Note: L1 < 0 signals to skip non-negativity constraint in nnls2_device.
        // But our nnls2_device always enforces non-negativity. For nonneg=false
        // we need the unconstrained path. For now, we always use nonneg=true
        // which is the bipartition default.

        // L1-norm scale H
        scale_rows_l1_k2_gpu(ctx, H, d);

        // --- W update ---
        // Compute Gram G = H·H' (2×2)
        if constexpr (std::is_same_v<Scalar, double>) {
            CUBLAS_CHECK(cublasDdot(ctx.cublas, n_sub, H.data.get(), 2,
                                    H.data.get(), 2, &G00));
            CUBLAS_CHECK(cublasDdot(ctx.cublas, n_sub, H.data.get(), 2,
                                    H.data.get() + 1, 2, &G01));
            CUBLAS_CHECK(cublasDdot(ctx.cublas, n_sub, H.data.get() + 1, 2,
                                    H.data.get() + 1, 2, &G11));
        } else {
            CUBLAS_CHECK(cublasSdot(ctx.cublas, n_sub, H.data.get(), 2,
                                    H.data.get(), 2, &G00));
            CUBLAS_CHECK(cublasSdot(ctx.cublas, n_sub, H.data.get(), 2,
                                    H.data.get() + 1, 2, &G01));
            CUBLAS_CHECK(cublasSdot(ctx.cublas, n_sub, H.data.get() + 1, 2,
                                    H.data.get() + 1, 2, &G11));
        }
        ctx.sync();

        G00 += static_cast<Scalar>(1e-15);
        G11 += static_cast<Scalar>(1e-15);
        det = G00 * G11 - G01 * G01;
        inv_det = Scalar(1) / det;

        // Fused RHS + NNLS for W via A^T
        blocks = (m + threads - 1) / threads;
        fused_rhs_nnls_k2_w_kernel<Scalar><<<blocks, threads, 0, ctx.stream>>>(
            H.data.get(),
            At_sub.col_ptr.get(), At_sub.row_indices.get(), At_sub.values.get(),
            W.data.get(), m,
            G00, G01, G11, inv_det, L1_zero);

        // L1-norm scale W
        scale_rows_l1_k2_gpu(ctx, W, d);

        // --- Convergence check ---
        std::vector<Scalar> W_curr(2 * m);
        W.download_to(W_curr.data());
        ctx.sync();

        double tol_val = correlation_distance(W_curr.data(), W_prev.data(), 2 * m);

        if (verbose) {
            FACTORNET_GPU_LOG(verbose, "  bipartition iter %3d / %d  tol = %.2e\n",
                   iter + 1, maxit, tol_val);
        }

        if (tol_val < tol && iter > 0) {
            break;
        }
    }

    // ---- 4. Post-processing: compute bipartition vector ----
    std::vector<Scalar> H_host(2 * n_sub);
    Scalar d_host[2];
    H.download_to(H_host.data());
    d.download(d_host, 2);
    ctx.sync();

    GPUBipartitionResult result;
    result.v.resize(n_sub);

    // Determine factor ordering by diagonal scaling
    bool swap = (d_host[1] > d_host[0]);

    unsigned int cnt1 = 0, cnt2 = 0;
    for (int j = 0; j < n_sub; ++j) {
        double h0 = static_cast<double>(H_host[0 + j * 2]);
        double h1 = static_cast<double>(H_host[1 + j * 2]);
        result.v[j] = swap ? (h1 - h0) : (h0 - h1);
        if (result.v[j] > 0) cnt1++;
        else cnt2++;
    }

    result.size1 = cnt1;
    result.size2 = cnt2;
    result.samples1.resize(cnt1);
    result.samples2.resize(cnt2);

    unsigned int s1 = 0, s2 = 0;
    for (int j = 0; j < n_sub; ++j) {
        if (result.v[j] > 0) {
            result.samples1[s1++] = samples[j];
        } else {
            result.samples2[s2++] = samples[j];
        }
    }

    // ---- 5. Optional: centroids and distance ----
    result.dist = -1.0;
    if (calc_dist && cnt1 > 0 && cnt2 > 0) {
        result.center1 = compute_centroid_host(
            host_col_ptr, host_row_idx, host_values, m, result.samples1);
        result.center2 = compute_centroid_host(
            host_col_ptr, host_row_idx, host_values, m, result.samples2);
        result.dist = relative_cosine_distance(
            host_col_ptr, host_row_idx, host_values, m,
            result.samples1, result.samples2,
            result.center1, result.center2);
    }

    return result;
}

} // namespace gpu
} // namespace FactorNet
