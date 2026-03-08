/**
 * @file dispersion.cuh
 * @brief GPU-native dispersion parameter updates for IRLS distributions.
 *
 * Replaces the host-mediated dispersion update path that previously required
 * downloading W, H, d from the GPU, computing dispersion on the CPU, and
 * re-uploading. All computation now stays on the GPU.
 *
 * Supported distributions:
 *   - GP (Generalized Poisson): theta update via MM quadratic iterations
 *   - NB (Negative Binomial): size r update via method of moments
 *   - Gamma/InvGauss/Tweedie: phi update via Pearson MoM estimator
 *
 * @author RcppML
 */

#pragma once

#include "types.cuh"
#include "gram.cuh"
#include <cstdint>

namespace FactorNet {
namespace gpu {

// =========================================================================
// Phase 1 kernels: Per-nonzero prediction + per-row accumulation
// =========================================================================

/**
 * @brief Phi update (Gamma/IG/Tweedie): accumulate Pearson residuals per row.
 *
 * For each nonzero (i,j): compute pred = W_Td[:,i] · H[:,j],
 * then accumulate (y - pred)² / mu^p into sum_pearson_sq[i] and count[i].
 *
 * 1 block per column, threads cooperate on nonzeros within that column.
 */
template<typename Scalar>
__global__ void phi_accum_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ W_Td,     // k × m, column-major
    const Scalar* __restrict__ H,        // k × n, column-major
    int k, int n,
    double var_power,
    double* __restrict__ sum_pearson_sq,  // m-length, atomicAdd
    int*    __restrict__ count)           // m-length, atomicAdd
{
    const int j = blockIdx.x;
    if (j >= n) return;

    const int tid = threadIdx.x;
    const int start = col_ptr[j];
    const int end   = col_ptr[j + 1];

    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        int i = row_idx[idx];
        double y = static_cast<double>(values[idx]);
        if (y <= 0.0) continue;

        double pred = 0.0;
        for (int f = 0; f < k; ++f)
            pred += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H[f + j * k]);
        pred = fmax(pred, 1e-10);

        double resid = y - pred;
        double v_mu = pow(pred, var_power);
        double pearson_sq = (resid * resid) / fmax(v_mu, 1e-20);

        atomicAdd(&sum_pearson_sq[i], pearson_sq);
        atomicAdd(&count[i], 1);
    }
}

/**
 * @brief Phi finalize: compute phi[i] = sum_pearson_sq[i] / count[i], clamped.
 */
template<typename Scalar>
__global__ void phi_finalize_kernel(
    Scalar* __restrict__ phi_vec,
    const double* __restrict__ sum_pearson_sq,
    const int*    __restrict__ count,
    int m,
    Scalar phi_min, Scalar phi_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    if (count[i] > 0) {
        double phi_new = sum_pearson_sq[i] / static_cast<double>(count[i]);
        phi_new = fmax(static_cast<double>(phi_min), fmin(phi_new, static_cast<double>(phi_max)));
        if (isfinite(phi_new))
            phi_vec[i] = static_cast<Scalar>(phi_new);
    }
}

/**
 * @brief NB accumulation: per-nonzero stats for method of moments.
 *
 * Accumulates sum_mu_nz[i], sum_mu_sq_nz[i], sum_resid_sq_nz[i] per row.
 */
template<typename Scalar>
__global__ void nb_accum_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ W_Td,     // k × m, column-major
    const Scalar* __restrict__ H,        // k × n, column-major
    int k, int n,
    double* __restrict__ sum_mu_nz,      // m-length
    double* __restrict__ sum_mu_sq_nz,   // m-length
    double* __restrict__ sum_resid_sq_nz) // m-length
{
    const int j = blockIdx.x;
    if (j >= n) return;

    const int tid = threadIdx.x;
    const int start = col_ptr[j];
    const int end   = col_ptr[j + 1];

    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        int i = row_idx[idx];
        double y = static_cast<double>(values[idx]);

        double mu = 0.0;
        for (int f = 0; f < k; ++f)
            mu += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H[f + j * k]);
        mu = fmax(mu, 1e-10);

        double resid = y - mu;
        atomicAdd(&sum_mu_nz[i], mu);
        atomicAdd(&sum_mu_sq_nz[i], mu * mu);
        atomicAdd(&sum_resid_sq_nz[i], resid * resid);
    }
}

/**
 * @brief NB finalize: compute r[i] = total_mu_sq / excess_var.
 *
 * Uses pre-computed total_mu and total_mu_sq (from Gram trick, computed
 * by a separate kernel) combined with the nonzero-only accumulators.
 *
 * @param total_mu     m-length: Σ_j μ_ij = w_i · (H · 1), from Gram trick
 * @param total_mu_sq  m-length: w_i^T G_H w_i, from Gram trick
 */
template<typename Scalar>
__global__ void nb_finalize_kernel(
    Scalar* __restrict__ nb_size_vec,
    const double* __restrict__ sum_mu_nz,
    const double* __restrict__ sum_mu_sq_nz,
    const double* __restrict__ sum_resid_sq_nz,
    const double* __restrict__ total_mu_sq,
    const double* __restrict__ total_mu,
    int m,
    Scalar r_min, Scalar r_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    double t_mu_sq = total_mu_sq[i];
    double t_mu = total_mu[i];

    // total_resid_sq = sum_resid_sq_nz + (total_mu_sq - sum_mu_sq_nz)
    // The zero entries contribute (0 - μ_ij)² = μ_ij², summed = total_mu_sq - sum_mu_sq_nz
    double total_resid_sq = sum_resid_sq_nz[i] + (t_mu_sq - sum_mu_sq_nz[i]);
    double excess_var = total_resid_sq - t_mu;

    double r_min_d = static_cast<double>(r_min);
    double r_max_d = static_cast<double>(r_max);

    if (excess_var > 1e-10 && t_mu_sq > 1e-10) {
        double r_new = t_mu_sq / excess_var;
        r_new = fmax(r_min_d, fmin(r_new, r_max_d));
        if (isfinite(r_new))
            nb_size_vec[i] = static_cast<Scalar>(r_new);
    } else {
        nb_size_vec[i] = r_max;
    }
}

/**
 * @brief Compute total_mu and total_mu_sq per row via Gram trick (for NB).
 *
 * total_mu[i]    = w_i · (H · 1)  = Σ_f W_Td(f,i) * H_rowsum(f)
 * total_mu_sq[i] = w_i' G_H w_i   = Σ_a Σ_b W_Td(a,i) * G_H(a,b) * W_Td(b,i)
 *
 * where G_H = H · H' (k × k, pre-computed).
 *
 * @param G_H      k × k Gram of H (device, column-major)
 * @param H_rowsum k-length column sums of H (device)
 */
template<typename Scalar>
__global__ void nb_gram_trick_kernel(
    const Scalar* __restrict__ W_Td,     // k × m, column-major
    const Scalar* __restrict__ G_H,      // k × k, column-major
    const Scalar* __restrict__ H_rowsum, // k-length
    int k, int m,
    double* __restrict__ total_mu,       // m-length output
    double* __restrict__ total_mu_sq)    // m-length output
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    // total_mu[i] = dot(W_Td[:,i], H_rowsum)
    double mu_total = 0.0;
    for (int f = 0; f < k; ++f)
        mu_total += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H_rowsum[f]);
    total_mu[i] = mu_total;

    // total_mu_sq[i] = W_Td[:,i]' * G_H * W_Td[:,i]
    double mu_sq_total = 0.0;
    for (int a = 0; a < k; ++a) {
        double w_a = static_cast<double>(W_Td[a + i * k]);
        for (int b = 0; b < k; ++b)
            mu_sq_total += w_a * static_cast<double>(G_H[a + b * k]) * static_cast<double>(W_Td[b + i * k]);
    }
    total_mu_sq[i] = mu_sq_total;
}

// =========================================================================
// GP theta update kernels
// =========================================================================

/**
 * @brief GP Phase 1: Compute predictions for all nonzeros + per-row sums.
 *
 * For each nonzero: pred = W_Td[:,i] · H[:,j].
 * Stores pred in d_pred_cache (nnz-length).
 * Accumulates sum_s_d[i] (total pred per row) and sum_y_d[i] (total observed).
 * Also counts n_nz[i] (number of nonzeros with y >= 1).
 */
template<typename Scalar>
__global__ void gp_phase1_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ W_Td,     // k × m, column-major
    const Scalar* __restrict__ H,        // k × n, column-major
    int k, int n,
    double* __restrict__ d_pred_cache,   // nnz-length: cached predictions
    double* __restrict__ sum_y_d,        // m-length: atomicAdd
    int*    __restrict__ n_nz)           // m-length: atomicAdd (count y >= 1)
{
    const int j = blockIdx.x;
    if (j >= n) return;

    const int tid = threadIdx.x;
    const int start = col_ptr[j];
    const int end   = col_ptr[j + 1];

    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        int i = row_idx[idx];
        double y = static_cast<double>(values[idx]);

        double pred = 0.0;
        for (int f = 0; f < k; ++f)
            pred += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H[f + j * k]);
        pred = fmax(pred, 1e-10);

        d_pred_cache[idx] = pred;
        atomicAdd(&sum_y_d[i], y);
        if (y >= 1.0) atomicAdd(&n_nz[i], 1);
    }
}

/**
 * @brief GP total mu per row via Gram trick: sum_s_d[i] = w_i · (H · 1).
 */
template<typename Scalar>
__global__ void gp_total_mu_kernel(
    const Scalar* __restrict__ W_Td,     // k × m, column-major
    const Scalar* __restrict__ H_rowsum, // k-length
    int k, int m,
    double* __restrict__ sum_s_d)        // m-length output
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    double s = 0.0;
    for (int f = 0; f < k; ++f)
        s += static_cast<double>(W_Td[f + i * k]) * static_cast<double>(H_rowsum[f]);
    sum_s_d[i] = s;
}

/**
 * @brief GP MM inner iteration: accumulate alpha and gamma from nonzeros.
 *
 * For each nonzero with y >= 1:
 *   theta = theta_vec[row]
 *   denom = max(s + theta * y, 1e-10)
 *   eta1 = s / denom
 *   alpha_d[row] += (y - 1) * eta1
 *   gamma_d[row] += (y - 1) * (1 - eta1)
 */
template<typename Scalar>
__global__ void gp_mm_accum_kernel(
    const int*    __restrict__ col_ptr,
    const int*    __restrict__ row_idx,
    const Scalar* __restrict__ values,
    const double* __restrict__ d_pred_cache,  // nnz-length cached predictions
    const Scalar* __restrict__ theta_vec,     // m-length current theta
    int n,
    double* __restrict__ alpha_d,             // m-length, atomicAdd (pre-zeroed)
    double* __restrict__ gamma_d)             // m-length, atomicAdd (pre-zeroed)
{
    const int j = blockIdx.x;
    if (j >= n) return;

    const int tid = threadIdx.x;
    const int start = col_ptr[j];
    const int end   = col_ptr[j + 1];

    for (int idx = start + tid; idx < end; idx += blockDim.x) {
        int i = row_idx[idx];
        double y = static_cast<double>(values[idx]);

        if (y >= 1.0) {
            double s = d_pred_cache[idx];
            double th = static_cast<double>(theta_vec[i]);
            double denom = fmax(s + th * y, 1e-10);
            double eta1 = s / denom;
            double ym1 = y - 1.0;
            atomicAdd(&alpha_d[i], ym1 * eta1);
            atomicAdd(&gamma_d[i], ym1 * (1.0 - eta1));
        }
    }
}

/**
 * @brief GP MM finalize: update theta per row using quadratic formula.
 *
 * alpha_i = alpha_d[i] + n_nz[i]
 * beta_i = (sum_y_d[i] - sum_s_d[i]) - gamma_d[i] + alpha_i
 * disc = beta^2 + 4 * alpha * gamma
 * theta_new = (-beta + sqrt(disc)) / (2 * alpha)
 */
template<typename Scalar>
__global__ void gp_mm_finalize_kernel(
    Scalar* __restrict__ theta_vec,
    const double* __restrict__ alpha_d,
    const double* __restrict__ gamma_d,
    const double* __restrict__ sum_y_d,
    const double* __restrict__ sum_s_d,
    const int*    __restrict__ n_nz,
    int m,
    Scalar theta_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    double alpha_i = alpha_d[i] + static_cast<double>(n_nz[i]);
    double gamma_i = gamma_d[i];
    double beta_i = (sum_y_d[i] - sum_s_d[i]) - gamma_i + alpha_i;

    if (alpha_i > 1e-15) {
        double disc = beta_i * beta_i + 4.0 * alpha_i * gamma_i;
        if (disc > 0.0 && isfinite(disc)) {
            double new_theta = (-beta_i + sqrt(disc)) / (2.0 * alpha_i);
            double cap = static_cast<double>(theta_max);
            if (isfinite(new_theta) && new_theta >= 0.0)
                theta_vec[i] = static_cast<Scalar>(fmin(new_theta, cap));
        }
    }
}

// =========================================================================
// Host dispatch functions
// =========================================================================

/**
 * @brief GPU-native phi (Gamma/IG/Tweedie) dispersion update.
 *
 * Entirely on GPU — no host downloads of W, H, or d.
 */
template<typename Scalar>
void phi_update_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W_Td,   // k × m (d-scaled)
    const DenseMatrixGPU<Scalar>& H,      // k × n
    DeviceMemory<Scalar>& d_phi_vec,      // m-length, updated in-place
    double var_power,
    Scalar phi_min, Scalar phi_max,
    int disp_mode = 0)  // 0 = per_row, 1 = global
{
    const int k = W_Td.rows;
    const int m = W_Td.cols;
    const int n = H.cols;

    // Temporary per-row accumulators
    DeviceMemory<double> d_sum_pearson_sq(m);
    DeviceMemory<int> d_count(m);
    CUDA_CHECK(cudaMemsetAsync(d_sum_pearson_sq.get(), 0, m * sizeof(double), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(d_count.get(), 0, m * sizeof(int), ctx.stream));

    // Phase 1: accumulate Pearson residuals
    int threads = 256;
    phi_accum_kernel<Scalar><<<n, threads, 0, ctx.stream>>>(
        A.col_ptr.get(), A.row_indices.get(), A.values.get(),
        W_Td.data.get(), H.data.get(), k, n,
        var_power, d_sum_pearson_sq.get(), d_count.get());

    // Phase 2: finalize phi per row
    int blocks = (m + 255) / 256;
    phi_finalize_kernel<Scalar><<<blocks, 256, 0, ctx.stream>>>(
        d_phi_vec.get(), d_sum_pearson_sq.get(), d_count.get(),
        m, phi_min, phi_max);

    // Global mode: compute median on host (rare path, acceptable)
    if (disp_mode == 1) {
        ctx.sync();
        std::vector<Scalar> h_phi(m);
        d_phi_vec.download(h_phi.data(), m);
        std::nth_element(h_phi.begin(), h_phi.begin() + m / 2, h_phi.end());
        Scalar median = h_phi[m / 2];
        std::fill(h_phi.begin(), h_phi.end(), median);
        d_phi_vec.upload(h_phi.data(), m);
    }
}

/**
 * @brief GPU-native NB size (r) update via method of moments.
 */
template<typename Scalar>
void nb_size_update_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W_Td,   // k × m (d-scaled)
    const DenseMatrixGPU<Scalar>& H,      // k × n
    const DenseMatrixGPU<Scalar>& G_H,    // k × k Gram of H (pre-computed)
    DeviceMemory<Scalar>& d_nb_size_vec,  // m-length, updated in-place
    Scalar r_min, Scalar r_max,
    int disp_mode = 0)
{
    const int k = W_Td.rows;
    const int m = W_Td.cols;
    const int n = H.cols;

    // Per-row accumulators from nonzeros
    DeviceMemory<double> d_sum_mu_nz(m);
    DeviceMemory<double> d_sum_mu_sq_nz(m);
    DeviceMemory<double> d_sum_resid_sq_nz(m);
    CUDA_CHECK(cudaMemsetAsync(d_sum_mu_nz.get(), 0, m * sizeof(double), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(d_sum_mu_sq_nz.get(), 0, m * sizeof(double), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(d_sum_resid_sq_nz.get(), 0, m * sizeof(double), ctx.stream));

    // Phase 1: accumulate nonzero statistics
    int threads = 256;
    nb_accum_kernel<Scalar><<<n, threads, 0, ctx.stream>>>(
        A.col_ptr.get(), A.row_indices.get(), A.values.get(),
        W_Td.data.get(), H.data.get(), k, n,
        d_sum_mu_nz.get(), d_sum_mu_sq_nz.get(), d_sum_resid_sq_nz.get());

    // Phase 2: Gram trick for total_mu and total_mu_sq per row
    // Need H_rowsum = H · 1 (k-length)
    DeviceMemory<Scalar> d_H_rowsum(k);
    DeviceMemory<double> d_total_mu(m);
    DeviceMemory<double> d_total_mu_sq(m);

    // Compute H_rowsum on device: sum each row of H (k × n matrix)
    // Use a simple kernel or cuBLAS gemv with ones vector
    // For simplicity, use a ones vector + cuBLAS gemv
    DeviceMemory<Scalar> d_ones(n);
    {
        std::vector<Scalar> h_ones(n, static_cast<Scalar>(1));
        d_ones.upload(h_ones.data(), n);
    }

    // H_rowsum = H · 1_n  (k × n) · (n × 1) = (k × 1)
    Scalar alpha_one = static_cast<Scalar>(1);
    Scalar beta_zero = static_cast<Scalar>(0);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N,
            k, n, &alpha_one,
            H.data.get(), k,
            d_ones.get(), 1,
            &beta_zero,
            d_H_rowsum.get(), 1));
    } else {
        CUBLAS_CHECK(cublasSgemv(ctx.cublas, CUBLAS_OP_N,
            k, n, &alpha_one,
            H.data.get(), k,
            d_ones.get(), 1,
            &beta_zero,
            d_H_rowsum.get(), 1));
    }

    // Gram trick per row
    int blocks = (m + 255) / 256;
    nb_gram_trick_kernel<Scalar><<<blocks, 256, 0, ctx.stream>>>(
        W_Td.data.get(), G_H.data.get(), d_H_rowsum.get(), k, m,
        d_total_mu.get(), d_total_mu_sq.get());

    // Phase 3: finalize r per row
    nb_finalize_kernel<Scalar><<<blocks, 256, 0, ctx.stream>>>(
        d_nb_size_vec.get(),
        d_sum_mu_nz.get(), d_sum_mu_sq_nz.get(), d_sum_resid_sq_nz.get(),
        d_total_mu_sq.get(), d_total_mu.get(),
        m, r_min, r_max);

    // Global mode
    if (disp_mode == 1) {
        ctx.sync();
        std::vector<Scalar> h_r(m);
        d_nb_size_vec.download(h_r.data(), m);
        std::nth_element(h_r.begin(), h_r.begin() + m / 2, h_r.end());
        Scalar median = h_r[m / 2];
        std::fill(h_r.begin(), h_r.end(), median);
        d_nb_size_vec.upload(h_r.data(), m);
    }
}

/**
 * @brief GPU-native GP theta update via MM quadratic iterations.
 */
template<typename Scalar>
void gp_theta_update_gpu(
    const GPUContext& ctx,
    const SparseMatrixGPU<Scalar>& A,
    const DenseMatrixGPU<Scalar>& W_Td,   // k × m (d-scaled)
    const DenseMatrixGPU<Scalar>& H,      // k × n
    DeviceMemory<Scalar>& d_theta_vec,    // m-length, updated in-place
    Scalar theta_max,
    int disp_mode = 0)
{
    const int k = W_Td.rows;
    const int m = W_Td.cols;
    const int n = H.cols;
    const int nnz = A.nnz;
    constexpr int THETA_INNER_ITERS = 5;

    // Allocate GPU buffers
    DeviceMemory<double> d_pred_cache(nnz);  // cached predictions
    DeviceMemory<double> d_sum_y_d(m);
    DeviceMemory<double> d_sum_s_d(m);
    DeviceMemory<int>    d_n_nz(m);
    DeviceMemory<double> d_alpha_d(m);
    DeviceMemory<double> d_gamma_d(m);

    CUDA_CHECK(cudaMemsetAsync(d_sum_y_d.get(), 0, m * sizeof(double), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(d_n_nz.get(), 0, m * sizeof(int), ctx.stream));

    // Phase 1: compute predictions + per-row sums (once)
    int threads = 256;
    gp_phase1_kernel<Scalar><<<n, threads, 0, ctx.stream>>>(
        A.col_ptr.get(), A.row_indices.get(), A.values.get(),
        W_Td.data.get(), H.data.get(), k, n,
        d_pred_cache.get(), d_sum_y_d.get(), d_n_nz.get());

    // Compute total_mu per row via Gram trick: sum_s_d[i] = w_i · (H · 1)
    DeviceMemory<Scalar> d_H_rowsum(k);
    DeviceMemory<Scalar> d_ones(n);
    {
        std::vector<Scalar> h_ones(n, static_cast<Scalar>(1));
        d_ones.upload(h_ones.data(), n);
    }
    Scalar alpha_one = static_cast<Scalar>(1);
    Scalar beta_zero = static_cast<Scalar>(0);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N,
            k, n, &alpha_one,
            H.data.get(), k, d_ones.get(), 1,
            &beta_zero, d_H_rowsum.get(), 1));
    } else {
        CUBLAS_CHECK(cublasSgemv(ctx.cublas, CUBLAS_OP_N,
            k, n, &alpha_one,
            H.data.get(), k, d_ones.get(), 1,
            &beta_zero, d_H_rowsum.get(), 1));
    }

    int blocks = (m + 255) / 256;
    gp_total_mu_kernel<Scalar><<<blocks, 256, 0, ctx.stream>>>(
        W_Td.data.get(), d_H_rowsum.get(), k, m, d_sum_s_d.get());

    // Phase 2: MM iterations (5x)
    for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS; ++mm_iter) {
        CUDA_CHECK(cudaMemsetAsync(d_alpha_d.get(), 0, m * sizeof(double), ctx.stream));
        CUDA_CHECK(cudaMemsetAsync(d_gamma_d.get(), 0, m * sizeof(double), ctx.stream));

        gp_mm_accum_kernel<Scalar><<<n, threads, 0, ctx.stream>>>(
            A.col_ptr.get(), A.row_indices.get(), A.values.get(),
            d_pred_cache.get(), d_theta_vec.get(), n,
            d_alpha_d.get(), d_gamma_d.get());

        gp_mm_finalize_kernel<Scalar><<<blocks, 256, 0, ctx.stream>>>(
            d_theta_vec.get(),
            d_alpha_d.get(), d_gamma_d.get(),
            d_sum_y_d.get(), d_sum_s_d.get(), d_n_nz.get(),
            m, theta_max);
    }

    // Global mode
    if (disp_mode == 1) {
        ctx.sync();
        std::vector<Scalar> h_theta(m);
        d_theta_vec.download(h_theta.data(), m);
        Scalar sum = 0;
        for (int i = 0; i < m; ++i) sum += h_theta[i];
        Scalar mean_theta = sum / static_cast<Scalar>(m);
        std::fill(h_theta.begin(), h_theta.end(), mean_theta);
        d_theta_vec.upload(h_theta.data(), m);
    }
}

} // namespace gpu
} // namespace FactorNet
