// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file randomized_gpu.cuh
 * @brief GPU-accelerated Randomized SVD (Halko-Martinsson-Tropp, 2011)
 *
 * Key GPU optimizations:
 *   - cuSPARSE SpMM for batched forward/transpose products (optimal for k=50)
 *   - cuSOLVER for QR factorization (Householder on small l×l)
 *   - cuBLAS GEMM for dense products (Q'·A steps)
 *   - All intermediate matrices device-resident
 *   - Single sparse matrix upload, reused across all power iterations
 *
 * Algorithm:
 *   1. Draw random Omega (n × l) where l = k + oversampling
 *   2. Y = A · Omega                   (forward SpMM, O(nnz·l))
 *   3. For q power iterations:
 *        QR(Y);  Z = A' · Y;  QR(Z);  Y = A · Z   (4 SpMMs per iter)
 *   4. Q = QR(Y)                       (thin QR, O(m·l²))
 *   5. B = Q' · A (via B' = A' · Q)   (transpose SpMM, O(nnz·l))
 *   6. SVD(B), extract top k           (l × n dense SVD on host)
 *
 * Total SpMMs: 4q + 2, each O(nnz · l).
 * For q=2, k=50: only 10 SpMMs total — very GPU-friendly.
 *
 * PCA centering: implicit, via correction terms in SpMM kernels.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/gpu/types.cuh>
#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/rng/rng.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <chrono>
#include <cmath>

// cuSOLVER for QR
#include <cusolverDn.h>

namespace FactorNet {
namespace svd {
namespace detail {

// ============================================================================
// cuBLAS typed wrappers
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_cublas_gemm
#define FACTORNET_SVD_GPU_DEF_cublas_gemm
template<typename Scalar>
inline void cublas_gemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    Scalar alpha, const Scalar* A, int lda,
    const Scalar* B, int ldb,
    Scalar beta, Scalar* C, int ldc);

template<>
inline void cublas_gemm<double>(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    double alpha, const double* A, int lda,
    const double* B, int ldb,
    double beta, double* C, int ldc)
{
    CUBLAS_CHECK(cublasDgemm(handle, transa, transb, m, n, k,
        &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template<>
inline void cublas_gemm<float>(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc)
{
    CUBLAS_CHECK(cublasSgemm(handle, transa, transb, m, n, k,
        &alpha, A, lda, B, ldb, &beta, C, ldc));
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_gemm

// ============================================================================
// PCA centering kernels for SpMM (multi-column)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_pca_correct_forward_kernel
#define FACTORNET_SVD_GPU_DEF_pca_correct_forward_kernel
/**
 * @brief PCA correction for forward SpMM: Y[:, j] -= mu * colsum_X[j]
 *
 * For Y = A_c · X = A·X - mu · (1'X) where colsum_X[j] = sum(X[:, j])
 * Y(i, j) -= mu(i) * colsum_X(j)
 */
template<typename Scalar>
__global__ void pca_correct_forward_kernel(
    Scalar* __restrict__ Y,       // m × l, column-major
    const Scalar* __restrict__ mu, // m
    const Scalar* __restrict__ colsums, // l
    int m, int l)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * l) return;
    int i = idx % m;
    int j = idx / m;
    Y[idx] -= mu[i] * colsums[j];
}
#endif // FACTORNET_SVD_GPU_DEF_pca_correct_forward_kernel

#ifndef FACTORNET_SVD_GPU_DEF_pca_correct_transpose_kernel
#define FACTORNET_SVD_GPU_DEF_pca_correct_transpose_kernel
/**
 * @brief PCA correction for transpose SpMM: Z[:, j] -= (mu'·Y[:, j]) · 1
 *
 * For Z = A_c' · Y = A'·Y - 1 · (mu' · Y)
 * Z(i, j) -= dot(mu, Y[:, j])
 */
template<typename Scalar>
__global__ void pca_correct_transpose_kernel(
    Scalar* __restrict__ Z,         // n × l, column-major
    const Scalar* __restrict__ dots, // l — dot(mu, Y[:, j]) for each j
    int n, int l)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * l) return;
    int j = idx / n;
    Z[idx] -= dots[j];
}
#endif // FACTORNET_SVD_GPU_DEF_pca_correct_transpose_kernel

#ifndef FACTORNET_SVD_GPU_DEF_compute_colsums_kernel
#define FACTORNET_SVD_GPU_DEF_compute_colsums_kernel
/**
 * @brief Compute column sums of X (n × l) for PCA correction.
 *
 * colsums[j] = sum_i X[i + j*n]  for j = 0..l-1
 *
 * We use a simple per-column parallel reduction.
 */
template<typename Scalar>
__global__ void compute_colsums_kernel(
    const Scalar* __restrict__ X, // n × l
    Scalar* __restrict__ colsums, // l
    int n, int l)
{
    int j = blockIdx.x;
    if (j >= l) return;

    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);

    int tid = threadIdx.x;
    Scalar val = Scalar(0);
    for (int i = tid; i < n; i += blockDim.x) {
        val += X[i + static_cast<size_t>(j) * n];
    }
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) colsums[j] = sdata[0];
}
#endif // FACTORNET_SVD_GPU_DEF_compute_colsums_kernel

#ifndef FACTORNET_SVD_GPU_DEF_compute_mu_dots_kernel
#define FACTORNET_SVD_GPU_DEF_compute_mu_dots_kernel
/**
 * @brief Compute dots[j] = mu' · Y[:, j] for PCA transpose correction.
 */
template<typename Scalar>
__global__ void compute_mu_dots_kernel(
    const Scalar* __restrict__ mu,  // m
    const Scalar* __restrict__ Y,   // m × l
    Scalar* __restrict__ dots,      // l
    int m, int l)
{
    int j = blockIdx.x;
    if (j >= l) return;

    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);

    int tid = threadIdx.x;
    Scalar val = Scalar(0);
    for (int i = tid; i < m; i += blockDim.x) {
        val += mu[i] * Y[i + static_cast<size_t>(j) * m];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) dots[j] = sdata[0];
}
#endif // FACTORNET_SVD_GPU_DEF_compute_mu_dots_kernel

// ============================================================================
// cuSOLVER QR wrapper — fully on-device thin QR
// ============================================================================

// Typed cuSOLVER geqrf wrappers
template<typename Scalar>
inline int cusolver_geqrf_bufferSize(cusolverDnHandle_t h, int m, int n, Scalar* A, int lda);
template<>
inline int cusolver_geqrf_bufferSize<double>(cusolverDnHandle_t h, int m, int n, double* A, int lda) {
    int size; CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(h, m, n, A, lda, &size)); return size;
}
template<>
inline int cusolver_geqrf_bufferSize<float>(cusolverDnHandle_t h, int m, int n, float* A, int lda) {
    int size; CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(h, m, n, A, lda, &size)); return size;
}

template<typename Scalar>
inline void cusolver_geqrf(cusolverDnHandle_t h, int m, int n, Scalar* A, int lda,
                           Scalar* tau, Scalar* work, int lwork, int* devInfo);
template<>
inline void cusolver_geqrf<double>(cusolverDnHandle_t h, int m, int n, double* A, int lda,
                                   double* tau, double* work, int lwork, int* devInfo) {
    CUSOLVER_CHECK(cusolverDnDgeqrf(h, m, n, A, lda, tau, work, lwork, devInfo));
}
template<>
inline void cusolver_geqrf<float>(cusolverDnHandle_t h, int m, int n, float* A, int lda,
                                  float* tau, float* work, int lwork, int* devInfo) {
    CUSOLVER_CHECK(cusolverDnSgeqrf(h, m, n, A, lda, tau, work, lwork, devInfo));
}

// Typed cuSOLVER orgqr wrappers (generate explicit Q from Householder reflectors)
template<typename Scalar>
inline int cusolver_orgqr_bufferSize(cusolverDnHandle_t h, int m, int n, int k,
                                     const Scalar* A, int lda, const Scalar* tau);
template<>
inline int cusolver_orgqr_bufferSize<double>(cusolverDnHandle_t h, int m, int n, int k,
                                             const double* A, int lda, const double* tau) {
    int size; CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(h, m, n, k, A, lda, tau, &size)); return size;
}
template<>
inline int cusolver_orgqr_bufferSize<float>(cusolverDnHandle_t h, int m, int n, int k,
                                            const float* A, int lda, const float* tau) {
    int size; CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(h, m, n, k, A, lda, tau, &size)); return size;
}

template<typename Scalar>
inline void cusolver_orgqr(cusolverDnHandle_t h, int m, int n, int k,
                           Scalar* A, int lda, const Scalar* tau,
                           Scalar* work, int lwork, int* devInfo);
template<>
inline void cusolver_orgqr<double>(cusolverDnHandle_t h, int m, int n, int k,
                                   double* A, int lda, const double* tau,
                                   double* work, int lwork, int* devInfo) {
    CUSOLVER_CHECK(cusolverDnDorgqr(h, m, n, k, A, lda, tau, work, lwork, devInfo));
}
template<>
inline void cusolver_orgqr<float>(cusolverDnHandle_t h, int m, int n, int k,
                                  float* A, int lda, const float* tau,
                                  float* work, int lwork, int* devInfo) {
    CUSOLVER_CHECK(cusolverDnSorgqr(h, m, n, k, A, lda, tau, work, lwork, devInfo));
}

// ============================================================================
// cuSOLVER gesvdj wrappers — device-side Jacobi SVD (optimal for tall-skinny)
// ============================================================================

// Typed cuSOLVER gesvdj buffer size wrappers
template<typename Scalar>
inline int cusolver_gesvdj_bufferSize(
    cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n,
    const Scalar* A, int lda, const Scalar* S,
    const Scalar* U, int ldu, const Scalar* V, int ldv,
    gesvdjInfo_t params);

template<>
inline int cusolver_gesvdj_bufferSize<double>(
    cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n,
    const double* A, int lda, const double* S,
    const double* U, int ldu, const double* V, int ldv,
    gesvdjInfo_t params)
{
    int lwork;
    CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
        h, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));
    return lwork;
}

template<>
inline int cusolver_gesvdj_bufferSize<float>(
    cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n,
    const float* A, int lda, const float* S,
    const float* U, int ldu, const float* V, int ldv,
    gesvdjInfo_t params)
{
    int lwork;
    CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(
        h, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));
    return lwork;
}

// Typed cuSOLVER gesvdj compute wrappers
template<typename Scalar>
inline void cusolver_gesvdj(
    cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n,
    Scalar* A, int lda, Scalar* S,
    Scalar* U, int ldu, Scalar* V, int ldv,
    Scalar* work, int lwork, int* devInfo,
    gesvdjInfo_t params);

template<>
inline void cusolver_gesvdj<double>(
    cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n,
    double* A, int lda, double* S,
    double* U, int ldu, double* V, int ldv,
    double* work, int lwork, int* devInfo,
    gesvdjInfo_t params)
{
    CUSOLVER_CHECK(cusolverDnDgesvdj(
        h, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
        work, lwork, devInfo, params));
}

template<>
inline void cusolver_gesvdj<float>(
    cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n,
    float* A, int lda, float* S,
    float* U, int ldu, float* V, int ldv,
    float* work, int lwork, int* devInfo,
    gesvdjInfo_t params)
{
    CUSOLVER_CHECK(cusolverDnSgesvdj(
        h, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
        work, lwork, devInfo, params));
}

/// In-place thin QR on device: overwrite d_A (rows × cols) with orthonormal Q.
/// Pre-allocated workspace buffers must be provided by the caller.
template<typename Scalar>
void device_thin_qr(
    cusolverDnHandle_t cusolver,
    Scalar* d_A,         // device: rows × cols, column-major → overwritten with Q
    int rows, int cols,
    Scalar* d_tau,       // device workspace: min(rows, cols)
    Scalar* d_work,      // device workspace: >= max(geqrf, orgqr) buffer size
    int lwork,
    int* d_info)
{
    // Step 1: Householder QR factorization in-place
    cusolver_geqrf<Scalar>(cusolver, rows, cols, d_A, rows, d_tau, d_work, lwork, d_info);

    // Step 2: Generate explicit thin Q from Householder reflectors
    cusolver_orgqr<Scalar>(cusolver, rows, cols, cols, d_A, rows, d_tau, d_work, lwork, d_info);
}

}  // namespace detail


// ============================================================================
// Main GPU Randomized SVD
// ============================================================================

/**
 * @brief GPU randomized truncated SVD/PCA (Halko-Martinsson-Tropp)
 *
 * @tparam Scalar     float or double
 * @param h_col_ptr   Host CSC column pointers (length n+1)
 * @param h_row_idx   Host CSC row indices (length nnz)
 * @param h_values    Host CSC values (length nnz)
 * @param m           Number of rows
 * @param n           Number of columns
 * @param nnz         Number of non-zeros
 * @param config      SVD configuration (k_max, tol, max_iter for power iters)
 * @return SVDResult with factors
 */
template<typename Scalar>
SVDResult<Scalar> randomized_svd_gpu(
    const int* h_col_ptr,
    const int* h_row_idx,
    const Scalar* h_values,
    int m, int n, int nnz,
    const SVDConfig<Scalar>& config)
{
    using namespace FactorNet::gpu;
    using DenseMatrixS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVectorS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    auto t_start = std::chrono::high_resolution_clock::now();

    const int k = config.k_max;
    // Oversampling: max(10, k/5) — Halko et al. recommend p=5..10 for small k,
    // but for k=50 we need p≥10 to capture the subspace boundary well.
    // k/5 scales gracefully: k=10→10, k=50→10, k=100→20.
    const int p = std::min(std::max(10, k / 5), std::min(m, n) - k);
    const int l = k + std::max(p, 0);   // sketch dimension
    // Power iterations: max_iter directly controls this.
    // Default 3 (not 2): q=3 significantly improves tail SV accuracy
    // at modest cost (2 extra SpMMs). For k=50 on scRNA-seq, q=2 gives
    // ~5% error on d[k] while q=3 reduces it to <1%.
    const int q = (config.max_iter <= 0) ? 3 : std::min(config.max_iter, 10);

    SVDResult<Scalar> result;
    result.centered = config.center;

    const int BLOCK_SIZE = 256;

    // ------------------------------------------------------------------
    // 1. GPU context and sparse matrix upload (dual-CSR for all-gather SpMM)
    // ------------------------------------------------------------------

    GPUContext ctx;

    SparseMatrixGPU<Scalar> d_A(m, n, nnz, h_col_ptr, h_row_idx, h_values);

    // Build dual-CSR: CSR(A) for forward SpMM, CSR(A^T) for transpose SpMM
    // Both use NON_TRANSPOSE (gather) — eliminates slow TRANSPOSE scatter path
    DualCSR<Scalar> dual_csr;
    dual_csr.init(ctx, d_A, m, n, nnz);

    // ------------------------------------------------------------------
    // 2. Row means for PCA centering
    // ------------------------------------------------------------------

    DenseVectorS row_means;
    DeviceMemory<Scalar> d_mu;
    DeviceMemory<Scalar> d_colsums(l);   // workspace for PCA forward correction
    DeviceMemory<Scalar> d_mu_dots(l);   // workspace for PCA transpose correction

    if (config.center) {
        Eigen::Map<const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>>
            A_map(m, n, nnz, h_col_ptr, h_row_idx, h_values);

        row_means.resize(m);
        row_means.setZero();
        for (int j = 0; j < n; ++j)
            for (typename decltype(A_map)::InnerIterator it(A_map, j); it; ++it)
                row_means(it.row()) += static_cast<Scalar>(it.value());
        if (n > 0) row_means /= static_cast<Scalar>(n);

        result.row_means = row_means;

        d_mu = DeviceMemory<Scalar>(m);
        d_mu.upload(row_means.data(), m);
    }

    // ------------------------------------------------------------------
    // 3. Compute ||A||^2_F
    // ------------------------------------------------------------------

    Scalar A_norm_sq = 0;
    for (int i = 0; i < nnz; ++i) A_norm_sq += h_values[i] * h_values[i];
    if (config.center) A_norm_sq -= static_cast<Scalar>(n) * row_means.squaredNorm();
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // ------------------------------------------------------------------
    // 4. Allocate dense matrices on GPU
    // ------------------------------------------------------------------

    // Y = A · X  output: m × l
    // Z = A' · Y output: n × l
    DeviceMemory<Scalar> d_Y(static_cast<size_t>(m) * l);
    DeviceMemory<Scalar> d_Z(static_cast<size_t>(n) * l);

    // Create cuSPARSE dense matrix descriptors for SpMM
    cusparseDnMatDescr_t dnDescr_Y, dnDescr_Z;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_Y, m, l, m /* lda */, d_Y.get(),
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_Z, n, l, n /* lda */, d_Z.get(),
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

    // ------------------------------------------------------------------
    // 5. Pre-allocate SpMM buffers (dual-CSR: both use NON_TRANSPOSE gather)
    //    a) Forward:   Y (m×l) = A · X (n×l)   via NON_TRANSPOSE on CSR(A)
    //    b) Transpose: Z (n×l) = A'· Y (m×l)   via NON_TRANSPOSE on CSR(A^T)
    // ------------------------------------------------------------------

    Scalar spmm_alpha = Scalar(1), spmm_beta = Scalar(0);

    cusparseDnMatDescr_t dnDescr_input_n;  // n × l (input for forward multiply)
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_input_n, n, l, n, d_Z.get(),  // temporary pointer
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t dnDescr_input_m;  // m × l (input for transpose multiply)
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_input_m, m, l, m, d_Y.get(),  // temporary pointer
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

    size_t bufSize_fwd = 0, bufSize_trans = 0;

    // Forward: NON_TRANSPOSE on CSR(A) [m rows × n cols] * X [n × l] = Y [m × l]
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmm_alpha, dual_csr.descr_A, dnDescr_input_n,
        &spmm_beta, dnDescr_Y,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufSize_fwd));

    // Transpose: NON_TRANSPOSE on CSR(A^T) [n rows × m cols] * Y [m × l] = Z [n × l]
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmm_alpha, dual_csr.descr_AT, dnDescr_input_m,
        &spmm_beta, dnDescr_Z,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufSize_trans));

    size_t bufSize = std::max(bufSize_fwd, bufSize_trans);
    DeviceMemory<char> d_spmm_buffer(bufSize > 0 ? bufSize : 1);

    // ------------------------------------------------------------------
    // Helper lambdas for SpMM with PCA correction
    // ------------------------------------------------------------------

    // Forward SpMM: Y = A · X via NON_TRANSPOSE on CSR(A) [m×n] * X [n×l] = Y [m×l]
    auto spmm_forward = [&](Scalar* d_src_n) {
        // Update dense descriptor pointer for input
        CUSPARSE_CHECK(cusparseDestroyDnMat(dnDescr_input_n));
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &dnDescr_input_n, n, l, n, d_src_n,
            CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

        Scalar beta_zero = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMM(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmm_alpha, dual_csr.descr_A, dnDescr_input_n,
            &beta_zero, dnDescr_Y,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_buffer.get()));

        // PCA correction: Y -= mu * colsums(X)
        if (config.center) {
            int smem = BLOCK_SIZE * sizeof(Scalar);
            detail::compute_colsums_kernel<<<l, BLOCK_SIZE, smem, ctx.stream>>>(
                d_src_n, d_colsums.get(), n, l);

            int total = m * l;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::pca_correct_forward_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_Y.get(), d_mu.get(), d_colsums.get(), m, l);
        }
    };

    // Transpose SpMM: Z = A^T · Y via NON_TRANSPOSE on CSR(A^T) [n×m] * Y [m×l] = Z [n×l]
    auto spmm_transpose = [&](Scalar* d_src_m) {
        CUSPARSE_CHECK(cusparseDestroyDnMat(dnDescr_input_m));
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &dnDescr_input_m, m, l, m, d_src_m,
            CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

        Scalar beta_zero = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMM(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmm_alpha, dual_csr.descr_AT, dnDescr_input_m,
            &beta_zero, dnDescr_Z,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_buffer.get()));

        // PCA correction: Z -= 1 * (mu' · Y[:, j]) for each j
        if (config.center) {
            int smem = BLOCK_SIZE * sizeof(Scalar);
            detail::compute_mu_dots_kernel<<<l, BLOCK_SIZE, smem, ctx.stream>>>(
                d_mu.get(), d_src_m, d_mu_dots.get(), m, l);

            int total = n * l;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::pca_correct_transpose_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_Z.get(), d_mu_dots.get(), n, l);
        }
    };

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU Randomized SVD: k=%d, l=%d, power_iter=%d\n", k, l, q);
    }

    // ========================================================================
    // Step 1: Random sketch Omega (n × l) → upload to d_Z
    // ========================================================================
    {
        DenseMatrixS Omega(n, l);
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        rng.fill_uniform(Omega.data(), Omega.rows(), Omega.cols());
        Omega.array() -= static_cast<Scalar>(0.5);
        d_Z.upload(Omega.data(), static_cast<size_t>(n) * l);
    }

    // ========================================================================
    // Step 2: Y = A_c · Omega  (m × l)
    // ========================================================================
    spmm_forward(d_Z.get());

    // ========================================================================
    // cuSOLVER QR workspace (pre-allocated once, reused for all QR calls)
    //   QR is called on Y (m × l) and Z (n × l), so workspace must fit both.
    // ========================================================================
    const int tau_len = l;   // min(m, l) = l since m >> l and n >> l typically
    DeviceMemory<Scalar> d_tau(tau_len);
    DeviceMemory<int> d_qr_info(1);

    // Query workspace sizes for both Y (m×l) and Z (n×l), take the max
    int lwork_y_geqrf = detail::cusolver_geqrf_bufferSize<Scalar>(
        ctx.cusolver, m, l, d_Y.get(), m);
    int lwork_y_orgqr = detail::cusolver_orgqr_bufferSize<Scalar>(
        ctx.cusolver, m, l, l, d_Y.get(), m, d_tau.get());
    int lwork_z_geqrf = detail::cusolver_geqrf_bufferSize<Scalar>(
        ctx.cusolver, n, l, d_Z.get(), n);
    int lwork_z_orgqr = detail::cusolver_orgqr_bufferSize<Scalar>(
        ctx.cusolver, n, l, l, d_Z.get(), n, d_tau.get());
    int qr_lwork = std::max({lwork_y_geqrf, lwork_y_orgqr, lwork_z_geqrf, lwork_z_orgqr});
    DeviceMemory<Scalar> d_qr_work(qr_lwork > 0 ? qr_lwork : 1);

    // ========================================================================
    // Step 3: Power iterations
    // ========================================================================
    for (int i = 0; i < q; ++i) {
        // QR(Y) in-place — fully on device via cuSOLVER
        detail::device_thin_qr<Scalar>(
            ctx.cusolver, d_Y.get(), m, l, d_tau.get(), d_qr_work.get(), qr_lwork, d_qr_info.get());

        // Z = A_c' · Y  (n × l)
        spmm_transpose(d_Y.get());

        // QR(Z) in-place — fully on device via cuSOLVER
        detail::device_thin_qr<Scalar>(
            ctx.cusolver, d_Z.get(), n, l, d_tau.get(), d_qr_work.get(), qr_lwork, d_qr_info.get());

        // Y = A_c · Z  (m × l)
        spmm_forward(d_Z.get());

        if (config.verbose) {
            FACTORNET_PRINT_IMPL("    power iter %d/%d done\n", i + 1, q);
        }
    }

    // ========================================================================
    // Step 4: Final QR of Y → orthonormal Q (m × l)
    // ========================================================================
    detail::device_thin_qr<Scalar>(
        ctx.cusolver, d_Y.get(), m, l, d_tau.get(), d_qr_work.get(), qr_lwork, d_qr_info.get());
    // d_Y now holds Q (m × l)

    // ========================================================================
    // Step 5: Bt = A_c' · Q  (n × l)  — stays on device for Step 6
    // ========================================================================
    spmm_transpose(d_Y.get());   // d_Z = A_c' · Q (n × l) = Bt

    // ========================================================================
    // Step 6: Device SVD of Bt (n × l) via cuSOLVER gesvdj
    //
    //   SVD(Bt) = U_bt · diag(S) · V_bt'
    //     where U_bt: n×l, S: l, V_bt: l×l  (econ mode, n >> l)
    //
    //   Since B = Bt', the SVD of B is:
    //     Left singular vectors of B  = V_bt  (l × l)
    //     Right singular vectors of B = U_bt  (n × l)
    //     Singular values of B        = S     (same)
    //
    //   Final results:
    //     result.d = S[:k]
    //     result.V = U_bt[:, :k]               (n × k)
    //     result.U = Q · V_bt[:, :k]           (m × k, via cuBLAS GEMM)
    // ========================================================================
    {
        // Create gesvdj parameters
        gesvdjInfo_t gesvdj_params = nullptr;
        CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1e-12));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 100));

        // Allocate device output buffers
        DeviceMemory<Scalar> d_svd_U(static_cast<size_t>(n) * l);   // U_bt: n × l
        DeviceMemory<Scalar> d_svd_S(l);                              // S: l
        DeviceMemory<Scalar> d_svd_V(static_cast<size_t>(l) * l);    // V_bt: l × l
        DeviceMemory<int> d_svd_info(1);

        // Query workspace size
        int svd_lwork = detail::cusolver_gesvdj_bufferSize<Scalar>(
            ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR, 1 /* econ */,
            n, l, d_Z.get(), n,
            d_svd_S.get(), d_svd_U.get(), n, d_svd_V.get(), l,
            gesvdj_params);
        DeviceMemory<Scalar> d_svd_work(svd_lwork > 0 ? svd_lwork : 1);

        // Run gesvdj: Bt = U_bt · diag(S) · V_bt'
        // Note: d_Z (input A) is overwritten by gesvdj internals
        detail::cusolver_gesvdj<Scalar>(
            ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR, 1 /* econ */,
            n, l, d_Z.get(), n,
            d_svd_S.get(), d_svd_U.get(), n, d_svd_V.get(), l,
            d_svd_work.get(), svd_lwork, d_svd_info.get(),
            gesvdj_params);

        CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));

        int k_actual = std::min(k, l);

        // GEMM on device: result_U = Q · V_bt[:, :k]
        //   Q = d_Y (m × l), V_bt[:,:k] = first k cols of d_svd_V (l × l)
        //   Column-major: first k cols are contiguous (l * k elements)
        DeviceMemory<Scalar> d_U_final(static_cast<size_t>(m) * k_actual);
        detail::cublas_gemm<Scalar>(
            ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            m, k_actual, l,
            Scalar(1), d_Y.get(), m,           // Q: m × l
            d_svd_V.get(), l,                  // V_bt[:, :k]: l × k
            Scalar(0), d_U_final.get(), m      // U_result: m × k
        );

        // Download results to host
        result.d.resize(k_actual);
        result.U.resize(m, k_actual);
        result.V.resize(n, k_actual);

        // Sync to ensure gesvdj and GEMM are complete
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        d_svd_S.download(result.d.data(), k_actual);
        d_U_final.download(result.U.data(), static_cast<size_t>(m) * k_actual);
        d_svd_U.download(result.V.data(), static_cast<size_t>(n) * k_actual);
    }

    int k_actual = std::min(k, l);
    result.k_selected = k_actual;
    result.iters_per_factor.push_back(q);

    // ------------------------------------------------------------------
    // Cleanup cuSPARSE descriptors
    // ------------------------------------------------------------------
    cusparseDestroyDnMat(dnDescr_Y);
    cusparseDestroyDnMat(dnDescr_Z);
    cusparseDestroyDnMat(dnDescr_input_n);
    cusparseDestroyDnMat(dnDescr_input_m);
    // dual_csr destructor handles descr_A and descr_AT cleanup

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU Randomized SVD done: %.1f ms, d[0]=%.4e d[k-1]=%.4e\n",
            result.wall_time_ms,
            static_cast<double>(result.d(0)),
            static_cast<double>(result.d(k_actual - 1)));
    }

    return result;
}


// ============================================================================
// Dense GPU Randomized SVD (cuBLAS GEMM instead of cuSPARSE SpMM)
// ============================================================================

/**
 * @brief GPU randomized truncated SVD/PCA for dense matrices.
 *
 * Uses cuBLAS GEMM for forward (A*X) and transpose (A'*Y) products,
 * which is significantly faster than converting dense to CSC and using
 * cuSPARSE SpMM.
 *
 * @tparam Scalar     float or double
 * @param h_A         Host column-major dense matrix (m × n)
 * @param m           Number of rows
 * @param n           Number of columns
 * @param config      SVD configuration
 * @return SVDResult with factors
 */
template<typename Scalar>
SVDResult<Scalar> randomized_svd_gpu_dense(
    const Scalar* h_A,
    int m, int n,
    const SVDConfig<Scalar>& config)
{
    using namespace FactorNet::gpu;
    using DenseMatrixS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVectorS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    auto t_start = std::chrono::high_resolution_clock::now();

    const int k = config.k_max;
    const int p = std::min(std::max(10, k / 5), std::min(m, n) - k);
    const int l = k + std::max(p, 0);
    const int q = (config.max_iter <= 0) ? 3 : std::min(config.max_iter, 10);

    SVDResult<Scalar> result;
    result.centered = config.center;

    const int BLOCK_SIZE = 256;

    // ------------------------------------------------------------------
    // 1. GPU context and dense matrix upload
    // ------------------------------------------------------------------
    GPUContext ctx;

    const size_t mn = static_cast<size_t>(m) * n;
    DeviceMemory<Scalar> d_A(mn);
    d_A.upload(h_A, mn);

    // ------------------------------------------------------------------
    // 2. Row means and Frobenius norm
    // ------------------------------------------------------------------
    DenseVectorS row_means;
    DeviceMemory<Scalar> d_mu;
    DeviceMemory<Scalar> d_colsums(l);
    DeviceMemory<Scalar> d_mu_dots(l);

    double A_norm_sq = 0;
    for (size_t i = 0; i < mn; ++i)
        A_norm_sq += static_cast<double>(h_A[i]) * static_cast<double>(h_A[i]);

    if (config.center) {
        row_means.resize(m);
        row_means.setZero();
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                row_means(i) += h_A[i + static_cast<size_t>(j) * m];
        if (n > 0) row_means /= static_cast<Scalar>(n);
        A_norm_sq -= static_cast<double>(n) * static_cast<double>(row_means.squaredNorm());

        result.row_means = row_means;
        d_mu = DeviceMemory<Scalar>(m);
        d_mu.upload(row_means.data(), m);
    }
    result.frobenius_norm_sq = A_norm_sq;

    // ------------------------------------------------------------------
    // 3. Allocate dense matrices on GPU
    // ------------------------------------------------------------------
    DeviceMemory<Scalar> d_Y(static_cast<size_t>(m) * l);  // m × l
    DeviceMemory<Scalar> d_Z(static_cast<size_t>(n) * l);  // n × l

    // ------------------------------------------------------------------
    // Helper lambdas for GEMM with PCA correction
    // ------------------------------------------------------------------

    // Forward GEMM: Y = A * X, then PCA correct
    // X is d_src_n (n × l), Y is d_Y (m × l)
    auto gemm_forward = [&](Scalar* d_src_n) {
        // Y = A * X: A is m×n, X is n×l → Y is m×l
        detail::cublas_gemm<Scalar>(
            ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            m, l, n,
            Scalar(1), d_A.get(), m,
            d_src_n, n,
            Scalar(0), d_Y.get(), m);

        if (config.center) {
            int smem = BLOCK_SIZE * sizeof(Scalar);
            detail::compute_colsums_kernel<<<l, BLOCK_SIZE, smem, ctx.stream>>>(
                d_src_n, d_colsums.get(), n, l);
            int total = m * l;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::pca_correct_forward_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_Y.get(), d_mu.get(), d_colsums.get(), m, l);
        }
    };

    // Transpose GEMM: Z = A' * Y, then PCA correct
    // Y is d_src_m (m × l), Z is d_Z (n × l)
    auto gemm_transpose = [&](Scalar* d_src_m) {
        // Z = A' * Y: A' is n×m, Y is m×l → Z is n×l
        detail::cublas_gemm<Scalar>(
            ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n, l, m,
            Scalar(1), d_A.get(), m,
            d_src_m, m,
            Scalar(0), d_Z.get(), n);

        if (config.center) {
            int smem = BLOCK_SIZE * sizeof(Scalar);
            detail::compute_mu_dots_kernel<<<l, BLOCK_SIZE, smem, ctx.stream>>>(
                d_mu.get(), d_src_m, d_mu_dots.get(), m, l);
            int total = n * l;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::pca_correct_transpose_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_Z.get(), d_mu_dots.get(), n, l);
        }
    };

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU Randomized SVD (dense): k=%d, l=%d, power_iter=%d\n", k, l, q);
    }

    // ========================================================================
    // Step 1: Random sketch Omega (n × l) → upload to d_Z
    // ========================================================================
    {
        DenseMatrixS Omega(n, l);
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        rng.fill_uniform(Omega.data(), Omega.rows(), Omega.cols());
        Omega.array() -= static_cast<Scalar>(0.5);
        d_Z.upload(Omega.data(), static_cast<size_t>(n) * l);
    }

    // ========================================================================
    // Step 2: Y = A_c * Omega  (m × l)
    // ========================================================================
    gemm_forward(d_Z.get());

    // ========================================================================
    // cuSOLVER QR workspace (pre-allocated once, reused)
    // ========================================================================
    const int tau_len = l;
    DeviceMemory<Scalar> d_tau(tau_len);
    DeviceMemory<int> d_qr_info(1);

    int lwork_y_geqrf = detail::cusolver_geqrf_bufferSize<Scalar>(
        ctx.cusolver, m, l, d_Y.get(), m);
    int lwork_y_orgqr = detail::cusolver_orgqr_bufferSize<Scalar>(
        ctx.cusolver, m, l, l, d_Y.get(), m, d_tau.get());
    int lwork_z_geqrf = detail::cusolver_geqrf_bufferSize<Scalar>(
        ctx.cusolver, n, l, d_Z.get(), n);
    int lwork_z_orgqr = detail::cusolver_orgqr_bufferSize<Scalar>(
        ctx.cusolver, n, l, l, d_Z.get(), n, d_tau.get());
    int qr_lwork = std::max({lwork_y_geqrf, lwork_y_orgqr, lwork_z_geqrf, lwork_z_orgqr});
    DeviceMemory<Scalar> d_qr_work(qr_lwork > 0 ? qr_lwork : 1);

    // ========================================================================
    // Step 3: Power iterations
    // ========================================================================
    for (int i = 0; i < q; ++i) {
        detail::device_thin_qr<Scalar>(
            ctx.cusolver, d_Y.get(), m, l, d_tau.get(), d_qr_work.get(), qr_lwork, d_qr_info.get());
        gemm_transpose(d_Y.get());
        detail::device_thin_qr<Scalar>(
            ctx.cusolver, d_Z.get(), n, l, d_tau.get(), d_qr_work.get(), qr_lwork, d_qr_info.get());
        gemm_forward(d_Z.get());

        if (config.verbose) {
            FACTORNET_PRINT_IMPL("    power iter %d/%d done\n", i + 1, q);
        }
    }

    // ========================================================================
    // Step 4: Final QR of Y → orthonormal Q (m × l)
    // ========================================================================
    detail::device_thin_qr<Scalar>(
        ctx.cusolver, d_Y.get(), m, l, d_tau.get(), d_qr_work.get(), qr_lwork, d_qr_info.get());

    // ========================================================================
    // Step 5: Bt = A_c' * Q  (n × l)
    // ========================================================================
    gemm_transpose(d_Y.get());

    // ========================================================================
    // Step 6: Device SVD of Bt (n × l) via cuSOLVER gesvdj
    // ========================================================================
    {
        gesvdjInfo_t gesvdj_params = nullptr;
        CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1e-12));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 100));

        DeviceMemory<Scalar> d_svd_U(static_cast<size_t>(n) * l);
        DeviceMemory<Scalar> d_svd_S(l);
        DeviceMemory<Scalar> d_svd_V(static_cast<size_t>(l) * l);
        DeviceMemory<int> d_svd_info(1);

        int svd_lwork = detail::cusolver_gesvdj_bufferSize<Scalar>(
            ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR, 1,
            n, l, d_Z.get(), n,
            d_svd_S.get(), d_svd_U.get(), n, d_svd_V.get(), l,
            gesvdj_params);
        DeviceMemory<Scalar> d_svd_work(svd_lwork > 0 ? svd_lwork : 1);

        detail::cusolver_gesvdj<Scalar>(
            ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR, 1,
            n, l, d_Z.get(), n,
            d_svd_S.get(), d_svd_U.get(), n, d_svd_V.get(), l,
            d_svd_work.get(), svd_lwork, d_svd_info.get(),
            gesvdj_params);

        CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));

        int k_actual = std::min(k, l);

        // GEMM: result_U = Q * V_bt[:, :k]
        DeviceMemory<Scalar> d_U_final(static_cast<size_t>(m) * k_actual);
        detail::cublas_gemm<Scalar>(
            ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            m, k_actual, l,
            Scalar(1), d_Y.get(), m,
            d_svd_V.get(), l,
            Scalar(0), d_U_final.get(), m);

        // Download results
        result.d.resize(k_actual);
        result.U.resize(m, k_actual);
        result.V.resize(n, k_actual);

        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        d_svd_S.download(result.d.data(), k_actual);
        d_U_final.download(result.U.data(), static_cast<size_t>(m) * k_actual);
        d_svd_U.download(result.V.data(), static_cast<size_t>(n) * k_actual);
    }

    int k_actual = std::min(k, l);
    result.k_selected = k_actual;
    result.iters_per_factor.push_back(q);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU Randomized SVD (dense) done: %.1f ms, d[0]=%.4e d[k-1]=%.4e\n",
            result.wall_time_ms,
            static_cast<double>(result.d(0)),
            static_cast<double>(result.d(k_actual - 1)));
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet
