// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file lanczos_gpu.cuh
 * @brief GPU-accelerated Golub-Kahan-Lanczos bidiagonalization for truncated SVD
 *
 * GPU counterpart to lanczos_svd.hpp. Runs the Lanczos bidiagonalization
 * on GPU using cuSPARSE SpMV for matvecs and cuBLAS GEMV for reorthog.
 *
 * Key GPU optimizations:
 *   - cuSPARSE SpMV for A*p and A'*q (CSC-as-CSR trick)
 *   - cuBLAS GEMV for full two-pass reorthogonalization
 *   - P (n × jmax+1) and Q (m × jmax) device-resident throughout
 *   - Bidiagonal B on host — JacobiSVD for small matrix
 *   - Adaptive early stopping via Ritz error bounds
 *   - Pre-allocated SpMV buffer reused across all steps
 *   - PCA centering via lightweight GPU kernels
 *
 * Algorithm:
 *   A · P_j = Q_j · B_j  (approximately)
 *   where P (n × j) and Q (m × j) have orthonormal columns,
 *   B (j × j) is upper bidiagonal.
 *
 * Cost per step: 2 SpMVs O(nnz) + reorthog O((m+n)·j).
 * Total: O(nnz · jmax + (m+n) · jmax²) where jmax = 2k.
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
#include <limits>
#include <chrono>
#include <cmath>

namespace FactorNet {
namespace svd {
namespace detail {

// ============================================================================
// cuBLAS typed wrappers (same pattern as IRLBA GPU)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_cublas_nrm2
#define FACTORNET_SVD_GPU_DEF_cublas_nrm2
template<typename Scalar>
inline Scalar cublas_nrm2(cublasHandle_t h, int n, const Scalar* x);
template<>
inline double cublas_nrm2<double>(cublasHandle_t h, int n, const double* x) {
    double r; CUBLAS_CHECK(cublasDnrm2(h, n, x, 1, &r)); return r;
}
template<>
inline float cublas_nrm2<float>(cublasHandle_t h, int n, const float* x) {
    float r; CUBLAS_CHECK(cublasSnrm2(h, n, x, 1, &r)); return r;
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_nrm2

#ifndef FACTORNET_SVD_GPU_DEF_cublas_scal
#define FACTORNET_SVD_GPU_DEF_cublas_scal
template<typename Scalar>
inline void cublas_scal(cublasHandle_t h, int n, Scalar alpha, Scalar* x);
template<>
inline void cublas_scal<double>(cublasHandle_t h, int n, double alpha, double* x) {
    CUBLAS_CHECK(cublasDscal(h, n, &alpha, x, 1));
}
template<>
inline void cublas_scal<float>(cublasHandle_t h, int n, float alpha, float* x) {
    CUBLAS_CHECK(cublasSscal(h, n, &alpha, x, 1));
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_scal

#ifndef FACTORNET_SVD_GPU_DEF_cublas_axpy
#define FACTORNET_SVD_GPU_DEF_cublas_axpy
template<typename Scalar>
inline void cublas_axpy(cublasHandle_t h, int n, Scalar alpha, const Scalar* x, Scalar* y);
template<>
inline void cublas_axpy<double>(cublasHandle_t h, int n, double alpha, const double* x, double* y) {
    CUBLAS_CHECK(cublasDaxpy(h, n, &alpha, x, 1, y, 1));
}
template<>
inline void cublas_axpy<float>(cublasHandle_t h, int n, float alpha, const float* x, float* y) {
    CUBLAS_CHECK(cublasSaxpy(h, n, &alpha, x, 1, y, 1));
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_axpy

#ifndef FACTORNET_SVD_GPU_DEF_cublas_copy
#define FACTORNET_SVD_GPU_DEF_cublas_copy
template<typename Scalar>
inline void cublas_copy(cublasHandle_t h, int n, const Scalar* x, Scalar* y);
template<>
inline void cublas_copy<double>(cublasHandle_t h, int n, const double* x, double* y) {
    CUBLAS_CHECK(cublasDcopy(h, n, x, 1, y, 1));
}
template<>
inline void cublas_copy<float>(cublasHandle_t h, int n, const float* x, float* y) {
    CUBLAS_CHECK(cublasScopy(h, n, x, 1, y, 1));
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_copy

#ifndef FACTORNET_SVD_GPU_DEF_cublas_dot
#define FACTORNET_SVD_GPU_DEF_cublas_dot
template<typename Scalar>
inline Scalar cublas_dot(cublasHandle_t h, int n, const Scalar* x, const Scalar* y);
template<>
inline double cublas_dot<double>(cublasHandle_t h, int n, const double* x, const double* y) {
    double r; CUBLAS_CHECK(cublasDdot(h, n, x, 1, y, 1, &r)); return r;
}
template<>
inline float cublas_dot<float>(cublasHandle_t h, int n, const float* x, const float* y) {
    float r; CUBLAS_CHECK(cublasSdot(h, n, x, 1, y, 1, &r)); return r;
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_dot

#ifndef FACTORNET_SVD_GPU_DEF_cublas_gemv
#define FACTORNET_SVD_GPU_DEF_cublas_gemv
template<typename Scalar>
inline void cublas_gemv(cublasHandle_t h, cublasOperation_t trans,
    int m, int n, Scalar alpha, const Scalar* A, int lda,
    const Scalar* x, Scalar beta, Scalar* y);
template<>
inline void cublas_gemv<double>(cublasHandle_t h, cublasOperation_t trans,
    int m, int n, double alpha, const double* A, int lda,
    const double* x, double beta, double* y) {
    CUBLAS_CHECK(cublasDgemv(h, trans, m, n, &alpha, A, lda, x, 1, &beta, y, 1));
}
template<>
inline void cublas_gemv<float>(cublasHandle_t h, cublasOperation_t trans,
    int m, int n, float alpha, const float* A, int lda,
    const float* x, float beta, float* y) {
    CUBLAS_CHECK(cublasSgemv(h, trans, m, n, &alpha, A, lda, x, 1, &beta, y, 1));
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_gemv

#ifndef FACTORNET_SVD_GPU_DEF_cublas_gemm
#define FACTORNET_SVD_GPU_DEF_cublas_gemm
template<typename Scalar>
inline void cublas_gemm(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, Scalar alpha, const Scalar* A, int lda,
    const Scalar* B, int ldb, Scalar beta, Scalar* C, int ldc);
template<>
inline void cublas_gemm<double>(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double* C, int ldc) {
    CUBLAS_CHECK(cublasDgemm(h, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}
template<>
inline void cublas_gemm<float>(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc) {
    CUBLAS_CHECK(cublasSgemm(h, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_gemm

// ============================================================================
// PCA centering kernels
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_subtract_scalar_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_scalar_kernel
template<typename Scalar>
__global__ void subtract_scalar_kernel(Scalar* vec, int n, Scalar val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] -= val;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_scalar_kernel

#ifndef FACTORNET_SVD_GPU_DEF_subtract_scaled_vector_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_scaled_vector_kernel
template<typename Scalar>
__global__ void subtract_scaled_vector_kernel(
    Scalar* result, const Scalar* mu, int m, Scalar scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) result[i] -= mu[i] * scale;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_scaled_vector_kernel

#ifndef FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel
/// PCA centering: result[i] -= mu[i] * d_scale[0]  (reads scale from device memory, fully async)
template<typename Scalar>
__global__ void subtract_scaled_vector_device_scalar_kernel(
    Scalar* __restrict__ result, const Scalar* __restrict__ mu,
    int m, const Scalar* __restrict__ d_scale)
{
    Scalar scale = d_scale[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) result[i] -= mu[i] * scale;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel

#ifndef FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel
/// PCA centering: vec[i] -= d_val[0]  (reads val from device memory, fully async)
template<typename Scalar>
__global__ void subtract_device_scalar_kernel(
    Scalar* __restrict__ vec, int n, const Scalar* __restrict__ d_val)
{
    Scalar val = d_val[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] -= val;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel

#ifndef FACTORNET_SVD_GPU_DEF_cublas_dot_device
#define FACTORNET_SVD_GPU_DEF_cublas_dot_device
/// cuBLAS device-pointer dot (non-blocking result on device)
template<typename Scalar>
inline void cublas_dot_device(cublasHandle_t h, int n, const Scalar* x, const Scalar* y, Scalar* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDdot(h, n, x, 1, y, 1, d_result));
    } else {
        CUBLAS_CHECK(cublasSdot(h, n, x, 1, y, 1, d_result));
    }
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_dot_device

#ifndef FACTORNET_SVD_GPU_DEF_fused_recurrence_kernel
#define FACTORNET_SVD_GPU_DEF_fused_recurrence_kernel
/// Fused three-term recurrence: dst = spmv_result - beta * prev
template<typename Scalar>
__global__ void fused_recurrence_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ spmv_result,
    Scalar neg_beta,
    const Scalar* __restrict__ prev,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = spmv_result[i] + neg_beta * prev[i];
}
#endif // FACTORNET_SVD_GPU_DEF_fused_recurrence_kernel

/// dst = src_a - d_scalar[0] * src_b (reads scalar from device memory, fully async)
template<typename Scalar>
__global__ void fused_recurrence_device_ptr_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src_a,
    const Scalar* __restrict__ d_scalar,
    const Scalar* __restrict__ src_b,
    int n)
{
    Scalar s = d_scalar[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src_a[i] - s * src_b[i];
}

/// dst = src / d_norm[0], safe against near-zero norm (fully async)
template<typename Scalar>
__global__ void normalize_copy_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src,
    const Scalar* __restrict__ d_norm,
    Scalar eps,
    int n)
{
    Scalar nrm = d_norm[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = (nrm < eps) ? Scalar(0) : src[i] / nrm;
}

#ifndef FACTORNET_SVD_GPU_DEF_cublas_nrm2_device
#define FACTORNET_SVD_GPU_DEF_cublas_nrm2_device
/// cuBLAS nrm2 with result written to device memory (no host sync)
template<typename Scalar>
inline void cublas_nrm2_device(cublasHandle_t h, int n, const Scalar* x, Scalar* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    if constexpr (std::is_same_v<Scalar, double>) {
        CUBLAS_CHECK(cublasDnrm2(h, n, x, 1, d_result));
    } else {
        CUBLAS_CHECK(cublasSnrm2(h, n, x, 1, d_result));
    }
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_nrm2_device

// ============================================================================
// GPU two-pass Gram-Schmidt (identical to IRLBA's)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_gpu_orthog
#define FACTORNET_SVD_GPU_DEF_gpu_orthog
template<typename Scalar>
void gpu_orthog(cublasHandle_t cublas,
                const Scalar* d_X, Scalar* d_y,
                int dim, int ncols, Scalar* d_h)
{
    if (ncols <= 0) return;
    Scalar one = Scalar(1), zero = Scalar(0), neg_one = Scalar(-1);

    cublas_gemv<Scalar>(cublas, CUBLAS_OP_T, dim, ncols,
        one, d_X, dim, d_y, zero, d_h);
    cublas_gemv<Scalar>(cublas, CUBLAS_OP_N, dim, ncols,
        neg_one, d_X, dim, d_h, one, d_y);

    cublas_gemv<Scalar>(cublas, CUBLAS_OP_T, dim, ncols,
        one, d_X, dim, d_y, zero, d_h);
    cublas_gemv<Scalar>(cublas, CUBLAS_OP_N, dim, ncols,
        neg_one, d_X, dim, d_h, one, d_y);
}
#endif // FACTORNET_SVD_GPU_DEF_gpu_orthog

}  // namespace detail


// ============================================================================
// Main GPU Lanczos SVD — internal impl with pre-built GPU resources
// ============================================================================

/**
 * @brief GPU Lanczos SVD with pre-built GPU context and DualCSR.
 *
 * Internal function called by both lanczos_svd_gpu (public API) and
 * krylov_svd_gpu (shared resources to avoid double-uploading the matrix).
 *
 * Parameters ctx, dual_csr, d_mu, row_means, A_norm_sq_in are pre-built
 * by the caller.  This avoids the expensive GPUContext creation, PCIe
 * upload, and csr2csc transpose when krylov calls lanczos as Phase 1.
 */
template<typename Scalar>
SVDResult<Scalar> lanczos_svd_gpu_impl(
    const int* h_col_ptr,
    const int* h_row_idx,
    const Scalar* h_values,
    int m, int n, int nnz,
    const SVDConfig<Scalar>& config,
    FactorNet::gpu::GPUContext& ctx,
    FactorNet::gpu::DualCSR<Scalar>& dual_csr,
    FactorNet::gpu::DeviceMemory<Scalar>& d_mu,         // size m if center, else empty
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& row_means,  // size m if center
    double A_norm_sq_in)
{
    using namespace FactorNet::gpu;
    using DenseMatrixS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVectorS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    auto t_start = std::chrono::high_resolution_clock::now();

    const int k = config.k_max;
    // Default: max(3k, k+50) steps — 2k is too tight for bottom SVs at large k.
    // Reference: for k=50, this gives 150 vs the old 100, ensuring better convergence
    // of all k Ritz values. Early stopping avoids waste when convergence is fast.
    const int default_steps = std::max(3 * k, k + 50);
    const int user_steps = config.max_iter > 0 ? config.max_iter : 0;
    const int jmax = std::min(std::min(m, n) - 1,
                              std::max(default_steps, user_steps));

    SVDResult<Scalar> result;
    result.centered = config.center;
    // Use pre-provided Frobenius norm and row means (no re-computation)
    result.frobenius_norm_sq = A_norm_sq_in;
    if (config.center && row_means.size() > 0) result.row_means = row_means;

    const Scalar eps = std::numeric_limits<Scalar>::epsilon() * 100;
    const Scalar conv_tol = config.tol > 0 ? config.tol : static_cast<Scalar>(1e-10);
    const int check_interval = 5;
    const int min_steps_before_check = k + 2;
    const int BLOCK_SIZE = 256;

    // ------------------------------------------------------------------
    // SpMV setup (dual-CSR: both directions use NON_TRANSPOSE gather)
    // Uses the pre-built dual_csr — no matrix upload or csr2csc needed.
    // ------------------------------------------------------------------
    DeviceMemory<Scalar> d_spmv_out_n(n);
    DeviceMemory<Scalar> d_spmv_out_m(m);

    cusparseDnVecDescr_t vecDescr_in_m, vecDescr_in_n;
    cusparseDnVecDescr_t vecDescr_out_n, vecDescr_out_m;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_out_n, n, d_spmv_out_n.get(), CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_out_m, m, d_spmv_out_m.get(), CudaDataType<Scalar>::value));

    DeviceMemory<Scalar> d_dummy_n(n), d_dummy_m(m);
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_in_n, n, d_dummy_n.get(), CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_in_m, m, d_dummy_m.get(), CudaDataType<Scalar>::value));

    Scalar spmv_alpha = Scalar(1), spmv_beta = Scalar(0);
    size_t bufSize_fwd = 0, bufSize_trans = 0;

    // Forward: A*v via NON_TRANSPOSE on CSR(A) [m rows × n cols]
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_A, vecDescr_in_n,
        &spmv_beta, vecDescr_out_m,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufSize_fwd));

    // Transpose: A^T*w via NON_TRANSPOSE on CSR(A^T) [n rows × m cols]
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_AT, vecDescr_in_m,
        &spmv_beta, vecDescr_out_n,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufSize_trans));

    // CRITICAL: use SEPARATE preprocess buffers for forward and transpose.
    // Sharing a single buffer causes the second preprocess to overwrite the
    // first's analysis state, corrupting SpMV results for the other direction.
    DeviceMemory<char> d_spmv_buffer_fwd(bufSize_fwd > 0 ? bufSize_fwd : 1);
    DeviceMemory<char> d_spmv_buffer_trans(bufSize_trans > 0 ? bufSize_trans : 1);

    // SpMV preprocess (CUDA 12+): precompute analysis for repeated SpMV calls
#if CUDART_VERSION >= 12000
    cusparseSpMV_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_A, vecDescr_in_n,
        &spmv_beta, vecDescr_out_m,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_fwd.get());
    cusparseSpMV_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, dual_csr.descr_AT, vecDescr_in_m,
        &spmv_beta, vecDescr_out_n,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_trans.get());
#endif

    // ------------------------------------------------------------------
    // 5. Allocate Lanczos vectors on device
    // ------------------------------------------------------------------
    DeviceMemory<Scalar> d_P(static_cast<size_t>(n) * (jmax + 1)); // n × (jmax+1)
    DeviceMemory<Scalar> d_Q(static_cast<size_t>(m) * jmax);        // m × jmax
    DeviceMemory<Scalar> d_r(m);  // working vector

    // Device-side ones vector for PCA sum(v) via cublas_dot
    DeviceMemory<Scalar> d_ones_n;
    if (config.center) {
        d_ones_n = DeviceMemory<Scalar>(n);
        std::vector<Scalar> h_ones(n, Scalar(1));
        d_ones_n.upload(h_ones.data(), n);
    }
    DeviceMemory<Scalar> d_s(n);  // working vector

    // Workspace for orthogonalization
    DeviceMemory<Scalar> d_orth_h(jmax + 1);

    // Device-scalar workspace for PCA dot results (async, no host sync)
    DeviceMemory<Scalar> d_pca_dot_result(1);

    // Host bidiagonal entries
    DenseVectorS alpha_vec(jmax);
    DenseVectorS beta_vec(jmax + 1);
    beta_vec(0) = 0;

    // Device-side bidiagonal vectors: nrm2 writes directly here (zero host sync)
    DeviceMemory<Scalar> d_alpha_dev(jmax);
    DeviceMemory<Scalar> d_beta_dev(jmax + 1);

    auto d_P_col = [&](int j) -> Scalar* { return d_P.get() + static_cast<size_t>(j) * n; };
    auto d_Q_col = [&](int j) -> Scalar* { return d_Q.get() + static_cast<size_t>(j) * m; };

    // ------------------------------------------------------------------
    // SpMV helper lambdas (dual-CSR: both use NON_TRANSPOSE gather)
    // ------------------------------------------------------------------
    auto spmv_forward = [&](Scalar* d_v_ptr) {
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecDescr_in_n, d_v_ptr));
        Scalar bz = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMV(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, dual_csr.descr_A, vecDescr_in_n,
            &bz, vecDescr_out_m,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_fwd.get()));

        if (config.center) {
            // sum(v) = dot(ones, v) → device scalar (no host sync)
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, n, d_ones_n.get(), d_v_ptr, d_pca_dot_result.get());
            int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::subtract_scaled_vector_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_spmv_out_m.get(), d_mu.get(), m, d_pca_dot_result.get());
        }
    };

    auto spmv_transpose = [&](Scalar* d_w_ptr) {
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecDescr_in_m, d_w_ptr));
        Scalar bz = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMV(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, dual_csr.descr_AT, vecDescr_in_m,
            &bz, vecDescr_out_n,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_trans.get()));

        if (config.center) {
            // mu_dot_w = dot(mu, w) → device scalar (no host sync)
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, m, d_mu.get(), d_w_ptr, d_pca_dot_result.get());
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::subtract_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_spmv_out_n.get(), n, d_pca_dot_result.get());
        }
    };

    // ------------------------------------------------------------------
    // 6. Initialize p_0 = random unit vector
    // ------------------------------------------------------------------
    {
        DenseVectorS p0(n);
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        DenseMatrixS p0_mat(n, 1);
        rng.fill_uniform(p0_mat.data(), p0_mat.rows(), p0_mat.cols());
        p0 = p0_mat.col(0);
        p0.array() -= Scalar(0.5);
        p0.normalize();
        CUDA_CHECK(cudaMemcpy(d_P_col(0), p0.data(), n * sizeof(Scalar), cudaMemcpyHostToDevice));
    }

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU Lanczos bidiag: k=%d, jmax=%d, tol=%.1e\n",
            k, jmax, (double)conv_tol);
    }

    // ========================================================================
    // Golub-Kahan-Lanczos bidiagonalization
    // ========================================================================
    int j_actual = 0;

    for (int j = 0; j < jmax; ++j) {
        // Step 1: r = A · p_j - beta_j · q_{j-1}
        spmv_forward(d_P_col(j));

        if (j > 0) {
            // Fused: r = spmv_out - beta_j * q_{j-1}  (reads beta from device)
            int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::fused_recurrence_device_ptr_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_r.get(), d_spmv_out_m.get(), d_beta_dev.get() + j, d_Q_col(j - 1), m);
        } else {
            // r = spmv_out_m (simple copy)
            detail::cublas_copy<Scalar>(ctx.cublas, m, d_spmv_out_m.get(), d_r.get());
        }

        // Full reorthogonalization against Q[:, 0:j-1] (two-pass)
        if (j > 0) {
            detail::gpu_orthog<Scalar>(
                ctx.cublas, d_Q.get(), d_r.get(), m, j, d_orth_h.get());
        }

        // alpha_j = ||r|| → device (no sync)
        detail::cublas_nrm2_device<Scalar>(
            ctx.cublas, m, d_r.get(), d_alpha_dev.get() + j);

        // q_j = r / alpha_j (reads norm from device, handles breakdown)
        {
            int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::normalize_copy_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_Q_col(j), d_r.get(), d_alpha_dev.get() + j, eps, m);
        }

        // Step 2: s = A' · q_j - alpha_j · p_j (reads alpha from device)
        spmv_transpose(d_Q_col(j));
        {
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::fused_recurrence_device_ptr_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_s.get(), d_spmv_out_n.get(), d_alpha_dev.get() + j, d_P_col(j), n);
        }

        // Full reorthogonalization against P[:, 0:j] (two-pass)
        detail::gpu_orthog<Scalar>(
            ctx.cublas, d_P.get(), d_s.get(), n, j + 1, d_orth_h.get());

        // beta_{j+1} = ||s|| → device (no sync)
        detail::cublas_nrm2_device<Scalar>(
            ctx.cublas, n, d_s.get(), d_beta_dev.get() + (j + 1));

        // p_{j+1} = s / beta_{j+1} (reads norm from device, handles breakdown)
        {
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::normalize_copy_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_P_col(j + 1), d_s.get(), d_beta_dev.get() + (j + 1), eps, n);
        }

        j_actual = j + 1;

        // Adaptive early stopping (periodic sync point)
        if (j_actual >= min_steps_before_check &&
            j_actual % check_interval == 0 &&
            j_actual < jmax) {
            // Sync + download device bidiagonal for convergence check
            ctx.sync();
            CUDA_CHECK(cudaMemcpy(alpha_vec.data(), d_alpha_dev.get(),
                j_actual * sizeof(Scalar), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(beta_vec.data() + 1, d_beta_dev.get() + 1,
                j_actual * sizeof(Scalar), cudaMemcpyDeviceToHost));

            // Check for breakdowns (alpha ≈ 0 → lucky breakdown)
            bool had_breakdown = false;
            for (int jj = 0; jj < j_actual; ++jj) {
                if (std::abs(alpha_vec(jj)) < eps) {
                    j_actual = jj;
                    had_breakdown = true;
                    if (config.verbose)
                        FACTORNET_PRINT_IMPL("    Lucky breakdown at step %d\n", jj);
                    break;
                }
            }
            if (had_breakdown) break;

            DenseMatrixS B_check = DenseMatrixS::Zero(j_actual, j_actual);
            for (int jj = 0; jj < j_actual; ++jj) {
                B_check(jj, jj) = alpha_vec(jj);
                if (jj < j_actual - 1)
                    B_check(jj, jj + 1) = beta_vec(jj + 1);
            }
            Eigen::JacobiSVD<DenseMatrixS> svd_check(B_check, Eigen::ComputeFullU);
            auto sigmas = svd_check.singularValues();
            auto U_B = svd_check.matrixU();

            int k_check = std::min(k, j_actual);
            Scalar beta_last = std::abs(beta_vec(j_actual));
            bool all_converged = true;
            for (int i = 0; i < k_check; ++i) {
                Scalar err = beta_last * std::abs(U_B(j_actual - 1, i));
                if (err > conv_tol * sigmas(0)) {
                    all_converged = false;
                    break;
                }
            }
            if (all_converged) {
                if (config.verbose)
                    FACTORNET_PRINT_IMPL("    Early stop at step %d: all %d Ritz values converged\n",
                        j_actual, k_check);
                break;
            }
        }
    }

    if (config.verbose)
        FACTORNET_PRINT_IMPL("    Completed %d Lanczos steps\n", j_actual);

    // Download bidiagonal entries from device for final B-matrix SVD
    ctx.sync();
    if (j_actual > 0) {
        CUDA_CHECK(cudaMemcpy(alpha_vec.data(), d_alpha_dev.get(),
            j_actual * sizeof(Scalar), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(beta_vec.data() + 1, d_beta_dev.get() + 1,
            j_actual * sizeof(Scalar), cudaMemcpyDeviceToHost));
    }

    // ========================================================================
    // Build bidiagonal B (j_actual × j_actual), SVD on host
    // ========================================================================
    DenseMatrixS B = DenseMatrixS::Zero(j_actual, j_actual);
    for (int j = 0; j < j_actual; ++j) {
        B(j, j) = alpha_vec(j);
        if (j < j_actual - 1) B(j, j + 1) = beta_vec(j + 1);
    }

    Eigen::JacobiSVD<DenseMatrixS> svd_b(B, Eigen::ComputeFullU | Eigen::ComputeFullV);

    int k_actual = std::min(k, j_actual);
    result.d = svd_b.singularValues().head(k_actual);

    // ========================================================================
    // Final rotation on GPU: U = Q * U_B, V = P * V_B
    // ========================================================================
    {
        DenseMatrixS U_B = svd_b.matrixU().leftCols(k_actual);     // j_actual × k_actual
        DenseMatrixS V_B = svd_b.matrixV().leftCols(k_actual);     // j_actual × k_actual

        DeviceMemory<Scalar> d_UB(static_cast<size_t>(j_actual) * k_actual);
        DeviceMemory<Scalar> d_VB(static_cast<size_t>(j_actual) * k_actual);
        d_UB.upload(U_B.data(), static_cast<size_t>(j_actual) * k_actual);
        d_VB.upload(V_B.data(), static_cast<size_t>(j_actual) * k_actual);

        // U_final (m × k_actual) = Q[:, 0:j_actual] * U_B
        DeviceMemory<Scalar> d_U_final(static_cast<size_t>(m) * k_actual);
        detail::cublas_gemm<Scalar>(ctx.cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, k_actual, j_actual,
            Scalar(1), d_Q.get(), m,
            d_UB.get(), j_actual,
            Scalar(0), d_U_final.get(), m);

        // V_final (n × k_actual) = P[:, 0:j_actual] * V_B
        DeviceMemory<Scalar> d_V_final(static_cast<size_t>(n) * k_actual);
        detail::cublas_gemm<Scalar>(ctx.cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, k_actual, j_actual,
            Scalar(1), d_P.get(), n,
            d_VB.get(), j_actual,
            Scalar(0), d_V_final.get(), n);

        ctx.sync();

        result.U.resize(m, k_actual);
        result.V.resize(n, k_actual);
        d_U_final.download(result.U.data(), static_cast<size_t>(m) * k_actual);
        d_V_final.download(result.V.data(), static_cast<size_t>(n) * k_actual);
    }

    result.k_selected = k_actual;
    result.iters_per_factor.push_back(j_actual);

    // ------------------------------------------------------------------
    // Convergence diagnostic
    // ------------------------------------------------------------------
    if (config.verbose && j_actual > 0) {
        Scalar resid_bound = std::abs(beta_vec(j_actual));
        FACTORNET_PRINT_IMPL("    Residual norm (beta_%d) = %.4e\n", j_actual, (double)resid_bound);
        for (int i = 0; i < std::min(k_actual, 5); ++i) {
            Scalar err_bound = resid_bound *
                std::abs(svd_b.matrixU()(j_actual - 1, i));
            FACTORNET_PRINT_IMPL("    sigma_%d = %.4e  err_bound = %.4e\n",
                i + 1, (double)result.d(i), (double)err_bound);
        }
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    cusparseDestroyDnVec(vecDescr_in_m);
    cusparseDestroyDnVec(vecDescr_in_n);
    cusparseDestroyDnVec(vecDescr_out_n);
    cusparseDestroyDnVec(vecDescr_out_m);
    // dual_csr destructor handles descr_A and descr_AT cleanup

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU Lanczos SVD done: %d steps, %.1f ms\n",
            j_actual, result.wall_time_ms);
    }

    return result;
}

// ============================================================================
// Public API: build GPU resources once, then call lanczos_svd_gpu_impl.
// Other callers (krylov_svd_gpu) that pre-build these resources can call
// lanczos_svd_gpu_impl directly to avoid redundant upload + context creation.
// ============================================================================
template<typename Scalar>
SVDResult<Scalar> lanczos_svd_gpu(
    const int* h_col_ptr,
    const int* h_row_idx,
    const Scalar* h_values,
    int m, int n, int nnz,
    const SVDConfig<Scalar>& config)
{
    using namespace FactorNet::gpu;
    using DenseVectorS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Build GPU context, upload matrix, build dual-CSR transpose
    GPUContext ctx;
    SparseMatrixGPU<Scalar> d_A(m, n, nnz, h_col_ptr, h_row_idx, h_values);
    DualCSR<Scalar> dual_csr;
    dual_csr.init(ctx, d_A, m, n, nnz);

    // Compute row means and Frobenius norm (can be shared if caller pre-computes)
    DenseVectorS row_means;
    DeviceMemory<Scalar> d_mu;
    double A_norm_sq = 0;
    for (int i = 0; i < nnz; ++i)
        A_norm_sq += static_cast<double>(h_values[i]) * static_cast<double>(h_values[i]);
    if (config.center) {
        Eigen::Map<const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>>
            A_map(m, n, nnz, h_col_ptr, h_row_idx, h_values);
        row_means.resize(m);
        row_means.setZero();
        for (int j = 0; j < n; ++j)
            for (typename decltype(A_map)::InnerIterator it(A_map, j); it; ++it)
                row_means(it.row()) += static_cast<Scalar>(it.value());
        if (n > 0) row_means /= static_cast<Scalar>(n);
        A_norm_sq -= static_cast<double>(n) * static_cast<double>(row_means.squaredNorm());
        d_mu = DeviceMemory<Scalar>(m);
        d_mu.upload(row_means.data(), m);
    }

    return lanczos_svd_gpu_impl(
        h_col_ptr, h_row_idx, h_values, m, n, nnz, config,
        ctx, dual_csr, d_mu, row_means, A_norm_sq);
}

}  // namespace svd
}  // namespace FactorNet
