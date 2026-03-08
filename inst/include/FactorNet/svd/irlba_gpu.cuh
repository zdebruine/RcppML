// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file irlba_gpu.cuh
 * @brief GPU-accelerated IRLBA (Augmented Implicitly Restarted Lanczos Bidiag)
 *
 * GPU counterpart to irlba_svd.hpp. Runs the augmented Lanczos bidiag
 * on GPU using cuSPARSE SpMV and cuBLAS GEMV/GEMM.
 *
 * Key GPU optimizations:
 *   - cuSPARSE SpMV for A*v and A'*w (CSC-as-CSR trick, dual-mode)
 *   - cuBLAS GEMV for reorthogonalization (two-pass Gram-Schmidt)
 *   - cuBLAS GEMM for restart compression (V, W rotation)
 *   - V (n×work) and W (m×work) device-resident throughout
 *   - Pre-allocated SpMV buffer reused across all iterations
 *   - PCA centering via lightweight GPU kernels
 *   - B (work×work) on host — JacobiSVD is fast for small matrices
 *
 * Algorithm overview (Baglama & Reichel 2005):
 *   1. Short Lanczos bidiagonalization: work steps (k+15)
 *   2. SVD of bidiagonal B → Ritz values + error bounds
 *   3. Convergence test (residual + SV ratio)
 *   4. Implicit restart: compress to k best directions
 *   5. Repeat until nu singular triplets converge
 *
 * Memory: O((m+n) * work) on device, work = k+15 typically
 * Matvecs per restart: 2 * work
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
// cuBLAS typed wrappers
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

// ============================================================================
// cuBLAS device-pointer mode helpers (non-blocking result on device)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_cublas_nrm2_device
#define FACTORNET_SVD_GPU_DEF_cublas_nrm2_device
/// nrm2 with result written to device memory (no host sync)
template<typename Scalar>
inline void cublas_nrm2_device(cublasHandle_t h, int n, const Scalar* x, Scalar* d_result);
template<>
inline void cublas_nrm2_device<double>(cublasHandle_t h, int n, const double* x, double* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    CUBLAS_CHECK(cublasDnrm2(h, n, x, 1, d_result));
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
template<>
inline void cublas_nrm2_device<float>(cublasHandle_t h, int n, const float* x, float* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    CUBLAS_CHECK(cublasSnrm2(h, n, x, 1, d_result));
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_nrm2_device

#ifndef FACTORNET_SVD_GPU_DEF_cublas_dot_device
#define FACTORNET_SVD_GPU_DEF_cublas_dot_device
/// dot product with result written to device memory (no host sync)
template<typename Scalar>
inline void cublas_dot_device(cublasHandle_t h, int n, const Scalar* x, const Scalar* y, Scalar* d_result);
template<>
inline void cublas_dot_device<double>(cublasHandle_t h, int n, const double* x, const double* y, double* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    CUBLAS_CHECK(cublasDdot(h, n, x, 1, y, 1, d_result));
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
template<>
inline void cublas_dot_device<float>(cublasHandle_t h, int n, const float* x, const float* y, float* d_result) {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    CUBLAS_CHECK(cublasSdot(h, n, x, 1, y, 1, d_result));
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
}
#endif // FACTORNET_SVD_GPU_DEF_cublas_dot_device

// ============================================================================
// PCA centering kernels (same as deflation)
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

// ============================================================================
// Fused kernels to reduce launch overhead and memory passes
// ============================================================================

/**
 * @brief Fused copy + axpy: dst = src_a + alpha * src_b
 * Replaces separate cublas_copy + cublas_axpy (2 kernels, 2 memory passes)
 * with a single kernel (1 launch, 1 pass).
 */
template<typename Scalar>
__global__ void fused_copy_axpy_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src_a,
    Scalar alpha,
    const Scalar* __restrict__ src_b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src_a[i] + alpha * src_b[i];
}

/**
 * @brief Fused normalize + copy: dst = src / norm
 * Replaces separate cublas_copy + cublas_scal with one kernel.
 */
template<typename Scalar>
__global__ void fused_normalize_copy_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src,
    Scalar inv_norm,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * inv_norm;
}

#ifndef FACTORNET_SVD_GPU_DEF_fused_recurrence_kernel
#define FACTORNET_SVD_GPU_DEF_fused_recurrence_kernel
/**
 * @brief Fused three-term recurrence: dst = src_a - beta * src_b
 * For W[:, j+1] = SpMV_result - R_F * W[:, j]
 */
template<typename Scalar>
__global__ void fused_recurrence_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ spmv_result,
    Scalar neg_beta,
    const Scalar* __restrict__ w_prev,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = spmv_result[i] + neg_beta * w_prev[i];
}
#endif // FACTORNET_SVD_GPU_DEF_fused_recurrence_kernel

// ============================================================================
// Device-scalar kernels (read scalars from device memory, fully async)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel
#define FACTORNET_SVD_GPU_DEF_subtract_ssvd_kernel
/// PCA forward centering: result[i] -= mu[i] * d_scale[0]  (reads scale from device)
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
/// PCA transpose centering: vec[i] -= d_val[0]  (reads val from device)
template<typename Scalar>
__global__ void subtract_device_scalar_kernel(
    Scalar* __restrict__ vec, int n, const Scalar* __restrict__ d_val)
{
    Scalar val = d_val[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] -= val;
}
#endif // FACTORNET_SVD_GPU_DEF_subtract_device_scalar_kernel

/// F = spmv_out - S * V[:,j], reading S from device memory
template<typename Scalar>
__global__ void fused_copy_axpy_device_scalar_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src_a,
    const Scalar* __restrict__ d_s,   // positive S on device
    const Scalar* __restrict__ src_b,
    int n)
{
    Scalar neg_s = -d_s[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src_a[i] + neg_s * src_b[i];
}

/// W[:,j+1] = spmv - R_F * W[:,j], reading R_F from device memory
template<typename Scalar>
__global__ void fused_recurrence_device_scalar_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ spmv_result,
    const Scalar* __restrict__ d_rf,   // positive R_F on device
    const Scalar* __restrict__ w_prev,
    int n)
{
    Scalar neg_rf = -d_rf[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = spmv_result[i] + neg_rf * w_prev[i];
}

/// V[:,j+1] = F / ||F||, with norm read from device; stores norm, flags degenerate
template<typename Scalar>
__global__ void normalize_copy_or_flag_kernel(
    Scalar* __restrict__ dst,
    const Scalar* __restrict__ src,
    const Scalar* __restrict__ d_norm,
    Scalar* __restrict__ d_norm_out,    // store norm for B construction
    Scalar eps23,
    Scalar* __restrict__ d_flag,        // set > 0 if degenerate
    int n)
{
    Scalar nrm = d_norm[0];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_norm_out[0] = nrm;
        if (nrm < eps23) d_flag[0] = Scalar(1);
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (nrm < eps23) { dst[i] = src[i]; return; } // copy without normalizing
    dst[i] = src[i] / nrm;
}

/// Normalize vec in-place; stores norm to two locations, flags degenerate
template<typename Scalar>
__global__ void normalize_or_flag_kernel(
    Scalar* __restrict__ vec, int n,
    const Scalar* __restrict__ d_norm,
    Scalar* __restrict__ d_scalar_out,  // d_S_scalar for next iteration
    Scalar* __restrict__ d_array_out,   // d_S_vals[j+1] for B construction
    Scalar eps23,
    Scalar* __restrict__ d_flag)
{
    Scalar nrm = d_norm[0];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_scalar_out[0] = nrm;
        d_array_out[0] = nrm;
        if (nrm < eps23) d_flag[0] = Scalar(1);
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (nrm < eps23) return; // leave non-normalized; flag was set
    vec[i] /= nrm;
}

// ============================================================================
// GPU Gram-Schmidt orthogonalization (single-pass for speed)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_gpu_orthog
#define FACTORNET_SVD_GPU_DEF_gpu_orthog
/**
 * @brief Orthogonalize d_y against the first ncols columns of d_X.
 *
 * Single pass of: h = X[:,0:ncols]' * y; y -= X[:,0:ncols] * h
 * (R's irlba uses single-pass; two-pass enabled via parameter)
 */
template<typename Scalar>
void gpu_orthog(cublasHandle_t cublas,
                const Scalar* d_X, Scalar* d_y,
                int dim, int ncols, Scalar* d_h,
                bool two_pass = true)
{
    if (ncols <= 0) return;
    Scalar one = Scalar(1), zero = Scalar(0), neg_one = Scalar(-1);

    // Pass 1: h = X' * y; y -= X * h
    cublas_gemv<Scalar>(cublas, CUBLAS_OP_T, dim, ncols,
        one, d_X, dim, d_y, zero, d_h);
    cublas_gemv<Scalar>(cublas, CUBLAS_OP_N, dim, ncols,
        neg_one, d_X, dim, d_h, one, d_y);

    if (two_pass) {
        // Pass 2 (extra numerical stability)
        cublas_gemv<Scalar>(cublas, CUBLAS_OP_T, dim, ncols,
            one, d_X, dim, d_y, zero, d_h);
        cublas_gemv<Scalar>(cublas, CUBLAS_OP_N, dim, ncols,
            neg_one, d_X, dim, d_h, one, d_y);
    }
}
#endif // FACTORNET_SVD_GPU_DEF_gpu_orthog

// ============================================================================
// Convergence test (matches CPU irlba)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_conv_test
#define FACTORNET_SVD_GPU_DEF_conv_test
template<typename Scalar>
inline bool conv_test(int work, int n_wanted, Scalar tol, Scalar svtol,
                      Scalar Smax, const DenseVector<Scalar>& svratio,
                      const DenseVector<Scalar>& residuals,
                      int& k_restart, Scalar S_norm) {
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    int n_converged = 0;
    for (int j = 0; j < work; ++j) {
        if (std::abs(residuals(j)) < tol * Smax && svratio(j) < svtol)
            n_converged++;
    }
    if (n_converged >= n_wanted || S_norm < eps) return true;
    if (k_restart < n_wanted + n_converged)
        k_restart = n_wanted + n_converged;
    if (k_restart > work - 3) k_restart = work - 3;
    if (k_restart < 1) k_restart = 1;
    return false;
}
#endif // FACTORNET_SVD_GPU_DEF_conv_test

}  // namespace detail


// ============================================================================
// Main GPU IRLBA
// ============================================================================

/**
 * @tparam Scalar  Compute precision (float or double).
 *
 * Note on CUDA Graphs: The inner Lanczos loop has strict serial data
 * dependencies (each SpMV depends on the previous step's output) and
 * variable-size reorthogonalization (j columns, j changes each step).
 * This makes CUDA Graph capture ineffective — there is no repeating
 * identical subgraph to replay, and SpMV dominates runtime (making
 * ~5µs launch overhead negligible). Similarly, multi-stream overlap
 * provides no benefit because every operation depends on the previous.
 */
template<typename Scalar>
SVDResult<Scalar> irlba_svd_gpu(
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

    const int nu = config.k_max;
    const int mn_min = std::min(m, n);
    // Augmentation size: max(20, k) — for large k, a bigger working
    // subspace reduces the number of restarts needed and improves accuracy
    // of boundary singular values.  Standard irlba uses k+7; we use
    // k+20 for small k but scale up for large k.
    // k=8→28, k=16→36, k=32→64, k=64→128, k=100→200
    const int aug = std::max(20, nu);
    const int work = std::min(nu + aug, mn_min);
    const int maxit = config.max_iter > 0 ? config.max_iter : 1000;
    const Scalar tol = config.tol > 0 ? config.tol : static_cast<Scalar>(1e-7);
    const Scalar svtol = tol;
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const Scalar eps23 = std::pow(eps, static_cast<Scalar>(2.0 / 3.0));

    SVDResult<Scalar> result;
    result.centered = config.center;

    const int BLOCK_SIZE = 256;

    // ------------------------------------------------------------------
    // 1. GPU context and sparse matrix upload
    // ------------------------------------------------------------------
    GPUContext ctx;

    SparseMatrixGPU<Scalar> d_A(m, n, nnz, h_col_ptr, h_row_idx, h_values);

    // ------------------------------------------------------------------
    // 1a. Construct CSR(A) from CSC(A) on device.
    //
    //     We already have CSC(A) = CSR(A^T). Converting to CSR(A) lets
    //     us use NON_TRANSPOSE (gather) for BOTH SpMV directions:
    //       A*v  → NON_TRANSPOSE on CSR(A)    (gather, ~0.12ms/call)
    //       A^T*w → NON_TRANSPOSE on CSR(A^T) (gather, ~0.12ms/call)
    //     instead of TRANSPOSE (scatter, ~0.72ms/call) for forward.
    //     This alone is a 6x speedup on the forward SpMV path.
    // ------------------------------------------------------------------
    DeviceMemory<int> d_csr_A_row_ptr(m + 1);
    DeviceMemory<int> d_csr_A_col_idx(nnz);
    DeviceMemory<Scalar> d_csr_A_values(nnz);

    // cusparseCsr2cscEx2: input = CSR(A^T) [n rows, m cols], output = CSC(A^T) = CSR(A) [m rows, n cols]
    {
        size_t csr2csc_bufSize = 0;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            ctx.cusparse, n, m, nnz,
            d_A.values.get(), d_A.col_ptr.get(), d_A.row_indices.get(),
            d_csr_A_values.get(), d_csr_A_row_ptr.get(), d_csr_A_col_idx.get(),
            CudaDataType<Scalar>::value,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            &csr2csc_bufSize));
        DeviceMemory<char> d_csr2csc_buf(csr2csc_bufSize > 0 ? csr2csc_bufSize : 1);
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            ctx.cusparse, n, m, nnz,
            d_A.values.get(), d_A.col_ptr.get(), d_A.row_indices.get(),
            d_csr_A_values.get(), d_csr_A_row_ptr.get(), d_csr_A_col_idx.get(),
            CudaDataType<Scalar>::value,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            d_csr2csc_buf.get()));
    }

    // cuSPARSE descriptors for both CSR matrices
    cusparseSpMatDescr_t matDescr_AT = nullptr;   // CSR(A^T): n rows × m cols
    cusparseSpMatDescr_t matDescr_A  = nullptr;   // CSR(A):   m rows × n cols
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matDescr_AT, n, m, nnz,
        d_A.col_ptr.get(), d_A.row_indices.get(), d_A.values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matDescr_A, m, n, nnz,
        d_csr_A_row_ptr.get(), d_csr_A_col_idx.get(), d_csr_A_values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));

    // ------------------------------------------------------------------
    // 2. Row means for PCA centering
    // ------------------------------------------------------------------
    DenseVectorS row_means;
    DeviceMemory<Scalar> d_mu;
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
    // 3. ||A||²_F
    // ------------------------------------------------------------------
    Scalar A_norm_sq = 0;
    for (int i = 0; i < nnz; ++i) A_norm_sq += h_values[i] * h_values[i];
    if (config.center) A_norm_sq -= static_cast<Scalar>(n) * row_means.squaredNorm();
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // ------------------------------------------------------------------
    // 4. Allocate device vectors and matrices
    // ------------------------------------------------------------------

    // SpMV output buffers — always FP32
    DeviceMemory<Scalar> d_spmv_out_n(n);  // for A^T * w result (length n)
    DeviceMemory<Scalar> d_spmv_out_m(m);  // for A * v result (length m)

    // Device-side ones vector for PCA sum(v)
    DeviceMemory<Scalar> d_ones_n;
    if (config.center) {
        d_ones_n = DeviceMemory<Scalar>(n);
        std::vector<Scalar> h_ones(n, Scalar(1));
        d_ones_n.upload(h_ones.data(), n);
    }

    // cuSPARSE SpMV setup — needed for ALL paths.
    // Both use NON_TRANSPOSE (gather) for maximum performance.
    // Use dummy device vectors for buffer-size query and preprocess analysis;
    // the actual input pointer is updated via cusparseDnVecSetValues before each call.
    cusparseDnVecDescr_t vecDescr_in_m = nullptr, vecDescr_in_n = nullptr;
    cusparseDnVecDescr_t vecDescr_out_n = nullptr, vecDescr_out_m = nullptr;
    Scalar spmv_alpha = Scalar(1), spmv_beta = Scalar(0);

    // Dummy device vectors for setup (kept alive for descriptor lifetime)
    DeviceMemory<Scalar> d_dummy_n(n), d_dummy_m(m);
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_out_n, n, d_spmv_out_n.get(), CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_out_m, m, d_spmv_out_m.get(), CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_in_n, n, d_dummy_n.get(), CudaDataType<Scalar>::value));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDescr_in_m, m, d_dummy_m.get(), CudaDataType<Scalar>::value));

    // Pre-allocate SpMV buffers — both NON_TRANSPOSE.
    // CRITICAL: use SEPARATE buffers for forward and transpose preprocess.
    // Sharing a single buffer causes the second preprocess to overwrite the
    // first's analysis state, corrupting SpMV results for the other direction.
    size_t bufSize_fwd = 0, bufSize_trans = 0;
    // Transpose: A^T*w = NON_TRANSPOSE on CSR(A^T) [n×m] * w[m] → y[n]
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, matDescr_AT, vecDescr_in_m,
        &spmv_beta, vecDescr_out_n,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufSize_trans));
    // Forward: A*v = NON_TRANSPOSE on CSR(A) [m×n] * v[n] → y[m]
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, matDescr_A, vecDescr_in_n,
        &spmv_beta, vecDescr_out_m,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufSize_fwd));

    // Separate device buffers for each preprocess result
    DeviceMemory<char> d_spmv_buffer_fwd(bufSize_fwd > 0 ? bufSize_fwd : 1);
    DeviceMemory<char> d_spmv_buffer_trans(bufSize_trans > 0 ? bufSize_trans : 1);

#if CUDART_VERSION >= 12000
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, matDescr_A, vecDescr_in_n,
        &spmv_beta, vecDescr_out_m,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_fwd.get()));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, matDescr_AT, vecDescr_in_m,
        &spmv_beta, vecDescr_out_n,
        CudaDataType<Scalar>::value,
        CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_trans.get()));
#endif

    // Lanczos subspace matrices — always Scalar (FP32 or FP64)
    DeviceMemory<Scalar> d_V(static_cast<size_t>(n) * work);
    DeviceMemory<Scalar> d_W(static_cast<size_t>(m) * work);
    DeviceMemory<Scalar> d_F(n);
    DeviceMemory<Scalar> d_V1(static_cast<size_t>(n) * work);
    DeviceMemory<Scalar> d_W1(static_cast<size_t>(m) * work);

    // Restart rotation matrices — always Scalar
    DeviceMemory<Scalar> d_BV_block(static_cast<size_t>(work) * work);
    DeviceMemory<Scalar> d_BU_block(static_cast<size_t>(work) * work);

    // Orthogonalization scratch — always Scalar
    DeviceMemory<Scalar> d_orth_h(work);

    // Device-side scalar storage for fully async inner loop (always float)
    DeviceMemory<Scalar> d_nrm2_result(1);    // async nrm2 output
    DeviceMemory<Scalar> d_pca_dot_result(1);  // async PCA dot output
    DeviceMemory<Scalar> d_S_scalar(1);        // current S for next iteration
    DeviceMemory<Scalar> d_S_vals(work);       // all S values for deferred B
    DeviceMemory<Scalar> d_RF_vals(work);      // all R_F values for deferred B
    DeviceMemory<Scalar> d_degenerate_flag(1); // > 0 if invariant subspace hit
    d_degenerate_flag.zero();

    // Host-side bidiagonal and SVD workspace
    DenseMatrixS B = DenseMatrixS::Zero(work, work);
    DenseVectorS svratio_prev = DenseVectorS::Zero(work);
    DenseVectorS svratio_comp(work);
    DenseVectorS residuals_vec(work);
    DenseVectorS signed_res(work);

    // ------------------------------------------------------------------
    // SpMV helper lambdas
    // ------------------------------------------------------------------

    // Forward SpMV: d_spmv_out_m = A * v (fully async, no host sync)
    // cuSPARSE NON_TRANSPOSE (gather) on CSR(A) — 6x faster than TRANSPOSE (scatter).
    auto spmv_forward = [&](Scalar* d_v_ptr) {
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecDescr_in_n, d_v_ptr));
        Scalar beta_zero = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMV(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, matDescr_A, vecDescr_in_n,
            &beta_zero, vecDescr_out_m,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_fwd.get()));

        // PCA centering: spmv_out -= mu * sum(v), fully async
        if (config.center) {
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, n, d_ones_n.get(), d_v_ptr, d_pca_dot_result.get());
            int blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::subtract_scaled_vector_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_spmv_out_m.get(), d_mu.get(), m, d_pca_dot_result.get());
        }
    };

    // Transpose SpMV: d_spmv_out_n = A^T * w (fully async, no host sync)
    // cuSPARSE NON_TRANSPOSE (gather) on CSR(A^T) = CSC(A).
    auto spmv_transpose = [&](Scalar* d_w_ptr) {
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecDescr_in_m, d_w_ptr));
        Scalar beta_zero = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMV(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, matDescr_AT, vecDescr_in_m,
            &beta_zero, vecDescr_out_n,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_trans.get()));

        // PCA centering: spmv_out -= dot(mu, w) * 1, fully async
        if (config.center) {
            detail::cublas_dot_device<Scalar>(
                ctx.cublas, m, d_mu.get(), d_w_ptr, d_pca_dot_result.get());
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::subtract_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_spmv_out_n.get(), n, d_pca_dot_result.get());
        }
    };

    // Column pointer helpers — always Scalar*
    auto d_V_col = [&](int j) { return d_V.get() + static_cast<size_t>(j) * n; };
    auto d_W_col = [&](int j) { return d_W.get() + static_cast<size_t>(j) * m; };

    // Orthogonalization helpers — always standard FP32 cuBLAS
    auto orthog_V = [&](Scalar* d_y, int ncols) {
        detail::gpu_orthog<Scalar>(ctx.cublas, d_V.get(), d_y, n, ncols, d_orth_h.get());
    };
    auto orthog_W = [&](Scalar* d_y, int ncols) {
        detail::gpu_orthog<Scalar>(ctx.cublas, d_W.get(), d_y, m, ncols, d_orth_h.get());
    };

    // ------------------------------------------------------------------
    // Initialize V[:, 0] = random unit vector
    // ------------------------------------------------------------------
    {
        DenseVectorS v0(n);
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        DenseMatrixS v0_mat(n, 1);
        rng.fill_uniform(v0_mat.data(), v0_mat.rows(), v0_mat.cols());
        v0 = v0_mat.col(0);
        v0.array() -= static_cast<Scalar>(0.5);
        v0.normalize();
        CUDA_CHECK(cudaMemcpy(d_V_col(0), v0.data(), n * sizeof(Scalar), cudaMemcpyHostToDevice));
    }

    int k = nu;
    int iter = 0;
    int mprod = 0;
    Scalar Smax = 0;
    bool converged = false;

    DenseVectorS BS;
    DenseMatrixS BU, BV_mat;

    // Phase-level CUDA event profiling (active only when verbose)
    cudaEvent_t ev_start, ev_stop;
    float phase_ms = 0;
    double t_spmv_fwd = 0, t_spmv_trans = 0;
    double t_orthog_V = 0, t_orthog_W = 0;
    double t_norms = 0, t_fused = 0;
    double t_host_svd = 0, t_restart = 0, t_setup = 0;
    int n_spmv_fwd = 0, n_spmv_trans = 0;
    if (config.verbose) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }
    // Macros for phase timing (only when verbose)
    #define PHASE_BEGIN() do { if (config.verbose) { CUDA_CHECK(cudaEventRecord(ev_start, ctx.stream)); } } while(0)
    #define PHASE_END(accum) do { if (config.verbose) { \
        CUDA_CHECK(cudaEventRecord(ev_stop, ctx.stream)); \
        CUDA_CHECK(cudaEventSynchronize(ev_stop)); \
        CUDA_CHECK(cudaEventElapsedTime(&phase_ms, ev_start, ev_stop)); \
        (accum) += phase_ms; \
    } } while(0)

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU IRLBA: k=%d, work=%d, tol=%.1e, maxit=%d\n",
            nu, work, static_cast<double>(tol), maxit);
    }

    // ========================================================================
    // Main IRLBA loop
    // ========================================================================
    {
        auto t_setup_end = std::chrono::high_resolution_clock::now();
        t_setup = std::chrono::duration<double, std::milli>(t_setup_end - t_start).count();
    }
    while (iter < maxit) {
        int j_start = (iter == 0) ? 0 : k;
        int j = j_start;

        // ==========================================================
        // Step 1: First forward multiply W[:, j] = A * V[:, j]
        //   All vectors are Scalar (FP32/FP64) in both mixed and
        //   non-mixed modes. SpMV lambda handles the dispatch.
        // ==========================================================

        PHASE_BEGIN();
        spmv_forward(d_V_col(j));
        PHASE_END(t_spmv_fwd); n_spmv_fwd++;
        mprod++;

        // Copy SpMV result to W[:, j]
        PHASE_BEGIN();
        detail::cublas_copy<Scalar>(ctx.cublas, m, d_spmv_out_m.get(), d_W_col(j));

        // Reorthogonalize W[:, j] against W[:, 0:j-1]
        if (j > 0) {
            orthog_W(d_W_col(j), j);
        }
        PHASE_END(t_orthog_W);

        // nrm2 (host sync for convergence)
        PHASE_BEGIN();
        Scalar S = detail::cublas_nrm2<Scalar>(ctx.cublas, m, d_W_col(j));
        PHASE_END(t_norms);

        if (S < eps23 && j == 0) {
            if (config.verbose)
                FACTORNET_PRINT_IMPL("    Starting vector near null space (S=%.2e)\n", (double)S);
            break;
        }
        if (S < eps23) {
            // Replace with random vector
            DenseMatrixS r_mat(m, 1);
            rng::SplitMix64 rng2(config.seed + iter * 1000 + 1);
            rng2.fill_uniform(r_mat.data(), r_mat.rows(), r_mat.cols());
            r_mat.array() -= Scalar(0.5);
            CUDA_CHECK(cudaMemcpy(d_W_col(j), r_mat.data(), m * sizeof(Scalar), cudaMemcpyHostToDevice));
            orthog_W(d_W_col(j), j);
            Scalar nrm = detail::cublas_nrm2<Scalar>(ctx.cublas, m, d_W_col(j));
            detail::cublas_scal<Scalar>(ctx.cublas, m, Scalar(1) / nrm, d_W_col(j));
            S = 0;
        } else {
            detail::cublas_scal<Scalar>(ctx.cublas, m, Scalar(1) / S, d_W_col(j));
        }

        // Upload initial S to device for the fully-async inner loop
        CUDA_CHECK(cudaMemcpyAsync(d_S_scalar.get(), &S, sizeof(Scalar),
            cudaMemcpyHostToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_S_vals.get() + j_start, &S, sizeof(Scalar),
            cudaMemcpyHostToDevice, ctx.stream));
        d_degenerate_flag.zero();

        // ==========================================================
        // Step 2: Inner Lanczos bidiagonalization loop
        //   FULLY ASYNC: zero host sync points in the inner loop.
        //   nrm2 results stored on device; B built after loop.
        // ==========================================================
        while (j < work) {
            // F = A' * W[:, j]  (async)
            PHASE_BEGIN();
            spmv_transpose(d_W_col(j));
            PHASE_END(t_spmv_trans); n_spmv_trans++;
            mprod++;

            // F = spmv_out - S * V[:, j]  (reads S from device memory)
            PHASE_BEGIN();
            {
                int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                detail::fused_copy_axpy_device_scalar_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                    d_F.get(), d_spmv_out_n.get(), d_S_scalar.get(), d_V_col(j), n);
            }
            PHASE_END(t_fused);

            // Reorthogonalize F against V[:, 0:j]  (two-pass GS)
            PHASE_BEGIN();
            orthog_V(d_F.get(), j + 1);
            PHASE_END(t_orthog_V);

            if (j + 1 < work) {
                // R_F = ||F|| → device (no host sync)
                PHASE_BEGIN();
                detail::cublas_nrm2_device<Scalar>(
                    ctx.cublas, n, d_F.get(), d_nrm2_result.get());

                // V[:, j+1] = F / R_F; store R_F → d_RF_vals[j]; flag if degenerate
                {
                    int vblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    detail::normalize_copy_or_flag_kernel<<<vblocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_V_col(j + 1), d_F.get(), d_nrm2_result.get(),
                        d_RF_vals.get() + j, eps23, d_degenerate_flag.get(), n);
                }
                PHASE_END(t_norms);

                // W[:, j+1] = A * V[:, j+1]  (async)
                PHASE_BEGIN();
                spmv_forward(d_V_col(j + 1));
                PHASE_END(t_spmv_fwd); n_spmv_fwd++;
                mprod++;

                // W[:, j+1] = spmv_out - R_F * W[:, j]  (reads R_F from device)
                PHASE_BEGIN();
                {
                    int wblocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    detail::fused_recurrence_device_scalar_kernel<<<wblocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_W_col(j + 1), d_spmv_out_m.get(), d_RF_vals.get() + j,
                        d_W_col(j), m);
                }
                PHASE_END(t_fused);

                // Reorthogonalize W[:, j+1] against W[:, 0:j]  (two-pass GS)
                PHASE_BEGIN();
                orthog_W(d_W_col(j + 1), j + 1);
                PHASE_END(t_orthog_W);

                // S = ||W[:, j+1]|| → device (no host sync)
                PHASE_BEGIN();
                detail::cublas_nrm2_device<Scalar>(
                    ctx.cublas, m, d_W_col(j + 1), d_nrm2_result.get());

                // Normalize W[:, j+1]; store S → d_S_scalar + d_S_vals[j+1]
                {
                    int wblocks2 = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    detail::normalize_or_flag_kernel<<<wblocks2, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_W_col(j + 1), m, d_nrm2_result.get(),
                        d_S_scalar.get(), d_S_vals.get() + j + 1,
                        eps23, d_degenerate_flag.get());
                }
                PHASE_END(t_norms);
            }
            j++;
        }

        // Sync and download scalars for deferred B construction
        ctx.sync();

        // Check degenerate flag (extremely rare — invariant subspace)
        {
            Scalar h_flag = Scalar(0);
            d_degenerate_flag.download(&h_flag, 1);
            if (h_flag > Scalar(0) && config.verbose) {
                FACTORNET_PRINT_IMPL("    Warning: invariant subspace detected in inner loop\n");
            }
        }

        // Download S and R_F arrays, build bidiagonal B
        {
            std::vector<Scalar> h_S_vals(work), h_RF_vals(work);
            CUDA_CHECK(cudaMemcpy(h_S_vals.data(), d_S_vals.get(),
                work * sizeof(Scalar), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_RF_vals.data(), d_RF_vals.get(),
                work * sizeof(Scalar), cudaMemcpyDeviceToHost));

            for (int jj = j_start; jj < work; ++jj) {
                B(jj, jj) = h_S_vals[jj];
                if (jj + 1 < work) B(jj + 1, jj) = h_RF_vals[jj];
            }
        }

        // ==========================================================
        // Step 3: SVD of bidiagonal B (work × work) on host
        // ==========================================================
        {
            auto t_svd_start = std::chrono::high_resolution_clock::now();
            Eigen::JacobiSVD<DenseMatrixS> svd_b(
                B.topLeftCorner(work, work),
                Eigen::ComputeFullU | Eigen::ComputeFullV);

            BS = svd_b.singularValues();
            BU = svd_b.matrixV();       // RIGHT SVs of B_lower → for W + residuals
            BV_mat = svd_b.matrixU();   // LEFT SVs of B_lower → for V
            auto t_svd_end = std::chrono::high_resolution_clock::now();
            t_host_svd += std::chrono::duration<double, std::milli>(t_svd_end - t_svd_start).count();
        }

        // Normalize residual vector F (sync already done above)
        Scalar R_F = detail::cublas_nrm2<Scalar>(ctx.cublas, n, d_F.get());
        if (R_F > eps) {
            detail::cublas_scal<Scalar>(ctx.cublas, n, Scalar(1) / R_F, d_F.get());
        } else {
            R_F = 0;
        }

        // ==========================================================
        // Step 4: Convergence test
        // ==========================================================
        Smax = std::max(Smax, BS(0));

        for (int i = 0; i < work; ++i) {
            signed_res(i) = R_F * BU(work - 1, i);
            residuals_vec(i) = std::abs(signed_res(i));
        }

        for (int i = 0; i < work; ++i) {
            if (svratio_prev(i) > 0 && BS(i) > eps) {
                svratio_comp(i) = std::abs(svratio_prev(i) - BS(i)) / BS(i);
            } else {
                svratio_comp(i) = Scalar(1);
            }
        }

        k = work;
        converged = detail::conv_test(work, nu, tol, svtol, Smax,
            svratio_comp, residuals_vec, k, S);

        if (config.verbose) {
            int n_conv = 0;
            for (int i = 0; i < work; ++i)
                if (std::abs(residuals_vec(i)) < tol * Smax && svratio_comp(i) < svtol)
                    n_conv++;
            FACTORNET_PRINT_IMPL("    iter %d: sigma_1=%.6e, |res_1|=%.2e, converged=%d/%d, mprod=%d\n",
                iter + 1, (double)BS(0), (double)std::abs(residuals_vec(0)),
                n_conv, nu, mprod);
        }

        if (converged) { iter++; break; }
        if (iter >= maxit - 1) {
            iter++;
            if (config.verbose)
                FACTORNET_PRINT_IMPL("    IRLBA: reached maxit=%d without full convergence\n", maxit);
            break;
        }

        // ==========================================================
        // Step 5: Store previous BS for svratio
        // ==========================================================
        for (int i = 0; i < work; ++i) svratio_prev(i) = BS(i);

        // ==========================================================
        // Step 6: Implicit restart — compress to k directions on GPU
        //
        // V1[:, 0:k] = V[:, 0:work] * BV_mat[0:work, 0:k]
        // W1[:, 0:k] = W[:, 0:work] * BU[0:work, 0:k]
        // ==========================================================
        PHASE_BEGIN();
        {
            DenseMatrixS BV_block = BV_mat.topLeftCorner(work, k);
            DenseMatrixS BU_block_data = BU.topLeftCorner(work, k);

            d_BV_block.upload(BV_block.data(), static_cast<size_t>(work) * k);
            d_BU_block.upload(BU_block_data.data(), static_cast<size_t>(work) * k);

            detail::cublas_gemm<Scalar>(ctx.cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, k, work,
                Scalar(1), d_V.get(), n,
                d_BV_block.get(), work,
                Scalar(0), d_V1.get(), n);

            CUDA_CHECK(cudaMemcpyAsync(d_V.get(), d_V1.get(),
                static_cast<size_t>(n) * k * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));

            detail::cublas_copy<Scalar>(ctx.cublas, n, d_F.get(), d_V_col(k));

            detail::cublas_gemm<Scalar>(ctx.cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, k, work,
                Scalar(1), d_W.get(), m,
                d_BU_block.get(), work,
                Scalar(0), d_W1.get(), m);

            CUDA_CHECK(cudaMemcpyAsync(d_W.get(), d_W1.get(),
                static_cast<size_t>(m) * k * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        }
        PHASE_END(t_restart);

        // Rebuild B
        B.setZero();
        for (int i = 0; i < k; ++i) {
            B(i, i) = BS(i);
            B(i, k) = signed_res(i);
        }

        iter++;
    }

    // ========================================================================
    // Extract final results (from device V and W, rotated by B's SVD)
    // ========================================================================
    int k_actual = std::min(nu, work);

    if (BS.size() >= k_actual && BU.rows() >= work && BV_mat.rows() >= work) {
        DenseMatrixS BU_k = BU.leftCols(k_actual);
        DenseMatrixS BV_k = BV_mat.leftCols(k_actual);

        DeviceMemory<Scalar> d_BU_k(static_cast<size_t>(work) * k_actual);
        DeviceMemory<Scalar> d_BV_k(static_cast<size_t>(work) * k_actual);
        d_BU_k.upload(BU_k.data(), static_cast<size_t>(work) * k_actual);
        d_BV_k.upload(BV_k.data(), static_cast<size_t>(work) * k_actual);

        DeviceMemory<Scalar> d_U_final(static_cast<size_t>(m) * k_actual);
        detail::cublas_gemm<Scalar>(ctx.cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, k_actual, work,
            Scalar(1), d_W.get(), m,
            d_BU_k.get(), work,
            Scalar(0), d_U_final.get(), m);

        DeviceMemory<Scalar> d_V_final(static_cast<size_t>(n) * k_actual);
        detail::cublas_gemm<Scalar>(ctx.cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, k_actual, work,
            Scalar(1), d_V.get(), n,
            d_BV_k.get(), work,
            Scalar(0), d_V_final.get(), n);

        ctx.sync();

        result.U.resize(m, k_actual);
        result.V.resize(n, k_actual);
        d_U_final.download(result.U.data(), static_cast<size_t>(m) * k_actual);
        d_V_final.download(result.V.data(), static_cast<size_t>(n) * k_actual);
        result.d = BS.head(k_actual);
    } else {
        result.U = DenseMatrixS::Zero(m, k_actual);
        result.V = DenseMatrixS::Zero(n, k_actual);
        result.d = DenseVectorS::Zero(k_actual);
    }

    result.k_selected = k_actual;
    result.iters_per_factor.push_back(iter);

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    if (vecDescr_in_m)  cusparseDestroyDnVec(vecDescr_in_m);
    if (vecDescr_in_n)  cusparseDestroyDnVec(vecDescr_in_n);
    if (vecDescr_out_n) cusparseDestroyDnVec(vecDescr_out_n);
    if (vecDescr_out_m) cusparseDestroyDnVec(vecDescr_out_m);
    if (matDescr_AT) cusparseDestroySpMat(matDescr_AT);
    if (matDescr_A)  cusparseDestroySpMat(matDescr_A);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config.verbose) {
        FACTORNET_PRINT_IMPL("  GPU IRLBA done: %d restarts, %d matvecs, %.1f ms, %s\n",
            iter, mprod, result.wall_time_ms,
            converged ? "converged" : "NOT converged");

        // Phase breakdown
        double t_total_phased = t_spmv_fwd + t_spmv_trans + t_orthog_V + t_orthog_W
                                + t_norms + t_fused + t_host_svd + t_restart;
        FACTORNET_PRINT_IMPL("  Phase breakdown (GPU event timing, %.1f ms tracked of %.1f ms wall):\n",
            t_total_phased + t_setup, result.wall_time_ms);
        FACTORNET_PRINT_IMPL("    Setup (upload+init):  %7.2f ms\n", t_setup);
        FACTORNET_PRINT_IMPL("    SpMV forward  (A*v):  %7.2f ms  (%4.1f%%)  [%d calls, %.3f ms/call]\n",
            t_spmv_fwd, 100.0*t_spmv_fwd/t_total_phased, n_spmv_fwd,
            n_spmv_fwd > 0 ? t_spmv_fwd/n_spmv_fwd : 0.0);
        FACTORNET_PRINT_IMPL("    SpMV transpose(A'w):  %7.2f ms  (%4.1f%%)  [%d calls, %.3f ms/call]\n",
            t_spmv_trans, 100.0*t_spmv_trans/t_total_phased, n_spmv_trans,
            n_spmv_trans > 0 ? t_spmv_trans/n_spmv_trans : 0.0);
        FACTORNET_PRINT_IMPL("    Orthog V (GEMV):      %7.2f ms  (%4.1f%%)\n",
            t_orthog_V, 100.0*t_orthog_V/t_total_phased);
        FACTORNET_PRINT_IMPL("    Orthog W (GEMV):      %7.2f ms  (%4.1f%%)\n",
            t_orthog_W, 100.0*t_orthog_W/t_total_phased);
        FACTORNET_PRINT_IMPL("    Norms + normalize:    %7.2f ms  (%4.1f%%)\n",
            t_norms, 100.0*t_norms/t_total_phased);
        FACTORNET_PRINT_IMPL("    Fused kernels:        %7.2f ms  (%4.1f%%)\n",
            t_fused, 100.0*t_fused/t_total_phased);
        FACTORNET_PRINT_IMPL("    Host SVD (B):         %7.2f ms  (%4.1f%%)\n",
            t_host_svd, 100.0*t_host_svd/t_total_phased);
        FACTORNET_PRINT_IMPL("    Restart GEMM:         %7.2f ms  (%4.1f%%)\n",
            t_restart, 100.0*t_restart/t_total_phased);

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
    }

    #undef PHASE_BEGIN
    #undef PHASE_END

    return result;
}

}  // namespace svd
}  // namespace FactorNet
