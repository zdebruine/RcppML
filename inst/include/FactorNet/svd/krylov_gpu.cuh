// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file krylov_gpu.cuh
 * @brief GPU-accelerated Krylov-seeded projected refinement (KSPR) SVD
 *
 * Phase 1: GPU Lanczos seed (delegates to lanczos_svd_gpu)
 * Phase 2: Gram-solve-then-project refinement (fully on-device pipeline)
 *   - cuSPARSE SpMM for A*V (m×k) and A'*W (n×k)
 *   - cuBLAS SYRK for Gram matrices (V'V, W'W) on device
 *   - cuSOLVER potrf + cuBLAS TRSM for Cholesky solve on device
 *   - GPU-side constraint projection kernels (L1, nonneg, bounds)
 *   - GPU-side column normalize kernel
 *   - Host-side fallback for Gram regularization (L21, angular, graph)
 *
 * Supports all constrained features: L1, L2, nonneg, bounds, L21, angular,
 * graph Laplacian, CV/auto-rank, PCA centering.
 *
 * Architecture notes:
 *   - Sparse matrix uploaded once, reused across all phases
 *   - W and V stay on-device throughout refinement; only downloaded for
 *     convergence checks (W per-pass) and final result extraction
 *   - SYRK→potrf→TRSM pipeline eliminates host Gram/Cholesky bottleneck
 *   - Feature headers (ortho.hpp, graph_reg.hpp, L21.hpp) operate on small
 *     k×k Gram matrices on HOST as fallback — negligible cost, full NMF reuse
 *   - Constraint projection + column normalize are GPU-side kernels
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
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/svd/lanczos_gpu.cuh>
#include <FactorNet/svd/test_entries.hpp>

// Feature headers — backend-agnostic, operate on small Gram matrices (host)
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/graph_reg.hpp>

#include <Eigen/Dense>
// Eigen/Sparse intentionally omitted: sparse data accessed via raw CSC arrays,
// not Eigen::Map, keeping GPU headers free of the heavy Sparse module.

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <chrono>
#include <cmath>

namespace FactorNet {
namespace svd {
namespace detail {

// ============================================================================
// cuBLAS typed wrappers
// ============================================================================

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

// cuBLAS SYRK typed wrappers (C = alpha * A^T * A + beta * C, fills LOWER)
template<typename Scalar>
inline void cublas_syrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, Scalar alpha, const Scalar* A, int lda, Scalar beta, Scalar* C, int ldc);
template<>
inline void cublas_syrk<double>(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, double alpha, const double* A, int lda, double beta, double* C, int ldc) {
    CUBLAS_CHECK(cublasDsyrk(h, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc));
}
template<>
inline void cublas_syrk<float>(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, float alpha, const float* A, int lda, float beta, float* C, int ldc) {
    CUBLAS_CHECK(cublasSsyrk(h, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc));
}

// cuBLAS TRSM typed wrappers (triangular solve: op(A) * X = alpha*B or X * op(A) = alpha*B)
template<typename Scalar>
inline void cublas_trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, Scalar alpha, const Scalar* A, int lda, Scalar* B, int ldb);
template<>
inline void cublas_trsm<double>(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
    CUBLAS_CHECK(cublasDtrsm(h, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb));
}
template<>
inline void cublas_trsm<float>(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
    CUBLAS_CHECK(cublasStrsm(h, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb));
}

#ifndef FACTORNET_SVD_GPU_DEF_cublas_axpy
#define FACTORNET_SVD_GPU_DEF_cublas_axpy
// cuBLAS AXPY typed wrappers (y = alpha*x + y)
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

// cuSOLVER potrf typed wrappers (Cholesky factorization: A = L * L')
template<typename Scalar>
inline void cusolver_potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
    int n, Scalar* A, int lda, Scalar* work, int lwork, int* devInfo);
template<>
inline void cusolver_potrf<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo,
    int n, double* A, int lda, double* work, int lwork, int* devInfo) {
    CUSOLVER_CHECK(cusolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo));
}
template<>
inline void cusolver_potrf<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo,
    int n, float* A, int lda, float* work, int lwork, int* devInfo) {
    CUSOLVER_CHECK(cusolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo));
}

template<typename Scalar>
inline int cusolver_potrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
    int n, Scalar* A, int lda);
template<>
inline int cusolver_potrf_bufferSize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo,
    int n, double* A, int lda) {
    int lwork; CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, &lwork));
    return lwork;
}
template<>
inline int cusolver_potrf_bufferSize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo,
    int n, float* A, int lda) {
    int lwork; CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, &lwork));
    return lwork;
}

// ============================================================================
// PCA centering kernels (SpMM multi-column)
// ============================================================================

#ifndef FACTORNET_SVD_GPU_DEF_pca_correct_forward_kernel
#define FACTORNET_SVD_GPU_DEF_pca_correct_forward_kernel
template<typename Scalar>
__global__ void pca_correct_forward_kernel(
    Scalar* __restrict__ Y,
    const Scalar* __restrict__ mu,
    const Scalar* __restrict__ colsums,
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
template<typename Scalar>
__global__ void pca_correct_transpose_kernel(
    Scalar* __restrict__ Z,
    const Scalar* __restrict__ dots,
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
template<typename Scalar>
__global__ void compute_colsums_kernel(
    const Scalar* __restrict__ X,
    Scalar* __restrict__ colsums,
    int n, int l)
{
    int j = blockIdx.x;
    if (j >= l) return;
    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);
    int tid = threadIdx.x;
    Scalar val = Scalar(0);
    for (int i = tid; i < n; i += blockDim.x)
        val += X[i + static_cast<size_t>(j) * n];
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) colsums[j] = sdata[0];
}
#endif // FACTORNET_SVD_GPU_DEF_compute_colsums_kernel

#ifndef FACTORNET_SVD_GPU_DEF_compute_mu_dots_kernel
#define FACTORNET_SVD_GPU_DEF_compute_mu_dots_kernel
template<typename Scalar>
__global__ void compute_mu_dots_kernel(
    const Scalar* __restrict__ mu,
    const Scalar* __restrict__ Y,
    Scalar* __restrict__ dots,
    int m, int l)
{
    int j = blockIdx.x;
    if (j >= l) return;
    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);
    int tid = threadIdx.x;
    Scalar val = Scalar(0);
    for (int i = tid; i < m; i += blockDim.x)
        val += mu[i] * Y[i + static_cast<size_t>(j) * m];
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
// Column squared-norm + diagonal regularization kernels
// ============================================================================

/**
 * @brief Compute squared L2 norm of each column of X (column-major).
 *
 * One block per column with shared-memory parallel reduction.
 * sqnorms[c] = sum_i X[i + c*rows]^2.
 */
template<typename Scalar>
__global__ void column_sqnorm_kernel(
    const Scalar* __restrict__ X,
    Scalar* __restrict__ sqnorms,
    int rows, int k)
{
    int c = blockIdx.x;
    if (c >= k) return;

    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);
    int tid = threadIdx.x;

    Scalar sum = Scalar(0);
    for (int i = tid; i < rows; i += blockDim.x) {
        Scalar v = X[i + static_cast<size_t>(c) * rows];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) sqnorms[c] = sdata[0];
}

/**
 * @brief Add a scalar value to every diagonal element of a k×k column-major matrix.
 *
 * Used for Tikhonov regularization (eps + L2) on the Gram matrix.
 */
template<typename Scalar>
inline __global__ void add_to_diagonal_kernel(
    Scalar* __restrict__ G, int k, Scalar val)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < k) G[i * (k + 1)] += val;  // column-major (i,i) = i + i*k
}

// ============================================================================
// Constraint projection kernels
// ============================================================================

/**
 * @brief Soft-threshold + nonneg + upper bound projection for one column
 *
 * For each element x in W[:, c]:
 *   if L1 > 0: x = sign(x) * max(|x| - thr, 0)      [soft threshold]
 *   if nonneg:  x = max(x, 0)
 *   if ub > 0:  x = min(x, ub)
 */
template<typename Scalar>
__global__ void project_constraints_kernel(
    Scalar* __restrict__ X,           // m × k, column-major
    int dim, int k,
    Scalar L1, bool nonneg, Scalar ub,
    const Scalar* __restrict__ norm_sq_per_col)  // k scalars
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * k;
    if (idx >= total) return;

    int c = idx / dim;
    Scalar nsq = norm_sq_per_col[c];
    if (nsq <= Scalar(0)) { X[idx] = Scalar(0); return; }

    Scalar val = X[idx];

    // Soft threshold
    if (L1 > Scalar(0)) {
        Scalar thr = L1 / (Scalar(2) * nsq);
        val = (val > thr) ? val - thr : (val < -thr) ? val + thr : Scalar(0);
    }

    // Nonneg
    if (nonneg && val < Scalar(0)) val = Scalar(0);

    // Upper bound
    if (ub > Scalar(0) && val > ub) val = ub;

    X[idx] = val;
}

/**
 * @brief Normalize each column of X in-place; write norms to d_norms.
 *
 * One block per column. Uses shared-memory reduction to compute column norm,
 * then divides each element by that norm (in-place). If norm <= eps the column
 * is left as-is and d_norms[c] is set to 0.
 *
 * This kernel replaces the D→H download / host normalize / H→D upload cycle
 * in the Phase 2 fast path, eliminating 2× large PCIe transfers per pass.
 */
template<typename Scalar>
__global__ void normalize_columns_kernel(
    Scalar* __restrict__ X,    // dim × cols column-major
    int rows, int cols,
    Scalar* __restrict__ d_norms,  // output: norm[c] = ||X[:,c]|| before normalize
    Scalar eps)
{
    int c = blockIdx.x;
    if (c >= cols) return;

    Scalar* col = X + static_cast<size_t>(c) * rows;

    extern __shared__ char smem[];
    Scalar* sdata = reinterpret_cast<Scalar*>(smem);
    int tid = threadIdx.x;

    // Parallel sum-of-squares
    Scalar sum = Scalar(0);
    for (int i = tid; i < rows; i += blockDim.x)
        sum += col[i] * col[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 computes norm and stores to smem[0] as the reciprocal (or 1 if zero)
    if (tid == 0) {
        Scalar nrm = sqrt(sdata[0]);
        d_norms[c] = (nrm > eps) ? nrm : Scalar(0);
        // Store reciprocal for scaling (avoids second __syncthreads for broadcast)
        sdata[0] = (nrm > eps) ? Scalar(1) / nrm : Scalar(1);
    }
    __syncthreads();

    Scalar inv_nrm = sdata[0];
    for (int i = tid; i < rows; i += blockDim.x)
        col[i] *= inv_nrm;
}

/**
 * @brief Inline CSC CV test-MSE kernel — zero materialization, zero PCIe for mask.
 *
 * Iterates ALL nonzeros of A (CSC format, already on device from SparseMatrixGPU).
 * For each nonzero, calls SplitMix64::is_holdout inline — same deterministic hash
 * as the CPU LazySpeckledMask, so results are identical with no data transfer.
 *
 * One GPU thread per column. Inner loop walks the column's nonzeros.
 * Holdout entries accumulate (actual - recon)^2 into a 2-element double buffer:
 *   mse_buf[0] = sum of squared residuals
 *   mse_buf[1] = count of holdout entries
 *
 * Layout: W is m×k col-major (W[i + l*m]), V is n×k col-major (V[j + l*n])
 */
template<typename Scalar>
__global__ void krylov_test_mse_csc_kernel(
    const int* __restrict__ col_ptr,     // CSC col pointer, n+1
    const int* __restrict__ row_idx,     // CSC row indices, nnz
    const Scalar* __restrict__ values,   // CSC values, nnz
    int n,
    const Scalar* __restrict__ W,        // m × k, column-major
    const Scalar* __restrict__ d_vec,    // k singular values
    const Scalar* __restrict__ V,        // n × k, column-major
    int m, int k,
    uint64_t cv_seed, uint64_t inv_prob,
    const Scalar* __restrict__ d_mu,     // row means (m), or null if no centering
    double* __restrict__ mse_buf)        // [2]: sum_sq + count (double atomicAdd)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
        int i = row_idx[p];
        if (!FactorNet::rng::SplitMix64::is_holdout(
                cv_seed, static_cast<uint32_t>(i), static_cast<uint32_t>(j), inv_prob))
            continue;

        Scalar actual = values[p];
        if (d_mu) actual -= d_mu[i];

        Scalar recon = Scalar(0);
        for (int l = 0; l < k; ++l)
            recon += d_vec[l] * W[i + static_cast<size_t>(l) * m]
                              * V[j + static_cast<size_t>(l) * n];

        double diff = static_cast<double>(actual - recon);
        atomicAdd(&mse_buf[0], diff * diff);
        atomicAdd(&mse_buf[1], 1.0);
    }
}

/**
 * @brief Apply L2,1 group sparsity penalty to G diagonal — GPU-side.
 *
 * G[i,i] += lambda / ||W[:,i]||  using pre-computed column squared norms.
 * Replaces the host-side apply_L21(G, Wt, lambda) that required downloading
 * the full m×k W matrix.
 */
template<typename Scalar>
__global__ void apply_L21_diag_gpu_kernel(
    Scalar* __restrict__ G,           // k×k, column-major (lower triangle from SYRK)
    const Scalar* __restrict__ col_sqnorms,  // k: ||W[:,i]||^2
    int k, Scalar lambda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    Scalar norm = sqrt(col_sqnorms[i]);
    if (norm > Scalar(1e-10)) {
        G[i + static_cast<size_t>(i) * k] += lambda / norm;
    }
}

/**
 * @brief Add angular (orthogonality) penalty to Gram matrix G — GPU-side.
 *
 * G += lambda * (WtW - diag(WtW))
 * where WtW[i,j] for i != j is the off-diagonal overlap.
 * The diagonal of WtW is not penalized (only cross-column correlation).
 *
 * WtW is a FULL k×k matrix (result of a separate SYRK on W).
 * We add only the off-diagonal part of WtW (scaled by lambda) to G.
 */
template<typename Scalar>
__global__ void apply_angular_add_gpu_kernel(
    Scalar* __restrict__ G,          // k×k column-major, lower triangle (SYRK output)
    const Scalar* __restrict__ WtW,  // k×k column-major, W^T*W (all entries filled)
    int k, Scalar lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * k) return;
    int col = idx / k;
    int row = idx % k;
    if (row != col) {
        G[idx] += lambda * WtW[idx];
    }
}

}  // namespace detail


// ============================================================================
// Main GPU Krylov Constrained SVD
// ============================================================================

/**
 * @brief GPU Krylov-seeded projected refinement SVD
 *
 * Phase 1: Run GPU Lanczos for unconstrained seed
 * Phase 2: Iterative Gram-solve-then-project refinement using SpMM
 *
 * When no constraints are present, falls through to lanczos_svd_gpu.
 */
template<typename Scalar>
SVDResult<Scalar> krylov_svd_gpu(
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
    const Scalar eps100 = std::numeric_limits<Scalar>::epsilon() * 100;
    const int BLOCK_SIZE = 256;

    // Check for constraints
    const bool has_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.L2_u > 0 || config.L2_v > 0 ||
        config.upper_bound_u > 0 || config.upper_bound_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        config.angular_u > 0 || config.angular_v > 0 ||
        config.has_gram_level_reg();

    // No constraints and no CV → just run GPU Lanczos (optimal)
    if (!has_constraints && !config.is_cv()) {
        return lanczos_svd_gpu<Scalar>(
            h_col_ptr, h_row_idx, h_values, m, n, nnz, config);
    }

    // ==================================================================
    // Shared GPU setup — context, sparse matrix upload, DualCSR, row means.
    // Created ONCE here so Phase 1 (Lanczos seed) and Phase 2 (Krylov
    // refinement) share the same device memory and library handles,
    // eliminating a redundant second PCIe upload + csr2csc transpose.
    // ==================================================================
    GPUContext ctx;
    SparseMatrixGPU<Scalar> d_A(m, n, nnz, h_col_ptr, h_row_idx, h_values);
    DualCSR<Scalar> dual_csr;
    dual_csr.init(ctx, d_A, m, n, nnz);

    DenseVectorS row_means_shared;
    DeviceMemory<Scalar> d_mu_shared;
    double A_norm_sq_shared = 0;
    for (int i = 0; i < nnz; ++i)
        A_norm_sq_shared += static_cast<double>(h_values[i])
                          * static_cast<double>(h_values[i]);
    if (config.center) {
        // Direct raw-pointer loop — no Eigen::Map needed since h_col_ptr/h_row_idx
        // are already accessible as plain arrays of the correct type.
        row_means_shared.resize(m);
        row_means_shared.setZero();
        for (int j = 0; j < n; ++j)
            for (int p = h_col_ptr[j]; p < h_col_ptr[j + 1]; ++p)
                row_means_shared(h_row_idx[p]) += static_cast<Scalar>(h_values[p]);
        if (n > 0) row_means_shared /= static_cast<Scalar>(n);
        A_norm_sq_shared -= static_cast<double>(n)
                          * static_cast<double>(row_means_shared.squaredNorm());
        d_mu_shared = DeviceMemory<Scalar>(m);
        d_mu_shared.upload(row_means_shared.data(), m);
    }

    // ==================================================================
    // Phase 1: GPU Lanczos seed (reuses shared context — no re-upload)
    // ==================================================================
    SVDConfig<Scalar> lanczos_config;
    lanczos_config.k_max = k;
    lanczos_config.tol = config.tol;
    lanczos_config.max_iter = 0;
    lanczos_config.center = config.center;
    lanczos_config.verbose = false;
    lanczos_config.seed = config.seed;
    lanczos_config.threads = config.threads;

    auto t_p1 = std::chrono::high_resolution_clock::now();
    SVDResult<Scalar> seed = lanczos_svd_gpu_impl<Scalar>(
        h_col_ptr, h_row_idx, h_values, m, n, nnz, lanczos_config,
        ctx, dual_csr, d_mu_shared, row_means_shared, A_norm_sq_shared);
    auto t_p1e = std::chrono::high_resolution_clock::now();
    double p1_ms = std::chrono::duration<double, std::milli>(t_p1e - t_p1).count();

    if (config.verbose) {
        FACTORNET_GPU_LOG(true, "Krylov GPU Phase 1 (Lanczos seed): %.1f ms, k=%d\n",
                        p1_ms, seed.k_selected);
    }

    const int k_actual = seed.k_selected;
    if (k_actual == 0) {
        seed.wall_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        return seed;
    }

    // ==================================================================
    // Phase 2: Gram-solve-then-project refinement
    // ==================================================================

    // Host factor matrices (initialized from seed)
    DenseMatrixS W = seed.U.leftCols(k_actual);   // m × k
    DenseMatrixS V = seed.V.leftCols(k_actual);   // n × k
    DenseVectorS d = seed.d.head(k_actual);

    // ctx, d_A, and dual_csr are shared from above — no re-upload needed.

    // Dense device matrices for W (m×k) and V (n×k)
    DeviceMemory<Scalar> d_W(static_cast<size_t>(m) * k_actual);
    DeviceMemory<Scalar> d_V(static_cast<size_t>(n) * k_actual);
    DeviceMemory<Scalar> d_AV(static_cast<size_t>(m) * k_actual);   // A*V result
    DeviceMemory<Scalar> d_AtW(static_cast<size_t>(n) * k_actual);  // A'*W result
    DeviceMemory<Scalar> d_norm_sq(k_actual);  // column norm² for projection

    // cuSPARSE dense descriptors for SpMM
    cusparseDnMatDescr_t dnDescr_V, dnDescr_W, dnDescr_AV, dnDescr_AtW;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_V, n, k_actual, n, d_V.get(),
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_W, m, k_actual, m, d_W.get(),
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_AV, m, k_actual, m, d_AV.get(),
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnDescr_AtW, n, k_actual, n, d_AtW.get(),
        CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

    // SpMM buffer (dual-CSR: both directions use NON_TRANSPOSE gather)
    Scalar spmm_alpha = Scalar(1), spmm_beta = Scalar(0);
    size_t bufSize_fwd = 0, bufSize_trans = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmm_alpha, dual_csr.descr_A, dnDescr_V,
        &spmm_beta, dnDescr_AV,
        CudaDataType<Scalar>::value, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize_fwd));
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmm_alpha, dual_csr.descr_AT, dnDescr_W,
        &spmm_beta, dnDescr_AtW,
        CudaDataType<Scalar>::value, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize_trans));
    // CRITICAL: use SEPARATE preprocess buffers for forward and transpose.
    // Sharing a single buffer causes the second preprocess to overwrite the
    // first's analysis state, corrupting SpMM results for the other direction.
    DeviceMemory<char> d_spmm_buffer_fwd(bufSize_fwd > 0 ? bufSize_fwd : 1);
    DeviceMemory<char> d_spmm_buffer_trans(bufSize_trans > 0 ? bufSize_trans : 1);

    // Preprocess SpMM descriptors (CUDA 12+): pre-analyzes sparsity pattern once,
    // amortizing the cost across all refinement passes.
#if CUSPARSE_VERSION >= 12000
    CUSPARSE_CHECK(cusparseSpMM_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmm_alpha, dual_csr.descr_A, dnDescr_V,
        &spmm_beta, dnDescr_AV,
        CudaDataType<Scalar>::value, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_buffer_fwd.get()));
    CUSPARSE_CHECK(cusparseSpMM_preprocess(
        ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmm_alpha, dual_csr.descr_AT, dnDescr_W,
        &spmm_beta, dnDescr_AtW,
        CudaDataType<Scalar>::value, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_buffer_trans.get()));
#endif

    // PCA centering: row_means and d_mu pre-computed in shared setup above.
    // Aliases to shared variables so all downstream code uses the same names.
    DenseVectorS& row_means = row_means_shared;
    DeviceMemory<Scalar>& d_mu = d_mu_shared;
    DeviceMemory<Scalar> d_colsums(k_actual);
    DeviceMemory<Scalar> d_mu_dots(k_actual);

    // SpMM with PCA correction lambdas (dual-CSR: both use NON_TRANSPOSE gather)
    auto spmm_forward = [&]() {
        // AV = A * V  via NON_TRANSPOSE on CSR(A) [m×n]
        Scalar bz = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMM(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmm_alpha, dual_csr.descr_A, dnDescr_V,
            &bz, dnDescr_AV,
            CudaDataType<Scalar>::value, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_buffer_fwd.get()));
        if (config.center) {
            int smem = BLOCK_SIZE * sizeof(Scalar);
            detail::compute_colsums_kernel<<<k_actual, BLOCK_SIZE, smem, ctx.stream>>>(
                d_V.get(), d_colsums.get(), n, k_actual);
            int total = m * k_actual;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::pca_correct_forward_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_AV.get(), d_mu.get(), d_colsums.get(), m, k_actual);
        }
    };

    auto spmm_transpose = [&]() {
        // AtW = A' * W  via NON_TRANSPOSE on CSR(A^T) [n×m]
        Scalar bz = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMM(
            ctx.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmm_alpha, dual_csr.descr_AT, dnDescr_W,
            &bz, dnDescr_AtW,
            CudaDataType<Scalar>::value, CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_buffer_trans.get()));
        if (config.center) {
            int smem = BLOCK_SIZE * sizeof(Scalar);
            detail::compute_mu_dots_kernel<<<k_actual, BLOCK_SIZE, smem, ctx.stream>>>(
                d_mu.get(), d_W.get(), d_mu_dots.get(), m, k_actual);
            int total = n * k_actual;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::pca_correct_transpose_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                d_AtW.get(), d_mu_dots.get(), n, k_actual);
        }
    };

    // Refinement iterations
    const int max_passes = config.max_iter > 0
        ? config.max_iter
        : std::max(10, static_cast<int>(std::ceil(std::log2(
              static_cast<double>(k_actual)))) * 2 + 3);

    // Working matrices (host — graph-reg fallback only; L21/angular now GPU-side)
    DenseMatrixS G(k_actual, k_actual);  // k×k: only for host graph-reg fallback

    // GPU Gram/Cholesky workspace
    DeviceMemory<Scalar> d_G(static_cast<size_t>(k_actual) * k_actual);
    DeviceMemory<Scalar> d_d(k_actual);   // column norms from normalize (= singular values)
    DeviceMemory<int>    d_potrf_info(1);
    int potrf_lwork = detail::cusolver_potrf_bufferSize<Scalar>(
        ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k_actual, d_G.get(), k_actual);
    DeviceMemory<Scalar> d_potrf_work(potrf_lwork > 0 ? potrf_lwork : 1);

    // GPU workspaces for L21/angular Gram regularization (avoid large W/V downloads)
    //   d_L21_norms: k column norms of W or V — replaces m×k or n×k host download
    //   d_WtW: k×k Gram W^T*W or V^T*V for angular — extra GPU SYRK, no host sync
    const bool need_L21_u     = (config.L21_u > 0);
    const bool need_L21_v     = (config.L21_v > 0);
    const bool need_angular_u = (config.angular_u > 0);
    const bool need_angular_v = (config.angular_v > 0);
    const bool need_graph_u   = (config.graph_u && config.graph_u_lambda > 0);
    const bool need_graph_v   = (config.graph_v && config.graph_v_lambda > 0);
    DeviceMemory<Scalar> d_L21_norms(
        (need_L21_u || need_L21_v) ? static_cast<size_t>(k_actual) : 1);
    DeviceMemory<Scalar> d_WtW(
        (need_angular_u || need_angular_v)
            ? static_cast<size_t>(k_actual) * k_actual : 1);

    // Device-side convergence: ||W - prev_W|| / ||prev_W|| on GPU — 2 scalars only
    const size_t Wk_elems = static_cast<size_t>(m) * k_actual;
    DeviceMemory<Scalar> d_prev_W(Wk_elems);
    DeviceMemory<Scalar> d_conv_tmp(Wk_elems);  // scratch for diff

    // Upload initial factors from Lanczos seed
    d_W.upload(W.data(), static_cast<size_t>(m) * k_actual);
    d_V.upload(V.data(), static_cast<size_t>(n) * k_actual);

    // Gram-level regularization flags (for loop dispatch)
    const bool need_gram_reg_u = need_L21_u || need_angular_u || need_graph_u;
    const bool need_gram_reg_v = need_L21_v || need_angular_v || need_graph_v;

    // ==================================================================
    // CV setup: Phase 2.5 per-rank auto-rank runs on host after convergence
    // using init_test_entries_csc + LazySpeckledMask.
    // ==================================================================
    const bool do_cv = config.is_cv();

    // CV denominator correction — accounts for zeroed holdout fraction in Gram
    Scalar cv_denom_correction = Scalar(1);
    if (do_cv) {
        if (config.mask_zeros) {
            Scalar density = static_cast<Scalar>(nnz) /
                            (static_cast<Scalar>(m) * static_cast<Scalar>(n));
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction) * density;
        } else {
            cv_denom_correction = Scalar(1) - static_cast<Scalar>(config.test_fraction);
        }
        if (config.verbose) {
            FACTORNET_GPU_LOG(true, "  Krylov GPU CV: denom_correction=%.4f\n",
                             static_cast<double>(cv_denom_correction));
        }
    }

    // Phase 2 timing accumulators (verbose only — zero overhead in production)
    double t2_spmm_ms = 0;
    double t2_gram_ms = 0;        // SYRK: G = X'X
    double t2_chol_solve_ms = 0;  // Cholesky + triangular solve
    double t2_normalize_ms = 0;   // column normalization
    double t2_pcie_ms = 0;        // convergence check (device-side)
    auto t2_wall_start = std::chrono::high_resolution_clock::now();
    int actual_passes = 0;

    for (int pass = 0; pass < max_passes; ++pass) {
        ++actual_passes;

        // ==========================================================
        // W-update: W = (A*V) * inv(V'V), fully on device
        // ==========================================================

        // 1. SpMM forward: d_AV = A * d_V (d_V already on device)
        { auto ts = std::chrono::high_resolution_clock::now();
          spmm_forward();
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_spmm_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - ts).count(); }
        }

        // 2. SYRK: d_G = d_V' * d_V  (lower triangle, k×k)
        { auto tg = std::chrono::high_resolution_clock::now();
          detail::cublas_syrk<Scalar>(ctx.cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
              k_actual, n, Scalar(1), d_V.get(), n, Scalar(0), d_G.get(), k_actual);
          // CV correction: scale data-derived Gram BEFORE adding regularization
          if (do_cv)
              detail::cublas_scal<Scalar>(ctx.cublas, k_actual * k_actual,
                  cv_denom_correction, d_G.get());
          { int blk = (k_actual + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::add_to_diagonal_kernel<<<blk, BLOCK_SIZE, 0, ctx.stream>>>(
                d_G.get(), k_actual,
                static_cast<Scalar>(1e-12) + config.L2_u); }
          // Gram-level regularization (GPU-side L21/angular, host-only for graph Laplacian)
          if (need_gram_reg_u) {
              // L21: add lambda/||w_i|| to diagonal using GPU kernel (no W download)
              if (need_L21_u) {
                  detail::column_sqnorm_kernel<<<k_actual, BLOCK_SIZE,
                      BLOCK_SIZE * sizeof(Scalar), ctx.stream>>>(
                      d_W.get(), d_L21_norms.get(), m, k_actual);
                  { int blk = (k_actual + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    detail::apply_L21_diag_gpu_kernel<<<blk, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_G.get(), d_L21_norms.get(), k_actual, config.L21_u); }
              }
              // Angular: G += lambda * lower-tri(W^T*W) via SYRK + kernel (no W download)
              if (need_angular_u) {
                  detail::cublas_syrk<Scalar>(ctx.cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                      k_actual, m, Scalar(1), d_W.get(), m, Scalar(0), d_WtW.get(), k_actual);
                  { int blk2d = (k_actual + 15) / 16;
                    dim3 gb(blk2d, blk2d), tb(16, 16);
                    detail::apply_angular_add_gpu_kernel<<<gb, tb, 0, ctx.stream>>>(
                        d_G.get(), d_WtW.get(), k_actual, config.angular_u); }
              }
              // Graph Laplacian: unavoidable host roundtrip (downloads G [k×k] + W [m×k])
              if (need_graph_u) {
                  d_G.download(G.data(), static_cast<size_t>(k_actual) * k_actual);
                  d_W.download(W.data(), static_cast<size_t>(m) * k_actual);
                  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
                  for (int i = 0; i < k_actual; ++i)
                      for (int j = i+1; j < k_actual; ++j) G(i,j) = G(j,i);
                  DenseMatrixS Wt = W.transpose();
                  features::apply_graph_reg(G, *config.graph_u, Wt, config.graph_u_lambda);
                  d_G.upload(G.data(), static_cast<size_t>(k_actual) * k_actual);
              }
          }
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_gram_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - tg).count(); }
        }

        // 3. Cholesky + TRSM solve: W = AV * inv(G), on device
        { auto tc = std::chrono::high_resolution_clock::now();
          detail::cusolver_potrf<Scalar>(ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k_actual,
              d_G.get(), k_actual, d_potrf_work.get(), potrf_lwork, d_potrf_info.get());
          // Copy d_AV → d_W (TRSM solves in-place)
          CUDA_CHECK(cudaMemcpyAsync(d_W.get(), d_AV.get(),
              static_cast<size_t>(m) * k_actual * sizeof(Scalar),
              cudaMemcpyDeviceToDevice, ctx.stream));
          // Solve W * G = AV via: W * L * L' = AV  →  two right-side TRSM
          detail::cublas_trsm<Scalar>(ctx.cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
              CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
              m, k_actual, Scalar(1), d_G.get(), k_actual, d_W.get(), m);
          detail::cublas_trsm<Scalar>(ctx.cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
              CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
              m, k_actual, Scalar(1), d_G.get(), k_actual, d_W.get(), m);
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_chol_solve_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - tc).count(); }
        }

        // 4. Constraint projection + column normalize (GPU)
        { auto tn = std::chrono::high_resolution_clock::now();
          if (config.L1_u > 0 || config.nonneg_u || config.upper_bound_u > 0) {
              detail::column_sqnorm_kernel<<<k_actual, BLOCK_SIZE,
                  BLOCK_SIZE * sizeof(Scalar), ctx.stream>>>(
                  d_V.get(), d_norm_sq.get(), n, k_actual);
              // CV correction: scale norm_sq to match corrected Gram
              if (do_cv)
                  detail::cublas_scal<Scalar>(ctx.cublas, k_actual,
                      cv_denom_correction, d_norm_sq.get());
              int total = m * k_actual;
              int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
              detail::project_constraints_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                  d_W.get(), m, k_actual,
                  config.L1_u, config.nonneg_u, config.upper_bound_u,
                  d_norm_sq.get());
          }
          detail::normalize_columns_kernel<<<k_actual, BLOCK_SIZE,
              BLOCK_SIZE * sizeof(Scalar), ctx.stream>>>(
              d_W.get(), m, k_actual, d_d.get(), eps100);
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_normalize_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - tn).count(); }
        }

        // ==========================================================
        // V-update: V = (A'*W) * inv(W'W), fully on device
        // ==========================================================

        // 1. SpMM transpose: d_AtW = A' * d_W (d_W on device from W-update)
        { auto ts = std::chrono::high_resolution_clock::now();
          spmm_transpose();
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_spmm_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - ts).count(); }
        }

        // 2. SYRK: d_G = d_W' * d_W  (lower triangle, k×k)
        { auto tg = std::chrono::high_resolution_clock::now();
          detail::cublas_syrk<Scalar>(ctx.cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
              k_actual, m, Scalar(1), d_W.get(), m, Scalar(0), d_G.get(), k_actual);
          // CV correction: scale data-derived Gram BEFORE adding regularization
          if (do_cv)
              detail::cublas_scal<Scalar>(ctx.cublas, k_actual * k_actual,
                  cv_denom_correction, d_G.get());
          { int blk = (k_actual + BLOCK_SIZE - 1) / BLOCK_SIZE;
            detail::add_to_diagonal_kernel<<<blk, BLOCK_SIZE, 0, ctx.stream>>>(
                d_G.get(), k_actual,
                static_cast<Scalar>(1e-12) + config.L2_v); }
          if (need_gram_reg_v) {
              if (need_L21_v) {
                  detail::column_sqnorm_kernel<<<k_actual, BLOCK_SIZE,
                      BLOCK_SIZE * sizeof(Scalar), ctx.stream>>>(
                      d_V.get(), d_L21_norms.get(), n, k_actual);
                  { int blk = (k_actual + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    detail::apply_L21_diag_gpu_kernel<<<blk, BLOCK_SIZE, 0, ctx.stream>>>(
                        d_G.get(), d_L21_norms.get(), k_actual, config.L21_v); }
              }
              if (need_angular_v) {
                  detail::cublas_syrk<Scalar>(ctx.cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                      k_actual, n, Scalar(1), d_V.get(), n, Scalar(0), d_WtW.get(), k_actual);
                  { int blk2d = (k_actual + 15) / 16;
                    dim3 gb(blk2d, blk2d), tb(16, 16);
                    detail::apply_angular_add_gpu_kernel<<<gb, tb, 0, ctx.stream>>>(
                        d_G.get(), d_WtW.get(), k_actual, config.angular_v); }
              }
              if (need_graph_v) {
                  d_G.download(G.data(), static_cast<size_t>(k_actual) * k_actual);
                  d_V.download(V.data(), static_cast<size_t>(n) * k_actual);
                  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
                  for (int i = 0; i < k_actual; ++i)
                      for (int j = i+1; j < k_actual; ++j) G(i,j) = G(j,i);
                  DenseMatrixS Vt = V.transpose();
                  features::apply_graph_reg(G, *config.graph_v, Vt, config.graph_v_lambda);
                  d_G.upload(G.data(), static_cast<size_t>(k_actual) * k_actual);
              }
          }
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_gram_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - tg).count(); }
        }

        // 3. Cholesky + TRSM solve: V = AtW * inv(G), on device
        { auto tc = std::chrono::high_resolution_clock::now();
          detail::cusolver_potrf<Scalar>(ctx.cusolver, CUBLAS_FILL_MODE_LOWER, k_actual,
              d_G.get(), k_actual, d_potrf_work.get(), potrf_lwork, d_potrf_info.get());
          CUDA_CHECK(cudaMemcpyAsync(d_V.get(), d_AtW.get(),
              static_cast<size_t>(n) * k_actual * sizeof(Scalar),
              cudaMemcpyDeviceToDevice, ctx.stream));
          detail::cublas_trsm<Scalar>(ctx.cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
              CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
              n, k_actual, Scalar(1), d_G.get(), k_actual, d_V.get(), n);
          detail::cublas_trsm<Scalar>(ctx.cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
              CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
              n, k_actual, Scalar(1), d_G.get(), k_actual, d_V.get(), n);
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_chol_solve_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - tc).count(); }
        }

        // 4. Constraint projection + column normalize (GPU)
        { auto tn = std::chrono::high_resolution_clock::now();
          if (config.L1_v > 0 || config.nonneg_v || config.upper_bound_v > 0) {
              detail::column_sqnorm_kernel<<<k_actual, BLOCK_SIZE,
                  BLOCK_SIZE * sizeof(Scalar), ctx.stream>>>(
                  d_W.get(), d_norm_sq.get(), m, k_actual);
              // CV correction: scale norm_sq to match corrected Gram
              if (do_cv)
                  detail::cublas_scal<Scalar>(ctx.cublas, k_actual,
                      cv_denom_correction, d_norm_sq.get());
              int total = n * k_actual;
              int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
              detail::project_constraints_kernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                  d_V.get(), n, k_actual,
                  config.L1_v, config.nonneg_v, config.upper_bound_v,
                  d_norm_sq.get());
          }
          detail::normalize_columns_kernel<<<k_actual, BLOCK_SIZE,
              BLOCK_SIZE * sizeof(Scalar), ctx.stream>>>(
              d_V.get(), n, k_actual, d_d.get(), eps100);
          if (config.verbose) { CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
              t2_normalize_ms += std::chrono::duration<double,
                  std::milli>(std::chrono::high_resolution_clock::now() - tn).count(); }
        }

        // Convergence check — entirely on device.
        // Compute ||W - prev_W|| / ||prev_W|| using cuBLAS:
        //   1. d_conv_tmp = d_W  (D2D copy)
        //   2. d_conv_tmp -= d_prev_W  (axpy: alpha=-1)
        //   3. diff_norm  = nrm2(d_conv_tmp)
        //   4. prev_norm  = nrm2(d_prev_W)
        //   5. dW = diff_norm / (prev_norm + eps)
        // Then download only 2 scalars instead of m×k matrix.
        { auto tp = std::chrono::high_resolution_clock::now();
          if (pass > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_conv_tmp.get(), d_W.get(),
                Wk_elems * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
            Scalar neg1 = Scalar(-1);
            detail::cublas_axpy<Scalar>(ctx.cublas,
                static_cast<int>(Wk_elems), neg1, d_prev_W.get(), d_conv_tmp.get());
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            Scalar diff_nrm = detail::cublas_nrm2<Scalar>(
                ctx.cublas, static_cast<int>(Wk_elems), d_conv_tmp.get());
            Scalar prev_nrm = detail::cublas_nrm2<Scalar>(
                ctx.cublas, static_cast<int>(Wk_elems), d_prev_W.get());
            Scalar dW = diff_nrm / (prev_nrm + eps100);
            if (config.verbose) {
                FACTORNET_GPU_LOG(true, "  Krylov GPU pass %d: dW=%.2e\n",
                                pass + 1, static_cast<double>(dW));
            }
            if (dW < config.tol) {
                if (config.verbose) t2_pcie_ms += std::chrono::duration<double,
                    std::milli>(std::chrono::high_resolution_clock::now() - tp).count();
                break;
            }
          }
          // Save current W as prev_W for next pass (D2D copy, no host involved)
          CUDA_CHECK(cudaMemcpyAsync(d_prev_W.get(), d_W.get(),
              Wk_elems * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
          if (config.verbose) t2_pcie_ms += std::chrono::duration<double,
              std::milli>(std::chrono::high_resolution_clock::now() - tp).count();
        }
    }

    // Download final W, V and d from device
    d_W.download(W.data(), static_cast<size_t>(m) * k_actual);
    d_V.download(V.data(), static_cast<size_t>(n) * k_actual);
    d_d.download(d.data(), k_actual);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Phase 2 verbose timing summary
    if (config.verbose) {
        double t2_wall = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t2_wall_start).count();
        FACTORNET_GPU_LOG(true,
            "Krylov GPU Phase 2: %.1f ms over %d passes\n"
            "  SpMM (GPU):    %6.1f ms (%4.1f%%)\n"
            "  SYRK+reg:      %6.1f ms (%4.1f%%)\n"
            "  Potrf+TRSM:    %6.1f ms (%4.1f%%)\n"
            "  Constr+Norm:   %6.1f ms (%4.1f%%)\n"
            "  Convergence:   %6.1f ms (%4.1f%%)\n"
            "  Other:         %6.1f ms (%4.1f%%)\n",
            t2_wall, actual_passes,
            t2_spmm_ms, t2_wall > 0 ? 100.0 * t2_spmm_ms / t2_wall : 0.0,
            t2_gram_ms, t2_wall > 0 ? 100.0 * t2_gram_ms / t2_wall : 0.0,
            t2_chol_solve_ms, t2_wall > 0 ? 100.0 * t2_chol_solve_ms / t2_wall : 0.0,
            t2_normalize_ms, t2_wall > 0 ? 100.0 * t2_normalize_ms / t2_wall : 0.0,
            t2_pcie_ms, t2_wall > 0 ? 100.0 * t2_pcie_ms / t2_wall : 0.0,
            t2_wall - t2_spmm_ms - t2_gram_ms - t2_chol_solve_ms - t2_normalize_ms - t2_pcie_ms,
            t2_wall > 0 ? 100.0 * (t2_wall - t2_spmm_ms - t2_gram_ms - t2_chol_solve_ms - t2_normalize_ms - t2_pcie_ms) / t2_wall : 0.0);
    }

    // ==================================================================
    // Cleanup cuSPARSE descriptors
    // ==================================================================
    // dual_csr destructor handles descr_A and descr_AT cleanup
    CUSPARSE_CHECK(cusparseDestroyDnMat(dnDescr_V));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dnDescr_W));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dnDescr_AV));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dnDescr_AtW));

    // ==================================================================
    // Build result — sort by decreasing d
    // ==================================================================
    SVDResult<Scalar> result;
    result.centered = config.center;
    if (config.center) result.row_means = row_means.size() > 0 ? row_means : seed.row_means;
    result.frobenius_norm_sq = seed.frobenius_norm_sq;

    std::vector<int> order(k_actual);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return d(a) > d(b); });

    // ==================================================================
    // Phase 2.5: Per-rank CV auto-rank selection (host-side)
    // ==================================================================
    int best_k = k_actual;
    if (do_cv) {
        const DenseVectorS* rm_ptr = (config.center && row_means.size() > 0)
                                     ? &row_means : nullptr;
        FactorNet::nmf::LazySpeckledMask<Scalar> mask(
            m, n, static_cast<int64_t>(nnz),
            static_cast<double>(config.test_fraction),
            config.effective_cv_seed(), config.mask_zeros);
        auto test_entries = FactorNet::svd::init_test_entries_csc<Scalar>(
            h_col_ptr, h_row_idx, h_values,
            m, n, static_cast<int64_t>(nnz), mask, rm_ptr);

        Scalar best_mse = std::numeric_limits<Scalar>::max();
        int patience_counter = 0;
        for (int kk = 0; kk < k_actual; ++kk) {
            int idx = order[kk];
            test_entries.update(W.col(idx), d(idx), V.col(idx), 0);
            Scalar mse = test_entries.compute_mse(0);
            result.test_loss_trajectory.push_back(mse);
            if (mse < best_mse) {
                best_mse = mse;
                best_k = kk + 1;
                patience_counter = 0;
            } else if (++patience_counter >= config.patience) {
                break;
            }
        }
    }

    result.k_selected = best_k;
    result.d.resize(best_k);
    result.U.resize(m, best_k);
    result.V.resize(n, best_k);
    for (int i = 0; i < best_k; ++i) {
        result.d(i) = d(order[i]);
        result.U.col(i) = W.col(order[i]);
        result.V.col(i) = V.col(order[i]);
    }

    result.wall_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_start).count();

    if (config.verbose) {
        FACTORNET_GPU_LOG(true, "Krylov GPU done: %.1f ms (P1: %.1f ms), k_selected=%d/%d\n",
                        result.wall_time_ms, p1_ms, best_k, k_actual);
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet
