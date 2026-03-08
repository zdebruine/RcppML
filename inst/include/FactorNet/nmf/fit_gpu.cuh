// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit_gpu.cuh
 * @brief GPU NMF with full feature support via unified architecture
 *
 * This is the GPU counterpart to fit_unified.hpp. It runs the NMF iteration
 * loop on GPU using the existing CUDA kernels (gram, rhs, nnls_batch, loss),
 * but adds Tier 1/2 feature application by downloading the small k×k Gram
 * matrix G to host, applying features on CPU, then uploading back.
 *
 * Data residency:
 *   - A (sparse): uploaded once at start, stays on device
 *   - W (k×m), H (k×n): allocated on device, stay resident
 *   - G (k×k): computed on device, downloaded for features, uploaded back
 *   - B (k×n or k×m): stays on device for standard features
 *     (downloaded only for Tier 2 features that touch full factors)
 *   - d (k): stays on device, downloaded at end
 *   - Final W, H, d: downloaded once at end
 *
 * Features supported (all of them — parity with CPU by construction):
 *   Tier 1 (on k×k G, no data movement): L1, L2, ortho
 *   Tier 2 (download factor to host): graph_reg, L21, bounds
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

// GPU kernel implementations
#include <FactorNet/gpu/types.cuh>
#include <FactorNet/gpu/gram.cuh>
#include <FactorNet/gpu/rhs.cuh>
#include <FactorNet/gpu/nnls.cuh>
#include <FactorNet/gpu/loss.cuh>
#include <FactorNet/gpu/dispersion.cuh>
#include <FactorNet/gpu/cv_delta.cuh>
#include <FactorNet/gpu/fused_cv.cuh>
// mixed_precision.cuh removed — RHS now uses cuSPARSE SpMM (see gpu/rhs.cuh)

// GPU Cholesky NNLS primitives
#include <FactorNet/primitives/gpu/cholesky_nnls.cuh>

// Batch CD + PGD GEMM NNLS + GPU L1 norm
#include <FactorNet/gpu/batch_nnls.cuh>


// Unified primitive wrappers
#include <FactorNet/primitives/gpu/context.cuh>

// Unified types
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/core/constants.hpp>

// Features (backend-agnostic — operate on host Eigen matrices)
#include <FactorNet/features/sparsity.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>
#include <FactorNet/gpu/graph_reg_gpu.cuh>

// Shared algorithm variant dispatch and helpers
#include <FactorNet/nmf/variant_helpers.hpp>
#include <FactorNet/nmf/gpu_features.cuh>

// IRLS for non-MSE losses
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>    // host-mediated fallback
#include <FactorNet/primitives/gpu/nnls_batch_irls.cuh>    // GPU-native IRLS
#include <FactorNet/primitives/gpu/nnls_batch_zi_irls.cuh> // GPU-native ZI IRLS

// CPU masked NNLS (host-mediated via CPU solver)
#include <FactorNet/nmf/fit_cpu.hpp>

// GPU profiling timer
#include <FactorNet/profiling/gpu_timer.cuh>

// RNG
#include <FactorNet/rng/rng.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <memory>
#include <vector>

namespace FactorNet {
namespace nmf {


namespace gpu_detail {

// ========================================================================
// Helper: explicit loss for non-MSE losses (host-side, raw CSC)
// ========================================================================

/**
 * @brief Compute explicit per-element loss over sparse nonzeros.
 *
 * Used by host-mediated IRLS path when loss != MSE.
 * For each nonzero A(i,j): predicted = W_Td.col(i) · H.col(j),
 * then sums loss(A(i,j), predicted).
 */
template<typename Scalar>
inline Scalar explicit_loss_raw_csc(
    const int* col_ptr, const int* row_idx, const Scalar* values,
    int n,
    const DenseMatrix<Scalar>& W_Td,  // k × m (with scaling absorbed)
    const DenseMatrix<Scalar>& H,     // k × n
    const LossConfig<Scalar>& loss_config,
    int num_threads = 1,
    const Scalar* disp_ptr = nullptr)  // per-row dispersion (GP theta / NB r)
{
    const int k = static_cast<int>(W_Td.rows());
    Scalar total_loss = 0;

    #pragma omp parallel for reduction(+:total_loss) num_threads(num_threads) \
                             schedule(dynamic, 64) if(num_threads > 1)
    for (int j = 0; j < n; ++j) {
        for (int idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
            int row = row_idx[idx];
            Scalar val = values[idx];
            Scalar predicted = 0;
            for (int f = 0; f < k; ++f)
                predicted += W_Td(f, row) * H(f, j);
            Scalar theta = disp_ptr ? disp_ptr[row] : Scalar(0);
            total_loss += compute_loss(val, predicted, loss_config, theta);
        }
    }
    return total_loss;
}

// ========================================================================
// Helper: GP theta update (host-mediated, MM quadratic per row)
// ========================================================================

/**
 * @brief Update GP dispersion theta per row using MM approach.
 *
 * Operates on host CSC arrays + Eigen factor matrices.
 * theta_vec is updated in-place. Uses fp64 accumulation for stability.
 */
template<typename Scalar>
inline void gp_theta_update_host(
    const int* col_ptr, const int* row_idx, const Scalar* values,
    int m, int n, int nnz,
    const DenseMatrix<Scalar>& W_Td,  // k × m (d-scaled)
    const DenseMatrix<Scalar>& H,     // k × n
    Scalar* theta_vec,                // m-length, updated in-place
    Scalar theta_max,
    DispersionMode mode,
    int num_threads = 1)
{
    const int k = static_cast<int>(W_Td.rows());
    constexpr int THETA_INNER_ITERS = 5;

    // Pre-compute total mu per row via Gram trick: sum_s_i = w_i · (H · 1)
    DenseVector<Scalar> h_rowsum = H.rowwise().sum();

    // Cache nonzero entries for inner iterations
    struct NzEntry { int row; double y; double s; };
    std::vector<NzEntry> nz_cache;
    nz_cache.reserve(nnz);

    Eigen::VectorXd sum_y_d = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd sum_s_d = Eigen::VectorXd::Zero(m);
    Eigen::VectorXi n_nz = Eigen::VectorXi::Zero(m);

    // Total mu per row via Gram trick
    for (int i = 0; i < m; ++i)
        sum_s_d(i) = static_cast<double>(W_Td.col(i).dot(h_rowsum));

    // Single pass over nonzeros
    for (int j = 0; j < n; ++j) {
        for (int idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
            int i = row_idx[idx];
            double y = static_cast<double>(values[idx]);
            double s = std::max(static_cast<double>(W_Td.col(i).dot(H.col(j))), 1e-10);
            sum_y_d(i) += y;
            if (y >= 1.0) n_nz(i)++;
            nz_cache.push_back({i, y, s});
        }
    }

    // Inner MM iterations
    double cap = static_cast<double>(theta_max);
    for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS; ++mm_iter) {
        Eigen::VectorXd alpha_d = Eigen::VectorXd::Zero(m);
        Eigen::VectorXd gamma_d = Eigen::VectorXd::Zero(m);

        for (const auto& nz : nz_cache) {
            if (nz.y >= 1.0) {
                double th = static_cast<double>(theta_vec[nz.row]);
                double denom = std::max(nz.s + th * nz.y, 1e-10);
                double eta1 = nz.s / denom;
                alpha_d(nz.row) += (nz.y - 1.0) * eta1;
                gamma_d(nz.row) += (nz.y - 1.0) * (1.0 - eta1);
            }
        }

        for (int i = 0; i < m; ++i) {
            double alpha_i = alpha_d(i) + static_cast<double>(n_nz(i));
            double beta_i = (sum_y_d(i) - sum_s_d(i)) - gamma_d(i) + alpha_i;
            if (alpha_i > 1e-15) {
                double disc = beta_i * beta_i + 4.0 * alpha_i * gamma_d(i);
                if (disc > 0.0 && std::isfinite(disc)) {
                    double new_theta = (-beta_i + std::sqrt(disc)) / (2.0 * alpha_i);
                    if (std::isfinite(new_theta) && new_theta >= 0.0)
                        theta_vec[i] = static_cast<Scalar>(std::min(new_theta, cap));
                }
            }
        }
    }

    if (mode == DispersionMode::GLOBAL) {
        Scalar sum = 0;
        for (int i = 0; i < m; ++i) sum += theta_vec[i];
        Scalar mean_theta = sum / static_cast<Scalar>(m);
        for (int i = 0; i < m; ++i) theta_vec[i] = mean_theta;
    }
}

// ========================================================================
// Helper: NB dispersion (r) update (host-mediated, method of moments)
// ========================================================================

/**
 * @brief Update NB size parameter r per row using method of moments.
 *
 * r_i = Σ_j μ_ij² / max(Σ_j [(y_ij - μ_ij)² - μ_ij], ε)
 * Uses Gram trick for total sums over all entries (including zeros).
 */
template<typename Scalar>
inline void nb_size_update_host(
    const int* col_ptr, const int* row_idx, const Scalar* values,
    int m, int n, int k,
    const DenseMatrix<Scalar>& W_Td,  // k × m (d-scaled)
    const DenseMatrix<Scalar>& H,     // k × n
    Scalar* nb_size_vec,              // m-length, updated in-place
    Scalar r_min, Scalar r_max,
    DispersionMode mode,
    int num_threads = 1)
{
    double r_min_d = static_cast<double>(r_min);
    double r_max_d = static_cast<double>(r_max);

    // Per-row accumulators from nonzero entries
    Eigen::VectorXd sum_mu_nz = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd sum_mu_sq_nz = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd sum_resid_sq_nz = Eigen::VectorXd::Zero(m);

    for (int j = 0; j < n; ++j) {
        for (int idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
            int i = row_idx[idx];
            double y = static_cast<double>(values[idx]);
            double mu = 0;
            for (int f = 0; f < k; ++f)
                mu += static_cast<double>(W_Td(f, i)) * static_cast<double>(H(f, j));
            mu = std::max(mu, 1e-10);
            double resid = y - mu;
            sum_mu_nz(i) += mu;
            sum_mu_sq_nz(i) += mu * mu;
            sum_resid_sq_nz(i) += resid * resid;
        }
    }

    // Gram trick for total sums
    DenseVector<Scalar> h_rs = H.rowwise().sum();
    DenseMatrix<Scalar> G_H = H * H.transpose();  // k × k

    for (int i = 0; i < m; ++i) {
        double total_mu = static_cast<double>(W_Td.col(i).dot(h_rs));
        double total_mu_sq = 0.0;
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                total_mu_sq += static_cast<double>(W_Td(a, i)) *
                               static_cast<double>(G_H(a, b)) *
                               static_cast<double>(W_Td(b, i));

        double total_resid_sq = sum_resid_sq_nz(i) + (total_mu_sq - sum_mu_sq_nz(i));
        double excess_var = total_resid_sq - total_mu;

        if (excess_var > 1e-10 && total_mu_sq > 1e-10) {
            double r_new = total_mu_sq / excess_var;
            r_new = std::max(r_min_d, std::min(r_new, r_max_d));
            if (std::isfinite(r_new))
                nb_size_vec[i] = static_cast<Scalar>(r_new);
        } else {
            nb_size_vec[i] = r_max;
        }
    }

    if (mode == DispersionMode::GLOBAL) {
        std::vector<Scalar> r_vals(nb_size_vec, nb_size_vec + m);
        std::nth_element(r_vals.begin(), r_vals.begin() + m / 2, r_vals.end());
        Scalar median_r = r_vals[m / 2];
        for (int i = 0; i < m; ++i) nb_size_vec[i] = median_r;
    }
}

/**
 * @brief Host-side Gamma/IG/Tweedie dispersion (φ) update — MoM Pearson estimator.
 *
 * φ̂_i = (1/n_i) Σ_j (y_ij − μ_ij)² / μ_ij^p
 * where p = 2 (Gamma), 3 (InvGauss), or power_param (Tweedie).
 */
template<typename Scalar>
inline void phi_update_host(
    const int* col_ptr, const int* row_idx, const Scalar* values,
    int m, int n,
    const DenseMatrix<Scalar>& W_Td,  // k × m (d-scaled)
    const DenseMatrix<Scalar>& H,     // k × n
    Scalar* phi_vec,                  // m-length, updated in-place
    double var_power,
    Scalar phi_min, Scalar phi_max,
    DispersionMode mode)
{
    const int k = static_cast<int>(W_Td.rows());
    double phi_min_d = static_cast<double>(phi_min);
    double phi_max_d = static_cast<double>(phi_max);

    Eigen::VectorXd sum_pearson_sq = Eigen::VectorXd::Zero(m);
    Eigen::VectorXi count_vec = Eigen::VectorXi::Zero(m);

    for (int j = 0; j < n; ++j) {
        for (int idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
            int i = row_idx[idx];
            double y = static_cast<double>(values[idx]);
            if (y <= 0.0) continue;
            double mu = 0;
            for (int f = 0; f < k; ++f)
                mu += static_cast<double>(W_Td(f, i)) * static_cast<double>(H(f, j));
            mu = std::max(mu, 1e-10);
            double resid = y - mu;
            double v_mu = std::pow(mu, var_power);
            sum_pearson_sq(i) += (resid * resid) / std::max(v_mu, 1e-20);
            count_vec(i)++;
        }
    }

    for (int i = 0; i < m; ++i) {
        if (count_vec(i) > 0) {
            double phi_new = sum_pearson_sq(i) / static_cast<double>(count_vec(i));
            phi_new = std::max(phi_min_d, std::min(phi_new, phi_max_d));
            if (std::isfinite(phi_new))
                phi_vec[i] = static_cast<Scalar>(phi_new);
        }
    }

    if (mode == DispersionMode::GLOBAL) {
        std::vector<Scalar> phi_vals(phi_vec, phi_vec + m);
        std::nth_element(phi_vals.begin(), phi_vals.begin() + m / 2, phi_vals.end());
        Scalar median_phi = phi_vals[m / 2];
        for (int i = 0; i < m; ++i) phi_vec[i] = median_phi;
    }
}

// ========================================================================
// Helper: check if any Tier 2 features are active
// ========================================================================

template<typename Scalar>
inline bool has_tier2_features(const ::FactorNet::NMFConfig<Scalar>& config) {
    return (config.W.graph != nullptr)
        || (config.H.graph != nullptr)
        || (config.W.L21 > 0)
        || (config.H.L21 > 0)
        || (config.W.upper_bound > 0)
        || (config.H.upper_bound > 0);
}

// ========================================================================
// Helper: download DenseMatrixGPU to Eigen and upload back
// ========================================================================

template<typename Scalar>
inline void download_to_eigen(const FactorNet::gpu::DenseMatrixGPU<Scalar>& d_mat,
                              DenseMatrix<Scalar>& h_mat) {
    h_mat.resize(d_mat.rows, d_mat.cols);
    d_mat.download_to(h_mat.data());
}

template<typename Scalar>
inline void upload_from_eigen(FactorNet::gpu::DenseMatrixGPU<Scalar>& d_mat,
                              const DenseMatrix<Scalar>& h_mat) {
    d_mat.upload_from(h_mat.data());
}

}  // namespace gpu_detail

// ========================================================================
// GPU NMF fit — full feature support
// ========================================================================

/**
 * @brief Internal implementation of GPU NMF with all features.
 *
 * Accepts a pre-constructed SparseMatrixGPU (on device) plus host CSC
 * arrays for host-mediated paths (IRLS, masking, Eigen::Map access).
 */
template<typename Scalar>
::FactorNet::NMFResult<Scalar> nmf_fit_gpu_impl(
    ::FactorNet::gpu::SparseMatrixGPU<Scalar>& d_A,
    const int* A_csc_ptr,
    const int* A_row_idx,
    const Scalar* A_values,
    int m, int n, int nnz,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    using namespace FactorNet::gpu;

    const int k = config.rank;

    // GPU context and guard (RAII)
    GPUContext ctx;
    primitives::gpu_detail::GPUGuard guard;

    // ------------------------------------------------------------------
    // 3. Initialize factors on host, then upload to device
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> h_W_T, h_H;  // host copies for init / features
    DenseVector<Scalar> h_d;

    if (W_init && H_init) {
        h_W_T = W_init->transpose();  // m×k → k×m
        h_H = *H_init;
        h_d = DenseVector<Scalar>::Ones(k);
    } else if (config.init_mode == 1) {
        // Lanczos SVD initialization on host using CSC arrays
        using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
        Eigen::Map<const SpMat> h_A_map(m, n, nnz, A_csc_ptr, A_row_idx, A_values);
        detail::initialize_lanczos(h_W_T, h_H, h_d, h_A_map, k, m, n,
                                   config.seed, config.threads, config.verbose);
    } else {
        // Random initialization
        rng::SplitMix64 rng(config.seed);
        h_W_T.resize(k, m);
        rng.fill_uniform(h_W_T.data(), k, m);
        h_H.resize(k, n);
        rng.fill_uniform(h_H.data(), k, n);
        h_d = DenseVector<Scalar>::Ones(k);
    }

    // Upload to device
    DenseMatrixGPU<Scalar> d_W(k, m, h_W_T.data());
    DenseMatrixGPU<Scalar> d_H(k, n, h_H.data());
    DeviceMemory<Scalar> d_d(k);
    d_d.upload(h_d.data(), k);

    // ------------------------------------------------------------------
    // 4. Allocate GPU workspace
    // ------------------------------------------------------------------

    DenseMatrixGPU<Scalar> d_G(k, k);                    // Gram
    DenseMatrixGPU<Scalar> d_B(k, std::max(m, n));       // RHS
    DenseMatrixGPU<Scalar> d_B_save(k, n);               // Saved RHS for loss
    DenseMatrixGPU<Scalar> d_Hd(k, n);                   // d-scaled H for W update
    DeviceMemory<double> d_scratch(4);                    // Loss reduction (double: avoids float overflow)

    // d-scaled W for GPU-side non-MSE loss computation
    DenseMatrixGPU<Scalar> d_W_Td_loss;
    if (config.requires_irls()) d_W_Td_loss = DenseMatrixGPU<Scalar>(k, m);

    // ------------------------------------------------------------------
    // 4b. cuSPARSE SpMM handles for RHS (created once, reused per iter)
    // ------------------------------------------------------------------
    //
    // Forward (H-update): B(k×n) = W(k×m) · A(m×n)
    //   Uses A's CSC as CSR of A^T (n×m)
    //
    // Backward (W-update): B(k×m) = H(k×n) · A^T(n×m)
    //   Uses A^T's CSC as CSR of A (m×n)
    //   A^T CSC is built on host and uploaded once.

    // Build and upload A^T CSC for the W-update SpMM handle
    std::vector<int> h_At_col_ptr, h_At_row_idx;
    std::vector<Scalar> h_At_values;
    build_transpose_csc_host(m, n, nnz, A_csc_ptr, A_row_idx, A_values,
                             h_At_col_ptr, h_At_row_idx, h_At_values);
    SparseMatrixGPU<Scalar> d_A_T(n, m, nnz,
        h_At_col_ptr.data(), h_At_row_idx.data(), h_At_values.data());

    // Forward SpMM: B = W · A
    SpMMHandle<Scalar> spmm_fwd;
    spmm_fwd.init(ctx, d_A, k, /*csc_cols=*/n, /*csc_rows=*/m,
                  d_W.data.get(), d_B.data.get());

    // Backward SpMM: B = H · A^T
    SpMMHandle<Scalar> spmm_bwd;
    spmm_bwd.init(ctx, d_A_T, k, /*csc_cols=*/m, /*csc_rows=*/n,
                  d_H.data.get(), d_B.data.get());

    if (config.verbose) {
        FACTORNET_GPU_LOG(true, "[SpMM] cuSPARSE handles initialized (fwd: buf=%zuB, bwd: buf=%zuB)\n",
            spmm_fwd.buffer_size, spmm_bwd.buffer_size);
    }

    // ------------------------------------------------------------------
    // 4c. GPU graph regularization handles (cuSPARSE SpMM + cuBLAS GEMM)
    // ------------------------------------------------------------------
    GpuGraphRegHandle<Scalar> graph_reg_H, graph_reg_W;
    if (config.H.graph != nullptr) {
        const auto& L = *config.H.graph;
        graph_reg_H.init(ctx, L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
                         static_cast<int>(L.rows()), static_cast<int>(L.nonZeros()),
                         k, d_H.data.get());
    }
    if (config.W.graph != nullptr) {
        const auto& L = *config.W.graph;
        graph_reg_W.init(ctx, L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
                         static_cast<int>(L.rows()), static_cast<int>(L.nonZeros()),
                         k, d_W.data.get());
    }

    // Host buffers for feature application (only used when features are active)
    DenseMatrix<Scalar> h_G(k, k);    // Downloaded G for features

    // ------------------------------------------------------------------
    // 5. IRLS pre-computation (host-mediated non-MSE loss support)
    // ------------------------------------------------------------------

    const bool use_irls = config.requires_irls();
    const bool use_mask = config.has_mask();
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
    SpMat h_At_sparse;  // A^T for W-update IRLS / mask
    SpMat h_A_sparse;   // A for H-update mask

    if (use_irls || use_mask) {
        // Pre-compute A and A^T on host for IRLS / masked NNLS
        Eigen::Map<const SpMat> h_A_map(m, n, nnz, A_csc_ptr, A_row_idx, A_values);
        h_A_sparse = h_A_map;  // copy Map → SparseMatrix
        h_At_sparse = h_A_map.transpose();
        h_At_sparse.makeCompressed();
    }

    // Pre-compute mask transpose for W-update
    SpMat h_mask_T;
    if (use_mask) {
        h_mask_T = config.mask->transpose();
        h_mask_T.makeCompressed();
    }

    // Upload user mask to device for GPU-native mask path (MSE only)
    // Stores CSC arrays of the mask on device; null when no mask.
    std::unique_ptr<FactorNet::gpu::SparseMatrixGPU<Scalar>> d_mask;
    if (use_mask && !use_irls) {
        // config.mask is already CSC (SpMat = Eigen::SparseMatrix, ColMajor)
        const auto& mask_ref = *config.mask;
        d_mask = std::make_unique<FactorNet::gpu::SparseMatrixGPU<Scalar>>(
            mask_ref.rows(), mask_ref.cols(),
            static_cast<int>(mask_ref.nonZeros()),
            mask_ref.outerIndexPtr(),
            mask_ref.innerIndexPtr(),
            mask_ref.valuePtr());
    }

    // Workspace for GPU mask delta kernels (reused across iterations)
    DenseMatrixGPU<Scalar> d_mask_delta_G_h(k * k, n);
    DenseMatrixGPU<Scalar> d_mask_delta_b_h(k, n);
    DenseMatrixGPU<Scalar> d_mask_B_full_h(k, n);
    DenseMatrixGPU<Scalar> d_mask_delta_G_w(k * k, m);
    DenseMatrixGPU<Scalar> d_mask_delta_b_w(k, m);
    DenseMatrixGPU<Scalar> d_mask_B_full_w(k, m);

    // ------------------------------------------------------------------
    // 5b. Per-row dispersion for GP/NB/Gamma/IG/Tweedie (device-resident)
    // ------------------------------------------------------------------
    // For H-update: disp is per-row of A (m-length), indexed by row_idx
    // For W-update on A^T: disp is per-row of A (m-length), indexed by column of A^T
    // So H-update uses disp_row, W-update uses disp_col (same data, different indexing)
    DeviceMemory<Scalar> d_disp_vec;  // m-length dispersion on device
    const bool is_gp_gpu = (config.loss.type == LossType::GP);
    const bool is_nb_gpu = (config.loss.type == LossType::NB);
    const bool is_gamma_gpu = (config.loss.type == LossType::GAMMA);
    const bool is_invgauss_gpu = (config.loss.type == LossType::INVGAUSS);
    const bool is_tweedie_gpu = (config.loss.type == LossType::TWEEDIE);
    const bool needs_disp = (is_gp_gpu || is_nb_gpu || is_gamma_gpu ||
                             is_invgauss_gpu || is_tweedie_gpu) &&
                            config.gp_dispersion != DispersionMode::NONE;
    if (needs_disp) {
        d_disp_vec.allocate(m);
        Scalar init_val = is_nb_gpu ? config.nb_size_init
                        : (is_gamma_gpu || is_invgauss_gpu || is_tweedie_gpu)
                          ? config.gamma_phi_init
                          : config.loss.gp_theta_init;
        std::vector<Scalar> h_disp(m, init_val);
        d_disp_vec.upload(h_disp.data(), m);
    }

    // ------------------------------------------------------------------
    // 5c. Zero-inflated buffers (GPU-resident A_imputed, π vectors)
    // ------------------------------------------------------------------
    const bool is_zi_gpu = (is_gp_gpu || is_nb_gpu) && config.has_zi();
    DenseMatrixGPU<Scalar> d_A_imputed;    // m × n dense (ZI-imputed A)
    DenseMatrixGPU<Scalar> d_A_imputed_T;  // n × m dense (transposed for W-update)
    DeviceMemory<Scalar> d_pi_row;  // m-length π_row (ZI_ROW or ZI_TWOWAY)
    DeviceMemory<Scalar> d_pi_col;  // n-length π_col (ZI_COL or ZI_TWOWAY)

    if (is_zi_gpu) {
        // Memory guard: ZI requires 2 dense m×n matrices on GPU
        const size_t zi_dense_bytes = static_cast<size_t>(m) * n * sizeof(Scalar) * 2;
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (zi_dense_bytes > free_mem * 8 / 10) {  // >80% of free VRAM
            char buf[256];
            snprintf(buf, sizeof(buf),
                "ZI-NMF requires %.1f GB for dense A_imputed matrices "
                "(m=%d, n=%d) but only %.1f GB free on GPU. "
                "Use resource=\"cpu\" or reduce matrix dimensions.",
                zi_dense_bytes / (1024.0 * 1024.0 * 1024.0), m, n,
                free_mem / (1024.0 * 1024.0 * 1024.0));
            throw std::runtime_error(buf);
        }

        // Allocate dense imputed matrices
        d_A_imputed = DenseMatrixGPU<Scalar>(m, n);
        d_A_imputed_T = DenseMatrixGPU<Scalar>(n, m);

        // Initialize A_imputed from sparse A (copy nonzeros, zeros stay zero)
        d_A_imputed.zero();
        // Scatter sparse values into dense (simple kernel / host init)
        {
            std::vector<Scalar> h_A_dense(static_cast<size_t>(m) * n, Scalar(0));
            for (int j = 0; j < n; ++j)
                for (int p = A_csc_ptr[j]; p < A_csc_ptr[j + 1]; ++p)
                    h_A_dense[A_row_idx[p] + static_cast<size_t>(j) * m] = A_values[p];
            d_A_imputed.upload_from(h_A_dense.data());
        }

        // Allocate and initialize π vectors
        const int zi_mode_int = static_cast<int>(config.zi_mode);
        if (zi_mode_int == 1 || zi_mode_int == 3) {  // ZI_ROW or ZI_TWOWAY
            d_pi_row.allocate(m);
            // Initialize from data: pi_row[i] = min(zero_rate_i * 0.5, 0.3)
            std::vector<int> nnz_per_row(m, 0);
            for (int p = 0; p < nnz; ++p) nnz_per_row[A_row_idx[p]]++;
            std::vector<Scalar> h_pi_row(m);
            for (int i = 0; i < m; ++i) {
                Scalar zero_rate = Scalar(1) - static_cast<Scalar>(nnz_per_row[i]) / static_cast<Scalar>(n);
                h_pi_row[i] = std::min(zero_rate * Scalar(0.5), Scalar(0.3));
            }
            d_pi_row.upload(h_pi_row.data(), m);
        }
        if (zi_mode_int == 2 || zi_mode_int == 3) {  // ZI_COL or ZI_TWOWAY
            d_pi_col.allocate(n);
            std::vector<Scalar> h_pi_col(n);
            for (int j = 0; j < n; ++j) {
                int nnz_col = A_csc_ptr[j + 1] - A_csc_ptr[j];
                Scalar zero_rate = Scalar(1) - static_cast<Scalar>(nnz_col) / static_cast<Scalar>(m);
                h_pi_col[j] = std::min(zero_rate * Scalar(0.5), Scalar(0.3));
            }
            d_pi_col.upload(h_pi_col.data(), n);
        }

        if (config.verbose) {
            size_t zi_mem = d_A_imputed.elements() * sizeof(Scalar) * 2  // A_imputed + transposed
                          + d_pi_row.size() * sizeof(Scalar)
                          + d_pi_col.size() * sizeof(Scalar);
            FACTORNET_GPU_LOG(true, "[ZI] Allocated %.1f MB for dense A_imputed + π vectors\n",
                zi_mem / (1024.0 * 1024.0));
        }
    }

    // Pre-allocate d-scaled W buffer for ZI E-step (reused every iteration)
    DenseMatrixGPU<Scalar> d_W_Td_zi;
    if (is_zi_gpu) d_W_Td_zi = DenseMatrixGPU<Scalar>(k, m);

    // Pre-allocate d-scaled W buffer for projective H-update (reused every iteration)
    const bool is_projective =
        (nmf::variant::h_update_mode(config) == nmf::variant::HUpdateMode::PROJECTIVE);
    DenseMatrixGPU<Scalar> d_W_scaled;
    if (is_projective) d_W_scaled = DenseMatrixGPU<Scalar>(k, m);

    // ------------------------------------------------------------------
    // 6. Iteration loop
    // ------------------------------------------------------------------

    ::FactorNet::NMFResult<Scalar> result;
    Scalar prev_loss = std::numeric_limits<Scalar>::max();
    int patience_counter = 0;

    // Per-phase CUDA event timing (active only when config.verbose = true)
    cudaEvent_t _ev_start = nullptr, _ev_stop = nullptr;
    float _phase_ms = 0.0f;
    double _t_gram_h = 0, _t_rhs_h = 0, _t_nnls_h = 0, _t_norm_h = 0;
    double _t_gram_w = 0, _t_rhs_w = 0, _t_nnls_w = 0, _t_norm_w = 0;
    double _t_loss = 0;
    if (config.verbose) {
        CUDA_CHECK(cudaEventCreate(&_ev_start));
        CUDA_CHECK(cudaEventCreate(&_ev_stop));
    }

    // Wall-clock overhead tracking (std::chrono, verbose only)
    // Measures the CPU-side cost of sections NOT timed by CUDA events:
    //   wt_norm_h / wt_norm_w : L1/None normalization (host-mediated paths)
    //   wt_iter               : full iteration wall time
    using _clk = std::chrono::high_resolution_clock;
    using _ns  = std::chrono::nanoseconds;
    double _wt_h_norm_sync = 0, _wt_h_norm_cpu = 0, _wt_h_norm_ul = 0;
    double _wt_w_norm_sync = 0, _wt_w_norm_cpu = 0, _wt_w_norm_ul = 0;
    double _wt_iter_sum = 0;  // sum of full-iteration wall times (ms)
#define _WC_NOW() std::chrono::duration_cast<_ns>(_clk::now().time_since_epoch()).count()
#define GPU_PBEGIN() do { if (config.verbose) { CUDA_CHECK(cudaEventRecord(_ev_start, ctx.stream)); } } while(0)
#define GPU_PEND(acc) do { if (config.verbose) { \
    CUDA_CHECK(cudaEventRecord(_ev_stop, ctx.stream)); \
    CUDA_CHECK(cudaEventSynchronize(_ev_stop)); \
    CUDA_CHECK(cudaEventElapsedTime(&_phase_ms, _ev_start, _ev_stop)); \
    acc += _phase_ms; } } while(0)

    // Structured GPU section timer for profiling (mirrors existing timing)
    FactorNet::profiling::GpuSectionTimer _prof_timer;
    _prof_timer.init();
    _prof_timer.active = config.verbose;
    double _t_features_h = 0, _t_features_w = 0;
    double _t_dispersion = 0, _t_zi_estep = 0;
    double _t_irls_h = 0, _t_irls_w = 0;

    for (int iter = 0; iter < config.max_iter; ++iter) {

        // ==============================================================
        // H update  (unit-norm W for Gram/RHS — matches CPU approach)
        // ==============================================================
        // CPU insight: use unit-norm factors for well-conditioned Gram.
        // d is NOT applied before Gram/RHS — this preserves conditioning.

        bool compute_loss = (iter % config.loss_every == 0)
                          || (iter == config.max_iter - 1);

        // Wall-clock start of iteration
        int64_t _wt_iter_t0 = config.verbose ? _WC_NOW() : 0;

        const auto h_mode = nmf::variant::h_update_mode(config);

        if (h_mode == nmf::variant::HUpdateMode::PROJECTIVE) {
            // PROJECTIVE: H = diag(d) · W · A — GPU-native (zero PCIe transfers)
            // 1. Copy W → W_scaled, then scale rows: W_scaled[i,:] *= d[i]
            CUDA_CHECK(cudaMemcpyAsync(
                d_W_scaled.data.get(), d_W.data.get(),
                sizeof(Scalar) * k * m, cudaMemcpyDeviceToDevice, ctx.stream));
            ::FactorNet::gpu::scale_rows_by_d_gpu(ctx, d_W_scaled, d_d);

            // 2. H = W_scaled · A (SpMM with temporary factor pointer)
            spmm_fwd.compute(ctx, d_W_scaled.data.get(), d_H.data.get());

            // 3. Normalize H → d
            if (config.norm_type == ::FactorNet::NormType::L2) {
                scale_rows_opt_gpu(ctx, d_H, d_d);
            } else if (config.norm_type == ::FactorNet::NormType::L1) {
                ::FactorNet::gpu::l1_norm_rows_gpu(ctx, d_H, d_d);
            } else {
                ::FactorNet::gpu::set_ones_gpu(ctx, d_d, k);
            }

        } else if (h_mode == nmf::variant::HUpdateMode::SYMMETRIC_SKIP) {
            // SYMMETRIC: skip H update (H = W enforced after W update)

        } else if (use_mask && !use_irls) {
            // --- GPU-native masked H-update: delta-kernel path ---
            // Treats user-masked entries like CV test entries: excluded from
            // B_full and delta_b, but added to delta_G so G_train is correct.
            GPU_PBEGIN();
            compute_gram_regularized_gpu(ctx, d_W, d_G,
                                         static_cast<Scalar>(config.H.L2));
            compute_cv_deltas_h_gpu(ctx, d_A, d_W, k, n,
                                    /*rng_state=*/0ULL, /*inv_prob=*/0ULL,
                                    d_mask_delta_G_h, d_mask_delta_b_h, &d_mask_B_full_h,
                                    /*W_half=*/nullptr,
                                    d_mask->col_ptr.get(), d_mask->row_indices.get());
            primitives::cholesky_cv_tiled_gpu(ctx, d_G, d_mask_delta_G_h,
                                d_mask_B_full_h, d_mask_delta_b_h, d_H,
                                config.H.nonneg);
            GPU_PEND(_t_nnls_h);

            if (config.H.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_H,
                    static_cast<Scalar>(config.H.upper_bound));

        } else if (use_mask && use_irls) {
            // --- Masked IRLS H-update: host-mediated (per-column weights needed) ---
            ctx.sync();
            gpu_detail::download_to_eigen(d_W, h_W_T);
            gpu_detail::download_to_eigen(d_H, h_H);

            DenseMatrix<Scalar> G_base = h_W_T * h_W_T.transpose();

            nmf::mask_detail::masked_nnls_h(
                h_A_sparse, h_W_T, G_base, h_H, *config.mask, config,
                config.threads > 0 ? config.threads : 1, iter > 0);

            if (config.H.upper_bound > 0)
                features::apply_upper_bound(h_H, config.H.upper_bound);

            gpu_detail::upload_from_eigen(d_H, h_H);

        } else if (use_irls) {
            // --- IRLS H-update: GPU-native IRLS solver ---
            GPU_PBEGIN();

            if (is_zi_gpu && iter > 0) {
                // ZI path: dense IRLS on A_imputed (all weights differ from 1)
                const DeviceMemory<Scalar>* disp_row_h = d_disp_vec.size() > 0 ? &d_disp_vec : nullptr;
                primitives::gpu_zi::nnls_batch_zi_irls_gpu(
                    ctx, d_A_imputed, d_W, d_H,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    disp_row_h,                  // per-row dispersion
                    static_cast<const DeviceMemory<Scalar>*>(nullptr),  // no per-col
                    static_cast<Scalar>(config.H.L1),
                    static_cast<Scalar>(config.H.L2),
                    config.H.nonneg,
                    config.cd_max_iter,
                    config.irls_max_iter,
                    static_cast<Scalar>(config.cd_tol),
                    static_cast<Scalar>(config.irls_tol),
                    config.threads > 0 ? config.threads : 1);
            } else {
                // Standard sparse IRLS (or first ZI iteration before imputation)
                compute_gram_gpu(ctx, d_W, d_G);  // G_base = W^T W (no L2 in base)

                const DeviceMemory<Scalar>* disp_row_h = d_disp_vec.size() > 0 ? &d_disp_vec : nullptr;
                primitives::gpu_irls::nnls_batch_irls_gpu(
                    ctx, d_A, d_W, d_H, d_G,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    config.loss.robust_delta,
                    config.loss.gp_blend,
                    config.loss.power_param,
                    disp_row_h,                  // per-row dispersion
                    static_cast<const DeviceMemory<Scalar>*>(nullptr),  // no per-col
                    static_cast<Scalar>(config.H.L1),
                    static_cast<Scalar>(config.H.L2),
                    config.H.nonneg,
                    config.cd_max_iter,
                    config.irls_max_iter);
            }
            GPU_PEND(_t_nnls_h);

            // Upper bounds — GPU clamp kernel (no PCIe round-trip)
            if (config.H.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_H,
                    static_cast<Scalar>(config.H.upper_bound));

        } else {
            // --- Standard GPU MSE H-update ---

            // 1. Gram: G = WᵀW + eps·I  (unit-norm W, well-conditioned)
            GPU_PBEGIN();
            compute_gram_regularized_gpu(ctx, d_W, d_G,
                                         static_cast<Scalar>(config.H.L2));
            GPU_PEND(_t_gram_h);

            // 2. RHS: B = W · A  (unit-norm W) — cuSPARSE SpMM
            GPU_PBEGIN();
            spmm_fwd.compute(ctx, d_W.data.get(), d_B.data.get());
            GPU_PEND(_t_rhs_h);

            // Save B for Gram-trick loss (pre-NNLS, pre-L1)
            if (compute_loss) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_B_save.data.get(), d_B.data.get(),
                    static_cast<size_t>(k) * n * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
            }

            // 3. Feature application on B (RHS)
            if (config.H.L1 > 0) {
                apply_l1_gpu(ctx, d_B, static_cast<Scalar>(config.H.L1));
            }

            // Tier 2 features: GPU-native L21 and angular, host fallback for graph
            _prof_timer.begin(ctx.stream, "features_tier2_H");
            if (config.H.angular > 0) {
                nmf::gpu_features::apply_angular_gpu(ctx, d_G, d_H,
                    static_cast<Scalar>(config.H.angular));
            }
            if (config.H.L21 > 0) {
                nmf::gpu_features::apply_L21_gpu(ctx, d_G, d_H,
                    static_cast<Scalar>(config.H.L21));
            }
            if (config.H.graph != nullptr) {
                graph_reg_H.apply(ctx, d_G.data.get(), d_H.data.get(),
                                  d_H.cols, static_cast<Scalar>(config.H.graph_lambda));
            }
            _prof_timer.end(ctx.stream);

            // 4. NNLS solve: dispatch by solver_mode
            GPU_PBEGIN();
            if (config.solver_mode == 1) {
                // GPU-native Cholesky: pure GPU solve, zero transfers
                primitives::cholesky_clip_batch_gpu(ctx, d_G, d_B, d_H, config.H.nonneg);
            } else {
                // Fixed-iteration batch CD: default GPU solver
                batch_cd_nnls_gpu(ctx, d_G, d_B, d_H, config.cd_max_iter,
                                  config.H.nonneg, /*warm_start=*/(iter > 0));
            }
            GPU_PEND(_t_nnls_h);

            // 5. Post-NNLS: upper bounds — GPU clamp kernel (no PCIe round-trip)
            if (config.H.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_H,
                    static_cast<Scalar>(config.H.upper_bound));
        }

        // 6. Scale H → d  (skip for projective [already normalized] and symmetric [no-op])
        if (h_mode == nmf::variant::HUpdateMode::STANDARD) {
            if (config.norm_type == ::FactorNet::NormType::L2) {
                GPU_PBEGIN();
                scale_rows_opt_gpu(ctx, d_H, d_d);
                GPU_PEND(_t_norm_h);
            } else if (config.norm_type == ::FactorNet::NormType::L1) {
                // GPU L1 norm: no PCIe transfer, runs entirely on device
                GPU_PBEGIN();
                ::FactorNet::gpu::l1_norm_rows_gpu(ctx, d_H, d_d);
                GPU_PEND(_t_norm_h);
            } else {
                // NormType::None — GPU fill: d := 1 (no sync, no PCIe transfer)
                ::FactorNet::gpu::set_ones_gpu(ctx, d_d, k);
            }
        }

        // ==============================================================
        // W update  (unit-norm H for Gram/RHS — matches CPU approach)
        // ==============================================================

        const auto w_mode = nmf::variant::w_update_mode(config);

        if (w_mode == nmf::variant::WUpdateMode::SYMMETRIC) {
            // SYMMETRIC W-update: Gram(W), ForwardRHS(W,A), features, NNLS
            // Uses W as factor (not H) since A ≈ W·diag(d)·W^T

            compute_gram_regularized_gpu(ctx, d_W, d_G,
                                         static_cast<Scalar>(config.W.L2));

            spmm_fwd.compute(ctx, d_W.data.get(), d_B.data.get());

            if (config.W.L1 > 0)
                apply_l1_gpu(ctx, d_B, static_cast<Scalar>(config.W.L1));

            if (config.W.angular > 0) {
                nmf::gpu_features::apply_angular_gpu(ctx, d_G, d_W,
                    static_cast<Scalar>(config.W.angular));
            }
            if (config.W.L21 > 0) {
                nmf::gpu_features::apply_L21_gpu(ctx, d_G, d_W,
                    static_cast<Scalar>(config.W.L21));
            }
            if (config.W.graph != nullptr) {
                graph_reg_W.apply(ctx, d_G.data.get(), d_W.data.get(),
                                  d_W.cols, static_cast<Scalar>(config.W.graph_lambda));
            }

            if (config.solver_mode == 1) {
                // GPU-native Cholesky for symmetric W
                primitives::cholesky_clip_batch_gpu(ctx, d_G, d_B, d_W, config.W.nonneg);
            } else {
                // Fixed-iteration batch CD: default GPU solver
                batch_cd_nnls_gpu(ctx, d_G, d_B, d_W, config.cd_max_iter,
                                  config.W.nonneg, /*warm_start=*/(iter > 0));
            }

            if (config.W.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                    static_cast<Scalar>(config.W.upper_bound));

        } else if (use_mask && !use_irls) {
            // --- GPU-native masked W-update: delta-kernel path ---
            GPU_PBEGIN();
            compute_gram_regularized_gpu(ctx, d_H, d_G,
                                         static_cast<Scalar>(config.W.L2));
            compute_cv_deltas_w_gpu(ctx, d_A_T, d_H, k, m,
                                    /*rng_state=*/0ULL, /*inv_prob=*/0ULL,
                                    d_mask_delta_G_w, d_mask_delta_b_w, &d_mask_B_full_w,
                                    /*H_half=*/nullptr,
                                    d_mask->col_ptr.get(), d_mask->row_indices.get());
            primitives::cholesky_cv_tiled_gpu(ctx, d_G, d_mask_delta_G_w,
                                d_mask_B_full_w, d_mask_delta_b_w, d_W,
                                config.W.nonneg);
            GPU_PEND(_t_nnls_w);

            if (config.W.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                    static_cast<Scalar>(config.W.upper_bound));

        } else if (use_mask && use_irls) {
            // --- Masked IRLS W-update: host-mediated (per-column weights needed) ---
            ctx.sync();
            gpu_detail::download_to_eigen(d_H, h_H);
            gpu_detail::download_to_eigen(d_W, h_W_T);

            DenseMatrix<Scalar> G_base = h_H * h_H.transpose();

            nmf::mask_detail::masked_nnls_w(
                h_At_sparse, h_H, G_base, h_W_T, h_mask_T, config,
                config.threads > 0 ? config.threads : 1, iter > 0);

            if (config.W.upper_bound > 0)
                features::apply_upper_bound(h_W_T, config.W.upper_bound);

            gpu_detail::upload_from_eigen(d_W, h_W_T);

        } else if (use_irls) {
            // --- IRLS W-update: GPU-native IRLS solver ---
            GPU_PBEGIN();

            if (is_zi_gpu && iter > 0) {
                // ZI path: transpose A_imputed, then dense IRLS on transposed
                primitives::gpu_zi::transpose_dense_gpu(ctx, d_A_imputed, d_A_imputed_T);

                // W-update: solve for columns of W using A_imputed^T (n×m)
                // disp is per-column of A_imputed^T = per-row of A = m-length
                const DeviceMemory<Scalar>* disp_col_w = d_disp_vec.size() > 0 ? &d_disp_vec : nullptr;
                primitives::gpu_zi::nnls_batch_zi_irls_gpu(
                    ctx, d_A_imputed_T, d_H, d_W,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    static_cast<const DeviceMemory<Scalar>*>(nullptr),  // no per-row (transposed)
                    disp_col_w,                  // per-column = per-row of A
                    static_cast<Scalar>(config.W.L1),
                    static_cast<Scalar>(config.W.L2),
                    config.W.nonneg,
                    config.cd_max_iter,
                    config.irls_max_iter,
                    static_cast<Scalar>(config.cd_tol),
                    static_cast<Scalar>(config.irls_tol),
                    config.threads > 0 ? config.threads : 1);
            } else {
                // Standard sparse IRLS (or first ZI iteration)
                compute_gram_gpu(ctx, d_H, d_G);  // G_base = H H^T (no L2 in base)

                const DeviceMemory<Scalar>* disp_col_w = d_disp_vec.size() > 0 ? &d_disp_vec : nullptr;
                primitives::gpu_irls::nnls_batch_irls_gpu(
                    ctx, d_A_T, d_H, d_W, d_G,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    config.loss.robust_delta,
                    config.loss.gp_blend,
                    config.loss.power_param,
                    static_cast<const DeviceMemory<Scalar>*>(nullptr),  // no per-row
                    disp_col_w,                  // per-column dispersion (= per-row of A)
                    static_cast<Scalar>(config.W.L1),
                    static_cast<Scalar>(config.W.L2),
                    config.W.nonneg,
                    config.cd_max_iter,
                    config.irls_max_iter);
            }
            GPU_PEND(_t_nnls_w);

            // Post-NNLS bounds — GPU clamp kernel (no PCIe round-trip)
            if (config.W.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                    static_cast<Scalar>(config.W.upper_bound));

        } else {
            // --- Standard GPU MSE W-update ---

            // 1. Gram: G = HᵀH + eps·I  (unit-norm H, well-conditioned)
            GPU_PBEGIN();
            compute_gram_regularized_gpu(ctx, d_H, d_G,
                                         static_cast<Scalar>(config.W.L2));
            GPU_PEND(_t_gram_w);

            // 2. RHS: B = H · Aᵀ (k × m, unit-norm H) — cuSPARSE SpMM
            GPU_PBEGIN();
            spmm_bwd.compute(ctx, d_H.data.get(), d_B.data.get());
            GPU_PEND(_t_rhs_w);

            // 3. Features on B (RHS)
            if (config.W.L1 > 0) {
                apply_l1_gpu(ctx, d_B, static_cast<Scalar>(config.W.L1));
            }

            // Tier 2 features: GPU-native L21 and angular, host fallback for graph
            _prof_timer.begin(ctx.stream, "features_tier2_W");
            if (config.W.angular > 0) {
                nmf::gpu_features::apply_angular_gpu(ctx, d_G, d_W,
                    static_cast<Scalar>(config.W.angular));
            }
            if (config.W.L21 > 0) {
                nmf::gpu_features::apply_L21_gpu(ctx, d_G, d_W,
                    static_cast<Scalar>(config.W.L21));
            }
            if (config.W.graph != nullptr) {
                graph_reg_W.apply(ctx, d_G.data.get(), d_W.data.get(),
                                  d_W.cols, static_cast<Scalar>(config.W.graph_lambda));
            }
            _prof_timer.end(ctx.stream);

            // 4. NNLS solve: dispatch by solver_mode
            GPU_PBEGIN();
            if (config.solver_mode == 1) {
                // GPU-native Cholesky for standard W
                primitives::cholesky_clip_batch_gpu(ctx, d_G, d_B, d_W, config.W.nonneg);
            } else {
                // Fixed-iteration batch CD: default GPU solver
                batch_cd_nnls_gpu(ctx, d_G, d_B, d_W, config.cd_max_iter,
                                  config.W.nonneg, /*warm_start=*/(iter > 0));
            }
            GPU_PEND(_t_nnls_w);

            // 5. Post-NNLS bounds — GPU clamp kernel (no PCIe round-trip)
            if (config.W.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                    static_cast<Scalar>(config.W.upper_bound));
        }

        // 6. Scale W → d  (extract row norms, normalize)
        if (config.norm_type == ::FactorNet::NormType::L2) {
            GPU_PBEGIN();
            scale_rows_opt_gpu(ctx, d_W, d_d);
            GPU_PEND(_t_norm_w);
        } else if (config.norm_type == ::FactorNet::NormType::L1) {
            // GPU L1 norm for W: no PCIe transfer
            // NOTE: d_W is k×m (W transposed); accumulate over m columns
            GPU_PBEGIN();
            ::FactorNet::gpu::l1_norm_rows_gpu(ctx, d_W, d_d);
            GPU_PEND(_t_norm_w);
        } else {
            // NormType::None — GPU fill: d := 1 (no sync, no PCIe transfer)
            ::FactorNet::gpu::set_ones_gpu(ctx, d_d, k);
        }

        // Enforce H = W (symmetric only, on device)
        if (w_mode == nmf::variant::WUpdateMode::SYMMETRIC) {
            CUDA_CHECK(cudaMemcpyAsync(d_H.data.get(), d_W.data.get(),
                static_cast<size_t>(k) * n * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        }

        // ==============================================================
        // GP/NB dispersion update (GPU-native, once per iteration)
        // ==============================================================
        _prof_timer.begin(ctx.stream, "dispersion");
        if (d_disp_vec.size() > 0) {
            // Build d-scaled W on device (reuses d_W_Td_loss buffer)
            CUDA_CHECK(cudaMemcpyAsync(
                d_W_Td_loss.data.get(), d_W.data.get(),
                static_cast<size_t>(k) * m * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            absorb_d_gpu(ctx, d_W_Td_loss, d_d);

            int disp_mode = (config.gp_dispersion == DispersionMode::GLOBAL) ? 1 : 0;

            if (is_gp_gpu) {
                // Compute Gram of H for NB (GP doesn't need it but reuse is possible)
                gpu::gp_theta_update_gpu<Scalar>(
                    ctx, d_A, d_W_Td_loss, d_H, d_disp_vec,
                    config.gp_theta_max, disp_mode);
            } else if (is_nb_gpu) {
                // NB needs G_H = H · H' for Gram trick
                DenseMatrixGPU<Scalar> d_G_H(k, k);
                compute_gram_gpu(ctx, d_H, d_G_H);
                gpu::nb_size_update_gpu<Scalar>(
                    ctx, d_A, d_W_Td_loss, d_H, d_G_H, d_disp_vec,
                    config.nb_size_min, config.nb_size_max, disp_mode);
            } else if (is_gamma_gpu || is_invgauss_gpu || is_tweedie_gpu) {
                double var_power = is_tweedie_gpu
                    ? static_cast<double>(config.loss.power_param)
                    : (is_gamma_gpu ? 2.0 : 3.0);
                gpu::phi_update_gpu<Scalar>(
                    ctx, d_A, d_W_Td_loss, d_H, d_disp_vec,
                    var_power, config.gamma_phi_min, config.gamma_phi_max,
                    disp_mode);
            }
        }
        _prof_timer.end(ctx.stream);

        // ==============================================================
        // ZI E-step + M-step + soft imputation (after dispersion update)
        // ==============================================================
        _prof_timer.begin(ctx.stream, "zi_estep");
        if (is_zi_gpu) {
            // Compute d-scaled W on device. When dispersion is active,
            // d_W_Td_loss already has the scaled W; copy it.
            // Otherwise compute fresh.
            if (d_disp_vec.size() > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_W_Td_zi.data.get(), d_W_Td_loss.data.get(),
                    static_cast<size_t>(k) * m * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_W_Td_zi.data.get(), d_W.data.get(),
                    static_cast<size_t>(k) * m * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
                absorb_d_gpu(ctx, d_W_Td_zi, d_d);
            }

            primitives::gpu_zi::zi_em_step_gpu(
                ctx, d_A, d_W_Td_zi, d_H, d_disp_vec,
                d_pi_row, d_pi_col,
                d_A_imputed,
                k, m, n,
                static_cast<int>(config.loss.type),
                static_cast<int>(config.zi_mode),
                config.zi_em_iters,
                config.gp_theta_min);
        }
        _prof_timer.end(ctx.stream);

        // ==============================================================
        // Loss computation (after both updates, matching CPU)
        //
        // Note: ZI-NMF computes loss against the ORIGINAL sparse A on
        // nonzero entries only (matching CPU behavior). The solver
        // operates on A_imputed where structural zeros are softened,
        // but convergence is monitored against real observations.
        // ==============================================================

        // Ensure B_save is available for Gram trick loss
        // (projective/symmetric skip the standard H-update that saves B)
        if (compute_loss && !use_mask && !use_irls &&
            h_mode != nmf::variant::HUpdateMode::STANDARD) {
            spmm_fwd.compute(ctx, d_W.data.get(), d_B.data.get());
            CUDA_CHECK(cudaMemcpyAsync(
                d_B_save.data.get(), d_B.data.get(),
                static_cast<size_t>(k) * n * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        }

        if (compute_loss) {
            double loss_val;

            if (use_mask) {
                // Masked loss: explicit computation skipping masked entries
                ctx.sync();
                gpu_detail::download_to_eigen(d_W, h_W_T);
                gpu_detail::download_to_eigen(d_H, h_H);

                DenseMatrix<Scalar> h_W_Td = h_W_T;
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h_d_vec(k);
                d_d.download(h_d_vec.data(), k);
                for (int f = 0; f < k; ++f)
                    h_W_Td.row(f) *= h_d_vec(f);

                loss_val = static_cast<double>(
                    nmf::mask_detail::masked_loss(
                        h_A_sparse, h_W_Td, h_H, *config.mask, config.loss,
                        config.threads > 0 ? config.threads : 1));

            } else if (use_irls) {
                // Non-MSE loss: GPU-native explicit per-element computation
                GPU_PBEGIN();

                // Build d-scaled W on device: W_Td = W * diag(d)
                CUDA_CHECK(cudaMemcpyAsync(
                    d_W_Td_loss.data.get(), d_W.data.get(),
                    static_cast<size_t>(k) * m * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
                absorb_d_gpu(ctx, d_W_Td_loss, d_d);

                loss_val = gpu::compute_explicit_loss_gpu<Scalar>(
                    ctx, d_A, d_W_Td_loss, d_H, d_disp_vec,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    config.loss.power_param);
                GPU_PEND(_t_loss);

            } else {
                // MSE: Gram trick (O(k²) — essentially free)
                // Uses B_save = W*A (pre-feature) from H update.
                GPU_PBEGIN();
                CUDA_CHECK(cudaMemcpyAsync(
                    d_Hd.data.get(), d_H.data.get(),
                    static_cast<size_t>(k) * n * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
                absorb_d_gpu(ctx, d_Hd, d_d);

                compute_gram_gpu(ctx, d_W, d_G);  // G = WᵀW, no eps

                loss_val = compute_total_loss_gpu(
                    ctx, d_A, d_W, d_Hd, d_G, d_B_save, d_scratch, 0, 0);
                GPU_PEND(_t_loss);
            }

            if (config.track_loss_history)
                result.loss_history.push_back(static_cast<Scalar>(loss_val));

            // Convergence check
            if (iter > 0) {
                Scalar rel_change = std::abs(
                    static_cast<Scalar>(prev_loss - loss_val))
                    / (std::abs(prev_loss) + static_cast<Scalar>(1e-15));
                result.final_tol = rel_change;

                if (rel_change < config.tol) {
                    patience_counter++;
                    if (patience_counter >= config.patience) {
                        result.converged = true;
                        result.train_loss = static_cast<Scalar>(loss_val);
                        result.iterations = iter + 1;
                        break;
                    }
                } else {
                    patience_counter = 0;
                }
            }
            prev_loss = static_cast<Scalar>(loss_val);
        }

        // Per-iteration callback (fires every iteration using last known loss)
        if (config.on_iteration)
            config.on_iteration(iter + 1, static_cast<Scalar>(prev_loss), Scalar(0));

        result.iterations = iter + 1;

        // Accumulate full-iteration wall time
        if (config.verbose) {
            _wt_iter_sum += (_WC_NOW() - _wt_iter_t0) * 1e-6;
        }
    }

    // Per-phase timing report (verbose mode)
    if (config.verbose && result.iterations > 0) {
        const int ni = result.iterations;
        double iter_ms = (_t_gram_h + _t_rhs_h + _t_nnls_h + _t_norm_h +
                          _t_gram_w + _t_rhs_w + _t_nnls_w + _t_norm_w + _t_loss) / ni;
        FACTORNET_GPU_LOG(true, "[GPU NMF timing — avg per iter over %d iters]\n", ni);
        FACTORNET_GPU_LOG(true, "  H:    gram=%7.3fms  rhs=%7.3fms  nnls=%7.3fms  norm=%7.3fms\n",
                _t_gram_h/ni, _t_rhs_h/ni, _t_nnls_h/ni, _t_norm_h/ni);
        FACTORNET_GPU_LOG(true, "  W:    gram=%7.3fms  rhs=%7.3fms  nnls=%7.3fms  norm=%7.3fms\n",
                _t_gram_w/ni, _t_rhs_w/ni, _t_nnls_w/ni, _t_norm_w/ni);
        FACTORNET_GPU_LOG(true, "  loss: %7.3fms  (every %d iters)\n", _t_loss/ni, config.loss_every);
        FACTORNET_GPU_LOG(true, "  total timed: %.3fms/iter  (%.1fms total across %d iters)\n",
                iter_ms, iter_ms * ni, ni);
        // Breakdown percentages
        double tot = _t_gram_h + _t_rhs_h + _t_nnls_h + _t_norm_h +
                     _t_gram_w + _t_rhs_w + _t_nnls_w + _t_norm_w + _t_loss;
        if (tot > 0) {
            FACTORNET_GPU_LOG(true, "  breakdown: gram_H=%.1f%%  rhs_H=%.1f%%  nnls_H=%.1f%%  norm_H=%.1f%%  "
                    "gram_W=%.1f%%  rhs_W=%.1f%%  nnls_W=%.1f%%  norm_W=%.1f%%  loss=%.1f%%\n",
                    100.*_t_gram_h/tot, 100.*_t_rhs_h/tot, 100.*_t_nnls_h/tot, 100.*_t_norm_h/tot,
                    100.*_t_gram_w/tot, 100.*_t_rhs_w/tot, 100.*_t_nnls_w/tot, 100.*_t_norm_w/tot,
                    100.*_t_loss/tot);
        }
        // -- Wall-clock overhead breakdown --
        const int ni_wt = result.iterations;
        double timed_total = _t_gram_h + _t_rhs_h + _t_nnls_h + _t_norm_h +
                             _t_gram_w + _t_rhs_w + _t_nnls_w + _t_norm_w + _t_loss;
        double wall_ms_per_iter = _wt_iter_sum / ni_wt;
        double overhead_ms = wall_ms_per_iter - timed_total / ni_wt;
        FACTORNET_GPU_LOG(true, "[GPU NMF overhead — wall vs timed (avg per iter)]\n");
        FACTORNET_GPU_LOG(true, "  wall/iter:     %7.3fms    timed/iter:  %7.3fms\n",
                wall_ms_per_iter, timed_total / ni_wt);
        FACTORNET_GPU_LOG(true, "  unaccounted:   %7.3fms  (%.0f%% of wall — gap between timed kernels)\n",
                overhead_ms, 100. * overhead_ms / wall_ms_per_iter);
        // norm breakdown (only meaningful for NormType::None; L1 and L2 go through GPU path)
        double h_none_total = _wt_h_norm_sync + _wt_h_norm_cpu + _wt_h_norm_ul;
        double w_none_total = _wt_w_norm_sync + _wt_w_norm_cpu + _wt_w_norm_ul;
        if (h_none_total + w_none_total > 0) {
            FACTORNET_GPU_LOG(true, "  norm-H(none):  %7.3fms/iter  (sync=%.3f  cpu=%.3f  upload=%.3f)\n",
                    (h_none_total) / ni_wt,
                    _wt_h_norm_sync / ni_wt, _wt_h_norm_cpu / ni_wt, _wt_h_norm_ul / ni_wt);
            FACTORNET_GPU_LOG(true, "  norm-W(none):  %7.3fms/iter  (sync=%.3f  cpu=%.3f  upload=%.3f)\n",
                    (w_none_total) / ni_wt,
                    _wt_w_norm_sync / ni_wt, _wt_w_norm_cpu / ni_wt, _wt_w_norm_ul / ni_wt);
        }
        FACTORNET_GPU_LOG(true, "  rhs path:  %s\n",
            "cuSPARSE SpMM (CSR_ALG2, preprocessed)");
        FACTORNET_GPU_LOG(true, "  norm path: %s\n",
            config.norm_type == ::FactorNet::NormType::L2 ? "L2 (GPU scale_rows)"
          : config.norm_type == ::FactorNet::NormType::L1 ? "L1 (GPU l1_norm_rows — no PCIe)"
          : "None (GPU d=ones, no PCIe)");
        FACTORNET_GPU_LOG(true, "  solver:    mode=%d  cd_maxit=%d\n",
            config.solver_mode, config.cd_max_iter);

        // Merge existing accumulator data into structured timer
        _prof_timer.sections["gram_H"] = {static_cast<float>(_t_gram_h), ni};
        _prof_timer.sections["rhs_H"] = {static_cast<float>(_t_rhs_h), ni};
        _prof_timer.sections["nnls_H"] = {static_cast<float>(_t_nnls_h), ni};
        _prof_timer.sections["scaling_H"] = {static_cast<float>(_t_norm_h), ni};
        _prof_timer.sections["gram_W"] = {static_cast<float>(_t_gram_w), ni};
        _prof_timer.sections["rhs_W"] = {static_cast<float>(_t_rhs_w), ni};
        _prof_timer.sections["nnls_W"] = {static_cast<float>(_t_nnls_w), ni};
        _prof_timer.sections["scaling_W"] = {static_cast<float>(_t_norm_w), ni};
        _prof_timer.sections["loss"] = {static_cast<float>(_t_loss), ni};

        // Print structured profiling summary
        _prof_timer.report(ni, "GPU NMF (structured)");

        CUDA_CHECK(cudaEventDestroy(_ev_start));
        CUDA_CHECK(cudaEventDestroy(_ev_stop));
    }
    _prof_timer.destroy();
#undef GPU_PBEGIN
#undef GPU_PEND
#undef _WC_NOW

    // ------------------------------------------------------------------
    // 6. Download final factors to host
    // ------------------------------------------------------------------

    ctx.sync();

    // W: download k×m, transpose to m×k
    gpu_detail::download_to_eigen(d_W, h_W_T);
    result.W = h_W_T.transpose();  // k×m → m×k

    // H: download k×n
    gpu_detail::download_to_eigen(d_H, h_H);
    result.H = h_H;

    // d: download k
    h_d.resize(k);
    d_d.download(h_d.data(), k);
    result.d = h_d;

    // Dispersion parameters: download from device to result
    if (d_disp_vec.size() > 0) {
        DenseVector<Scalar> h_disp_final(m);
        d_disp_vec.download(h_disp_final.data(), m);
        if (is_gp_gpu) {
            result.theta = h_disp_final;
        } else if (is_nb_gpu) {
            result.theta = h_disp_final;
        } else if (is_gamma_gpu || is_invgauss_gpu || is_tweedie_gpu) {
            result.dispersion = h_disp_final;
        }
    }

    // ZI pi vectors
    if (d_pi_row.size() > 0) {
        DenseVector<Scalar> h_pi_row(m);
        d_pi_row.download(h_pi_row.data(), m);
        result.pi_row = h_pi_row;
    }
    if (d_pi_col.size() > 0) {
        DenseVector<Scalar> h_pi_col(n);
        d_pi_col.download(h_pi_col.data(), n);
        result.pi_col = h_pi_col;
    }

    if (result.train_loss == 0 && !result.loss_history.empty())
        result.train_loss = result.loss_history.back();
    else if (result.train_loss == 0)
        result.train_loss = prev_loss;

    if (config.sort_model)
        result.sort();

    return result;
}

// ========================================================================
// Public API: upload from host
// ========================================================================

/**
 * @brief GPU-accelerated NMF from host CSC arrays (standard path).
 *
 * Uploads the sparse matrix A from host to device, then runs
 * the full-featured NMF loop on GPU.
 */
template<typename Scalar>
::FactorNet::NMFResult<Scalar> nmf_fit_gpu(
    const int* A_csc_ptr,
    const int* A_row_idx,
    const Scalar* A_values,
    int m, int n, int nnz,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    using namespace FactorNet::gpu;

    ::FactorNet::gpu::SparseMatrixGPU<Scalar> d_A(m, n, nnz, A_csc_ptr, A_row_idx, A_values);

    return nmf_fit_gpu_impl(d_A, A_csc_ptr, A_row_idx, A_values,
                            m, n, nnz, config, W_init, H_init);
}

// ========================================================================
// Public API: zero-copy from device pointers (sp_read_gpu path)
// ========================================================================

/**
 * @brief GPU NMF using data already on device (zero-copy).
 *
 * Wraps existing device pointers in a non-owning SparseMatrixGPU.
 * Downloads A to host for host-mediated paths (IRLS, masking).
 * Avoids the PCIe upload step for the main data matrix.
 */
template<typename Scalar>
::FactorNet::NMFResult<Scalar> nmf_fit_gpu_zerocopy(
    int* d_col_ptr,
    int* d_row_idx,
    Scalar* d_values,
    int m, int n, int nnz,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    using namespace FactorNet::gpu;

    // Wrap device pointers without allocating/copying (non-owning)
    ::FactorNet::gpu::SparseMatrixGPU<Scalar> d_A = ::FactorNet::gpu::SparseMatrixGPU<Scalar>::from_device_ptrs(
        m, n, nnz, d_col_ptr, d_row_idx, d_values);

    // Download to host for host-mediated paths (IRLS, masking, Eigen::Map)
    std::vector<int> h_col_ptr(n + 1);
    std::vector<int> h_row_idx(nnz);
    std::vector<Scalar> h_values(nnz);
    CUDA_CHECK(cudaMemcpy(h_col_ptr.data(), d_col_ptr,
                          (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_idx.data(), d_row_idx,
                          nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values,
                          nnz * sizeof(Scalar), cudaMemcpyDeviceToHost));

    return nmf_fit_gpu_impl(d_A, h_col_ptr.data(), h_row_idx.data(),
                            h_values.data(), m, n, nnz, config,
                            W_init, H_init);
}


}  // namespace nmf
}  // namespace FactorNet
