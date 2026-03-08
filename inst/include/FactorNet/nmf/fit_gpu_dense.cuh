// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/fit_gpu_dense.cuh
 * @brief GPU NMF for dense input matrices
 *
 * Counterpart to fit_gpu_unified.cuh for dense (non-sparse) input.
 * Uses cuBLAS GEMM directly for RHS computation — no sparse kernels needed.
 * All other operations (Gram, NNLS, loss, features) are identical.
 *
 * Data residency:
 *   - A (dense m×n): uploaded once, stays on device
 *   - A^T: computed via CUBLAS_OP_T on A (no separate upload needed)
 *   - W (k×m), H (k×n): on device, stay resident
 *   - G (k×k), B (k×max(m,n)): on device
 *   - d (k): on device, downloaded at end
 *
 * Memory: O(2·m·n + k·(m+n)) on GPU — the full dense matrix is resident.
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

// GPU Cholesky NNLS primitives
#include <FactorNet/primitives/gpu/cholesky_nnls.cuh>

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

// GPU-native Tier 2 features (angular, L21)
#include <FactorNet/nmf/gpu_features.cuh>

// IRLS for non-MSE losses (host-mediated via CPU solver)
// NOTE: Dense GPU path supports NB/GP dispersion estimation (host-mediated)
//   and zero-inflation (GPU-native ZI E-step + IRLS on A_imputed).
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>

// GPU-native ZI (zero-inflation): E-step kernel, transpose, IRLS on A_imputed
#include <FactorNet/primitives/gpu/nnls_batch_zi_irls.cuh>

// CPU fit_unified.hpp for mask_detail (host-mediated masking)
#include <FactorNet/nmf/fit_cpu.hpp>

// GPU profiling timer
#include <FactorNet/profiling/gpu_timer.cuh>

// RNG
#include <FactorNet/rng/rng.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace FactorNet {
namespace nmf {


// ---------------------------------------------------------------------------
// cuBLAS GEMM dispatch for dense-input RHS: B(k×n) = W(k×m) · A(m×n)
// ---------------------------------------------------------------------------
namespace dense_detail {
inline void dense_gemm_dispatch(cublasHandle_t handle,
    int k, int n, int m,
    const float* W, const float* A, float* B) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        k, n, m, &alpha, W, k, A, m, &beta, B, k);
}
inline void dense_gemm_dispatch(cublasHandle_t handle,
    int k, int n, int m,
    const double* W, const double* A, double* B) {
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        k, n, m, &alpha, W, k, A, m, &beta, B, k);
}
// Transposed variant: B(k×n) = W(k×m) · A^T where A is stored as n×m col-major
// Uses CUBLAS_OP_T on second operand to avoid explicit transpose storage.
inline void dense_gemm_dispatch_T(cublasHandle_t handle,
    int k, int n, int m,
    const float* W, const float* A, float* B) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        k, n, m, &alpha, W, k, A, n, &beta, B, k);
}
inline void dense_gemm_dispatch_T(cublasHandle_t handle,
    int k, int n, int m,
    const double* W, const double* A, double* B) {
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        k, n, m, &alpha, W, k, A, n, &beta, B, k);
}
}  // namespace dense_detail

/**
 * @brief GPU-accelerated NMF for dense input matrices
 *
 * Same algorithm as nmf_fit_gpu but accepts a dense matrix directly,
 * avoiding the sparse → dense materialization cost. Uses cuBLAS GEMM
 * for all RHS computations.
 *
 * @tparam Scalar float or double
 * @param A_data     Dense matrix data (m × n, column-major, host)
 * @param m          Rows of A
 * @param n          Columns of A
 * @param config     Unified NMF configuration
 * @param W_init     Optional W initialization (m × k). nullptr = random.
 * @param H_init     Optional H initialization (k × n). nullptr = random.
 * @return           NMFResult with W (m×k), d (k), H (k×n)
 */
template<typename Scalar>
::FactorNet::NMFResult<Scalar> nmf_fit_gpu_dense(
    const Scalar* A_data,
    int m, int n,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    using namespace FactorNet::gpu;

    const int k = config.rank;

    // ------------------------------------------------------------------
    // 1. Create GPU context and guard
    // ------------------------------------------------------------------

    GPUContext ctx;
    primitives::gpu_detail::GPUGuard guard;

    // ------------------------------------------------------------------
    // 2. Upload dense A and A^T to device
    // ------------------------------------------------------------------

    // A is m×n column-major: A[i,j] = A_data[i + j*m]
    DenseMatrixGPU<Scalar> d_A(m, n, A_data);
    // Note: A^T is NOT uploaded separately. W-update RHS uses CUBLAS_OP_T
    // on d_A directly, saving ~50% GPU memory and PCIe upload time.

    // ------------------------------------------------------------------
    // 3. Initialize factors on host, then upload to device
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> h_W_T, h_H;  // host copies
    DenseVector<Scalar> h_d;

    if (W_init && H_init) {
        h_W_T = W_init->transpose();  // m×k → k×m
        h_H = *H_init;
        h_d = DenseVector<Scalar>::Ones(k);
    } else if (config.init_mode == 1) {
        // Lanczos SVD initialization on host using dense matrix
        Eigen::Map<const DenseMatrix<Scalar>> h_A_map(A_data, m, n);
        detail::initialize_lanczos(h_W_T, h_H, h_d, h_A_map, k, m, n,
                                   config.seed, config.threads, config.verbose);
    } else {
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

    // Host buffers for feature application
    DenseMatrix<Scalar> h_G(k, k);

    // ------------------------------------------------------------------
    // 4b. GPU graph regularization handles (cuSPARSE SpMM + cuBLAS GEMM)
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

    // ------------------------------------------------------------------
    // 4c. Zero-inflation buffers (GPU-resident A_imputed, π vectors)
    // ------------------------------------------------------------------

    const bool is_zi = (config.loss.type == LossType::GP || config.loss.type == LossType::NB)
                       && config.has_zi();
    DenseMatrixGPU<Scalar> d_A_imputed;     // m × n dense (ZI-imputed A)
    DenseMatrixGPU<Scalar> d_A_imputed_T;   // n × m dense (transposed for W-update)
    DeviceMemory<Scalar> d_pi_row;
    DeviceMemory<Scalar> d_pi_col;
    DenseMatrixGPU<Scalar> d_W_Td_zi;
    DeviceMemory<Scalar> d_disp_vec;        // device copy of theta for GPU ZI E-step

    if (is_zi) {
        // Memory guard: ZI requires 2 dense m×n matrices on GPU
        const size_t zi_dense_bytes = static_cast<size_t>(m) * n * sizeof(Scalar) * 2;
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (zi_dense_bytes > free_mem * 8 / 10) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "ZI-NMF requires %.1f GB for dense A_imputed matrices "
                "(m=%d, n=%d) but only %.1f GB free on GPU.",
                zi_dense_bytes / (1024.0 * 1024.0 * 1024.0), m, n,
                free_mem / (1024.0 * 1024.0 * 1024.0));
            throw std::runtime_error(buf);
        }

        d_A_imputed = DenseMatrixGPU<Scalar>(m, n);
        d_A_imputed_T = DenseMatrixGPU<Scalar>(n, m);

        // Initialize A_imputed from dense A (D2D copy)
        CUDA_CHECK(cudaMemcpyAsync(d_A_imputed.data.get(), d_A.data.get(),
            static_cast<size_t>(m) * n * sizeof(Scalar),
            cudaMemcpyDeviceToDevice, ctx.stream));

        // Initialize π vectors from dense zero rates
        const int zi_mode_int = static_cast<int>(config.zi_mode);
        if (zi_mode_int == 1 || zi_mode_int == 3) {  // ZI_ROW or ZI_TWOWAY
            d_pi_row.allocate(m);
            std::vector<Scalar> h_pi_row(m);
            for (int i = 0; i < m; ++i) {
                int nz = 0;
                for (int j2 = 0; j2 < n; ++j2)
                    if (A_data[i + static_cast<size_t>(j2) * m] != Scalar(0)) ++nz;
                Scalar zero_rate = Scalar(1) - static_cast<Scalar>(nz) / static_cast<Scalar>(n);
                h_pi_row[i] = std::min(zero_rate * Scalar(0.5), Scalar(0.3));
            }
            d_pi_row.upload(h_pi_row.data(), m);
        }
        if (zi_mode_int == 2 || zi_mode_int == 3) {  // ZI_COL or ZI_TWOWAY
            d_pi_col.allocate(n);
            std::vector<Scalar> h_pi_col(n);
            for (int j2 = 0; j2 < n; ++j2) {
                int nz = 0;
                for (int i2 = 0; i2 < m; ++i2)
                    if (A_data[i2 + static_cast<size_t>(j2) * m] != Scalar(0)) ++nz;
                Scalar zero_rate = Scalar(1) - static_cast<Scalar>(nz) / static_cast<Scalar>(m);
                h_pi_col[j2] = std::min(zero_rate * Scalar(0.5), Scalar(0.3));
            }
            d_pi_col.upload(h_pi_col.data(), n);
        }

        d_W_Td_zi = DenseMatrixGPU<Scalar>(k, m);
    }

    // ------------------------------------------------------------------
    // 5. IRLS pre-computation (host-mediated non-MSE loss support)
    // ------------------------------------------------------------------

    const bool use_irls = config.requires_irls();
    const bool use_mask = config.has_mask();

    // Create Eigen wraps of host data for IRLS
    Eigen::Map<const DenseMatrix<Scalar>> h_A_map(A_data, m, n);
    DenseMatrix<Scalar> h_A_T;
    if (use_irls) {
        h_A_T = h_A_map.transpose();  // n × m for W-update IRLS
    }

    // Pre-compute mask transpose for W-update
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
    SpMat h_mask_T;
    if (use_mask) {
        h_mask_T = config.mask->transpose();
        h_mask_T.makeCompressed();
    }

    // ------------------------------------------------------------------
    // 5b. Dispersion vectors for IRLS (NB size, GP theta)
    // ------------------------------------------------------------------

    const bool is_gp = (config.loss.type == LossType::GP);
    const bool is_nb = (config.loss.type == LossType::NB);

    DenseVector<Scalar> h_theta_vec;  // GP theta or NB size (r)

    if (is_nb) {
        if (config.gp_dispersion == DispersionMode::PER_ROW) {
            h_theta_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            h_theta_vec = DenseVector<Scalar>::Constant(n, config.nb_size_init);
        } else if (config.gp_dispersion == DispersionMode::GLOBAL) {
            h_theta_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        } else {
            h_theta_vec = DenseVector<Scalar>::Constant(m, config.nb_size_max);
        }
    }

    if (is_gp) {
        if (config.gp_dispersion == DispersionMode::NONE) {
            h_theta_vec = DenseVector<Scalar>::Zero(m);
        } else if (config.gp_dispersion == DispersionMode::PER_COL) {
            h_theta_vec = DenseVector<Scalar>::Constant(n, config.gp_theta_init);
        } else {
            h_theta_vec = DenseVector<Scalar>::Constant(m, config.gp_theta_init);
        }
    }

    // Active loss for IRLS solver: GP uses KL weights (stable, same fixed point)
    FactorNet::LossConfig<Scalar> irls_active_loss = config.loss;
    if (is_gp) irls_active_loss.type = LossType::KL;

    // ------------------------------------------------------------------
    // 5c. IRLS prep: upload initial dispersion for GPU ZI path
    // ------------------------------------------------------------------

    if (use_irls && is_nb && h_theta_vec.size() > 0 && d_disp_vec.size() == 0) {
        d_disp_vec.allocate(h_theta_vec.size());
        d_disp_vec.upload(h_theta_vec.data(), h_theta_vec.size());
    }

    // ------------------------------------------------------------------
    // 6. Iteration loop
    // ------------------------------------------------------------------

    ::FactorNet::NMFResult<Scalar> result;
    Scalar prev_loss = std::numeric_limits<Scalar>::max();
    int patience_counter = 0;

    // GPU section profiling timer
    FactorNet::profiling::GpuSectionTimer _timer;
    _timer.init();
    _timer.active = config.verbose;

    // Precompute ||A||² once (constant for dense input) — used by GPU Gram trick loss
    const double A_sq_norm = FactorNet::gpu::compute_dense_sq_norm_gpu(ctx, d_A, d_scratch);

    // Pre-allocate d-scaled W buffer for projective H-update (reused every iteration)
    const bool is_projective =
        (nmf::variant::h_update_mode(config) == nmf::variant::HUpdateMode::PROJECTIVE);
    DenseMatrixGPU<Scalar> d_W_scaled;
    if (is_projective) d_W_scaled = DenseMatrixGPU<Scalar>(k, m);

    for (int iter = 0; iter < config.max_iter; ++iter) {

        // ==============================================================
        // H update  (unit-norm W — matches CPU approach)
        // ==============================================================

        bool do_loss = (iter % config.loss_every == 0)
                     || (iter == config.max_iter - 1);

        const auto h_mode = nmf::variant::h_update_mode(config);

        if (h_mode == nmf::variant::HUpdateMode::PROJECTIVE) {
            // PROJECTIVE: H = diag(d) · W · A — GPU-native (zero PCIe transfers)
            // 1. Copy W → W_scaled, then scale rows: W_scaled[i,:] *= d[i]
            CUDA_CHECK(cudaMemcpyAsync(
                d_W_scaled.data.get(), d_W.data.get(),
                sizeof(Scalar) * k * m, cudaMemcpyDeviceToDevice, ctx.stream));
            ::FactorNet::gpu::scale_rows_by_d_gpu(ctx, d_W_scaled, d_d);

            // 2. H = W_scaled · A (cuBLAS GEMM, no transfers)
            dense_detail::dense_gemm_dispatch(ctx.cublas, k, n, m,
                d_W_scaled.data.get(), d_A.data.get(), d_H.data.get());

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

        } else if (use_mask) {
            // --- Masked H-update: host-mediated per-column NNLS ---
            ctx.sync();
            gpu_detail::download_to_eigen(d_W, h_W_T);
            gpu_detail::download_to_eigen(d_H, h_H);

            DenseMatrix<Scalar> G_base = h_W_T * h_W_T.transpose();

            nmf::mask_detail::masked_nnls_h<Scalar>(
                h_A_map, h_W_T, G_base, h_H, *config.mask, config,
                config.threads > 0 ? config.threads : 1, iter > 0);

            if (config.H.upper_bound > 0)
                features::apply_upper_bound(h_H, config.H.upper_bound);

            gpu_detail::upload_from_eigen(d_H, h_H);

        } else if (use_irls) {
            // --- IRLS H-update ---
            if (is_zi && iter > 0) {
                // ZI path: GPU-native dense IRLS on A_imputed
                const DeviceMemory<Scalar>* disp_row_h =
                    d_disp_vec.size() > 0 ? &d_disp_vec : nullptr;
                primitives::gpu_zi::nnls_batch_zi_irls_gpu(
                    ctx, d_A_imputed, d_W, d_H,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    disp_row_h,
                    static_cast<const DeviceMemory<Scalar>*>(nullptr),
                    static_cast<Scalar>(config.H.L1),
                    static_cast<Scalar>(config.H.L2),
                    config.H.nonneg,
                    config.cd_max_iter,
                    config.irls_max_iter,
                    static_cast<Scalar>(config.cd_tol),
                    static_cast<Scalar>(config.irls_tol),
                    config.threads > 0 ? config.threads : 1);

                if (config.H.upper_bound > 0)
                    ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_H,
                        static_cast<Scalar>(config.H.upper_bound));
            } else {
                // Host-mediated CPU IRLS on original A (non-ZI or first ZI iter)
                ctx.sync();
                gpu_detail::download_to_eigen(d_W, h_W_T);
                gpu_detail::download_to_eigen(d_H, h_H);
                DenseMatrix<Scalar> G_irls = h_W_T * h_W_T.transpose();

                const Scalar* irls_theta_ptr = nullptr;
                const Scalar* irls_theta_col_ptr = nullptr;
                if (is_nb) {
                    if (config.gp_dispersion == DispersionMode::PER_COL)
                        irls_theta_col_ptr = h_theta_vec.data();
                    else
                        irls_theta_ptr = h_theta_vec.data();
                }

                primitives::nnls_batch_irls_dense(
                    h_A_map, h_W_T, G_irls, h_H, irls_active_loss,
                    static_cast<Scalar>(config.H.L1),
                    static_cast<Scalar>(config.H.L2),
                    config.H.nonneg,
                    config.cd_max_iter, static_cast<Scalar>(config.cd_tol),
                    config.irls_max_iter, static_cast<Scalar>(config.irls_tol),
                    config.threads > 0 ? config.threads : 1,
                    irls_theta_ptr, irls_theta_col_ptr);

                if (config.H.upper_bound > 0)
                    features::apply_upper_bound(h_H, config.H.upper_bound);

                gpu_detail::upload_from_eigen(d_H, h_H);
            }

        } else {
            // --- Standard GPU MSE H-update ---

            // 1. Gram: G = WᵀW + L2*I  (unit-norm W, well-conditioned)
            _timer.begin(ctx.stream, "gram_H");
            compute_gram_regularized_gpu(ctx, d_W, d_G,
                                         static_cast<Scalar>(config.H.L2));
            _timer.end(ctx.stream);

            // 2. RHS: B(k×n) = W(k×m) · A(m×n) via cuBLAS GEMM
            _timer.begin(ctx.stream, "gemm_rhs_H");
            dense_detail::dense_gemm_dispatch(ctx.cublas,
                k, n, m,
                d_W.data.get(), d_A.data.get(), d_B.data.get());
            _timer.end(ctx.stream);

            // Save B for Gram-trick loss (pre-feature, pre-L1)
            if (do_loss) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_B_save.data.get(), d_B.data.get(),
                    static_cast<size_t>(k) * n * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
            }

            // 3. Features on B (RHS)
            if (config.H.L1 > 0) {
                apply_l1_gpu(ctx, d_B, static_cast<Scalar>(config.H.L1));
            }

            // Tier 2 features: GPU-native angular/L21, host fallback for graph
            _timer.begin(ctx.stream, "features_tier2_H");
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
            _timer.end(ctx.stream);

            // 4. NNLS solve: dispatch by solver_mode
            _timer.begin(ctx.stream, "nnls_H");
            if (config.solver_mode == 1) {
                // GPU-native Cholesky: pure GPU solve, zero transfers
                primitives::cholesky_clip_batch_gpu(ctx, d_G, d_B, d_H, config.H.nonneg);
            } else {
                // Fixed-iteration batch CD: default GPU solver
                batch_cd_nnls_gpu(ctx, d_G, d_B, d_H, config.cd_max_iter,
                                  config.H.nonneg, /*warm_start=*/(iter > 0));
            }
            _timer.end(ctx.stream);

            // 5. Upper bounds — GPU clamp kernel (no PCIe round-trip)
            if (config.H.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_H,
                    static_cast<Scalar>(config.H.upper_bound));
        }

        // 6. Scale H → d  (skip for projective [already normalized] and symmetric [no-op])
        _timer.begin(ctx.stream, "scaling_H");
        if (h_mode == nmf::variant::HUpdateMode::STANDARD) {
            scale_rows_opt_gpu(ctx, d_H, d_d);
        }
        _timer.end(ctx.stream);

        // ==============================================================
        // W update  (unit-norm H — matches CPU approach)
        // ==============================================================

        const auto w_mode = nmf::variant::w_update_mode(config);

        if (w_mode == nmf::variant::WUpdateMode::SYMMETRIC) {
            // SYMMETRIC W-update: Gram(W), ForwardRHS(W,A), features, NNLS
            compute_gram_regularized_gpu(ctx, d_W, d_G,
                                         static_cast<Scalar>(config.W.L2));

            // Forward RHS: B(k×n) = W(k×m) · A(m×n) via cuBLAS GEMM
            dense_detail::dense_gemm_dispatch(ctx.cublas,
                k, n, m,
                d_W.data.get(), d_A.data.get(), d_B.data.get());

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

            // Upper bounds — GPU clamp kernel (no PCIe round-trip)
            if (config.W.upper_bound > 0)
                ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                    static_cast<Scalar>(config.W.upper_bound));

        } else if (use_mask) {
            // --- Masked W-update: host-mediated per-column NNLS ---
            ctx.sync();
            gpu_detail::download_to_eigen(d_H, h_H);
            gpu_detail::download_to_eigen(d_W, h_W_T);

            DenseMatrix<Scalar> G_base = h_H * h_H.transpose();

            nmf::mask_detail::masked_nnls_w<Scalar>(
                h_A_map, h_H, G_base, h_W_T, h_mask_T, config,
                config.threads > 0 ? config.threads : 1, iter > 0);

            if (config.W.upper_bound > 0)
                features::apply_upper_bound(h_W_T, config.W.upper_bound);

            gpu_detail::upload_from_eigen(d_W, h_W_T);

        } else if (use_irls) {
            // --- IRLS W-update ---
            if (is_zi && iter > 0) {
                // ZI path: transpose A_imputed, then GPU-native dense IRLS
                primitives::gpu_zi::transpose_dense_gpu(ctx, d_A_imputed, d_A_imputed_T);

                const DeviceMemory<Scalar>* disp_col_w =
                    d_disp_vec.size() > 0 ? &d_disp_vec : nullptr;
                primitives::gpu_zi::nnls_batch_zi_irls_gpu(
                    ctx, d_A_imputed_T, d_H, d_W,
                    static_cast<int>(config.loss.type),
                    config.loss.huber_delta,
                    static_cast<const DeviceMemory<Scalar>*>(nullptr),
                    disp_col_w,              // per-column = per-row of A
                    static_cast<Scalar>(config.W.L1),
                    static_cast<Scalar>(config.W.L2),
                    config.W.nonneg,
                    config.cd_max_iter,
                    config.irls_max_iter,
                    static_cast<Scalar>(config.cd_tol),
                    static_cast<Scalar>(config.irls_tol),
                    config.threads > 0 ? config.threads : 1);

                if (config.W.upper_bound > 0)
                    ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                        static_cast<Scalar>(config.W.upper_bound));
            } else {
                // Host-mediated CPU IRLS on A^T (non-ZI or first ZI iteration)
                ctx.sync();
                gpu_detail::download_to_eigen(d_H, h_H);
                gpu_detail::download_to_eigen(d_W, h_W_T);
                DenseMatrix<Scalar> G_irls_w = h_H * h_H.transpose();

                const Scalar* irls_theta_ptr_w = nullptr;
                const Scalar* irls_theta_col_ptr_w = nullptr;
                if (is_nb) {
                    // For W-update on A^T: rows of A^T = cols of A
                    if (config.gp_dispersion == DispersionMode::PER_COL)
                        irls_theta_ptr_w = h_theta_vec.data();    // per-col of A = per-row of A^T
                    else
                        irls_theta_col_ptr_w = h_theta_vec.data(); // per-row of A = per-col of A^T
                }

                primitives::nnls_batch_irls_dense(
                    h_A_T, h_H, G_irls_w, h_W_T, irls_active_loss,
                    static_cast<Scalar>(config.W.L1),
                    static_cast<Scalar>(config.W.L2),
                    config.W.nonneg,
                    config.cd_max_iter, static_cast<Scalar>(config.cd_tol),
                    config.irls_max_iter, static_cast<Scalar>(config.irls_tol),
                    config.threads > 0 ? config.threads : 1,
                    irls_theta_ptr_w, irls_theta_col_ptr_w);

                if (config.W.upper_bound > 0)
                    features::apply_upper_bound(h_W_T, config.W.upper_bound);

                gpu_detail::upload_from_eigen(d_W, h_W_T);
            }

        } else {
            // --- Standard GPU MSE W-update ---

            // 1. Gram: G = HᵀH + L2*I  (unit-norm H, well-conditioned)
            _timer.begin(ctx.stream, "gram_W");
            compute_gram_regularized_gpu(ctx, d_H, d_G,
                                         static_cast<Scalar>(config.W.L2));
            _timer.end(ctx.stream);

            // 2. RHS: B(k×m) = H(k×n) · A^T via cuBLAS GEMM with CUBLAS_OP_T
            _timer.begin(ctx.stream, "gemm_rhs_W");
            dense_detail::dense_gemm_dispatch_T(ctx.cublas,
                k, m, n,
                d_H.data.get(), d_A.data.get(), d_B.data.get());
            _timer.end(ctx.stream);

            // 3. Features on B (RHS)
            if (config.W.L1 > 0) {
                apply_l1_gpu(ctx, d_B, static_cast<Scalar>(config.W.L1));
            }

            // Tier 2 features: GPU-native angular/L21, host fallback for graph
            _timer.begin(ctx.stream, "features_tier2_W");
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
            _timer.end(ctx.stream);

            // 4. NNLS solve: dispatch by solver_mode
            _timer.begin(ctx.stream, "nnls_W");
            if (config.solver_mode == 1) {
                // GPU-native Cholesky for standard W
                primitives::cholesky_clip_batch_gpu(ctx, d_G, d_B, d_W, config.W.nonneg);
            } else {
                // Fixed-iteration batch CD: default GPU solver
                batch_cd_nnls_gpu(ctx, d_G, d_B, d_W, config.cd_max_iter,
                                  config.W.nonneg, /*warm_start=*/(iter > 0));
            }
            _timer.end(ctx.stream);

            // 5. Upper bounds
            if (config.W.upper_bound > 0) {
                ctx.sync();
                gpu_detail::download_to_eigen(d_W, h_W_T);
                features::apply_upper_bound(h_W_T, config.W.upper_bound);
                gpu_detail::upload_from_eigen(d_W, h_W_T);
            }
        }

        // 6. Scale W → d
        _timer.begin(ctx.stream, "scaling_W");
        scale_rows_opt_gpu(ctx, d_W, d_d);
        _timer.end(ctx.stream);

        // Enforce H = W (symmetric only, on device)
        if (w_mode == nmf::variant::WUpdateMode::SYMMETRIC) {
            CUDA_CHECK(cudaMemcpyAsync(d_H.data.get(), d_W.data.get(),
                static_cast<size_t>(k) * n * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        }

        // ==============================================================
        // NB dispersion estimation (method of moments, host-mediated)
        // ==============================================================

        if (is_nb && config.gp_dispersion != DispersionMode::NONE) {
            ctx.sync();
            gpu_detail::download_to_eigen(d_W, h_W_T);
            gpu_detail::download_to_eigen(d_H, h_H);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h_d_vec(k);
            d_d.download(h_d_vec.data(), k);

            // W_Td = W * diag(d)
            DenseMatrix<Scalar> W_Td_nb = h_W_T;
            for (int f = 0; f < k; ++f) W_Td_nb.row(f) *= h_d_vec(f);

            double r_min = static_cast<double>(config.nb_size_min);
            double r_max = static_cast<double>(config.nb_size_max);

            if (config.gp_dispersion == DispersionMode::PER_COL) {
                for (int j = 0; j < n; ++j) {
                    double sum_mu_sq = 0, sum_excess_var = 0;
                    for (int i = 0; i < m; ++i) {
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(i).dot(h_H.col(j))), 1e-10);
                        double y = static_cast<double>(h_A_map(i, j));
                        double resid = y - mu;
                        sum_mu_sq += mu * mu;
                        sum_excess_var += resid * resid - mu;
                    }
                    if (sum_excess_var > 1e-10 && sum_mu_sq > 1e-10) {
                        double r_new = sum_mu_sq / sum_excess_var;
                        r_new = std::max(r_min, std::min(r_new, r_max));
                        if (std::isfinite(r_new))
                            h_theta_vec(j) = static_cast<Scalar>(r_new);
                    } else {
                        h_theta_vec(j) = static_cast<Scalar>(r_max);
                    }
                }
            } else {
                // PER_ROW or GLOBAL
                for (int i = 0; i < m; ++i) {
                    double sum_mu_sq = 0, sum_excess_var = 0;
                    for (int j = 0; j < n; ++j) {
                        double mu = std::max(static_cast<double>(
                            W_Td_nb.col(i).dot(h_H.col(j))), 1e-10);
                        double y = static_cast<double>(h_A_map(i, j));
                        double resid = y - mu;
                        sum_mu_sq += mu * mu;
                        sum_excess_var += resid * resid - mu;
                    }
                    if (sum_excess_var > 1e-10 && sum_mu_sq > 1e-10) {
                        double r_new = sum_mu_sq / sum_excess_var;
                        r_new = std::max(r_min, std::min(r_new, r_max));
                        if (std::isfinite(r_new))
                            h_theta_vec(i) = static_cast<Scalar>(r_new);
                    } else {
                        h_theta_vec(i) = static_cast<Scalar>(r_max);
                    }
                }
                if (config.gp_dispersion == DispersionMode::GLOBAL) {
                    std::vector<Scalar> r_vals(h_theta_vec.data(), h_theta_vec.data() + m);
                    std::nth_element(r_vals.begin(), r_vals.begin() + m / 2, r_vals.end());
                    h_theta_vec.setConstant(r_vals[m / 2]);
                }
            }
        }

        // ==============================================================
        // GP theta estimation (MM update, per-row, host-mediated)
        // ==============================================================

        if (is_gp && config.gp_dispersion != DispersionMode::NONE) {
            ctx.sync();
            gpu_detail::download_to_eigen(d_W, h_W_T);
            gpu_detail::download_to_eigen(d_H, h_H);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h_d_vec(k);
            d_d.download(h_d_vec.data(), k);

            DenseMatrix<Scalar> W_Td_gp = h_W_T;
            for (int f = 0; f < k; ++f) W_Td_gp.row(f) *= h_d_vec(f);

            // Reconstruct S = W_Td^T * H  (m × n)
            DenseMatrix<Scalar> S = W_Td_gp.transpose() * h_H;

            const int disp_dim = (config.gp_dispersion == DispersionMode::PER_COL) ? n : m;
            double cap = static_cast<double>(config.gp_theta_max);

            if (config.gp_dispersion == DispersionMode::PER_COL) {
                // PER_COL MM update
                struct NzEntryCol { int col; double y; double s; };
                std::vector<NzEntryCol> nz_cache_col;
                Eigen::VectorXd sum_y_col = Eigen::VectorXd::Zero(n);
                Eigen::VectorXd sum_s_col = Eigen::VectorXd::Zero(n);
                Eigen::VectorXi n_nz_col = Eigen::VectorXi::Zero(n);

                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        double y = static_cast<double>(h_A_map(i, j));
                        double s = std::max(static_cast<double>(S(i, j)), 1e-10);
                        sum_y_col(j) += y;
                        sum_s_col(j) += s;
                        if (y >= 1.0) { n_nz_col(j)++; nz_cache_col.push_back({j, y, s}); }
                    }
                }

                constexpr int THETA_INNER_ITERS_COL = 5;
                for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS_COL; ++mm_iter) {
                    Eigen::VectorXd alpha_col = Eigen::VectorXd::Zero(n);
                    Eigen::VectorXd gamma_col = Eigen::VectorXd::Zero(n);
                    for (const auto& nz : nz_cache_col) {
                        if (nz.y >= 1.0) {
                            double th = static_cast<double>(h_theta_vec(nz.col));
                            double denom = std::max(nz.s + th * nz.y, 1e-10);
                            double eta1 = nz.s / denom;
                            alpha_col(nz.col) += (nz.y - 1.0) * eta1;
                            gamma_col(nz.col) += (nz.y - 1.0) * (1.0 - eta1);
                        }
                    }
                    for (int j = 0; j < n; ++j) {
                        double alpha_j = alpha_col(j) + static_cast<double>(n_nz_col(j));
                        double beta_j = (sum_y_col(j) - sum_s_col(j)) - gamma_col(j) + alpha_j;
                        if (alpha_j > 1e-15) {
                            double disc = beta_j * beta_j + 4.0 * alpha_j * gamma_col(j);
                            if (disc > 0.0 && std::isfinite(disc)) {
                                double new_theta = (-beta_j + std::sqrt(disc)) / (2.0 * alpha_j);
                                if (std::isfinite(new_theta) && new_theta >= 0.0)
                                    h_theta_vec(j) = static_cast<Scalar>(std::min(new_theta, cap));
                            }
                        }
                    }
                }
            } else {
                // PER_ROW or GLOBAL MM update
                struct NzEntry { int row; double y; double s; };
                std::vector<NzEntry> nz_cache;
                Eigen::VectorXd sum_y_d = Eigen::VectorXd::Zero(m);
                Eigen::VectorXd sum_s_d = Eigen::VectorXd::Zero(m);
                Eigen::VectorXi n_nz = Eigen::VectorXi::Zero(m);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double y = static_cast<double>(h_A_map(i, j));
                        double s = std::max(static_cast<double>(S(i, j)), 1e-10);
                        sum_y_d(i) += y;
                        sum_s_d(i) += s;
                        if (y >= 1.0) { n_nz(i)++; nz_cache.push_back({i, y, s}); }
                    }
                }

                constexpr int THETA_INNER_ITERS = 5;
                for (int mm_iter = 0; mm_iter < THETA_INNER_ITERS; ++mm_iter) {
                    Eigen::VectorXd alpha_d = Eigen::VectorXd::Zero(m);
                    Eigen::VectorXd gamma_d = Eigen::VectorXd::Zero(m);
                    for (const auto& nz : nz_cache) {
                        if (nz.y >= 1.0) {
                            double th = static_cast<double>(h_theta_vec(nz.row));
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
                                    h_theta_vec(i) = static_cast<Scalar>(std::min(new_theta, cap));
                            }
                        }
                    }
                }

                if (config.gp_dispersion == DispersionMode::GLOBAL) {
                    Scalar mean_theta = h_theta_vec.mean();
                    h_theta_vec.setConstant(mean_theta);
                }
            }
        }

        // ==============================================================
        // ZI E-step + M-step + soft imputation (after dispersion update)
        // ==============================================================

        // Re-upload dispersion to device after host-side update (non-ZI GPU IRLS)
        if (!is_zi && is_nb && h_theta_vec.size() > 0 && d_disp_vec.size() > 0) {
            d_disp_vec.upload(h_theta_vec.data(), h_theta_vec.size());
        }

        if (is_zi) {
            // Upload updated theta to device for GPU ZI E-step
            if (h_theta_vec.size() > 0) {
                if (d_disp_vec.size() == 0)
                    d_disp_vec.allocate(h_theta_vec.size());
                d_disp_vec.upload(h_theta_vec.data(), h_theta_vec.size());
            }

            // Compute d-scaled W: W_Td_zi = W * diag(d)
            CUDA_CHECK(cudaMemcpyAsync(d_W_Td_zi.data.get(), d_W.data.get(),
                static_cast<size_t>(k) * m * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            absorb_d_gpu(ctx, d_W_Td_zi, d_d);

            primitives::gpu_zi::zi_em_step_gpu_dense(
                ctx, d_A, d_W_Td_zi, d_H, d_disp_vec,
                d_pi_row, d_pi_col,
                d_A_imputed,
                k, m, n,
                static_cast<int>(config.loss.type),
                static_cast<int>(config.zi_mode),
                config.zi_em_iters, config.gp_theta_min);
        }

        // ==============================================================
        // Loss computation (after both updates, matching CPU)
        // ==============================================================

        // Ensure B_save is available for Gram trick loss
        if (do_loss && !use_mask && !use_irls &&
            h_mode != nmf::variant::HUpdateMode::STANDARD) {
            dense_detail::dense_gemm_dispatch(ctx.cublas,
                k, n, m,
                d_W.data.get(), d_A.data.get(), d_B.data.get());
            CUDA_CHECK(cudaMemcpyAsync(
                d_B_save.data.get(), d_B.data.get(),
                static_cast<size_t>(k) * n * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        }

        if (do_loss) {
            double loss_val;

            if (use_mask) {
                // Masked loss: explicit per-element, skip masked entries (host-mediated)
                ctx.sync();
                gpu_detail::download_to_eigen(d_W, h_W_T);
                gpu_detail::download_to_eigen(d_H, h_H);

                // Apply scaling: W_Td = W * diag(d)
                DenseMatrix<Scalar> h_W_Td = h_W_T;
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h_d_vec(k);
                d_d.download(h_d_vec.data(), k);
                for (int f = 0; f < k; ++f)
                    h_W_Td.row(f) *= h_d_vec(f);

                loss_val = static_cast<double>(
                    nmf::mask_detail::masked_loss<Scalar>(
                        h_A_map, h_W_Td, h_H, *config.mask, config.loss));
            } else if (use_irls) {
                // Non-MSE loss: explicit per-element computation on host
                ctx.sync();
                gpu_detail::download_to_eigen(d_W, h_W_T);
                gpu_detail::download_to_eigen(d_H, h_H);

                // Apply scaling: W_Td = W * diag(d)
                DenseMatrix<Scalar> h_W_Td = h_W_T;
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h_d_vec(k);
                d_d.download(h_d_vec.data(), k);
                for (int f = 0; f < k; ++f)
                    h_W_Td.row(f) *= h_d_vec(f);

                // Explicit loss with dispersion
                DenseMatrix<Scalar> R = h_W_Td.transpose() * h_H;  // m × n
                const bool disp_is_per_col = (config.gp_dispersion == DispersionMode::PER_COL);
                Scalar total = 0;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        Scalar theta = (is_gp || is_nb)
                            ? (disp_is_per_col ? h_theta_vec(j) : h_theta_vec(i))
                            : static_cast<Scalar>(0);
                        total += compute_robust_loss(h_A_map(i, j), R(i, j),
                                                     config.loss, theta);
                    }
                loss_val = static_cast<double>(total);

            } else if (h_mode != nmf::variant::HUpdateMode::STANDARD) {
                // Projective/symmetric: explicit loss (Gram trick has
                // numerical issues with non-standard H updates)
                ctx.sync();
                gpu_detail::download_to_eigen(d_W, h_W_T);
                gpu_detail::download_to_eigen(d_H, h_H);

                DenseMatrix<Scalar> h_W_Td = h_W_T;
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h_d_vec(k);
                d_d.download(h_d_vec.data(), k);
                for (int f = 0; f < k; ++f)
                    h_W_Td.row(f) *= h_d_vec(f);

                DenseMatrix<Scalar> R = h_W_Td.transpose() * h_H;  // m × n
                Scalar total = 0;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        Scalar diff = h_A_map(i, j) - R(i, j);
                        total += diff * diff;
                    }
                loss_val = static_cast<double>(total);

            } else {
                // MSE: Gram trick — all on GPU (cross + recon kernels)
                _timer.begin(ctx.stream, "loss");

                CUDA_CHECK(cudaMemcpyAsync(
                    d_Hd.data.get(), d_H.data.get(),
                    static_cast<size_t>(k) * n * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
                absorb_d_gpu(ctx, d_Hd, d_d);

                compute_gram_gpu(ctx, d_W, d_G);

                loss_val = FactorNet::gpu::compute_total_loss_gpu_dense(
                    ctx, A_sq_norm, d_Hd, d_G, d_B_save, d_scratch);
                if (loss_val < 0) loss_val = 0;
                _timer.end(ctx.stream);
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

        result.iterations = iter + 1;
    }

    // Profiling report (verbose mode)
    if (config.verbose && result.iterations > 0) {
        _timer.report(result.iterations, "GPU Dense NMF");
    }
    _timer.destroy();

    // ------------------------------------------------------------------
    // 6. Download final factors to host
    // ------------------------------------------------------------------

    ctx.sync();

    gpu_detail::download_to_eigen(d_W, h_W_T);
    result.W = h_W_T.transpose();  // k×m → m×k

    gpu_detail::download_to_eigen(d_H, h_H);
    result.H = h_H;

    h_d.resize(k);
    d_d.download(h_d.data(), k);
    result.d = h_d;

    if (result.train_loss == 0 && !result.loss_history.empty())
        result.train_loss = result.loss_history.back();
    else if (result.train_loss == 0)
        result.train_loss = prev_loss;

    // Store dispersion estimates
    if (is_gp || is_nb) result.theta = h_theta_vec;

    // ZI pi vectors
    if (d_pi_row.size() > 0) {
        DenseVector<Scalar> h_pi_row_final(m);
        d_pi_row.download(h_pi_row_final.data(), m);
        result.pi_row = h_pi_row_final;
    }
    if (d_pi_col.size() > 0) {
        DenseVector<Scalar> h_pi_col_final(n);
        d_pi_col.download(h_pi_col_final.data(), n);
        result.pi_col = h_pi_col_final;
    }

    if (config.sort_model)
        result.sort();

    return result;
}


}  // namespace nmf
}  // namespace FactorNet
