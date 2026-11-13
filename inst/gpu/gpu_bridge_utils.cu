/**
 * @file gpu_bridge_utils.cu
 * @brief GPU NMF profiler entry point (phase timing)
 *
 * Part of RcppML GPU bridge. Compiled by nvcc into RcppML_gpu.so.
 * Extracted from gpu_bridge.cu (2026).
 *
 * Build: make -f Makefile.gpu
 */

#include "gpu_bridge_common.cuh"

extern "C" {

// ==========================================================================
// GPU NMF Phase Profiler v2 -- batch CD NNLS (solver_mode = 5)
//
// Instruments every NMF iteration with CUDA events and accumulates
// per-phase wall time:
//   [0]  gram_H          — G = W·Wᵀ  (GEMM, k×k)
//   [1]  rhs_H           — B = W·A   (sparse SpMM forward, k×n)
//   [2]  nnls_H          — batch CD NNLS on H  (k×n columns)
//   [3]  norm_H          — scale_rows H → d
//   [4]  gram_W          — G = H·Hᵀ  (GEMM, k×k)
//   [5]  rhs_W_atomic    — B = H·Aᵀ  atomicAdd kernel (k×m)
//   [6]  rhs_W_noatomic  — B = H·Aᵀ  forward kernel on A^T CSC (no atomics)
//   [7]  nnls_W          — batch CD NNLS on W  (uses noatomic result)
//   [8]  norm_W          — scale_rows W → d
//   [9]  loss_fast       — O(k²) GramFIP + cuBLAS-dot (precomputed ||A||²)
//   [10] total           — wall time per full iteration (noatomic path)
//
// Outputs:  out_phase_ms_total[11]    — accumulated ms per phase over all iters
//           out_phase_ms_per_iter[11] — mean ms per iteration
//           out_n_iters               — actual number of iterations run
//           out_status                — 0 = success, -1 = error
// ==========================================================================

/**
 * @brief CUDA-event profiler for GPU NMF batch-CD inner loop (v2).
 *
 * New in v2:
 *   - Precomputes A^T CSC on host, uploads once → eliminates atomicAdd in rhs_W
 *   - Precomputes ||A||² once (constant, not recomputed per iteration)
 *   - Uses compute_total_loss_fast_gpu: O(k²) recon via Gram FIP + cuBLAS dot
 *   - Separately times atomic vs non-atomic rhs_W to quantify speedup
 *   - 11 phases instead of 10
 */
void rcppml_gpu_nmf_profile_double(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m_ptr, int* n_ptr, int* nnz_ptr, int* k_ptr,
    int* max_iter_ptr, double* tol_ptr,
    int* cd_maxit_ptr, int* seed_ptr,
    double* out_phase_ms_total,    // [11] — accumulated ms per phase
    double* out_phase_ms_per_iter, // [11] — mean ms per phase per iteration
    int*    out_n_iters,
    int*    out_status)
{
    const int m = *m_ptr, n = *n_ptr, nnz = *nnz_ptr, k = *k_ptr;
    const int max_iter = *max_iter_ptr;
    const double tol = *tol_ptr;
    const int cd_maxit = *cd_maxit_ptr;
    const uint32_t seed = static_cast<uint32_t>(*seed_ptr);

    for (int p = 0; p < 11; ++p) {
        out_phase_ms_total[p] = 0.0;
        out_phase_ms_per_iter[p] = 0.0;
    }
    *out_n_iters = 0;
    *out_status = -1;

    try {
        using namespace FactorNet;
        using namespace FactorNet::gpu;
        using Scalar = double;

        // ---- GPU context ----
        GPUContext ctx;

        // ---- Upload sparse A ----
        SparseMatrixGPU<Scalar> d_A(m, n, nnz, col_ptr, row_idx, values);

        // ---- Precompute A^T CSC on host, upload to GPU once ----
        // d_A_T is n×m (A transposed): n rows, m "columns"
        std::vector<int>    At_col_ptr_h, At_row_idx_h;
        std::vector<Scalar> At_values_h;
        build_transpose_csc_host(m, n, nnz,
                                 col_ptr, row_idx, values,
                                 At_col_ptr_h, At_row_idx_h, At_values_h);
        SparseMatrixGPU<Scalar> d_A_T(n, m, nnz,
            At_col_ptr_h.data(), At_row_idx_h.data(), At_values_h.data());

        // ---- Initialize W, H with random seed ----
        rng::SplitMix64 rng_init(seed);
        DenseMatrix<Scalar> h_W_T(k, m), h_H(k, n);
        rng_init.fill_uniform(h_W_T.data(), k, m);
        rng_init.fill_uniform(h_H.data(), k, n);

        DenseMatrixGPU<Scalar> d_W(k, m, h_W_T.data());
        DenseMatrixGPU<Scalar> d_H(k, n, h_H.data());
        DeviceMemory<Scalar> d_d(k);
        {
            DenseVector<Scalar> h_d = DenseVector<Scalar>::Ones(k);
            d_d.upload(h_d.data(), k);
        }

        // ---- Allocate GPU workspace ----
        DenseMatrixGPU<Scalar> d_G(k, k);
        DenseMatrixGPU<Scalar> d_G_tmp(k, k);             // scratch for GramFIP loss
        DenseMatrixGPU<Scalar> d_B(k, std::max(m, n));
        DenseMatrixGPU<Scalar> d_B_W(k, m);               // rhs_W noatomic result
        DenseMatrixGPU<Scalar> d_B_save(k, n);            // saved rhs_H for loss
        DenseMatrixGPU<Scalar> d_Hd(k, n);
        DeviceMemory<Scalar> d_scratch(4);

        // ---- Precompute ||A||² once ----
        DeviceMemory<Scalar> d_scratch1(4);
        double A_sq = precompute_A_sq_gpu(ctx, d_A, d_scratch1);
        FACTORNET_GPU_WARN("  [profiler] ||A||^2 = %.6e  (A: %d x %d, nnz=%d)\n",
                        A_sq, m, n, nnz);

        // ---- CUDA events for 11 phases ----
        cudaEvent_t ev_start[11], ev_stop[11];
        for (int p = 0; p < 11; ++p) {
            CUDA_CHECK(cudaEventCreate(&ev_start[p]));
            CUDA_CHECK(cudaEventCreate(&ev_stop[p]));
        }

        // ---- Convergence state ----
        Scalar prev_loss = std::numeric_limits<Scalar>::max();
        int n_iters_run = 0;
        bool converged = false;

        // ---- Warm-up iteration (unchained, no timing) ----
        compute_gram_regularized_gpu(ctx, d_W, d_G, Scalar(0));
        compute_rhs_forward_gpu(ctx, d_W, d_A, d_B);
        CUDA_CHECK(cudaMemcpyAsync(d_B_save.data.get(), d_B.data.get(),
            (size_t)k * n * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
        batch_cd_nnls_gpu(ctx, d_G, d_B, d_H, cd_maxit, true, false);
        scale_rows_opt_gpu(ctx, d_H, d_d);
        compute_gram_regularized_gpu(ctx, d_H, d_G, Scalar(0));
        compute_rhs_forward_gpu(ctx, d_H, d_A_T, d_B_W);
        batch_cd_nnls_gpu(ctx, d_G, d_B_W, d_W, cd_maxit, true, false);
        scale_rows_opt_gpu(ctx, d_W, d_d);
        ctx.sync();

        // ---- Iteration loop ----
        for (int iter = 0; iter < max_iter && !converged; ++iter) {

            // ================================================================
            // [10] Total iteration — START
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[10], ctx.stream));

            // ================================================================
            // [0] Gram H  (G = W·Wᵀ)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[0], ctx.stream));
            compute_gram_regularized_gpu(ctx, d_W, d_G, Scalar(0));
            CUDA_CHECK(cudaEventRecord(ev_stop[0], ctx.stream));

            // ================================================================
            // [1] RHS H  (B = W·A, save copy for loss)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[1], ctx.stream));
            compute_rhs_forward_gpu(ctx, d_W, d_A, d_B);
            CUDA_CHECK(cudaEventRecord(ev_stop[1], ctx.stream));

            CUDA_CHECK(cudaMemcpyAsync(d_B_save.data.get(), d_B.data.get(),
                (size_t)k * n * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));

            // ================================================================
            // [2] NNLS H
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[2], ctx.stream));
            batch_cd_nnls_gpu(ctx, d_G, d_B, d_H, cd_maxit, true, (iter > 0));
            CUDA_CHECK(cudaEventRecord(ev_stop[2], ctx.stream));

            // ================================================================
            // [3] Normalize H
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[3], ctx.stream));
            scale_rows_opt_gpu(ctx, d_H, d_d);
            CUDA_CHECK(cudaEventRecord(ev_stop[3], ctx.stream));

            // ================================================================
            // [4] Gram W  (G = H·Hᵀ)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[4], ctx.stream));
            compute_gram_regularized_gpu(ctx, d_H, d_G, Scalar(0));
            CUDA_CHECK(cudaEventRecord(ev_stop[4], ctx.stream));

            // ================================================================
            // [5] rhs_W atomic  (atomicAdd transpose kernel, baseline)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[5], ctx.stream));
            compute_rhs_transpose_gpu(ctx, d_H, d_A, d_B);   // result in d_B (overwritten)
            CUDA_CHECK(cudaEventRecord(ev_stop[5], ctx.stream));

            // ================================================================
            // [6] rhs_W noatomic  (forward SpMM on A^T CSC, no atomicAdd)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[6], ctx.stream));
            compute_rhs_forward_gpu(ctx, d_H, d_A_T, d_B_W);
            CUDA_CHECK(cudaEventRecord(ev_stop[6], ctx.stream));

            // Use d_B_W (noatomic) for NNLS_W
            // ================================================================
            // [7] NNLS W  (uses noatomic rhs result)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[7], ctx.stream));
            batch_cd_nnls_gpu(ctx, d_G, d_B_W, d_W, cd_maxit, true, (iter > 0));
            CUDA_CHECK(cudaEventRecord(ev_stop[7], ctx.stream));

            // ================================================================
            // [8] Normalize W
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[8], ctx.stream));
            scale_rows_opt_gpu(ctx, d_W, d_d);
            CUDA_CHECK(cudaEventRecord(ev_stop[8], ctx.stream));

            // ================================================================
            // [9] Fast loss  (O(k²) GramFIP, cuBLAS dot, precomputed A_sq)
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_start[9], ctx.stream));

            // Build Hd = H * diag(d)
            CUDA_CHECK(cudaMemcpyAsync(d_Hd.data.get(), d_H.data.get(),
                (size_t)k * n * sizeof(Scalar), cudaMemcpyDeviceToDevice, ctx.stream));
            absorb_d_gpu(ctx, d_Hd, d_d);

            // Recompute G = W'W for loss (W was updated in place)
            compute_gram_gpu(ctx, d_W, d_G);

            double loss_val = compute_total_loss_fast_gpu(
                ctx, d_G, d_B_save, d_Hd, d_G_tmp, d_scratch, A_sq, 0, 0);

            CUDA_CHECK(cudaEventRecord(ev_stop[9], ctx.stream));

            // ================================================================
            // [10] Total iteration — STOP
            // ================================================================
            CUDA_CHECK(cudaEventRecord(ev_stop[10], ctx.stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop[10]));

            // ---- Accumulate phase times ----
            for (int p = 0; p < 11; ++p) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start[p], ev_stop[p]));
                out_phase_ms_total[p] += static_cast<double>(ms);
            }

            ++n_iters_run;

            // ---- Convergence check ----
            if (iter > 0 && prev_loss > 0) {
                double rel = std::abs(prev_loss - loss_val)
                           / (std::abs(prev_loss) + 1e-15);
                if (rel < tol) converged = true;
            }
            prev_loss = loss_val;
        }  // iteration loop

        // ---- Destroy events ----
        for (int p = 0; p < 11; ++p) {
            cudaEventDestroy(ev_start[p]);
            cudaEventDestroy(ev_stop[p]);
        }

        // ---- Per-iteration means ----
        *out_n_iters = n_iters_run;
        if (n_iters_run > 0) {
            for (int p = 0; p < 11; ++p)
                out_phase_ms_per_iter[p] = out_phase_ms_total[p] / n_iters_run;
        }

        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU NMF profiler error: %s\n", e.what());
        *out_status = -1;
    }
}


} // extern "C"
