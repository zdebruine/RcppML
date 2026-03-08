/**
 * @file nmf/fit_chunked_gpu.cuh
 * @brief GPU-accelerated chunked NMF using the DataLoader abstraction.
 *
 * Combines the DataLoader interface (Phase 10) with GPU kernels for
 * streaming NMF that minimises PCIe/NVLink traffic:
 *
 *   - W (k×m) stays on GPU for the entire run — never downloaded
 *     except for Tier-2 features (graph_reg, L21, bounds)
 *   - Gram matrices G (k×k) are tiny; PCIe cost ≈ 0
 *   - H panels are uploaded/downloaded per chunk (unavoidable for
 *     streaming), but this is O(k × chunk_cols) per panel
 *   - A panels come from DataLoader and are uploaded once per chunk
 *   - Loss is computed via the Gram trick entirely on GPU
 *
 * Algorithm per iteration:
 *   H-update:
 *     1. G_W = W'W on GPU (device-resident W)
 *     2. For each forward chunk from DataLoader:
 *        a. Upload A_panel CSC → SparseMatrixGPU
 *        b. B_panel = W · A_panel (one-shot cuSPARSE SpMM)
 *        c. H_panel = NNLS(G_W, B_panel) on GPU
 *        d. Download H_panel → host H
 *     3. Normalize H → d on host
 *   W-update:
 *     1. For each forward chunk (re-stream A):
 *        a. Upload A_panel + corresponding H_panel to GPU
 *        b. G_H_partial = H_panel · H_panel' (cuBLAS GEMM)
 *        c. B_W_partial = H_panel · A_panel' (one-shot transpose SpMM)
 *        d. Accumulate G_H and B_W on GPU
 *     2. W = NNLS(G_H, B_W) on GPU
 *     3. Scale W → d on GPU
 *   Loss:
 *     Gram trick: ||A||² − 2·cross + recon, all on GPU per panel
 *
 * Memory (GPU):
 *   Fixed:  W(k×m) + G(k×k) + B_W(k×m) + d(k) + scratch ≈ 2km + k²
 *   Per-panel: A_panel(nnz) + H_panel(k×nc) + B_panel(k×nc)
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#ifdef FACTORNET_HAS_GPU

#include <FactorNet/gpu/types.cuh>
#include <FactorNet/gpu/gram.cuh>
#include <FactorNet/gpu/rhs.cuh>
#include <FactorNet/gpu/nnls.cuh>
#include <FactorNet/gpu/loss.cuh>
#include <FactorNet/gpu/fused_cv.cuh>
#include <FactorNet/gpu/batch_nnls.cuh>
#include <FactorNet/primitives/gpu/cholesky_nnls.cuh>

// IRLS for non-MSE losses (host-mediated via CPU solver)
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/constants.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/rng/rng.hpp>

#include <FactorNet/features/sparsity.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>
#include <FactorNet/gpu/graph_reg_gpu.cuh>

#include <FactorNet/nmf/variant_helpers.hpp>
#include <FactorNet/nmf/gpu_features.cuh>
#include <FactorNet/io/loader.hpp>
#include <FactorNet/profiling/stream_timer.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace FactorNet {
namespace nmf {

namespace chunked_gpu_detail {

/**
 * @brief Upload an Eigen CSC sparse matrix chunk to GPU as SparseMatrixGPU.
 *
 * DataLoader provides Eigen::SparseMatrix<Scalar> chunks in CSC format.
 * We extract the raw CSC arrays and upload them.
 */
template<typename Scalar>
gpu::SparseMatrixGPU<Scalar> upload_chunk(
    const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>& panel)
{
    return gpu::SparseMatrixGPU<Scalar>(
        static_cast<int>(panel.rows()),
        static_cast<int>(panel.cols()),
        static_cast<int>(panel.nonZeros()),
        panel.outerIndexPtr(),
        panel.innerIndexPtr(),
        panel.valuePtr());
}

}  // namespace chunked_gpu_detail

/**
 * @brief GPU-accelerated chunked NMF using DataLoader.
 *
 * @tparam Scalar float or double
 * @param loader  DataLoader providing forward and transpose chunks
 * @param config  NMF configuration
 * @param W_init  Optional warm-start W (m × k). nullptr = random.
 * @param H_init  Optional warm-start H (k × n). nullptr = random.
 * @return NMFResult with W, d, H factors
 */
template<typename Scalar = float>
NMFResult<Scalar> nmf_chunked_gpu(
    io::DataLoader<Scalar>& loader,
    const NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    using namespace FactorNet::gpu;
    using MatS = DenseMatrix<Scalar>;
    using VecS = DenseVector<Scalar>;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    const uint32_t m = loader.rows();
    const uint32_t n = loader.cols();
    const int k = config.rank;
    const int verbose = config.verbose;

    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "GPU Chunked NMF: %u x %u, nnz=%llu, %u fwd chunks\n",
        m, n, (unsigned long long)loader.nnz(),
        loader.num_forward_chunks());
    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "Rank: %d, maxit: %d, tol: %.1e\n",
        k, config.max_iter, config.tol);

    // Validate variant config
    variant::validate_variant_config(config, "GPU chunked NMF");

    // ------------------------------------------------------------------
    // 1. Create GPU context
    // ------------------------------------------------------------------
    GPUContext ctx;

    // ------------------------------------------------------------------
    // 2. Initialize factors
    // ------------------------------------------------------------------
    MatS h_W_T(k, m);   // k × m (row-major W, matching GPU layout)
    MatS h_H(k, n);     // k × n (always on host — may be too big for VRAM)
    VecS h_d(k);

    if (W_init) {
        // User-provided W initialization (m×k → transpose to k×m)
        h_W_T = W_init->transpose();
        h_d.setOnes();
        if (H_init) {
            h_H = *H_init;
        } else {
            rng::SplitMix64 rng(config.seed);
            rng.fill_uniform(h_H.data(), k, n);
        }
    } else {
        rng::SplitMix64 rng(config.seed);
        rng.fill_uniform(h_W_T.data(), k, m);
        rng.fill_uniform(h_H.data(), k, n);
        h_d.setOnes();
    }

    // ------------------------------------------------------------------
    // 3. Allocate persistent GPU memory
    // ------------------------------------------------------------------
    // W stays on GPU for the entire run
    DenseMatrixGPU<Scalar> d_W(k, m, h_W_T.data());
    DeviceMemory<Scalar> d_d(k);
    d_d.upload(h_d.data(), k);

    // Gram matrices (k×k — tiny, reused)
    DenseMatrixGPU<Scalar> d_G(k, k);

    // B_W accumulator for W-update (k×m — stays on GPU)
    DenseMatrixGPU<Scalar> d_B_W(k, m);

    // G_H accumulator for W-update Gram (k×k — stays on GPU)
    DenseMatrixGPU<Scalar> d_G_H(k, k);

    // Loss scratch (3 doubles for Gram trick)
    DeviceMemory<double> d_scratch(4);

    // Host buffers for Tier-2 features and normalization
    MatS h_G(k, k);

    // GPU graph regularization handle for W (H stays on host in chunked mode)
    GpuGraphRegHandle<Scalar> graph_reg_W;
    if (config.W.graph && config.W.graph_lambda > 0) {
        const auto& L = *config.W.graph;
        graph_reg_W.init(ctx, L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
                         static_cast<int>(L.rows()), static_cast<int>(L.nonZeros()),
                         k, d_W.data.get());
    }

    // ------------------------------------------------------------------
    // 4. Iteration loop
    // ------------------------------------------------------------------
    NMFResult<Scalar> result;
    Scalar prev_loss = std::numeric_limits<Scalar>::max();
    int patience_counter = 0;

    // IRLS for non-MSE losses: host-mediated CPU solver (no GPU IRLS kernel)
    const bool use_irls = config.requires_irls();
    const bool is_gp = (config.loss.type == LossType::GP);

    FactorNet::profiling::StreamTimer stream_timer(config.enable_profiling);

    for (int iter = 0; iter < config.max_iter; ++iter) {
        stream_timer.begin("total_iter");

        // ==============================================================
        // H-update
        // ==============================================================
        const auto h_mode = variant::h_update_mode(config);

        if (h_mode == variant::HUpdateMode::SYMMETRIC_SKIP) {
            // Symmetric: H = W^T (download W once)
            ctx.sync();
            d_W.download_to(h_W_T.data());
            h_H = h_W_T;

        } else if (h_mode == variant::HUpdateMode::PROJECTIVE) {
            // Projective: H = diag(d) * W^T * A (SpMM only, no NNLS)
            // Build d-scaled W on GPU
            DenseMatrixGPU<Scalar> d_Wd(k, m);
            CUDA_CHECK(cudaMemcpyAsync(
                d_Wd.data.get(), d_W.data.get(),
                static_cast<size_t>(k) * m * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            absorb_d_gpu(ctx, d_Wd, d_d);

            io::Chunk<Scalar> chunk;
            loader.reset_forward();
            while (loader.next_forward(chunk)) {
                const uint32_t col_start = chunk.col_start;
                const uint32_t nc = chunk.num_cols;
                auto d_A_panel = chunked_gpu_detail::upload_chunk(chunk.matrix);

                DenseMatrixGPU<Scalar> d_H_panel(k, nc);
                compute_rhs_forward_gpu(ctx, d_Wd, d_A_panel, d_H_panel);

                ctx.sync();
                d_H_panel.download_to(
                    h_H.data() + static_cast<size_t>(k) * col_start);
            }

            // Normalize H → d on host
            variant::extract_scaling(h_H, h_d, config.norm_type);
            d_d.upload(h_d.data(), k);

        } else {
            // Standard or IRLS H-update
            if (use_irls) {
                // --- IRLS H-update: host-mediated CPU IRLS, streamed chunk-at-a-time ---
                ctx.sync();
                d_W.download_to(h_W_T.data());

                // Rebuild base Gram on host (IRLS rebuilds weighted Gram internally,
                // but irls_nnls_col_sparse requires a G_base param — pass W'W)
                MatS G_base = h_W_T * h_W_T.transpose();
                FactorNet::LossConfig<Scalar> active_loss = config.loss;
                if (is_gp) active_loss.type = LossType::KL;

                io::Chunk<Scalar> chunk;
                loader.reset_forward();
                while (loader.next_forward(chunk)) {
                    const uint32_t col_start = chunk.col_start;
                    const uint32_t nc = chunk.num_cols;
                    const SpMat& A_panel = chunk.matrix;

                    #pragma omp parallel for num_threads(config.threads) schedule(dynamic,16) if(config.threads > 1)
                    for (uint32_t jj = 0; jj < nc; ++jj) {
                        Scalar* h_col = h_H.data() + static_cast<size_t>(k) * (col_start + jj);
                        FactorNet::primitives::detail::irls_nnls_col_sparse(
                            A_panel, h_W_T, static_cast<int>(jj), h_col,
                            G_base, active_loss,
                            static_cast<Scalar>(config.H.L1),
                            static_cast<Scalar>(config.H.L2),
                            config.H.nonneg, config.cd_max_iter,
                            static_cast<Scalar>(config.cd_tol),
                            config.irls_max_iter,
                            static_cast<Scalar>(config.irls_tol),
                            /* weight_zeros = */ true);
                    }
                }

                if (config.H.upper_bound > 0)
                    features::apply_upper_bound(h_H, config.H.upper_bound);

            } else {
            // Standard GPU MSE H-update
            stream_timer.begin("h_gram");
            compute_gram_regularized_gpu(ctx, d_W, d_G,
                                         static_cast<Scalar>(config.H.L2));

            // Apply Tier-2 features to Gram
            const bool has_tier2_H = (config.H.angular > 0) ||
                                     (config.H.L21 > 0) ||
                                     (config.H.graph && config.H.graph_lambda > 0);
            if (has_tier2_H) {
                // L21/angular on host (H is host-resident; G is k×k — cheap round-trip)
                if (config.H.angular > 0 || config.H.L21 > 0 ||
                    (config.H.graph && config.H.graph_lambda > 0)) {
                    ctx.sync();
                    d_G.download_to(h_G.data());
                    if (config.H.angular > 0)
                        features::apply_angular(h_G, h_H, static_cast<Scalar>(config.H.angular));
                    if (config.H.L21 > 0)
                        features::apply_L21(h_G, h_H, static_cast<Scalar>(config.H.L21));
                    if (config.H.graph && config.H.graph_lambda > 0) {
                        if constexpr (std::is_same_v<Scalar, double>) {
                            features::apply_graph_reg(h_G, *config.H.graph, h_H,
                                                      static_cast<Scalar>(config.H.graph_lambda));
                        } else {
                            Eigen::SparseMatrix<Scalar> g_h = config.H.graph->template cast<Scalar>();
                            features::apply_graph_reg(h_G, g_h, h_H,
                                                      static_cast<Scalar>(config.H.graph_lambda));
                        }
                    }
                    d_G.upload_from(h_G.data());
                }
            }
            stream_timer.end();  // h_gram

            io::Chunk<Scalar> chunk;
            loader.reset_forward();
            while (loader.next_forward(chunk)) {
                const uint32_t col_start = chunk.col_start;
                const uint32_t nc = chunk.num_cols;

                stream_timer.begin("chunk_h2d");
                auto d_A_panel = chunked_gpu_detail::upload_chunk(chunk.matrix);
                stream_timer.end();

                // RHS: B = W · A_panel (one-shot SpMM)
                stream_timer.begin("chunk_spmm");
                DenseMatrixGPU<Scalar> d_B_panel(k, nc);
                compute_rhs_forward_gpu(ctx, d_W, d_A_panel, d_B_panel);
                stream_timer.end();

                // L1 regularization on RHS
                if (config.H.L1 > 0)
                    apply_l1_gpu(ctx, d_B_panel, static_cast<Scalar>(config.H.L1));

                // NNLS solve on GPU
                stream_timer.begin("chunk_nnls");
                DenseMatrixGPU<Scalar> d_H_panel(k, nc);
                CUDA_CHECK(cudaMemsetAsync(d_H_panel.data.get(), 0,
                    static_cast<size_t>(k) * nc * sizeof(Scalar), ctx.stream));

                if (config.solver_mode == 1) {
                    // GPU Cholesky: need a copy of G since it's overwritten
                    DenseMatrixGPU<Scalar> d_G_copy(k, k);
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_G_copy.data.get(), d_G.data.get(),
                        static_cast<size_t>(k) * k * sizeof(Scalar),
                        cudaMemcpyDeviceToDevice, ctx.stream));
                    primitives::cholesky_clip_batch_gpu(
                        ctx, d_G_copy, d_B_panel, d_H_panel, config.H.nonneg);
                } else {
                    batch_cd_nnls_gpu(ctx, d_G, d_B_panel, d_H_panel,
                                      config.cd_max_iter, config.H.nonneg,
                                      /*warm_start=*/false);
                }
                stream_timer.end();  // chunk_nnls

                // Upper bounds — GPU clamp kernel (no PCIe round-trip)
                if (config.H.upper_bound > 0)
                    ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_H_panel,
                        static_cast<Scalar>(config.H.upper_bound));

                // Download H_panel to host
                stream_timer.begin("chunk_d2h");
                ctx.sync();
                d_H_panel.download_to(
                    h_H.data() + static_cast<size_t>(k) * col_start);
                stream_timer.end();
            }

            // Normalize H → d on host
            variant::extract_scaling(h_H, h_d, config.norm_type);
            d_d.upload(h_d.data(), k);
            }  // end standard GPU H-update
        }  // end standard-or-IRLS H-update

        // ==============================================================
        // W-update: IRLS (CPU per-row) or standard GPU accumulate+solve
        // ==============================================================
        if (use_irls) {
            // --- IRLS W-update: host-mediated CPU IRLS, streamed transpose chunks ---
            ctx.sync();
            d_W.download_to(h_W_T.data());

            MatS G_base = h_H * h_H.transpose();
            FactorNet::LossConfig<Scalar> active_loss_w = config.loss;
            if (is_gp) active_loss_w.type = LossType::KL;

            io::Chunk<Scalar> chunk;
            loader.reset_transpose();
            while (loader.next_transpose(chunk)) {
                const uint32_t row_start = chunk.col_start;
                const uint32_t nr = chunk.num_cols;
                const SpMat& AT_panel = chunk.matrix;

                #pragma omp parallel for num_threads(config.threads) schedule(dynamic,16) if(config.threads > 1)
                for (uint32_t jj = 0; jj < nr; ++jj) {
                    Scalar* w_col = h_W_T.data() + static_cast<size_t>(k) * (row_start + jj);
                    FactorNet::primitives::detail::irls_nnls_col_sparse(
                        AT_panel, h_H, static_cast<int>(jj), w_col,
                        G_base, active_loss_w,
                        static_cast<Scalar>(config.W.L1),
                        static_cast<Scalar>(config.W.L2),
                        config.W.nonneg, config.cd_max_iter,
                        static_cast<Scalar>(config.cd_tol),
                        config.irls_max_iter,
                        static_cast<Scalar>(config.irls_tol),
                        /* weight_zeros = */ true);
                }
            }

            if (config.W.upper_bound > 0)
                features::apply_upper_bound(h_W_T, config.W.upper_bound);

            d_W.upload_from(h_W_T.data());

        } else {
        // Standard GPU W-update: accumulate Gram + RHS on GPU, then NNLS
        stream_timer.begin("w_accumulate");
        d_G_H.zero();
        d_B_W.zero();

        {
            io::Chunk<Scalar> chunk;
            loader.reset_forward();  // Re-stream A in forward order
            while (loader.next_forward(chunk)) {
                const uint32_t col_start = chunk.col_start;
                const uint32_t nc = chunk.num_cols;

                // Upload A panel to GPU
                auto d_A_panel = chunked_gpu_detail::upload_chunk(chunk.matrix);

                // Upload corresponding H panel to GPU
                DenseMatrixGPU<Scalar> d_H_panel(
                    k, nc,
                    h_H.data() + static_cast<size_t>(k) * col_start);

                // Accumulate partial Gram: G_H += H_panel · H_panel'
                // Use cuBLAS GEMM: G_H += H_panel * H_panel^T (rank-k update)
                {
                    Scalar alpha = Scalar(1), beta = Scalar(1);
                    auto cublas_dt = std::is_same<Scalar, float>::value
                        ? CUDA_R_32F : CUDA_R_64F;
                    // G_H(k×k) += H_panel(k×nc) * H_panel^T(nc×k)
                    CUBLAS_CHECK(cublasGemmEx(
                        ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        k, k, nc,
                        &alpha,
                        d_H_panel.data.get(), cublas_dt, k,
                        d_H_panel.data.get(), cublas_dt, k,
                        &beta,
                        d_G_H.data.get(), cublas_dt, k,
                        cublas_dt, CUBLAS_GEMM_DEFAULT));
                }

                // Accumulate partial RHS: B_W += H_panel · A_panel^T
                // This gives k × m contribution
                {
                    DenseMatrixGPU<Scalar> d_B_partial(k, m);
                    compute_rhs_transpose_gpu(ctx, d_H_panel, d_A_panel, d_B_partial);

                    // B_W += B_partial (element-wise add on GPU via cuBLAS axpy)
                    Scalar alpha = Scalar(1);
                    if constexpr (std::is_same_v<Scalar, float>) {
                        CUBLAS_CHECK(cublasSaxpy(
                            ctx.cublas, k * m, &alpha,
                            d_B_partial.data.get(), 1,
                            d_B_W.data.get(), 1));
                    } else {
                        CUBLAS_CHECK(cublasDaxpy(
                            ctx.cublas, k * m, &alpha,
                            d_B_partial.data.get(), 1,
                            d_B_W.data.get(), 1));
                    }
                }
            }
        }

        // Apply L21/angular on GPU (no PCIe round-trip; W and G both device-resident)
        if (config.W.L21 > 0)
            ::FactorNet::nmf::gpu_features::apply_L21_gpu(ctx, d_G_H, d_W, static_cast<Scalar>(config.W.L21));
        if (config.W.angular > 0)
            ::FactorNet::nmf::gpu_features::apply_angular_gpu(ctx, d_G_H, d_W, static_cast<Scalar>(config.W.angular));

        // Add L2 regularization + eps to Gram diagonal (GPU kernel — no host round-trip)
        {
            Scalar diag_add = static_cast<Scalar>(config.W.L2)
                            + static_cast<Scalar>(constants::EPS);
            int threads = (k + 255) / 256 * 256;
            if constexpr (std::is_same_v<Scalar, double>) {
                gpu::add_to_diagonal_kernel<<<1, threads, 0, ctx.stream>>>(
                    d_G_H.data.get(), k, diag_add);
            } else {
                gpu::add_to_diagonal_kernel_f<<<1, threads, 0, ctx.stream>>>(
                    d_G_H.data.get(), k, diag_add);
            }

            // Apply graph regularization on GPU (cuSPARSE SpMM + cuBLAS GEMM)
            if (config.W.graph && config.W.graph_lambda > 0) {
                graph_reg_W.apply(ctx, d_G_H.data.get(), d_W.data.get(),
                                  d_W.cols, static_cast<Scalar>(config.W.graph_lambda));
            }
        }
        stream_timer.end();  // w_accumulate

        // L1 on accumulated RHS
        if (config.W.L1 > 0)
            apply_l1_gpu(ctx, d_B_W, static_cast<Scalar>(config.W.L1));

        // NNLS solve for W on GPU: W(k×m) = solve(G_H, B_W)
        stream_timer.begin("w_nnls");
        if (config.solver_mode == 1) {
            DenseMatrixGPU<Scalar> d_G_copy(k, k);
            CUDA_CHECK(cudaMemcpyAsync(
                d_G_copy.data.get(), d_G_H.data.get(),
                static_cast<size_t>(k) * k * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            DenseMatrixGPU<Scalar> d_B_copy(k, m);
            CUDA_CHECK(cudaMemcpyAsync(
                d_B_copy.data.get(), d_B_W.data.get(),
                static_cast<size_t>(k) * m * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            primitives::cholesky_clip_batch_gpu(
                ctx, d_G_copy, d_B_copy, d_W, config.W.nonneg);
        } else {
            batch_cd_nnls_gpu(ctx, d_G_H, d_B_W, d_W,
                              config.cd_max_iter, config.W.nonneg,
                              /*warm_start=*/(iter > 0));
        }
        stream_timer.end();  // w_nnls

        // Upper bounds for W — GPU clamp kernel (no PCIe round-trip)
        if (!use_irls && config.W.upper_bound > 0)
            ::FactorNet::gpu::apply_upper_bound_gpu(ctx, d_W,
                static_cast<Scalar>(config.W.upper_bound));

        }  // end standard GPU W-update

        // Scale W → d on GPU (stays device-resident)
        scale_rows_opt_gpu(ctx, d_W, d_d);

        // Symmetric: H = W^T after normalization
        if (config.symmetric) {
            ctx.sync();
            d_W.download_to(h_W_T.data());
            d_d.download(h_d.data(), k);
            h_H = h_W_T;
        } else {
            // Download d for host-side H normalization tracking
            ctx.sync();
            d_d.download(h_d.data(), k);
        }

        // ==============================================================
        // Loss (Gram trick, streamed over panels, all on GPU)
        // ==============================================================
        bool compute_loss = (iter % config.loss_every == 0)
                          || (iter == config.max_iter - 1);

        double total_loss = 0.0;
        if (compute_loss) {
            stream_timer.begin("loss");
            // Build d-scaled W for loss on GPU
            DenseMatrixGPU<Scalar> d_Wd(k, m);
            CUDA_CHECK(cudaMemcpyAsync(
                d_Wd.data.get(), d_W.data.get(),
                static_cast<size_t>(k) * m * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            absorb_d_gpu(ctx, d_Wd, d_d);

            // Gram of d-scaled W for recon term
            DenseMatrixGPU<Scalar> d_G_loss(k, k);
            compute_gram_gpu(ctx, d_Wd, d_G_loss);

            io::Chunk<Scalar> chunk;
            loader.reset_forward();
            while (loader.next_forward(chunk)) {
                const uint32_t col_start = chunk.col_start;
                const uint32_t nc = chunk.num_cols;

                auto d_A_panel = chunked_gpu_detail::upload_chunk(chunk.matrix);
                DenseMatrixGPU<Scalar> d_H_panel(
                    k, nc,
                    h_H.data() + static_cast<size_t>(k) * col_start);

                // B for cross term: W_d · A_panel
                DenseMatrixGPU<Scalar> d_B_loss(k, nc);
                compute_rhs_forward_gpu(ctx, d_Wd, d_A_panel, d_B_loss);

                total_loss += compute_total_loss_gpu(
                    ctx, d_A_panel, d_Wd, d_H_panel,
                    d_G_loss, d_B_loss, d_scratch, 0, 0);
            }

            if (config.track_loss_history)
                result.loss_history.push_back(static_cast<Scalar>(total_loss));

            FACTORNET_LOG_NMF(FactorNet::LogLevel::DETAILED, verbose,
                "Iter %3d: loss=%.6e\n", iter + 1, total_loss);

            if (iter > 0) {
                Scalar rel_change = std::abs(
                    static_cast<Scalar>(prev_loss - total_loss))
                    / (std::abs(prev_loss) + static_cast<Scalar>(1e-15));
                result.final_tol = rel_change;
                if (rel_change < config.tol) {
                    patience_counter++;
                    if (patience_counter >= config.patience) {
                        result.converged = true;
                        result.train_loss = static_cast<Scalar>(total_loss);
                        result.iterations = iter + 1;
                        stream_timer.end();  // loss
                        stream_timer.end();  // total_iter
                        break;
                    }
                } else {
                    patience_counter = 0;
                }
            }
            prev_loss = static_cast<Scalar>(total_loss);
            stream_timer.end();  // loss
        }

        // Per-iteration callback (fires every iteration using last known loss)
        if (config.on_iteration)
            config.on_iteration(iter + 1, static_cast<Scalar>(prev_loss), Scalar(0));

        result.iterations = iter + 1;
        stream_timer.end();  // total_iter
    }

    result.profile = stream_timer.results();

    // ------------------------------------------------------------------
    // 5. Collect results
    // ------------------------------------------------------------------
    ctx.sync();
    d_W.download_to(h_W_T.data());
    d_d.download(h_d.data(), k);

    result.W = h_W_T.transpose();  // k×m → m×k
    result.d = h_d;
    result.H = h_H;

    if (result.train_loss == 0 && !result.loss_history.empty())
        result.train_loss = result.loss_history.back();
    else if (result.train_loss == 0)
        result.train_loss = prev_loss;

    if (config.sort_model)
        result.sort();

    result.diagnostics.plan_chosen = "GPU (chunked streaming)";

    return result;
}

}  // namespace nmf
}  // namespace FactorNet

#endif  // FACTORNET_HAS_GPU
