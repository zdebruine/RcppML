/**
 * @file nmf/fit_chunked.hpp
 * @brief Chunked NMF using the DataLoader abstraction.
 *
 * This replaces the SPZ-specific streaming logic with a generic chunked
 * NMF that works with ANY DataLoader (InMemoryLoader, SpzLoader, or
 * future implementations like GPU loaders).
 *
 * The NMF algorithm is identical to fit_streaming_spz.hpp:
 *   1. Pre-compute G_W = W'W
 *   2. For each forward chunk: H-update + Gram accumulation
 *   3. For each transpose chunk: W-update
 *   4. Scale and check convergence
 *
 * Memory: O(m*k + n*k + max_chunk_nnz) — never holds full A.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/rng/rng.hpp>
#include <FactorNet/io/loader.hpp>
#include <FactorNet/io/ping_pong_prefetch.hpp>

// Feature headers
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>

#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/nmf/variant_helpers.hpp>
#include <FactorNet/math/loss.hpp>
#include <FactorNet/profiling/stream_timer.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <algorithm>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace nmf {

/**
 * @brief Run NMF using a DataLoader for chunk-at-a-time matrix access.
 *
 * Works identically for both in-memory and streaming (.spz) data sources.
 * The caller is responsible for constructing and owning the DataLoader.
 *
 * @tparam Scalar float or double
 * @param loader  DataLoader providing forward and transpose chunks
 * @param config  Unified NMF configuration
 * @param W_init  Optional warm-start W (m × k). nullptr = random.
 * @param H_init  Optional warm-start H (k × n). nullptr = random.
 * @return NMFResult with W, d, H factors
 */
template<typename Scalar = float>
FactorNet::NMFResult<Scalar> nmf_chunked(
    FactorNet::io::DataLoader<Scalar>& loader,
    const FactorNet::NMFConfig<Scalar>& config,
    const FactorNet::DenseMatrix<Scalar>* W_init = nullptr,
    const FactorNet::DenseMatrix<Scalar>* H_init = nullptr)
{
    using MatS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    const uint32_t m = loader.rows();
    const uint32_t n = loader.cols();
    const int k = config.rank;
    const int verbose = config.verbose;

    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "Chunked NMF: %u x %u, nnz=%llu, %u fwd chunks, %u trans chunks\n",
        m, n, (unsigned long long)loader.nnz(),
        loader.num_forward_chunks(), loader.num_transpose_chunks());
    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "Rank: %d, maxit: %d, tol: %.1e, threads: %d\n",
        k, config.max_iter, config.tol, config.threads);

    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "XYZZY_BUILD_CHECK: entering init phase\n");

    // ---- Initialize W, d, H ----
    FactorNet::NMFResult<Scalar> result;
    result.W.resize(m, k);
    result.d.resize(k);
    result.H.resize(k, n);

    if (W_init) {
        // User-provided W initialization
        result.W = *W_init;
        result.d.setOnes();
        if (H_init) {
            result.H = *H_init;
        } else {
            FactorNet::rng::SplitMix64 rng(config.seed);
            for (uint32_t j = 0; j < n; ++j)
                for (int i = 0; i < k; ++i)
                    result.H(i, j) = static_cast<Scalar>(0.01 + 0.99 * rng.uniform());
        }
    } else {
        // Random initialization
        FactorNet::rng::SplitMix64 rng(config.seed);
        for (int j = 0; j < k; ++j) {
            for (uint32_t i = 0; i < m; ++i)
                result.W(i, j) = static_cast<Scalar>(0.01 + 0.99 * rng.uniform());
            result.d(j) = 1.0;
        }
        for (uint32_t j = 0; j < n; ++j)
            for (int i = 0; i < k; ++i)
                result.H(i, j) = static_cast<Scalar>(0.01 + 0.99 * rng.uniform());
    }

    // ---- Masking / CV setup ----
    const bool has_cv = config.holdout_fraction > 0;
    const bool has_mask = config.mask != nullptr;
    const bool use_irls = config.requires_irls();
    const bool is_gp = (config.loss.type == LossType::GP);
    const bool is_nb = (config.loss.type == LossType::NB);
    const bool needs_per_column = has_cv || has_mask || use_irls;

    std::unique_ptr<LazySpeckledMask<Scalar>> cv_mask;
    if (has_cv) {
        cv_mask = std::make_unique<LazySpeckledMask<Scalar>>(
            static_cast<int>(m), static_cast<int>(n),
            static_cast<int64_t>(loader.nnz()),
            config.holdout_fraction, config.cv_seed, config.mask_zeros);
        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
            "CV: holdout=%.1f%%, mask_zeros=%s, est_test=%lld\n",
            config.holdout_fraction * 100.0,
            config.mask_zeros ? "true" : "false",
            (long long)cv_mask->estimated_n_test());
    }

    // ---- Variant detection ----
    const bool is_projective = config.projective;
    const bool is_symmetric = config.symmetric;

    // ---- Symmetric init: enforce H = W^T from the start ----
    if (is_symmetric) {
        result.H = result.W.transpose().eval();
    }

    // ---- Iterative NMF ----
    MatS G_W(k, k), G_H(k, k);
    MatS B_W(k, m);
    MatS B_panel(k, 0);
    MatS H_panel;
    MatS W_T(k, m);  // Pre-transposed W for cache-friendly RHS computation

    // NB: per-row inverse-dispersion (size) vector, initialized from config
    DenseVector<Scalar> nb_size_vec;
    if (is_nb) {
        if (config.gp_dispersion == DispersionMode::PER_COL) {
            nb_size_vec = DenseVector<Scalar>::Constant(n, config.nb_size_init);
        } else {
            // PER_ROW or GLOBAL: one entry per row of A
            nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
        }
    }

    // Gram-trick loss: tr(A^T A) is constant, computed once in first iteration
    double trAtA = 0;
    bool trAtA_computed = false;

    // Streaming profiler (zero overhead when inactive)
    FactorNet::profiling::StreamTimer stream_timer(config.enable_profiling);

    for (int iter = 0; iter < config.max_iter; ++iter) {
        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
            "XYZZY iter %d: gram\n", iter);
        stream_timer.iter_begin();
        // ---- G_W = W'W + regularization ----
        // Pre-transpose W for cache-friendly sparse RHS: W_T(fi, row) is
        // contiguous in fi, avoiding stride-m access on column-major W(m×k).
        stream_timer.begin("chunk_gram");
        W_T.noalias() = result.W.transpose();
        G_W.noalias() = W_T * result.W;
        stream_timer.end();
        // Save clean Gram for Gram-trick loss (before L2/tier2 modifications)
        MatS G_W_clean = G_W;
        if (config.H.L2 > 0) {
            for (int i = 0; i < k; ++i) G_W(i, i) += static_cast<Scalar>(config.H.L2);
        }

        // Tier 2 Gram features for H-update
        MatS W_T_feat = W_T;  // Copy for potential modification
        variant::apply_h_tier2_features(G_W, result.H, config);

        G_H.setZero();
        if (!needs_per_column) B_W.setZero();
        double train_loss_accum = 0, test_loss_accum = 0;
        int64_t n_train_total = 0, n_test_total = 0;
        // Gram-trick accumulators for batch path (O(k²) instead of O(nnz·k))
        double cross_term_accum = 0;
        double chunk_trAtA_accum = 0;

        if (is_symmetric) {
            // ---- Symmetric NMF: skip H-update, use W_T for Gram (matches in-memory) ----
            G_H.noalias() = W_T * W_T.transpose();
        } else {

        // ---- Process each forward chunk (H-update, PingPongPrefetcher) ----
        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
            "XYZZY iter %d: fwd reset\n", iter);
        loader.reset_forward();
        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
            "XYZZY iter %d: fwd pp ctor\n", iter);
        {
            FactorNet::io::PingPongPrefetcher<FactorNet::io::Chunk<Scalar>> fwd_pp(
                [&loader](FactorNet::io::Chunk<Scalar>& c) {
                    return loader.next_forward(c);
                });
            FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
                "XYZZY iter %d: fwd pp next\n", iter);
            FactorNet::io::Chunk<Scalar>* chunk_ptr = nullptr;

        while (fwd_pp.next(chunk_ptr)) {
            const uint32_t col_start = chunk_ptr->col_start;
            const uint32_t nc = chunk_ptr->num_cols;
            const SpMat& A_panel = chunk_ptr->matrix;

            FACTORNET_LOG_NMF(FactorNet::LogLevel::DEBUG, verbose,
                "  Fwd chunk: cols %u-%u, nnz=%llu\n",
                col_start, col_start + nc - 1, (unsigned long long)chunk_ptr->nnz);

            H_panel.resize(k, nc);
            H_panel.setZero();

            if (needs_per_column) {
                // ---- Per-column H-update with Gram correction ----
                double chunk_train = 0, chunk_test = 0;
                int64_t chunk_n_train = 0, chunk_n_test = 0;

                #pragma omp parallel num_threads(config.threads) if(config.threads > 1)
                {
                    VecS b_loc(k), x_loc(k);
                    MatS G_loc(k, k);
                    std::vector<int> excl_rows;

                    #pragma omp for schedule(dynamic, 64) \
                        reduction(+:chunk_train, chunk_test, chunk_n_train, chunk_n_test)
                    for (uint32_t jj = 0; jj < nc; ++jj) {
                        const uint32_t gj = col_start + jj;
                        b_loc.setZero();
                        excl_rows.clear();

                        // Classify nonzeros as train or excluded
                        for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it) {
                            bool excl = false;
                            if (has_mask && config.mask->coeff(it.row(), gj) != 0)
                                excl = true;
                            if (!excl && has_cv && cv_mask->is_holdout(it.row(), gj))
                                excl = true;
                            if (excl) {
                                excl_rows.push_back(static_cast<int>(it.row()));
                            } else {
                                for (int fi = 0; fi < k; ++fi)
                                    b_loc(fi) += W_T_feat(fi, it.row()) * it.value();
                            }
                        }

                        // mask_zeros=false CV: zero entries can also be holdout
                        if (has_cv && !config.mask_zeros) {
                            std::vector<int> nz_rows;
                            for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it)
                                nz_rows.push_back(static_cast<int>(it.row()));
                            cv_mask->holdout_zero_rows(gj, m, nz_rows, excl_rows);
                        }

                        // Gram correction: G_loc = G_W - W_excl * W_excl^T
                        G_loc = G_W;
                        if (!excl_rows.empty()) {
                            const int ne = static_cast<int>(excl_rows.size());
                            MatS W_ex(k, ne);
                            for (int ei = 0; ei < ne; ++ei)
                                W_ex.col(ei) = W_T_feat.col(excl_rows[ei]);
                            G_loc.template selfadjointView<Eigen::Lower>()
                                .rankUpdate(W_ex, Scalar(-1));
                            G_loc.template triangularView<Eigen::Upper>() =
                                G_loc.transpose();
                        }

                        // Solver dispatch
                        x_loc.setZero();
                        if (use_irls) {
                            FactorNet::LossConfig<Scalar> active_loss = config.loss;
                            if (is_gp) active_loss.type = LossType::KL;
                            // NB: pass per-row size vector so irls_weight_nb gets r > 0
                            const Scalar* h_theta_row = is_nb ? nb_size_vec.data() : static_cast<const Scalar*>(nullptr);
                            const Scalar* h_theta_col = static_cast<const Scalar*>(nullptr);
                            FactorNet::primitives::detail::irls_nnls_col_sparse(
                                A_panel, W_T_feat, static_cast<int>(jj), x_loc.data(),
                                G_W, active_loss,
                                static_cast<Scalar>(config.H.L1),
                                static_cast<Scalar>(config.H.L2),
                                config.H.nonneg, config.cd_max_iter,
                                static_cast<Scalar>(config.cd_tol),
                                config.irls_max_iter,
                                static_cast<Scalar>(config.irls_tol),
                                /* weight_zeros = */ true,
                                /* theta_per_row = */ h_theta_row,
                                /* theta_per_col = */ h_theta_col);
                        } else if (config.solver_mode == 1) {
                            FactorNet::primitives::detail::cholesky_clip_col(
                                G_loc, b_loc.data(), x_loc.data(), k,
                                static_cast<Scalar>(config.H.L1), static_cast<Scalar>(0),
                                config.H.nonneg, config.cd_max_iter,
                                static_cast<Scalar>(config.cd_tol),
                                static_cast<Scalar>(config.H.upper_bound));
                        } else {
                            FactorNet::primitives::detail::cd_nnls_col_fixed(
                                G_loc, b_loc.data(), x_loc.data(), k,
                                static_cast<Scalar>(config.H.L1), static_cast<Scalar>(0),
                                config.H.nonneg, config.cd_max_iter,
                                static_cast<Scalar>(config.H.upper_bound));
                        }
                        H_panel.col(jj) = x_loc;

                        // Loss at nonzero entries
                        // NOTE: when use_irls=true, loss_history contains the IRLS objective
                        // (distribution-specific deviance/NLL), not MSE.
                        for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it) {
                            if (has_mask && config.mask->coeff(it.row(), gj) != 0)
                                continue;
                            Scalar recon = 0;
                            for (int fi = 0; fi < k; ++fi)
                                recon += result.W(it.row(), fi) * result.d(fi) * x_loc(fi);
                            double loss_val;
                            if (use_irls) {
                                // Distribution-specific loss (GP NLL, NB NLL, Gamma deviance, etc.)
                                Scalar theta_val = static_cast<Scalar>(0);
                                if (is_nb && static_cast<uint32_t>(it.row()) < static_cast<uint32_t>(nb_size_vec.size()))
                                    theta_val = nb_size_vec(it.row());
                                loss_val = static_cast<double>(FactorNet::compute_loss(
                                    it.value(), recon, config.loss, theta_val));
                            } else {
                                Scalar diff = it.value() - recon;
                                loss_val = static_cast<double>(diff * diff);
                            }
                            if (has_cv && cv_mask->is_holdout(it.row(), gj)) {
                                chunk_test += loss_val; chunk_n_test++;
                            } else {
                                chunk_train += loss_val; chunk_n_train++;
                            }
                        }

                        // mask_zeros=false CV: loss on zero entries
                        if (has_cv && !config.mask_zeros) {
                            VecS dh(k);
                            for (int fi = 0; fi < k; ++fi)
                                dh(fi) = result.d(fi) * x_loc(fi);

                            if (use_irls) {
                                // IRLS: distribution-specific loss at zero entries (GP/NB/Gamma/etc.)
                                // Cannot use Gram trick since loss(0, recon) is non-quadratic.
                                std::vector<int> nz_set;
                                for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it)
                                    nz_set.push_back(static_cast<int>(it.row()));
                                size_t ni = 0;
                                for (uint32_t i = 0; i < m; ++i) {
                                    if (ni < nz_set.size() && nz_set[ni] == static_cast<int>(i)) { ++ni; continue; }
                                    if (has_mask && config.mask->coeff(i, gj) != 0) continue;
                                    Scalar recon = 0;
                                    for (int fi = 0; fi < k; ++fi)
                                        recon += result.W(i, fi) * dh(fi);
                                    Scalar theta_val = static_cast<Scalar>(0);
                                    if (is_nb && i < static_cast<uint32_t>(nb_size_vec.size()))
                                        theta_val = nb_size_vec(i);
                                    double loss_val = static_cast<double>(FactorNet::compute_loss(
                                        static_cast<Scalar>(0), recon, config.loss, theta_val));
                                    if (cv_mask->is_holdout(i, gj)) {
                                        chunk_test += loss_val; chunk_n_test++;
                                    } else {
                                        chunk_train += loss_val; chunk_n_train++;
                                    }
                                }
                            } else {
                                // MSE: Gram trick for O(k²) aggregate instead of O(m·k)
                                // total zero-entry MSE = h^T*D*G_W*D*h - Σ_{nz}(recon_i²)
                                double total_recon_sq = 0;
                                for (int fi = 0; fi < k; ++fi)
                                    for (int fj = 0; fj < k; ++fj)
                                        total_recon_sq += static_cast<double>(dh(fi)) *
                                            static_cast<double>(G_W_clean(fi, fj)) *
                                            static_cast<double>(dh(fj));

                                double nz_recon_sq = 0;
                                for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it) {
                                    Scalar recon = 0;
                                    for (int fi = 0; fi < k; ++fi)
                                        recon += result.W(it.row(), fi) * dh(fi);
                                    nz_recon_sq += static_cast<double>(recon * recon);
                                }
                                double zero_recon_sq = total_recon_sq - nz_recon_sq;

                                // Holdout zeros → test loss
                                std::vector<int> nz_rows;
                                for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it)
                                    nz_rows.push_back(static_cast<int>(it.row()));
                                std::vector<int> holdout_zeros;
                                cv_mask->holdout_zero_rows(gj, m, nz_rows, holdout_zeros);

                                double holdout_zero_sq = 0;
                                for (int hi : holdout_zeros) {
                                    if (has_mask && config.mask->coeff(hi, gj) != 0) continue;
                                    Scalar recon = 0;
                                    for (int fi = 0; fi < k; ++fi)
                                        recon += result.W(hi, fi) * dh(fi);
                                    holdout_zero_sq += static_cast<double>(recon * recon);
                                }

                                int n_zeros = static_cast<int>(m) - static_cast<int>(nz_rows.size());
                                int n_holdout_zeros = static_cast<int>(holdout_zeros.size());
                                int n_train_zeros = n_zeros - n_holdout_zeros;

                                chunk_test += holdout_zero_sq;
                                chunk_n_test += n_holdout_zeros;
                                chunk_train += (zero_recon_sq - holdout_zero_sq);
                                chunk_n_train += n_train_zeros;
                            }
                        }
                    }
                }  // end omp parallel

                train_loss_accum += chunk_train;
                test_loss_accum += chunk_test;
                n_train_total += chunk_n_train;
                n_test_total += chunk_n_test;

            } else {
                // ---- Batch H-update (fast path, no masking) ----
                // Uses Eigen column expressions for SIMD-vectorized accumulation.
                stream_timer.begin("chunk_rhs");
                B_panel.resize(k, nc);
                #pragma omp parallel for num_threads(config.threads) schedule(dynamic, 64) if(config.threads > 1)
                for (uint32_t jj = 0; jj < nc; ++jj) {
                    auto b_col = B_panel.col(jj);
                    b_col.setZero();
                    for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it) {
                        b_col.noalias() += static_cast<Scalar>(it.value()) * W_T.col(it.row());
                    }
                }
                stream_timer.end();

                if (is_projective) {
                    // ---- Projective H-update: H = W^T · A ----
                    // B_panel already contains W_T * A_panel = W^T * A_panel.
                    // Do NOT scale by d here; the loss computation already
                    // applies d, and extract_scaling normalizes H afterward.
                    H_panel = B_panel;
                } else if (config.solver_mode == 1) {
                    stream_timer.begin("chunk_nnls");
                    if (config.H.L1 > 0)
                        B_panel.array() -= static_cast<Scalar>(config.H.L1);
                    MatS G_solve = G_W;
                    if (config.H.L2 > 0) G_solve.diagonal().array() += static_cast<Scalar>(config.H.L2);
                    FactorNet::primitives::cholesky_clip_batch(
                        G_solve, B_panel, H_panel,
                        config.H.nonneg, config.threads, false);
                    stream_timer.end();
                } else {
                    stream_timer.begin("chunk_nnls");
                    FactorNet::primitives::nnls_batch<FactorNet::primitives::CPU, Scalar>(
                        G_W, B_panel, H_panel,
                        config.cd_max_iter, static_cast<Scalar>(config.cd_tol),
                        static_cast<Scalar>(config.H.L1), static_cast<Scalar>(config.H.L2),
                        config.H.nonneg, config.threads,
                        static_cast<Scalar>(config.H.upper_bound), false);
                    stream_timer.end();
                }

            }

            // Store H_panel into full H
            result.H.block(0, col_start, k, nc) = H_panel;

            // Gram-trick loss accumulation (batch path only; per-column computes inline)
            // cross_term += trace(D * B_panel * H_panel^T) — O(k·nc) per chunk
            // trAtA += sum(A_ij^2) — O(nnz), computed once in first iteration
            if (!needs_per_column) {
                stream_timer.begin("loss");
                // Cross term: Σ_f d_f * Σ_j B_panel(f,j) * H_panel(f,j)
                for (int f = 0; f < k; ++f)
                    cross_term_accum += static_cast<double>(result.d(f)) *
                        static_cast<double>(B_panel.row(f).dot(H_panel.row(f)));

                // tr(A^T A) — only computed once across all iterations
                if (!trAtA_computed) {
                    double chunk_ata = 0;
                    for (int jj = 0; jj < static_cast<int>(A_panel.outerSize()); ++jj)
                        for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it)
                            chunk_ata += static_cast<double>(it.value()) *
                                         static_cast<double>(it.value());
                    chunk_trAtA_accum += chunk_ata;
                }
                stream_timer.end();
            }

            // Accumulate G_H for W-update
            stream_timer.begin("gram_accumulate");
            G_H.noalias() += H_panel * H_panel.transpose();
            stream_timer.end();

        }  // end forward chunk loop
        }  // end PingPongPrefetcher scope

        // ---- Gram-trick loss finalization (batch path) ----
        // loss = trAtA - 2*cross_term + recon_norm
        // where recon_norm = trace(D * G_W * D * G_H)
        if (!needs_per_column && !is_symmetric) {
            // Cache trAtA once (it's a constant across iterations)
            if (!trAtA_computed) {
                trAtA = chunk_trAtA_accum;
                trAtA_computed = true;
            }
            // recon_norm = Σ_ij d_i * d_j * G_W_clean(i,j) * G_H(i,j)  —  O(k²)
            double recon_norm = 0;
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    recon_norm += static_cast<double>(result.d(i)) *
                                  static_cast<double>(result.d(j)) *
                                  static_cast<double>(G_W_clean(i, j)) *
                                  static_cast<double>(G_H(i, j));
            train_loss_accum = trAtA - 2.0 * cross_term_accum + recon_norm;
            // n_train_total = m*n for full MSE normalization
            n_train_total = static_cast<int64_t>(m) * static_cast<int64_t>(n);
        }

        // ---- Post-NNLS angular decorrelation on H ----
        if (config.H.angular > 0)
            features::apply_angular_posthoc(result.H, static_cast<Scalar>(config.H.angular));

        // ---- Projective: normalize H and extract d ----
        if (is_projective) {
            variant::extract_scaling(result.H, result.d, config.norm_type);
            // Recompute G_H from normalized H so W-update Gram/RHS are consistent
            G_H.noalias() = result.H * result.H.transpose();
        }

        }  // end if (!is_symmetric)

        // ---- Gram features for W-update ----
        if (config.W.L2 > 0) {
            for (int i = 0; i < k; ++i) G_H(i, i) += static_cast<Scalar>(config.W.L2);
        }
        MatS W_T_feat2 = W_T;  // Copy for potential modification
        variant::apply_w_tier2_features(G_H, W_T_feat2, config);

        // ---- W-update via transpose chunks ----
        loader.reset_transpose();
        if (needs_per_column) {
            // ---- Per-row W-update with Gram correction (PingPongPrefetcher) ----
            {
            FactorNet::io::PingPongPrefetcher<FactorNet::io::Chunk<Scalar>> t_pp(
                [&loader](FactorNet::io::Chunk<Scalar>& c) {
                    return loader.next_transpose(c);
                });
            FactorNet::io::Chunk<Scalar>* t_chunk_ptr = nullptr;
            while (t_pp.next(t_chunk_ptr)) {
                const uint32_t tc_col_start = t_chunk_ptr->col_start;
                const uint32_t tc_nc = t_chunk_ptr->num_cols;
                const SpMat& AT_panel = t_chunk_ptr->matrix;

                #pragma omp parallel num_threads(config.threads) if(config.threads > 1)
                {
                    VecS b_loc(k), x_loc(k);
                    MatS G_loc(k, k);
                    std::vector<int> excl_cols;

                    #pragma omp for schedule(dynamic, 64)
                    for (uint32_t jj = 0; jj < tc_nc; ++jj) {
                        const uint32_t row_A = tc_col_start + jj;
                        b_loc.setZero();
                        excl_cols.clear();

                        for (typename SpMat::InnerIterator it(AT_panel, jj); it; ++it) {
                            Scalar val = it.value();
                            uint32_t col_A = static_cast<uint32_t>(it.row());

                            bool excl = false;
                            if (has_mask && config.mask->coeff(row_A, col_A) != 0)
                                excl = true;
                            if (!excl && has_cv && cv_mask->is_holdout(row_A, col_A))
                                excl = true;
                            if (excl) {
                                excl_cols.push_back(static_cast<int>(col_A));
                            } else {
                                for (int fi = 0; fi < k; ++fi)
                                    b_loc(fi) += val * result.H(fi, col_A);
                            }
                        }

                        // mask_zeros=false: zero entries can be holdout
                        if (has_cv && !config.mask_zeros) {
                            std::vector<int> nz_cols;
                            for (typename SpMat::InnerIterator it(AT_panel, jj); it; ++it)
                                nz_cols.push_back(static_cast<int>(it.row()));
                            std::sort(nz_cols.begin(), nz_cols.end());
                            cv_mask->holdout_zero_cols(row_A, n, nz_cols, excl_cols);
                        }

                        // Gram correction
                        G_loc = G_H;
                        if (!excl_cols.empty()) {
                            const int ne = static_cast<int>(excl_cols.size());
                            MatS H_ex(k, ne);
                            for (int ei = 0; ei < ne; ++ei)
                                H_ex.col(ei) = result.H.col(excl_cols[ei]);
                            G_loc.template selfadjointView<Eigen::Lower>()
                                .rankUpdate(H_ex, Scalar(-1));
                            G_loc.template triangularView<Eigen::Upper>() =
                                G_loc.transpose();
                        }

                        x_loc.setZero();
                        if (use_irls) {
                            FactorNet::LossConfig<Scalar> active_loss = config.loss;
                            if (is_gp) active_loss.type = LossType::KL;
                            // NB W-update: column jj of AT corresponds to row (tc_col_start+jj) of A.
                            // Pass nb_size_vec offset by tc_col_start so theta_per_col[jj] = r_{row_A}.
                            const Scalar* w_theta_row = static_cast<const Scalar*>(nullptr);
                            const Scalar* w_theta_col = is_nb ? (nb_size_vec.data() + tc_col_start) : static_cast<const Scalar*>(nullptr);
                            FactorNet::primitives::detail::irls_nnls_col_sparse(
                                AT_panel, result.H, static_cast<int>(jj), x_loc.data(),
                                G_H, active_loss,
                                static_cast<Scalar>(config.W.L1),
                                static_cast<Scalar>(config.W.L2),
                                config.W.nonneg, config.cd_max_iter,
                                static_cast<Scalar>(config.cd_tol),
                                config.irls_max_iter,
                                static_cast<Scalar>(config.irls_tol),
                                /* weight_zeros = */ true,
                                /* theta_per_row = */ w_theta_row,
                                /* theta_per_col = */ w_theta_col);
                        } else if (config.solver_mode == 1) {
                            FactorNet::primitives::detail::cholesky_clip_col(
                                G_loc, b_loc.data(), x_loc.data(), k,
                                static_cast<Scalar>(config.W.L1), static_cast<Scalar>(0),
                                config.W.nonneg, config.cd_max_iter,
                                static_cast<Scalar>(config.cd_tol),
                                static_cast<Scalar>(config.W.upper_bound));
                        } else {
                            FactorNet::primitives::detail::cd_nnls_col_fixed(
                                G_loc, b_loc.data(), x_loc.data(), k,
                                static_cast<Scalar>(config.W.L1), static_cast<Scalar>(0),
                                config.W.nonneg, config.cd_max_iter,
                                static_cast<Scalar>(config.W.upper_bound));
                        }

                        for (int fi = 0; fi < k; ++fi)
                            result.W(row_A, fi) = x_loc(fi);
                    }
                }  // end omp parallel

            }  // end per-column transpose loop
            }  // end PingPongPrefetcher transpose scope
        } else {
            // ---- Batch W-update (PingPongPrefetcher) ----
            B_W.setZero();
            {
            FactorNet::io::PingPongPrefetcher<FactorNet::io::Chunk<Scalar>> bt_pp(
                [&loader](FactorNet::io::Chunk<Scalar>& c) {
                    return loader.next_transpose(c);
                });
            FactorNet::io::Chunk<Scalar>* bt_chunk_ptr = nullptr;
            while (bt_pp.next(bt_chunk_ptr)) {
                const uint32_t tc_col_start = bt_chunk_ptr->col_start;
                const uint32_t tc_nc = bt_chunk_ptr->num_cols;
                const SpMat& AT_panel = bt_chunk_ptr->matrix;

                stream_timer.begin("w_rhs");
                #pragma omp parallel for num_threads(config.threads) schedule(dynamic, 64) if(config.threads > 1)
                for (uint32_t jj = 0; jj < tc_nc; ++jj) {
                    uint32_t global_j = tc_col_start + jj;
                    auto bw_col = B_W.col(global_j);
                    for (typename SpMat::InnerIterator it(AT_panel, jj); it; ++it) {
                        bw_col.noalias() += static_cast<Scalar>(it.value()) * result.H.col(it.row());
                    }
                }
                stream_timer.end();

            }  // end batch transpose loop
            }  // end batch PingPongPrefetcher scope

            stream_timer.begin("w_nnls");
            MatS W_T(k, m);
            W_T.setZero();
            if (config.solver_mode == 1) {
                if (config.W.L1 > 0)
                    B_W.array() -= static_cast<Scalar>(config.W.L1);
                MatS G_solve = G_H;
                if (config.W.L2 > 0) G_solve.diagonal().array() += static_cast<Scalar>(config.W.L2);
                FactorNet::primitives::cholesky_clip_batch(
                    G_solve, B_W, W_T,
                    config.W.nonneg, config.threads, false);
            } else {
                FactorNet::primitives::nnls_batch<FactorNet::primitives::CPU, Scalar>(
                    G_H, B_W, W_T,
                    config.cd_max_iter, static_cast<Scalar>(config.cd_tol),
                    static_cast<Scalar>(config.W.L1), static_cast<Scalar>(config.W.L2),
                    config.W.nonneg, config.threads,
                    static_cast<Scalar>(config.W.upper_bound), false);
            }
            result.W = W_T.transpose();
            stream_timer.end();
        }

        // ---- Post-NNLS angular decorrelation on W ----
        if (config.W.angular > 0) {
            MatS W_T_tmp = result.W.transpose();
            features::apply_angular_posthoc(W_T_tmp, static_cast<Scalar>(config.W.angular));
            result.W = W_T_tmp.transpose();
        }

        // ---- Scale ----
        stream_timer.begin("scaling");
        if (config.norm_type == FactorNet::NormType::None) {
            result.d.setOnes();
        } else {
            for (int i = 0; i < k; ++i) {
                if (config.norm_type == FactorNet::NormType::L1) {
                    result.d(i) = result.W.col(i).template lpNorm<1>();
                } else {
                    result.d(i) = result.W.col(i).norm();
                }
                if (result.d(i) > 1e-15) {
                    result.W.col(i) /= result.d(i);
                } else {
                    result.d(i) = 1e-15;
                }
            }
        }
        stream_timer.end();

        // ---- Symmetric enforcement: H = W^T ----
        if (is_symmetric) {
            W_T.noalias() = result.W.transpose();
            variant::symmetric_enforce_h(result.H, W_T);
        }

        // ---- Convergence ----
        double train_mse = train_loss_accum / std::max(static_cast<double>(n_train_total), 1.0);
        result.loss_history.push_back(train_mse);
        result.train_loss = train_mse;
        result.iterations = iter + 1;

        if (has_cv) {
            double test_mse = test_loss_accum / std::max(static_cast<double>(n_test_total), 1.0);
            result.test_loss_history.push_back(test_mse);
            result.test_loss = test_mse;
            FACTORNET_LOG_NMF(FactorNet::LogLevel::DETAILED, verbose,
                "Iter %3d: train=%.6e  test=%.6e\n", iter + 1, train_mse, test_mse);
        } else {
            FACTORNET_LOG_NMF(FactorNet::LogLevel::DETAILED, verbose,
                "Iter %3d: loss=%.6e\n", iter + 1, train_mse);
        }

        // Fire per-iteration callback
        if (config.on_iteration) {
            Scalar test_val = has_cv ? static_cast<Scalar>(result.test_loss)
                                     : std::numeric_limits<Scalar>::quiet_NaN();
            config.on_iteration(iter + 1, static_cast<Scalar>(train_mse), test_val);
        }

        // Symmetric NMF: no loss is computed (forward pass skipped),
        // so loss-based convergence is disabled. Always run all iterations.
        if (!is_symmetric && iter >= 1) {
            double prev = result.loss_history[iter - 1];
            double rel_change = std::abs(train_mse - prev) / std::max(std::abs(prev), 1e-15);
            if (rel_change < config.tol) {
                result.converged = true;
                FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
                    "Converged at iteration %d (rel_change=%.2e)\n", iter + 1, rel_change);
                stream_timer.iter_end();
                break;
            }
        }
        stream_timer.iter_end();
    }  // end iteration loop

    // Store profiling results in the NMF result
    if (config.enable_profiling) {
        result.profile = stream_timer.results();
    }

    return result;
}

}  // namespace nmf
}  // namespace FactorNet

