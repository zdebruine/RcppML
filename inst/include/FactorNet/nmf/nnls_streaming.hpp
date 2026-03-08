/**
 * @file nmf/nnls_streaming.hpp
 * @brief Streaming NNLS: solve for H given fixed W by streaming A from .spz.
 *
 * Given a fixed factor matrix W (m x k) and a .spz file containing A (m x n),
 * solve the NNLS problem:  min ||A - W*H||  s.t. H >= 0  (when nonneg=true)
 * by streaming forward chunks of A and solving per-chunk NNLS.
 *
 * This is a single-pass extraction of the H-update from nmf_chunked().
 * Memory: O(m*k + n*k + max_chunk_cols*k) — never holds full A.
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/io/loader.hpp>
#include <FactorNet/io/spz_loader.hpp>
#include <FactorNet/io/dense_spz_loader.hpp>
#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>
#include <sparsepress/sparsepress_v3.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace nmf {

/**
 * @brief Solve for H by streaming columns of A from a .spz file.
 *
 * @tparam Scalar float or double
 * @param path         Path to .spz file (v2 sparse or v3 dense)
 * @param W            Factor matrix (m x k), fixed
 * @param cd_maxit     Max CD iterations per column
 * @param cd_tol       CD convergence tolerance
 * @param L1           L1 penalty on H
 * @param L2           L2 penalty on H
 * @param upper_bound  Upper bound on H entries (0 = none)
 * @param nonneg       Enforce non-negativity
 * @param solver_mode  0 = CD, 1 = Cholesky
 * @param threads      Number of threads (0 = auto)
 * @return H matrix (k x n)
 */
template<typename Scalar = double>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> nnls_streaming_spz(
    const std::string& path,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W,
    int cd_maxit = 100,
    Scalar cd_tol = static_cast<Scalar>(1e-8),
    Scalar L1 = 0,
    Scalar L2 = 0,
    Scalar upper_bound = 0,
    bool nonneg = true,
    int solver_mode = 1,
    int threads = 0)
{
    using MatS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    // ---- 1. Detect file version and create loader ----
    std::ifstream probe(path, std::ios::binary);
    if (!probe.is_open())
        throw std::runtime_error("Cannot open SPZ file: " + path);

    uint8_t header_buf[8];
    probe.read(reinterpret_cast<char*>(header_buf), 8);
    probe.close();

    uint32_t version = sparsepress::v3::detect_version(header_buf, 8);

    std::unique_ptr<FactorNet::io::DataLoader<Scalar>> loader;
    if (version == 3) {
        loader = std::make_unique<FactorNet::io::DenseSpzLoader<Scalar>>(
            path, /* require_transpose = */ false);
    } else {
        loader = std::make_unique<FactorNet::io::SpzLoader<Scalar>>(
            path, /* require_transpose = */ false);
    }

    const uint32_t m = loader->rows();
    const uint32_t n = loader->cols();
    const int k = static_cast<int>(W.cols());

    if (static_cast<uint32_t>(W.rows()) != m) {
        throw std::runtime_error(
            "W rows (" + std::to_string(W.rows()) +
            ") != data rows (" + std::to_string(m) + ")");
    }

    // ---- 2. Compute Gram matrix G_W = W'W + tiny_num + L2 ----
    MatS W_T = W.transpose();  // k x m
    MatS G_W(k, k);
    G_W.noalias() = W_T * W;
    G_W.diagonal().array() += FactorNet::tiny_num<Scalar>();  // match c_nnls
    if (L2 > 0) {
        for (int i = 0; i < k; ++i) G_W(i, i) += L2;
    }

    // ---- 3. Allocate output H (k x n) ----
    MatS H = MatS::Zero(k, n);

    // ---- 4. Stream forward chunks, solve NNLS per chunk ----
    FactorNet::io::Chunk<Scalar> chunk;
    MatS B_panel, H_panel;

    loader->reset_forward();
    while (loader->next_forward(chunk)) {
        const uint32_t col_start = chunk.col_start;
        const uint32_t nc = chunk.num_cols;
        const SpMat& A_panel = chunk.matrix;

        // Compute RHS: B_panel = W_T * A_panel
        B_panel.resize(k, nc);
        #pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(dynamic, 64) if(threads != 1)
        for (uint32_t jj = 0; jj < nc; ++jj) {
            B_panel.col(jj).setZero();
            for (typename SpMat::InnerIterator it(A_panel, jj); it; ++it) {
                const Scalar val = it.value();
                const Scalar* w_col = W_T.data() + static_cast<size_t>(it.row()) * k;
                Scalar* b_col = B_panel.data() + static_cast<size_t>(jj) * k;
                for (int fi = 0; fi < k; ++fi)
                    b_col[fi] += w_col[fi] * val;
            }
        }

        // Solve NNLS: G_W * H_panel = B_panel
        H_panel.resize(k, nc);
        H_panel.setZero();

        if (solver_mode == 1) {
            if (L1 > 0)
                B_panel.array() -= L1;
            MatS G_solve = G_W;
            FactorNet::primitives::cholesky_clip_batch(
                G_solve, B_panel, H_panel,
                nonneg, threads, false);
        } else {
            FactorNet::primitives::nnls_batch<FactorNet::primitives::CPU, Scalar>(
                G_W, B_panel, H_panel,
                cd_maxit, cd_tol,
                L1, Scalar(0),  // L2 already in G_W
                nonneg, threads,
                upper_bound, false);
        }

        // Store into output H
        H.block(0, col_start, k, nc) = H_panel;
    }

    return H;
}

}  // namespace nmf
}  // namespace FactorNet
