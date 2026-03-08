// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file randomized.hpp
 * @brief Randomized SVD (Halko-Martinsson-Tropp, 2011)
 *
 * Algorithm:
 *   1. Draw random Omega (n × l) where l = k + oversampling
 *   2. Y = A · Omega                   (forward SpMM, O(nnz·l))
 *   3. For q power iterations:
 *        QR(Y);  Z = A' · Y;  QR(Z);  Y = A · Z   (4 SpMMs per iter)
 *   4. Q = QR(Y)                       (thin QR, O(m·l²))
 *   5. B = Q' · A                      (backward SpMM, O(nnz·l))
 *   6. SVD(B), extract top k           (l × n small SVD)
 *
 * Total cost: O(nnz · l · (2q + 2)) + O(m·l²·q) + O(l²·n)
 *
 * With q=2 power iterations, accuracy matches Krylov methods for
 * well-separated spectra. Constant cost (no iteration count to optimize).
 *
 * Supports: centering (PCA).
 */

#pragma once

#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/resource_tags.hpp>
#include <FactorNet/svd/spmv.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>  // for trace_AtA
#include <FactorNet/rng/rng.hpp>

#include <cmath>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(RCPP_VERSION) || __has_include(<Rcpp.h>)
#include <Rcpp.h>
#define RSVD_PRINT(...) { char buf[512]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); Rcpp::Rcout << buf; }
#define RSVD_CHECK_INTERRUPT() Rcpp::checkUserInterrupt()
#else
#include <cstdio>
#define RSVD_PRINT(...) std::printf(__VA_ARGS__)
#define RSVD_CHECK_INTERRUPT() ((void)0)
#endif

namespace FactorNet {
namespace svd {

template<typename MatrixType, typename Scalar>
SVDResult<Scalar> randomized_svd(const MatrixType& A,
                                 const SVDConfig<Scalar>& config)
{
    using FactorNet::primitives::CPU;
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_start = Clock::now();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.k_max;
    // Oversampling: max(10, k/5) — scales gracefully for larger k
    const int p = std::min(std::max(10, k / 5), std::min(m, n) - k);
    const int l = k + std::max(p, 0);                   // sketch dimension
    // Power iterations: default 3 (q=3 gives <1% tail SV error for scRNA-seq)
    const int q = (config.max_iter <= 0) ? 3 : std::min(config.max_iter, 10);

    SVDResult<Scalar> result;
    result.centered = config.center;

    int eff_threads = config.threads;
#ifdef _OPENMP
    if (eff_threads <= 0) eff_threads = omp_get_max_threads();
#else
    eff_threads = 1;
#endif

    // --- Row means for PCA centering ---
    DenseVector<Scalar> mu;
    const DenseVector<Scalar>* mu_ptr = nullptr;
    if (config.center) {
        mu = compute_row_means<MatrixType, Scalar>(A, eff_threads);
        result.row_means = mu;
        mu_ptr = &mu;
    }

    // --- Row SDs for correlation PCA scaling ---
    DenseVector<Scalar> row_inv_sds;
    const DenseVector<Scalar>* sd_ptr = nullptr;
    if (config.scale) {
        DenseVector<Scalar> row_sds = compute_row_sds<MatrixType, Scalar>(A, mu, eff_threads);
        row_inv_sds = row_sds.cwiseInverse();
        result.row_sds = row_sds;
        result.scaled = true;
        sd_ptr = &row_inv_sds;
    }

    // --- ||A||²_F ---
    Scalar A_norm_sq;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        SparseMatrix<Scalar> A_cast = A.template cast<Scalar>();
        A_cast.makeCompressed();
        A_norm_sq = primitives::trace_AtA<CPU, Scalar, SparseMatrix<Scalar>>(A_cast);
    } else {
        A_norm_sq = primitives::trace_AtA<CPU, Scalar, MatrixType>(A);
    }
    if (config.center) {
        A_norm_sq -= static_cast<Scalar>(n) * mu.squaredNorm();
    }
    if (config.scale) {
        A_norm_sq = static_cast<Scalar>(m) * static_cast<Scalar>(n);
    }
    result.frobenius_norm_sq = static_cast<double>(A_norm_sq);

    // Pre-allocate SpMM workspace to avoid repeated thread-local allocs
    SpmmWorkspace<Scalar> spmm_ws;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        spmm_ws.init(m, l, eff_threads);
    }

    // --- Helper: centered+scaled forward SpMM  Y = Ã · X  (m×l) ---
    //     Uses batched SpMM from spmv.hpp — reads each nonzero once per l cols
    auto forward_mult = [&](const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y) {
        spmm_forward(A, X, Y, mu_ptr, eff_threads, &spmm_ws, sd_ptr);
    };

    // --- Helper: centered+scaled transpose SpMM  Z = Ã' · X  (n×l) ---
    auto transpose_mult = [&](const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Z) {
        spmm_transpose(A, X, Z, mu_ptr, eff_threads, sd_ptr);
    };

    // ========================================================================
    // Step 1: Random sketch  Omega (n × l)
    // ========================================================================
    DenseMatrix<Scalar> Omega(n, l);
    {
        rng::SplitMix64 rng(config.seed == 0 ? 42ULL : static_cast<uint64_t>(config.seed));
        rng.fill_uniform(Omega);
        // Shift to [-0.5, 0.5) for better conditioning
        Omega.array() -= static_cast<Scalar>(0.5);
    }

    // ========================================================================
    // Step 2: Y = A_c · Omega  (m × l)
    // ========================================================================
    DenseMatrix<Scalar> Y(m, l);
    forward_mult(Omega, Y);

    if (config.verbose) {
        RSVD_PRINT("  randomized SVD: k=%d, l=%d (oversample=%d), power_iter=%d\n",
                   k, l, l - k, q);
    }

    // ========================================================================
    // Step 3: Power iterations for spectral decay amplification
    // ========================================================================
    DenseMatrix<Scalar> Z(n, l);

    // Pre-allocate Identity matrices for QR thin-Q extraction
    // (avoids allocating ~14MB per QR call: 2 QRs per power iter × q iters)
    DenseMatrix<Scalar> I_m = DenseMatrix<Scalar>::Identity(m, l);
    DenseMatrix<Scalar> I_n = DenseMatrix<Scalar>::Identity(n, l);

    for (int i = 0; i < q; ++i) {
        RSVD_CHECK_INTERRUPT();

        // QR(Y) → thin orthonormal basis
        Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_y(Y);
        Y = qr_y.householderQ() * I_m;

        // Z = A_c' · Y  (n × l)
        transpose_mult(Y, Z);

        // QR(Z) → thin orthonormal basis
        Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_z(Z);
        Z = qr_z.householderQ() * I_n;

        // Y = A_c · Z  (m × l)
        forward_mult(Z, Y);

        if (config.verbose) {
            RSVD_PRINT("    power iter %d/%d done\n", i + 1, q);
        }
    }

    // ========================================================================
    // Step 4: Final QR of Y → orthonormal Q (m × l)
    // ========================================================================
    Eigen::HouseholderQR<DenseMatrix<Scalar>> qr_final(Y);
    DenseMatrix<Scalar> Q = qr_final.householderQ() * I_m;

    // ========================================================================
    // Step 5: Bt = A_c' · Q  (n × l), then SVD(Bt) with U/V swap to get SVD(B)
    //
    // Previously: B = Bt.transpose() then SVD(B), which copies the entire n×l matrix.
    // Now: SVD(Bt) directly and swap U↔V, since SVD(X') has same singular values
    //      and swaps left/right singular vectors.
    // ========================================================================
    DenseMatrix<Scalar> Bt(n, l);
    transpose_mult(Q, Bt);

    // ========================================================================
    // Step 6: Small SVD of Bt (n × l), extract top k
    //   Bt = Ub · Σ · Vb'  ⟹  B = Bt' = Vb · Σ · Ub'
    //   So B's left singular vectors = Vb, B's right singular vectors = Ub
    // ========================================================================
    Eigen::JacobiSVD<DenseMatrix<Scalar>> svd_bt(
        Bt, Eigen::ComputeThinU | Eigen::ComputeThinV);

    int k_actual = std::min(k, static_cast<int>(svd_bt.singularValues().size()));

    result.d = svd_bt.singularValues().head(k_actual);
    // U = Q · (Bt's right singular vectors) = Q · Vb  → projects back to m dimensions
    result.U = Q * svd_bt.matrixV().leftCols(k_actual);   // m × k
    // V = Bt's left singular vectors = Ub
    result.V = svd_bt.matrixU().leftCols(k_actual);         // n × k
    result.k_selected = k_actual;
    result.iters_per_factor.push_back(q);

    auto t_end = Clock::now();
    result.wall_time_ms = Ms(t_end - t_start).count();

    if (config.verbose) {
        RSVD_PRINT("  randomized SVD done: %.1f ms, d[0]=%.4e d[k-1]=%.4e\n",
                   result.wall_time_ms,
                   static_cast<double>(result.d(0)),
                   static_cast<double>(result.d(k_actual - 1)));
    }

    return result;
}

}  // namespace svd
}  // namespace FactorNet

