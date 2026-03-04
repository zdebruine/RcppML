// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file algorithms/nmf/fit_unified.hpp
 * @brief THE single NMF iteration loop for all backends (unified architecture)
 *
 * This file contains ONE generic NMF implementation that works identically
 * on CPU and GPU. The Resource template parameter selects which primitive
 * implementations (gram, rhs, nnls_batch) are called.
 *
 * Features are applied via runtime if-checks on small k×k matrices —
 * the branch cost is negligible compared to O(nnz·k) primitives.
 *
 * Loss computation uses the Gram trick by default:
 *   ||A - WH||² = tr(AᵀA) - 2·tr(BᵀH) + tr(G · HHᵀ)
 * Cost: O(k²) using already-computed G and B — essentially free.
 *
 * Convergence uses relative tolerance with patience:
 *   |Δloss| / |loss| < tol  must hold for `patience` consecutive checks.
 *
 * Internally, W is stored as k × m (transposed) for cache-efficient
 * column-major iteration. It is transposed to m × k in the result.
 *
 * This is the NEW unified implementation, coexisting with the old
 * algorithms/nmf/fit.hpp during the migration period.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#ifndef RCPPML_ALGORITHMS_NMF_FIT_UNIFIED_HPP
#define RCPPML_ALGORITHMS_NMF_FIT_UNIFIED_HPP

#include <chrono>
#include <RcppML/core/config.hpp>
#include <RcppML/core/result.hpp>
#include <RcppML/core/types.hpp>
#include <RcppML/core/traits.hpp>
#include <RcppML/core/constants.hpp>
#include <RcppML/core/logging.hpp>
#include <RcppML/core/memory.hpp>

// Primitives (template-specialized per Resource)
#include <RcppML/primitives/primitives.hpp>
#include <RcppML/primitives/cpu/gram.hpp>
#include <RcppML/primitives/cpu/rhs.hpp>
#include <RcppML/primitives/cpu/nnls_batch.hpp>
#include <RcppML/primitives/cpu/nnls_batch_irls.hpp>
#include <RcppML/primitives/cpu/cholesky_clip.hpp>
#include <RcppML/primitives/cpu/loss.hpp>
#include <RcppML/primitives/cpu/fused_nnls.hpp>

// Features (backend-agnostic)
#include <RcppML/features/sparsity.hpp>
#include <RcppML/features/ortho.hpp>
#include <RcppML/features/L21.hpp>
#include <RcppML/features/graph_reg.hpp>
#include <RcppML/features/bounds.hpp>

// Shared algorithm variant dispatch and helpers
#include <RcppML/algorithms/nmf/variant_helpers.hpp>

// RNG
#include <RcppML/rng/splitmix64.hpp>

// SVD (for Lanczos and IRLBA initialization)
#include <RcppML/algorithms/svd/lanczos_svd.hpp>
#include <RcppML/algorithms/svd/irlba_svd.hpp>

#include <cmath>
#include <limits>
#include <algorithm>

namespace RcppML {
namespace algorithms {
namespace unified {

// ========================================================================
// Internal helpers
// ========================================================================

namespace detail {

/**
 * @brief Initialize factors from Lanczos SVD seed
 *
 * Computes a rank-k Lanczos SVD of A, then sets:
 *   W_T = (|U| * sqrt(Sigma))^T   (k x m)
 *   H   = sqrt(Sigma) * |V|^T     (k x n)
 *   d   = ones (scaling extracted during first iteration)
 *
 * This provides a much better starting point than random init,
 * typically reducing iteration count by 30-50%.
 */
template<typename Scalar, typename MatrixType>
inline void initialize_lanczos(DenseMatrix<Scalar>& W_T,
                               DenseMatrix<Scalar>& H,
                               DenseVector<Scalar>& d,
                               const MatrixType& A,
                               int k, int m, int n,
                               uint32_t seed, int threads,
                               bool verbose)
{
    // Run Lanczos SVD
    RcppML::SVDConfig<Scalar> svd_config;
    svd_config.k_max = k;
    svd_config.tol = static_cast<Scalar>(1e-10);
    svd_config.max_iter = 0;  // auto
    svd_config.center = false;
    svd_config.verbose = false;
    svd_config.seed = seed;
    svd_config.threads = threads;

    auto svd_result = svd::lanczos_svd<MatrixType, Scalar>(A, svd_config);
    const int k_actual = svd_result.k_selected;

    W_T.resize(k, m);
    H.resize(k, n);
    d = DenseVector<Scalar>::Ones(k);

    // W_T(i,:) = |U(:,i)| * sqrt(sigma_i)
    // H(i,:)  = |V(:,i)| * sqrt(sigma_i)
    for (int i = 0; i < std::min(k, k_actual); ++i) {
        Scalar sqrtd = std::sqrt(svd_result.d(i));
        for (int j = 0; j < m; ++j)
            W_T(i, j) = std::abs(svd_result.U(j, i)) * sqrtd;
        for (int j = 0; j < n; ++j)
            H(i, j) = std::abs(svd_result.V(j, i)) * sqrtd;
    }

    // Fill remaining ranks with random if Lanczos returned fewer
    if (k_actual < k) {
        rng::SplitMix64 rng(seed == 0 ? 54321U : seed + 999);
        DenseMatrix<Scalar> fill_W(k - k_actual, m);
        DenseMatrix<Scalar> fill_H(k - k_actual, n);
        rng.fill_uniform(fill_W.data(), fill_W.rows(), fill_W.cols());
        rng.fill_uniform(fill_H.data(), fill_H.rows(), fill_H.cols());
        W_T.bottomRows(k - k_actual) = fill_W;
        H.bottomRows(k - k_actual) = fill_H;
    }

    RCPPML_LOG_NMF(RcppML::LogLevel::SUMMARY, verbose,
        "Lanczos init: k_svd=%d, seed=%u\n", k_actual, seed);
}

/**
 * @brief Initialize NMF factors using IRLBA (Implicitly Restarted Lanczos)
 *
 * IRLBA is more accurate than basic Lanczos for high-rank decompositions
 * (k >= 32) since it uses implicit restarts to maintain orthogonality.
 * For low rank (k <= 16), both methods are similar.
 *
 * Uses same interface as initialize_lanczos: W_T(i,:) = |U(:,i)|·√σ_i,
 * H(i,:) = |V(:,i)|·√σ_i.
 *
 * This provides a high-quality starting point, typically reducing iteration
 * count by 30-50% vs random init, especially for high rank.
 */
template<typename Scalar, typename MatrixType>
inline void initialize_irlba(DenseMatrix<Scalar>& W_T,
                             DenseMatrix<Scalar>& H,
                             DenseVector<Scalar>& d,
                             const MatrixType& A,
                             int k, int m, int n,
                             uint32_t seed, int threads,
                             bool verbose)
{
    // Run IRLBA SVD
    RcppML::SVDConfig<Scalar> svd_config;
    svd_config.k_max = k;
    svd_config.tol = static_cast<Scalar>(1e-10);
    svd_config.max_iter = 0;  // auto
    svd_config.center = false;
    svd_config.verbose = false;
    svd_config.seed = seed;
    svd_config.threads = threads;

    auto svd_result = svd::irlba_svd<MatrixType, Scalar>(A, svd_config);
    const int k_actual = svd_result.k_selected;

    W_T.resize(k, m);
    H.resize(k, n);
    d = DenseVector<Scalar>::Ones(k);

    // W_T(i,:) = |U(:,i)| * sqrt(sigma_i)
    // H(i,:)  = |V(:,i)| * sqrt(sigma_i)
    for (int i = 0; i < std::min(k, k_actual); ++i) {
        Scalar sqrtd = std::sqrt(svd_result.d(i));
        for (int j = 0; j < m; ++j)
            W_T(i, j) = std::abs(svd_result.U(j, i)) * sqrtd;
        for (int j = 0; j < n; ++j)
            H(i, j) = std::abs(svd_result.V(j, i)) * sqrtd;
    }

    // Fill remaining ranks with random if IRLBA returned fewer
    if (k_actual < k) {
        rng::SplitMix64 rng(seed == 0 ? 54321U : seed + 999);
        DenseMatrix<Scalar> fill_W(k - k_actual, m);
        DenseMatrix<Scalar> fill_H(k - k_actual, n);
        rng.fill_uniform(fill_W.data(), fill_W.rows(), fill_W.cols());
        rng.fill_uniform(fill_H.data(), fill_H.rows(), fill_H.cols());
        W_T.bottomRows(k - k_actual) = fill_W;
        H.bottomRows(k - k_actual) = fill_H;
    }

    RCPPML_LOG_NMF(RcppML::LogLevel::SUMMARY, verbose,
        "IRLBA init: k_svd=%d, seed=%u\n", k_actual, seed);
}

/**
 * @brief Initialize factors W (k×m) and H (k×n) with random values
 *
 * Uses SplitMix64 for reproducible initialization.
 * W_T is k × m (transposed storage for cache efficiency).
 */
template<typename Scalar>
inline void initialize_factors(DenseMatrix<Scalar>& W_T,
                               DenseMatrix<Scalar>& H,
                               DenseVector<Scalar>& d,
                               int k, int m, int n,
                               uint32_t seed)
{
    rng::SplitMix64 rng(seed);

    W_T.resize(k, m);
    rng.fill_uniform(W_T.data(), W_T.rows(), W_T.cols());

    H.resize(k, n);
    rng.fill_uniform(H.data(), H.rows(), H.cols());

    d = DenseVector<Scalar>::Ones(k);
}

/**
 * @brief Gram-trick loss — delegates to primitives::cpu::gram_trick_loss
 */
template<typename Scalar>
inline Scalar gram_trick_loss(Scalar trAtA,
                              const DenseMatrix<Scalar>& G,
                              const DenseMatrix<Scalar>& B_original,
                              const DenseMatrix<Scalar>& H)
{
    return primitives::cpu::gram_trick_loss(trAtA, G, B_original, H);
}

/**
 * @brief Explicit loss for non-MSE losses over sparse nonzeros
 *
 * For each nonzero A(i,j), computes predicted = W_Td.col(i)^T * H.col(j),
 * then sums loss(A(i,j), predicted) over all nonzeros.
 * Cost: O(nnz·k) — same as one RHS computation.
 */
template<typename Scalar, typename SparseMat>
inline Scalar explicit_loss_sparse(const SparseMat& A,
                                    const DenseMatrix<Scalar>& W_Td,
                                    const DenseMatrix<Scalar>& H,
                                    const LossConfig<Scalar>& loss_config,
                                    int num_threads = 1)
{
    const int n = static_cast<int>(A.cols());
    Scalar total_loss = 0;

    #pragma omp parallel for reduction(+:total_loss) num_threads(num_threads) \
                             schedule(dynamic, 64) if(num_threads > 1)
    for (int j = 0; j < n; ++j) {
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            Scalar predicted = W_Td.col(it.row()).dot(H.col(j));
            total_loss += compute_loss(it.value(), predicted, loss_config);
        }
    }
    return total_loss;
}

/**
 * @brief Explicit loss for non-MSE losses over all elements (dense)
 *
 * Reconstructs WH and sums element-wise loss.
 * Cost: O(m·n·k) for reconstruction + O(m·n) for loss.
 */
template<typename Scalar>
inline Scalar explicit_loss_dense(const DenseMatrix<Scalar>& A,
                                   const DenseMatrix<Scalar>& W_Td,
                                   const DenseMatrix<Scalar>& H,
                                   const LossConfig<Scalar>& loss_config)
{
    // Compute reconstruction: R = W_Td^T * H  (m × n)
    DenseMatrix<Scalar> R = W_Td.transpose() * H;
    Scalar total_loss = 0;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            total_loss += compute_loss(A(i, j), R(i, j), loss_config);
        }
    }
    return total_loss;
}

/**
 * @brief Scale rows of F by d: F.row(i) *= d(i)
 */
template<typename Scalar>
inline void apply_scaling(DenseMatrix<Scalar>& F,
                          const DenseVector<Scalar>& d)
{
    for (int i = 0; i < d.size(); ++i) {
        F.row(i) *= d(i);
    }
}

/**
 * @brief Extract scaling: d[i] = norm(F.row(i)),  F.row(i) /= d[i]
 *
 * Supports L1, L2, and None normalization:
 *   - L1:   d[i] = sum(|F.row(i)|)
 *   - L2:   d[i] = sqrt(sum(F.row(i)^2))
 *   - None: d[i] = 1 (no normalization)
 *
 * Adds 1e-15 epsilon to prevent component death, matching old backend's
 * scale_rows() behavior.
 */
template<typename Scalar>
inline void extract_scaling(DenseMatrix<Scalar>& F,
                            DenseVector<Scalar>& d,
                            NormType norm_type = NormType::L1)
{
    if (norm_type == NormType::None) {
        d.setOnes();
        return;
    }
    if (norm_type == NormType::L1) {
        d = F.array().abs().rowwise().sum().matrix();
    } else {
        d = F.rowwise().norm();
    }
    d.array() += static_cast<Scalar>(1e-15);
    for (int i = 0; i < d.size(); ++i) {
        F.row(i) /= d(i);
    }
}

/**
 * @brief Compute transposed RHS for W update: B = H · Aᵗ (k × m)
 *
 * For sparse A: uses gather pattern on pre-computed CSC(Aᵀ)
 *      - No write contention, embarrassingly parallel
 *      - Same access pattern as H-update
 *
 * For dense A: B = H * A.transpose() (always BLAS).
 */
template<typename Scalar, typename MatrixType>
inline void rhs_transpose(const MatrixType& A,
                           const DenseMatrix<Scalar>& H,
                           DenseMatrix<Scalar>& B,
                           int m, int n,
                           const SparseMatrix<Scalar>* At = nullptr,
                           int num_threads = 1)
{
    const int k = static_cast<int>(H.rows());
    B.resize(k, m);
    B.setZero();

    if constexpr (is_sparse_v<MatrixType>) {
        // Gather path: iterate columns of Aᵀ (rows of A)
        // B(:, j) = Σ over nnz in Aᵀ(:, j): val * H(:, row)
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64) if(num_threads > 1)
        for (int j = 0; j < m; ++j) {
            for (typename SparseMatrix<Scalar>::InnerIterator it(*At, j); it; ++it) {
                B.col(j).noalias() += it.value() * H.col(it.row());
            }
        }
    } else {
        B.noalias() = H * A.transpose();
    }
}

}  // namespace detail

// ========================================================================
// Masked NNLS helpers — per-column solve with delta-G correction
// ========================================================================

namespace mask_detail {

/**
 * @brief Per-column masked NNLS for H-update (sparse A)
 *
 * For each column j, identifies masked rows from the mask matrix,
 * builds delta_G = Σ_{i ∈ masked(j)} w_i * w_i^T, and solves:
 *   (G_full - delta_G) * h_j = b_j
 * where b_j only includes non-masked entries.
 *
 * @param A       Sparse data matrix (m × n)
 * @param W_T     Factor matrix (k × m), transposed W
 * @param G_full  Pre-computed Gram matrix W_T * W_T^T (k × k)
 * @param H       Solution matrix (k × n), updated in-place
 * @param mask    Sparse mask matrix (m × n), nonzeros = masked
 * @param config  NMF configuration (L1, L2, nonneg, cd params)
 * @param threads Number of OpenMP threads
 * @param warm_start Whether to warm-start from current H
 */
template<typename Scalar, typename SparseMat>
void masked_nnls_h_sparse(
    const SparseMat& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_full,
    DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const ::RcppML::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start)
{
    const int k = static_cast<int>(W_T.rows());
    const int n = static_cast<int>(A.cols());

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < n; ++j) {
        // Build b_j from non-masked entries
        DenseVector<Scalar> b(k);
        b.setZero();

        // Collect masked rows for this column
        std::vector<int> masked_rows;
        // Iterate mask column j to find masked rows
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit) {
            if (mit.value() != 0) {
                masked_rows.push_back(static_cast<int>(mit.row()));
            }
        }

        // Build RHS from A column, skipping masked entries
        // For efficiency, use a set for O(1) lookup
        std::vector<bool> is_masked(A.rows(), false);
        for (int r : masked_rows) is_masked[r] = true;

        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            if (!is_masked[it.row()]) {
                b.noalias() += it.value() * W_T.col(it.row());
            }
        }

        // If no masked entries in this column, use full Gram
        if (masked_rows.empty()) {
            // Standard solve: G_full * h_j = b
            DenseMatrix<Scalar> G_local = G_full;

            // Apply L1/L2 to b/G
            for (int i = 0; i < k; ++i) {
                b(i) -= config.L1_H;
                G_local(i, i) += config.L2_H;
            }

            DenseVector<Scalar> x = warm_start ? DenseVector<Scalar>(H.col(j)) : DenseVector<Scalar>::Zero(k);
            RcppML::primitives::detail::cd_nnls_col_fixed(
                G_local, b.data(), x.data(), k,
                static_cast<Scalar>(0), static_cast<Scalar>(0),
                config.nonneg_H, config.cd_max_iter);
            H.col(j) = x;
        } else {
            // Compute delta_G from masked rows
            DenseMatrix<Scalar> G_local = G_full;
            for (int r : masked_rows) {
                G_local.noalias() -= W_T.col(r) * W_T.col(r).transpose();
            }

            // Apply L1/L2
            for (int i = 0; i < k; ++i) {
                b(i) -= config.L1_H;
                G_local(i, i) += config.L2_H;
            }

            DenseVector<Scalar> x = warm_start ? DenseVector<Scalar>(H.col(j)) : DenseVector<Scalar>::Zero(k);
            RcppML::primitives::detail::cd_nnls_col_fixed(
                G_local, b.data(), x.data(), k,
                static_cast<Scalar>(0), static_cast<Scalar>(0),
                config.nonneg_H, config.cd_max_iter);
            H.col(j) = x;
        }
    }
}

/**
 * @brief Per-column masked NNLS for H-update (dense A)
 */
template<typename Scalar>
void masked_nnls_h_dense(
    const DenseMatrix<Scalar>& A,
    const DenseMatrix<Scalar>& W_T,
    const DenseMatrix<Scalar>& G_full,
    DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const ::RcppML::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start)
{
    const int k = static_cast<int>(W_T.rows());
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < n; ++j) {
        std::vector<int> masked_rows;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit) {
            if (mit.value() != 0)
                masked_rows.push_back(static_cast<int>(mit.row()));
        }

        // Build b_j: W_T * A(:,j) excluding masked rows
        DenseVector<Scalar> b(k);
        b.setZero();

        if (masked_rows.empty()) {
            b.noalias() = W_T * A.col(j);
        } else {
            std::vector<bool> is_masked(m, false);
            for (int r : masked_rows) is_masked[r] = true;
            for (int i = 0; i < m; ++i) {
                if (!is_masked[i])
                    b.noalias() += A(i, j) * W_T.col(i);
            }
        }

        DenseMatrix<Scalar> G_local = G_full;
        for (int r : masked_rows)
            G_local.noalias() -= W_T.col(r) * W_T.col(r).transpose();

        for (int i = 0; i < k; ++i) {
            b(i) -= config.L1_H;
            G_local(i, i) += config.L2_H;
        }

        DenseVector<Scalar> x = warm_start ? DenseVector<Scalar>(H.col(j)) : DenseVector<Scalar>::Zero(k);
        RcppML::primitives::detail::cd_nnls_col_fixed(
            G_local, b.data(), x.data(), k,
            static_cast<Scalar>(0), static_cast<Scalar>(0),
            config.nonneg_H, config.cd_max_iter);
        H.col(j) = x;
    }
}

/**
 * @brief Per-column masked NNLS for W-update (sparse A^T)
 *
 * For each row j of A (= column j of A^T), identifies masked columns,
 * builds delta_G from masked columns' H factor rows, and solves.
 *
 * @param At     Transposed sparse data matrix A^T (n × m in CSC)
 * @param H      Factor matrix (k × n)
 * @param G_full Pre-computed Gram matrix H * H^T (k × k)
 * @param W_T    Solution matrix (k × m), updated in-place
 * @param mask   Sparse mask matrix (m × n), nonzeros = masked
 *               For W-update, we need mask row j (i.e., which columns are masked for row j)
 * @param config NMF configuration
 * @param threads Number of threads
 * @param warm_start Whether to warm-start
 */
template<typename Scalar>
void masked_nnls_w_sparse(
    const SparseMatrix<Scalar>& At,
    const DenseMatrix<Scalar>& H,
    const DenseMatrix<Scalar>& G_full,
    DenseMatrix<Scalar>& W_T,
    const SparseMatrix<Scalar>& mask_T,   // TRANSPOSED mask (n × m)
    const ::RcppML::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start)
{
    const int k = static_cast<int>(H.rows());
    const int m = static_cast<int>(At.cols());

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < m; ++j) {
        DenseVector<Scalar> b(k);
        b.setZero();

        // Masked columns for row j of A = column j of mask_T
        std::vector<int> masked_cols;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask_T, j); mit; ++mit) {
            if (mit.value() != 0)
                masked_cols.push_back(static_cast<int>(mit.row()));
        }

        std::vector<bool> is_masked(At.rows(), false);
        for (int c : masked_cols) is_masked[c] = true;

        // b_j from A^T column j, skipping masked entries
        for (typename SparseMatrix<Scalar>::InnerIterator it(At, j); it; ++it) {
            if (!is_masked[it.row()])
                b.noalias() += it.value() * H.col(it.row());
        }

        DenseMatrix<Scalar> G_local = G_full;
        for (int c : masked_cols)
            G_local.noalias() -= H.col(c) * H.col(c).transpose();

        for (int i = 0; i < k; ++i) {
            b(i) -= config.L1_W;
            G_local(i, i) += config.L2_W;
        }

        DenseVector<Scalar> x = warm_start ? DenseVector<Scalar>(W_T.col(j)) : DenseVector<Scalar>::Zero(k);
        RcppML::primitives::detail::cd_nnls_col_fixed(
            G_local, b.data(), x.data(), k,
            static_cast<Scalar>(0), static_cast<Scalar>(0),
            config.nonneg_W, config.cd_max_iter);
        W_T.col(j) = x;
    }
}

/**
 * @brief Per-column masked NNLS for W-update (dense A)
 */
template<typename Scalar>
void masked_nnls_w_dense(
    const DenseMatrix<Scalar>& A,
    const DenseMatrix<Scalar>& H,
    const DenseMatrix<Scalar>& G_full,
    DenseMatrix<Scalar>& W_T,
    const SparseMatrix<Scalar>& mask_T,  // TRANSPOSED mask (n × m)
    const ::RcppML::NMFConfig<Scalar>& config,
    int threads,
    bool warm_start)
{
    const int k = static_cast<int>(H.rows());
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < m; ++j) {
        std::vector<int> masked_cols;
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask_T, j); mit; ++mit) {
            if (mit.value() != 0)
                masked_cols.push_back(static_cast<int>(mit.row()));
        }

        // Build b_j = H * A(j, :)^T, skipping masked columns
        DenseVector<Scalar> b(k);
        b.setZero();
        if (masked_cols.empty()) {
            for (int c = 0; c < n; ++c)
                b.noalias() += A(j, c) * H.col(c);
        } else {
            std::vector<bool> is_masked(n, false);
            for (int c : masked_cols) is_masked[c] = true;
            for (int c = 0; c < n; ++c) {
                if (!is_masked[c])
                    b.noalias() += A(j, c) * H.col(c);
            }
        }

        DenseMatrix<Scalar> G_local = G_full;
        for (int c : masked_cols)
            G_local.noalias() -= H.col(c) * H.col(c).transpose();

        for (int i = 0; i < k; ++i) {
            b(i) -= config.L1_W;
            G_local(i, i) += config.L2_W;
        }

        DenseVector<Scalar> x = warm_start ? DenseVector<Scalar>(W_T.col(j)) : DenseVector<Scalar>::Zero(k);
        RcppML::primitives::detail::cd_nnls_col_fixed(
            G_local, b.data(), x.data(), k,
            static_cast<Scalar>(0), static_cast<Scalar>(0),
            config.nonneg_W, config.cd_max_iter);
        W_T.col(j) = x;
    }
}

/**
 * @brief Compute loss excluding masked entries (sparse A)
 */
template<typename Scalar, typename SparseMat>
Scalar masked_loss_sparse(
    const SparseMat& A,
    const DenseMatrix<Scalar>& W_Td,
    const DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const LossConfig<Scalar>& loss_config,
    int threads)
{
    const int n = static_cast<int>(A.cols());
    const int k = static_cast<int>(W_Td.rows());
    Scalar total_loss = 0;

    #pragma omp parallel for reduction(+:total_loss) num_threads(threads) \
        schedule(dynamic, 64) if(threads > 1)
    for (int j = 0; j < n; ++j) {
        // Build is_masked for this column
        std::vector<bool> is_masked(A.rows(), false);
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit)
            if (mit.value() != 0)
                is_masked[mit.row()] = true;

        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            if (!is_masked[it.row()]) {
                Scalar pred = 0;
                for (int f = 0; f < k; ++f)
                    pred += W_Td(f, it.row()) * H(f, j);
                total_loss += compute_loss(it.value(), pred, loss_config);
            }
        }
    }
    return total_loss;
}

/**
 * @brief Compute loss excluding masked entries (dense A)
 */
template<typename Scalar>
Scalar masked_loss_dense(
    const DenseMatrix<Scalar>& A,
    const DenseMatrix<Scalar>& W_Td,
    const DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask,
    const LossConfig<Scalar>& loss_config)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = static_cast<int>(W_Td.rows());
    Scalar total_loss = 0;

    for (int j = 0; j < n; ++j) {
        std::vector<bool> is_masked(m, false);
        for (typename SparseMatrix<Scalar>::InnerIterator mit(mask, j); mit; ++mit)
            if (mit.value() != 0)
                is_masked[mit.row()] = true;

        for (int i = 0; i < m; ++i) {
            if (!is_masked[i]) {
                Scalar pred = 0;
                for (int f = 0; f < k; ++f)
                    pred += W_Td(f, i) * H(f, j);
                total_loss += compute_loss(A(i, j), pred, loss_config);
            }
        }
    }
    return total_loss;
}

}  // namespace mask_detail

// ========================================================================
// Main NMF fitting function
// ========================================================================

/**
 * @brief NMF fitting — THE single implementation for all backends
 *
 * Template parameter Resource selects CPU or GPU primitives.
 * Template parameter MatrixType selects sparse or dense A.
 * Features are applied via runtime checks (negligible cost on k×k ops).
 *
 * @tparam Resource    primitives::CPU or primitives::GPU
 * @tparam Scalar      float or double
 * @tparam MatrixType  SparseMatrix<Scalar>, MappedSparseMatrix<Scalar>, or DenseMatrix<Scalar>
 *
 * @param A       Data matrix (m × n)
 * @param config  NMF configuration
 * @param W_init  Optional initialization for W (m × k). nullptr = random.
 * @param H_init  Optional initialization for H (k × n). nullptr = random.
 * @return        NMFResult with W (m×k), d (k), H (k×n), convergence info
 */
template<typename Resource, typename Scalar, typename MatrixType>
::RcppML::NMFResult<Scalar> nmf_fit(
    const MatrixType& A,
    const ::RcppML::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.rank;

    // ------------------------------------------------------------------
    // 1. Initialize factors
    // ------------------------------------------------------------------

    // W_T: k × m (transposed storage for cache-efficient row iteration)
    // H:   k × n
    // d:   k (diagonal scaling)
    DenseMatrix<Scalar> W_T, H;
    DenseVector<Scalar> d;

    if (W_init) {
        // User-provided W initialization — transpose from m×k to k×m
        W_T = W_init->transpose();
        d = DenseVector<Scalar>::Ones(k);

        if (H_init) {
            H = *H_init;
        } else {
            // Initialize H randomly using seed
            H.resize(k, n);
            rng::SplitMix64 hrng(config.seed == 0 ? 12345U : config.seed);
            hrng.fill_uniform(H.data(), H.rows(), H.cols());
        }
    } else if (config.init_mode == 1) {
        // Lanczos SVD initialization: W = |U|·√Σ, H = √Σ·|V|^T
        detail::initialize_lanczos(W_T, H, d, A, k, m, n,
                                   config.seed, config.threads, config.verbose);
    } else if (config.init_mode == 2) {
        // IRLBA SVD initialization: W = |U|·√Σ, H = √Σ·|V|^T
        detail::initialize_irlba(W_T, H, d, A, k, m, n,
                                 config.seed, config.threads, config.verbose);
    } else {
        detail::initialize_factors(W_T, H, d, k, m, n, config.seed);
    }

    // ------------------------------------------------------------------
    // 2. Precompute tr(AᵗA) for Gram-trick loss
    // ------------------------------------------------------------------

    const Scalar trAtA = primitives::trace_AtA<Resource, Scalar, MatrixType>(A);

    // ------------------------------------------------------------------
    // 2b. Pre-compute CSC(Aᵀ) for gather-based W update
    // ------------------------------------------------------------------
    // For sparse input, compute the transpose once.
    // This eliminates write contention in the W-update RHS computation
    // (embarrassingly parallel, no atomics).
    // Cost: O(nnz + m + n) time, O(nnz) memory. Amortized over all iterations.

    const SparseMatrix<Scalar>* At_ptr = nullptr;
    SparseMatrix<Scalar> At_storage;

    if constexpr (is_sparse_v<MatrixType>) {
        // Memory safety check: ensure we have enough RAM for the transpose
        const size_t nnz = static_cast<size_t>(A.nonZeros());
        auto mem_check = RcppML::memory::check_transpose_memory<Scalar>(nnz, m, n);

        RCPPML_LOG_NMF(RcppML::LogLevel::SUMMARY, config.verbose,
            "Transpose: %s\n", mem_check.message.c_str());

        if (!mem_check.fits) {
            throw std::runtime_error(
                std::string("Cannot compute in-memory transpose of sparse matrix.\n") +
                mem_check.message);
        }

        At_storage = A.transpose();
        At_storage.makeCompressed();
        At_ptr = &At_storage;
    }

    int eff_threads = config.threads;
    if (eff_threads <= 0) {
        #ifdef _OPENMP
        eff_threads = omp_get_max_threads();
        #else
        eff_threads = 1;
        #endif
    }

    // ------------------------------------------------------------------
    // 3. Allocate working buffers
    // ------------------------------------------------------------------

    DenseMatrix<Scalar> G(k, k);      // Gram matrix — always host-resident
    DenseMatrix<Scalar> B;             // RHS buffer

    // Pre-compute mask transpose for W-update (needed for per-row masking)
    SparseMatrix<Scalar> mask_T_storage;
    const bool use_mask = config.has_mask();
    if (use_mask) {
        mask_T_storage = config.mask->transpose();
        mask_T_storage.makeCompressed();
    }

    ::RcppML::NMFResult<Scalar> result;
    Scalar prev_loss = std::numeric_limits<Scalar>::max();
    int patience_counter = 0;

    // Factor convergence: keep previous W_T and H for change measurement
    DenseMatrix<Scalar> prev_W_T, prev_H;
    if (config.convergence_mode == 1 || config.convergence_mode == 2) {
        prev_W_T = W_T;
        prev_H   = H;
    }

    // ------------------------------------------------------------------
    // 3b. Acceleration state
    // ------------------------------------------------------------------

    // Adaptive solver: determine the effective solver mode per iteration
    const bool use_adaptive_solver = (config.adaptive_solver_switch > 0 &&
                                      config.solver_mode == 0);

    // ------------------------------------------------------------------
    // 3c. Fused RHS+NNLS eligibility
    // ------------------------------------------------------------------
    // Use fused path when: sparse data, CD solver (mode 0), no mask, no IRLS,
    // standard NMF (not projective/symmetric), CPU backend.
    // The fused path computes RHS + warm start + NNLS in a single parallel
    // loop, eliminating the global B matrix and BLAS-3 warm start GEMM.
    const bool can_fuse = is_sparse_v<MatrixType>
                       && config.solver_mode == 0
                       && !use_adaptive_solver
                       && !use_mask
                       && !config.requires_irls();

    if (config.verbose && can_fuse) {
        RCPPML_LOG_NMF(RcppML::LogLevel::SUMMARY, true,
            "Using fused RHS+NNLS path (sparse CD)\n");
    }

    // ------------------------------------------------------------------
    // 4. Iteration loop
    // ------------------------------------------------------------------

    for (int iter = 0; iter < config.max_iter; ++iter) {

        // --- Timing instrumentation ---
        auto _t0 = std::chrono::high_resolution_clock::now();
        auto _t_gram_h = _t0, _t_rhs_h = _t0, _t_nnls_h = _t0,
             _t_norm_h = _t0, _t_gram_w = _t0, _t_rhs_w = _t0,
             _t_nnls_w = _t0, _t_norm_w = _t0, _t_loss = _t0, _t_end = _t0;

        bool compute_loss_this_iter = (iter % config.loss_every == 0)
                                    || (iter == config.max_iter - 1);

        // Pre-compute whether loss is needed this iteration
        const int conv_mode = config.convergence_mode;
        bool need_loss = (conv_mode == 0 || conv_mode == 2)
                       || config.track_loss_history;

        // Effective solver mode: adaptive switching from CD → hierarchical Cholesky
        int eff_solver_mode = config.solver_mode;
        if (use_adaptive_solver && iter >= config.adaptive_solver_switch) {
            eff_solver_mode = 2;  // switch to hierarchical Cholesky
        }

        // ==============================================================
        // Update H — dispatch on algorithm variant
        // ==============================================================

        const auto h_mode = algorithms::variant::h_update_mode(config);

        if (h_mode == algorithms::variant::HUpdateMode::PROJECTIVE) {
            // --------------------------------------------------------
            // PROJECTIVE NMF: H = diag(d) · W_T · A  (tied-weight)
            // No NNLS solve — H is deterministic given W and A.
            // --------------------------------------------------------
            algorithms::variant::projective_h_update(
                W_T, H, d,
                [&](const DenseMatrix<Scalar>& factor, DenseMatrix<Scalar>& result) {
                    primitives::rhs<Resource>(A, factor, result);
                },
                config.norm_type);

        } else if (h_mode == algorithms::variant::HUpdateMode::SYMMETRIC_SKIP) {
            // --------------------------------------------------------
            // SYMMETRIC NMF: skip H update; H = W_T set after W update
            // --------------------------------------------------------

        } else {
            // --------------------------------------------------------
            // STANDARD NMF: Gram → RHS → features → NNLS → normalize
            // --------------------------------------------------------

            // Use normalized W_T directly (unit-norm rows) for well-conditioned
            // Gram matrix, matching old backend behavior.
            // d is NOT applied before Gram/RHS — this preserves conditioning.

            // 1. Compute Gram — always needed
            _t_gram_h = std::chrono::high_resolution_clock::now();
            primitives::gram<Resource>(W_T, G);

            // The fused path is sparse-only: guard with if constexpr so that
            // fused_rhs_nnls_sparse (which calls sparse-only Eigen API) is never
            // instantiated when MatrixType is a dense matrix.
            bool h_used_fused = false;
            if constexpr (is_sparse_v<MatrixType>) {
            if (can_fuse) {
                // ====================================================
                // FUSED PATH: RHS + warm start + NNLS in one parallel loop
                // No global B matrix needed.
                // ====================================================
                _t_rhs_h = std::chrono::high_resolution_clock::now();

                // Apply G-modifying features before the fused loop
                // (L2 on diagonal, ortho, graph_reg, L21)
                if (config.L2_H > 0) G.diagonal().array() += config.L2_H;
                if (config.ortho_H > 0)
                    features::apply_ortho(G, H, config.ortho_H);
                if (config.graph_H)
                    features::apply_graph_reg(G, *config.graph_H, H, config.graph_H_lambda);
                if (config.L21_H > 0)
                    features::apply_L21(G, H, config.L21_H);

                // Fused RHS+NNLS (L1 applied per-column inside the loop)
                primitives::fused_rhs_nnls_sparse(
                    A, W_T, G, H,
                    config.cd_max_iter, config.cd_tol,
                    config.L1_H,          // L1 subtracted from each column's b
                    config.nonneg_H,
                    eff_threads,
                    iter > 0,             // warm_start
                    static_cast<Scalar>(0),  // upper_bound
                    config.cd_abs_tol,
                    config.adaptive_cd,
                    config.cd_n_probe);

                _t_nnls_h = std::chrono::high_resolution_clock::now();
                h_used_fused = true;
            }
            } // end if constexpr sparse
            if (!h_used_fused) {
                // ====================================================
                // STANDARD PATH: separate RHS → features → NNLS
                // Used for dense data, non-CD solvers, mask, IRLS
                // ====================================================
                _t_rhs_h = std::chrono::high_resolution_clock::now();
                primitives::rhs<Resource>(A, W_T, B);

                // 2. Apply all H-side features (shared sequence)
                algorithms::variant::apply_h_features(G, B, H, config);

                _t_nnls_h = std::chrono::high_resolution_clock::now();
                // 4. NNLS solve — branch on mask / loss type
                if (use_mask) {
                    // Masked path: per-column NNLS with delta-G correction
                    primitives::gram<Resource>(W_T, G);  // Rebuild unmodified G
                    if constexpr (is_sparse_v<MatrixType>) {
                        mask_detail::masked_nnls_h_sparse(
                            A, W_T, G, H, *config.mask, config, eff_threads, iter > 0);
                    } else {
                        mask_detail::masked_nnls_h_dense(
                            A, W_T, G, H, *config.mask, config, eff_threads, iter > 0);
                    }
                } else if (config.requires_irls()) {
                    // IRLS path: per-column weighted NNLS
                    primitives::gram<Resource>(W_T, G);  // Rebuild unmodified G for IRLS base
                    if constexpr (std::is_base_of_v<Eigen::SparseMatrixBase<std::decay_t<decltype(A)>>,
                                                    std::decay_t<decltype(A)>>) {
                        primitives::nnls_batch_irls_sparse(
                            A, W_T, G, H, config.loss,
                            config.L1_H, config.L2_H,
                            config.nonneg_H,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads);
                    } else {
                        primitives::nnls_batch_irls_dense(
                            A, W_T, G, H, config.loss,
                            config.L1_H, config.L2_H,
                            config.nonneg_H,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads);
                    }
                } else if (eff_solver_mode == 1) {
                    // Cholesky + clip solver: unconstrained LS then max(0,x)
                    primitives::cholesky_clip_batch(G, B, H,
                                                   config.nonneg_H,
                                                   config.threads,
                                                   iter > 0);
                } else if (eff_solver_mode == 2) {
                    // Hierarchical Cholesky: one active-set iteration
                    primitives::hierarchical_cholesky_batch(G, B, H,
                                                           config.nonneg_H,
                                                           config.threads,
                                                           iter > 0);
                } else if (eff_solver_mode == 3) {
                    // Adaptive Cholesky: hierarchical only when violation rate is high
                    primitives::adaptive_cholesky_batch(G, B, H,
                                                       config.nonneg_H,
                                                       config.threads,
                                                       iter > 0,
                                                       config.chol_adaptive_threshold);
                } else {
                    primitives::nnls_batch<Resource>(G, B, H,
                                                     config.cd_max_iter, config.cd_tol,
                                                     static_cast<Scalar>(0),
                                                     static_cast<Scalar>(0),
                                                     config.nonneg_H,
                                                     config.threads,
                                                     static_cast<Scalar>(0),  // upper_bound
                                                     iter > 0,                // warm_start
                                                     config.cd_abs_tol);
                }
            }  // end fused vs standard H-update NNLS

            // 5. Post-NNLS bounds
            if (config.upper_bound_H > 0)
                features::apply_upper_bound(H, config.upper_bound_H);

            _t_norm_h = std::chrono::high_resolution_clock::now();
            // 6. Normalize H → d
            detail::extract_scaling(H, d, config.norm_type);
        }  // end standard H-update

        // ==============================================================
        // Update W
        // ==============================================================

        // Declare loss-saving variables before the branch so they remain in scope
        DenseMatrix<Scalar> G_w_saved, B_w_saved;
        bool saved_for_loss = false;
        bool used_fused_w = false;  // Track if fused path was used (no B_w_saved)

        const auto w_mode = algorithms::variant::w_update_mode(config);

        if (w_mode == algorithms::variant::WUpdateMode::SYMMETRIC) {
            // --------------------------------------------------------
            // SYMMETRIC NMF: A ≈ W^T·diag(d)·W, only one update needed.
            // Gram = W_T·W_T^T, RHS = W_T·A (forward, since A = A^T).
            // After solving for W_T, enforce H = W_T.
            // --------------------------------------------------------
            primitives::gram<Resource>(W_T, G);
            primitives::rhs<Resource>(A, W_T, B);

            // Save pre-feature Gram and RHS for loss computation
            if (need_loss && compute_loss_this_iter && !use_mask && !config.requires_irls()) {
                G_w_saved = G;
                B_w_saved = B;
                saved_for_loss = true;
            }

            // Apply W-side features (shared sequence)
            algorithms::variant::apply_w_features(G, B, W_T, config);

            // NNLS solve for W_T
            if (config.solver_mode == 1) {
                primitives::cholesky_clip_batch(G, B, W_T,
                                               config.nonneg_W,
                                               config.threads,
                                               iter > 0);
            } else if (config.solver_mode == 2) {
                primitives::hierarchical_cholesky_batch(G, B, W_T,
                                                       config.nonneg_W,
                                                       config.threads,
                                                       iter > 0);
            } else if (config.solver_mode == 3) {
                primitives::adaptive_cholesky_batch(G, B, W_T,
                                                   config.nonneg_W,
                                                   config.threads,
                                                   iter > 0,
                                                   config.chol_adaptive_threshold);
            } else {
                primitives::nnls_batch<Resource>(G, B, W_T,
                                                 config.cd_max_iter, config.cd_tol,
                                                 static_cast<Scalar>(0),
                                                 static_cast<Scalar>(0),
                                                 config.nonneg_W,
                                                 config.threads,
                                                 static_cast<Scalar>(0),
                                                 iter > 0,
                                                 config.cd_abs_tol);
            }

            if (config.upper_bound_W > 0)
                features::apply_upper_bound(W_T, config.upper_bound_W);

            // Normalize W_T → d
            detail::extract_scaling(W_T, d, config.norm_type);

            // Enforce symmetry: H = W_T (shared helper)
            algorithms::variant::symmetric_enforce_h(H, W_T);

        } else {
            // --------------------------------------------------------
            // STANDARD or PROJECTIVE W update
            // --------------------------------------------------------

            // Use normalized H directly (unit-norm rows) for well-conditioned
            // Gram matrix, matching old backend behavior.
            _t_gram_w = std::chrono::high_resolution_clock::now();
            primitives::gram<Resource>(H, G);

            // Save G_w before features modify it (needed for loss in both paths)
            if (need_loss && compute_loss_this_iter && !use_mask && !config.requires_irls()) {
                G_w_saved = G;  // k × k (cheap)
                saved_for_loss = true;
            }

            // The fused path is sparse-only: guard with if constexpr so that
            // fused_rhs_nnls_sparse (which calls sparse-only Eigen API) is never
            // instantiated when MatrixType is a dense matrix.
            if constexpr (is_sparse_v<MatrixType>) {
            if (can_fuse && At_ptr) {
                // ====================================================
                // FUSED PATH (W-update): RHS + warm start + NNLS in one loop
                // Iterates columns of A^T (= rows of A).
                // No global B matrix needed.
                // ====================================================
                _t_rhs_w = std::chrono::high_resolution_clock::now();

                // Apply G-modifying features before the fused loop
                if (config.L2_W > 0) G.diagonal().array() += config.L2_W;
                if (config.ortho_W > 0)
                    features::apply_ortho(G, W_T, config.ortho_W);
                if (config.graph_W)
                    features::apply_graph_reg(G, *config.graph_W, W_T, config.graph_W_lambda);
                if (config.L21_W > 0)
                    features::apply_L21(G, W_T, config.L21_W);

                // Fused RHS+NNLS on columns of A^T
                primitives::fused_rhs_nnls_sparse(
                    *At_ptr, H, G, W_T,
                    config.cd_max_iter, config.cd_tol,
                    config.L1_W,
                    config.nonneg_W,
                    eff_threads,
                    iter > 0,             // warm_start
                    static_cast<Scalar>(0),  // upper_bound
                    config.cd_abs_tol,
                    config.adaptive_cd,
                    config.cd_n_probe);

                _t_nnls_w = std::chrono::high_resolution_clock::now();
                used_fused_w = true;
            }
            } // end if constexpr sparse
            if (!used_fused_w) {
                // ====================================================
                // STANDARD PATH (W-update): separate RHS → features → NNLS
                // ====================================================

                // Transposed RHS: B = H · Aᵗ  (k × m)
                // Uses gather pattern when transpose is available (no atomics)
                _t_rhs_w = std::chrono::high_resolution_clock::now();
                detail::rhs_transpose(A, H, B, m, n, At_ptr, eff_threads);

                // Save B_w = H·A^T BEFORE features modify it.
                if (saved_for_loss) {
                    B_w_saved = B;  // k × m
                }

                // Apply W-side features (shared sequence)
                algorithms::variant::apply_w_features(G, B, W_T, config);

                _t_nnls_w = std::chrono::high_resolution_clock::now();
                // NNLS solve — branch on mask / loss type
                if (use_mask) {
                    // Masked path: per-column NNLS with delta-G correction
                    primitives::gram<Resource>(H, G);  // Rebuild unmodified G
                    if constexpr (is_sparse_v<MatrixType>) {
                        mask_detail::masked_nnls_w_sparse(
                            *At_ptr, H, G, W_T, mask_T_storage, config, eff_threads, iter > 0);
                    } else {
                        mask_detail::masked_nnls_w_dense(
                            A, H, G, W_T, mask_T_storage, config, eff_threads, iter > 0);
                    }
                } else if (config.requires_irls()) {
                    // IRLS W-update: uses A^T as the "data" with H as the "factor"
                    primitives::gram<Resource>(H, G);  // Rebuild unmodified G
                    if constexpr (is_sparse_v<MatrixType>) {
                        // Use pre-computed transpose
                        if (At_ptr) {
                            primitives::nnls_batch_irls_sparse(
                                *At_ptr, H, G, W_T, config.loss,
                                config.L1_W, config.L2_W,
                                config.nonneg_W,
                                config.cd_max_iter, config.cd_tol,
                                config.irls_max_iter, config.irls_tol,
                                config.threads);
                        }
                    } else {
                        // Dense case: use A^T directly
                        DenseMatrix<Scalar> A_T = A.transpose();
                        primitives::nnls_batch_irls_dense(
                            A_T, H, G, W_T, config.loss,
                            config.L1_W, config.L2_W,
                            config.nonneg_W,
                            config.cd_max_iter, config.cd_tol,
                            config.irls_max_iter, config.irls_tol,
                            config.threads);
                    }
                } else if (eff_solver_mode == 1) {
                    // Cholesky + clip solver
                    primitives::cholesky_clip_batch(G, B, W_T,
                                                   config.nonneg_W,
                                                   config.threads,
                                                   iter > 0);
                } else if (eff_solver_mode == 2) {
                    // Hierarchical Cholesky: one active-set iteration
                    primitives::hierarchical_cholesky_batch(G, B, W_T,
                                                           config.nonneg_W,
                                                           config.threads,
                                                           iter > 0);
                } else if (eff_solver_mode == 3) {
                    // Adaptive Cholesky: hierarchical only when violation rate is high
                    primitives::adaptive_cholesky_batch(G, B, W_T,
                                                       config.nonneg_W,
                                                       config.threads,
                                                       iter > 0,
                                                       config.chol_adaptive_threshold);
                } else {
                    primitives::nnls_batch<Resource>(G, B, W_T,
                                                     config.cd_max_iter, config.cd_tol,
                                                     static_cast<Scalar>(0),
                                                     static_cast<Scalar>(0),
                                                     config.nonneg_W,
                                                     config.threads,
                                                     static_cast<Scalar>(0),  // upper_bound
                                                     iter > 0,                // warm_start
                                                     config.cd_abs_tol);
                }
            }  // end fused vs standard W-update NNLS

            if (config.upper_bound_W > 0)
                features::apply_upper_bound(W_T, config.upper_bound_W);

            _t_norm_w = std::chrono::high_resolution_clock::now();
            // Normalize W_T → d
            detail::extract_scaling(W_T, d, config.norm_type);
        }  // end W update

        // ==============================================================
        // Convergence check
        // ==============================================================

        _t_loss = std::chrono::high_resolution_clock::now();

        bool loss_converged = false;
        bool factor_converged = false;

        // --- Compute loss when needed (modes 0/2, or tracking) ---
        if (need_loss && compute_loss_this_iter) {
            Scalar loss_val;
            if (use_mask) {
                // Masked loss: explicit computation skipping masked entries
                DenseMatrix<Scalar> W_Td_loss = W_T;
                detail::apply_scaling(W_Td_loss, d);
                if constexpr (is_sparse_v<MatrixType>) {
                    loss_val = mask_detail::masked_loss_sparse(
                        A, W_Td_loss, H, *config.mask, config.loss, eff_threads);
                } else {
                    loss_val = mask_detail::masked_loss_dense(
                        A, W_Td_loss, H, *config.mask, config.loss);
                }
            } else if (config.requires_irls()) {
                // Non-MSE loss: explicit per-element computation
                DenseMatrix<Scalar> W_Td_loss = W_T;
                detail::apply_scaling(W_Td_loss, d);
                if constexpr (is_sparse_v<MatrixType>) {
                    loss_val = detail::explicit_loss_sparse(
                        A, W_Td_loss, H, config.loss, config.threads);
                } else {
                    loss_val = detail::explicit_loss_dense(
                        A, W_Td_loss, H, config.loss);
                }
            } else if (saved_for_loss && !used_fused_w) {
                // MSE: Optimized Gram trick reusing W-update matrices.
                // B_w_saved was materialized in the standard path.
                // Gram of new W_T: O(k²·m)
                DenseMatrix<Scalar> G_wt;
                primitives::gram<Resource>(W_T, G_wt);

                // Cross term: Σᵢ dᵢ · dot(W_T.row(i), B_w_saved.row(i)): O(k·m)
                Scalar cross_term = 0;
                for (int i = 0; i < k; ++i)
                    cross_term += d(i) * W_T.row(i).dot(B_w_saved.row(i));

                // Recon norm: Σᵢⱼ dᵢ·dⱼ · G_wt(i,j) · G_w_saved(i,j): O(k²)
                Scalar recon_norm = 0;
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)
                        recon_norm += d(i) * d(j) * G_wt(i, j) * G_w_saved(i, j);

                loss_val = trAtA - static_cast<Scalar>(2) * cross_term + recon_norm;
            } else if (saved_for_loss && used_fused_w) {
                // MSE: Fused path — B_w was never materialized.
                // Use sparse cross-term computation: O(nnz·k) + O(k²·m).

                // Gram of new W_T: O(k²·m)
                DenseMatrix<Scalar> G_wt;
                primitives::gram<Resource>(W_T, G_wt);

                // Cross term via sparse traversal of A^T: O(nnz·k)
                Scalar cross_term;
                if constexpr (is_sparse_v<MatrixType>) {
                    cross_term = primitives::loss_cross_term_sparse_via_At(
                        *At_ptr, W_T, H, d, eff_threads);
                } else {
                    // Dense fallback (shouldn't reach here since can_fuse requires sparse)
                    cross_term = 0;
                }

                // Recon norm: Σᵢⱼ dᵢ·dⱼ · G_wt(i,j) · G_w_saved(i,j): O(k²)
                Scalar recon_norm = 0;
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)
                        recon_norm += d(i) * d(j) * G_wt(i, j) * G_w_saved(i, j);

                loss_val = trAtA - static_cast<Scalar>(2) * cross_term + recon_norm;
            } else {
                // MSE fallback: full recomputation
                DenseMatrix<Scalar> W_Td_loss = W_T;
                detail::apply_scaling(W_Td_loss, d);
                DenseMatrix<Scalar> G_loss;
                primitives::gram<Resource>(W_Td_loss, G_loss);
                DenseMatrix<Scalar> B_loss;
                primitives::rhs<Resource>(A, W_Td_loss, B_loss);
                loss_val = detail::gram_trick_loss(
                    trAtA, G_loss, B_loss, H);
            }

            if (config.track_loss_history)
                result.loss_history.push_back(loss_val);

            if (iter > 0 && (conv_mode == 0 || conv_mode == 2)) {
                Scalar rel_change = std::abs(prev_loss - loss_val)
                                  / (std::abs(prev_loss) + static_cast<Scalar>(1e-15));
                result.final_tol = rel_change;
                if (rel_change < config.tol) loss_converged = true;
            }
            prev_loss = loss_val;
        }

        // --- Factor-change convergence (modes 1 and 2) ---
        if (conv_mode == 1 || conv_mode == 2) {
            if (iter > 0) {
                Scalar eps = static_cast<Scalar>(1e-15);
                Scalar dW = (W_T - prev_W_T).norm() / (prev_W_T.norm() + eps);
                Scalar dH = (H - prev_H).norm() / (prev_H.norm() + eps);
                Scalar factor_change = std::max(dW, dH);

                if (conv_mode == 1) result.final_tol = factor_change;
                if (factor_change < config.tol) factor_converged = true;
            }
            prev_W_T = W_T;
            prev_H   = H;
        }

        // --- Apply patience ---
        bool iter_converged = false;
        if (conv_mode == 0) iter_converged = loss_converged;
        else if (conv_mode == 1) iter_converged = factor_converged;
        else if (conv_mode == 2) iter_converged = (loss_converged && factor_converged);

        if (iter > 0) {
            if (iter_converged) {
                patience_counter++;
                if (patience_counter >= config.patience) {
                    result.converged = true;
                    result.train_loss = prev_loss;
                    result.iterations = iter + 1;
                    break;
                }
            } else {
                patience_counter = 0;
            }
        }

        result.iterations = iter + 1;

        // --- Timer report ---
        _t_end = std::chrono::high_resolution_clock::now();
        if (config.verbose) {
            auto us = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };
            Rprintf("[iter %d] gram_h=%ldus rhs_h=%ldus nnls_h=%ldus norm_h=%ldus | "
                    "gram_w=%ldus rhs_w=%ldus nnls_w=%ldus norm_w=%ldus | loss=%ldus | total=%ldus\n",
                    iter, us(_t_gram_h, _t_rhs_h), us(_t_rhs_h, _t_nnls_h),
                    us(_t_nnls_h, _t_norm_h), us(_t_norm_h, _t_gram_w),
                    us(_t_gram_w, _t_rhs_w), us(_t_rhs_w, _t_nnls_w),
                    us(_t_nnls_w, _t_norm_w), us(_t_norm_w, _t_loss),
                    us(_t_loss, _t_end), us(_t0, _t_end));
        }
    }

    // ------------------------------------------------------------------
    // 5. Package result
    // ------------------------------------------------------------------

    result.W = W_T.transpose();  // k×m → m×k
    result.d = d;
    result.H = H;

    if (result.train_loss == 0 && !result.loss_history.empty())
        result.train_loss = result.loss_history.back();
    else if (result.train_loss == 0)
        result.train_loss = prev_loss;

    if (config.sort_model)
        result.sort();

    return result;
}

}  // namespace unified
}  // namespace algorithms
}  // namespace RcppML

#endif  // RCPPML_ALGORITHMS_NMF_FIT_UNIFIED_HPP
