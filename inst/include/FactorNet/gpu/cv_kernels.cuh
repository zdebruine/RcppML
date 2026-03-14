/**
 * @file cv_kernels.cuh
 * @brief Production GPU Cross-Validation NMF.
 *
 * GPU pipeline for CV NMF:
 *   H update: Full Gram on GPU → per-column delta on GPU → batch CV NNLS on GPU
 *   W update: Full Gram on GPU → per-row delta on GPU (via A^T) → batch CV NNLS on GPU
 *   Loss:     Gram trick (mask_zeros=FALSE) or direct sparse (mask_zeros=TRUE)
 *
 * Key improvements over benchmark version:
 *   - NNLS runs on GPU (no per-column CPU NNLS with GPU download per iteration)
 *   - Gram trick loss for mask_zeros=FALSE on GPU
 *   - Precomputed deltas for batch solve
 *   - Proper double precision throughout (fixed sqrtf bug)
 *   - W-update deltas computed on GPU via precomputed A^T (eliminates CPU bottleneck)
 *
 * For singlet loss parity:
 *   - mask_zeros=FALSE + per-column-average MSE matches singlet test_error
 *   - Use compute_cv_loss_dense_gpu for this mode
 */

#pragma once

#include "types.cuh"
#include "gram.cuh"
#include "rhs.cuh"
#include "nnls.cuh"
#include "loss.cuh"
#include <FactorNet/rng/rng.hpp>
#include "cv_delta.cuh"
#include "fused_cv.cuh"
// mixed_precision.cuh removed — RHS now uses cuSPARSE SpMM (see gpu/rhs.cuh)

#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <FactorNet/core/logging.hpp>

// Tier 2 features (host-side Gram modifications)
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>
#include <FactorNet/gpu/graph_reg_gpu.cuh>
#include "batch_nnls.cuh"
#include <FactorNet/nmf/gpu_features.cuh>

// IRLS for non-MSE losses (host-mediated via CPU solver)
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>
#include <FactorNet/math/loss.hpp>

// Batched Cholesky NNLS for CV fallback path
#include <FactorNet/primitives/gpu/cholesky_nnls.cuh>

namespace FactorNet {
namespace gpu {

// =============================================================================
// CV Configuration
// =============================================================================

struct CVNMFConfig {
    // Per-factor config (matches NMFConfig::FactorSettings access pattern)
    struct FactorConfig {
        double L1 = 0, L2 = 0, L21 = 0;
        double upper_bound = 0;
        double graph_lambda = 0;
        bool nonneg = true;
    };

    int k;                   // Factorization rank
    int max_iter;            // Maximum NMF iterations
    double tol;              // Convergence tolerance (factor correlation)
    FactorConfig H, W;       // Per-factor penalties and constraints
    double ortho_H, ortho_W;// Orthogonality penalties
    double holdout_fraction; // Fraction held out (e.g. 0.1)
    uint64_t cv_seed;        // RNG seed for mask
    bool mask_zeros;         // TRUE: test set from non-zeros only
    bool verbose;
    int cd_maxit;            // NNLS max iterations
    int threads;             // CPU threads (for W update deltas)
    double col_subsample_frac; // Fraction of columns eligible for test (1.0 = all)

    // Loss function (IRLS for non-MSE)
    int loss_type;            // 0=MSE, 1=MAE, 2=Huber, 3=KL, 4=GP, 5=NB, 6=Gamma, 7=InvGauss
    double huber_delta;       // Huber transition threshold
    double robust_delta;      // Huber robustness on top of any distribution (0=off)
    double dispersion;        // Dispersion parameter (for GP/NB/Gamma/IG)
    int irls_max_iter;        // IRLS outer iterations per column
    double irls_tol;          // IRLS convergence tolerance
    double cd_tol;            // CD inner convergence tolerance

    // Graph regularization (host-side, nullable pointers)
    const void* graph_W_data;     // Pointer to Eigen::SparseMatrix<Scalar> for W graph
    const void* graph_H_data;     // Pointer to Eigen::SparseMatrix<Scalar> for H graph

    // User-supplied mask (type-erased Eigen::SparseMatrix<Scalar>*).
    // Nonzero entries are masked/missing — excluded from train, test, and loss.
    const void* mask_data;

    // Normalization type for the scaling diagonal (L1, L2, or None)
    int norm_type_int;  // 0=L1, 1=L2, 2=None (int to avoid CUDA enum issues)

    // Algorithm variants
    bool projective;   // Projective NMF: H = diag(d) * W^T * A (no NNLS)
    bool symmetric;    // Symmetric NMF: H = W (enforced copy)

    CVNMFConfig()
        : k(10), max_iter(100), tol(1e-4)
        , ortho_H(0), ortho_W(0)
        , holdout_fraction(0.1), cv_seed(42), mask_zeros(true)
        , verbose(false), cd_maxit(10), threads(1)
        , col_subsample_frac(1.0)
        , loss_type(0), huber_delta(1.0), robust_delta(0), dispersion(1.0)
        , irls_max_iter(20), irls_tol(1e-4), cd_tol(1e-10)
        , graph_W_data(nullptr), graph_H_data(nullptr)
        , mask_data(nullptr)
        , norm_type_int(0)
        , projective(false), symmetric(false) {}

    bool requires_irls() const { return loss_type != 0 || robust_delta > 0; }
    bool has_mask() const { return mask_data != nullptr; }

    bool has_tier2() const {
        return H.L21 > 0 || W.L21 > 0
            || ortho_H > 0 || ortho_W > 0
            || H.upper_bound > 0 || W.upper_bound > 0
            || (graph_W_data != nullptr && W.graph_lambda > 0)
            || (graph_H_data != nullptr && H.graph_lambda > 0);
    }
};

// =============================================================================
// CV Result
// =============================================================================

struct CVNMFResult {
    double train_loss;
    double test_loss;
    double best_test_loss;
    int best_iter;
    int iterations;
    bool converged;
    std::vector<double> train_history;
    std::vector<double> test_history;

    CVNMFResult() : train_loss(0), test_loss(0), best_test_loss(1e30),
                    best_iter(0), iterations(0), converged(false) {}
};

// =============================================================================
// CV Norms: precomputed ||A_train||² and ||A_test||²
// =============================================================================

struct CVNorms {
    double A_train_sq;
    double A_test_sq;
    int64_t n_train;
    int64_t n_test;

    CVNorms() : A_train_sq(0), A_test_sq(0), n_train(0), n_test(0) {}
};

/**
 * @brief Precompute CV norms on host.
 *
 * Splits ||A||² into train and test based on RNG mask.
 * For mask_zeros=TRUE: only non-zeros are classified.
 * For mask_zeros=FALSE: all m×n entries are classified.
 */
inline CVNorms precompute_cv_norms(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<double>& values,
    int m, int n,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    bool mask_zeros,
    uint64_t col_sub_thresh = 0xFFFFFFFFFFFFFFFFULL)
{
    CVNorms norms;

    // Column subsample helper: check if column j is in subsample
    auto is_holdout_with_sub = [&](int i, int j) -> bool {
        if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
            uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                        ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
            ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
            ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
            ch ^= ch >> 31;
            if (ch >= col_sub_thresh) return false;
        }
        return rng.is_holdout(i, j, inv_prob);
    };

    if (mask_zeros) {
        for (int j = 0; j < n; ++j) {
            for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
                int i = row_idx[p];
                double v = values[p];
                if (is_holdout_with_sub(i, j)) {
                    norms.A_test_sq += v * v;
                    norms.n_test++;
                } else {
                    norms.A_train_sq += v * v;
                    norms.n_train++;
                }
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            int p = col_ptr[j];
            for (int i = 0; i < m; ++i) {
                bool has_nz = (p < col_ptr[j + 1]) && (row_idx[p] == i);
                double v = has_nz ? values[p] : 0.0;
                if (is_holdout_with_sub(i, j)) {
                    norms.A_test_sq += v * v;
                    norms.n_test++;
                } else {
                    norms.A_train_sq += v * v;
                    norms.n_train++;
                }
                if (has_nz) ++p;
            }
        }
    }

    return norms;
}

// =============================================================================
// Host-mediated IRLS for CV (per-column/row with train/test split)
// =============================================================================

namespace cv_irls_detail {

/**
 * @brief Build LossConfig from CVNMFConfig loss fields
 */
template<typename Scalar>
FactorNet::LossConfig<Scalar> make_loss_config(const CVNMFConfig& config) {
    FactorNet::LossConfig<Scalar> lc;
    lc.type = static_cast<FactorNet::LossType>(config.loss_type);
    lc.huber_delta = static_cast<Scalar>(config.huber_delta);
    lc.robust_delta = static_cast<Scalar>(config.robust_delta);
    lc.dispersion = static_cast<Scalar>(config.dispersion);
    return lc;
}

/**
 * @brief Compute IRLS weight for a single residual/predicted value
 *
 * Supports robust_delta composition: when loss.robust_delta > 0 and the
 * loss type is a distribution (not legacy MAE/HUBER), the weight is
 * distribution_weight * robust_huber_modifier(pearson_residual).
 */
template<typename Scalar>
inline Scalar cv_irls_weight(Scalar residual, Scalar predicted,
                              const FactorNet::LossConfig<Scalar>& loss) {
    // Legacy standalone MAE/HUBER (old API)
    switch (loss.type) {
        case FactorNet::LossType::MAE:
            return FactorNet::irls_weight_mae(residual);
        case FactorNet::LossType::HUBER:
            return FactorNet::irls_weight_huber(residual, loss.huber_delta);
        default:
            break;
    }

    // Distribution-derived weight
    Scalar w_dist;
    switch (loss.type) {
        case FactorNet::LossType::KL:
            w_dist = FactorNet::irls_weight_kl(predicted);
            break;
        case FactorNet::LossType::GP: {
            Scalar observed = residual + predicted;
            w_dist = FactorNet::irls_weight_gp(observed, predicted, loss.dispersion);
            break;
        }
        case FactorNet::LossType::NB:
            w_dist = FactorNet::irls_weight_nb(predicted, loss.dispersion);
            break;
        case FactorNet::LossType::GAMMA:
            w_dist = FactorNet::irls_weight_power(predicted, static_cast<Scalar>(2));
            break;
        case FactorNet::LossType::INVGAUSS:
            w_dist = FactorNet::irls_weight_power(predicted, static_cast<Scalar>(3));
            break;
        case FactorNet::LossType::TWEEDIE:
            w_dist = FactorNet::irls_weight_power(predicted, loss.power_param);
            break;
        default:
            w_dist = static_cast<Scalar>(1);
            break;
    }

    // Apply robust Huber modifier if active
    if (loss.robust_delta > static_cast<Scalar>(0)) {
        constexpr Scalar eps = static_cast<Scalar>(1e-10);
        Scalar sd_inv = std::sqrt(std::max(w_dist, eps));
        Scalar pearson_r = residual * sd_inv;
        Scalar abs_pr = std::abs(pearson_r);
        Scalar w_robust = (abs_pr <= loss.robust_delta)
                        ? static_cast<Scalar>(1)
                        : loss.robust_delta / (abs_pr + eps);
        return w_dist * w_robust;
    }

    return w_dist;
}

/**
 * @brief IRLS per-column solve for H update in CV (sparse A, host data)
 *
 * Solves for column j of H using only TRAIN entries (not held out).
 * Uses IRLS to handle non-MSE losses.
 *
 * @param col_ptr    CSC column pointers
 * @param row_idx    CSC row indices
 * @param values     CSC values
 * @param W_T        Factor W (k × m, Eigen, column-major)
 * @param j          Column index in H
 * @param x          Solution vector (k), modified in-place
 * @param k          Rank
 * @param rng        RNG for holdout mask
 * @param inv_prob   1/holdout_fraction as uint64
 * @param mask_zeros Whether to mask zeros
 * @param loss       Loss configuration
 * @param L1         L1 penalty
 * @param L2         L2 penalty
 * @param nonneg     Non-negativity constraint
 * @param cd_maxit   CD iterations per IRLS step
 * @param cd_tol     CD tolerance
 * @param irls_max_iter Max IRLS iterations
 * @param irls_tol   IRLS convergence tolerance
 * @param m          Number of rows
 * @param col_sub_thresh Column subsample threshold (0xFFF... = all)
 */
template<typename Scalar>
void irls_solve_col_cv(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_T,
    int j,
    Scalar* x,
    int k,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    bool mask_zeros,
    const FactorNet::LossConfig<Scalar>& loss,
    Scalar L1, Scalar L2,
    bool nonneg,
    int cd_maxit, Scalar cd_tol,
    int irls_max_iter, Scalar irls_tol,
    int m,
    uint64_t col_sub_thresh)
{
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Check column subsample: if this column is NOT in subsample,
    // all entries are training (no holdout)
    bool col_in_sub = true;
    if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
        uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                    ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
        ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
        ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
        ch ^= ch >> 31;
        col_in_sub = (ch < col_sub_thresh);
    }

    // Collect train entries
    struct TrainEntry { int row; Scalar value; };
    std::vector<TrainEntry> train_entries;

    if (mask_zeros) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int i = row_idx[p];
            bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
            if (!is_test) {
                train_entries.push_back({i, values[p]});
            }
        }
    } else {
        int p = col_ptr[j];
        for (int i = 0; i < m; ++i) {
            Scalar val = Scalar(0);
            if (p < col_ptr[j + 1] && row_idx[p] == i) { val = values[p]; ++p; }
            bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
            if (!is_test) {
                train_entries.push_back({i, val});
            }
        }
    }

    // Working buffers
    DenseMat G_w(k, k);
    DenseVec b_w(k);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_old(k);

    for (int irls_iter = 0; irls_iter < irls_max_iter; ++irls_iter) {
        G_w.setZero();
        b_w.setZero();

        for (const auto& te : train_entries) {
            // Predicted value
            Scalar predicted = Scalar(0);
            for (int f = 0; f < k; ++f)
                predicted += W_T(f, te.row) * x[f];

            Scalar residual = te.value - predicted;
            Scalar w = cv_irls_weight(residual, predicted, loss);
            Scalar wv = w * te.value;

            // Accumulate weighted Gram and RHS
            for (int a = 0; a < k; ++a) {
                b_w(a) += W_T(a, te.row) * wv;
                for (int bb = a; bb < k; ++bb) {
                    G_w(a, bb) += w * W_T(a, te.row) * W_T(bb, te.row);
                }
            }
        }

        // Symmetrize
        for (int a = 0; a < k; ++a)
            for (int bb = a + 1; bb < k; ++bb)
                G_w(bb, a) = G_w(a, bb);

        // Regularization and stabilization
        if (L2 > 0) G_w.diagonal().array() += L2;
        G_w.diagonal().array() += static_cast<Scalar>(1e-15);

        // Save for convergence check
        for (int f = 0; f < k; ++f) x_old(f) = x[f];

        // CD solve (warm start from current x)
        FactorNet::primitives::detail::cd_nnls_col_fixed(
            G_w, b_w.data(), x, k, L1, Scalar(0), nonneg, cd_maxit);

        // Check IRLS convergence
        Scalar max_change = 0;
        for (int f = 0; f < k; ++f) {
            Scalar rel = std::abs(x[f] - x_old(f))
                       / (std::abs(x_old(f)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < irls_tol) break;
    }
}

/**
 * @brief Host-mediated IRLS for CV: H update (all columns)
 */
template<typename Scalar>
void irls_h_update_cv(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_T,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H_mat,
    int k, int m, int n,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    const CVNMFConfig& config,
    uint64_t col_sub_thresh)
{
    FactorNet::LossConfig<Scalar> loss = make_loss_config<Scalar>(config);

#ifdef _OPENMP
    int n_threads = (config.threads > 0) ? config.threads : 1;
    #pragma omp parallel for schedule(dynamic, 64) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        irls_solve_col_cv(col_ptr, row_idx, values, W_T, j,
                          H_mat.col(j).data(), k,
                          rng, inv_prob,
                          config.mask_zeros, loss,
                          static_cast<Scalar>(config.H.L1),
                          static_cast<Scalar>(config.H.L2),
                          config.H.nonneg,
                          config.cd_maxit,
                          static_cast<Scalar>(config.cd_tol),
                          config.irls_max_iter,
                          static_cast<Scalar>(config.irls_tol),
                          m, col_sub_thresh);
    }
}

/**
 * @brief Host-mediated IRLS for CV: W update (all rows, using A^T)
 *
 * A^T is stored in CSC format (rows→columns) so we can reuse the same
 * per-column IRLS solver, just with the transposed matrix.
 */
template<typename Scalar>
void irls_w_update_cv(
    const std::vector<int>& at_col_ptr,
    const std::vector<int>& at_row_idx,
    const std::vector<Scalar>& at_values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H_mat,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_T_mat,
    int k, int n, int m,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    const CVNMFConfig& config,
    uint64_t col_sub_thresh)
{
    FactorNet::LossConfig<Scalar> loss = make_loss_config<Scalar>(config);

    // For W update, the "columns" are rows of A (via A^T)
    // The holdout test is rng.is_holdout(row_of_At=col_of_A, col_of_At=row_of_A, inv_prob)
    // In the transposed representation: iterating column i of A^T corresponds to row i of A
    // The holdout mask is defined on (row, col) = (row_idx_in_At_col_i, i)
    // But wait — for W update, we iterate over rows of A, i.e., columns of A^T
    // For column i of A^T: entries correspond to A's nonzeros in row i
    //   row_idx[p] for A^T column i = column index j in original A
    //   The holdout test for entry (i,j) in original A is rng.is_holdout(i, j, inv_prob)

    // We need a custom version that swaps i,j for the holdout check
    // since irls_solve_col_cv does rng.is_holdout(row, col, inv_prob)
    // and here col = "column of A^T" = row of A = i, row = "row_idx in A^T" = column of A = j
    // So the holdout check should be rng.is_holdout(j, i, inv_prob) → but the function does
    // rng.is_holdout(row_idx[p], j_of_At, inv_prob) = rng.is_holdout(j_orig, i_orig, inv_prob)
    // This is WRONG — we need rng.is_holdout(i_orig, j_orig, inv_prob)

    // Solution: Don't reuse the same function. Write the W-update IRLS directly.
    // Actually, looking at how the holdout mask works, it's a hash of (i, j) that's symmetric
    // if the hash function is symmetric. Let me check rng.is_holdout()...
    // From splitmix64.hpp: is_holdout(row, col, inv_prob) uses hash(row, col)
    // This is NOT symmetric — is_holdout(i,j) != is_holdout(j,i) in general.

    // So for W update through A^T, we need is_holdout(row_of_A, col_of_A) = is_holdout(i, j)
    // When iterating column i of A^T, the entry's row in A^T is j (column of A)
    // So the holdout check must be: is_holdout(i, row_idx_of_At[p]) — swapped from the H-update

#ifdef _OPENMP
    int n_threads = (config.threads > 0) ? config.threads : 1;
    #pragma omp parallel for schedule(dynamic, 64) num_threads(n_threads)
#endif
    for (int i = 0; i < m; ++i) {
        // Column i of A^T = row i of A
        // Collect train entries for this row
        struct TrainEntry { int col; Scalar value; };
        std::vector<TrainEntry> train_entries;

        // Column subsample check applies to column j of A, not row i
        // For each nonzero in A^T column i (= row i of A), we check
        // if its column (= row of A^T) j is in subsample AND is holdout

        if (config.mask_zeros) {
            for (int p = at_col_ptr[i]; p < at_col_ptr[i + 1]; ++p) {
                int j = at_row_idx[p];

                // Check column subsample
                bool col_in_sub = true;
                if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
                    uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                                ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
                    ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
                    ch ^= ch >> 31;
                    col_in_sub = (ch < col_sub_thresh);
                }

                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (!is_test) {
                    train_entries.push_back({j, at_values[p]});
                }
            }
        } else {
            int p = at_col_ptr[i];
            for (int j = 0; j < n; ++j) {
                Scalar val = Scalar(0);
                if (p < at_col_ptr[i + 1] && at_row_idx[p] == j) { val = at_values[p]; ++p; }

                bool col_in_sub = true;
                if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
                    uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                                ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
                    ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
                    ch ^= ch >> 31;
                    col_in_sub = (ch < col_sub_thresh);
                }

                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (!is_test) {
                    train_entries.push_back({j, val});
                }
            }
        }

        // IRLS solve for this row of W
        using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using DenseVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        DenseMat G_w(k, k);
        DenseVec b_w(k);
        DenseVec x_old(k);
        Scalar* w_row = W_T_mat.col(i).data();

        for (int irls_iter = 0; irls_iter < config.irls_max_iter; ++irls_iter) {
            G_w.setZero();
            b_w.setZero();

            for (const auto& te : train_entries) {
                // col te.col of A, row i → H col te.col
                Scalar predicted = Scalar(0);
                for (int f = 0; f < k; ++f)
                    predicted += H_mat(f, te.col) * w_row[f];

                Scalar residual = te.value - predicted;
                Scalar w = cv_irls_weight(residual, predicted, loss);
                Scalar wv = w * te.value;

                for (int a = 0; a < k; ++a) {
                    b_w(a) += H_mat(a, te.col) * wv;
                    for (int bb = a; bb < k; ++bb) {
                        G_w(a, bb) += w * H_mat(a, te.col) * H_mat(bb, te.col);
                    }
                }
            }

            // Symmetrize
            for (int a = 0; a < k; ++a)
                for (int bb = a + 1; bb < k; ++bb)
                    G_w(bb, a) = G_w(a, bb);

            if (config.W.L2 > 0) G_w.diagonal().array() += static_cast<Scalar>(config.W.L2);
            G_w.diagonal().array() += static_cast<Scalar>(1e-15);

            for (int f = 0; f < k; ++f) x_old(f) = w_row[f];

            FactorNet::primitives::detail::cd_nnls_col_fixed(
                G_w, b_w.data(), w_row, k,
                static_cast<Scalar>(config.W.L1), Scalar(0),
                config.W.nonneg, config.cd_maxit);

            Scalar max_change = 0;
            for (int f = 0; f < k; ++f) {
                Scalar rel = std::abs(w_row[f] - x_old(f))
                           / (std::abs(x_old(f)) + static_cast<Scalar>(1e-12));
                if (rel > max_change) max_change = rel;
            }
            if (max_change < static_cast<Scalar>(config.irls_tol)) break;
        }
    }
}

/**
 * @brief Compute explicit loss for non-MSE functions (host-side)
 *
 * Computes both train and test loss using the appropriate loss function.
 * Result is per-element average loss.
 */
template<typename Scalar>
void compute_cv_loss_irls(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_Td,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H_mat,
    int m, int n, int k,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    bool mask_zeros,
    const FactorNet::LossConfig<Scalar>& loss,
    uint64_t col_sub_thresh,
    double& train_loss, double& test_loss)
{
    double train_sum = 0, test_sum = 0;
    int64_t train_count = 0, test_count = 0;

    for (int j = 0; j < n; ++j) {
        bool col_in_sub = true;
        if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
            uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                        ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
            ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
            ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
            ch ^= ch >> 31;
            col_in_sub = (ch < col_sub_thresh);
        }

        if (mask_zeros) {
            for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
                int i = row_idx[p];
                Scalar predicted = Scalar(0);
                for (int f = 0; f < k; ++f)
                    predicted += W_Td(f, i) * H_mat(f, j);
                double entry_loss = static_cast<double>(
                    FactorNet::compute_loss(values[p], predicted, loss));

                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (is_test) { test_sum += entry_loss; test_count++; }
                else { train_sum += entry_loss; train_count++; }
            }
        } else {
            int p = col_ptr[j];
            for (int i = 0; i < m; ++i) {
                Scalar val = Scalar(0);
                if (p < col_ptr[j + 1] && row_idx[p] == i) { val = values[p]; ++p; }
                Scalar predicted = Scalar(0);
                for (int f = 0; f < k; ++f)
                    predicted += W_Td(f, i) * H_mat(f, j);
                double entry_loss = static_cast<double>(
                    FactorNet::compute_loss(val, predicted, loss));

                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (is_test) { test_sum += entry_loss; test_count++; }
                else { train_sum += entry_loss; train_count++; }
            }
        }
    }

    train_loss = (train_count > 0) ? train_sum / train_count : 0;
    test_loss = (test_count > 0) ? test_sum / test_count : 0;
}

}  // namespace cv_irls_detail

// =============================================================================
// Host-mediated masked CV helpers (user-supplied mask + speckled holdout)
// =============================================================================
namespace cv_mask_detail {

/**
 * @brief O(log n) check: is entry (row, col) in the user mask?
 *
 * Binary search in the mask's CSC column.
 */
template<typename Scalar>
inline bool is_user_masked(int row, int col,
                           const Eigen::SparseMatrix<Scalar>& mask) {
    const int start = mask.outerIndexPtr()[col];
    const int end   = mask.outerIndexPtr()[col + 1];
    return std::binary_search(mask.innerIndexPtr() + start,
                              mask.innerIndexPtr() + end, row);
}

/**
 * @brief Host-mediated masked H-update for CV (all columns)
 *
 * For each column j of H:
 *   1. Collect "train" entries: not holdout AND not user-masked
 *   2. Build Gram from train entries only (correct per-column correction)
 *   3. Solve CD NNLS
 *
 * For MSE loss (no IRLS), weights are all 1 → one solve per column.
 * For non-MSE (IRLS), iteratively reweighted solves per column.
 */
template<typename Scalar>
void masked_h_update_cv(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_T,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H_mat,
    int k, int m, int n,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    const CVNMFConfig& config,
    uint64_t col_sub_thresh,
    const Eigen::SparseMatrix<Scalar>& mask)
{
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    const bool is_mse = (config.loss_type == 0);
    FactorNet::LossConfig<Scalar> loss = cv_irls_detail::make_loss_config<Scalar>(config);
    const Scalar L1 = static_cast<Scalar>(config.H.L1);
    const Scalar L2 = static_cast<Scalar>(config.H.L2);
    const Scalar cd_tol = static_cast<Scalar>(config.cd_tol);
    const Scalar irls_tol_s = static_cast<Scalar>(config.irls_tol);

#ifdef _OPENMP
    int n_threads = (config.threads > 0) ? config.threads : 1;
    #pragma omp parallel for schedule(dynamic, 64) num_threads(n_threads)
#endif
    for (int j = 0; j < n; ++j) {
        // Check column subsample
        bool col_in_sub = true;
        if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
            uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                        ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
            ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
            ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
            ch ^= ch >> 31;
            col_in_sub = (ch < col_sub_thresh);
        }

        // Collect train entries: NOT holdout AND NOT user-masked
        struct TrainEntry { int row; Scalar value; };
        std::vector<TrainEntry> train_entries;

        if (config.mask_zeros) {
            for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
                int i = row_idx[p];
                if (is_user_masked(i, j, mask)) continue;
                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (!is_test) {
                    train_entries.push_back({i, values[p]});
                }
            }
        } else {
            int p = col_ptr[j];
            for (int i = 0; i < m; ++i) {
                if (is_user_masked(i, j, mask)) {
                    if (p < col_ptr[j + 1] && row_idx[p] == i) ++p;
                    continue;
                }
                Scalar val = Scalar(0);
                if (p < col_ptr[j + 1] && row_idx[p] == i) { val = values[p]; ++p; }
                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (!is_test) {
                    train_entries.push_back({i, val});
                }
            }
        }

        DenseMat G_w(k, k);
        DenseVec b_w(k);
        Scalar* x = H_mat.col(j).data();

        if (is_mse) {
            // MSE: build G and b once (weights = 1), single solve
            G_w.setZero();
            b_w.setZero();
            for (const auto& te : train_entries) {
                b_w.noalias() += te.value * W_T.col(te.row);
                for (int a = 0; a < k; ++a)
                    for (int bb = a; bb < k; ++bb)
                        G_w(a, bb) += W_T(a, te.row) * W_T(bb, te.row);
            }
            for (int a = 0; a < k; ++a)
                for (int bb = a + 1; bb < k; ++bb)
                    G_w(bb, a) = G_w(a, bb);
            if (L2 > 0) G_w.diagonal().array() += L2;
            G_w.diagonal().array() += static_cast<Scalar>(1e-15);

            FactorNet::primitives::detail::cd_nnls_col_fixed(
                G_w, b_w.data(), x, k, L1, Scalar(0),
                config.H.nonneg, config.cd_maxit);
        } else {
            // IRLS: iteratively reweighted solves
            DenseVec x_old(k);
            for (int irls_iter = 0; irls_iter < config.irls_max_iter; ++irls_iter) {
                G_w.setZero();
                b_w.setZero();
                for (const auto& te : train_entries) {
                    Scalar predicted = Scalar(0);
                    for (int f = 0; f < k; ++f)
                        predicted += W_T(f, te.row) * x[f];
                    Scalar residual = te.value - predicted;
                    Scalar w = cv_irls_detail::cv_irls_weight(residual, predicted, loss);
                    Scalar wv = w * te.value;
                    for (int a = 0; a < k; ++a) {
                        b_w(a) += W_T(a, te.row) * wv;
                        for (int bb = a; bb < k; ++bb)
                            G_w(a, bb) += w * W_T(a, te.row) * W_T(bb, te.row);
                    }
                }
                for (int a = 0; a < k; ++a)
                    for (int bb = a + 1; bb < k; ++bb)
                        G_w(bb, a) = G_w(a, bb);
                if (L2 > 0) G_w.diagonal().array() += L2;
                G_w.diagonal().array() += static_cast<Scalar>(1e-15);

                for (int f = 0; f < k; ++f) x_old(f) = x[f];
                FactorNet::primitives::detail::cd_nnls_col_fixed(
                    G_w, b_w.data(), x, k, L1, Scalar(0),
                    config.H.nonneg, config.cd_maxit);

                Scalar max_change = 0;
                for (int f = 0; f < k; ++f) {
                    Scalar rel = std::abs(x[f] - x_old(f))
                               / (std::abs(x_old(f)) + static_cast<Scalar>(1e-12));
                    if (rel > max_change) max_change = rel;
                }
                if (max_change < irls_tol_s) break;
            }
        }
    }
}

/**
 * @brief Host-mediated masked W-update for CV (all rows, via A^T)
 */
template<typename Scalar>
void masked_w_update_cv(
    const std::vector<int>& at_col_ptr,
    const std::vector<int>& at_row_idx,
    const std::vector<Scalar>& at_values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H_mat,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_T_mat,
    int k, int n, int m,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    const CVNMFConfig& config,
    uint64_t col_sub_thresh,
    const Eigen::SparseMatrix<Scalar>& mask)
{
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Precompute mask transpose for efficient row access
    Eigen::SparseMatrix<Scalar> mask_T = mask.transpose();

    const bool is_mse = (config.loss_type == 0);
    FactorNet::LossConfig<Scalar> loss = cv_irls_detail::make_loss_config<Scalar>(config);
    const Scalar L1 = static_cast<Scalar>(config.W.L1);
    const Scalar L2 = static_cast<Scalar>(config.W.L2);
    const Scalar cd_tol = static_cast<Scalar>(config.cd_tol);
    const Scalar irls_tol_s = static_cast<Scalar>(config.irls_tol);

#ifdef _OPENMP
    int n_threads = (config.threads > 0) ? config.threads : 1;
    #pragma omp parallel for schedule(dynamic, 64) num_threads(n_threads)
#endif
    for (int i = 0; i < m; ++i) {
        // Column i of A^T ≡ row i of A
        // For W-update, holdout(i, j) uses original (row=i, col=j) indexing

        struct TrainEntry { int col; Scalar value; };
        std::vector<TrainEntry> train_entries;

        // User mask for row i: check via mask_T column i
        auto is_row_masked = [&](int j) -> bool {
            const int start = mask_T.outerIndexPtr()[i];
            const int end   = mask_T.outerIndexPtr()[i + 1];
            return std::binary_search(mask_T.innerIndexPtr() + start,
                                      mask_T.innerIndexPtr() + end, j);
        };

        if (config.mask_zeros) {
            for (int p = at_col_ptr[i]; p < at_col_ptr[i + 1]; ++p) {
                int j = at_row_idx[p];
                if (is_row_masked(j)) continue;

                bool col_in_sub = true;
                if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
                    uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                                ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
                    ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
                    ch ^= ch >> 31;
                    col_in_sub = (ch < col_sub_thresh);
                }
                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (!is_test) {
                    train_entries.push_back({j, at_values[p]});
                }
            }
        } else {
            int p = at_col_ptr[i];
            for (int j = 0; j < n; ++j) {
                if (is_row_masked(j)) {
                    if (p < at_col_ptr[i + 1] && at_row_idx[p] == j) ++p;
                    continue;
                }
                Scalar val = Scalar(0);
                if (p < at_col_ptr[i + 1] && at_row_idx[p] == j) { val = at_values[p]; ++p; }

                bool col_in_sub = true;
                if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
                    uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                                ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
                    ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
                    ch ^= ch >> 31;
                    col_in_sub = (ch < col_sub_thresh);
                }
                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (!is_test) {
                    train_entries.push_back({j, val});
                }
            }
        }

        DenseMat G_w(k, k);
        DenseVec b_w(k);
        Scalar* w_row = W_T_mat.col(i).data();

        if (is_mse) {
            G_w.setZero();
            b_w.setZero();
            for (const auto& te : train_entries) {
                b_w.noalias() += te.value * H_mat.col(te.col);
                for (int a = 0; a < k; ++a)
                    for (int bb = a; bb < k; ++bb)
                        G_w(a, bb) += H_mat(a, te.col) * H_mat(bb, te.col);
            }
            for (int a = 0; a < k; ++a)
                for (int bb = a + 1; bb < k; ++bb)
                    G_w(bb, a) = G_w(a, bb);
            if (L2 > 0) G_w.diagonal().array() += L2;
            G_w.diagonal().array() += static_cast<Scalar>(1e-15);

            FactorNet::primitives::detail::cd_nnls_col_fixed(
                G_w, b_w.data(), w_row, k, L1, Scalar(0),
                config.W.nonneg, config.cd_maxit);
        } else {
            DenseVec x_old(k);
            for (int irls_iter = 0; irls_iter < config.irls_max_iter; ++irls_iter) {
                G_w.setZero();
                b_w.setZero();
                for (const auto& te : train_entries) {
                    Scalar predicted = Scalar(0);
                    for (int f = 0; f < k; ++f)
                        predicted += H_mat(f, te.col) * w_row[f];
                    Scalar residual = te.value - predicted;
                    Scalar w = cv_irls_detail::cv_irls_weight(residual, predicted, loss);
                    Scalar wv = w * te.value;
                    for (int a = 0; a < k; ++a) {
                        b_w(a) += H_mat(a, te.col) * wv;
                        for (int bb = a; bb < k; ++bb)
                            G_w(a, bb) += w * H_mat(a, te.col) * H_mat(bb, te.col);
                    }
                }
                for (int a = 0; a < k; ++a)
                    for (int bb = a + 1; bb < k; ++bb)
                        G_w(bb, a) = G_w(a, bb);
                if (L2 > 0) G_w.diagonal().array() += L2;
                G_w.diagonal().array() += static_cast<Scalar>(1e-15);

                for (int f = 0; f < k; ++f) x_old(f) = w_row[f];
                FactorNet::primitives::detail::cd_nnls_col_fixed(
                    G_w, b_w.data(), w_row, k, L1, Scalar(0),
                    config.W.nonneg, config.cd_maxit);

                Scalar max_change = 0;
                for (int f = 0; f < k; ++f) {
                    Scalar rel = std::abs(w_row[f] - x_old(f))
                               / (std::abs(x_old(f)) + static_cast<Scalar>(1e-12));
                    if (rel > max_change) max_change = rel;
                }
                if (max_change < irls_tol_s) break;
            }
        }
    }
}

/**
 * @brief Compute CV loss excluding user-masked entries
 *
 * Three-way partition: train entries → train loss,
 * holdout entries → test loss, masked entries → excluded.
 */
template<typename Scalar>
void masked_cv_loss(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<Scalar>& values,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& W_Td,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& H_mat,
    int m, int n, int k,
    const FactorNet::rng::SplitMix64& rng,
    uint64_t inv_prob,
    bool mask_zeros,
    const FactorNet::LossConfig<Scalar>& loss,
    uint64_t col_sub_thresh,
    const Eigen::SparseMatrix<Scalar>& mask,
    double& train_loss, double& test_loss)
{
    double train_sum = 0, test_sum = 0;
    int64_t train_count = 0, test_count = 0;

    for (int j = 0; j < n; ++j) {
        bool col_in_sub = true;
        if (col_sub_thresh < 0xFFFFFFFFFFFFFFFFULL) {
            uint64_t ch = (rng.state() ^ 0xa953b1c7217cc550ULL)
                        ^ (static_cast<uint64_t>(j) * 0x517cc1b727220a95ULL);
            ch = (ch ^ (ch >> 30)) * 0xbf58476d1ce4e5b9ULL;
            ch = (ch ^ (ch >> 27)) * 0x94d049bb133111ebULL;
            ch ^= ch >> 31;
            col_in_sub = (ch < col_sub_thresh);
        }

        if (mask_zeros) {
            for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
                int i = row_idx[p];
                if (is_user_masked(i, j, mask)) continue;
                Scalar predicted = Scalar(0);
                for (int f = 0; f < k; ++f)
                    predicted += W_Td(f, i) * H_mat(f, j);
                double entry_loss = static_cast<double>(
                    FactorNet::compute_loss(values[p], predicted, loss));

                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (is_test) { test_sum += entry_loss; test_count++; }
                else { train_sum += entry_loss; train_count++; }
            }
        } else {
            int p = col_ptr[j];
            for (int i = 0; i < m; ++i) {
                Scalar val = Scalar(0);
                if (p < col_ptr[j + 1] && row_idx[p] == i) { val = values[p]; ++p; }
                if (is_user_masked(i, j, mask)) continue;
                Scalar predicted = Scalar(0);
                for (int f = 0; f < k; ++f)
                    predicted += W_Td(f, i) * H_mat(f, j);
                double entry_loss = static_cast<double>(
                    FactorNet::compute_loss(val, predicted, loss));

                bool is_test = col_in_sub && rng.is_holdout(i, j, inv_prob);
                if (is_test) { test_sum += entry_loss; test_count++; }
                else { train_sum += entry_loss; train_count++; }
            }
        }
    }

    train_loss = (train_count > 0) ? train_sum / train_count : 0;
    test_loss = (test_count > 0) ? test_sum / test_count : 0;
}

}  // namespace cv_mask_detail

// =============================================================================
// Main CV NMF fitting function
// =============================================================================

/**
 * @brief Fit NMF with cross-validation on GPU.
 *
 * H update pipeline (per-column modified Gram on GPU):
 *   1. Compute G_full = W'W on GPU (SYRK)
 *   2. Compute delta_G, delta_b, B_full per column on GPU
 *   3. Batch CV NNLS on GPU: each column solves with G_full - delta_G_j
 *   4. Scale rows of H
 *
 * W update pipeline (per-row modified Gram on GPU via A^T):
 *   1. Compute G_full = H'H on GPU (SYRK)
 *   2. Compute per-row deltas on GPU using A^T (CSR of A)
 *   3. Batch CV NNLS on GPU
 *   4. Scale rows of W
 *
 * Loss computation:
 *   mask_zeros=TRUE:  GPU kernel iterating non-zeros, O(nnz·k)
 *   mask_zeros=FALSE: Gram trick total loss, approximate test loss
 */
template<typename Scalar>
CVNMFResult nmf_cv_fit_gpu(
    GPUContext& ctx,
    // Host-side sparse matrix (for transpose computation)
    const std::vector<int>& h_col_ptr,
    const std::vector<int>& h_row_idx,
    const std::vector<Scalar>& h_values,
    int m, int n,
    // Initial factors (host, k × m, k × n, k)
    Scalar* W_host,
    Scalar* H_host,
    Scalar* d_host,
    const CVNMFConfig& config)
{
    const int k = config.k;
    const int nnz = static_cast<int>(h_values.size());

    CVNMFResult result;

    // Create RNG
    FactorNet::rng::SplitMix64 rng(config.cv_seed);
    uint64_t inv_prob = config.holdout_fraction > 0
                      ? static_cast<uint64_t>(1.0 / config.holdout_fraction) : 0;

    // Column subsample threshold
    uint64_t col_sub_thresh = (config.col_subsample_frac >= 1.0)
        ? 0xFFFFFFFFFFFFFFFFULL
        : static_cast<uint64_t>(config.col_subsample_frac * 18446744073709551615.0);

    // Precompute norms
    CVNorms norms = precompute_cv_norms(
        h_col_ptr, h_row_idx,
        std::vector<double>(h_values.begin(), h_values.end()),
        m, n, rng, inv_prob, config.mask_zeros, col_sub_thresh);

    if (config.verbose) {
        FACTORNET_GPU_LOG(config.verbose, "CV NMF GPU: %dx%d, k=%d, nnz=%d\n", m, n, k, nnz);
        FACTORNET_GPU_LOG(config.verbose, "  Train: %lld entries, ||A_train||²=%.3e\n",
               (long long)norms.n_train, norms.A_train_sq);
        FACTORNET_GPU_LOG(config.verbose, "  Test:  %lld entries, ||A_test||²=%.3e\n",
               (long long)norms.n_test, norms.A_test_sq);
    }

    // Upload sparse matrix to GPU
    SparseMatrixGPU<Scalar> A(m, n, nnz,
                               h_col_ptr.data(), h_row_idx.data(), h_values.data());

    // Compute and upload A^T (CSR of A = CSC of A^T) for GPU W-update deltas
    std::vector<int> at_row_ptr, at_col_idx;
    std::vector<Scalar> at_values;
    compute_csc_transpose(h_col_ptr, h_row_idx, h_values, m, n, nnz,
                          at_row_ptr, at_col_idx, at_values);
    // A^T has dimensions n×m, stored in CSC with m "columns" (rows of A)
    SparseMatrixGPU<Scalar> At(n, m, nnz,
                                at_row_ptr.data(), at_col_idx.data(), at_values.data());
    // Free host transpose memory — GPU copy is all we need
    // (unless IRLS or masking needs it for W-update on host)
    const bool use_irls = config.requires_irls();
    const bool use_mask = config.has_mask();
    if (!use_irls && !use_mask) {
        std::vector<int>().swap(at_row_ptr); std::vector<int>().swap(at_col_idx);
        std::vector<Scalar>().swap(at_values);
    }

    // Upload factors to GPU
    DenseMatrixGPU<Scalar> W(k, m, W_host);
    DenseMatrixGPU<Scalar> H(k, n, H_host);
    DeviceMemory<Scalar> d_vec(k);
    d_vec.upload(d_host, k);

    // Allocate GPU workspace
    DenseMatrixGPU<Scalar> G_full(k, k);

    // Workspace for batched Cholesky CV pipeline
    DenseMatrixGPU<Scalar> delta_G_h(k * k, n);
    DenseMatrixGPU<Scalar> delta_b_h(k, n);
    DenseMatrixGPU<Scalar> B_full_h(k, n);
    DenseMatrixGPU<Scalar> delta_G_w(k * k, m);
    DenseMatrixGPU<Scalar> delta_b_w(k, m);
    DenseMatrixGPU<Scalar> B_full_w(k, m);
    DenseMatrixGPU<Scalar> H_out(k, n);
    DenseMatrixGPU<Scalar> W_out(k, m);

    DeviceMemory<double> scratch(8);

    // Host buffers removed — W-update deltas now computed on GPU

    // Pre-allocate loss workspace (outside loop for efficiency)
    DenseMatrixGPU<Scalar> G_loss(k, k);
    // W_D: temporary d-scaled W for Gram trick loss (reuse W_out buffer)
    // W_out is unused after NNLS writes directly to W
    DenseMatrixGPU<Scalar>& W_D = W_out;

    // For loss with mask_zeros=FALSE, need the RHS for cross-term
    DenseMatrixGPU<Scalar> B_save(k, n);

    // Previous W for convergence check (GPU-resident — no PCIe download)
    DenseMatrixGPU<Scalar> W_prev_gpu(k, m);

    // Tier 2 host-side buffers (for Gram modification and bounds)
    const bool has_t2 = config.has_tier2();
    using DenseMatHost = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVecHost = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    DenseMatHost G_host(k, k);          // for downloading/uploading Gram
    DenseMatHost H_host_mat(k, n);      // current H (Eigen wrapper)
    DenseMatHost W_host_mat(k, m);      // current W (Eigen wrapper)

    // GPU graph regularization handles (cuSPARSE SpMM + cuBLAS GEMM)
    GpuGraphRegHandle<Scalar> graph_reg_H, graph_reg_W;
    if (config.graph_H_data && config.H.graph_lambda > 0) {
        using SpMat = Eigen::SparseMatrix<Scalar>;
        const SpMat* graph_H = static_cast<const SpMat*>(config.graph_H_data);
        graph_reg_H.init(ctx, graph_H->outerIndexPtr(), graph_H->innerIndexPtr(),
                         graph_H->valuePtr(), static_cast<int>(graph_H->rows()),
                         static_cast<int>(graph_H->nonZeros()), k, H.data.get());
    }
    if (config.graph_W_data && config.W.graph_lambda > 0) {
        using SpMat = Eigen::SparseMatrix<Scalar>;
        const SpMat* graph_W = static_cast<const SpMat*>(config.graph_W_data);
        graph_reg_W.init(ctx, graph_W->outerIndexPtr(), graph_W->innerIndexPtr(),
                         graph_W->valuePtr(), static_cast<int>(graph_W->rows()),
                         static_cast<int>(graph_W->nonZeros()), k, W.data.get());
    }

    // Upload user mask to device for GPU-native masked updates (MSE path only)
    std::unique_ptr<SparseMatrixGPU<Scalar>> d_user_mask;
    if (use_mask && !use_irls) {
        const auto* user_mask = static_cast<const Eigen::SparseMatrix<Scalar>*>(config.mask_data);
        d_user_mask = std::make_unique<SparseMatrixGPU<Scalar>>(
            user_mask->rows(), user_mask->cols(),
            static_cast<int>(user_mask->nonZeros()),
            user_mask->outerIndexPtr(),
            user_mask->innerIndexPtr(),
            user_mask->valuePtr());
    }

    double best_test = 1e30;
    int best_iter = 0;

    for (int iter = 0; iter < config.max_iter; ++iter) {

        // Save W for convergence check (device-to-device copy — no PCIe)
        CUDA_CHECK(cudaMemcpyAsync(
            W_prev_gpu.data.get(), W.data.get(),
            static_cast<size_t>(k) * m * sizeof(Scalar),
            cudaMemcpyDeviceToDevice, ctx.stream));

        // =============================================================
        // H UPDATE
        // =============================================================

        // 1. Gram: G_full = W'W + (eps + L2_H) * I (skip for projective/symmetric)
        if (!config.projective && !config.symmetric) {
        compute_gram_regularized_gpu(ctx, W, G_full,
                                     static_cast<Scalar>(config.H.L2));

        // Tier 2 Gram modifications for H update
        if (has_t2 && (config.H.L21 > 0 || config.ortho_H > 0
                       || (config.graph_H_data && config.H.graph_lambda > 0))) {
            // Apply L21 and angular — GPU-native (no PCIe round-trip)
            if (config.H.L21 > 0)
                ::FactorNet::nmf::gpu_features::apply_L21_gpu(ctx, G_full, H, static_cast<Scalar>(config.H.L21));
            if (config.ortho_H > 0)
                ::FactorNet::nmf::gpu_features::apply_angular_gpu(ctx, G_full, H, static_cast<Scalar>(config.ortho_H));
            // Apply graph regularization — GPU-native (cuSPARSE SpMM + cuBLAS GEMM)
            if (config.graph_H_data && config.H.graph_lambda > 0) {
                graph_reg_H.apply(ctx, G_full.data.get(), H.data.get(),
                                  H.cols, static_cast<Scalar>(config.H.graph_lambda));
            }
        }
        } // end !projective && !symmetric guard for Gram + tier2

        if (config.projective) {
            // Projective NMF: H = diag(d) · W^T · A (SpMM, uses full A — no CV holdout)
            // Step 1: B = W · A via cuSPARSE SpMM (k×n result)
            compute_rhs_forward_gpu(ctx, W, A, B_full_h);
            // Step 2: Apply d scaling: H(f,j) = d(f) * B(f,j)
            {
                std::vector<Scalar> h_d(k);
                d_vec.download(h_d.data(), k);
                CUDA_CHECK(cudaMemcpyAsync(
                    H.data.get(), B_full_h.data.get(),
                    static_cast<size_t>(k) * n * sizeof(Scalar),
                    cudaMemcpyDeviceToDevice, ctx.stream));
                for (int f = 0; f < k; ++f) {
                    if (std::abs(h_d[f]) > Scalar(1e-15)) {
                        Scalar alpha = h_d[f];
                        if constexpr (std::is_same_v<Scalar, float>) {
                            cublasSscal(ctx.cublas, n, &alpha, H.data.get() + f, k);
                        } else {
                            cublasDscal(ctx.cublas, n, &alpha, H.data.get() + f, k);
                        }
                    }
                }
            }
        } else if (config.symmetric) {
            // Symmetric NMF: H = W (device-to-device copy, k×min(m,n))
            int copy_cols = std::min(m, n);
            CUDA_CHECK(cudaMemcpyAsync(
                H.data.get(), W.data.get(),
                static_cast<size_t>(k) * copy_cols * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        } else if (use_mask && !use_irls) {
            // The delta kernels handle both simultaneously: user-masked entries add to
            // delta_G (Gram correction) but not B_full; CV-holdout entries add to both
            // delta_G and delta_b.
            compute_cv_deltas_h_gpu(ctx, A, W, k, n,
                                    rng.state(), inv_prob,
                                    delta_G_h, delta_b_h, &B_full_h,
                                    /*W_half=*/nullptr,
                                    d_user_mask->col_ptr.get(),
                                    d_user_mask->row_indices.get(),
                                    config.mask_zeros);
            cholesky_cv_tiled_gpu(ctx, G_full, delta_G_h, B_full_h, delta_b_h,
                                  H, config.H.nonneg);
        } else if (use_mask && use_irls) {
            // --- Masked IRLS H-update: host-mediated (per-column weights needed) ---
            ctx.sync();
            W.download_to(W_host_mat.data());
            H.download_to(H_host_mat.data());

            const auto* user_mask = static_cast<const Eigen::SparseMatrix<Scalar>*>(config.mask_data);
            cv_mask_detail::masked_h_update_cv(
                h_col_ptr, h_row_idx, h_values,
                W_host_mat, H_host_mat,
                k, m, n, rng, inv_prob, config, col_sub_thresh, *user_mask);

            H.upload_from(H_host_mat.data());
        } else if (use_irls) {
            // --- IRLS H-update: host-mediated CPU IRLS solver ---
            ctx.sync();
            W.download_to(W_host_mat.data());
            H.download_to(H_host_mat.data());

            cv_irls_detail::irls_h_update_cv(
                h_col_ptr, h_row_idx, h_values,
                W_host_mat, H_host_mat,
                k, m, n, rng, inv_prob, config, col_sub_thresh);

            H.upload_from(H_host_mat.data());
        } else {
            // Batched Cholesky CV pipeline
            compute_cv_deltas_h_gpu(ctx, A, W, k, n,
                                    rng.state(), inv_prob,
                                    delta_G_h, delta_b_h, &B_full_h,
                                    nullptr,
                                    nullptr, nullptr,
                                    config.mask_zeros);
            cholesky_cv_tiled_gpu(ctx, G_full, delta_G_h, B_full_h, delta_b_h,
                                  H, config.H.nonneg);
        }

        // Tier 2: apply upper bounds to H — GPU clamp kernel (no PCIe round-trip)
        if (config.H.upper_bound > 0)
            ::FactorNet::gpu::apply_upper_bound_gpu(ctx, H,
                static_cast<Scalar>(config.H.upper_bound));

        // Scale rows of H, update d (skip for projective — H is a projection, not optimized)
        if (!config.projective) {
        if (config.norm_type_int == 1) {
            // L2: use GPU kernel
            scale_rows_opt_gpu(ctx, H, d_vec);
        } else if (config.norm_type_int == 0) {
            // L1: use GPU kernel (no PCIe round-trip)
            l1_norm_rows_gpu(ctx, H, d_vec);
        } else {
            // None: d = ones (GPU kernel)
            set_ones_gpu(ctx, d_vec, k);
        }
        } // end !projective

        // =============================================================
        // W UPDATE (GPU deltas via A^T)
        // =============================================================

        // 1. Gram: G_full = H'H + (eps + L2_W) * I
        compute_gram_regularized_gpu(ctx, H, G_full,
                                     static_cast<Scalar>(config.W.L2));

        // Tier 2 Gram modifications for W update
        if (has_t2 && (config.W.L21 > 0 || config.ortho_W > 0
                       || (config.graph_W_data && config.W.graph_lambda > 0))) {
            // Apply L21 and angular — GPU-native (no PCIe round-trip)
            if (config.W.L21 > 0)
                ::FactorNet::nmf::gpu_features::apply_L21_gpu(ctx, G_full, W, static_cast<Scalar>(config.W.L21));
            if (config.ortho_W > 0)
                ::FactorNet::nmf::gpu_features::apply_angular_gpu(ctx, G_full, W, static_cast<Scalar>(config.ortho_W));
            // Apply graph regularization — GPU-native (cuSPARSE SpMM + cuBLAS GEMM)
            if (config.graph_W_data && config.W.graph_lambda > 0) {
                graph_reg_W.apply(ctx, G_full.data.get(), W.data.get(),
                                  W.cols, static_cast<Scalar>(config.W.graph_lambda));
            }
        }

        if (use_mask && !use_irls) {
            // --- GPU-native masked CV W-update: delta-kernel with user mask + CV holdout ---
            compute_cv_deltas_w_gpu(ctx, At, H, k, m,
                                    rng.state(), inv_prob,
                                    delta_G_w, delta_b_w, &B_full_w,
                                    /*H_half=*/nullptr,
                                    d_user_mask->col_ptr.get(),
                                    d_user_mask->row_indices.get(),
                                    config.mask_zeros);
            cholesky_cv_tiled_gpu(ctx, G_full, delta_G_w, B_full_w, delta_b_w,
                                  W, config.W.nonneg);
        } else if (use_mask && use_irls) {
            // --- Masked IRLS W-update: host-mediated (per-column weights needed) ---
            ctx.sync();
            H.download_to(H_host_mat.data());
            W.download_to(W_host_mat.data());

            const auto* user_mask = static_cast<const Eigen::SparseMatrix<Scalar>*>(config.mask_data);
            cv_mask_detail::masked_w_update_cv(
                at_row_ptr, at_col_idx, at_values,
                H_host_mat, W_host_mat,
                k, n, m, rng, inv_prob, config, col_sub_thresh, *user_mask);

            W.upload_from(W_host_mat.data());
        } else if (use_irls) {
            // --- IRLS W-update: host-mediated CPU IRLS solver ---
            ctx.sync();
            H.download_to(H_host_mat.data());
            W.download_to(W_host_mat.data());

            cv_irls_detail::irls_w_update_cv(
                at_row_ptr, at_col_idx, at_values,
                H_host_mat, W_host_mat,
                k, n, m, rng, inv_prob, config, col_sub_thresh);

            W.upload_from(W_host_mat.data());
        } else {
            // Batched Cholesky CV pipeline
            compute_cv_deltas_w_gpu(ctx, At, H, k, m,
                                    rng.state(), inv_prob,
                                    delta_G_w, delta_b_w, &B_full_w,
                                    nullptr,
                                    nullptr, nullptr,
                                    config.mask_zeros);
            cholesky_cv_tiled_gpu(ctx, G_full, delta_G_w, B_full_w, delta_b_w,
                                  W, config.W.nonneg);
        }

        // Tier 2: apply upper bounds to W — GPU clamp kernel (no PCIe round-trip)
        if (config.W.upper_bound > 0)
            ::FactorNet::gpu::apply_upper_bound_gpu(ctx, W,
                static_cast<Scalar>(config.W.upper_bound));

        // Scale rows of W
        if (config.norm_type_int == 1) {
            // L2: use GPU kernel
            scale_rows_opt_gpu(ctx, W, d_vec);
        } else if (config.norm_type_int == 0) {
            // L1: use GPU kernel (no PCIe round-trip)
            l1_norm_rows_gpu(ctx, W, d_vec);
        } else {
            // None: d = ones (GPU kernel)
            set_ones_gpu(ctx, d_vec, k);
        }

        // Enforce H = W after W-update for symmetric NMF
        if (config.symmetric) {
            int copy_cols = std::min(m, n);
            CUDA_CHECK(cudaMemcpyAsync(
                H.data.get(), W.data.get(),
                static_cast<size_t>(k) * copy_cols * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
        }

        // =============================================================
        // LOSS COMPUTATION
        // =============================================================

        double train_mse = 0, test_mse = 0;

        if (use_mask) {
            // Masked loss: explicit per-element, skip user-masked entries
            ctx.sync();
            W.download_to(W_host_mat.data());
            H.download_to(H_host_mat.data());

            DenseMatHost W_Td = W_host_mat;
            DenseVecHost d_host_vec(k);
            d_vec.download(d_host_vec.data(), k);
            for (int f = 0; f < k; ++f)
                W_Td.row(f) *= d_host_vec(f);

            const auto* user_mask = static_cast<const Eigen::SparseMatrix<Scalar>*>(config.mask_data);
            FactorNet::LossConfig<Scalar> loss = cv_irls_detail::make_loss_config<Scalar>(config);
            cv_mask_detail::masked_cv_loss(
                h_col_ptr, h_row_idx, h_values,
                W_Td, H_host_mat, m, n, k,
                rng, inv_prob, config.mask_zeros, loss,
                col_sub_thresh, *user_mask, train_mse, test_mse);
        } else if (use_irls) {
            // Non-MSE loss: explicit per-element computation on host
            ctx.sync();
            W.download_to(W_host_mat.data());
            H.download_to(H_host_mat.data());

            // Apply scaling: W_Td = diag(d) * W
            DenseMatHost W_Td = W_host_mat;
            DenseVecHost d_host_vec(k);
            d_vec.download(d_host_vec.data(), k);
            for (int f = 0; f < k; ++f)
                W_Td.row(f) *= d_host_vec(f);

            FactorNet::LossConfig<Scalar> loss = cv_irls_detail::make_loss_config<Scalar>(config);
            cv_irls_detail::compute_cv_loss_irls(
                h_col_ptr, h_row_idx, h_values,
                W_Td, H_host_mat, m, n, k,
                rng, inv_prob, config.mask_zeros, loss,
                col_sub_thresh, train_mse, test_mse);
        } else if (config.mask_zeros) {
            // Direct computation at non-zeros on GPU
            compute_cv_loss_sparse_gpu(ctx, A, W, H, d_vec,
                                       rng.state(), inv_prob,
                                       train_mse, test_mse, col_sub_thresh);
        } else {
            // Gram trick for total loss, approximate test loss
            // Must use d-scaled W for correct predictions:
            //   pred[i,j] = Σ_f d[f] * W[f,i] * H[f,j]
            //   = (D*W)[:,i]' * H[:,j]  where D = diag(d)
            //
            // Create W_D = diag(d) * W (temporary, reuses W_out buffer)
            CUDA_CHECK(cudaMemcpyAsync(
                W_D.data.get(), W.data.get(),
                static_cast<size_t>(k) * m * sizeof(Scalar),
                cudaMemcpyDeviceToDevice, ctx.stream));
            absorb_d_gpu(ctx, W_D, d_vec);

            // Gram and RHS with d-scaled W
            compute_gram_gpu(ctx, W_D, G_loss);
            compute_rhs_forward_gpu(ctx, W_D, A, B_save);

            compute_cv_loss_dense_gpu(ctx, A, W, H, d_vec,
                                      G_loss, B_save,
                                      config.holdout_fraction,
                                      rng.state(), inv_prob,
                                      norms.n_train, norms.n_test,
                                      norms.A_train_sq, norms.A_test_sq,
                                      scratch,
                                      train_mse, test_mse);
        }

        result.train_loss = train_mse;
        result.test_loss = test_mse;
        result.train_history.push_back(train_mse);
        result.test_history.push_back(test_mse);
        result.iterations = iter + 1;

        if (test_mse < best_test) {
            best_test = test_mse;
            best_iter = iter + 1;
        }

        FACTORNET_GPU_LOG(config.verbose, "  iter %3d: train=%.6e  test=%.6e\n",
                   iter + 1, train_mse, test_mse);

        // Convergence check via factor correlation (GPU-native — single scalar download)
        if (iter > 0) {
            double corr = compute_factor_correlation_gpu(ctx, W, W_prev_gpu);
            if ((1.0 - corr) < config.tol) {
                result.converged = true;
                FACTORNET_GPU_LOG(config.verbose, "  Converged at iter %d (corr=%.6f)\n",
                           iter + 1, corr);
                break;
            }
        }
    }

    result.best_test_loss = best_test;
    result.best_iter = best_iter;

    // Download final factors
    W.download_to(W_host);
    H.download_to(H_host);
    d_vec.download(d_host, k);

    return result;
}

} // namespace gpu
} // namespace FactorNet
