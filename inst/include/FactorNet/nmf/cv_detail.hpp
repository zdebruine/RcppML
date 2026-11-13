// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/cv_detail.hpp
 * @brief Internal helper functions for cross-validation NMF
 *
 * Extracted from fit_cv.hpp to keep the main fitting function readable.
 * Contains:
 *   - CVWorkspace             : per-thread buffers for one column/row solve
 *   - apply_gram_correction   : per-column Gram down-date (G - F_excl F_excl^T)
 *   - gram_from_observed_col/row : per-column Gram from observed entries only
 *   - cv_solve_col            : one k-variable solve (Cholesky+clip or exact CD)
 *   - irls_solve_col_cv       : IRLS-wrapped H-column solve
 *   - irls_solve_row_cv       : IRLS-wrapped W-row solve
 *   - list_holdout_col/row    : held-out indices of a column / row
 *   - rhs_excluding_col/row   : right-hand side over the non-excluded entries
 *   - CVMaskCache             : holdout lists built once per fit
 *   - User mask helpers       : adjust_rhs_for_user_mask_h/w, is_user_masked_h
 *
 * Conventions used throughout:
 *   - "excluded" index lists are sorted ascending.
 *   - With mask_zeros = true, zeros are MISSING: they enter neither the Gram
 *     nor the right-hand side nor any loss. For sparse input every stored
 *     entry counts as observed; for dense input every non-zero entry does.
 *   - cd_nnls_col_fixed() is called with the residual b - G x, as its
 *     contract requires when x is a warm start.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/primitives/cpu/data_accessor.hpp>
#include <FactorNet/primitives/cpu/nnls_batch_irls.hpp>
#include <FactorNet/primitives/cpu/cholesky_clip.hpp>
#include <FactorNet/primitives/cpu/nnls_batch.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/nmf/variant_helpers.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace FactorNet {
namespace nmf {

// ========================================================================
// CV-specific internal helpers
// ========================================================================

namespace detail {

/// One training entry of a column (idx = row) or of a row (idx = column).
template<typename Scalar>
struct CVEntry {
    int idx;
    Scalar value;
};

/**
 * @brief Per-thread buffers for one per-column (or per-row) CV solve.
 *
 * The index buffer `buf` grows on demand, so the number of excluded or
 * observed entries in a column is never limited by the initial estimate.
 */
template<typename Scalar>
struct CVWorkspace {
    DenseVector<Scalar> b;        ///< train RHS (k)
    DenseVector<Scalar> b_test;   ///< RHS contribution of held-out entries (k)
    DenseVector<Scalar> x;        ///< solution (k)
    DenseVector<Scalar> x_old;    ///< previous IRLS iterate (k)
    DenseVector<Scalar> x_try;    ///< full IRLS step before halving (k)
    DenseVector<Scalar> scratch;  ///< residual / RHS copy (k)
    DenseMatrix<Scalar> G_local;  ///< per-column Gram (k × k)
    DenseMatrix<Scalar> buf;      ///< gathered factor columns (k × n_idx)
    std::vector<int> test_idx;
    std::vector<int> excluded_idx;
    std::vector<CVEntry<Scalar>> entries;

    CVWorkspace(int k, int est_cols)
        : b(k), b_test(k), x(k), x_old(k), x_try(k), scratch(k),
          G_local(k, k), buf(k, std::max(est_cols, 1))
    {
        test_idx.reserve(static_cast<size_t>(std::max(est_cols, 1)));
    }
};

// ========================================================================
// Gram helpers (MSE path)
// ========================================================================

/**
 * @brief Apply Gram correction for CV: G_local = G - F_excl * F_excl^T
 *
 * Copies the full Gram G into G_local, then subtracts the outer products of
 * the factor columns at the excluded indices (held-out and user-masked
 * entries of this column).
 *
 * @param G           Full Gram matrix F * F^T (k×k, symmetric)
 * @param W_T         Factor in column storage (k × m for the H update,
 *                    k × n for the W update)
 * @param test_rows   Excluded indices for this column
 * @param G_local     Output corrected Gram (k×k)
 * @param W_test_buf  Workspace matrix (k × anything); resized when too small
 */
template<typename Scalar>
inline void apply_gram_correction(
    const DenseMatrix<Scalar>& G,
    const DenseMatrix<Scalar>& W_T,
    const std::vector<int>& test_rows,
    DenseMatrix<Scalar>& G_local,
    DenseMatrix<Scalar>& W_test_buf)
{
    G_local = G;
    if (!test_rows.empty()) {
        const int n_test = static_cast<int>(test_rows.size());
        if (W_test_buf.rows() != W_T.rows() || W_test_buf.cols() < n_test)
            W_test_buf.resize(W_T.rows(), n_test);
        for (int idx = 0; idx < n_test; ++idx)
            W_test_buf.col(idx) = W_T.col(test_rows[idx]);
        auto W_sub = W_test_buf.leftCols(n_test);
        G_local.template selfadjointView<Eigen::Lower>()
            .rankUpdate(W_sub, Scalar(-1));
        G_local.template triangularView<Eigen::Upper>() =
            G_local.transpose();
    }
}

/// Fill G_local = buf[:, 0:cnt] * buf[:, 0:cnt]^T (zero when cnt == 0).
template<typename Scalar>
inline void gram_from_buffer(
    const DenseMatrix<Scalar>& buf, int cnt, DenseMatrix<Scalar>& G_local)
{
    G_local.setZero();
    if (cnt > 0) {
        auto sub = buf.leftCols(cnt);
        G_local.template selfadjointView<Eigen::Lower>()
            .rankUpdate(sub, Scalar(1));
        G_local.template triangularView<Eigen::Upper>() =
            G_local.transpose();
    }
}

/// Append F.col(idx) to buf (growing buf when needed).
template<typename Scalar>
inline void push_factor_col(
    const DenseMatrix<Scalar>& F, int idx, DenseMatrix<Scalar>& buf, int& cnt)
{
    if (buf.rows() != F.rows()) {
        // Only possible before anything was pushed
        buf.resize(F.rows(), std::max<Eigen::Index>(buf.cols(), 1));
    }
    if (cnt >= buf.cols())
        buf.conservativeResize(Eigen::NoChange, std::max<Eigen::Index>(2 * buf.cols(), cnt + 1));
    buf.col(cnt) = F.col(idx);
    ++cnt;
}

/**
 * @brief Gram over the OBSERVED training entries of column j (mask_zeros = true).
 *
 * G_local = sum over rows i that are observed (stored for sparse, non-zero for
 * dense) and not excluded of W_T.col(i) W_T.col(i)^T.
 *
 * @return number of training entries in the column
 */
template<typename Scalar, typename MatrixType>
inline int gram_from_observed_col(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    int j,
    const std::vector<int>& excluded_rows,
    DenseMatrix<Scalar>& G_local,
    DenseMatrix<Scalar>& buf)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    int cnt = 0;
    size_t ex = 0;
    const size_t n_ex = excluded_rows.size();
    DA::for_col(A, j, [&](int i, Scalar val) {
        if (!DA::is_sparse && val == Scalar(0)) return;
        while (ex < n_ex && excluded_rows[ex] < i) ++ex;
        if (ex < n_ex && excluded_rows[ex] == i) return;
        push_factor_col(W_T, i, buf, cnt);
    });
    gram_from_buffer(buf, cnt, G_local);
    return cnt;
}

/**
 * @brief Gram over the OBSERVED training entries of row i (mask_zeros = true).
 *
 * For sparse: pass A^T (n×m CSC); column i of A^T is row i of A.
 * For dense:  pass A (m×n).
 *
 * @return number of training entries in the row
 */
template<typename Scalar, typename MatrixType>
inline int gram_from_observed_row(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    int i,
    const std::vector<int>& excluded_cols,
    DenseMatrix<Scalar>& G_local,
    DenseMatrix<Scalar>& buf)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    int cnt = 0;
    size_t ex = 0;
    const size_t n_ex = excluded_cols.size();
    auto visit = [&](int col, Scalar val) {
        if (!DA::is_sparse && val == Scalar(0)) return;
        while (ex < n_ex && excluded_cols[ex] < col) ++ex;
        if (ex < n_ex && excluded_cols[ex] == col) return;
        push_factor_col(H, col, buf, cnt);
    };
    if constexpr (DA::is_sparse) {
        DA::for_col(data, i, visit);
    } else {
        DA::for_row(data, i, visit);
    }
    gram_from_buffer(buf, cnt, G_local);
    return cnt;
}

// ========================================================================
// Single k-variable solve
// ========================================================================

/**
 * @brief Solve min_x 0.5 x'Gx - b'x (+ L1, x >= 0 when nonneg) for one column.
 *
 * @param G        System matrix (k × k); L2 and other diagonal penalties are
 *                 already included
 * @param b        Raw right-hand side (NOT a residual); left unchanged
 * @param x        In: warm start. Out: solution
 * @param L1       L1 penalty, forwarded to the solver
 * @param upper_bound  Box constraint x <= upper_bound enforced inside the
 *                 solver (0 = none)
 * @param scratch  k-vector workspace
 * @param allow_cholesky  If false, coordinate descent is used even when
 *                 config.solver_mode == 1 (e.g. rank-deficient systems)
 *
 * solver_mode 1 uses the Cholesky + clip solver; if it produces a non-finite
 * result (singular system) the solve falls back to coordinate descent.
 * Coordinate descent is given the residual b - G x for the warm start x, as
 * cd_nnls_col_fixed() requires.
 */
template<typename Scalar>
inline void cv_solve_col(
    const DenseMatrix<Scalar>& G,
    const DenseVector<Scalar>& b,
    DenseVector<Scalar>& x,
    Scalar L1,
    bool nonneg,
    Scalar upper_bound,
    const ::FactorNet::NMFConfig<Scalar>& config,
    DenseVector<Scalar>& scratch,
    bool allow_cholesky = true)
{
    const int k = static_cast<int>(x.size());
    if (config.solver_mode == 1 && allow_cholesky) {
        scratch = b;
        primitives::detail::cholesky_clip_col(
            G, scratch.data(), x.data(), k,
            L1, Scalar(0), nonneg,
            config.cd_max_iter, config.cd_tol, upper_bound);
        if (x.allFinite()) return;
        x.setZero();
    }
    if (!x.allFinite()) x.setZero();
    // Coordinate descent needs a feasible warm start
    if (nonneg) x = x.cwiseMax(Scalar(0));
    if (upper_bound > Scalar(0)) x = x.cwiseMin(upper_bound);

    // Residual for the warm start: r = b - G x
    scratch = b;
    scratch.noalias() -= G * x;
    primitives::detail::cd_nnls_col_fixed(
        G, scratch.data(), x.data(), k,
        L1, Scalar(0), nonneg,
        config.cd_max_iter, upper_bound, config.cd_tol);
}

// ========================================================================
// IRLS helpers for per-column/row CV NNLS
// ========================================================================

/**
 * @brief Penalized objective of one column for the IRLS step control.
 *
 *   F(x) = sum_e f(y_e, f_e . x) + L1 |x|_1
 *          + (1/2) x'(G_reg + ridge I) x - b_reg' x
 *
 * with f = unit deviance / 2 (or the Huber function of the deviance residual
 * for robust fits), the same merit the non-CV IRLS solver monitors.
 */
template<typename Scalar>
inline double irls_merit_entries_cv(
    const std::vector<CVEntry<Scalar>>& entries,
    const DenseMatrix<Scalar>& F,
    const DenseVector<Scalar>& x,
    const LossConfig<Scalar>& loss,
    const DenseMatrix<Scalar>& G_reg,
    const Scalar* b_reg,
    Scalar ridge_col,
    Scalar L1,
    bool nonneg,
    Scalar theta_const,
    const Scalar* theta_by_idx)
{
    const int k = static_cast<int>(F.rows());
    double total = 0.0;
    for (const auto& te : entries) {
        const Scalar predicted = F.col(te.idx).dot(x);
        const Scalar theta = theta_by_idx ? theta_by_idx[te.idx] : theta_const;
        total += primitives::detail::irls_entry_merit(te.value, predicted, loss, theta);
    }
    double quad = 0.0, lin = 0.0, l1 = 0.0, sq = 0.0;
    for (int a = 0; a < k; ++a) {
        const double xa = static_cast<double>(x(a));
        double row = 0.0;
        for (int bb = 0; bb < k; ++bb)
            row += static_cast<double>(G_reg(a, bb)) * static_cast<double>(x(bb));
        quad += xa * row;
        sq += xa * xa;
        l1 += nonneg ? xa : std::abs(xa);
        if (b_reg != nullptr) lin += static_cast<double>(b_reg[a]) * xa;
    }
    total += 0.5 * quad + 0.5 * static_cast<double>(ridge_col) * sq - lin;
    if (L1 != Scalar(0)) total += static_cast<double>(L1) * l1;
    return total;
}

/**
 * @brief IRLS solve of one column (or row) from its list of training entries.
 *
 * Each IRLS pass builds the weighted normal equations from the training
 * entries only,
 *   G_w = sum_e w_e f_e f_e^T + G_reg + ridge_col I,
 *   b_w = sum_e w_e y_e f_e + b_reg,
 * with w_e the distribution weight evaluated at the observed value y_e, the
 * current prediction and the entry's dispersion, and solves them from the
 * current iterate (warm start).
 *
 * Step control, as in the non-CV IRLS solver: a step is accepted only if the
 * penalized objective F does not rise; otherwise it is halved back toward
 * the current iterate (at most IRLS_MAX_HALVINGS times) and, failing that,
 * the current iterate is kept. Not applied to the legacy MAE / Huber loss
 * types, whose weights do not derive from a deviance.
 *
 * @param entries       Training entries (idx indexes columns of F)
 * @param F             Fixed factor (k × m for the H update, k × n for W)
 * @param x             In: warm start. Out: solution
 * @param loss          Loss used for the IRLS weights and the merit
 * @param G_reg         k × k penalty matrix shared by all columns
 * @param b_reg         k-vector added to the RHS (target, graph), or nullptr
 * @param ridge_col     Ridge of this column (graph diagonal)
 * @param upper_bound   Box constraint (0 = none)
 * @param theta_const   Dispersion shared by all entries of this column/row
 * @param theta_by_idx  Per-entry dispersion indexed like F's columns, or
 *                      nullptr to use theta_const
 */
template<typename Scalar>
inline void irls_solve_entries_cv(
    const std::vector<CVEntry<Scalar>>& entries,
    const DenseMatrix<Scalar>& F,
    DenseVector<Scalar>& x,
    const LossConfig<Scalar>& loss,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>& G_reg,
    const Scalar* b_reg,
    Scalar ridge_col,
    Scalar L1,
    bool nonneg,
    Scalar upper_bound,
    Scalar theta_const,
    const Scalar* theta_by_idx,
    CVWorkspace<Scalar>& ws)
{
    const int k = static_cast<int>(F.rows());

    if (entries.empty() && b_reg == nullptr) {
        // No training data for this column: nothing identifies it
        x.setZero();
        return;
    }

    DenseMatrix<Scalar>& G_w = ws.G_local;
    DenseVector<Scalar>& b_w = ws.b;
    if (G_w.rows() != k || G_w.cols() != k) G_w.resize(k, k);
    if (b_w.size() != k) b_w.resize(k);

    // Feasible, finite starting point
    if (!x.allFinite()) x.setZero();
    if (nonneg) x = x.cwiseMax(Scalar(0));
    if (upper_bound > Scalar(0)) x = x.cwiseMin(upper_bound);

    const bool step_control =
        !(loss.type == LossType::MAE || loss.type == LossType::HUBER);
    double F_old = step_control
        ? irls_merit_entries_cv(entries, F, x, loss, G_reg, b_reg, ridge_col,
                                L1, nonneg, theta_const, theta_by_idx)
        : 0.0;

    for (int irls_iter = 0; irls_iter < config.irls_max_iter; ++irls_iter) {
        G_w.setZero();
        b_w.setZero();

        for (const auto& te : entries) {
            const auto f = F.col(te.idx);
            const Scalar predicted = f.dot(x);
            const Scalar residual = te.value - predicted;
            const Scalar theta = theta_by_idx ? theta_by_idx[te.idx] : theta_const;
            const Scalar w = primitives::detail::compute_irls_weight(
                residual, predicted, loss, te.value, theta);

            const Scalar wv = w * te.value;
            for (int a = 0; a < k; ++a) {
                const Scalar wfa = w * f(a);
                b_w(a) += f(a) * wv;
                for (int bb = a; bb < k; ++bb)
                    G_w(a, bb) += wfa * f(bb);
            }
        }
        for (int a = 0; a < k; ++a)
            for (int bb = a + 1; bb < k; ++bb)
                G_w(bb, a) = G_w(a, bb);

        G_w += G_reg;
        G_w.diagonal().array() += ridge_col + static_cast<Scalar>(1e-15);
        if (b_reg != nullptr) {
            for (int a = 0; a < k; ++a) b_w(a) += b_reg[a];
        }

        ws.x_old = x;
        cv_solve_col(G_w, b_w, x, L1, nonneg, upper_bound, config, ws.scratch);

        // Step control: never accept an increase of the penalized objective
        if (step_control) {
            double F_new = irls_merit_entries_cv(
                entries, F, x, loss, G_reg, b_reg, ridge_col,
                L1, nonneg, theta_const, theta_by_idx);
            if (std::isfinite(F_old)) {
                const double slack = 1e-6 * (std::abs(F_old) + 1e-12);
                if (!(F_new <= F_old + slack)) {
                    ws.x_try = x;
                    bool accepted = false;
                    Scalar t = static_cast<Scalar>(1);
                    for (int h = 0; h < primitives::detail::IRLS_MAX_HALVINGS; ++h) {
                        t *= static_cast<Scalar>(0.5);
                        x = ws.x_old + t * (ws.x_try - ws.x_old);
                        F_new = irls_merit_entries_cv(
                            entries, F, x, loss, G_reg, b_reg, ridge_col,
                            L1, nonneg, theta_const, theta_by_idx);
                        if (F_new <= F_old + slack) { accepted = true; break; }
                    }
                    if (!accepted) {
                        x = ws.x_old;
                        break;
                    }
                }
            }
            F_old = F_new;
        }

        Scalar max_change = 0;
        for (int a = 0; a < k; ++a) {
            Scalar rel = std::abs(x(a) - ws.x_old(a)) /
                         (std::abs(ws.x_old(a)) + static_cast<Scalar>(1e-12));
            if (rel > max_change) max_change = rel;
        }
        if (max_change < config.irls_tol) break;
    }
}

/**
 * @brief IRLS-wrapped per-column solve for the H update in CV.
 *
 * Training entries of column j are all rows that are not in `excluded_rows`
 * (held-out and user-masked rows, sorted) and, when mask_zeros is true, are
 * observed (stored for sparse input, non-zero for dense input).
 *
 * @param theta_col     Dispersion of column j (used when theta_by_row is null)
 * @param theta_by_row  Per-row dispersion (length m) or nullptr
 */
template<typename Scalar, typename MatrixType>
void irls_solve_col_cv(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    int j,
    DenseVector<Scalar>& x,
    const std::vector<int>& excluded_rows,
    bool mask_zeros,
    const LossConfig<Scalar>& loss,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>& G_reg,
    const Scalar* b_reg,
    Scalar ridge_col,
    Scalar theta_col,
    const Scalar* theta_by_row,
    int m,
    CVWorkspace<Scalar>& ws)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;

    std::vector<CVEntry<Scalar>>& train_entries = ws.entries;
    train_entries.clear();

    size_t ex = 0;
    const size_t n_ex = excluded_rows.size();
    auto is_excluded = [&](int row) -> bool {
        while (ex < n_ex && excluded_rows[ex] < row) ++ex;
        return ex < n_ex && excluded_rows[ex] == row;
    };

    if (mask_zeros) {
        // Zeros are missing: only observed entries can be training entries
        DA::for_col(A, j, [&](int row, Scalar val) {
            if (!DA::is_sparse && val == Scalar(0)) return;
            if (!is_excluded(row)) train_entries.push_back({row, val});
        });
    } else {
        // ALL m entries (including zeros) can be training entries
        if constexpr (DA::is_sparse) {
            typename MatrixType::InnerIterator it(A, j);
            for (int row = 0; row < m; ++row) {
                Scalar val = 0;
                if (it && it.row() == row) { val = it.value(); ++it; }
                if (!is_excluded(row)) train_entries.push_back({row, val});
            }
        } else {
            for (int row = 0; row < m; ++row) {
                if (!is_excluded(row)) train_entries.push_back({row, A(row, j)});
            }
        }
    }

    irls_solve_entries_cv(train_entries, W_T, x, loss, config, G_reg, b_reg,
                          ridge_col, config.H.L1, config.H.nonneg,
                          config.H.upper_bound,
                          theta_col, theta_by_row, ws);
}

/**
 * @brief IRLS-wrapped per-row solve for the W update in CV.
 *
 * For sparse: pass A^T (n×m CSC); DA::for_col(A_T, i) iterates row i of A.
 * For dense:  pass A (m×n);       DA::for_row(A, i) iterates row i of A.
 *
 * @param theta_row     Dispersion of row i (used when theta_by_col is null)
 * @param theta_by_col  Per-column dispersion (length n) or nullptr
 */
template<typename Scalar, typename MatrixType>
void irls_solve_row_cv(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    int i,
    DenseVector<Scalar>& x,
    const std::vector<int>& excluded_cols,
    bool mask_zeros,
    const LossConfig<Scalar>& loss,
    const ::FactorNet::NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>& G_reg,
    const Scalar* b_reg,
    Scalar ridge_col,
    Scalar theta_row,
    const Scalar* theta_by_col,
    int n,
    CVWorkspace<Scalar>& ws)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;

    std::vector<CVEntry<Scalar>>& train_entries = ws.entries;
    train_entries.clear();

    size_t ex = 0;
    const size_t n_ex = excluded_cols.size();
    auto is_excluded = [&](int col) -> bool {
        while (ex < n_ex && excluded_cols[ex] < col) ++ex;
        return ex < n_ex && excluded_cols[ex] == col;
    };

    if (mask_zeros) {
        auto visit = [&](int col, Scalar val) {
            if (!DA::is_sparse && val == Scalar(0)) return;
            if (!is_excluded(col)) train_entries.push_back({col, val});
        };
        if constexpr (DA::is_sparse) {
            DA::for_col(data, i, visit);
        } else {
            DA::for_row(data, i, visit);
        }
    } else {
        if constexpr (DA::is_sparse) {
            typename MatrixType::InnerIterator it(data, i);
            for (int col = 0; col < n; ++col) {
                Scalar val = 0;
                if (it && it.row() == col) { val = it.value(); ++it; }
                if (!is_excluded(col)) train_entries.push_back({col, val});
            }
        } else {
            DA::for_row(data, i, [&](int col, Scalar val) {
                if (!is_excluded(col)) train_entries.push_back({col, val});
            });
        }
    }

    irls_solve_entries_cv(train_entries, H, x, loss, config, G_reg, b_reg,
                          ridge_col, config.W.L1, config.W.nonneg,
                          config.W.upper_bound,
                          theta_row, theta_by_col, ws);
}

// ========================================================================
// Held-out index lists and train-only right-hand sides
// ========================================================================
//
// The holdout set of a column (or row) is produced by list_holdout_col()
// (list_holdout_row()). The right-hand side is then accumulated by
// rhs_excluding_col() (rhs_excluding_row()) from the entries that are not in
// that list. The lists can be recomputed from the hash on every call or be
// built once and cached (CVMaskCache); both routes run the same two
// functions, so they hold out identical entries and give identical sums.

/**
 * @brief Held-out rows of column j, ascending.
 *
 * mask_zeros = true: only observed entries (stored for sparse input,
 * non-zero for dense input) can be held out. Otherwise every row can.
 */
template<typename Scalar, typename MatrixType>
inline void list_holdout_col(
    const MatrixType& A,
    int j,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    bool mask_zeros,
    int m,
    std::vector<int>& out)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    out.clear();
    if (mask_zeros) {
        DA::for_col(A, j, [&](int i, Scalar val) {
            if (!DA::is_sparse && val == Scalar(0)) return;
            if (mask.is_holdout(i, j)) out.push_back(i);
        });
    } else {
        for (int i = 0; i < m; ++i) {
            if (mask.is_holdout(i, j)) out.push_back(i);
        }
    }
}

/**
 * @brief Held-out columns of row i, ascending.
 *
 * For sparse: pass A^T (n×m CSC); column i of A^T is row i of A.
 * For dense:  pass A (m×n).
 */
template<typename Scalar, typename MatrixType>
inline void list_holdout_row(
    const MatrixType& data,
    int i,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    bool mask_zeros,
    int n,
    std::vector<int>& out)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    out.clear();
    if (mask_zeros) {
        auto visit = [&](int col, Scalar val) {
            if (!DA::is_sparse && val == Scalar(0)) return;
            if (mask.is_holdout(i, col)) out.push_back(col);
        };
        if constexpr (DA::is_sparse) {
            DA::for_col(data, i, visit);
        } else {
            DA::for_row(data, i, visit);
        }
    } else {
        for (int col = 0; col < n; ++col) {
            if (mask.is_holdout(i, col)) out.push_back(col);
        }
    }
}

/**
 * @brief b = sum over the entries of column j that are not in `excluded_rows`
 *        of A(i, j) * W_T.col(i).
 *
 * `excluded_rows` must be ascending. Cost O(nnz_j + |excluded|) for sparse
 * input, O(m) for dense input.
 */
template<typename Scalar, typename MatrixType>
inline void rhs_excluding_col(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    int j,
    const std::vector<int>& excluded_rows,
    DenseVector<Scalar>& b)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    b.setZero();
    size_t ex = 0;
    const size_t n_ex = excluded_rows.size();
    DA::for_col(A, j, [&](int i, Scalar val) {
        if (val == Scalar(0)) return;
        while (ex < n_ex && excluded_rows[ex] < i) ++ex;
        if (ex < n_ex && excluded_rows[ex] == i) return;
        b.noalias() += val * W_T.col(i);
    });
}

/**
 * @brief b = sum over the entries of row i that are not in `excluded_cols`
 *        of A(i, col) * H.col(col).
 *
 * For sparse: pass A^T (n×m CSC). For dense: pass A (m×n).
 *
 * @param b_test  Optional: receives the same sum over the excluded entries,
 *                so that b + b_test is the right-hand side over all entries
 *                of the row (used for the Gram-trick train loss)
 */
template<typename Scalar, typename MatrixType>
inline void rhs_excluding_row(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    int i,
    const std::vector<int>& excluded_cols,
    DenseVector<Scalar>& b,
    DenseVector<Scalar>* b_test = nullptr)
{
    using DA = FactorNet::primitives::DataAccessor<MatrixType>;
    b.setZero();
    if (b_test != nullptr) b_test->setZero();
    size_t ex = 0;
    const size_t n_ex = excluded_cols.size();
    auto visit = [&](int col, Scalar val) {
        if (val == Scalar(0)) return;
        while (ex < n_ex && excluded_cols[ex] < col) ++ex;
        if (ex < n_ex && excluded_cols[ex] == col) {
            if (b_test != nullptr) b_test->noalias() += val * H.col(col);
        } else {
            b.noalias() += val * H.col(col);
        }
    };
    if constexpr (DA::is_sparse) {
        DA::for_col(data, i, visit);
    } else {
        DA::for_row(data, i, visit);
    }
}

/**
 * @brief Compute RHS for column j using only TRAIN entries (H update).
 *
 * Fills `test_rows` (ascending) with the held-out rows of the column, then
 * accumulates b from the remaining entries.
 */
template<typename Scalar, typename MatrixType>
inline void compute_train_rhs(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    int j,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    DenseVector<Scalar>& b,
    std::vector<int>& test_rows,
    bool mask_zeros,
    int m)
{
    list_holdout_col(A, j, mask, mask_zeros, m, test_rows);
    rhs_excluding_col(A, W_T, j, test_rows, b);
}

/**
 * @brief Compute RHS for row i using only TRAIN entries (W update).
 *
 * For sparse: pass A^T (n×m CSC); column i of A_T = row i of A.
 * For dense:  pass A (m×n).
 */
template<typename Scalar, typename MatrixType>
inline void compute_train_rhs_W(
    const MatrixType& data,
    const DenseMatrix<Scalar>& H,
    int i,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    DenseVector<Scalar>& b,
    std::vector<int>& test_cols,
    bool mask_zeros,
    int n,
    DenseVector<Scalar>* b_test = nullptr)
{
    list_holdout_row(data, i, mask, mask_zeros, n, test_cols);
    rhs_excluding_row(data, H, i, test_cols, b, b_test);
}

// ========================================================================
// Cached holdout lists
// ========================================================================

/**
 * @brief Held-out indices per column and per row, built once per fit.
 *
 * Removes the three m × n hash passes per iteration (H update, W update,
 * loss). Memory: 4 bytes per held-out entry in each orientation, i.e. about
 * 8 bytes × holdout_fraction × (number of entries that can be held out).
 *
 * The cache is built when that estimate is at most cv_mask_cache_limit_bytes()
 * (1 GB unless the environment variable RCPPML_CV_MASK_CACHE_MB says
 * otherwise; 0 disables the cache). Above the limit the fit recomputes the
 * lists from the hash on every pass, which needs no extra memory and holds
 * out exactly the same entries.
 */
struct CVMaskCache {
    bool active = false;
    std::vector<std::vector<int>> col;  ///< held-out rows of each column
    std::vector<std::vector<int>> row;  ///< held-out columns of each row
};

/// Memory limit for CVMaskCache in bytes (see CVMaskCache).
inline double cv_mask_cache_limit_bytes()
{
    double mb = 1024.0;
    const char* env = std::getenv("RCPPML_CV_MASK_CACHE_MB");
    if (env != nullptr && *env != '\0') {
        char* end = nullptr;
        const double v = std::strtod(env, &end);
        if (end != env && v >= 0.0) mb = v;
    }
    return mb * 1048576.0;
}

// ========================================================================
// User mask helpers for CV with user-supplied mask
// ========================================================================

/**
 * @brief O(log n) check: is entry (row, col) in the user mask?
 */
template<typename Scalar>
inline bool is_user_masked_h(int row, int col,
                             const SparseMatrix<Scalar>& mask) {
    const int start = mask.outerIndexPtr()[col];
    const int end   = mask.outerIndexPtr()[col + 1];
    return std::binary_search(mask.innerIndexPtr() + start,
                              mask.innerIndexPtr() + end, row);
}

/**
 * @brief Adjust b and build excluded_rows after compute_train_rhs for H-update
 *
 * After compute_train_rhs fills b and test_rows, this function:
 * 1. Subtracts user-masked entries' contributions from b (they were included as train)
 * 2. Builds excluded_rows = test_rows ∪ masked_rows (sorted)
 *
 * @param A           The same data matrix that compute_train_rhs was given
 * @param W_T         Factor W in transposed storage (k × m)
 * @param user_mask   User-supplied sparse mask (nonzero = masked)
 * @param j           Column index
 * @param b           RHS vector to adjust (in/out)
 * @param test_rows   Holdout rows (from compute_train_rhs)
 * @param excluded_rows Output: holdout ∪ masked rows
 */
template<typename Scalar, typename MatrixType>
inline void adjust_rhs_for_user_mask_h(
    const MatrixType& A,
    const DenseMatrix<Scalar>& W_T,
    const SparseMatrix<Scalar>& user_mask,
    int j,
    DenseVector<Scalar>& b,
    const std::vector<int>& test_rows,
    std::vector<int>& excluded_rows)
{
    excluded_rows = test_rows;
    const int* inner = user_mask.innerIndexPtr();
    const int start = user_mask.outerIndexPtr()[j];
    const int end   = user_mask.outerIndexPtr()[j + 1];

    for (int p = start; p < end; ++p) {
        int row = inner[p];
        // If already in test_rows, it was already excluded from b
        if (std::binary_search(test_rows.begin(), test_rows.end(), row))
            continue;

        // Subtract from b (was included as a train entry)
        Scalar a_val;
        if constexpr (is_sparse_v<MatrixType>) {
            a_val = A.coeff(row, j);
        } else {
            a_val = A(row, j);
        }
        if (a_val != Scalar(0))
            b.noalias() -= a_val * W_T.col(row);

        excluded_rows.push_back(row);
    }
    std::sort(excluded_rows.begin(), excluded_rows.end());
}

/**
 * @brief Adjust b and build excluded_cols for W-update with user mask
 *
 * Same as adjust_rhs_for_user_mask_h but for the W-update which uses
 * A^T (or rows of A), H instead of W_T, and mask^T instead of mask.
 */
template<typename Scalar, typename MatrixType>
inline void adjust_rhs_for_user_mask_w(
    const MatrixType& A_or_At,
    const DenseMatrix<Scalar>& H,
    const SparseMatrix<Scalar>& mask_T,
    int i,
    DenseVector<Scalar>& b,
    const std::vector<int>& test_cols,
    std::vector<int>& excluded_cols)
{
    excluded_cols = test_cols;
    const int* inner = mask_T.innerIndexPtr();
    const int start = mask_T.outerIndexPtr()[i];
    const int end   = mask_T.outerIndexPtr()[i + 1];

    for (int p = start; p < end; ++p) {
        int col = inner[p];
        if (std::binary_search(test_cols.begin(), test_cols.end(), col))
            continue;

        // For sparse A^T: A_or_At.coeff(col, i) = A_T(col, i) = A(i, col)
        // For dense A:    A_or_At(i, col) = A(i, col)
        Scalar a_val;
        if constexpr (is_sparse_v<MatrixType>) {
            a_val = A_or_At.coeff(col, i);
        } else {
            a_val = A_or_At(i, col);
        }
        if (a_val != Scalar(0))
            b.noalias() -= a_val * H.col(col);

        excluded_cols.push_back(col);
    }
    std::sort(excluded_cols.begin(), excluded_cols.end());
}

}  // namespace detail
}  // namespace nmf
}  // namespace FactorNet
