// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file test_entries.hpp
 * @brief Holdout test entries for speckled cross-validation in SVD
 *
 * Shared infrastructure for test loss computation used by both
 * deflation and krylov SVD algorithms.
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/nmf/speckled_cv.hpp>
#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/svd/spmv.hpp>  // is_sparse_matrix

#include <vector>
#include <stdexcept>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace svd {

/**
 * @brief Store and evaluate test entries for speckled CV
 *
 * For each holdout entry (i,j), stores the residual A(i,j) - sum_k sigma_k*u_k(i)*v_k(j).
 * After each factor, update residuals and compute MSE.
 */
template<typename Scalar>
struct TestEntries {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<Scalar> residuals;
    int count = 0;

    Scalar compute_mse(int threads = 0) const {
        if (count == 0) return 0;
        Scalar sum_sq = 0;
#ifdef _OPENMP
        const int nthreads = threads > 0 ? threads : omp_get_max_threads();
        #pragma omp parallel for reduction(+:sum_sq) num_threads(nthreads) schedule(static)
#endif
        for (int i = 0; i < count; ++i) {
            sum_sq += residuals[i] * residuals[i];
        }
        return sum_sq / static_cast<Scalar>(count);
    }

    void update(const DenseVector<Scalar>& u, Scalar sigma,
                const DenseVector<Scalar>& v, int threads = 0) {
#ifdef _OPENMP
        const int nthreads = threads > 0 ? threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (int i = 0; i < count; ++i) {
            residuals[i] -= sigma * u(rows[i]) * v(cols[i]);
        }
    }
};

/**
 * @brief Initialize test entries from sparse matrix using LazySpeckledMask
 */
template<typename Scalar, typename SparseMat>
TestEntries<Scalar> init_test_entries_sparse(
    const SparseMat& A,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    const int n = static_cast<int>(A.cols());

    te.rows.reserve(static_cast<size_t>(A.nonZeros() * 0.15));
    te.cols.reserve(static_cast<size_t>(A.nonZeros() * 0.15));
    te.residuals.reserve(static_cast<size_t>(A.nonZeros() * 0.15));

    if (mask.mask_zeros()) {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                if (mask.is_holdout(it.row(), j)) {
                    Scalar val = static_cast<Scalar>(it.value());
                    if (row_means) val -= (*row_means)(it.row());
                    te.rows.push_back(it.row());
                    te.cols.push_back(j);
                    te.residuals.push_back(val);
                }
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                if (mask.is_holdout(it.row(), j)) {
                    Scalar val = static_cast<Scalar>(it.value());
                    if (row_means) val -= (*row_means)(it.row());
                    te.rows.push_back(it.row());
                    te.cols.push_back(j);
                    te.residuals.push_back(val);
                }
            }
        }
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

/**
 * @brief Initialize test entries from dense matrix using LazySpeckledMask
 */
template<typename Scalar>
TestEntries<Scalar> init_test_entries_dense(
    const DenseMatrix<Scalar>& A,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    te.rows.reserve(static_cast<size_t>(m) * n / 20);
    te.cols.reserve(static_cast<size_t>(m) * n / 20);
    te.residuals.reserve(static_cast<size_t>(m) * n / 20);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (mask.is_holdout(i, j)) {
                Scalar val = A(i, j);
                if (row_means) val -= (*row_means)(i);
                te.rows.push_back(i);
                te.cols.push_back(j);
                te.residuals.push_back(val);
            }
        }
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

/**
 * @brief Initialize test entries from streaming cached panels
 *
 * Iterates over all cached Eigen sparse panels in a StreamingContext
 * to collect holdout entries via LazySpeckledMask.
 * Only supports mask_zeros=true mode (sparse data — only nonzeros are candidates).
 *
 * @tparam Scalar  float or double
 * @tparam CtxType StreamingContext<Scalar> (templated to avoid circular include)
 */
template<typename Scalar, typename CtxType>
TestEntries<Scalar> init_test_entries_streaming(
    const CtxType& ctx,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    ctx.ensure_cache();

    // Estimate reservation: ~test_fraction of total nnz
    int64_t total_nnz = 0;
    for (uint32_t c = 0; c < ctx.num_chunks; ++c)
        total_nnz += ctx.cached_panels[c].nonZeros();
    te.rows.reserve(static_cast<size_t>(total_nnz * 0.15));
    te.cols.reserve(static_cast<size_t>(total_nnz * 0.15));
    te.residuals.reserve(static_cast<size_t>(total_nnz * 0.15));

    uint32_t col_offset = 0;
    for (uint32_t c = 0; c < ctx.num_chunks; ++c) {
        const auto& panel = ctx.cached_panels[c];
        uint32_t nc = ctx.chunks[c].num_cols;
        for (int lj = 0; lj < static_cast<int>(nc); ++lj) {
            int j = static_cast<int>(col_offset) + lj;  // global column
            for (typename std::remove_const<
                    typename std::remove_reference<decltype(panel)>::type
                 >::type::InnerIterator it(panel, lj); it; ++it)
            {
                int i = static_cast<int>(it.row());
                if (mask.is_holdout(i, j)) {
                    Scalar val = static_cast<Scalar>(it.value());
                    if (row_means) val -= (*row_means)(i);
                    te.rows.push_back(i);
                    te.cols.push_back(j);
                    te.residuals.push_back(val);
                }
            }
        }
        col_offset += nc;
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

/**
 * @brief Initialize test entries from raw CSC arrays
 *
 * Used when the matrix is available as raw CSC pointers (e.g., from GPU path)
 * without needing to construct an Eigen sparse matrix.
 * Only supports mask_zeros=true (only nonzeros are holdout candidates).
 */
template<typename Scalar>
TestEntries<Scalar> init_test_entries_csc(
    const int* col_ptr, const int* row_idx, const Scalar* values,
    int m, int n, int64_t nnz,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    const DenseVector<Scalar>* row_means = nullptr)
{
    TestEntries<Scalar> te;
    te.rows.reserve(static_cast<size_t>(nnz * 0.15));
    te.cols.reserve(static_cast<size_t>(nnz * 0.15));
    te.residuals.reserve(static_cast<size_t>(nnz * 0.15));

    for (int j = 0; j < n; ++j) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int i = row_idx[p];
            if (mask.is_holdout(i, j)) {
                Scalar val = values[p];
                if (row_means) val -= (*row_means)(i);
                te.rows.push_back(i);
                te.cols.push_back(j);
                te.residuals.push_back(val);
            }
        }
    }

    te.count = static_cast<int>(te.rows.size());
    return te;
}

}  // namespace svd
}  // namespace FactorNet

// ============================================================================
// Training matrix construction — zero out holdout entries
// ============================================================================
namespace FactorNet {
namespace svd {

/**
 * @brief Create a training copy of a sparse matrix with holdout entries zeroed
 *
 * The sparsity pattern is preserved (structural zeros remain), but their
 * values are set to 0 so they contribute nothing to SpMV during training.
 * This ensures the model never sees the held-out entries.
 *
 * The output uses the input matrix's own scalar type (e.g. double) even when
 * the working Scalar is float, because Eigen forbids mixed-type sparse
 * construction.  The SpMV/SpMM helpers use static_cast internally to handle
 * the double→float conversion at element level.
 *
 * @tparam MaskScalar  Scalar type used for the mask/config (float or double)
 * @param A          Original sparse matrix (CSC); may be Map<const ...>
 * @param mask       Speckled mask identifying holdout entries
 * @param n_zeroed   [out] Number of entries zeroed (for diagnostics)
 * @return Deep copy of A with holdout values set to 0 (same scalar as A)
 */
template<typename MaskScalar, typename InputMat>
Eigen::SparseMatrix<typename InputMat::Scalar, Eigen::ColMajor, typename InputMat::StorageIndex>
create_training_matrix_sparse(
    const InputMat& A,
    const FactorNet::nmf::LazySpeckledMask<MaskScalar>& mask,
    int& n_zeroed)
{
    // Always return an owned SparseMatrix matching A's scalar type
    using MatScalar = typename InputMat::Scalar;
    using OwnedSp = Eigen::SparseMatrix<MatScalar, Eigen::ColMajor,
                                         typename InputMat::StorageIndex>;
    OwnedSp A_train(A);  // deep copy from Map or owned
    n_zeroed = 0;

    const int n = static_cast<int>(A_train.cols());
    for (int j = 0; j < n; ++j) {
        for (typename OwnedSp::InnerIterator it(A_train, j); it; ++it) {
            if (mask.is_holdout(it.row(), j)) {
                it.valueRef() = 0;
                ++n_zeroed;
            }
        }
    }

    return A_train;
}

/**
 * @brief Create a training copy of a dense matrix with holdout entries zeroed
 */
template<typename Scalar>
DenseMatrix<Scalar> create_training_matrix_dense(
    const DenseMatrix<Scalar>& A,
    const FactorNet::nmf::LazySpeckledMask<Scalar>& mask,
    int& n_zeroed)
{
    DenseMatrix<Scalar> A_train(A);
    n_zeroed = 0;

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (mask.is_holdout(i, j)) {
                A_train(i, j) = 0;
                ++n_zeroed;
            }
        }
    }

    return A_train;
}

}  // namespace svd
}  // namespace FactorNet

// ============================================================================
// Missing entries (user mask and CV holdout) handled as MISSING, not as zeros
// ============================================================================
namespace FactorNet {
namespace svd {

/**
 * @brief Entries excluded from a fit: the user mask and/or the CV holdout.
 *
 * The fit works on the transformed matrix X = D^{-1}(A - mu 1') and replaces
 * every listed entry by its current model prediction (EM / iterative
 * imputation). The plain operators in spmv.hpp use the stored value
 * `base` = (raw - mu_i) / sd_i at those positions, so the imputed operator is
 *
 *   X_imputed = X + sum_e (fill_e - base_e) e_i e_j'
 *
 * and only needs an O(count) correction after each product. At a fixed point
 * of alternating least squares this is the least-squares fit over the
 * observed entries only, so singular values are not shrunk by the masking.
 *
 * Row means and SDs are computed from observed entries only.
 */
template<typename Scalar>
struct MaskedEntries {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<Scalar> raw;      ///< value of A at the entry (0 if structurally zero)
    std::vector<char> is_test;    ///< 1 = CV holdout (truth known), 0 = user mask
    std::vector<Scalar> base;     ///< transformed value used by the plain operator
    std::vector<Scalar> fill;     ///< current imputed value (transformed space)
    int count = 0;
    int n_test = 0;

    void push(int i, int j, Scalar value, bool test) {
        rows.push_back(i);
        cols.push_back(j);
        raw.push_back(value);
        is_test.push_back(test ? 1 : 0);
        base.push_back(static_cast<Scalar>(0));
        fill.push_back(static_cast<Scalar>(0));
        ++count;
        if (test) ++n_test;
    }

    void set_fill_zero() {
        for (int e = 0; e < count; ++e) fill[e] = static_cast<Scalar>(0);
    }

    /// out += (X_imputed - X) v
    void add_forward(const DenseVector<Scalar>& v, DenseVector<Scalar>& out) const {
        for (int e = 0; e < count; ++e)
            out(rows[e]) += (fill[e] - base[e]) * v(cols[e]);
    }

    /// out += (X_imputed - X)' u
    void add_transpose(const DenseVector<Scalar>& u, DenseVector<Scalar>& out) const {
        for (int e = 0; e < count; ++e)
            out(cols[e]) += (fill[e] - base[e]) * u(rows[e]);
    }

    /// Y (m x k) += (X_imputed - X) X_in (n x k)
    void add_forward_block(const DenseMatrix<Scalar>& X_in, DenseMatrix<Scalar>& Y) const {
        const int k = static_cast<int>(X_in.cols());
        for (int e = 0; e < count; ++e) {
            const Scalar c = fill[e] - base[e];
            const int i = rows[e], j = cols[e];
            for (int f = 0; f < k; ++f) Y(i, f) += c * X_in(j, f);
        }
    }

    /// Z (n x k) += (X_imputed - X)' X_in (m x k)
    void add_transpose_block(const DenseMatrix<Scalar>& X_in, DenseMatrix<Scalar>& Z) const {
        const int k = static_cast<int>(X_in.cols());
        for (int e = 0; e < count; ++e) {
            const Scalar c = fill[e] - base[e];
            const int i = rows[e], j = cols[e];
            for (int f = 0; f < k; ++f) Z(j, f) += c * X_in(i, f);
        }
    }

    /// Held-out entries as TestEntries: residual starts at the transformed truth
    TestEntries<Scalar> test_entries() const {
        TestEntries<Scalar> te;
        te.rows.reserve(static_cast<size_t>(n_test));
        te.cols.reserve(static_cast<size_t>(n_test));
        te.residuals.reserve(static_cast<size_t>(n_test));
        for (int e = 0; e < count; ++e) {
            if (is_test[e]) {
                te.rows.push_back(rows[e]);
                te.cols.push_back(cols[e]);
                te.residuals.push_back(base[e]);
            }
        }
        te.count = static_cast<int>(te.rows.size());
        return te;
    }
};

/**
 * @brief Collect the user mask (config.obs_mask, non-zero = missing) and the
 *        speckled CV holdout (config.test_fraction) for matrix A.
 *
 * The holdout set is defined identically for sparse and dense input: every
 * (i, j) selected by the position hash, restricted to non-zero entries when
 * config.mask_zeros is set. Entries in the user mask are never test entries.
 */
template<typename MatrixType, typename Scalar>
inline MaskedEntries<Scalar> build_masked_entries(const MatrixType& A,
                                                  const SVDConfig<Scalar>& config)
{
    MaskedEntries<Scalar> me;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const bool do_cv = config.is_cv();
    const bool mask_zeros = config.mask_zeros;
    const SparseMatrix<Scalar>* um = config.obs_mask;
    const bool have_user = (um != nullptr);

    if (have_user && (static_cast<int>(um->rows()) != m || static_cast<int>(um->cols()) != n))
        throw std::invalid_argument("svd: 'mask' dimensions must match the input matrix");

    int64_t nnz_est = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        nnz_est = static_cast<int64_t>(A.nonZeros());
    }

    if (do_cv) {
        const double candidates = mask_zeros
            ? static_cast<double>(nnz_est)
            : static_cast<double>(m) * static_cast<double>(n);
        if (candidates * static_cast<double>(config.test_fraction) > 4.0e8)
            throw std::runtime_error(
                "svd: the cross-validation holdout set is too large to enumerate; "
                "use mask = \"zeros\" to hold out non-zero entries only, or a smaller test_fraction");
    }

    FactorNet::nmf::LazySpeckledMask<Scalar> mask(m, n, nnz_est,
                          static_cast<double>(config.test_fraction),
                          config.effective_cv_seed(),
                          mask_zeros);

    std::vector<char> user_mark(static_cast<size_t>(m), 0);
    std::vector<Scalar> colbuf;
    if constexpr (is_sparse_matrix<MatrixType>::value) {
        colbuf.assign(static_cast<size_t>(m), static_cast<Scalar>(0));
    }

    for (int j = 0; j < n; ++j) {
        int n_user_col = 0;
        if (have_user) {
            for (typename SparseMatrix<Scalar>::InnerIterator mit(*um, j); mit; ++mit) {
                if (mit.value() != static_cast<Scalar>(0)) {
                    user_mark[static_cast<size_t>(mit.row())] = 1;
                    ++n_user_col;
                }
            }
        }

        bool scan_all_rows = (do_cv && !mask_zeros) || n_user_col > 0;
        if constexpr (is_sparse_matrix<MatrixType>::value) {
            if (scan_all_rows) {
                for (typename MatrixType::InnerIterator it(A, j); it; ++it)
                    colbuf[static_cast<size_t>(it.row())] = static_cast<Scalar>(it.value());
            }
        } else {
            scan_all_rows = scan_all_rows || do_cv;
        }

        if (scan_all_rows) {
            for (int i = 0; i < m; ++i) {
                Scalar val;
                if constexpr (is_sparse_matrix<MatrixType>::value) {
                    val = colbuf[static_cast<size_t>(i)];
                } else {
                    val = static_cast<Scalar>(A(i, j));
                }
                if (user_mark[static_cast<size_t>(i)]) {
                    me.push(i, j, val, false);
                    continue;
                }
                if (do_cv && mask.is_holdout(i, j)) {
                    if (mask_zeros && val == static_cast<Scalar>(0)) continue;
                    me.push(i, j, val, true);
                }
            }
            if constexpr (is_sparse_matrix<MatrixType>::value) {
                for (typename MatrixType::InnerIterator it(A, j); it; ++it)
                    colbuf[static_cast<size_t>(it.row())] = static_cast<Scalar>(0);
            }
        } else if (do_cv) {
            // Sparse input, non-zeros only, no user-masked rows in this column
            if constexpr (is_sparse_matrix<MatrixType>::value) {
                for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
                    const Scalar val = static_cast<Scalar>(it.value());
                    if (val != static_cast<Scalar>(0) &&
                        mask.is_holdout(static_cast<int>(it.row()), j)) {
                        me.push(static_cast<int>(it.row()), j, val, true);
                    }
                }
            }
        }

        if (n_user_col > 0) {
            for (typename SparseMatrix<Scalar>::InnerIterator mit(*um, j); mit; ++mit)
                user_mark[static_cast<size_t>(mit.row())] = 0;
        }
    }

    return me;
}

/**
 * @brief Row means / sample SDs over OBSERVED entries, the transformed values
 *        of the masked entries, and the observed sum of squares.
 *
 * @param[out] row_means   size m (zero when !center)
 * @param[out] row_sds     size m (one when !scale)
 * @param[out] total_ss    sum over observed entries of ((A_ij - mu_i)/sd_i)^2
 * @param[out] n_observed  number of observed entries
 */
template<typename MatrixType, typename Scalar>
inline void masked_row_stats(const MatrixType& A,
                             MaskedEntries<Scalar>& me,
                             bool center, bool scale,
                             DenseVector<Scalar>& row_means,
                             DenseVector<Scalar>& row_sds,
                             double& total_ss,
                             double& n_observed,
                             int threads = 1)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    Eigen::VectorXd sum, sum_sq;
    row_sums_sumsq(A, sum, sum_sq, threads);
    Eigen::VectorXd nobs = Eigen::VectorXd::Constant(m, static_cast<double>(n));
    for (int e = 0; e < me.count; ++e) {
        const int i = me.rows[e];
        const double v = static_cast<double>(me.raw[e]);
        sum(i) -= v;
        sum_sq(i) -= v * v;
        nobs(i) -= 1.0;
    }
    for (int i = 0; i < m; ++i) {
        if (sum_sq(i) < 0.0) sum_sq(i) = 0.0;
        if (nobs(i) <= 0.0) { sum(i) = 0.0; sum_sq(i) = 0.0; }
    }

    row_means = DenseVector<Scalar>::Zero(m);
    row_sds = DenseVector<Scalar>::Ones(m);
    if (scale) row_sds = row_sds_from_moments<Scalar>(sum, sum_sq, nobs);

    total_ss = 0.0;
    n_observed = 0.0;
    for (int i = 0; i < m; ++i) {
        const double no = nobs(i);
        if (no <= 0.0) continue;
        n_observed += no;
        const double mu = center ? sum(i) / no : 0.0;
        if (center) row_means(i) = static_cast<Scalar>(mu);
        double css = sum_sq(i) - (center ? no * mu * mu : 0.0);
        if (css < 0.0) css = 0.0;
        const double sd = static_cast<double>(row_sds(i));
        total_ss += css / (sd * sd);
    }

    for (int e = 0; e < me.count; ++e) {
        const int i = me.rows[e];
        me.base[e] = (me.raw[e] - row_means(i)) / row_sds(i);
        me.fill[e] = static_cast<Scalar>(0);
    }
}

}  // namespace svd
}  // namespace FactorNet

