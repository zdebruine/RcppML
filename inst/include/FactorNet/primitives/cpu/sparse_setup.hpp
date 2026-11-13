// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file cpu/sparse_setup.hpp
 * @brief Parallel scalar conversion and transpose of a CSC sparse matrix
 *
 * The two fixed setup costs of a sparse fit are the single-precision copy of
 * the input and the CSC transpose used by the W-update.  Eigen does both
 * serially.  The routines here produce the same result bit for bit (same
 * outer index, same sorted inner indices, same values) with OpenMP:
 *
 *   sparse_cast       copies the structure and converts the values with a
 *                     parallel loop (which also first-touches the memory on
 *                     the threads that will read it).
 *
 *   sparse_transpose  counts entries per row with per-thread histograms,
 *                     takes a prefix sum, splits the rows into contiguous
 *                     ranges of about equal nnz, and lets each range be
 *                     filled independently: a range scans every column,
 *                     binary-searches its first row and stops past its last.
 *                     Ranges write disjoint parts of the output, so there
 *                     are no atomics and the output order does not depend on
 *                     the number of threads.
 *
 * Both fall back to Eigen's serial code when the matrix is small (thread
 * start-up costs more than it saves), when one thread is requested, when the
 * build has no OpenMP, when the input is not in compressed form, or (for the
 * transpose) when the inner indices of a column are not strictly increasing.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace primitives {

/// Below this many non-zeros the serial Eigen path is used.
constexpr long long SPARSE_SETUP_MIN_PARALLEL_NNZ = 1000000LL;

namespace detail {

/// Number of threads to use for a setup pass (1 = serial path).
inline int sparse_setup_threads(int threads, long long nnz, long long min_parallel_nnz)
{
#ifdef _OPENMP
    int t = (threads > 0) ? threads : omp_get_max_threads();
    if (t < 1) t = 1;
    if (nnz < min_parallel_nnz) t = 1;
    return t;
#else
    (void)threads;
    (void)nnz;
    (void)min_parallel_nnz;
    return 1;
#endif
}

/// Eigen's storage index is int: a matrix with more non-zeros cannot be held.
inline void check_nnz_fits_int(long long nnz)
{
    if (nnz > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Sparse matrix has more than INT_MAX non-zeros, which the "
            "32-bit sparse index used by RcppML cannot address.");
    }
}

}  // namespace detail

/**
 * @brief Copy a compressed sparse matrix, converting the scalar type
 *
 * Equal to `A.cast<OutScalar>()` entry for entry.
 *
 * @tparam OutScalar  Scalar type of the result
 * @tparam SparseIn   Eigen::SparseMatrix or Eigen::Map of one (column-major)
 * @param threads     OpenMP threads (0 = all available)
 * @param min_parallel_nnz  Serial path below this many non-zeros
 */
template<typename OutScalar, typename SparseIn>
inline SparseMatrix<OutScalar> sparse_cast(
    const SparseIn& A,
    int threads,
    long long min_parallel_nnz = SPARSE_SETUP_MIN_PARALLEL_NNZ)
{
    const long long nnz_ll = static_cast<long long>(A.nonZeros());
    detail::check_nnz_fits_int(nnz_ll);
    const int n_threads = detail::sparse_setup_threads(threads, nnz_ll, min_parallel_nnz);

    if (n_threads <= 1 || !A.isCompressed()) {
        SparseMatrix<OutScalar> out = A.template cast<OutScalar>();
        return out;
    }

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int nnz = static_cast<int>(nnz_ll);
    const auto* p = A.outerIndexPtr();
    const auto* i = A.innerIndexPtr();
    const auto* x = A.valuePtr();

    SparseMatrix<OutScalar> out(m, n);
    out.resizeNonZeros(nnz);
    int* op = out.outerIndexPtr();
    int* oi = out.innerIndexPtr();
    OutScalar* ox = out.valuePtr();

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int j = 0; j <= n; ++j) {
        op[j] = static_cast<int>(p[j]);
    }
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int q = 0; q < nnz; ++q) {
        oi[q] = static_cast<int>(i[q]);
        ox[q] = static_cast<OutScalar>(x[q]);
    }
    return out;
}

/**
 * @brief Transpose a compressed sparse matrix (optionally converting scalars)
 *
 * Equal to `SparseMatrix<OutScalar>(A.cast<OutScalar>().transpose())` in
 * compressed form, entry for entry.
 *
 * @tparam OutScalar  Scalar type of the result
 * @tparam SparseIn   Eigen::SparseMatrix or Eigen::Map of one (column-major)
 * @param threads     OpenMP threads (0 = all available)
 * @param min_parallel_nnz  Serial path below this many non-zeros
 */
template<typename OutScalar, typename SparseIn>
inline SparseMatrix<OutScalar> sparse_transpose(
    const SparseIn& A,
    int threads,
    long long min_parallel_nnz = SPARSE_SETUP_MIN_PARALLEL_NNZ)
{
    const long long nnz_ll = static_cast<long long>(A.nonZeros());
    detail::check_nnz_fits_int(nnz_ll);
    const int n_threads = detail::sparse_setup_threads(threads, nnz_ll, min_parallel_nnz);

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int nnz = static_cast<int>(nnz_ll);

    bool use_parallel = (n_threads > 1) && A.isCompressed() && m > 0 && n > 0;

    // Per-row counts (shifted by one for the prefix sum), filled below.
    std::vector<int> cnt;

    if (use_parallel) {
        const auto* p = A.outerIndexPtr();
        const auto* i = A.innerIndexPtr();

        // 1. Row counts with per-thread histograms.  The same pass checks
        //    that every column has strictly increasing row indices, which
        //    step 3 relies on.
        cnt.assign(static_cast<std::size_t>(m) + 1, 0);
        bool sorted = true;
        #pragma omp parallel num_threads(n_threads)
        {
            std::vector<int> local(static_cast<std::size_t>(m), 0);
            bool local_sorted = true;
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < n; ++j) {
                const int begin = static_cast<int>(p[j]);
                const int end = static_cast<int>(p[j + 1]);
                for (int q = begin; q < end; ++q) {
                    const int r = static_cast<int>(i[q]);
                    if (r < 0 || r >= m) {
                        local_sorted = false;
                        continue;
                    }
                    ++local[static_cast<std::size_t>(r)];
                    if (q > begin && r <= static_cast<int>(i[q - 1]))
                        local_sorted = false;
                }
            }
            #pragma omp critical(factornet_sparse_setup_hist)
            {
                for (int r = 0; r < m; ++r)
                    cnt[static_cast<std::size_t>(r) + 1] += local[static_cast<std::size_t>(r)];
                if (!local_sorted) sorted = false;
            }
        }
        if (!sorted) use_parallel = false;
    }

    if (!use_parallel) {
        // Serial Eigen path
        SparseMatrix<OutScalar> out;
        SparseMatrix<OutScalar> tmp = A.template cast<OutScalar>();
        out = tmp.transpose();
        out.makeCompressed();
        return out;
    }

    const auto* p = A.outerIndexPtr();
    const auto* i = A.innerIndexPtr();
    const auto* x = A.valuePtr();

    SparseMatrix<OutScalar> out(n, m);
    out.resizeNonZeros(nnz);
    int* op = out.outerIndexPtr();   // length m + 1
    int* oi = out.innerIndexPtr();
    OutScalar* ox = out.valuePtr();

    // 2. Prefix sum: start of each output column (= row of A).
    op[0] = 0;
    for (int r = 0; r < m; ++r)
        op[r + 1] = op[r] + cnt[static_cast<std::size_t>(r) + 1];

    // 3. Split the rows into n_threads contiguous ranges of about equal nnz.
    std::vector<int> bound(static_cast<std::size_t>(n_threads) + 1, m);
    bound[0] = 0;
    for (int t = 1; t < n_threads; ++t) {
        const long long target = nnz_ll * t / n_threads;
        int b = static_cast<int>(
            std::lower_bound(op, op + m + 1, static_cast<int>(target)) - op);
        if (b > m) b = m;
        if (b < bound[static_cast<std::size_t>(t) - 1])
            b = bound[static_cast<std::size_t>(t) - 1];
        bound[static_cast<std::size_t>(t)] = b;
    }

    // 4. Fill.  The loop runs over RANGES, not over threads, so every range
    //    is filled whatever team size the runtime actually grants.
    #pragma omp parallel for schedule(static, 1) num_threads(n_threads)
    for (int t = 0; t < n_threads; ++t) {
        const int lo = bound[static_cast<std::size_t>(t)];
        const int hi = bound[static_cast<std::size_t>(t) + 1];
        if (hi <= lo) continue;
        // Next free slot of each row in [lo, hi)
        std::vector<int> cur(op + lo, op + hi);
        for (int j = 0; j < n; ++j) {
            const auto* col_begin = i + p[j];
            const auto* col_end = i + p[j + 1];
            // rows within a column are sorted: first row >= lo
            const auto* first = std::lower_bound(col_begin, col_end, lo);
            const int q_end = static_cast<int>(p[j + 1]);
            for (int q = static_cast<int>(first - i); q < q_end; ++q) {
                const int r = static_cast<int>(i[q]);
                if (r >= hi) break;
                const int dst = cur[static_cast<std::size_t>(r - lo)]++;
                oi[dst] = j;
                ox[dst] = static_cast<OutScalar>(x[q]);
            }
        }
    }
    return out;
}

}  // namespace primitives
}  // namespace FactorNet
