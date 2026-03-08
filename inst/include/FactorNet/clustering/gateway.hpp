// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file clustering/bipartition.hpp
 * @brief Single entry point for clustering operations (bipartition, dclust)
 *
 * Provides gateway functions that handle:
 *   1. Resource detection (CPU cores, GPU availability)
 *   2. OpenMP thread management
 *   3. GPU dispatch with CPU fallback (future)
 *   4. Consistent interface for R layer
 *
 * Currently wraps bipartition() and dclust() with thread management.
 * GPU clustering dispatch can be added here when GPU primitives support
 * rank-2 NMF and centroid computation.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/clustering/bipartition.hpp>
#include <FactorNet/clustering/dclust.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace clustering {

/**
 * @brief RAII guard for OpenMP thread count
 *
 * Sets the thread count on construction and restores it on destruction.
 * No-op when OpenMP is not available or threads == 0 (use all).
 */
struct ThreadGuard {
#ifdef _OPENMP
    int prev_threads;
    bool active;

    explicit ThreadGuard(int threads) : prev_threads(omp_get_max_threads()), active(threads > 0) {
        if (active) omp_set_num_threads(threads);
    }

    ~ThreadGuard() {
        if (active) omp_set_num_threads(prev_threads);
    }
#else
    explicit ThreadGuard(int) {}
#endif

    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

/**
 * @brief Gateway for bipartition — spectral bipartitioning
 *
 * @tparam MatrixType  Sparse (dgCMatrix-mapped) or dense Eigen matrix
 * @param A            Input data matrix (features × samples)
 * @param tol          Convergence tolerance for rank-2 NMF
 * @param maxit        Maximum NMF iterations
 * @param nonneg       Enforce non-negativity
 * @param samples      Indices of columns to cluster
 * @param seed         RNG seed
 * @param verbose      Print progress
 * @param calc_dist    Compute inter-cluster distance
 * @param diag         Use diagonal scaling
 * @param threads      Number of OpenMP threads (0 = all available)
 * @return bipartitionModel with cluster assignments and diagnostics
 */
template<typename MatrixType>
auto bipartition_gateway(
    const MatrixType& A,
    double tol,
    unsigned int maxit,
    bool nonneg,
    const std::vector<unsigned int>& samples,
    unsigned int seed,
    bool verbose,
    bool calc_dist,
    bool diag,
    int threads = 0
) {
    ThreadGuard guard(threads);
    return clustering::bipartition(A, tol, maxit, nonneg, samples, seed, verbose, calc_dist, diag);
}

/**
 * @brief Gateway for dclust — divisive hierarchical clustering
 *
 * @tparam Scalar      Floating point type
 * @tparam MatrixType  Sparse or dense Eigen matrix
 * @param A            Input data matrix (features × samples)
 * @param min_samples  Minimum cluster size to attempt splitting
 * @param min_dist     Minimum cosine distance for a valid split
 * @param verbose      Print progress
 * @param tol          Convergence tolerance for rank-2 NMF
 * @param maxit        Maximum NMF iterations per split
 * @param nonneg       Enforce non-negativity
 * @param seed         RNG seed
 * @param threads      Number of OpenMP threads (0 = all available)
 * @return Vector of cluster descriptors
 */
template<typename Scalar, typename MatrixType>
auto dclust_gateway(
    const MatrixType& A,
    unsigned int min_samples,
    double min_dist,
    bool verbose,
    double tol,
    unsigned int maxit,
    bool nonneg,
    unsigned int seed,
    unsigned int threads = 0
) {
    ThreadGuard guard(static_cast<int>(threads));
    return clustering::dclust<Scalar>(A, min_samples, min_dist, verbose, tol, maxit, nonneg, seed, threads);
}

} // namespace clustering
}  // namespace FactorNet

