// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file gateway.hpp
 * @brief Unified SVD/PCA dispatch gateway.
 *
 * Single-include umbrella that routes calls to the right SVD algorithm based
 * on a method string.  Mirroring the NMF gateway pattern, this centralises:
 *  - all algorithm includes (replaces 6 per-algorithm includes at the Rcpp boundary)
 *  - in-memory sparse/dense dispatch  (svd_gateway)
 *  - streaming SPZ dispatch            (svd_streaming_gateway)
 *  - auto-method selection             (auto_select_svd_method, re-exported)
 *  - SVDConfig builder from raw scalars (build_svd_config_scalars)
 *
 * Usage in RcppFunctions_svd.cpp:
 * @code
 *   #include <FactorNet/svd/gateway.hpp>
 *   auto config = FactorNet::svd::build_svd_config_scalars<float>(...);
 *   result = FactorNet::svd::svd_gateway(A_f, config, method);
 *   result = FactorNet::svd::svd_streaming_gateway<float>(config, method);
 * @endcode
 */

#pragma once

// Algorithm headers (all algorithms available through this single include)
#include <FactorNet/svd/auto_select.hpp>
#include <FactorNet/svd/deflation.hpp>
#include <FactorNet/svd/randomized.hpp>
#include <FactorNet/svd/lanczos.hpp>
#include <FactorNet/svd/irlba.hpp>
#include <FactorNet/svd/krylov.hpp>
#include <FactorNet/svd/streaming_matvec.hpp>
#include <FactorNet/svd/streaming.hpp>
#include <FactorNet/core/svd_config.hpp>
#include <FactorNet/core/svd_result.hpp>



#include <string>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace FactorNet {
namespace svd {

// ============================================================================
// build_svd_config_scalars  —  build SVDConfig from plain-old-data parameters
// (graph Laplacians and obs_mask must be added by the caller afterwards because
//  their SparseMatrix values need caller-scope lifetime management)
// ============================================================================

template<typename Scalar>
inline SVDConfig<Scalar> build_svd_config_scalars(
    int k_max,
    Scalar tol,
    int max_iter,
    bool center,
    bool scale,
    bool verbose,
    uint32_t seed,
    int threads,
    SVDConvergence convergence,
    Scalar L1_u, Scalar L1_v,
    Scalar L2_u, Scalar L2_v,
    bool nonneg_u, bool nonneg_v,
    Scalar upper_bound_u, Scalar upper_bound_v,
    Scalar L21_u, Scalar L21_v,
    Scalar angular_u, Scalar angular_v,
    Scalar test_fraction,
    uint32_t cv_seed,
    int patience,
    bool mask_zeros,
    const std::string& spz_path = "",
    Scalar robust_delta = 0,
    int irls_max_iter = 5,
    Scalar irls_tol = static_cast<Scalar>(1e-4))
{
    SVDConfig<Scalar> config;
    config.k_max           = k_max;
    config.tol             = tol;
    config.max_iter        = max_iter;
    config.center          = center;
    config.scale           = scale;
    config.verbose         = verbose;
    config.seed            = seed;
    config.threads         = threads;
    config.convergence     = convergence;
    config.L1_u            = L1_u;
    config.L1_v            = L1_v;
    config.L2_u            = L2_u;
    config.L2_v            = L2_v;
    config.nonneg_u        = nonneg_u;
    config.nonneg_v        = nonneg_v;
    config.upper_bound_u   = upper_bound_u;
    config.upper_bound_v   = upper_bound_v;
    config.L21_u           = L21_u;
    config.L21_v           = L21_v;
    config.angular_u       = angular_u;
    config.angular_v       = angular_v;
    config.test_fraction   = test_fraction;
    config.cv_seed         = cv_seed;
    config.patience        = patience;
    config.mask_zeros      = mask_zeros;
    config.spz_path        = spz_path;
    config.robust_delta    = robust_delta;
    config.irls_max_iter   = irls_max_iter;
    config.irls_tol        = irls_tol;
    return config;
}

// ============================================================================
// parse_svd_convergence  —  shared helper to map string → SVDConvergence enum
// ============================================================================

inline SVDConvergence parse_svd_convergence(const std::string& s) noexcept {
    if (s == "loss") return SVDConvergence::LOSS;
    if (s == "both") return SVDConvergence::BOTH;
    return SVDConvergence::FACTOR;
}

// ============================================================================

namespace detail {

/// True for the three unconstrained Krylov/sketching solvers
inline bool is_plain_svd_method(const std::string& method) {
    return method == "lanczos" || method == "irlba" || method == "randomized";
}

/**
 * Options that a method cannot honor are rejected here instead of being
 * ignored. (The R layer performs the same checks with friendlier messages;
 * these guard direct C++ callers and the factor-network path.)
 */
template<typename Scalar>
inline void validate_svd_request(const SVDConfig<Scalar>& config,
                                 const std::string& method,
                                 bool streaming)
{
    if (config.upper_bound_u > 0 || config.upper_bound_v > 0) {
        throw std::invalid_argument(
            "svd: 'upper_bound' is not supported: factors are returned with unit "
            "norm, so an element-wise bound has no scale-free meaning");
    }

    const bool has_missing = config.is_cv() || (config.obs_mask != nullptr);
    const bool has_constraints =
        config.nonneg_u || config.nonneg_v ||
        config.L1_u > 0 || config.L1_v > 0 ||
        config.L2_u > 0 || config.L2_v > 0 ||
        config.L21_u > 0 || config.L21_v > 0 ||
        config.has_gram_level_reg();

    if (is_plain_svd_method(method)) {
        if (has_missing) {
            throw std::invalid_argument(
                "svd: method '" + method + "' cannot treat masked or held-out entries "
                "as missing; use method = \"deflation\" or \"krylov\"");
        }
        if (has_constraints) {
            throw std::invalid_argument(
                "svd: method '" + method + "' does not support constraints or "
                "penalties; use method = \"deflation\" or \"krylov\"");
        }
        if (config.convergence != SVDConvergence::FACTOR) {
            throw std::invalid_argument(
                "svd: convergence = \"loss\"/\"both\" is only implemented for "
                "method = \"deflation\" and \"krylov\"");
        }
    }
    if (config.is_robust() && method != "deflation") {
        throw std::invalid_argument(
            "svd: robust (Huber) fitting is only implemented for method = \"deflation\"");
    }
    (void)streaming;
}

}  // namespace detail

/**
 * Dispatch an in-memory SVD/PCA run to the algorithm named by `method`.
 * Valid method strings: "deflation" (default), "randomized", "lanczos",
 * "irlba", "krylov".  Unknown strings fall through to deflation.
 *
 * Constrained SVD (nonneg, L1, L2, L21, angular, graph), masks and
 * cross-validation are handled by "deflation" and "krylov". Requests that the
 * chosen method cannot honor throw std::invalid_argument.
 *
 * Robust SVD (Huber-weighted IRLS) is only supported by "deflation".
 */
template<typename MatrixType, typename Scalar>
inline SVDResult<Scalar> svd_gateway(
    const MatrixType& A,
    const SVDConfig<Scalar>& config,
    const std::string& method)
{
    detail::validate_svd_request(config, method, false);

    if (method == "randomized")
        return randomized_svd<MatrixType, Scalar>(A, config);
    if (method == "lanczos")
        return lanczos_svd<MatrixType, Scalar>(A, config, true);
    if (method == "irlba")
        return irlba_svd<MatrixType, Scalar>(A, config, true);
    if (method == "krylov")
        return krylov_constrained_svd<MatrixType, Scalar>(A, config);
    // default — deflation
    return deflation_svd<MatrixType, Scalar>(A, config);
}

// ============================================================================
// svd_streaming_gateway  —  out-of-core dispatch (reads config.spz_path)
// ============================================================================

/**
 * Dispatch a streaming (out-of-core SPZ) SVD run.
 * config.spz_path must be set before this call.
 * Valid method strings mirror svd_gateway; unknown → streaming deflation.
 */
template<typename Scalar>
inline SVDResult<Scalar> svd_streaming_gateway(
    const SVDConfig<Scalar>& config,
    const std::string& method)
{
    detail::validate_svd_request(config, method, true);

    // Missing entries (mask, CV holdout) and projection deflation for
    // constrained factors are implemented once, in the in-memory solvers.
    // The streaming context holds all decoded panels in memory anyway, so
    // assemble them and use that path instead of ignoring the request.
    const bool has_missing = config.is_cv() || (config.obs_mask != nullptr);
    const bool constrained_deflation =
        method != "randomized" && method != "lanczos" && method != "irlba" &&
        method != "krylov" &&
        (config.nonneg_u || config.nonneg_v || config.L1_u > 0 || config.L1_v > 0);
    if (has_missing || constrained_deflation) {
        detail::StreamingContext<Scalar> ctx;
        ctx.load(config.spz_path, false, false);
        ctx.ensure_cache();

        SVDConfig<Scalar> cfg = config;
        cfg.spz_path.clear();

        const int m = static_cast<int>(ctx.m);
        const int n = static_cast<int>(ctx.n);
        if (ctx.is_v3) {
            DenseMatrix<Scalar> A_dense(m, n);
            int col_offset = 0;
            for (uint32_t c = 0; c < ctx.num_chunks; ++c) {
                const int nc = static_cast<int>(ctx.chunk_num_cols[c]);
                A_dense.middleCols(col_offset, nc) = ctx.cached_dense_panels[c];
                col_offset += nc;
            }
            return svd_gateway<DenseMatrix<Scalar>, Scalar>(A_dense, cfg, method);
        } else {
            typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> SpMat;
            std::vector<Eigen::Triplet<Scalar>> trips;
            size_t total_nnz = 0;
            for (uint32_t c = 0; c < ctx.num_chunks; ++c)
                total_nnz += static_cast<size_t>(ctx.cached_panels[c].nonZeros());
            trips.reserve(total_nnz);
            int col_offset = 0;
            for (uint32_t c = 0; c < ctx.num_chunks; ++c) {
                const SpMat& panel = ctx.cached_panels[c];
                const int nc = static_cast<int>(ctx.chunk_num_cols[c]);
                for (int lj = 0; lj < nc; ++lj) {
                    for (typename SpMat::InnerIterator it(panel, lj); it; ++it) {
                        trips.push_back(Eigen::Triplet<Scalar>(
                            static_cast<int>(it.row()), col_offset + lj, it.value()));
                    }
                }
                col_offset += nc;
            }
            SpMat A_sparse(m, n);
            A_sparse.setFromTriplets(trips.begin(), trips.end());
            A_sparse.makeCompressed();
            return svd_gateway<SpMat, Scalar>(A_sparse, cfg, method);
        }
    }

    if (method == "randomized")
        return streaming_randomized_svd<Scalar>(config);
    if (method == "lanczos")
        return streaming_lanczos_svd<Scalar>(config);
    if (method == "irlba")
        return streaming_irlba_svd<Scalar>(config);
    if (method == "krylov")
        return streaming_krylov_svd<Scalar>(config);
    // default — streaming deflation
    return streaming_deflation_svd<Scalar>(config);
}

}  // namespace svd
}  // namespace FactorNet
