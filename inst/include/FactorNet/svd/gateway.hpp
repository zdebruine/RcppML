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

// NMF redirect for constrained SVD (non-negativity, L1/L2, bounds, etc.)
// When constraints are active, the SVD gateway redirects to NMF fit, which
// properly solves the constrained matrix factorization problem.
#include <FactorNet/nmf/fit_cpu.hpp>
#include <FactorNet/core/resource_tags.hpp>

#include <string>
#include <cstdint>

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
// svd_config_to_nmf_config  —  build an NMFConfig from a constrained SVDConfig
// ============================================================================

/**
 * @brief Convert a constrained SVDConfig into an NMFConfig for the redirect path.
 *
 * When SVD is called with regularization/non-negativity constraints, we route
 * to NMF which properly handles the constrained factorization problem.
 * SVD convention: A \u2248 U \u00b7 diag(d) \u00b7 V^T  →  NMF convention: A \u2248 W \u00b7 diag(d) \u00b7 H
 * Mapping: W \u2248 U (m\u00d7k), H \u2248 V^T (k\u00d7n), d (k).
 * Factor correspondence: U \u2192 W,  V \u2192 H^T.
 */
template<typename Scalar>
inline FactorNet::NMFConfig<Scalar> svd_config_to_nmf_config(
    const SVDConfig<Scalar>& svd)
{
    FactorNet::NMFConfig<Scalar> nmf;
    nmf.rank          = svd.k_max;
    nmf.tol           = svd.tol;
    nmf.max_iter      = svd.max_iter > 0 ? svd.max_iter : 200;
    nmf.seed          = svd.seed;
    nmf.threads       = svd.threads;
    nmf.verbose       = svd.verbose;
    // Use SVD init (Lanczos) to seed NMF from the spectral structure
    nmf.init_mode     = 1;
    // Constraints: u \u2248 H (columns), v \u2248 W (rows)
    nmf.H.L1          = svd.L1_u;
    nmf.H.L2          = svd.L2_u;
    nmf.H.nonneg      = svd.nonneg_u;
    nmf.H.upper_bound = svd.upper_bound_u;
    nmf.H.L21         = svd.L21_u;
    nmf.H.angular     = svd.angular_u;
    nmf.W.L1          = svd.L1_v;
    nmf.W.L2          = svd.L2_v;
    nmf.W.nonneg      = svd.nonneg_v;
    nmf.W.upper_bound = svd.upper_bound_v;
    nmf.W.L21         = svd.L21_v;
    nmf.W.angular     = svd.angular_v;
    // Preserve graph regularization pointers (lifetime managed by caller)
    nmf.H.graph        = svd.graph_u;
    nmf.H.graph_lambda = svd.graph_u_lambda;
    nmf.W.graph        = svd.graph_v;
    nmf.W.graph_lambda = svd.graph_v_lambda;
    // No norm scaling — preserve SVD scaling structure (d from NMF L2 norm)
    nmf.norm_type = FactorNet::NormType::L2;
    return nmf;
}

// ============================================================================
// svd_gateway  —  in-memory dispatch (sparse *or* dense A)
// ============================================================================

/**
 * Dispatch an in-memory SVD/PCA run to the algorithm named by `method`.
 * Valid method strings: "deflation" (default), "randomized", "lanczos",
 * "irlba", "krylov".  Unknown strings fall through to deflation.
 *
 * When any regularization constraint is active (nonneg, L1, L2, bounds,
 * L21, angular, graph), the call is redirected to NMF which properly
 * handles the constrained factorization.  The NMF result is converted
 * back to SVDResult: U = W, d = d, V = H^T.
 */
template<typename MatrixType, typename Scalar>
inline SVDResult<Scalar> svd_gateway(
    const MatrixType& A,
    const SVDConfig<Scalar>& config,
    const std::string& method)
{
    // Robust SVD requires deflation (only method with IRLS support)
    if (config.is_robust() && method != "deflation") {
        // Route robust to deflation regardless of requested method
        return deflation_svd<MatrixType, Scalar>(A, config);
    }

    // Constrained SVD → NMF redirect
    if (config.has_regularization_u() || config.has_regularization_v()) {
        auto nmf_config = svd_config_to_nmf_config(config);
        auto nmf_result = FactorNet::nmf::nmf_fit<FactorNet::primitives::CPU>(
            A, nmf_config);

        // Convert NMFResult → SVDResult
        // NMF: A \u2248 W \u00b7 diag(d) \u00b7 H   (W: m\u00d7k, H: k\u00d7n, d: k)
        // SVD: A \u2248 U \u00b7 diag(d) \u00b7 V^T  (U: m\u00d7k, V: n\u00d7k, d: k)
        SVDResult<Scalar> svd_result;
        svd_result.U = nmf_result.W;
        svd_result.d = nmf_result.d;
        svd_result.V = nmf_result.H.transpose();  // k\u00d7n → n\u00d7k
        svd_result.k_selected = nmf_config.rank;
        return svd_result;
    }

    if (method == "randomized")
        return randomized_svd<MatrixType, Scalar>(A, config);
    if (method == "lanczos")
        return lanczos_svd<MatrixType, Scalar>(A, config);
    if (method == "irlba")
        return irlba_svd<MatrixType, Scalar>(A, config);
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
