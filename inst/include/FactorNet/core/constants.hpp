// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file constants.hpp
 * @brief Numerical constants and algorithm parameters
 * 
 * Defines all compile-time constants used throughout RcppML including:
 * - Numerical stability epsilons
 * - Algorithm convergence tolerances
 * - Maximum iteration limits
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

#ifndef FactorNet_CORE_CONSTANTS_HPP
#define FactorNet_CORE_CONSTANTS_HPP

#include <limits>
#include <type_traits>

namespace FactorNet {

// ============================================================================
// Numerical Stability Constants
// ============================================================================

/**
 * @brief Machine epsilon for numerical stability
 * 
 * Used to prevent division by zero, ensure positive definiteness in Gram matrices,
 * and floor IRLS weights. Suitable for double precision computations.
 * 
 * @tparam Scalar Precision type
 * @return Epsilon value (1e-15)
 */
template<typename Scalar>
constexpr Scalar tiny_num() noexcept {
    return static_cast<Scalar>(1e-15);
}

/**
 * @brief Floor value for KL divergence to avoid log(0)
 * 
 * @tparam Scalar Precision type
 * @return KL epsilon value (1e-10)
 */
template<typename Scalar>
constexpr Scalar kl_epsilon() noexcept {
    return static_cast<Scalar>(1e-10);
}

// ============================================================================
// Coordinate Descent Parameters
// ============================================================================

// Default convergence tolerance (average relative change per coordinate)
// Matches singlet's criterion: sum |diff / (x[i] + 1e-15)| / k < 1e-8.
// This is a relative criterion that naturally adapts to solution scale.
constexpr double CD_TOL = 1e-8;

// Maximum inner CD iterations per column (safety cap)
// Matches singlet's default of 100.  With relative convergence, most columns
// converge in 3-10 sweeps with warm starts; the cap of 100 ensures quality
// on hard problems.
constexpr unsigned int CD_MAXIT = 100;

// Absolute tolerance floor for CD convergence denominator
// The CD convergence check computes |diff / (x[i] + cd_abs_tol)|.
// With the default 1e-15, tiny x values produce huge relative changes.
// Setting to ~1.0 gives a "mixed" tolerance: absolute for small x, relative for large x.
constexpr double CD_ABS_TOL = 1e-15;

// ============================================================================
// NMF Algorithm Parameters
// ============================================================================

// Default convergence tolerance (relative change in loss)
constexpr double NMF_TOL = 1e-4;

// Default maximum iterations
constexpr unsigned int NMF_MAXIT = 100;

// Early stopping patience (iterations without CV improvement)
constexpr unsigned int NMF_PATIENCE = 5;

// ============================================================================
// Regularization Defaults
// ============================================================================

// Default regularization strengths (0 = no regularization)
constexpr double DEFAULT_L1 = 0.0;
constexpr double DEFAULT_L2 = 0.0;
constexpr double DEFAULT_L21 = 0.0;
constexpr double DEFAULT_GRAPH_LAMBDA = 0.0;

// Default Huber loss transition (quadratic → linear at |residual| = delta)
constexpr double DEFAULT_HUBER_DELTA = 1.0;

// ============================================================================
// Parallelization Defaults
// ============================================================================

// Default number of threads (0 = use all available)
constexpr unsigned int DEFAULT_THREADS = 0;

// Minimum problem size for parallelization (smaller = overhead exceeds benefit)
constexpr unsigned int MIN_SIZE_FOR_THREADING = 100;

}  // namespace FactorNet

#endif // FactorNet_CORE_CONSTANTS_HPP
