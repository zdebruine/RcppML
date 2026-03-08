// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file rng.hpp
 * @brief Unified random number generator for all RcppML backends
 *
 * Provides a single RNG class that handles:
 * - Sequential operations (W/H initialization)
 * - Position-dependent operations (CV speckled holdout masks)
 * - CPU and GPU execution (__host__ __device__)
 *
 * Replaces three prior RNG classes:
 *   - UnifiedRNG / NMF_RNG   (xorshift64)  — CPU sequential + position-dependent
 *   - CV_RNG / SpeckledRNG   (SplitMix64)  — CPU cross-validation masks
 *   - DeviceCVRNG            (SplitMix64)  — GPU cross-validation masks
 *
 * Reproducibility guarantees:
 *   - Same seed → identical initialization matrices and CV masks on all backends
 *   - Same seed + same backend + same thread count → bitwise reproducible
 *   - Cross-backend (CPU vs GPU): STATISTICALLY EQUIVALENT, not bitwise identical
 *     (floating-point non-associativity in GEMM, reductions causes numeric drift)
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <cstdint>
#include <limits>

// Host/device annotation for CUDA portability
#ifdef __CUDACC__
#define FACTORNET_HOST_DEVICE __host__ __device__
#else
#define FACTORNET_HOST_DEVICE
#endif

namespace FactorNet {
namespace rng {

/**
 * @brief SplitMix64 RNG — unified engine for all backends
 *
 * Supports two modes:
 *  1. **Sequential** — `next()`, `uniform()`, `fill_uniform()` advance internal
 *     state and produce a stream of pseudo-random values.
 *  2. **Position-dependent** — `hash()`, `is_holdout()` are PURE functions of
 *     (seed, i, j) and NEVER mutate state. This makes them trivially thread-safe
 *     and GPU-safe.
 *
 * Thread safety:
 *   - Sequential ops are NOT thread-safe (each thread needs its own instance).
 *   - Position-dependent ops ARE thread-safe (pure functions).
 */
class SplitMix64 {
private:
    uint64_t state_;

public:
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /**
     * @brief Construct with seed
     * @param seed  Random seed. 0 is remapped to 12345 to avoid degenerate state.
     */
    FACTORNET_HOST_DEVICE explicit SplitMix64(uint64_t seed = 12345) noexcept
        : state_(seed == 0 ? 12345ULL : seed) {}

    /// Get current state (for debugging / testing)
    FACTORNET_HOST_DEVICE uint64_t state() const noexcept { return state_; }

    /// Set state directly (for testing)
    FACTORNET_HOST_DEVICE void set_state(uint64_t s) noexcept { state_ = s; }

    // -------------------------------------------------------------------
    // Sequential operations — advance state on each call
    // -------------------------------------------------------------------

    /**
     * @brief Generate next 64-bit value (advances state)
     */
    FACTORNET_HOST_DEVICE uint64_t next() noexcept {
        state_ += 0x9e3779b97f4a7c15ULL;          // golden-ratio increment
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    /**
     * @brief Uniform random in [0, 1)
     * @tparam T  Return type (float or double)
     */
    template<typename T = double>
    FACTORNET_HOST_DEVICE T uniform() noexcept {
        return static_cast<T>(next()) / static_cast<T>(UINT64_MAX);
    }

    /**
     * @brief Random integer in [0, max)
     */
    FACTORNET_HOST_DEVICE uint64_t rand_int(uint64_t max) noexcept {
        return next() % max;
    }

    // -------------------------------------------------------------------
    // Position-dependent operations — pure, no state mutation
    // Thread-safe and GPU-safe.
    // -------------------------------------------------------------------

    /**
     * @brief Position-dependent hash (i, j) → deterministic 64-bit value
     *
     * Uses SplitMix64-style mixing with golden-ratio primes.
     * Excellent avalanche properties and statistical quality.
     *
     * @param seed  Base seed
     * @param i     Row index
     * @param j     Column index
     * @return      64-bit hash
     */
    FACTORNET_HOST_DEVICE static uint64_t hash(uint64_t seed,
                                             uint32_t i,
                                             uint32_t j) noexcept {
        uint64_t h = seed
                   + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL
                   + static_cast<uint64_t>(j) * 0x6c62272e07bb0142ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        return h ^ (h >> 31);
    }

    /**
     * @brief Position-dependent uniform in [0, 1)
     *
     * @tparam T  Return type (float or double)
     */
    template<typename T = double>
    FACTORNET_HOST_DEVICE static T uniform(uint64_t seed,
                                         uint32_t i,
                                         uint32_t j) noexcept {
        return static_cast<T>(hash(seed, i, j)) / static_cast<T>(UINT64_MAX);
    }

    /**
     * @brief Deterministic holdout test: is entry (i, j) in the test set?
     *
     * Returns true with probability ~1/inv_prob.
     * Uses threshold comparison (faster than modulo).
     *
     * @param seed      Base seed
     * @param i         Row index
     * @param j         Column index
     * @param inv_prob  Inverse probability (e.g. 10 → 10% holdout)
     * @return          true if (i,j) is held out
     */
    FACTORNET_HOST_DEVICE static bool is_holdout(uint64_t seed,
                                               uint32_t i,
                                               uint32_t j,
                                               uint64_t inv_prob) noexcept {
        if (inv_prob == 0) return false;
        return hash(seed, i, j) < (UINT64_MAX / inv_prob);
    }

    /**
     * @brief Instance-method overload using internal state as seed.
     *
     * Drop-in replacement for DeviceCVRNG::is_holdout().
     * Does NOT advance state (position-dependent, pure function).
     */
    FACTORNET_HOST_DEVICE bool is_holdout(int row, int col,
                                        uint64_t inv_prob) const noexcept {
        return is_holdout(state_, static_cast<uint32_t>(row),
                          static_cast<uint32_t>(col), inv_prob);
    }

    // -------------------------------------------------------------------
    // Matrix initialization helpers
    // -------------------------------------------------------------------

    /**
     * @brief Fill a raw column-major buffer with uniform random values in [0, 1)
     *
     * Column-major iteration for Eigen compatibility.
     * This ADVANCES state (sequential mode).
     */
    template<typename Scalar>
    void fill_uniform(Scalar* data, int rows, int cols) noexcept {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                data[j * rows + i] = uniform<Scalar>();
            }
        }
    }

#ifndef __CUDACC__
    /**
     * @brief Fill an Eigen matrix with uniform random values in [0, 1)
     *
     * Only available on host (Eigen not available on device).
     *
     * @tparam Scalar  Matrix element type
     * @param M        Matrix to fill (resized if needed)
     */
    template<typename Derived>
    void fill_uniform(Eigen::DenseBase<Derived>& M) {
        for (int j = 0; j < M.cols(); ++j) {
            for (int i = 0; i < M.rows(); ++i) {
                M(i, j) = uniform<typename Derived::Scalar>();
            }
        }
    }
#endif
};

/// Convenience alias
using Engine = SplitMix64;

}  // namespace rng
}  // namespace FactorNet

