// Automatic rank determination via cross-validation
//
// Strategy (all evaluations are cached, the same holdout mask is used for
// every rank):
//   1. Coarse geometric grid k_min, 2 k_min, 4 k_min, ..., k_max. Expansion
//      stops once the test loss has risen at two consecutive grid points, or
//      when the range ends.
//   2. Refinement between the grid neighbours of the best grid point by
//      repeated bisection of the two sub-intervals around the current best
//      rank, until the best rank's integer neighbours have been evaluated
//      (or the bracket around it is no wider than `tolerance`).
//   3. The answer is the rank with the lowest test loss among ALL ranks that
//      were evaluated (ties go to the smaller rank). Any integer in
//      [k_min, k_max] can be returned.
//
// Routes through nmf::nmf() for unified CV infrastructure:
// - Uses the CV fit with per-column Gram correction
// - Supports GPU fallback via gateway resource detection
// - Uses core::NMFConfig and NMFResult types
//
// Copyright (C) 2021-2026 Zach DeBruine

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/nmf/fit.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <FactorNet/core/logging.hpp>

namespace FactorNet {
namespace nmf {

/**
 * @brief Cross-validation evaluation for a single rank
 *
 * Contains train/test losses at each iteration and final values.
 * Losses are extracted from NMFResult (computed by nmf::fit).
 */
template<typename Scalar>
struct CVEvaluation {
    int rank;
    Scalar train_loss_final;
    Scalar test_loss_final;
    std::vector<Scalar> train_loss_history;   // Per-iteration train loss
    std::vector<Scalar> test_loss_history;    // Per-iteration test loss
    int iterations;
};

/**
 * @brief Results from automatic rank search
 *
 * Contains optimal rank, all evaluations, and search metadata.
 *
 * k_optimal             Rank with the lowest test loss among `evaluations`
 * k_low, k_high         Closest evaluated ranks below and above k_optimal
 *                       (equal to k_optimal at the ends of the range)
 * overfitting_detected  True if some evaluated rank above k_optimal has a
 *                       higher test loss, i.e. the minimum is interior
 */
template<typename Scalar>
struct RankSearchResult {
    int k_optimal;
    std::vector<CVEvaluation<Scalar>> evaluations;  // All tested ranks, in evaluation order
    int k_low;
    int k_high;
    bool overfitting_detected;
};

/**
 * @brief Fit NMF at specified rank with CV tracking
 *
 * Routes through nmf::nmf() for unified CV with per-column
 * Gram correction and GPU fallback support.
 *
 * @param A Input matrix
 * @param rank Factorization rank to test
 * @param base_config NMF configuration (must have cv_seed and holdout_fraction set)
 * @return CVEvaluation with train/test losses from nmf::fit
 */
template<typename Scalar, typename MatrixType>
CVEvaluation<Scalar> evaluate_rank_with_cv(
    const MatrixType& A,
    int rank,
    const ::FactorNet::NMFConfig<Scalar>& base_config)
{
    CVEvaluation<Scalar> result;
    result.rank = rank;

    // Build config for this rank
    ::FactorNet::NMFConfig<Scalar> config = base_config;
    config.rank = rank;
    config.track_loss_history = true;  // Need per-iteration losses

    // Use rank-dependent seed for initialization diversity
    if (config.seed > 0) {
        config.seed = base_config.seed + static_cast<uint32_t>(rank);
    }

    // Route through gateway (handles resource detection, GPU fallback, etc.)
    auto nmf_result = nmf::nmf(A, config);

    // Map NMFResult to CVEvaluation. test_loss describes the returned model
    // (the best test iteration when early stopping is on).
    result.iterations = nmf_result.iterations;
    result.train_loss_final = nmf_result.train_loss;
    result.test_loss_final = nmf_result.test_loss;
    result.train_loss_history = nmf_result.loss_history;
    result.test_loss_history = nmf_result.test_loss_history;

    // Fallback if loss history is empty
    if (result.train_loss_history.empty()) {
        result.train_loss_history.push_back(result.train_loss_final);
    }
    if (result.test_loss_history.empty()) {
        result.test_loss_history.push_back(result.test_loss_final);
    }

    return result;
}

namespace rank_detail {

/**
 * @brief Cached rank evaluator: each rank is fitted at most once.
 */
template<typename Scalar, typename MatrixType>
class RankEvaluator {
public:
    RankEvaluator(const MatrixType& A,
                  const ::FactorNet::NMFConfig<Scalar>& config,
                  std::vector<CVEvaluation<Scalar>>& evaluations,
                  bool verbose)
        : A_(A), config_(config), evaluations_(evaluations), verbose_(verbose) {}

    /// Test loss at rank k (+infinity when it is not finite)
    double loss(int k) {
        auto it = cache_.find(k);
        if (it != cache_.end()) return it->second;

        CVEvaluation<Scalar> eval = evaluate_rank_with_cv(A_, k, config_);
        double value = static_cast<double>(eval.test_loss_final);
        if (!std::isfinite(value)) value = std::numeric_limits<double>::infinity();

        FACTORNET_LOG_INFO(verbose_, "Rank %d: train=%g, test=%g\n", k,
                           static_cast<double>(eval.train_loss_final),
                           static_cast<double>(eval.test_loss_final));

        evaluations_.push_back(eval);
        cache_[k] = value;
        return value;
    }

    bool evaluated(int k) const { return cache_.find(k) != cache_.end(); }

    /// Rank with the lowest test loss among all evaluated ranks
    /// (ties go to the smaller rank; std::map iterates in increasing rank)
    int argmin() const {
        int best_k = cache_.begin()->first;
        double best_v = cache_.begin()->second;
        for (const auto& kv : cache_) {
            if (kv.second < best_v) {
                best_v = kv.second;
                best_k = kv.first;
            }
        }
        return best_k;
    }

    /// Closest evaluated rank strictly below k (k itself if none)
    int lower_neighbour(int k) const {
        int out = k;
        for (const auto& kv : cache_) {
            if (kv.first < k) out = kv.first;
            else break;
        }
        return out;
    }

    /// Closest evaluated rank strictly above k (k itself if none)
    int upper_neighbour(int k) const {
        for (const auto& kv : cache_) {
            if (kv.first > k) return kv.first;
        }
        return k;
    }

    /// True if some evaluated rank above k has a strictly higher loss than k
    bool rises_above(int k) const {
        auto it = cache_.find(k);
        if (it == cache_.end()) return false;
        for (const auto& kv : cache_) {
            if (kv.first > k && kv.second > it->second) return true;
        }
        return false;
    }

private:
    const MatrixType& A_;
    const ::FactorNet::NMFConfig<Scalar>& config_;
    std::vector<CVEvaluation<Scalar>>& evaluations_;
    bool verbose_;
    std::map<int, double> cache_;
};

}  // namespace rank_detail

/**
 * @brief Main entry point: automatic rank determination via CV
 *
 * Coarse geometric grid followed by bisection refinement around the best
 * grid point; returns the rank with the lowest test loss among all ranks
 * evaluated. The same holdout mask (config.cv_seed) is used for every rank.
 *
 * @param A Input matrix
 * @param k_init Smallest rank to consider (k_min)
 * @param max_k Largest rank to consider (k_max); capped at min(nrow, ncol)
 * @param tolerance Allowed width of the final bracket [k_low, k_high] around
 *                  the best rank. Values <= 3 refine until both integer
 *                  neighbours of the best rank have been evaluated
 * @param config NMF configuration (must have holdout_fraction in (0,1))
 * @param verbose Print progress
 * @return RankSearchResult with optimal rank and all evaluations
 */
template<typename Scalar, typename MatrixType>
RankSearchResult<Scalar> find_optimal_rank(
    const MatrixType& A,
    int k_init = 2,
    int max_k = 50,
    int tolerance = 2,
    const ::FactorNet::NMFConfig<Scalar>& config = ::FactorNet::NMFConfig<Scalar>(),
    bool verbose = false)
{
    if (!config.is_cv()) {
        throw std::invalid_argument(
            "Automatic rank selection requires cross-validation: "
            "holdout_fraction (test_fraction) must be in (0, 1).");
    }

    const int dim_cap = static_cast<int>(std::min(A.rows(), A.cols()));
    const int k_min = std::max(1, std::min(k_init, dim_cap));
    const int k_max = std::max(k_min, std::min(max_k, dim_cap));
    // tolerance is the allowed width of the final bracket [k_low, k_high];
    // each side of the best rank may therefore be tolerance / 2 away.
    const int gap_tol = std::max(1, tolerance / 2);

    RankSearchResult<Scalar> result;
    result.k_optimal = k_min;
    result.k_low = k_min;
    result.k_high = k_min;
    result.overfitting_detected = false;

    rank_detail::RankEvaluator<Scalar, MatrixType> eval(
        A, config, result.evaluations, verbose);

    // ------------------------------------------------------------------
    // Phase 1: coarse geometric grid
    // ------------------------------------------------------------------
    int consecutive_rises = 0;
    double prev_loss = eval.loss(k_min);
    int k_current = k_min;
    while (k_current < k_max && consecutive_rises < 2) {
        // Next grid point: double, at least +1, never beyond k_max
        int k_next = (k_current > k_max / 2) ? k_max : std::max(k_current + 1, 2 * k_current);
        if (k_next > k_max) k_next = k_max;

        const double loss_next = eval.loss(k_next);
        if (loss_next > prev_loss) {
            ++consecutive_rises;
        } else {
            consecutive_rises = 0;
        }
        prev_loss = loss_next;
        k_current = k_next;
    }

    // ------------------------------------------------------------------
    // Phase 2: refine around the best evaluated rank
    // ------------------------------------------------------------------
    // Invariant: lo < best < hi are evaluated ranks (lo == best or hi == best
    // at the ends of the range). Each pass evaluates the midpoints of
    // [lo, best] and [best, hi]; the gaps at least halve until they close.
    int best = eval.argmin();
    for (;;) {
        const int lo = eval.lower_neighbour(best);
        const int hi = eval.upper_neighbour(best);
        const bool left_open = (best - lo) > gap_tol;
        const bool right_open = (hi - best) > gap_tol;
        if (!left_open && !right_open) break;

        if (left_open) {
            const int mid = lo + (best - lo) / 2;
            if (mid > lo && mid < best) eval.loss(mid);
        }
        if (right_open) {
            const int mid = best + (hi - best + 1) / 2;
            if (mid > best && mid < hi) eval.loss(mid);
        }
        best = eval.argmin();
    }

    // ------------------------------------------------------------------
    // Result: argmin of the test loss over everything that was evaluated
    // ------------------------------------------------------------------
    result.k_optimal = eval.argmin();
    result.k_low = eval.lower_neighbour(result.k_optimal);
    result.k_high = eval.upper_neighbour(result.k_optimal);
    result.overfitting_detected = eval.rises_above(result.k_optimal);

    if (result.overfitting_detected) {
        FACTORNET_LOG_INFO(verbose, "Optimal rank: %d\n", result.k_optimal);
    } else {
        FACTORNET_LOG_INFO(verbose,
            "Optimal rank: %d (test loss did not rise within the range searched)\n",
            result.k_optimal);
    }

    return result;
}

} // namespace nmf
}  // namespace FactorNet
