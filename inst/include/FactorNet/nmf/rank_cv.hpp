// Automatic rank determination via cross-validation
// Uses exponential search + golden section refinement
//
// Routes through nmf::nmf() for unified CV infrastructure:
// - Uses fit_cv_unified with per-column Gram correction
// - Supports GPU fallback via gateway resource detection
// - Uses core::NMFConfig and NMFResult types
//
// Copyright (C) 2021-2026 Zach DeBruine

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/nmf/fit.hpp>
#include <cmath>
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
 */
template<typename Scalar>
struct RankSearchResult {
    int k_optimal;
    std::vector<CVEvaluation<Scalar>> evaluations;  // All tested ranks
    int k_low;   // Lower bound from exponential search
    int k_high;  // Upper bound from exponential search
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
    
    // Map NMFResult to CVEvaluation
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

/**
 * @brief Phase 1: Exponential search to find overfitting boundary
 * 
 * Tests ranks: k_init, k_init*2, k_init*4, ... until overfitting detected.
 * Overfitting = test loss increases while train loss converged.
 * 
 * @return RankSearchResult with k_low, k_high bracket if overfitting found
 */
template<typename Scalar, typename MatrixType>
RankSearchResult<Scalar> exponential_search(
    const MatrixType& A,
    int k_init,
    int max_k,
    const ::FactorNet::NMFConfig<Scalar>& config,
    bool verbose = false)
{
    RankSearchResult<Scalar> result;
    result.k_optimal = k_init;
    result.k_low = -1;
    result.k_high = -1;
    result.overfitting_detected = false;
    
    int k_current = k_init;
    
    while (k_current <= max_k) {
        FACTORNET_LOG_INFO(verbose, "Testing rank %d... ", k_current);
        
        // Evaluate this rank using masked CV
        CVEvaluation<Scalar> eval = evaluate_rank_with_cv(A, k_current, config);
        result.evaluations.push_back(eval);
        
        FACTORNET_LOG_INFO(verbose, "train=%g, test=%g\n", 
                       static_cast<double>(eval.train_loss_final),
                       static_cast<double>(eval.test_loss_final));
        
        // Check for overfitting (need at least 2 evaluations)
        if (result.evaluations.size() >= 2) {
            const auto& prev = result.evaluations[result.evaluations.size() - 2];
            const auto& curr = result.evaluations[result.evaluations.size() - 1];
            
            // Train loss converged?
            Scalar train_rel_change = std::abs(curr.train_loss_final - prev.train_loss_final) 
                                     / (prev.train_loss_final + tiny_num<Scalar>());
            bool train_converged = train_rel_change < Scalar(0.01);
            
            // Test loss increased?
            bool test_increased = curr.test_loss_final > prev.test_loss_final;
            
            if (train_converged && test_increased) {
                FACTORNET_LOG_INFO(verbose, "Overfitting detected!\n");
                result.k_low = prev.rank;
                result.k_high = curr.rank;
                result.overfitting_detected = true;
                return result;
            }
        }
        
        // Next power of 2
        if (k_current * 2 > max_k && k_current < max_k) {
            k_current = max_k;  // Test max_k before stopping
        } else {
            k_current *= 2;
        }
    }
    
    // No overfitting detected - use highest tested rank
    if (!result.evaluations.empty()) {
        result.k_optimal = result.evaluations.back().rank;
    }
    
    return result;
}

/**
 * @brief Phase 2: Golden section search to refine rank selection
 * 
 * Narrows down [k_low, k_high] bracket using golden ratio.
 * 
 * @return Optimal rank (conservative choice = lower bound)
 */
template<typename Scalar, typename MatrixType>
int golden_section_search(
    const MatrixType& A,
    int k_low,
    int k_high,
    int tolerance,
    const ::FactorNet::NMFConfig<Scalar>& config,
    std::vector<CVEvaluation<Scalar>>& evaluations,
    bool verbose = false)
{
    const Scalar phi = (Scalar(1) + std::sqrt(Scalar(5))) / Scalar(2);  // Golden ratio
    
    FACTORNET_LOG_INFO(verbose, "\nRefining search in [%d, %d]...\n", k_low, k_high);
    
    while ((k_high - k_low) > tolerance) {
        int k1 = static_cast<int>(k_high - (k_high - k_low) / phi + Scalar(0.5));
        int k2 = static_cast<int>(k_low + (k_high - k_low) / phi + Scalar(0.5));
        
        // Avoid duplicates and ensure valid range
        if (k1 <= k_low || k2 >= k_high || k1 >= k2) break;
        
        FACTORNET_LOG_INFO(verbose, "Testing ranks %d and %d... ", k1, k2);
        
        // Evaluate both ranks using masked CV
        CVEvaluation<Scalar> eval1 = evaluate_rank_with_cv(A, k1, config);
        CVEvaluation<Scalar> eval2 = evaluate_rank_with_cv(A, k2, config);
        
        evaluations.push_back(eval1);
        evaluations.push_back(eval2);
        
        FACTORNET_LOG_INFO(verbose, "test[%d]=%g, test[%d]=%g\n", 
                       k1, static_cast<double>(eval1.test_loss_final),
                       k2, static_cast<double>(eval2.test_loss_final));
        
        // Update bracket
        if (eval1.test_loss_final < eval2.test_loss_final) {
            k_high = k2;  // Optimal is in [k_low, k2]
        } else {
            k_low = k1;   // Optimal is in [k1, k_high]
        }
    }
    
    // Return conservative choice (lower bound)
    return k_low;
}

/**
 * @brief Main entry point: automatic rank determination via CV
 * 
 * Uses exponential search followed by golden section refinement.
 * Leverages existing 2D speckled mask infrastructure for CV.
 * 
 * @param A Input matrix
 * @param k_init Starting rank for exponential search
 * @param max_k Maximum rank to test
 * @param tolerance Stop refining when bracket narrows to this size
 * @param config NMF configuration (should have cv_seed and holdout_fraction set)
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
    // Phase 1: Exponential search
    auto result = exponential_search(A, k_init, max_k, config, verbose);
    
    // Phase 2: Golden section refinement (if overfitting detected)
    if (result.overfitting_detected) {
        result.k_optimal = golden_section_search(
            A, result.k_low, result.k_high, tolerance,
            config, result.evaluations, verbose
        );
        
        FACTORNET_LOG_INFO(verbose, "Optimal rank: %d\n", result.k_optimal);
    } else {
        FACTORNET_LOG_INFO(verbose, "No overfitting detected. Using rank: %d\n", 
                       result.k_optimal);
    }
    
    return result;
}

} // namespace nmf
}  // namespace FactorNet

