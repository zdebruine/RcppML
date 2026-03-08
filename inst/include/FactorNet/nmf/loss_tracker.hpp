// This file is part of RcppML
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file loss_tracker.hpp
 * @brief Comprehensive loss tracking for NMF fitting
 * 
 * Tracks training loss, test loss (CV mode), and regularization losses
 * using efficient Gram matrix computations with minimal overhead.
 * 
 * Key formulas:
 *   Training MSE = (||A||² - 2*cross_term + recon_norm²) / n_train
 *   Test MSE = computed similarly for held-out entries
 *   L1 loss = λ₁ * (||W||₁ + ||H||₁)
 *   L2 loss = λ₂ * (||W||²_F + ||H||²_F)
 * 
 * @author Zach DeBruine
 * @date 2026-01-29
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace FactorNet {
namespace nmf {

// Import type aliases from parent namespace (core/types.hpp)
template<typename S> using DenseMatrix = FactorNet::DenseMatrix<S>;
template<typename S> using DenseVector = FactorNet::DenseVector<S>;

/**
 * Configuration for loss tracking
 */
struct LossTrackConfig {
    bool track_loss = true;           // Enable loss tracking
    bool track_test_loss = false;     // Track test loss (CV mode)
    bool track_reg_loss = false;      // Track regularization loss
    bool store_history = false;       // Store per-iteration losses
    
    // Regularization weights (for computing regularization loss)
    double L1_W = 0, L1_H = 0;
    double L2_W = 0, L2_H = 0;
    
    // Test set indicators (for CV mode)
    std::vector<bool> is_test_column;  // Column-wise CV
    int n_test_entries = 0;
    int n_train_entries = 0;
};

/**
 * Per-epoch loss values
 */
template<typename Scalar>
struct EpochLoss {
    Scalar train_mse = 0;
    Scalar test_mse = 0;
    Scalar L1_loss = 0;
    Scalar L2_loss = 0;
    Scalar total = 0;
    
    void compute_total() {
        total = train_mse + L1_loss + L2_loss;
    }
};

/**
 * Comprehensive loss tracker for NMF fitting
 */
template<typename Scalar>
class LossTracker {
public:
    // ===== Configuration =====
    bool enabled = true;
    bool track_test = false;
    bool track_reg = false;
    bool store_history = false;
    
    // ===== Precomputed values =====
    Scalar A_norm_sq = 0;           // ||A||²_F (computed once at start)
    Scalar A_train_norm_sq = 0;     // ||A_train||²_F (for CV mode)
    int total_entries = 0;          // m * n
    int n_train_entries = 0;        // Number of training entries
    int n_test_entries = 0;         // Number of test entries
    
    // ===== Per-epoch accumulators (H update) =====
    Scalar train_cross_term = 0;    // Σⱼ∈train bⱼ'hⱼ
    Scalar train_recon_sq = 0;      // Σⱼ∈train hⱼ'Ghⱼ
    Scalar test_cross_term = 0;     // Σⱼ∈test bⱼ'hⱼ  
    Scalar test_recon_sq = 0;       // Σⱼ∈test hⱼ'Ghⱼ
    
    // ===== Per-epoch H norms (for regularization) =====
    Scalar H_L1_norm = 0;           // ||H||₁ = Σᵢⱼ |Hᵢⱼ|
    Scalar H_L2_norm_sq = 0;        // ||H||²_F = Σᵢⱼ H²ᵢⱼ
    
    // ===== Factor norms (for regularization computation in fit loop) =====
    Scalar train_W_L1 = 0;          // ||W||₁
    Scalar train_W_L2 = 0;          // ||W||²_F
    Scalar train_H_L1 = 0;          // ||H||₁
    Scalar train_H_L2 = 0;          // ||H||²_F
    
    // ===== Regularization weights =====
    Scalar L1_W = 0, L1_H = 0;
    Scalar L2_W = 0, L2_H = 0;
    
    // ===== Loss history =====
    std::vector<EpochLoss<Scalar>> history;
    
    // ===== Current epoch loss =====
    EpochLoss<Scalar> current;
    Scalar prev_total_loss = std::numeric_limits<Scalar>::max();
    
    /**
     * Initialize tracker with matrix dimensions and precomputed ||A||²
     */
    void initialize(int m, int n, Scalar A_sq_norm) {
        A_norm_sq = A_sq_norm;
        total_entries = m * n;
        n_train_entries = total_entries;
        n_test_entries = 0;
        reset_epoch();
    }
    
    /**
     * Initialize for CV mode with test column indicators
     */
    void initialize_cv(int m, int n, Scalar A_sq_norm, 
                      const std::vector<bool>& is_test_col,
                      Scalar A_train_sq_norm = 0) {
        A_norm_sq = A_sq_norm;
        A_train_norm_sq = A_train_sq_norm;
        total_entries = m * n;
        track_test = true;
        
        // Count test/train entries
        n_test_entries = 0;
        for (int j = 0; j < n; ++j) {
            if (is_test_col[j]) {
                n_test_entries += m;
            }
        }
        n_train_entries = total_entries - n_test_entries;
        
        reset_epoch();
    }
    
    /**
     * Reset accumulators for a new epoch
     */
    void reset_epoch() {
        train_cross_term = 0;
        train_recon_sq = 0;
        test_cross_term = 0;
        test_recon_sq = 0;
        H_L1_norm = 0;
        H_L2_norm_sq = 0;
    }
    
    /**
     * Accumulate loss for a single H column
     * 
     * @param b_orig   Original RHS vector (W'A[:,j]) BEFORE CD solve modified it
     * @param h_j      Solved H column
     * @param G        Gram matrix W'W (NOT including epsilon/L2 on diagonal)
     * @param is_test  Whether this is a test column (CV mode)
     */
    void accumulate_H_column(
        const DenseVector<Scalar>& b_orig,
        const DenseVector<Scalar>& h_j,
        const DenseMatrix<Scalar>& G,
        bool is_test = false
    ) {
        if (!enabled) return;
        
        // Cross term: b'h = O(k)
        Scalar cross = b_orig.dot(h_j);
        
        // Reconstruction norm: h'Gh = O(k²)
        Scalar recon = h_j.dot(G * h_j);
        
        if (is_test && track_test) {
            test_cross_term += cross;
            test_recon_sq += recon;
        } else {
            train_cross_term += cross;
            train_recon_sq += recon;
        }
        
        // Regularization norms: O(k)
        if (track_reg) {
            H_L1_norm += h_j.template lpNorm<1>();
            H_L2_norm_sq += h_j.squaredNorm();
        }
    }
    
    /**
     * Accumulate loss for a single H column (unscaled version - no d diagonal)
     * Overload when tracking is definitely training only
     */
    void accumulate_H_column_train(
        const DenseVector<Scalar>& b_orig,
        const DenseVector<Scalar>& h_j,
        const DenseMatrix<Scalar>& G
    ) {
        if (!enabled) return;
        
        train_cross_term += b_orig.dot(h_j);
        train_recon_sq += h_j.dot(G * h_j);
        
        if (track_reg) {
            H_L1_norm += h_j.template lpNorm<1>();
            H_L2_norm_sq += h_j.squaredNorm();
        }
    }
    
    /**
     * Compute W regularization norms after W update
     * Called once per epoch (not per column) for efficiency
     */
    void compute_W_reg_norms(const DenseMatrix<Scalar>& W_T) {
        if (!enabled || !track_reg) return;
        
        // W_T is k × m, compute norms: O(km)
        current.L1_loss = L1_W * W_T.template lpNorm<1>() + L1_H * H_L1_norm;
        current.L2_loss = L2_W * W_T.squaredNorm() + L2_H * H_L2_norm_sq;
    }
    
    /**
     * Finalize epoch and compute losses
     * 
     * @return Relative change in total loss for convergence check
     */
    Scalar finalize_epoch() {
        if (!enabled) {
            return std::numeric_limits<Scalar>::max();
        }
        
        // Training MSE: (||A||² - 2*cross + recon) / n_train
        // For full matrix (no CV): n_train = total_entries
        Scalar train_sq_err = A_norm_sq - 2 * train_cross_term + train_recon_sq;
        if (track_test) {
            // In CV mode, subtract test contribution from A_norm_sq
            train_sq_err = A_train_norm_sq - 2 * train_cross_term + train_recon_sq;
        }
        current.train_mse = (n_train_entries > 0) ? 
            std::max(Scalar(0), train_sq_err) / n_train_entries : 0;
        
        // Test MSE (CV mode)
        if (track_test && n_test_entries > 0) {
            Scalar test_A_norm = A_norm_sq - A_train_norm_sq;  // ||A_test||²
            Scalar test_sq_err = test_A_norm - 2 * test_cross_term + test_recon_sq;
            current.test_mse = std::max(Scalar(0), test_sq_err) / n_test_entries;
        }
        
        // Total loss
        current.compute_total();
        
        // Relative change for convergence
        Scalar rel_change = std::abs(current.total - prev_total_loss) / 
                           (std::abs(prev_total_loss) + static_cast<Scalar>(1e-15));
        prev_total_loss = current.total;
        
        // Store history
        if (store_history) {
            history.push_back(current);
        }
        
        return rel_change;
    }
    
    /**
     * Get verbose output string for current epoch
     */
    std::string verbose_string(int iter) const {
        std::ostringstream oss;
        oss << "Iter " << (iter + 1) << ": ";
        oss << std::fixed << std::setprecision(6);
        oss << "train_mse=" << current.train_mse;
        
        if (track_reg && (L1_W > 0 || L1_H > 0 || L2_W > 0 || L2_H > 0)) {
            oss << " reg=" << (current.L1_loss + current.L2_loss);
        }
        
        if (track_test) {
            oss << " test_mse=" << current.test_mse;
        }
        
        oss << " total=" << current.total;
        return oss.str();
    }
    
    /**
     * Extract loss history vectors for R return
     */
    std::vector<Scalar> get_train_loss_history() const {
        std::vector<Scalar> result;
        result.reserve(history.size());
        for (const auto& h : history) {
            result.push_back(h.train_mse);
        }
        return result;
    }
    
    std::vector<Scalar> get_test_loss_history() const {
        std::vector<Scalar> result;
        result.reserve(history.size());
        for (const auto& h : history) {
            result.push_back(h.test_mse);
        }
        return result;
    }
    
    std::vector<Scalar> get_total_loss_history() const {
        std::vector<Scalar> result;
        result.reserve(history.size());
        for (const auto& h : history) {
            result.push_back(h.total);
        }
        return result;
    }
};

/**
 * Helper: Compute ||A||² for dense matrix
 */
template<typename Scalar>
inline Scalar compute_A_norm_sq(const DenseMatrix<Scalar>& A) {
    return A.squaredNorm();
}

/**
 * Helper: Compute ||A||² for sparse matrix
 * Template on SparseMat to support both Eigen::SparseMatrix and Eigen::Map
 */
template<typename SparseMat>
inline auto compute_A_norm_sq(const SparseMat& A) -> typename SparseMat::Scalar {
    using Scalar = typename SparseMat::Scalar;
    Scalar norm_sq = 0;
    for (int j = 0; j < A.outerSize(); ++j) {
        for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
            norm_sq += it.value() * it.value();
        }
    }
    return norm_sq;
}

/**
 * Helper: Compute ||A_train||² for CV mode with test columns
 */
template<typename Scalar>
inline Scalar compute_train_norm_sq(
    const DenseMatrix<Scalar>& A,
    const std::vector<bool>& is_test_col
) {
    Scalar norm_sq = 0;
    for (int j = 0; j < A.cols(); ++j) {
        if (!is_test_col[j]) {
            norm_sq += A.col(j).squaredNorm();
        }
    }
    return norm_sq;
}

/**
 * Helper: Compute ||A_train||² for CV mode with test columns
 * Template on SparseMat to support both Eigen::SparseMatrix and Eigen::Map
 */
template<typename SparseMat>
inline auto compute_train_norm_sq(
    const SparseMat& A,
    const std::vector<bool>& is_test_col
) -> typename SparseMat::Scalar {
    using Scalar = typename SparseMat::Scalar;
    Scalar norm_sq = 0;
    for (int j = 0; j < A.outerSize(); ++j) {
        if (!is_test_col[j]) {
            for (typename SparseMat::InnerIterator it(A, j); it; ++it) {
                norm_sq += it.value() * it.value();
            }
        }
    }
    return norm_sq;
}

} // namespace nmf
} // namespace FactorNet

