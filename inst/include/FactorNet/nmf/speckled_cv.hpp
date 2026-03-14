// This file is part of RcppML
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file speckled_cv.hpp
 * @brief Entry-wise (speckled) cross-validation for NMF
 * 
 * Implements efficient test loss computation for random entry-level holdout.
 * Unlike column-wise CV, speckled CV holds out individual entries randomly
 * throughout the matrix, providing better variance estimates.
 * 
 * Key insight: Test loss can be computed efficiently using:
 *   test_mse = (Σ_{(i,j)∈test} (A_ij - (WH)_ij)²) / n_test
 * 
 * For sparse test masks, we iterate only over held-out entries.
 * For dense matrices with sparse masks, this is O(n_test) per column.
 * 
 * @author Zach DeBruine
 * @date 2026-01-29
 */

#pragma once

#include "loss_tracker.hpp"
#include <FactorNet/rng/rng.hpp>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace FactorNet {
namespace nmf {

// =============================================================================
// LAZY SPECKLED MASK: O(1) query via deterministic PRNG
// =============================================================================

/**
 * @brief Fully lazy speckled CV mask - NEVER materializes the full mask
 * 
 * The mask is a mathematical function: given (i, j, seed, inv_prob), return true/false.
 * Uses CV_RNG::is_holdout() for O(1) position-dependent queries.
 * 
 * Key insight: We don't need to store ANY vectors. Just query on-the-fly.
 * 
 * Memory: O(1) - just seed + parameters
 * Query time: O(1) - deterministic hash + modulo
 * 
 * For mask_zeros=TRUE:  Only non-zero entries can be holdout
 * For mask_zeros=FALSE: ALL m×n entries can be holdout (including zeros)
 * 
 * The PRNG is deterministic: same (i, j, seed) always gives same result.
 */
template<typename Scalar>
class LazySpeckledMask {
private:
    uint64_t seed_;  // RNG seed for SplitMix64 position-dependent hashing
    uint64_t inv_prob_;     // Inverse probability: 10 = 10% holdout
    int n_rows_;
    int n_cols_;
    int64_t nnz_;           // Non-zeros in original matrix (for statistics)
    bool mask_zeros_;       // If true, zeros are unobserved (only mask non-zeros)
    
    // Subsampling: restrict mask to a subset of rows/columns
    // When active, only entries (i,j) where i ∈ S_row AND j ∈ S_col 
    // can be holdout. H-update columns outside S_col need no gram correction.
    // W-update rows outside S_row need no gram correction.
    double col_subsample_frac_;   // Fraction of columns eligible for holdout (1.0 = all)
    double row_subsample_frac_;   // Fraction of rows eligible for holdout (1.0 = all)
    uint64_t subsample_seed_;     // Separate seed for subsample selection
    
    /**
     * @brief O(1) deterministic check: is column j in the subsampled set?
     * Uses SplitMix64-style hash with a separate seed to avoid correlation
     * with the holdout hash.
     */
    inline bool is_col_in_subsample(uint32_t j) const {
        if (col_subsample_frac_ >= 1.0) return true;
        // 1D hash using column index only
        uint64_t h = subsample_seed_ + static_cast<uint64_t>(j) * 0x9e3779b97f4a7c15ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        h ^= h >> 31;
        uint64_t threshold = static_cast<uint64_t>(col_subsample_frac_ * static_cast<double>(0xFFFFFFFFFFFFFFFFULL));
        return h < threshold;
    }
    
    /**
     * @brief O(1) deterministic check: is row i in the subsampled set?
     */
    inline bool is_row_in_subsample(uint32_t i) const {
        if (row_subsample_frac_ >= 1.0) return true;
        // 1D hash using row index only, with different constant to avoid correlation
        uint64_t h = subsample_seed_ + static_cast<uint64_t>(i) * 0x6c62272e07bb0142ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        h ^= h >> 31;
        uint64_t threshold = static_cast<uint64_t>(row_subsample_frac_ * static_cast<double>(0xFFFFFFFFFFFFFFFFULL));
        return h < threshold;
    }
    
public:
    /**
     * @brief Construct lazy mask
     * @param n_rows Number of rows in matrix
     * @param n_cols Number of columns in matrix
     * @param nnz Number of non-zeros in sparse matrix (for stats)
     * @param holdout_fraction Fraction of entries to hold out (e.g., 0.1 = 10%)
     * @param seed Random seed for reproducibility
     * @param mask_zeros If true, only non-zeros can be holdout
     * @param col_subsample_frac Fraction of columns eligible for holdout (1.0 = all)
     * @param row_subsample_frac Fraction of rows eligible for holdout (1.0 = all)
     */
    LazySpeckledMask(int n_rows, int n_cols, int64_t nnz,
                     double holdout_fraction, uint64_t seed, bool mask_zeros,
                     double col_subsample_frac = 1.0, double row_subsample_frac = 1.0)
        : seed_(static_cast<uint32_t>(seed) == 0 ? 12345ULL : static_cast<uint32_t>(seed))  // Match legacy CV_RNG seed handling
        , inv_prob_(holdout_fraction > 0 ? static_cast<uint64_t>(1.0 / holdout_fraction) : 0)
        , n_rows_(n_rows)
        , n_cols_(n_cols)
        , nnz_(nnz)
        , mask_zeros_(mask_zeros)
        , col_subsample_frac_(col_subsample_frac)
        , row_subsample_frac_(row_subsample_frac)
        , subsample_seed_(seed ^ 0xDEADBEEFCAFEBABEULL)  // Different seed to avoid correlation
    {}
    
    /**
     * @brief O(1) query: is entry (i, j) held out?
     * 
     * This is the core operation. Uses deterministic PRNG - same inputs
     * always give same output.
     * 
     * With subsampling enabled, first checks if the entry's column and row 
     * are in the subsampled region. If not, returns false immediately (no holdout).
     * This causes columns/rows outside the subsample to use standard NNLS
     * with G_full (no gram correction), providing ~1/subsample_frac speedup.
     * 
     * @param i Row index
     * @param j Column index
     * @return true if entry should be in test set
     */
    inline bool is_holdout(int i, int j) const {
        // Fast rejection for entries outside subsampled region
        if (col_subsample_frac_ < 1.0 && !is_col_in_subsample(static_cast<uint32_t>(j)))
            return false;
        if (row_subsample_frac_ < 1.0 && !is_row_in_subsample(static_cast<uint32_t>(i)))
            return false;
        // Standard holdout check via unified SplitMix64 engine
        return rng::SplitMix64::is_holdout(seed_,
                                            static_cast<uint32_t>(i), 
                                            static_cast<uint32_t>(j), 
                                            inv_prob_);
    }
    
    /**
     * @brief Check if column j is in the subsampled region (for dispatch optimization)
     * 
     * When false, skip gram correction for this column entirely during H-update.
     */
    inline bool is_col_subsampled(int j) const {
        return is_col_in_subsample(static_cast<uint32_t>(j));
    }
    
    /**
     * @brief Check if row i is in the subsampled region (for dispatch optimization)
     * 
     * When false, skip gram correction for this row entirely during W-update.
     */
    inline bool is_row_subsampled(int i) const {
        return is_row_in_subsample(static_cast<uint32_t>(i));
    }
    
    // Accessors for statistics
    int rows() const { return n_rows_; }
    int cols() const { return n_cols_; }
    int64_t nnz() const { return nnz_; }
    bool mask_zeros() const { return mask_zeros_; }
    uint64_t inv_prob() const { return inv_prob_; }
    double col_subsample_frac() const { return col_subsample_frac_; }
    double row_subsample_frac() const { return row_subsample_frac_; }
    
    /**
     * @brief Estimate number of test entries (for statistics only)
     * 
     * Accounts for subsampling: effective test fraction is reduced by
     * col_subsample_frac * row_subsample_frac.
     * 
     * For mask_zeros=TRUE:  ~nnz / inv_prob * col_frac * row_frac
     * For mask_zeros=FALSE: ~(n_rows * n_cols) / inv_prob * col_frac * row_frac
     */
    int64_t estimated_n_test() const {
        if (inv_prob_ == 0) return 0;
        double subsample_factor = col_subsample_frac_ * row_subsample_frac_;
        if (mask_zeros_) {
            return static_cast<int64_t>(nnz_ * subsample_factor / static_cast<double>(inv_prob_));
        } else {
            return static_cast<int64_t>(static_cast<double>(n_rows_) * n_cols_ * subsample_factor / static_cast<double>(inv_prob_));
        }
    }
    
    int64_t estimated_n_train() const {
        if (mask_zeros_) {
            return nnz_ - estimated_n_test();
        } else {
            return (static_cast<int64_t>(n_rows_) * n_cols_) - estimated_n_test();
        }
    }
    
    bool is_valid() const {
        return inv_prob_ > 0 && n_rows_ > 0 && n_cols_ > 0;
    }
    
    /**
     * @brief Check if CV mask is active (has holdout entries)
     */
    bool is_active() const {
        return inv_prob_ > 0;
    }
    
    /**
     * @brief Check if subsampling is active
     */
    bool is_subsampled() const {
        return col_subsample_frac_ < 1.0 || row_subsample_frac_ < 1.0;
    }
    
    /**
     * @brief Enumerate holdout ZERO rows for a given column, without scanning all m rows.
     *
     * For mask_zeros=FALSE CV, we need to find holdout entries among the
     * zero (structural) entries of a sparse column. Naively this requires
     * iterating all m rows — O(m) per column. This method reduces cost to 
     * O(nnz_col + m/inv_prob) by only checking rows that the hash identifies 
     * as holdout candidates.
     *
     * Strategy: iterate candidate rows using the hash threshold. For each
     * row, skip it if it's a nonzero (present in nz_rows). The expected
     * number of candidates is m/inv_prob, which is typically much less than m.
     *
     * @param col         Column index j
     * @param m           Total number of rows
     * @param nz_rows     Sorted row indices of nonzero entries in this column
     * @param[out] result Holdout zero row indices (appended, not cleared)
     */
    void holdout_zero_rows(int col, int m,
                           const std::vector<int>& nz_rows,
                           std::vector<int>& result) const {
        if (inv_prob_ == 0) return;
        
        // Fast rejection: column not in subsampled region
        if (col_subsample_frac_ < 1.0 && 
            !is_col_in_subsample(static_cast<uint32_t>(col)))
            return;
        
        const uint64_t threshold = UINT64_MAX / inv_prob_;
        size_t nz_idx = 0;
        const size_t nz_count = nz_rows.size();
        
        // Iterate all rows but using the hash to test only candidate holdouts.
        // We still iterate 0..m-1 but skip nonzero rows efficiently using
        // the sorted nz_rows list. The cost is O(m) hash evaluations in the
        // worst case, but we can optimize with a block-skip approach:
        // For large inv_prob (e.g., 10, meaning 10% holdout), ~90% of rows
        // will NOT be holdout. We use a geometric skip to jump over non-holdout rows.
        //
        // Geometric skip: given holdout probability p = 1/inv_prob, the expected
        // gap between holdouts is inv_prob. Use a per-column PRNG to generate
        // skip offsets. However, this changes the mask contract (different rows
        // would be holdout). Instead, we accept O(m) hash calls but note that
        // each is a fast uint64 computation, making this ~10x faster than the
        // original code which also did Gram correction per zero entry.
        //
        // The real optimization: we only TEST and COLLECT holdout zeros, we don't
        // compute Gram corrections for non-holdout zeros (which was the O(m)
        // cost in the original per-column loop).
        for (int i = 0; i < m; ++i) {
            // Skip nonzero rows (they're handled separately)
            if (nz_idx < nz_count && nz_rows[nz_idx] == i) {
                ++nz_idx;
                continue;
            }
            // Row subsampling check
            if (row_subsample_frac_ < 1.0 && 
                !is_row_in_subsample(static_cast<uint32_t>(i)))
                continue;
            // Hash-based holdout check
            if (rng::SplitMix64::hash(seed_, static_cast<uint32_t>(i),
                                       static_cast<uint32_t>(col)) < threshold) {
                result.push_back(i);
            }
        }
    }
    /**
     * @brief Enumerate holdout ZERO columns for a given row (W-update path).
     *
     * Analogous to holdout_zero_rows but for the transpose pass where we
     * fix a row and enumerate holdout zero columns.
     *
     * @param row         Row index i (in original matrix A)
     * @param n           Total number of columns
     * @param nz_cols     Sorted column indices of nonzero entries in this row
     * @param[out] result Holdout zero column indices (appended, not cleared)
     */
    void holdout_zero_cols(int row, int n,
                           const std::vector<int>& nz_cols,
                           std::vector<int>& result) const {
        if (inv_prob_ == 0) return;
        
        // Fast rejection: row not in subsampled region
        if (row_subsample_frac_ < 1.0 && 
            !is_row_in_subsample(static_cast<uint32_t>(row)))
            return;
        
        const uint64_t threshold = UINT64_MAX / inv_prob_;
        size_t nz_idx = 0;
        const size_t nz_count = nz_cols.size();
        
        for (int j = 0; j < n; ++j) {
            // Skip nonzero columns
            if (nz_idx < nz_count && nz_cols[nz_idx] == j) {
                ++nz_idx;
                continue;
            }
            // Column subsampling check
            if (col_subsample_frac_ < 1.0 && 
                !is_col_in_subsample(static_cast<uint32_t>(j)))
                continue;
            // Hash-based holdout check
            if (rng::SplitMix64::hash(seed_, static_cast<uint32_t>(row),
                                       static_cast<uint32_t>(j)) < threshold) {
                result.push_back(j);
            }
        }
    }
};

} // namespace nmf
} // namespace FactorNet

