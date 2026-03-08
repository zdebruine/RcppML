// sparsepress - Matrix analyzer (compute statistics for model building)
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_ANALYZER_HPP
#define SPARSEPRESS_ANALYZER_HPP

#include <streampress/core/types.hpp>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace streampress {

// ============================================================================
// Matrix statistics computed in a single O(nnz) pass
// ============================================================================
struct MatrixStats {
    uint32_t m = 0;
    uint32_t n = 0;
    uint64_t nnz = 0;

    // Row statistics
    std::vector<uint32_t> row_nnz;      // [m]: non-zeros per row
    std::vector<double>   row_sum;      // [m]: sum of values per row

    // Column statistics (derivable from p[], but stored explicitly)
    std::vector<uint32_t> col_nnz;      // [n]: non-zeros per column

    // Value distribution
    uint32_t max_value = 0;             // Maximum integer value
    bool all_integer = true;            // All x[] are exact integers?
    bool all_nonneg = true;             // All x[] >= 0?
    uint32_t n_distinct_values = 0;     // Number of distinct values
    std::vector<uint64_t> value_hist;   // [max_value+1]: frequency of each int value

    // Aggregates
    double total_sum = 0.0;
    double density = 0.0;              // nnz / (m * n)

    // Entropy estimate (bits per value symbol)
    double value_entropy = 0.0;
    // Entropy estimate (bits per gap)
    double gap_entropy = 0.0;
};

// ============================================================================
// Analyze a CSC matrix and produce statistics
// ============================================================================
inline MatrixStats analyze(const CSCMatrix& mat) {
    MatrixStats s;
    s.m = mat.m;
    s.n = mat.n;
    s.nnz = mat.nnz;
    s.density = static_cast<double>(mat.nnz) / (static_cast<double>(mat.m) * mat.n);

    s.row_nnz.resize(mat.m, 0);
    s.row_sum.resize(mat.m, 0.0);
    s.col_nnz.resize(mat.n, 0);

    // Single pass over all non-zeros
    // First check max value and integer-ness
    double max_val = 0.0;
    for (uint64_t k = 0; k < mat.nnz; ++k) {
        double v = mat.x[k];
        if (v < 0) s.all_nonneg = false;
        if (v != std::floor(v)) s.all_integer = false;
        if (v > max_val) max_val = v;
        s.total_sum += v;
    }

    if (s.all_integer && s.all_nonneg && max_val <= 4294967295.0) {
        s.max_value = static_cast<uint32_t>(max_val);
    } else if (s.all_integer && s.all_nonneg) {
        s.max_value = UINT32_MAX;
    } else {
        // Fallback: not integer data
        s.max_value = 0;
    }

    // Build value histogram (only for integer data)
    if (s.all_integer && s.all_nonneg && s.max_value <= 65536) {
        s.value_hist.resize(s.max_value + 1, 0);
        for (uint64_t k = 0; k < mat.nnz; ++k) {
            uint32_t v = static_cast<uint32_t>(mat.x[k]);
            s.value_hist[v]++;
        }
        s.n_distinct_values = 0;
        for (auto c : s.value_hist)
            if (c > 0) s.n_distinct_values++;
    } else {
        // For non-integer data, build a coarser histogram
        s.n_distinct_values = 0; // unknown
    }

    // Row statistics
    for (uint32_t j = 0; j < mat.n; ++j) {
        uint32_t col_start = mat.p[j];
        uint32_t col_end = mat.p[j + 1];
        s.col_nnz[j] = col_end - col_start;

        for (uint32_t k = col_start; k < col_end; ++k) {
            uint32_t row = mat.i[k];
            s.row_nnz[row]++;
            s.row_sum[row] += mat.x[k];
        }
    }

    // Compute value entropy
    if (!s.value_hist.empty()) {
        s.value_entropy = 0.0;
        for (uint32_t v = 0; v <= s.max_value; ++v) {
            if (s.value_hist[v] == 0) continue;
            double p = static_cast<double>(s.value_hist[v]) / s.nnz;
            s.value_entropy -= p * std::log2(p);
        }
    }

    // Compute gap entropy (empirical - sample from first few columns)
    {
        uint64_t gap_sum = 0;
        uint64_t gap_count = 0;
        uint32_t cols_to_sample = std::min(mat.n, 1000u);
        for (uint32_t j = 0; j < cols_to_sample; ++j) {
            uint32_t col_start = mat.p[j];
            uint32_t col_end = mat.p[j + 1];
            if (col_end <= col_start) continue;

            uint32_t prev = 0;
            for (uint32_t k = col_start; k < col_end; ++k) {
                uint32_t gap = mat.i[k] - prev;
                gap_sum += gap;
                gap_count++;
                prev = mat.i[k] + 1;
            }
        }
        if (gap_count > 0) {
            double mean_gap = static_cast<double>(gap_sum) / gap_count;
            // For geometric distribution, entropy = -log2(p)/p - log2(1-p)/(1-p)
            // Approximation: ≈ log2(mean_gap + 1) + 1
            s.gap_entropy = std::log2(mean_gap + 1.0) + 1.0;
        }
    }

    return s;
}

} // namespace streampress

#endif // SPARSEPRESS_ANALYZER_HPP
