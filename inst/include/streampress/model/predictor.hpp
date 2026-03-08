// sparsepress - Predictive models for delta and value compression
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)
//
// Uses row/column marginals to predict values and index gaps,
// encoding residuals instead of raw values for lower entropy.

#ifndef SPARSEPRESS_PREDICTOR_HPP
#define SPARSEPRESS_PREDICTOR_HPP

#include <streampress/core/types.hpp>
#include <streampress/core/prng.hpp>
#include <streampress/model/analyzer.hpp>
#include <streampress/codec/varint.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>

namespace streampress {

// ============================================================================
// Row-density block model for delta prediction
//
// Divides rows into B blocks, each with an aggregate density.
// For a gap starting at row r in column j, the predicted gap length is:
//   predicted = (1 - block_density) / block_density
// where block_density is the density of the block containing row r.
//
// The residual (actual_gap - predicted) has lower variance than raw gaps.
// ============================================================================
struct DensityModel {
    uint32_t n_blocks;
    uint32_t block_size;
    std::vector<uint16_t> block_density_q;  // Quantized density × 65535

    // Build from row nnz statistics
    void build(const std::vector<uint32_t>& row_nnz, uint32_t m, uint32_t n,
               uint32_t n_blocks_hint) {
        n_blocks = std::min(n_blocks_hint, m);
        if (n_blocks == 0) n_blocks = 1;
        block_size = (m + n_blocks - 1) / n_blocks;

        block_density_q.resize(n_blocks);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            uint32_t row_start = b * block_size;
            uint32_t row_end = std::min(row_start + block_size, m);
            uint64_t block_nnz = 0;
            for (uint32_t r = row_start; r < row_end; ++r) {
                block_nnz += row_nnz[r];
            }
            double density = static_cast<double>(block_nnz)
                           / (static_cast<double>(row_end - row_start) * n);
            // Quantize to uint16 (0 = zero density, 65535 = full density)
            // Clamp to [1/65535, 1] to avoid division by zero
            density = std::max(density, 1.0 / 65535.0);
            density = std::min(density, 1.0);
            block_density_q[b] = static_cast<uint16_t>(density * 65535.0 + 0.5);
            if (block_density_q[b] == 0) block_density_q[b] = 1;
        }
    }

    // Get density for a given row
    double density_at(uint32_t row) const {
        uint32_t b = row / block_size;
        if (b >= n_blocks) b = n_blocks - 1;
        return static_cast<double>(block_density_q[b]) / 65535.0;
    }

    // Predict gap length given current row position
    int32_t predict_gap(uint32_t row) const {
        double d = density_at(row);
        // Expected gap for geometric with success prob = d: (1-d)/d
        double predicted = (1.0 - d) / d;
        return static_cast<int32_t>(std::round(predicted));
    }

    // Serialize the density model
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> out;
        varint::encode(n_blocks, out);
        varint::encode(block_size, out);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            out.push_back(static_cast<uint8_t>(block_density_q[b] & 0xFF));
            out.push_back(static_cast<uint8_t>((block_density_q[b] >> 8) & 0xFF));
        }
        return out;
    }

    // Deserialize
    static DensityModel deserialize(const uint8_t*& ptr) {
        DensityModel dm;
        dm.n_blocks = static_cast<uint32_t>(varint::decode(ptr));
        dm.block_size = static_cast<uint32_t>(varint::decode(ptr));
        dm.block_density_q.resize(dm.n_blocks);
        for (uint32_t b = 0; b < dm.n_blocks; ++b) {
            dm.block_density_q[b] = static_cast<uint16_t>(ptr[0])
                                  | (static_cast<uint16_t>(ptr[1]) << 8);
            ptr += 2;
        }
        return dm;
    }
};

// ============================================================================
// Value predictor using independence model
//
// predicted_value = max(1, round(row_mean * col_ratio))
// where row_mean = row_sum / row_nnz
//       col_ratio = col_nnz / mean(col_nnz)
//
// Residual = actual - predicted (signed, ZigZag encoded)
// ============================================================================
struct ValuePredictor {
    std::vector<uint16_t> row_mean_q;   // Quantized row means × 256
    double global_col_mean;              // Mean col_nnz

    void build(const MatrixStats& stats) {
        row_mean_q.resize(stats.m);
        for (uint32_t r = 0; r < stats.m; ++r) {
            double rm = (stats.row_nnz[r] > 0)
                ? stats.row_sum[r] / stats.row_nnz[r]
                : 1.0;
            // Quantize: store as fixed-point × 256, cap at 65535 (= mean of 255.99)
            uint32_t q = static_cast<uint32_t>(rm * 256.0 + 0.5);
            row_mean_q[r] = static_cast<uint16_t>(std::min(q, 65535u));
        }

        uint64_t total_col_nnz = 0;
        for (uint32_t j = 0; j < stats.n; ++j)
            total_col_nnz += stats.col_nnz[j];
        global_col_mean = static_cast<double>(total_col_nnz) / stats.n;
        if (global_col_mean < 1.0) global_col_mean = 1.0;

        // Apply quantization now so compress uses the same quantized value
        // that decompress will get after deserialize
        uint32_t gcm_q = static_cast<uint32_t>(global_col_mean * 256.0 + 0.5);
        global_col_mean = static_cast<double>(gcm_q) / 256.0;
    }

    // Predict value for entry at (row, col_nnz)
    uint32_t predict(uint32_t row, uint32_t col_nnz) const {
        double rm = static_cast<double>(row_mean_q[row]) / 256.0;
        double col_ratio = static_cast<double>(col_nnz) / global_col_mean;
        int32_t predicted = static_cast<int32_t>(std::round(rm * col_ratio));
        return static_cast<uint32_t>(std::max(1, predicted));
    }

    // Serialize
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> out;
        uint32_t m = static_cast<uint32_t>(row_mean_q.size());
        varint::encode(m, out);
        // Encode global_col_mean as fixed-point × 256
        uint32_t gcm_q = static_cast<uint32_t>(global_col_mean * 256.0 + 0.5);
        varint::encode(gcm_q, out);
        // Delta-encode row means for compactness
        int32_t prev = 0;
        for (uint32_t r = 0; r < m; ++r) {
            int32_t d = static_cast<int32_t>(row_mean_q[r]) - prev;
            varint::encode_signed(d, out);
            prev = static_cast<int32_t>(row_mean_q[r]);
        }
        return out;
    }

    // Deserialize
    static ValuePredictor deserialize(const uint8_t*& ptr) {
        ValuePredictor vp;
        uint32_t m = static_cast<uint32_t>(varint::decode(ptr));
        uint32_t gcm_q = static_cast<uint32_t>(varint::decode(ptr));
        vp.global_col_mean = static_cast<double>(gcm_q) / 256.0;
        vp.row_mean_q.resize(m);
        int32_t prev = 0;
        for (uint32_t r = 0; r < m; ++r) {
            int32_t d = static_cast<int32_t>(varint::decode_signed(ptr));
            prev += d;
            vp.row_mean_q[r] = static_cast<uint16_t>(prev & 0xFFFF);
        }
        return vp;
    }
};

} // namespace streampress

#endif // SPARSEPRESS_PREDICTOR_HPP
