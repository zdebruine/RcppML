// sparsepress - Main public API header
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_HPP
#define SPARSEPRESS_HPP

#include <streampress/core/types.hpp>
#include <streampress/core/prng.hpp>
#include <streampress/codec/bitstream.hpp>
#include <streampress/codec/varint.hpp>
#include <streampress/codec/golomb_rice.hpp>
#include <streampress/codec/rans.hpp>
#include <streampress/format/header.hpp>
#include <streampress/format/checksum.hpp>
#include <streampress/model/analyzer.hpp>
#include <streampress/model/predictor.hpp>
#include <streampress/transform/delta.hpp>
#include <streampress/transform/value_map.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>

// When compiled within R packages, use R's messaging instead of stderr
#ifdef SPARSEPRESS_USE_R
#include <R_ext/Print.h>
#define SPZ_FPRINTF(stream, ...) REprintf(__VA_ARGS__)
#else
#define SPZ_FPRINTF(stream, ...) std::fprintf(stream, __VA_ARGS__)
#endif

namespace streampress {

// ============================================================================
// Compress a CSC sparse matrix
// ============================================================================
inline std::vector<uint8_t> compress(const CSCMatrix& mat,
                                      const CompressConfig& cfg = {},
                                      CompressStats* stats_out = nullptr) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Phase 0: Analyze
    // ========================================================================
    MatrixStats mstats = analyze(mat);

    if (cfg.verbose) {
        SPZ_FPRINTF(stderr, "[sparsepress] Matrix: %u × %u, nnz=%lu, density=%.4f%%\n",
                     mstats.m, mstats.n, (unsigned long)mstats.nnz,
                     mstats.density * 100.0);
        SPZ_FPRINTF(stderr, "[sparsepress] Max value: %u, all_integer=%d, distinct=%u\n",
                     mstats.max_value, mstats.all_integer, mstats.n_distinct_values);
        SPZ_FPRINTF(stderr, "[sparsepress] Value entropy: %.3f bits, Gap entropy est: %.3f bits\n",
                     mstats.value_entropy, mstats.gap_entropy);
    }

    // ========================================================================
    // Phase 1: Build models
    // ========================================================================
    DensityModel density_model;
    ValuePredictor value_pred;

    bool use_delta_pred = cfg.use_delta_prediction && mstats.nnz > 0;
    bool use_value_pred = cfg.use_value_prediction && mstats.all_integer
                          && mstats.all_nonneg && mstats.nnz > 0;

    if (use_delta_pred) {
        density_model.build(mstats.row_nnz, mstats.m, mstats.n, cfg.density_blocks);
    }
    if (use_value_pred) {
        value_pred.build(mstats);
    }

    // Serialize models
    std::vector<uint8_t> model_data;
    if (use_delta_pred) {
        auto dm_ser = density_model.serialize();
        model_data.insert(model_data.end(), dm_ser.begin(), dm_ser.end());
    }
    if (use_value_pred) {
        auto vp_ser = value_pred.serialize();
        model_data.insert(model_data.end(), vp_ser.begin(), vp_ser.end());
    }

    // ========================================================================
    // Phase 2: Encode column counts (delta of col_ptrs as VarInt)
    // ========================================================================
    std::vector<uint8_t> col_counts_data;
    for (uint32_t j = 0; j < mat.n; ++j) {
        varint::encode(mat.p[j + 1] - mat.p[j], col_counts_data);
    }

    // ========================================================================
    // Phase 3: Encode structure (row index gaps)
    // ========================================================================
    // Delta encode all indices
    std::vector<uint32_t> all_gaps = delta::encode_all(mat);

    // Apply delta prediction if enabled
    std::vector<uint32_t> structure_symbols;  // What we'll actually encode
    if (use_delta_pred) {
        // Convert to signed residuals, then ZigZag
        structure_symbols.resize(mat.nnz);
        for (uint32_t j = 0; j < mat.n; ++j) {
            uint32_t col_start = mat.p[j];
            uint32_t col_end = mat.p[j + 1];
            uint32_t current_row = 0;
            for (uint32_t k = col_start; k < col_end; ++k) {
                int32_t predicted = density_model.predict_gap(current_row);
                int32_t residual = static_cast<int32_t>(all_gaps[k]) - predicted;
                // ZigZag encode
                structure_symbols[k] = static_cast<uint32_t>(
                    (residual << 1) ^ (residual >> 31));
                // Advance current row position
                current_row += all_gaps[k] + 1;
            }
        }
    } else {
        structure_symbols = std::move(all_gaps);
    }

    // Encode with rANS + escape coding (same approach as values)
    // Symbols 0..254 encode directly; 255 = escape + VarInt
    constexpr uint32_t STRUCT_RANS_MAX = 255;
    constexpr uint32_t STRUCT_ESCAPE = 255;

    std::vector<uint8_t> struct_encoded;
    std::vector<uint8_t> struct_k_data;  // repurposed: overflow data for structure

    if (structure_symbols.size() > 0) {
        std::vector<uint32_t> rans_syms(structure_symbols.size());
        std::vector<uint8_t> overflow_buf;

        bool has_overflow = false;
        for (uint64_t k = 0; k < structure_symbols.size(); ++k) {
            if (structure_symbols[k] >= STRUCT_ESCAPE) {
                rans_syms[k] = STRUCT_ESCAPE;
                varint::encode(structure_symbols[k], overflow_buf);
                has_overflow = true;
            } else {
                rans_syms[k] = structure_symbols[k];
            }
        }

        uint32_t max_sym = has_overflow ? STRUCT_RANS_MAX : 0;
        for (uint64_t k = 0; k < rans_syms.size(); ++k)
            if (rans_syms[k] > max_sym) max_sym = rans_syms[k];

        RansTable table = rans::build_table(rans_syms.data(),
                                             rans_syms.size(), max_sym);
        struct_encoded = table.serialize();
        auto encoded = rans::encode_array(rans_syms.data(),
                                           rans_syms.size(), table);

        // Append: enc_size(4) + encoded + overflow_size(4) + overflow
        uint32_t enc_sz = static_cast<uint32_t>(encoded.size());
        struct_encoded.push_back(static_cast<uint8_t>(enc_sz & 0xFF));
        struct_encoded.push_back(static_cast<uint8_t>((enc_sz >> 8) & 0xFF));
        struct_encoded.push_back(static_cast<uint8_t>((enc_sz >> 16) & 0xFF));
        struct_encoded.push_back(static_cast<uint8_t>((enc_sz >> 24) & 0xFF));
        struct_encoded.insert(struct_encoded.end(), encoded.begin(), encoded.end());

        uint32_t ov_sz = static_cast<uint32_t>(overflow_buf.size());
        struct_encoded.push_back(static_cast<uint8_t>(ov_sz & 0xFF));
        struct_encoded.push_back(static_cast<uint8_t>((ov_sz >> 8) & 0xFF));
        struct_encoded.push_back(static_cast<uint8_t>((ov_sz >> 16) & 0xFF));
        struct_encoded.push_back(static_cast<uint8_t>((ov_sz >> 24) & 0xFF));
        struct_encoded.insert(struct_encoded.end(),
                              overflow_buf.begin(), overflow_buf.end());
    }

    // ========================================================================
    // Phase 4: Encode values
    // ========================================================================
    std::vector<uint8_t> values_encoded;

    ValueType vtype = classify_values(mat.x.data(), mat.nnz,
                                       mstats.max_value,
                                       mstats.all_integer, mstats.all_nonneg);

    if (vtype != ValueType::FLOAT64 && mstats.nnz > 0) {
        // Integer path: convert doubles to uint32, optionally predict, rANS encode
        std::vector<uint32_t> int_values = doubles_to_ints(mat.x.data(), mat.nnz);

        std::vector<uint32_t> value_symbols;
        uint32_t max_symbol;

        if (use_value_pred) {
            // Compute residuals
            value_symbols.resize(mat.nnz);
            int32_t max_residual = 0;
            int32_t min_residual = 0;

            for (uint32_t j = 0; j < mat.n; ++j) {
                uint32_t col_start = mat.p[j];
                uint32_t col_end = mat.p[j + 1];
                uint32_t col_nnz_ = col_end - col_start;

                for (uint32_t k = col_start; k < col_end; ++k) {
                    uint32_t row = mat.i[k];
                    uint32_t predicted = value_pred.predict(row, col_nnz_);
                    int32_t residual = static_cast<int32_t>(int_values[k])
                                     - static_cast<int32_t>(predicted);
                    // ZigZag encode to unsigned
                    uint32_t zz = static_cast<uint32_t>(
                        (residual << 1) ^ (residual >> 31));
                    value_symbols[k] = zz;
                    if (residual > max_residual) max_residual = residual;
                    if (residual < min_residual) min_residual = residual;
                }
            }
            // Max symbol for ZigZag: max of zigzag(min_residual), zigzag(max_residual)
            uint32_t zz_max_pos = static_cast<uint32_t>(
                (max_residual << 1) ^ (max_residual >> 31));
            uint32_t zz_max_neg = static_cast<uint32_t>(
                (min_residual << 1) ^ (min_residual >> 31));
            max_symbol = std::max(zz_max_pos, zz_max_neg);
        } else {
            value_symbols = std::move(int_values);
            max_symbol = mstats.max_value;
        }

        // Cap max_symbol for rANS table (use escape for very rare large symbols)
        // If max_symbol > 255, we'll use a two-pass approach:
        //   - Symbols 0..254 encode directly
        //   - Symbol 255 = escape, followed by VarInt of actual value
        constexpr uint32_t RANS_MAX_SYMBOL = 255;
        constexpr uint32_t ESCAPE_SYMBOL = 255;

        std::vector<uint8_t> overflow_data;  // VarInt-encoded overflow values

        if (max_symbol <= RANS_MAX_SYMBOL) {
            // All symbols fit in [0, 255] — direct rANS encoding
            RansTable table = rans::build_table(value_symbols.data(), mat.nnz, max_symbol);
            values_encoded = table.serialize();
            auto encoded = rans::encode_array(value_symbols.data(), mat.nnz, table);
            // Append encoded size + data
            uint32_t enc_size = static_cast<uint32_t>(encoded.size());
            values_encoded.push_back(static_cast<uint8_t>(enc_size & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_size >> 8) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_size >> 16) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_size >> 24) & 0xFF));
            values_encoded.insert(values_encoded.end(), encoded.begin(), encoded.end());
            // No overflow data
            uint32_t ov_size = 0;
            values_encoded.push_back(static_cast<uint8_t>(ov_size & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((ov_size >> 8) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((ov_size >> 16) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((ov_size >> 24) & 0xFF));
        } else {
            // Escape coding for large symbols
            std::vector<uint32_t> rans_symbols(mat.nnz);
            for (uint64_t k = 0; k < mat.nnz; ++k) {
                if (value_symbols[k] >= ESCAPE_SYMBOL) {
                    rans_symbols[k] = ESCAPE_SYMBOL;
                    varint::encode(value_symbols[k], overflow_data);
                } else {
                    rans_symbols[k] = value_symbols[k];
                }
            }

            RansTable table = rans::build_table(rans_symbols.data(), mat.nnz, RANS_MAX_SYMBOL);
            values_encoded = table.serialize();
            auto encoded = rans::encode_array(rans_symbols.data(), mat.nnz, table);
            uint32_t enc_size = static_cast<uint32_t>(encoded.size());
            values_encoded.push_back(static_cast<uint8_t>(enc_size & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_size >> 8) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_size >> 16) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_size >> 24) & 0xFF));
            values_encoded.insert(values_encoded.end(), encoded.begin(), encoded.end());
            // Overflow data
            uint32_t ov_size = static_cast<uint32_t>(overflow_data.size());
            values_encoded.push_back(static_cast<uint8_t>(ov_size & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((ov_size >> 8) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((ov_size >> 16) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((ov_size >> 24) & 0xFF));
            values_encoded.insert(values_encoded.end(),
                                  overflow_data.begin(), overflow_data.end());
        }
    } else if (mstats.nnz > 0) {
        // Float64 path: XOR-delta on raw uint64 representation,
        // then byte-shuffle + per-stream rANS.
        // XOR of adjacent doubles zeros out shared exponent bits,
        // making high bytes very compressible for clustered data.
        uint64_t N = mat.nnz;

        // Column-aware XOR-delta: within each column, XOR consecutive values.
        // This zeros out shared exponent bits between adjacent entries
        // in the same column, which is much better than flat XOR.
        std::vector<uint64_t> xor_vals(N);
        {
            const uint64_t* raw64 = reinterpret_cast<const uint64_t*>(mat.x.data());
            for (uint32_t j = 0; j < mat.n; ++j) {
                uint32_t col_start = mat.p[j];
                uint32_t col_end = mat.p[j + 1];
                if (col_start < col_end) {
                    xor_vals[col_start] = raw64[col_start]; // first in column: store raw
                    for (uint32_t k = col_start + 1; k < col_end; ++k) {
                        xor_vals[k] = raw64[k] ^ raw64[k - 1];
                    }
                }
            }
        }

        const uint8_t* raw = reinterpret_cast<const uint8_t*>(xor_vals.data());

        // Write stream count marker
        constexpr uint8_t N_STREAMS = 8;
        values_encoded.push_back(N_STREAMS);

        for (int s = 0; s < N_STREAMS; ++s) {
            // Extract byte s from each XOR'd value
            std::vector<uint32_t> stream(N);
            for (uint64_t k = 0; k < N; ++k) {
                stream[k] = raw[k * 8 + s];
            }

            // Build frequency table and rANS encode
            RansTable table = rans::build_table(stream.data(), N, 255);
            auto table_bytes = table.serialize();
            auto encoded = rans::encode_array(stream.data(), N, table);

            // Write: table_size(4) + table + enc_size(4) + encoded
            uint32_t table_sz = static_cast<uint32_t>(table_bytes.size());
            uint32_t enc_sz = static_cast<uint32_t>(encoded.size());

            values_encoded.push_back(static_cast<uint8_t>(table_sz & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((table_sz >> 8) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((table_sz >> 16) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((table_sz >> 24) & 0xFF));
            values_encoded.insert(values_encoded.end(),
                                  table_bytes.begin(), table_bytes.end());

            values_encoded.push_back(static_cast<uint8_t>(enc_sz & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_sz >> 8) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_sz >> 16) & 0xFF));
            values_encoded.push_back(static_cast<uint8_t>((enc_sz >> 24) & 0xFF));
            values_encoded.insert(values_encoded.end(),
                                  encoded.begin(), encoded.end());
        }
    }

    // ========================================================================
    // Phase 5: Assemble output
    // ========================================================================
    FileHeader header;
    header.m = mat.m;
    header.n = mat.n;
    header.nnz = mat.nnz;
    header.max_value = mstats.max_value;
    header.value_type = static_cast<uint8_t>(vtype);
    header.rice_block_size = static_cast<uint16_t>(cfg.rice_block_size);
    header.density_blocks = static_cast<uint16_t>(cfg.density_blocks);
    header.prng_seed = cfg.prng_seed;
    header.model_size = static_cast<uint32_t>(model_data.size());
    header.struct_size = static_cast<uint32_t>(struct_encoded.size());
    header.values_size = static_cast<uint32_t>(values_encoded.size());
    header.col_counts_size = static_cast<uint32_t>(col_counts_data.size());
    header.struct_k_size = static_cast<uint32_t>(struct_k_data.size());
    header.flags = 0;
    if (use_delta_pred)  header.flags |= FLAG_DELTA_PREDICTION;
    if (use_value_pred)  header.flags |= FLAG_VALUE_PREDICTION;
    if (mstats.all_integer && mstats.all_nonneg)
        header.flags |= FLAG_INTEGER_VALUES;

    // Compute total size
    size_t total = HEADER_SIZE
        + model_data.size()
        + col_counts_data.size()
        + struct_k_data.size()
        + struct_encoded.size()
        + values_encoded.size();

    std::vector<uint8_t> output;
    output.reserve(total);

    // Placeholder header (CRC computed after)
    auto header_bytes = header.serialize();
    output.insert(output.end(), header_bytes.begin(), header_bytes.end());

    // Sections
    output.insert(output.end(), model_data.begin(), model_data.end());
    output.insert(output.end(), col_counts_data.begin(), col_counts_data.end());
    output.insert(output.end(), struct_k_data.begin(), struct_k_data.end());
    output.insert(output.end(), struct_encoded.begin(), struct_encoded.end());
    output.insert(output.end(), values_encoded.begin(), values_encoded.end());

    // CRC32 over everything after header
    uint32_t crc = CRC32::compute(output.data() + HEADER_SIZE,
                                   output.size() - HEADER_SIZE);
    header.crc32 = crc;
    header_bytes = header.serialize();
    std::memcpy(output.data(), header_bytes.data(), HEADER_SIZE);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (stats_out) {
        stats_out->raw_size = mat.raw_size();
        stats_out->compressed_size = output.size();
        stats_out->header_size = HEADER_SIZE;
        stats_out->model_size = model_data.size();
        stats_out->structure_size = struct_k_data.size() + struct_encoded.size();
        stats_out->values_size = values_encoded.size();
        stats_out->compress_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    if (cfg.verbose) {
        SPZ_FPRINTF(stderr, "[sparsepress] Compressed: %zu → %zu bytes (%.2f× ratio)\n",
                     mat.raw_size(), output.size(),
                     static_cast<double>(mat.raw_size()) / output.size());
        SPZ_FPRINTF(stderr, "[sparsepress]   Header: %zu, Model: %zu, ColCounts: %zu\n",
                     (size_t)HEADER_SIZE, model_data.size(), col_counts_data.size());
        SPZ_FPRINTF(stderr, "[sparsepress]   Structure K: %zu, Structure: %zu, Values: %zu\n",
                     struct_k_data.size(), struct_encoded.size(), values_encoded.size());
        SPZ_FPRINTF(stderr, "[sparsepress]   Bits/nnz: %.3f (structure: %.3f, values: %.3f)\n",
                     static_cast<double>(output.size()) * 8.0 / mat.nnz,
                     static_cast<double>(struct_k_data.size() + struct_encoded.size()) * 8.0 / mat.nnz,
                     static_cast<double>(values_encoded.size()) * 8.0 / mat.nnz);
        SPZ_FPRINTF(stderr, "[sparsepress]   Compress time: %.1f ms\n",
                     std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    return output;
}

// ============================================================================
// Decompress back to CSC sparse matrix
// ============================================================================
inline CSCMatrix decompress(const std::vector<uint8_t>& data,
                             CompressStats* stats_out = nullptr) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (data.size() < HEADER_SIZE)
        throw std::runtime_error("Data too small to contain header");

    // ========================================================================
    // Phase 0: Read header
    // ========================================================================
    FileHeader header = FileHeader::deserialize(data.data());

    // Verify CRC
    uint32_t expected_crc = header.crc32;
    uint32_t actual_crc = CRC32::compute(data.data() + HEADER_SIZE,
                                          data.size() - HEADER_SIZE);
    if (expected_crc != actual_crc)
        throw std::runtime_error("CRC32 mismatch: data corrupted");

    bool use_delta_pred  = (header.flags & FLAG_DELTA_PREDICTION) != 0;
    bool use_value_pred  = (header.flags & FLAG_VALUE_PREDICTION) != 0;
    bool is_integer      = (header.flags & FLAG_INTEGER_VALUES) != 0;
    (void)is_integer;  // reserved for future per-type decode paths
    ValueType vtype      = static_cast<ValueType>(header.value_type);

    CSCMatrix mat(header.m, header.n, header.nnz);

    // ========================================================================
    // Phase 1: Read sections (pointer arithmetic)
    // ========================================================================
    const uint8_t* ptr = data.data() + HEADER_SIZE;

    // Model section
    const uint8_t* model_ptr = ptr;
    ptr += header.model_size;

    // Column counts section
    const uint8_t* col_counts_ptr = ptr;
    ptr += header.col_counts_size;

    // Structure k params section
    const uint8_t* struct_k_ptr = ptr;
    (void)struct_k_ptr;  // pointer used for section boundary; data read via struct_data_ptr
    ptr += header.struct_k_size;

    // Structure data section
    const uint8_t* struct_data_ptr = ptr;
    ptr += header.struct_size;

    // Values section
    const uint8_t* values_ptr = ptr;

    // ========================================================================
    // Phase 2: Decode models
    // ========================================================================
    const uint8_t* mp = model_ptr;
    DensityModel density_model;
    ValuePredictor value_pred;

    if (use_delta_pred) {
        density_model = DensityModel::deserialize(mp);
    }
    if (use_value_pred) {
        value_pred = ValuePredictor::deserialize(mp);
    }

    // ========================================================================
    // Phase 3: Decode column counts → reconstruct column pointers
    // ========================================================================
    {
        const uint8_t* cp = col_counts_ptr;
        mat.p[0] = 0;
        for (uint32_t j = 0; j < header.n; ++j) {
            uint32_t count = static_cast<uint32_t>(varint::decode(cp));
            mat.p[j + 1] = mat.p[j] + count;
        }
    }

    // ========================================================================
    // Phase 4: Decode structure (row index gaps) via rANS + escape
    // ========================================================================
    {
        // The struct_k_size field is 0 (no separate k data for rANS mode)
        // The structure data contains: rANS table + enc_size(4) + encoded +
        //                              overflow_size(4) + overflow
        const uint8_t* sp = struct_data_ptr;

        // Read rANS table
        RansTable struct_table = RansTable::deserialize(sp);

        // Read encoded data
        uint32_t enc_sz = static_cast<uint32_t>(sp[0])
                        | (static_cast<uint32_t>(sp[1]) << 8)
                        | (static_cast<uint32_t>(sp[2]) << 16)
                        | (static_cast<uint32_t>(sp[3]) << 24);
        sp += 4;

        std::vector<uint32_t> rans_syms(header.nnz);
        rans::decode_array(sp, enc_sz, rans_syms.data(), header.nnz, struct_table);
        sp += enc_sz;

        // Read overflow data
        uint32_t ov_sz = static_cast<uint32_t>(sp[0])
                       | (static_cast<uint32_t>(sp[1]) << 8)
                       | (static_cast<uint32_t>(sp[2]) << 16)
                       | (static_cast<uint32_t>(sp[3]) << 24);
        sp += 4;
        const uint8_t* ov_ptr = sp;

        // Reconstruct structure symbols with escape decoding
        constexpr uint32_t STRUCT_ESCAPE = 255;
        std::vector<uint32_t> structure_symbols(header.nnz);
        for (uint64_t k = 0; k < header.nnz; ++k) {
            if (rans_syms[k] == STRUCT_ESCAPE && ov_sz > 0) {
                structure_symbols[k] = static_cast<uint32_t>(varint::decode(ov_ptr));
            } else {
                structure_symbols[k] = rans_syms[k];
            }
        }

        // Undo delta prediction if applied
        std::vector<uint32_t> gaps(header.nnz);
        if (use_delta_pred) {
            for (uint32_t j = 0; j < header.n; ++j) {
                uint32_t col_start = mat.p[j];
                uint32_t col_end = mat.p[j + 1];
                uint32_t current_row = 0;
                for (uint32_t k = col_start; k < col_end; ++k) {
                    // Undo ZigZag
                    uint32_t zz = structure_symbols[k];
                    int32_t residual = static_cast<int32_t>((zz >> 1) ^ -(zz & 1));
                    // Add prediction
                    int32_t predicted = density_model.predict_gap(current_row);
                    int32_t gap_signed = residual + predicted;
                    // Gap must be non-negative
                    gaps[k] = static_cast<uint32_t>(std::max(0, gap_signed));
                    current_row += gaps[k] + 1;
                }
            }
        } else {
            gaps = std::move(structure_symbols);
        }

        // Reconstruct row indices from gaps
        delta::decode_all(gaps.data(), mat.p.data(), header.n, mat.i.data());
    }

    // ========================================================================
    // Phase 5: Decode values
    // ========================================================================
    if (vtype != ValueType::FLOAT64 && header.nnz > 0) {
        const uint8_t* vp = values_ptr;

        // Read rANS table
        RansTable table = RansTable::deserialize(vp);

        // Read encoded data size
        uint32_t enc_size = static_cast<uint32_t>(vp[0])
                          | (static_cast<uint32_t>(vp[1]) << 8)
                          | (static_cast<uint32_t>(vp[2]) << 16)
                          | (static_cast<uint32_t>(vp[3]) << 24);
        vp += 4;

        // Decode rANS symbols
        std::vector<uint32_t> rans_symbols(header.nnz);
        rans::decode_array(vp, enc_size, rans_symbols.data(), header.nnz, table);
        vp += enc_size;

        // Read overflow data size
        uint32_t ov_size = static_cast<uint32_t>(vp[0])
                         | (static_cast<uint32_t>(vp[1]) << 8)
                         | (static_cast<uint32_t>(vp[2]) << 16)
                         | (static_cast<uint32_t>(vp[3]) << 24);
        vp += 4;

        // Decode overflow values
        const uint8_t* ov_ptr = vp;
        std::vector<uint32_t> value_symbols(header.nnz);

        constexpr uint32_t ESCAPE_SYMBOL = 255;
        for (uint64_t k = 0; k < header.nnz; ++k) {
            if (rans_symbols[k] == ESCAPE_SYMBOL && ov_size > 0) {
                value_symbols[k] = static_cast<uint32_t>(varint::decode(ov_ptr));
            } else {
                value_symbols[k] = rans_symbols[k];
            }
        }

        // Undo value prediction if applied
        if (use_value_pred) {
            for (uint32_t j = 0; j < header.n; ++j) {
                uint32_t col_start = mat.p[j];
                uint32_t col_end = mat.p[j + 1];
                uint32_t col_nnz_ = col_end - col_start;

                for (uint32_t k = col_start; k < col_end; ++k) {
                    // Undo ZigZag
                    uint32_t zz = value_symbols[k];
                    int32_t residual = static_cast<int32_t>((zz >> 1) ^ -(zz & 1));
                    // Add prediction
                    uint32_t predicted = value_pred.predict(mat.i[k], col_nnz_);
                    int32_t actual = residual + static_cast<int32_t>(predicted);
                    mat.x[k] = static_cast<double>(std::max(0, actual));
                }
            }
        } else {
            // Direct conversion
            for (uint64_t k = 0; k < header.nnz; ++k) {
                mat.x[k] = static_cast<double>(value_symbols[k]);
            }
        }
    } else if (header.nnz > 0) {
        // Float64 path: byte-unshuffle + XOR-undelta
        const uint8_t* vp = values_ptr;
        uint64_t N = header.nnz;

        uint8_t n_streams = *vp++;

        // Decode byte streams into XOR'd representation
        std::vector<uint64_t> xor_vals(N, 0);
        uint8_t* raw = reinterpret_cast<uint8_t*>(xor_vals.data());

        for (int s = 0; s < n_streams; ++s) {
            // Read table_size + table
            uint32_t table_sz = static_cast<uint32_t>(vp[0])
                              | (static_cast<uint32_t>(vp[1]) << 8)
                              | (static_cast<uint32_t>(vp[2]) << 16)
                              | (static_cast<uint32_t>(vp[3]) << 24);
            (void)table_sz;  // size embedded in table; deserialize advances vp
            vp += 4;
            RansTable table = RansTable::deserialize(vp);

            // Read enc_size + encoded
            uint32_t enc_sz = static_cast<uint32_t>(vp[0])
                            | (static_cast<uint32_t>(vp[1]) << 8)
                            | (static_cast<uint32_t>(vp[2]) << 16)
                            | (static_cast<uint32_t>(vp[3]) << 24);
            vp += 4;

            std::vector<uint32_t> stream(N);
            rans::decode_array(vp, enc_sz, stream.data(), N, table);
            vp += enc_sz;

            // Scatter byte s back
            for (uint64_t k = 0; k < N; ++k) {
                raw[k * 8 + s] = static_cast<uint8_t>(stream[k]);
            }
        }

        // Undo column-aware XOR-delta
        uint64_t* out64 = reinterpret_cast<uint64_t*>(mat.x.data());
        for (uint32_t j = 0; j < header.n; ++j) {
            uint32_t col_start = mat.p[j];
            uint32_t col_end = mat.p[j + 1];
            if (col_start < col_end) {
                out64[col_start] = xor_vals[col_start]; // first in column: raw
                for (uint32_t k = col_start + 1; k < col_end; ++k) {
                    out64[k] = xor_vals[k] ^ out64[k - 1];
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (stats_out) {
        stats_out->raw_size = mat.raw_size();
        stats_out->compressed_size = data.size();
        stats_out->decompress_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return mat;
}

// ============================================================================
// File I/O helpers
// ============================================================================
inline void write_compressed(const std::string& path,
                              const std::vector<uint8_t>& data) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
}

inline std::vector<uint8_t> read_compressed(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open for reading: " + path);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(size);
    if (fread(data.data(), 1, size, f) != static_cast<size_t>(size)) {
        fclose(f);
        throw std::runtime_error("Failed to read file: " + path);
    }
    fclose(f);
    return data;
}

} // namespace streampress

#endif // SPARSEPRESS_HPP
