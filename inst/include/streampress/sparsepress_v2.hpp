/**
 * @file sparsepress_v2.hpp
 * @brief SparsePress v2 chunked encoder/decoder.
 *
 * Implements 256-column chunked compression with:
 *   - Per-chunk gap encoding (delta + rANS + escape)
 *   - Multiple value types (uint8/16/32, fp32, fp16, quant8, fp64)
 *   - Byte-shuffle + rANS for float values
 *   - Optional row-sorting for better compression
 *   - Metadata section (dimnames, row permutation)
 *   - Parallel decode via OpenMP
 *   - Partial column reads
 */

#ifndef SPARSEPRESS_V2_HPP
#define SPARSEPRESS_V2_HPP

#include <streampress/format/header_v2.hpp>
#include <streampress/format/obs_var_table.hpp>
#include <streampress/core/types.hpp>
#include <streampress/codec/varint.hpp>
#include <streampress/codec/rans.hpp>
#include <streampress/format/checksum.hpp>
#include <streampress/core/checks.hpp>

#include <atomic>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstdio>

// Logging support
#if defined(SPARSEPRESS_USE_R)
#include <R_ext/Print.h>
#define SPZ2_LOG(verbose, level, ...) do { \
    if (verbose >= level) REprintf(__VA_ARGS__); \
} while(0)
#else
#define SPZ2_LOG(verbose, level, ...) do { \
    if (verbose >= level) std::fprintf(stderr, __VA_ARGS__); \
} while(0)
#endif

namespace streampress {
namespace v2 {

// ============================================================================
// Compress config
// ============================================================================

struct CompressConfig_v2 {
    std::string precision = "auto";       // "auto", "fp32", "fp16", "quant8", "fp64"
    bool row_sort = false;                // Sort rows by nnz
    bool include_transpose = false;       // Store CSC(Aᵀ) in file
    int  verbose = 0;                     // 0=silent, 1=summary, 2=detailed, 3=debug
    uint32_t chunk_cols = DEFAULT_CHUNK_COLS;
    std::vector<uint8_t> obs_buf;         // Pre-serialized obs table (empty = no obs)
    std::vector<uint8_t> var_buf;         // Pre-serialized var table (empty = no var)
};

struct CompressStats_v2 {
    size_t raw_size = 0;
    size_t compressed_size = 0;
    double compress_time_ms = 0;
    uint32_t num_chunks = 0;
    double ratio() const { return raw_size > 0 ? static_cast<double>(raw_size) / compressed_size : 0; }
};

// ============================================================================
// Internal: encode a single chunk's gaps (delta + rANS + escape)
// ============================================================================

namespace detail {

inline std::vector<uint8_t> encode_gaps_chunk(
    const uint32_t* col_ptrs,   // local col ptrs (num_cols+1 entries, starting at 0)
    const uint32_t* row_indices,
    uint32_t num_cols, uint32_t nnz_chunk)
{
    std::vector<uint8_t> result;

    // First: VarInt-encode column counts
    for (uint32_t j = 0; j < num_cols; ++j) {
        uint32_t col_nnz = col_ptrs[j + 1] - col_ptrs[j];
        varint::encode(col_nnz, result);
    }
    // Store col_counts section size for decoder
    uint32_t cc_size = static_cast<uint32_t>(result.size());

    if (nnz_chunk == 0) return result;

    // Compute delta gaps (per-column reset)
    std::vector<uint32_t> gaps(nnz_chunk);
    uint32_t gap_idx = 0;
    for (uint32_t j = 0; j < num_cols; ++j) {
        uint32_t prev = 0;
        for (uint32_t k = col_ptrs[j]; k < col_ptrs[j + 1]; ++k) {
            uint32_t row = row_indices[k];
            gaps[gap_idx++] = row - prev;
            prev = row + 1;
        }
    }

    // rANS encode with escape for values >= 255
    constexpr uint32_t MAX_SYM = 255;
    constexpr uint32_t ESCAPE = 255;

    std::vector<uint32_t> rans_syms(nnz_chunk);
    std::vector<uint8_t> overflow;
    uint32_t max_seen = 0;

    for (uint32_t i = 0; i < nnz_chunk; ++i) {
        if (gaps[i] >= ESCAPE) {
            rans_syms[i] = ESCAPE;
            varint::encode(gaps[i], overflow);
        } else {
            rans_syms[i] = gaps[i];
            if (gaps[i] > max_seen) max_seen = gaps[i];
        }
    }
    if (!overflow.empty()) max_seen = MAX_SYM;

    RansTable table = rans::build_table(rans_syms.data(), nnz_chunk, max_seen);
    auto table_bytes = table.serialize();
    auto encoded = rans::encode_array(rans_syms.data(), nnz_chunk, table);

    // Append: [cc_size(4)][table][enc_size(4)][encoded][overflow_size(4)][overflow]
    // Prepend cc_size so decoder knows where col counts end
    std::vector<uint8_t> prefix;
    prefix.push_back(cc_size & 0xFF);
    prefix.push_back((cc_size >> 8) & 0xFF);
    prefix.push_back((cc_size >> 16) & 0xFF);
    prefix.push_back((cc_size >> 24) & 0xFF);
    // Insert at beginning (after col counts already in result)
    // Actually restructure: [cc_size(4)][col_counts_varint][table][enc_sz(4)][encoded][ov_sz(4)][overflow]
    std::vector<uint8_t> final_result;
    final_result.reserve(4 + result.size() + table_bytes.size() + 4 + encoded.size() + 4 + overflow.size());

    // cc_size
    final_result.push_back(cc_size & 0xFF);
    final_result.push_back((cc_size >> 8) & 0xFF);
    final_result.push_back((cc_size >> 16) & 0xFF);
    final_result.push_back((cc_size >> 24) & 0xFF);

    // col counts (VarInt)
    final_result.insert(final_result.end(), result.begin(), result.end());

    // rANS table
    final_result.insert(final_result.end(), table_bytes.begin(), table_bytes.end());

    // encoded data size + data
    uint32_t enc_sz = static_cast<uint32_t>(encoded.size());
    final_result.push_back(enc_sz & 0xFF);
    final_result.push_back((enc_sz >> 8) & 0xFF);
    final_result.push_back((enc_sz >> 16) & 0xFF);
    final_result.push_back((enc_sz >> 24) & 0xFF);
    final_result.insert(final_result.end(), encoded.begin(), encoded.end());

    // overflow size + data
    uint32_t ov_sz = static_cast<uint32_t>(overflow.size());
    final_result.push_back(ov_sz & 0xFF);
    final_result.push_back((ov_sz >> 8) & 0xFF);
    final_result.push_back((ov_sz >> 16) & 0xFF);
    final_result.push_back((ov_sz >> 24) & 0xFF);
    final_result.insert(final_result.end(), overflow.begin(), overflow.end());

    return final_result;
}

// ============================================================================
// Internal: encode values for a chunk
// ============================================================================

// Integer values: rANS + escape
inline std::vector<uint8_t> encode_values_int_chunk(
    const uint32_t* int_values, uint32_t count)
{
    if (count == 0) return {};

    constexpr uint32_t MAX_SYM = 255;
    constexpr uint32_t ESCAPE = 255;

    std::vector<uint32_t> syms(count);
    std::vector<uint8_t> overflow;
    uint32_t max_seen = 0;

    for (uint32_t i = 0; i < count; ++i) {
        if (int_values[i] >= ESCAPE) {
            syms[i] = ESCAPE;
            varint::encode(int_values[i], overflow);
        } else {
            syms[i] = int_values[i];
            if (int_values[i] > max_seen) max_seen = int_values[i];
        }
    }
    if (!overflow.empty()) max_seen = MAX_SYM;

    RansTable table = rans::build_table(syms.data(), count, max_seen);
    auto table_bytes = table.serialize();
    auto encoded = rans::encode_array(syms.data(), count, table);

    std::vector<uint8_t> result;
    result.insert(result.end(), table_bytes.begin(), table_bytes.end());

    uint32_t enc_sz = static_cast<uint32_t>(encoded.size());
    result.push_back(enc_sz & 0xFF);
    result.push_back((enc_sz >> 8) & 0xFF);
    result.push_back((enc_sz >> 16) & 0xFF);
    result.push_back((enc_sz >> 24) & 0xFF);
    result.insert(result.end(), encoded.begin(), encoded.end());

    uint32_t ov_sz = static_cast<uint32_t>(overflow.size());
    result.push_back(ov_sz & 0xFF);
    result.push_back((ov_sz >> 8) & 0xFF);
    result.push_back((ov_sz >> 16) & 0xFF);
    result.push_back((ov_sz >> 24) & 0xFF);
    result.insert(result.end(), overflow.begin(), overflow.end());

    return result;
}

// Float32 values: byte-shuffle (4 streams) + per-stream rANS
inline std::vector<uint8_t> encode_values_f32_chunk(
    const float* values, uint32_t count)
{
    if (count == 0) return {};

    constexpr uint8_t N_STREAMS = 4;
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(values);

    std::vector<uint8_t> result;
    result.push_back(N_STREAMS);

    for (int s = 0; s < N_STREAMS; ++s) {
        std::vector<uint32_t> stream(count);
        for (uint32_t k = 0; k < count; ++k)
            stream[k] = raw[k * 4 + s];

        RansTable table = rans::build_table(stream.data(), count, 255);
        auto table_bytes = table.serialize();
        auto encoded = rans::encode_array(stream.data(), count, table);

        uint32_t table_sz = static_cast<uint32_t>(table_bytes.size());
        uint32_t enc_sz = static_cast<uint32_t>(encoded.size());

        result.push_back(table_sz & 0xFF);
        result.push_back((table_sz >> 8) & 0xFF);
        result.push_back((table_sz >> 16) & 0xFF);
        result.push_back((table_sz >> 24) & 0xFF);
        result.insert(result.end(), table_bytes.begin(), table_bytes.end());

        result.push_back(enc_sz & 0xFF);
        result.push_back((enc_sz >> 8) & 0xFF);
        result.push_back((enc_sz >> 16) & 0xFF);
        result.push_back((enc_sz >> 24) & 0xFF);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }

    return result;
}

// Float64 values: byte-shuffle (8 streams) + per-stream rANS
inline std::vector<uint8_t> encode_values_f64_chunk(
    const double* values, uint32_t count)
{
    if (count == 0) return {};

    constexpr uint8_t N_STREAMS = 8;
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(values);

    std::vector<uint8_t> result;
    result.push_back(N_STREAMS);

    for (int s = 0; s < N_STREAMS; ++s) {
        std::vector<uint32_t> stream(count);
        for (uint32_t k = 0; k < count; ++k)
            stream[k] = raw[k * 8 + s];

        RansTable table = rans::build_table(stream.data(), count, 255);
        auto table_bytes = table.serialize();
        auto encoded = rans::encode_array(stream.data(), count, table);

        uint32_t table_sz = static_cast<uint32_t>(table_bytes.size());
        uint32_t enc_sz = static_cast<uint32_t>(encoded.size());

        result.push_back(table_sz & 0xFF);
        result.push_back((table_sz >> 8) & 0xFF);
        result.push_back((table_sz >> 16) & 0xFF);
        result.push_back((table_sz >> 24) & 0xFF);
        result.insert(result.end(), table_bytes.begin(), table_bytes.end());

        result.push_back(enc_sz & 0xFF);
        result.push_back((enc_sz >> 8) & 0xFF);
        result.push_back((enc_sz >> 16) & 0xFF);
        result.push_back((enc_sz >> 24) & 0xFF);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }

    return result;
}

// Float16 values: convert fp32→fp16 + byte-shuffle (2 streams) + rANS
inline std::vector<uint8_t> encode_values_f16_chunk(
    const float* values, uint32_t count)
{
    if (count == 0) return {};

    // Convert to fp16
    std::vector<uint16_t> fp16_vals(count);
    for (uint32_t k = 0; k < count; ++k)
        fp16_vals[k] = float_to_half(values[k]);

    constexpr uint8_t N_STREAMS = 2;
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(fp16_vals.data());

    std::vector<uint8_t> result;
    result.push_back(N_STREAMS);

    for (int s = 0; s < N_STREAMS; ++s) {
        std::vector<uint32_t> stream(count);
        for (uint32_t k = 0; k < count; ++k)
            stream[k] = raw[k * 2 + s];

        RansTable table = rans::build_table(stream.data(), count, 255);
        auto table_bytes = table.serialize();
        auto encoded = rans::encode_array(stream.data(), count, table);

        uint32_t table_sz = static_cast<uint32_t>(table_bytes.size());
        uint32_t enc_sz = static_cast<uint32_t>(encoded.size());

        result.push_back(table_sz & 0xFF);
        result.push_back((table_sz >> 8) & 0xFF);
        result.push_back((table_sz >> 16) & 0xFF);
        result.push_back((table_sz >> 24) & 0xFF);
        result.insert(result.end(), table_bytes.begin(), table_bytes.end());

        result.push_back(enc_sz & 0xFF);
        result.push_back((enc_sz >> 8) & 0xFF);
        result.push_back((enc_sz >> 16) & 0xFF);
        result.push_back((enc_sz >> 24) & 0xFF);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }

    return result;
}

// Quant8 values: affine quantize + rANS
inline std::vector<uint8_t> encode_values_quant8_chunk(
    const float* values, uint32_t count,
    float& out_scale, float& out_offset)
{
    if (count == 0) { out_scale = 1.0f; out_offset = 0.0f; return {}; }

    // Find min/max
    float vmin = values[0], vmax = values[0];
    for (uint32_t k = 1; k < count; ++k) {
        if (values[k] < vmin) vmin = values[k];
        if (values[k] > vmax) vmax = values[k];
    }

    out_offset = vmin;
    out_scale = (vmax > vmin) ? (vmax - vmin) / 255.0f : 1.0f;

    std::vector<uint32_t> quantized(count);
    for (uint32_t k = 0; k < count; ++k) {
        float q = (values[k] - out_offset) / out_scale;
        quantized[k] = static_cast<uint32_t>(std::min(255.0f, std::max(0.0f, q + 0.5f)));
    }

    // rANS encode (all symbols in [0, 255])
    RansTable table = rans::build_table(quantized.data(), count, 255);
    auto table_bytes = table.serialize();
    auto encoded = rans::encode_array(quantized.data(), count, table);

    std::vector<uint8_t> result;
    result.insert(result.end(), table_bytes.begin(), table_bytes.end());

    uint32_t enc_sz = static_cast<uint32_t>(encoded.size());
    result.push_back(enc_sz & 0xFF);
    result.push_back((enc_sz >> 8) & 0xFF);
    result.push_back((enc_sz >> 16) & 0xFF);
    result.push_back((enc_sz >> 24) & 0xFF);
    result.insert(result.end(), encoded.begin(), encoded.end());

    return result;
}

// ============================================================================
// Decode helpers (mirror encoders)
// ============================================================================

// Read a uint32_t from buffer
inline uint32_t read_u32(const uint8_t* ptr) {
    uint32_t v;
    std::memcpy(&v, ptr, 4);
    return v;
}

// Decode rANS + escape encoded data
inline std::vector<uint32_t> decode_rans_escape(
    const uint8_t* data, size_t data_size, uint32_t expected_count)
{
    if (data_size == 0 || expected_count == 0)
        return std::vector<uint32_t>(expected_count, 0);

    // Every read below is bounded by `end`; a short or inconsistent stream
    // throws. The result always holds exactly expected_count symbols.
    const uint8_t* const end = data + data_size;
    const uint8_t* cur = data;

    // Deserialize table
    RansTable table = RansTable::deserialize(cur, end);

    // Read encoded size
    if (static_cast<size_t>(end - cur) < 4)
        spz_corrupt("rANS stream is truncated before its size field");
    uint32_t enc_sz = read_u32(cur); cur += 4;
    if (enc_sz > static_cast<size_t>(end - cur))
        spz_corrupt("rANS payload extends past its stream");

    // Decode rANS
    std::vector<uint32_t> symbols(expected_count);
    rans::decode_array(cur, enc_sz, symbols.data(), expected_count, table);
    cur += enc_sz;

    // Read overflow size (absent in streams without an escape section)
    if (static_cast<size_t>(end - cur) < 4) return symbols;
    uint32_t ov_sz = read_u32(cur); cur += 4;
    if (ov_sz > static_cast<size_t>(end - cur))
        spz_corrupt("escape section extends past its stream");

    // Apply escapes
    if (ov_sz > 0) {
        const uint8_t* ov_scan = cur;
        const uint8_t* const ov_end = cur + ov_sz;
        for (uint32_t i = 0; i < expected_count; ++i) {
            if (symbols[i] == 255) {
                symbols[i] = static_cast<uint32_t>(
                    varint::decode(ov_scan, ov_end));
            }
        }
    }

    return symbols;
}

// Decode byte-shuffled float values (N streams)
inline void decode_values_byteshuffle(
    const uint8_t* data, size_t data_size, uint32_t count,
    void* output, uint32_t bytes_per_value)
{
    if (data_size == 0 || count == 0) return;

    const uint8_t* const end = data + data_size;
    const uint8_t* cur = data;
    uint8_t n_streams = *cur++;
    // One stream per byte of the value type; more would write past `output`.
    if (n_streams > bytes_per_value)
        spz_corrupt("value stream count does not match the value type");
    uint8_t* out = static_cast<uint8_t*>(output);

    for (int s = 0; s < n_streams; ++s) {
        if (static_cast<size_t>(end - cur) < 4)
            spz_corrupt("value stream is truncated");
        uint32_t table_sz = read_u32(cur); cur += 4;
        if (table_sz > static_cast<size_t>(end - cur))
            spz_corrupt("value stream table extends past its stream");

        const uint8_t* tbl_ptr2 = cur;
        RansTable table = RansTable::deserialize(tbl_ptr2, cur + table_sz);
        cur += table_sz;

        if (static_cast<size_t>(end - cur) < 4)
            spz_corrupt("value stream is truncated");
        uint32_t enc_sz = read_u32(cur); cur += 4;
        if (enc_sz > static_cast<size_t>(end - cur))
            spz_corrupt("value stream payload extends past its stream");

        std::vector<uint32_t> stream(count);
        rans::decode_array(cur, enc_sz, stream.data(), count, table);
        cur += enc_sz;

        // De-shuffle: place each byte back
        for (uint32_t k = 0; k < count; ++k) {
            out[static_cast<size_t>(k) * bytes_per_value + s] =
                static_cast<uint8_t>(stream[k]);
        }
    }
}

// ============================================================================
// Validation of layout fields read from the file
// ============================================================================

/// Header fields every reader relies on. `data_size` is the number of bytes
/// in the caller's buffer. Streaming readers pass a prefix of the file
/// (header + chunk index + the chunks they need), so only the chunk index
/// itself must be present here; chunk streams are checked when decoded.
inline void validate_header_v2(const FileHeader_v2& hdr, size_t data_size) {
    if (hdr.m > 0x7FFFFFFFu || hdr.n > 0x7FFFFFFFu)
        spz_corrupt("matrix dimensions exceed 2^31 - 1");
    if (hdr.nnz > 0xFFFFFFFFull)
        spz_corrupt("more than 2^32 - 1 non-zeros");
    if (hdr.nnz > static_cast<uint64_t>(hdr.m) * hdr.n)
        spz_corrupt("more non-zeros than matrix cells");
    require_range(hdr.chunk_index_offset,
                  static_cast<uint64_t>(hdr.num_chunks) * sizeof(ChunkDescriptor_v2),
                  data_size, "chunk index lies outside the file");
    if (hdr.data_offset > data_size)
        spz_corrupt("data section lies outside the file");
}

/// A chunk index must tile the columns [0, n_cols) in order, and its
/// non-zero counts must be consistent with the matrix. Called before any
/// output is allocated or written.
inline void validate_chunk_index(const std::vector<ChunkDescriptor_v2>& descs,
                                 uint64_t n_cols, uint64_t n_rows,
                                 bool check_nnz, uint64_t total_nnz) {
    uint64_t col = 0, nnz = 0;
    for (const auto& d : descs) {
        if (d.col_start != col)
            spz_corrupt("chunk index is not contiguous");
        col += d.num_cols;
        if (col > n_cols)
            spz_corrupt("chunk index covers more columns than the matrix has");
        if (static_cast<uint64_t>(d.nnz) >
            static_cast<uint64_t>(d.num_cols) * n_rows)
            spz_corrupt("chunk holds more non-zeros than cells");
        if (d.nnz > 0 && d.stream_size[STREAM_GAPS] < 4)
            spz_corrupt("chunk with non-zeros has no structure stream");
        nnz += d.nnz;
    }
    if (col != n_cols)
        spz_corrupt("chunk index does not cover every column");
    if (check_nnz && nnz != total_nnz)
        spz_corrupt("chunk non-zero counts do not sum to the header total");
    if (nnz > 0xFFFFFFFFull)
        spz_corrupt("more than 2^32 - 1 non-zeros");
}

/// Both streams of a chunk must lie within the `avail` bytes that follow the
/// base of the data section in the caller's buffer.
inline void require_chunk_streams(const ChunkDescriptor_v2& d, uint64_t avail) {
    require_range(d.stream_offset[STREAM_GAPS], d.stream_size[STREAM_GAPS],
                  avail, "chunk structure stream lies outside the file");
    require_range(d.stream_offset[STREAM_VALUES], d.stream_size[STREAM_VALUES],
                  avail, "chunk value stream lies outside the file");
}

/// One per-column count from the structure stream.
inline uint32_t read_col_count(const uint8_t*& ptr, const uint8_t* end,
                               uint32_t chunk_nnz) {
    const uint64_t c = varint::decode(ptr, end);
    if (c > chunk_nnz)
        spz_corrupt("column count exceeds the chunk's non-zeros");
    return static_cast<uint32_t>(c);
}

/// Column counts of a chunk must add up to its non-zero count.
inline void check_col_counts(const std::vector<uint32_t>& counts,
                             uint32_t chunk_nnz) {
    uint64_t total = 0;
    for (uint32_t c : counts) total += c;
    if (total != chunk_nnz)
        spz_corrupt("column counts do not sum to the chunk's non-zeros");
}

/// Row index from the previous row and a decoded gap, checked against the
/// number of rows.
inline uint32_t next_row(uint32_t prev_row, uint32_t gap, uint32_t n_rows) {
    const uint64_t r = static_cast<uint64_t>(prev_row) + gap;
    if (r >= n_rows) spz_corrupt("row index out of range");
    return static_cast<uint32_t>(r);
}

/// Whole-file checks for full reads: the footer must be present, and the
/// CRC32 it records (when non-zero) must match the bytes before it.
inline void verify_footer_v2(const uint8_t* data, size_t data_size,
                             bool check_crc) {
    if (data_size < HEADER_SIZE_V2 + FOOTER_SIZE)
        spz_corrupt("file is shorter than its header and footer");
    Footer_v2 footer = Footer_v2::deserialize(data + data_size - FOOTER_SIZE);
    if (check_crc && footer.file_crc32 != 0) {
        const uint32_t actual = CRC32::compute(data, data_size - FOOTER_SIZE);
        if (actual != footer.file_crc32)
            spz_corrupt("CRC32 mismatch");
    }
}

} // namespace detail

// ============================================================================
// Public API: compress_v2
// ============================================================================

inline std::vector<uint8_t> compress_v2(
    const CSCMatrix& mat,              // v1 CSCMatrix (double values)
    const CompressConfig_v2& cfg = {},
    CompressStats_v2* stats_out = nullptr)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const uint32_t m = mat.m;
    const uint32_t n = mat.n;
    const uint64_t nnz = mat.nnz;

    // Determine value type
    ValueType_v2 vtype = classify_values_v2(mat.x.data(), nnz, cfg.precision);

    SPZ2_LOG(cfg.verbose, 1,
        "[spz] Compressing v2: %u × %u, nnz=%lu (%.1f%%), type=%s, chunks=%u cols\n",
        m, n, (unsigned long)nnz,
        (m > 0 && n > 0) ? 100.0 * nnz / ((double)m * n) : 0.0,
        value_type_name(vtype), cfg.chunk_cols);

    // Optionally sort rows
    RowSortResult row_sort;
    std::vector<uint32_t> sorted_rows;
    if (cfg.row_sort && nnz > 0) {
        row_sort = compute_row_sort(mat.p.data(), mat.i.data(), m, n, nnz);
        sorted_rows.resize(nnz);
        for (uint64_t k = 0; k < nnz; ++k)
            sorted_rows[k] = row_sort.perm[mat.i[k]];
        SPZ2_LOG(cfg.verbose, 2, "[spz] Row sorting applied\n");
    }

    const uint32_t* row_idx = cfg.row_sort ? sorted_rows.data() : mat.i.data();

    // Convert values to target type
    std::vector<float> f32_values;
    if (vtype == ValueType_v2::FLOAT32 || vtype == ValueType_v2::FLOAT16 ||
        vtype == ValueType_v2::QUANT8) {
        f32_values.resize(nnz);
        for (uint64_t k = 0; k < nnz; ++k)
            f32_values[k] = static_cast<float>(mat.x[k]);
    }

    std::vector<uint32_t> int_values;
    if (vtype == ValueType_v2::UINT8 || vtype == ValueType_v2::UINT16 ||
        vtype == ValueType_v2::UINT32) {
        int_values.resize(nnz);
        for (uint64_t k = 0; k < nnz; ++k)
            int_values[k] = static_cast<uint32_t>(mat.x[k]);
    }

    // Compute chunk boundaries
    uint32_t num_chunks = (n + cfg.chunk_cols - 1) / cfg.chunk_cols;

    // Encode each chunk
    struct ChunkData {
        std::vector<uint8_t> gap_data;
        std::vector<uint8_t> value_data;
        ChunkDescriptor_v2 desc;
    };
    std::vector<ChunkData> chunks(num_chunks);

    // Parallel encode
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(num_chunks > 4)
    #endif
    for (uint32_t c = 0; c < num_chunks; ++c) {
        uint32_t col_start = c * cfg.chunk_cols;
        uint32_t col_end = std::min(col_start + cfg.chunk_cols, n);
        uint32_t nc = col_end - col_start;

        // Local column pointers (rebased to 0)
        std::vector<uint32_t> local_p(nc + 1);
        uint32_t chunk_nnz_start = mat.p[col_start];
        for (uint32_t j = 0; j <= nc; ++j)
            local_p[j] = mat.p[col_start + j] - chunk_nnz_start;
        uint32_t chunk_nnz = local_p[nc];

        chunks[c].desc.col_start = col_start;
        chunks[c].desc.num_cols = nc;
        chunks[c].desc.nnz = chunk_nnz;

        // Encode gaps
        chunks[c].gap_data = detail::encode_gaps_chunk(
            local_p.data(), row_idx + chunk_nnz_start, nc, chunk_nnz);

        // Encode values
        switch (vtype) {
            case ValueType_v2::UINT8:
            case ValueType_v2::UINT16:
            case ValueType_v2::UINT32:
                chunks[c].value_data = detail::encode_values_int_chunk(
                    int_values.data() + chunk_nnz_start, chunk_nnz);
                break;
            case ValueType_v2::FLOAT32:
                chunks[c].value_data = detail::encode_values_f32_chunk(
                    f32_values.data() + chunk_nnz_start, chunk_nnz);
                break;
            case ValueType_v2::FLOAT16:
                chunks[c].value_data = detail::encode_values_f16_chunk(
                    f32_values.data() + chunk_nnz_start, chunk_nnz);
                break;
            case ValueType_v2::QUANT8:
                chunks[c].value_data = detail::encode_values_quant8_chunk(
                    f32_values.data() + chunk_nnz_start, chunk_nnz,
                    chunks[c].desc.quant_scale, chunks[c].desc.quant_offset);
                break;
            case ValueType_v2::FLOAT64:
                chunks[c].value_data = detail::encode_values_f64_chunk(
                    mat.x.data() + chunk_nnz_start, chunk_nnz);
                break;
        }

        SPZ2_LOG(cfg.verbose, 3,
            "[spz] Chunk %u: cols %u-%u, nnz=%u, gap=%zu B, val=%zu B\n",
            c, col_start, col_end - 1, chunk_nnz,
            chunks[c].gap_data.size(), chunks[c].value_data.size());
    }

    // Compute data layout
    // Data section: all chunk gap and value streams concatenated
    uint64_t data_size = 0;
    for (uint32_t c = 0; c < num_chunks; ++c) {
        chunks[c].desc.stream_offset[STREAM_GAPS] = static_cast<uint32_t>(data_size);
        chunks[c].desc.stream_size[STREAM_GAPS] = static_cast<uint32_t>(chunks[c].gap_data.size());
        data_size += chunks[c].gap_data.size();

        chunks[c].desc.stream_offset[STREAM_VALUES] = static_cast<uint32_t>(data_size);
        chunks[c].desc.stream_size[STREAM_VALUES] = static_cast<uint32_t>(chunks[c].value_data.size());
        data_size += chunks[c].value_data.size();

        chunks[c].desc.decoded_gap_bytes = chunks[c].desc.nnz * sizeof(uint32_t);
        chunks[c].desc.decoded_value_bytes = chunks[c].desc.nnz *
            value_type_bytes(vtype);
    }

    // Build metadata
    Metadata meta;
    if (cfg.row_sort) {
        meta.set_row_permutation(row_sort.perm);
    }

    auto meta_bytes = meta.serialize();

    // Build header
    FileHeader_v2 hdr;
    hdr.m = m;
    hdr.n = n;
    hdr.nnz = nnz;
    hdr.chunk_cols = cfg.chunk_cols;
    hdr.num_chunks = num_chunks;
    hdr.num_tables = 0;  // tables embedded in chunk data
    hdr.table_log = 11;
    hdr.value_type = static_cast<uint8_t>(vtype);
    hdr.compression_level = static_cast<uint8_t>(CompressionLevel::DEFAULT);
    hdr.row_sorted = cfg.row_sort ? 1 : 0;
    hdr.col_sorted = 0;
    hdr.most_common_value = 0;
    hdr.density = (m > 0 && n > 0) ? static_cast<float>(nnz) / (static_cast<float>(m) * n) : 0.0f;
    hdr.max_value = 0;

    // Compute section offsets
    uint64_t chunk_index_size = static_cast<uint64_t>(num_chunks) * sizeof(ChunkDescriptor_v2);
    hdr.chunk_index_offset = HEADER_SIZE_V2;
    hdr.tables_offset = hdr.chunk_index_offset + chunk_index_size;  // no separate tables
    hdr.data_offset = hdr.tables_offset;  // tables_size = 0
    hdr.transpose_offset = 0;

    // Optionally encode transpose (CSC(Aᵀ))
    std::vector<uint8_t> transpose_data;
    uint32_t num_transpose_chunks = 0;
    std::vector<ChunkDescriptor_v2> transpose_descs;

    if (cfg.include_transpose && nnz > 0) {
        SPZ2_LOG(cfg.verbose, 1, "[spz] Building transpose for pre-computed CSC(Aᵀ)...\n");

        // Build CSC(Aᵀ) = CSR of original matrix
        // In CSC(Aᵀ), columns are original rows, rows are original columns
        uint32_t t_n = m;  // transpose columns = original rows
        uint32_t t_m = n;  // transpose rows = original columns
        (void)t_m;  // documented for symmetry with t_n; not referenced below

        // COO for transpose
        std::vector<uint32_t> t_p(t_n + 1, 0);
        // Count per-column nnz
        for (uint64_t k = 0; k < nnz; ++k) {
            uint32_t orig_row = mat.i[k];
            if (cfg.row_sort) orig_row = row_sort.perm[orig_row];
            t_p[orig_row + 1]++;
        }
        // Prefix sum
        for (uint32_t j = 0; j < t_n; ++j)
            t_p[j + 1] += t_p[j];

        std::vector<uint32_t> t_i(nnz);
        std::vector<double> t_x(nnz);
        std::vector<uint32_t> write_pos(t_n, 0);

        for (uint32_t col = 0; col < n; ++col) {
            for (uint32_t k = mat.p[col]; k < mat.p[col + 1]; ++k) {
                uint32_t orig_row = mat.i[k];
                if (cfg.row_sort) orig_row = row_sort.perm[orig_row];
                uint32_t dst = t_p[orig_row] + write_pos[orig_row]++;
                t_i[dst] = col;  // original column becomes row in transpose
                t_x[dst] = mat.x[k];
            }
        }

        // Chunk the transpose
        num_transpose_chunks = (t_n + cfg.chunk_cols - 1) / cfg.chunk_cols;
        transpose_descs.resize(num_transpose_chunks);

        struct TChunkData {
            std::vector<uint8_t> gap_data;
            std::vector<uint8_t> value_data;
        };
        std::vector<TChunkData> t_chunks(num_transpose_chunks);

        // Convert transpose values
        std::vector<float> t_f32;
        if (vtype == ValueType_v2::FLOAT32 || vtype == ValueType_v2::FLOAT16 ||
            vtype == ValueType_v2::QUANT8) {
            t_f32.resize(nnz);
            for (uint64_t k = 0; k < nnz; ++k)
                t_f32[k] = static_cast<float>(t_x[k]);
        }
        std::vector<uint32_t> t_int;
        if (vtype == ValueType_v2::UINT8 || vtype == ValueType_v2::UINT16 ||
            vtype == ValueType_v2::UINT32) {
            t_int.resize(nnz);
            for (uint64_t k = 0; k < nnz; ++k)
                t_int[k] = static_cast<uint32_t>(t_x[k]);
        }

        // Parallel encode transpose chunks
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) if(num_transpose_chunks > 4)
        #endif
        for (uint32_t tc = 0; tc < num_transpose_chunks; ++tc) {
            uint32_t tc_start = tc * cfg.chunk_cols;
            uint32_t tc_end = std::min(tc_start + cfg.chunk_cols, t_n);
            uint32_t tc_nc = tc_end - tc_start;

            std::vector<uint32_t> local_tp(tc_nc + 1);
            uint32_t tc_nnz_start = t_p[tc_start];
            for (uint32_t j = 0; j <= tc_nc; ++j)
                local_tp[j] = t_p[tc_start + j] - tc_nnz_start;
            uint32_t tc_nnz = local_tp[tc_nc];

            transpose_descs[tc].col_start = tc_start;
            transpose_descs[tc].num_cols = tc_nc;
            transpose_descs[tc].nnz = tc_nnz;

            t_chunks[tc].gap_data = detail::encode_gaps_chunk(
                local_tp.data(), t_i.data() + tc_nnz_start, tc_nc, tc_nnz);

            switch (vtype) {
                case ValueType_v2::UINT8:
                case ValueType_v2::UINT16:
                case ValueType_v2::UINT32:
                    t_chunks[tc].value_data = detail::encode_values_int_chunk(
                        t_int.data() + tc_nnz_start, tc_nnz);
                    break;
                case ValueType_v2::FLOAT32:
                    t_chunks[tc].value_data = detail::encode_values_f32_chunk(
                        t_f32.data() + tc_nnz_start, tc_nnz);
                    break;
                case ValueType_v2::FLOAT16:
                    t_chunks[tc].value_data = detail::encode_values_f16_chunk(
                        t_f32.data() + tc_nnz_start, tc_nnz);
                    break;
                case ValueType_v2::QUANT8:
                    t_chunks[tc].value_data = detail::encode_values_quant8_chunk(
                        t_f32.data() + tc_nnz_start, tc_nnz,
                        transpose_descs[tc].quant_scale, transpose_descs[tc].quant_offset);
                    break;
                case ValueType_v2::FLOAT64:
                    t_chunks[tc].value_data = detail::encode_values_f64_chunk(
                        t_x.data() + tc_nnz_start, tc_nnz);
                    break;
            }
        }

        // Build transpose data layout
        uint64_t t_index_size = static_cast<uint64_t>(num_transpose_chunks) * sizeof(ChunkDescriptor_v2);
        uint64_t t_data_offset_local = 0;
        for (uint32_t tc = 0; tc < num_transpose_chunks; ++tc) {
            transpose_descs[tc].stream_offset[STREAM_GAPS] = static_cast<uint32_t>(t_data_offset_local);
            transpose_descs[tc].stream_size[STREAM_GAPS] = static_cast<uint32_t>(t_chunks[tc].gap_data.size());
            t_data_offset_local += t_chunks[tc].gap_data.size();

            transpose_descs[tc].stream_offset[STREAM_VALUES] = static_cast<uint32_t>(t_data_offset_local);
            transpose_descs[tc].stream_size[STREAM_VALUES] = static_cast<uint32_t>(t_chunks[tc].value_data.size());
            t_data_offset_local += t_chunks[tc].value_data.size();

            transpose_descs[tc].decoded_gap_bytes = transpose_descs[tc].nnz * sizeof(uint32_t);
            transpose_descs[tc].decoded_value_bytes = transpose_descs[tc].nnz * value_type_bytes(vtype);
        }

        // Assemble transpose section: [num_transpose_chunks:u32][descs][data]
        transpose_data.reserve(4 + t_index_size + t_data_offset_local);

        // Number of transpose chunks
        uint32_t ntc = num_transpose_chunks;
        const uint8_t* ntc_ptr = reinterpret_cast<const uint8_t*>(&ntc);
        transpose_data.insert(transpose_data.end(), ntc_ptr, ntc_ptr + 4);

        // Transpose descriptors
        for (uint32_t tc = 0; tc < num_transpose_chunks; ++tc) {
            const uint8_t* desc_ptr = reinterpret_cast<const uint8_t*>(&transpose_descs[tc]);
            transpose_data.insert(transpose_data.end(), desc_ptr, desc_ptr + sizeof(ChunkDescriptor_v2));
        }

        // Transpose chunk data
        for (uint32_t tc = 0; tc < num_transpose_chunks; ++tc) {
            transpose_data.insert(transpose_data.end(),
                t_chunks[tc].gap_data.begin(), t_chunks[tc].gap_data.end());
            transpose_data.insert(transpose_data.end(),
                t_chunks[tc].value_data.begin(), t_chunks[tc].value_data.end());
        }

        SPZ2_LOG(cfg.verbose, 1, "[spz] Transpose: %u chunks, %zu bytes\n",
                 num_transpose_chunks, transpose_data.size());
    }

    hdr.metadata_offset = hdr.data_offset + data_size;
    if (!transpose_data.empty()) {
        hdr.transpose_offset = hdr.metadata_offset;
        hdr.metadata_offset = hdr.transpose_offset + transpose_data.size();
    }

    // Obs/var table offsets (after transpose, before metadata)
    if (!cfg.obs_buf.empty()) {
        hdr.set_obs_table_offset(hdr.metadata_offset);
        hdr.metadata_offset += cfg.obs_buf.size();
    }
    if (!cfg.var_buf.empty()) {
        hdr.set_var_table_offset(hdr.metadata_offset);
        hdr.metadata_offset += cfg.var_buf.size();
    }

    // Assemble output
    uint64_t total_size = hdr.metadata_offset + meta_bytes.size() + FOOTER_SIZE;
    std::vector<uint8_t> output;
    output.reserve(total_size);

    // Header
    auto hdr_bytes = hdr.serialize();
    output.insert(output.end(), hdr_bytes.begin(), hdr_bytes.end());

    // Chunk index
    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint8_t* desc_ptr = reinterpret_cast<const uint8_t*>(&chunks[c].desc);
        output.insert(output.end(), desc_ptr, desc_ptr + sizeof(ChunkDescriptor_v2));
    }

    // Data section
    for (uint32_t c = 0; c < num_chunks; ++c) {
        output.insert(output.end(), chunks[c].gap_data.begin(), chunks[c].gap_data.end());
        output.insert(output.end(), chunks[c].value_data.begin(), chunks[c].value_data.end());
    }

    // Transpose section (optional)
    if (!transpose_data.empty()) {
        output.insert(output.end(), transpose_data.begin(), transpose_data.end());
    }

    // Obs/var tables (optional)
    if (!cfg.obs_buf.empty()) {
        output.insert(output.end(), cfg.obs_buf.begin(), cfg.obs_buf.end());
    }
    if (!cfg.var_buf.empty()) {
        output.insert(output.end(), cfg.var_buf.begin(), cfg.var_buf.end());
    }

    // Metadata
    output.insert(output.end(), meta_bytes.begin(), meta_bytes.end());

    // Footer
    Footer_v2 footer;
    footer.metadata_size = static_cast<uint32_t>(meta_bytes.size());
    footer.total_chunks = num_chunks;
    footer.file_crc32 = CRC32::compute(output.data(), output.size());
    auto footer_bytes = footer.serialize();
    output.insert(output.end(), footer_bytes.begin(), footer_bytes.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (stats_out) {
        stats_out->raw_size = 12 + (n + 1) * 4 + nnz * 4 + nnz * value_type_bytes(vtype);
        stats_out->compressed_size = output.size();
        stats_out->compress_time_ms = elapsed_ms;
        stats_out->num_chunks = num_chunks;
    }

    SPZ2_LOG(cfg.verbose, 1,
        "[spz] Compressed: %zu → %zu bytes (%.1fx), %u chunks, %.1f ms\n",
        stats_out ? stats_out->raw_size : 0, output.size(),
        stats_out ? stats_out->ratio() : 0.0,
        num_chunks, elapsed_ms);

    return output;
}

// ============================================================================
// Public API: decompress_v2
// ============================================================================

struct DecompressConfig_v2 {
    bool reorder = true;    // Undo row permutation
    int  verbose = 0;
    int  num_threads = 0;   // 0 = auto
    // Partial read: specify column range (0-indexed, -1 = all)
    int  col_start = -1;
    int  col_end = -1;
    // Full reads check the footer CRC32 against the file contents
    bool verify_crc = true;
};

inline CSCMatrix decompress_v2(
    const uint8_t* data, size_t data_size,
    const DecompressConfig_v2& cfg = {},
    Metadata* meta_out = nullptr)
{
    if (data_size < HEADER_SIZE_V2)
        throw std::runtime_error("Data too small for v2 header");

    FileHeader_v2 hdr = FileHeader_v2::deserialize(data);

    SPZ2_LOG(cfg.verbose, 1,
        "[spz] Decompressing v2: %u × %u, nnz=%lu, %u chunks, type=%s\n",
        hdr.m, hdr.n, (unsigned long)hdr.nnz, hdr.num_chunks,
        value_type_name(static_cast<ValueType_v2>(hdr.value_type)));

    // Read chunk index
    detail::validate_header_v2(hdr, data_size);
    // A full read sees the whole file: require the footer and check its CRC
    if (cfg.col_start < 0 || cfg.col_end < 0)
        detail::verify_footer_v2(data, data_size, cfg.verify_crc);

    std::vector<ChunkDescriptor_v2> chunk_descs(hdr.num_chunks);
    std::memcpy(chunk_descs.data(),
                data + hdr.chunk_index_offset,
                hdr.num_chunks * sizeof(ChunkDescriptor_v2));
    detail::validate_chunk_index(chunk_descs, hdr.n, hdr.m, true, hdr.nnz);

    // Determine which chunks to decode
    uint32_t first_chunk = 0, last_chunk = hdr.num_chunks;  // [first, last)
    uint32_t out_col_start = 0;
    uint32_t out_n = hdr.n;
    uint64_t out_nnz = hdr.nnz;

    if (cfg.col_start >= 0 && cfg.col_end >= 0) {
        // Find chunks covering requested columns
        uint32_t req_start = static_cast<uint32_t>(cfg.col_start);
        uint32_t req_end = static_cast<uint32_t>(cfg.col_end);
        if (req_end > hdr.n) req_end = hdr.n;

        for (uint32_t c = 0; c < hdr.num_chunks; ++c) {
            if (chunk_descs[c].col_start + chunk_descs[c].num_cols <= req_start)
                first_chunk = c + 1;
            if (chunk_descs[c].col_start < req_end)
                last_chunk = c + 1;
        }

        if (first_chunk >= last_chunk)
            throw std::runtime_error(
                "Requested column range is empty or outside the matrix");
        out_col_start = chunk_descs[first_chunk].col_start;
        uint32_t end_chunk = last_chunk - 1;
        out_n = chunk_descs[end_chunk].col_start + chunk_descs[end_chunk].num_cols - out_col_start;
        out_nnz = 0;
        for (uint32_t c = first_chunk; c < last_chunk; ++c)
            out_nnz += chunk_descs[c].nnz;

        SPZ2_LOG(cfg.verbose, 2,
            "[spz] Partial read: chunks %u-%u, cols %u-%u, nnz=%lu\n",
            first_chunk, last_chunk - 1, out_col_start,
            out_col_start + out_n - 1, (unsigned long)out_nnz);
    }

    ValueType_v2 vtype = static_cast<ValueType_v2>(hdr.value_type);

    // Allocate output
    CSCMatrix result(hdr.m, out_n, out_nnz);

    // Compute nnz offsets per chunk
    uint32_t n_decode = last_chunk - first_chunk;
    std::vector<uint64_t> chunk_nnz_offset(n_decode + 1, 0);
    for (uint32_t c = 0; c < n_decode; ++c)
        chunk_nnz_offset[c + 1] = chunk_nnz_offset[c] + chunk_descs[first_chunk + c].nnz;

    int num_threads = cfg.num_threads;
    (void)num_threads;
    #ifdef _OPENMP
    num_threads = (num_threads <= 0) ? omp_get_max_threads() : num_threads;
    // Clamp to actual chunk count: idle threads beyond n_decode cause race
    // conditions in repeated calls (e.g., loading many files in a loop).
    num_threads = std::min(num_threads, static_cast<int>(n_decode));
    if (num_threads < 1) num_threads = 1;
    #else
    num_threads = 1;
    #endif

    // An exception must not leave an OpenMP region (that calls
    // std::terminate), so the first decode error is recorded and rethrown
    // after the loop.
    std::atomic<bool> decode_failed(false);
    std::string decode_error;

    // Decode chunks in parallel
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic) if(n_decode > 1)
    #endif
    for (uint32_t ci = 0; ci < n_decode; ++ci) {
        if (decode_failed.load()) continue;
        try {
        uint32_t c = first_chunk + ci;
        const auto& desc = chunk_descs[c];
        uint64_t out_offset = chunk_nnz_offset[ci];

        // Decode gap stream: [cc_size(4)][col_counts varint][rANS table][enc(4)][data][ov(4)][overflow]
        detail::require_chunk_streams(desc, data_size - hdr.data_offset);
        const uint8_t* gap_ptr = data + hdr.data_offset + desc.stream_offset[STREAM_GAPS];
        size_t gap_size = desc.stream_size[STREAM_GAPS];

        uint32_t local_col_base = desc.col_start - out_col_start;

        if (gap_size >= 4) {
            // Read col counts section size
            uint32_t cc_size = detail::read_u32(gap_ptr);
            if (cc_size > gap_size - 4)
                spz_corrupt("column-count section exceeds its chunk");
            const uint8_t* cc_ptr = gap_ptr + 4;
            const uint8_t* rans_ptr = cc_ptr + cc_size;
            size_t rans_size = gap_size - 4 - cc_size;

            // Decode column counts
            std::vector<uint32_t> col_counts(desc.num_cols);
            const uint8_t* cc_scan = cc_ptr;
            for (uint32_t j = 0; j < desc.num_cols; ++j) {
                col_counts[j] = detail::read_col_count(cc_scan, rans_ptr, desc.nnz);
            }

            // Set column pointers
            uint32_t running = static_cast<uint32_t>(out_offset);
            for (uint32_t j = 0; j < desc.num_cols; ++j) {
                result.p[local_col_base + j] = running;
                running += col_counts[j];
            }

            // Decode gaps via rANS
            if (desc.nnz > 0) {
                detail::check_col_counts(col_counts, desc.nnz);
                auto gaps = detail::decode_rans_escape(rans_ptr, rans_size, desc.nnz);

                // Reconstruct row indices per column
                uint32_t gap_idx = 0;
                for (uint32_t j = 0; j < desc.num_cols; ++j) {
                    uint32_t prev_row = 0;
                    for (uint32_t ki = 0; ki < col_counts[j]; ++ki) {
                        uint32_t row = detail::next_row(prev_row, gaps[gap_idx], hdr.m);
                        result.i[out_offset + gap_idx] = row;
                        prev_row = row + 1;
                        gap_idx++;
                    }
                }
            }
        } else {
            // Empty chunk
            for (uint32_t j = 0; j < desc.num_cols; ++j)
                result.p[local_col_base + j] = static_cast<uint32_t>(out_offset);
        }

        // Decode values
        const uint8_t* val_ptr = data + hdr.data_offset + desc.stream_offset[STREAM_VALUES];
        size_t val_size = desc.stream_size[STREAM_VALUES];

        switch (vtype) {
            case ValueType_v2::UINT8:
            case ValueType_v2::UINT16:
            case ValueType_v2::UINT32: {
                auto int_vals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(int_vals[k]);
                break;
            }
            case ValueType_v2::FLOAT32: {
                std::vector<float> f32(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                   f32.data(), 4);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(f32[k]);
                break;
            }
            case ValueType_v2::FLOAT16: {
                std::vector<uint16_t> f16(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                   f16.data(), 2);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(half_to_float(f16[k]));
                break;
            }
            case ValueType_v2::QUANT8: {
                auto qvals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                for (uint32_t k = 0; k < desc.nnz; ++k) {
                    result.x[out_offset + k] = static_cast<double>(
                        desc.quant_offset + desc.quant_scale * qvals[k]);
                }
                break;
            }
            case ValueType_v2::FLOAT64: {
                std::vector<double> f64(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                   f64.data(), 8);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = f64[k];
                break;
            }
        }

        SPZ2_LOG(cfg.verbose, 3,
            "[spz] Decoded chunk %u: %u nnz\n", c, desc.nnz);
        } catch (const std::exception& e) {
            #ifdef _OPENMP
            #pragma omp critical(spz_decode_error)
            #endif
            {
                if (!decode_failed.load()) decode_error = e.what();
                decode_failed.store(true);
            }
        }
    }
    if (decode_failed.load()) throw std::runtime_error(decode_error);

    // Set final column pointer
    result.p[out_n] = static_cast<uint32_t>(out_nnz);

    // Read metadata
    if (hdr.metadata_offset > 0 && data_size >= FOOTER_SIZE &&
        hdr.metadata_offset <= data_size - FOOTER_SIZE) {
        size_t meta_avail = data_size - FOOTER_SIZE - hdr.metadata_offset;
        Metadata meta = Metadata::deserialize(data + hdr.metadata_offset, meta_avail);

        // Undo row permutation if requested
        if (cfg.reorder && hdr.row_sorted && meta.has_row_permutation()) {
            auto inv_perm = meta.get_row_permutation();
            // inv_perm maps sorted→original
            // We need to undo: for each nnz, result.i[k] = inv_perm[result.i[k]]
            for (uint64_t k = 0; k < out_nnz; ++k) {
                if (result.i[k] < inv_perm.size()) {
                    const uint32_t orig = inv_perm[result.i[k]];
                    if (orig >= hdr.m)
                        spz_corrupt("row permutation entry out of range");
                    result.i[k] = orig;
                }
            }
        }

        if (meta_out) *meta_out = meta;
    }

    return result;
}

// ============================================================================
// Public API: decompress_v2_typed — read CSC with native Scalar values
// ============================================================================

/**
 * @brief Decompress a v2 file into a typed CSCMatrixTyped<Scalar>.
 *
 * When Scalar=float and the file stores FLOAT32 data, this avoids the
 * intermediate conversion to double, saving memory for large fp32 datasets.
 * For other value types (UINT8/16/32, FLOAT16, FLOAT64, QUANT8), values
 * are decoded to their native type and then cast to Scalar.
 *
 * @tparam Scalar  float or double
 */
template<typename Scalar>
inline CSCMatrixTyped<Scalar> decompress_v2_typed(
    const uint8_t* data, size_t data_size,
    const DecompressConfig_v2& cfg = {})
{
    if (data_size < HEADER_SIZE_V2)
        throw std::runtime_error("Data too small for v2 header");

    FileHeader_v2 hdr = FileHeader_v2::deserialize(data);

    // Read chunk index
    detail::validate_header_v2(hdr, data_size);
    // A full read sees the whole file: require the footer and check its CRC
    if (cfg.col_start < 0 || cfg.col_end < 0)
        detail::verify_footer_v2(data, data_size, cfg.verify_crc);

    std::vector<ChunkDescriptor_v2> chunk_descs(hdr.num_chunks);
    std::memcpy(chunk_descs.data(),
                data + hdr.chunk_index_offset,
                hdr.num_chunks * sizeof(ChunkDescriptor_v2));
    detail::validate_chunk_index(chunk_descs, hdr.n, hdr.m, true, hdr.nnz);

    // Determine which chunks to decode
    uint32_t first_chunk = 0, last_chunk = hdr.num_chunks;
    uint32_t out_col_start = 0;
    uint32_t out_n = hdr.n;
    uint64_t out_nnz = hdr.nnz;

    if (cfg.col_start >= 0 && cfg.col_end >= 0) {
        uint32_t req_start = static_cast<uint32_t>(cfg.col_start);
        uint32_t req_end = static_cast<uint32_t>(cfg.col_end);
        if (req_end > hdr.n) req_end = hdr.n;

        for (uint32_t c = 0; c < hdr.num_chunks; ++c) {
            if (chunk_descs[c].col_start + chunk_descs[c].num_cols <= req_start)
                first_chunk = c + 1;
            if (chunk_descs[c].col_start < req_end)
                last_chunk = c + 1;
        }

        if (first_chunk >= last_chunk)
            throw std::runtime_error(
                "Requested column range is empty or outside the matrix");
        out_col_start = chunk_descs[first_chunk].col_start;
        uint32_t end_chunk = last_chunk - 1;
        out_n = (chunk_descs[end_chunk].col_start + chunk_descs[end_chunk].num_cols) - out_col_start;
        out_nnz = 0;
        for (uint32_t c = first_chunk; c < last_chunk; ++c)
            out_nnz += chunk_descs[c].nnz;
    }

    CSCMatrixTyped<Scalar> result(hdr.m, out_n, out_nnz);
    ValueType_v2 vtype = static_cast<ValueType_v2>(hdr.value_type);
    uint64_t out_offset = 0;

    for (uint32_t c = first_chunk; c < last_chunk; ++c) {
        const auto& desc = chunk_descs[c];
        uint32_t local_col_base = desc.col_start - out_col_start;

        // Decode structure (col pointers and row indices) — same as decompress_v2
        if (desc.nnz > 0) {
            // Decode gap stream: [cc_size(4)][col_counts varint][rANS table][enc(4)][data][ov(4)][overflow]
            detail::require_chunk_streams(desc, data_size - hdr.data_offset);
            const uint8_t* gap_ptr = data + hdr.data_offset + desc.stream_offset[STREAM_GAPS];
            size_t gap_size = desc.stream_size[STREAM_GAPS];

            if (gap_size >= 4) {
                // Read col counts section size
                uint32_t cc_size = detail::read_u32(gap_ptr);
                if (cc_size > gap_size - 4)
                    spz_corrupt("column-count section exceeds its chunk");
                const uint8_t* cc_ptr = gap_ptr + 4;
                const uint8_t* rans_ptr = cc_ptr + cc_size;
                size_t rans_size = gap_size - 4 - cc_size;

                // Decode column counts
                std::vector<uint32_t> col_counts(desc.num_cols);
                const uint8_t* cc_scan = cc_ptr;
                for (uint32_t j = 0; j < desc.num_cols; ++j) {
                    col_counts[j] = detail::read_col_count(cc_scan, rans_ptr, desc.nnz);
                }

                // Set column pointers
                uint32_t running = static_cast<uint32_t>(out_offset);
                for (uint32_t j = 0; j < desc.num_cols; ++j) {
                    result.p[local_col_base + j] = running;
                    running += col_counts[j];
                }

                // Decode gaps via rANS
                detail::check_col_counts(col_counts, desc.nnz);
                auto gaps = detail::decode_rans_escape(rans_ptr, rans_size, desc.nnz);

                // Reconstruct row indices per column
                uint32_t gap_idx = 0;
                for (uint32_t j = 0; j < desc.num_cols; ++j) {
                    uint32_t prev_row = 0;
                    for (uint32_t ki = 0; ki < col_counts[j]; ++ki) {
                        uint32_t row = detail::next_row(prev_row, gaps[gap_idx], hdr.m);
                        result.i[out_offset + gap_idx] = row;
                        prev_row = row + 1;
                        gap_idx++;
                    }
                }
            } else {
                // Empty gap stream — set column pointers only
                for (uint32_t j = 0; j < desc.num_cols; ++j)
                    result.p[local_col_base + j] = static_cast<uint32_t>(out_offset);
            }
        } else {
            for (uint32_t j = 0; j < desc.num_cols; ++j)
                result.p[local_col_base + j] = static_cast<uint32_t>(out_offset);
        }

        // Decode values — native Scalar path
        const uint8_t* val_ptr = data + hdr.data_offset + desc.stream_offset[STREAM_VALUES];
        size_t val_size = desc.stream_size[STREAM_VALUES];

        switch (vtype) {
            case ValueType_v2::UINT8:
            case ValueType_v2::UINT16:
            case ValueType_v2::UINT32: {
                auto int_vals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<Scalar>(int_vals[k]);
                break;
            }
            case ValueType_v2::FLOAT32: {
                if constexpr (std::is_same_v<Scalar, float>) {
                    // Native fp32 path — no intermediate double conversion
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                      result.x.data() + out_offset, 4);
                } else {
                    std::vector<float> f32(desc.nnz);
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                      f32.data(), 4);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[out_offset + k] = static_cast<Scalar>(f32[k]);
                }
                break;
            }
            case ValueType_v2::FLOAT16: {
                std::vector<uint16_t> f16(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                  f16.data(), 2);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<Scalar>(half_to_float(f16[k]));
                break;
            }
            case ValueType_v2::QUANT8: {
                auto qvals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                for (uint32_t k = 0; k < desc.nnz; ++k) {
                    result.x[out_offset + k] = static_cast<Scalar>(
                        desc.quant_offset + desc.quant_scale * qvals[k]);
                }
                break;
            }
            case ValueType_v2::FLOAT64: {
                if constexpr (std::is_same_v<Scalar, double>) {
                    // Native fp64 path — no conversion
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                      result.x.data() + out_offset, 8);
                } else {
                    std::vector<double> f64(desc.nnz);
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz,
                                                      f64.data(), 8);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[out_offset + k] = static_cast<Scalar>(f64[k]);
                }
                break;
            }
        }

        out_offset += desc.nnz;
    }

    // Set final column pointer
    result.p[out_n] = static_cast<uint32_t>(out_nnz);

    return result;
}

// ============================================================================
// Public API: decompress_v2_transpose — read pre-stored CSC(Aᵀ)
// ============================================================================

/**
 * @brief Check if a v2 file contains a pre-stored transpose.
 */
inline bool has_transpose(const uint8_t* data, size_t data_size) {
    if (data_size < HEADER_SIZE_V2) return false;
    FileHeader_v2 hdr = FileHeader_v2::deserialize(data);
    return hdr.transpose_offset != 0;
}

/**
 * @brief Decompress the pre-stored transpose section of a v2 file.
 *
 * Returns CSC(Aᵀ) with dimensions (n × m) where the original
 * matrix is (m × n). This avoids runtime transpose computation.
 *
 * @throws std::runtime_error if the file has no transpose section.
 */
inline CSCMatrix decompress_v2_transpose(
    const uint8_t* data, size_t data_size,
    const DecompressConfig_v2& cfg = {})
{
    if (data_size < HEADER_SIZE_V2)
        throw std::runtime_error("Data too small for v2 header");

    FileHeader_v2 hdr = FileHeader_v2::deserialize(data);

    if (hdr.transpose_offset == 0)
        throw std::runtime_error("File does not contain a pre-stored transpose");
    detail::validate_header_v2(hdr, data_size);
    // This entry point takes the whole file: require the footer and check
    // its CRC, as the forward reader does
    detail::verify_footer_v2(data, data_size, cfg.verify_crc);

    ValueType_v2 vtype = static_cast<ValueType_v2>(hdr.value_type);

    // Transpose dimensions: columns = original rows, rows = original columns
    uint32_t t_m = hdr.n;   // rows of Aᵀ
    uint32_t t_n = hdr.m;   // cols of Aᵀ

    // Read transpose section
    const uint8_t* t_section = data + hdr.transpose_offset;
    // Bytes of the transpose section actually present in the buffer
    if (hdr.transpose_offset > data_size)
        spz_corrupt("transpose section lies outside the file");
    size_t t_avail = data_size - static_cast<size_t>(hdr.transpose_offset);
    if (hdr.metadata_offset > hdr.transpose_offset &&
        hdr.metadata_offset - hdr.transpose_offset < t_avail)
        t_avail = static_cast<size_t>(hdr.metadata_offset - hdr.transpose_offset);

    if (t_avail < 4)
        throw std::runtime_error("Transpose section too small");

    // Read num_transpose_chunks
    uint32_t num_t_chunks;
    std::memcpy(&num_t_chunks, t_section, 4);
    const uint8_t* t_idx_ptr = t_section + 4;

    // Read transpose chunk descriptors
    const uint64_t t_idx_size64 =
        static_cast<uint64_t>(num_t_chunks) * sizeof(ChunkDescriptor_v2);
    if (t_idx_size64 > t_avail - 4)
        throw std::runtime_error("Transpose section truncated at index");
    size_t t_idx_size = static_cast<size_t>(t_idx_size64);

    std::vector<ChunkDescriptor_v2> t_descs(num_t_chunks);
    if (t_idx_size > 0)
        std::memcpy(t_descs.data(), t_idx_ptr, t_idx_size);
    // Chunks of the transpose tile its columns (the rows of A)
    detail::validate_chunk_index(t_descs, t_n, t_m, false, 0);

    const uint8_t* t_data_base = t_idx_ptr + t_idx_size;
    // Bytes that follow the transpose index in the buffer
    const uint64_t t_data_avail =
        data_size - static_cast<size_t>(t_data_base - data);

    // Compute output
    uint64_t t_total_nnz = 0;
    for (uint32_t tc = 0; tc < num_t_chunks; ++tc)
        t_total_nnz += t_descs[tc].nnz;

    CSCMatrix result(t_m, t_n, t_total_nnz);

    // Compute per-chunk offsets
    std::vector<uint64_t> chunk_nnz_off(num_t_chunks + 1, 0);
    for (uint32_t tc = 0; tc < num_t_chunks; ++tc)
        chunk_nnz_off[tc + 1] = chunk_nnz_off[tc] + t_descs[tc].nnz;

    // Decode transpose chunks (same logic as main decode)
    std::atomic<bool> decode_failed(false);
    std::string decode_error;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(num_t_chunks > 1)
    #endif
    for (uint32_t tc = 0; tc < num_t_chunks; ++tc) {
        if (decode_failed.load()) continue;
        try {
        const auto& desc = t_descs[tc];
        uint64_t out_offset = chunk_nnz_off[tc];
        uint32_t local_col_base = desc.col_start;

        // Decode gaps
        detail::require_chunk_streams(desc, t_data_avail);
        const uint8_t* gap_ptr = t_data_base + desc.stream_offset[STREAM_GAPS];
        size_t gap_size = desc.stream_size[STREAM_GAPS];

        if (gap_size >= 4) {
            uint32_t cc_size = detail::read_u32(gap_ptr);
            if (cc_size > gap_size - 4)
                spz_corrupt("column-count section exceeds its chunk");
            const uint8_t* cc_ptr = gap_ptr + 4;
            const uint8_t* rans_ptr = cc_ptr + cc_size;
            size_t rans_size = gap_size - 4 - cc_size;

            std::vector<uint32_t> col_counts(desc.num_cols);
            const uint8_t* cc_scan = cc_ptr;
            for (uint32_t j = 0; j < desc.num_cols; ++j)
                col_counts[j] = detail::read_col_count(cc_scan, rans_ptr, desc.nnz);

            uint32_t running = static_cast<uint32_t>(out_offset);
            for (uint32_t j = 0; j < desc.num_cols; ++j) {
                result.p[local_col_base + j] = running;
                running += col_counts[j];
            }

            if (desc.nnz > 0) {
                detail::check_col_counts(col_counts, desc.nnz);
                auto gaps = detail::decode_rans_escape(rans_ptr, rans_size, desc.nnz);
                uint32_t gap_idx = 0;
                for (uint32_t j = 0; j < desc.num_cols; ++j) {
                    uint32_t prev_row = 0;
                    for (uint32_t ki = 0; ki < col_counts[j]; ++ki) {
                        const uint32_t row =
                            detail::next_row(prev_row, gaps[gap_idx], t_m);
                        result.i[out_offset + gap_idx] = row;
                        prev_row = row + 1;
                        gap_idx++;
                    }
                }
            }
        } else {
            for (uint32_t j = 0; j < desc.num_cols; ++j)
                result.p[local_col_base + j] = static_cast<uint32_t>(out_offset);
        }

        // Decode values
        const uint8_t* val_ptr = t_data_base + desc.stream_offset[STREAM_VALUES];
        size_t val_size = desc.stream_size[STREAM_VALUES];

        switch (vtype) {
            case ValueType_v2::UINT8:
            case ValueType_v2::UINT16:
            case ValueType_v2::UINT32: {
                auto int_vals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(int_vals[k]);
                break;
            }
            case ValueType_v2::FLOAT32: {
                std::vector<float> f32(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz, f32.data(), 4);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(f32[k]);
                break;
            }
            case ValueType_v2::FLOAT16: {
                std::vector<uint16_t> f16(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz, f16.data(), 2);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(half_to_float(f16[k]));
                break;
            }
            case ValueType_v2::QUANT8: {
                auto qvals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = static_cast<double>(
                        desc.quant_offset + desc.quant_scale * qvals[k]);
                break;
            }
            case ValueType_v2::FLOAT64: {
                std::vector<double> f64(desc.nnz);
                detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz, f64.data(), 8);
                for (uint32_t k = 0; k < desc.nnz; ++k)
                    result.x[out_offset + k] = f64[k];
                break;
            }
        }
        } catch (const std::exception& e) {
            #ifdef _OPENMP
            #pragma omp critical(spz_decode_error)
            #endif
            {
                if (!decode_failed.load()) decode_error = e.what();
                decode_failed.store(true);
            }
        }
    }
    if (decode_failed.load()) throw std::runtime_error(decode_error);

    result.p[t_n] = static_cast<uint32_t>(t_total_nnz);

    SPZ2_LOG(cfg.verbose, 1,
        "[spz] Decoded transpose: %u × %u, nnz=%lu, %u chunks\n",
        t_m, t_n, (unsigned long)t_total_nnz, num_t_chunks);

    return result;
}

// ============================================================================
// Public API: TransposeChunkReader — stream transpose chunks one at a time
// ============================================================================

/**
 * @brief Lightweight reader for iterating transpose chunks without
 *        materializing the full CSC(Aᵀ).
 *
 * Usage:
 *   TransposeChunkReader reader(data, size);
 *   for (uint32_t tc = 0; tc < reader.num_chunks(); ++tc) {
 *       CSCMatrix chunk = reader.decompress_chunk(tc);
 *       // chunk covers columns [chunk.col_start .. chunk.col_start + chunk.n - 1]
 *       // of the transpose (= original rows col_start..col_start+n-1)
 *       // process chunk...
 *   }
 *
 * Memory: only one chunk materialized at a time → O(max_chunk_nnz).
 */
class TransposeChunkReader {
public:
    TransposeChunkReader(const uint8_t* data, size_t data_size)
    {
        if (data_size < HEADER_SIZE_V2)
            throw std::runtime_error("Data too small for v2 header");

        hdr_ = FileHeader_v2::deserialize(data);
        if (hdr_.transpose_offset == 0)
            throw std::runtime_error("File has no transpose section");
        detail::validate_header_v2(hdr_, data_size);

        vtype_ = static_cast<ValueType_v2>(hdr_.value_type);

        const uint8_t* t_section = data + hdr_.transpose_offset;
        // Bytes of the transpose section actually present in the buffer
        if (hdr_.transpose_offset > data_size)
            spz_corrupt("transpose section lies outside the file");
        size_t t_avail = data_size - static_cast<size_t>(hdr_.transpose_offset);
        if (hdr_.metadata_offset > hdr_.transpose_offset &&
            hdr_.metadata_offset - hdr_.transpose_offset < t_avail)
            t_avail = static_cast<size_t>(hdr_.metadata_offset - hdr_.transpose_offset);

        if (t_avail < 4)
            throw std::runtime_error("Transpose section too small");

        std::memcpy(&num_t_chunks_, t_section, 4);

        const uint64_t t_idx_size64 =
            static_cast<uint64_t>(num_t_chunks_) * sizeof(ChunkDescriptor_v2);
        if (t_idx_size64 > t_avail - 4)
            throw std::runtime_error("Transpose section truncated at index");
        size_t t_idx_size = static_cast<size_t>(t_idx_size64);

        t_descs_.resize(num_t_chunks_);
        if (t_idx_size > 0)
            std::memcpy(t_descs_.data(), t_section + 4, t_idx_size);
        // Chunks of the transpose tile its columns (the rows of A)
        detail::validate_chunk_index(t_descs_, hdr_.m, hdr_.n, false, 0);

        t_data_base_ = t_section + 4 + t_idx_size;
        // Bytes that follow the transpose index in the buffer
        t_data_avail_ = data_size - static_cast<size_t>(t_data_base_ - data);

        // Pre-compute cumulative nnz offsets (needed for col pointers)
        chunk_nnz_off_.resize(num_t_chunks_ + 1, 0);
        for (uint32_t tc = 0; tc < num_t_chunks_; ++tc)
            chunk_nnz_off_[tc + 1] = chunk_nnz_off_[tc] + t_descs_[tc].nnz;

        // transpose dimensions
        t_m_ = hdr_.n;  // rows of Aᵀ = columns of A
        t_n_ = hdr_.m;  // cols of Aᵀ = rows of A
    }

    uint32_t num_chunks() const { return num_t_chunks_; }
    uint32_t t_m() const { return t_m_; }  // rows of Aᵀ
    uint32_t t_n() const { return t_n_; }  // cols of Aᵀ

    /// Column range for chunk tc (of the transpose = row range of original)
    uint32_t chunk_col_start(uint32_t tc) const { return t_descs_[tc].col_start; }
    uint32_t chunk_num_cols(uint32_t tc) const { return t_descs_[tc].num_cols; }
    uint32_t chunk_nnz(uint32_t tc) const { return t_descs_[tc].nnz; }

    /**
     * @brief Decompress a single transpose chunk into a CSCMatrix.
     *
     * The returned matrix has dimensions (t_m_ × num_cols) where
     * columns correspond to original-matrix rows [col_start, col_start + num_cols).
     * Row indices in the returned matrix correspond to original-matrix columns.
     *
     * Column pointers are 0-based (local to this chunk).
     */
    CSCMatrix decompress_chunk(uint32_t tc) const {
        if (tc >= num_t_chunks_)
            throw std::runtime_error("Transpose chunk index out of range");

        const auto& desc = t_descs_[tc];
        CSCMatrix result(t_m_, desc.num_cols, desc.nnz);

        // Decode gaps → row indices and column pointers
        detail::require_chunk_streams(desc, t_data_avail_);
        const uint8_t* gap_ptr = t_data_base_ + desc.stream_offset[STREAM_GAPS];
        size_t gap_size = desc.stream_size[STREAM_GAPS];

        if (gap_size >= 4 && desc.nnz > 0) {
            uint32_t cc_size = detail::read_u32(gap_ptr);
            if (cc_size > gap_size - 4)
                spz_corrupt("column-count section exceeds its chunk");
            const uint8_t* cc_ptr = gap_ptr + 4;
            const uint8_t* rans_ptr = cc_ptr + cc_size;
            size_t rans_size = gap_size - 4 - cc_size;

            // Decode column counts
            std::vector<uint32_t> col_counts(desc.num_cols);
            const uint8_t* cc_scan = cc_ptr;
            for (uint32_t j = 0; j < desc.num_cols; ++j)
                col_counts[j] = detail::read_col_count(cc_scan, rans_ptr, desc.nnz);

            // Set column pointers (0-based)
            uint32_t running = 0;
            for (uint32_t j = 0; j < desc.num_cols; ++j) {
                result.p[j] = running;
                running += col_counts[j];
            }
            result.p[desc.num_cols] = running;

            // Decode row gaps → row indices
            detail::check_col_counts(col_counts, desc.nnz);
            auto gaps = detail::decode_rans_escape(rans_ptr, rans_size, desc.nnz);
            uint32_t gap_idx = 0;
            for (uint32_t j = 0; j < desc.num_cols; ++j) {
                uint32_t prev_row = 0;
                for (uint32_t ki = 0; ki < col_counts[j]; ++ki) {
                    const uint32_t row =
                        detail::next_row(prev_row, gaps[gap_idx], t_m_);
                    result.i[gap_idx] = row;
                    prev_row = row + 1;
                    gap_idx++;
                }
            }
        } else {
            // Empty chunk
            for (uint32_t j = 0; j <= desc.num_cols; ++j)
                result.p[j] = 0;
        }

        // Decode values
        const uint8_t* val_ptr = t_data_base_ + desc.stream_offset[STREAM_VALUES];
        size_t val_size = desc.stream_size[STREAM_VALUES];

        if (desc.nnz > 0) {
            switch (vtype_) {
                case ValueType_v2::UINT8:
                case ValueType_v2::UINT16:
                case ValueType_v2::UINT32: {
                    auto int_vals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[k] = static_cast<double>(int_vals[k]);
                    break;
                }
                case ValueType_v2::FLOAT32: {
                    std::vector<float> f32(desc.nnz);
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz, f32.data(), 4);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[k] = static_cast<double>(f32[k]);
                    break;
                }
                case ValueType_v2::FLOAT16: {
                    std::vector<uint16_t> f16(desc.nnz);
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz, f16.data(), 2);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[k] = static_cast<double>(half_to_float(f16[k]));
                    break;
                }
                case ValueType_v2::QUANT8: {
                    auto qvals = detail::decode_rans_escape(val_ptr, val_size, desc.nnz);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[k] = static_cast<double>(
                            desc.quant_offset + desc.quant_scale * qvals[k]);
                    break;
                }
                case ValueType_v2::FLOAT64: {
                    std::vector<double> f64(desc.nnz);
                    detail::decode_values_byteshuffle(val_ptr, val_size, desc.nnz, f64.data(), 8);
                    for (uint32_t k = 0; k < desc.nnz; ++k)
                        result.x[k] = f64[k];
                    break;
                }
            }
        }

        return result;
    }

private:
    FileHeader_v2 hdr_;
    ValueType_v2 vtype_;
    uint32_t num_t_chunks_;
    uint32_t t_m_, t_n_;
    std::vector<ChunkDescriptor_v2> t_descs_;
    std::vector<uint64_t> chunk_nnz_off_;
    const uint8_t* t_data_base_;
    uint64_t t_data_avail_ = 0;
};

// ============================================================================

inline std::vector<uint8_t> compress_v2(
    const CSCMatrix_v2& mat,
    const CompressConfig_v2& cfg = {},
    CompressStats_v2* stats_out = nullptr)
{
    // Convert to v1 CSCMatrix for now (native fp32 path is a future optimization)
    CSCMatrix v1;
    v1.m = mat.m;
    v1.n = mat.n;
    v1.nnz = mat.nnz;
    v1.p = mat.p;
    v1.i = mat.i;
    v1.x.resize(mat.nnz);
    for (uint64_t k = 0; k < mat.nnz; ++k) {
        v1.x[k] = (mat.stored_type == ValueType_v2::FLOAT64) ?
            mat.x_f64[k] : static_cast<double>(mat.x_f32[k]);
    }
    return compress_v2(v1, cfg, stats_out);
}

// ============================================================================
// File I/O with v2
// ============================================================================

inline void write_v2(const std::string& path,
                     const std::vector<uint8_t>& compressed) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
    if (std::fwrite(compressed.data(), 1, compressed.size(), f) != compressed.size()) {
        std::fclose(f);
        throw std::runtime_error("Write failed: " + path);
    }
    std::fclose(f);
}

inline std::vector<uint8_t> read_v2(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    const long long sz = spz_file_size(f);
    if (sz < 0) {
        std::fclose(f);
        throw std::runtime_error("Cannot determine file size: " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(sz));
    if (sz > 0 &&
        std::fread(data.data(), 1, data.size(), f) != data.size()) {
        std::fclose(f);
        throw std::runtime_error("Read failed: " + path);
    }
    std::fclose(f);
    return data;
}

// ============================================================================
// Version dispatch: read v1 or v2
// ============================================================================

inline uint16_t detect_version(const uint8_t* data, size_t data_size) {
    if (data_size < 6) throw std::runtime_error("File too small");
    if (data[0] != 'S' || data[1] != 'P' || data[2] != 'R' || data[3] != 'Z')
        throw std::runtime_error("Invalid magic bytes");
    uint16_t version;
    std::memcpy(&version, data + 4, 2);
    return version;
}

} // namespace v2
} // namespace streampress

#endif // SPARSEPRESS_V2_HPP
