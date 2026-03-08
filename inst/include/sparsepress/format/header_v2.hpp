/**
 * @file header_v2.hpp
 * @brief SparsePress v2 format specification.
 *
 * Chunked format with 256-column chunks, random-access decompression,
 * metadata support (dimnames, row permutation), and multiple value types
 * including fp32, fp16, and quant8.
 *
 * File layout:
 *   [FileHeader_v2: 128 bytes]
 *   [ChunkDescriptor_v2 × num_chunks]
 *   [rANS decode tables]
 *   [Chunk data (gap + value streams per chunk)]
 *   [Transpose chunk data (optional)]
 *   [Metadata section (dimnames, row permutation)]
 *   [Footer: 16 bytes]
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace sparsepress {
namespace v2 {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint8_t  HEADER_MAGIC_V2[4] = {'S', 'P', 'R', 'Z'};
static constexpr uint16_t FORMAT_VERSION_V2  = 2;
static constexpr uint16_t HEADER_SIZE_V2     = 128;
static constexpr uint32_t DEFAULT_CHUNK_COLS = 256;
static constexpr uint8_t  FOOTER_MAGIC[4]    = {'S', 'P', 'E', 'N'};  // "SPEN" = SParsepress END
static constexpr size_t   FOOTER_SIZE        = 16;

// ============================================================================
// Value types
// ============================================================================

enum class ValueType_v2 : uint8_t {
    UINT8    = 0,   // 1-byte unsigned integer (count data)
    UINT16   = 1,   // 2-byte unsigned integer (count data)
    UINT32   = 2,   // 4-byte unsigned integer (count data)
    FLOAT32  = 3,   // 4-byte IEEE 754 float (default for continuous)
    FLOAT16  = 4,   // 2-byte IEEE 754 half (lossy, user-requested)
    QUANT8   = 5,   // 1-byte affine quantized (per-chunk scale + offset)
    FLOAT64  = 6,   // 8-byte IEEE 754 double (legacy, discouraged)
};

inline const char* value_type_name(ValueType_v2 vt) {
    switch (vt) {
        case ValueType_v2::UINT8:   return "uint8";
        case ValueType_v2::UINT16:  return "uint16";
        case ValueType_v2::UINT32:  return "uint32";
        case ValueType_v2::FLOAT32: return "float32";
        case ValueType_v2::FLOAT16: return "float16";
        case ValueType_v2::QUANT8:  return "quant8";
        case ValueType_v2::FLOAT64: return "float64";
        default: return "unknown";
    }
}

inline uint8_t value_type_bytes(ValueType_v2 vt) {
    switch (vt) {
        case ValueType_v2::UINT8:   return 1;
        case ValueType_v2::UINT16:  return 2;
        case ValueType_v2::UINT32:  return 4;
        case ValueType_v2::FLOAT32: return 4;
        case ValueType_v2::FLOAT16: return 2;
        case ValueType_v2::QUANT8:  return 1;
        case ValueType_v2::FLOAT64: return 8;
        default: return 0;
    }
}

// ============================================================================
// Compression level
// ============================================================================

enum class CompressionLevel : uint8_t {
    FAST    = 0,   // Fastest encode, still fast decode
    DEFAULT = 1,   // Balanced
    HIGH    = 2,   // Maximum compression, slower encode
};

// ============================================================================
// Stream types within each chunk
// ============================================================================

enum StreamType_v2 : uint32_t {
    STREAM_GAPS     = 0,   // Gap-encoded row indices
    STREAM_VALUES   = 1,   // Compressed values
    NUM_STREAMS_V2  = 2
};

// ============================================================================
// Metadata entry types
// ============================================================================

enum class MetadataKey : uint8_t {
    ROWNAMES        = 0,
    COLNAMES        = 1,
    ROW_PERMUTATION = 2,
    CUSTOM          = 3,
};

// ============================================================================
// File Header (128 bytes)
// ============================================================================

struct FileHeader_v2 {
    // Identity (8 bytes)
    uint8_t  magic[4];           // "SPRZ"
    uint16_t version;            // 2
    uint16_t header_size;        // 128 (for forward compat)

    // Dimensions (20 bytes)
    uint32_t m;                  // rows
    uint32_t n;                  // columns
    uint64_t nnz;                // total non-zeros
    uint32_t chunk_cols;         // columns per chunk (256)

    // Chunk layout (12 bytes)
    uint32_t num_chunks;         // ceil(n / chunk_cols)
    uint32_t num_tables;         // number of rANS decode tables
    uint32_t table_log;          // log2(table_size), e.g. 11

    // Value encoding (8 bytes)
    uint8_t  value_type;         // ValueType_v2 enum
    uint8_t  compression_level;  // CompressionLevel enum
    uint8_t  row_sorted;         // 1 if rows sorted by nnz
    uint8_t  col_sorted;         // 1 if columns sorted by nnz
    uint32_t most_common_value;  // for flag-based value encoding

    // Section offsets (40 bytes)
    uint64_t chunk_index_offset; // byte offset to chunk index
    uint64_t tables_offset;      // byte offset to rANS tables
    uint64_t data_offset;        // byte offset to chunk data
    uint64_t transpose_offset;   // byte offset to transpose data (0 = none)
    uint64_t metadata_offset;    // byte offset to metadata section

    // Statistics (8 bytes)
    uint32_t max_value;          // max integer value (for integer types)
    float    density;            // nnz / (m * n), stored for quick reference

    // Reserved (32 bytes)
    uint8_t  reserved[32];       // future use, zero-filled

    FileHeader_v2() {
        std::memset(this, 0, sizeof(FileHeader_v2));
        std::memcpy(magic, HEADER_MAGIC_V2, 4);
        version = FORMAT_VERSION_V2;
        header_size = HEADER_SIZE_V2;
        chunk_cols = DEFAULT_CHUNK_COLS;
        table_log = 11;
    }

    bool valid() const {
        return std::memcmp(magic, HEADER_MAGIC_V2, 4) == 0
            && version == FORMAT_VERSION_V2;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf(HEADER_SIZE_V2, 0);
        std::memcpy(buf.data(), this, HEADER_SIZE_V2);
        return buf;
    }

    static FileHeader_v2 deserialize(const uint8_t* buf) {
        // Verify magic
        if (std::memcmp(buf, HEADER_MAGIC_V2, 4) != 0) {
            throw std::runtime_error("Invalid magic bytes (not a SparsePress file)");
        }
        // Check version
        uint16_t ver;
        std::memcpy(&ver, buf + 4, 2);
        if (ver != FORMAT_VERSION_V2) {
            throw std::runtime_error("Not a v2 SparsePress file (version=" +
                                     std::to_string(ver) + ")");
        }

        FileHeader_v2 h;
        std::memcpy(&h, buf, HEADER_SIZE_V2);
        return h;
    }
};

static_assert(sizeof(FileHeader_v2) == 128, "FileHeader_v2 must be 128 bytes");

// ============================================================================
// Chunk Descriptor (48 bytes)
// ============================================================================

struct ChunkDescriptor_v2 {
    uint32_t col_start;          // first column index
    uint32_t num_cols;           // columns in this chunk (≤ 256)
    uint32_t nnz;                // non-zeros in this chunk

    // Stream layout (offsets relative to data_offset)
    uint32_t stream_offset[NUM_STREAMS_V2];
    uint32_t stream_size[NUM_STREAMS_V2];

    // Decoded sizes for allocation
    uint32_t decoded_gap_bytes;
    uint32_t decoded_value_bytes;

    // Quantization parameters (for QUANT8 value type)
    float    quant_scale;        // per-chunk scale factor
    float    quant_offset;       // per-chunk offset

    // Padding to 48 bytes
    uint32_t reserved;

    ChunkDescriptor_v2() {
        std::memset(this, 0, sizeof(ChunkDescriptor_v2));
    }
};

static_assert(sizeof(ChunkDescriptor_v2) == 48, "ChunkDescriptor_v2 must be 48 bytes");

// ============================================================================
// Footer (16 bytes) — at end of file
// ============================================================================

struct Footer_v2 {
    uint32_t metadata_size;      // size of metadata section in bytes
    uint32_t file_crc32;         // CRC32 of entire file (everything before footer)
    uint32_t total_chunks;       // redundant with header (for quick validation)
    uint8_t  footer_magic[4];    // "SPEN"

    Footer_v2() {
        std::memset(this, 0, sizeof(Footer_v2));
        std::memcpy(footer_magic, FOOTER_MAGIC, 4);
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf(FOOTER_SIZE, 0);
        std::memcpy(buf.data(), this, FOOTER_SIZE);
        return buf;
    }

    static Footer_v2 deserialize(const uint8_t* buf) {
        Footer_v2 f;
        std::memcpy(&f, buf, FOOTER_SIZE);
        if (std::memcmp(f.footer_magic, FOOTER_MAGIC, 4) != 0) {
            throw std::runtime_error("Invalid footer magic (file may be truncated)");
        }
        return f;
    }

    bool valid() const {
        return std::memcmp(footer_magic, FOOTER_MAGIC, 4) == 0;
    }
};

static_assert(sizeof(Footer_v2) == 16, "Footer_v2 must be 16 bytes");

// ============================================================================
// Metadata section — variable-length, self-describing
// ============================================================================

struct MetadataEntry {
    MetadataKey key;
    std::vector<uint8_t> data;
};

struct Metadata {
    std::vector<MetadataEntry> entries;

    // ---------- Builders ----------

    void set_rownames(const std::vector<std::string>& names) {
        MetadataEntry e;
        e.key = MetadataKey::ROWNAMES;
        for (const auto& name : names) {
            e.data.insert(e.data.end(), name.begin(), name.end());
            e.data.push_back('\0');
        }
        // Replace existing or append
        for (auto& existing : entries) {
            if (existing.key == MetadataKey::ROWNAMES) {
                existing = e;
                return;
            }
        }
        entries.push_back(e);
    }

    void set_colnames(const std::vector<std::string>& names) {
        MetadataEntry e;
        e.key = MetadataKey::COLNAMES;
        for (const auto& name : names) {
            e.data.insert(e.data.end(), name.begin(), name.end());
            e.data.push_back('\0');
        }
        for (auto& existing : entries) {
            if (existing.key == MetadataKey::COLNAMES) {
                existing = e;
                return;
            }
        }
        entries.push_back(e);
    }

    void set_row_permutation(const std::vector<uint32_t>& perm) {
        MetadataEntry e;
        e.key = MetadataKey::ROW_PERMUTATION;
        e.data.resize(perm.size() * sizeof(uint32_t));
        std::memcpy(e.data.data(), perm.data(), e.data.size());
        for (auto& existing : entries) {
            if (existing.key == MetadataKey::ROW_PERMUTATION) {
                existing = e;
                return;
            }
        }
        entries.push_back(e);
    }

    // ---------- Getters ----------

    std::vector<std::string> get_names(MetadataKey key) const {
        for (const auto& e : entries) {
            if (e.key == key) {
                std::vector<std::string> names;
                const char* ptr = reinterpret_cast<const char*>(e.data.data());
                const char* end = ptr + e.data.size();
                while (ptr < end) {
                    std::string s(ptr);
                    names.push_back(s);
                    ptr += s.size() + 1;
                }
                return names;
            }
        }
        return {};
    }

    std::vector<std::string> get_rownames() const {
        return get_names(MetadataKey::ROWNAMES);
    }

    std::vector<std::string> get_colnames() const {
        return get_names(MetadataKey::COLNAMES);
    }

    std::vector<uint32_t> get_row_permutation() const {
        for (const auto& e : entries) {
            if (e.key == MetadataKey::ROW_PERMUTATION) {
                size_t count = e.data.size() / sizeof(uint32_t);
                std::vector<uint32_t> perm(count);
                std::memcpy(perm.data(), e.data.data(), e.data.size());
                return perm;
            }
        }
        return {};
    }

    bool has_rownames() const {
        for (const auto& e : entries) {
            if (e.key == MetadataKey::ROWNAMES) return true;
        }
        return false;
    }

    bool has_colnames() const {
        for (const auto& e : entries) {
            if (e.key == MetadataKey::COLNAMES) return true;
        }
        return false;
    }

    bool has_row_permutation() const {
        for (const auto& e : entries) {
            if (e.key == MetadataKey::ROW_PERMUTATION) return true;
        }
        return false;
    }

    // ---------- Serialization ----------

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf;

        // Number of entries
        uint32_t n = static_cast<uint32_t>(entries.size());
        buf.resize(4);
        std::memcpy(buf.data(), &n, 4);

        for (const auto& e : entries) {
            // Key type (1 byte)
            buf.push_back(static_cast<uint8_t>(e.key));

            // Data length (4 bytes)
            uint32_t len = static_cast<uint32_t>(e.data.size());
            size_t pos = buf.size();
            buf.resize(pos + 4);
            std::memcpy(buf.data() + pos, &len, 4);

            // Data
            buf.insert(buf.end(), e.data.begin(), e.data.end());
        }

        return buf;
    }

    static Metadata deserialize(const uint8_t* buf, size_t buf_size) {
        Metadata meta;
        if (buf_size < 4) return meta;

        const uint8_t* p = buf;
        const uint8_t* end = buf + buf_size;

        uint32_t n;
        std::memcpy(&n, p, 4); p += 4;

        for (uint32_t i = 0; i < n && p < end; ++i) {
            MetadataEntry e;
            e.key = static_cast<MetadataKey>(*p++);

            if (p + 4 > end) break;
            uint32_t len;
            std::memcpy(&len, p, 4); p += 4;

            if (p + len > end) break;
            e.data.assign(p, p + len);
            p += len;

            meta.entries.push_back(std::move(e));
        }

        return meta;
    }
};

// ============================================================================
// CSCMatrix_v2 — extended with float32 values
// ============================================================================

struct CSCMatrix_v2 {
    uint32_t m = 0;
    uint32_t n = 0;
    uint64_t nnz = 0;

    std::vector<uint32_t> p;       // column pointers, size n+1
    std::vector<uint32_t> i;       // row indices, size nnz
    std::vector<float>    x_f32;   // float32 values, size nnz (primary)
    std::vector<double>   x_f64;   // float64 values (only if FLOAT64 type)

    ValueType_v2 stored_type = ValueType_v2::FLOAT32;

    Metadata metadata;

    CSCMatrix_v2() = default;

    CSCMatrix_v2(uint32_t m_, uint32_t n_, uint64_t nnz_)
        : m(m_), n(n_), nnz(nnz_), p(n_ + 1), i(nnz_), x_f32(nnz_) {}

    // Get value as float32 (always available)
    float value_f32(uint64_t idx) const {
        if (stored_type == ValueType_v2::FLOAT64) {
            return static_cast<float>(x_f64[idx]);
        }
        return x_f32[idx];
    }

    // Get value as float64
    double value_f64(uint64_t idx) const {
        if (stored_type == ValueType_v2::FLOAT64) {
            return x_f64[idx];
        }
        return static_cast<double>(x_f32[idx]);
    }

    size_t raw_size() const {
        size_t val_bytes = (stored_type == ValueType_v2::FLOAT64) ? 8 : 4;
        return 12 + (n + 1) * 4 + nnz * 4 + nnz * val_bytes;
    }
};

// ============================================================================
// Classify values — determine best value type
// ============================================================================

inline ValueType_v2 classify_values_v2(const double* values, uint64_t nnz,
                                        const std::string& precision) {
    if (precision == "fp64") return ValueType_v2::FLOAT64;
    if (precision == "fp16") return ValueType_v2::FLOAT16;
    if (precision == "quant8") return ValueType_v2::QUANT8;
    if (precision == "fp32") {
        // Check if all values are non-negative integers that fit in integer types
        bool all_integer = true;
        bool all_nonneg = true;
        double max_val = 0;
        for (uint64_t k = 0; k < nnz && (all_integer || all_nonneg); ++k) {
            double v = values[k];
            if (v < 0) all_nonneg = false;
            if (v != std::floor(v) || v < 0) all_integer = false;
            if (v > max_val) max_val = v;
        }
        if (all_integer && all_nonneg) {
            if (max_val <= 255) return ValueType_v2::UINT8;
            if (max_val <= 65535) return ValueType_v2::UINT16;
            if (max_val <= 4294967295.0) return ValueType_v2::UINT32;
        }
        return ValueType_v2::FLOAT32;
    }

    // "auto" mode
    bool all_integer = true;
    bool all_nonneg = true;
    double max_val = 0;
    for (uint64_t k = 0; k < nnz; ++k) {
        double v = values[k];
        if (v < 0) all_nonneg = false;
        if (v != std::floor(v) || v < 0) all_integer = false;
        if (v > max_val) max_val = v;
    }
    if (all_integer && all_nonneg) {
        if (max_val <= 255) return ValueType_v2::UINT8;
        if (max_val <= 65535) return ValueType_v2::UINT16;
        if (max_val <= 4294967295.0) return ValueType_v2::UINT32;
    }
    return ValueType_v2::FLOAT32;
}

// ============================================================================
// Row sorting utilities
// ============================================================================

struct RowSortResult {
    std::vector<uint32_t> perm;          // original_row → sorted_row
    std::vector<uint32_t> inv_perm;      // sorted_row → original_row
};

/**
 * Sort rows by ascending nnz count.
 * Returns permutation vectors for reindexing.
 */
inline RowSortResult compute_row_sort(const uint32_t* col_ptrs,
                                       const uint32_t* row_indices,
                                       uint32_t m, uint32_t n, uint64_t nnz) {
    // Count nnz per row
    std::vector<uint32_t> row_nnz(m, 0);
    for (uint64_t k = 0; k < nnz; ++k) {
        row_nnz[row_indices[k]]++;
    }

    // Create sort permutation (ascending nnz)
    RowSortResult result;
    result.inv_perm.resize(m);
    for (uint32_t i = 0; i < m; ++i) result.inv_perm[i] = i;

    std::sort(result.inv_perm.begin(), result.inv_perm.end(),
              [&](uint32_t a, uint32_t b) {
                  return row_nnz[a] < row_nnz[b];
              });

    // Inverse permutation: perm[original] = sorted_position
    result.perm.resize(m);
    for (uint32_t sorted = 0; sorted < m; ++sorted) {
        result.perm[result.inv_perm[sorted]] = sorted;
    }

    return result;
}

/**
 * Apply row permutation to row indices in-place.
 */
inline void apply_row_permutation(uint32_t* row_indices, uint64_t nnz,
                                   const std::vector<uint32_t>& perm) {
    for (uint64_t k = 0; k < nnz; ++k) {
        row_indices[k] = perm[row_indices[k]];
    }
}

/**
 * Undo row permutation on row indices in-place.
 */
inline void undo_row_permutation(uint32_t* row_indices, uint64_t nnz,
                                  const std::vector<uint32_t>& inv_perm) {
    for (uint64_t k = 0; k < nnz; ++k) {
        row_indices[k] = inv_perm[row_indices[k]];
    }
}

// ============================================================================
// Half-precision (fp16) conversion utilities
// ============================================================================

/**
 * Convert float32 to IEEE 754 half-precision (lossy).
 * Uses standard rounding. Handles denorms, inf, nan.
 */
inline uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);

    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;

    uint16_t h_sign = sign << 15;

    if (exp > 15) {
        // Overflow → infinity
        return h_sign | 0x7C00;
    }
    if (exp < -14) {
        // Underflow → zero (skip denormals for speed)
        return h_sign;
    }

    uint16_t h_exp  = static_cast<uint16_t>((exp + 15) & 0x1F) << 10;
    uint16_t h_frac = static_cast<uint16_t>(frac >> 13);

    // Round to nearest even
    uint32_t round_bit = (frac >> 12) & 1;
    uint32_t sticky = frac & 0xFFF;
    if (round_bit && (sticky || (h_frac & 1))) {
        h_frac++;
        if (h_frac > 0x3FF) {
            h_frac = 0;
            h_exp += (1 << 10);
        }
    }

    return h_sign | h_exp | h_frac;
}

/**
 * Convert IEEE 754 half-precision to float32.
 */
inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign << 31;
        } else {
            // Denormalized
            exp = 1;
            while (!(frac & 0x400)) { frac <<= 1; exp--; }
            frac &= 0x3FF;
            bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }

    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// ============================================================================
// File layout computation
// ============================================================================

struct FileLayout_v2 {
    uint64_t header_offset;       // 0
    uint64_t chunk_index_offset;  // 128
    uint64_t tables_offset;
    uint64_t data_offset;
    uint64_t transpose_offset;    // 0 if no transpose
    uint64_t metadata_offset;
    uint64_t footer_offset;

    static FileLayout_v2 compute(const FileHeader_v2& hdr,
                                  uint64_t tables_size,
                                  uint64_t data_size,
                                  uint64_t transpose_size,
                                  uint64_t metadata_size) {
        FileLayout_v2 layout;
        layout.header_offset = 0;
        layout.chunk_index_offset = HEADER_SIZE_V2;
        layout.tables_offset = layout.chunk_index_offset +
            static_cast<uint64_t>(hdr.num_chunks) * sizeof(ChunkDescriptor_v2);
        layout.data_offset = layout.tables_offset + tables_size;
        if (transpose_size > 0) {
            layout.transpose_offset = layout.data_offset + data_size;
            layout.metadata_offset = layout.transpose_offset + transpose_size;
        } else {
            layout.transpose_offset = 0;
            layout.metadata_offset = layout.data_offset + data_size;
        }
        layout.footer_offset = layout.metadata_offset + metadata_size;
        return layout;
    }
};

} // namespace v2
} // namespace sparsepress
