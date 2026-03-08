/**
 * @file header_v3.hpp
 * @brief SparsePress v3 format specification — dense column-panel storage.
 *
 * Extends SparsePress to handle dense matrices via chunked column-panels.
 * Each chunk stores a contiguous column-major block of m × chunk_cols values.
 * No gap/index encoding is needed (all entries are stored).
 *
 * File layout:
 *   [FileHeader_v3: 128 bytes]
 *   [DenseChunkDescriptor × num_chunks]
 *   [Dense panel data (column-major)]
 *   [Transpose panel data (optional, column-major)]
 *   [Metadata section (dimnames, etc.)]
 *   [Footer_v2: 16 bytes]  (reuses v2 footer format)
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <streampress/format/header_v2.hpp>  // reuse Footer_v2, FOOTER_MAGIC, etc.

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

namespace streampress {
namespace v3 {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint8_t  HEADER_MAGIC_V3[4] = {'S', 'P', 'R', 'Z'};
static constexpr uint16_t FORMAT_VERSION_V3  = 3;
static constexpr uint16_t HEADER_SIZE_V3     = 128;
static constexpr uint32_t DEFAULT_CHUNK_COLS = 2048;

// ============================================================================
// Dense value types (subset of v2 types that make sense for dense)
// ============================================================================

enum class DenseValueType : uint8_t {
    FLOAT32 = 3,   // 4-byte IEEE 754 float (default)
    FLOAT64 = 6,   // 8-byte IEEE 754 double
};

inline uint8_t dense_value_bytes(DenseValueType vt) {
    switch (vt) {
        case DenseValueType::FLOAT32: return 4;
        case DenseValueType::FLOAT64: return 8;
        default: return 0;
    }
}

inline const char* dense_value_type_name(DenseValueType vt) {
    switch (vt) {
        case DenseValueType::FLOAT32: return "float32";
        case DenseValueType::FLOAT64: return "float64";
        default: return "unknown";
    }
}

// ============================================================================
// Dense codec for compressed v3 panels
// Stored in FileHeader_v3::reserved[0]
// ============================================================================

enum class DenseCodec : uint8_t {
    RAW_FP32  = 0,  // original format, no compression (backwards compatible)
    FP16      = 1,  // 50% size, lossy fp16 truncation
    QUANT8    = 2,  // 75% size, 8-bit range quantization
    FP16_RANS = 3,  // fp16 + XOR-column-delta + rANS entropy coding
    FP32_RANS = 4,  // fp32 + XOR-column-delta + rANS entropy coding
};

inline const char* dense_codec_name(DenseCodec c) {
    switch (c) {
        case DenseCodec::RAW_FP32:  return "raw_fp32";
        case DenseCodec::FP16:      return "fp16";
        case DenseCodec::QUANT8:    return "quant8";
        case DenseCodec::FP16_RANS: return "fp16_rans";
        case DenseCodec::FP32_RANS: return "fp32_rans";
        default: return "unknown";
    }
}

inline DenseCodec parse_dense_codec(const std::string& s) {
    if (s == "raw" || s == "raw_fp32" || s == "none") return DenseCodec::RAW_FP32;
    if (s == "fp16")      return DenseCodec::FP16;
    if (s == "quant8")    return DenseCodec::QUANT8;
    if (s == "fp16_rans") return DenseCodec::FP16_RANS;
    if (s == "fp32_rans") return DenseCodec::FP32_RANS;
    return DenseCodec::RAW_FP32;  // default
}

// ============================================================================
// File Header (128 bytes)
// ============================================================================

struct FileHeader_v3 {
    // Identity (8 bytes)
    uint8_t  magic[4];           // "SPRZ"
    uint16_t version;            // 3
    uint16_t header_size;        // 128

    // Dimensions (20 bytes)
    uint32_t m;                  // rows
    uint32_t n;                  // columns
    uint64_t nnz;                // m * n (all entries, since dense)
    uint32_t chunk_cols;         // columns per chunk (default 256)

    // Chunk layout (8 bytes)
    uint32_t num_chunks;         // ceil(n / chunk_cols)
    uint32_t num_transpose_chunks; // ceil(m / chunk_cols), or 0 if no transpose

    // Value encoding (4 bytes)
    uint8_t  value_type;         // DenseValueType enum
    uint8_t  has_transpose;      // 1 if transpose panels are stored
    uint8_t  reserved_flags[2];  // future use

    // Section offsets (40 bytes)
    uint64_t chunk_index_offset; // byte offset to chunk descriptor array
    uint64_t data_offset;        // byte offset to forward panel data
    uint64_t transpose_index_offset; // byte offset to transpose chunk descriptors (0 = none)
    uint64_t transpose_data_offset;  // byte offset to transpose panel data (0 = none)
    uint64_t metadata_offset;    // byte offset to metadata section (0 = none)

    // Reserved (48 bytes — pad to 128 total)
    // reserved[0]: DenseCodec enum (0 = RAW_FP32, default, backwards compat)
    // reserved[1]: delta_encode flag (0 = none, 1 = XOR-delta between columns)
    // reserved[2..47]: future use
    uint8_t  reserved[48];

    FileHeader_v3() {
        std::memset(this, 0, sizeof(FileHeader_v3));
        std::memcpy(magic, HEADER_MAGIC_V3, 4);
        version = FORMAT_VERSION_V3;
        header_size = HEADER_SIZE_V3;
        chunk_cols = DEFAULT_CHUNK_COLS;
        value_type = static_cast<uint8_t>(DenseValueType::FLOAT32);
    }

    /// Get the dense codec from reserved[0]
    DenseCodec codec() const { return static_cast<DenseCodec>(reserved[0]); }
    void set_codec(DenseCodec c) { reserved[0] = static_cast<uint8_t>(c); }

    /// Get the delta encoding flag from reserved[1]
    bool delta_encode() const { return reserved[1] != 0; }
    void set_delta_encode(bool d) { reserved[1] = d ? 1 : 0; }

    bool valid() const {
        return std::memcmp(magic, HEADER_MAGIC_V3, 4) == 0
            && version == FORMAT_VERSION_V3;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf(HEADER_SIZE_V3, 0);
        std::memcpy(buf.data(), this, HEADER_SIZE_V3);
        return buf;
    }

    static FileHeader_v3 deserialize(const uint8_t* buf) {
        if (std::memcmp(buf, HEADER_MAGIC_V3, 4) != 0) {
            throw std::runtime_error("Invalid magic bytes (not a SparsePress file)");
        }
        uint16_t ver;
        std::memcpy(&ver, buf + 4, 2);
        if (ver != FORMAT_VERSION_V3) {
            throw std::runtime_error("Not a v3 SparsePress file (version=" +
                                     std::to_string(ver) + ")");
        }
        FileHeader_v3 h;
        std::memcpy(&h, buf, HEADER_SIZE_V3);
        return h;
    }
};

static_assert(sizeof(FileHeader_v3) == 128, "FileHeader_v3 must be 128 bytes");

// ============================================================================
// Dense Chunk Descriptor (24 bytes) — used for RAW_FP32 (codec=0)
// ============================================================================

struct DenseChunkDescriptor {
    uint32_t col_start;          // first column index in the full matrix
    uint32_t num_cols;           // columns in this chunk (≤ chunk_cols)
    uint64_t byte_offset;        // offset into data section (relative to data_offset)
    uint64_t byte_size;          // size in bytes = m * num_cols * value_bytes (or compressed)

    DenseChunkDescriptor() {
        std::memset(this, 0, sizeof(DenseChunkDescriptor));
    }
};

static_assert(sizeof(DenseChunkDescriptor) == 24, "DenseChunkDescriptor must be 24 bytes");

// ============================================================================
// Extended Dense Chunk Descriptor (40 bytes) — used for compressed codecs
// ============================================================================

struct DenseChunkDescriptorExt {
    uint32_t col_start;          // first column index
    uint32_t num_cols;           // number of columns
    uint32_t _pad;               // alignment padding
    uint32_t _pad2;              // alignment padding
    uint64_t byte_offset;        // file offset of compressed bytes
    uint64_t byte_size;          // compressed byte size
    uint64_t uncompressed_size;  // uncompressed byte size (m * num_cols * val_bytes)

    DenseChunkDescriptorExt() {
        std::memset(this, 0, sizeof(DenseChunkDescriptorExt));
    }

    /// Convert from base descriptor (for RAW_FP32: compressed == uncompressed)
    explicit DenseChunkDescriptorExt(const DenseChunkDescriptor& d) {
        col_start = d.col_start;
        num_cols = d.num_cols;
        _pad = 0;
        _pad2 = 0;
        byte_offset = d.byte_offset;
        byte_size = d.byte_size;
        uncompressed_size = d.byte_size;  // RAW: same
    }
};

static_assert(sizeof(DenseChunkDescriptorExt) == 40, "DenseChunkDescriptorExt size check");

}  // namespace v3
}  // namespace streampress

