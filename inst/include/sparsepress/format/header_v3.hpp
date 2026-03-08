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

#include <sparsepress/format/header_v2.hpp>  // reuse Footer_v2, FOOTER_MAGIC, etc.

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

namespace sparsepress {
namespace v3 {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint8_t  HEADER_MAGIC_V3[4] = {'S', 'P', 'R', 'Z'};
static constexpr uint16_t FORMAT_VERSION_V3  = 3;
static constexpr uint16_t HEADER_SIZE_V3     = 128;
static constexpr uint32_t DEFAULT_CHUNK_COLS = 256;

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
    uint8_t  reserved[48];

    FileHeader_v3() {
        std::memset(this, 0, sizeof(FileHeader_v3));
        std::memcpy(magic, HEADER_MAGIC_V3, 4);
        version = FORMAT_VERSION_V3;
        header_size = HEADER_SIZE_V3;
        chunk_cols = DEFAULT_CHUNK_COLS;
        value_type = static_cast<uint8_t>(DenseValueType::FLOAT32);
    }

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
// Dense Chunk Descriptor (24 bytes)
// ============================================================================

struct DenseChunkDescriptor {
    uint32_t col_start;          // first column index in the full matrix
    uint32_t num_cols;           // columns in this chunk (≤ chunk_cols)
    uint64_t byte_offset;        // offset into data section (relative to data_offset)
    uint64_t byte_size;          // size in bytes = m * num_cols * value_bytes

    DenseChunkDescriptor() {
        std::memset(this, 0, sizeof(DenseChunkDescriptor));
    }
};

static_assert(sizeof(DenseChunkDescriptor) == 24, "DenseChunkDescriptor must be 24 bytes");

}  // namespace v3
}  // namespace sparsepress

