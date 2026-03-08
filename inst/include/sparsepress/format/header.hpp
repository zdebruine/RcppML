// sparsepress - Binary format header definition
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_HEADER_HPP
#define SPARSEPRESS_HEADER_HPP

#include <sparsepress/transform/value_map.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace sparsepress {

// ============================================================================
// SparsePress compressed file format
//
// Layout:
//   [Header: 64 bytes fixed]
//   [Model section: variable]
//   [Structure section: variable (encoded gaps)]
//   [Values section: variable (encoded values)]
//
// Header fields (64 bytes):
//   magic[4]      = "SPRZ"
//   version       = uint16
//   flags         = uint16
//   m             = uint32 (rows)
//   n             = uint32 (cols)
//   nnz           = uint64
//   max_value     = uint32
//   value_type    = uint8
//   rice_block_sz = uint16
//   density_blocks= uint16
//   _reserved     = uint8
//   prng_seed     = uint64
//   model_size    = uint32 (bytes)
//   struct_size   = uint32 (bytes)
//   values_size   = uint32 (bytes)
//   col_counts_sz = uint32 (bytes)
//   struct_k_size = uint32 (bytes) -- rice k params for structure
//   crc32         = uint32
//   _pad          = 6 bytes (total=72)
// ============================================================================

static constexpr uint8_t HEADER_MAGIC[4] = {'S', 'P', 'R', 'Z'};
static constexpr uint16_t FORMAT_VERSION = 1;
static constexpr size_t HEADER_SIZE = 72;

// Flags
static constexpr uint16_t FLAG_DELTA_PREDICTION = 0x0001;
static constexpr uint16_t FLAG_VALUE_PREDICTION = 0x0002;
static constexpr uint16_t FLAG_INTEGER_VALUES   = 0x0004;

struct FileHeader {
    uint8_t  magic[4];
    uint16_t version;
    uint16_t flags;
    uint32_t m;
    uint32_t n;
    uint64_t nnz;
    uint32_t max_value;
    uint8_t  value_type;       // ValueType enum
    uint16_t rice_block_size;
    uint16_t density_blocks;
    uint8_t  reserved1;
    uint64_t prng_seed;
    uint32_t model_size;
    uint32_t struct_size;
    uint32_t values_size;
    uint32_t col_counts_size;
    uint32_t struct_k_size;
    uint32_t crc32;

    FileHeader() {
        std::memset(this, 0, sizeof(FileHeader));
        std::memcpy(magic, HEADER_MAGIC, 4);
        version = FORMAT_VERSION;
    }

    // Serialize to 64 bytes
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf(HEADER_SIZE, 0);
        uint8_t* p = buf.data();

        std::memcpy(p, magic, 4); p += 4;
        std::memcpy(p, &version, 2); p += 2;
        std::memcpy(p, &flags, 2); p += 2;
        std::memcpy(p, &m, 4); p += 4;
        std::memcpy(p, &n, 4); p += 4;
        std::memcpy(p, &nnz, 8); p += 8;
        std::memcpy(p, &max_value, 4); p += 4;
        *p++ = value_type;
        std::memcpy(p, &rice_block_size, 2); p += 2;
        std::memcpy(p, &density_blocks, 2); p += 2;
        *p++ = reserved1;
        std::memcpy(p, &prng_seed, 8); p += 8;
        std::memcpy(p, &model_size, 4); p += 4;
        std::memcpy(p, &struct_size, 4); p += 4;
        std::memcpy(p, &values_size, 4); p += 4;
        std::memcpy(p, &col_counts_size, 4); p += 4;
        std::memcpy(p, &struct_k_size, 4); p += 4;
        std::memcpy(p, &crc32, 4); p += 4;

        return buf;
    }

    // Deserialize from 64 bytes
    static FileHeader deserialize(const uint8_t* buf) {
        FileHeader h;
        const uint8_t* p = buf;

        std::memcpy(h.magic, p, 4); p += 4;
        if (std::memcmp(h.magic, HEADER_MAGIC, 4) != 0)
            throw std::runtime_error("Invalid magic bytes (not a SparsePress file)");

        std::memcpy(&h.version, p, 2); p += 2;
        if (h.version != FORMAT_VERSION)
            throw std::runtime_error("Unsupported format version");

        std::memcpy(&h.flags, p, 2); p += 2;
        std::memcpy(&h.m, p, 4); p += 4;
        std::memcpy(&h.n, p, 4); p += 4;
        std::memcpy(&h.nnz, p, 8); p += 8;
        std::memcpy(&h.max_value, p, 4); p += 4;
        h.value_type = *p++;
        std::memcpy(&h.rice_block_size, p, 2); p += 2;
        std::memcpy(&h.density_blocks, p, 2); p += 2;
        h.reserved1 = *p++;
        std::memcpy(&h.prng_seed, p, 8); p += 8;
        std::memcpy(&h.model_size, p, 4); p += 4;
        std::memcpy(&h.struct_size, p, 4); p += 4;
        std::memcpy(&h.values_size, p, 4); p += 4;
        std::memcpy(&h.col_counts_size, p, 4); p += 4;
        std::memcpy(&h.struct_k_size, p, 4); p += 4;
        std::memcpy(&h.crc32, p, 4); p += 4;

        return h;
    }
};

} // namespace sparsepress

#endif // SPARSEPRESS_HEADER_HPP
