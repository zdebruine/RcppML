// sparsepress - CRC32 checksum (no external dependencies)
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_CHECKSUM_HPP
#define SPARSEPRESS_CHECKSUM_HPP

#include <cstdint>
#include <cstddef>
#include <vector>

namespace streampress {

// ============================================================================
// CRC32 with precomputed table (ITU V.42 / ISO 3309 polynomial 0xEDB88320)
// ============================================================================
class CRC32 {
public:
    CRC32() : crc_(0xFFFFFFFF) {
        init_table();
    }

    void update(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            crc_ = table_[(crc_ ^ data[i]) & 0xFF] ^ (crc_ >> 8);
        }
    }

    void update(const std::vector<uint8_t>& data) {
        update(data.data(), data.size());
    }

    uint32_t finalize() const {
        return crc_ ^ 0xFFFFFFFF;
    }

    void reset() {
        crc_ = 0xFFFFFFFF;
    }

    // One-shot
    static uint32_t compute(const uint8_t* data, size_t len) {
        CRC32 c;
        c.update(data, len);
        return c.finalize();
    }

    static uint32_t compute(const std::vector<uint8_t>& data) {
        return compute(data.data(), data.size());
    }

private:
    uint32_t crc_;
    static uint32_t table_[256];
    static bool table_init_;

    static void init_table() {
        if (table_init_) return;
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                if (c & 1)
                    c = 0xEDB88320u ^ (c >> 1);
                else
                    c >>= 1;
            }
            table_[i] = c;
        }
        table_init_ = true;
    }
};

// Static member initialization
inline uint32_t CRC32::table_[256] = {};
inline bool CRC32::table_init_ = false;

} // namespace streampress

#endif // SPARSEPRESS_CHECKSUM_HPP
