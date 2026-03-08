// sparsepress - Lossless double-to-integer value mapping
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_VALUE_MAP_HPP
#define SPARSEPRESS_VALUE_MAP_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

namespace streampress {

// ============================================================================
// Value type categories for optimal encoding
// ============================================================================
enum class ValueType : uint8_t {
    UINT8   = 0,   // All values fit in [0, 255]
    UINT16  = 1,   // All values fit in [0, 65535]
    UINT32  = 2,   // All values fit in [0, 2^32-1]
    FLOAT64 = 3,   // Non-integer or negative values (fallback)
};

// Determine narrowest value type
inline ValueType classify_values(const double* x, size_t nnz, uint32_t max_value,
                                  bool all_integer, bool all_nonneg) {
    if (!all_integer || !all_nonneg)
        return ValueType::FLOAT64;
    if (max_value <= 255)
        return ValueType::UINT8;
    if (max_value <= 65535)
        return ValueType::UINT16;
    return ValueType::UINT32;
}

// ============================================================================
// Convert double array to uint32 array (lossless for integer data)
// ============================================================================
inline std::vector<uint32_t> doubles_to_ints(const double* x, size_t nnz) {
    std::vector<uint32_t> out(nnz);
    for (size_t k = 0; k < nnz; ++k) {
        out[k] = static_cast<uint32_t>(x[k]);
    }
    return out;
}

// Convert uint32 array back to double array
inline std::vector<double> ints_to_doubles(const uint32_t* vals, size_t nnz) {
    std::vector<double> out(nnz);
    for (size_t k = 0; k < nnz; ++k) {
        out[k] = static_cast<double>(vals[k]);
    }
    return out;
}

// ============================================================================
// Fallback: store doubles as raw 8-byte IEEE 754
// ============================================================================
inline std::vector<uint64_t> doubles_to_bits(const double* x, size_t nnz) {
    std::vector<uint64_t> out(nnz);
    for (size_t k = 0; k < nnz; ++k) {
        std::memcpy(&out[k], &x[k], 8);
    }
    return out;
}

inline std::vector<double> bits_to_doubles(const uint64_t* bits, size_t nnz) {
    std::vector<double> out(nnz);
    for (size_t k = 0; k < nnz; ++k) {
        std::memcpy(&out[k], &bits[k], 8);
    }
    return out;
}

} // namespace streampress

#endif // SPARSEPRESS_VALUE_MAP_HPP
