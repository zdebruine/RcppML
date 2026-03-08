// sparsepress - Variable-length integer codec
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_VARINT_HPP
#define SPARSEPRESS_VARINT_HPP

#include <cstdint>
#include <cstddef>
#include <vector>

namespace streampress {

// ============================================================================
// VarInt codec - 7-bit continuation encoding
// ============================================================================
namespace varint {

// Encode unsigned integer to buffer, return number of bytes written
inline size_t encode(uint64_t value, uint8_t* out) {
    size_t n = 0;
    do {
        uint8_t byte = value & 0x7F;
        value >>= 7;
        if (value > 0) byte |= 0x80;
        out[n++] = byte;
    } while (value > 0);
    return n;
}

// Encode to vector
inline void encode(uint64_t value, std::vector<uint8_t>& out) {
    do {
        uint8_t byte = value & 0x7F;
        value >>= 7;
        if (value > 0) byte |= 0x80;
        out.push_back(byte);
    } while (value > 0);
}

// Decode unsigned integer from buffer, return number of bytes consumed
inline size_t decode(const uint8_t* in, uint64_t& value) {
    value = 0;
    size_t n = 0;
    int shift = 0;
    do {
        value |= static_cast<uint64_t>(in[n] & 0x7F) << shift;
        shift += 7;
    } while (in[n++] & 0x80);
    return n;
}

// Decode from pointer, advance pointer
inline uint64_t decode(const uint8_t*& ptr) {
    uint64_t value = 0;
    int shift = 0;
    do {
        value |= static_cast<uint64_t>(*ptr & 0x7F) << shift;
        shift += 7;
    } while (*ptr++ & 0x80);
    return value;
}

// ZigZag encoding for signed integers: (n << 1) ^ (n >> 63)
inline uint64_t zigzag_encode(int64_t n) {
    return static_cast<uint64_t>((n << 1) ^ (n >> 63));
}

// ZigZag decoding
inline int64_t zigzag_decode(uint64_t n) {
    return static_cast<int64_t>((n >> 1) ^ -(n & 1));
}

// Encode signed integer as ZigZag + VarInt
inline void encode_signed(int64_t value, std::vector<uint8_t>& out) {
    encode(zigzag_encode(value), out);
}

// Decode signed integer
inline int64_t decode_signed(const uint8_t*& ptr) {
    return zigzag_decode(decode(ptr));
}

// Encode a vector of uint32_t values
inline std::vector<uint8_t> encode_array(const std::vector<uint32_t>& values) {
    std::vector<uint8_t> out;
    out.reserve(values.size() * 2);
    for (auto v : values)
        encode(v, out);
    return out;
}

// Decode N uint32_t values
inline std::vector<uint32_t> decode_array(const uint8_t*& ptr, size_t count) {
    std::vector<uint32_t> values(count);
    for (size_t k = 0; k < count; ++k)
        values[k] = static_cast<uint32_t>(decode(ptr));
    return values;
}

} // namespace varint
} // namespace streampress

#endif // SPARSEPRESS_VARINT_HPP
