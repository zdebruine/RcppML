// sparsepress - Golomb-Rice codec for geometric-distributed values
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_GOLOMB_RICE_HPP
#define SPARSEPRESS_GOLOMB_RICE_HPP

#include <sparsepress/codec/bitstream.hpp>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace sparsepress {

// ============================================================================
// Golomb-Rice codec
//
// Optimal for geometric distributions. Encodes value v with parameter k:
//   quotient q = v >> k        (written in unary: q ones + zero)
//   remainder r = v & ((1<<k)-1)  (written as k raw bits)
//
// Optimal k ≈ floor(log2(0.6931 * mean)) for geometric(p) with mean=(1-p)/p
// ============================================================================
namespace rice {

// Compute optimal Rice parameter from distribution mean
inline uint32_t optimal_k(double mean) {
    if (mean <= 0.0) return 0;
    double k_real = std::log2(0.6931471805599453 * mean);
    return static_cast<uint32_t>(std::max(0.0, std::floor(k_real)));
}

// Compute optimal Rice parameter from sample data
inline uint32_t optimal_k_from_data(const uint32_t* data, size_t count) {
    if (count == 0) return 0;
    uint64_t sum = 0;
    for (size_t i = 0; i < count; ++i)
        sum += data[i];
    double mean = static_cast<double>(sum) / count;
    return optimal_k(mean);
}

// Encode a single value with parameter k
inline void encode(BitWriter& bw, uint32_t value, uint32_t k) {
    uint32_t q = value >> k;
    uint32_t r = value & ((1u << k) - 1);

    // Unary code for quotient: q ones followed by a zero
    bw.write_unary(q);

    // Binary code for remainder: k bits
    if (k > 0) {
        bw.write(r, k);
    }
}

// Decode a single value with parameter k
inline uint32_t decode(BitReader& br, uint32_t k) {
    uint32_t q = br.read_unary();
    uint32_t r = 0;
    if (k > 0) {
        r = static_cast<uint32_t>(br.read(k));
    }
    return (q << k) | r;
}

// Encode ZigZag-encoded signed value
inline void encode_signed(BitWriter& bw, int32_t value, uint32_t k) {
    uint32_t zz = static_cast<uint32_t>((value << 1) ^ (value >> 31));
    encode(bw, zz, k);
}

// Decode ZigZag-encoded signed value
inline int32_t decode_signed(BitReader& br, uint32_t k) {
    uint32_t zz = decode(br, k);
    return static_cast<int32_t>((zz >> 1) ^ -(zz & 1));
}

// ============================================================================
// Block-adaptive Rice encoding
//
// Splits data into blocks, computes optimal k per block, stores k values
// in the header for each block.
// ============================================================================
struct BlockRiceEncoder {
    uint32_t block_size;
    std::vector<uint8_t> k_params;  // k for each block

    explicit BlockRiceEncoder(uint32_t bs = 256) : block_size(bs) {}

    // Encode array of unsigned values with block-adaptive k
    std::vector<uint8_t> encode_array(const uint32_t* data, size_t count) {
        if (count == 0) return {};

        size_t n_blocks = (count + block_size - 1) / block_size;
        k_params.resize(n_blocks);

        // Phase 1: Compute optimal k per block
        for (size_t b = 0; b < n_blocks; ++b) {
            size_t start = b * block_size;
            size_t end = std::min(start + block_size, count);
            k_params[b] = static_cast<uint8_t>(
                optimal_k_from_data(data + start, end - start));
            // Cap k to avoid degenerate cases
            if (k_params[b] > 24) k_params[b] = 24;
        }

        // Phase 2: Encode
        BitWriter bw;
        for (size_t b = 0; b < n_blocks; ++b) {
            size_t start = b * block_size;
            size_t end = std::min(start + block_size, count);
            uint32_t k = k_params[b];
            for (size_t i = start; i < end; ++i) {
                rice::encode(bw, data[i], k);
            }
        }

        return bw.release();
    }

    // Encode array of signed values (ZigZag + Rice)
    std::vector<uint8_t> encode_signed_array(const int32_t* data, size_t count) {
        if (count == 0) return {};

        // Convert to unsigned via ZigZag
        std::vector<uint32_t> zz(count);
        for (size_t i = 0; i < count; ++i)
            zz[i] = static_cast<uint32_t>((data[i] << 1) ^ (data[i] >> 31));

        return encode_array(zz.data(), count);
    }
};

struct BlockRiceDecoder {
    uint32_t block_size;
    std::vector<uint8_t> k_params;

    BlockRiceDecoder(uint32_t bs, std::vector<uint8_t> ks)
        : block_size(bs), k_params(std::move(ks)) {}

    // Decode array of unsigned values
    void decode_array(const uint8_t* encoded_data, size_t encoded_size,
                      uint32_t* out, size_t count) {
        BitReader br(encoded_data, encoded_size);
        size_t n_blocks = k_params.size();

        for (size_t b = 0; b < n_blocks; ++b) {
            size_t start = b * block_size;
            size_t end = std::min(start + block_size, count);
            uint32_t k = k_params[b];
            for (size_t i = start; i < end; ++i) {
                out[i] = rice::decode(br, k);
            }
        }
    }

    // Decode array of signed values (undo ZigZag)
    void decode_signed_array(const uint8_t* encoded_data, size_t encoded_size,
                             int32_t* out, size_t count) {
        std::vector<uint32_t> zz(count);
        decode_array(encoded_data, encoded_size, zz.data(), count);
        for (size_t i = 0; i < count; ++i)
            out[i] = static_cast<int32_t>((zz[i] >> 1) ^ -(zz[i] & 1));
    }
};

} // namespace rice
} // namespace sparsepress

#endif // SPARSEPRESS_GOLOMB_RICE_HPP
