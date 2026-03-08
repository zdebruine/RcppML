// sparsepress - xorshift64 PRNG (matches RcppML's UnifiedRNG)
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_PRNG_HPP
#define SPARSEPRESS_PRNG_HPP

#include <cstdint>

namespace sparsepress {

// ============================================================================
// xorshift64 PRNG - deterministic, fast, position-dependent capable
// ============================================================================
class PRNG {
public:
    explicit PRNG(uint64_t seed = 1) : state_(seed ? seed : 1) {}

    // Advance state and return next value
    uint64_t next() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return state_;
    }

    // Uniform double in [0, 1)
    double uniform() {
        return static_cast<double>(next()) / 18446744073709551616.0;
    }

    // Position-dependent hash (does NOT advance state)
    // Deterministic function of (seed, i, j)
    static uint64_t position_hash(uint64_t seed, uint32_t i, uint32_t j) {
        uint64_t h = seed;
        h ^= static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<uint64_t>(j) * 0x517CC1B727220A95ULL;
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27;
        h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;
        return h;
    }

    // Seed combiner for per-column seeds
    static uint64_t combine_seed(uint64_t base_seed, uint32_t col) {
        return position_hash(base_seed, 0, col);
    }

    uint64_t state() const { return state_; }
    void set_state(uint64_t s) { state_ = s ? s : 1; }

private:
    uint64_t state_;
};

} // namespace sparsepress

#endif // SPARSEPRESS_PRNG_HPP
