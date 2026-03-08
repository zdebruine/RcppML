// sparsepress - rANS (range Asymmetric Numeral Systems) entropy codec
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)
//
// From-scratch implementation of a byte-renormalized rANS codec.
// Approaches Shannon entropy within ~0.01 bits/symbol for highly skewed
// distributions (e.g., scRNA-seq UMI count residuals).

#ifndef SPARSEPRESS_RANS_HPP
#define SPARSEPRESS_RANS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

namespace streampress {

// ============================================================================
// rANS symbol frequency table
// ============================================================================

static constexpr int RANS_PROB_BITS = 14;           // Probability scale: 2^14 = 16384
static constexpr uint32_t RANS_PROB_SCALE = 1u << RANS_PROB_BITS;

struct RansSymbol {
    uint16_t freq;       // Normalized frequency (sums to RANS_PROB_SCALE)
    uint16_t cum_freq;   // Cumulative frequency (exclusive prefix sum)
};

struct RansTable {
    uint32_t n_symbols;                       // Number of distinct symbols
    std::vector<RansSymbol> symbols;          // Indexed by symbol value
    std::vector<uint16_t> cum_to_symbol;       // Lookup: cum_freq → symbol (size RANS_PROB_SCALE)

    RansTable() : n_symbols(0) {}

    // Build from raw frequency counts
    // freq_counts[i] = count of symbol i, for i in [0, max_symbol]
    void build(const std::vector<uint64_t>& freq_counts) {
        n_symbols = static_cast<uint32_t>(freq_counts.size());
        symbols.resize(n_symbols);

        uint64_t total = 0;
        for (auto c : freq_counts) total += c;
        if (total == 0) {
            // degenerate: no symbols
            for (uint32_t i = 0; i < n_symbols; ++i) {
                symbols[i].freq = 0;
                symbols[i].cum_freq = 0;
            }
            return;
        }

        // Count non-zero symbols
        uint32_t n_nonzero = 0;
        for (auto c : freq_counts)
            if (c > 0) n_nonzero++;

        if (n_nonzero == 0) return;

        // Proportional scaling: each non-zero symbol gets at least freq=1
        // Reserve n_nonzero slots (1 each), distribute remaining proportionally
        uint32_t remaining = RANS_PROB_SCALE - n_nonzero;
        std::vector<uint16_t> freqs(n_symbols, 0);

        // First pass: assign proportional frequencies
        uint64_t assigned = 0;
        for (uint32_t i = 0; i < n_symbols; ++i) {
            if (freq_counts[i] == 0) continue;
            double frac = static_cast<double>(freq_counts[i]) / total;
            uint32_t f = 1 + static_cast<uint32_t>(frac * remaining);
            freqs[i] = static_cast<uint16_t>(f);
            assigned += f;
        }

        // Fix rounding: adjust to sum exactly to RANS_PROB_SCALE
        // Sort symbols by fractional error, trim/add from those with largest error
        int64_t diff = static_cast<int64_t>(assigned) - RANS_PROB_SCALE;

        if (diff != 0) {
            // Build list of adjustable symbols sorted by frequency (largest first for removal)
            std::vector<uint32_t> adjustable;
            for (uint32_t i = 0; i < n_symbols; ++i)
                if (freqs[i] > 0) adjustable.push_back(i);

            if (diff > 0) {
                // Need to remove 'diff' counts — take from largest frequencies
                std::sort(adjustable.begin(), adjustable.end(),
                    [&](uint32_t a, uint32_t b) { return freqs[a] > freqs[b]; });
                for (size_t k = 0; diff > 0 && k < adjustable.size(); ++k) {
                    uint32_t sym = adjustable[k % adjustable.size()];
                    if (freqs[sym] > 1) {
                        freqs[sym]--;
                        diff--;
                    }
                    if (k == adjustable.size() - 1 && diff > 0) k = static_cast<size_t>(-1); // restart
                }
            } else {
                // Need to add |diff| counts — add to smallest frequencies
                std::sort(adjustable.begin(), adjustable.end(),
                    [&](uint32_t a, uint32_t b) { return freqs[a] < freqs[b]; });
                for (size_t k = 0; diff < 0; ++k) {
                    uint32_t sym = adjustable[k % adjustable.size()];
                    freqs[sym]++;
                    diff++;
                    if (k == adjustable.size() - 1 && diff < 0) k = static_cast<size_t>(-1);
                }
            }
        }

        // Build cumulative frequencies
        uint16_t cum = 0;
        for (uint32_t i = 0; i < n_symbols; ++i) {
            symbols[i].freq = freqs[i];
            symbols[i].cum_freq = cum;
            cum += freqs[i];
        }

        // Build reverse lookup table
        build_lookup();
    }

    void build_lookup() {
        cum_to_symbol.resize(RANS_PROB_SCALE);
        for (uint32_t s = 0; s < n_symbols; ++s) {
            if (symbols[s].freq == 0) continue;
            for (uint16_t k = 0; k < symbols[s].freq; ++k) {
                cum_to_symbol[symbols[s].cum_freq + k] = static_cast<uint16_t>(s);
            }
        }
    }

    // Serialize frequency table (just the freq array, n_symbols known from context)
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> out;
        // Write number of symbols as uint16
        out.push_back(static_cast<uint8_t>(n_symbols & 0xFF));
        out.push_back(static_cast<uint8_t>((n_symbols >> 8) & 0xFF));
        // Write each freq as uint16_t LE
        for (uint32_t i = 0; i < n_symbols; ++i) {
            out.push_back(static_cast<uint8_t>(symbols[i].freq & 0xFF));
            out.push_back(static_cast<uint8_t>((symbols[i].freq >> 8) & 0xFF));
        }
        return out;
    }

    // Deserialize
    static RansTable deserialize(const uint8_t*& ptr) {
        RansTable t;
        t.n_symbols = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8);
        ptr += 2;
        t.symbols.resize(t.n_symbols);
        uint16_t cum = 0;
        for (uint32_t i = 0; i < t.n_symbols; ++i) {
            t.symbols[i].freq = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
            t.symbols[i].cum_freq = cum;
            cum += t.symbols[i].freq;
            ptr += 2;
        }
        t.build_lookup();
        return t;
    }
};

// ============================================================================
// rANS encoder (encodes in reverse, outputs bytes)
// ============================================================================

static constexpr uint32_t RANS_BYTE_L = 1u << 23;  // Lower bound of state

class RansEncoder {
public:
    RansEncoder() : state_(RANS_BYTE_L) {}

    // Encode a single symbol (call in REVERSE order)
    void encode(uint32_t sym, const RansTable& table) {
        const auto& s = table.symbols[sym];
        if (s.freq == 0)
            throw std::runtime_error("rANS: cannot encode zero-frequency symbol");

        // Renormalize: while state is too large, emit bytes
        uint32_t freq = s.freq;
        uint32_t x_max = ((RANS_BYTE_L >> RANS_PROB_BITS) << 8) * freq;
        while (state_ >= x_max) {
            output_.push_back(static_cast<uint8_t>(state_ & 0xFF));
            state_ >>= 8;
        }

        // Encode: state = ((state / freq) << PROB_BITS) + (state % freq) + cum_freq
        state_ = ((state_ / freq) << RANS_PROB_BITS) + (state_ % freq) + s.cum_freq;
    }

    // Flush final state (4 bytes, big-endian)
    std::vector<uint8_t> finish() {
        // Write final state as 4 bytes LE
        for (int i = 0; i < 4; ++i) {
            output_.push_back(static_cast<uint8_t>(state_ & 0xFF));
            state_ >>= 8;
        }
        // Reverse the output (since we encoded backwards)
        std::reverse(output_.begin(), output_.end());
        auto result = std::move(output_);
        output_.clear();
        state_ = RANS_BYTE_L;
        return result;
    }

private:
    uint32_t state_;
    std::vector<uint8_t> output_;
};

// ============================================================================
// rANS decoder (reads forward)
// ============================================================================

class RansDecoder {
public:
    RansDecoder() : state_(0), ptr_(nullptr), end_(nullptr) {}

    void init(const uint8_t* data, size_t size) {
        ptr_ = data;
        end_ = data + size;
        // Read initial state (4 bytes LE, stored at front after reverse)
        state_ = 0;
        for (int i = 0; i < 4; ++i) {
            state_ = (state_ << 8) | *ptr_++;
        }
    }

    // Decode a single symbol
    uint32_t decode(const RansTable& table) {
        // Extract cumulative frequency from state
        uint32_t cum = state_ & (RANS_PROB_SCALE - 1);

        // Look up symbol
        uint32_t sym = table.cum_to_symbol[cum];

        // Advance state
        const auto& s = table.symbols[sym];
        state_ = s.freq * (state_ >> RANS_PROB_BITS) + cum - s.cum_freq;

        // Renormalize
        while (state_ < RANS_BYTE_L && ptr_ < end_) {
            state_ = (state_ << 8) | *ptr_++;
        }

        return sym;
    }

private:
    uint32_t state_;
    const uint8_t* ptr_;
    const uint8_t* end_;
};

// ============================================================================
// Convenience: encode/decode entire arrays
// ============================================================================

namespace rans {

// Encode an array of symbols. Returns encoded bytes.
// Symbols must be in [0, table.n_symbols).
inline std::vector<uint8_t> encode_array(const uint32_t* symbols, size_t count,
                                          const RansTable& table) {
    if (count == 0) return {};

    RansEncoder enc;
    // Encode in reverse order
    for (size_t i = count; i > 0; --i) {
        enc.encode(symbols[i - 1], table);
    }
    return enc.finish();
}

// Decode count symbols from encoded data
inline void decode_array(const uint8_t* data, size_t size,
                         uint32_t* out, size_t count,
                         const RansTable& table) {
    RansDecoder dec;
    dec.init(data, size);
    for (size_t i = 0; i < count; ++i) {
        out[i] = dec.decode(table);
    }
}

// Build frequency table from raw data
inline RansTable build_table(const uint32_t* data, size_t count, uint32_t max_symbol) {
    std::vector<uint64_t> freq(max_symbol + 1, 0);
    for (size_t i = 0; i < count; ++i) {
        if (data[i] <= max_symbol)
            freq[data[i]]++;
    }
    RansTable table;
    table.build(freq);
    return table;
}

} // namespace rans
} // namespace streampress

#endif // SPARSEPRESS_RANS_HPP
