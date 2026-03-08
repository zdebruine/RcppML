// sparsepress - Bit-level I/O stream
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_BITSTREAM_HPP
#define SPARSEPRESS_BITSTREAM_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace sparsepress {

// ============================================================================
// BitWriter - writes bits into a byte buffer, LSB-first
// ============================================================================
class BitWriter {
public:
    BitWriter() : buffer_(0), bits_in_buffer_(0) {}

    // Write n bits (1..64) from value (LSB first)
    void write(uint64_t value, int n) {
        while (n > 0) {
            int space = 64 - bits_in_buffer_;
            int take = n < space ? n : space;
            uint64_t mask = take < 64 ? ((1ULL << take) - 1) : ~0ULL;
            buffer_ |= (value & mask) << bits_in_buffer_;
            bits_in_buffer_ += take;
            value >>= take;
            n -= take;

            // Flush complete bytes
            while (bits_in_buffer_ >= 8) {
                data_.push_back(static_cast<uint8_t>(buffer_ & 0xFF));
                buffer_ >>= 8;
                bits_in_buffer_ -= 8;
            }
        }
    }

    // Write a single bit
    void write_bit(uint32_t bit) {
        buffer_ |= (static_cast<uint64_t>(bit & 1)) << bits_in_buffer_;
        bits_in_buffer_++;
        if (bits_in_buffer_ >= 8) {
            data_.push_back(static_cast<uint8_t>(buffer_ & 0xFF));
            buffer_ >>= 8;
            bits_in_buffer_ -= 8;
        }
    }

    // Write unary code: value ones followed by a zero
    void write_unary(uint32_t value) {
        for (uint32_t k = 0; k < value; ++k)
            write_bit(1);
        write_bit(0);
    }

    // Flush remaining bits (pad with zeros)
    void flush() {
        if (bits_in_buffer_ > 0) {
            data_.push_back(static_cast<uint8_t>(buffer_ & 0xFF));
            buffer_ = 0;
            bits_in_buffer_ = 0;
        }
    }

    const std::vector<uint8_t>& data() const { return data_; }
    std::vector<uint8_t>& data() { return data_; }
    size_t size_bytes() const { return data_.size(); }
    size_t total_bits() const { return data_.size() * 8; }

    // Move data out
    std::vector<uint8_t> release() {
        flush();
        auto result = std::move(data_);
        data_.clear();
        buffer_ = 0;
        bits_in_buffer_ = 0;
        return result;
    }

private:
    std::vector<uint8_t> data_;
    uint64_t buffer_;
    int bits_in_buffer_;
};

// ============================================================================
// BitReader - reads bits from a byte buffer, LSB-first
// ============================================================================
class BitReader {
public:
    BitReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), byte_pos_(0), buffer_(0), bits_in_buffer_(0)
    {
        refill();
    }

    BitReader(const std::vector<uint8_t>& v)
        : BitReader(v.data(), v.size()) {}

    // Read n bits (1..56)
    uint64_t read(int n) {
        if (bits_in_buffer_ < n) refill();
        uint64_t mask = n < 64 ? ((1ULL << n) - 1) : ~0ULL;
        uint64_t val = buffer_ & mask;
        buffer_ >>= n;
        bits_in_buffer_ -= n;
        return val;
    }

    // Read a single bit
    uint32_t read_bit() {
        if (bits_in_buffer_ < 1) refill();
        uint32_t bit = static_cast<uint32_t>(buffer_ & 1);
        buffer_ >>= 1;
        bits_in_buffer_--;
        return bit;
    }

    // Read unary code: count ones until zero
    uint32_t read_unary() {
        uint32_t count = 0;
        while (read_bit() == 1)
            ++count;
        return count;
    }

    bool eof() const {
        return byte_pos_ >= size_ && bits_in_buffer_ == 0;
    }

private:
    void refill() {
        while (bits_in_buffer_ <= 56 && byte_pos_ < size_) {
            buffer_ |= static_cast<uint64_t>(data_[byte_pos_]) << bits_in_buffer_;
            byte_pos_++;
            bits_in_buffer_ += 8;
        }
    }

    const uint8_t* data_;
    size_t size_;
    size_t byte_pos_;
    uint64_t buffer_;
    int bits_in_buffer_;
};

} // namespace sparsepress

#endif // SPARSEPRESS_BITSTREAM_HPP
