/**
 * @file checks.hpp
 * @brief Validation helpers shared by the StreamPress (.spz) readers.
 *
 * Everything read from a file (sizes, offsets, counts) is untrusted. The
 * readers use these helpers to check each value against the bytes actually
 * available before using it, so that a truncated or corrupted file produces
 * an exception (an R error) instead of an out-of-bounds access.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace streampress {

/// Throw the standard "corrupt file" error.
[[noreturn]] inline void spz_corrupt(const char* what) {
    throw std::runtime_error(
        std::string("Corrupt or truncated .spz file: ") + what);
}

/// True on a little-endian host.
inline bool host_is_little_endian() noexcept {
    const uint16_t one = 1;
    uint8_t first = 0;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

/// The on-disk format is little-endian and is read with memcpy into native
/// integers, so big-endian hosts are refused up front with a clear message.
inline void require_little_endian() {
    if (!host_is_little_endian())
        throw std::runtime_error(
            "StreamPress (.spz) files are little-endian and are not "
            "supported on big-endian platforms");
}

/// True iff the byte range [offset, offset + size) lies within [0, total).
/// Written so that it cannot overflow.
inline bool range_ok(uint64_t offset, uint64_t size, uint64_t total) noexcept {
    return offset <= total && size <= total - offset;
}

/// Throw unless [offset, offset + size) lies within [0, total).
inline void require_range(uint64_t offset, uint64_t size, uint64_t total,
                          const char* what) {
    if (!range_ok(offset, size, total)) spz_corrupt(what);
}

/// Length of a string that may lack a terminator within `max_len` bytes.
inline size_t bounded_strlen(const char* str, size_t max_len) noexcept {
    const void* nul = std::memchr(str, 0, max_len);
    return nul ? static_cast<size_t>(static_cast<const char*>(nul) - str)
               : max_len;
}

/// Unaligned little-endian loads (memcpy; safe for any address).
inline uint16_t load_u16(const uint8_t* p) noexcept {
    uint16_t v; std::memcpy(&v, p, sizeof(v)); return v;
}
inline uint32_t load_u32(const uint8_t* p) noexcept {
    uint32_t v; std::memcpy(&v, p, sizeof(v)); return v;
}
inline uint64_t load_u64(const uint8_t* p) noexcept {
    uint64_t v; std::memcpy(&v, p, sizeof(v)); return v;
}
inline float load_f32(const uint8_t* p) noexcept {
    float v; std::memcpy(&v, p, sizeof(v)); return v;
}
inline double load_f64(const uint8_t* p) noexcept {
    double v; std::memcpy(&v, p, sizeof(v)); return v;
}

// ---------------------------------------------------------------------------
// 64-bit file positioning. `long` is 32 bits on Windows, so fseek/ftell
// cannot address files of 2 GB or more there.
// ---------------------------------------------------------------------------

/// Seek to an absolute offset. Returns 0 on success.
inline int spz_fseek(std::FILE* f, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, static_cast<long long>(offset), SEEK_SET);
#else
    return std::fseek(f, static_cast<long>(offset), SEEK_SET);
#endif
}

/// Size of an open file in bytes, or -1 on error. Leaves the position at 0.
inline long long spz_file_size(std::FILE* f) {
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    const long long sz = _ftelli64(f);
    if (_fseeki64(f, 0, SEEK_SET) != 0) return -1;
    return sz;
#else
    if (std::fseek(f, 0, SEEK_END) != 0) return -1;
    const long long sz = static_cast<long long>(std::ftell(f));
    if (std::fseek(f, 0, SEEK_SET) != 0) return -1;
    return sz;
#endif
}

}  // namespace streampress
