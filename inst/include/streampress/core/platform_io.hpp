/**
 * @file platform_io.hpp
 * @brief Platform-abstracted positional read (pread) for thread-safe I/O.
 *
 * Provides safe_pread() that works on Linux/macOS (pread) and Windows
 * (ReadFile with OVERLAPPED). This enables concurrent reads without mutexes.
 */

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace streampress {

#ifdef _WIN32
static inline ssize_t safe_pread(int fd, void* buf, size_t count, uint64_t offset) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    OVERLAPPED ov = {};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD nread = 0;
    if (!ReadFile(h, buf, static_cast<DWORD>(count), &nread, &ov)) return -1;
    return static_cast<ssize_t>(nread);
}
#else
static inline ssize_t safe_pread(int fd, void* buf, size_t count, uint64_t offset) {
    return ::pread(fd, buf, count, static_cast<off_t>(offset));
}
#endif

// Helper: resolve thread count (0 = all available)
inline int resolve_threads(int threads) {
#ifdef _OPENMP
    return (threads <= 0) ? omp_get_max_threads() : threads;
#else
    (void)threads;
    return 1;
#endif
}

}  // namespace streampress
