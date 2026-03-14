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
// Avoid <windows.h> entirely — MinGW C++17+ std::byte conflicts with
// rpcndr.h byte typedef.  Use _lseeki64 + _read under a mutex instead.
#include <io.h>
#include <fcntl.h>
#include <mutex>
#else
#include <unistd.h>
#endif

namespace streampress {

#ifdef _WIN32
inline ssize_t safe_pread(int fd, void* buf, size_t count, uint64_t offset) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) < 0) return -1;
    int n = _read(fd, buf, static_cast<unsigned int>(count));
    return static_cast<ssize_t>(n);
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
