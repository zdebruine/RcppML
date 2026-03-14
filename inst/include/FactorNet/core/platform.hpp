/**
 * @file platform.hpp
 * @brief Platform-conditional utilities: RAM detection, OS queries
 *
 * Shared header for memory-aware dispatch (streaming auto-dispatch,
 * SVD init RAM check, chunk sizing). All functions are inline and
 * header-only — no link dependencies beyond system headers.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <cstdint>
#include <string>

#if defined(_WIN32)
// Workaround for MinGW C++17+: std::byte (from <cstddef>) conflicts
// with rpcndr.h "typedef unsigned char byte;" inside <windows.h>.
// Temporarily rename the Windows byte to avoid the ambiguity.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define byte win_byte_override
#include <windows.h>
#undef byte
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
// Linux / POSIX
#include <fstream>
#include <cstdio>
#endif

namespace FactorNet {

/**
 * @brief Query available physical RAM in bytes (platform-conditional).
 *
 * - Linux: reads MemAvailable from /proc/meminfo
 * - macOS: sysctl vm.page_free_count * page_size
 * - Windows: GlobalMemoryStatusEx().ullAvailPhys
 *
 * @return Available RAM in bytes, or 0 if detection fails
 */
inline uint64_t get_available_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return static_cast<uint64_t>(status.ullAvailPhys);
    return 0;
#elif defined(__APPLE__)
    int64_t page_size = sysconf(_SC_PAGE_SIZE);
    int64_t free_pages = 0;
    size_t len = sizeof(free_pages);
    if (sysctlbyname("vm.page_free_count", &free_pages, &len, nullptr, 0) == 0)
        return static_cast<uint64_t>(page_size * free_pages);
    return 0;
#else
    // Linux: read MemAvailable from /proc/meminfo
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            unsigned long long kb = 0;
            if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1)
                return kb * 1024ULL;
        }
    }
    return 0;  // unknown
#endif
}

/**
 * @brief Format bytes as a human-readable GB string (e.g., "12.3")
 */
inline std::string to_gb_str(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return std::string(buf);
}

}  // namespace FactorNet
