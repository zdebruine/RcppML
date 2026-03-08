/**
 * @file memory.hpp
 * @brief CPU memory estimation and safety checks for RcppML.
 *
 * Provides utilities to:
 *   - Query available system memory (Linux /proc/meminfo, macOS sysctl)
 *   - Estimate memory cost of sparse matrix transposition
 *   - Guard against OOM before allocating large buffers
 *
 * Used by NMF algorithms that unconditionally compute CSC(Aᵀ) for the
 * gather-based W update.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(__linux__)
  #include <cstdio>
#elif defined(__APPLE__)
  #include <sys/types.h>
  #include <sys/sysctl.h>
  #include <mach/mach.h>
#endif

namespace FactorNet {
namespace memory {

// ============================================================================
// System memory queries
// ============================================================================

/**
 * @brief Query the available (free + reclaimable) memory on the system.
 *
 * On Linux: parses /proc/meminfo for MemAvailable.
 * On macOS: uses mach_host_statistics for free + inactive pages.
 * Returns 0 on failure or unsupported platforms (caller should treat as unknown).
 */
inline size_t available_memory_bytes() {
#if defined(__linux__)
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;

    char line[256];
    size_t avail_kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "MemAvailable: %zu kB", &avail_kb) == 1) {
            break;
        }
    }
    std::fclose(f);
    return avail_kb * 1024ULL;  // kB → bytes

#elif defined(__APPLE__)
    // Use Mach host statistics for available memory
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        mach_host_self(), HOST_VM_INFO64,
        reinterpret_cast<host_info64_t>(&vm_stats), &count);
    if (kr != KERN_SUCCESS) return 0;

    // Page size
    vm_size_t page_size;
    host_page_size(mach_host_self(), &page_size);

    // Available = free + inactive (reclaimable)
    size_t avail = static_cast<size_t>(vm_stats.free_count + vm_stats.inactive_count)
                   * static_cast<size_t>(page_size);
    return avail;

#else
    return 0;  // Unknown platform — caller handles gracefully
#endif
}

// ============================================================================
// Sparse transpose memory estimation
// ============================================================================

/**
 * @brief Estimate the memory cost (bytes) of a CSC sparse transpose.
 *
 * The transpose of an m×n CSC matrix with `nnz` non-zeros produces an
 * n×m CSC matrix with the same `nnz`. The storage cost is:
 *   - values:  nnz × sizeof(Scalar)
 *   - indices: nnz × sizeof(int)          (InnerIndexType = int in Eigen)
 *   - pointers: (m + 1) × sizeof(int)     (OuterIndexPtr for n×m → m+1 entries)
 *
 * @tparam Scalar  Element type (float or double)
 * @param nnz   Number of non-zero entries
 * @param m     Rows of the original matrix (becomes cols of transpose)
 * @param n     Columns of the original matrix (becomes rows of transpose)
 */
template<typename Scalar>
inline size_t sparse_transpose_bytes(size_t nnz, int m, int /*n*/) {
    return nnz * sizeof(Scalar)          // value array
         + nnz * sizeof(int)             // inner index array
         + static_cast<size_t>(m + 1) * sizeof(int);  // outer index array
}

/**
 * @brief Format a byte count as a human-readable string (e.g., "2.4 GB").
 */
inline std::string format_bytes(size_t bytes) {
    char buf[64];
    if (bytes >= 1024ULL * 1024 * 1024) {
        std::snprintf(buf, sizeof(buf), "%.1f GB",
                      static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024ULL * 1024) {
        std::snprintf(buf, sizeof(buf), "%.1f MB",
                      static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f KB",
                      static_cast<double>(bytes) / 1024.0);
    }
    return std::string(buf);
}

// ============================================================================
// Memory safety check result
// ============================================================================

struct MemoryCheckResult {
    bool fits;                      ///< true if the transpose fits in available memory
    size_t transpose_bytes;         ///< estimated transpose allocation size
    size_t available_bytes;         ///< system available memory (0 = unknown)
    double headroom_fraction;       ///< available / transpose (>1 means fits)
    std::string message;            ///< human-readable diagnostic
};

/**
 * @brief Check whether computing A.transpose() in-memory is safe.
 *
 * Returns a MemoryCheckResult with:
 *   - fits:  true if we can safely allocate the transpose
 *   - message: diagnostic suitable for user display
 *
 * The safety margin is 2x — we require at least twice the transpose size
 * to be available, because Eigen's transpose() briefly holds both the
 * original and the result, plus other algorithm buffers need space.
 *
 * @tparam Scalar  float or double
 * @param nnz   Non-zero count of the sparse matrix
 * @param m     Rows
 * @param n     Columns
 */
template<typename Scalar>
inline MemoryCheckResult check_transpose_memory(size_t nnz, int m, int n) {
    MemoryCheckResult result;
    result.transpose_bytes = sparse_transpose_bytes<Scalar>(nnz, m, n);
    result.available_bytes = available_memory_bytes();

    if (result.available_bytes == 0) {
        // Cannot determine available memory — proceed with warning
        result.fits = true;
        result.headroom_fraction = 0;
        result.message =
            "Transpose allocation: " + format_bytes(result.transpose_bytes) +
            " (available memory unknown — proceeding)";
        return result;
    }

    // Safety factor: require 2x the transpose size to be available.
    // The algorithm also needs space for W (k×m), H (k×n), Gram (k×k), etc.
    constexpr double SAFETY_FACTOR = 2.0;
    result.headroom_fraction =
        static_cast<double>(result.available_bytes) /
        static_cast<double>(result.transpose_bytes);
    result.fits = result.headroom_fraction >= SAFETY_FACTOR;

    if (result.fits) {
        result.message =
            "Transpose allocation: " + format_bytes(result.transpose_bytes) +
            " of " + format_bytes(result.available_bytes) + " available"
            " (headroom: " + std::to_string(static_cast<int>(result.headroom_fraction)) + "x)";
    } else {
        result.message =
            "INSUFFICIENT MEMORY for in-memory transpose.\n"
            "  Transpose requires: " + format_bytes(result.transpose_bytes) + "\n"
            "  Available memory:   " + format_bytes(result.available_bytes) + "\n"
            "  Required headroom:  2x (for transpose + algorithm buffers)\n"
            "\n"
            "Options to resolve this:\n"
            "  1. Request more memory (e.g., SLURM: --mem=<size>)\n"
            "  2. Use out-of-core NMF with a .spz file that has a pre-stored\n"
            "     transpose section:\n"
            "       sp_write(x, 'data.spz', include_transpose = TRUE)\n"
            "     Then call nmf() on the .spz file path instead of the matrix.\n"
            "  3. Reduce matrix size (subsample columns or filter low-count features)\n";
    }

    return result;
}

}  // namespace memory
}  // namespace FactorNet
