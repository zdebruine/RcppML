/**
 * @file logging.hpp
 * @brief Unified logging infrastructure for RcppML.
 *
 * Cross-platform logging that works identically from R (Rcpp),
 * Python (pybind11), and standalone C++.
 *
 * Log levels:
 *   SILENT   (0) — No output
 *   SUMMARY  (1) — Backend selection, dimensions, final result
 *   DETAILED (2) — Per-iteration progress, timing
 *   DEBUG    (3) — Memory allocation, kernel launches, chunk decode timing
 *
 * Usage:
 *   FACTORNET_LOG(LogLevel::SUMMARY, verbose, "[nmf] Backend: %s\n", name);
 *   FACTORNET_LOG_NMF(LogLevel::DETAILED, verbose, "Iter %d: loss=%.6f\n", iter, loss);
 */

#pragma once

#include <cstdio>

namespace FactorNet {

enum class LogLevel : int {
    SILENT   = 0,
    SUMMARY  = 1,
    DETAILED = 2,
    DEBUG    = 3
};

} // namespace FactorNet

// ============================================================================
// Platform-specific output primitives
// ============================================================================

#if defined(FACTORNET_USE_R) || defined(SPARSEPRESS_USE_R)
  #include <R_ext/Print.h>
  #define FACTORNET_PRINT_IMPL(...)  Rprintf(__VA_ARGS__)
  #define FACTORNET_WARN_IMPL(...)   REprintf(__VA_ARGS__)
#elif defined(FACTORNET_USE_PYTHON)
  #include <cstdio>
  #define FACTORNET_PRINT_IMPL(...)  std::fprintf(stdout, __VA_ARGS__)
  #define FACTORNET_WARN_IMPL(...)   std::fprintf(stderr, __VA_ARGS__)
#else
  #define FACTORNET_PRINT_IMPL(...)  std::fprintf(stdout, __VA_ARGS__)
  #define FACTORNET_WARN_IMPL(...)   std::fprintf(stderr, __VA_ARGS__)
#endif

// ============================================================================
// Level-gated logging macros
// ============================================================================

/**
 * Primary logging macro — gated by runtime verbosity level.
 * @param level  FactorNet::LogLevel threshold
 * @param verbose  int verbosity setting (0-3)
 * @param ...  printf-style format string and arguments
 */
#define FACTORNET_LOG(level, verbose, ...) \
    do { \
        if (static_cast<int>(verbose) >= static_cast<int>(level)) { \
            FACTORNET_PRINT_IMPL(__VA_ARGS__); \
        } \
    } while (0)

#define FACTORNET_WARN(level, verbose, ...) \
    do { \
        if (static_cast<int>(verbose) >= static_cast<int>(level)) { \
            FACTORNET_WARN_IMPL(__VA_ARGS__); \
        } \
    } while (0)

// ============================================================================
// Component-prefixed convenience macros
// ============================================================================

#define FACTORNET_LOG_NMF(level, verbose, ...)  FACTORNET_LOG(level, verbose, "[nmf] " __VA_ARGS__)
#define FACTORNET_LOG_SPZ(level, verbose, ...)  FACTORNET_LOG(level, verbose, "[spz] " __VA_ARGS__)
#define FACTORNET_LOG_GPU(level, verbose, ...)  FACTORNET_LOG(level, verbose, "[gpu] " __VA_ARGS__)
#define FACTORNET_LOG_IO(level, verbose, ...)   FACTORNET_LOG(level, verbose, "[io]  " __VA_ARGS__)

// ============================================================================
// Backward-compatible SPZ_FPRINTF replacement
// ============================================================================

// Redefine SPZ_FPRINTF to use the unified logging system
#ifdef SPZ_FPRINTF
#undef SPZ_FPRINTF
#endif
#define SPZ_FPRINTF(stream, ...) FACTORNET_WARN_IMPL(__VA_ARGS__)

// ============================================================================
// Convenience: level-less logging (verbose as bool or int >= 1)
// ============================================================================

/**
 * Verbose-gated info log (prints when verbose >= 1)
 * @param verbose  bool or int
 * @param ...  printf-style format string and arguments
 */
#define FACTORNET_LOG_INFO(verbose, ...) FACTORNET_LOG(FactorNet::LogLevel::SUMMARY, verbose, __VA_ARGS__)

/**
 * Verbose-gated debug log (prints when verbose >= 2)
 */
#define FACTORNET_LOG_DETAIL(verbose, ...) FACTORNET_LOG(FactorNet::LogLevel::DETAILED, verbose, __VA_ARGS__)

/**
 * Verbose-gated trace log (prints when verbose >= 3)
 */
#define FACTORNET_LOG_TRACE(verbose, ...) FACTORNET_LOG(FactorNet::LogLevel::DEBUG, verbose, __VA_ARGS__)

/**
 * Warning/error output (always to stderr, gated by verbose)
 */
#define FACTORNET_WARN_INFO(verbose, ...) FACTORNET_WARN(FactorNet::LogLevel::SUMMARY, verbose, __VA_ARGS__)

// ============================================================================
// GPU-host logging (for .cuh files compiled by nvcc)
// ============================================================================
// These use printf/fprintf directly since R's Rprintf is not available
// in nvcc compilation units. The verbosity gating still works.

#ifndef FACTORNET_GPU_LOG
#define FACTORNET_GPU_LOG(verbose, ...) \
    do { if (verbose) std::fprintf(stdout, __VA_ARGS__); } while(0)
#endif

#ifndef FACTORNET_GPU_WARN
#define FACTORNET_GPU_WARN(...) \
    do { std::fprintf(stderr, __VA_ARGS__); } while(0)
#endif
