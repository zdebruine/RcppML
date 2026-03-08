// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file profiling/stream_timer.hpp
 * @brief Section profiling for streaming/chunked NMF pipelines
 *
 * Identical interface to SectionTimer in cpu_timer.hpp but in a separate
 * file to avoid ownership conflicts between agents. Agent 7 owns this file;
 * Agent 5 owns cpu_timer.hpp.
 *
 * Sections tracked:
 *   chunk_read     — Time to read + decompress one SPZ chunk from disk
 *   chunk_decode   — Time to decode chunk into sparse column format
 *   chunk_gram     — Per-chunk Gram contribution computation
 *   chunk_rhs      — Per-chunk RHS computation
 *   chunk_nnls     — Per-chunk NNLS solve
 *   gram_accumulate— Accumulating Gram across chunks
 *   total_iter     — Full iteration time
 *   chunk_h2d      — (GPU) Host-to-device chunk transfer time
 *   sync_per_chunk — (GPU) Per-chunk GPU synchronization time
 *   kernel_launch  — (GPU) Kernel dispatch overhead per chunk
 *
 * Usage:
 *   StreamTimer timer(config.enable_profiling);
 *   timer.begin("chunk_read");
 *   ... read chunk ...
 *   timer.end();
 *   auto profile = timer.results();
 */

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace FactorNet {
namespace profiling {

struct StreamTimer {
    bool active;

    struct Entry {
        double total_ms = 0.0;
        int count = 0;
    };

    std::unordered_map<std::string, Entry> sections;
    std::chrono::high_resolution_clock::time_point current_start;
    std::string current_section;

    // For wall-clock total iteration timing (not nested with sections)
    std::chrono::high_resolution_clock::time_point iter_start;
    double iter_total_ms = 0.0;
    int iter_count = 0;

    explicit StreamTimer(bool enable = false) : active(enable) {}

    void begin(const std::string& name) {
        if (__builtin_expect(!active, 1)) return;
        current_section = name;
        current_start = std::chrono::high_resolution_clock::now();
    }

    void end() {
        if (__builtin_expect(!active, 1)) return;
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - current_start).count();
        sections[current_section].total_ms += elapsed;
        sections[current_section].count++;
    }

    /// Mark the beginning of one NMF iteration (wall clock, not nested)
    void iter_begin() {
        if (__builtin_expect(!active, 1)) return;
        iter_start = std::chrono::high_resolution_clock::now();
    }

    /// Mark the end of one NMF iteration
    void iter_end() {
        if (__builtin_expect(!active, 1)) return;
        iter_total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - iter_start).count();
        iter_count++;
    }

    /// Return cumulative ms per section (includes "total_iter" from iter timing)
    std::unordered_map<std::string, double> results() const {
        std::unordered_map<std::string, double> out;
        for (const auto& kv : sections) {
            out[kv.first] = kv.second.total_ms;
        }
        if (iter_count > 0) {
            out["total_iter"] = iter_total_ms;
        }
        return out;
    }

    /// Return counts per section (includes "total_iter" from iter timing)
    std::unordered_map<std::string, int> counts() const {
        std::unordered_map<std::string, int> out;
        for (const auto& kv : sections) {
            out[kv.first] = kv.second.count;
        }
        if (iter_count > 0) {
            out["total_iter"] = iter_count;
        }
        return out;
    }

    /// Reset all accumulated data
    void reset() {
        sections.clear();
        iter_total_ms = 0.0;
        iter_count = 0;
    }
};

}  // namespace profiling
}  // namespace FactorNet
