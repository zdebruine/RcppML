// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file profiling/gpu_timer.cuh
 * @brief CUDA event-based section timer for GPU profiling
 *
 * Provides per-section GPU timing with string-keyed accumulation.
 * Activated at runtime via `active` flag (typically gated on config.verbose).
 *
 * IMPORTANT: The end() call inserts a cudaEventSynchronize, which serializes
 * the GPU pipeline. This is intentional for profiling — it gives accurate
 * per-section timing. Timer overhead is ~0 when inactive (early return).
 *
 * Usage:
 *   GpuSectionTimer timer;
 *   timer.init();
 *   timer.active = true;
 *
 *   timer.begin(stream, "gram_H");
 *   compute_gram(...);
 *   timer.end(stream);
 *
 *   timer.begin(stream, "rhs_H");
 *   compute_rhs(...);
 *   timer.end(stream);
 *
 *   // At end, retrieve structured results
 *   auto sections = timer.results();  // map<string, float> of total_ms
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <map>
#include <string>
#include <cstdio>

namespace FactorNet {
namespace profiling {

/**
 * @brief CUDA event-based section timer
 *
 * Tracks cumulative GPU time per named section. Each begin()/end()
 * pair records one measurement that gets accumulated into the section.
 */
struct GpuSectionTimer {
    bool active = false;

    struct Entry {
        float total_ms = 0.0f;
        int count = 0;
    };

    std::unordered_map<std::string, Entry> sections;

    void init() {
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_stop_);
        initialized_ = true;
    }

    void destroy() {
        if (initialized_) {
            cudaEventDestroy(ev_start_);
            cudaEventDestroy(ev_stop_);
            initialized_ = false;
        }
    }

    ~GpuSectionTimer() {
        destroy();
    }

    /**
     * @brief Start timing a named section on the given stream.
     *
     * No-op if timer is inactive.
     */
    void begin(cudaStream_t stream, const std::string& name) {
        if (!active) return;
        current_ = name;
        cudaEventRecord(ev_start_, stream);
    }

    /**
     * @brief End the current section and accumulate elapsed time.
     *
     * Calls cudaEventSynchronize (serializes pipeline). No-op if inactive.
     */
    void end(cudaStream_t stream) {
        if (!active) return;
        cudaEventRecord(ev_stop_, stream);
        cudaEventSynchronize(ev_stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_, ev_stop_);
        sections[current_].total_ms += ms;
        sections[current_].count++;
    }

    /**
     * @brief Get total accumulated time per section (sorted by name).
     */
    std::map<std::string, float> results() const {
        std::map<std::string, float> out;
        for (const auto& kv : sections)
            out[kv.first] = kv.second.total_ms;
        return out;
    }

    /**
     * @brief Print a formatted timing report to stderr.
     *
     * Shows per-section total, average, count, and percentage of grand total.
     */
    void report(int n_iters, const char* label = "GPU NMF") const {
        if (sections.empty()) return;

        float grand_total = 0.0f;
        for (const auto& kv : sections)
            grand_total += kv.second.total_ms;

        // Sorted output
        std::map<std::string, Entry> sorted(sections.begin(), sections.end());

        fprintf(stderr, "[%s profiling — %d iters, %.1fms total]\n",
                label, n_iters, grand_total);
        fprintf(stderr, "  %-20s %10s %10s %8s %6s\n",
                "Section", "Total(ms)", "Avg(ms)", "Count", "%%");
        fprintf(stderr, "  %-20s %10s %10s %8s %6s\n",
                "-------", "---------", "-------", "-----", "--");

        for (const auto& kv : sorted) {
            float pct = (grand_total > 0) ? (100.0f * kv.second.total_ms / grand_total) : 0.0f;
            fprintf(stderr, "  %-20s %10.2f %10.3f %8d %5.1f%%\n",
                    kv.first.c_str(),
                    kv.second.total_ms,
                    (kv.second.count > 0) ? (kv.second.total_ms / kv.second.count) : 0.0f,
                    kv.second.count,
                    pct);
        }
    }

    /**
     * @brief Reset all sections (keep events alive).
     */
    void reset() {
        sections.clear();
    }

private:
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_stop_ = nullptr;
    std::string current_;
    bool initialized_ = false;
};

}  // namespace profiling
}  // namespace FactorNet
