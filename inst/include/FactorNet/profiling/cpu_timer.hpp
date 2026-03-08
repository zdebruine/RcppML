// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file profiling/cpu_timer.hpp
 * @brief Runtime-activated section profiling with zero overhead when inactive
 *
 * Usage:
 *   SectionTimer timer(config.enable_profiling);
 *   timer.begin("gram_H");
 *   ... expensive operation ...
 *   timer.end();
 *   // After loop:
 *   auto profile = timer.results();  // map<string, double> of total_ms
 */

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace FactorNet {
namespace profiling {

struct SectionTimer {
    bool active;

    struct Entry {
        double total_ms = 0.0;
        int count = 0;
    };

    std::unordered_map<std::string, Entry> sections;
    std::chrono::high_resolution_clock::time_point current_start;
    std::string current_section;

    explicit SectionTimer(bool enable = false) : active(enable) {}

    void begin(const std::string& name) {
        if (!active) return;
        current_section = name;
        current_start = std::chrono::high_resolution_clock::now();
    }

    void end() {
        if (!active) return;
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - current_start).count();
        sections[current_section].total_ms += elapsed;
        sections[current_section].count++;
    }

    /// Return cumulative ms per section (suitable for R named numeric vector)
    std::unordered_map<std::string, double> results() const {
        std::unordered_map<std::string, double> out;
        for (const auto& kv : sections) {
            out[kv.first] = kv.second.total_ms;
        }
        return out;
    }
};

}  // namespace profiling
}  // namespace FactorNet
