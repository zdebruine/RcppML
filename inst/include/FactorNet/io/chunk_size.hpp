#pragma once
#include <cstdint>
#include <algorithm>

namespace streampress {

/// Heuristic chunk column count for streaming NMF.
/// Targets ~1% of available RAM per chunk.
inline uint32_t choose_chunk_cols(uint64_t m, uint64_t ram_avail_bytes) {
    if (ram_avail_bytes == 0) return 2048;
    uint64_t bytes_per_col = m * sizeof(float);
    if (bytes_per_col == 0) return 2048;
    uint64_t target_bytes = ram_avail_bytes / 100;
    uint64_t auto_cols = target_bytes / bytes_per_col;
    return static_cast<uint32_t>(std::clamp(auto_cols, (uint64_t)256, (uint64_t)32768));
}

} // namespace streampress
