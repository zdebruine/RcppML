// sparsepress - Delta encoding for sorted row indices
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_DELTA_HPP
#define SPARSEPRESS_DELTA_HPP

#include <sparsepress/core/types.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace sparsepress {

// ============================================================================
// Delta encoding for sorted row indices within a column
//
// For sorted indices [i0, i1, i2, ...]:
//   gap[0] = i0
//   gap[k] = i[k] - i[k-1] - 1    (subtract 1 since strictly increasing)
//
// All gaps are non-negative. Distribution is approximately geometric
// with parameter related to column density.
// ============================================================================
namespace delta {

// Compute gaps from sorted row indices for one column
inline void encode_column(const uint32_t* indices, uint32_t count,
                          uint32_t* gaps) {
    if (count == 0) return;
    gaps[0] = indices[0];
    for (uint32_t k = 1; k < count; ++k) {
        gaps[k] = indices[k] - indices[k - 1] - 1;
    }
}

// Reconstruct sorted row indices from gaps for one column
inline void decode_column(const uint32_t* gaps, uint32_t count,
                          uint32_t* indices) {
    if (count == 0) return;
    indices[0] = gaps[0];
    for (uint32_t k = 1; k < count; ++k) {
        indices[k] = indices[k - 1] + 1 + gaps[k];
    }
}

// Encode all columns, writing gaps into a flat output array
// Returns the flat gap array (same size as nnz)
inline std::vector<uint32_t> encode_all(const CSCMatrix& mat) {
    std::vector<uint32_t> gaps(mat.nnz);
    for (uint32_t j = 0; j < mat.n; ++j) {
        uint32_t start = mat.p[j];
        uint32_t end = mat.p[j + 1];
        encode_column(mat.i.data() + start, end - start, gaps.data() + start);
    }
    return gaps;
}

// Decode all columns from flat gap array back to row indices
inline void decode_all(const uint32_t* gaps, const uint32_t* col_ptrs,
                       uint32_t n, uint32_t* indices) {
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t start = col_ptrs[j];
        uint32_t end = col_ptrs[j + 1];
        decode_column(gaps + start, end - start, indices + start);
    }
}

} // namespace delta
} // namespace sparsepress

#endif // SPARSEPRESS_DELTA_HPP
