#!/usr/bin/env python3
"""Fix transpose.hpp to use correct streampress API."""

TRANSPOSE_HPP = "/mnt/home/debruinz/RcppML-2/inst/include/streampress/transpose.hpp"

CONTENT = r'''#pragma once
// streampress/transpose.hpp — Post-hoc transpose addition for v2 .spz files
//
// Reads an existing v2 .spz file without a transpose section,
// decompresses the forward data, builds CSC(A^T) via the compress_v2 pipeline,
// and rewrites the file with the transpose section included.

#include <streampress/sparsepress_v2.hpp>
#include <streampress/format/header_v2.hpp>
#include <string>
#include <cstdio>
#include <cstring>

#ifdef SPARSEPRESS_USE_R
#include <Rcpp.h>
#define SP_TRANSPOSE_MSG(...) Rprintf(__VA_ARGS__)
#else
#define SP_TRANSPOSE_MSG(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace streampress {

/// Add a pre-computed transpose section to an existing v2 .spz file.
///
/// Reads the file, decompresses it, builds CSC(A^T) via compress_v2 with
/// include_transpose=true, and rewrites the file. The original compression
/// settings (precision, row_sort, etc.) are preserved.
///
/// @param path     Path to the .spz v2 file
/// @param verbose  Print progress messages
/// @return true on success
inline bool add_transpose(const std::string& path, bool verbose = true) {
    // Read whole file
    std::vector<uint8_t> data = v2::read_v2(path);
    if (data.size() < v2::HEADER_SIZE_V2) {
        SP_TRANSPOSE_MSG("Error: file too small for v2 header\n");
        return false;
    }

    // Parse header to check state and extract settings
    v2::FileHeader_v2 hdr = v2::FileHeader_v2::deserialize(data.data());
    if (hdr.magic != v2::MAGIC_SPRZ) {
        SP_TRANSPOSE_MSG("Error: not a v2 .spz file\n");
        return false;
    }
    if (hdr.transpose_offset != 0) {
        if (verbose) SP_TRANSPOSE_MSG("File already has a transpose section.\n");
        return true;
    }

    if (verbose) {
        SP_TRANSPOSE_MSG("[transpose] Reading %u x %u matrix (nnz=%lu)...\n",
                         hdr.m, hdr.n, (unsigned long)hdr.nnz);
    }

    // Decompress forward data
    v2::DecompressConfig_v2 dcfg;
    dcfg.reorder = false;  // keep permuted order for faithful re-encode
    CSCMatrix mat = v2::decompress_v2(data.data(), data.size(), dcfg);
    data.clear();  // free memory

    if (verbose) {
        SP_TRANSPOSE_MSG("[transpose] Recompressing with transpose...\n");
    }

    // Reconstruct compression config from header
    v2::CompressConfig_v2 cfg;
    cfg.include_transpose = true;
    cfg.verbose = verbose ? 1 : 0;
    cfg.chunk_cols = hdr.chunk_cols;
    cfg.row_sort = (hdr.flags & 0x04) != 0;
    cfg.use_delta_prediction = (hdr.flags & 0x01) != 0;
    cfg.use_value_prediction = (hdr.flags & 0x02) != 0;

    // Reconstruct precision string from value_type
    switch (static_cast<v2::ValueType_v2>(hdr.value_type)) {
        case v2::ValueType_v2::FLOAT32: cfg.precision = "float"; break;
        case v2::ValueType_v2::FLOAT16: cfg.precision = "half"; break;
        case v2::ValueType_v2::QUANT8:  cfg.precision = "quant8"; break;
        case v2::ValueType_v2::UINT8:
        case v2::ValueType_v2::UINT16:
        case v2::ValueType_v2::UINT32:  cfg.precision = "auto"; break;
        default:                        cfg.precision = "auto"; break;
    }

    // Compress with transpose
    v2::CompressStats_v2 stats;
    std::vector<uint8_t> compressed = v2::compress_v2(mat, cfg, &stats);

    // Overwrite original file
    v2::write_v2(path, compressed);

    if (verbose) {
        SP_TRANSPOSE_MSG("[transpose] Done. Transpose added to %s (%u chunks)\n",
                         path.c_str(), stats.num_chunks);
    }

    return true;
}

}  // namespace streampress
'''

with open(TRANSPOSE_HPP, 'w') as f:
    f.write(CONTENT)
print("[OK] Rewrote transpose.hpp")
