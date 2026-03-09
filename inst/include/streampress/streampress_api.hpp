// StreamPress Public C++ API
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)
//
// This header provides a clean C++ interface to StreamPress functionality
// for downstream C++ users (e.g., FactorNet) without requiring Rcpp.
//
// TODO(python): Python bindings should wrap these C++ API types.

#ifndef STREAMPRESS_API_HPP
#define STREAMPRESS_API_HPP

#include <streampress/core/types.hpp>
#include <streampress/sparsepress_v2.hpp>
#include <streampress/format/header_v2.hpp>
#include <streampress/format/obs_var_table.hpp>
#include <streampress/chunk_iterator.hpp>

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>

namespace streampress {
namespace api {

// ============================================================================
// Option structs
// ============================================================================

struct WriteOptions {
    bool     include_transpose = true;
    bool     use_delta         = true;
    bool     value_pred        = false;
    bool     row_sort          = false;
    uint32_t chunk_cols        = 0;     ///< 0 = compute from chunk_bytes
    uint64_t chunk_bytes       = 64ULL * 1024 * 1024;  ///< 64 MB default
    uint32_t transp_chunk_cols = 0;     ///< 0 = compute from chunk_bytes
    int      threads           = 0;     ///< 0 = all available
    std::string precision      = "auto";
};

struct ReadOptions {
    bool reorder  = true;
    int  threads  = 0;   ///< 0 = all available
};

// ============================================================================
// Result structs
// ============================================================================

struct WriteStats {
    size_t   raw_size         = 0;
    size_t   compressed_size  = 0;
    double   compress_time_ms = 0;
    uint32_t num_chunks       = 0;
    bool     has_transpose    = false;
    bool     has_obs          = false;
    bool     has_var          = false;
    double ratio() const {
        return raw_size > 0 ? static_cast<double>(raw_size) / compressed_size : 0;
    }
};

struct ReadResult {
    std::vector<uint32_t> col_ptr;     ///< CSC column pointers (length n+1)
    std::vector<uint32_t> row_ind;     ///< CSC row indices
    std::vector<float>    values;      ///< Non-zero values (fp32)
    uint32_t m = 0;                    ///< Number of rows
    uint32_t n = 0;                    ///< Number of columns
    uint64_t nnz = 0;                  ///< Number of non-zeros
    std::vector<std::string> rownames;
    std::vector<std::string> colnames;
};

struct FileInfo {
    uint32_t m = 0;                    ///< Number of rows
    uint32_t n = 0;                    ///< Number of columns
    uint64_t nnz = 0;                  ///< Number of non-zeros
    uint32_t chunk_cols = 0;
    uint32_t transp_chunk_cols = 0;
    bool     has_transpose     = false;
    bool     has_obs_table     = false;
    bool     has_var_table     = false;
    bool     has_dimnames      = false;
    float    density           = 0.0f;
    uint64_t file_bytes        = 0;
    uint16_t version           = 0;
};

// ============================================================================
// API Functions
// ============================================================================

/// Write a CSCMatrix to a .spz file
inline WriteStats write_sparse(
    const std::string& path,
    const CSCMatrix& mat,
    const std::vector<uint8_t>& obs_buf = {},
    const std::vector<uint8_t>& var_buf = {},
    const WriteOptions& opts = WriteOptions{})
{
    v2::CompressConfig_v2 cfg;
    cfg.precision = opts.precision;
    cfg.row_sort = opts.row_sort;
    cfg.include_transpose = opts.include_transpose;
    cfg.verbose = 0;

    // Chunk sizing: use explicit chunk_cols or compute from chunk_bytes
    if (opts.chunk_cols > 0) {
        cfg.chunk_cols = opts.chunk_cols;
    } else if (opts.chunk_bytes > 0 && mat.m > 0) {
        uint64_t bytes_per_col = static_cast<uint64_t>(mat.m) * 4;  // fp32
        uint32_t cc = static_cast<uint32_t>(opts.chunk_bytes / std::max(bytes_per_col, uint64_t(1)));
        cfg.chunk_cols = std::max(cc, uint32_t(1));
        cfg.chunk_cols = std::min(cfg.chunk_cols, mat.n);
    }

    cfg.obs_buf = obs_buf;
    cfg.var_buf = var_buf;

    v2::CompressStats_v2 stats2;
    std::vector<uint8_t> compressed = v2::compress_v2(mat, cfg, &stats2);
    v2::write_v2(path, compressed);

    WriteStats ws;
    ws.raw_size = stats2.raw_size;
    ws.compressed_size = stats2.compressed_size;
    ws.compress_time_ms = stats2.compress_time_ms;
    ws.num_chunks = stats2.num_chunks;
    ws.has_transpose = opts.include_transpose;
    ws.has_obs = !obs_buf.empty();
    ws.has_var = !var_buf.empty();
    return ws;
}

/// Read a full sparse matrix from a .spz file
inline ReadResult read_sparse(
    const std::string& path,
    const ReadOptions& opts = ReadOptions{})
{
    std::vector<uint8_t> data = read_compressed(path);

    v2::DecompressConfig_v2 dcfg;
    dcfg.reorder = opts.reorder;
    dcfg.num_threads = opts.threads;

    v2::Metadata meta;
    CSCMatrix mat = v2::decompress_v2(data.data(), data.size(), dcfg, &meta);

    ReadResult rr;
    rr.col_ptr = std::move(mat.p);
    rr.row_ind = std::move(mat.i);
    rr.values.resize(mat.x.size());
    for (size_t k = 0; k < mat.x.size(); ++k)
        rr.values[k] = static_cast<float>(mat.x[k]);
    rr.m = mat.m;
    rr.n = mat.n;
    rr.nnz = mat.nnz;
    rr.rownames = meta.get_rownames();
    rr.colnames = meta.get_colnames();
    return rr;
}

/// Read a contiguous range of columns [col_start, col_end) (0-indexed)
inline ReadResult slice_cols(
    const std::string& path,
    uint32_t col_start, uint32_t col_end,
    const ReadOptions& opts = ReadOptions{})
{
    std::vector<uint8_t> data = read_compressed(path);

    v2::DecompressConfig_v2 dcfg;
    dcfg.reorder = opts.reorder;
    dcfg.num_threads = opts.threads;
    dcfg.col_start = static_cast<int>(col_start);
    dcfg.col_end = static_cast<int>(col_end);

    v2::Metadata meta;
    CSCMatrix mat = v2::decompress_v2(data.data(), data.size(), dcfg, &meta);

    ReadResult rr;
    rr.col_ptr = std::move(mat.p);
    rr.row_ind = std::move(mat.i);
    rr.values.resize(mat.x.size());
    for (size_t k = 0; k < mat.x.size(); ++k)
        rr.values[k] = static_cast<float>(mat.x[k]);
    rr.m = mat.m;
    rr.n = mat.n;
    rr.nnz = mat.nnz;
    rr.rownames = meta.get_rownames();
    auto all_colnames = meta.get_colnames();
    if (!all_colnames.empty() && col_end <= static_cast<uint32_t>(all_colnames.size())) {
        rr.colnames.assign(all_colnames.begin() + col_start,
                           all_colnames.begin() + col_end);
    }
    return rr;
}

/// Get file info without decompression
inline FileInfo info(const std::string& path) {
    std::vector<uint8_t> data = read_compressed(path);
    if (data.size() < v2::HEADER_SIZE_V2)
        throw std::runtime_error("File too small for v2 header: " + path);

    v2::FileHeader_v2 hdr = v2::FileHeader_v2::deserialize(data.data());
    if (!hdr.valid())
        throw std::runtime_error("Invalid v2 file header: " + path);

    FileInfo fi;
    fi.m = hdr.m;
    fi.n = hdr.n;
    fi.nnz = hdr.nnz;
    fi.chunk_cols = hdr.chunk_cols;
    fi.transp_chunk_cols = hdr.transp_chunk_cols();
    fi.has_transpose = (hdr.transpose_offset != 0);
    fi.has_obs_table = (hdr.obs_table_offset() != 0);
    fi.has_var_table = (hdr.var_table_offset() != 0);
    fi.has_dimnames = hdr.has_dimnames();
    fi.density = (hdr.m > 0 && hdr.n > 0)
        ? static_cast<float>(hdr.nnz) / (static_cast<double>(hdr.m) * hdr.n)
        : 0.0f;
    fi.file_bytes = data.size();
    fi.version = hdr.format_version;
    return fi;
}

} // namespace api
} // namespace streampress

#endif // STREAMPRESS_API_HPP
