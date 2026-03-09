// StreamPress Chunk Iterator — sequential chunk access for C++ users
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef STREAMPRESS_CHUNK_ITERATOR_HPP
#define STREAMPRESS_CHUNK_ITERATOR_HPP

#include <streampress/sparsepress_v2.hpp>
#include <streampress/core/platform_io.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace streampress {

/// Result from decoding one chunk
struct ChunkRef {
    std::vector<uint32_t> col_ptr;   ///< CSC column pointers (length num_cols+1)
    std::vector<uint32_t> row_ind;   ///< CSC row indices
    std::vector<float>    values;    ///< Non-zero values (fp32)
    uint32_t col_start;              ///< First column index (0-based) in original matrix
    uint32_t col_end;                ///< One-past-last column index (0-based)
    uint32_t nrows;                  ///< Number of rows in original matrix
};

/// Sequential iterator over chunks of a v2 .spz file.
///
/// Usage:
///   ChunkIterator it("file.spz");
///   while (it.has_next()) {
///       ChunkRef chunk = it.next();
///       // process chunk...
///   }
class ChunkIterator {
public:
    /// Construct for forward (column) iteration
    /// @param path Path to .spz file
    /// @param threads Number of decode threads (0 = all available)
    explicit ChunkIterator(const std::string& path, int threads = 0)
        : path_(path), threads_(threads), current_idx_(0), use_transpose_(false) {
        load_file();
    }

    /// Construct with transpose option
    /// @param path Path to .spz file
    /// @param threads Number of decode threads (0 = all available)
    /// @param use_transpose If true, iterate over transpose chunks (row chunks)
    ChunkIterator(const std::string& path, int threads, bool use_transpose)
        : path_(path), threads_(threads), current_idx_(0), use_transpose_(use_transpose) {
        load_file();
    }

    /// Check if more chunks remain
    bool has_next() const { return current_idx_ < num_chunks_; }

    /// Decode and return the next chunk; advances cursor
    ChunkRef next() {
        if (!has_next())
            throw std::runtime_error("ChunkIterator::next() called past end");

        // Use the full decompress path with column range to get just this chunk
        uint32_t chunk_cols = use_transpose_ ? transp_chunk_cols_ : chunk_cols_;
        uint32_t col_start = current_idx_ * chunk_cols;
        uint32_t num_total_cols = use_transpose_ ? hdr_.m : hdr_.n;
        uint32_t col_end = std::min(col_start + chunk_cols, num_total_cols);

        v2::DecompressConfig_v2 dcfg;
        dcfg.reorder = true;
        dcfg.num_threads = threads_;
        dcfg.col_start = static_cast<int>(col_start);
        dcfg.col_end = static_cast<int>(col_end);

        CSCMatrix mat;
        if (use_transpose_) {
            // For transpose iteration, read the transpose section
            // and extract columns col_start..col_end of A^T
            // (which are rows of A)
            mat = v2::decompress_v2(data_.data(), data_.size(), dcfg, nullptr);
        } else {
            mat = v2::decompress_v2(data_.data(), data_.size(), dcfg, nullptr);
        }

        ChunkRef ref;
        ref.col_ptr = std::move(mat.p);
        ref.row_ind = std::move(mat.i);
        ref.values.resize(mat.x.size());
        for (size_t k = 0; k < mat.x.size(); ++k)
            ref.values[k] = mat.x[k];
        ref.col_start = col_start;
        ref.col_end = col_end;
        ref.nrows = use_transpose_ ? hdr_.n : hdr_.m;

        ++current_idx_;
        return ref;
    }

    /// Rewind to first chunk
    void reset() { current_idx_ = 0; }

    /// Total number of chunks
    uint32_t num_chunks() const { return num_chunks_; }

    /// Current chunk index (0-based)
    uint32_t current_chunk_idx() const { return current_idx_; }

    /// Number of rows in the original matrix
    uint32_t nrows() const { return hdr_.m; }

    /// Number of columns in the original matrix
    uint32_t ncols() const { return hdr_.n; }

    /// Number of non-zeros in the full matrix
    uint64_t nnz() const { return hdr_.nnz; }

private:
    void load_file() {
        data_ = read_compressed(path_);
        if (data_.size() < v2::HEADER_SIZE_V2)
            throw std::runtime_error("File too small for v2 header: " + path_);

        hdr_ = v2::FileHeader_v2::deserialize(data_.data());
        if (!hdr_.valid())
            throw std::runtime_error("Invalid v2 file header: " + path_);

        chunk_cols_ = hdr_.chunk_cols;
        transp_chunk_cols_ = hdr_.transp_chunk_cols();
        if (transp_chunk_cols_ == 0) transp_chunk_cols_ = chunk_cols_;

        if (use_transpose_ && hdr_.transpose_offset == 0)
            throw std::runtime_error(
                "File has no transpose section; re-write with include_transpose=TRUE: " + path_);

        uint32_t total = use_transpose_ ? hdr_.m : hdr_.n;
        uint32_t cc = use_transpose_ ? transp_chunk_cols_ : chunk_cols_;
        num_chunks_ = (total + cc - 1) / cc;
    }

    std::string path_;
    int threads_;
    uint32_t current_idx_;
    bool use_transpose_;
    std::vector<uint8_t> data_;
    v2::FileHeader_v2 hdr_;
    uint32_t chunk_cols_;
    uint32_t transp_chunk_cols_;
    uint32_t num_chunks_;
};

} // namespace streampress

#endif // STREAMPRESS_CHUNK_ITERATOR_HPP
