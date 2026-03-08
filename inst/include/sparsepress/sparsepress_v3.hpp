/**
 * @file sparsepress_v3.hpp
 * @brief SparsePress v3 dense format — write and read dense column-panel files.
 *
 * Write: Takes a column-major dense matrix, splits into chunk_cols-wide panels,
 *        writes each panel's raw bytes sequentially. Optional transpose.
 *
 * Read: Reads header, chunk index, and decompresses individual panels on demand.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <sparsepress/format/header_v3.hpp>
#include <sparsepress/format/header_v2.hpp>  // Footer_v2

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

namespace sparsepress {
namespace v3 {

// ============================================================================
// Write API
// ============================================================================

/**
 * @brief Write a dense matrix to an SPZ v3 file.
 *
 * @tparam Scalar  float or double
 * @param path     Output file path
 * @param data     Column-major dense data, m * n values
 * @param m        Number of rows
 * @param n        Number of columns
 * @param chunk_cols  Columns per chunk (default 256)
 * @param include_transpose  If true, store A^T panels for streaming NMF
 */
template<typename Scalar>
void write_v3(const std::string& path,
              const Scalar* data,
              uint32_t m, uint32_t n,
              uint32_t chunk_cols = DEFAULT_CHUNK_COLS,
              bool include_transpose = false)
{
    static_assert(sizeof(Scalar) == 4 || sizeof(Scalar) == 8,
                  "Only float32 and float64 are supported");

    const DenseValueType vtype = (sizeof(Scalar) == 4)
        ? DenseValueType::FLOAT32
        : DenseValueType::FLOAT64;
    const uint8_t vbytes = dense_value_bytes(vtype);

    // Compute chunks
    const uint32_t num_fwd_chunks = (n + chunk_cols - 1) / chunk_cols;
    const uint32_t num_trans_chunks = include_transpose
        ? (m + chunk_cols - 1) / chunk_cols
        : 0;

    // Build forward chunk descriptors
    std::vector<DenseChunkDescriptor> fwd_descs(num_fwd_chunks);
    uint64_t fwd_data_size = 0;
    for (uint32_t c = 0; c < num_fwd_chunks; ++c) {
        const uint32_t col_start = c * chunk_cols;
        const uint32_t nc = std::min(chunk_cols, n - col_start);
        fwd_descs[c].col_start = col_start;
        fwd_descs[c].num_cols = nc;
        fwd_descs[c].byte_offset = fwd_data_size;
        fwd_descs[c].byte_size = static_cast<uint64_t>(m) * nc * vbytes;
        fwd_data_size += fwd_descs[c].byte_size;
    }

    // Build transpose chunk descriptors (A^T is n × m, column-major)
    std::vector<DenseChunkDescriptor> trans_descs(num_trans_chunks);
    uint64_t trans_data_size = 0;
    for (uint32_t c = 0; c < num_trans_chunks; ++c) {
        const uint32_t col_start = c * chunk_cols;
        const uint32_t nc = std::min(chunk_cols, m - col_start);
        trans_descs[c].col_start = col_start;
        trans_descs[c].num_cols = nc;
        trans_descs[c].byte_offset = trans_data_size;
        trans_descs[c].byte_size = static_cast<uint64_t>(n) * nc * vbytes;
        trans_data_size += trans_descs[c].byte_size;
    }

    // Compute section offsets
    const uint64_t chunk_index_offset = HEADER_SIZE_V3;
    const uint64_t chunk_index_size =
        static_cast<uint64_t>(num_fwd_chunks) * sizeof(DenseChunkDescriptor);
    const uint64_t data_offset = chunk_index_offset + chunk_index_size;

    uint64_t transpose_index_offset = 0;
    uint64_t transpose_data_offset = 0;
    if (include_transpose) {
        transpose_index_offset = data_offset + fwd_data_size;
        const uint64_t trans_index_size =
            static_cast<uint64_t>(num_trans_chunks) * sizeof(DenseChunkDescriptor);
        transpose_data_offset = transpose_index_offset + trans_index_size;
    }

    const uint64_t metadata_offset = include_transpose
        ? transpose_data_offset + trans_data_size
        : data_offset + fwd_data_size;

    // Build header
    FileHeader_v3 header;
    header.m = m;
    header.n = n;
    header.nnz = static_cast<uint64_t>(m) * n;
    header.chunk_cols = chunk_cols;
    header.num_chunks = num_fwd_chunks;
    header.num_transpose_chunks = num_trans_chunks;
    header.value_type = static_cast<uint8_t>(vtype);
    header.has_transpose = include_transpose ? 1 : 0;
    header.chunk_index_offset = chunk_index_offset;
    header.data_offset = data_offset;
    header.transpose_index_offset = transpose_index_offset;
    header.transpose_data_offset = transpose_data_offset;
    header.metadata_offset = metadata_offset;

    // Write file
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    // Header
    auto hdr_buf = header.serialize();
    if (fwrite(hdr_buf.data(), 1, HEADER_SIZE_V3, f) != HEADER_SIZE_V3) {
        fclose(f);
        throw std::runtime_error("Failed to write header");
    }

    // Forward chunk index
    if (num_fwd_chunks > 0) {
        size_t idx_bytes = num_fwd_chunks * sizeof(DenseChunkDescriptor);
        if (fwrite(fwd_descs.data(), 1, idx_bytes, f) != idx_bytes) {
            fclose(f);
            throw std::runtime_error("Failed to write forward chunk index");
        }
    }

    // Forward panel data (column-major slices of the input)
    for (uint32_t c = 0; c < num_fwd_chunks; ++c) {
        const uint32_t col_start = fwd_descs[c].col_start;
        const uint32_t nc = fwd_descs[c].num_cols;
        // data is column-major: column j starts at data + j * m
        const Scalar* panel = data + static_cast<uint64_t>(col_start) * m;
        size_t panel_bytes = static_cast<size_t>(m) * nc * vbytes;
        if (fwrite(panel, 1, panel_bytes, f) != panel_bytes) {
            fclose(f);
            throw std::runtime_error("Failed to write forward panel " + std::to_string(c));
        }
    }

    // Transpose panels (A^T is n × m, but we store it column-major)
    if (include_transpose) {
        // Write transpose chunk index
        size_t tidx_bytes = num_trans_chunks * sizeof(DenseChunkDescriptor);
        if (fwrite(trans_descs.data(), 1, tidx_bytes, f) != tidx_bytes) {
            fclose(f);
            throw std::runtime_error("Failed to write transpose chunk index");
        }

        // Compute A^T in column-major order: A^T[i][j] = A[j][i]
        // A^T is n × m. Column c of A^T (rows 0..n-1) = row c of A.
        // A chunk of A^T columns [col_start, col_start+nc) corresponds to
        // rows [col_start, col_start+nc) of A.
        for (uint32_t c = 0; c < num_trans_chunks; ++c) {
            const uint32_t col_start = trans_descs[c].col_start;
            const uint32_t nc = trans_descs[c].num_cols;
            // Build transpose panel: n rows × nc columns
            // Column j of this panel (j = col_start + local_j) is row j of A
            std::vector<Scalar> panel(static_cast<size_t>(n) * nc);
            for (uint32_t lj = 0; lj < nc; ++lj) {
                const uint32_t row_in_A = col_start + lj;
                for (uint32_t i = 0; i < n; ++i) {
                    // A^T[i, lj] = A[row_in_A, i]
                    // A is column-major: A[row, col] = data[col * m + row]
                    panel[static_cast<size_t>(lj) * n + i] =
                        data[static_cast<uint64_t>(i) * m + row_in_A];
                }
            }
            size_t panel_bytes = static_cast<size_t>(n) * nc * vbytes;
            if (fwrite(panel.data(), 1, panel_bytes, f) != panel_bytes) {
                fclose(f);
                throw std::runtime_error("Failed to write transpose panel " +
                                         std::to_string(c));
            }
        }
    }

    // Footer (reuse v2 footer structure)
    sparsepress::v2::Footer_v2 footer;
    footer.metadata_size = 0;
    footer.file_crc32 = 0;  // TODO: compute CRC32
    footer.total_chunks = num_fwd_chunks;
    auto ftr_buf = footer.serialize();
    if (fwrite(ftr_buf.data(), 1, sparsepress::v2::FOOTER_SIZE, f) !=
        sparsepress::v2::FOOTER_SIZE) {
        fclose(f);
        throw std::runtime_error("Failed to write footer");
    }

    fclose(f);
}

// ============================================================================
// Read API — full matrix read (for testing / sp_read equivalent)
// ============================================================================

/**
 * @brief Read header from an SPZ v3 file.
 *
 * @param buf  Pointer to file data in memory
 * @param size File size in bytes
 * @return Parsed FileHeader_v3
 */
inline FileHeader_v3 read_header_v3(const uint8_t* buf, size_t size) {
    if (size < HEADER_SIZE_V3) {
        throw std::runtime_error("File too small for v3 header");
    }
    return FileHeader_v3::deserialize(buf);
}

/**
 * @brief Read a single dense forward chunk from in-memory file data.
 *
 * @tparam Scalar float or double
 * @param buf        File data pointer
 * @param size       File size
 * @param header     Parsed v3 header
 * @param chunk_idx  Which chunk (0-based)
 * @param[out] out_data  Filled with m × num_cols column-major values
 * @param[out] col_start First column index
 * @param[out] num_cols  Number of columns
 */
template<typename Scalar>
void read_forward_chunk(const uint8_t* buf, size_t size,
                        const FileHeader_v3& header,
                        uint32_t chunk_idx,
                        std::vector<Scalar>& out_data,
                        uint32_t& col_start,
                        uint32_t& num_cols)
{
    if (chunk_idx >= header.num_chunks) {
        throw std::runtime_error("Chunk index out of range");
    }

    // Read chunk descriptor
    DenseChunkDescriptor desc;
    const uint64_t desc_offset = header.chunk_index_offset +
        static_cast<uint64_t>(chunk_idx) * sizeof(DenseChunkDescriptor);
    if (desc_offset + sizeof(DenseChunkDescriptor) > size) {
        throw std::runtime_error("Chunk descriptor beyond file bounds");
    }
    std::memcpy(&desc, buf + desc_offset, sizeof(DenseChunkDescriptor));

    col_start = desc.col_start;
    num_cols = desc.num_cols;

    // Read panel data
    const uint64_t panel_offset = header.data_offset + desc.byte_offset;
    if (panel_offset + desc.byte_size > size) {
        throw std::runtime_error("Panel data beyond file bounds");
    }

    const uint32_t m = header.m;
    const size_t n_elems = static_cast<size_t>(m) * num_cols;
    out_data.resize(n_elems);

    // Handle type conversion: file may store float32 but Scalar may be double
    const uint8_t on_disk_bytes = dense_value_bytes(
        static_cast<DenseValueType>(header.value_type));
    if (on_disk_bytes == sizeof(Scalar)) {
        // Same type — direct memcpy
        std::memcpy(out_data.data(), buf + panel_offset, desc.byte_size);
    } else if (on_disk_bytes == 4 && sizeof(Scalar) == 8) {
        // float32 on disk → double in memory
        const float* src = reinterpret_cast<const float*>(buf + panel_offset);
        for (size_t i = 0; i < n_elems; ++i)
            out_data[i] = static_cast<Scalar>(src[i]);
    } else if (on_disk_bytes == 8 && sizeof(Scalar) == 4) {
        // float64 on disk → float in memory
        const double* src = reinterpret_cast<const double*>(buf + panel_offset);
        for (size_t i = 0; i < n_elems; ++i)
            out_data[i] = static_cast<Scalar>(src[i]);
    } else {
        throw std::runtime_error("Unsupported on-disk value size");
    }
}

/**
 * @brief Read a single dense transpose chunk from in-memory file data.
 *
 * @tparam Scalar float or double
 * @param buf        File data pointer
 * @param size       File size
 * @param header     Parsed v3 header
 * @param chunk_idx  Which transpose chunk (0-based)
 * @param[out] out_data  Filled with n × num_cols column-major values
 * @param[out] col_start First column of transpose (= first row of A)
 * @param[out] num_cols  Number of columns
 */
template<typename Scalar>
void read_transpose_chunk(const uint8_t* buf, size_t size,
                          const FileHeader_v3& header,
                          uint32_t chunk_idx,
                          std::vector<Scalar>& out_data,
                          uint32_t& col_start,
                          uint32_t& num_cols)
{
    if (!header.has_transpose) {
        throw std::runtime_error("File does not contain transpose data");
    }
    if (chunk_idx >= header.num_transpose_chunks) {
        throw std::runtime_error("Transpose chunk index out of range");
    }

    // Read chunk descriptor from transpose index
    DenseChunkDescriptor desc;
    const uint64_t desc_offset = header.transpose_index_offset +
        static_cast<uint64_t>(chunk_idx) * sizeof(DenseChunkDescriptor);
    if (desc_offset + sizeof(DenseChunkDescriptor) > size) {
        throw std::runtime_error("Transpose chunk descriptor beyond file bounds");
    }
    std::memcpy(&desc, buf + desc_offset, sizeof(DenseChunkDescriptor));

    col_start = desc.col_start;
    num_cols = desc.num_cols;

    // Read panel data
    const uint64_t panel_offset = header.transpose_data_offset + desc.byte_offset;
    if (panel_offset + desc.byte_size > size) {
        throw std::runtime_error("Transpose panel data beyond file bounds");
    }

    const uint32_t n = header.n;
    const size_t n_elems = static_cast<size_t>(n) * num_cols;
    out_data.resize(n_elems);

    // Handle type conversion for transpose chunks
    const uint8_t on_disk_bytes = dense_value_bytes(
        static_cast<DenseValueType>(header.value_type));
    if (on_disk_bytes == sizeof(Scalar)) {
        std::memcpy(out_data.data(), buf + panel_offset, desc.byte_size);
    } else if (on_disk_bytes == 4 && sizeof(Scalar) == 8) {
        const float* src = reinterpret_cast<const float*>(buf + panel_offset);
        for (size_t i = 0; i < n_elems; ++i)
            out_data[i] = static_cast<Scalar>(src[i]);
    } else if (on_disk_bytes == 8 && sizeof(Scalar) == 4) {
        const double* src = reinterpret_cast<const double*>(buf + panel_offset);
        for (size_t i = 0; i < n_elems; ++i)
            out_data[i] = static_cast<Scalar>(src[i]);
    } else {
        throw std::runtime_error("Unsupported on-disk value size");
    }
}

/**
 * @brief Read entire dense matrix from SPZ v3 file.
 *
 * @tparam Scalar float or double
 * @param buf   File data in memory
 * @param size  File size
 * @param[out] out_data  Column-major m × n values
 * @param[out] m  Rows
 * @param[out] n  Columns
 */
template<typename Scalar>
void read_full_matrix(const uint8_t* buf, size_t size,
                      std::vector<Scalar>& out_data,
                      uint32_t& m, uint32_t& n)
{
    FileHeader_v3 header = read_header_v3(buf, size);
    m = header.m;
    n = header.n;
    out_data.resize(static_cast<size_t>(m) * n);

    for (uint32_t c = 0; c < header.num_chunks; ++c) {
        std::vector<Scalar> chunk_data;
        uint32_t col_start, num_cols;
        read_forward_chunk<Scalar>(buf, size, header, c,
                                   chunk_data, col_start, num_cols);
        // Copy into output at correct column position
        std::memcpy(out_data.data() + static_cast<size_t>(col_start) * m,
                    chunk_data.data(),
                    static_cast<size_t>(m) * num_cols * sizeof(Scalar));
    }
}

// ============================================================================
// Version detection helper
// ============================================================================

/**
 * @brief Detect SPZ version from file data.
 *
 * @param buf   File data
 * @param size  File size (must be >= 8)
 * @return Version number (1, 2, or 3), or 0 if not a valid SPZ file
 */
inline uint16_t detect_version(const uint8_t* buf, size_t size) {
    if (size < 8) return 0;
    if (std::memcmp(buf, "SPRZ", 4) != 0) return 0;
    uint16_t ver;
    std::memcpy(&ver, buf + 4, 2);
    return ver;
}

}  // namespace v3
}  // namespace sparsepress

