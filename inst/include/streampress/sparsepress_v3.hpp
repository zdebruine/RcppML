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

#include <streampress/format/header_v3.hpp>
#include <streampress/format/header_v2.hpp>  // Footer_v2, float_to_half, half_to_float
#include <streampress/format/checksum.hpp>    // CRC32
#include <streampress/codec/rans.hpp>

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

namespace streampress {
namespace v3 {

// ============================================================================
// Dense Compression Pipeline
// ============================================================================

/// Apply XOR-delta encoding to a byte stream (in-place).
/// XOR each element with the previous to reduce entropy for smooth data.
inline void xor_delta_encode(uint8_t* data, size_t size) {
    if (size < 2) return;
    for (size_t i = size - 1; i >= 1; --i)
        data[i] ^= data[i - 1];
}

/// Undo XOR-delta encoding (in-place).
inline void xor_delta_decode(uint8_t* data, size_t size) {
    for (size_t i = 1; i < size; ++i)
        data[i] ^= data[i - 1];
}

/// Encode a dense column-major panel with the given codec.
/// Returns compressed bytes. `m` = rows, `num_cols` = columns in this panel.
inline std::vector<uint8_t> encode_dense_chunk(
    const float* col_data, uint32_t m, uint32_t num_cols,
    DenseCodec codec, bool delta_encode)
{
    const size_t n_elems = static_cast<size_t>(m) * num_cols;

    if (codec == DenseCodec::RAW_FP32) {
        // No compression — return raw bytes
        std::vector<uint8_t> out(n_elems * 4);
        std::memcpy(out.data(), col_data, out.size());
        return out;
    }

    if (codec == DenseCodec::FP16 || codec == DenseCodec::FP16_RANS) {
        // Convert fp32 → fp16
        std::vector<uint16_t> fp16(n_elems);
        for (size_t i = 0; i < n_elems; ++i)
            fp16[i] = streampress::v2::float_to_half(col_data[i]);

        // Get raw bytes of fp16 data
        std::vector<uint8_t> raw(n_elems * 2);
        std::memcpy(raw.data(), fp16.data(), raw.size());

        if (delta_encode)
            xor_delta_encode(raw.data(), raw.size());

        if (codec == DenseCodec::FP16) {
            return raw;  // fp16 without rANS
        }

        // FP16_RANS: apply rANS entropy coding on bytes
        std::vector<uint32_t> symbols(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            symbols[i] = raw[i];

        RansTable table = rans::build_table(symbols.data(), symbols.size(), 255);
        std::vector<uint8_t> table_bytes = table.serialize();
        std::vector<uint8_t> encoded = rans::encode_array(symbols.data(), symbols.size(), table);

        // Format: [table_size(4)][table_bytes][encoded_bytes]
        std::vector<uint8_t> out;
        out.reserve(4 + table_bytes.size() + encoded.size());
        uint32_t tbl_sz = static_cast<uint32_t>(table_bytes.size());
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&tbl_sz),
                   reinterpret_cast<uint8_t*>(&tbl_sz) + 4);
        out.insert(out.end(), table_bytes.begin(), table_bytes.end());
        out.insert(out.end(), encoded.begin(), encoded.end());
        return out;
    }

    if (codec == DenseCodec::FP32_RANS) {
        std::vector<uint8_t> raw(n_elems * 4);
        std::memcpy(raw.data(), col_data, raw.size());

        if (delta_encode)
            xor_delta_encode(raw.data(), raw.size());

        std::vector<uint32_t> symbols(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            symbols[i] = raw[i];

        RansTable table = rans::build_table(symbols.data(), symbols.size(), 255);
        std::vector<uint8_t> table_bytes = table.serialize();
        std::vector<uint8_t> encoded = rans::encode_array(symbols.data(), symbols.size(), table);

        std::vector<uint8_t> out;
        out.reserve(4 + table_bytes.size() + encoded.size());
        uint32_t tbl_sz = static_cast<uint32_t>(table_bytes.size());
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&tbl_sz),
                   reinterpret_cast<uint8_t*>(&tbl_sz) + 4);
        out.insert(out.end(), table_bytes.begin(), table_bytes.end());
        out.insert(out.end(), encoded.begin(), encoded.end());
        return out;
    }

    if (codec == DenseCodec::QUANT8) {
        // Range quantization to 8 bits per value
        float vmin = col_data[0], vmax = col_data[0];
        for (size_t i = 1; i < n_elems; ++i) {
            if (col_data[i] < vmin) vmin = col_data[i];
            if (col_data[i] > vmax) vmax = col_data[i];
        }
        float range = vmax - vmin;
        float scale = (range > 0) ? 255.0f / range : 0.0f;

        // Format: [vmin(4)][vmax(4)][quant8_bytes(n_elems)]
        std::vector<uint8_t> out(8 + n_elems);
        std::memcpy(out.data(), &vmin, 4);
        std::memcpy(out.data() + 4, &vmax, 4);
        for (size_t i = 0; i < n_elems; ++i) {
            float q = (col_data[i] - vmin) * scale;
            out[8 + i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, q + 0.5f)));
        }
        return out;
    }

    throw std::runtime_error("Unknown DenseCodec");
}

/// Decode a compressed dense chunk back to float32.
inline void decode_dense_chunk(
    const uint8_t* compressed, uint64_t compressed_size,
    float* out, uint32_t m, uint32_t num_cols,
    DenseCodec codec, bool delta_encode,
    uint64_t uncompressed_size)
{
    const size_t n_elems = static_cast<size_t>(m) * num_cols;

    if (codec == DenseCodec::RAW_FP32) {
        std::memcpy(out, compressed, n_elems * 4);
        return;
    }

    if (codec == DenseCodec::FP16) {
        std::vector<uint8_t> raw(compressed, compressed + compressed_size);
        if (delta_encode)
            xor_delta_decode(raw.data(), raw.size());
        const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n_elems; ++i)
            out[i] = streampress::v2::half_to_float(fp16[i]);
        return;
    }

    if (codec == DenseCodec::FP16_RANS) {
        // Parse: [table_size(4)][table_bytes][encoded_bytes]
        uint32_t tbl_sz;
        std::memcpy(&tbl_sz, compressed, 4);
        const uint8_t* tbl_ptr = compressed + 4;
        RansTable table = RansTable::deserialize(tbl_ptr);
        const uint8_t* enc_ptr = compressed + 4 + tbl_sz;
        size_t enc_size = compressed_size - 4 - tbl_sz;

        size_t n_bytes = n_elems * 2;  // fp16 = 2 bytes per elem
        std::vector<uint32_t> symbols(n_bytes);
        rans::decode_array(enc_ptr, enc_size, symbols.data(), n_bytes, table);

        std::vector<uint8_t> raw(n_bytes);
        for (size_t i = 0; i < n_bytes; ++i)
            raw[i] = static_cast<uint8_t>(symbols[i]);

        if (delta_encode)
            xor_delta_decode(raw.data(), raw.size());

        const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n_elems; ++i)
            out[i] = streampress::v2::half_to_float(fp16[i]);
        return;
    }

    if (codec == DenseCodec::FP32_RANS) {
        uint32_t tbl_sz;
        std::memcpy(&tbl_sz, compressed, 4);
        const uint8_t* tbl_ptr = compressed + 4;
        RansTable table = RansTable::deserialize(tbl_ptr);
        const uint8_t* enc_ptr = compressed + 4 + tbl_sz;
        size_t enc_size = compressed_size - 4 - tbl_sz;

        size_t n_bytes = n_elems * 4;
        std::vector<uint32_t> symbols(n_bytes);
        rans::decode_array(enc_ptr, enc_size, symbols.data(), n_bytes, table);

        std::vector<uint8_t> raw(n_bytes);
        for (size_t i = 0; i < n_bytes; ++i)
            raw[i] = static_cast<uint8_t>(symbols[i]);

        if (delta_encode)
            xor_delta_decode(raw.data(), raw.size());

        std::memcpy(out, raw.data(), n_bytes);
        return;
    }

    if (codec == DenseCodec::QUANT8) {
        float vmin, vmax;
        std::memcpy(&vmin, compressed, 4);
        std::memcpy(&vmax, compressed + 4, 4);
        float range = vmax - vmin;
        float inv_scale = (range > 0) ? range / 255.0f : 0.0f;
        for (size_t i = 0; i < n_elems; ++i)
            out[i] = vmin + compressed[8 + i] * inv_scale;
        return;
    }

    throw std::runtime_error("Unknown DenseCodec for decode");
}

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
 * @param codec    Compression codec (default RAW_FP32 = uncompressed)
 * @param use_delta  If true, apply XOR-delta before entropy coding
 */
template<typename Scalar>
void write_v3(const std::string& path,
              const Scalar* data,
              uint32_t m, uint32_t n,
              uint32_t chunk_cols = DEFAULT_CHUNK_COLS,
              bool include_transpose = false,
              DenseCodec codec = DenseCodec::RAW_FP32,
              bool use_delta = false)
{
    static_assert(sizeof(Scalar) == 4 || sizeof(Scalar) == 8,
                  "Only float32 and float64 are supported");

    const DenseValueType vtype = (sizeof(Scalar) == 4)
        ? DenseValueType::FLOAT32
        : DenseValueType::FLOAT64;
    const uint8_t vbytes = dense_value_bytes(vtype);
    const bool compressed = (codec != DenseCodec::RAW_FP32);

    // Descriptor sizes differ for compressed vs raw
    const size_t desc_size = compressed
        ? sizeof(DenseChunkDescriptorExt)
        : sizeof(DenseChunkDescriptor);

    // Compute chunk counts
    const uint32_t num_fwd_chunks = (n + chunk_cols - 1) / chunk_cols;
    const uint32_t num_trans_chunks = include_transpose
        ? (m + chunk_cols - 1) / chunk_cols
        : 0;

    // -- Compress forward panels --
    std::vector<std::vector<uint8_t>> fwd_compressed(num_fwd_chunks);
    std::vector<uint32_t> fwd_col_start(num_fwd_chunks);
    std::vector<uint32_t> fwd_num_cols(num_fwd_chunks);

    for (uint32_t c = 0; c < num_fwd_chunks; ++c) {
        fwd_col_start[c] = c * chunk_cols;
        fwd_num_cols[c] = std::min(chunk_cols, n - fwd_col_start[c]);
        const size_t n_elems = static_cast<size_t>(m) * fwd_num_cols[c];

        if (compressed) {
            // Need float32 for codec pipeline
            std::vector<float> panel_f32(n_elems);
            const Scalar* panel = data + static_cast<uint64_t>(fwd_col_start[c]) * m;
            for (size_t i = 0; i < n_elems; ++i)
                panel_f32[i] = static_cast<float>(panel[i]);
            fwd_compressed[c] = encode_dense_chunk(
                panel_f32.data(), m, fwd_num_cols[c], codec, use_delta);
        } else {
            // Raw: just copy bytes
            fwd_compressed[c].resize(n_elems * vbytes);
            const Scalar* panel = data + static_cast<uint64_t>(fwd_col_start[c]) * m;
            std::memcpy(fwd_compressed[c].data(), panel, n_elems * vbytes);
        }
    }

    // -- Compress transpose panels --
    std::vector<std::vector<uint8_t>> trans_compressed(num_trans_chunks);
    std::vector<uint32_t> trans_col_start(num_trans_chunks);
    std::vector<uint32_t> trans_num_cols(num_trans_chunks);

    for (uint32_t c = 0; c < num_trans_chunks; ++c) {
        trans_col_start[c] = c * chunk_cols;
        trans_num_cols[c] = std::min(chunk_cols, m - trans_col_start[c]);
        const size_t n_elems = static_cast<size_t>(n) * trans_num_cols[c];

        // Build transpose panel
        std::vector<float> panel_f32(n_elems);
        for (uint32_t lj = 0; lj < trans_num_cols[c]; ++lj) {
            const uint32_t row_in_A = trans_col_start[c] + lj;
            for (uint32_t i = 0; i < n; ++i) {
                panel_f32[static_cast<size_t>(lj) * n + i] =
                    static_cast<float>(data[static_cast<uint64_t>(i) * m + row_in_A]);
            }
        }

        if (compressed) {
            trans_compressed[c] = encode_dense_chunk(
                panel_f32.data(), n, trans_num_cols[c], codec, use_delta);
        } else {
            // Raw: convert panel_f32 to Scalar and store
            trans_compressed[c].resize(n_elems * vbytes);
            if (sizeof(Scalar) == 4) {
                std::memcpy(trans_compressed[c].data(), panel_f32.data(), n_elems * 4);
            } else {
                std::vector<Scalar> panel_typed(n_elems);
                for (size_t i = 0; i < n_elems; ++i)
                    panel_typed[i] = static_cast<Scalar>(panel_f32[i]);
                std::memcpy(trans_compressed[c].data(), panel_typed.data(), n_elems * vbytes);
            }
        }
    }

    // -- Build descriptors with actual compressed sizes --
    uint64_t fwd_data_size = 0;
    std::vector<DenseChunkDescriptorExt> fwd_descs_ext(num_fwd_chunks);
    std::vector<DenseChunkDescriptor> fwd_descs(num_fwd_chunks);

    for (uint32_t c = 0; c < num_fwd_chunks; ++c) {
        if (compressed) {
            fwd_descs_ext[c].col_start = fwd_col_start[c];
            fwd_descs_ext[c].num_cols = fwd_num_cols[c];
            fwd_descs_ext[c]._pad = 0;
            fwd_descs_ext[c]._pad2 = 0;
            fwd_descs_ext[c].byte_offset = fwd_data_size;
            fwd_descs_ext[c].byte_size = fwd_compressed[c].size();
            fwd_descs_ext[c].uncompressed_size =
                static_cast<uint64_t>(m) * fwd_num_cols[c] * sizeof(float);
        } else {
            fwd_descs[c].col_start = fwd_col_start[c];
            fwd_descs[c].num_cols = fwd_num_cols[c];
            fwd_descs[c].byte_offset = fwd_data_size;
            fwd_descs[c].byte_size = fwd_compressed[c].size();
        }
        fwd_data_size += fwd_compressed[c].size();
    }

    uint64_t trans_data_size = 0;
    std::vector<DenseChunkDescriptorExt> trans_descs_ext(num_trans_chunks);
    std::vector<DenseChunkDescriptor> trans_descs(num_trans_chunks);

    for (uint32_t c = 0; c < num_trans_chunks; ++c) {
        if (compressed) {
            trans_descs_ext[c].col_start = trans_col_start[c];
            trans_descs_ext[c].num_cols = trans_num_cols[c];
            trans_descs_ext[c]._pad = 0;
            trans_descs_ext[c]._pad2 = 0;
            trans_descs_ext[c].byte_offset = trans_data_size;
            trans_descs_ext[c].byte_size = trans_compressed[c].size();
            trans_descs_ext[c].uncompressed_size =
                static_cast<uint64_t>(n) * trans_num_cols[c] * sizeof(float);
        } else {
            trans_descs[c].col_start = trans_col_start[c];
            trans_descs[c].num_cols = trans_num_cols[c];
            trans_descs[c].byte_offset = trans_data_size;
            trans_descs[c].byte_size = trans_compressed[c].size();
        }
        trans_data_size += trans_compressed[c].size();
    }

    // -- Compute section offsets --
    const uint64_t chunk_index_offset = HEADER_SIZE_V3;
    const uint64_t chunk_index_size =
        static_cast<uint64_t>(num_fwd_chunks) * desc_size;
    const uint64_t data_offset = chunk_index_offset + chunk_index_size;

    uint64_t transpose_index_offset = 0;
    uint64_t transpose_data_offset = 0;
    if (include_transpose) {
        transpose_index_offset = data_offset + fwd_data_size;
        const uint64_t trans_index_size =
            static_cast<uint64_t>(num_trans_chunks) * desc_size;
        transpose_data_offset = transpose_index_offset + trans_index_size;
    }

    const uint64_t metadata_offset = include_transpose
        ? transpose_data_offset + trans_data_size
        : data_offset + fwd_data_size;

    // -- Build header --
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
    header.set_codec(codec);
    header.set_delta_encode(use_delta);

    // -- Write file --
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    // Accumulate CRC32 over everything before the footer
    streampress::CRC32 crc;

    // Header
    auto hdr_buf = header.serialize();
    crc.update(hdr_buf.data(), HEADER_SIZE_V3);
    if (fwrite(hdr_buf.data(), 1, HEADER_SIZE_V3, f) != HEADER_SIZE_V3) {
        fclose(f);
        throw std::runtime_error("Failed to write header");
    }

    // Forward chunk index
    if (num_fwd_chunks > 0) {
        size_t idx_bytes = num_fwd_chunks * desc_size;
        const uint8_t* idx_ptr = compressed
            ? reinterpret_cast<const uint8_t*>(fwd_descs_ext.data())
            : reinterpret_cast<const uint8_t*>(fwd_descs.data());
        crc.update(idx_ptr, idx_bytes);
        if (fwrite(idx_ptr, 1, idx_bytes, f) != idx_bytes) {
            fclose(f);
            throw std::runtime_error("Failed to write forward chunk index");
        }
    }

    // Forward panel data
    for (uint32_t c = 0; c < num_fwd_chunks; ++c) {
        crc.update(fwd_compressed[c]);
        if (fwrite(fwd_compressed[c].data(), 1, fwd_compressed[c].size(), f)
                != fwd_compressed[c].size()) {
            fclose(f);
            throw std::runtime_error("Failed to write forward panel " + std::to_string(c));
        }
    }

    // Transpose section
    if (include_transpose) {
        // Transpose chunk index
        size_t tidx_bytes = num_trans_chunks * desc_size;
        const uint8_t* tidx_ptr = compressed
            ? reinterpret_cast<const uint8_t*>(trans_descs_ext.data())
            : reinterpret_cast<const uint8_t*>(trans_descs.data());
        crc.update(tidx_ptr, tidx_bytes);
        if (fwrite(tidx_ptr, 1, tidx_bytes, f) != tidx_bytes) {
            fclose(f);
            throw std::runtime_error("Failed to write transpose chunk index");
        }

        // Transpose panel data
        for (uint32_t c = 0; c < num_trans_chunks; ++c) {
            crc.update(trans_compressed[c]);
            if (fwrite(trans_compressed[c].data(), 1, trans_compressed[c].size(), f)
                    != trans_compressed[c].size()) {
                fclose(f);
                throw std::runtime_error("Failed to write transpose panel " +
                                         std::to_string(c));
            }
        }
    }

    // Footer (reuse v2 footer structure)
    streampress::v2::Footer_v2 footer;
    footer.metadata_size = 0;
    footer.file_crc32 = crc.finalize();
    footer.total_chunks = num_fwd_chunks;
    auto ftr_buf = footer.serialize();
    if (fwrite(ftr_buf.data(), 1, streampress::v2::FOOTER_SIZE, f) !=
        streampress::v2::FOOTER_SIZE) {
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

    const DenseCodec codec = header.codec();
    const bool is_delta = header.delta_encode();
    const bool compressed = (codec != DenseCodec::RAW_FP32);
    const size_t desc_size = compressed
        ? sizeof(DenseChunkDescriptorExt)
        : sizeof(DenseChunkDescriptor);

    const uint64_t desc_offset = header.chunk_index_offset +
        static_cast<uint64_t>(chunk_idx) * desc_size;

    // Read the appropriate descriptor type
    uint64_t byte_offset, byte_size, uncompressed_sz;

    if (compressed) {
        if (desc_offset + sizeof(DenseChunkDescriptorExt) > size)
            throw std::runtime_error("Chunk descriptor beyond file bounds");
        DenseChunkDescriptorExt desc;
        std::memcpy(&desc, buf + desc_offset, sizeof(desc));
        col_start = desc.col_start;
        num_cols = desc.num_cols;
        byte_offset = desc.byte_offset;
        byte_size = desc.byte_size;
        uncompressed_sz = desc.uncompressed_size;
    } else {
        if (desc_offset + sizeof(DenseChunkDescriptor) > size)
            throw std::runtime_error("Chunk descriptor beyond file bounds");
        DenseChunkDescriptor desc;
        std::memcpy(&desc, buf + desc_offset, sizeof(desc));
        col_start = desc.col_start;
        num_cols = desc.num_cols;
        byte_offset = desc.byte_offset;
        byte_size = desc.byte_size;
        uncompressed_sz = byte_size;
    }

    const uint64_t panel_offset = header.data_offset + byte_offset;
    if (panel_offset + byte_size > size)
        throw std::runtime_error("Panel data beyond file bounds");

    const uint32_t m = header.m;
    const size_t n_elems = static_cast<size_t>(m) * num_cols;
    out_data.resize(n_elems);

    if (compressed) {
        // Decompress to float, then convert to Scalar
        std::vector<float> tmp(n_elems);
        decode_dense_chunk(buf + panel_offset, byte_size,
                           tmp.data(), m, num_cols, codec, is_delta,
                           uncompressed_sz);
        for (size_t i = 0; i < n_elems; ++i)
            out_data[i] = static_cast<Scalar>(tmp[i]);
    } else {
        // Uncompressed: handle type conversion
        const uint8_t on_disk_bytes = dense_value_bytes(
            static_cast<DenseValueType>(header.value_type));
        if (on_disk_bytes == sizeof(Scalar)) {
            std::memcpy(out_data.data(), buf + panel_offset, byte_size);
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

    const DenseCodec codec = header.codec();
    const bool is_delta = header.delta_encode();
    const bool compressed = (codec != DenseCodec::RAW_FP32);
    const size_t desc_size = compressed
        ? sizeof(DenseChunkDescriptorExt)
        : sizeof(DenseChunkDescriptor);

    const uint64_t desc_offset = header.transpose_index_offset +
        static_cast<uint64_t>(chunk_idx) * desc_size;

    uint64_t byte_offset, byte_size, uncompressed_sz;

    if (compressed) {
        if (desc_offset + sizeof(DenseChunkDescriptorExt) > size)
            throw std::runtime_error("Transpose chunk descriptor beyond file bounds");
        DenseChunkDescriptorExt desc;
        std::memcpy(&desc, buf + desc_offset, sizeof(desc));
        col_start = desc.col_start;
        num_cols = desc.num_cols;
        byte_offset = desc.byte_offset;
        byte_size = desc.byte_size;
        uncompressed_sz = desc.uncompressed_size;
    } else {
        if (desc_offset + sizeof(DenseChunkDescriptor) > size)
            throw std::runtime_error("Transpose chunk descriptor beyond file bounds");
        DenseChunkDescriptor desc;
        std::memcpy(&desc, buf + desc_offset, sizeof(desc));
        col_start = desc.col_start;
        num_cols = desc.num_cols;
        byte_offset = desc.byte_offset;
        byte_size = desc.byte_size;
        uncompressed_sz = byte_size;
    }

    const uint64_t panel_offset = header.transpose_data_offset + byte_offset;
    if (panel_offset + byte_size > size)
        throw std::runtime_error("Transpose panel data beyond file bounds");

    const uint32_t n = header.n;
    const size_t n_elems = static_cast<size_t>(n) * num_cols;
    out_data.resize(n_elems);

    if (compressed) {
        std::vector<float> tmp(n_elems);
        decode_dense_chunk(buf + panel_offset, byte_size,
                           tmp.data(), n, num_cols, codec, is_delta,
                           uncompressed_sz);
        for (size_t i = 0; i < n_elems; ++i)
            out_data[i] = static_cast<Scalar>(tmp[i]);
    } else {
        const uint8_t on_disk_bytes = dense_value_bytes(
            static_cast<DenseValueType>(header.value_type));
        if (on_disk_bytes == sizeof(Scalar)) {
            std::memcpy(out_data.data(), buf + panel_offset, byte_size);
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
}  // namespace streampress

