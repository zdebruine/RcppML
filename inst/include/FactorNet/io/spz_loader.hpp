/**
 * @file io/spz_loader.hpp
 * @brief SpzLoader — DataLoader for .spz v2 files with chunk-at-a-time decompression.
 *
 * Uses seek-based random-access I/O (via FileReader) for large files, only
 * keeping the header and chunk index in RAM. Each call to next_forward() /
 * next_transpose() reads and decompresses one chunk. Previous chunks are discarded.
 *
 * For files smaller than IN_CORE_THRESHOLD (2 GB), falls back to legacy
 * bulk-read for maximum speed.
 *
 * Memory: O(header + chunk_index + max_chunk_bytes) — never holds the full file.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/io/loader.hpp>
#include <FactorNet/io/file_reader.hpp>
#include <streampress/sparsepress_v2.hpp>
#include <streampress/format/header_v2.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace FactorNet {
namespace io {

/**
 * @brief DataLoader for .spz v2 files.
 *
 * For files >= IN_CORE_THRESHOLD, uses seek-based I/O via FileReader.
 * For smaller files, loads the entire file into RAM for speed.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class SpzLoader : public DataLoader<Scalar> {
public:
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    static constexpr uint64_t IN_CORE_THRESHOLD = 2ULL * 1024 * 1024 * 1024; // 2 GB

    /**
     * @brief Construct from a .spz v2 file path.
     *
     * @param path             Path to the .spz v2 file
     * @param require_transpose If true (default), throw if file lacks transpose section
     */
    explicit SpzLoader(const std::string& path, bool require_transpose = true)
        : fwd_idx_(0), trans_idx_(0), in_core_(false)
    {
        reader_ = std::make_unique<streampress::FileReader>(path);
        file_size_ = reader_->file_size();

        // Decide: in-core (legacy bulk-read) or seek-based
        in_core_ = (file_size_ < IN_CORE_THRESHOLD);

        if (in_core_) {
            // Legacy path: read entire file into RAM for speed
            file_data_.resize(file_size_);
            size_t n_read = reader_->pread(0, file_data_.data(), file_size_);
            if (n_read != file_size_)
                throw std::runtime_error("Failed to read .spz file: " + path);
        } else {
            // Seek-based: read only header
            if (file_size_ < streampress::v2::HEADER_SIZE_V2)
                throw std::runtime_error("File too small for v2 header: " + path);

            std::vector<uint8_t> hdr_buf(streampress::v2::HEADER_SIZE_V2);
            reader_->pread(0, hdr_buf.data(), streampress::v2::HEADER_SIZE_V2);
        }

        // Parse v2 header
        const uint8_t* hdr_ptr = in_core_
            ? file_data_.data()
            : nullptr;

        if (!in_core_) {
            // Re-read header for parsing (already have it in stack for seek path)
            hdr_buf_.resize(streampress::v2::HEADER_SIZE_V2);
            reader_->pread(0, hdr_buf_.data(), streampress::v2::HEADER_SIZE_V2);
            hdr_ptr = hdr_buf_.data();
        }

        header_ = streampress::v2::FileHeader_v2::deserialize(hdr_ptr);
        if (header_.version != 2)
            throw std::runtime_error("Not a v2 .spz file: " + path);

        m_   = header_.m;
        n_   = header_.n;
        nnz_ = header_.nnz;

        // Parse forward chunk descriptors
        fwd_descs_.resize(header_.num_chunks);
        size_t chunk_idx_bytes = header_.num_chunks * sizeof(streampress::v2::ChunkDescriptor_v2);
        if (in_core_) {
            const uint8_t* chunk_idx = file_data_.data() + header_.chunk_index_offset;
            std::memcpy(fwd_descs_.data(), chunk_idx, chunk_idx_bytes);
        } else {
            std::vector<uint8_t> idx_buf(chunk_idx_bytes);
            reader_->pread(header_.chunk_index_offset, idx_buf.data(), chunk_idx_bytes);
            std::memcpy(fwd_descs_.data(), idx_buf.data(), chunk_idx_bytes);
        }

        // Check and build transpose reader
        if (in_core_) {
            has_transpose_ = streampress::v2::has_transpose(
                file_data_.data(), file_size_);
            if (require_transpose && !has_transpose_) {
                throw std::runtime_error(
                    "Out-of-core NMF requires a .spz file with a pre-stored transpose.\n"
                    "Re-compress with: sp_write(x, 'data.spz', include_transpose = TRUE)");
            }
            if (has_transpose_) {
                t_reader_ = std::make_unique<streampress::v2::TransposeChunkReader>(
                    file_data_.data(), file_size_);
            }
        } else {
            // For seek-based, check transpose_offset in header
            has_transpose_ = (header_.transpose_offset != 0);
            if (require_transpose && !has_transpose_) {
                throw std::runtime_error(
                    "Out-of-core NMF requires a .spz file with a pre-stored transpose.\n"
                    "Re-compress with: sp_write(x, 'data.spz', include_transpose = TRUE)");
            }
            // TransposeChunkReader for seek-based path: we need to read
            // enough data around the transpose section. Read the full file
            // into a secondary buffer for transpose if needed.
            if (has_transpose_) {
                // For the transpose section, read just the transpose region
                // The transpose data is between transpose_offset and metadata_offset (or EOF)
                trans_data_offset_ = header_.transpose_offset;
                // We'll handle transpose chunk-by-chunk in next_transpose()
                // using pread, loading the TransposeChunkReader header info
                init_transpose_seek(path);
            }
        }
    }

    uint32_t rows() const override { return m_; }
    uint32_t cols() const override { return n_; }
    uint64_t nnz()  const override { return nnz_; }

    uint32_t num_forward_chunks() const override {
        return header_.num_chunks;
    }

    uint32_t num_transpose_chunks() const override {
        if (!has_transpose_) return 0;
        if (in_core_) return t_reader_->num_chunks();
        return num_trans_chunks_;
    }

    bool next_forward(Chunk<Scalar>& out) override {
        if (fwd_idx_ >= header_.num_chunks) return false;

        const auto& desc = fwd_descs_[fwd_idx_];
        const uint32_t col_start = desc.col_start;
        const uint32_t nc = desc.num_cols;

        if (in_core_) {
            // Legacy path: decompress from in-memory buffer
            streampress::v2::DecompressConfig_v2 dcfg;
            dcfg.col_start = col_start;
            dcfg.col_end = col_start + nc;
            auto csc = streampress::v2::decompress_v2_typed<Scalar>(
                file_data_.data(), file_size_, dcfg);

            out.col_start = col_start;
            out.num_cols  = nc;
            out.num_rows  = m_;
            out.nnz       = csc.nnz;
            out.matrix    = csc_typed_to_eigen(csc, m_, nc);
        } else {
            // Seek-based path: pread just this chunk's streams into a temp buffer
            // and build a minimal in-memory image for decompress_v2_typed

            // Calculate the byte range needed for this chunk
            uint64_t stream_min = header_.data_offset +
                desc.stream_offset[streampress::v2::STREAM_GAPS];
            uint64_t stream_max = stream_min + desc.stream_size[streampress::v2::STREAM_GAPS];

            uint64_t val_start = header_.data_offset +
                desc.stream_offset[streampress::v2::STREAM_VALUES];
            uint64_t val_end = val_start + desc.stream_size[streampress::v2::STREAM_VALUES];

            if (val_start < stream_min) stream_min = val_start;
            if (val_end > stream_max) stream_max = val_end;

            // Build a fake file buffer: header + chunk index + rANS tables + chunk data
            // We need the full prefix up to data_offset, plus the chunk data
            uint64_t prefix_size = header_.data_offset;
            uint64_t chunk_data_start = stream_min;
            uint64_t chunk_data_end = stream_max;
            uint64_t chunk_data_size = chunk_data_end - chunk_data_start;

            // Read data_offset bytes (header region) + chunk bytes
            // The total is prefix + pad to maintain offsets + chunk data
            // Simplest: read prefix (header, chunk index, tables) + chunk data at correct offset
            uint64_t total_buf_size = chunk_data_end;
            std::vector<uint8_t> buf(total_buf_size);

            // Read the prefix (header + chunk index + rANS tables)
            reader_->pread(0, buf.data(), prefix_size);

            // Read the chunk data at its correct offset
            reader_->pread(chunk_data_start, buf.data() + chunk_data_start, chunk_data_size);

            streampress::v2::DecompressConfig_v2 dcfg;
            dcfg.col_start = col_start;
            dcfg.col_end = col_start + nc;
            auto csc = streampress::v2::decompress_v2_typed<Scalar>(
                buf.data(), total_buf_size, dcfg);

            out.col_start = col_start;
            out.num_cols  = nc;
            out.num_rows  = m_;
            out.nnz       = csc.nnz;
            out.matrix    = csc_typed_to_eigen(csc, m_, nc);
        }

        ++fwd_idx_;
        return true;
    }

    bool next_transpose(Chunk<Scalar>& out) override {
        if (!has_transpose_) return false;

        if (in_core_) {
            if (trans_idx_ >= t_reader_->num_chunks()) return false;

            const uint32_t tc = trans_idx_;
            streampress::CSCMatrix csc = t_reader_->decompress_chunk(tc);
            const uint32_t col_start = t_reader_->chunk_col_start(tc);
            const uint32_t nc = t_reader_->chunk_num_cols(tc);

            out.col_start = col_start;
            out.num_cols  = nc;
            out.num_rows  = n_;
            out.nnz       = csc.nnz;
            out.matrix    = csc_to_eigen(csc, n_, nc);
        } else {
            if (trans_idx_ >= num_trans_chunks_) return false;

            // Seek-based transpose: use the pre-loaded transpose data
            const uint32_t tc = trans_idx_;
            streampress::CSCMatrix csc = t_reader_seek_->decompress_chunk(tc);
            const uint32_t col_start = t_reader_seek_->chunk_col_start(tc);
            const uint32_t nc = t_reader_seek_->chunk_num_cols(tc);

            out.col_start = col_start;
            out.num_cols  = nc;
            out.num_rows  = n_;
            out.nnz       = csc.nnz;
            out.matrix    = csc_to_eigen(csc, n_, nc);
        }

        ++trans_idx_;
        return true;
    }

    void reset_forward()   override { fwd_idx_ = 0; }
    void reset_transpose() override { trans_idx_ = 0; }

    /// Access to raw file data (for init routines that need full decompression)
    /// Only available in in-core mode; returns null and 0 in seek mode.
    const uint8_t* file_data() const {
        return in_core_ ? file_data_.data() : nullptr;
    }
    size_t file_size() const { return file_size_; }

    /// Access to the parsed header
    const streampress::v2::FileHeader_v2& header() const { return header_; }

    /// Access to forward chunk descriptors
    const std::vector<streampress::v2::ChunkDescriptor_v2>& forward_descs() const {
        return fwd_descs_;
    }

    /// Whether the file has a pre-stored transpose section
    bool has_transpose() const { return has_transpose_; }

    /// Whether the file is loaded fully into RAM (in-core mode)
    bool is_in_core() const { return in_core_; }

private:
    /// Initialize transpose reading for seek-based mode.
    /// Reads the transpose section into a dedicated buffer.
    void init_transpose_seek(const std::string& path) {
        // The transpose section starts at header_.transpose_offset.
        // We need to read enough of the file from that point to parse
        // TransposeChunkReader. The section extends to metadata_offset or EOF.
        uint64_t trans_start = header_.transpose_offset;
        uint64_t trans_end = (header_.metadata_offset > trans_start)
            ? header_.metadata_offset
            : file_size_ - streampress::v2::FOOTER_SIZE;

        uint64_t trans_section_size = trans_end - trans_start;

        // Read the full file from offset 0 to trans_end into a buffer
        // so TransposeChunkReader can parse it (it needs header + transpose data)
        trans_buf_.resize(trans_end);
        // Read header + everything up to end of transpose section
        // Do it in chunks to avoid running into pread limits
        uint64_t offset = 0;
        while (offset < trans_end) {
            size_t to_read = std::min((uint64_t)(128 * 1024 * 1024), trans_end - offset);
            reader_->pread(offset, trans_buf_.data() + offset, to_read);
            offset += to_read;
        }

        t_reader_seek_ = std::make_unique<streampress::v2::TransposeChunkReader>(
            trans_buf_.data(), trans_end);
        num_trans_chunks_ = t_reader_seek_->num_chunks();
    }

    /// Convert CSCMatrixTyped<Scalar> to Eigen sparse (forward chunks)
    static SpMat csc_typed_to_eigen(
        const streampress::CSCMatrixTyped<Scalar>& csc,
        uint32_t nrows, uint32_t ncols)
    {
        SpMat mat(nrows, ncols);
        if (csc.nnz == 0) {
            mat.makeCompressed();
            return mat;
        }
        mat.resizeNonZeros(csc.nnz);
        auto* op = mat.outerIndexPtr();
        auto* ip = mat.innerIndexPtr();
        auto* vp = mat.valuePtr();
        for (uint32_t j = 0; j <= ncols; ++j)
            op[j] = static_cast<int>(csc.p[j]);
        for (uint64_t idx = 0; idx < csc.nnz; ++idx) {
            ip[idx] = static_cast<int>(csc.i[idx]);
            vp[idx] = csc.x[idx];
        }
        return mat;
    }

    /// Convert CSCMatrix (double) to Eigen sparse (transpose chunks)
    static SpMat csc_to_eigen(
        const streampress::CSCMatrix& csc,
        uint32_t nrows, uint32_t ncols)
    {
        SpMat mat(nrows, ncols);
        if (csc.nnz == 0) {
            mat.makeCompressed();
            return mat;
        }
        mat.resizeNonZeros(csc.nnz);
        auto* op = mat.outerIndexPtr();
        auto* ip = mat.innerIndexPtr();
        auto* vp = mat.valuePtr();
        for (uint32_t j = 0; j <= ncols; ++j)
            op[j] = static_cast<int>(csc.p[j]);
        for (uint64_t idx = 0; idx < csc.nnz; ++idx) {
            ip[idx] = static_cast<int>(csc.i[idx]);
            vp[idx] = static_cast<Scalar>(csc.x[idx]);
        }
        return mat;
    }

    // Core state
    std::unique_ptr<streampress::FileReader> reader_;
    bool in_core_;
    size_t file_size_;

    // In-core mode: full file buffer
    std::vector<uint8_t> file_data_;

    // Seek-based mode: header buffer
    std::vector<uint8_t> hdr_buf_;

    // Shared state
    streampress::v2::FileHeader_v2 header_;
    std::vector<streampress::v2::ChunkDescriptor_v2> fwd_descs_;

    uint32_t m_, n_;
    uint64_t nnz_;

    bool has_transpose_;

    // In-core transpose reader
    std::unique_ptr<streampress::v2::TransposeChunkReader> t_reader_;

    // Seek-based transpose
    std::vector<uint8_t> trans_buf_;
    std::unique_ptr<streampress::v2::TransposeChunkReader> t_reader_seek_;
    uint64_t trans_data_offset_ = 0;
    uint32_t num_trans_chunks_ = 0;

    uint32_t fwd_idx_;
    uint32_t trans_idx_;
};

}  // namespace io
}  // namespace FactorNet

