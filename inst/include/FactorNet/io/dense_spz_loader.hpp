/**
 * @file io/dense_spz_loader.hpp
 * @brief DenseSpzLoader — DataLoader for .spz v3 dense files.
 *
 * Uses seek-based random-access I/O (via FileReader) for large files, only
 * keeping the header and chunk index in RAM. For files smaller than
 * IN_CORE_THRESHOLD (2 GB), falls back to legacy bulk-read for speed.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/io/loader.hpp>
#include <FactorNet/io/file_reader.hpp>
#include <streampress/sparsepress_v3.hpp>
#include <streampress/format/header_v3.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace FactorNet {
namespace io {

/**
 * @brief DataLoader for .spz v3 (dense) files.
 *
 * For files >= IN_CORE_THRESHOLD, uses seek-based I/O via FileReader.
 * For smaller files, loads the entire file into RAM for speed.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class DenseSpzLoader : public DataLoader<Scalar> {
public:
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    static constexpr uint64_t IN_CORE_THRESHOLD = 2ULL * 1024 * 1024 * 1024; // 2 GB

    /**
     * @brief Construct from an SPZ v3 file path.
     *
     * @param path              Path to the .spz v3 file
     * @param require_transpose If true (default), throw if file lacks transpose
     */
    explicit DenseSpzLoader(const std::string& path,
                            bool require_transpose = true)
        : fwd_idx_(0), trans_idx_(0), in_core_(false)
    {
        reader_ = std::make_unique<streampress::FileReader>(path);
        file_size_ = reader_->file_size();
        in_core_ = (file_size_ < IN_CORE_THRESHOLD);

        if (in_core_) {
            file_data_.resize(file_size_);
            size_t n_read = reader_->pread(0, file_data_.data(), file_size_);
            if (n_read != file_size_)
                throw std::runtime_error("Failed to read .spz file: " + path);
        }

        // Parse v3 header
        if (in_core_) {
            header_ = streampress::v3::read_header_v3(file_data_.data(), file_size_);
        } else {
            std::vector<uint8_t> hdr_buf(streampress::v3::HEADER_SIZE_V3);
            reader_->pread(0, hdr_buf.data(), streampress::v3::HEADER_SIZE_V3);
            header_ = streampress::v3::FileHeader_v3::deserialize(hdr_buf.data());
        }

        m_ = header_.m;
        n_ = header_.n;
        nnz_ = static_cast<uint64_t>(m_) * n_;

        // Parse forward chunk descriptors
        fwd_descs_.resize(header_.num_chunks);
        size_t fwd_idx_bytes = header_.num_chunks * sizeof(streampress::v3::DenseChunkDescriptor);
        if (in_core_) {
            std::memcpy(fwd_descs_.data(),
                        file_data_.data() + header_.chunk_index_offset,
                        fwd_idx_bytes);
        } else {
            std::vector<uint8_t> idx_buf(fwd_idx_bytes);
            reader_->pread(header_.chunk_index_offset, idx_buf.data(), fwd_idx_bytes);
            std::memcpy(fwd_descs_.data(), idx_buf.data(), fwd_idx_bytes);
        }

        // Parse transpose chunk descriptors if present
        has_transpose_ = (header_.has_transpose != 0);
        if (require_transpose && !has_transpose_) {
            throw std::runtime_error(
                "Out-of-core NMF requires a .spz file with a pre-stored transpose.\n"
                "Re-write with: sp_write_dense(x, 'data.spz', include_transpose = TRUE)");
        }
        if (has_transpose_) {
            trans_descs_.resize(header_.num_transpose_chunks);
            size_t trans_idx_bytes = header_.num_transpose_chunks *
                sizeof(streampress::v3::DenseChunkDescriptor);
            if (in_core_) {
                std::memcpy(trans_descs_.data(),
                            file_data_.data() + header_.transpose_index_offset,
                            trans_idx_bytes);
            } else {
                std::vector<uint8_t> idx_buf(trans_idx_bytes);
                reader_->pread(header_.transpose_index_offset, idx_buf.data(), trans_idx_bytes);
                std::memcpy(trans_descs_.data(), idx_buf.data(), trans_idx_bytes);
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
        return has_transpose_ ? header_.num_transpose_chunks : 0;
    }

    bool next_forward(Chunk<Scalar>& out) override {
        if (fwd_idx_ >= header_.num_chunks) return false;

        const auto& desc = fwd_descs_[fwd_idx_];
        uint32_t col_start = desc.col_start;
        uint32_t num_cols = desc.num_cols;

        std::vector<Scalar> panel;
        if (in_core_) {
            streampress::v3::read_forward_chunk<Scalar>(
                file_data_.data(), file_size_, header_,
                fwd_idx_, panel, col_start, num_cols);
        } else {
            read_dense_panel_seek(desc, header_.data_offset, m_, panel);
        }

        out.col_start = col_start;
        out.num_cols  = num_cols;
        out.num_rows  = m_;
        out.nnz       = static_cast<uint64_t>(m_) * num_cols;
        out.matrix    = dense_to_sparse(panel.data(), m_, num_cols);

        ++fwd_idx_;
        return true;
    }

    bool next_transpose(Chunk<Scalar>& out) override {
        if (!has_transpose_) return false;
        if (trans_idx_ >= header_.num_transpose_chunks) return false;

        const auto& desc = trans_descs_[trans_idx_];
        uint32_t col_start = desc.col_start;
        uint32_t num_cols = desc.num_cols;

        std::vector<Scalar> panel;
        if (in_core_) {
            streampress::v3::read_transpose_chunk<Scalar>(
                file_data_.data(), file_size_, header_,
                trans_idx_, panel, col_start, num_cols);
        } else {
            read_dense_panel_seek(desc, header_.transpose_data_offset, n_, panel);
        }

        out.col_start = col_start;
        out.num_cols  = num_cols;
        out.num_rows  = n_;
        out.nnz       = static_cast<uint64_t>(n_) * num_cols;
        out.matrix    = dense_to_sparse(panel.data(), n_, num_cols);

        ++trans_idx_;
        return true;
    }

    void reset_forward()   override { fwd_idx_ = 0; }
    void reset_transpose() override { trans_idx_ = 0; }

    const streampress::v3::FileHeader_v3& header() const { return header_; }
    bool has_transpose() const { return has_transpose_; }
    bool is_in_core() const { return in_core_; }

    const uint8_t* file_data() const {
        return in_core_ ? file_data_.data() : nullptr;
    }
    size_t file_size() const { return file_size_; }

private:
    /// Read a dense panel via seek for the out-of-core path
    void read_dense_panel_seek(const streampress::v3::DenseChunkDescriptor& desc,
                               uint64_t section_offset,
                               uint32_t panel_rows,
                               std::vector<Scalar>& out_data) const
    {
        const uint64_t panel_offset = section_offset + desc.byte_offset;
        const size_t n_elems = static_cast<size_t>(panel_rows) * desc.num_cols;
        const uint8_t on_disk_bytes = streampress::v3::dense_value_bytes(
            static_cast<streampress::v3::DenseValueType>(header_.value_type));

        out_data.resize(n_elems);

        if (on_disk_bytes == sizeof(Scalar)) {
            reader_->pread(panel_offset, out_data.data(), desc.byte_size);
        } else if (on_disk_bytes == 4 && sizeof(Scalar) == 8) {
            std::vector<float> buf(n_elems);
            reader_->pread(panel_offset, buf.data(), desc.byte_size);
            for (size_t i = 0; i < n_elems; ++i)
                out_data[i] = static_cast<Scalar>(buf[i]);
        } else if (on_disk_bytes == 8 && sizeof(Scalar) == 4) {
            std::vector<double> buf(n_elems);
            reader_->pread(panel_offset, buf.data(), desc.byte_size);
            for (size_t i = 0; i < n_elems; ++i)
                out_data[i] = static_cast<Scalar>(buf[i]);
        } else {
            throw std::runtime_error("Unsupported on-disk value size");
        }
    }

    static SpMat dense_to_sparse(const Scalar* data,
                                 uint32_t nrows, uint32_t ncols)
    {
        using Triplet = Eigen::Triplet<Scalar>;
        std::vector<Triplet> trips;
        trips.reserve(static_cast<size_t>(nrows) * ncols);

        for (uint32_t j = 0; j < ncols; ++j) {
            const Scalar* col = data + static_cast<size_t>(j) * nrows;
            for (uint32_t i = 0; i < nrows; ++i) {
                if (col[i] != Scalar(0)) {
                    trips.emplace_back(static_cast<int>(i),
                                       static_cast<int>(j),
                                       col[i]);
                }
            }
        }

        SpMat mat(nrows, ncols);
        mat.setFromTriplets(trips.begin(), trips.end());
        mat.makeCompressed();
        return mat;
    }

    // Core state
    std::unique_ptr<streampress::FileReader> reader_;
    bool in_core_;
    size_t file_size_;

    // In-core mode: full file buffer
    std::vector<uint8_t> file_data_;

    // Shared state
    streampress::v3::FileHeader_v3 header_;
    std::vector<streampress::v3::DenseChunkDescriptor> fwd_descs_;
    std::vector<streampress::v3::DenseChunkDescriptor> trans_descs_;

    uint32_t m_, n_;
    uint64_t nnz_;

    bool has_transpose_;

    uint32_t fwd_idx_;
    uint32_t trans_idx_;
};

}  // namespace io
}  // namespace FactorNet

