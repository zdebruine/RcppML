/**
 * @file io/spz_loader.hpp
 * @brief SpzLoader — DataLoader for .spz v2 files with chunk-at-a-time decompression.
 *
 * Reads the entire .spz file into memory once, then decompresses column-panels
 * on demand. Each call to next_forward() / next_transpose() decompresses
 * one chunk into an Eigen::SparseMatrix. Previous chunks are discarded.
 *
 * Memory: O(file_size + max_chunk_nnz) — never holds the full decompressed matrix.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/io/loader.hpp>
#include <sparsepress/sparsepress_v2.hpp>
#include <sparsepress/format/header_v2.hpp>

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
 * Reads the file into a byte buffer at construction time, parses the
 * v2 header and chunk index, and decompresses one chunk per next_*() call.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class SpzLoader : public DataLoader<Scalar> {
public:
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    /**
     * @brief Construct from a .spz v2 file path.
     *
     * @param path             Path to the .spz v2 file
     * @param require_transpose If true (default), throw if file lacks transpose section
     */
    explicit SpzLoader(const std::string& path, bool require_transpose = true)
        : fwd_idx_(0), trans_idx_(0)
    {
        // Read entire file into memory
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open file: " + path);
        fseek(f, 0, SEEK_END);
        file_size_ = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);

        file_data_.resize(file_size_);
        if (fread(file_data_.data(), 1, file_size_, f) != file_size_) {
            fclose(f);
            throw std::runtime_error("Failed to read .spz file: " + path);
        }
        fclose(f);

        // Parse v2 header
        if (file_size_ < sparsepress::v2::HEADER_SIZE_V2) {
            throw std::runtime_error("File too small for v2 header: " + path);
        }
        header_ = sparsepress::v2::FileHeader_v2::deserialize(file_data_.data());
        if (header_.version != 2) {
            throw std::runtime_error("Not a v2 .spz file: " + path);
        }

        m_   = header_.m;
        n_   = header_.n;
        nnz_ = header_.nnz;

        // Parse forward chunk descriptors
        fwd_descs_.resize(header_.num_chunks);
        const uint8_t* chunk_idx = file_data_.data() + header_.chunk_index_offset;
        for (uint32_t c = 0; c < header_.num_chunks; ++c) {
            std::memcpy(&fwd_descs_[c],
                        chunk_idx + c * sizeof(sparsepress::v2::ChunkDescriptor_v2),
                        sizeof(sparsepress::v2::ChunkDescriptor_v2));
        }

        // Check and build transpose reader
        has_transpose_ = sparsepress::v2::has_transpose(
            file_data_.data(), file_size_);
        if (require_transpose && !has_transpose_) {
            throw std::runtime_error(
                "Out-of-core NMF requires a .spz file with a pre-stored transpose.\n"
                "Re-compress with: sp_write(x, 'data.spz', include_transpose = TRUE)");
        }
        if (has_transpose_) {
            t_reader_ = std::make_unique<sparsepress::v2::TransposeChunkReader>(
                file_data_.data(), file_size_);
        }
    }

    uint32_t rows() const override { return m_; }
    uint32_t cols() const override { return n_; }
    uint64_t nnz()  const override { return nnz_; }

    uint32_t num_forward_chunks() const override {
        return header_.num_chunks;
    }

    uint32_t num_transpose_chunks() const override {
        return has_transpose_
            ? t_reader_->num_chunks()
            : 0;
    }

    bool next_forward(Chunk<Scalar>& out) override {
        if (fwd_idx_ >= header_.num_chunks) return false;

        const auto& desc = fwd_descs_[fwd_idx_];
        const uint32_t col_start = desc.col_start;
        const uint32_t nc = desc.num_cols;

        // Decompress with native Scalar type
        sparsepress::v2::DecompressConfig_v2 dcfg;
        dcfg.col_start = col_start;
        dcfg.col_end = col_start + nc;
        auto csc = sparsepress::v2::decompress_v2_typed<Scalar>(
            file_data_.data(), file_size_, dcfg);

        // Build Eigen sparse matrix from CSCMatrixTyped
        out.col_start = col_start;
        out.num_cols  = nc;
        out.num_rows  = m_;
        out.nnz       = csc.nnz;
        out.matrix    = csc_typed_to_eigen(csc, m_, nc);

        ++fwd_idx_;
        return true;
    }

    bool next_transpose(Chunk<Scalar>& out) override {
        if (!has_transpose_) return false;
        if (trans_idx_ >= t_reader_->num_chunks()) return false;

        const uint32_t tc = trans_idx_;
        sparsepress::CSCMatrix csc = t_reader_->decompress_chunk(tc);
        const uint32_t col_start = t_reader_->chunk_col_start(tc);
        const uint32_t nc = t_reader_->chunk_num_cols(tc);

        out.col_start = col_start;
        out.num_cols  = nc;
        out.num_rows  = n_;   // transpose: n rows (original columns)
        out.nnz       = csc.nnz;
        out.matrix    = csc_to_eigen(csc, n_, nc);

        ++trans_idx_;
        return true;
    }

    void reset_forward()   override { fwd_idx_ = 0; }
    void reset_transpose() override { trans_idx_ = 0; }

    /// Access to raw file data (for init routines that need full decompression)
    const uint8_t* file_data() const { return file_data_.data(); }
    size_t         file_size() const { return file_size_; }

    /// Access to the parsed header
    const sparsepress::v2::FileHeader_v2& header() const { return header_; }

    /// Access to forward chunk descriptors
    const std::vector<sparsepress::v2::ChunkDescriptor_v2>& forward_descs() const {
        return fwd_descs_;
    }

    /// Whether the file has a pre-stored transpose section
    bool has_transpose() const { return has_transpose_; }

private:
    /// Convert CSCMatrixTyped<Scalar> to Eigen sparse (forward chunks)
    static SpMat csc_typed_to_eigen(
        const sparsepress::CSCMatrixTyped<Scalar>& csc,
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
        const sparsepress::CSCMatrix& csc,
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

    std::vector<uint8_t> file_data_;
    size_t file_size_;
    sparsepress::v2::FileHeader_v2 header_;
    std::vector<sparsepress::v2::ChunkDescriptor_v2> fwd_descs_;

    uint32_t m_, n_;
    uint64_t nnz_;

    bool has_transpose_;
    std::unique_ptr<sparsepress::v2::TransposeChunkReader> t_reader_;

    uint32_t fwd_idx_;
    uint32_t trans_idx_;
};

}  // namespace io
}  // namespace FactorNet

