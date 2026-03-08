/**
 * @file io/dense_spz_loader.hpp
 * @brief DenseSpzLoader — DataLoader for .spz v3 dense files.
 *
 * Reads an SPZ v3 (dense) file into memory and serves chunks via the
 * DataLoader interface. Dense panels are returned as Eigen::SparseMatrix
 * for compatibility with the existing chunked NMF pipeline.
 *
 * Memory: O(file_size + max_panel_size) — never holds the full matrix
 * in a separate buffer.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/io/loader.hpp>
#include <sparsepress/sparsepress_v3.hpp>
#include <sparsepress/format/header_v3.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace FactorNet {
namespace io {

/**
 * @brief DataLoader for .spz v3 (dense) files.
 *
 * Reads the file into a byte buffer at construction time, parses the
 * v3 header and chunk index, and returns dense panels wrapped as sparse
 * matrices for each next_*() call.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class DenseSpzLoader : public DataLoader<Scalar> {
public:
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
    using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * @brief Construct from an SPZ v3 file path.
     *
     * @param path              Path to the .spz v3 file
     * @param require_transpose If true (default), throw if file lacks transpose
     */
    explicit DenseSpzLoader(const std::string& path,
                            bool require_transpose = true)
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

        // Parse v3 header
        header_ = sparsepress::v3::read_header_v3(file_data_.data(), file_size_);

        m_ = header_.m;
        n_ = header_.n;
        nnz_ = static_cast<uint64_t>(m_) * n_;  // dense: all entries

        has_transpose_ = (header_.has_transpose != 0);
        if (require_transpose && !has_transpose_) {
            throw std::runtime_error(
                "Out-of-core NMF requires a .spz file with a pre-stored transpose.\n"
                "Re-write with: sp_write_dense(x, 'data.spz', include_transpose = TRUE)");
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

        std::vector<Scalar> panel;
        uint32_t col_start, num_cols;
        sparsepress::v3::read_forward_chunk<Scalar>(
            file_data_.data(), file_size_, header_,
            fwd_idx_, panel, col_start, num_cols);

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

        std::vector<Scalar> panel;
        uint32_t col_start, num_cols;
        sparsepress::v3::read_transpose_chunk<Scalar>(
            file_data_.data(), file_size_, header_,
            trans_idx_, panel, col_start, num_cols);

        out.col_start = col_start;
        out.num_cols  = num_cols;
        out.num_rows  = n_;   // transpose: n rows (original columns)
        out.nnz       = static_cast<uint64_t>(n_) * num_cols;
        out.matrix    = dense_to_sparse(panel.data(), n_, num_cols);

        ++trans_idx_;
        return true;
    }

    void reset_forward()   override { fwd_idx_ = 0; }
    void reset_transpose() override { trans_idx_ = 0; }

    /// Access to the parsed header
    const sparsepress::v3::FileHeader_v3& header() const { return header_; }

    /// Whether the file has a pre-stored transpose section
    bool has_transpose() const { return has_transpose_; }

    /// Direct access to raw file data
    const uint8_t* file_data() const { return file_data_.data(); }
    size_t file_size() const { return file_size_; }

private:
    /**
     * @brief Convert a column-major dense panel to an Eigen sparse matrix.
     *
     * Dense panels include all entries (zeros and nonzeros), so the sparse
     * matrix is "fully dense". This provides compatibility with the existing
     * chunked NMF pipeline that expects Eigen::SparseMatrix.
     */
    static SpMat dense_to_sparse(const Scalar* data,
                                 uint32_t nrows, uint32_t ncols)
    {
        // Count nonzeros per column (for triplet-free construction)
        // For a dense panel, every entry is stored.
        // Use setFromTriplets for simplicity — the NMF pipeline will handle it.
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

    std::vector<uint8_t> file_data_;
    size_t file_size_;
    sparsepress::v3::FileHeader_v3 header_;

    uint32_t m_, n_;
    uint64_t nnz_;

    bool has_transpose_;

    uint32_t fwd_idx_;
    uint32_t trans_idx_;
};

}  // namespace io
}  // namespace FactorNet

