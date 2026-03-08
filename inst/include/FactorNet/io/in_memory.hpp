/**
 * @file io/in_memory.hpp
 * @brief InMemoryLoader — zero-copy DataLoader over Eigen sparse matrices.
 *
 * Wraps an existing Eigen::SparseMatrix<Scalar> (or Map) and emits
 * fixed-size column-panels. No data is copied — each chunk references
 * a contiguous block of the original matrix's CSC arrays.
 *
 * For the transpose, a pre-computed A^T is stored (computed once at
 * construction). This is the standard path for in-memory NMF.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/io/loader.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>

namespace FactorNet {
namespace io {

/**
 * @brief DataLoader over an in-memory Eigen sparse matrix.
 *
 * Chunks are contiguous column-panels of size `chunk_cols` (default 256,
 * matching SPZ chunk size). The last chunk may be smaller.
 *
 * The forward matrix is borrowed (non-owning pointer). The transpose
 * is computed and owned by this loader.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class InMemoryLoader : public DataLoader<Scalar> {
public:
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    /**
     * @brief Construct from an Eigen sparse matrix.
     *
     * @param A          The data matrix (m x n). Must outlive this loader.
     * @param chunk_cols Number of columns per chunk (default 256)
     */
    InMemoryLoader(const SpMat& A, uint32_t chunk_cols = 256)
        : A_(A),
          A_T_(A.transpose()),
          chunk_cols_(chunk_cols),
          fwd_idx_(0),
          trans_idx_(0)
    {
        m_ = static_cast<uint32_t>(A.rows());
        n_ = static_cast<uint32_t>(A.cols());
        nnz_ = static_cast<uint64_t>(A.nonZeros());

        // Precompute chunk boundaries for forward
        for (uint32_t c = 0; c < n_; c += chunk_cols_) {
            fwd_starts_.push_back(c);
        }
        // Precompute chunk boundaries for transpose
        for (uint32_t c = 0; c < m_; c += chunk_cols_) {
            trans_starts_.push_back(c);
        }
    }

    uint32_t rows() const override { return m_; }
    uint32_t cols() const override { return n_; }
    uint64_t nnz()  const override { return nnz_; }

    uint32_t num_forward_chunks()   const override {
        return static_cast<uint32_t>(fwd_starts_.size());
    }
    uint32_t num_transpose_chunks() const override {
        return static_cast<uint32_t>(trans_starts_.size());
    }

    bool next_forward(Chunk<Scalar>& out) override {
        if (fwd_idx_ >= fwd_starts_.size()) return false;

        const uint32_t col_start = fwd_starts_[fwd_idx_];
        const uint32_t nc = std::min(chunk_cols_, n_ - col_start);

        out.col_start = col_start;
        out.num_cols  = nc;
        out.num_rows  = m_;

        // Extract the column-panel as a block of the sparse matrix
        out.matrix = A_.middleCols(col_start, nc);
        out.nnz = static_cast<uint64_t>(out.matrix.nonZeros());

        ++fwd_idx_;
        return true;
    }

    bool next_transpose(Chunk<Scalar>& out) override {
        if (trans_idx_ >= trans_starts_.size()) return false;

        const uint32_t col_start = trans_starts_[trans_idx_];
        const uint32_t nc = std::min(chunk_cols_, m_ - col_start);

        out.col_start = col_start;
        out.num_cols  = nc;
        out.num_rows  = n_;  // transpose: n rows

        out.matrix = A_T_.middleCols(col_start, nc);
        out.nnz = static_cast<uint64_t>(out.matrix.nonZeros());

        ++trans_idx_;
        return true;
    }

    void reset_forward()   override { fwd_idx_ = 0; }
    void reset_transpose() override { trans_idx_ = 0; }

    /// Direct access to the full forward matrix (for init/loss that needs it)
    const SpMat& forward_matrix()   const { return A_; }

    /// Direct access to the full transpose matrix
    const SpMat& transpose_matrix() const { return A_T_; }

private:
    const SpMat& A_;        ///< Borrowed reference to original data
    SpMat A_T_;             ///< Owned transpose
    uint32_t m_, n_;
    uint64_t nnz_;
    uint32_t chunk_cols_;

    std::vector<uint32_t> fwd_starts_;
    std::vector<uint32_t> trans_starts_;
    size_t fwd_idx_;
    size_t trans_idx_;
};

}  // namespace io
}  // namespace FactorNet

