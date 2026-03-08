/**
 * @file io/loader.hpp
 * @brief DataLoader interface for chunked matrix access in NMF.
 *
 * The DataLoader abstraction unifies in-memory and streaming (SPZ) matrix
 * access behind a single interface. NMF fit loops iterate over column-panels
 * ("chunks") of the data matrix and its transpose without knowing the
 * underlying storage format.
 *
 * Two concrete implementations:
 *   - InMemoryLoader: Zero-copy views over Eigen sparse/dense matrices
 *   - SpzLoader:      Chunk-at-a-time decompression from .spz v2 files
 *
 * Data flow in NMF:
 *   1. H-update: iterate forward chunks (columns of A)
 *   2. W-update: iterate transpose chunks (columns of A^T = rows of A)
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>

namespace FactorNet {
namespace io {

/**
 * @brief A column-panel (chunk) of a sparse CSC matrix.
 *
 * Non-owning view when used with InMemoryLoader; owning storage when used
 * with SpzLoader (the Eigen::SparseMatrix holds the decompressed data).
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
struct Chunk {
    uint32_t col_start;   ///< First column index in the full matrix
    uint32_t num_cols;    ///< Number of columns in this chunk
    uint32_t num_rows;    ///< Number of rows (same as full matrix)
    uint64_t nnz;         ///< Number of nonzeros in this chunk

    /// The chunk data as an Eigen sparse matrix (num_rows x num_cols).
    /// For InMemoryLoader this is a lightweight view (Map);
    /// for SpzLoader this owns the decompressed data.
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int> matrix;
};

/**
 * @brief Abstract interface for chunked matrix access.
 *
 * Implementations provide forward iteration over column-panels of A
 * and column-panels of A^T, with reset for multi-iteration NMF.
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class DataLoader {
public:
    virtual ~DataLoader() = default;

    /// Number of rows in the full matrix A (m)
    virtual uint32_t rows() const = 0;

    /// Number of columns in the full matrix A (n)
    virtual uint32_t cols() const = 0;

    /// Total number of nonzeros in A
    virtual uint64_t nnz() const = 0;

    /// Number of forward chunks (column-panels of A)
    virtual uint32_t num_forward_chunks() const = 0;

    /// Number of transpose chunks (column-panels of A^T)
    virtual uint32_t num_transpose_chunks() const = 0;

    /**
     * @brief Get the next forward chunk (columns of A).
     *
     * Called sequentially from chunk 0 to num_forward_chunks()-1.
     * After all chunks are consumed, call reset_forward() before
     * iterating again.
     *
     * @param[out] out  Filled with the next chunk's data
     * @return true if a chunk was produced, false if exhausted
     */
    virtual bool next_forward(Chunk<Scalar>& out) = 0;

    /**
     * @brief Get the next transpose chunk (columns of A^T = rows of A).
     *
     * Called sequentially from chunk 0 to num_transpose_chunks()-1.
     * After all chunks are consumed, call reset_transpose() before
     * iterating again.
     *
     * @param[out] out  Filled with the next chunk's data
     * @return true if a chunk was produced, false if exhausted
     */
    virtual bool next_transpose(Chunk<Scalar>& out) = 0;

    /// Reset forward iteration to the first chunk
    virtual void reset_forward() = 0;

    /// Reset transpose iteration to the first chunk
    virtual void reset_transpose() = 0;
};

}  // namespace io
}  // namespace FactorNet

