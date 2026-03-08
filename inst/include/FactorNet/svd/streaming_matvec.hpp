// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file streaming_matvec.hpp
 * @brief Reusable streaming SpMV/SpMM primitives for out-of-core SVD from .spz v2 files
 *
 * Central abstraction: StreamingContext holds the file data and chunk metadata.
 * All algorithms access A only through:
 *   - streaming_spmm_forward():   result = A * X   (m × k)
 *   - streaming_spmm_transpose(): result = A' * X  (n × k)
 *   - streaming_spmv_forward():   result = A * v   (m × 1)
 *   - streaming_spmv_transpose(): result = A' * u  (n × 1)
 *
 * This layer is algorithm-agnostic — deflation, krylov, lanczos, irlba, and
 * randomized SVD all use these same primitives.
 *
 * PCA centering is handled implicitly via optional row_means correction.
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <sparsepress/sparsepress_v2.hpp>
#include <sparsepress/sparsepress_v3.hpp>
#include <sparsepress/format/header_v2.hpp>
#include <sparsepress/format/header_v3.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cstdio>
#include <memory>
#include <cstring>

namespace FactorNet {
namespace svd {
namespace detail {

// ============================================================================
// Zero-copy CSC → Eigen conversion
// ============================================================================

template<typename Scalar>
inline Eigen::SparseMatrix<Scalar, Eigen::ColMajor>
csc_typed_to_eigen(const sparsepress::CSCMatrixTyped<Scalar>& csc,
                   uint32_t nrows, uint32_t ncols)
{
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    SpMat mat(nrows, ncols);
    mat.resizeNonZeros(csc.nnz);

    auto* outerPtr = mat.outerIndexPtr();
    auto* innerPtr = mat.innerIndexPtr();
    auto* valuePtr = mat.valuePtr();

    for (uint32_t j = 0; j <= ncols; ++j)
        outerPtr[j] = static_cast<typename SpMat::StorageIndex>(csc.p[j]);
    for (uint64_t idx = 0; idx < csc.nnz; ++idx) {
        innerPtr[idx] = static_cast<typename SpMat::StorageIndex>(csc.i[idx]);
        valuePtr[idx] = csc.x[idx];
    }
    return mat;
}

// ============================================================================
// StreamingContext — holds file data + parsed metadata
// ============================================================================

template<typename Scalar>
struct StreamingContext {
    std::vector<uint8_t> file_data;
    size_t file_size = 0;
    uint32_t m = 0;      ///< rows
    uint32_t n = 0;      ///< cols
    uint32_t num_chunks = 0;
    bool is_v3 = false;   ///< true for v3 dense format

    // v2-only
    std::vector<sparsepress::v2::ChunkDescriptor_v2> chunks;

    // v3-only
    sparsepress::v3::FileHeader_v3 v3_header;

    /// Row means (populated if center=true)
    DenseVector<Scalar> row_means;
    const DenseVector<Scalar>* row_means_ptr = nullptr;

    /// Row inverse SDs (populated if scale=true)
    DenseVector<Scalar> row_inv_sds;
    const DenseVector<Scalar>* row_inv_sds_ptr = nullptr;

    /// Row SDs (stored for result output)
    DenseVector<Scalar> row_sds;

    /// ||A||²_F (or ||A_c||²_F for centered, or m*n for scaled)
    Scalar frobenius_norm_sq = 0;

    /// Cached decompressed Eigen sparse panels — avoids re-decompressing
    /// from file_data on every SpMM call. For 10 chunks × 200 Lanczos steps
    /// × 2 SpMV/step, this eliminates ~4000 decompress calls.
    mutable bool cache_initialized = false;
    mutable std::vector<Eigen::SparseMatrix<Scalar, Eigen::ColMajor>> cached_panels;

    /// Cached dense panels for v3 format
    mutable std::vector<DenseMatrix<Scalar>> cached_dense_panels;
    /// Column start offset per chunk (used for both v2 and v3)
    mutable std::vector<uint32_t> chunk_col_starts;
    mutable std::vector<uint32_t> chunk_num_cols;

    /**
     * @brief Pre-decompress all chunks and cache as Eigen sparse matrices.
     * Called lazily on first SpMM/SpMV access. Thread-safe via flag check.
     */
    void ensure_cache() const {
        if (cache_initialized) return;
        chunk_col_starts.resize(num_chunks);
        chunk_num_cols.resize(num_chunks);

        if (is_v3) {
            cached_dense_panels.resize(num_chunks);
            for (uint32_t c = 0; c < num_chunks; ++c) {
                std::vector<Scalar> panel;
                uint32_t col_start, nc;
                sparsepress::v3::read_forward_chunk<Scalar>(
                    file_data.data(), file_size, v3_header,
                    c, panel, col_start, nc);
                // Map to Eigen dense matrix (column-major: m rows × nc cols)
                cached_dense_panels[c] = Eigen::Map<const DenseMatrix<Scalar>>(
                    panel.data(), m, nc);
                chunk_col_starts[c] = col_start;
                chunk_num_cols[c] = nc;
            }
        } else {
            cached_panels.resize(num_chunks);
            for (uint32_t c = 0; c < num_chunks; ++c) {
                uint32_t nc = chunks[c].num_cols;
                sparsepress::v2::DecompressConfig_v2 dcfg;
                dcfg.col_start = static_cast<int>(chunks[c].col_start);
                dcfg.col_end = static_cast<int>(chunks[c].col_start + nc);
                auto csc = sparsepress::v2::decompress_v2_typed<Scalar>(
                    file_data.data(), file_data.size(), dcfg);
                cached_panels[c] = csc_typed_to_eigen<Scalar>(csc, m, nc);
                chunk_col_starts[c] = chunks[c].col_start;
                chunk_num_cols[c] = nc;
            }
        }
        cache_initialized = true;
    }

    /**
     * @brief Get chunk panel — returns cached sparse version (v2 only)
     */
    const Eigen::SparseMatrix<Scalar, Eigen::ColMajor>& get_panel(uint32_t c) const {
        ensure_cache();
        return cached_panels[c];
    }

    /**
     * @brief Get chunk as dense panel (v3 only)
     */
    const DenseMatrix<Scalar>& get_dense_panel(uint32_t c) const {
        ensure_cache();
        return cached_dense_panels[c];
    }

    /**
     * @brief Load a .spz v2 file into memory and parse header
     */
    void load(const std::string& spz_path, bool center = false, bool scale = false) {
        FILE* f = fopen(spz_path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open .spz file: " + spz_path);
        fseek(f, 0, SEEK_END);
        file_size = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);
        file_data.resize(file_size);
        if (fread(file_data.data(), 1, file_size, f) != file_size) {
            fclose(f);
            throw std::runtime_error("Failed to read .spz file");
        }
        fclose(f);

        // Detect format version
        uint16_t version = sparsepress::v3::detect_version(file_data.data(), file_size);

        if (version == 3) {
            is_v3 = true;
            v3_header = sparsepress::v3::read_header_v3(file_data.data(), file_size);
            m = v3_header.m;
            n = v3_header.n;
            num_chunks = v3_header.num_chunks;
        } else {
            is_v3 = false;
            auto header = sparsepress::v2::FileHeader_v2::deserialize(file_data.data());
            m = header.m;
            n = header.n;
            num_chunks = header.num_chunks;

            chunks.resize(num_chunks);
            const uint8_t* chunk_idx_ptr = file_data.data() + header.chunk_index_offset;
            for (uint32_t c = 0; c < num_chunks; ++c) {
                std::memcpy(&chunks[c],
                            chunk_idx_ptr + c * sizeof(sparsepress::v2::ChunkDescriptor_v2),
                            sizeof(sparsepress::v2::ChunkDescriptor_v2));
            }
        }

        // Compute row means if centering
        if (center) {
            row_means = compute_row_means();
            row_means_ptr = &row_means;
        }

        // Compute row SDs if scaling (requires row_means)
        if (scale) {
            row_sds = compute_row_sds_streaming();
            row_inv_sds = row_sds.cwiseInverse();
            row_inv_sds_ptr = &row_inv_sds;
        }

        // Compute ||A||²_F
        frobenius_norm_sq = compute_frobenius_sq();
        if (center) {
            frobenius_norm_sq -= static_cast<Scalar>(n) * row_means.squaredNorm();
        }
        if (scale) {
            frobenius_norm_sq = static_cast<Scalar>(m) * static_cast<Scalar>(n);
        }
    }

    /**
     * @brief Compute row means by streaming through all chunks
     */
    DenseVector<Scalar> compute_row_means() const {
        DenseVector<Scalar> sums = DenseVector<Scalar>::Zero(m);
        if (is_v3) {
            ensure_cache();
            for (uint32_t c = 0; c < num_chunks; ++c) {
                sums += cached_dense_panels[c].rowwise().sum();
            }
        } else {
            for (uint32_t c = 0; c < num_chunks; ++c) {
                sparsepress::v2::DecompressConfig_v2 dcfg;
                dcfg.col_start = static_cast<int>(chunks[c].col_start);
                dcfg.col_end = static_cast<int>(chunks[c].col_start + chunks[c].num_cols);
                auto csc = sparsepress::v2::decompress_v2_typed<Scalar>(
                    file_data.data(), file_data.size(), dcfg);
                for (uint64_t idx = 0; idx < csc.nnz; ++idx) {
                    sums(csc.i[idx]) += csc.x[idx];
                }
            }
        }
        return sums / static_cast<Scalar>(n);
    }

    /**
     * @brief Compute ||A||²_F by streaming
     */
    Scalar compute_frobenius_sq() const {
        Scalar sum_sq = 0;
        if (is_v3) {
            ensure_cache();
            for (uint32_t c = 0; c < num_chunks; ++c) {
                sum_sq += cached_dense_panels[c].squaredNorm();
            }
        } else {
            for (uint32_t c = 0; c < num_chunks; ++c) {
                sparsepress::v2::DecompressConfig_v2 dcfg;
                dcfg.col_start = static_cast<int>(chunks[c].col_start);
                dcfg.col_end = static_cast<int>(chunks[c].col_start + chunks[c].num_cols);
                auto csc = sparsepress::v2::decompress_v2_typed<Scalar>(
                    file_data.data(), file_data.size(), dcfg);
                for (uint64_t idx = 0; idx < csc.nnz; ++idx) {
                    sum_sq += csc.x[idx] * csc.x[idx];
                }
            }
        }
        return sum_sq;
    }

    /**
     * @brief Compute row standard deviations by streaming (requires row_means)
     *
     * Uses Var(x_i) = E[x_i²] - μ_i², streaming through all chunks to
     * accumulate E[x_i²] per row.
     */
    DenseVector<Scalar> compute_row_sds_streaming() const {
        DenseVector<Scalar> sum_sq = DenseVector<Scalar>::Zero(m);
        if (is_v3) {
            ensure_cache();
            for (uint32_t c = 0; c < num_chunks; ++c) {
                sum_sq += cached_dense_panels[c].cwiseProduct(cached_dense_panels[c]).rowwise().sum();
            }
        } else {
            for (uint32_t c = 0; c < num_chunks; ++c) {
                sparsepress::v2::DecompressConfig_v2 dcfg;
                dcfg.col_start = static_cast<int>(chunks[c].col_start);
                dcfg.col_end = static_cast<int>(chunks[c].col_start + chunks[c].num_cols);
                auto csc = sparsepress::v2::decompress_v2_typed<Scalar>(
                    file_data.data(), file_data.size(), dcfg);
                for (uint64_t idx = 0; idx < csc.nnz; ++idx) {
                    sum_sq(csc.i[idx]) += csc.x[idx] * csc.x[idx];
                }
            }
        }
        DenseVector<Scalar> variance = sum_sq / static_cast<Scalar>(n) -
                                       row_means.cwiseProduct(row_means);
        // Clamp away from zero to avoid division by zero
        const Scalar eps = std::numeric_limits<Scalar>::epsilon() * 100;
        return variance.cwiseMax(eps).cwiseSqrt();
    }
};

// ============================================================================
// Streaming SpMM: result = A * X  (m × k)
// ============================================================================

template<typename Scalar>
void spmm_forward(const StreamingContext<Scalar>& ctx,
                  const DenseMatrix<Scalar>& X,      // n × k
                  DenseMatrix<Scalar>& result)        // m × k (output)
{
    const int k = static_cast<int>(X.cols());
    result.setZero(ctx.m, k);

    ctx.ensure_cache();
    uint32_t col_offset = 0;
    for (uint32_t c = 0; c < ctx.num_chunks; ++c) {
        uint32_t nc = ctx.chunk_num_cols[c];
        auto X_panel = X.middleRows(col_offset, nc);
        if (ctx.is_v3) {
            const auto& D_panel = ctx.get_dense_panel(c);
            result.noalias() += D_panel * X_panel;
        } else {
            const auto& A_panel = ctx.get_panel(c);
            result.noalias() += A_panel * X_panel;
        }
        col_offset += nc;
    }

    // PCA centering correction
    if (ctx.row_means_ptr && ctx.row_means_ptr->size() > 0) {
        DenseVector<Scalar> col_sums = X.colwise().sum();
        result -= (*ctx.row_means_ptr) * col_sums.transpose();
    }
    // Correlation PCA scaling: multiply each row by 1/sigma_i
    if (ctx.row_inv_sds_ptr && ctx.row_inv_sds_ptr->size() > 0) {
        result.array().colwise() *= ctx.row_inv_sds_ptr->array();
    }
}
// ============================================================================

template<typename Scalar>
void spmm_transpose(const StreamingContext<Scalar>& ctx,
                    const DenseMatrix<Scalar>& X,      // m × k
                    DenseMatrix<Scalar>& result)        // n × k (output)
{
    const int k = static_cast<int>(X.cols());
    result.resize(ctx.n, k);

    // PCA centering correction: (A - µ1')'X = A'X - 1µ'X
    // For scaling: Ã = D^{-1}(A-µ1'), Ã'X = (A-µ1')'D^{-1}X = A'(D^{-1}X) - 1µ'(D^{-1}X)
    const DenseMatrix<Scalar>* X_eff = &X;
    DenseMatrix<Scalar> X_scaled_storage;
    if (ctx.row_inv_sds_ptr && ctx.row_inv_sds_ptr->size() > 0) {
        X_scaled_storage = X.array().colwise() * ctx.row_inv_sds_ptr->array();
        X_eff = &X_scaled_storage;
    }

    uint32_t col_offset = 0;
    ctx.ensure_cache();
    for (uint32_t c = 0; c < ctx.num_chunks; ++c) {
        uint32_t nc = ctx.chunk_num_cols[c];
        if (ctx.is_v3) {
            const auto& D_panel = ctx.get_dense_panel(c);
            result.middleRows(col_offset, nc).noalias() = D_panel.transpose() * (*X_eff);
        } else {
            const auto& A_panel = ctx.get_panel(c);
            result.middleRows(col_offset, nc).noalias() = A_panel.transpose() * (*X_eff);
        }
        col_offset += nc;
    }

    if (ctx.row_means_ptr && ctx.row_means_ptr->size() > 0) {
        DenseVector<Scalar> mu_X = X_eff->transpose() * (*ctx.row_means_ptr);  // k × 1
        result.rowwise() -= mu_X.transpose();
    }
}

// ============================================================================
// Streaming SpMV (single vector): y = A * v  (m × 1)
// ============================================================================

template<typename Scalar>
void spmv_forward(const StreamingContext<Scalar>& ctx,
                  const DenseVector<Scalar>& v,       // n × 1
                  DenseVector<Scalar>& result)         // m × 1 (output)
{
    DenseMatrix<Scalar> V_mat = v;   // n × 1 as matrix
    DenseMatrix<Scalar> R_mat;
    spmm_forward(ctx, V_mat, R_mat);
    result = R_mat.col(0);
}

// ============================================================================
// Streaming SpMV transpose: y = A' * u  (n × 1)
// ============================================================================

template<typename Scalar>
void spmv_transpose(const StreamingContext<Scalar>& ctx,
                    const DenseVector<Scalar>& u,       // m × 1
                    DenseVector<Scalar>& result)         // n × 1 (output)
{
    DenseMatrix<Scalar> U_mat = u;   // m × 1 as matrix
    DenseMatrix<Scalar> R_mat;
    spmm_transpose(ctx, U_mat, R_mat);
    result = R_mat.col(0);
}

}  // namespace streaming
}  // namespace svd
}  // namespace FactorNet

