// sparsepress - Deep compression for sparse CSC matrices
// Copyright (C) 2026 Zach DeBruine
// License: GPL (>= 3)

#ifndef SPARSEPRESS_TYPES_HPP
#define SPARSEPRESS_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cassert>

namespace streampress {

// ============================================================================
// CSC sparse matrix representation (matches R's dgCMatrix)
// ============================================================================
struct CSCMatrix {
    uint32_t m = 0;           // rows (genes)
    uint32_t n = 0;           // cols (cells)
    uint64_t nnz = 0;         // number of non-zeros

    std::vector<uint32_t> p;  // column pointers, size n+1
    std::vector<uint32_t> i;  // row indices, size nnz
    std::vector<double>   x;  // values, size nnz

    CSCMatrix() = default;

    CSCMatrix(uint32_t m_, uint32_t n_, uint64_t nnz_)
        : m(m_), n(n_), nnz(nnz_), p(n_ + 1), i(nnz_), x(nnz_) {}

    // Read from binary file (format from export_pbmc3k.R)
    static CSCMatrix from_binary(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open file: " + path);

        int32_t m32, n32, nnz32;
        if (fread(&m32, 4, 1, f) != 1 ||
            fread(&n32, 4, 1, f) != 1 ||
            fread(&nnz32, 4, 1, f) != 1) {
            fclose(f);
            throw std::runtime_error("Failed to read header from: " + path);
        }

        CSCMatrix mat(static_cast<uint32_t>(m32),
                      static_cast<uint32_t>(n32),
                      static_cast<uint64_t>(nnz32));

        // Read column pointers (n+1 int32s)
        std::vector<int32_t> p_tmp(mat.n + 1);
        if (fread(p_tmp.data(), 4, mat.n + 1, f) != mat.n + 1) {
            fclose(f);
            throw std::runtime_error("Failed to read col ptrs");
        }
        for (uint32_t j = 0; j <= mat.n; ++j)
            mat.p[j] = static_cast<uint32_t>(p_tmp[j]);

        // Read row indices (nnz int32s)
        std::vector<int32_t> i_tmp(mat.nnz);
        if (fread(i_tmp.data(), 4, mat.nnz, f) != mat.nnz) {
            fclose(f);
            throw std::runtime_error("Failed to read row indices");
        }
        for (uint64_t k = 0; k < mat.nnz; ++k)
            mat.i[k] = static_cast<uint32_t>(i_tmp[k]);

        // Read values (nnz doubles)
        if (fread(mat.x.data(), 8, mat.nnz, f) != mat.nnz) {
            fclose(f);
            throw std::runtime_error("Failed to read values");
        }

        fclose(f);
        return mat;
    }

    // Write to binary file (same format)
    void to_binary(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("Cannot open file for writing: " + path);

        int32_t m32 = static_cast<int32_t>(m);
        int32_t n32 = static_cast<int32_t>(n);
        int32_t nnz32 = static_cast<int32_t>(nnz);
        fwrite(&m32, 4, 1, f);
        fwrite(&n32, 4, 1, f);
        fwrite(&nnz32, 4, 1, f);

        std::vector<int32_t> p_tmp(n + 1);
        for (uint32_t j = 0; j <= n; ++j)
            p_tmp[j] = static_cast<int32_t>(p[j]);
        fwrite(p_tmp.data(), 4, n + 1, f);

        std::vector<int32_t> i_tmp(nnz);
        for (uint64_t k = 0; k < nnz; ++k)
            i_tmp[k] = static_cast<int32_t>(i[k]);
        fwrite(i_tmp.data(), 4, nnz, f);

        fwrite(x.data(), 8, nnz, f);
        fclose(f);
    }

    // Raw size in bytes
    size_t raw_size() const {
        return 12 + (n + 1) * 4 + nnz * 4 + nnz * 8;
    }

    // Verify two matrices are identical
    bool equals(const CSCMatrix& other) const {
        if (m != other.m || n != other.n || nnz != other.nnz) return false;
        if (p != other.p) return false;
        if (i != other.i) return false;
        if (x.size() != other.x.size()) return false;
        return std::memcmp(x.data(), other.x.data(), nnz * sizeof(double)) == 0;
    }
};

// ============================================================================
// Typed CSC sparse matrix — avoids float→double conversion for fp32 workflows
// ============================================================================
template<typename Scalar>
struct CSCMatrixTyped {
    uint32_t m = 0;
    uint32_t n = 0;
    uint64_t nnz = 0;

    std::vector<uint32_t>  p;  // column pointers, size n+1
    std::vector<uint32_t>  i;  // row indices, size nnz
    std::vector<Scalar>    x;  // values, size nnz

    CSCMatrixTyped() = default;

    CSCMatrixTyped(uint32_t m_, uint32_t n_, uint64_t nnz_)
        : m(m_), n(n_), nnz(nnz_), p(n_ + 1), i(nnz_), x(nnz_) {}

    // Construct from double CSCMatrix with optional cast
    explicit CSCMatrixTyped(const CSCMatrix& src)
        : m(src.m), n(src.n), nnz(src.nnz), p(src.p), i(src.i), x(src.nnz) {
        for (uint64_t k = 0; k < nnz; ++k)
            x[k] = static_cast<Scalar>(src.x[k]);
    }
};

// ============================================================================
// Compression configuration
// ============================================================================
struct CompressConfig {
    uint32_t rice_block_size = 1024;     // Entries per adaptive Rice block
    uint32_t density_blocks = 128;      // Number of row-density blocks for predictor
    uint64_t prng_seed = 0x5A434F4D50ULL; // Default PRNG seed ("ZCOMP")
    bool use_value_prediction = false;   // Use independence-model value prediction
    bool use_delta_prediction = false;   // Use density-based delta prediction
    bool verbose = false;
};

// ============================================================================
// Compression statistics
// ============================================================================
struct CompressStats {
    size_t raw_size = 0;
    size_t compressed_size = 0;
    size_t header_size = 0;
    size_t model_size = 0;
    size_t structure_size = 0;
    size_t values_size = 0;
    double compress_time_ms = 0;
    double decompress_time_ms = 0;

    double ratio() const {
        return compressed_size > 0
            ? static_cast<double>(raw_size) / compressed_size
            : 0.0;
    }

    double bits_per_nnz(uint64_t nnz) const {
        return nnz > 0
            ? static_cast<double>(compressed_size) * 8.0 / nnz
            : 0.0;
    }
};

} // namespace streampress

#endif // SPARSEPRESS_TYPES_HPP
