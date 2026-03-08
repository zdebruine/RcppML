/**
 * @file fit_streaming_spz.hpp
 * @brief Out-of-core streaming NMF from .spz v2 (sparse) and v3 (dense) files.
 *
 * Thin entry point that creates the appropriate DataLoader and delegates to
 * nmf_chunked(). SVD-based initialization is handled here (requires full
 * matrix decompression). Random initialization and all NMF iteration are
 * handled by nmf_chunked().
 *
 * Memory: O(m*k + n*k + max_chunk_nnz) — never holds full A (except during
 * SVD init, which temporarily decompresses the full matrix).
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/logging.hpp>
#include <FactorNet/core/resources.hpp>
#include <FactorNet/io/spz_loader.hpp>
#include <FactorNet/io/dense_spz_loader.hpp>
#include <FactorNet/nmf/fit_chunked.hpp>
#include <sparsepress/sparsepress_v3.hpp>

#ifdef FACTORNET_HAS_GPU
#include <FactorNet/nmf/fit_chunked_gpu.cuh>
#endif

// SVD initialization (needs fit_cpu.hpp for initialize_lanczos/irlba)
#include <FactorNet/nmf/fit_cpu.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace FactorNet {
namespace nmf {

/**
 * @brief Run NMF directly from a .spz file (v2 sparse or v3 dense),
 *        processing chunk-by-chunk.
 *
 * Detects file version, creates the appropriate DataLoader, and delegates
 * iteration to nmf_chunked(). SVD-based initialization (init_mode 1 or 2)
 * temporarily decompresses the full matrix for the SVD computation.
 *
 * @tparam Scalar float or double
 * @param path    Path to .spz file (v2 or v3)
 * @param config  Unified NMF configuration
 * @return NMFResult with W, d, H factors
 */
template<typename Scalar = float>
FactorNet::NMFResult<Scalar> nmf_streaming_spz(
    const std::string& path,
    const FactorNet::NMFConfig<Scalar>& config)
{
    using MatS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    const int k = config.rank;
    const int verbose = config.verbose;

    // ---- 0. Detect file version ----
    // Read first 8 bytes to determine v2 vs v3
    FILE* vf = fopen(path.c_str(), "rb");
    if (!vf) throw std::runtime_error("Cannot open file: " + path);
    uint8_t ver_buf[8];
    if (fread(ver_buf, 1, 8, vf) != 8) {
        fclose(vf);
        throw std::runtime_error("File too small: " + path);
    }
    fclose(vf);
    const uint16_t file_version = sparsepress::v3::detect_version(ver_buf, 8);

    // ---- 1. Create appropriate DataLoader ----
    std::unique_ptr<FactorNet::io::DataLoader<Scalar>> loader_ptr;

    if (file_version == 3) {
        loader_ptr = std::make_unique<FactorNet::io::DenseSpzLoader<Scalar>>(
            path, /*require_transpose=*/true);
        FACTORNET_LOG_IO(FactorNet::LogLevel::SUMMARY, verbose,
            "Streaming NMF from dense .spz v3: %s\n", path.c_str());
    } else if (file_version == 2 || file_version == 1) {
        loader_ptr = std::make_unique<FactorNet::io::SpzLoader<Scalar>>(
            path, /*require_transpose=*/true);
        FACTORNET_LOG_IO(FactorNet::LogLevel::SUMMARY, verbose,
            "Streaming NMF from sparse .spz v%d: %s\n", file_version, path.c_str());
    } else {
        throw std::runtime_error("Unsupported .spz version: " +
                                 std::to_string(file_version));
    }

    auto& loader = *loader_ptr;
    const uint32_t m = loader.rows();
    const uint32_t n = loader.cols();
    const uint64_t total_nnz = loader.nnz();

    double density_pct = (m > 0 && n > 0)
        ? static_cast<double>(total_nnz) / (static_cast<double>(m) * n) * 100.0
        : 0.0;
    FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
        "Data: %u x %u, nnz=%llu (%.1f%% dense), %u fwd chunks, %u t chunks\n",
        m, n, (unsigned long long)total_nnz,
        density_pct,
        loader.num_forward_chunks(), loader.num_transpose_chunks());

    // ---- 2. SVD initialization (requires temporary full matrix) ----
    // W_init (m×k) and H_init (k×n) used by nmf_chunked / nmf_chunked_gpu.
    // nullptr means use random init.
    std::unique_ptr<MatS> W_init_ptr;
    std::unique_ptr<MatS> H_init_ptr;

    if (config.init_mode == 1 || config.init_mode == 2) {
        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
            "Building temporary matrix for %s init...\n",
            config.init_mode == 1 ? "Lanczos" : "IRLBA");

        // Decompress all forward chunks into a full sparse matrix
        std::vector<int> full_colptr(n + 1, 0);
        std::vector<int> full_rowIdx;
        std::vector<Scalar> full_values;
        full_rowIdx.reserve(total_nnz);
        full_values.reserve(total_nnz);

        FactorNet::io::Chunk<Scalar> chunk;
        loader.reset_forward();
        while (loader.next_forward(chunk)) {
            const SpMat& panel = chunk.matrix;
            for (uint32_t j_local = 0; j_local < chunk.num_cols; ++j_local) {
                uint32_t j_global = chunk.col_start + j_local;
                int col_nnz = panel.outerIndexPtr()[j_local + 1] - panel.outerIndexPtr()[j_local];
                full_colptr[j_global + 1] = full_colptr[j_global] + col_nnz;

                for (typename SpMat::InnerIterator it(panel, j_local); it; ++it) {
                    full_rowIdx.push_back(it.row());
                    full_values.push_back(it.value());
                }
            }
        }

        SpMat A_full(m, n);
        A_full.resizeNonZeros(full_rowIdx.size());
        std::copy(full_colptr.begin(), full_colptr.end(), A_full.outerIndexPtr());
        std::copy(full_rowIdx.begin(), full_rowIdx.end(), A_full.innerIndexPtr());
        std::copy(full_values.begin(), full_values.end(), A_full.valuePtr());

        FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
            "Running %s SVD (k=%d)...\n",
            config.init_mode == 1 ? "Lanczos" : "IRLBA", k);

        MatS W_T_svd(k, m);
        MatS H_svd(k, n);
        VecS d_svd(k);

        if (config.init_mode == 1) {
            detail::initialize_lanczos(W_T_svd, H_svd, d_svd, A_full,
                                       k, static_cast<int>(m), static_cast<int>(n),
                                       config.seed, config.threads, verbose);
        } else {
            detail::initialize_irlba(W_T_svd, H_svd, d_svd, A_full,
                                     k, static_cast<int>(m), static_cast<int>(n),
                                     config.seed, config.threads, verbose);
        }

        // nmf_chunked expects W as m×k (not transposed)
        W_init_ptr = std::make_unique<MatS>(W_T_svd.transpose());
        H_init_ptr = std::make_unique<MatS>(std::move(H_svd));
    }

    // ---- 3. Delegate to chunked NMF (GPU or CPU) ----
    // I/O is overlapped with compute via double-buffered async prefetch
    // in nmf_chunked / nmf_chunked_gpu — no in-memory caching needed.
    auto& loader_ref = *loader_ptr;
    loader_ref.reset_forward();
    loader_ref.reset_transpose();

#ifdef FACTORNET_HAS_GPU
    {
        // Check if GPU is requested or auto-selected
        bool use_gpu = false;
        if (config.resource_override == "gpu" || config.resource_override == "GPU") {
            use_gpu = true;
        } else if (config.resource_override == "auto" || config.resource_override.empty()) {
            FactorNet::Resources res = FactorNet::Resources::detect();
            use_gpu = res.has_gpu;
        }

        if (use_gpu) {
            FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
                "Dispatching to GPU chunked NMF\n");
            try {
                return FactorNet::nmf::nmf_chunked_gpu<Scalar>(loader_ref, config,
                    W_init_ptr.get(), H_init_ptr.get());
            } catch (const std::exception& e) {
                FACTORNET_LOG_NMF(FactorNet::LogLevel::SUMMARY, verbose,
                    "GPU chunked NMF failed (%s), falling back to CPU\n", e.what());
                loader_ref.reset_forward();
                loader_ref.reset_transpose();
            }
        }
    }
#endif

    return FactorNet::nmf::nmf_chunked<Scalar>(loader_ref, config,
        W_init_ptr.get(), H_init_ptr.get());
}

}  // namespace nmf
}  // namespace FactorNet

