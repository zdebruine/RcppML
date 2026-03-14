/**
 * @file sp_gpu_bridge.cu
 * @brief CUDA bridge for reading .spz files directly to GPU memory.
 *
 * Provides C-callable functions (via .C() from R) for:
 *   1. sp_read_gpu: Read .spz v2 file, decode on GPU, return device ptrs
 *   2. sp_free_gpu: Free GPU-resident CSC arrays
 *
 * The GPU-resident CSC can be passed directly to the NMF GPU pipeline,
 * bypassing the CPU→GPU transfer step entirely.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <RcppML/core/logging.hpp>

// v2 adapter: CPU decode + GPU upload
#include "streampress/gpu/spgpu_v2_adapter.cuh"

extern "C" {

/**
 * @brief Read a .spz file and decode to GPU memory.
 *
 * Called from R via .C(). All pointer args are in/out.
 * On success, col_ptr/row_idx/values point to device memory.
 *
 * @param path_ptr     Path to .spz file (char*)
 * @param device_id    CUDA device ID (int*)
 * @param out_col_ptr  Output: device pointer to col_ptr (int64_t*, cast from int*)
 * @param out_row_idx  Output: device pointer to row_idx (int64_t*)
 * @param out_values   Output: device pointer to values (int64_t*)
 * @param out_m        Output: number of rows
 * @param out_n        Output: number of columns
 * @param out_nnz      Output: number of non-zeros
 * @param out_status   Output: 0 = success, nonzero = error
 */
void rcppml_sp_read_gpu(
    const char** path_ptr,
    int* device_id,
    double* out_col_ptr_addr,   // device ptr stored as double for R
    double* out_row_idx_addr,
    double* out_values_addr,
    int* out_m,
    int* out_n,
    double* out_nnz,
    int* out_status)
{
    *out_status = -1;

    const char* path = *path_ptr;
    int dev = *device_id;

    // 1. Read file to host memory
    FILE* f = fopen(path, "rb");
    if (!f) {
        FACTORNET_GPU_WARN("[sp_read_gpu] Cannot open file: %s\n", path);
        *out_status = 1;
        return;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> file_data(file_size);
    if (fread(file_data.data(), 1, file_size, f) != file_size) {
        fclose(f);
        FACTORNET_GPU_WARN("[sp_read_gpu] Failed to read file\n");
        *out_status = 2;
        return;
    }
    fclose(f);

    // 2. Check version
    if (file_size < 6) {
        FACTORNET_GPU_WARN("[sp_read_gpu] File too small\n");
        *out_status = 3;
        return;
    }

    uint16_t version;
    memcpy(&version, file_data.data() + 4, 2);
    if (version != 2) {
        FACTORNET_GPU_WARN("[sp_read_gpu] Only v2 .spz files supported for GPU decode (got v%d)\n", version);
        *out_status = 4;
        return;
    }

    // 3. Build GPU decode context and decode
    try {
        auto* ctx = new spgpu::v2_adapter::V2GPUDecodeContext();
        ctx->build(file_data.data(), file_size, dev);
        ctx->decode();

        // 4. Return device pointers and dimensions
        *out_m = static_cast<int>(ctx->v2_header.m);
        *out_n = static_cast<int>(ctx->v2_header.n);
        *out_nnz = static_cast<double>(ctx->v2_header.nnz);

        // Store device pointers as doubles (R doesn't have int64)
        // These are opaque handles — user must call sp_free_gpu to release
        *out_col_ptr_addr = static_cast<double>(reinterpret_cast<uintptr_t>(ctx->d_col_ptr));
        *out_row_idx_addr = static_cast<double>(reinterpret_cast<uintptr_t>(ctx->d_row_idx));
        *out_values_addr  = static_cast<double>(reinterpret_cast<uintptr_t>(ctx->d_values));

        // Detach device pointers from context so they aren't freed on context destroy
        ctx->d_col_ptr = nullptr;
        ctx->d_row_idx = nullptr;
        ctx->d_values = nullptr;

        // Free everything else (decode buffers, etc.)
        delete ctx;

        *out_status = 0;
    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("[sp_read_gpu] GPU decode error: %s\n", e.what());
        *out_status = 5;
    }
}

/**
 * @brief Free GPU-resident CSC arrays previously allocated by sp_read_gpu.
 *
 * @param col_ptr_addr  Device pointer (as double)
 * @param row_idx_addr  Device pointer (as double)
 * @param values_addr   Device pointer (as double)
 * @param out_status    0 = success
 */
void rcppml_sp_free_gpu(
    double* col_ptr_addr,
    double* row_idx_addr,
    double* values_addr,
    int* out_status)
{
    *out_status = 0;

    auto to_ptr = [](double addr) -> void* {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
    };

    if (*col_ptr_addr != 0.0)
        cudaFree(to_ptr(*col_ptr_addr));
    if (*row_idx_addr != 0.0)
        cudaFree(to_ptr(*row_idx_addr));
    if (*values_addr != 0.0)
        cudaFree(to_ptr(*values_addr));

    *col_ptr_addr = 0.0;
    *row_idx_addr = 0.0;
    *values_addr = 0.0;
}

} // extern "C"
