/**
 * @file types.cuh
 * @brief Core CUDA types, RAII wrappers, and GPU context for RcppML GPU NMF.
 *
 * Provides:
 *   - CUDA error-checking macros
 *   - DeviceMemory<T>: RAII wrapper for device allocations
 *   - SparseMatrixGPU<T>: CSC sparse matrix on device
 *   - DenseMatrixGPU<T>: Column-major dense matrix on device
 *   - GPUContext: cuBLAS/cuSPARSE handles and stream management
 *   - CudaDataType<T>: Type traits for cuBLAS dispatch
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <stdexcept>

namespace FactorNet {
namespace gpu {

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                   \
        throw std::runtime_error(cudaGetErrorString(err));                     \
    }                                                                          \
} while(0)

#define CUBLAS_CHECK(call) do {                                                \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n",                          \
                (int)status, __FILE__, __LINE__);                              \
        throw std::runtime_error("cuBLAS error");                              \
    }                                                                          \
} while(0)

#define CUSPARSE_CHECK(call) do {                                              \
    cusparseStatus_t status = (call);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSPARSE error %d at %s:%d\n",                        \
                (int)status, __FILE__, __LINE__);                              \
        throw std::runtime_error("cuSPARSE error");                            \
    }                                                                          \
} while(0)

#define CUSOLVER_CHECK(call) do {                                              \
    cusolverStatus_t status = (call);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSOLVER error %d at %s:%d\n",                        \
                (int)status, __FILE__, __LINE__);                              \
        throw std::runtime_error("cuSOLVER error");                            \
    }                                                                          \
} while(0)

// ---------------------------------------------------------------------------
// CudaDataType: type traits mapping C++ types to cuBLAS enum values
// ---------------------------------------------------------------------------

template<typename T>
struct CudaDataType;

template<>
struct CudaDataType<float> {
    static constexpr cudaDataType_t value = CUDA_R_32F;
};

template<>
struct CudaDataType<double> {
    static constexpr cudaDataType_t value = CUDA_R_64F;
};

template<>
struct CudaDataType<__half> {
    static constexpr cudaDataType_t value = CUDA_R_16F;
};

// ---------------------------------------------------------------------------
// DeviceMemory<T>: RAII wrapper for device memory allocations
// ---------------------------------------------------------------------------

template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0), owns_(true) {}

    explicit DeviceMemory(size_t count)
        : ptr_(nullptr), size_(count), owns_(true)
    {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    /// Non-owning constructor: wraps an existing device pointer without
    /// allocating or freeing. Used for zero-copy paths (sp_read_gpu → NMF).
    static DeviceMemory wrap(T* device_ptr, size_t count) {
        DeviceMemory dm;
        dm.ptr_ = device_ptr;
        dm.size_ = count;
        dm.owns_ = false;
        return dm;
    }

    ~DeviceMemory() {
        if (ptr_ && owns_) cudaFree(ptr_);
    }

    // Move semantics
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), owns_(other.owns_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_ && owns_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            owns_ = other.owns_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // No copying
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool owns() const { return owns_; }

    void upload(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (ptr_ && size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
    }

    void allocate(size_t count) {
        if (ptr_ && owns_) cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = count;
        owns_ = true;
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

private:
    T* ptr_;
    size_t size_;
    bool owns_;
};

// ---------------------------------------------------------------------------
// SparseMatrixGPU<T>: CSC sparse matrix stored on device
// ---------------------------------------------------------------------------

template<typename T>
struct SparseMatrixGPU {
    int rows, cols;
    int nnz;
    DeviceMemory<int> col_ptr;     // size: cols + 1
    DeviceMemory<int> row_indices;  // size: nnz
    DeviceMemory<T>   values;       // size: nnz

    SparseMatrixGPU() : rows(0), cols(0), nnz(0) {}

    // Allocate device memory without uploading data
    SparseMatrixGPU(int m, int n, int nnz_)
        : rows(m), cols(n), nnz(nnz_),
          col_ptr(n + 1), row_indices(nnz_), values(nnz_) {}

    SparseMatrixGPU(int m, int n, int nnz_,
                    const int* h_col_ptr,
                    const int* h_row_indices,
                    const T* h_values)
        : rows(m), cols(n), nnz(nnz_),
          col_ptr(n + 1), row_indices(nnz_), values(nnz_)
    {
        col_ptr.upload(h_col_ptr, n + 1);
        row_indices.upload(h_row_indices, nnz_);
        values.upload(h_values, nnz_);
    }

    // Upload data to pre-allocated device memory
    void upload(const int* h_col_ptr, const int* h_row_indices, const T* h_values) {
        col_ptr.upload(h_col_ptr, cols + 1);
        row_indices.upload(h_row_indices, nnz);
        values.upload(h_values, nnz);
    }

    /// Create a non-owning SparseMatrixGPU that wraps existing device pointers.
    /// The caller is responsible for the lifetime of the device memory.
    /// Used for zero-copy NMF on gpu_sparse_matrix objects from sp_read_gpu().
    static SparseMatrixGPU from_device_ptrs(int m, int n, int nnz_,
                                            int* d_col_ptr,
                                            int* d_row_indices,
                                            T* d_values)
    {
        SparseMatrixGPU mat;
        mat.rows = m;
        mat.cols = n;
        mat.nnz = nnz_;
        mat.col_ptr = DeviceMemory<int>::wrap(d_col_ptr, n + 1);
        mat.row_indices = DeviceMemory<int>::wrap(d_row_indices, nnz_);
        mat.values = DeviceMemory<T>::wrap(d_values, nnz_);
        return mat;
    }
};

// ---------------------------------------------------------------------------
// DenseMatrixGPU<T>: Column-major dense matrix stored on device
// ---------------------------------------------------------------------------

template<typename T>
struct DenseMatrixGPU {
    int rows, cols;
    DeviceMemory<T> data;

    DenseMatrixGPU() : rows(0), cols(0) {}

    DenseMatrixGPU(int m, int n)
        : rows(m), cols(n), data(static_cast<size_t>(m) * n) {}

    DenseMatrixGPU(int m, int n, const T* host_data)
        : rows(m), cols(n), data(static_cast<size_t>(m) * n)
    {
        data.upload(host_data, static_cast<size_t>(m) * n);
    }

    void upload_from(const T* host_data) {
        data.upload(host_data, static_cast<size_t>(rows) * cols);
    }

    void download_to(T* host_data) const {
        data.download(host_data, static_cast<size_t>(rows) * cols);
    }

    void zero() { data.zero(); }

    size_t elements() const { return static_cast<size_t>(rows) * cols; }
};

// ---------------------------------------------------------------------------
// GPUContext: cuBLAS/cuSPARSE handle and stream management
// ---------------------------------------------------------------------------

struct GPUContext {
    cublasHandle_t cublas;
    cublasHandle_t& handle = cublas;  // backward-compat alias for legacy NMF code
    cusparseHandle_t cusparse;
    cusolverDnHandle_t cusolver;
    cudaStream_t stream;
    int sm_count;  // Number of SMs for kernel launch configuration

    GPUContext() {
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

        // Enable TF32 for float on Ampere+ GPUs
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));

        // Query SM count for adaptive kernel dispatch
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        sm_count = prop.multiProcessorCount;
    }

    ~GPUContext() {
        // Sync stream before destroying handles to avoid async errors
        cudaStreamSynchronize(stream);
        // Clear any sticky async errors so handle destruction succeeds
        cudaGetLastError();
        cusolverDnDestroy(cusolver);
        cublasDestroy(cublas);
        cusparseDestroy(cusparse);
        cudaStreamDestroy(stream);
    }

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    void sync() const {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// ---------------------------------------------------------------------------
// DualCSR: CSR(A) + CSR(A^T) from a CSC upload, for all-gather SpMV
// ---------------------------------------------------------------------------

/**
 * @brief Convert CSC(A) [uploaded as CSR(A^T)] to also have CSR(A).
 *
 * When A is m×n stored CSC, cuSPARSE sees it as CSR(A^T) with n rows, m cols.
 * Forward SpMV (A*v) via TRANSPOSE on CSR(A^T) uses the slow scatter path.
 * By converting to CSR(A), we can use NON_TRANSPOSE (gather) for both directions:
 *   A*v   → NON_TRANSPOSE on CSR(A)   [m rows × n cols]
 *   A^T*w → NON_TRANSPOSE on CSR(A^T) [n rows × m cols]
 *
 * The conversion uses cusparseCsr2cscEx2 on device (no host roundtrip).
 * Memory cost: nnz ints + nnz Scalars + (m+1) ints extra.
 */
template<typename Scalar>
struct DualCSR {
    // CSR(A): m rows × n cols — for forward SpMV/SpMM (A*v, A*X)
    DeviceMemory<int>    row_ptr_A;    // m+1
    DeviceMemory<int>    col_idx_A;    // nnz
    DeviceMemory<Scalar> values_A;     // nnz
    cusparseSpMatDescr_t descr_A  = nullptr;

    // CSR(A^T): n rows × m cols — for transpose SpMV/SpMM (A^T*w, A^T*Y)
    // These are just aliases to the original CSC upload.
    cusparseSpMatDescr_t descr_AT = nullptr;

    int m = 0, n = 0, nnz = 0;

    DualCSR() = default;

    /**
     * Build dual-CSR from a CSC upload (SparseMatrixGPU holds CSC = CSR(A^T)).
     * @param ctx       GPU context with cuSPARSE handle
     * @param d_A       The uploaded CSC matrix (col_ptr, row_indices, values)
     * @param rows      Number of rows in A (m)
     * @param cols      Number of columns in A (n)
     * @param nnz_      Number of nonzeros
     */
    void init(const GPUContext& ctx,
              const SparseMatrixGPU<Scalar>& d_A,
              int rows, int cols, int nnz_)
    {
        m = rows; n = cols; nnz = nnz_;

        // Allocate CSR(A) arrays
        row_ptr_A = DeviceMemory<int>(m + 1);
        col_idx_A = DeviceMemory<int>(nnz);
        values_A  = DeviceMemory<Scalar>(nnz);

        // cusparseCsr2cscEx2: treat input as CSR(A^T) [n rows, m cols]
        // output is CSC(A^T) = CSR(A) [m rows, n cols]
        size_t bufSize = 0;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            ctx.cusparse, n, m, nnz,
            d_A.values.get(), d_A.col_ptr.get(), d_A.row_indices.get(),
            values_A.get(), row_ptr_A.get(), col_idx_A.get(),
            CudaDataType<Scalar>::value,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            &bufSize));
        DeviceMemory<char> buf(bufSize > 0 ? bufSize : 1);
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            ctx.cusparse, n, m, nnz,
            d_A.values.get(), d_A.col_ptr.get(), d_A.row_indices.get(),
            values_A.get(), row_ptr_A.get(), col_idx_A.get(),
            CudaDataType<Scalar>::value,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            buf.get()));

        // CSR(A): m rows × n cols
        CUSPARSE_CHECK(cusparseCreateCsr(
            &descr_A, m, n, nnz,
            row_ptr_A.get(), col_idx_A.get(), values_A.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));

        // CSR(A^T): n rows × m cols (aliases to original CSC upload)
        CUSPARSE_CHECK(cusparseCreateCsr(
            &descr_AT, n, m, nnz,
            d_A.col_ptr.get(), d_A.row_indices.get(), d_A.values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));
    }

    /**
     * Refresh CSR(A) values after modifying d_A.values in-place (e.g., for CV holdout zeroing).
     * The sparsity pattern is unchanged; only numeric values are retransposed.
     * CSR(A^T) (descr_AT) automatically sees the change since it aliases d_A.values directly.
     */
    void refresh_values_from_csc(const GPUContext& ctx, const SparseMatrixGPU<Scalar>& d_A) {
        size_t bufSize = 0;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            ctx.cusparse, n, m, nnz,
            d_A.values.get(), d_A.col_ptr.get(), d_A.row_indices.get(),
            values_A.get(), row_ptr_A.get(), col_idx_A.get(),
            CudaDataType<Scalar>::value,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            &bufSize));
        DeviceMemory<char> buf(bufSize > 0 ? bufSize : 1);
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            ctx.cusparse, n, m, nnz,
            d_A.values.get(), d_A.col_ptr.get(), d_A.row_indices.get(),
            values_A.get(), row_ptr_A.get(), col_idx_A.get(),
            CudaDataType<Scalar>::value,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            buf.get()));
    }

    ~DualCSR() {
        if (descr_A)  cusparseDestroySpMat(descr_A);
        if (descr_AT) cusparseDestroySpMat(descr_AT);
    }

    DualCSR(const DualCSR&) = delete;
    DualCSR& operator=(const DualCSR&) = delete;
    DualCSR(DualCSR&& o) noexcept
        : row_ptr_A(std::move(o.row_ptr_A)), col_idx_A(std::move(o.col_idx_A)),
          values_A(std::move(o.values_A)), descr_A(o.descr_A), descr_AT(o.descr_AT),
          m(o.m), n(o.n), nnz(o.nnz)
    { o.descr_A = nullptr; o.descr_AT = nullptr; }
    DualCSR& operator=(DualCSR&& o) noexcept {
        if (this != &o) {
            if (descr_A)  cusparseDestroySpMat(descr_A);
            if (descr_AT) cusparseDestroySpMat(descr_AT);
            row_ptr_A = std::move(o.row_ptr_A);
            col_idx_A = std::move(o.col_idx_A);
            values_A  = std::move(o.values_A);
            descr_A = o.descr_A; descr_AT = o.descr_AT;
            m = o.m; n = o.n; nnz = o.nnz;
            o.descr_A = nullptr; o.descr_AT = nullptr;
        }
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Mixed-precision helper: load factor value from fp16 shadow or full precision
// ---------------------------------------------------------------------------

/**
 * @brief Read a factor matrix element, using fp16 shadow if available.
 *
 * When factor_half is non-null, reads from fp16 and converts to Scalar.
 * Otherwise reads directly from the full-precision array. The branch is
 * warp-uniform (all threads take the same path), so no divergence penalty.
 */
template<typename Scalar>
__device__ __forceinline__ Scalar load_factor(
    const Scalar* __restrict__ full,
    const __half* __restrict__ half_ptr,
    int idx)
{
    return half_ptr
        ? static_cast<Scalar>(__half2float(half_ptr[idx]))
        : full[idx];
}

// ---------------------------------------------------------------------------
// GPUMatrix<Scalar>: unified sparse/dense GPU matrix with fp32-first design.
//
// Design principles:
//   1. Always deep-copy from host — no mapped/aliased host memory in GPU paths.
//   2. Always convert to Scalar precision on upload (typically float = fp32).
//   3. Sparse: holds SparseMatrixGPU (CSC) + DualCSR (CSR(A) + CSR(A^T))
//              Transpose is built GPU-side via cusparseCsr2cscEx2 — zero PCIe.
//   4. Dense:  holds DenseMatrixGPU (col-major). Transpose handled by
//              cuBLAS CUBLAS_OP_T — no second allocation needed.
//   5. Single nnz()/m/n interface regardless of storage kind.
//
// Usage (preferred path in bridge layer):
//   GPUMatrix<float> gm = GPUMatrix<float>::from_csc(ctx, col_ptr, row_idx,
//                                                     vals_d, m, n, nnz);
//   // then pass gm.sparse / gm.dual to SVD kernels
// ---------------------------------------------------------------------------

template<typename Scalar = float>
struct GPUMatrix {
    enum class Kind { Sparse, Dense };

    Kind kind = Kind::Sparse;
    int  m    = 0;
    int  n    = 0;

    // Sparse storage (valid when kind == Kind::Sparse)
    SparseMatrixGPU<Scalar> sparse;  // CSC on device
    DualCSR<Scalar>         dual;    // CSR(A) + CSR(A^T) descriptors

    // Dense storage (valid when kind == Kind::Dense)
    DenseMatrixGPU<Scalar>  dense;   // col-major, m×n

    // Number of stored elements (nnz for sparse, m*n for dense)
    int64_t nnz() const noexcept {
        if (kind == Kind::Sparse) return static_cast<int64_t>(sparse.nnz);
        return static_cast<int64_t>(m) * n;
    }

    GPUMatrix() = default;

    // -----------------------------------------------------------------------
    // Sparse factories
    // -----------------------------------------------------------------------

    /// Build sparse GPU matrix from host CSC arrays already cast to Scalar.
    /// Performs one PCIe upload (col_ptr + row_idx + values) and builds
    /// DualCSR on GPU (transpose via cusparseCsr2cscEx2 — zero extra PCIe).
    static GPUMatrix from_csc(const GPUContext& ctx,
                               const int*    h_col_ptr,
                               const int*    h_row_idx,
                               const Scalar* h_values,
                               int rows, int cols, int nnz_)
    {
        GPUMatrix gm;
        gm.kind   = Kind::Sparse;
        gm.m      = rows;
        gm.n      = cols;
        gm.sparse = SparseMatrixGPU<Scalar>(rows, cols, nnz_,
                                             h_col_ptr, h_row_idx, h_values);
        gm.dual.init(ctx, gm.sparse, rows, cols, nnz_);
        return gm;
    }

    /// Build sparse GPU matrix from host CSC with double-precision values.
    /// Performs CPU-side cast to Scalar (fp32) before upload.
    /// This is the standard path from the R bridge (R always uses double).
    static GPUMatrix from_csc_double(const GPUContext& ctx,
                                      const int*    h_col_ptr,
                                      const int*    h_row_idx,
                                      const double* h_values_d,
                                      int rows, int cols, int nnz_)
    {
        std::vector<Scalar> h_vals(nnz_);
        for (int i = 0; i < nnz_; ++i)
            h_vals[i] = static_cast<Scalar>(h_values_d[i]);
        return from_csc(ctx, h_col_ptr, h_row_idx, h_vals.data(), rows, cols, nnz_);
    }

    // -----------------------------------------------------------------------
    // Dense factories
    // -----------------------------------------------------------------------

    /// Build dense GPU matrix from host col-major array already cast to Scalar.
    /// Single PCIe upload of m*n elements. No explicit A^T stored —
    /// cuBLAS CUBLAS_OP_T handles transposition at zero memory cost.
    static GPUMatrix from_dense(const GPUContext& /*ctx*/,
                                 const Scalar* h_data,
                                 int rows, int cols)
    {
        GPUMatrix gm;
        gm.kind  = Kind::Dense;
        gm.m     = rows;
        gm.n     = cols;
        gm.dense = DenseMatrixGPU<Scalar>(rows, cols, h_data);
        return gm;
    }

    /// Build dense GPU matrix from host col-major double array.
    /// CPU-side cast to Scalar (fp32) then single PCIe upload.
    static GPUMatrix from_dense_double(const GPUContext& ctx,
                                        const double* h_data_d,
                                        int rows, int cols)
    {
        size_t elems = static_cast<size_t>(rows) * cols;
        std::vector<Scalar> h_data(elems);
        for (size_t i = 0; i < elems; ++i)
            h_data[i] = static_cast<Scalar>(h_data_d[i]);
        return from_dense(ctx, h_data.data(), rows, cols);
    }

    // Non-copyable, movable (GPUContext-dependent resources are owned by members)
    GPUMatrix(const GPUMatrix&) = delete;
    GPUMatrix& operator=(const GPUMatrix&) = delete;
    GPUMatrix(GPUMatrix&&) = default;
    GPUMatrix& operator=(GPUMatrix&&) = default;
};

// ---------------------------------------------------------------------------
// Device helper: binary search on a sorted CSC column to check if a
// (row, col) entry appears in a user-supplied sparse mask.
//
// mask_col_ptr[col] .. mask_col_ptr[col+1]-1 contains sorted row indices.
// Returns true if row is found in that range.
// Pass nullptr for mask_col_ptr when there is no user mask.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool is_user_masked_device(
    int row, int col,
    const int* __restrict__ mask_col_ptr,
    const int* __restrict__ mask_row_idx)
{
    if (!mask_col_ptr) return false;
    int lo = mask_col_ptr[col];
    int hi = mask_col_ptr[col + 1] - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int r   = mask_row_idx[mid];
        if (r == row) return true;
        if (r <  row) lo = mid + 1;
        else          hi = mid - 1;
    }
    return false;
}

} // namespace gpu
} // namespace FactorNet
