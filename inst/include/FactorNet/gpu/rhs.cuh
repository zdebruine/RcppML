/**
 * @file rhs.cuh
 * @brief GPU RHS computation for NMF factor updates via cuSPARSE SpMM.
 *
 * Computes:
 *   B = W * A   (k×n, for H update — W is k×m, A is m×n sparse CSC)
 *   B = H * A^T (k×m, for W update — H is k×n, A^T is n×m sparse CSC)
 *
 * Uses a single implementation path: cuSPARSE SpMM with CSR_ALG2 (merge-based,
 * load-balanced).  The sparse CSC arrays are reinterpreted as CSR of the
 * transpose, and dense matrices are presented in row-major layout so that:
 *
 *   B^T(out × k) = Sparse^T(out × in, CSR) · Factor^T(in × k)
 *
 * which writes the correct B in column-major memory without an explicit
 * transpose.
 *
 * The SpMM handle is created once with preprocessing (one-time sparsity
 * analysis) and reused every NMF iteration.  Only the dense data pointers
 * are updated between calls.
 *
 * Design principles:
 *   - One code path for all k values, densities, and GPU architectures
 *   - Vendor-optimized (NVIDIA cuSPARSE), auto-tuned per architecture
 *   - No custom CUDA kernels — portable across compute capabilities
 *   - Preprocessing amortized over NMF iterations
 */

#pragma once

#include <cstdint>
#include <vector>
#include "types.cuh"

namespace FactorNet {
namespace gpu {

// ===========================================================================
// cuSPARSE SpMM handle — created once, reused each NMF iteration
// ===========================================================================

template<typename Scalar>
struct SpMMHandle {
    cusparseSpMatDescr_t sp_descr = nullptr;
    cusparseDnMatDescr_t dn_in    = nullptr;
    cusparseDnMatDescr_t dn_out   = nullptr;
    void*  buffer      = nullptr;
    size_t buffer_size = 0;
    bool   ready       = false;

    ~SpMMHandle() { destroy(); }

    SpMMHandle() = default;
    SpMMHandle(const SpMMHandle&) = delete;
    SpMMHandle& operator=(const SpMMHandle&) = delete;
    SpMMHandle(SpMMHandle&& o) noexcept
        : sp_descr(o.sp_descr), dn_in(o.dn_in), dn_out(o.dn_out),
          buffer(o.buffer), buffer_size(o.buffer_size), ready(o.ready) {
        o.sp_descr = nullptr; o.dn_in = nullptr; o.dn_out = nullptr;
        o.buffer = nullptr; o.ready = false;
    }
    SpMMHandle& operator=(SpMMHandle&& o) noexcept {
        if (this != &o) {
            destroy();
            sp_descr = o.sp_descr; dn_in = o.dn_in; dn_out = o.dn_out;
            buffer = o.buffer; buffer_size = o.buffer_size; ready = o.ready;
            o.sp_descr = nullptr; o.dn_in = nullptr; o.dn_out = nullptr;
            o.buffer = nullptr; o.ready = false;
        }
        return *this;
    }

    void destroy() {
        if (!ready) return;
        if (sp_descr) { cusparseDestroySpMat(sp_descr); sp_descr = nullptr; }
        if (dn_in)    { cusparseDestroyDnMat(dn_in);    dn_in = nullptr; }
        if (dn_out)   { cusparseDestroyDnMat(dn_out);   dn_out = nullptr; }
        if (buffer)   { cudaFree(buffer); buffer = nullptr; }
        ready = false;
    }

    /**
     * @brief Set up SpMM for: B(k × out_n) = Factor(k × in_n) · Sparse(in_n × out_n)
     *
     * Internally computes:
     *   B^T(out_n × k) = Sparse^T(out_n × in_n, CSR) · Factor^T(in_n × k)
     *
     * The sparse CSC input is reinterpreted as CSR of its transpose.
     * Dense matrices use row-major layout, equivalent to reading the
     * column-major GPU buffers as transposed.
     *
     * @param ctx       GPU context (cuSPARSE handle + stream)
     * @param sp        Sparse matrix in CSC format
     * @param k         Factorization rank
     * @param csc_cols  Number of CSC columns (= rows of the CSR view)
     * @param csc_rows  Number of CSC rows (= cols of the CSR view)
     * @param fac_ptr   Initial dense factor pointer (k × csc_rows, col-major)
     * @param out_ptr   Initial dense output pointer (k × csc_cols, col-major)
     */
    void init(const GPUContext& ctx,
              const SparseMatrixGPU<Scalar>& sp,
              int k, int csc_cols, int csc_rows,
              Scalar* fac_ptr, Scalar* out_ptr)
    {
        destroy();
        const cudaDataType dt =
            std::is_same<Scalar, float>::value ? CUDA_R_32F : CUDA_R_64F;

        // Sparse descriptor: CSR of the transpose (csc_cols × csc_rows)
        // CSC col_ptr → CSR row_ptr, CSC row_idx → CSR col_idx
        CUSPARSE_CHECK(cusparseCreateCsr(
            &sp_descr,
            (int64_t)csc_cols, (int64_t)csc_rows, (int64_t)sp.nnz,
            (void*)sp.col_ptr.get(),
            (void*)sp.row_indices.get(),
            (void*)sp.values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, dt));

        // Dense factor: (csc_rows × k) row-major = k × csc_rows col-major in memory
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &dn_in, (int64_t)csc_rows, (int64_t)k, (int64_t)k,
            (void*)fac_ptr, dt, CUSPARSE_ORDER_ROW));

        // Dense output: (csc_cols × k) row-major = k × csc_cols col-major in memory
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &dn_out, (int64_t)csc_cols, (int64_t)k, (int64_t)k,
            (void*)out_ptr, dt, CUSPARSE_ORDER_ROW));

        // Buffer allocation
        Scalar alpha = Scalar(1), beta = Scalar(0);
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_descr, dn_in, &beta, dn_out,
            dt, CUSPARSE_SPMM_CSR_ALG2, &buffer_size));

        CUDA_CHECK(cudaMalloc(&buffer, std::max(buffer_size, (size_t)1)));

        // One-time preprocessing: sparsity analysis (amortized over iterations)
        CUSPARSE_CHECK(cusparseSpMM_preprocess(
            ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_descr, dn_in, &beta, dn_out,
            dt, CUSPARSE_SPMM_CSR_ALG2, buffer));

        ready = true;
    }

    /**
     * @brief Execute SpMM: B = Factor · Sparse (via transposed row-major trick).
     *
     * Updates dense matrix value pointers (factor values change each iteration).
     */
    void compute(const GPUContext& ctx,
                 const Scalar* fac_ptr,
                 Scalar* out_ptr) const
    {
        CUSPARSE_CHECK(cusparseDnMatSetValues(dn_in, (void*)fac_ptr));
        CUSPARSE_CHECK(cusparseDnMatSetValues(dn_out, (void*)out_ptr));

        Scalar alpha = Scalar(1), beta = Scalar(0);
        const cudaDataType dt =
            std::is_same<Scalar, float>::value ? CUDA_R_32F : CUDA_R_64F;

        CUSPARSE_CHECK(cusparseSpMM(
            ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_descr, dn_in, &beta, dn_out,
            dt, CUSPARSE_SPMM_CSR_ALG2, buffer));
    }
};

// ===========================================================================
// Host dispatch wrappers
// ===========================================================================

/**
 * @brief Compute B(k×n) = W(k×m) · A(m×n) using an initialized SpMM handle.
 */
template<typename Scalar>
void compute_rhs_forward_gpu(const GPUContext& ctx,
                              const SpMMHandle<Scalar>& spmm,
                              const DenseMatrixGPU<Scalar>& W,
                              DenseMatrixGPU<Scalar>& B) {
    spmm.compute(ctx, W.data.get(), B.data.get());
}

/**
 * @brief Compute B(k×m) = H(k×n) · A^T(n×m) using an initialized SpMM handle.
 */
template<typename Scalar>
void compute_rhs_transpose_gpu(const GPUContext& ctx,
                                const SpMMHandle<Scalar>& spmm,
                                const DenseMatrixGPU<Scalar>& H,
                                DenseMatrixGPU<Scalar>& B) {
    spmm.compute(ctx, H.data.get(), B.data.get());
}

// ===========================================================================
// One-shot convenience wrappers (for callers without pre-built handles)
//
// These create a temporary cuSPARSE SpMM handle, compute, and destroy.
// Use when the sparse structure changes between calls (e.g., streaming
// panels) or for one-off computations (testing, loss re-computation).
// For iteration loops, prefer pre-built SpMMHandle for amortized cost.
// ===========================================================================

/**
 * @brief One-shot B(k×n) = W(k×m) · A(m×n, CSC).
 *
 * Creates a temporary SpMM handle with preprocessing, computes, and destroys.
 */
template<typename Scalar>
void compute_rhs_forward_gpu(const GPUContext& ctx,
                              const DenseMatrixGPU<Scalar>& W,
                              const SparseMatrixGPU<Scalar>& A,
                              DenseMatrixGPU<Scalar>& B) {
    SpMMHandle<Scalar> tmp;
    tmp.init(ctx, A, W.rows, /*csc_cols=*/A.cols, /*csc_rows=*/A.rows,
             const_cast<Scalar*>(W.data.get()), B.data.get());
    tmp.compute(ctx, W.data.get(), B.data.get());
}

/**
 * @brief One-shot B(k×m) = H(k×n) · A^T(n×m) given A in CSC.
 *
 * Uses cuSPARSE TRANSPOSE operation on A^T CSR (= A CSC arrays) with
 * CSR_ALG1 (which supports transpose, unlike ALG2).  No A^T CSC needed.
 */
template<typename Scalar>
void compute_rhs_transpose_gpu(const GPUContext& ctx,
                                const DenseMatrixGPU<Scalar>& H,
                                const SparseMatrixGPU<Scalar>& A,
                                DenseMatrixGPU<Scalar>& B) {
    const int k = H.rows;
    const int m = A.rows;
    const int n = A.cols;
    const cudaDataType dt =
        std::is_same<Scalar, float>::value ? CUDA_R_32F : CUDA_R_64F;

    // A's CSC arrays = CSR of A^T (n×m).
    // TRANSPOSE on this CSR gives (A^T)^T = A (m×n).
    // So: B^T(m×k) = A(m×n) × H^T(n×k).
    cusparseSpMatDescr_t sp = nullptr;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &sp, (int64_t)n, (int64_t)m, (int64_t)A.nnz,
        (void*)A.col_ptr.get(), (void*)A.row_indices.get(),
        (void*)A.values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, dt));

    // Dense H^T: (n×k) row-major = k×n col-major
    cusparseDnMatDescr_t dn_h = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dn_h, (int64_t)n, (int64_t)k, (int64_t)k,
        (void*)H.data.get(), dt, CUSPARSE_ORDER_ROW));

    // Dense B^T: (m×k) row-major = k×m col-major
    cusparseDnMatDescr_t dn_b = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dn_b, (int64_t)m, (int64_t)k, (int64_t)k,
        (void*)B.data.get(), dt, CUSPARSE_ORDER_ROW));

    Scalar alpha = Scalar(1), beta = Scalar(0);
    size_t buf_sz = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, sp, dn_h, &beta, dn_b,
        dt, CUSPARSE_SPMM_CSR_ALG1, &buf_sz));

    void* buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, std::max(buf_sz, (size_t)1)));

    CUSPARSE_CHECK(cusparseSpMM(
        ctx.cusparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, sp, dn_h, &beta, dn_b,
        dt, CUSPARSE_SPMM_CSR_ALG1, buf));

    // Cleanup
    cusparseDestroySpMat(sp);
    cusparseDestroyDnMat(dn_h);
    cusparseDestroyDnMat(dn_b);
    cudaFree(buf);
}

// ===========================================================================
// Build A^T CSC arrays on host (one-time, for W-update SpMM handle)
// ===========================================================================

/**
 * @brief Build A^T in CSC format from A's CSC arrays.
 *
 * A is m×n in CSC.  A^T is n×m in CSC (m "columns", each containing the
 * non-zeros from that row of A).  Computed once at NMF startup and uploaded
 * to GPU for the W-update SpMM handle.
 *
 * @param m, n     Dimensions of A
 * @param nnz      Number of non-zeros
 * @param col_ptr  A's CSC column pointer (length n+1)
 * @param row_idx  A's CSC row indices (length nnz)
 * @param values   A's CSC values (length nnz)
 * @param At_col_ptr  Output: A^T CSC column pointer (length m+1)
 * @param At_row_idx  Output: A^T CSC row indices (length nnz)
 * @param At_values   Output: A^T CSC values (length nnz)
 */
template<typename Scalar>
void build_transpose_csc_host(
    int m, int n, int nnz,
    const int* col_ptr, const int* row_idx, const Scalar* values,
    std::vector<int>& At_col_ptr,
    std::vector<int>& At_row_idx,
    std::vector<Scalar>& At_values)
{
    // Count non-zeros per row of A (= column nnz of A^T)
    At_col_ptr.assign(m + 1, 0);
    for (int p = 0; p < nnz; ++p)
        At_col_ptr[row_idx[p] + 1]++;

    // Prefix sum
    for (int i = 0; i < m; ++i)
        At_col_ptr[i + 1] += At_col_ptr[i];

    // Fill row indices and values
    At_row_idx.resize(nnz);
    At_values.resize(nnz);

    std::vector<int> write_pos(At_col_ptr.begin(), At_col_ptr.begin() + m);
    for (int j = 0; j < n; ++j) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int i = row_idx[p];
            int dest = write_pos[i]++;
            At_row_idx[dest] = j;
            At_values[dest] = values[p];
        }
    }
}

} // namespace gpu
} // namespace FactorNet
