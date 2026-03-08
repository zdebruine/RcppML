#pragma once
/**
 * @file graph_reg_gpu.cuh
 * @brief GPU graph regularization: G += λ · factor · L · factorᵀ
 *
 * Replaces the CPU fallback that downloads factor+Gram, applies Eigen
 * sparse multiply on host, and re-uploads. Uses cuSPARSE SpMM for the
 * sparse Laplacian multiply and cuBLAS GEMM for the dense Gram update.
 *
 * Usage:
 *   GpuGraphRegHandle<Scalar> handle;
 *   handle.init(ctx, L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
 *               L.rows(), L.nonZeros(), k);
 *   // per iteration:
 *   handle.apply(ctx, d_G.data.get(), d_H.data.get(), dim, lambda);
 */

#include <FactorNet/gpu/types.cuh>

namespace FactorNet { namespace gpu {

template<typename Scalar>
struct GpuGraphRegHandle {
    // cuSPARSE descriptors
    cusparseSpMatDescr_t desc_L = nullptr;
    cusparseDnMatDescr_t desc_B = nullptr;  // factor^T (dim×k, row-major view)
    cusparseDnMatDescr_t desc_C = nullptr;  // temp (dim×k, col-major)

    // Device storage for Laplacian CSC
    DeviceMemory<int>    d_col_ptr;
    DeviceMemory<int>    d_row_idx;
    DeviceMemory<Scalar> d_values;

    // SpMM workspace
    DeviceMemory<Scalar> d_temp;   // dim×k
    DeviceMemory<char>   d_buf;    // SpMM external buffer

    int dim_ = 0;
    int k_   = 0;

    GpuGraphRegHandle() = default;

    // Non-copyable, movable
    GpuGraphRegHandle(const GpuGraphRegHandle&) = delete;
    GpuGraphRegHandle& operator=(const GpuGraphRegHandle&) = delete;
    GpuGraphRegHandle(GpuGraphRegHandle&& o) noexcept { swap(o); }
    GpuGraphRegHandle& operator=(GpuGraphRegHandle&& o) noexcept {
        if (this != &o) { destroy(); swap(o); }
        return *this;
    }

    /**
     * Upload graph Laplacian to device and prepare SpMM pipeline.
     *
     * @param ctx       GPU context
     * @param h_outer   CSC outerIndexPtr (dim+1 ints)
     * @param h_inner   CSC innerIndexPtr (nnz ints)
     * @param h_vals    CSC valuePtr (nnz scalars)
     * @param dim       Laplacian dimension (n for H-graph, m for W-graph)
     * @param nnz       Number of non-zeros
     * @param k         Rank (number of factors)
     * @param factor_ptr Initial factor device pointer (for buffer sizing)
     */
    void init(const GPUContext& ctx,
              const int* h_outer, const int* h_inner, const Scalar* h_vals,
              int dim, int nnz, int k, const Scalar* factor_ptr)
    {
        dim_ = dim;
        k_   = k;

        // Upload CSC arrays
        d_col_ptr.allocate(dim + 1);
        d_row_idx.allocate(nnz);
        d_values.allocate(nnz);
        d_col_ptr.upload(h_outer, dim + 1);
        d_row_idx.upload(h_inner, nnz);
        d_values.upload(h_vals, nnz);

        // Sparse CSC descriptor: L (dim × dim)
        CUSPARSE_CHECK(cusparseCreateCsc(
            &desc_L, dim, dim, nnz,
            d_col_ptr.get(), d_row_idx.get(), d_values.get(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CudaDataType<Scalar>::value));

        // Temp buffer for SpMM result: dim × k, col-major
        d_temp.allocate(static_cast<size_t>(dim) * k);

        // Dense descriptor for factor^T: dim × k, row-major with ld=k
        // (reinterprets factor(k×dim, col-major, ld=k) as factor^T(dim×k, row-major))
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &desc_B, dim, k, k,
            const_cast<Scalar*>(factor_ptr),
            CudaDataType<Scalar>::value, CUSPARSE_ORDER_ROW));

        // Dense descriptor for temp: dim × k, col-major with ld=dim
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &desc_C, dim, k, dim,
            d_temp.get(),
            CudaDataType<Scalar>::value, CUSPARSE_ORDER_COL));

        // Query SpMM buffer size
        Scalar alpha_one = 1, beta_zero = 0;
        size_t buf_size = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, desc_L, desc_B, &beta_zero, desc_C,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMM_ALG_DEFAULT, &buf_size));
        if (buf_size > 0)
            d_buf.allocate(buf_size);
    }

    /**
     * Apply graph regularization on device:  G += lambda * factor * L * factor^T
     *
     * @param ctx         GPU context
     * @param d_G         Device pointer to k×k Gram (col-major, ld=k), modified in-place
     * @param d_factor    Device pointer to k×dim factor (col-major, ld=k)
     * @param dim         Factor dimension (must match init dim)
     * @param lambda      Regularization strength
     */
    void apply(const GPUContext& ctx,
               Scalar* d_G, const Scalar* d_factor,
               int dim, Scalar lambda) const
    {
        Scalar alpha_one = 1, beta_zero = 0, one = 1;

        // Update factor pointer in B descriptor
        CUSPARSE_CHECK(cusparseDnMatSetValues(desc_B, const_cast<Scalar*>(d_factor)));

        // Step 1: SpMM  temp(dim×k) = L(dim×dim) * factor^T(dim×k)
        CUSPARSE_CHECK(cusparseSpMM(
            ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, desc_L, desc_B, &beta_zero, desc_C,
            CudaDataType<Scalar>::value,
            CUSPARSE_SPMM_ALG_DEFAULT,
            d_buf.get()));

        // Step 2: GEMM  G += lambda * factor(k×dim) * temp(dim×k)
        if constexpr (std::is_same_v<Scalar, float>) {
            CUBLAS_CHECK(cublasSgemm(ctx.cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                k_, k_, dim,
                &lambda, d_factor, k_,
                d_temp.get(), dim,
                &one, d_G, k_));
        } else {
            CUBLAS_CHECK(cublasDgemm(ctx.cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                k_, k_, dim,
                &lambda, d_factor, k_,
                d_temp.get(), dim,
                &one, d_G, k_));
        }
    }

    void destroy() {
        if (desc_L) { cusparseDestroySpMat(desc_L); desc_L = nullptr; }
        if (desc_B) { cusparseDestroyDnMat(desc_B); desc_B = nullptr; }
        if (desc_C) { cusparseDestroyDnMat(desc_C); desc_C = nullptr; }
    }

    ~GpuGraphRegHandle() { destroy(); }

private:
    void swap(GpuGraphRegHandle& o) noexcept {
        std::swap(desc_L, o.desc_L);
        std::swap(desc_B, o.desc_B);
        std::swap(desc_C, o.desc_C);
        std::swap(d_col_ptr, o.d_col_ptr);
        std::swap(d_row_idx, o.d_row_idx);
        std::swap(d_values, o.d_values);
        std::swap(d_temp, o.d_temp);
        std::swap(d_buf, o.d_buf);
        std::swap(dim_, o.dim_);
        std::swap(k_, o.k_);
    }
};

}}  // namespace FactorNet::gpu
