// GPU function stubs for non-CUDA builds
// These provide fallback implementations for GPU functions called
// from R via .C(). Only active entry points are stubbed here;
// dead legacy entry points (rcppml_gpu_nmf_double/float) have
// been removed since the unified path uses Rcpp gateway instead.

#ifndef FACTORNET_HAS_GPU

#include <Rcpp.h>
#include <cstring>

extern "C" {

void rcppml_gpu_bipartition_double(
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *)
{
    Rf_error("GPU support not available. Recompile with CUDA to enable GPU bipartition.");
}

void rcppml_gpu_dclust_double(
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *)
{
    Rf_error("GPU support not available. Recompile with CUDA to enable GPU dclust.");
}

void rcppml_gpu_detect(
    void *count_ptr, void *, void *, void *, void *)
{
    // No GPU available: set count to 0 and return
    if (count_ptr) {
        *((int*)count_ptr) = 0;
    }
}

// NOTE: rcppml_gpu_nmf_double/float stubs removed — these entry points
// are no longer callable from R. The unified path uses Rcpp gateway.

void rcppml_sp_free_gpu(void *, void *, void *, void *)
{
    Rf_error("GPU support not available. Recompile with CUDA.");
}

void rcppml_sp_read_gpu(
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *)
{
    Rf_error("GPU support not available. Recompile with CUDA.");
}

void rcppml_gpu_svd_pca_double(
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *,
    void *)
{
    Rf_error("GPU support not available. Recompile with CUDA to enable GPU SVD/PCA.");
}

}  // extern "C"

#endif  // !FACTORNET_HAS_GPU
