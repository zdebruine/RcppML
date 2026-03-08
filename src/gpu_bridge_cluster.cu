/**
 * @file gpu_bridge_cluster.cu
 * @brief Cluster/detection GPU entry points (bipartition, dclust, GPU detect)
 *
 * Part of RcppML GPU bridge. Compiled by nvcc into RcppML_gpu.so.
 * Extracted from gpu_bridge.cu (2026).
 *
 * Build: make -f Makefile.gpu
 */

#include "gpu_bridge_common.cuh"

extern "C" {

/**
 * @brief Query GPU availability and info.
 *
 * @param num_gpus     Output: number of suitable GPUs found
 * @param total_mem_mb Output: array of total memory per GPU in MB (pre-allocated)
 * @param free_mem_mb  Output: array of free memory per GPU in MB (pre-allocated)
 * @param max_gpus     Pointer to max entries in output arrays
 * @param out_status   Output: 0 on success, -1 if no GPUs
 */
void rcppml_gpu_detect(
    int* num_gpus,
    double* total_mem_mb,
    double* free_mem_mb,
    int* max_gpus,
    int* out_status)
{
    auto gpus = detect_gpus();
    *num_gpus = static_cast<int>(gpus.size());
    
    if (gpus.empty()) {
        *out_status = -1;
        return;
    }
    
    int count = std::min(*num_gpus, *max_gpus);
    for (int i = 0; i < count; ++i) {
        total_mem_mb[i] = gpus[i].total_memory / (1024.0 * 1024.0);
        free_mem_mb[i] = gpus[i].free_memory / (1024.0 * 1024.0);
    }
    
    *out_status = 0;
}

// NOTE: rcppml_gpu_nmf_double/float entry points REMOVED.
// These were the legacy NMF path (L1/L2 only, via nmf_gpu_dispatch).
// All NMF now goes through the unified path: gateway::nmf() → fit_gpu_unified.cuh.
// The .gpu_nmf_sparse() R function that called these was also deleted.

/**
 * @brief Run GPU bipartition (rank-2 NMF + partition).
 * All scalars are pointers per .C() convention.
 */
void rcppml_gpu_bipartition_double(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz,
    int* max_iter, double* tol, int* nonneg, double* seed,
    int* partition, double* v, double* center, double* dist,
    int* out_status)
{
    try {
        using namespace FactorNet::gpu;
        GPUContext ctx;
        
        // Build all-samples vector {0, 1, ..., n-1}
        std::vector<unsigned int> samples(*n);
        std::iota(samples.begin(), samples.end(), 0u);
        
        bool calc_dist = true;
        GPUBipartitionResult result = bipartition_gpu<double>(
            ctx, col_ptr, row_idx, values,
            *m, *n, samples,
            *tol, static_cast<unsigned int>(*max_iter),
            (*nonneg != 0),
            static_cast<unsigned int>(*seed),
            calc_dist, false);
        
        // Unpack results into output arrays
        *dist = result.dist;
        for (int i = 0; i < *n; ++i) partition[i] = 0;
        for (unsigned int s : result.samples1) {
            if (s < static_cast<unsigned int>(*n)) partition[s] = 0;
        }
        for (unsigned int s : result.samples2) {
            if (s < static_cast<unsigned int>(*n)) partition[s] = 1;
        }
        int v_len = static_cast<int>(result.v.size());
        for (int i = 0; i < v_len && i < *m; ++i) {
            v[i] = result.v[i];
        }
        *out_status = 0;
    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU bipartition error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Run GPU divisive clustering.
 * All scalars are pointers per .C() convention.
 */
void rcppml_gpu_dclust_double(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz,
    int* max_clusters, int* min_samples, double* min_dist,
    int* max_iter, double* tol, int* nonneg, double* seed,
    int* assignments, int* out_num_clusters, int* out_status)
{
    try {
        using namespace FactorNet::gpu;
        GPUContext ctx;
        
        std::vector<GPUCluster<double>> clusters = dclust_gpu<double>(
            ctx, col_ptr, row_idx, values,
            *m, *n,
            static_cast<unsigned int>(*min_samples),
            *min_dist, *tol,
            static_cast<unsigned int>(*max_iter),
            (*nonneg != 0),
            static_cast<unsigned int>(*seed),
            false);
        
        int nc = static_cast<int>(clusters.size());
        // Limit to max_clusters if specified
        if (*max_clusters > 0 && nc > *max_clusters) nc = *max_clusters;
        
        // Fill assignments array: sample i → cluster id
        for (int i = 0; i < *n; ++i) assignments[i] = -1;
        for (int c = 0; c < nc; ++c) {
            for (unsigned int s : clusters[c].samples) {
                if (s < static_cast<unsigned int>(*n)) {
                    assignments[s] = static_cast<int>(clusters[c].id);
                }
            }
        }
        *out_num_clusters = nc;
        *out_status = 0;
    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU dclust error: %s\n", e.what());
        *out_status = -1;
    }
}

} // extern "C" (legacy entry points)
