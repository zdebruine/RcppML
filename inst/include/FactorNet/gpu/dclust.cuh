/**
 * @file dclust.cuh
 * @brief GPU-accelerated divisive hierarchical clustering.
 *
 * Implements the same iterative stack-based algorithm as the CPU dclust,
 * but delegates each bipartition to the GPU via gpu_bipartition.cuh.
 *
 * For large datasets (e.g., millions of cells), each bipartition is
 * expensive enough to warrant GPU acceleration. The dclust driver
 * itself runs on the host (managing the partition queue), calling
 * bipartition_gpu for each split.
 *
 * Algorithm:
 *   1. Start with all samples in one cluster
 *   2. While partition queue is non-empty:
 *      a. Pop a cluster from the queue
 *      b. If too small (< 2×min_samples), finalize as leaf
 *      c. Call bipartition_gpu → get samples1, samples2, dist
 *      d. If partition valid (sizes ≥ min_samples, dist ≥ min_dist):
 *         push samples1 and samples2 back onto queue
 *      e. Else: finalize as leaf
 *
 * Future optimization: for deeper tree levels where many small clusters
 * exist, batch multiple bipartitions into a single GPU call. This is
 * deferred to a later pass since the first few splits dominate runtime
 * (halving a billion-cell dataset is the expensive part).
 */

#pragma once

#include "bipartition.cuh"
#include <vector>
#include <numeric>
#include <cstdio>
#include <FactorNet/core/logging.hpp>

namespace FactorNet {
namespace gpu {

// ==========================================================================
// GPU dclust result (matches CPU Cluster<Scalar>)
// ==========================================================================

template<typename Scalar>
struct GPUCluster {
    std::vector<unsigned int> samples;  // Sample indices in cluster (0-based)
    std::vector<Scalar> center;         // Cluster centroid
    unsigned int id;                    // Cluster identifier
    unsigned int size;                  // Number of samples
    Scalar radius;                      // Cluster radius (optional, cosine dist)
};

// ==========================================================================
// Main GPU dclust function
// ==========================================================================

/**
 * @brief GPU-accelerated divisive hierarchical clustering.
 *
 * @param ctx           GPU context
 * @param host_col_ptr  Host CSC column pointers for matrix A
 * @param host_row_idx  Host CSC row indices
 * @param host_values   Host CSC values
 * @param m             Number of rows (features)
 * @param n             Number of columns (samples)
 * @param min_samples   Minimum cluster size before stopping
 * @param min_dist      Minimum cosine distance for valid partition (0 = disabled)
 * @param tol           Convergence tolerance for rank-2 NMF
 * @param maxit         Maximum ALS iterations per bipartition
 * @param nonneg        Enforce non-negativity
 * @param seed          Random seed
 * @param verbose       Print progress
 *
 * @return Vector of GPUCluster objects
 */
template<typename Scalar>
std::vector<GPUCluster<Scalar>> dclust_gpu(
    GPUContext& ctx,
    const int* host_col_ptr,
    const int* host_row_idx,
    const Scalar* host_values,
    int m,
    int n,
    unsigned int min_samples,
    double min_dist = 0.0,
    double tol = 1e-5,
    unsigned int maxit = 100,
    bool nonneg = true,
    unsigned int seed = 0,
    bool verbose = false)
{
    std::vector<GPUCluster<Scalar>> clusters;
    std::vector<std::vector<unsigned int>> to_partition;

    // Start with all samples
    std::vector<unsigned int> all_samples(n);
    std::iota(all_samples.begin(), all_samples.end(), 0);
    to_partition.push_back(std::move(all_samples));

    unsigned int cluster_id = 0;
    bool calc_dist = (min_dist > 0.0);
    int split_count = 0;

    while (!to_partition.empty()) {
        std::vector<unsigned int> current_samples = std::move(to_partition.back());
        to_partition.pop_back();

        // Check if cluster is too small to partition
        if (current_samples.size() < static_cast<size_t>(min_samples * 2)) {
            GPUCluster<Scalar> c;
            c.samples = std::move(current_samples);
            c.size = static_cast<unsigned int>(c.samples.size());
            c.id = cluster_id++;
            c.center = compute_centroid_host(
                host_col_ptr, host_row_idx, host_values, m, c.samples);
            c.radius = 0;
            clusters.push_back(std::move(c));
            continue;
        }

        // Try bipartition on GPU
        if (verbose) {
            FACTORNET_GPU_LOG(verbose, "dclust: splitting cluster with %zu samples (split #%d)\n",
                   current_samples.size(), split_count + 1);
        }

        auto model = bipartition_gpu<Scalar>(
            ctx,
            host_col_ptr, host_row_idx, host_values,
            m, n,
            current_samples,
            tol, maxit, nonneg,
            seed + split_count,
            calc_dist,
            verbose);

        split_count++;

        // Check if partition is valid
        bool valid_partition = (model.size1 >= min_samples &&
                                model.size2 >= min_samples);

        if (valid_partition && calc_dist && model.dist >= 0 &&
            model.dist < min_dist) {
            valid_partition = false;
        }

        if (valid_partition) {
            // Successful partition — queue both subclusters
            to_partition.push_back(std::move(model.samples1));
            to_partition.push_back(std::move(model.samples2));

            if (verbose) {
                FACTORNET_GPU_LOG(verbose, "  → split into %u + %u samples",
                       model.size1, model.size2);
                if (calc_dist) {
                    FACTORNET_GPU_LOG(verbose, " (dist = %.4f)", model.dist);
                }
                FACTORNET_GPU_LOG(verbose, "\n");
            }
        } else {
            // Cannot partition further — create leaf cluster
            GPUCluster<Scalar> c;
            c.samples = std::move(current_samples);
            c.size = static_cast<unsigned int>(c.samples.size());
            c.id = cluster_id++;

            if (calc_dist && !model.center1.empty()) {
                // Recompute centroid for the whole (un-split) cluster
                c.center = compute_centroid_host(
                    host_col_ptr, host_row_idx, host_values, m, c.samples);
                c.radius = static_cast<Scalar>(model.dist);
            } else {
                c.center = compute_centroid_host(
                    host_col_ptr, host_row_idx, host_values, m, c.samples);
                c.radius = 0;
            }

            clusters.push_back(std::move(c));

            if (verbose) {
                FACTORNET_GPU_LOG(verbose, "  → leaf cluster #%u (%u samples)\n",
                       c.id, c.size);
            }
        }
    }

    FACTORNET_GPU_LOG(verbose, "dclust complete: %zu clusters, %d total splits\n",
               clusters.size(), split_count);

    return clusters;
}

// ==========================================================================
// Level-parallel GPU dclust (advanced: multiple bipartitions per level)
// ==========================================================================

/**
 * @brief Level-parallel dclust using CUDA streams.
 *
 * For tree levels with multiple pending bipartitions, runs them
 * concurrently using separate CUDA streams. This is beneficial when
 * individual clusters are small enough that a single GPU bipartition
 * doesn't saturate the GPU.
 *
 * Heuristic:
 *   - If any pending cluster has > 100K samples, run serially
 *     (one bipartition saturates the GPU)
 *   - Otherwise, batch up to MAX_CONCURRENT streams
 *
 * @note Requires multiple GPUContext objects (one per stream).
 *       For the initial version, we use sequential processing.
 *       Stream-parallel version is a future optimization.
 */
template<typename Scalar>
std::vector<GPUCluster<Scalar>> dclust_level_parallel_gpu(
    GPUContext& ctx,
    const int* host_col_ptr,
    const int* host_row_idx,
    const Scalar* host_values,
    int m,
    int n,
    unsigned int min_samples,
    double min_dist = 0.0,
    double tol = 1e-5,
    unsigned int maxit = 100,
    bool nonneg = true,
    unsigned int seed = 0,
    bool verbose = false)
{
    static constexpr int MAX_CONCURRENT = 4;

    std::vector<GPUCluster<Scalar>> clusters;
    std::vector<std::vector<unsigned int>> current_level;

    std::vector<unsigned int> all_samples(n);
    std::iota(all_samples.begin(), all_samples.end(), 0);
    current_level.push_back(std::move(all_samples));

    unsigned int cluster_id = 0;
    bool calc_dist = (min_dist > 0.0);
    int level = 0;

    while (!current_level.empty()) {
        std::vector<std::vector<unsigned int>> next_level;

        FACTORNET_GPU_LOG(verbose, "dclust level %d: %zu pending clusters\n",
                   level, current_level.size());

        // Check if any cluster is large enough to saturate GPU alone
        size_t max_size = 0;
        for (const auto& cs : current_level) {
            max_size = std::max(max_size, cs.size());
        }

        bool use_serial = (max_size > 100000) ||
                          (current_level.size() <= 1);

        if (use_serial) {
            // Process sequentially (each bipartition saturates GPU)
            for (size_t i = 0; i < current_level.size(); ++i) {
                auto& current_samples = current_level[i];

                if (current_samples.size() < static_cast<size_t>(min_samples * 2)) {
                    GPUCluster<Scalar> c;
                    c.samples = std::move(current_samples);
                    c.size = static_cast<unsigned int>(c.samples.size());
                    c.id = cluster_id++;
                    c.center = compute_centroid_host(
                        host_col_ptr, host_row_idx, host_values, m, c.samples);
                    c.radius = 0;
                    clusters.push_back(std::move(c));
                    continue;
                }

                auto model = bipartition_gpu<Scalar>(
                    ctx, host_col_ptr, host_row_idx, host_values,
                    m, n, current_samples,
                    tol, maxit, nonneg, seed + level * 1000 + i,
                    calc_dist, false);

                bool valid = (model.size1 >= min_samples &&
                              model.size2 >= min_samples);
                if (valid && calc_dist && model.dist >= 0 &&
                    model.dist < min_dist) {
                    valid = false;
                }

                if (valid) {
                    next_level.push_back(std::move(model.samples1));
                    next_level.push_back(std::move(model.samples2));
                } else {
                    GPUCluster<Scalar> c;
                    c.samples = std::move(current_samples);
                    c.size = static_cast<unsigned int>(c.samples.size());
                    c.id = cluster_id++;
                    c.center = compute_centroid_host(
                        host_col_ptr, host_row_idx, host_values, m, c.samples);
                    c.radius = calc_dist ? static_cast<Scalar>(model.dist) : 0;
                    clusters.push_back(std::move(c));
                }
            }
        } else {
            // Small clusters — process multiple concurrently
            // For now, process sequentially but in batches
            // (multi-stream optimization deferred)
            for (size_t i = 0; i < current_level.size(); ++i) {
                auto& current_samples = current_level[i];

                if (current_samples.size() < static_cast<size_t>(min_samples * 2)) {
                    GPUCluster<Scalar> c;
                    c.samples = std::move(current_samples);
                    c.size = static_cast<unsigned int>(c.samples.size());
                    c.id = cluster_id++;
                    c.center = compute_centroid_host(
                        host_col_ptr, host_row_idx, host_values, m, c.samples);
                    c.radius = 0;
                    clusters.push_back(std::move(c));
                    continue;
                }

                auto model = bipartition_gpu<Scalar>(
                    ctx, host_col_ptr, host_row_idx, host_values,
                    m, n, current_samples,
                    tol, maxit, nonneg, seed + level * 1000 + i,
                    calc_dist, false);

                bool valid = (model.size1 >= min_samples &&
                              model.size2 >= min_samples);
                if (valid && calc_dist && model.dist >= 0 &&
                    model.dist < min_dist) {
                    valid = false;
                }

                if (valid) {
                    next_level.push_back(std::move(model.samples1));
                    next_level.push_back(std::move(model.samples2));
                } else {
                    GPUCluster<Scalar> c;
                    c.samples = std::move(current_samples);
                    c.size = static_cast<unsigned int>(c.samples.size());
                    c.id = cluster_id++;
                    c.center = compute_centroid_host(
                        host_col_ptr, host_row_idx, host_values, m, c.samples);
                    c.radius = calc_dist ? static_cast<Scalar>(model.dist) : 0;
                    clusters.push_back(std::move(c));
                }
            }
        }

        current_level = std::move(next_level);
        level++;
    }

    FACTORNET_GPU_LOG(verbose, "dclust complete: %zu clusters across %d levels\n",
               clusters.size(), level);

    return clusters;
}

} // namespace gpu
} // namespace FactorNet
