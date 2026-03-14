// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file dclust.hpp
 * @brief Divisive hierarchical clustering
 * 
 * Implements top-down hierarchical clustering using bipartitioning.
 * Recursively splits clusters until stopping criteria are met.
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

#ifndef FactorNet_CLUSTERING_DCLUST_HPP
#define FactorNet_CLUSTERING_DCLUST_HPP

#include <FactorNet/clustering/bipartition.hpp>
#include <numeric>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FactorNet {
namespace clustering {


/**
 * @brief Cluster result structure
 * 
 * @tparam Scalar Computation precision
 */
template<typename Scalar = double>
struct Cluster {
    std::vector<unsigned int> samples; ///< Sample indices in cluster
    std::vector<Scalar> center;        ///< Cluster centroid
    std::string id;                    ///< Binary path encoding split hierarchy ("0"/"1" sequence)
    unsigned int size;                 ///< Number of samples
    Scalar radius;                     ///< Cluster radius (optional)
};

/**
 * @brief Divisive hierarchical clustering
 * 
 * Recursively partitions data using bipartitioning until:
 * - Clusters are smaller than min_samples
 * - Distance between subclusters is less than min_dist
 * - Maximum depth is reached
 * 
 * @tparam Scalar Computation precision
 * @tparam Mat Matrix type (sparse or dense)
 * 
 * @param A Data matrix (features × samples)
 * @param min_samples Minimum cluster size (stop splitting below this)
 * @param min_dist Minimum distance between subclusters (stop if below this)
 * @param verbose Print progress information
 * @param tol Convergence tolerance for bipartitioning
 * @param maxit Maximum iterations for bipartitioning
 * @param nonneg Apply non-negativity constraint
 * @param seed Random seed
 * @param threads Number of OpenMP threads (0 = auto-detect)
 * 
 * @return Vector of Cluster objects
 */
template<typename Scalar, typename Mat>
inline std::vector<Cluster<Scalar>> dclust(
    const Mat& A,
    const unsigned int min_samples,
    const Scalar min_dist,
    const bool verbose,
    const Scalar tol,
    const unsigned int maxit,
    const bool nonneg,
    const unsigned int seed,
    const unsigned int threads = 0) {
    
    std::vector<Cluster<Scalar>> clusters;
    // Stack tracks (samples, binary_path) pairs
    std::vector<std::pair<std::vector<unsigned int>, std::string>> to_partition;
    
    // Start with all samples, empty path
    std::vector<unsigned int> all_samples(A.cols());
    std::iota(all_samples.begin(), all_samples.end(), 0);
    to_partition.push_back({std::move(all_samples), std::string()});
    
    bool calc_dist = (min_dist > static_cast<Scalar>(0));
    
    while (!to_partition.empty()) {
        auto current = std::move(to_partition.back());
        to_partition.pop_back();
        auto& current_samples = current.first;
        auto& path = current.second;
        
        // Check if cluster is too small to partition
        if (current_samples.size() < min_samples * 2) {
            Cluster<Scalar> c;
            c.samples = std::move(current_samples);
            c.size = c.samples.size();
            c.id = path;
            c.center = compute_centroid<Scalar>(A, c.samples);
            c.radius = 0;
            clusters.push_back(std::move(c));
            continue;
        }
        
        // Try bipartition
        auto model = bipartition(A, tol, maxit, nonneg, current_samples,
                                seed, verbose, calc_dist, true);
        
        // Check if partition is valid
        bool valid_partition = (model.size1 >= min_samples && 
                               model.size2 >= min_samples);
        
        if (valid_partition && calc_dist && model.dist < min_dist) {
            valid_partition = false;
        }
        
        if (valid_partition) {
            // Successful partition - queue both subclusters with extended paths
            to_partition.push_back({std::move(model.samples1), path + "0"});
            to_partition.push_back({std::move(model.samples2), path + "1"});
        } else {
            // Cannot partition further - create cluster
            Cluster<Scalar> c;
            c.samples = std::move(current_samples);
            c.size = c.samples.size();
            c.id = path;
            c.center = compute_centroid<Scalar>(A, c.samples);
            c.radius = calc_dist ? model.dist : static_cast<Scalar>(0);
            clusters.push_back(std::move(c));
        }
    }
    
    return clusters;
}

/**
 * @brief Divisive clustering with OpenMP parallelization (advanced)
 * 
 * This version attempts to partition multiple clusters in parallel.
 * Note: This is more complex and may not always be faster for small datasets.
 * 
 * @tparam Scalar Computation precision
 * @tparam Mat Matrix type (sparse or dense)
 */
template<typename Scalar, typename Mat>
inline std::vector<Cluster<Scalar>> dclust_parallel(
    const Mat& A,
    const unsigned int min_samples,
    const Scalar min_dist,
    const bool verbose,
    const Scalar tol,
    const unsigned int maxit,
    const bool nonneg,
    const unsigned int seed,
    const unsigned int threads = 0) {
    
    std::vector<Cluster<Scalar>> clusters;
    // Level tracks (samples, binary_path) pairs 
    std::vector<std::pair<std::vector<unsigned int>, std::string>> current_level;
    
    // Start with all samples, empty path
    std::vector<unsigned int> all_samples(A.cols());
    std::iota(all_samples.begin(), all_samples.end(), 0);
    current_level.push_back({std::move(all_samples), std::string()});
    
    bool calc_dist = (min_dist > static_cast<Scalar>(0));
    
    // Process clusters level by level
    while (!current_level.empty()) {
        std::vector<std::pair<std::vector<unsigned int>, std::string>> next_level;
        std::vector<Cluster<Scalar>> level_clusters(current_level.size());
        std::vector<bool> should_split(current_level.size(), false);
        
#ifdef _OPENMP
        // Determine number of threads
        int num_threads = threads > 0 ? threads : omp_get_max_threads();
        
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
#endif
        for (size_t i = 0; i < current_level.size(); ++i) {
            const auto& current_samples = current_level[i].first;
            
            // Check if too small to partition
            if (current_samples.size() < min_samples * 2) {
                level_clusters[i].samples = current_samples;
                level_clusters[i].size = current_samples.size();
                level_clusters[i].center = compute_centroid<Scalar>(A, current_samples);
                level_clusters[i].radius = 0;
                should_split[i] = false;
                continue;
            }
            
            // Try bipartition
            auto model = bipartition(A, tol, maxit, nonneg, current_samples,
                                    seed + i + 1000, verbose, calc_dist, true);
            
            // Check if partition is valid
            bool valid_partition = (model.size1 >= min_samples && 
                                   model.size2 >= min_samples);
            
            if (valid_partition && calc_dist && model.dist < min_dist) {
                valid_partition = false;
            }
            
            if (valid_partition) {
                should_split[i] = true;
                level_clusters[i].samples = model.samples1;  // Temp storage
                level_clusters[i].center = model.center1;    // Store both for next level
                level_clusters[i].radius = model.dist;
                // Store samples2 in size field temporarily
                level_clusters[i].size = i;  // Index marker
            } else {
                level_clusters[i].samples = current_samples;
                level_clusters[i].size = current_samples.size();
                level_clusters[i].center = compute_centroid<Scalar>(A, current_samples);
                level_clusters[i].radius = calc_dist ? model.dist : static_cast<Scalar>(0);
                should_split[i] = false;
            }
        }
        
        // Process results (single-threaded for path assignment)
        for (size_t i = 0; i < current_level.size(); ++i) {
            const auto& path = current_level[i].second;
            if (should_split[i]) {
                // Re-run bipartition to get both partitions (this is inefficient but safe)
                auto model = bipartition(A, tol, maxit, nonneg, current_level[i].first,
                                        seed + i, false, calc_dist, true);
                next_level.push_back({std::move(model.samples1), path + "0"});
                next_level.push_back({std::move(model.samples2), path + "1"});
            } else {
                level_clusters[i].id = path;
                clusters.push_back(std::move(level_clusters[i]));
            }
        }
        
        current_level = std::move(next_level);
    }
    
    return clusters;
}

}  // namespace clustering

}  // namespace FactorNet

#endif // FactorNet_CLUSTERING_DCLUST_HPP
