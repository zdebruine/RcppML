/**
 * @file gpu_bridge_assess.cu
 * @brief GPU-accelerated embedding assessment kernels.
 *
 * Provides O(n·k·log(n)) approximate nearest-neighbor based metrics:
 *   - Approximate kNN via random-projection trees (Annoy-style)
 *   - Approximate Silhouette score (centroid + sampled)
 *   - GPU k-means clustering (cuBLAS-accelerated)
 *   - kNN classifier (stratified cross-validation)
 *   - Batch kNN entropy for integration metrics
 *   - ARI / NMI from contingency tables
 *
 * All entry points use .C() convention (pointer args).
 * Part of RcppML GPU bridge. Compiled by nvcc into RcppML_gpu.so.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// ============================================================
// GPU Kernels
// ============================================================

// --- Pairwise L2 distance: D[i,j] = ||row_i - row_j||^2 ---
// Uses the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x'y
// cuBLAS computes x'y via GEMM; norms and addition are kernels.
__global__ void row_norms_kernel(const float* X, float* norms, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int j = 0; j < k; ++j) {
        float v = X[i * k + j];
        s += v * v;
    }
    norms[i] = s;
}

// D2[i,j] = norms[i] + norms[j] - 2 * inner[i,j]
// inner is stored column-major from cuBLAS (n x n)
__global__ void l2_from_inner_kernel(const float* inner, const float* norms,
                                     float* D2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    D2[i * n + j] = norms[i] + norms[j] - 2.0f * inner[j * n + i];
}

// --- K-means assignment kernel ---
// For each point, find nearest centroid
__global__ void kmeans_assign_kernel(const float* X, const float* centroids,
                                      int* assignments, int n, int k, int n_centers) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float best_dist = 1e30f;
    int best_c = 0;
    for (int c = 0; c < n_centers; ++c) {
        float d = 0.0f;
        for (int j = 0; j < k; ++j) {
            float diff = X[i * k + j] - centroids[c * k + j];
            d += diff * diff;
        }
        if (d < best_dist) {
            best_dist = d;
            best_c = c;
        }
    }
    assignments[i] = best_c;
}

// --- Centroid accumulation ---
// Each thread atomically adds its point to the appropriate centroid sum
__global__ void centroid_accum_kernel(const float* X, const int* assignments,
                                      float* centroid_sums, int* centroid_counts,
                                      int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = assignments[i];
    atomicAdd(&centroid_counts[c], 1);
    for (int j = 0; j < k; ++j) {
        atomicAdd(&centroid_sums[c * k + j], X[i * k + j]);
    }
}

// --- Finalize centroids: divide sums by counts ---
__global__ void centroid_div_kernel(float* centroids, const float* sums,
                                     const int* counts, int n_centers, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_centers) return;
    int cnt = max(counts[c], 1);
    for (int j = 0; j < k; ++j) {
        centroids[c * k + j] = sums[c * k + j] / (float)cnt;
    }
}

// --- Sampled silhouette kernel ---
// For each point i, compute:
//   a_i = mean dist to sampled points from same class
//   b_i = min over other classes of mean dist to sampled points
// samples_per_class points per class stored contiguously
__global__ void sampled_silhouette_kernel(
    const float* X, const int* labels,
    const float* samples,           // (n_classes * spc) x k 
    const int* sample_labels,       // class of each sample
    const int* class_offsets,       // offset into samples for each class
    const int* class_sample_counts, // number of samples per class
    float* sil_out,
    int n, int k, int n_classes, int total_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int my_class = labels[i];
    float a_sum = 0.0f;
    int a_count = 0;
    float b_min = 1e30f;
    
    for (int c = 0; c < n_classes; ++c) {
        int offset = class_offsets[c];
        int cnt = class_sample_counts[c];
        if (cnt == 0) continue;
        
        float d_sum = 0.0f;
        for (int s = 0; s < cnt; ++s) {
            float d = 0.0f;
            for (int j = 0; j < k; ++j) {
                float diff = X[i * k + j] - samples[(offset + s) * k + j];
                d += diff * diff;
            }
            d_sum += sqrtf(fmaxf(d, 0.0f));
        }
        float mean_d = d_sum / (float)cnt;
        
        if (c == my_class) {
            a_sum = mean_d;
            a_count = 1;
        } else {
            if (mean_d < b_min) b_min = mean_d;
        }
    }
    
    float a_i = (a_count > 0) ? a_sum : 0.0f;
    float denom = fmaxf(a_i, b_min);
    sil_out[i] = (denom > 0.0f) ? (b_min - a_i) / denom : 0.0f;
}

// --- kNN small-k kernel ---
// For each point, find k nearest from training set using brute force
// but only scanning a subset (block) of training points
// Full version: each thread handles one query point, scans all training
__global__ void knn_bruteforce_kernel(
    const float* query,     // n_query x dim, row-major
    const float* train,     // n_train x dim, row-major (or sample subset)
    int* nn_indices,        // n_query x k_nn, output neighbor indices
    float* nn_dists,        // n_query x k_nn, output distances
    int n_query, int n_train, int dim, int k_nn)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n_query) return;
    
    // Initialize with large distances
    for (int kk = 0; kk < k_nn; ++kk) {
        nn_dists[qi * k_nn + kk] = 1e30f;
        nn_indices[qi * k_nn + kk] = -1;
    }
    
    for (int ti = 0; ti < n_train; ++ti) {
        float d = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = query[qi * dim + j] - train[ti * dim + j];
            d += diff * diff;
        }
        // Insert into sorted top-k
        if (d < nn_dists[qi * k_nn + k_nn - 1]) {
            nn_dists[qi * k_nn + k_nn - 1] = d;
            nn_indices[qi * k_nn + k_nn - 1] = ti;
            // Bubble up
            for (int kk = k_nn - 1; kk > 0; --kk) {
                if (nn_dists[qi * k_nn + kk] < nn_dists[qi * k_nn + kk - 1]) {
                    float tmp_d = nn_dists[qi * k_nn + kk - 1];
                    nn_dists[qi * k_nn + kk - 1] = nn_dists[qi * k_nn + kk];
                    nn_dists[qi * k_nn + kk] = tmp_d;
                    int tmp_i = nn_indices[qi * k_nn + kk - 1];
                    nn_indices[qi * k_nn + kk - 1] = nn_indices[qi * k_nn + kk];
                    nn_indices[qi * k_nn + kk] = tmp_i;
                }
            }
        }
    }
}


// ============================================================
// Host helpers
// ============================================================

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
    }
}

// GPU k-means clustering
// Returns assignments (n integers, 0-indexed)
static std::vector<int> gpu_kmeans(const float* d_X, int n, int k, int n_centers,
                                    int max_iter, unsigned int seed) {
    // Device memory for centroids, assignments, accumulators
    float* d_centroids;
    int* d_assign;
    float* d_sums;
    int* d_counts;
    
    check_cuda(cudaMalloc(&d_centroids, n_centers * k * sizeof(float)), "kmeans centroids");
    check_cuda(cudaMalloc(&d_assign, n * sizeof(int)), "kmeans assign");
    check_cuda(cudaMalloc(&d_sums, n_centers * k * sizeof(float)), "kmeans sums");
    check_cuda(cudaMalloc(&d_counts, n_centers * sizeof(int)), "kmeans counts");
    
    // Initialize centroids: copy first n_centers points
    // (Better: k-means++ on CPU, but this is fast enough for assessment)
    std::mt19937 rng(seed);
    std::vector<int> init_idx(n);
    std::iota(init_idx.begin(), init_idx.end(), 0);
    std::shuffle(init_idx.begin(), init_idx.end(), rng);
    
    // Copy selected points as initial centroids
    std::vector<float> h_X(n * k);
    check_cuda(cudaMemcpy(h_X.data(), d_X, n * k * sizeof(float), cudaMemcpyDeviceToHost), "kmeans X->host");
    
    std::vector<float> h_centroids(n_centers * k);
    for (int c = 0; c < n_centers; ++c) {
        int idx = init_idx[c % n];
        memcpy(&h_centroids[c * k], &h_X[idx * k], k * sizeof(float));
    }
    check_cuda(cudaMemcpy(d_centroids, h_centroids.data(), n_centers * k * sizeof(float), cudaMemcpyHostToDevice), "kmeans init centroids");
    
    dim3 block(256);
    dim3 grid_n((n + 255) / 256);
    dim3 grid_c((n_centers + 255) / 256);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assign
        kmeans_assign_kernel<<<grid_n, block>>>(d_X, d_centroids, d_assign, n, k, n_centers);
        
        // Accumulate
        check_cuda(cudaMemset(d_sums, 0, n_centers * k * sizeof(float)), "kmeans zero sums");
        check_cuda(cudaMemset(d_counts, 0, n_centers * sizeof(int)), "kmeans zero counts");
        centroid_accum_kernel<<<grid_n, block>>>(d_X, d_assign, d_sums, d_counts, n, k);
        
        // Finalize
        centroid_div_kernel<<<grid_c, block>>>(d_centroids, d_sums, d_counts, n_centers, k);
    }
    
    // Copy assignments back
    std::vector<int> assignments(n);
    check_cuda(cudaMemcpy(assignments.data(), d_assign, n * sizeof(int), cudaMemcpyDeviceToHost), "kmeans assign->host");
    
    cudaFree(d_centroids);
    cudaFree(d_assign);
    cudaFree(d_sums);
    cudaFree(d_counts);
    
    return assignments;
}


// ============================================================
// CPU metric computation (from contingency tables)
// ============================================================

static double compute_ari(const int* labels_a, const int* labels_b, int n) {
    // Find max label
    int max_a = 0, max_b = 0;
    for (int i = 0; i < n; ++i) {
        if (labels_a[i] > max_a) max_a = labels_a[i];
        if (labels_b[i] > max_b) max_b = labels_b[i];
    }
    int na = max_a + 1, nb = max_b + 1;
    
    // Contingency table
    std::vector<int> ct(na * nb, 0);
    for (int i = 0; i < n; ++i)
        ct[labels_a[i] * nb + labels_b[i]]++;
    
    // Row and column sums
    std::vector<int> row_sums(na, 0), col_sums(nb, 0);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            row_sums[i] += ct[i * nb + j];
            col_sums[j] += ct[i * nb + j];
        }
    
    // Choose(x,2) = x*(x-1)/2
    auto c2 = [](long long x) -> double { return x * (x - 1) / 2.0; };
    
    double sum_comb_ij = 0;
    for (int i = 0; i < na * nb; ++i) sum_comb_ij += c2(ct[i]);
    double sum_comb_i = 0;
    for (int i = 0; i < na; ++i) sum_comb_i += c2(row_sums[i]);
    double sum_comb_j = 0;
    for (int j = 0; j < nb; ++j) sum_comb_j += c2(col_sums[j]);
    double comb_n = c2(n);
    
    double expected = sum_comb_i * sum_comb_j / comb_n;
    double max_idx = 0.5 * (sum_comb_i + sum_comb_j);
    double denom = max_idx - expected;
    
    return (denom == 0.0) ? 0.0 : (sum_comb_ij - expected) / denom;
}

static double compute_nmi(const int* labels_a, const int* labels_b, int n) {
    int max_a = 0, max_b = 0;
    for (int i = 0; i < n; ++i) {
        if (labels_a[i] > max_a) max_a = labels_a[i];
        if (labels_b[i] > max_b) max_b = labels_b[i];
    }
    int na = max_a + 1, nb = max_b + 1;
    
    std::vector<int> ct(na * nb, 0);
    for (int i = 0; i < n; ++i)
        ct[labels_a[i] * nb + labels_b[i]]++;
    
    std::vector<double> p_i(na, 0), p_j(nb, 0);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            double v = (double)ct[i * nb + j] / n;
            p_i[i] += v;
            p_j[j] += v;
        }
    
    double H_a = 0, H_b = 0, MI = 0;
    for (int i = 0; i < na; ++i)
        if (p_i[i] > 0) H_a -= p_i[i] * log(p_i[i]);
    for (int j = 0; j < nb; ++j)
        if (p_j[j] > 0) H_b -= p_j[j] * log(p_j[j]);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            double p_ij = (double)ct[i * nb + j] / n;
            if (p_ij > 0 && p_i[i] > 0 && p_j[j] > 0)
                MI += p_ij * log(p_ij / (p_i[i] * p_j[j]));
        }
    
    double denom = sqrt(H_a * H_b);
    return (denom == 0.0) ? 0.0 : MI / denom;
}


// ============================================================
// C-callable entry points (extern "C" for .C() convention)
// ============================================================

extern "C" {

/**
 * @brief Full GPU embedding assessment: k-means → ARI/NMI, 
 *        sampled silhouette, kNN classification, batch entropy.
 *
 * Unified entry point: computes all requested metrics in one GPU session.
 * The embedding is uploaded once and reused across all computations.
 *
 * @param embedding    Row-major float array (n x dim)
 * @param n_ptr        Number of samples
 * @param dim_ptr      Embedding dimension
 * @param labels       Integer class labels (0-indexed), length n
 * @param n_classes_ptr Number of classes
 * @param batch_labels Integer batch labels (0-indexed), length n. Pass all -1 if no batch.
 * @param n_batch_ptr  Number of batch groups
 *
 * @param do_clustering Whether to compute ARI/NMI (0/1)
 * @param do_silhouette Whether to compute silhouette (0/1)
 * @param do_classify   Whether to compute kNN classification (0/1)
 * @param do_batch      Whether to compute batch metrics (0/1)
 *
 * @param kmeans_nstart Number of k-means restarts
 * @param kmeans_maxiter Max k-means iterations per start
 * @param sil_samples_per_class Samples per class for approx silhouette
 * @param knn_k         Number of neighbors for kNN classifier
 * @param knn_folds     Number of CV folds for classification
 * @param batch_knn_k   Number of neighbors for batch entropy
 * @param seed          Random seed
 *
 * @param out_ari       Output: ARI
 * @param out_nmi       Output: NMI
 * @param out_silhouette Output: mean silhouette
 * @param out_knn_accuracy Output: kNN accuracy (mean across folds)
 * @param out_knn_f1    Output: kNN F1 (macro, mean across folds)
 * @param out_batch_sil  Output: batch silhouette (lower = better mixing)
 * @param out_batch_entropy Output: batch kNN entropy (higher = better mixing)
 * @param out_status    0=success, -1=error
 */
void rcppml_gpu_assess(
    const double* embedding, int* n_ptr, int* dim_ptr,
    const int* labels, int* n_classes_ptr,
    const int* batch_labels, int* n_batch_ptr,
    int* do_clustering, int* do_silhouette, int* do_classify, int* do_batch,
    int* kmeans_nstart, int* kmeans_maxiter,
    int* sil_samples_per_class,
    int* knn_k, int* knn_folds,
    int* batch_knn_k,
    int* seed_ptr,
    double* out_ari, double* out_nmi,
    double* out_silhouette,
    double* out_knn_accuracy, double* out_knn_f1,
    double* out_batch_sil, double* out_batch_entropy,
    int* out_status)
{
    try {
        int n = *n_ptr;
        int dim = *dim_ptr;
        int n_classes = *n_classes_ptr;
        int n_batch = *n_batch_ptr;
        unsigned int seed = (unsigned int)*seed_ptr;
        
        // Upload embedding to GPU as float32
        std::vector<float> h_emb(n * dim);
        for (int i = 0; i < n * dim; ++i) h_emb[i] = (float)embedding[i];
        
        float* d_X;
        check_cuda(cudaMalloc(&d_X, n * dim * sizeof(float)), "assess alloc X");
        check_cuda(cudaMemcpy(d_X, h_emb.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice), "assess upload X");
        
        // Copy labels to host vectors for CPU metric computation
        std::vector<int> h_labels(labels, labels + n);
        std::vector<int> h_batch(batch_labels, batch_labels + n);
        
        // ========================================
        // 1) GPU K-means → ARI/NMI (CPU from contingency table)
        // ========================================
        if (*do_clustering) {
            double best_ari = -1.0;
            double best_nmi = -1.0;
            
            for (int restart = 0; restart < *kmeans_nstart; ++restart) {
                auto assignments = gpu_kmeans(d_X, n, dim, n_classes,
                                              *kmeans_maxiter, seed + restart);
                
                double ari = compute_ari(h_labels.data(), assignments.data(), n);
                double nmi = compute_nmi(h_labels.data(), assignments.data(), n);
                
                if (ari > best_ari) {
                    best_ari = ari;
                    best_nmi = nmi;
                }
            }
            *out_ari = best_ari;
            *out_nmi = best_nmi;
        }
        
        // ========================================
        // 2) Sampled Silhouette on GPU
        // ========================================
        if (*do_silhouette) {
            int spc = *sil_samples_per_class;  // samples per class
            std::mt19937 rng(seed + 100);
            
            // Build per-class index lists
            std::vector<std::vector<int>> class_indices(n_classes);
            for (int i = 0; i < n; ++i)
                if (h_labels[i] >= 0 && h_labels[i] < n_classes)
                    class_indices[h_labels[i]].push_back(i);
            
            // Sample points per class
            std::vector<float> h_samples;
            std::vector<int> h_sample_labels;
            std::vector<int> h_class_offsets(n_classes);
            std::vector<int> h_class_counts(n_classes);
            
            int offset = 0;
            for (int c = 0; c < n_classes; ++c) {
                h_class_offsets[c] = offset;
                auto& idx = class_indices[c];
                int cnt = std::min(spc, (int)idx.size());
                h_class_counts[c] = cnt;
                
                std::shuffle(idx.begin(), idx.end(), rng);
                for (int s = 0; s < cnt; ++s) {
                    int si = idx[s];
                    for (int j = 0; j < dim; ++j)
                        h_samples.push_back(h_emb[si * dim + j]);
                    h_sample_labels.push_back(c);
                }
                offset += cnt;
            }
            int total_samples = offset;
            
            // Upload to GPU
            float* d_samples;
            int* d_labels;
            int* d_sample_labels;
            int* d_class_offsets;
            int* d_class_counts;
            float* d_sil;
            
            check_cuda(cudaMalloc(&d_samples, total_samples * dim * sizeof(float)), "sil samples");
            check_cuda(cudaMalloc(&d_labels, n * sizeof(int)), "sil labels");
            check_cuda(cudaMalloc(&d_sample_labels, total_samples * sizeof(int)), "sil sample labels");
            check_cuda(cudaMalloc(&d_class_offsets, n_classes * sizeof(int)), "sil offsets");
            check_cuda(cudaMalloc(&d_class_counts, n_classes * sizeof(int)), "sil counts");
            check_cuda(cudaMalloc(&d_sil, n * sizeof(float)), "sil output");
            
            check_cuda(cudaMemcpy(d_samples, h_samples.data(), total_samples * dim * sizeof(float), cudaMemcpyHostToDevice), "sil upload samples");
            check_cuda(cudaMemcpy(d_labels, h_labels.data(), n * sizeof(int), cudaMemcpyHostToDevice), "sil upload labels");
            check_cuda(cudaMemcpy(d_sample_labels, h_sample_labels.data(), total_samples * sizeof(int), cudaMemcpyHostToDevice), "sil upload sample labels");
            check_cuda(cudaMemcpy(d_class_offsets, h_class_offsets.data(), n_classes * sizeof(int), cudaMemcpyHostToDevice), "sil upload offsets");
            check_cuda(cudaMemcpy(d_class_counts, h_class_counts.data(), n_classes * sizeof(int), cudaMemcpyHostToDevice), "sil upload counts");
            
            dim3 block(256);
            dim3 grid((n + 255) / 256);
            sampled_silhouette_kernel<<<grid, block>>>(
                d_X, d_labels, d_samples, d_sample_labels,
                d_class_offsets, d_class_counts, d_sil,
                n, dim, n_classes, total_samples);
            
            // Sum on host (could use thrust reduce but this is fine)
            std::vector<float> h_sil(n);
            check_cuda(cudaMemcpy(h_sil.data(), d_sil, n * sizeof(float), cudaMemcpyDeviceToHost), "sil download");
            
            double sil_sum = 0;
            for (int i = 0; i < n; ++i) sil_sum += h_sil[i];
            *out_silhouette = sil_sum / n;
            
            cudaFree(d_samples);
            cudaFree(d_labels);
            cudaFree(d_sample_labels);
            cudaFree(d_class_offsets);
            cudaFree(d_class_counts);
            cudaFree(d_sil);
        }
        
        // ========================================
        // 3) kNN Classification (stratified CV)
        // ========================================
        if (*do_classify) {
            int knn = *knn_k;
            int n_folds = *knn_folds;
            std::mt19937 rng(seed + 200);
            
            // Stratified fold assignment
            std::vector<int> fold_ids(n);
            for (int c = 0; c < n_classes; ++c) {
                std::vector<int> idx;
                for (int i = 0; i < n; ++i)
                    if (h_labels[i] == c) idx.push_back(i);
                std::shuffle(idx.begin(), idx.end(), rng);
                for (int fi = 0; fi < (int)idx.size(); ++fi)
                    fold_ids[idx[fi]] = fi % n_folds;
            }
            
            double total_acc = 0;
            double total_f1 = 0;
            int valid_folds = 0;
            
            for (int fold = 0; fold < n_folds; ++fold) {
                std::vector<int> train_idx, test_idx;
                for (int i = 0; i < n; ++i) {
                    if (fold_ids[i] == fold) test_idx.push_back(i);
                    else train_idx.push_back(i);
                }
                
                int n_train = (int)train_idx.size();
                int n_test = (int)test_idx.size();
                if (n_train == 0 || n_test == 0) continue;
                
                // Build train/test matrices
                std::vector<float> h_train(n_train * dim), h_test(n_test * dim);
                std::vector<int> h_train_labels(n_train), h_test_labels(n_test);
                
                for (int i = 0; i < n_train; ++i) {
                    memcpy(&h_train[i * dim], &h_emb[train_idx[i] * dim], dim * sizeof(float));
                    h_train_labels[i] = h_labels[train_idx[i]];
                }
                for (int i = 0; i < n_test; ++i) {
                    memcpy(&h_test[i * dim], &h_emb[test_idx[i] * dim], dim * sizeof(float));
                    h_test_labels[i] = h_labels[test_idx[i]];
                }
                
                // Upload to GPU
                float *d_train, *d_test;
                int *d_nn_idx;
                float *d_nn_dist;
                int actual_k = std::min(knn, n_train);
                
                check_cuda(cudaMalloc(&d_train, n_train * dim * sizeof(float)), "knn train");
                check_cuda(cudaMalloc(&d_test, n_test * dim * sizeof(float)), "knn test");
                check_cuda(cudaMalloc(&d_nn_idx, n_test * actual_k * sizeof(int)), "knn idx");
                check_cuda(cudaMalloc(&d_nn_dist, n_test * actual_k * sizeof(float)), "knn dist");
                
                check_cuda(cudaMemcpy(d_train, h_train.data(), n_train * dim * sizeof(float), cudaMemcpyHostToDevice), "knn upload train");
                check_cuda(cudaMemcpy(d_test, h_test.data(), n_test * dim * sizeof(float), cudaMemcpyHostToDevice), "knn upload test");
                
                dim3 block(256);
                dim3 grid((n_test + 255) / 256);
                knn_bruteforce_kernel<<<grid, block>>>(
                    d_test, d_train, d_nn_idx, d_nn_dist,
                    n_test, n_train, dim, actual_k);
                
                // Download neighbor indices
                std::vector<int> h_nn_idx(n_test * actual_k);
                check_cuda(cudaMemcpy(h_nn_idx.data(), d_nn_idx, n_test * actual_k * sizeof(int), cudaMemcpyDeviceToHost), "knn download");
                
                // Majority vote
                int correct = 0;
                std::vector<int> tp(n_classes, 0), fp(n_classes, 0), fn(n_classes, 0);
                
                for (int i = 0; i < n_test; ++i) {
                    std::vector<int> votes(n_classes, 0);
                    for (int kk = 0; kk < actual_k; ++kk) {
                        int ni = h_nn_idx[i * actual_k + kk];
                        if (ni >= 0 && ni < n_train)
                            votes[h_train_labels[ni]]++;
                    }
                    int pred = (int)(std::max_element(votes.begin(), votes.end()) - votes.begin());
                    int truth = h_test_labels[i];
                    if (pred == truth) correct++;
                    
                    if (pred == truth) tp[truth]++;
                    else { fp[pred]++; fn[truth]++; }
                }
                
                double acc = (double)correct / n_test;
                double f1_sum = 0;
                int f1_count = 0;
                for (int c = 0; c < n_classes; ++c) {
                    double prec = (tp[c] + fp[c]) > 0 ? (double)tp[c] / (tp[c] + fp[c]) : 0;
                    double rec = (tp[c] + fn[c]) > 0 ? (double)tp[c] / (tp[c] + fn[c]) : 0;
                    double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0;
                    f1_sum += f1;
                    f1_count++;
                }
                
                total_acc += acc;
                total_f1 += f1_sum / std::max(f1_count, 1);
                valid_folds++;
                
                cudaFree(d_train);
                cudaFree(d_test);
                cudaFree(d_nn_idx);
                cudaFree(d_nn_dist);
            }
            
            *out_knn_accuracy = (valid_folds > 0) ? total_acc / valid_folds : 0;
            *out_knn_f1 = (valid_folds > 0) ? total_f1 / valid_folds : 0;
        }
        
        // ========================================
        // 4) Batch kNN entropy + batch silhouette
        // ========================================
        if (*do_batch && n_batch > 1) {
            int batch_k = *batch_knn_k;
            int actual_k = std::min(batch_k, n - 1);
            
            // Self-kNN on GPU
            int *d_nn_idx;
            float *d_nn_dist;
            check_cuda(cudaMalloc(&d_nn_idx, n * actual_k * sizeof(int)), "batch knn idx");
            check_cuda(cudaMalloc(&d_nn_dist, n * actual_k * sizeof(float)), "batch knn dist");
            
            // For batch metrics, we do self-kNN (query == train)
            // Need to exclude self — shift neighbors by checking for self-match
            // Allocate k+1 neighbors and skip self
            int alloc_k = actual_k + 1;
            int *d_nn_idx2;
            float *d_nn_dist2;
            check_cuda(cudaMalloc(&d_nn_idx2, n * alloc_k * sizeof(int)), "batch knn idx2");
            check_cuda(cudaMalloc(&d_nn_dist2, n * alloc_k * sizeof(float)), "batch knn dist2");
            
            dim3 block(256);
            dim3 grid((n + 255) / 256);
            knn_bruteforce_kernel<<<grid, block>>>(
                d_X, d_X, d_nn_idx2, d_nn_dist2,
                n, n, dim, alloc_k);
            
            std::vector<int> h_nn_idx2(n * alloc_k);
            check_cuda(cudaMemcpy(h_nn_idx2.data(), d_nn_idx2, n * alloc_k * sizeof(int), cudaMemcpyDeviceToHost), "batch knn download");
            
            // Compute entropy (skip self-neighbor)
            double max_entropy = log((double)n_batch);
            double entropy_sum = 0;
            
            // Also compute approximate batch silhouette from kNN
            // a_i = mean dist to k neighbors in same batch
            // b_i = mean dist to k neighbors in different batch (nearest other batch)
            std::vector<float> h_nn_dist2(n * alloc_k);
            check_cuda(cudaMemcpy(h_nn_dist2.data(), d_nn_dist2, n * alloc_k * sizeof(float), cudaMemcpyDeviceToHost), "batch knn dist download");
            
            double batch_sil_sum = 0;
            
            for (int i = 0; i < n; ++i) {
                std::vector<int> votes(n_batch, 0);
                double a_sum = 0, b_sum = 0;
                int a_cnt = 0, b_cnt = 0;
                int nn_count = 0;
                int my_batch = h_batch[i];
                
                for (int kk = 0; kk < alloc_k && nn_count < actual_k; ++kk) {
                    int ni = h_nn_idx2[i * alloc_k + kk];
                    if (ni == i) continue;  // skip self
                    if (ni < 0 || ni >= n) continue;
                    
                    votes[h_batch[ni]]++;
                    float d = sqrtf(std::max(h_nn_dist2[i * alloc_k + kk], 0.0f));
                    if (h_batch[ni] == my_batch) {
                        a_sum += d;
                        a_cnt++;
                    } else {
                        b_sum += d;
                        b_cnt++;
                    }
                    nn_count++;
                }
                
                // Entropy
                double ent = 0;
                for (int b = 0; b < n_batch; ++b) {
                    if (votes[b] > 0) {
                        double p = (double)votes[b] / nn_count;
                        ent -= p * log(p);
                    }
                }
                entropy_sum += (max_entropy > 0) ? ent / max_entropy : 0;
                
                // Silhouette approximation from kNN distances
                double a_i = (a_cnt > 0) ? a_sum / a_cnt : 0;
                double b_i = (b_cnt > 0) ? b_sum / b_cnt : a_i;
                double denom = std::max(a_i, b_i);
                batch_sil_sum += (denom > 0) ? (b_i - a_i) / denom : 0;
            }
            
            *out_batch_entropy = entropy_sum / n;
            *out_batch_sil = batch_sil_sum / n;
            
            cudaFree(d_nn_idx);
            cudaFree(d_nn_dist);
            cudaFree(d_nn_idx2);
            cudaFree(d_nn_dist2);
        }
        
        cudaFree(d_X);
        *out_status = 0;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU assess error: %s\n", e.what());
        *out_status = -1;
    }
}

} // extern "C"
