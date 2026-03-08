// RcppFunctions_utils.cpp
// Lightweight utility exports: prediction, loss evaluation, bipartition,
// divisive clustering, NNLS solvers, RNG helpers, bipartite matching,
// and KNN/Jaccard connectivity.
// No NMF or SVD algorithm headers — keeps this TU fast to compile.

#include "RcppFunctions_common.h"
#include "../inst/include/FactorNet/clustering/gateway.hpp"
#include "../inst/include/FactorNet/primitives/cpu/nnls_batch.hpp"
#include "../inst/include/FactorNet/primitives/cpu/gram.hpp"
#include "../inst/include/FactorNet/primitives/cpu/rhs.hpp"
#include "../inst/include/FactorNet/nmf/nnls_streaming.hpp"


using namespace FactorNet::clustering;
using FactorNet::tiny_num;

// ============================================================================
// PREDICTION (PROJECT NEW SAMPLES)
// ============================================================================

// [[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict(SEXP A_sexp, const Rcpp::S4& mask, Eigen::MatrixXd w,
                             const double L1, const double L2, const unsigned int threads,
                             const bool mask_zeros, const double upper_bound = 0) {
    FactorNet::clustering::ThreadGuard guard(static_cast<int>(threads));
    // R predict method passes w as k×m (already transposed)
    const int k = static_cast<int>(w.rows());
    
    // Compute Gram matrix (shared for sparse and dense)
    Eigen::MatrixXd G(k, k);
    FactorNet::primitives::gram<FactorNet::primitives::CPU, double>(w, G);
    G.diagonal().array() += tiny_num<double>();
    if (L2 > 0) G.diagonal().array() += L2;
    
    // Compute RHS — dispatch on sparse vs dense
    Eigen::MatrixXd B;
    if (Rf_isS4(A_sexp)) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        Eigen::SparseMatrix<double> A_copy(A_mapped);
        B.resize(k, A_copy.cols());
        FactorNet::primitives::rhs<FactorNet::primitives::CPU, double>(A_copy, w, B);
    } else {
        Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        B.resize(k, A_dense.cols());
        FactorNet::primitives::rhs<FactorNet::primitives::CPU, double>(A_dense, w, B);
    }
    
    Eigen::MatrixXd h(k, B.cols());
    FactorNet::primitives::nnls_batch<FactorNet::primitives::CPU, double>(
        G, B, h, 100, 1e-8, L1, 0.0, true, static_cast<int>(threads), upper_bound);
    return h;
}

// ============================================================================
// MSE LOSS CALCULATION
// ============================================================================

template<typename MatrixType>
double compute_mse(const MatrixType& A, const Eigen::MatrixXd& W, 
                   const Eigen::VectorXd& d, const Eigen::MatrixXd& H,
                   bool mask_zeros = false) {
    using namespace FactorNet;
    // Reconstruct: A_hat = W * diag(d) * H
    Eigen::MatrixXd WH = W * d.asDiagonal() * H;
    
    if constexpr (FactorNet::is_sparse_v<MatrixType>) {
        if (mask_zeros) {
            // Only compute MSE over nonzero entries
            double mse = 0;
            int nnz = 0;
            for (int k = 0; k < A.outerSize(); ++k) {
                for (typename MatrixType::InnerIterator it(A, k); it; ++it) {
                    double diff = it.value() - WH(it.row(), it.col());
                    mse += diff * diff;
                    nnz++;
                }
            }
            return nnz > 0 ? mse / nnz : 0;
        } else {
            // Compute MSE over ALL entries (dense evaluation)
            Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
            Eigen::MatrixXd diff = A_dense - WH;
            return diff.array().square().mean();
        }
    } else {
        Eigen::MatrixXd diff = A - WH;
        return diff.array().square().mean();
    }
}

// ============================================================================
// GENERAL LOSS CALCULATION (MSE, MAE, HUBER, KL)
// ============================================================================

template<typename MatrixType>
double compute_loss_general(const MatrixType& A, const Eigen::MatrixXd& W, 
                           const Eigen::VectorXd& d, const Eigen::MatrixXd& H,
                           int loss_type, double huber_delta,
                           bool mask_zeros = false) {
    using namespace FactorNet;
    
    // Set up loss configuration
    FactorNet::LossConfig<double> loss_config;
    loss_config.type = static_cast<LossType>(loss_type);
    loss_config.huber_delta = huber_delta;
    
    // Reconstruct: A_hat = W * diag(d) * H
    Eigen::MatrixXd WH = W * d.asDiagonal() * H;
    
    double total_loss = 0;
    int count = 0;
    
    if constexpr (FactorNet::is_sparse_v<MatrixType>) {
        if (mask_zeros) {
            // Only compute loss over nonzero entries in A
            for (int k = 0; k < A.outerSize(); ++k) {
                for (typename MatrixType::InnerIterator it(A, k); it; ++it) {
                    double observed = it.value();
                    double predicted = WH(it.row(), it.col());
                    total_loss += compute_loss(observed, predicted, loss_config);
                    count++;
                }
            }
        } else {
            // Compute loss over ALL entries
            for (int i = 0; i < A.rows(); ++i) {
                for (int j = 0; j < A.cols(); ++j) {
                    double observed = A.coeff(i, j);
                    double predicted = WH(i, j);
                    total_loss += compute_loss(observed, predicted, loss_config);
                    count++;
                }
            }
        }
    } else {
        // For dense matrices, iterate over all entries
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                double observed = A(i, j);
                double predicted = WH(i, j);
                total_loss += compute_loss(observed, predicted, loss_config);
                count++;
            }
        }
    }
    
    return count > 0 ? total_loss / count : 0;
}

// [[Rcpp::export]]
double Rcpp_evaluate_loss(SEXP A_sexp, const Rcpp::S4& mask, 
                          Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                          int loss_type, double huber_delta,
                          const unsigned int threads, const bool mask_zeros) {
    if (Rf_isS4(A_sexp)) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        return compute_loss_general(A_mapped, w, d, h, loss_type, huber_delta, mask_zeros);
    } else {
        Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        return compute_loss_general(A_dense, w, d, h, loss_type, huber_delta, mask_zeros);
    }
}

// [[Rcpp::export]]
double Rcpp_evaluate_loss_missing(SEXP A_sexp, const Rcpp::S4& mask,
                                  Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                                  int loss_type, double huber_delta,
                                  const unsigned int threads) {
    using namespace FactorNet;
    auto mask_mapped = mapSparseMatrix(mask);
    
    // Set up loss configuration
    FactorNet::LossConfig<double> loss_config;
    loss_config.type = static_cast<LossType>(loss_type);
    loss_config.huber_delta = huber_delta;
    
    // Reconstruct WH (shared for both paths)
    Eigen::MatrixXd WH = w * d.asDiagonal() * h;
    double total_loss = 0;
    int count = 0;
    
    if (Rf_isS4(A_sexp)) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        for (int k = 0; k < mask_mapped.outerSize(); ++k) {
            for (MappedSpMat::InnerIterator it(mask_mapped, k); it; ++it) {
                if (it.value() > 0) {
                    double observed = A_mapped.coeff(it.row(), it.col());
                    double predicted = WH(it.row(), it.col());
                    total_loss += compute_loss(observed, predicted, loss_config);
                    count++;
                }
            }
        }
    } else {
        Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        for (int k = 0; k < mask_mapped.outerSize(); ++k) {
            for (MappedSpMat::InnerIterator it(mask_mapped, k); it; ++it) {
                if (it.value() > 0) {
                    double observed = A_dense(it.row(), it.col());
                    double predicted = WH(it.row(), it.col());
                    total_loss += compute_loss(observed, predicted, loss_config);
                    count++;
                }
            }
        }
    }
    
    return count > 0 ? total_loss / count : 0;
}

// ============================================================================
// BIPARTITION
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_bipartition(SEXP A_sexp, const double tol, const unsigned int maxit,
                            const bool nonneg, const std::vector<unsigned int>& samples,
                            const unsigned int seed, const bool verbose = false,
                            const bool calc_dist = false, const bool diag = true,
                            const int threads = 0) {
    // Use lambda to capture result from either branch
    auto make_model = [&](auto& A) {
        return FactorNet::clustering::bipartition_gateway(
            A, tol, maxit, nonneg, samples, seed,
            verbose, calc_dist, diag, threads);
    };
    
    auto to_list = [](const auto& model) {
        return Rcpp::List::create(
            Rcpp::Named("v") = model.v,
            Rcpp::Named("dist") = model.dist,
            Rcpp::Named("size1") = model.size1,
            Rcpp::Named("size2") = model.size2,
            Rcpp::Named("samples1") = model.samples1,
            Rcpp::Named("samples2") = model.samples2,
            Rcpp::Named("center1") = model.center1,
            Rcpp::Named("center2") = model.center2
        );
    };
    
    if (Rf_isS4(A_sexp)) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        auto model = make_model(A_mapped);
        return to_list(model);
    } else {
        Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        auto model = make_model(A_dense);
        return to_list(model);
    }
}

// ============================================================================
// DIVISIVE CLUSTERING
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_dclust_sparse(const Rcpp::S4& A, const unsigned int min_samples,
                              const double min_dist, const bool verbose, const double tol,
                              const unsigned int maxit, const bool nonneg,
                              const unsigned int seed, const unsigned int threads) {
    
    auto A_mapped = mapSparseMatrix(A);
    
    // Route through gateway for RAII thread management
    auto clusters = FactorNet::clustering::dclust_gateway<double>(
        A_mapped, min_samples, min_dist, verbose,
        tol, maxit, nonneg, seed, threads);
    
    // Convert to R list
    Rcpp::List result(clusters.size());
    for (size_t i = 0; i < clusters.size(); ++i) {
        result[i] = Rcpp::List::create(
            Rcpp::Named("samples") = clusters[i].samples,
            Rcpp::Named("center") = clusters[i].center,
            Rcpp::Named("id") = clusters[i].id,
            Rcpp::Named("size") = clusters[i].size
        );
    }
    
    return result;
}

// ============================================================================
// NNLS SOLVERS
// ============================================================================

// [[Rcpp::export]]
Eigen::MatrixXd c_solve(Eigen::MatrixXd a, Eigen::MatrixXd b,
                        unsigned int cd_maxit = 100, const double cd_tol = 1e-8,
                        const double L1 = 0, const double L2 = 0,
                        const double upper_bound = 0, const bool nonneg = true) {
    
    // Apply L2 externally to Gram diagonal (ridge penalty)
    // L1 is passed internally to cd_nnls_col_fixed for proper iterative sparsity
    a.diagonal().array() += tiny_num<double>();
    if (L2 > 0) a.diagonal().array() += L2;
    
    const int k = static_cast<int>(b.rows());
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(k, b.cols());
    for (int col = 0; col < b.cols(); ++col) {
        Eigen::VectorXd b_col = b.col(col);
        Eigen::VectorXd x_col = x.col(col);
        FactorNet::primitives::detail::cd_nnls_col_fixed(
            a, b_col.data(), x_col.data(), k,
            L1, 0.0, nonneg, static_cast<int>(cd_maxit), upper_bound);
        x.col(col) = x_col;
    }
    
    return x;
}

// [[Rcpp::export]]
Eigen::MatrixXd c_nnls(const Eigen::MatrixXd& w, SEXP A_sexp,
                       unsigned int cd_maxit = 100, const double cd_tol = 1e-8,
                       const double L1 = 0, const double L2 = 0,
                       const double upper_bound = 0, const bool nonneg = true,
                       const unsigned int threads = 0,
                       Rcpp::Nullable<Eigen::MatrixXd> warm_start = R_NilValue) {
    
    FactorNet::clustering::ThreadGuard guard(static_cast<int>(threads));
    const int k = static_cast<int>(w.cols());
    Eigen::MatrixXd w_T = w.transpose();
    
    // Compute Gram matrix (shared for sparse and dense)
    Eigen::MatrixXd G(k, k);
    FactorNet::primitives::gram<FactorNet::primitives::CPU, double>(w_T, G);
    G.diagonal().array() += tiny_num<double>();
    if (L2 > 0) G.diagonal().array() += L2;
    
    // Compute RHS — dispatch on sparse vs dense
    Eigen::MatrixXd B;
    int n;
    if (Rf_isS4(A_sexp)) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        Eigen::SparseMatrix<double> A_copy(A_mapped);
        n = static_cast<int>(A_copy.cols());
        B.resize(k, n);
        FactorNet::primitives::rhs<FactorNet::primitives::CPU, double>(A_copy, w_T, B);
    } else {
        Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        n = static_cast<int>(A_dense.cols());
        B.resize(k, n);
        FactorNet::primitives::rhs<FactorNet::primitives::CPU, double>(A_dense, w_T, B);
    }
    
    Eigen::MatrixXd h;
    if (warm_start.isNotNull()) {
        h = Rcpp::as<Eigen::MatrixXd>(warm_start);
        B.noalias() -= G * h;
        #ifdef _OPENMP
        int n_threads = (static_cast<int>(threads) > 0) ? static_cast<int>(threads) : omp_get_max_threads();
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        #endif
        for (int j = 0; j < n; ++j) {
            FactorNet::primitives::detail::cd_nnls_col_fixed(
                G, B.col(j).data(), h.col(j).data(), k,
                L1, 0.0, nonneg, static_cast<int>(cd_maxit), upper_bound);
        }
    } else {
        FactorNet::primitives::nnls_batch<FactorNet::primitives::CPU, double>(
            G, B, h, static_cast<int>(cd_maxit), cd_tol,
            L1, 0.0, nonneg, static_cast<int>(threads), upper_bound);
    }
    return h;
}

// ============================================================================
// STREAMING NNLS (for SPZ files)
// ============================================================================

// [[Rcpp::export]]
Eigen::MatrixXd c_nnls_streaming(const std::string& path,
                                  const Eigen::MatrixXd& w,
                                  unsigned int cd_maxit = 100,
                                  const double cd_tol = 1e-8,
                                  const double L1 = 0, const double L2 = 0,
                                  const double upper_bound = 0,
                                  const bool nonneg = true,
                                  const unsigned int threads = 0) {
    return FactorNet::nmf::nnls_streaming_spz<double>(
        path, w,
        static_cast<int>(cd_maxit), cd_tol,
        L1, L2, upper_bound, nonneg,
        /* solver_mode = */ 0, static_cast<int>(threads));
}

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

// [[Rcpp::export]]
Rcpp::NumericMatrix c_rmatrix(uint32_t nrow, uint32_t ncol, uint32_t rng_seed) {
    Eigen::MatrixXd mat(nrow, ncol);
    FactorNet::rng::SplitMix64 rng(rng_seed);
    rng.fill_uniform(mat);
    return Rcpp::wrap(mat);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix c_rtimatrix(uint32_t nrow, uint32_t ncol, uint32_t rng_seed) {
    Eigen::MatrixXd mat(nrow, ncol);
    FactorNet::rng::SplitMix64 rng(rng_seed);
    rng.fill_uniform(mat);
    // Make upper triangular
    for (uint32_t i = 0; i < nrow; ++i) {
        for (uint32_t j = 0; j < i && j < ncol; ++j) {
            mat(i, j) = 0;
        }
    }
    return Rcpp::wrap(mat);
}

// [[Rcpp::export]]
Rcpp::NumericVector c_runif(const uint32_t n, const float min, const float max,
                            const uint32_t rng_seed, const uint32_t rng2) {
    FactorNet::rng::SplitMix64 rng(rng_seed);
    Rcpp::NumericVector result(n);
    for (uint32_t i = 0; i < n; ++i) {
        result[i] = rng.uniform() * (max - min) + min;
    }
    return result;
}

// [[Rcpp::export]]
Rcpp::IntegerVector c_rbinom(const uint32_t n, uint32_t size, const uint32_t inv_probability,
                             const uint32_t rng_seed, const uint32_t rng2) {
    FactorNet::rng::SplitMix64 rng(rng_seed);
    Rcpp::IntegerVector result(n);
    double p = 1.0 / inv_probability;
    for (uint32_t i = 0; i < n; ++i) {
        int count = 0;
        for (uint32_t t = 0; t < size; ++t) {
            if (rng.uniform() < p) ++count;
        }
        result[i] = count;
    }
    return result;
}

// [[Rcpp::export]]
std::vector<uint32_t> c_sample(const uint32_t n, const uint32_t size,
                                const bool replace, const uint32_t rng_seed,
                                const uint32_t rng2) {
    FactorNet::rng::SplitMix64 rng(rng_seed);
    std::vector<uint32_t> result;
    result.reserve(size);
    
    if (replace) {
        for (uint32_t i = 0; i < size; ++i) {
            result.push_back(static_cast<uint32_t>(rng.rand_int(n)));
        }
    } else {
        std::vector<uint32_t> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        for (uint32_t i = 0; i < size && i < n; ++i) {
            uint32_t idx = static_cast<uint32_t>(rng.rand_int(pool.size()));
            result.push_back(pool[idx]);
            pool.erase(pool.begin() + idx);
        }
    }
    
    return result;
}

// [[Rcpp::export]]
Rcpp::S4 c_rtisparsematrix(const uint32_t nrow, const uint32_t ncol,
                           const uint32_t inv_probability, const bool pattern_only,
                           uint32_t rng_seed) {
    FactorNet::rng::SplitMix64 rng(rng_seed);
    double p = 1.0 / inv_probability;
    
    std::vector<int> i_vec, p_vec;
    std::vector<double> x_vec;
    p_vec.push_back(0);
    
    for (uint32_t col = 0; col < ncol; ++col) {
        for (uint32_t row = 0; row <= col && row < nrow; ++row) {
            if (rng.uniform() < p) {
                i_vec.push_back(row);
                x_vec.push_back(pattern_only ? 1.0 : rng.uniform());
            }
        }
        p_vec.push_back(i_vec.size());
    }
    
    Rcpp::S4 mat("dgCMatrix");
    mat.slot("i") = Rcpp::wrap(i_vec);
    mat.slot("p") = Rcpp::wrap(p_vec);
    mat.slot("x") = Rcpp::wrap(x_vec);
    mat.slot("Dim") = Rcpp::IntegerVector::create(nrow, ncol);
    
    return mat;
}

// [[Rcpp::export]]
Rcpp::S4 c_rsparsematrix(const uint32_t nrow, const uint32_t ncol,
                         const uint32_t inv_probability, const bool pattern_only,
                         uint32_t rng_seed) {
    FactorNet::rng::SplitMix64 rng(rng_seed);
    double p = 1.0 / inv_probability;
    
    std::vector<int> i_vec, p_vec;
    std::vector<double> x_vec;
    p_vec.push_back(0);
    
    for (uint32_t col = 0; col < ncol; ++col) {
        for (uint32_t row = 0; row < nrow; ++row) {
            if (rng.uniform() < p) {
                i_vec.push_back(row);
                x_vec.push_back(pattern_only ? 1.0 : rng.uniform());
            }
        }
        p_vec.push_back(i_vec.size());
    }
    
    Rcpp::S4 mat("dgCMatrix");
    mat.slot("i") = Rcpp::wrap(i_vec);
    mat.slot("p") = Rcpp::wrap(p_vec);
    mat.slot("x") = Rcpp::wrap(x_vec);
    mat.slot("Dim") = Rcpp::IntegerVector::create(nrow, ncol);
    
    return mat;
}

// ============================================================================
// BIPARTITE MATCHING (Hungarian algorithm)
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_bipartite_match(Rcpp::NumericMatrix x) {
    // Convert matrix to vector<vector<double>>
    std::vector<std::vector<double>> cost(x.nrow());
    for (int i = 0; i < x.nrow(); ++i) {
        cost[i].resize(x.ncol());
        for (int j = 0; j < x.ncol(); ++j) {
            cost[i][j] = x(i, j);
        }
    }
    
    HungarianAlgorithm solver;
    std::vector<int> assignment;
    double total_cost = solver.Solve(cost, assignment);
    
    return Rcpp::List::create(
        Rcpp::Named("assignment") = assignment,
        Rcpp::Named("cost") = total_cost
    );
}

// ============================================================================
// CONSENSUS CLUSTERING HELPERS
// ============================================================================

// [[Rcpp::export]]
Eigen::MatrixXd c_knn_jaccard(const Eigen::MatrixXd& sim_matrix, int knn) {
    const int n = static_cast<int>(sim_matrix.rows());
    const int actual_k = std::min(knn, n - 1);
    
    // Step 1: Build KNN sets (each row's top-k indices by similarity)
    std::vector<std::vector<int>> knn_sets(n);
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        // Partial sort to find top-k indices (exclude self)
        std::vector<std::pair<double, int>> sims(n - 1);
        int idx = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            sims[idx++] = {sim_matrix(i, j), j};
        }
        std::partial_sort(sims.begin(), sims.begin() + actual_k, sims.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        
        knn_sets[i].resize(actual_k);
        for (int t = 0; t < actual_k; ++t) {
            knn_sets[i][t] = sims[t].second;
        }
        std::sort(knn_sets[i].begin(), knn_sets[i].end());
    }
    
    // Step 2: Pairwise Jaccard on sorted KNN sets
    Eigen::MatrixXd connectivity = Eigen::MatrixXd::Zero(n, n);
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int j1 = 0; j1 < n - 1; ++j1) {
        for (int j2 = j1 + 1; j2 < n; ++j2) {
            const auto& s1 = knn_sets[j1];
            const auto& s2 = knn_sets[j2];
            
            // Sorted set intersection count
            int intersection = 0;
            int i1 = 0, i2 = 0;
            while (i1 < actual_k && i2 < actual_k) {
                if (s1[i1] == s2[i2]) { ++intersection; ++i1; ++i2; }
                else if (s1[i1] < s2[i2]) ++i1;
                else ++i2;
            }
            
            int union_size = 2 * actual_k - intersection;
            double jaccard = union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
            
            connectivity(j1, j2) = jaccard;
            connectivity(j2, j1) = jaccard;
        }
    }
    
    connectivity.diagonal().setOnes();
    return connectivity;
}
