// Complete rewrite of RcppFunctions.cpp to use new modular RcppML architecture
// This file contains all R bindings adapted to use the new API

#include <RcppEigen.h>
#include "../inst/include/RcppML.h"
#include <RcppHungarian.h>
#include <numeric>  // for std::iota

// Phase 2: Unified gateway includes
#include "../inst/include/RcppML/gateway/nmf.hpp"
#include "../inst/include/RcppML/gateway/clustering.hpp"
#include "../inst/include/RcppML/nmf/fit_streaming_spz.hpp"
#include "../inst/include/RcppML/primitives/cpu/nnls_batch.hpp"
#include "../inst/include/RcppML/primitives/cpu/gram.hpp"
#include "../inst/include/RcppML/primitives/cpu/rhs.hpp"

// SVD algorithm implementations — moved here from RcppML.h so that
// RcppExports.cpp (which includes only RcppML.h) does not parse them.
// Include guards make re-inclusion of lanczos/irlba (already in
// fit_unified.hpp via gateway/nmf.hpp) zero-cost.
#include "../inst/include/RcppML/algorithms/nmf/rank_cv.hpp"
#include "../inst/include/RcppML/algorithms/svd/deflation_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/randomized_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/lanczos_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/irlba_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/krylov_constrained_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/streaming_matvec.hpp"
#include "../inst/include/RcppML/algorithms/svd/streaming_svd.hpp"

// Bring new API namespaces into scope
using namespace RcppML::algorithms;
using namespace RcppML::algorithms::clustering;
using RcppML::tiny_num;

// ============================================================================
// Utility: Create Eigen::Map from R dgCMatrix (zero-copy)
// ============================================================================


using MappedSpMat = Eigen::Map<const Eigen::SparseMatrix<double>>;

inline MappedSpMat mapSparseMatrix(const Rcpp::S4& s4) {
    Rcpp::IntegerVector dims = s4.slot("Dim");
    Rcpp::IntegerVector i = s4.slot("i");
    Rcpp::IntegerVector p = s4.slot("p");
    Rcpp::NumericVector x = s4.slot("x");
    return MappedSpMat(dims[0], dims[1], x.size(), p.begin(), i.begin(), x.begin());
}

// Print and interrupt functions for Rcpp integration
inline void rcpp_print(const std::string& s) { Rcpp::Rcout << s; }
inline void rcpp_check_interrupt() { Rcpp::checkUserInterrupt(); }

// ============================================================================
// Shared helpers to reduce duplication
// ============================================================================

// Helper: extract graph matrix from R S4 and compute degree vector
struct GraphData {
    Eigen::SparseMatrix<double> matrix;
    Eigen::VectorXd degrees;
    bool valid = false;
};

inline GraphData extract_graph(const Rcpp::Nullable<Rcpp::S4>& graph, double lambda) {
    GraphData gd;
    if (graph.isNotNull() && lambda > 0) {
        gd.matrix = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(graph));
        gd.degrees = Eigen::VectorXd::Zero(gd.matrix.cols());
        for (int j = 0; j < gd.matrix.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(gd.matrix, j); it; ++it) {
                if (it.row() != j) gd.degrees(j) += 1;
            }
        }
        gd.valid = true;
    }
    return gd;
}

// ============================================================================
// PREDICTION (PROJECT NEW SAMPLES)
// ============================================================================

// [[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w,
                                    const double L1, const double L2, const unsigned int threads,
                                    const bool mask_zeros, const double upper_bound = 0) {
    RcppML::gateway::ThreadGuard guard(static_cast<int>(threads));
    auto A_mapped = mapSparseMatrix(A);
    Eigen::SparseMatrix<double> A_copy(A_mapped);
    
    // R predict method passes w as k×m (already transposed)
    // gram(H, G) computes G = H·H^T, rhs(A, H, B) computes B = H·A
    // So we pass w directly (k×m), no need to transpose
    const int k = static_cast<int>(w.rows());
    const int n = static_cast<int>(A_copy.cols());
    
    // Compute Gram and RHS using unified primitives
    Eigen::MatrixXd G(k, k);
    RcppML::primitives::gram<RcppML::primitives::CPU, double>(w, G);
    G.diagonal().array() += tiny_num<double>();
    
    Eigen::MatrixXd B(k, n);
    RcppML::primitives::rhs<RcppML::primitives::CPU, double>(A_copy, w, B);
    
    // Apply L2 externally to Gram diagonal; pass L1 internally for proper sparsity
    if (L2 > 0) G.diagonal().array() += L2;
    
    Eigen::MatrixXd h(k, n);
    RcppML::primitives::nnls_batch<RcppML::primitives::CPU, double>(
        G, B, h, 100, 1e-8, L1, 0.0, true, static_cast<int>(threads), upper_bound);
    return h;
}

// [[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict_dense(const Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w,
                                   const double L1, const double L2, const unsigned int threads,
                                   const bool mask_zeros, const double upper_bound = 0) {
    RcppML::gateway::ThreadGuard guard(static_cast<int>(threads));
    // R predict method passes w as k×m (already transposed)
    const int k = static_cast<int>(w.rows());
    const int n = static_cast<int>(A_.cols());
    
    // Compute Gram and RHS using unified primitives
    Eigen::MatrixXd G(k, k);
    RcppML::primitives::gram<RcppML::primitives::CPU, double>(w, G);
    G.diagonal().array() += tiny_num<double>();
    
    Eigen::MatrixXd B(k, n);
    RcppML::primitives::rhs<RcppML::primitives::CPU, double>(A_, w, B);
    
    // Apply L2 externally to Gram diagonal; pass L1 internally for proper sparsity
    if (L2 > 0) G.diagonal().array() += L2;
    
    Eigen::MatrixXd h(k, n);
    RcppML::primitives::nnls_batch<RcppML::primitives::CPU, double>(
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
    using namespace RcppML;
    // Reconstruct: A_hat = W * diag(d) * H
    Eigen::MatrixXd WH = W * d.asDiagonal() * H;
    
    if constexpr (RcppML::is_sparse_v<MatrixType>) {
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
    using namespace RcppML;
    
    // Set up loss configuration
    LossConfig<double> loss_config;
    loss_config.type = static_cast<LossType>(loss_type);
    loss_config.huber_delta = huber_delta;
    
    // Reconstruct: A_hat = W * diag(d) * H
    Eigen::MatrixXd WH = W * d.asDiagonal() * H;
    
    double total_loss = 0;
    int count = 0;
    
    if constexpr (RcppML::is_sparse_v<MatrixType>) {
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
double Rcpp_evaluate_loss_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, 
                                 Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                                 int loss_type, double huber_delta,
                                 const unsigned int threads, const bool mask_zeros) {
    auto A_mapped = mapSparseMatrix(A);
    return compute_loss_general(A_mapped, w, d, h, loss_type, huber_delta, mask_zeros);
}

// [[Rcpp::export]]
double Rcpp_evaluate_loss_dense(const Eigen::MatrixXd& A_, const Rcpp::S4& mask,
                                Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                                int loss_type, double huber_delta,
                                const unsigned int threads, const bool mask_zeros) {
    return compute_loss_general(A_, w, d, h, loss_type, huber_delta, mask_zeros);
}

// [[Rcpp::export]]
double Rcpp_evaluate_loss_missing_sparse(const Rcpp::S4& A, const Rcpp::S4& mask,
                                         Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                                         int loss_type, double huber_delta,
                                         const unsigned int threads) {
    using namespace RcppML;
    auto A_mapped = mapSparseMatrix(A);
    auto mask_mapped = mapSparseMatrix(mask);
    
    // Set up loss configuration
    LossConfig<double> loss_config;
    loss_config.type = static_cast<LossType>(loss_type);
    loss_config.huber_delta = huber_delta;
    
    // Compute loss only on masked locations
    Eigen::MatrixXd WH = w * d.asDiagonal() * h;
    double total_loss = 0;
    int count = 0;
    
    for (int k = 0; k < mask_mapped.outerSize(); ++k) {
        for (MappedSpMat::InnerIterator it(mask_mapped, k); it; ++it) {
            if (it.value() > 0) {  // Masked position
                double observed = A_mapped.coeff(it.row(), it.col());
                double predicted = WH(it.row(), it.col());
                total_loss += compute_loss(observed, predicted, loss_config);
                count++;
            }
        }
    }
    
    return count > 0 ? total_loss / count : 0;
}

// [[Rcpp::export]]
double Rcpp_evaluate_loss_missing_dense(const Eigen::MatrixXd& A_, const Rcpp::S4& mask,
                                        Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                                        int loss_type, double huber_delta,
                                        const unsigned int threads) {
    using namespace RcppML;
    auto mask_mapped = mapSparseMatrix(mask);
    
    // Set up loss configuration
    LossConfig<double> loss_config;
    loss_config.type = static_cast<LossType>(loss_type);
    loss_config.huber_delta = huber_delta;
    
    Eigen::MatrixXd WH = w * d.asDiagonal() * h;
    double total_loss = 0;
    int count = 0;
    
    for (int k = 0; k < mask_mapped.outerSize(); ++k) {
        for (MappedSpMat::InnerIterator it(mask_mapped, k); it; ++it) {
            if (it.value() > 0) {
                double observed = A_(it.row(), it.col());
                double predicted = WH(it.row(), it.col());
                total_loss += compute_loss(observed, predicted, loss_config);
                count++;
            }
        }
    }
    
    return count > 0 ? total_loss / count : 0;
}

// ============================================================================
// BIPARTITION
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_bipartition_sparse(const Rcpp::S4& A, const double tol, const unsigned int maxit,
                                   const bool nonneg, const std::vector<unsigned int>& samples,
                                   const unsigned int seed, const bool verbose = false,
                                   const bool calc_dist = false, const bool diag = true,
                                   const int threads = 0) {
    auto A_mapped = mapSparseMatrix(A);
    // Route through gateway for RAII thread management
    auto model = RcppML::gateway::bipartition_gateway(
        A_mapped, tol, maxit, nonneg, samples, seed,
        verbose, calc_dist, diag, threads);

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
}

// [[Rcpp::export]]
Rcpp::List Rcpp_bipartition_dense(Eigen::MatrixXd& A, const double tol, const unsigned int maxit,
                                  const bool nonneg, const std::vector<unsigned int>& samples,
                                  const unsigned int seed, const bool verbose = false,
                                  const bool calc_dist = false, const bool diag = true,
                                  const int threads = 0) {
    // Route through gateway for RAII thread management
    auto model = RcppML::gateway::bipartition_gateway(
        A, tol, maxit, nonneg, samples, seed,
        verbose, calc_dist, diag, threads);

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
    auto clusters = RcppML::gateway::dclust_gateway<double>(
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
        RcppML::primitives::detail::cd_nnls_col_fixed(
            a, b_col.data(), x_col.data(), k,
            L1, 0.0, nonneg, static_cast<int>(cd_maxit), upper_bound);
        x.col(col) = x_col;
    }
    
    return x;
}

// [[Rcpp::export]]
Eigen::MatrixXd c_nnls_dense(const Eigen::MatrixXd& w, const Eigen::MatrixXd& A,
                             unsigned int cd_maxit = 100, const double cd_tol = 1e-8,
                             const double L1 = 0, const double L2 = 0,
                             const double upper_bound = 0, const bool nonneg = true,
                             const unsigned int threads = 0,
                             Rcpp::Nullable<Eigen::MatrixXd> warm_start = R_NilValue) {
    
    RcppML::gateway::ThreadGuard guard(static_cast<int>(threads));
    const int k = static_cast<int>(w.cols());
    const int n = static_cast<int>(A.cols());
    Eigen::MatrixXd w_T = w.transpose();
    
    // Compute Gram and RHS using unified primitives
    Eigen::MatrixXd G(k, k);
    RcppML::primitives::gram<RcppML::primitives::CPU, double>(w_T, G);
    G.diagonal().array() += tiny_num<double>();
    
    Eigen::MatrixXd B(k, n);
    RcppML::primitives::rhs<RcppML::primitives::CPU, double>(A, w_T, B);
    
    // Apply L2 externally to Gram diagonal; pass L1 internally for proper sparsity
    if (L2 > 0) G.diagonal().array() += L2;
    
    Eigen::MatrixXd h;
    if (warm_start.isNotNull()) {
        // Warm start: use cd_nnls_col_fixed directly since nnls_batch always cold starts
        h = Rcpp::as<Eigen::MatrixXd>(warm_start);
        B.noalias() -= G * h;
        #ifdef _OPENMP
        int n_threads = (static_cast<int>(threads) > 0) ? static_cast<int>(threads) : omp_get_max_threads();
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        #endif
        for (int j = 0; j < n; ++j) {
            RcppML::primitives::detail::cd_nnls_col_fixed(
                G, B.col(j).data(), h.col(j).data(), k,
                L1, 0.0, nonneg, static_cast<int>(cd_maxit), upper_bound);
        }
    } else {
        RcppML::primitives::nnls_batch<RcppML::primitives::CPU, double>(
            G, B, h, static_cast<int>(cd_maxit), cd_tol,
            L1, 0.0, nonneg, static_cast<int>(threads), upper_bound);
    }
    return h;
}

// [[Rcpp::export]]
Eigen::MatrixXd c_nnls_sparse(const Eigen::MatrixXd& w, const Rcpp::S4& A,
                              unsigned int cd_maxit = 100, const double cd_tol = 1e-8,
                              const double L1 = 0, const double L2 = 0,
                              const double upper_bound = 0, const bool nonneg = true,
                              const unsigned int threads = 0,
                              Rcpp::Nullable<Eigen::MatrixXd> warm_start = R_NilValue) {
    
    RcppML::gateway::ThreadGuard guard(static_cast<int>(threads));
    auto A_mapped = mapSparseMatrix(A);
    Eigen::SparseMatrix<double> A_copy(A_mapped);
    
    const int k = static_cast<int>(w.cols());
    const int n = static_cast<int>(A_copy.cols());
    Eigen::MatrixXd w_T = w.transpose();
    
    // Compute Gram and RHS using unified primitives
    Eigen::MatrixXd G(k, k);
    RcppML::primitives::gram<RcppML::primitives::CPU, double>(w_T, G);
    G.diagonal().array() += tiny_num<double>();
    
    Eigen::MatrixXd B(k, n);
    RcppML::primitives::rhs<RcppML::primitives::CPU, double>(A_copy, w_T, B);
    
    // Apply L2 externally to Gram diagonal; pass L1 internally for proper sparsity
    if (L2 > 0) G.diagonal().array() += L2;
    
    Eigen::MatrixXd h;
    if (warm_start.isNotNull()) {
        // Warm start: use cd_nnls_col_fixed directly since nnls_batch always cold starts
        h = Rcpp::as<Eigen::MatrixXd>(warm_start);
        B.noalias() -= G * h;
        #ifdef _OPENMP
        int n_threads = (static_cast<int>(threads) > 0) ? static_cast<int>(threads) : omp_get_max_threads();
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        #endif
        for (int j = 0; j < n; ++j) {
            RcppML::primitives::detail::cd_nnls_col_fixed(
                G, B.col(j).data(), h.col(j).data(), k,
                L1, 0.0, nonneg, static_cast<int>(cd_maxit), upper_bound);
        }
    } else {
        RcppML::primitives::nnls_batch<RcppML::primitives::CPU, double>(
            G, B, h, static_cast<int>(cd_maxit), cd_tol,
            L1, 0.0, nonneg, static_cast<int>(threads), upper_bound);
    }
    return h;
}

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

// [[Rcpp::export]]
Rcpp::NumericMatrix c_rmatrix(uint32_t nrow, uint32_t ncol, uint32_t rng_seed) {
    Eigen::MatrixXd mat(nrow, ncol);
    RcppML::rng::SplitMix64 rng(rng_seed);
    rng.fill_uniform(mat);
    return Rcpp::wrap(mat);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix c_rtimatrix(uint32_t nrow, uint32_t ncol, uint32_t rng_seed) {
    Eigen::MatrixXd mat(nrow, ncol);
    RcppML::rng::SplitMix64 rng(rng_seed);
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
    RcppML::rng::SplitMix64 rng(rng_seed);
    Rcpp::NumericVector result(n);
    for (uint32_t i = 0; i < n; ++i) {
        result[i] = rng.uniform() * (max - min) + min;
    }
    return result;
}

// [[Rcpp::export]]
Rcpp::IntegerVector c_rbinom(const uint32_t n, uint32_t size, const uint32_t inv_probability,
                             const uint32_t rng_seed, const uint32_t rng2) {
    RcppML::rng::SplitMix64 rng(rng_seed);
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
    RcppML::rng::SplitMix64 rng(rng_seed);
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
    RcppML::rng::SplitMix64 rng(rng_seed);
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
    RcppML::rng::SplitMix64 rng(rng_seed);
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
// AUTOMATIC RANK DETERMINATION VIA CROSS-VALIDATION
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_nmf_rank_cv_sparse(const Rcpp::S4& A,
                                   int k_init = 2,
                                   int max_k = 50,
                                   int tolerance = 2,
                                   double holdout_fraction = 0.1,
                                   unsigned int cv_seed = 1,
                                   double tol = 1e-4,
                                   unsigned int maxit = 100,
                                   bool verbose = false,
                                   const std::vector<double> L1 = std::vector<double>(),
                                   const std::vector<double> L2 = std::vector<double>(),
                                   unsigned int threads = 0) {
    
    auto A_mapped = mapSparseMatrix(A);
    Eigen::SparseMatrix<double> A_copy(A_mapped);
    
    // Build core::NMFConfig with CV parameters
    RcppML::NMFConfig<double> config;
    config.tol = tol;
    config.max_iter = static_cast<int>(maxit);
    config.cv_seed = cv_seed;
    config.holdout_fraction = holdout_fraction;
    config.verbose = verbose;
    config.threads = static_cast<int>(threads);
    config.track_loss_history = true;
    if (L1.size() >= 1) { config.L1_W = L1[0]; config.L1_H = L1[0]; }
    if (L2.size() >= 1) { config.L2_W = L2[0]; config.L2_H = L2[0]; }
    
    // Run rank search (routes through gateway::nmf for unified CV)
    auto result = find_optimal_rank(A_copy, k_init, max_k, tolerance, config, verbose);
    
    // Build return list
    Rcpp::List evaluations_list(result.evaluations.size());
    for (size_t i = 0; i < result.evaluations.size(); ++i) {
        const auto& eval = result.evaluations[i];
        evaluations_list[i] = Rcpp::List::create(
            Rcpp::Named("rank") = eval.rank,
            Rcpp::Named("train_loss") = eval.train_loss_final,
            Rcpp::Named("test_loss") = eval.test_loss_final,
            Rcpp::Named("train_history") = eval.train_loss_history,
            Rcpp::Named("test_history") = eval.test_loss_history,
            Rcpp::Named("iterations") = eval.iterations
        );
    }
    
    return Rcpp::List::create(
        Rcpp::Named("k_optimal") = result.k_optimal,
        Rcpp::Named("evaluations") = evaluations_list,
        Rcpp::Named("k_low") = result.k_low,
        Rcpp::Named("k_high") = result.k_high,
        Rcpp::Named("overfitting_detected") = result.overfitting_detected
    );
}

// ============================================================================
// UNIFIED GATEWAY HELPERS
// ============================================================================
// These helper functions translate R parameters to unified C++ types.
// They must be defined before Rcpp_nmf_full and Rcpp_nmf_gateway which use them.

/**
 * @brief Build NMFConfig from R parameter list
 * 
 * Maps R parameters to the unified NMFConfig struct. This is the single
 * translation layer between R types and C++ types.
 */
template<typename Scalar>
RcppML::NMFConfig<Scalar> build_config_from_params(
    int rank, double tol, int maxit, bool verbose,
    const std::vector<double>& L1, const std::vector<double>& L2,
    const std::vector<double>& L21, const std::vector<double>& ortho,
    int threads, bool nonneg_w, bool nonneg_h,
    const std::vector<double>& upper_bound,
    int cd_maxit, double cd_tol,
    double cd_abs_tol,
    double holdout_fraction, unsigned int cv_seed,
    int cv_patience, bool mask_zeros,
    bool track_loss_history, bool sort_model,
    unsigned int seed, const std::string& resource_override,
    const std::string& norm_str = "L1",
    bool projective = false,
    bool symmetric = false)
{
    RcppML::NMFConfig<Scalar> config;
    
    // Core
    config.rank = rank;
    config.tol = static_cast<Scalar>(tol);
    config.max_iter = maxit;
    config.verbose = verbose;
    config.seed = seed;
    config.threads = threads;
    config.sort_model = sort_model;
    
    // Normalization type
    if (norm_str == "L2") {
        config.norm_type = RcppML::NormType::L2;
    } else if (norm_str == "none") {
        config.norm_type = RcppML::NormType::None;
    } else {
        config.norm_type = RcppML::NormType::L1;  // default
    }
    
    // Regularization (W/H split)
    config.L1_W = L1.size() > 0 ? static_cast<Scalar>(L1[0]) : 0;
    config.L1_H = L1.size() > 1 ? static_cast<Scalar>(L1[1]) : config.L1_W;
    config.L2_W = L2.size() > 0 ? static_cast<Scalar>(L2[0]) : 0;
    config.L2_H = L2.size() > 1 ? static_cast<Scalar>(L2[1]) : config.L2_W;
    config.L21_W = L21.size() > 0 ? static_cast<Scalar>(L21[0]) : 0;
    config.L21_H = L21.size() > 1 ? static_cast<Scalar>(L21[1]) : config.L21_W;
    config.ortho_W = ortho.size() > 0 ? static_cast<Scalar>(ortho[0]) : 0;
    config.ortho_H = ortho.size() > 1 ? static_cast<Scalar>(ortho[1]) : config.ortho_W;
    
    // Non-negativity and bounds
    config.nonneg_W = nonneg_w;
    config.nonneg_H = nonneg_h;
    config.upper_bound_W = upper_bound.size() > 0 ? static_cast<Scalar>(upper_bound[0]) : 0;
    config.upper_bound_H = upper_bound.size() > 1 ? static_cast<Scalar>(upper_bound[1]) : config.upper_bound_W;
    
    // NNLS solver
    config.cd_max_iter = cd_maxit > 0 ? cd_maxit : 10;
    config.cd_tol = static_cast<Scalar>(cd_tol > 0 ? cd_tol : 5e-3);
    config.cd_abs_tol = static_cast<Scalar>(cd_abs_tol > 0 ? cd_abs_tol : 1e-15);
    
    // Cross-validation
    config.holdout_fraction = static_cast<Scalar>(holdout_fraction);
    config.cv_seed = cv_seed;
    config.cv_patience = cv_patience;
    config.mask_zeros = mask_zeros;
    
    // Algorithm variants
    config.projective = projective;
    config.symmetric = symmetric;
    
    // Loss tracking
    config.track_loss_history = track_loss_history;
    
    // Resource override
    config.resource_override = resource_override;
    
    return config;
}

/**
 * @brief Convert NMFResult to R list
 */
template<typename Scalar>
Rcpp::List result_to_list(const RcppML::NMFResult<Scalar>& result) {
    Rcpp::List ret = Rcpp::List::create(
        Rcpp::Named("w") = result.W.template cast<double>(),
        Rcpp::Named("d") = result.d.template cast<double>(),
        Rcpp::Named("h") = result.H.template cast<double>(),
        Rcpp::Named("tol") = static_cast<double>(result.final_tol),
        Rcpp::Named("iter") = result.iterations,
        Rcpp::Named("train_loss") = static_cast<double>(result.train_loss),
        Rcpp::Named("test_loss") = static_cast<double>(result.test_loss),
        Rcpp::Named("best_test_loss") = static_cast<double>(result.best_test_loss),
        Rcpp::Named("best_iter") = result.best_iter + 1,
        Rcpp::Named("converged") = result.converged,
        Rcpp::Named("loss") = static_cast<double>(result.train_loss)
    );
    
    // Add loss history if tracked
    if (!result.loss_history.empty()) {
        std::vector<double> train_hist(result.loss_history.begin(), 
                                        result.loss_history.end());
        ret["train_loss_history"] = train_hist;
    }
    if (!result.test_loss_history.empty()) {
        std::vector<double> test_hist(result.test_loss_history.begin(), 
                                       result.test_loss_history.end());
        ret["test_loss_history"] = test_hist;
    }
    
    // Add resource diagnostics
    Rcpp::List diag = Rcpp::List::create(
        Rcpp::Named("cpu_cores") = result.diagnostics.cpu_cores_detected,
        Rcpp::Named("gpus") = result.diagnostics.gpus_detected,
        Rcpp::Named("plan") = result.diagnostics.plan_chosen,
        Rcpp::Named("reason") = result.diagnostics.plan_reason
    );
    ret.attr("diagnostics") = diag;
    
    return ret;
}

// [[Rcpp::export]]
Rcpp::List Rcpp_nmf_full(
    SEXP A_sexp,
    std::vector<int> k_vec,
    bool k_auto,
    int k_auto_min,
    int k_auto_max,
    std::vector<double> L1,
    std::vector<double> L2,
    std::vector<double> L21,
    std::vector<double> ortho,
    std::vector<double> upper_bound,
    std::vector<double> graph_lambda,
    double tol,
    int maxit,
    int cd_maxit,
    double cd_tol,
    double cd_abs_tol,
    int threads,
    unsigned int seed,
    bool nonneg_w,
    bool nonneg_h,
    bool sort_model,
    bool mask_zeros,
    bool track_loss_history,
    bool track_train_loss,
    std::string loss_str,
    double huber_delta,
    std::string convergence_str,
    double holdout_fraction,
    std::vector<unsigned int> cv_seeds,
    int cv_patience,
    std::string resource_override,
    std::string norm_str = "L1",
    Rcpp::Nullable<Rcpp::S4> graph_W_sexp = R_NilValue,
    Rcpp::Nullable<Rcpp::S4> graph_H_sexp = R_NilValue,
    Rcpp::Nullable<Eigen::MatrixXd> w_init_sexp = R_NilValue,
    bool verbose = false,
    Rcpp::Nullable<Rcpp::S4> mask_sexp = R_NilValue,
    std::string precision = "double",
    bool projective = false,
    bool symmetric_nmf = false,
    int solver_mode = 0,
    int init_mode = 0,
    int adaptive_solver_switch = 0,
    double chol_adaptive_threshold = 0.05)
{
    // --- Extract graphs with degree vectors ---
    double lambda_W = graph_lambda.size() > 0 ? graph_lambda[0] : 0.0;
    double lambda_H = graph_lambda.size() > 1 ? graph_lambda[1] : 0.0;
    auto gd_W = extract_graph(graph_W_sexp, lambda_W);
    auto gd_H = extract_graph(graph_H_sexp, lambda_H);

    const Eigen::SparseMatrix<double>* graph_W_ptr = gd_W.valid ? &gd_W.matrix : nullptr;
    const Eigen::SparseMatrix<double>* graph_H_ptr = gd_H.valid ? &gd_H.matrix : nullptr;
    const Eigen::VectorXd* graph_W_deg_ptr = gd_W.valid ? &gd_W.degrees : nullptr;
    const Eigen::VectorXd* graph_H_deg_ptr = gd_H.valid ? &gd_H.degrees : nullptr;

    // --- Extract W_init if provided ---
    Eigen::MatrixXd W_init_mat;
    if (w_init_sexp.isNotNull()) {
        W_init_mat = Rcpp::as<Eigen::MatrixXd>(w_init_sexp);
    }

    // Detect sparse vs dense
    bool is_sparse = Rf_isS4(A_sexp);

    // --- Parse loss and convergence ---
    int loss_type = 0;
    if (loss_str == "mse") loss_type = 0;
    else if (loss_str == "mae") loss_type = 1;
    else if (loss_str == "huber") loss_type = 2;
    else if (loss_str == "kl") loss_type = 3;
    int convergence_mode = 0;
    if (convergence_str == "loss") convergence_mode = 0;
    else if (convergence_str == "factor") convergence_mode = 1;
    else if (convergence_str == "both") convergence_mode = 2;

    // --- Build unified NMFConfig ---
    auto uconfig = build_config_from_params<double>(
        k_vec[0], tol, maxit, verbose,
        L1, L2, L21, ortho, threads,
        nonneg_w, nonneg_h, upper_bound,
        cd_maxit, cd_tol, cd_abs_tol,
        holdout_fraction,
        cv_seeds.empty() ? 0u : cv_seeds[0],
        cv_patience, mask_zeros,
        track_loss_history, sort_model, seed, resource_override,
        norm_str,
        projective, symmetric_nmf);

    // Wire convergence mode, loss config, and solver mode
    uconfig.convergence_mode = convergence_mode;
    uconfig.solver_mode = solver_mode;
    uconfig.init_mode = init_mode;
    uconfig.adaptive_solver_switch = adaptive_solver_switch;
    uconfig.chol_adaptive_threshold = static_cast<double>(chol_adaptive_threshold);
    uconfig.loss.type = static_cast<RcppML::LossType>(loss_type);
    uconfig.loss.huber_delta = huber_delta;

    // Set graph configs (using stack-local storage for lifetime)
    Eigen::SparseMatrix<double> graph_W_cfg, graph_H_cfg;
    if (gd_W.valid) {
        graph_W_cfg = gd_W.matrix;
        uconfig.graph_W = &graph_W_cfg;
        uconfig.graph_W_lambda = lambda_W;
    }
    if (gd_H.valid) {
        graph_H_cfg = gd_H.matrix;
        uconfig.graph_H = &graph_H_cfg;
        uconfig.graph_H_lambda = lambda_H;
    }

    // Prepare W_init pointer
    Eigen::MatrixXd* W_init_ptr = W_init_mat.size() > 0 ? &W_init_mat : nullptr;

    // --- Extract mask if provided ---
    Eigen::SparseMatrix<double> mask_cfg;
    if (mask_sexp.isNotNull()) {
        mask_cfg = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(mask_sexp));
        uconfig.mask = &mask_cfg;
    }

    // --- Dispatch via unified gateway ---
    // fp32 path: convert data/init/graphs to float, run float gateway
    if (precision == "float") {
        auto fconfig = build_config_from_params<float>(
            k_vec[0], tol, maxit, verbose,
            L1, L2, L21, ortho, threads,
            nonneg_w, nonneg_h, upper_bound,
            cd_maxit, cd_tol, cd_abs_tol,
            holdout_fraction,
            cv_seeds.empty() ? 0u : cv_seeds[0],
            cv_patience, mask_zeros,
            track_loss_history, sort_model, seed, resource_override,
            norm_str,
            projective, symmetric_nmf);
        fconfig.convergence_mode = convergence_mode;
        fconfig.solver_mode = solver_mode;
        fconfig.init_mode = init_mode;
        fconfig.adaptive_solver_switch = adaptive_solver_switch;
        fconfig.chol_adaptive_threshold = static_cast<float>(chol_adaptive_threshold);
        fconfig.loss.type = static_cast<RcppML::LossType>(loss_type);
        fconfig.loss.huber_delta = static_cast<float>(huber_delta);

        // Convert graphs to float if present
        Eigen::SparseMatrix<float> graph_W_f, graph_H_f;
        if (gd_W.valid) {
            graph_W_f = gd_W.matrix.cast<float>();
            fconfig.graph_W = &graph_W_f;
            fconfig.graph_W_lambda = static_cast<float>(lambda_W);
        }
        if (gd_H.valid) {
            graph_H_f = gd_H.matrix.cast<float>();
            fconfig.graph_H = &graph_H_f;
            fconfig.graph_H_lambda = static_cast<float>(lambda_H);
        }

        // Convert W_init to float
        Eigen::MatrixXf W_init_f;
        Eigen::MatrixXf* W_init_f_ptr = nullptr;
        if (W_init_ptr) {
            W_init_f = W_init_mat.cast<float>();
            W_init_f_ptr = &W_init_f;
        }

        // Convert mask to float if present
        Eigen::SparseMatrix<float> mask_f;
        if (mask_sexp.isNotNull()) {
            mask_f = mask_cfg.cast<float>();
            fconfig.mask = &mask_f;
        }

        if (is_sparse) {
            auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
            Eigen::SparseMatrix<float> A_f = A_mapped.cast<float>();
            auto result = RcppML::gateway::nmf<float>(
                A_f, fconfig, W_init_f_ptr, nullptr);
            return result_to_list(result);
        } else {
            Eigen::MatrixXd A_d = Rcpp::as<Eigen::MatrixXd>(A_sexp);
            Eigen::MatrixXf A_f = A_d.cast<float>();
            auto result = RcppML::gateway::nmf<float>(
                A_f, fconfig, W_init_f_ptr, nullptr);
            return result_to_list(result);
        }
    }

    // --- fp64 path (default) ---
    if (is_sparse) {
        // Zero-copy: map R's dgCMatrix directly into Eigen without deep copy.
        // MappedSparseMatrix is recognized as sparse by is_sparse_v<> trait,
        // and all CPU primitives (rhs, trace_AtA) have Map specializations.
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));

        auto result = RcppML::gateway::nmf<double>(
            A_mapped, uconfig, W_init_ptr, nullptr);
        return result_to_list(result);
    } else {
        Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);

        auto result = RcppML::gateway::nmf<double>(
            A_dense, uconfig, W_init_ptr, nullptr);
        return result_to_list(result);
    }
}

// [[Rcpp::export]]
Rcpp::List Rcpp_nmf_rank_cv_dense(Eigen::MatrixXd A,
                                  int k_init = 2,
                                  int max_k = 50,
                                  int tolerance = 2,
                                  double holdout_fraction = 0.1,
                                  unsigned int cv_seed = 1,
                                  double tol = 1e-4,
                                  unsigned int maxit = 100,
                                  bool verbose = false,
                                  const std::vector<double> L1 = std::vector<double>(),
                                  const std::vector<double> L2 = std::vector<double>(),
                                  unsigned int threads = 0) {
    
    // Build core::NMFConfig with CV parameters
    RcppML::NMFConfig<double> config;
    config.tol = tol;
    config.max_iter = static_cast<int>(maxit);
    config.cv_seed = cv_seed;
    config.holdout_fraction = holdout_fraction;
    config.verbose = verbose;
    config.threads = static_cast<int>(threads);
    config.track_loss_history = true;
    if (L1.size() >= 1) { config.L1_W = L1[0]; config.L1_H = L1[0]; }
    if (L2.size() >= 1) { config.L2_W = L2[0]; config.L2_H = L2[0]; }
    
    // Run rank search (routes through gateway::nmf for unified CV)
    auto result = find_optimal_rank(A, k_init, max_k, tolerance, config, verbose);
    
    // Build return list
    Rcpp::List evaluations_list(result.evaluations.size());
    for (size_t i = 0; i < result.evaluations.size(); ++i) {
        const auto& eval = result.evaluations[i];
        evaluations_list[i] = Rcpp::List::create(
            Rcpp::Named("rank") = eval.rank,
            Rcpp::Named("train_loss") = eval.train_loss_final,
            Rcpp::Named("test_loss") = eval.test_loss_final,
            Rcpp::Named("train_history") = eval.train_loss_history,
            Rcpp::Named("test_history") = eval.test_loss_history,
            Rcpp::Named("iterations") = eval.iterations
        );
    }
    
    return Rcpp::List::create(
        Rcpp::Named("k_optimal") = result.k_optimal,
        Rcpp::Named("evaluations") = evaluations_list,
        Rcpp::Named("k_low") = result.k_low,
        Rcpp::Named("k_high") = result.k_high,
        Rcpp::Named("overfitting_detected") = result.overfitting_detected
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

// [[Rcpp::export]]
Rcpp::List Rcpp_nmf_streaming_spz(
    std::string path,
    int k,
    double tol,
    int maxit,
    std::vector<double> L1,
    std::vector<double> L2,
    std::vector<double> L21,
    std::vector<double> angular,
    std::vector<double> upper_bound,
    std::vector<double> graph_lambda,
    bool nonneg_w,
    bool nonneg_h,
    int cd_maxit,
    double cd_tol,
    int threads,
    unsigned int seed,
    bool verbose,
    std::string precision,
    std::string loss_str,
    double huber_delta,
    double holdout_fraction,
    bool mask_zeros,
    unsigned int cv_seed,
    std::string norm_str = "L1",
    Rcpp::Nullable<Rcpp::S4> mask_sexp = R_NilValue,
    Rcpp::Nullable<Rcpp::S4> graph_W_sexp = R_NilValue,
    Rcpp::Nullable<Rcpp::S4> graph_H_sexp = R_NilValue,
    int solver_mode = 0,
    int init_mode = 0)
{
    // Build config
    RcppML::nmf::StreamingSpzConfig config;
    config.k = k;
    config.tol = tol;
    config.maxit = maxit;
    config.solver_mode = solver_mode;
    config.init_mode = init_mode;
    config.L1_w = L1.size() > 0 ? L1[0] : 0;
    config.L1_h = L1.size() > 1 ? L1[1] : config.L1_w;
    config.L2_w = L2.size() > 0 ? L2[0] : 0;
    config.L2_h = L2.size() > 1 ? L2[1] : config.L2_w;
    config.L21_w = L21.size() > 0 ? L21[0] : 0;
    config.L21_h = L21.size() > 1 ? L21[1] : config.L21_w;
    config.ortho_w = angular.size() > 0 ? angular[0] : 0;
    config.ortho_h = angular.size() > 1 ? angular[1] : config.ortho_w;
    config.upper_bound_w = upper_bound.size() > 0 ? upper_bound[0] : 0;
    config.upper_bound_h = upper_bound.size() > 1 ? upper_bound[1] : config.upper_bound_w;
    config.nonneg_w = nonneg_w;
    config.nonneg_h = nonneg_h;
    config.cd_maxit = cd_maxit;
    config.cd_tol = cd_tol;
    config.num_threads = threads;
    config.verbose = verbose ? 1 : 0;
    config.seed = seed;

    // Normalization type
    if (norm_str == "L2") {
        config.norm_type = RcppML::NormType::L2;
    } else if (norm_str == "none") {
        config.norm_type = RcppML::NormType::None;
    } else {
        config.norm_type = RcppML::NormType::L1;
    }

    // Streaming NMF only supports MSE loss (non-MSE IRLS is fundamentally
    // incompatible with sparse streaming — see PRODUCTION_STATUS.md B1).
    // The R layer already validates this, but defense-in-depth:
    if (loss_str != "mse") {
        Rcpp::stop("Streaming SPZ NMF only supports loss = \"mse\". "
                    "Non-MSE losses require the full matrix in memory.");
    }

    // Cross-validation config
    config.holdout_fraction = holdout_fraction;
    config.mask_zeros = mask_zeros;
    config.cv_seed = static_cast<uint32_t>(cv_seed);

    // User-provided NA mask
    Eigen::SparseMatrix<double> mask_cfg;
    if (mask_sexp.isNotNull()) {
        mask_cfg = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(mask_sexp));
        config.mask = &mask_cfg;
    }

    // Extract graphs
    double gw_lambda = graph_lambda.size() > 0 ? graph_lambda[0] : 0.0;
    double gh_lambda = graph_lambda.size() > 1 ? graph_lambda[1] : 0.0;
    auto gd_W = extract_graph(graph_W_sexp, gw_lambda);
    auto gd_H = extract_graph(graph_H_sexp, gh_lambda);
    if (gd_W.valid) {
        config.graph_w = &gd_W.matrix;
        config.graph_w_lambda = gw_lambda;
    }
    if (gd_H.valid) {
        config.graph_h = &gd_H.matrix;
        config.graph_h_lambda = gh_lambda;
    }

    if (precision == "float") {
        auto result = RcppML::nmf::nmf_streaming_spz<float>(path, config);
        Rcpp::List ret = Rcpp::List::create(
            Rcpp::Named("w") = result.W.template cast<double>(),
            Rcpp::Named("d") = result.d.template cast<double>(),
            Rcpp::Named("h") = result.H.template cast<double>(),
            Rcpp::Named("loss") = result.loss,
            Rcpp::Named("test_loss") = result.test_loss,
            Rcpp::Named("iter") = result.iter,
            Rcpp::Named("converged") = result.converged,
            Rcpp::Named("train_loss_history") = result.loss_history,
            Rcpp::Named("test_loss_history") = result.test_loss_history
        );
        return ret;
    } else {
        auto result = RcppML::nmf::nmf_streaming_spz<double>(path, config);
        Rcpp::List ret = Rcpp::List::create(
            Rcpp::Named("w") = result.W,
            Rcpp::Named("d") = result.d,
            Rcpp::Named("h") = result.H,
            Rcpp::Named("loss") = result.loss,
            Rcpp::Named("test_loss") = result.test_loss,
            Rcpp::Named("iter") = result.iter,
            Rcpp::Named("converged") = result.converged,
            Rcpp::Named("train_loss_history") = result.loss_history,
            Rcpp::Named("test_loss_history") = result.test_loss_history
        );
        return ret;
    }
}

// ============================================================================
// Deflation SVD / PCA
// ============================================================================

template<typename Scalar>
Rcpp::List svd_result_to_list(const RcppML::SVDResult<Scalar>& result) {
    Rcpp::List ret = Rcpp::List::create(
        Rcpp::Named("u") = result.U.template cast<double>(),
        Rcpp::Named("d") = result.d.template cast<double>(),
        Rcpp::Named("v") = result.V.template cast<double>(),
        Rcpp::Named("k") = result.k_selected,
        Rcpp::Named("centered") = result.centered,
        Rcpp::Named("iters_per_factor") = result.iters_per_factor,
        Rcpp::Named("wall_time_ms") = result.wall_time_ms,
        Rcpp::Named("frobenius_norm_sq") = result.frobenius_norm_sq
    );

    if (result.centered && result.row_means.size() > 0) {
        ret["row_means"] = result.row_means.template cast<double>();
    }

    if (!result.test_loss_trajectory.empty()) {
        std::vector<double> tl(result.test_loss_trajectory.begin(),
                               result.test_loss_trajectory.end());
        ret["test_loss"] = tl;
    }

    if (!result.train_loss_trajectory.empty()) {
        std::vector<double> trl(result.train_loss_trajectory.begin(),
                                result.train_loss_trajectory.end());
        ret["train_loss"] = trl;
    }

    return ret;
}

// [[Rcpp::export]]
Rcpp::List Rcpp_svd_pca(
    SEXP A_sexp,
    int k_max,
    double tol,
    int max_iter,
    bool center,
    bool verbose,
    unsigned int seed,
    int threads,
    double L1_u, double L1_v,
    double L2_u, double L2_v,
    bool nonneg_u, bool nonneg_v,
    double upper_bound_u, double upper_bound_v,
    double L21_u, double L21_v,
    double angular_u, double angular_v,
    Rcpp::Nullable<Rcpp::S4> graph_u,
    Rcpp::Nullable<Rcpp::S4> graph_v,
    double graph_u_lambda, double graph_v_lambda,
    double test_fraction,
    unsigned int cv_seed,
    int patience,
    bool mask_zeros,
    std::string convergence = "factor",
    std::string precision = "double",
    std::string method = "deflation")
{
    bool is_sparse = Rf_isS4(A_sexp);

    // Extract graph Laplacians (if provided)
    GraphData graph_u_data = extract_graph(graph_u, graph_u_lambda);
    GraphData graph_v_data = extract_graph(graph_v, graph_v_lambda);

    // Parse convergence mode
    RcppML::SVDConvergence conv_mode = RcppML::SVDConvergence::FACTOR;
    if (convergence == "loss") conv_mode = RcppML::SVDConvergence::LOSS;
    else if (convergence == "both") conv_mode = RcppML::SVDConvergence::BOTH;

    auto run = [&](auto scalar_tag) -> Rcpp::List {
        using Scalar = decltype(scalar_tag);

        RcppML::SVDConfig<Scalar> config;
        config.k_max = k_max;
        config.tol = static_cast<Scalar>(tol);
        config.max_iter = max_iter;
        config.center = center;
        config.verbose = verbose;
        config.seed = seed;
        config.threads = threads;
        config.convergence = conv_mode;

        // Per-element regularization
        config.L1_u = static_cast<Scalar>(L1_u);
        config.L1_v = static_cast<Scalar>(L1_v);
        config.L2_u = static_cast<Scalar>(L2_u);
        config.L2_v = static_cast<Scalar>(L2_v);
        config.nonneg_u = nonneg_u;
        config.nonneg_v = nonneg_v;
        config.upper_bound_u = static_cast<Scalar>(upper_bound_u);
        config.upper_bound_v = static_cast<Scalar>(upper_bound_v);

        // Gram-level regularization
        config.L21_u = static_cast<Scalar>(L21_u);
        config.L21_v = static_cast<Scalar>(L21_v);
        config.angular_u = static_cast<Scalar>(angular_u);
        config.angular_v = static_cast<Scalar>(angular_v);

        // Graph Laplacian pointers (owned by graph_u_data/graph_v_data)
        // Note: For float precision, we'd need to cast the sparse matrices.
        // For now, graph reg only supports double precision.
        if (graph_u_data.valid) {
            config.graph_u = reinterpret_cast<const RcppML::SparseMatrix<Scalar>*>(&graph_u_data.matrix);
            config.graph_u_lambda = static_cast<Scalar>(graph_u_lambda);
        }
        if (graph_v_data.valid) {
            config.graph_v = reinterpret_cast<const RcppML::SparseMatrix<Scalar>*>(&graph_v_data.matrix);
            config.graph_v_lambda = static_cast<Scalar>(graph_v_lambda);
        }

        // CV fields
        config.test_fraction = static_cast<Scalar>(test_fraction);
        config.cv_seed = cv_seed;
        config.patience = patience;
        config.mask_zeros = mask_zeros;

        // Dispatch helper: run an SVD algorithm on sparse or dense A
        auto dispatch_svd = [&](auto svd_func) -> Rcpp::List {
            if (is_sparse) {
                auto A = mapSparseMatrix(Rcpp::S4(A_sexp));
                auto result = svd_func(A, config);
                return svd_result_to_list(result);
            } else {
                Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
                if constexpr (std::is_same_v<Scalar, float>) {
                    Eigen::MatrixXf A_f = A_dense.cast<float>();
                    auto result = svd_func(A_f, config);
                    return svd_result_to_list(result);
                } else {
                    auto result = svd_func(A_dense, config);
                    return svd_result_to_list(result);
                }
            }
        };

        if (method == "randomized") {
            return dispatch_svd([](const auto& A, const auto& c) {
                using AType = std::remove_cv_t<std::remove_reference_t<decltype(A)>>;
                return RcppML::svd::randomized_svd<AType, Scalar>(A, c);
            });
        } else if (method == "lanczos") {
            return dispatch_svd([](const auto& A, const auto& c) {
                using AType = std::remove_cv_t<std::remove_reference_t<decltype(A)>>;
                return RcppML::svd::lanczos_svd<AType, Scalar>(A, c);
            });
        } else if (method == "irlba") {
            return dispatch_svd([](const auto& A, const auto& c) {
                using AType = std::remove_cv_t<std::remove_reference_t<decltype(A)>>;
                return RcppML::svd::irlba_svd<AType, Scalar>(A, c);
            });
        } else if (method == "krylov") {
            return dispatch_svd([](const auto& A, const auto& c) {
                using AType = std::remove_cv_t<std::remove_reference_t<decltype(A)>>;
                return RcppML::svd::krylov_constrained_svd<AType, Scalar>(A, c);
            });
        } else {
            // Default: deflation SVD
            return dispatch_svd([](const auto& A, const auto& c) {
                using AType = std::remove_cv_t<std::remove_reference_t<decltype(A)>>;
                return RcppML::svd::deflation_svd<AType, Scalar>(A, c);
            });
        }
    };

    if (precision == "float") {
        return run(float{});
    } else {
        return run(double{});
    }
}
// ============================================================================
// Streaming SVD / PCA (out-of-core via SPZ files)
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_svd_streaming_spz(
    std::string path,
    int k_max,
    double tol,
    int max_iter,
    bool center,
    bool verbose,
    unsigned int seed,
    int threads,
    double L1_u, double L1_v,
    double L2_u, double L2_v,
    bool nonneg_u, bool nonneg_v,
    double upper_bound_u, double upper_bound_v,
    double L21_u, double L21_v,
    double angular_u, double angular_v,
    Rcpp::Nullable<Rcpp::S4> graph_u,
    Rcpp::Nullable<Rcpp::S4> graph_v,
    double graph_u_lambda, double graph_v_lambda,
    double test_fraction,
    unsigned int cv_seed,
    int patience,
    bool mask_zeros,
    std::string convergence = "factor",
    std::string precision = "double",
    std::string method = "deflation")
{
    // Extract graph Laplacians
    GraphData graph_u_data = extract_graph(graph_u, graph_u_lambda);
    GraphData graph_v_data = extract_graph(graph_v, graph_v_lambda);

    // Parse convergence mode
    RcppML::SVDConvergence conv_mode = RcppML::SVDConvergence::FACTOR;
    if (convergence == "loss") conv_mode = RcppML::SVDConvergence::LOSS;
    else if (convergence == "both") conv_mode = RcppML::SVDConvergence::BOTH;

    auto run = [&](auto scalar_tag) -> Rcpp::List {
        using Scalar = decltype(scalar_tag);

        RcppML::SVDConfig<Scalar> config;
        config.k_max = k_max;
        config.tol = static_cast<Scalar>(tol);
        config.max_iter = max_iter;
        config.center = center;
        config.verbose = verbose;
        config.seed = seed;
        config.threads = threads;
        config.convergence = conv_mode;
        config.spz_path = path;

        // Per-element regularization
        config.L1_u = static_cast<Scalar>(L1_u);
        config.L1_v = static_cast<Scalar>(L1_v);
        config.L2_u = static_cast<Scalar>(L2_u);
        config.L2_v = static_cast<Scalar>(L2_v);
        config.nonneg_u = nonneg_u;
        config.nonneg_v = nonneg_v;
        config.upper_bound_u = static_cast<Scalar>(upper_bound_u);
        config.upper_bound_v = static_cast<Scalar>(upper_bound_v);

        // Gram-level regularization
        config.L21_u = static_cast<Scalar>(L21_u);
        config.L21_v = static_cast<Scalar>(L21_v);
        config.angular_u = static_cast<Scalar>(angular_u);
        config.angular_v = static_cast<Scalar>(angular_v);

        if (graph_u_data.valid) {
            config.graph_u = reinterpret_cast<const RcppML::SparseMatrix<Scalar>*>(&graph_u_data.matrix);
            config.graph_u_lambda = static_cast<Scalar>(graph_u_lambda);
        }
        if (graph_v_data.valid) {
            config.graph_v = reinterpret_cast<const RcppML::SparseMatrix<Scalar>*>(&graph_v_data.matrix);
            config.graph_v_lambda = static_cast<Scalar>(graph_v_lambda);
        }

        // CV fields
        config.test_fraction = static_cast<Scalar>(test_fraction);
        config.cv_seed = cv_seed;
        config.patience = patience;
        config.mask_zeros = mask_zeros;

        RcppML::SVDResult<Scalar> result;

        if (method == "randomized") {
            result = RcppML::svd::streaming_randomized_svd<Scalar>(config);
        } else if (method == "lanczos") {
            result = RcppML::svd::streaming_lanczos_svd<Scalar>(config);
        } else if (method == "irlba") {
            result = RcppML::svd::streaming_irlba_svd<Scalar>(config);
        } else if (method == "krylov") {
            result = RcppML::svd::streaming_krylov_svd<Scalar>(config);
        } else {
            // Default: streaming deflation
            result = RcppML::svd::streaming_deflation_svd<Scalar>(config);
        }

        return svd_result_to_list(result);
    };

    if (precision == "float") {
        return run(float{});
    } else {
        return run(double{});
    }
}