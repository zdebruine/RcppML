// RcppFunctions_svd.cpp
// SVD/PCA exports: in-memory (sparse+dense) and streaming (SPZ) variants.
// All method dispatch routes through svd/gateway.hpp — one include handles
// deflation, randomized, lanczos, irlba, krylov, and streaming paths.

#include "RcppFunctions_common.h"
#include "../inst/include/FactorNet/svd/gateway.hpp"
// trace_AtA is inline in cpu/rhs.hpp; not reachable from SVD headers
#include "../inst/include/FactorNet/primitives/cpu/rhs.hpp"

using FactorNet::tiny_num;

// ============================================================================
// Helper: SVD result → R list (casts to double at R boundary)
// ============================================================================

template<typename Scalar>
static Rcpp::List svd_result_to_list(const FactorNet::SVDResult<Scalar>& result) {
    Rcpp::List ret = Rcpp::List::create(
        Rcpp::Named("u") = result.U.template cast<double>(),
        Rcpp::Named("d") = result.d.template cast<double>(),
        Rcpp::Named("v") = result.V.template cast<double>(),
        Rcpp::Named("k") = result.k_selected,
        Rcpp::Named("centered") = result.centered,
        Rcpp::Named("scaled") = result.scaled,
        Rcpp::Named("iters_per_factor") = result.iters_per_factor,
        Rcpp::Named("wall_time_ms") = result.wall_time_ms,
        Rcpp::Named("frobenius_norm_sq") = result.frobenius_norm_sq
    );

    if (result.centered && result.row_means.size() > 0) {
        ret["row_means"] = result.row_means.template cast<double>();
    }

    if (result.scaled && result.row_sds.size() > 0) {
        ret["row_sds"] = result.row_sds.template cast<double>();
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

// ============================================================================
// Deflation SVD / PCA (in-memory, sparse or dense A)
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_svd_pca(
    SEXP A_sexp,
    int k_max,
    double tol,
    int max_iter,
    bool center,
    bool scale,
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
    Rcpp::Nullable<Rcpp::S4> obs_mask,
    std::string convergence = "factor",
    std::string method = "deflation",
    double robust_delta = 0.0,
    int irls_max_iter = 5,
    double irls_tol = 1e-4)
{
    using Scalar = float;

    // Build config from plain-old-data parameters
    auto config = FactorNet::svd::build_svd_config_scalars<Scalar>(
        k_max, static_cast<Scalar>(tol), max_iter, center, scale,
        verbose, seed, threads,
        FactorNet::svd::parse_svd_convergence(convergence),
        static_cast<Scalar>(L1_u), static_cast<Scalar>(L1_v),
        static_cast<Scalar>(L2_u), static_cast<Scalar>(L2_v),
        nonneg_u, nonneg_v,
        static_cast<Scalar>(upper_bound_u), static_cast<Scalar>(upper_bound_v),
        static_cast<Scalar>(L21_u), static_cast<Scalar>(L21_v),
        static_cast<Scalar>(angular_u), static_cast<Scalar>(angular_v),
        static_cast<Scalar>(test_fraction), cv_seed, patience, mask_zeros,
        "",  // spz_path (not streaming)
        static_cast<Scalar>(robust_delta),
        irls_max_iter,
        static_cast<Scalar>(irls_tol));

    // Graph Laplacians: extract + cast to float (must outlive config dispatch)
    GraphData graph_u_data = extract_graph(graph_u, graph_u_lambda);
    GraphData graph_v_data = extract_graph(graph_v, graph_v_lambda);
    Eigen::SparseMatrix<Scalar> graph_u_f, graph_v_f;
    if (graph_u_data.valid) {
        graph_u_f = graph_u_data.matrix.cast<Scalar>();
        config.graph_u        = &graph_u_f;
        config.graph_u_lambda = static_cast<Scalar>(graph_u_lambda);
    }
    if (graph_v_data.valid) {
        graph_v_f = graph_v_data.matrix.cast<Scalar>();
        config.graph_v        = &graph_v_f;
        config.graph_v_lambda = static_cast<Scalar>(graph_v_lambda);
    }

    // Observation mask
    Eigen::SparseMatrix<Scalar> obs_mask_f;
    if (obs_mask.isNotNull()) {
        obs_mask_f = mapSparseMatrix(Rcpp::S4(obs_mask.get())).cast<Scalar>();
        config.obs_mask = &obs_mask_f;
    }

    // Dispatch: sparse or dense A → gateway → algorithm
    if (Rf_isS4(A_sexp)) {
        Eigen::SparseMatrix<Scalar> A_f =
            mapSparseMatrix(Rcpp::S4(A_sexp)).cast<Scalar>();
        return svd_result_to_list(
            FactorNet::svd::svd_gateway(A_f, config, method));
    } else {
        Eigen::MatrixXf A_f =
            Rcpp::as<Eigen::MatrixXd>(A_sexp).cast<Scalar>();
        return svd_result_to_list(
            FactorNet::svd::svd_gateway(A_f, config, method));
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
    bool scale,
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
    Rcpp::Nullable<Rcpp::S4> obs_mask,
    std::string convergence = "factor",
    std::string method = "deflation",
    double robust_delta = 0.0,
    int irls_max_iter = 5,
    double irls_tol = 1e-4)
{
    using Scalar = float;

    // Build config — spz_path supplied here
    auto config = FactorNet::svd::build_svd_config_scalars<Scalar>(
        k_max, static_cast<Scalar>(tol), max_iter, center, scale,
        verbose, seed, threads,
        FactorNet::svd::parse_svd_convergence(convergence),
        static_cast<Scalar>(L1_u), static_cast<Scalar>(L1_v),
        static_cast<Scalar>(L2_u), static_cast<Scalar>(L2_v),
        nonneg_u, nonneg_v,
        static_cast<Scalar>(upper_bound_u), static_cast<Scalar>(upper_bound_v),
        static_cast<Scalar>(L21_u), static_cast<Scalar>(L21_v),
        static_cast<Scalar>(angular_u), static_cast<Scalar>(angular_v),
        static_cast<Scalar>(test_fraction), cv_seed, patience, mask_zeros,
        path,  // <-- streaming path
        static_cast<Scalar>(robust_delta),
        irls_max_iter,
        static_cast<Scalar>(irls_tol));

    // Graph Laplacians
    GraphData graph_u_data = extract_graph(graph_u, graph_u_lambda);
    GraphData graph_v_data = extract_graph(graph_v, graph_v_lambda);
    Eigen::SparseMatrix<Scalar> graph_u_f, graph_v_f;
    if (graph_u_data.valid) {
        graph_u_f = graph_u_data.matrix.cast<Scalar>();
        config.graph_u        = &graph_u_f;
        config.graph_u_lambda = static_cast<Scalar>(graph_u_lambda);
    }
    if (graph_v_data.valid) {
        graph_v_f = graph_v_data.matrix.cast<Scalar>();
        config.graph_v        = &graph_v_f;
        config.graph_v_lambda = static_cast<Scalar>(graph_v_lambda);
    }

    // Observation mask
    Eigen::SparseMatrix<Scalar> obs_mask_f;
    if (obs_mask.isNotNull()) {
        obs_mask_f = mapSparseMatrix(Rcpp::S4(obs_mask.get())).cast<Scalar>();
        config.obs_mask = &obs_mask_f;
    }

    return svd_result_to_list(
        FactorNet::svd::svd_streaming_gateway<Scalar>(config, method));
}
