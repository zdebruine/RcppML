// RcppFunctions_svd.cpp
// SVD/PCA exports: in-memory (sparse+dense) and streaming (SPZ) variants.
// fp32-only: always runs float<> path. Fixes the reinterpret_cast UB that
// was present in the original monolithic file for graph Laplacians.

#include "RcppFunctions_common.h"
#include "../inst/include/RcppML/algorithms/svd/deflation_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/randomized_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/lanczos_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/irlba_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/krylov_constrained_svd.hpp"
#include "../inst/include/RcppML/algorithms/svd/streaming_matvec.hpp"
#include "../inst/include/RcppML/algorithms/svd/streaming_svd.hpp"
// trace_AtA is inline in cpu/rhs.hpp; not reachable from SVD headers
#include "../inst/include/RcppML/primitives/cpu/rhs.hpp"

using RcppML::tiny_num;

// ============================================================================
// SVD result → R list (casts to double at R boundary)
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

// ============================================================================
// Deflation SVD / PCA (in-memory)
// ============================================================================

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
    std::string method = "deflation")
{
    bool is_sparse = Rf_isS4(A_sexp);

    // Extract graph Laplacians (double storage; cast to float below)
    GraphData graph_u_data = extract_graph(graph_u, graph_u_lambda);
    GraphData graph_v_data = extract_graph(graph_v, graph_v_lambda);

    // Parse convergence mode
    RcppML::SVDConvergence conv_mode = RcppML::SVDConvergence::FACTOR;
    if (convergence == "loss") conv_mode = RcppML::SVDConvergence::LOSS;
    else if (convergence == "both") conv_mode = RcppML::SVDConvergence::BOTH;

    // Always float (fp32-only)
    using Scalar = float;

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

    // Graph Laplacians: proper float cast (fixes reinterpret_cast UB from old code)
    Eigen::SparseMatrix<float> graph_u_f, graph_v_f;
    if (graph_u_data.valid) {
        graph_u_f = graph_u_data.matrix.cast<float>();
        config.graph_u = &graph_u_f;
        config.graph_u_lambda = static_cast<Scalar>(graph_u_lambda);
    }
    if (graph_v_data.valid) {
        graph_v_f = graph_v_data.matrix.cast<float>();
        config.graph_v = &graph_v_f;
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
            // Cast sparse to float at R boundary
            Eigen::SparseMatrix<float> A_f = A.cast<float>();
            auto result = svd_func(A_f, config);
            return svd_result_to_list(result);
        } else {
            Eigen::MatrixXd A_dense = Rcpp::as<Eigen::MatrixXd>(A_sexp);
            Eigen::MatrixXf A_f = A_dense.cast<float>();
            auto result = svd_func(A_f, config);
            return svd_result_to_list(result);
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
    std::string method = "deflation")
{
    // Extract graph Laplacians (double storage; cast to float below)
    GraphData graph_u_data = extract_graph(graph_u, graph_u_lambda);
    GraphData graph_v_data = extract_graph(graph_v, graph_v_lambda);

    // Parse convergence mode
    RcppML::SVDConvergence conv_mode = RcppML::SVDConvergence::FACTOR;
    if (convergence == "loss") conv_mode = RcppML::SVDConvergence::LOSS;
    else if (convergence == "both") conv_mode = RcppML::SVDConvergence::BOTH;

    // Always float (fp32-only)
    using Scalar = float;

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

    // Graph Laplacians: proper float cast (fixes reinterpret_cast UB from old code)
    Eigen::SparseMatrix<float> graph_u_f, graph_v_f;
    if (graph_u_data.valid) {
        graph_u_f = graph_u_data.matrix.cast<float>();
        config.graph_u = &graph_u_f;
        config.graph_u_lambda = static_cast<Scalar>(graph_u_lambda);
    }
    if (graph_v_data.valid) {
        graph_v_f = graph_v_data.matrix.cast<float>();
        config.graph_v = &graph_v_f;
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
}
