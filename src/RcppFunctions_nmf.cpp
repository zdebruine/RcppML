// RcppFunctions_nmf.cpp
// NMF exports: rank-CV search (sparse+dense), unified NMF gateway,
// and streaming SPZ NMF.
// fp32-only: all internal computation uses float; R boundary casts
// double→float on entry and float→double on exit.

#include "RcppFunctions_common.h"
#include "../inst/include/RcppML/gateway/nmf.hpp"
#include "../inst/include/RcppML/nmf/fit_streaming_spz.hpp"
#include "../inst/include/RcppML/algorithms/nmf/rank_cv.hpp"

using namespace RcppML::algorithms;
using RcppML::tiny_num;

// ============================================================================
// UNIFIED GATEWAY HELPERS
// ============================================================================

/**
 * @brief Build NMFConfig<float> from R parameter list
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
    config.cd_tol = static_cast<Scalar>(cd_tol > 0 ? cd_tol : 1e-8);
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
 * @brief Convert NMFResult<Scalar> to R list (casts to double for R boundary)
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
    // Cast to float at the R boundary
    Eigen::SparseMatrix<float> A_f = A_mapped.cast<float>();
    
    // Build NMFConfig<float>
    RcppML::NMFConfig<float> config;
    config.tol = static_cast<float>(tol);
    config.max_iter = static_cast<int>(maxit);
    config.cv_seed = cv_seed;
    config.holdout_fraction = static_cast<float>(holdout_fraction);
    config.verbose = verbose;
    config.threads = static_cast<int>(threads);
    config.track_loss_history = true;
    if (L1.size() >= 1) { config.L1_W = static_cast<float>(L1[0]); config.L1_H = static_cast<float>(L1[0]); }
    if (L2.size() >= 1) { config.L2_W = static_cast<float>(L2[0]); config.L2_H = static_cast<float>(L2[0]); }
    
    // Run rank search
    auto result = find_optimal_rank(A_f, k_init, max_k, tolerance, config, verbose);
    
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

    // --- Build NMFConfig<float> (fp32-only path) ---
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

    // Convert graphs to float (proper cast, not reinterpret_cast)
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
    if (W_init_mat.size() > 0) {
        W_init_f = W_init_mat.cast<float>();
        W_init_f_ptr = &W_init_f;
    }

    // Convert mask to float if present
    Eigen::SparseMatrix<float> mask_f;
    if (mask_sexp.isNotNull()) {
        Eigen::SparseMatrix<double> mask_d = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(mask_sexp));
        mask_f = mask_d.cast<float>();
        fconfig.mask = &mask_f;
    }

    // --- Dispatch: sparse or dense, always float ---
    if (is_sparse) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        Eigen::SparseMatrix<float> A_f = A_mapped.cast<float>();
        auto result = RcppML::gateway::nmf<float>(A_f, fconfig, W_init_f_ptr, nullptr);
        return result_to_list(result);
    } else {
        Eigen::MatrixXd A_d = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        Eigen::MatrixXf A_f = A_d.cast<float>();
        auto result = RcppML::gateway::nmf<float>(A_f, fconfig, W_init_f_ptr, nullptr);
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
    
    // Cast to float at the R boundary
    Eigen::MatrixXf A_f = A.cast<float>();

    // Build NMFConfig<float>
    RcppML::NMFConfig<float> config;
    config.tol = static_cast<float>(tol);
    config.max_iter = static_cast<int>(maxit);
    config.cv_seed = cv_seed;
    config.holdout_fraction = static_cast<float>(holdout_fraction);
    config.verbose = verbose;
    config.threads = static_cast<int>(threads);
    config.track_loss_history = true;
    if (L1.size() >= 1) { config.L1_W = static_cast<float>(L1[0]); config.L1_H = static_cast<float>(L1[0]); }
    if (L2.size() >= 1) { config.L2_W = static_cast<float>(L2[0]); config.L2_H = static_cast<float>(L2[0]); }
    
    // Run rank search
    auto result = find_optimal_rank(A_f, k_init, max_k, tolerance, config, verbose);
    
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
    // Build config (StreamingSpzConfig is not templated; scalar dispatched below)
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

    // Always float (fp32-only)
    auto result = RcppML::nmf::nmf_streaming_spz<float>(path, config);
    return Rcpp::List::create(
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
}
