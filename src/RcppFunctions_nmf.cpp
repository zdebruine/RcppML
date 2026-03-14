// RcppFunctions_nmf.cpp
// NMF exports: rank-CV search (sparse+dense), unified NMF gateway,
// and streaming SPZ NMF.
// fp32-only: all internal computation uses float; R boundary casts
// double→float on entry and float→double on exit.

#include "RcppFunctions_common.h"
#include "../inst/include/FactorNet/nmf/fit.hpp"
#include "../inst/include/FactorNet/nmf/fit_streaming_spz.hpp"
#include "../inst/include/FactorNet/nmf/rank_cv.hpp"

using namespace FactorNet::nmf;
using FactorNet::tiny_num;

// ============================================================================
// UNIFIED GATEWAY HELPERS
// ============================================================================

/**
 * @brief Build NMFConfig<float> from R parameter list
 */
template<typename Scalar>
FactorNet::NMFConfig<Scalar> build_config_from_params(
    int rank, double tol, int maxit, bool verbose,
    const std::vector<double>& L1, const std::vector<double>& L2,
    const std::vector<double>& L21, const std::vector<double>& angular,
    int threads, bool nonneg_w, bool nonneg_h,
    const std::vector<double>& upper_bound,
    int cd_maxit, double cd_tol,
    double holdout_fraction, unsigned int cv_seed,
    int cv_patience, bool mask_zeros,
    bool track_loss_history, bool sort_model,
    unsigned int seed, const std::string& resource_override,
    const std::string& norm_str = "L1",
    bool projective = false,
    bool symmetric = false)
{
    FactorNet::NMFConfig<Scalar> config;
    
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
        config.norm_type = FactorNet::NormType::L2;
    } else if (norm_str == "none") {
        config.norm_type = FactorNet::NormType::None;
    } else {
        config.norm_type = FactorNet::NormType::L1;  // default
    }
    
    // Regularization (W/H split)
    config.W.L1 = L1.size() > 0 ? static_cast<Scalar>(L1[0]) : 0;
    config.H.L1 = L1.size() > 1 ? static_cast<Scalar>(L1[1]) : config.W.L1;
    config.W.L2 = L2.size() > 0 ? static_cast<Scalar>(L2[0]) : 0;
    config.H.L2 = L2.size() > 1 ? static_cast<Scalar>(L2[1]) : config.W.L2;
    config.W.L21 = L21.size() > 0 ? static_cast<Scalar>(L21[0]) : 0;
    config.H.L21 = L21.size() > 1 ? static_cast<Scalar>(L21[1]) : config.W.L21;
    config.W.angular = angular.size() > 0 ? static_cast<Scalar>(angular[0]) : 0;
    config.H.angular = angular.size() > 1 ? static_cast<Scalar>(angular[1]) : config.W.angular;
    
    // Non-negativity and bounds
    config.W.nonneg = nonneg_w;
    config.H.nonneg = nonneg_h;
    config.W.upper_bound = upper_bound.size() > 0 ? static_cast<Scalar>(upper_bound[0]) : 0;
    config.H.upper_bound = upper_bound.size() > 1 ? static_cast<Scalar>(upper_bound[1]) : config.W.upper_bound;
    
    // NNLS solver
    config.cd_max_iter = cd_maxit > 0 ? cd_maxit : 10;
    config.cd_tol = static_cast<Scalar>(cd_tol > 0 ? cd_tol : 1e-8);
    
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

// Overload that also sets track_train_loss (called from Rcpp_nmf_full)
template<typename Scalar>
void apply_track_train_loss(FactorNet::NMFConfig<Scalar>& config, bool track_train_loss) {
    config.track_train_loss = track_train_loss;
}

/**
 * @brief Convert NMFResult<Scalar> to R list (casts to double for R boundary)
 */
template<typename Scalar>
Rcpp::List result_to_list(const FactorNet::NMFResult<Scalar>& result) {
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
    
    // Add FitHistory as a data.frame if populated
    if (!result.history.empty()) {
        Rcpp::DataFrame history_df = Rcpp::DataFrame::create(
            Rcpp::Named("iteration") = result.history.iterations(),
            Rcpp::Named("train_loss") = result.history.train_loss(),
            Rcpp::Named("test_loss") = result.history.test_loss(),
            Rcpp::Named("delta_loss") = result.history.delta_loss(),
            Rcpp::Named("wall_ms") = result.history.wall_ms()
        );
        ret["history"] = history_df;
    }

    // Add resource diagnostics
    Rcpp::List diag = Rcpp::List::create(
        Rcpp::Named("cpu_cores") = result.diagnostics.cpu_cores_detected,
        Rcpp::Named("gpus") = result.diagnostics.gpus_detected,
        Rcpp::Named("plan") = result.diagnostics.plan_chosen,
        Rcpp::Named("reason") = result.diagnostics.plan_reason
    );
    ret.attr("diagnostics") = diag;

    // Add GP theta if present
    if (result.theta.size() > 0) {
        ret["theta"] = result.theta.template cast<double>();
    }

    // Add Gamma/InvGauss dispersion if present
    if (result.dispersion.size() > 0) {
        ret["dispersion"] = result.dispersion.template cast<double>();
    }

    // Add ZIGP pi vectors if present
    if (result.pi_row.size() > 0) {
        ret["pi_row"] = result.pi_row.template cast<double>();
    }
    if (result.pi_col.size() > 0) {
        ret["pi_col"] = result.pi_col.template cast<double>();
    }

    // Add profiling data if present
    if (!result.profile.empty()) {
        Rcpp::NumericVector prof_vec(result.profile.size());
        Rcpp::CharacterVector prof_names(result.profile.size());
        int idx = 0;
        for (const auto& kv : result.profile) {
            prof_names[idx] = kv.first;
            prof_vec[idx] = kv.second;
            idx++;
        }
        prof_vec.attr("names") = prof_names;
        ret["profile"] = prof_vec;
    }
    
    return ret;
}

// ============================================================================
// AUTOMATIC RANK DETERMINATION VIA CROSS-VALIDATION
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_nmf_rank_cv(SEXP A_sexp,
                            int k_init = 2,
                            int max_k = 50,
                            int tolerance = 2,
                            double holdout_fraction = 0.1,
                            unsigned int cv_seed = 1,
                            double tol = 1e-4,
                            unsigned int maxit = 100,
                            bool verbose = false,
                            Rcpp::Nullable<Rcpp::NumericVector> L1 = R_NilValue,
                            Rcpp::Nullable<Rcpp::NumericVector> L2 = R_NilValue,
                            unsigned int threads = 0) {
    
    // Build NMFConfig<float>
    FactorNet::NMFConfig<float> config;
    config.tol = static_cast<float>(tol);
    config.max_iter = static_cast<int>(maxit);
    config.cv_seed = cv_seed;
    config.holdout_fraction = static_cast<float>(holdout_fraction);
    config.verbose = verbose;
    config.threads = static_cast<int>(threads);
    config.track_loss_history = true;
    config.solver_mode = 2;  // hierarchical_cholesky (match R default)
    if (L1.isNotNull()) { Rcpp::NumericVector v(L1); if (v.size() >= 1) { config.W.L1 = static_cast<float>(v[0]); config.H.L1 = static_cast<float>(v[0]); } }
    if (L2.isNotNull()) { Rcpp::NumericVector v(L2); if (v.size() >= 1) { config.W.L2 = static_cast<float>(v[0]); config.H.L2 = static_cast<float>(v[0]); } }
    
    // Dispatch: sparse or dense, always cast to float
    RankSearchResult<float> result;
    if (Rf_isS4(A_sexp)) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        Eigen::SparseMatrix<float> A_f = A_mapped.cast<float>();
        result = find_optimal_rank(A_f, k_init, max_k, tolerance, config, verbose);
    } else {
        Eigen::MatrixXd A_d = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        Eigen::MatrixXf A_f = A_d.cast<float>();
        result = find_optimal_rank(A_f, k_init, max_k, tolerance, config, verbose);
    }
    
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
    std::vector<double> angular,
    std::vector<double> upper_bound,
    std::vector<double> graph_lambda,
    double tol,
    int maxit,
    int cd_maxit,
    double cd_tol,
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
    double holdout_fraction,
    std::vector<unsigned int> cv_seeds,
    int cv_patience,
    std::string resource_override,
    std::string norm_str = "L1",
    Rcpp::Nullable<Rcpp::S4> graph_W_sexp = R_NilValue,
    Rcpp::Nullable<Rcpp::S4> graph_H_sexp = R_NilValue,
    Rcpp::Nullable<Eigen::MatrixXd> w_init_sexp = R_NilValue,
    Rcpp::Nullable<Eigen::MatrixXd> h_init_sexp = R_NilValue,
    bool verbose = false,
    Rcpp::Nullable<Rcpp::S4> mask_sexp = R_NilValue,
    bool projective = false,
    bool symmetric_nmf = false,
    int solver_mode = 0,
    int init_mode = 0,
    int irls_max_iter = 20,
    double irls_tol = 1e-4,
    int gp_dispersion_mode = 1,
    double gp_theta_init = 0.1,
    double gp_theta_max = 5.0,
    int zi_mode = 0,
    int zi_em_iters = 1,
    double gp_theta_min = 0.0,
    double nb_size_init = 10.0,
    double nb_size_max = 1e6,
    double nb_size_min = 0.01,
    double gamma_phi_init = 1.0,
    double gamma_phi_max = 1e4,
    double gamma_phi_min = 1e-6,
    double robust_delta = 0.0,
    double tweedie_power = 1.5,
    SEXP on_iteration_r = R_NilValue,
    Rcpp::Nullable<Eigen::MatrixXd> target_H_sexp = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> target_lambda = R_NilValue,
    bool enable_profiling = false)
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

    // --- Extract H_init if provided ---
    Eigen::MatrixXd H_init_mat;
    if (h_init_sexp.isNotNull()) {
        H_init_mat = Rcpp::as<Eigen::MatrixXd>(h_init_sexp);
    }

    // Detect sparse vs dense
    bool is_sparse = Rf_isS4(A_sexp);

    // --- Parse loss and convergence ---
    int loss_type = 0;
    if (loss_str == "mse") loss_type = 0;
    else if (loss_str == "mae") loss_type = 1;
    else if (loss_str == "huber") loss_type = 2;
    else if (loss_str == "kl") loss_type = 3;
    else if (loss_str == "gp") loss_type = 4;
    else if (loss_str == "nb") loss_type = 5;
    else if (loss_str == "gamma") loss_type = 6;
    else if (loss_str == "inverse_gaussian") loss_type = 7;
    else if (loss_str == "tweedie") loss_type = 8;
    // --- Build NMFConfig<float> (fp32-only path) ---
    auto fconfig = build_config_from_params<float>(
        k_vec[0], tol, maxit, verbose,
        L1, L2, L21, angular, threads,
        nonneg_w, nonneg_h, upper_bound,
        cd_maxit, cd_tol,
        holdout_fraction,
        cv_seeds.empty() ? 0u : cv_seeds[0],
        cv_patience, mask_zeros,
        track_loss_history, sort_model, seed, resource_override,
        norm_str,
        projective, symmetric_nmf);
    fconfig.solver_mode = solver_mode;
    fconfig.init_mode = init_mode;
    fconfig.irls_max_iter = irls_max_iter;
    fconfig.irls_tol = static_cast<float>(irls_tol);
    fconfig.track_train_loss = track_train_loss;
    fconfig.loss.type = static_cast<FactorNet::LossType>(loss_type);
    fconfig.loss.huber_delta = static_cast<float>(huber_delta);
    fconfig.loss.robust_delta = static_cast<float>(robust_delta);
    fconfig.gp_dispersion = static_cast<FactorNet::DispersionMode>(gp_dispersion_mode);
    fconfig.gp_theta_init = static_cast<float>(gp_theta_init);
    fconfig.gp_theta_max = static_cast<float>(gp_theta_max);
    fconfig.zi_mode = static_cast<FactorNet::ZIMode>(zi_mode);
    fconfig.zi_em_iters = zi_em_iters;
    fconfig.gp_theta_min = static_cast<float>(gp_theta_min);
    fconfig.nb_size_init = static_cast<float>(nb_size_init);
    fconfig.nb_size_max = static_cast<float>(nb_size_max);
    fconfig.nb_size_min = static_cast<float>(nb_size_min);
    fconfig.gamma_phi_init = static_cast<float>(gamma_phi_init);
    fconfig.gamma_phi_max = static_cast<float>(gamma_phi_max);
    fconfig.gamma_phi_min = static_cast<float>(gamma_phi_min);
    fconfig.loss.power_param = static_cast<float>(tweedie_power);
    fconfig.enable_profiling = enable_profiling;

    // Wire R callback if provided
    if (!Rf_isNull(on_iteration_r)) {
        Rcpp::Function r_fn(on_iteration_r);
        fconfig.on_iteration = [r_fn](int iter, float tl, float te) {
            r_fn(iter, static_cast<double>(tl), static_cast<double>(te));
        };
    }

    // Convert graphs to float (proper cast, not reinterpret_cast)
    Eigen::SparseMatrix<float> graph_W_f, graph_H_f;
    if (gd_W.valid) {
        graph_W_f = gd_W.matrix.cast<float>();
        fconfig.W.graph = &graph_W_f;
        fconfig.W.graph_lambda = static_cast<float>(lambda_W);
    }
    if (gd_H.valid) {
        graph_H_f = gd_H.matrix.cast<float>();
        fconfig.H.graph = &graph_H_f;
        fconfig.H.graph_lambda = static_cast<float>(lambda_H);
    }

    // Convert W_init to float
    Eigen::MatrixXf W_init_f;
    Eigen::MatrixXf* W_init_f_ptr = nullptr;
    if (W_init_mat.size() > 0) {
        W_init_f = W_init_mat.cast<float>();
        W_init_f_ptr = &W_init_f;
    }

    // Convert H_init to float
    Eigen::MatrixXf H_init_f;
    Eigen::MatrixXf* H_init_f_ptr = nullptr;
    if (H_init_mat.size() > 0) {
        H_init_f = H_init_mat.cast<float>();
        H_init_f_ptr = &H_init_f;
    }

    // Convert mask to float if present
    Eigen::SparseMatrix<float> mask_f;
    if (mask_sexp.isNotNull()) {
        Eigen::SparseMatrix<double> mask_d = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(mask_sexp));
        mask_f = mask_d.cast<float>();
        fconfig.mask = &mask_f;
    }


    // --- Parse target regularization ---
    Eigen::MatrixXf target_H_f;
    if (target_H_sexp.isNotNull()) {
        Eigen::MatrixXd target_H_d = Rcpp::as<Eigen::MatrixXd>(target_H_sexp);
        target_H_f = target_H_d.cast<float>();
        fconfig.H.target = &target_H_f;
    }
    if (target_lambda.isNotNull()) {
        Rcpp::NumericVector tl(target_lambda);
        if (tl.size() > 0)
            fconfig.W.target_lambda = static_cast<float>(tl[0]);
        if (tl.size() > 1)
            fconfig.H.target_lambda = static_cast<float>(tl[1]);
    }

    if (is_sparse) {
        auto A_mapped = mapSparseMatrix(Rcpp::S4(A_sexp));
        Eigen::SparseMatrix<float> A_f = A_mapped.cast<float>();
        auto result = FactorNet::nmf::nmf<float>(A_f, fconfig, W_init_f_ptr, H_init_f_ptr);
        return result_to_list(result);
    } else {
        Eigen::MatrixXd A_d = Rcpp::as<Eigen::MatrixXd>(A_sexp);
        Eigen::MatrixXf A_f = A_d.cast<float>();
        auto result = FactorNet::nmf::nmf<float>(A_f, fconfig, W_init_f_ptr, H_init_f_ptr);
        return result_to_list(result);
    }
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
    int init_mode = 0,
    bool projective = false,
    bool symmetric = false,
    bool enable_profiling = false)
{
    // Build unified NMFConfig<float> (Phase 1: replaces StreamingSpzConfig)
    FactorNet::NMFConfig<float> config;
    config.rank = k;
    config.tol = static_cast<float>(tol);
    config.max_iter = maxit;
    config.solver_mode = solver_mode;
    config.init_mode = init_mode;
    config.projective = projective;
    config.symmetric = symmetric;
    config.W.L1 = L1.size() > 0 ? static_cast<float>(L1[0]) : 0;
    config.H.L1 = L1.size() > 1 ? static_cast<float>(L1[1]) : config.W.L1;
    config.W.L2 = L2.size() > 0 ? static_cast<float>(L2[0]) : 0;
    config.H.L2 = L2.size() > 1 ? static_cast<float>(L2[1]) : config.W.L2;
    config.W.L21 = L21.size() > 0 ? static_cast<float>(L21[0]) : 0;
    config.H.L21 = L21.size() > 1 ? static_cast<float>(L21[1]) : config.W.L21;
    config.W.angular = angular.size() > 0 ? static_cast<float>(angular[0]) : 0;
    config.H.angular = angular.size() > 1 ? static_cast<float>(angular[1]) : config.W.angular;
    config.W.upper_bound = upper_bound.size() > 0 ? static_cast<float>(upper_bound[0]) : 0;
    config.H.upper_bound = upper_bound.size() > 1 ? static_cast<float>(upper_bound[1]) : config.W.upper_bound;
    config.W.nonneg = nonneg_w;
    config.H.nonneg = nonneg_h;
    config.cd_max_iter = cd_maxit;
    config.cd_tol = static_cast<float>(cd_tol);
    config.threads = threads;
    config.verbose = verbose;
    config.seed = seed;

    // Normalization type
    if (norm_str == "L2") {
        config.norm_type = FactorNet::NormType::L2;
    } else if (norm_str == "none") {
        config.norm_type = FactorNet::NormType::None;
    } else {
        config.norm_type = FactorNet::NormType::L1;
    }

    // Cross-validation config
    config.holdout_fraction = static_cast<float>(holdout_fraction);
    config.mask_zeros = mask_zeros;
    config.cv_seed = static_cast<uint32_t>(cv_seed);

    // User-provided NA mask — cast to float for NMFConfig<float>
    Eigen::SparseMatrix<float> mask_f;
    if (mask_sexp.isNotNull()) {
        Eigen::SparseMatrix<double> mask_d = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(mask_sexp));
        mask_f = mask_d.cast<float>();
        config.mask = &mask_f;
    }

    // Extract graphs and cast to float
    double gw_lambda = graph_lambda.size() > 0 ? graph_lambda[0] : 0.0;
    double gh_lambda = graph_lambda.size() > 1 ? graph_lambda[1] : 0.0;
    auto gd_W = extract_graph(graph_W_sexp, gw_lambda);
    auto gd_H = extract_graph(graph_H_sexp, gh_lambda);
    Eigen::SparseMatrix<float> graph_W_f, graph_H_f;
    if (gd_W.valid) {
        graph_W_f = gd_W.matrix.cast<float>();
        config.W.graph = &graph_W_f;
        config.W.graph_lambda = static_cast<float>(gw_lambda);
    }
    if (gd_H.valid) {
        graph_H_f = gd_H.matrix.cast<float>();
        config.H.graph = &graph_H_f;
        config.H.graph_lambda = static_cast<float>(gh_lambda);
    }

    config.enable_profiling = enable_profiling;

    // Loss type (was missing — would default to MSE if not set)
    {
        int loss_type = 0;
        if (loss_str == "mse") loss_type = 0;
        else if (loss_str == "mae") loss_type = 1;
        else if (loss_str == "huber") loss_type = 2;
        else if (loss_str == "kl") loss_type = 3;
        else if (loss_str == "gp") loss_type = 4;
        else if (loss_str == "nb") loss_type = 5;
        else if (loss_str == "gamma") loss_type = 6;
        else if (loss_str == "inverse_gaussian") loss_type = 7;
        else if (loss_str == "tweedie") loss_type = 8;
        config.loss.type = static_cast<FactorNet::LossType>(loss_type);
        config.loss.huber_delta = static_cast<float>(huber_delta);
    }

    // Always float (fp32-only)
    auto result = FactorNet::nmf::nmf_streaming_spz<float>(path, config);
    Rcpp::List ret = Rcpp::List::create(
        Rcpp::Named("w") = result.W.template cast<double>(),
        Rcpp::Named("d") = result.d.template cast<double>(),
        Rcpp::Named("h") = result.H.template cast<double>(),
        Rcpp::Named("loss") = static_cast<double>(result.train_loss),
        Rcpp::Named("test_loss") = static_cast<double>(result.test_loss),
        Rcpp::Named("iter") = result.iterations,
        Rcpp::Named("converged") = result.converged,
        Rcpp::Named("train_loss_history") = std::vector<double>(
            result.loss_history.begin(), result.loss_history.end()),
        Rcpp::Named("test_loss_history") = std::vector<double>(
            result.test_loss_history.begin(), result.test_loss_history.end())
    );
    if (!result.profile.empty()) {
        Rcpp::NumericVector prof_vec(result.profile.size());
        Rcpp::CharacterVector prof_names(result.profile.size());
        int idx = 0;
        for (const auto& kv : result.profile) {
            prof_names[idx] = kv.first;
            prof_vec[idx] = kv.second;
            idx++;
        }
        prof_vec.attr("names") = prof_names;
        ret["profile"] = prof_vec;
    }
    return ret;
}
