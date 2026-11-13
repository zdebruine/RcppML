// RcppFunctions_graph.cpp
// Rcpp exports for the FactorNet graph API.
// Parses an R graph descriptor into C++ FactorGraph, executes fit(),
// and returns results to R.
//
// fp32-only: all internal computation uses float; R boundary casts
// double→float on entry and float→double on exit.

#include "RcppFunctions_common.h"
#include "../inst/include/FactorNet/graph/graph_all.hpp"
#include "../inst/include/FactorNet/nmf/fit_streaming_spz.hpp"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>

using namespace FactorNet;
using namespace FactorNet::graph;

// ============================================================================
// Helpers: parse R list → C++ descriptor
// ============================================================================

namespace {

/// Parse a per-factor config from an R list
FactorDesc<float> parse_factor_desc(Rcpp::List fc, bool default_nonneg) {
    FactorDesc<float> desc;
    desc.L1 = fc.containsElementNamed("L1") ?
        static_cast<float>(Rcpp::as<double>(fc["L1"])) : 0.0f;
    desc.L2 = fc.containsElementNamed("L2") ?
        static_cast<float>(Rcpp::as<double>(fc["L2"])) : 0.0f;
    desc.L21 = fc.containsElementNamed("L21") ?
        static_cast<float>(Rcpp::as<double>(fc["L21"])) : 0.0f;
    desc.angular = fc.containsElementNamed("angular") ?
        static_cast<float>(Rcpp::as<double>(fc["angular"])) : 0.0f;
    desc.upper_bound = fc.containsElementNamed("upper_bound") ?
        static_cast<float>(Rcpp::as<double>(fc["upper_bound"])) : 0.0f;
    desc.nonneg = fc.containsElementNamed("nonneg") ?
        Rcpp::as<bool>(fc["nonneg"]) : default_nonneg;
    desc.graph_lambda = fc.containsElementNamed("graph_lambda") ?
        static_cast<float>(Rcpp::as<double>(fc["graph_lambda"])) : 0.0f;
    return desc;
}

/// Parse loss config from R list
LossConfig<float> parse_loss(const std::string& loss_str, double huber_delta,
                              double robust_delta = 0.0) {
    LossConfig<float> loss;
    if (loss_str == "mse") loss.type = LossType::MSE;
    else if (loss_str == "mae") loss.type = LossType::MAE;
    else if (loss_str == "huber") {
        loss.type = LossType::HUBER;
        loss.huber_delta = static_cast<float>(huber_delta);
    }
    else if (loss_str == "kl") loss.type = LossType::KL;
    else if (loss_str == "gp") loss.type = LossType::GP;
    else if (loss_str == "nb") loss.type = LossType::NB;
    else if (loss_str == "gamma") loss.type = LossType::GAMMA;
    else if (loss_str == "inverse_gaussian") loss.type = LossType::INVGAUSS;
    else if (loss_str == "tweedie") loss.type = LossType::TWEEDIE;
    else Rcpp::stop("Unknown loss type: '" + loss_str + "' (supported: mse, gp, nb, gamma, inverse_gaussian, tweedie)");
    loss.robust_delta = static_cast<float>(robust_delta);
    return loss;
}

/// Split "a,b,c" into its comma-separated names (surrounding blanks removed)
std::vector<std::string> split_csv_names(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        const size_t start = token.find_first_not_of(" \t");
        const size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos)
            out.push_back(token.substr(start, end - start + 1));
    }
    return out;
}

/// Read an optional target matrix and its strength from a per-factor R list
void parse_target(Rcpp::List fc, Eigen::MatrixXf& storage, FactorDesc<float>& desc) {
    if (!fc.containsElementNamed("target") || Rf_isNull(fc["target"])) return;
    Eigen::MatrixXd t = Rcpp::as<Eigen::MatrixXd>(fc["target"]);
    storage = t.cast<float>();
    desc.target = &storage;
    desc.target_lambda =
        (fc.containsElementNamed("target_lambda") && !Rf_isNull(fc["target_lambda"]))
        ? static_cast<float>(Rcpp::as<double>(fc["target_lambda"])) : 0.0f;
}

}  // anonymous namespace

// ============================================================================
// Main Rcpp export: fit a factor_net graph
// ============================================================================

// [[Rcpp::export]]
Rcpp::List Rcpp_factor_net_fit(Rcpp::List descriptor) {
    // -------------------------------------------------------------------
    // 1. Parse global config
    // -------------------------------------------------------------------
    Rcpp::List cfg = Rcpp::as<Rcpp::List>(descriptor["config"]);
    int max_iter = Rcpp::as<int>(cfg["maxit"]);
    double tol = Rcpp::as<double>(cfg["tol"]);
    bool verbose = cfg.containsElementNamed("verbose") ?
        Rcpp::as<bool>(cfg["verbose"]) : false;
    int threads = cfg.containsElementNamed("threads") ?
        Rcpp::as<int>(cfg["threads"]) : 0;
    uint32_t seed = cfg.containsElementNamed("seed") ?
        static_cast<uint32_t>(Rcpp::as<int>(cfg["seed"])) : 0u;
    int solver_mode = cfg.containsElementNamed("solver_mode") ?
        Rcpp::as<int>(cfg["solver_mode"]) : 1;
    std::string resource = cfg.containsElementNamed("resource") ?
        Rcpp::as<std::string>(cfg["resource"]) : "auto";
    std::string norm_str = cfg.containsElementNamed("norm") ?
        Rcpp::as<std::string>(cfg["norm"]) : "L1";
    std::string loss_str = cfg.containsElementNamed("loss") ?
        Rcpp::as<std::string>(cfg["loss"]) : "mse";
    double huber_delta = cfg.containsElementNamed("huber_delta") ?
        Rcpp::as<double>(cfg["huber_delta"]) : 1.0;

    // Cross-validation config
    double holdout_fraction = cfg.containsElementNamed("holdout_fraction") ?
        Rcpp::as<double>(cfg["holdout_fraction"]) : 0.0;
    uint32_t cv_seed_val = cfg.containsElementNamed("cv_seed") ?
        static_cast<uint32_t>(Rcpp::as<int>(cfg["cv_seed"])) : 0u;
    bool mask_zeros = cfg.containsElementNamed("mask_zeros") ?
        Rcpp::as<bool>(cfg["mask_zeros"]) : false;
    int cv_patience = cfg.containsElementNamed("cv_patience") ?
        Rcpp::as<int>(cfg["cv_patience"]) : 5;

    // -------------------------------------------------------------------
    // 2. Parse inputs — keep data in R-managed memory
    // -------------------------------------------------------------------
    Rcpp::List inputs_list = Rcpp::as<Rcpp::List>(descriptor["inputs"]);
    int n_inputs = inputs_list.size();

    // Storage for Eigen maps/copies (must outlive the graph)
    std::vector<Eigen::SparseMatrix<float>> sparse_storage(n_inputs);
    std::vector<Eigen::MatrixXf> dense_storage(n_inputs);

    GraphDesc<float> gdesc;
    gdesc.max_iter = max_iter;
    gdesc.tol = static_cast<float>(tol);
    gdesc.verbose = verbose;
    gdesc.threads = threads;
    gdesc.seed = seed;
    gdesc.solver_mode = solver_mode;
    gdesc.resource = resource;
    gdesc.norm = norm_str;
    gdesc.loss = parse_loss(loss_str, huber_delta);

    // Cross-validation
    gdesc.holdout_fraction = static_cast<float>(holdout_fraction);
    gdesc.cv_seed = cv_seed_val;
    gdesc.mask_zeros = mask_zeros;
    gdesc.cv_patience = cv_patience;

    for (int i = 0; i < n_inputs; ++i) {
        Rcpp::List inp = Rcpp::as<Rcpp::List>(inputs_list[i]);
        InputDesc<float> idesc;
        idesc.name = Rcpp::as<std::string>(inp["name"]);

        if (inp.containsElementNamed("spz_path") && !Rf_isNull(inp["spz_path"])) {
            idesc.spz_path = Rcpp::as<std::string>(inp["spz_path"]);
            idesc.is_sparse = true;
        } else {
            SEXP data_sexp = inp["data"];
            if (Rf_isS4(data_sexp)) {
                // Sparse matrix (dgCMatrix)
                auto mapped = mapSparseMatrix(Rcpp::S4(data_sexp));
                sparse_storage[i] = mapped.cast<float>();
                idesc.sparse_data = &sparse_storage[i];
                idesc.is_sparse = true;
            } else {
                // Dense matrix
                Eigen::MatrixXd dmat = Rcpp::as<Eigen::MatrixXd>(data_sexp);
                dense_storage[i] = dmat.cast<float>();
                idesc.dense_data = &dense_storage[i];
                idesc.is_sparse = false;
            }
        }
        gdesc.inputs.push_back(std::move(idesc));
    }

    // -------------------------------------------------------------------
    // 3. Parse conditions (if any)
    // -------------------------------------------------------------------
    if (descriptor.containsElementNamed("conditions") &&
        !Rf_isNull(descriptor["conditions"])) {
        Rcpp::List conds = Rcpp::as<Rcpp::List>(descriptor["conditions"]);
        for (int i = 0; i < conds.size(); ++i) {
            Rcpp::List c = Rcpp::as<Rcpp::List>(conds[i]);
            ConditionDesc<float> cdesc;
            cdesc.name = Rcpp::as<std::string>(c["name"]);
            cdesc.input_ref = Rcpp::as<std::string>(c["input_ref"]);
            Eigen::MatrixXd Z_d = Rcpp::as<Eigen::MatrixXd>(c["Z"]);
            cdesc.Z = Z_d.cast<float>();
            gdesc.conditions.push_back(std::move(cdesc));
        }
    }

    // -------------------------------------------------------------------
    // 4. Parse layers
    // -------------------------------------------------------------------
    Rcpp::List layers_list = Rcpp::as<Rcpp::List>(descriptor["layers"]);
    // Storage for graph Laplacians (must outlive the graph)
    std::vector<Eigen::SparseMatrix<float>> w_graph_storage(layers_list.size());
    std::vector<Eigen::SparseMatrix<float>> h_graph_storage(layers_list.size());
    // Storage for target matrices (must outlive the graph; sized up front so
    // that the addresses handed to the layer descriptors stay valid)
    std::vector<Eigen::MatrixXf> w_target_storage(layers_list.size());
    std::vector<Eigen::MatrixXf> h_target_storage(layers_list.size());

    for (int i = 0; i < layers_list.size(); ++i) {
        Rcpp::List layer = Rcpp::as<Rcpp::List>(layers_list[i]);
        LayerDesc<float> ldesc;
        ldesc.name = Rcpp::as<std::string>(layer["name"]);
        ldesc.type = Rcpp::as<std::string>(layer["type"]);
        ldesc.k = Rcpp::as<int>(layer["k"]);
        ldesc.input_ref = Rcpp::as<std::string>(layer["input_ref"]);

        bool default_nonneg = (ldesc.type == "nmf");

        if (layer.containsElementNamed("W_config") && !Rf_isNull(layer["W_config"])) {
            Rcpp::List wc = Rcpp::as<Rcpp::List>(layer["W_config"]);
            ldesc.W_config = parse_factor_desc(wc, default_nonneg);
            // Handle graph Laplacian for W
            if (wc.containsElementNamed("graph") && !Rf_isNull(wc["graph"])) {
                Rcpp::S4 graph_s4(wc["graph"]);
                auto gd = Rcpp::as<Eigen::SparseMatrix<double>>(graph_s4);
                w_graph_storage[i] = gd.cast<float>();
                ldesc.W_config.graph = &w_graph_storage[i];
            }
            parse_target(wc, w_target_storage[i], ldesc.W_config);
        } else {
            ldesc.W_config.nonneg = default_nonneg;
        }

        if (layer.containsElementNamed("H_config") && !Rf_isNull(layer["H_config"])) {
            Rcpp::List hc = Rcpp::as<Rcpp::List>(layer["H_config"]);
            ldesc.H_config = parse_factor_desc(hc, default_nonneg);
            // Handle graph Laplacian for H
            if (hc.containsElementNamed("graph") && !Rf_isNull(hc["graph"])) {
                Rcpp::S4 graph_s4(hc["graph"]);
                auto gd = Rcpp::as<Eigen::SparseMatrix<double>>(graph_s4);
                h_graph_storage[i] = gd.cast<float>();
                ldesc.H_config.graph = &h_graph_storage[i];
            }
            parse_target(hc, h_target_storage[i], ldesc.H_config);
        } else {
            ldesc.H_config.nonneg = default_nonneg;
        }

        gdesc.layers.push_back(std::move(ldesc));
    }

    // -------------------------------------------------------------------
    // 5b. Parse W_init if provided (R-generated initialization)
    // -------------------------------------------------------------------
    Eigen::MatrixXf w_init_f;
    Eigen::MatrixXf* w_init_ptr = nullptr;
    if (descriptor.containsElementNamed("w_init") &&
        !Rf_isNull(descriptor["w_init"])) {
        Eigen::MatrixXd w_d = Rcpp::as<Eigen::MatrixXd>(descriptor["w_init"]);
        w_init_f = w_d.cast<float>();
        w_init_ptr = &w_init_f;
    }

    // -------------------------------------------------------------------
    // 6. Build the C++ graph and fit
    // -------------------------------------------------------------------
    auto instance = GraphInstance<float>::build(gdesc);
    auto& graph = instance.graph();

    // --- Refuse graphs that the engine cannot execute faithfully ---
    {
        std::set<std::string> distinct_inputs;
        for (const auto& idesc : gdesc.inputs) distinct_inputs.insert(idesc.name);
        if (graph.n_layers() > 1 && distinct_inputs.size() > 1)
            Rcpp::stop("Multi-layer factor networks with more than one input are not "
                       "supported: every branch would be fit to the first input.");
        if (graph.n_layers() == 1 && !gdesc.conditions.empty())
            Rcpp::stop("factor_condition() directly on an input is not supported.");
    }

    bool any_target = false;
    for (const auto& ld : gdesc.layers)
        if (ld.W_config.target || ld.H_config.target) any_target = true;
    if (any_target && (graph.n_layers() != 1 || instance.is_streaming()))
        Rcpp::stop("Target regularization in a factor network is supported for "
                   "single-layer networks on in-memory data only.");

    // Dimension checks for the single-layer path (m x n is the matrix that
    // the layer factorizes; for shared inputs m is the total row count)
    auto check_single_layer_dims = [&](int m, int n) {
        if (graph.n_layers() != 1) return;
        const auto& l0 = gdesc.layers[0];
        if (l0.H_config.target &&
            (l0.H_config.target->rows() != l0.k || l0.H_config.target->cols() != n))
            Rcpp::stop("H(target) must be a k x n matrix (%d x %d)", l0.k, n);
        if (l0.W_config.target &&
            (l0.W_config.target->rows() != l0.k || l0.W_config.target->cols() != m))
            Rcpp::stop("W(target) must be a k x m matrix (%d x %d)", l0.k, m);
        if (w_init_ptr && (w_init_ptr->rows() != m || w_init_ptr->cols() != l0.k))
            Rcpp::stop("internal error: w_init is %d x %d but the layer input has %d rows and rank %d",
                       static_cast<int>(w_init_ptr->rows()), static_cast<int>(w_init_ptr->cols()),
                       m, l0.k);
    };

    GraphResult<float> result;

    if (instance.is_streaming()) {
        // Streaming SPZ path — single layer only
        if (graph.n_layers() != 1)
            Rcpp::stop("Streaming SPZ only supports single-layer graphs");
        NMFConfig<float> scfg = graph.build_layer_config(graph.layers()[0]);
        scfg.verbose = verbose;
        scfg.sort_model = true;
        auto nmf_result = nmf::nmf_streaming_spz<float>(
            instance.streaming_path(), scfg);

        LayerResult<float> lr;
        lr.W = std::move(nmf_result.W);
        lr.d = std::move(nmf_result.d);
        lr.H = std::move(nmf_result.H);
        lr.iterations = nmf_result.iterations;
        lr.loss = nmf_result.train_loss;
        lr.test_loss = nmf_result.test_loss;
        lr.best_test_loss = nmf_result.best_test_loss;
        lr.converged = nmf_result.converged;

        result.layers[graph.layers()[0]->name] = std::move(lr);
        result.total_iterations = nmf_result.iterations;
        result.total_loss = nmf_result.train_loss;
        result.converged = nmf_result.converged;
    } else if (instance.has_concatenated()) {
        // Multi-modal shared-H path
        if (instance.concat_is_sparse()) {
            check_single_layer_dims(static_cast<int>(instance.concatenated_sparse().rows()),
                                    static_cast<int>(instance.concatenated_sparse().cols()));
            result = fit(graph, instance.concatenated_sparse(), w_init_ptr);
        } else {
            check_single_layer_dims(static_cast<int>(instance.concatenated_data().rows()),
                                    static_cast<int>(instance.concatenated_data().cols()));
            result = fit(graph, instance.concatenated_data(), w_init_ptr);
        }
    } else if (instance.first_input_is_sparse()) {
        check_single_layer_dims(static_cast<int>(instance.first_sparse().rows()),
                                static_cast<int>(instance.first_sparse().cols()));
        result = fit(graph, instance.first_sparse(), w_init_ptr);
    } else {
        check_single_layer_dims(static_cast<int>(instance.first_dense().rows()),
                                static_cast<int>(instance.first_dense().cols()));
        result = fit(graph, instance.first_dense(), w_init_ptr);
    }

    // -------------------------------------------------------------------
    // 7. Package results as R list
    // -------------------------------------------------------------------
    // Layers are returned in declared (topological) order. GraphResult keeps
    // them in a std::map, whose iteration order is alphabetical by name.
    Rcpp::List layers_result;
    Rcpp::CharacterVector layer_names;
    bool is_multi_modal = false;
    bool first_layer = true;

    for (const auto& ldesc_out : gdesc.layers) {
        auto found = result.layers.find(ldesc_out.name);
        if (found == result.layers.end()) continue;
        const std::string& name = found->first;
        const LayerResult<float>& lr = found->second;

        Eigen::MatrixXd W_full = lr.W.template cast<double>();

        Rcpp::List lr_list = Rcpp::List::create(
            Rcpp::Named("W") = W_full,
            Rcpp::Named("d") = lr.d.template cast<double>(),
            Rcpp::Named("H") = lr.H.template cast<double>(),
            Rcpp::Named("iterations") = lr.iterations,
            Rcpp::Named("loss") = static_cast<double>(lr.loss),
            Rcpp::Named("test_loss") = static_cast<double>(lr.test_loss),
            Rcpp::Named("best_test_loss") = static_cast<double>(lr.best_test_loss),
            Rcpp::Named("converged") = lr.converged
        );

        // Multi-modal: split the rows of the concatenated W back into one
        // block per input, in CONCATENATION order (the order of the names in
        // the "shared:a,b,..." reference), not in std::map (alphabetical) order.
        if (first_layer && instance.has_concatenated() &&
            ldesc_out.input_ref.size() > 7 &&
            ldesc_out.input_ref.substr(0, 7) == "shared:") {
            const std::vector<std::string> shared_names =
                split_csv_names(ldesc_out.input_ref.substr(7));
            const auto& row_counts = instance.input_row_counts();
            const int n_shared = static_cast<int>(shared_names.size());

            Rcpp::List splits(n_shared);
            Rcpp::CharacterVector split_names(n_shared);
            int row_offset = 0;
            for (int si = 0; si < n_shared; ++si) {
                auto rc = row_counts.find(shared_names[si]);
                if (rc == row_counts.end())
                    Rcpp::stop("internal error: shared input '" + shared_names[si] + "' not found");
                const int nr = rc->second;
                if (row_offset + nr > static_cast<int>(W_full.rows()))
                    Rcpp::stop("internal error: shared inputs have more rows than W");
                Eigen::MatrixXd block = W_full.middleRows(row_offset, nr);
                splits[si] = Rcpp::wrap(block);
                split_names[si] = shared_names[si];
                row_offset += nr;
            }
            if (row_offset != static_cast<int>(W_full.rows()))
                Rcpp::stop("internal error: rows of W do not match the shared inputs");
            splits.attr("names") = split_names;
            lr_list.push_back(splits, "W_splits");
            is_multi_modal = true;
        }

        layers_result.push_back(lr_list);
        layer_names.push_back(name);
        first_layer = false;
    }
    layers_result.attr("names") = layer_names;

    return Rcpp::List::create(
        Rcpp::Named("layers") = layers_result,
        Rcpp::Named("total_iterations") = result.total_iterations,
        Rcpp::Named("total_loss") = static_cast<double>(result.total_loss),
        Rcpp::Named("converged") = result.converged,
        Rcpp::Named("multi_modal") = is_multi_modal
    );
}
