/**
 * @file graph/builder.hpp
 * @brief Graph builder — constructs FactorGraph from a flat descriptor
 *
 * Parses a serialized graph description (from R or any external caller)
 * into a concrete FactorGraph ready for compilation and execution.
 *
 * The builder owns all node objects and the graph itself, managing their
 * lifetimes through the GraphInstance wrapper.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/graph/node.hpp>
#include <FactorNet/graph/graph.hpp>
#include <FactorNet/graph/result.hpp>
#include <FactorNet/graph/fit.hpp>
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/types.hpp>
#include <FactorNet/math/loss.hpp>

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace FactorNet {
namespace graph {

// ============================================================================
// Descriptor types — plain data structures for graph specification
// ============================================================================

/**
 * @brief Descriptor for a per-factor configuration (W or H side)
 */
template<typename Scalar>
struct FactorDesc {
    Scalar L1 = 0;
    Scalar L2 = 0;
    Scalar L21 = 0;
    Scalar angular = 0;
    Scalar upper_bound = 0;
    bool nonneg = true;
    Scalar graph_lambda = 0;
    const SparseMatrix<Scalar>* graph = nullptr;
    // Target regularization handled at Rcpp level
};

/**
 * @brief Descriptor for a data input
 */
template<typename Scalar>
struct InputDesc {
    std::string name;
    const SparseMatrix<Scalar>* sparse_data = nullptr;   ///< CSC sparse matrix
    const DenseMatrix<Scalar>*  dense_data  = nullptr;    ///< Dense matrix
    std::string spz_path;                                  ///< Path for streaming
    bool is_sparse = true;

    int rows() const {
        if (sparse_data) return static_cast<int>(sparse_data->rows());
        if (dense_data)  return static_cast<int>(dense_data->rows());
        return 0;
    }
    int cols() const {
        if (sparse_data) return static_cast<int>(sparse_data->cols());
        if (dense_data)  return static_cast<int>(dense_data->cols());
        return 0;
    }
};

/**
 * @brief Descriptor for a layer (NMF or SVD)
 */
template<typename Scalar>
struct LayerDesc {
    std::string name;
    std::string type;       ///< "nmf" or "svd"
    int k = 10;             ///< Factorization rank
    std::string input_ref;  ///< Reference to input: "input:NAME", "layer:NAME",
                            ///< "shared:A,B", "concat:A,B", "add:A,B", "cond:LAYER"
    FactorDesc<Scalar> W_config;
    FactorDesc<Scalar> H_config;
};

/**
 * @brief Descriptor for a conditioner (appends Z to H)
 */
template<typename Scalar>
struct ConditionDesc {
    std::string name;
    std::string input_ref;          ///< Layer whose H gets augmented
    DenseMatrix<Scalar> Z;          ///< Conditioning matrix (p × n)
};

/**
 * @brief Complete graph descriptor
 */
template<typename Scalar>
struct GraphDesc {
    std::vector<InputDesc<Scalar>> inputs;
    std::vector<LayerDesc<Scalar>> layers;
    std::vector<ConditionDesc<Scalar>> conditions;

    // Global config
    int max_iter = 100;
    Scalar tol = static_cast<Scalar>(1e-4);
    bool verbose = false;
    int threads = 0;
    uint32_t seed = 0;
    int solver_mode = 1;
    std::string resource = "auto";
    std::string norm = "L1";
    LossConfig<Scalar> loss;

    // Cross-validation
    Scalar holdout_fraction = 0;
    uint32_t cv_seed = 0;
    bool mask_zeros = false;
    int cv_patience = 5;
};

// ============================================================================
// GraphInstance — owns all nodes and the graph, manages lifetimes
// ============================================================================

/**
 * @brief Self-contained graph instance with owned nodes
 *
 * All nodes are heap-allocated and managed by unique_ptr. The
 * FactorGraph holds raw pointers that remain valid for the lifetime
 * of this instance.
 */
template<typename Scalar>
class GraphInstance {
public:
    /// Build from a descriptor
    static GraphInstance build(const GraphDesc<Scalar>& desc) {
        GraphInstance inst;
        inst.build_impl(desc);
        return inst;
    }

    /// Access the compiled graph
    FactorGraph<Scalar>& graph() { return *graph_; }
    const FactorGraph<Scalar>& graph() const { return *graph_; }

    /// Get the primary data matrix (for single-input graphs)
    /// Returns the first input's data as a reference
    bool first_input_is_sparse() const { return first_sparse_ != nullptr; }
    const SparseMatrix<Scalar>& first_sparse() const { return *first_sparse_; }
    const DenseMatrix<Scalar>& first_dense() const { return *first_dense_; }

    /// Get the concatenated data for shared-H (multi-modal)
    bool has_concatenated() const { return concat_data_.rows() > 0; }
    const DenseMatrix<Scalar>& concatenated_data() const { return concat_data_; }
    const SparseMatrix<Scalar>& concatenated_sparse() const { return concat_sparse_; }
    bool concat_is_sparse() const { return concat_is_sparse_; }

    /// Input name → row split sizes (for multi-modal W splits)
    const std::map<std::string, int>& input_row_counts() const { return input_row_counts_; }

    /// Whether first layer is streaming
    bool is_streaming() const { return !streaming_path_.empty(); }
    const std::string& streaming_path() const { return streaming_path_; }

private:
    // Owned storage
    std::vector<std::unique_ptr<Node<Scalar>>> nodes_;
    std::unique_ptr<FactorGraph<Scalar>> graph_;

    // Data references (not owned — caller must keep alive)
    const SparseMatrix<Scalar>* first_sparse_ = nullptr;
    const DenseMatrix<Scalar>* first_dense_ = nullptr;
    DenseMatrix<Scalar> concat_data_;
    SparseMatrix<Scalar> concat_sparse_;
    bool concat_is_sparse_ = false;
    std::map<std::string, int> input_row_counts_;
    std::string streaming_path_;

    // Name → node lookup
    std::map<std::string, Node<Scalar>*> node_map_;

    void build_impl(const GraphDesc<Scalar>& desc) {
        // --- 1. Create input nodes ---
        for (const auto& inp_desc : desc.inputs) {
            Node<Scalar>* inp_node;
            if (inp_desc.sparse_data) {
                auto node = std::make_unique<InputNode<Scalar, SparseMatrix<Scalar>>>(
                    *inp_desc.sparse_data, inp_desc.name);
                inp_node = node.get();
                if (!first_sparse_) first_sparse_ = inp_desc.sparse_data;
                nodes_.push_back(std::move(node));
            } else if (inp_desc.dense_data) {
                auto node = std::make_unique<InputNode<Scalar, DenseMatrix<Scalar>>>(
                    *inp_desc.dense_data, inp_desc.name);
                inp_node = node.get();
                if (!first_dense_) first_dense_ = inp_desc.dense_data;
                nodes_.push_back(std::move(node));
            } else if (!inp_desc.spz_path.empty()) {
                // Streaming: create a placeholder input node
                // The actual data loading happens in fit_streaming
                streaming_path_ = inp_desc.spz_path;
                // Create a dummy dense input — streaming path handles data
                auto node = std::make_unique<InputNode<Scalar, DenseMatrix<Scalar>>>(
                    placeholder_dense_, inp_desc.name);
                inp_node = node.get();
                nodes_.push_back(std::move(node));
            } else {
                throw std::invalid_argument(
                    "Input '" + inp_desc.name + "' has no data");
            }
            node_map_["input:" + inp_desc.name] = inp_node;
            input_row_counts_[inp_desc.name] = inp_desc.rows();
        }

        // --- 2. Create layers and conditions in topological order ---
        // Conditions may reference layers (e.g., "layer:enc"), so we must
        // create each condition just before the layer that needs it.
        // Layers are in topological order from the R descriptor.
        // Build a lookup of conditions by name for on-demand creation.
        std::map<std::string, const ConditionDesc<Scalar>*> cond_lookup;
        for (const auto& cond_desc : desc.conditions) {
            cond_lookup[cond_desc.name] = &cond_desc;
        }

        auto ensure_condition = [&](const std::string& cond_name) {
            // Create condition node if not already created
            std::string key = "cond:" + cond_name;
            if (node_map_.count(key)) return;

            auto it = cond_lookup.find(cond_name);
            if (it == cond_lookup.end())
                throw std::invalid_argument("Condition not found: " + cond_name);

            const auto& cond_desc = *(it->second);
            Node<Scalar>* upstream = resolve_ref(cond_desc.input_ref);
            auto node = std::make_unique<ConditionNode<Scalar>>(upstream, cond_desc.Z);
            node->name = cond_desc.name;
            node_map_[key] = node.get();
            nodes_.push_back(std::move(node));
        };

        Node<Scalar>* last_layer = nullptr;
        for (const auto& layer_desc : desc.layers) {
            // If this layer's input references a condition, create it first
            if (layer_desc.input_ref.substr(0, 5) == "cond:") {
                std::string cond_name = layer_desc.input_ref.substr(5);
                ensure_condition(cond_name);
            }

            Node<Scalar>* input = resolve_input_ref(layer_desc.input_ref);

            if (layer_desc.type == "nmf") {
                auto node = std::make_unique<NMFLayerNode<Scalar>>(
                    input, layer_desc.k, layer_desc.name);
                apply_factor_desc(node->W_config, layer_desc.W_config);
                apply_factor_desc(node->H_config, layer_desc.H_config);
                last_layer = node.get();
                node_map_["layer:" + layer_desc.name] = node.get();
                nodes_.push_back(std::move(node));
            } else if (layer_desc.type == "svd") {
                auto node = std::make_unique<SVDLayerNode<Scalar>>(
                    input, layer_desc.k, layer_desc.name);
                apply_factor_desc(node->W_config, layer_desc.W_config);
                apply_factor_desc(node->H_config, layer_desc.H_config);
                // SVD defaults: nonneg = false
                node->W_config.nonneg = layer_desc.W_config.nonneg;
                node->H_config.nonneg = layer_desc.H_config.nonneg;
                last_layer = node.get();
                node_map_["layer:" + layer_desc.name] = node.get();
                nodes_.push_back(std::move(node));
            } else {
                throw std::invalid_argument(
                    "Unknown layer type: " + layer_desc.type);
            }
        }

        if (!last_layer)
            throw std::invalid_argument("Graph has no layers");

        // --- 4. Collect input node pointers ---
        std::vector<Node<Scalar>*> input_ptrs;
        for (const auto& inp_desc : desc.inputs) {
            input_ptrs.push_back(node_map_.at("input:" + inp_desc.name));
        }

        // --- 5. Build the FactorGraph ---
        graph_ = std::make_unique<FactorGraph<Scalar>>(input_ptrs, last_layer);

        // Apply global config
        graph_->max_iter = desc.max_iter;
        graph_->tol = desc.tol;
        graph_->verbose = desc.verbose;
        graph_->threads = desc.threads;
        graph_->seed = desc.seed;
        graph_->solver_mode = desc.solver_mode;
        graph_->resource = desc.resource;
        graph_->loss = desc.loss;

        // Cross-validation settings
        graph_->holdout_fraction = desc.holdout_fraction;
        graph_->cv_seed = desc.cv_seed;
        graph_->mask_zeros = desc.mask_zeros;
        graph_->cv_patience = desc.cv_patience;

        if (desc.norm == "L2") graph_->norm_type = NormType::L2;
        else if (desc.norm == "none") graph_->norm_type = NormType::None;
        else graph_->norm_type = NormType::L1;

        // --- 6. Compile ---
        graph_->compile();

        // --- 7. Build concatenated data for shared-H ---
        build_concatenated_if_shared(desc);
    }

    /// Resolve a node reference like "input:RNA" or "layer:L1"
    Node<Scalar>* resolve_ref(const std::string& ref) {
        auto it = node_map_.find(ref);
        if (it != node_map_.end()) return it->second;
        throw std::invalid_argument("Node reference not found: " + ref);
    }

    /// Resolve an input_ref for a layer, handling shared/concat/add/cond
    Node<Scalar>* resolve_input_ref(const std::string& ref) {
        // "input:NAME" or "layer:NAME" or "cond:NAME"
        if (ref.substr(0, 6) == "input:" || ref.substr(0, 6) == "layer:" ||
            ref.substr(0, 5) == "cond:") {
            return resolve_ref(ref);
        }

        // "shared:A,B,C" — create SharedNode on the fly
        if (ref.substr(0, 7) == "shared:") {
            auto names = split_names(ref.substr(7));
            std::vector<Node<Scalar>*> inputs;
            for (const auto& nm : names) {
                inputs.push_back(resolve_ref("input:" + nm));
            }
            auto node = std::make_unique<SharedNode<Scalar>>(inputs);
            Node<Scalar>* ptr = node.get();
            node_map_[ref] = ptr;
            nodes_.push_back(std::move(node));
            return ptr;
        }

        // "concat:L1,L2" — create ConcatNode
        if (ref.substr(0, 7) == "concat:") {
            auto names = split_names(ref.substr(7));
            std::vector<Node<Scalar>*> inputs;
            for (const auto& nm : names) {
                inputs.push_back(resolve_ref("layer:" + nm));
            }
            auto node = std::make_unique<ConcatNode<Scalar>>(inputs);
            Node<Scalar>* ptr = node.get();
            node_map_[ref] = ptr;
            nodes_.push_back(std::move(node));
            return ptr;
        }

        // "add:L1,L2" — create AddNode
        if (ref.substr(0, 4) == "add:") {
            auto names = split_names(ref.substr(4));
            std::vector<Node<Scalar>*> inputs;
            for (const auto& nm : names) {
                inputs.push_back(resolve_ref("layer:" + nm));
            }
            auto node = std::make_unique<AddNode<Scalar>>(inputs);
            Node<Scalar>* ptr = node.get();
            node_map_[ref] = ptr;
            nodes_.push_back(std::move(node));
            return ptr;
        }

        throw std::invalid_argument("Unknown input_ref format: " + ref);
    }

    void apply_factor_desc(FactorConfig<Scalar>& cfg, const FactorDesc<Scalar>& desc) {
        cfg.L1 = desc.L1;
        cfg.L2 = desc.L2;
        cfg.L21 = desc.L21;
        cfg.angular = desc.angular;
        cfg.upper_bound = desc.upper_bound;
        cfg.nonneg = desc.nonneg;
        cfg.graph_lambda = desc.graph_lambda;
        if (desc.graph) cfg.graph = desc.graph;
    }

    std::vector<std::string> split_names(const std::string& s) {
        std::vector<std::string> result;
        std::istringstream iss(s);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos)
                result.push_back(token.substr(start, end - start + 1));
        }
        return result;
    }

    void build_concatenated_if_shared(const GraphDesc<Scalar>& desc) {
        if (desc.inputs.size() < 2) return;

        // Check if first layer references a shared node
        if (desc.layers.empty()) return;
        const std::string& ref = desc.layers[0].input_ref;
        if (ref.substr(0, 7) != "shared:") return;

        // Build concatenated matrix
        auto names = split_names(ref.substr(7));
        bool all_sparse = true;
        int total_rows = 0;
        int n_cols = -1;

        for (const auto& nm : names) {
            auto it = std::find_if(desc.inputs.begin(), desc.inputs.end(),
                [&](const InputDesc<Scalar>& d) { return d.name == nm; });
            if (it == desc.inputs.end()) continue;
            total_rows += it->rows();
            if (n_cols < 0) {
                n_cols = it->cols();
            } else if (it->cols() != n_cols) {
                throw std::invalid_argument(
                    "All shared inputs must have same number of columns (samples). "
                    "Got " + std::to_string(n_cols) + " vs " + std::to_string(it->cols()));
            }
            if (!it->sparse_data) all_sparse = false;
        }

        if (all_sparse) {
            // Row-concatenate sparse matrices
            concat_is_sparse_ = true;
            std::vector<Eigen::Triplet<Scalar>> trips;
            int row_offset = 0;
            for (const auto& nm : names) {
                auto it = std::find_if(desc.inputs.begin(), desc.inputs.end(),
                    [&](const InputDesc<Scalar>& d) { return d.name == nm; });
                if (it == desc.inputs.end()) continue;
                const auto& sp = *it->sparse_data;
                for (int j = 0; j < sp.outerSize(); ++j) {
                    for (typename SparseMatrix<Scalar>::InnerIterator iter(sp, j); iter; ++iter) {
                        trips.emplace_back(iter.row() + row_offset, j, iter.value());
                    }
                }
                row_offset += static_cast<int>(sp.rows());
            }
            concat_sparse_.resize(total_rows, n_cols);
            concat_sparse_.setFromTriplets(trips.begin(), trips.end());
            concat_sparse_.makeCompressed();
        } else {
            // Dense concatenation
            concat_is_sparse_ = false;
            concat_data_.resize(total_rows, n_cols);
            int row_offset = 0;
            for (const auto& nm : names) {
                auto it = std::find_if(desc.inputs.begin(), desc.inputs.end(),
                    [&](const InputDesc<Scalar>& d) { return d.name == nm; });
                if (it == desc.inputs.end()) continue;
                int nr = it->rows();
                if (it->sparse_data) {
                    concat_data_.middleRows(row_offset, nr) =
                        DenseMatrix<Scalar>(*it->sparse_data);
                } else {
                    concat_data_.middleRows(row_offset, nr) = *it->dense_data;
                }
                row_offset += nr;
            }
        }
    }

    DenseMatrix<Scalar> placeholder_dense_;
};

}  // namespace graph
}  // namespace FactorNet
