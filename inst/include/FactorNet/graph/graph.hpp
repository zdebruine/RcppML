// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file graph/graph.hpp
 * @brief FactorGraph — compiled computation graph for multi-layer factorization
 *
 * A FactorGraph is the central abstraction of the FactorNet library.
 * It represents a directed acyclic graph of factorization layers
 * (NMF, SVD) connected by combinators (shared, concat, add, condition).
 *
 * Workflow:
 *   1. Create nodes (InputNode, NMFLayerNode, etc.)
 *   2. Wire them together via pointers
 *   3. Construct a FactorGraph with inputs and output
 *   4. Call compile() to validate topology and build execution plan
 *   5. Call fit() to execute the graph
 *
 * Example — single-layer NMF:
 * @code
 *   Eigen::SparseMatrix<float> A = load_data();
 *   InputNode<float, SparseMatrix<float>> inp(A, "X");
 *   NMFLayerNode<float> layer(&inp, 20, "L1");
 *   layer.H_config.L1 = 0.01;  // sparse H
 *
 *   FactorGraph<float> net({&inp}, &layer);
 *   net.max_iter = 100;
 *   net.tol = 1e-5;
 *   net.compile();
 *   auto result = net.fit();
 *   // result["L1"].W, .d, .H
 * @endcode
 *
 * Example — deep NMF (hierarchical):
 * @code
 *   InputNode<float, SparseMatrix<float>> inp(A, "X");
 *   NMFLayerNode<float> L1(&inp, 64, "encoder");
 *   NMFLayerNode<float> L2(&L1, 10, "bottleneck");
 *
 *   FactorGraph<float> net({&inp}, &L2);
 *   net.max_iter = 200;
 *   net.compile();
 *   auto result = net.fit();
 *   // result["encoder"].H → 64×n intermediate
 *   // result["bottleneck"].H → 10×n final embedding
 * @endcode
 *
 * Example — multi-modal (shared H):
 * @code
 *   InputNode<float, SparseMatrix<float>> rna(RNA, "RNA");
 *   InputNode<float, SparseMatrix<float>> atac(ATAC, "ATAC");
 *   SharedNode<float> shared({&rna, &atac});
 *   NMFLayerNode<float> layer(&shared, 20, "joint");
 *
 *   FactorGraph<float> net({&rna, &atac}, &layer);
 *   net.compile();
 *   auto result = net.fit();
 *   // result["joint"].W_splits["RNA"], .W_splits["ATAC"]
 *   // result["joint"].H → shared embedding
 * @endcode
 *
 * Example — branched network with concat:
 * @code
 *   InputNode<float, SparseMatrix<float>> inp(A, "X");
 *   NMFLayerNode<float> branch1(&inp, 32, "B1");
 *   SVDLayerNode<float> branch2(&inp, 16, "B2");
 *   ConcatNode<float> merged({&branch1, &branch2});
 *   NMFLayerNode<float> final(&merged, 10, "final");
 *
 *   FactorGraph<float> net({&inp}, &final);
 *   net.compile();
 *   auto result = net.fit();
 *   // result["final"].H → 10×n, fed by 48-dim concat of B1+B2
 * @endcode
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/graph/node.hpp>
#include <FactorNet/graph/result.hpp>
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/constants.hpp>

#include <vector>
#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace FactorNet {
namespace graph {

/**
 * @brief Compiled factorization graph — the primary FactorNet abstraction
 *
 * Owns the execution plan (layer ordering, topology validation) but does
 * NOT own the node objects — the caller must keep them alive for the
 * lifetime of the graph.
 *
 * Template parameter Scalar determines the arithmetic precision (float
 * or double) for all inner computation.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
class FactorGraph {
public:
    // ====================================================================
    // Global configuration (applies to all layers unless overridden)
    // ====================================================================

    int max_iter = NMF_MAXIT;               ///< Maximum outer iterations
    Scalar tol = static_cast<Scalar>(NMF_TOL);  ///< Convergence tolerance
    bool verbose = false;                    ///< Print per-iteration diagnostics
    int threads = 0;                         ///< OpenMP thread limit (0 = all)
    uint32_t seed = 0;                       ///< Random seed for initialization

    /// Loss function type (MSE, KL, Poisson, etc.)
    LossConfig<Scalar> loss;

    /// Normalization type for diagonal scaling
    NormType norm_type = NormType::L1;

    /// NNLS solver mode: 0 = CD, 1 = Cholesky+clip
    int solver_mode = 1;

    /// Resource override: "auto", "cpu", "gpu"
    std::string resource = "auto";

    // ====================================================================
    // Cross-validation (graph-level)
    // ====================================================================

    /// Fraction of entries held out for test set (0 = no CV)
    Scalar holdout_fraction = 0;

    /// Separate seed for CV mask (0 = derive from global seed)
    uint32_t cv_seed = 0;

    /// mask_zeros=true: test set is held-out nonzeros only
    bool mask_zeros = false;

    /// Early-stopping patience for CV
    int cv_patience = 5;

    // ====================================================================
    // Construction
    // ====================================================================

    /**
     * @brief Construct a FactorGraph from input nodes and an output node
     *
     * @param inputs  Vector of input node pointers (graph entry points)
     * @param output  The output node (graph exit point, typically the final layer)
     */
    FactorGraph(std::vector<Node<Scalar>*> inputs, Node<Scalar>* output)
        : inputs_(std::move(inputs)), output_(output)
    {
        if (inputs_.empty())
            throw std::invalid_argument("FactorGraph: at least one input required");
        if (!output_)
            throw std::invalid_argument("FactorGraph: output must not be null");
        for (auto* inp : inputs_) {
            if (!inp || inp->type != NodeType::INPUT)
                throw std::invalid_argument("FactorGraph: all inputs must be InputNode");
        }
    }

    // ====================================================================
    // Compilation
    // ====================================================================

    /**
     * @brief Validate topology and build execution plan
     *
     * Walks the graph from output backward to collect all layer nodes
     * in topological order. Validates that:
     *   - All declared inputs appear in the graph
     *   - Layer ranks are positive
     *   - Combinator nodes have valid children
     *   - No cycles exist (DAG property)
     *
     * Must be called before fit().
     */
    void compile() {
        layers_.clear();
        collect_layers(output_);

        // Assign default names to unnamed layers
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (layers_[i]->name.empty())
                layers_[i]->name = "L" + std::to_string(i + 1);
        }

        // Validate all declared inputs appear in the graph
        std::set<Node<Scalar>*> graph_inputs;
        collect_inputs(output_, graph_inputs);
        for (auto* inp : inputs_) {
            if (graph_inputs.find(inp) == graph_inputs.end())
                throw std::invalid_argument(
                    "FactorGraph: declared input '" + inp->name
                    + "' not reachable from output");
        }

        compiled_ = true;
    }

    /// True if compile() has been called successfully
    bool is_compiled() const noexcept { return compiled_; }

    /// Number of factorization layers in the graph
    int n_layers() const noexcept { return static_cast<int>(layers_.size()); }

    /// Access the ordered list of layer nodes (post-compilation)
    const std::vector<Node<Scalar>*>& layers() const { return layers_; }

    /// Access the output node
    Node<Scalar>* output() const { return output_; }

    // ====================================================================
    // Building NMFConfig from graph + layer config
    // ====================================================================

    /**
     * @brief Build an NMFConfig for a specific layer, merging graph-level
     *        and layer-level settings
     *
     * This is the config that gets passed to nmf::fit() for each layer.
     * Graph-level settings (max_iter, tol, threads, etc.) are used as
     * defaults; layer-level FactorConfig (W_config, H_config) overrides
     * per-factor regularization.
     *
     * @param layer  The layer node to build config for
     * @param maxit_override  Override max_iter (used for initialization passes)
     * @return NMFConfig ready for nmf::fit()
     */
    NMFConfig<Scalar> build_layer_config(
        const Node<Scalar>* layer,
        int maxit_override = -1) const
    {
        NMFConfig<Scalar> cfg;

        // Global settings — match nmf() defaults exactly
        cfg.max_iter = (maxit_override > 0) ? maxit_override : max_iter;
        cfg.tol = tol;
        cfg.verbose = false;  // deep loop manages its own verbosity
        cfg.threads = threads;
        cfg.seed = seed;
        cfg.loss = loss;
        cfg.norm_type = norm_type;
        cfg.solver_mode = solver_mode;
        cfg.resource_override = resource;

        // Cross-validation — propagate graph-level CV settings
        cfg.holdout_fraction = holdout_fraction;
        cfg.cv_seed = cv_seed;
        cfg.cv_patience = cv_patience;
        cfg.mask_zeros = mask_zeros;

        // Match nmf() R defaults for tracking
        cfg.track_loss_history = true;
        cfg.track_train_loss = true;

        // Per-factor config from the layer
        if (layer->type == NodeType::NMF_LAYER) {
            auto* nmf_layer = static_cast<const NMFLayerNode<Scalar>*>(layer);
            cfg.rank = nmf_layer->k;
            cfg.W = nmf_layer->W_config;
            cfg.H = nmf_layer->H_config;
        } else if (layer->type == NodeType::SVD_LAYER) {
            auto* svd_layer = static_cast<const SVDLayerNode<Scalar>*>(layer);
            cfg.rank = svd_layer->k;
            cfg.W = svd_layer->W_config;
            cfg.H = svd_layer->H_config;
        }

        return cfg;
    }

private:
    std::vector<Node<Scalar>*> inputs_;
    Node<Scalar>* output_;
    std::vector<Node<Scalar>*> layers_;  // topological order (input→output)
    bool compiled_ = false;

    /// Recursively collect layer nodes in topological order
    void collect_layers(Node<Scalar>* node) {
        if (!node) return;

        switch (node->type) {
            case NodeType::NMF_LAYER: {
                auto* n = static_cast<NMFLayerNode<Scalar>*>(node);
                collect_layers(n->input);
                layers_.push_back(node);
                break;
            }
            case NodeType::SVD_LAYER: {
                auto* n = static_cast<SVDLayerNode<Scalar>*>(node);
                collect_layers(n->input);
                layers_.push_back(node);
                break;
            }
            case NodeType::SHARED: {
                auto* n = static_cast<SharedNode<Scalar>*>(node);
                for (auto* child : n->inputs)
                    collect_layers(child);
                break;
            }
            case NodeType::CONCAT:
            case NodeType::ADD: {
                // Both have the same structure
                auto* concat = static_cast<ConcatNode<Scalar>*>(node);
                for (auto* child : concat->inputs)
                    collect_layers(child);
                break;
            }
            case NodeType::CONDITION: {
                auto* n = static_cast<ConditionNode<Scalar>*>(node);
                collect_layers(n->input);
                break;
            }
            case NodeType::INPUT:
                break;  // leaf — no layers
        }
    }

    /// Recursively collect input nodes
    void collect_inputs(Node<Scalar>* node, std::set<Node<Scalar>*>& out) {
        if (!node) return;

        if (node->type == NodeType::INPUT) {
            out.insert(node);
            return;
        }

        switch (node->type) {
            case NodeType::NMF_LAYER:
                collect_inputs(
                    static_cast<NMFLayerNode<Scalar>*>(node)->input, out);
                break;
            case NodeType::SVD_LAYER:
                collect_inputs(
                    static_cast<SVDLayerNode<Scalar>*>(node)->input, out);
                break;
            case NodeType::CONDITION:
                collect_inputs(
                    static_cast<ConditionNode<Scalar>*>(node)->input, out);
                break;
            case NodeType::SHARED:
                for (auto* child : static_cast<SharedNode<Scalar>*>(node)->inputs)
                    collect_inputs(child, out);
                break;
            case NodeType::CONCAT:
                for (auto* child : static_cast<ConcatNode<Scalar>*>(node)->inputs)
                    collect_inputs(child, out);
                break;
            case NodeType::ADD:
                for (auto* child : static_cast<AddNode<Scalar>*>(node)->inputs)
                    collect_inputs(child, out);
                break;
            default:
                break;
        }
    }
};

}  // namespace graph
}  // namespace FactorNet
