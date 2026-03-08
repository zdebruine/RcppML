// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file graph/fit.hpp
 * @brief Execution engine for FactorGraph — single-layer delegation and
 *        multi-layer outer ALS
 *
 * For single-layer graphs, delegates directly to nmf::fit() for maximum
 * performance (GPU, streaming all available).
 *
 * For multi-layer graphs, runs R-equivalent outer ALS:
 *   1. Initialize all layers with short warmup fits
 *   2. Alternate: for each layer, fix all others and run 1 NMF iteration
 *   3. Check convergence on total reconstruction loss
 *
 * The fit() function template requires the MatrixType of the original
 * input data (sparse or dense), which determines the backend dispatch
 * for the first layer.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/graph/graph.hpp>
#include <FactorNet/graph/node.hpp>
#include <FactorNet/graph/result.hpp>
#include <FactorNet/nmf/fit.hpp>
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace FactorNet {
namespace graph {

// ============================================================================
// Internal state for layer factors during outer ALS
// ============================================================================
namespace detail {

template<typename Scalar>
struct LayerState {
    DenseMatrix<Scalar> W;   // m × k
    DenseVector<Scalar> d;   // k
    DenseMatrix<Scalar> H;   // k × n
};

/// Resolve the original data from an input chain (walk through conditions)
template<typename Scalar, typename MatrixType>
const MatrixType& resolve_data(Node<Scalar>* node,
                               const std::map<Node<Scalar>*, const void*>& data_map)
{
    while (node->type == NodeType::CONDITION)
        node = static_cast<ConditionNode<Scalar>*>(node)->input;
    while (node->type == NodeType::NMF_LAYER)
        node = static_cast<NMFLayerNode<Scalar>*>(node)->input;
    while (node->type == NodeType::SVD_LAYER)
        node = static_cast<SVDLayerNode<Scalar>*>(node)->input;
    while (node->type == NodeType::CONDITION)
        node = static_cast<ConditionNode<Scalar>*>(node)->input;

    if (node->type != NodeType::INPUT)
        throw std::runtime_error("resolve_data: cannot reach input node");

    auto it = data_map.find(node);
    if (it == data_map.end())
        throw std::runtime_error("resolve_data: input node not in data map");

    return *static_cast<const MatrixType*>(it->second);
}

/// Find the layer index for a given node in the layers vector
template<typename Scalar>
int find_layer_idx(Node<Scalar>* node,
                   const std::vector<Node<Scalar>*>& layers)
{
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i] == node) return static_cast<int>(i);
    }
    return -1;
}

/// Compute the effective input matrix for layer i in a deep network.
/// For the first layer: returns the original data.
/// For deeper layers: returns t(H) from the upstream layer, possibly
/// with concat/add/condition applied.
template<typename Scalar>
DenseMatrix<Scalar> effective_input(
    int i,
    const std::vector<Node<Scalar>*>& layers,
    const std::vector<LayerState<Scalar>>& states,
    const DenseMatrix<Scalar>& original_data_dense)
{
    Node<Scalar>* layer = layers[i];
    Node<Scalar>* input_node = nullptr;

    // Get the input of this layer
    if (layer->type == NodeType::NMF_LAYER)
        input_node = static_cast<NMFLayerNode<Scalar>*>(layer)->input;
    else if (layer->type == NodeType::SVD_LAYER)
        input_node = static_cast<SVDLayerNode<Scalar>*>(layer)->input;
    else
        throw std::runtime_error("effective_input: not a layer node");

    // Walk through conditioning to find the real source
    DenseMatrix<Scalar> cond_Z;
    bool has_condition = false;
    while (input_node->type == NodeType::CONDITION) {
        auto* cond = static_cast<ConditionNode<Scalar>*>(input_node);
        cond_Z = cond->Z;  // last condition wins (they stack in R)
        has_condition = true;
        input_node = cond->input;
    }

    DenseMatrix<Scalar> result;

    if (input_node->type == NodeType::INPUT) {
        // First layer — use original data
        result = original_data_dense;
    } else if (input_node->type == NodeType::CONCAT) {
        auto* concat = static_cast<ConcatNode<Scalar>*>(input_node);
        // Row-bind t(H) from each branch → n × (k1+k2+...)
        int n = states[0].H.cols();
        int total_k = 0;
        for (auto* branch : concat->inputs) {
            int idx = find_layer_idx(branch, layers);
            if (idx < 0)
                throw std::runtime_error("concat: branch layer not found");
            total_k += static_cast<int>(states[idx].H.rows());
        }
        DenseMatrix<Scalar> merged(n, total_k);
        int col_offset = 0;
        for (auto* branch : concat->inputs) {
            int idx = find_layer_idx(branch, layers);
            int k_branch = static_cast<int>(states[idx].H.rows());
            merged.middleCols(col_offset, k_branch) = states[idx].H.transpose();
            col_offset += k_branch;
        }
        result = merged;
    } else if (input_node->type == NodeType::ADD) {
        auto* add = static_cast<AddNode<Scalar>*>(input_node);
        int first_idx = find_layer_idx(add->inputs[0], layers);
        if (first_idx < 0)
            throw std::runtime_error("add: branch layer not found");
        DenseMatrix<Scalar> summed = states[first_idx].H;
        for (size_t b = 1; b < add->inputs.size(); ++b) {
            int idx = find_layer_idx(add->inputs[b], layers);
            if (idx < 0)
                throw std::runtime_error("add: branch layer not found");
            summed += states[idx].H;
        }
        result = summed.transpose();  // n × k
    } else {
        // Sequential: previous layer's H
        int idx = find_layer_idx(input_node, layers);
        if (idx < 0) {
            if (i == 0)
                throw std::runtime_error("effective_input: cannot resolve input for first layer");
            idx = i - 1;  // fallback
        }
        result = states[idx].H.transpose();  // n × k
    }

    // Apply conditioning if present
    if (has_condition) {
        // Orient Z to p × n, then append columns to result (n × k → n × (k+p))
        DenseMatrix<Scalar> Z_oriented = cond_Z;
        int n = static_cast<int>(result.rows());
        if (Z_oriented.cols() != n) {
            if (Z_oriented.rows() == n)
                Z_oriented.transposeInPlace();
            else
                throw std::runtime_error("Conditioning Z dimension mismatch");
        }
        DenseMatrix<Scalar> augmented(n, result.cols() + Z_oriented.rows());
        augmented.leftCols(result.cols()) = result;
        augmented.rightCols(Z_oriented.rows()) = Z_oriented.transpose();
        result = augmented;
    }

    return result;
}

}  // namespace detail

// ============================================================================
// Single-layer fit — delegates to nmf::fit() for full backend support
// ============================================================================

/**
 * @brief Fit a single-layer FactorGraph
 *
 * Delegates directly to nmf::fit(), preserving access to GPU,
 * streaming, and all other backends. This is the fast path.
 *
 * @tparam Scalar     Element type
 * @tparam MatrixType Input matrix type (sparse or dense)
 * @param graph       Compiled single-layer FactorGraph
 * @return GraphResult with one layer
 */
template<typename Scalar, typename MatrixType>
GraphResult<Scalar> fit_single_layer(
    const FactorGraph<Scalar>& graph,
    const MatrixType& data,
    const DenseMatrix<Scalar>* W_init = nullptr)
{   
    auto* layer = graph.layers()[0];
    auto cfg = graph.build_layer_config(layer);
    cfg.verbose = graph.verbose;
    cfg.sort_model = true;

    auto nmf_result = FactorNet::nmf::nmf<Scalar>(data, cfg, W_init, nullptr);

    LayerResult<Scalar> lr;
    lr.W = std::move(nmf_result.W);
    lr.d = std::move(nmf_result.d);
    lr.H = std::move(nmf_result.H);
    lr.iterations = nmf_result.iterations;
    lr.loss = nmf_result.train_loss;
    lr.test_loss = nmf_result.test_loss;
    lr.best_test_loss = nmf_result.best_test_loss;
    lr.converged = nmf_result.converged;

    GraphResult<Scalar> result;
    result.layers[layer->name] = std::move(lr);
    result.total_iterations = nmf_result.iterations;
    result.total_loss = nmf_result.train_loss;
    result.converged = nmf_result.converged;
    return result;
}

// ============================================================================
// Multi-layer fit — outer ALS across layers
// ============================================================================

/**
 * @brief Fit a multi-layer FactorGraph using outer alternating least squares
 *
 * Iterates over all layers, fixing all but one and running a single
 * NMF iteration on each. Converges when the total reconstruction loss
 * stabilizes below tolerance.
 *
 * Inner layers operate on dense t(H) matrices, so GPU/sparse backends
 * are used only for the first layer (which sees the original data).
 *
 * @tparam Scalar     Element type
 * @tparam MatrixType Input matrix type for the first layer
 * @param graph       Compiled multi-layer FactorGraph
 * @param data        Original input data matrix
 * @return GraphResult with all layers
 */
template<typename Scalar, typename MatrixType>
GraphResult<Scalar> fit_deep(
    const FactorGraph<Scalar>& graph,
    const MatrixType& data)
{
    const auto& layers = graph.layers();
    int n_layers = static_cast<int>(layers.size());
    int max_iter = graph.max_iter;
    Scalar tol = graph.tol;
    uint32_t seed_base = graph.seed;
    if (seed_base == 0) seed_base = 42;

    // Dense copy of original data for intermediate computations
    DenseMatrix<Scalar> data_dense = DenseMatrix<Scalar>(data);

    // Initialize layer states
    std::vector<detail::LayerState<Scalar>> states(n_layers);
    int init_maxit = std::min(10, max_iter);

    for (int i = 0; i < n_layers; ++i) {
        DenseMatrix<Scalar> input_mat =
            detail::effective_input(i, layers, states, data_dense);

        auto cfg = graph.build_layer_config(layers[i], init_maxit);
        cfg.seed = seed_base + i;
        cfg.sort_model = false;
        cfg.rank = (layers[i]->type == NodeType::NMF_LAYER)
            ? static_cast<NMFLayerNode<Scalar>*>(layers[i])->k
            : static_cast<SVDLayerNode<Scalar>*>(layers[i])->k;

        auto init_result = FactorNet::nmf::nmf<Scalar>(input_mat, cfg, nullptr, nullptr);
        states[i].W = std::move(init_result.W);
        states[i].d = std::move(init_result.d);
        states[i].H = std::move(init_result.H);
    }

    // Outer ALS loop
    Scalar prev_loss = std::numeric_limits<Scalar>::infinity();
    int total_iter = 0;
    bool converged = false;

    for (int outer = 0; outer < max_iter; ++outer) {
        for (int i = 0; i < n_layers; ++i) {
            DenseMatrix<Scalar> input_mat =
                detail::effective_input(i, layers, states, data_dense);

            auto cfg = graph.build_layer_config(layers[i], 1);
            cfg.sort_model = false;
            cfg.tol = 0;
            cfg.rank = (layers[i]->type == NodeType::NMF_LAYER)
                ? static_cast<NMFLayerNode<Scalar>*>(layers[i])->k
                : static_cast<SVDLayerNode<Scalar>*>(layers[i])->k;

            // Warm-start with current W
            auto update_result = FactorNet::nmf::nmf<Scalar>(
                input_mat, cfg, &states[i].W, nullptr);

            states[i].W = std::move(update_result.W);
            states[i].d = std::move(update_result.d);
            states[i].H = std::move(update_result.H);
        }

        ++total_iter;

        // Compute per-layer reconstruction loss
        Scalar current_loss = 0;
        for (int i = 0; i < n_layers; ++i) {
            DenseMatrix<Scalar> eff =
                detail::effective_input(i, layers, states, data_dense);
            auto& s = states[i];
            DenseMatrix<Scalar> recon = s.W * s.d.asDiagonal() * s.H;
            Scalar layer_loss = (eff - recon).squaredNorm()
                / static_cast<Scalar>(eff.rows() * eff.cols());
            current_loss += layer_loss;
        }

        if (graph.verbose) {
            Rprintf("  outer iter %d: loss = %g\n", total_iter,
                    static_cast<double>(current_loss));
        }

        if (std::isfinite(static_cast<double>(prev_loss))) {
            Scalar rel_change = std::abs(prev_loss - current_loss)
                / (std::abs(prev_loss) + static_cast<Scalar>(1e-15));
            if (rel_change < tol) {
                converged = true;
                break;
            }
        }
        prev_loss = current_loss;
    }

    // Package results
    GraphResult<Scalar> result;
    result.total_iterations = total_iter;
    result.total_loss = prev_loss;
    result.converged = converged;

    for (int i = 0; i < n_layers; ++i) {
        const std::string& name = layers[i]->name;
        LayerResult<Scalar> lr;
        lr.W = std::move(states[i].W);
        lr.d = std::move(states[i].d);
        lr.H = std::move(states[i].H);
        lr.iterations = total_iter;
        lr.loss = prev_loss;
        lr.converged = converged;
        result.layers[name] = std::move(lr);
    }

    return result;
}

// ============================================================================
// Top-level fit() — dispatches to single-layer or deep
// ============================================================================

/**
 * @brief Fit a compiled FactorGraph
 *
 * Automatically dispatches to the optimal execution path:
 *   - Single-layer: delegates to nmf::fit() (GPU/streaming available)
 *   - Multi-layer: outer ALS with per-layer nmf::fit() calls
 *
 * @tparam Scalar     Element type (float or double)
 * @tparam MatrixType Eigen::SparseMatrix<Scalar> or DenseMatrix<Scalar>
 * @param graph       Compiled FactorGraph (must have called compile())
 * @param data        The original input data matrix
 * @return GraphResult containing per-layer W, d, H and convergence info
 *
 * @throws std::runtime_error if graph is not compiled
 */
template<typename Scalar, typename MatrixType>
GraphResult<Scalar> fit(
    const FactorGraph<Scalar>& graph,
    const MatrixType& data,
    const DenseMatrix<Scalar>* W_init = nullptr)
{
    if (!graph.is_compiled())
        throw std::runtime_error("FactorGraph::fit(): graph not compiled. "
                                 "Call compile() first.");

    if (graph.n_layers() == 1)
        return fit_single_layer(graph, data, W_init);
    else
        return fit_deep(graph, data);
}

}  // namespace graph
}  // namespace FactorNet
