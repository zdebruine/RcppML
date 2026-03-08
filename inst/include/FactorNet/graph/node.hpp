// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file graph/node.hpp
 * @brief Graph node types for FactorNet computation graphs
 *
 * Nodes are the building blocks of a FactorNet graph. Each node type
 * represents a different operation in the factorization pipeline:
 *
 *   - InputNode:     Wraps a data matrix (sparse or dense)
 *   - NMFLayerNode:  NMF factorization A ≈ W·diag(d)·H
 *   - SVDLayerNode:  SVD/PCA factorization (signed factors allowed)
 *   - SharedNode:    Multi-modal shared-H factorization
 *   - ConcatNode:    Row-bind H factors from branches
 *   - AddNode:       Element-wise H addition (skip/residual connection)
 *   - ConditionNode: Append conditioning metadata to H
 *
 * Nodes are connected via pointers to form a directed acyclic graph.
 * The graph is compiled into a FactorGraph for execution.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/factor_config.hpp>

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace FactorNet {
namespace graph {

// ============================================================================
// Node type enumeration
// ============================================================================

enum class NodeType {
    INPUT,        ///< Data input (sparse or dense matrix)
    NMF_LAYER,    ///< NMF factorization layer
    SVD_LAYER,    ///< SVD/PCA factorization layer
    SHARED,       ///< Multi-modal shared-H combinator
    CONCAT,       ///< Row-bind H factors from branches
    ADD,          ///< Element-wise H addition (skip/residual)
    CONDITION     ///< Append conditioning metadata to H
};

// ============================================================================
// Abstract base node
// ============================================================================

/**
 * @brief Abstract base class for all graph nodes
 *
 * Provides the common interface: type tag, name, and virtual destructor.
 * Concrete node types add their specific fields (data pointers, configs,
 * child connections).
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct Node {
    NodeType type;
    std::string name;

    explicit Node(NodeType t, const std::string& n = "")
        : type(t), name(n) {}

    virtual ~Node() = default;

    /// True if this is a factorization layer (NMF or SVD)
    bool is_layer() const noexcept {
        return type == NodeType::NMF_LAYER || type == NodeType::SVD_LAYER;
    }

    /// True if this node has multiple inputs (shared, concat, add)
    bool is_combinator() const noexcept {
        return type == NodeType::SHARED || type == NodeType::CONCAT
            || type == NodeType::ADD;
    }
};

// ============================================================================
// InputNode — wraps a data matrix
// ============================================================================

/**
 * @brief Input node that holds a reference to a data matrix
 *
 * Does not own the matrix — the caller must ensure the data outlives
 * the graph. Supports both sparse (CSC) and dense matrices via the
 * MatrixType template parameter.
 *
 * @tparam Scalar      Element type (float or double)
 * @tparam MatrixType  Eigen::SparseMatrix<Scalar> or DenseMatrix<Scalar>
 */
template<typename Scalar, typename MatrixType>
struct InputNode : Node<Scalar> {
    const MatrixType* data;   ///< Non-owning pointer to the data matrix
    int rows;                 ///< Cached row count
    int cols;                 ///< Cached column count

    InputNode(const MatrixType& mat, const std::string& name = "")
        : Node<Scalar>(NodeType::INPUT, name),
          data(&mat),
          rows(static_cast<int>(mat.rows())),
          cols(static_cast<int>(mat.cols()))
    {}
};

// ============================================================================
// NMFLayerNode — NMF factorization layer
// ============================================================================

/**
 * @brief NMF layer: factorizes input A ≈ W·diag(d)·H
 *
 * Holds per-factor configuration (regularization, constraints, guides)
 * for both W and H, plus the factorization rank k. The input field
 * points to the upstream node in the graph.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct NMFLayerNode : Node<Scalar> {
    Node<Scalar>* input = nullptr;   ///< Upstream node
    int k = 0;                        ///< Factorization rank

    FactorConfig<Scalar> W_config;    ///< W-side regularization & guides
    FactorConfig<Scalar> H_config;    ///< H-side regularization & guides

    NMFLayerNode(Node<Scalar>* in, int rank, const std::string& name = "")
        : Node<Scalar>(NodeType::NMF_LAYER, name),
          input(in), k(rank)
    {
        if (!in) throw std::invalid_argument("NMFLayerNode: input must not be null");
        if (rank <= 0) throw std::invalid_argument("NMFLayerNode: rank must be > 0");
        // NMF defaults: nonneg = true
        W_config.nonneg = true;
        H_config.nonneg = true;
    }
};

// ============================================================================
// SVDLayerNode — SVD/PCA factorization layer
// ============================================================================

/**
 * @brief SVD layer: factorizes input A ≈ U·diag(σ)·V^T
 *
 * Similar to NMFLayerNode but defaults to signed factors (nonneg=false).
 * Supports the full regularization suite (L1, L2, L21, angular, etc.)
 * but with signed initial values.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct SVDLayerNode : Node<Scalar> {
    Node<Scalar>* input = nullptr;   ///< Upstream node
    int k = 0;                        ///< Number of components

    FactorConfig<Scalar> W_config;    ///< Left factor config
    FactorConfig<Scalar> H_config;    ///< Right factor config

    SVDLayerNode(Node<Scalar>* in, int rank, const std::string& name = "")
        : Node<Scalar>(NodeType::SVD_LAYER, name),
          input(in), k(rank)
    {
        if (!in) throw std::invalid_argument("SVDLayerNode: input must not be null");
        if (rank <= 0) throw std::invalid_argument("SVDLayerNode: rank must be > 0");
        // SVD defaults: signed factors
        W_config.nonneg = false;
        H_config.nonneg = false;
    }
};

// ============================================================================
// SharedNode — multi-modal shared-H
// ============================================================================

/**
 * @brief Shared node: jointly factorizes multiple inputs with a common H
 *
 * Row-concatenates inputs [A1; A2; ...] and performs a single NMF,
 * yielding a shared H and per-input W splits: [W1; W2; ...] = W_concat.
 * All inputs must have the same number of columns (samples).
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct SharedNode : Node<Scalar> {
    std::vector<Node<Scalar>*> inputs;   ///< Multiple input nodes

    explicit SharedNode(std::vector<Node<Scalar>*> ins)
        : Node<Scalar>(NodeType::SHARED)
    {
        if (ins.size() < 2)
            throw std::invalid_argument("SharedNode requires at least 2 inputs");
        inputs = std::move(ins);
    }
};

// ============================================================================
// ConcatNode — row-bind H factors from branches
// ============================================================================

/**
 * @brief Concat node: row-binds H factors from multiple branches
 *
 * Given branches with ranks k1, k2, ..., produces an output H with
 * k_out = k1 + k2 + ... rows. Each branch's H is stacked vertically.
 * Used for merging parallel factorization streams before a downstream layer.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct ConcatNode : Node<Scalar> {
    std::vector<Node<Scalar>*> inputs;   ///< Branch layer nodes

    explicit ConcatNode(std::vector<Node<Scalar>*> ins)
        : Node<Scalar>(NodeType::CONCAT)
    {
        if (ins.size() < 2)
            throw std::invalid_argument("ConcatNode requires at least 2 inputs");
        inputs = std::move(ins);
    }
};

// ============================================================================
// AddNode — element-wise H addition (skip/residual connection)
// ============================================================================

/**
 * @brief Add node: element-wise addition of H factors from branches
 *
 * All branches must produce H factors of the same rank k.
 * The output H = H1 + H2 + ... element-wise.
 * Enables skip connections and residual architectures.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct AddNode : Node<Scalar> {
    std::vector<Node<Scalar>*> inputs;   ///< Branch layer nodes (must have same k)

    explicit AddNode(std::vector<Node<Scalar>*> ins)
        : Node<Scalar>(NodeType::ADD)
    {
        if (ins.size() < 2)
            throw std::invalid_argument("AddNode requires at least 2 inputs");
        inputs = std::move(ins);
    }
};

// ============================================================================
// ConditionNode — append conditioning metadata to H
// ============================================================================

/**
 * @brief Condition node: appends a metadata matrix Z to the H factor
 *
 * The downstream layer's input becomes [H; Z] (row-concatenated),
 * so the next W learns to factor out the conditioning variables.
 * Z must have the same number of columns as the upstream H.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct ConditionNode : Node<Scalar> {
    Node<Scalar>* input = nullptr;       ///< Upstream node
    DenseMatrix<Scalar> Z;               ///< Conditioning matrix (p × n)

    ConditionNode(Node<Scalar>* in, const DenseMatrix<Scalar>& conditioning)
        : Node<Scalar>(NodeType::CONDITION),
          input(in), Z(conditioning)
    {
        if (!in) throw std::invalid_argument("ConditionNode: input must not be null");
    }
};

}  // namespace graph
}  // namespace FactorNet
