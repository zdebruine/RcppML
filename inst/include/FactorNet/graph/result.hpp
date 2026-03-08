// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file graph/result.hpp
 * @brief Per-layer and whole-graph result types for FactorGraph::fit()
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>

#include <string>
#include <vector>
#include <map>

namespace FactorNet {
namespace graph {

/**
 * @brief Result for a single factorization layer
 *
 * Contains the W·diag(d)·H factors plus convergence metadata.
 * For multi-modal layers (SharedNode input), W_splits holds
 * the per-input W sub-matrices.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct LayerResult {
    DenseMatrix<Scalar> W;           ///< m × k (single-input) or full concatenated W
    DenseVector<Scalar> d;           ///< k (diagonal scaling)
    DenseMatrix<Scalar> H;           ///< k × n

    /// Per-input W splits for multi-modal (SharedNode) layers.
    /// Empty for standard single-input layers.
    std::map<std::string, DenseMatrix<Scalar>> W_splits;

    int iterations = 0;              ///< Iterations used for this layer
    Scalar loss = 0;                 ///< Final reconstruction loss (train)
    Scalar test_loss = 0;            ///< Final test loss (CV only, 0 if no CV)
    Scalar best_test_loss = 0;       ///< Best test loss seen (CV only)
    bool converged = false;          ///< Whether convergence was achieved
};

/**
 * @brief Result for the entire FactorGraph
 *
 * Contains per-layer results indexed by layer name, plus aggregate
 * convergence information for the whole network.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct GraphResult {
    /// Per-layer results, keyed by layer name (e.g. "L1", "encoder", "decoder")
    std::map<std::string, LayerResult<Scalar>> layers;

    int total_iterations = 0;        ///< Total outer iterations
    Scalar total_loss = 0;           ///< Sum of per-layer losses
    bool converged = false;          ///< Whether the network converged

    /// Access a layer result by name
    const LayerResult<Scalar>& operator[](const std::string& name) const {
        auto it = layers.find(name);
        if (it == layers.end())
            throw std::out_of_range("Layer '" + name + "' not found in result");
        return it->second;
    }
};

}  // namespace graph
}  // namespace FactorNet
