// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file graph/graph_all.hpp
 * @brief Single-include umbrella for the FactorNet graph API
 *
 * Include this one header to get the complete graph construction,
 * compilation, and execution API:
 *
 * @code
 *   #include <FactorNet/graph/graph_all.hpp>
 *   using namespace FactorNet::graph;
 *
 *   InputNode<float, SparseMatrix<float>> inp(A, "X");
 *   NMFLayerNode<float> layer(&inp, 20, "L1");
 *   FactorGraph<float> net({&inp}, &layer);
 *   net.compile();
 *   auto result = fit(net, A);
 * @endcode
 */

#pragma once

#include <FactorNet/graph/node.hpp>
#include <FactorNet/graph/graph.hpp>
#include <FactorNet/graph/result.hpp>
#include <FactorNet/graph/builder.hpp>
#include <FactorNet/graph/fit.hpp>
#include <FactorNet/graph/fit.hpp>
