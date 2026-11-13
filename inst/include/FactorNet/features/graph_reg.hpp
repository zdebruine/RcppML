// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file features/graph_reg.hpp
 * @brief Graph Laplacian regularization — Tier 2 (touches full factor)
 *
 * Adds a smoothness penalty that encourages neighboring features (or samples)
 * in a graph to have similar factor loadings:
 *
 *   G += λ · factor · L · factorᵀ  (k × k)
 *
 * where L is a sparse graph Laplacian (m × m or n × n).
 *
 * Classification: Tier 2 — accesses the full factor matrix (k × m or k × n)
 * multiplied by the sparse Laplacian. For GPU paths, this maps naturally to
 * a cuSPARSE SpMM as an "extended primitive".
 *
 * Cost: O(k² · nnz(L)) — depends on graph sparsity.
 *
 * Extracted from nmf/gram.hpp::compute_gram_regularized (graph path) and
 * nmf/cd_solver.hpp::cd_nnls_full_reg (graph path).
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/exceptions.hpp>
#include <FactorNet/core/logging.hpp>

#include <cmath>
#include <string>

namespace FactorNet {
namespace features {

/**
 * @brief Apply graph Laplacian regularization to Gram matrix
 *
 * Computes:  G += λ · factor · L · factorᵀ
 *
 * This smoothness penalty penalizes factor differences along graph edges.
 * The result is a k×k matrix added to the existing Gram.
 *
 * @param G          Gram matrix (k × k), modified in-place
 * @param laplacian  Sparse graph Laplacian (n × n, or m × m)
 * @param factor     Factor matrix (k × n, or k × m)
 * @param lambda     Graph regularization strength
 */
template<typename Scalar>
inline void apply_graph_reg(DenseMatrix<Scalar>& G,
                            const SparseMatrix<Scalar>& laplacian,
                            const DenseMatrix<Scalar>& factor,
                            Scalar lambda)
{
    if (lambda <= 0) return;

    // FL = factor · L  (k × n)  — sparse matrix multiply
    DenseMatrix<Scalar> FL = factor * laplacian;

    // G += λ · FL · factorᵀ  (k × k)
    G.noalias() += lambda * FL * factor.transpose();
}

/**
 * @brief True if the matrix looks like an adjacency matrix, not a Laplacian
 *
 * An adjacency matrix has an empty (all-zero) diagonal, non-negative
 * off-diagonal weights and at least one edge.  A Laplacian L = D − A of a
 * graph with at least one edge has a positive diagonal entry, so the two
 * cannot be confused.
 */
template<typename Scalar>
inline bool is_adjacency_graph(const SparseMatrix<Scalar>& L)
{
    bool any_edge = false;
    for (int j = 0; j < L.outerSize(); ++j) {
        for (typename SparseMatrix<Scalar>::InnerIterator it(L, j); it; ++it) {
            const Scalar v = it.value();
            if (static_cast<int>(it.row()) == j) {
                if (v != Scalar(0)) return false;
            } else {
                if (v < Scalar(0)) return false;
                if (v > Scalar(0)) any_edge = true;
            }
        }
    }
    return any_edge;
}

/**
 * @brief Check that a graph matrix is square, correctly sized and symmetric
 *
 * Accepts either a graph Laplacian L = D − A or a symmetric adjacency matrix
 * A (zero diagonal, non-negative weights).  An adjacency matrix is used as
 * its Laplacian D − A by graph_column_terms(); a note is printed the first
 * time this happens in a session.
 *
 * @param L     Sparse graph matrix supplied by the user
 * @param n     Expected dimension (number of columns of the factor)
 * @param name  Name used in messages ("graph_H" / "graph_W")
 * @throws InvalidArgument on a dimension mismatch or a non-symmetric matrix
 */
template<typename Scalar>
inline void validate_graph(const SparseMatrix<Scalar>& L, int n, const char* name)
{
    if (static_cast<int>(L.rows()) != n || static_cast<int>(L.cols()) != n) {
        throw InvalidArgument(
            std::string(name) + " must be a " + std::to_string(n) + " x " +
            std::to_string(n) + " graph Laplacian, got " +
            std::to_string(static_cast<long long>(L.rows())) + " x " +
            std::to_string(static_cast<long long>(L.cols())));
    }

    Scalar max_abs = 0;
    for (int j = 0; j < L.outerSize(); ++j) {
        for (typename SparseMatrix<Scalar>::InnerIterator it(L, j); it; ++it) {
            const Scalar a = std::abs(it.value());
            if (a > max_abs) max_abs = a;
        }
    }

    SparseMatrix<Scalar> Lt = L.transpose();
    SparseMatrix<Scalar> D = L - Lt;
    Scalar max_diff = 0;
    for (int j = 0; j < D.outerSize(); ++j) {
        for (typename SparseMatrix<Scalar>::InnerIterator it(D, j); it; ++it) {
            const Scalar a = std::abs(it.value());
            if (a > max_diff) max_diff = a;
        }
    }
    if (max_diff > static_cast<Scalar>(1e-5) * max_abs) {
        throw InvalidArgument(
            std::string(name) + " must be symmetric: supply a graph Laplacian "
            "L = D - A or a symmetric adjacency matrix A");
    }

    if (is_adjacency_graph(L)) {
        static bool noted = false;
        if (!noted) {
            noted = true;
            FACTORNET_WARN_IMPL(
                "RcppML: %s has a zero diagonal and non-negative weights, so it "
                "is treated as an adjacency matrix A and the graph Laplacian "
                "L = D - A is used. (This note is shown once per session.)\n",
                name);
        }
    }
}

/**
 * @brief Per-column (Jacobi) terms of the graph Laplacian penalty
 *
 * Penalty on the factor X (k × n, columns x_j):
 *
 *   (λ/2) · tr(X L Xᵀ) = (λ/2) · Σ_ij L_ij · x_iᵀx_j
 *
 * Holding the other columns at the previous iterate S, the subproblem for
 * column j gains
 *
 *   ridge_j   = λ · L_jj                     (added to the Gram diagonal)
 *   B_add_j  += λ · Σ_{i≠j} (−L_ij) · s_i    (added to the right-hand side)
 *
 * S must be on the scale of the variable being solved.  L must be symmetric.
 * Cost: O(nnz(L) · k).
 *
 * If L is an adjacency matrix A (see is_adjacency_graph), its Laplacian
 * D − A is used: L_jj = Σ_i A_ij and −L_ij = A_ij.
 *
 * @param L       Sparse symmetric Laplacian or adjacency matrix (n × n)
 * @param S       Previous iterate (k × n)
 * @param lambda  Graph regularization strength
 * @param ridge   Output: per-column ridge (length n), overwritten
 * @param B_add   k × n matrix to which the neighbor term is ADDED
 */
template<typename Scalar>
inline void graph_column_terms(const SparseMatrix<Scalar>& L,
                               const DenseMatrix<Scalar>& S,
                               Scalar lambda,
                               DenseVector<Scalar>& ridge,
                               DenseMatrix<Scalar>& B_add)
{
    const int n = static_cast<int>(S.cols());
    ridge.resize(n);
    ridge.setZero();
    if (lambda <= 0) return;

    if (is_adjacency_graph(L)) {
        // Adjacency A: Laplacian D − A, so L_jj = degree_j and −L_ij = A_ij
        for (int j = 0; j < n; ++j) {
            for (typename SparseMatrix<Scalar>::InnerIterator it(L, j); it; ++it) {
                const int i = static_cast<int>(it.row());
                if (i == j) continue;
                const Scalar v = lambda * static_cast<Scalar>(it.value());
                ridge(j) += v;
                B_add.col(j).noalias() += v * S.col(i);
            }
        }
        return;
    }

    for (int j = 0; j < n; ++j) {
        for (typename SparseMatrix<Scalar>::InnerIterator it(L, j); it; ++it) {
            const int i = static_cast<int>(it.row());
            const Scalar v = lambda * static_cast<Scalar>(it.value());
            if (i == j) {
                ridge(j) += v;
            } else {
                B_add.col(j).noalias() -= v * S.col(i);
            }
        }
        if (ridge(j) < Scalar(0)) ridge(j) = Scalar(0);
    }
}

}  // namespace features
}  // namespace FactorNet

