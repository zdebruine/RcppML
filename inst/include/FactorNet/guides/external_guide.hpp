// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file guides/external_guide.hpp
 * @brief External (static target) guide — steers a factor toward a fixed matrix
 *
 * Given a pre-computed target matrix T (same shape as the factor), applies:
 *
 *   G.diagonal() += |λ|
 *   B            += λ · T
 *
 * The target is fixed across iterations (no update() override needed).
 * Useful for:
 *   - Transfer learning: guide toward a reference factorization
 *   - Prior knowledge: steer factor toward known structure
 *   - Multi-layer coupling: cross-layer reference (set target externally)
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/guides/guide.hpp>

namespace FactorNet {
namespace guides {

template<typename Scalar>
struct ExternalGuide : Guide<Scalar> {

    /// Target matrix (k × n), same dimensions as the guided factor
    DenseMatrix<Scalar> target;

    /**
     * @brief Construct with a target matrix and strength
     *
     * @param target_  Target matrix (k × n_cols)
     * @param lambda_  Guide strength (positive = attract, negative = repel)
     */
    ExternalGuide(const DenseMatrix<Scalar>& target_, Scalar lambda_)
        : target(target_)
    {
        this->lambda = lambda_;
    }

    /**
     * @brief Apply external guide to Gram and RHS
     */
    void apply(DenseMatrix<Scalar>& G, DenseMatrix<Scalar>& B) const override {
        if (!this->is_active() || target.size() == 0) return;

        const Scalar abs_lambda = std::abs(this->lambda);
        G.diagonal().array() += abs_lambda;
        B.noalias() += this->lambda * target;
    }
};

}  // namespace guides
}  // namespace FactorNet
