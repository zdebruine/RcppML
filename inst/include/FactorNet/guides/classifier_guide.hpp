// This file is part of FactorNet, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file guides/classifier_guide.hpp
 * @brief Classifier guide — steers a factor toward class-consistent structure
 *
 * Given a label vector (integer class labels per column, -1 = unlabeled),
 * computes per-class centroid targets from the current factor and applies:
 *
 *   G.diagonal() += |λ|
 *   B.col(j)     += λ · centroid[label[j]]    (labeled columns only)
 *
 * For unlabeled columns (label == -1), only the diagonal modification
 * applies (equivalent to extra L2 regularization).
 *
 * The centroid targets are recomputed each iteration via update().
 *
 * Validated in benchmarks/cuda/guide_vs_blend.cu:
 *   Guide λ=2.0: 94.3% MNIST / 81.8% Fashion-MNIST test kNN
 *   vs. Blend-0.9: 93.7% / 81.2%
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/guides/guide.hpp>
#include <vector>

namespace FactorNet {
namespace guides {

template<typename Scalar>
struct ClassifierGuide : Guide<Scalar> {

    /// Per-column labels: label[j] ∈ {0,..,C-1} for labeled, -1 for unlabeled
    std::vector<int> labels;

    /// Number of classes (computed from labels)
    int n_classes = 0;

    /// Class centroids: centroids[c] is a k-length vector (the mean of all
    /// factor columns belonging to class c)
    std::vector<DenseVector<Scalar>> centroids;

    /// Target matrix (k × n) — precomputed in update(), applied in apply()
    DenseMatrix<Scalar> target;

    /// Count of labeled samples per class (for centroid computation)
    std::vector<int> class_counts;

    /**
     * @brief Construct a ClassifierGuide
     *
     * @param labels_  Per-column labels (-1 = unlabeled)
     * @param lambda_  Guide strength (positive = attract, negative = repel)
     */
    ClassifierGuide(const std::vector<int>& labels_, Scalar lambda_)
        : labels(labels_)
    {
        this->lambda = lambda_;
        // Compute number of classes
        n_classes = 0;
        for (int l : labels) {
            if (l >= n_classes) n_classes = l + 1;
        }
        class_counts.resize(n_classes, 0);
        for (int l : labels) {
            if (l >= 0 && l < n_classes) class_counts[l]++;
        }
    }

    /**
     * @brief Recompute class centroids from the current factor
     *
     * For each class c, centroid[c] = mean(factor[:, j] for j where label[j] == c).
     * Then target[:, j] = centroid[label[j]] for labeled j, zero for unlabeled.
     */
    void update(const DenseMatrix<Scalar>& factor, int /*iteration*/) override {
        const int k = factor.rows();
        const int n = factor.cols();

        // Reset centroids
        centroids.assign(n_classes, DenseVector<Scalar>::Zero(k));

        // Accumulate
        for (int j = 0; j < n && j < static_cast<int>(labels.size()); ++j) {
            int c = labels[j];
            if (c >= 0 && c < n_classes) {
                centroids[c] += factor.col(j);
            }
        }

        // Average
        for (int c = 0; c < n_classes; ++c) {
            if (class_counts[c] > 0) {
                centroids[c] /= static_cast<Scalar>(class_counts[c]);
            }
        }

        // Build target matrix
        target.resize(k, n);
        target.setZero();
        for (int j = 0; j < n && j < static_cast<int>(labels.size()); ++j) {
            int c = labels[j];
            if (c >= 0 && c < n_classes) {
                target.col(j) = centroids[c];
            }
        }
    }

    /**
     * @brief Apply classifier guide to Gram and RHS
     *
     * G.diagonal() += |λ|  (global — adds L2-like regularization)
     * B += λ · target       (per-column target shift)
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
