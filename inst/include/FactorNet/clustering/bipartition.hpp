// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

#ifndef FactorNet_CLUSTERING_BIPARTITION_HPP
#define FactorNet_CLUSTERING_BIPARTITION_HPP

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/core/constants.hpp>
#include <FactorNet/rng/rng.hpp>
#include <numeric>
#include <FactorNet/core/logging.hpp>

namespace FactorNet {
namespace clustering {


// Bipartition result structure
struct bipartitionModel {
    std::vector<double> v;
    double dist;
    unsigned int size1;
    unsigned int size2;
    std::vector<unsigned int> samples1;
    std::vector<unsigned int> samples2;
    std::vector<double> center1;
    std::vector<double> center2;
};

// ============================================================================
// HELPER FUNCTIONS (ported from original implementation)
// ============================================================================

// Compute centroid for sparse matrix
template<typename SpMat>
inline std::vector<double> centroid(const SpMat& A, const std::vector<unsigned int>& samples) {
    std::vector<double> center(A.rows(), 0.0);
    for (unsigned int s = 0; s < samples.size(); ++s) {
        for (typename SpMat::InnerIterator it(A, samples[s]); it; ++it) {
            center[it.row()] += it.value();
        }
    }
    for (int j = 0; j < A.rows(); ++j) {
        center[j] /= samples.size();
    }
    return center;
}

// Compute centroid for dense matrix
inline std::vector<double> centroid(const Eigen::MatrixXd& A, const std::vector<unsigned int>& samples) {
    std::vector<double> center(A.rows(), 0.0);
    for (unsigned int s = 0; s < samples.size(); ++s) {
        for (int r = 0; r < A.rows(); ++r) {
            center[r] += A(r, samples[s]);
        }
    }
    for (int j = 0; j < A.rows(); ++j) {
        center[j] /= samples.size();
    }
    return center;
}

// Template version for compatibility with dclust (returns Scalar type)
template<typename Scalar, typename Mat>
inline std::vector<Scalar> compute_centroid(const Mat& A, const std::vector<unsigned int>& samples) {
    std::vector<Scalar> center(A.rows(), static_cast<Scalar>(0));
    
    if constexpr (is_sparse_v<Mat>) {
        for (unsigned int s = 0; s < samples.size(); ++s) {
            for (typename Mat::InnerIterator it(A, samples[s]); it; ++it) {
                center[it.row()] += static_cast<Scalar>(it.value());
            }
        }
    } else {
        for (unsigned int s = 0; s < samples.size(); ++s) {
            for (int r = 0; r < A.rows(); ++r) {
                center[r] += A(r, samples[s]);
            }
        }
    }
    
    Scalar inv_size = static_cast<Scalar>(1) / static_cast<Scalar>(samples.size());
    for (int j = 0; j < A.rows(); ++j) {
        center[j] *= inv_size;
    }
    return center;
}

// Relative cosine distance for sparse matrix
template<typename SpMat>
inline double rel_cosine(const SpMat& A, 
                         const std::vector<unsigned int>& samples1,
                         const std::vector<unsigned int>& samples2,
                         const std::vector<double>& center1,
                         const std::vector<double>& center2) {
    double center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), 0.0));
    double center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), 0.0));
    double dist1 = 0, dist2 = 0;
    
    for (unsigned int s = 0; s < samples1.size(); ++s) {
        double x1_center1 = 0, x1_center2 = 0;
        for (typename SpMat::InnerIterator it(A, samples1[s]); it; ++it) {
            x1_center1 += center1[it.row()] * it.value();
            x1_center2 += center2[it.row()] * it.value();
        }
        dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
    }
    
    for (unsigned int s = 0; s < samples2.size(); ++s) {
        double x2_center1 = 0, x2_center2 = 0;
        for (typename SpMat::InnerIterator it(A, samples2[s]); it; ++it) {
            x2_center1 += center1[it.row()] * it.value();
            x2_center2 += center2[it.row()] * it.value();
        }
        dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
    }
    
    return (dist1 + dist2) / (2 * A.rows());
}

// Relative cosine distance for dense matrix
inline double rel_cosine(const Eigen::MatrixXd& A,
                         const std::vector<unsigned int>& samples1,
                         const std::vector<unsigned int>& samples2,
                         const std::vector<double>& center1,
                         const std::vector<double>& center2) {
    double center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), 0.0));
    double center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), 0.0));
    double dist1 = 0, dist2 = 0;
    
    for (unsigned int s = 0; s < samples1.size(); ++s) {
        double x1_center1 = 0, x1_center2 = 0;
        for (int r = 0; r < A.rows(); ++r) {
            x1_center1 += center1[r] * A(r, samples1[s]);
            x1_center2 += center2[r] * A(r, samples1[s]);
        }
        dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
    }
    
    for (unsigned int s = 0; s < samples2.size(); ++s) {
        double x2_center1 = 0, x2_center2 = 0;
        for (int r = 0; r < A.rows(); ++r) {
            x2_center1 += center1[r] * A(r, samples2[s]);
            x2_center2 += center2[r] * A(r, samples2[s]);
        }
        dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
    }
    
    return (dist1 + dist2) / (2 * A.rows());
}

// Scale matrix columns by row sums
inline void scale(Eigen::VectorXd& d, Eigen::MatrixXd& w) {
    d = w.rowwise().sum();
    d.array() += tiny_num<double>();
    for (unsigned int i = 0; i < w.rows(); ++i) {
        for (unsigned int j = 0; j < w.cols(); ++j) {
            w(i, j) /= d(i);
        }
    }
}

// Correlation between matrices for convergence check
inline double cor(const Eigen::MatrixXd& w, const Eigen::MatrixXd& w_it) {
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0;
    size_t n = w.rows() * w.cols();
    
    for (int i = 0; i < w.rows(); ++i) {
        for (int j = 0; j < w.cols(); ++j) {
            double x = w_it(i, j);
            double y = w(i, j);
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
        }
    }
    
    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y));
    
    return 1.0 - (numerator / denominator);
}

// Fast rank-2 NNLS solver (ported from original nnls2 function)
inline void nnls2(const Eigen::Matrix2d& a, const Eigen::Vector2d& b, 
                  const double denom, Eigen::MatrixXd& h, unsigned int col, bool nonneg) {
    if (!nonneg) {
        // Unconstrained least squares: h = inv(a) * b
        h(0, col) = (b(0) * a(1, 1) - b(1) * a(0, 1)) / denom;
        h(1, col) = (b(1) * a(0, 0) - b(0) * a(0, 1)) / denom;
    } else {
        // Non-negative least squares (simplified for 2x2 case)
        double h0 = (b(0) * a(1, 1) - b(1) * a(0, 1)) / denom;
        double h1 = (b(1) * a(0, 0) - b(0) * a(0, 1)) / denom;
        h(0, col) = std::max(0.0, h0);
        h(1, col) = std::max(0.0, h1);
    }
}

// Fast rank-2 NNLS solver in-place (ported from original nnls2InPlace)
inline void nnls2InPlace(const Eigen::Matrix2d& a, const double denom,
                          Eigen::MatrixXd& w, bool nonneg) {
    for (unsigned int j = 0; j < w.cols(); ++j) {
        double b0 = w(0, j);
        double b1 = w(1, j);
        
        if (!nonneg) {
            w(0, j) = (b0 * a(1, 1) - b1 * a(0, 1)) / denom;
            w(1, j) = (b1 * a(0, 0) - b0 * a(0, 1)) / denom;
        } else {
            w(0, j) = std::max(0.0, (b0 * a(1, 1) - b1 * a(0, 1)) / denom);
            w(1, j) = std::max(0.0, (b1 * a(0, 0) - b0 * a(0, 1)) / denom);
        }
    }
}

// ============================================================================
// BIPARTITION IMPLEMENTATIONS (ported from original c_bipartition_sparse/dense)
// ============================================================================

// Sparse matrix bipartition
template<typename SpMat>
inline bipartitionModel c_bipartition_sparse(
    const SpMat& A,
    Eigen::MatrixXd w,
    const std::vector<unsigned int> samples,
    const double tol,
    const bool nonneg,
    const bool calc_dist,
    const unsigned int maxit,
    const bool verbose) {
    
    // Rank-2 NMF
    Eigen::MatrixXd w_it, h(w.rows(), samples.size());
    Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    
    FACTORNET_LOG_INFO(verbose, "\n%4s | %8s \n---------------\n", "iter", "tol");
    
    double tol_ = 1;
    for (unsigned int iter = 0; iter < maxit && tol_ > tol; ++iter) {
        w_it = w;
        
        // Update h
        Eigen::Matrix2d a = w * w.transpose();
        double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::Vector2d b(0, 0);
            for (typename SpMat::InnerIterator it(A, samples[i]); it; ++it) {
                const double val = it.value();
                const unsigned int r = it.row();
                b(0) += val * w(0, r);
                b(1) += val * w(1, r);
            }
            nnls2(a, b, denom, h, i, nonneg);
        }
        scale(d, h);
        
        // Update w
        a = h * h.transpose();
        denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        w.setZero();
        for (unsigned int i = 0; i < h.cols(); ++i) {
            for (typename SpMat::InnerIterator it(A, samples[i]); it; ++it) {
                for (unsigned int j = 0; j < 2; ++j) {
                    w(j, it.row()) += it.value() * h(j, i);
                }
            }
        }
        nnls2InPlace(a, denom, w, nonneg);
        scale(d, w);
        
        tol_ = cor(w, w_it);
        FACTORNET_LOG_INFO(verbose, "%4d | %8.2e\n", iter + 1, tol_);
    }
    
    // Calculate bipartitioning vector
    unsigned int size1 = 0, size2 = 0;
    std::vector<double> v(h.cols()), center1(w.cols()), center2(w.cols());
    
    if (d(0) > d(1)) {
        for (unsigned int j = 0; j < h.cols(); ++j) {
            v[j] = h(0, j) - h(1, j);
            v[j] > 0 ? ++size1 : ++size2;
        }
    } else {
        for (unsigned int j = 0; j < h.cols(); ++j) {
            v[j] = h(1, j) - h(0, j);
            v[j] > 0 ? ++size1 : ++size2;
        }
    }
    
    std::vector<unsigned int> samples1(size1), samples2(size2);
    double dist = -1;
    
    // Get indices of samples in both clusters
    unsigned int s1 = 0, s2 = 0;
    for (unsigned int j = 0; j < h.cols(); ++j) {
        if (v[j] > 0) {
            samples1[s1] = samples[j];
            ++s1;
        } else {
            samples2[s2] = samples[j];
            ++s2;
        }
    }
    
    if (calc_dist) {
        // Calculate the centers of both clusters
        center1 = centroid(A, samples1);
        center2 = centroid(A, samples2);
        
        // Calculate relative cosine similarity
        dist = rel_cosine(A, samples1, samples2, center1, center2);
    }
    
    return bipartitionModel{v, dist, size1, size2, samples1, samples2, center1, center2};
}

// Dense matrix bipartition
inline bipartitionModel c_bipartition_dense(
    const Eigen::MatrixXd& A,
    Eigen::MatrixXd w,
    const std::vector<unsigned int> samples,
    const double tol,
    const bool nonneg,
    const bool calc_dist,
    const unsigned int maxit,
    const bool verbose) {
    
    // Rank-2 NMF
    Eigen::MatrixXd w_it, h(w.rows(), samples.size());
    Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    
    FACTORNET_LOG_INFO(verbose, "\n%4s | %8s \n---------------\n", "iter", "tol");
    
    double tol_ = 1;
    for (unsigned int iter = 0; iter < maxit && tol_ > tol; ++iter) {
        w_it = w;
        
        // Update h
        Eigen::Matrix2d a = w * w.transpose();
        double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        for (unsigned int i = 0; i < h.cols(); ++i) {
            Eigen::Vector2d b(0, 0);
            for (int j = 0; j < A.rows(); ++j) {
                const double val = A(j, samples[i]);
                b(0) += val * w(0, j);
                b(1) += val * w(1, j);
            }
            nnls2(a, b, denom, h, i, nonneg);
        }
        scale(d, h);
        
        // Update w
        a = h * h.transpose();
        denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        w.setZero();
        for (unsigned int i = 0; i < h.cols(); ++i) {
            for (int j = 0; j < A.rows(); ++j) {
                for (unsigned int l = 0; l < 2; ++l) {
                    w(l, j) += A(j, samples[i]) * h(l, i);
                }
            }
        }
        nnls2InPlace(a, denom, w, nonneg);
        scale(d, w);
        
        tol_ = cor(w, w_it);
        FACTORNET_LOG_INFO(verbose, "%4d | %8.2e\n", iter + 1, tol_);
    }
    
    // Calculate bipartitioning vector
    unsigned int size1 = 0, size2 = 0;
    std::vector<double> v(h.cols()), center1(w.cols()), center2(w.cols());
    
    if (d(0) > d(1)) {
        for (unsigned int j = 0; j < h.cols(); ++j) {
            v[j] = h(0, j) - h(1, j);
            v[j] > 0 ? ++size1 : ++size2;
        }
    } else {
        for (unsigned int j = 0; j < h.cols(); ++j) {
            v[j] = h(1, j) - h(0, j);
            v[j] > 0 ? ++size1 : ++size2;
        }
    }
    
    std::vector<unsigned int> samples1(size1), samples2(size2);
    double dist = -1;
    
    // Get indices of samples in both clusters
    unsigned int s1 = 0, s2 = 0;
    for (unsigned int j = 0; j < h.cols(); ++j) {
        if (v[j] > 0) {
            samples1[s1] = samples[j];
            ++s1;
        } else {
            samples2[s2] = samples[j];
            ++s2;
        }
    }
    
    if (calc_dist) {
        // Calculate the centers of both clusters
        center1 = centroid(A, samples1);
        center2 = centroid(A, samples2);
        
        // Calculate relative cosine similarity
        dist = rel_cosine(A, samples1, samples2, center1, center2);
    }
    
    return bipartitionModel{v, dist, size1, size2, samples1, samples2, center1, center2};
}

// ============================================================================
// MAIN BIPARTITION INTERFACE
// ============================================================================

// Generic bipartition function (dispatches to sparse/dense implementation)
template<typename Mat>
inline bipartitionModel bipartition(
    const Mat& A,
    const double tol,
    const unsigned int maxit,
    const bool nonneg,
    const std::vector<unsigned int>& samples,
    unsigned int seed,
    const bool verbose = false,
    const bool calc_dist = false,
    const bool diag = true) {
    
    // Initialize w (2 x n_features)
    FactorNet::rng::SplitMix64 rng(seed);
    Eigen::MatrixXd w(2, A.rows());
    for (int i = 0; i < w.rows(); ++i) {
        for (int j = 0; j < w.cols(); ++j) {
            w(i, j) = rng.uniform();
        }
    }
    
    // Dispatch to appropriate implementation
    if constexpr (is_sparse_v<Mat>) {
        return c_bipartition_sparse(A, w, samples, tol, nonneg, calc_dist, maxit, verbose);
    } else {
        return c_bipartition_dense(A, w, samples, tol, nonneg, calc_dist, maxit, verbose);
    }
}

}  // namespace clustering

}  // namespace FactorNet

#endif // FactorNet_CLUSTERING_BIPARTITION_HPP
