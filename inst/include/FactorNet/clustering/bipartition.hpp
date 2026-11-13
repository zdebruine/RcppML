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
#include <cmath>
#include <vector>
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
    if (samples.empty()) return center;
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
    if (samples.empty()) return center;
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
    if (samples.empty()) return center;
    
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

// ----------------------------------------------------------------------------
// Relative cosine distance of a bipartition
//
// For every sample x the cosine similarity to the center of its own cluster
// and to the center of the other cluster is computed. The measure is the mean
// over samples of
//
//     cosdist(x, c_other) - cosdist(x, c_own) = cos(x, c_own) - cos(x, c_other)
//
// It lies in [-1, 1], is close to 0 when the two centers point in the same
// direction (an arbitrary split of a homogeneous cluster) and approaches 1
// when the two clusters are orthogonal. Being a mean of cosines it does not
// depend on the number of features, the number of samples or the scale of the
// data. All-zero samples and all-zero centers contribute 0, so the result is
// always finite. A degenerate split with an empty child returns 0.
// ----------------------------------------------------------------------------
namespace detail {
inline double rel_cosine_term(double x_own, double x_other, double xx,
                              double norm_own, double norm_other) {
    if (!(xx > 0)) return 0.0;
    const double xn = std::sqrt(xx);
    const double cos_own = (norm_own > 0) ? x_own / (xn * norm_own) : 0.0;
    const double cos_other = (norm_other > 0) ? x_other / (xn * norm_other) : 0.0;
    const double val = cos_own - cos_other;
    return std::isfinite(val) ? val : 0.0;
}
}  // namespace detail

// Relative cosine distance for sparse matrix
template<typename SpMat>
inline double rel_cosine(const SpMat& A,
                         const std::vector<unsigned int>& samples1,
                         const std::vector<unsigned int>& samples2,
                         const std::vector<double>& center1,
                         const std::vector<double>& center2) {
    const double norm1 = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), 0.0));
    const double norm2 = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), 0.0));
    const size_t n_total = samples1.size() + samples2.size();
    // A split with an empty side is no split at all: report zero separation
    if (samples1.empty() || samples2.empty()) return 0.0;
    double total = 0;

    for (unsigned int s = 0; s < samples1.size(); ++s) {
        double x_c1 = 0, x_c2 = 0, xx = 0;
        for (typename SpMat::InnerIterator it(A, samples1[s]); it; ++it) {
            const double val = it.value();
            x_c1 += center1[it.row()] * val;
            x_c2 += center2[it.row()] * val;
            xx += val * val;
        }
        total += detail::rel_cosine_term(x_c1, x_c2, xx, norm1, norm2);
    }

    for (unsigned int s = 0; s < samples2.size(); ++s) {
        double x_c1 = 0, x_c2 = 0, xx = 0;
        for (typename SpMat::InnerIterator it(A, samples2[s]); it; ++it) {
            const double val = it.value();
            x_c1 += center1[it.row()] * val;
            x_c2 += center2[it.row()] * val;
            xx += val * val;
        }
        total += detail::rel_cosine_term(x_c2, x_c1, xx, norm2, norm1);
    }

    return total / static_cast<double>(n_total);
}

// Relative cosine distance for dense matrix
inline double rel_cosine(const Eigen::MatrixXd& A,
                         const std::vector<unsigned int>& samples1,
                         const std::vector<unsigned int>& samples2,
                         const std::vector<double>& center1,
                         const std::vector<double>& center2) {
    const double norm1 = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), 0.0));
    const double norm2 = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), 0.0));
    const size_t n_total = samples1.size() + samples2.size();
    // A split with an empty side is no split at all: report zero separation
    if (samples1.empty() || samples2.empty()) return 0.0;
    double total = 0;

    for (unsigned int s = 0; s < samples1.size(); ++s) {
        double x_c1 = 0, x_c2 = 0, xx = 0;
        for (int r = 0; r < A.rows(); ++r) {
            const double val = A(r, samples1[s]);
            x_c1 += center1[r] * val;
            x_c2 += center2[r] * val;
            xx += val * val;
        }
        total += detail::rel_cosine_term(x_c1, x_c2, xx, norm1, norm2);
    }

    for (unsigned int s = 0; s < samples2.size(); ++s) {
        double x_c1 = 0, x_c2 = 0, xx = 0;
        for (int r = 0; r < A.rows(); ++r) {
            const double val = A(r, samples2[s]);
            x_c1 += center1[r] * val;
            x_c2 += center2[r] * val;
            xx += val * val;
        }
        total += detail::rel_cosine_term(x_c2, x_c1, xx, norm2, norm1);
    }

    return total / static_cast<double>(n_total);
}

// Scale matrix rows by their L1 norms (sum of absolute values).
// For non-negative factors this is the row sum; for signed factors
// (nonneg = false) the absolute sum keeps the scaling positive.
inline void scale(Eigen::VectorXd& d, Eigen::MatrixXd& w) {
    d = w.cwiseAbs().rowwise().sum();
    d.array() += tiny_num<double>();
    for (unsigned int i = 0; i < w.rows(); ++i) {
        for (unsigned int j = 0; j < w.cols(); ++j) {
            w(i, j) /= d(i);
        }
    }
}

// Correlation distance (1 - Pearson r) between matrices for convergence check.
// Returns 0 (converged) when either matrix is constant or not finite.
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
    if (!(denominator > 0) || !std::isfinite(denominator) || !std::isfinite(numerator))
        return 0.0;

    return 1.0 - (numerator / denominator);
}

// ----------------------------------------------------------------------------
// Exact 2-variable least squares / non-negative least squares
//
// Minimizes 0.5 * x^T G x - b^T x for the symmetric positive semi-definite
// 2x2 matrix G = [a00 a01; a01 a11].
//
// nonneg = false: unconstrained solution. A singular G (collinear factors)
//                 is solved on the dominant coordinate only.
// nonneg = true : if the unconstrained solution is feasible it is returned.
//                 Otherwise the optimum lies on a face of the non-negative
//                 quadrant, i.e. it is (max(0, b0/a00), 0) or
//                 (0, max(0, b1/a11)); the one with the lower objective is
//                 returned. This is the exact NNLS solution in all sign cases.
// ----------------------------------------------------------------------------
inline void nnls2_solve(const double a00, const double a01, const double a11,
                        const double b0, const double b1, const bool nonneg,
                        double& x0, double& x1) {
    const double denom = a00 * a11 - a01 * a01;
    const bool regular = (a00 > 0) && (a11 > 0) && (denom > 1e-12 * a00 * a11);
    if (regular) {
        x0 = (b0 * a11 - b1 * a01) / denom;
        x1 = (b1 * a00 - b0 * a01) / denom;
        if (!nonneg || (x0 >= 0 && x1 >= 0)) return;
    } else if (!nonneg) {
        if (a00 >= a11) {
            x0 = (a00 > 0) ? b0 / a00 : 0.0;
            x1 = 0.0;
        } else {
            x0 = 0.0;
            x1 = b1 / a11;
        }
        return;
    }
    // Boundary candidates. Objective values: f(c0, 0) = -0.5 * b0 * c0 and
    // f(0, c1) = -0.5 * b1 * c1.
    const double c0 = (a00 > 0 && b0 > 0) ? b0 / a00 : 0.0;
    const double c1 = (a11 > 0 && b1 > 0) ? b1 / a11 : 0.0;
    if (b0 * c0 >= b1 * c1) {
        x0 = c0;
        x1 = 0.0;
    } else {
        x0 = 0.0;
        x1 = c1;
    }
}

// Rank-2 solver for one column (denom is kept for interface compatibility
// and is not used; see nnls2_solve)
inline void nnls2(const Eigen::Matrix2d& a, const Eigen::Vector2d& b,
                  const double /*denom*/, Eigen::MatrixXd& h, unsigned int col, bool nonneg) {
    double x0 = 0, x1 = 0;
    nnls2_solve(a(0, 0), a(0, 1), a(1, 1), b(0), b(1), nonneg, x0, x1);
    h(0, col) = x0;
    h(1, col) = x1;
}

// Rank-2 solver in-place: columns of w hold the right-hand sides on entry
// and the solutions on exit
inline void nnls2InPlace(const Eigen::Matrix2d& a, const double /*denom*/,
                          Eigen::MatrixXd& w, bool nonneg) {
    for (unsigned int j = 0; j < w.cols(); ++j) {
        double x0 = 0, x1 = 0;
        nnls2_solve(a(0, 0), a(0, 1), a(1, 1), w(0, j), w(1, j), nonneg, x0, x1);
        w(0, j) = x0;
        w(1, j) = x1;
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
    Eigen::MatrixXd w_it;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(w.rows(), samples.size());
    Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    
    FACTORNET_LOG_INFO(verbose, "\n%4s | %8s \n---------------\n", "iter", "tol");
    
    double tol_ = 1;
    for (unsigned int iter = 0; iter < maxit && tol_ > tol; ++iter) {
        w_it = w;
        
        // Update h
        Eigen::Matrix2d a = w * w.transpose();
        double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        // Columns are independent, so the loop is parallel and the result
        // does not depend on the number of threads. w is 2 x m column-major:
        // (w(0, r), w(1, r)) are adjacent in memory.
        {
            const int n_h = static_cast<int>(h.cols());
            const double* w_ptr = w.data();
            double* h_ptr = h.data();
            const double a00 = a(0, 0), a01 = a(0, 1), a11 = a(1, 1);
            (void)denom;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(n_h >= 256)
#endif
            for (int i = 0; i < n_h; ++i) {
                double b0 = 0, b1 = 0;
                for (typename SpMat::InnerIterator it(A, samples[i]); it; ++it) {
                    const double val = it.value();
                    const double* w_r = w_ptr + 2 * static_cast<size_t>(it.row());
                    b0 += val * w_r[0];
                    b1 += val * w_r[1];
                }
                double x0 = 0, x1 = 0;
                nnls2_solve(a00, a01, a11, b0, b1, nonneg, x0, x1);
                h_ptr[2 * static_cast<size_t>(i)] = x0;
                h_ptr[2 * static_cast<size_t>(i) + 1] = x1;
            }
        }
        scale(d, h);
        
        // Update w
        a = h * h.transpose();
        denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        w.setZero();
        // Serial on purpose: a parallel scatter would make the floating-point
        // sums depend on the number of threads.
        {
            const int n_h = static_cast<int>(h.cols());
            double* w_ptr = w.data();
            const double* h_ptr = h.data();
            for (int i = 0; i < n_h; ++i) {
                const double h0 = h_ptr[2 * static_cast<size_t>(i)];
                const double h1 = h_ptr[2 * static_cast<size_t>(i) + 1];
                if (h0 == 0 && h1 == 0) continue;
                for (typename SpMat::InnerIterator it(A, samples[i]); it; ++it) {
                    const double val = it.value();
                    double* w_r = w_ptr + 2 * static_cast<size_t>(it.row());
                    w_r[0] += val * h0;
                    w_r[1] += val * h1;
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
    Eigen::MatrixXd w_it;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(w.rows(), samples.size());
    Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    
    FACTORNET_LOG_INFO(verbose, "\n%4s | %8s \n---------------\n", "iter", "tol");
    
    double tol_ = 1;
    for (unsigned int iter = 0; iter < maxit && tol_ > tol; ++iter) {
        w_it = w;
        
        // Update h
        Eigen::Matrix2d a = w * w.transpose();
        double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
        {
            const int n_h = static_cast<int>(h.cols());
            const int m_rows = static_cast<int>(A.rows());
            const double* w_ptr = w.data();
            double* h_ptr = h.data();
            const double a00 = a(0, 0), a01 = a(0, 1), a11 = a(1, 1);
            (void)denom;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(n_h >= 256)
#endif
            for (int i = 0; i < n_h; ++i) {
                const double* a_col = A.data() + static_cast<size_t>(samples[i]) * static_cast<size_t>(m_rows);
                double b0 = 0, b1 = 0;
                for (int j = 0; j < m_rows; ++j) {
                    b0 += a_col[j] * w_ptr[2 * static_cast<size_t>(j)];
                    b1 += a_col[j] * w_ptr[2 * static_cast<size_t>(j) + 1];
                }
                double x0 = 0, x1 = 0;
                nnls2_solve(a00, a01, a11, b0, b1, nonneg, x0, x1);
                h_ptr[2 * static_cast<size_t>(i)] = x0;
                h_ptr[2 * static_cast<size_t>(i) + 1] = x1;
            }
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
