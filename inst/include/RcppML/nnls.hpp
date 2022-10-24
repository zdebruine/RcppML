// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_nnls
#define RcppML_nnls

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

// Non-Negative Least Squares solver
// solve ax = b given "a", "b", and h.col(sample) giving "x", subject to non-negativity. Coordinate descent.
inline void c_nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const unsigned int sample) {
    double tol = 1;
    for (unsigned int it = 0; it < CD_MAXIT && (tol / b.size()) > CD_TOL; ++it) {
        tol = 0;
        for (unsigned int i = 0; i < h.rows(); ++i) {
            double diff = b(i) / a(i, i);
            if (-diff > h(i, sample)) {
                if (h(i, sample) != 0) {
                    b -= a.col(i) * -h(i, sample);
                    tol = 1;
                    h(i, sample) = 0;
                }
            } else if (diff != 0) {
                h(i, sample) += diff;
                b -= a.col(i) * diff;
                tol += std::abs(diff / (h(i, sample) + TINY_NUM));
            }
        }
    }
}

// Upper-bounded Non-Negative Least Squares solver
// solve ax = b given "a", "b", and h.col(sample) giving "x", subject to non-negativity. Coordinate descent.
inline void c_bnnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const unsigned int sample, const double upper_bound = 1) {
    double tol = 1;
    for (unsigned int it = 0; it < CD_MAXIT && (tol / b.size()) > CD_TOL; ++it) {
        tol = 0;
        for (unsigned int i = 0; i < h.rows(); ++i) {
            double diff = b(i) / a(i, i);
            if (-diff > h(i, sample)) {
                if (h(i, sample) != 0) {
                    b -= a.col(i) * -h(i, sample);
                    tol = 1;
                    h(i, sample) = 0;
                }
            } else if (diff != 0) {
                if (h(i, sample) + diff > upper_bound) {
                    diff = upper_bound - h(i, sample);
                    h(i, sample) = upper_bound;
                } else {
                    h(i, sample) += diff;
                }
                b -= a.col(i) * diff;
                tol += std::abs(diff / (h(i, sample) + TINY_NUM));
            }
        }
    }
}

// 2-variable Non-Negative Least Squares solver
// solves ax_i = b for a two-variable system of equations, where the "i"th column in "x" contains the solution.
// "denom" is a pre-conditioned denominator for solving by direct substition.
inline void nnls2(const Eigen::Matrix2d& a, const Eigen::Vector2d& b, const double denom, Eigen::MatrixXd& x, const unsigned int i, const bool nonneg) {
    // solve least squares
    if (nonneg) {
        const double a01b1 = a(0, 1) * b(1);
        const double a11b0 = a(1, 1) * b(0);
        if (a11b0 < a01b1) {
            x(0, i) = 0;
            x(1, i) = b(1) / a(1, 1);
        } else {
            const double a01b0 = a(0, 1) * b(0);
            const double a00b1 = a(0, 0) * b(1);
            if (a00b1 < a01b0) {
                x(0, i) = b(0) / a(0, 0);
                x(1, i) = 0;
            } else {
                x(0, i) = (a11b0 - a01b1) / denom;
                x(1, i) = (a00b1 - a01b0) / denom;
            }
        }
    } else {
        x(0, i) = (a(1, 1) * b(0) - a(0, 1) * b(1)) / denom;
        x(1, i) = (a(0, 0) * b(1) - a(0, 1) * b(0)) / denom;
    }
}

// solves ax = b for a two-variable system of equations, where "b" is given in the "i"th column of "w", and replaced with the solution "x".
// "denom" is a pre-conditioned denominator for solving by direct substitution.
inline void nnls2InPlace(const Eigen::Matrix2d& a, const double denom, Eigen::MatrixXd& w, const bool nonneg) {
    for (unsigned int i = 0; i < w.cols(); ++i) {
        if (nonneg) {
            const double a01b1 = a(0, 1) * w(1, i);
            const double a11b0 = a(1, 1) * w(0, i);
            if (a11b0 < a01b1) {
                w(0, i) = 0;
                w(1, i) /= a(1, 1);
            } else {
                const double a01b0 = a(0, 1) * w(0, i);
                const double a00b1 = a(0, 0) * w(1, i);
                if (a00b1 < a01b0) {
                    w(0, i) /= a(0, 0);
                    w(1, i) = 0;
                } else {
                    w(0, i) = (a11b0 - a01b1) / denom;
                    w(1, i) = (a00b1 - a01b0) / denom;
                }
            }
        } else {
            double b0 = w(0, i);
            w(0, i) = (a(1, 1) * b0 - a(0, 1) * w(1, i)) / denom;
            w(1, i) = (a(0, 0) * w(1, i) - a(0, 1) * b0) / denom;
        }
    }
}

#endif
