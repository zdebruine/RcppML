// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_SOLVE2_H
#define RCPPML_SOLVE2_H

#ifndef RCPPML_H
#include "RcppML.h"
#endif

namespace RcppML {

    template <typename T>
    class CoeffMatrix2 {

    public:
        // CLASS CONSTRUCTOR
        CoeffMatrix2(Eigen::Matrix<T, 2, 2>& a, bool nonneg = false, int L0 = 2) :
            a(a), nonneg(nonneg), denom(a(0, 0)* a(1, 1) - a(0, 1) * a(0, 1)) {}

        // 2-variable systems of equations in the form "a * x = b" may be solved directly as follows:
        //
        // [a11 a12] [x1] = [b1]    x1 = (a22b1 - a12b2) / (a11a22 - a12a12)
        // [a12 a22] [x2] = [b2]    x2 = (a11b2 - a12b1) / (a11a22 - a12a12)
        //
        // Note that the denominator (a11a22 - a12a12) is constant and depends only on "a" and therefore can be calculated
        // just once for many systems in a linear model projection

        // 2-variable NNLS in-place in "b"
        template <typename Derived>
        void solveInPlace(Eigen::MatrixBase<Derived>& b) {
            if (nonneg) {
                T a01b1 = a(0, 1) * b(1);
                T a11b0 = a(1, 1) * b(0);
                if (a11b0 < a01b1) {
                    b(0) = 0;
                    b(1) /= a(1, 1);
                } else {
                    T a01b0 = a(0, 1) * b(0);
                    T a00b1 = a(0, 0) * b(1);
                    if (a00b1 < a01b0) {
                        b(0) /= a(0, 0);
                        b(1) = 0;
                    } else {
                        b(0) = (a11b0 - a01b1) / denom;
                        b(1) = (a00b1 - a01b0) / denom;
                    }
                }
            } else {
                T b0 = b[0];
                b(0) = (a(1, 1) * b(0) - a(0, 1) * b(1)) / denom;
                b(1) = (a(0, 0) * b(1) - a(0, 1) * b(0)) / denom;
            }
        }

        template <typename Derived, typename OtherDerived>
        void solveInPlace(Eigen::MatrixBase<Derived>& b, Eigen::MatrixBase<OtherDerived>& x) {
            if (nonneg) {
                T a01b1 = a(0, 1) * b(1);
                T a11b0 = a(1, 1) * b(0);
                if (a11b0 < a01b1) {
                    x(0) = 0;
                    x(1) = b(1) / a(1, 1);
                } else {
                    T a01b0 = a(0, 1) * b(0);
                    T a00b1 = a(0, 0) * b(1);
                    if (a00b1 < a01b0) {
                        x(0) = b(0) / a(0, 0);
                        x(1) = 0;
                    } else {
                        x(0) = (a11b0 - a01b1) / denom;
                        x(1) = (a00b1 - a01b0) / denom;
                    }
                }
            } else {
                x(0) = (a(1, 1) * b(0) - a(0, 1) * b(1)) / denom;
                x(1) = (a(0, 0) * b(1) - a(0, 1) * b(0)) / denom;
            }
        }

        template <typename Derived>
        Eigen::MatrixBase<Derived> solve(Eigen::MatrixBase<Derived>& b) {
            Eigen::MatrixBase<Derived> x = b;
            solveInPlace(x);
            return x;
        }

        // 2-variable least squares solution with L0 = 1
        template <typename Derived>
        Eigen::Matrix<T, 2, 1> solve_L0(Eigen::MatrixBase<Derived>& b) {
            Eigen::Matrix<T, 2, 1> x;
            x << b(0) / a(0, 0), b(1) / a(1, 1);
            if (nonneg) {
                if (x(0) < 0) x(0) = 0;
                if (x(1) < 0) x(1) = 0;
            }
            T err0 = std::pow(b(0) - a(0, 0) * x(0), 2) + std::pow(b(1) - a(1, 0) * x(0), 2);
            T err1 = std::pow(b(0) - a(0, 1) * x(1), 2) + std::pow(b(1) - a(1, 1) * x(1), 2);
            if (err0 < err1) x(1) = 0;
            else x(0) = 0;
            return x;
        }

        template <typename Derived>
        void solveInPlace_L0(Eigen::MatrixBase<Derived>& b) {
            Eigen::Matrix<T, 2, 1> x = solve_L0(b);
            b(0) = x(0);
            b(1) = x(1);
        }

    private:
        // CLASS MEMBERS
        Eigen::Matrix<T, 2, 2> a;
        bool nonneg;
        T denom;

    };

} // namespace RcppML

#endif // RCPPML_SOLVE2_H