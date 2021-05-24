// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_MATRIXFACTORIZATION2_H
#define RCPPML_MATRIXFACTORIZATION2_H

#ifndef RCPPML_SOLVE_H
#include "RcppML_Solve.h"
#endif

namespace RcppML {

    // specialized class for rank-2 matrix factorizations
    template <typename T>
    class MatrixFactorization2 {
    public:
        MatrixFactorization2(const Eigen::SparseMatrix<T>& A) : A(A) {
            W = Eigen::Matrix<T, 2, -1>(2, A.rows());
            H = Eigen::Matrix<T, 2, -1>(2, A.cols());
            H.setRandom();
            H.cwiseAbs();
        }
        void nonneg() { nonneg_W = true; nonneg_H = true; }
        void nonnegW() { nonneg_W = true; }
        void nonnegH() { nonneg_H = true; }
        void nonneg(bool n) { nonneg_W = n; nonneg_H = n; }
        void nonnegW(bool n) { nonneg_W = n; }
        void nonnegH(bool n) { nonneg_H = n; }
        void L0(int n) { if (n == 1) { L0_W = 1; L0_H = 1; } else { L0_W = 2; L0_H = 2; } }
        void L0W(int n) { L0_W = (n == 1) ? 1 : 2; }
        void L0H(int n) { L0_H = (n == 1) ? 1 : 2; }
        void withDiagonalization() { fit_with_diag = true; }
        void withNoDiagonalization() { fit_with_diag = false; }

        void MatrixW(Eigen::Matrix<T, 2, -1>& matrixW) { W = matrixW; }
        void MatrixH(Eigen::Matrix<T, 2, -1>& matrixH) { H = matrixH; }
        Eigen::Matrix<T, 2, 1> VectorD() { return D; }
        Eigen::Matrix<T, 2, -1> MatrixW() { return W.transpose(); }
        Eigen::Matrix<T, 2, -1> MatrixH() { return H; }
        Eigen::Matrix<T, -1, 1> VectorV2() { return (D(0) > D(1)) ? H.row(0) - H.row(1) : H.row(1) - H.row(0); }
        Eigen::Matrix<T, -1, 1> VectorU2() { return (D(0) > D(1)) ? W.row(0) - W.row(1) : W.row(1) - W.row(0); }

        // Solves At = W * H for W
        void solveW() {
            CoeffMatrix2<T> a(H * H.transpose(), nonneg_W, L0_W);
            for (unsigned int i = 0; i < A.cols(); ++i)
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                    W.col(it.row()) += it.value() * H.col(i);
            if (L0 == 2) {
                for (int i = 0; i < A.rows(); ++i)
                    a.solveInPlace(W.col(i));
            } else {
                for (int i = 0; i < A.rows(); ++i)
                    a.solveInPlace_L0(W.col(i));
            }
        }

        void solveH() {
            CoeffMatrix2<T> a(W * W.transpose(), nonneg_H, L0_H);
            if (L0 == 2) {
                for (int i = 0; i < A.cols(); ++i) {
                    Eigen::Matrix<T, 2, 1> b(2);
                    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                        b += W.col(it.row()) * it.value();
                    a.solveInPlace(b, H.col(i));
                }
            } else {
                for (int i = 0; i < A.cols(); ++i) {
                    Eigen::Matrix<T, 2, 1> b(2);
                    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                        b += W.col(it.row()) * it.value();
                    H.col(i) = a.solve_L0(b);
                }
            }
        }

        // Fits the model using alternating least squares projections of W and H
        void fit(T tol = 1e-4, int maxit = 100, bool verbose = false) {
            if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
            Eigen::Matrix<T, 2, -1> H_prev;
            for (int i = 1; i <= maxit; ++i) {
                H_prev = H;
                solveW();
                if (fit_with_diag) scaleW();
                solveH();
                if (fit_with_diag) scaleH();
                model_tol = cor(H, H_prev);
                if (verbose) {
                    Rcpp::checkUserInterrupt();
                    Rprintf("%4d | %8.2e\n", i, model_tol);
                }
                if (model_tol < tol) break;
            }
        }

    private:
        Eigen::SparseMatrix<T>& A;
        Eigen::Matrix<T, 2, -1> W, H;
        Eigen::Matrix<T, 2, 1> D;
        bool nonneg_W = false, nonneg_H = false, fit_with_diag = true;
        int L0_W = 2, L0_H = 2;
        T model_tol = -1;

        // calculate diagonal as rowsums of H, and normalize rows in H to sum to 1
        void scaleH() {
            D(0) = H.row(0).sum();
            D(1) = H.row(1).sum();
            H.row(0) /= D(0);
            H.row(1) /= D(1);
        }

        // calculate diagonal as rowsums of W, and normalize rows in W to sum to 1
        void scaleW() {
            D(0) = H.row(0).sum();
            D(1) = H.row(1).sum();
            W.row(0) /= D(0);
            W.row(1) /= D(1);
        }
    };

} // namespace RcppML

#endif // RCPPML_MATRIXFACTORIZATION2_H