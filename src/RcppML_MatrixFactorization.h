// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_MATRIXFACTORIZATION_H
#define RCPPML_MATRIXFACTORIZATION_H

#ifndef RCPPML_SOLVE_H
#include "RcppML_Solve.h"
#endif
#ifndef RCPPML_SPARSEMATRIX_H
#include "RcppML_SparseMatrix.h"
#endif

namespace RcppML {

    // correlation distance between H and another matrix (i.e. "H" from a previous model)
    template <typename T>
    T cor(Eigen::Matrix<T, -1, -1>& x, Eigen::Matrix<T, -1, -1>& y) {
        T x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
        const unsigned int n = x.size();
        for (unsigned int i = 0; i < n; ++i) {
            x_i = (*(x.data() + i));
            y_i = (*(y.data() + i));
            sum_x += x_i;
            sum_y += y_i;
            sum_xy += x_i * y_i;
            sum_x2 += x_i * x_i;
            sum_y2 += y_i * y_i;
        }
        return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    }

    template <typename T>
    class MatrixFactorization {
    public:
        MatrixFactorization(const Eigen::SparseMatrix<T>& A, int rank) : A(A), k(rank) {
            W = Eigen::Matrix<T, -1, -1>(k, A.rows());
            H = Eigen::Matrix<T, -1, -1>(k, A.cols());
            H.setRandom();
            H.cwiseAbs();
            D = Eigen::Matrix<T, -1, 1>(k);
            D.setOnes();
            threads = Eigen::nbThreads();
        }
        void nonneg() { nonneg_W = true; nonneg_H = true; }
        void nonnegW() { nonneg_W = true; }
        void nonnegH() { nonneg_H = true; }
        void nonneg(bool n) { nonneg_W = n; nonneg_H = n; }
        void nonnegW(bool n) { nonneg_W = n; }
        void nonnegH(bool n) { nonneg_H = n; }
        void L0(int n) { L0_W = n; L0_H = n; if (n >= k || n <= 0) { L0_W = -1; L0_H = -1; } }
        void L0W(int n) { L0_W = n; if (n >= k || n <= 0) L0_W = -1; }
        void L0H(int n) { L0_H = n; if (n >= k || n <= 0) L0_H = -1; }
        void L1(T n) { L1_W = n; L1_H = n; }
        void L1W(T n) { L1_W = n; }
        void L1H(T n) { L1_H = n; }
        void L2(T n) { L2_W = n; L2_H = n; }
        void L2W(T n) { L2_W = n; }
        void L2H(T n) { L2_H = n; }
        void withDiagonalization() { fit_with_diag = true; }
        void withNoDiagonalization() { fit_with_diag = false; }
        void L2H(T n) { L2_H = n; }
        void RL2(T n) { RL2_W = n; RL2_H = n; }
        void RL2W(T n) { RL2_W = n; }
        void RL2H(T n) { RL2_H = n; }
        void L0method(std::string s) { L0_method = s; }
        void solveMaxit(int n) { solve_maxit = n; }
        void solveTol(T n) { solve_tol = n; }
        void trackLoss() { calc_losses = true; }
        void trackLoss(bool n) { calc_losses = n; }

        int rank() { return k; }
        void MatrixW(Eigen::Matrix<T, -1, -1>& matrixW) { W = matrixW; }
        void MatrixH(Eigen::Matrix<T, -1, -1>& matrixH) { H = matrixH; }
        Eigen::Matrix<T, -1, 1> VectorD() { return D; }
        Eigen::Matrix<T, -1, -1> MatrixW() { return W.transpose(); }
        Eigen::Matrix<T, -1, -1> MatrixH() { return H; }
        T loss() { if (loss == -1) calcLoss(); return model_loss; }
        Eigen::Matrix<T, -1, 1> getLosses() { return losses; }
        Eigen::Matrix<T, -1, 1> getTols() { return tols; }

        // given A = WH and solving for "W", this is a slower alternative to solve for Wt but does not require transposition of A
        // thus, this solves A = WH for "W" rather than At = HWt for W.
        void solveWInPlace() {
            CoeffMatrix<T> a(H * H.transpose(), nonneg_W, L0_W, L1_W, L2_W, RL2_W, solve_maxit, solve_tol, L0_method);
            // compute right-hand side of linear system, "b" in place in w as b = h * A.row(i)
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < A.cols(); ++i)
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                    for (int j = 0; j < k; ++j) W(j, it.row()) += it.value * H(j, i);

            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < A.rows(); ++i)
                W.col(i) = a.solve(W.col(i));
        }

        // Solves At = W * H for W, requires a transposition of sparse matrix "A"
        void solveW() {
            if (!At_) {
                At = A.transpose();
                At_ = true;
            }
            CoeffMatrix<T> a(H * H.transpose(), nonneg_W, L0_W, L1_W, L2_W, RL2_W, solve_maxit, solve_tol, L0_method);
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < At.cols(); ++i) {
                Eigen::Matrix<T, -1, 1> b(k);
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(At, i); it; ++it)
                    for (int j = 0; j < k; ++j) b(j) += it.value() * H(j, it.row());
                W.col(i) = a.solve(b);
            }
        }

        // Solves A = Wt * H for H, where Wt.rows() and H.rows() = k
        // Equivalent to At = H * Wt
        void solveH() {
            CoeffMatrix<T> a(W * W.transpose(), nonneg_H, L0_H, L1_H, L2_H, RL2_H, solve_maxit, solve_tol, L0_method);
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < A.cols(); ++i) {
                // compute right-hand side of linear system, "b" as b = w * A.col(i)
                Eigen::Matrix<T, -1, 1> b(k);
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                    for (int j = 0; j < k; ++j) b(j) += it.value() * W(j, it.row());
                H.col(i) = a.solve(b);
            }
        }

        // Solves A = W * H for last factor in W (last row), does not require transposition of sparse matrix "A"
        void SolveLastFactorW() {
            Eigen::Matrix<T, -1, 1> a(rank);
            for (int i = 0; i <= rank; ++i)
                for (int j = 0; j < H.cols(); ++j)
                    a(i) += H(i, j) * H(rank, j);
            W.row(rank).setZero();
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < A.cols(); ++i)
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                    W(rank, it.row()) += it.value() * H(rank, i);
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < W.cols(); ++i) {
                for (int j = 0; j < rank; ++j)
                    W(rank, i) -= W(j, i) * a(j);
                W(rank, i) /= a(rank);
            }
        }

        // Solves A = W * H for last factor in H (last row)
        void SolveLastFactorH() {
            Eigen::Matrix<T, -1, 1> a(rank);
            for (int i = 0; i <= rank; ++i)
                for (int j = 0; j < W.cols(); ++j)
                    a(i) += W(i, j) * W(rank, j);
            H.row(rank).setZero();
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (int i = 0; i < A.cols(); ++i) {
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                    H(rank, i) += it.value() * W(rank, it.row());
                for (int j = 0; j < rank; ++j)
                    H(rank, i) -= H(j, i) * a(j);
                H(rank, i) /= a(rank);
            }
        }

        // Fits the model using alternating least squares projections of W and H
        void fit(T tol = 1e-4, int maxit = 100, bool verbose = false) {
            if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
            Eigen::Matrix<T, -1, -1> H_prev;
            for (int i = 1; i <= maxit; ++i) {
                H_prev = H;
                solveW();
                if (fit_with_diag) scaleW();
                solveH();
                if (fit_with_diag) scaleH();
                model_tol = cor(H, H_prev);
                tols.push(model_tol);
                if (calc_losses) losses.push(calcLoss());
                if (verbose) {
                    Rcpp::checkUserInterrupt();
                    Rprintf("%4d | %8.2e\n", i, model_tol);
                }
                if (model_tol < tol) break;
            }
            if (fit_with_diag) OrderByDiag();
        }

        // add a random factor to the model
        void addFactor() {
            ++rank;
            H.conservativeResize(rank, -1);
            W.conservativeResize(rank, -1);
            H.row(rank).setRandom();
            H.row(rank).cwiseAbs();
        }

    private:
        Eigen::SparseMatrix<T>& A;
        Eigen::SparseMatrix<T> At;
        Eigen::Matrix<T, -1, -1> W, H;
        Eigen::Matrix<T, -1, 1> D;
        bool nonneg_W = false, nonneg_H = false, At_ = false, fit_with_diag = true, calc_losses = false;
        int k, solve_maxit = 100, L0_W = -1, L0_H = -1, threads;
        T solve_tol = 1e-8, L1_W = 0, L1_H = 0, L2_W = 0, L2_H = 0, RL2_W = 0, RL2_H = 0, model_loss = -1, model_tol = -1;
        Eigen::Matrix<T, -1, 1> tols, losses;
        std::string L0_method = "ggperm";

        void calcLoss() {
            T model_loss = 0;
            Eigen::Matrix<T, -1, -1> Wt = W.transpose();
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < A.cols(); ++i) {
                Eigen::Matrix<T, -1, 1> WH_i = Wt * H.col(i);
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
                    WH_i(it.row()) -= it.value();
                for (unsigned int j = 0; j < WH_i.size(); ++j)
                    model_loss += std::pow(WH_i(j), 2);
            }
            model_loss /= (A.cols() * A.rows());
        }

        // calculate diagonal as rowsums of H, and normalize rows in H to sum to 1
        void scaleH() {
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < k; ++i) {
                D(i) = H.row(i).sum();
                H.row(i) /= D(i);
            }
        }

        // calculate diagonal as rowsums of W, and normalize rows in W to sum to 1
        void scaleW() {
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < k; ++i) {
                D(i) = W.row(i).sum();
                W.row(i) /= D(i);
            }
        }

        // reorder W, D, and H by decreasing diagonal value
        void OrderByDiag() {
            // calculate sort index based on decreasing diagonal values
            std::vector<int> sort_index(k);
            std::iota(sort_index.begin(), sort_index.end(), 0);
            std::vector<T> d(k);
            for (int i = 0; i < k; ++i) d[i] = D[i];
            sort(sort_index.begin(), sort_index.end(), [&d](size_t i1, size_t i2) {return d[i1] > d[i2];});

            // reorder W, D, and H by the sort index
            Eigen::Matrix<T, -1, -1> W_temp = W;
            Eigen::Matrix<T, -1, -1> H_temp = H;
            Eigen::Matrix<T, -1, 1> D_temp = D;
            for (unsigned int i = 0; i < k; ++i) {
                W.row(i) = W_temp.row(sort_index[i]);
                H.row(i) = W_temp.row(sort_index[i]);
                D(i) = D_temp(sort_index[i]);
            }
        }
    };

} // namespace RcppML

#endif // RCPPML_MATRIXFACTORIZATION_H