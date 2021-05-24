// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_SOLVE_H
#define RCPPML_SOLVE_H

#ifndef RCPPML_H
#include "RcppML.h"
#endif

namespace RcppML {

    template <typename T>
    class CoeffMatrix {

    public:
        // CLASS CONSTRUCTOR
        CoeffMatrix(Eigen::Matrix<T, -1, -1>& a, bool nonneg = false, int L0 = -1, T L1 = 0, T L2 = 0, T RL2 = 0, int maxit = 100, T tol = 1e-8, std::string L0_method = "ggperm") :
            nonneg(nonneg), rank(a.rows()), L0(L0), maxit(maxit), tol(tol), L1(L1), L2(L2), RL2(RL2), L0_method(L0_method), a(a) {

            if (L0 >= rank || L0 <= 0) L0 = -1;

            // add L2/Ridge regularization penalty
            if (L2 != 0)
                for (int i = 0; i < a.rows(); ++i)
                    a(i, i) += L2;

            // add Pattern Extraction/inverse L2/trough/angular regularization penalty
            if (RL2 != 0)
                for (int i = 0; i < a.rows(); ++i)
                    for (int j = 0; j < i; ++j)
                        if (i != j) {
                            a(i, j) += RL2;
                            a(j, i) += RL2;
                        }
            a_llt = a.llt();
        }

        // GENERAL SOLVER
        // Unconstrained or non-negative least squares solution for "x" in "a * x = b" where "a" is symmetric positive definite
        // generalized interface for private specialized subroutines
        template<typename Derived>
        Eigen::Matrix<T, -1, 1> solve(Eigen::MatrixBase<Derived>& b) {
            if (L1 != 0) for (int i = 0; i < rank; ++i) b[i] -= L1;

            if (L0 == 1) return solve_L0_1(b);

            // initialize with an unconstrained least squares solution
            Eigen::Matrix<T, -1, 1> x = a_llt.solve(b);

            // NNLS only: iterative feasible set selection while unconstrained solutions on the feasible set give any negative values
            if (nonneg) while ((x.array() < 0).any()) solve_nonzeros(b, x);

            // refine the solution by coordinate descent
            if (maxit > 0) solve_scd(b, x);

            if (L0 != -1 && cardinality(x) > L0) {
                if (L0_method == "exact") return solve_L0_exact(b);
                do reduce_rank(b, x); while (cardinality(x) > L0);
            }
            return x;
        }

        // COORDINATE DESCENT GENERAL SOLVER
        template<typename Derived, typename OtherDerived>
        void solve(const Eigen::MatrixBase<Derived>& b, Eigen::MatrixBase<OtherDerived>& x) {
            if (L1 != 0) for (int i = 0; i < rank; ++i) b[i] -= L1;
            if (L0 == 1) x = solve_L0_1(b);
            else {
                solve_scd(b, x);
                if (L0 != -1 && cardinality(x) > L0) {
                    if (L0_method == "exact") x = solve_L0_exact(b);
                    do reduce_rank(b, x); while (cardinality(x) > L0);
                }
            }
        }

    private:
        // CLASS MEMBERS
        bool nonneg;
        int rank, L0, maxit;
        T tol, L1, L2, RL2;
        std::string L0_method;
        Eigen::Matrix<T, -1, -1> a;
        Eigen::LLT <Eigen::Matrix<T, -1, -1>> a_llt;

        // NUMBER OF NON-ZERO VALUES IN SOLUTION
        int cardinality(const Eigen::Matrix<T, -1, 1>& x) {
            int k = 0;
            for (int i = 0; i < rank; ++i)
                if (x(i) != 0) ++k;
            return k;
        }

        // ERROR/RESIDUALS OF SOLUTION
        // calculate the error of "a * x - b" for a given solution of "x" to the system "a * x = b"
        template <typename Derived>
        T calc_error(const Eigen::MatrixBase<Derived>& b, const Eigen::Matrix<T, -1, 1>& x) {
            T x_err = 0;
            Eigen::Matrix<T, -1, 1> b0 = a * x;
            for (int i = 0; i < x.size(); ++i) x_err += std::pow(b0[i] - b[i], 2);
            return x_err;
        }

        // ITERATIVE FEASIBLE SET SELECTION
        // find the best solution to the system "a * x = b" where only positive indices in initial "x" are considered
        template<typename Derived>
        void solve_nonzeros(const Eigen::MatrixBase<Derived>& b, Eigen::Matrix<T, -1, 1>& x) {
            // get indices of positive values in x (i.e. the "feasible set")
            int n_gtz = 0;
            for (int i = 0; i < x.size(); ++i) if (x(i) > 0) ++n_gtz;
            Eigen::VectorXi feasible_set(n_gtz);
            Eigen::Matrix<T, -1, 1> bsub(n_gtz);
            for (int i = 0, j = 0; i < x.size(); ++i)
                if (x(i) > 0) {
                    feasible_set(j) = i;
                    bsub(j) = b(i);
                    ++j;
                }

            // subset "a" at feasible set indices
            Eigen::Matrix<T, -1, -1> asub = submat(a, feasible_set, feasible_set);

            // solve for x given "a" and "b" at feasible set indices and map the solution back to a full-length vector "x"
            Eigen::Matrix<T, -1, 1> xsub = asub.llt().solve(bsub);
            x.setZero();
            for (int i = 0; i < n_gtz; ++i) x(feasible_set(i)) = xsub(i);
        }

        // SEQUENTIAL COORDINATE DESCENT
        // Refines "x" in-place
        template<typename Derived>
        void solve_scd(const Eigen::MatrixBase<Derived>& b, Eigen::Matrix<T, -1, 1>& x) {
            Eigen::Matrix<T, -1, 1> gradient = a * x;
            for (int i = 0; i < rank; ++i) gradient(i) -= b(i);

            T tol_ = tol + 1;
            for (int it = 0; it < maxit && tol_ > tol; ++it) {
                tol_ = 0;
                for (int i = 0; i < rank; ++i) {
                    // update this solution coefficient to minimize difference to the gradient
                    T tmp = x(i) - gradient(i) / a(i, i);

                    // apply non-negativity constraint if applicable
                    if (nonneg && tmp < 0) tmp = 0;
                    T diff = tmp - x(i);

                    if (tmp != x(i)) {
                        // update the gradient based on the residuals between the current coefficient and the optimal coefficient
                        for (int j = 0; j < rank; ++j) gradient(j) += a(j, i) * diff;

                        // calculate tolerance of the update, used as stopping criteria
                        T tmp_tol = 2 * std::abs(diff) / (tmp + x(i));
                        if (tmp_tol > tol_) tol_ = tmp_tol;
                        x(i) = tmp;
                    }
                }
            }
        }

        // EXACT L0 = 1 SOLUTION
        template <typename Derived>
        Eigen::Matrix<T, -1, 1> solve_L0_1(const Eigen::MatrixBase<Derived>& b) {

            // find the exact least squares solution for each index if all other indices were zero
            Eigen::Matrix<T, -1, 1> solutions = b;
            for (int i = 0; i < solutions.size(); ++i) {
                if (nonneg && solutions(i) < 0) solutions(i) = 0;
                else solutions(i) /= a(i, i);
            }

            // calculate the squared error for each possible solution
            Eigen::Matrix<T, -1, 1> errors(solutions.size());
            for (int i = 0; i < solutions.size(); ++i)
                for (int j = 0; j < b.size(); ++j)
                    errors(i) += std::pow(b(j) - a(j, i) * solutions(i), 2);

            // find the index of the solution with the least error (index_min)
            T min_val = errors(0);
            int index_min = 0;
            for (int i = 0; i < errors.size(); ++i)
                if (errors(i) < min_val) {
                    min_val = errors(i);
                    index_min = i;
                }
            Eigen::Matrix<T, -1, 1> x(b.size());
            x(index_min) = solutions(index_min);
            return x;
        }

        // EXACT L0 SOLUTION FOR THE GENERAL CASE
        template<typename Derived>
        Eigen::Matrix<T, -1, 1> solve_L0_exact(const Eigen::MatrixBase<Derived>& b) {

            int iter = 0;
            std::vector<bool> v(rank);
            std::fill(v.end() - L0, v.end(), true);
            Eigen::VectorXi feasible_set(L0);
            Eigen::Matrix<T, -1, 1> x_new(rank), x(rank), bsub(L0);
            T best_err = b.sum();
            do {
                ++iter;
                if (iter % 1000 == 0) Rcpp::checkUserInterrupt();

                // solve unconstrained least squares for the given feasible set permutation
                int n_gtz = 0;
                for (int i = 0; i < rank; ++i)
                    if (v[i]) {
                        feasible_set(n_gtz) = i;
                        bsub(n_gtz) = b(i);
                        ++n_gtz;
                    }
                Eigen::Matrix<T, -1, -1> asub = submat(a, feasible_set, feasible_set);
                Eigen::Matrix<T, -1, 1> xsub = asub.llt().solve(bsub);
                x_new.setZero();
                for (int i = 0; i < n_gtz; ++i) x_new(feasible_set(i)) = xsub(i);

                // check error
                T curr_err = calc_error(b, x_new);
                if (curr_err < best_err) {
                    best_err = curr_err;
                    x = x_new;
                }
            } while (std::next_permutation(v.begin(), v.end()));
            return x;
        }

        // L0 SOLVER
        // reduce the cardinality/rank of the solution by 1
        template<typename Derived>
        void reduce_rank(Eigen::MatrixBase<Derived>& b, Eigen::Matrix<T, -1, 1>& x) {
            Eigen::Matrix<T, -1, 1> x_new(rank);

            // bind the non-zero coefficient to zero that increases error least when bound to zero
            T x_err = 0;
            for (int i = 0; i < rank; ++i) {
                if (x(i) != 0) {
                    x_new = x;
                    x_new(i) = 0;
                    solve_nonzeros(b, x_new);
                    if (nonneg) for (int j = 0; j < rank; ++j) if (x_new[j] < 0) x_new[j] = 0;
                    Eigen::Matrix<T, -1, 1> b0 = a * x_new;
                    T x_new_err = 0;
                    for (int j = 0; j < rank; ++j) x_new_err += std::pow(b0[j] - b[j], 2);
                    if (x_new_err < x_err || x_err == 0) {
                        x_err = x_new_err;
                        x = x_new;
                    }
                }
            }

            // exhaustive permutation
            // try permuting feasible set var "i" for active set var "j" for all i/j pairs
            // repeat permutation until no permutations which reduce error are discovered
            if (L0_method == "eperm") {
                for (;;) {
                    T x_new_err = x_err;
                    for (int i = 0; i < rank; ++i) {
                        if (x(i) != 0) { // variable i is in the feasible set
                            for (int j = 0; j < rank; ++j) {
                                if (x(j) == 0) { // variable j is not in the feasible set
                                    Eigen::Matrix<T, -1, 1> x_permutation = x;
                                    x_permutation(i) = 0;
                                    x_permutation(j) = 1;
                                    solve_nonzeros(b, x_permutation);
                                    if (nonneg) for (int k = 0; k < rank; ++k) if (x_permutation[k] < 0) x_permutation[k] = 0;
                                    T x_permutation_err = calc_error(b, x_permutation);
                                    if (x_permutation_err < x_new_err) {
                                        // i/j feasible set permutation reduced error
                                        x_new = x_permutation;
                                        x_new_err = x_permutation_err;
                                    }
                                }
                            }
                        }
                    }
                    if (x_new_err == x_err) break;
                    x = x_new;
                    x_err = calc_error(b, x);
                }
            } else if (L0_method == "ggperm") {
                // gradient-guided permutation
                for (;;) {
                    T x_new_err = x_err;
                    // consider removing each variable from the feasible set and finding the best possible substitution using coordinate descent
                    Eigen::Matrix<T, -1, 1> x_new(rank);
                    for (int i = 0; i < rank; ++i) {
                        if (x(i) != 0) { // variable i is in the feasible set
                            // compute gradient when removing variable i
                            Eigen::Matrix<T, -1, 1> x_permutation = x;
                            x_permutation(i) = 0;
                            Eigen::Matrix<T, -1, 1> b0 = a * x_permutation;
                            for (int j = 0; j < rank; ++j) b0[j] -= b[0];

                            // find coefficient with the greatest value in the gradient
                            int max_index = 0;
                            T max_value = b0[0];
                            for (int j = 0; j < rank; ++j) if (b0[j] > max_value) max_index = j;

                            // if the variable with the largest error is not the one we just removed, substitute and calculate change in error
                            if (max_index != i) {
                                x_permutation(max_index) = 1;
                                solve_nonzeros(b, x_permutation);
                                T x_permutation_err = calc_error(b, x_permutation);

                                // this substitution improved the error, keep it and set it as the new benchmark
                                if (x_permutation_err < x_new_err) {
                                    x_new = x_permutation;
                                    x_new_err = x_permutation_err;
                                }
                            }
                        }
                    }
                    if (x_new_err < x_err) {
                        x = x_new;
                        x_err = x_new_err;
                    } else break;
                }
            }
        }
    };

} // using namespace RcppML

#endif // RCPPML_SOLVE_H