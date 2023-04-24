// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_svd
#define RcppML_svd

#define DIV_OFFSET 1e-15

namespace RcppML
{
    template <class T>
    class svd
    {
    private:
        T &A;
        T t_A;
        Eigen::MatrixXd u;
        Eigen::MatrixXd v;
        Eigen::VectorXd d;
        double tol_ = -1, mse_ = 0;
        unsigned int iter_ = 0, best_model_ = 0;

    public:
        bool verbose = true;
        unsigned int maxit = 100, threads = 0;
        std::vector<double> L1 = std::vector<double>(2), L2 = std::vector<double>(2);

        double tol = 1e-4;

        std::vector<double> debug_errs;

        // CONSTRUCTORS
        // constructor for initialization with a randomly generated "w" matrix
        svd(T &A, const unsigned int k, const unsigned int seed = 0) : A(A)
        {
            u = randomMatrix(A.rows(), k, seed);
            v = Eigen::MatrixXd(A.cols(), k);
            d = Eigen::VectorXd::Ones(k);
        }

        // constructor for initialization with an initial "u" matrix
        svd(T &A, Eigen::MatrixXd u) : A(A), u(u)
        {
            if (A.rows() != u.rows())
                Rcpp::stop("number of rows in 'A' and 'u' are not equal!");
            v = Eigen::MatrixXd(A.cols(), u.cols());
            d = Eigen::VectorXd::Ones(u.cols());
        }

        // constructor for initialization with a fully-specified model
        svd(T &A, Eigen::MatrixXd u, Eigen::MatrixXd v) : A(A), u(u), v(v)
        {
            if (A.rows() != u.rows())
                Rcpp::stop("dimensions of 'u' and 'A' are not compatible");
            if (A.cols() != v.rows())
                Rcpp::stop("dimensions of 'v' and 'A' are not compatible");
            if (u.cols() != v.cols())
                Rcpp::stop("rank of 'u' and 'v' are not equal!");
            d = Eigen::VectorXd::Ones(u.cols());
        }

        // GETTERS
        Eigen::MatrixXd matrixU() { return u; }
        Eigen::MatrixXd matrixV() { return v; }
        Eigen::VectorXd vectorD() { return d; }
        double fit_tol() { return tol_; }
        unsigned int fit_iter() { return iter_; }
        double fit_mse() { return mse_; }
        unsigned int best_model() { return best_model_; }

        // requires specialized dense and sparse backends
        double mse();

        void fit();
        template <int k> void fit_rank_k();
        
        // fit the model multiple times and return the best one
        void fit_restarts(Rcpp::List &u_init)
        {
            Eigen::MatrixXd u_best = u;
            Eigen::MatrixXd v_best = v;
            double tol_best = tol_;
            double mse_best = 0;
            for (unsigned int i = 0; i < u_init.length(); ++i)
            {
                if (verbose)
                    Rprintf("Fitting model %i/%i:", i + 1, u_init.length());
                u = Rcpp::as<Eigen::MatrixXd>(u_init[i]);
                tol_ = 1;
                iter_ = 0;
                if (u.rows() != v.rows())
                    Rcpp::stop("rank of 'u' is not equal to rank of 'v'");
                if (u.cols() != A.rows())
                    Rcpp::stop("dimensions of 'u' and 'A' are not compatible");
                fit();
                mse_ = mse();
                if (verbose)
                    Rprintf("MSE: %8.4e\n\n", mse_);
                if (i == 0 || mse_ < mse_best)
                {
                    best_model_ = i;
                    u_best = u;
                    v_best = v;
                    tol_best = tol_;
                    mse_best = mse_;
                }
            }
            if (best_model_ != (u_init.length() - 1))
            {
                u = u_best;
                v = v_best;
                tol_ = tol_best;
                mse_ = mse_best;
            }
        }
    };

    // svd class methods with specialized dense/sparse backends
    template <>
    double svd<Rcpp::SparseMatrix>::mse()
    {
        Eigen::MatrixXd u0 = u.transpose();

        // compute losses across all samples in parallel
        Eigen::ArrayXd losses(v.cols());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (unsigned int i = 0; i < v.cols(); ++i)
        {
            Eigen::VectorXd uv_i = u0 * v.col(i);
            for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                uv_i(iter.row()) -= iter.value();
            losses(i) += uv_i.array().square().sum();
        }

        // divide total loss by number of applicable measurements
        return losses.sum() / ((v.cols() * u.cols()));
    }

    template <>
    double svd<Eigen::MatrixXd>::mse()
    {
        Eigen::MatrixXd u0 = u.transpose();
        // compute losses across all samples in parallel
        Eigen::ArrayXd losses(v.cols());

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (unsigned int i = 0; i < v.cols(); ++i)
        {
            Eigen::VectorXd uv_i = u0 * v.col(i);
            for (unsigned int iter = 0; iter < A.rows(); ++iter)
            {
                uv_i(iter) -= A(iter, i);
            }
            losses(i) += uv_i.array().square().sum();
        }

        // divide total loss by number of applicable measurements
        return losses.sum() / ((v.cols() * u.cols()));
    }

    template <>
    template <int k>
    void svd<Eigen::MatrixXd>::fit_rank_k()
    {
        fit_rank_k<k - 1>();
        // alternating least squares updates
        double d_k;
        for (; iter_ < maxit; ++iter_)
        {
            Eigen::MatrixXd u_it = u.col(k);

            double a = u.col(k).dot(u.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < v.rows(); ++i)
            {
                v(i, k) = (u.col(k).dot(A.col(i)));
                if (L1[1] > 0)
                    v(i, k) -= L1[1];

                if (k > 0)
                {
                    for (int _k = k - 1; _k >= 0; --_k)
                    {
                        // TODO: Precalculate a, so that dot product is not recalculated each loop
                        v(i, k) -= u.col(k).dot(u.col(_k)) * v(i, _k);
                    }
                }

                v(i, k) /= a;
            }

            // Scale V
            v.col(k) /= v.col(k).norm() + DIV_OFFSET;

            // Update U
            a = v.col(k).dot(v.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < u.rows(); ++i)
            {
                u(i, k) = v.col(k).dot(A.row(i));
                if (L1[0] > 0)
                    u(i, k) -= L1[0];

                if (k > 0)
                {
                    for (int _k = k - 1; _k >= 0; --_k)
                    {
                        // TODO: Precalculate a, so that dot product is not recalculated each loop
                        u(i, k) -= v.col(k).dot(v.col(_k)) * u(i, _k);
                    }
                }
                u(i, k) /= a;
            }

            // Scale U
            d_k = u.col(k).norm();
            u.col(k) /= (d_k + DIV_OFFSET);

            // Check exit criteria
            tol_ = (u.col(k) - u_it).array().square().sum() * d_k; // Undo scaling for tolerance check, to avoid instability when d(k) is basically 0
            if (verbose)
                Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);

            if (tol_ < tol)
                break;
            Rcpp::checkUserInterrupt();
        }

        // "unscale" U
        u.col(k) *= d_k;
        d(k) = d_k;
    };

    template <>
    template <>
    void svd<Eigen::MatrixXd>::fit_rank_k<0>()
    {
        // alternating least squares updates
        double d_k;
        for (; iter_ < maxit; ++iter_)
        {
            Eigen::MatrixXd u_it = u.col(0);

            double a = u.col(0).dot(u.col(0)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < v.rows(); ++i)
            {
                v(i, 0) = (u.col(0).dot(A.col(i)));
                if (L1[1] > 0)
                    v(i, 0) -= L1[1];

                v(i, 0) /= a;
            }

            // Scale V
            v.col(0) /= v.col(0).norm() + DIV_OFFSET;

            // Update U
            a = v.col(0).dot(v.col(0)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < u.rows(); ++i)
            {
                u(i, 0) = v.col(0).dot(A.row(i));
                if (L1[0] > 0)
                    u(i, 0) -= L1[0];

                u(i, 0) /= a;
            }

            // Scale U
            d_k = u.col(0).norm();
            u.col(0) /= (d_k + DIV_OFFSET);

            // Check exit criteria
            tol_ = (u.col(0) - u_it).array().square().sum() * d_k; // Undo scaling for tolerance check, to avoid instability when d(k) is basically 0
            if (verbose)
                Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);

            if (tol_ < tol)
                break;
            Rcpp::checkUserInterrupt();
        }

        // "unscale" U
        u.col(0) *= d_k;
        d(0) = d_k;
    };

    template <>
    void svd<Eigen::MatrixXd>::fit()
    {
        if (u.cols() == 1)
        {
            fit_rank_k<0>();
        }
        else if (u.cols() == 2)
        {
            fit_rank_k<1>();
        }
        else if (u.cols() == 3)
        {
            fit_rank_k<2>();
        }
        else
        {
            fit_rank_k<2>();
            for (int k = 0; k < u.cols(); ++k)
            {
                // alternating least squares updates
                double d_k;
                for (; iter_ < maxit; ++iter_)
                {
                    Eigen::VectorXd u_it = u.col(k);

                    double a = u.col(k).dot(u.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
                    for (int i = 0; i < v.rows(); ++i)
                    {
                        v(i, k) = (u.col(k).dot(A.col(i)));
                        if (L1[1] > 0)
                            v(i, k) -= L1[1];

                        for (int _k = k - 1; _k >= 0; --_k)
                        {
                            // TODO: Precalculate a, so that dot product is not recalculated each loop
                            v(i, k) -= u.col(k).dot(u.col(_k)) * v(i, _k);
                        }

                        v(i, k) /= a;
                    }

                    // Scale V
                    v.col(k) /= v.col(k).norm() + DIV_OFFSET;

                    // Update U
                    a = v.col(k).dot(v.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
                    for (int i = 0; i < u.rows(); ++i)
                    {
                        u(i, k) = v.col(k).dot(A.row(i));
                        if (L1[0] > 0)
                            u(i, k) -= L1[0];

                        if (k > 0)
                        {
                            for (int _k = k - 1; _k >= 0; --_k)
                            {
                                // TODO: Precalculate a, so that dot product is not recalculated each loop
                                u(i, k) -= v.col(k).dot(v.col(_k)) * u(i, _k);
                            }
                        }
                        u(i, k) /= a;
                    }

                    // Scale U
                    d_k = u.col(k).norm();
                    u.col(k) /= (d_k + DIV_OFFSET);

                    // Check exit criteria
                    tol_ = (u.col(k) - u_it).array().square().sum() * d_k; // Undo scaling for tolerance check, to avoid instability when d(k) is basically 0
                    if (verbose)
                        Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);

                    if (tol_ < tol)
                        break;
                    Rcpp::checkUserInterrupt();
                }

                // "unscale" U
                u.col(k) *= d_k;
                d(k) = d_k;
            }
        }

        // Scale u
        for (int i = 0; i < u.cols(); ++i)
            u.col(i) /= d(i);

        if (tol_ > tol && iter_ == maxit && verbose)
            Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
    };

    template <>
    template <int k>
    void svd<Rcpp::SparseMatrix>::fit_rank_k()
    {
        fit_rank_k<k - 1>();

        Rcpp::SparseMatrix At = A.transpose();

        // alternating least squares updates
        double d_k;
        for (; iter_ < maxit; ++iter_)
        {
            Eigen::MatrixXd u_it = u.col(k);

            double a = u.col(k).dot(u.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < v.rows(); ++i)
            {
                v(i, k) = 0.0;
                for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                {
                    v(i, k) += u(iter.row(), k) * iter.value();
                    for (int _k = k - 1; _k >= 0; --_k)
                    {
                        // TODO: Precalculate a, so that dot product is not recalculated each loop
                        v(i, k) -= u.col(k).dot(u.col(_k)) * v(i, _k);
                    }
                }

                if (L1[1] > 0)
                    v(i, k) -= L1[1];

                v(i, k) /= a;
            }

            // Scale V
            v.col(k) /= v.col(k).norm() + DIV_OFFSET;

            // Update U
            a = v.col(k).dot(v.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < u.rows(); ++i)
            {
                u(i, k) = 0.0;
                for (Rcpp::SparseMatrix::InnerIterator iter(At, i); iter; ++iter)
                {
                    u(i, k) += v(iter.row(), k) * iter.value();
                    for (int _k = k - 1; _k >= 0; --_k)
                    {
                        // TODO: Precalculate a, so that dot product is not recalculated each loop
                        u(i, k) -= v.col(k).dot(v.col(_k)) * u(i, _k);
                    }
                }
                if (L1[0] > 0)
                    u(i, k) -= L1[0];

                u(i, k) /= a;
            }

            // Scale U
            d_k = u.col(k).norm();
            u.col(k) /= (d_k + DIV_OFFSET);

            // Check early exit
            if (d_k < tol)
                break;

            // Check exit criteria
            tol_ = (u.col(k) - u_it).array().square().sum(); // Undo scaling for tolerance check, to avoid instability when d(k) is basically 0
            if (verbose)
                Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);

            if (tol_ < tol)
                break;
            Rcpp::checkUserInterrupt();
        }

        // "unscale" U
        u.col(k) *= d_k;
        d(k) = d_k;
    };

    template <>
    template <>
    void svd<Rcpp::SparseMatrix>::fit_rank_k<0>()
    {
        Rcpp::SparseMatrix At = A.transpose();

        // alternating least squares updates
        double d_k;
        for (; iter_ < maxit; ++iter_)
        {
            Eigen::MatrixXd u_it = u.col(0);

            double a = u.col(0).dot(u.col(0)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < v.rows(); ++i)
            {
                v(i, 0) = 0.0;
                for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                {
                    v(i, 0) += u(iter.row(), 0) * iter.value();
                }

                if (L1[1] > 0)
                    v(i, 0) -= L1[1];

                v(i, 0) /= a;
            }

            // Scale V
            v.col(0) /= v.col(0).norm() + DIV_OFFSET;

            // Update U
            a = v.col(0).dot(v.col(0)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
            for (int i = 0; i < u.rows(); ++i)
            {
                u(i, 0) = 0.0;
                for (Rcpp::SparseMatrix::InnerIterator iter(At, i); iter; ++iter)
                {
                    u(i, 0) += v(iter.row(), 0) * iter.value();
                }
                if (L1[0] > 0)
                    u(i, 0) -= L1[0];

                u(i, 0) /= a;
            }

            // Scale U
            d_k = u.col(0).norm();
            u.col(0) /= (d_k + DIV_OFFSET);

            // Check early exit
            if (d_k < tol)
                break;

            // Check exit criteria
            tol_ = (u.col(0) - u_it).array().square().sum(); // Undo scaling for tolerance check, to avoid instability when d(k) is basically 0
            if (verbose)
                Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);

            if (tol_ < tol)
                break;
            Rcpp::checkUserInterrupt();
        }

        // "unscale" U
        u.col(0) *= d_k;
        d(0) = d_k;
    }

    template <>
    void svd<Rcpp::SparseMatrix>::fit()
    {
        if (u.cols() == 1)
        {
            fit_rank_k<0>();
        }
        else if (u.cols() == 2)
        {
            fit_rank_k<1>();
        }
        else if (u.cols() == 3)
        {
            fit_rank_k<2>();
        }
        else
        {
            fit_rank_k<2>();
            Rcpp::SparseMatrix At = A.transpose();
            for (int k = 0; k < u.cols(); ++k)
            {
                // alternating least squares updates
                double d_k;
                for (; iter_ < maxit; ++iter_)
                {
                    Eigen::MatrixXd u_it = u.col(k);

                    double a = u.col(k).dot(u.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
                    for (int i = 0; i < v.rows(); ++i)
                    {
                        v(i, k) = 0.0;
                        for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
                        {
                            v(i, k) += u(iter.row(), k) * iter.value();
                            for (int _k = k - 1; _k >= 0; --_k)
                            {
                                // TODO: Precalculate a, so that dot product is not recalculated each loop
                                v(i, k) -= u.col(k).dot(u.col(_k)) * v(i, _k);
                            }
                        }

                        if (L1[1] > 0)
                            v(i, k) -= L1[1];

                        v(i, k) /= a;
                    }

                    // Scale V
                    v.col(k) /= v.col(k).norm() + DIV_OFFSET;

                    // Update U
                    a = v.col(k).dot(v.col(k)) + DIV_OFFSET;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
                    for (int i = 0; i < u.rows(); ++i)
                    {
                        u(i, k) = 0.0;
                        for (Rcpp::SparseMatrix::InnerIterator iter(At, i); iter; ++iter)
                        {
                            u(i, k) += v(iter.row(), k) * iter.value();
                            for (int _k = k - 1; _k >= 0; --_k)
                            {
                                // TODO: Precalculate a, so that dot product is not recalculated each loop
                                u(i, k) -= v.col(k).dot(v.col(_k)) * u(i, _k);
                            }
                        }
                        if (L1[0] > 0)
                            u(i, k) -= L1[0];

                        u(i, k) /= a;
                    }

                    // Scale U
                    d_k = u.col(k).norm();
                    u.col(k) /= (d_k + DIV_OFFSET);

                    // Check early exit
                    if (d_k < tol)
                        break;

                    // Check exit criteria
                    tol_ = (u.col(k) - u_it).array().square().sum(); // Undo scaling for tolerance check, to avoid instability when d(k) is basically 0
                    if (verbose)
                        Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);

                    if (tol_ < tol)
                        break;
                    Rcpp::checkUserInterrupt();
                }

                // "unscale" U
                u.col(k) *= d_k;
                d(k) = d_k;
            }

            // Scale u
            for (int i = 0; i < u.cols(); ++i)
                u.col(i) /= d(i);

            if (tol_ > tol && iter_ == maxit && verbose)
                Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
        }
    };

} // namespace RcppML

#endif