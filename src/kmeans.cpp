// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#include <RcppML.hpp>

// KMEANS CLUSTERING FOR DENSE AND SPARSE MATRICES

inline void calc_centers(RcppML::SparseMatrix& A, const Eigen::VectorXi& clusters, Eigen::MatrixXd& centers) {
    centers.setZero();
    unsigned int k = centers.cols();

    // add up center totals
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);
    for (unsigned int i = 0; i < A.cols(); ++i) {
        counts(clusters(i)) += 1;
        for (RcppML::SparseMatrix::InnerIterator it(A, i); it; ++it)
            centers(it.row(), clusters(i)) += it.value();
    }

    // divide cluster center sums by number of samples in cluster
    for (unsigned int i = 0; i < k; ++i)
        centers.col(i) /= counts(i);
}

inline void calc_centers(const Eigen::MatrixXd& A, const Eigen::VectorXi& clusters, Eigen::MatrixXd& centers) {
    centers.setZero();
    unsigned int k = centers.cols();
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);
    for (unsigned int i = 0; i < A.cols(); ++i) {
        counts(clusters(i)) += 1;
        centers.col(clusters(i)) += A.col(i);
    }
    for (unsigned int i = 0; i < k; ++i)
        centers.col(i) /= counts(i);
}

// function usage: intended simply as an interface to the R "kmeans" function
// if A_dense is an empty matrix with no rows or columns, then A_sp is used.
//[[Rcpp::export]]
Rcpp::List Rcpp_kmeans(const Rcpp::S4& A_sp, const Eigen::MatrixXd& A_dense, const unsigned int k, unsigned int maxit = 100, const double tol = 1e-6, const unsigned int seed = 0, bool verbose = true, const unsigned int threads = 0) {
    Rcpp::Clock clock;
    RcppML::SparseMatrix A_sparse(A_sp);
    bool use_sparse = false;
    if (A_dense.rows() <= 1 || A_dense.cols() <= 1)
        use_sparse = true;

    unsigned int iter = 0, n_reassignments = 1;
    unsigned int n = use_sparse ? A_sparse.cols() : A_dense.cols();
    unsigned int m = use_sparse ? A_sparse.rows() : A_dense.rows();
    double tot_withinss, tol_iter = 1 + tol, totss;
    Eigen::VectorXd ss(n), withinss(k);
    Eigen::VectorXi clusters(n);
    Eigen::MatrixXd centers(m, k);

    if (verbose) Rprintf("%4s %10s %10s\n", "iter", "num moves", "tol");
    // randomly assign samples to cluster centers
    std::vector<double> rand_init = getRandomValues(n, seed);
    for (unsigned int i = 0; i < n; ++i)
        clusters(i) = std::floor(rand_init[i] * k);

    // calculate centers given a random clustering
    use_sparse ? calc_centers(A_sparse, clusters, centers) : calc_centers(A_dense, clusters, centers);

    // while no sample permutations occur and less than maxit, 
    //   relocate each cluster to its best center
    for (; iter < maxit && tol_iter > tol && n_reassignments > 0; ++iter) {

        // calculate Euclidean distance between all samples and all clusters
        Eigen::MatrixXd dists = use_sparse ? distance(A_sparse, centers, "euclidean", threads) : distance(A_dense, centers, "euclidean", threads);

        // update clusterings based on distance matrix
        n_reassignments = 0;
        for (unsigned int i = 0; i < clusters.size(); ++i) {
            ss(i) = dists(i, clusters(i));
            bool is_reassigned = false;
            for (unsigned int j = 0; j < k; ++j) {
                if ((int)j != clusters(i)) {
                    if (dists(i, j) < ss(i)) {
                        clusters(i) = j;
                        ss(i) = dists(i, j);
                        is_reassigned = true;
                    }
                }
            }
            if (is_reassigned) ++n_reassignments;
        }
        ss.array().square();

        // recalculate cluster centers
        use_sparse ? calc_centers(A_sparse, clusters, centers) : calc_centers(A_dense, clusters, centers);

        // update within-cluster sum of squares vector
        withinss.setZero();
        for (unsigned int i = 0; i < n; ++i)
            withinss(clusters(i)) += ss(i);

        double tot_withinss_iter = withinss.array().sum();
        if (iter > 0) {
            tol_iter = std::abs(tot_withinss - tot_withinss_iter) / (2 * (tot_withinss + tot_withinss_iter));
            if (verbose) Rprintf("%4i %10i %10.2e\n", iter, n_reassignments, tol_iter);
            Rcpp::checkUserInterrupt();
        }

        tot_withinss = tot_withinss_iter;
        totss = std::pow(dists.sum(), 2);
    }
    Eigen::VectorXi size = Eigen::VectorXi::Zero(k);
    for (unsigned int i = 0; i < n; ++i)
        size(clusters(i)) += 1;
    clock.stop("timer");
    return Rcpp::List::create(
        Rcpp::Named("clusters") = clusters,
        Rcpp::Named("centers") = centers,
        Rcpp::Named("totss") = totss,
        Rcpp::Named("withinss") = withinss,
        Rcpp::Named("tot_withinss") = tot_withinss,
        Rcpp::Named("betweenss") = totss - tot_withinss,
        Rcpp::Named("size") = size,
        Rcpp::Named("iter") = iter);
}