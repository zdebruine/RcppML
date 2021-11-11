// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#include <RcppML.hpp>

// PROJECT LINEAR FACTOR MODELS

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, const bool nonneg,
                                    const double L1, const double L2,
                                    const unsigned int threads, const bool mask_zeros) {
    RcppML::SparseMatrix A_(A);
    RcppML::SparsePatternMatrix mask_(mask);
    RcppML::nmf<RcppML::SparseMatrix> m(A_, w);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);
    m.threads = threads;
    m.L1[1] = L1;
    m.L2[1] = L2;
    m.nonneg = nonneg;
    m.predictH();
    return m.matrixH();
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, const bool nonneg,
                                   const double L1, const double L2, const unsigned int threads, const bool mask_zeros) {
    RcppML::SparsePatternMatrix mask_(mask);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);
    m.threads = threads;
    m.L1[1] = L1;
    m.L2[1] = L1;
    m.nonneg = nonneg;
    m.predictH();
    return m.matrixH();
}

// MEAN SQUARED ERROR LOSS OF FACTORIZATION

//[[Rcpp::export]]
double Rcpp_mse_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                       const unsigned int threads, const bool mask_zeros) {
    RcppML::SparseMatrix A_(A);
    RcppML::SparsePatternMatrix mask_(mask);
    RcppML::nmf<RcppML::SparseMatrix> m(A_, w, d, h);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);
    m.threads = threads;
    return m.mse();
}

//[[Rcpp::export]]
double Rcpp_mse_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                      const unsigned int threads, const bool mask_zeros) {
    RcppML::SparsePatternMatrix mask_(mask);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w, d, h);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);
    m.threads = threads;
    return m.mse();
}

//[[Rcpp::export]]
double Rcpp_mse_missing_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                               const unsigned int threads) {
    RcppML::SparseMatrix A_(A);
    RcppML::SparsePatternMatrix mask_(mask);
    RcppML::nmf<RcppML::SparseMatrix> m(A_, w, d, h);
    m.maskMatrix(mask_);
    m.threads = threads;
    return m.mse_masked();
}

//[[Rcpp::export]]
double Rcpp_mse_missing_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                              const unsigned int threads) {
    RcppML::SparsePatternMatrix mask_(mask);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w, d, h);
    m.maskMatrix(mask_);
    m.threads = threads;
    return m.mse_masked();
}

// NON_NEGATIVE MATRIX FACTORIZATION

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, const double tol, const unsigned int maxit,
                           const bool verbose, const bool nonneg, const std::vector<double> L1, const std::vector<double> L2,
                           const bool diag, const unsigned int threads, Rcpp::List w_init, const Rcpp::S4& link_matrix_h,
                           const bool mask_zeros, const bool link_h, const bool sort_model) {
    RcppML::SparseMatrix A_(A);
    RcppML::SparsePatternMatrix mask_(mask), link_matrix_h_(link_matrix_h);
    Eigen::MatrixXd w_ = Rcpp::as<Eigen::MatrixXd>(w_init[0]);
    RcppML::nmf<RcppML::SparseMatrix> m(A_, w_);

    // set model parameters
    m.tol = tol;
    m.nonneg = nonneg;
    m.L1 = L1;
    m.L2 = L2;
    m.maxit = maxit;
    m.diag = diag;
    m.verbose = verbose;
    m.threads = threads;
    m.sort_model = sort_model;
    if (link_h) m.linkH(link_matrix_h_);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);

    if (w_init.length() == 1)
        m.fit();
    else
        m.fit_restarts(w_init);

    return Rcpp::List::create(Rcpp::Named("w") = m.matrixW().transpose(),
                              Rcpp::Named("d") = m.vectorD(),
                              Rcpp::Named("h") = m.matrixH(),
                              Rcpp::Named("tol") = m.fit_tol(),
                              Rcpp::Named("iter") = m.fit_iter(),
                              Rcpp::Named("mse") = m.fit_mse(),
                              Rcpp::Named("best_model") = m.best_model());
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, const double tol, const unsigned int maxit,
                          const bool verbose, const bool nonneg, const std::vector<double> L1, const std::vector<double> L2,
                          const bool diag, const unsigned int threads, Rcpp::List w_init, const Rcpp::S4& link_matrix_h, const bool mask_zeros,
                          const bool link_h, const bool sort_model) {
    RcppML::SparsePatternMatrix mask_(mask), link_matrix_h_(link_matrix_h);
    Eigen::MatrixXd w_ = Rcpp::as<Eigen::MatrixXd>(w_init[0]);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w_);

    // set model parameters
    m.tol = tol;
    m.nonneg = nonneg;
    m.L1 = L1;
    m.L2 = L2;
    m.maxit = maxit;
    m.diag = diag;
    m.verbose = verbose;
    m.threads = threads;
    m.sort_model = sort_model;
    if (link_h) m.linkH(link_matrix_h_);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);

    if (w_init.length() == 1)
        m.fit();
    else
        m.fit_restarts(w_init);

    return Rcpp::List::create(Rcpp::Named("w") = m.matrixW().transpose(),
                              Rcpp::Named("d") = m.vectorD(),
                              Rcpp::Named("h") = m.matrixH(),
                              Rcpp::Named("tol") = m.fit_tol(),
                              Rcpp::Named("iter") = m.fit_iter(),
                              Rcpp::Named("mse") = m.fit_mse(),
                              Rcpp::Named("best_model") = m.best_model());
}

// BIPARTITION A SAMPLE SET BY RANK-2 NMF

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_sparse(const Rcpp::S4& A, const double tol, const unsigned int maxit, const bool nonneg,
                                   const std::vector<unsigned int>& samples, const unsigned int seed, const bool verbose = false,
                                   const bool calc_dist = false, const bool diag = true) {
    RcppML::SparseMatrix A_(A);
    Eigen::MatrixXd w = randomMatrix(2, A_.rows(), seed);
    bipartitionModel m = c_bipartition_sparse(A_, w, samples, tol, nonneg, calc_dist, maxit, verbose);
    return Rcpp::List::create(Rcpp::Named("v") = m.v, Rcpp::Named("dist") = m.dist, Rcpp::Named("size1") = m.size1,
                              Rcpp::Named("size2") = m.size2, Rcpp::Named("samples1") = m.samples1, Rcpp::Named("samples2") = m.samples2,
                              Rcpp::Named("center1") = m.center1, Rcpp::Named("center2") = m.center2);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_dense(const Eigen::MatrixXd& A, const double tol, const unsigned int maxit, const bool nonneg,
                                  const std::vector<unsigned int>& samples, const unsigned int seed, const bool verbose = false,
                                  const bool calc_dist = false, const bool diag = true) {
    Eigen::MatrixXd w = randomMatrix(2, A.rows(), seed);
    bipartitionModel m = c_bipartition_dense(A, w, samples, tol, nonneg, calc_dist, maxit, verbose);
    return Rcpp::List::create(Rcpp::Named("v") = m.v, Rcpp::Named("dist") = m.dist, Rcpp::Named("size1") = m.size1,
                              Rcpp::Named("size2") = m.size2, Rcpp::Named("samples1") = m.samples1, Rcpp::Named("samples2") = m.samples2,
                              Rcpp::Named("center1") = m.center1, Rcpp::Named("center2") = m.center2);
}

// DIVISIVE CLUSTERING BY RECURSIVE BIPARTITIONING

//[[Rcpp::export]]
Rcpp::List Rcpp_dclust_sparse(const Rcpp::S4& A, const unsigned int min_samples, const double min_dist, const bool verbose,
                              const double tol, const unsigned int maxit, const bool nonneg, const unsigned int seed, const unsigned int threads) {
    RcppML::SparseMatrix A_(A);

    RcppML::clusterModel m = RcppML::clusterModel(A_, min_samples, min_dist);
    m.nonneg = nonneg;
    m.verbose = verbose;
    m.tol = tol;
    m.min_dist = min_dist;
    m.seed = seed;
    m.maxit = maxit;
    m.threads = threads;
    m.min_samples = min_samples;

    m.dclust();

    std::vector<cluster> clusters = m.getClusters();

    Rcpp::List result(clusters.size());
    for (unsigned int i = 0; i < clusters.size(); ++i) {
        result[i] = Rcpp::List::create(Rcpp::Named("id") = clusters[i].id, Rcpp::Named("samples") = clusters[i].samples,
                                       Rcpp::Named("center") = clusters[i].center, Rcpp::Named("dist") = clusters[i].dist,
                                       Rcpp::Named("leaf") = clusters[i].leaf);
    }
    return result;
}

//' @title Non-negative least squares
//'
//' @description Solves the equation \code{a %*% x = b} for \code{x} subject to \eqn{x > 0}.
//'
//' @details
//' This is a very fast implementation of non-negative least squares (NNLS), suitable for very small or very large systems.
//'
//' **Algorithm**. Sequential coordinate descent (CD) is at the core of this implementation, and requires an initialization of \eqn{x}. There are two supported methods for initialization of \eqn{x}:
//' 1. **Zero-filled initialization** when \code{fast_nnls = FALSE} and \code{cd_maxit > 0}. This is generally very efficient for well-conditioned and small systems.
//' 2. **Approximation with FAST** when \code{fast_nnls = TRUE}. Forward active set tuning (FAST), described below, finds an approximate active set using unconstrained least squares solutions found by Cholesky decomposition and substitution. To use only FAST approximation, set \code{cd_maxit = 0}.
//'
//' \code{a} must be symmetric positive definite if FAST NNLS is used, but this is not checked.
//'
//' See our BioRXiv manuscript (references) for benchmarking against Lawson-Hanson NNLS and for a more technical introduction to these methods.
//'
//' **Coordinate Descent NNLS**. Least squares by **sequential coordinate descent** is used to ensure the solution returned is exact. This algorithm was
//' introduced by Franc et al. (2005), and our implementation is a vectorized and optimized rendition of that found in the NNLM R package by Xihui Lin (2020).
//'
//' **FAST NNLS.** Forward active set tuning (FAST) is an exact or near-exact NNLS approximation initialized by an unconstrained
//' least squares solution. Negative values in this unconstrained solution are set to zero (the "active set"), and all
//' other values are added  to a "feasible set". An unconstrained least squares solution is then solved for the
//' "feasible set", any negative values in the resulting solution are set to zero, and the process is repeated until
//' the feasible set solution is strictly positive.
//'
//' The FAST algorithm has a definite convergence guarantee because the
//' feasible set will either converge or become smaller with each iteration. The result is generally exact or nearly
//' exact for small well-conditioned systems (< 50 variables) within 2 iterations and thus sets up coordinate
//' descent for very rapid convergence. The FAST method is similar to the first phase of the so-called "TNT-NN" algorithm (Myre et al., 2017),
//' but the latter half of that method relies heavily on heuristics to refine the approximate active set, which we avoid by using
//' coordinate descent instead.
//'
//' @param a symmetric positive definite matrix giving coefficients of the linear system
//' @param b matrix giving the right-hand side(s) of the linear system
//' @param L1 L1/LASSO penalty to be subtracted from \code{b}
//' @param L2 Ridge penalty to be added to diagonal of \code{a}
//' @param PE Pattern Extraction (angular) penalty to be added to off-diagonal values of \code{a}
//' @param fast_nnls initialize coordinate descent with a FAST NNLS approximation
//' @param cd_maxit maximum number of coordinate descent iterations
//' @param cd_tol stopping criteria, difference in \eqn{x} across consecutive solutions over the sum of \eqn{x}
//' @return vector or matrix giving solution for \code{x}
//' @export
//' @author Zach DeBruine
//' @seealso \code{\link{nmf}}, \code{\link{predict.nmf}}
//' @md
//'
//' @references
//'
//' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
//'
//' Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem. Proc. Int'l Conf. Computer Analysis of Images and Patterns."
//'
//' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
//'
//' Myre, JM, Frahm, E, Lilja DJ, and Saar, MO. (2017) "TNT-NN: A Fast Active Set Method for Solving Large Non-Negative Least Squares Problems". Proc. Computer Science.
//'
//' @examples
//' \dontrun{
//' # compare solution to base::solve for a random system
//' X <- matrix(runif(100), 10, 10)
//' a <- crossprod(X)
//' b <- crossprod(X, runif(10))
//' unconstrained_soln <- solve(a, b)
//' nonneg_soln <- nnls(a, b)
//' unconstrained_err <- mean((a %*% unconstrained_soln - b)^2)
//' nonnegative_err <- mean((a %*% nonneg_soln - b)^2)
//' unconstrained_err
//' nonnegative_err
//' all.equal(solve(a, b), nnls(a, b))
//'
//' # example adapted from multiway::fnnls example 1
//' X <- matrix(1:100,50,2)
//' y <- matrix(101:150,50,1)
//' beta <- solve(crossprod(X)) %*% crossprod(X, y)
//' beta
//' beta <- nnls(crossprod(X), crossprod(X, y))
//' beta
//' }
//[[Rcpp::export]]
Eigen::MatrixXd nnls(Eigen::MatrixXd a, Eigen::MatrixXd b, unsigned int cd_maxit = 100,
                     const double cd_tol = 1e-8, const bool fast_nnls = false, const double L1 = 0, const double L2 = 0, const double PE = 0) {
    if (a.rows() != a.cols()) Rcpp::stop("'a' is not symmetric");
    if (a.rows() != b.rows()) Rcpp::stop("dimensions of 'b' and 'a' are not compatible!");
    if (L1 != 0) b.array() -= L1;
    if (L2 != 0) a.diagonal().array() += L2;
    if (PE != 0) {
        a.array() += PE;
        a.diagonal().array() -= PE;
    }

    Eigen::LLT<Eigen::MatrixXd> a_llt;
    Eigen::MatrixXd x(b.rows(), b.cols());
    if (fast_nnls) a_llt = a.llt();
    for (unsigned int col = 0; col < b.cols(); ++col) {
        if (fast_nnls) {
            // initialize with unconstrained least squares solution
            x.col(col) = a_llt.solve(b.col(col));
            // iterative feasible set reduction while unconstrained least squares solutions at feasible indices contain negative values
            while ((x.col(col).array() < 0).any()) {
                Eigen::VectorXi gtz_ind = find_gtz(x, col);      // get indices in "x" greater than zero (the "feasible set")
                Eigen::VectorXd bsub = subvec(b, gtz_ind, col);  // subset "a" and "b" to those indices in the feasible set
                Eigen::MatrixXd asub = submat(a, gtz_ind, gtz_ind);
                Eigen::VectorXd xsub = asub.llt().solve(bsub);  // solve for those indices in "x"
                x.setZero();
                for (unsigned int i = 0; i < gtz_ind.size(); ++i) x(gtz_ind(i), col) = xsub(i);
            }
            b.col(col) -= a * x.col(col);  // adjust gradient for current solution
        }

        // refine FAST solution by coordinate descent, or find solution from zero-initialized "x" matrix
        if (cd_maxit > 0) {
            double tol = 1;
            for (unsigned int it = 0; it < cd_maxit && (tol / b.rows()) > cd_tol; ++it) {
                tol = 0;
                for (unsigned int i = 0; i < b.rows(); ++i) {
                    double diff = b(i, col) / a(i, i);
                    if (-diff > x(i, col)) {
                        if (x(i, col) != 0) {
                            b.col(col) -= a.col(i) * -x(i, col);
                            tol = 1;
                            x(i, col) = 0;
                        }
                    } else if (diff != 0) {
                        x(i, col) += diff;
                        b.col(col) -= a.col(i) * diff;
                        tol += std::abs(diff / (x(i, col) + TINY_NUM));
                    }
                }
            }
        }
    }
    return x;
}