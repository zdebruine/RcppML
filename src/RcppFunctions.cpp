// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#include "../inst/include/RcppEigen_bits.h"
#include "../inst/include/RcppML/SparseMatrix.h"

// #include <RcppML.h>
#include "../inst/include/RcppML/bipartition.hpp"
#include "../inst/include/RcppML/cluster.hpp"
#include "../inst/include/RcppML/distance.hpp"
#include "../inst/include/RcppML/nmf.hpp"
#include "../inst/include/RcppML/svd.hpp"
// PROJECT LINEAR FACTOR MODELS

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, const double L1, const double L2,
                                    const unsigned int threads, const bool mask_zeros, const double upper_bound = 0) {
    Rcpp::SparseMatrix A_(A);
    Rcpp::SparseMatrix mask_(mask);
    RcppML::nmf<Rcpp::SparseMatrix> m(A_, w);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);
    m.threads = threads;
    m.L1[1] = L1;
    m.L2[1] = L2;
    m.upper_bound = upper_bound;
    m.predictH();
    return m.matrixH();
}
//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_predict_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w,
                                   const double L1, const double L2, const unsigned int threads, const bool mask_zeros, const double upper_bound = 0) {
    Rcpp::SparseMatrix mask_(mask);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);
    m.threads = threads;
    m.L1[1] = L1;
    m.L2[1] = L1;
    m.upper_bound = upper_bound;
    m.predictH();
    return m.matrixH();
}

// MEAN SQUARED ERROR LOSS OF FACTORIZATION

//[[Rcpp::export]]
double Rcpp_mse_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                       const unsigned int threads, const bool mask_zeros) {
    Rcpp::SparseMatrix A_(A);
    Rcpp::SparseMatrix mask_(mask);
    RcppML::nmf<Rcpp::SparseMatrix> m(A_, w, d, h);
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
    Rcpp::SparseMatrix mask_(mask);
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
    Rcpp::SparseMatrix A_(A);
    Rcpp::SparseMatrix mask_(mask);
    RcppML::nmf<Rcpp::SparseMatrix> m(A_, w, d, h);
    m.maskMatrix(mask_);
    m.threads = threads;
    return m.mse_masked();
}

//[[Rcpp::export]]
double Rcpp_mse_missing_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h,
                              const unsigned int threads) {
    Rcpp::SparseMatrix mask_(mask);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w, d, h);
    m.maskMatrix(mask_);
    m.threads = threads;
    return m.mse_masked();
}

// NON_NEGATIVE MATRIX FACTORIZATION

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, const double tol, const unsigned int maxit,
                           const bool verbose, const std::vector<double> L1, const std::vector<double> L2,
                           const unsigned int threads, Rcpp::List w_init, const Rcpp::S4& link_matrix_h,
                           const bool mask_zeros, const bool link_h, const bool sort_model, const double upper_bound = 0) {
    Rcpp::SparseMatrix A_(A);
    Rcpp::SparseMatrix mask_(mask), link_matrix_h_(link_matrix_h);
    Eigen::MatrixXd w_ = Rcpp::as<Eigen::MatrixXd>(w_init[0]);
    RcppML::nmf<Rcpp::SparseMatrix> m(A_, w_);

    // set model parameters
    m.tol = tol;
    m.L1 = L1;
    m.L2 = L2;
    m.maxit = maxit;
    m.verbose = verbose;
    m.threads = threads;
    m.sort_model = sort_model;
    m.upper_bound = upper_bound;
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
                          const bool verbose, const std::vector<double> L1, const std::vector<double> L2,
                          const unsigned int threads, Rcpp::List w_init, const Rcpp::S4& link_matrix_h, const bool mask_zeros,
                          const bool link_h, const bool sort_model, const double upper_bound = 0) {
    Rcpp::SparseMatrix mask_(mask), link_matrix_h_(link_matrix_h);
    Eigen::MatrixXd w_ = Rcpp::as<Eigen::MatrixXd>(w_init[0]);
    RcppML::nmf<Eigen::MatrixXd> m(A_, w_);

    // set model parameters
    m.tol = tol;
    m.L1 = L1;
    m.L2 = L2;
    m.maxit = maxit;
    m.verbose = verbose;
    m.threads = threads;
    m.sort_model = sort_model;
    m.upper_bound = upper_bound;
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
    Rcpp::SparseMatrix A_(A);
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
    Rcpp::SparseMatrix A_(A);

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
//' This is a very fast implementation of sequential coordinate descent non-negative least squares (NNLS), suitable for very small or very large systems.
//' The algorithm begins with a zero-filled initialization of \code{x}.
//'
//' Least squares by **sequential coordinate descent** is used to ensure the solution returned is exact. This algorithm was
//' introduced by Franc et al. (2005), and our implementation is a vectorized and optimized rendition of that found in the NNLM R package by Xihui Lin (2020).
//'
//' @param a symmetric positive definite matrix giving coefficients of the linear system
//' @param b matrix giving the right-hand side(s) of the linear system
//' @param L1 L1/LASSO penalty to be subtracted from \code{b}
//' @param L2 Ridge penalty by which to shrink the diagonal of \code{a}
//' @param cd_maxit maximum number of coordinate descent iterations
//' @param cd_tol stopping criteria, difference in \eqn{x} across consecutive solutions over the sum of \eqn{x}
//' @param upper_bound maximum value permitted in solution, set to \code{0} to impose no upper bound
//' @return vector or matrix giving solution for \code{x}
//' @export
//' @author Zach DeBruine
//' @seealso \code{\link{nmf}}, \code{\link{project}}
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
//'
//' # learn nmf model and do bvls projection
//' data(hawaiibirds)
//' w <- nmf(hawaiibirds$counts, 10)@w
//' h <- project(w, hawaiibirds$counts)
//' # now impose upper bound on solutions
//' h2 <- project(w, hawaiibirds$counts, upper_bound = 2)
//' }
//[[Rcpp::export]]
Eigen::MatrixXd nnls(Eigen::MatrixXd a, Eigen::MatrixXd b, unsigned int cd_maxit = 100,
                     const double cd_tol = 1e-8, const double L1 = 0, const double L2 = 0, const double upper_bound = 0) {
    if (a.rows() != a.cols()) Rcpp::stop("'a' is not symmetric");
    if (a.rows() != b.rows()) Rcpp::stop("dimensions of 'b' and 'a' are not compatible!");
    a.diagonal().array() *= (1 - L2);
    b.array() -= L1;
    Eigen::MatrixXd h(b.rows(), b.cols());
    for (size_t sample = 0; sample < b.cols(); ++sample) {
        double tol = 1;
        for (unsigned int it = 0; it < cd_maxit && (tol / b.rows()) > cd_tol; ++it) {
            tol = 0;
            for (unsigned int i = 0; i < h.rows(); ++i) {
                double diff = b(i, sample) / a(i, i);
                if (-diff > h(i, sample)) {
                    if (h(i, sample) != 0) {
                        b.col(sample) -= a.col(i) * -h(i, sample);
                        tol = 1;
                        h(i, sample) = 0;
                    }
                } else if (diff != 0) {
                    if (upper_bound > 0) {
                        if (h(i, sample) + diff > upper_bound) {
                            diff = upper_bound - h(i, sample);
                            h(i, sample) = upper_bound;
                        } else {
                            h(i, sample) += diff;
                        }
                    } else {
                        h(i, sample) += diff;
                    }
                    b.col(sample) -= a.col(i) * diff;
                    tol += std::abs(diff / (h(i, sample) + TINY_NUM));
                }
            }
        }
    }
    return h;
}

//[[Rcpp::export]]
Rcpp::NumericMatrix c_rmatrix(uint32_t nrow, uint32_t ncol, uint32_t rng) {
    Rcpp::NumericMatrix m(nrow, ncol);
    RcppML::rng<false> s(rng);
    for (uint32_t i = 0; i < nrow; ++i)
        for (uint32_t j = 0; j < ncol; ++j)
            m(i, j) = s.runif<float>(i, j);
    return m;
}

//[[Rcpp::export]]
Rcpp::NumericMatrix c_rtimatrix(uint32_t nrow, uint32_t ncol, uint32_t rng) {
    Rcpp::NumericMatrix m(nrow, ncol);
    RcppML::rng<true> s(rng);

    // symmetric part first
    uint32_t n_sym = (nrow < ncol) ? nrow : ncol;

    for (uint32_t i = 0; i < n_sym; ++i) {
        for (uint32_t j = (i + 1); j < n_sym; ++j) {
            float tmp = s.runif<float>(i, j);
            m(i, j) = tmp;
            m(j, i) = tmp;
        }
    }

    // populate the diagonal of the symmetric part
    for (uint32_t i = 0; i < n_sym; ++i)
        m(i, i) = s.runif<float>(i, i);

    // asymmetric part (but still transpose-identical)
    if (nrow > ncol)
        for (uint32_t i = n_sym; i < nrow; ++i)
            for (uint32_t j = 0; j < ncol; ++j)
                m(i, j) = s.runif<float>(i, j);
    else if (ncol > nrow)
        for (uint32_t i = 0; i < nrow; ++i)
            for (uint32_t j = n_sym; j < ncol; ++j)
                m(i, j) = s.runif<float>(i, j);

    return m;
}

//[[Rcpp::export]]
Rcpp::NumericVector c_runif(const uint32_t n, const float min, const float max, const uint32_t rng, const uint32_t rng2) {
    Rcpp::NumericVector result(n);
    RcppML::rng<false> s(rng);
    float scale = max - min;
    for (uint32_t i = 0; i < n; ++i) {
        result[i] = s.runif<float>(i, rng2) * scale + min;
    }
    return result;
}

//[[Rcpp::export]]
Rcpp::IntegerVector c_rbinom(const uint32_t n, uint32_t size, const uint32_t inv_probability, const uint32_t rng, const uint32_t rng2) {
    Rcpp::IntegerVector result(n);
    RcppML::rng<false> s(rng);
    for (; size > 0; --size) {
        for (uint32_t i = 0; i < n; ++i) {
            if (s.sample(i, rng2, inv_probability) == 0)
                ++result[i];
        }
    }
    return result;
}

//[[Rcpp::export]]
std::vector<uint32_t> c_sample(const uint32_t n, const uint32_t size, const bool replace, const uint32_t rng, const uint32_t rng2) {
    RcppML::rng<false> s(rng);
    if (size == 1) {
        return std::vector<uint32_t>(1, s.sample(1, 1, n));
    }
    if (replace) {
        std::vector<uint32_t> result(size);
        for (uint32_t i = 0; i < size; ++i) {
            result[i] = s.sample(i, rng2, n);
        }
        return result;
    } else {
        if (size > n)
            Rcpp::stop("cannot take a sample larger than the population when 'replace = FALSE'");
        std::vector<uint32_t> result(n);
        std::iota(result.begin(), result.end(), 0);
        for (uint32_t i = 0; i < size; ++i) {
            std::swap(result[i], result[s.sample(i, rng2, n)]);
        }
        if (size < n)
            result.resize(size);
        return result;
    }
}

//[[Rcpp::export]]
Rcpp::S4 c_rtisparsematrix(const uint32_t nrow, const uint32_t ncol, const uint32_t inv_probability, const bool pattern_only, uint32_t rng) {
    RcppML::rng<true> s(rng);
    Rcpp::S4 result = pattern_only ? Rcpp::S4(std::string("ngCMatrix")) : Rcpp::S4(std::string("dgCMatrix"));
    Rcpp::IntegerVector p(ncol + 1);
    std::vector<uint32_t> i;
    i.reserve(nrow * ncol / inv_probability);
    if (pattern_only) {
        for (uint32_t col = 0; col < ncol; ++col) {
            for (uint32_t row = 0; row < nrow; ++row) {
                if (s.sample(row, col, inv_probability) == 0)
                    i.push_back(row);
            }
            p[col + 1] = i.size();
        }
    } else {
        std::vector<float> x;
        x.reserve(nrow * ncol / inv_probability);
        for (uint32_t col = 0; col < ncol; ++col) {
            for (uint32_t row = 0; row < nrow; ++row) {
                if (s.sample(row, col, inv_probability) == 0) {
                    i.push_back(row);
                    x.push_back(s.runif<float>(row, col));
                }
            }
            p[col + 1] = i.size();
        }
        Rcpp::NumericVector x_ = Rcpp::wrap(x);
        result.slot("x") = x_;
    }
    Rcpp::IntegerVector i_ = Rcpp::wrap(i);
    result.slot("Dim") = Rcpp::IntegerVector::create(nrow, ncol);
    result.slot("i") = i_;
    result.slot("p") = p;
    return result;
}

//[[Rcpp::export]]
Rcpp::S4 c_rsparsematrix(const uint32_t nrow, const uint32_t ncol, const uint32_t inv_probability, const bool pattern_only, uint32_t rng) {
    RcppML::rng<false> s(rng);
    Rcpp::S4 result = pattern_only ? Rcpp::S4(std::string("ngCMatrix")) : Rcpp::S4(std::string("dgCMatrix"));
    Rcpp::IntegerVector p(ncol + 1);
    std::vector<uint32_t> i;
    i.reserve(nrow * ncol / inv_probability);
    if (pattern_only) {
        for (uint32_t col = 0; col < ncol; ++col) {
            for (uint32_t row = 0; row < nrow; ++row) {
                if (s.sample(row, col, inv_probability) == 0)
                    i.push_back(row);
            }
            p[col + 1] = i.size();
        }
    } else {
        std::vector<float> x;
        x.reserve(nrow * ncol / inv_probability);
        for (uint32_t col = 0; col < ncol; ++col) {
            for (uint32_t row = 0; row < nrow; ++row) {
                if (s.sample(row, col, inv_probability) == 0) {
                    i.push_back(row);
                    x.push_back(s.runif<float>(row, col));
                }
            }
            p[col + 1] = i.size();
        }
        Rcpp::NumericVector x_ = Rcpp::wrap(x);
        result.slot("x") = x_;
    }
    Rcpp::IntegerVector i_ = Rcpp::wrap(i);
    result.slot("Dim") = Rcpp::IntegerVector::create(nrow, ncol);
    result.slot("i") = i_;
    result.slot("p") = p;
    return result;
}

//[[Rcpp::export]]
Rcpp::List Rcpp_svd_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, const double tol, const unsigned int maxit,
                          const bool verbose, const std::vector<double> L1, const std::vector<double> L2,
                          const unsigned int threads, Rcpp::List u_init, const Rcpp::S4& link_matrix_v, const bool mask_zeros,
                          const bool link_v, const double upper_bound = 0) {
    Rcpp::SparseMatrix mask_(mask), link_matrix_v_(link_matrix_v);
    Eigen::MatrixXd u_ = Rcpp::as<Eigen::MatrixXd>(u_init[0]);
    RcppML::svd<Eigen::MatrixXd> m(A_, u_);

    // set model parameters
    m.tol = tol;
    m.L1 = L1;
    m.L2 = L2;
    m.maxit = maxit;
    m.verbose = verbose;
    m.threads = threads;
    m.upper_bound = upper_bound;
    if (link_v) m.linkV(link_matrix_v_);
    if (mask_zeros)
        m.maskZeros();
    else if (mask_.rows() == A_.rows() && mask_.cols() == A_.cols())
        m.maskMatrix(mask_);

    if (u_init.length() == 1)
        m.fit();
    else
        m.fit_restarts(u_init);

    return Rcpp::List::create(Rcpp::Named("u") = m.matrixU(),
                              Rcpp::Named("v") = m.matrixV(),
                              Rcpp::Named("tol") = m.fit_tol(),
                              Rcpp::Named("iter") = m.fit_iter(),
                              Rcpp::Named("mse") = m.fit_mse(),
                              Rcpp::Named("best_model") = m.best_model());
}