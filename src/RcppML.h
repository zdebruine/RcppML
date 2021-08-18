// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#define _RCPPML

#ifdef _OPENMP
#include <omp.h>
#endif

// Rcpp sparse matrix class
#ifndef RCPP_DGCMATRIX_H
#include "dgCMatrix.h"
#endif

#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// STRUCTURES
struct wdhmodel {
    Eigen::MatrixXd w;
    Eigen::VectorXd d;
    Eigen::MatrixXd h;
    double tol;
    unsigned int it;
};

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

struct clusterModel { 
    std::string id;
    std::vector<unsigned int> samples;
    std::vector<double> center;
    double dist;
    bool leaf;
    bool agg;
};

/*
#include "helpers.cpp"
#include "nnls.cpp"
#include "project.cpp"
#include "mse.cpp"
#include "nmf.cpp"
#include "nmf2.cpp"
#include "bipartition.cpp"
#include "dclust.cpp"
*/

// FUNCTION FORWARD DECLARATIONS
// only functions that need to be called from separate files are included here

// helpers.cpp
// helper functions and subroutines
std::vector<int> sort_index(const Eigen::VectorXd& d);

Eigen::MatrixXd reorder_rows(const Eigen::MatrixXd& x, const std::vector<int>& ind);

Eigen::VectorXd reorder(const Eigen::VectorXd& x, const std::vector<int>& ind);

double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y);

std::vector<double> getRandomValues(const unsigned int len, const unsigned int seed);

Eigen::MatrixXd randomMatrix(const unsigned int nrow, const unsigned int ncol, const std::vector<double>& random_values);

// nnls.cpp
// generalized non-negative least squares
Eigen::VectorXd c_nnls(const Eigen::MatrixXd& a, const typename Eigen::VectorXd& b,
    const Eigen::LLT<Eigen::MatrixXd, 1>& a_llt, const unsigned int fast_maxit, const unsigned int cd_maxit,
    const double cd_tol, const bool nonneg);

// project.cpp
// projecting linear factor models
Eigen::MatrixXd c_project_sparse(Rcpp::dgCMatrix& A, Eigen::MatrixXd& w, const bool nonneg, const unsigned int fast_maxit,
    const unsigned int cd_maxit, const double cd_tol, const double L1, const unsigned int threads);

Eigen::MatrixXd Rcpp_project_dense(const Rcpp::NumericMatrix& A, Eigen::MatrixXd& w, const bool nonneg,
    const unsigned int fast_maxit, const unsigned int cd_maxit, const double cd_tol, const double L1,
    const unsigned int threads);

// nmf.cpp
// non-negative matrix factorization
wdhmodel c_nmf_sparse(Rcpp::dgCMatrix& A_S4, Rcpp::dgCMatrix& At_S4, const bool symmetric, Eigen::MatrixXd& w, 
    const double tol, const bool nonneg, const double L1_w, const double L1_h, const unsigned int maxit,
    const bool diag, const unsigned int fast_maxit, const unsigned int cd_maxit, const double cd_tol,
    const bool verbose, const unsigned int threads);
  
wdhmodel c_nmf_dense(const Rcpp::NumericMatrix& A, const Rcpp::NumericMatrix& At, const bool symmetric, Eigen::MatrixXd& w, 
    const double tol, const bool nonneg, const double L1_w, const double L1_h, const unsigned int maxit,
    const bool diag, const unsigned int fast_maxit, const unsigned int cd_maxit, const double cd_tol,
    const bool verbose, const unsigned int threads);

// nmf2.cpp
// rank-2 matrix factorization
wdhmodel c_nmf2_sparse(Rcpp::dgCMatrix& A, Eigen::MatrixXd& h, const double tol, const bool nonneg, const unsigned int maxit,
    const bool verbose, const bool diag, const std::vector<unsigned int> samples);

wdhmodel c_nmf2_dense(const Rcpp::NumericMatrix& A, Eigen::MatrixXd& h, const double tol, const bool nonneg,
    const unsigned int maxit, const bool verbose, const bool diag, const std::vector<unsigned int> samples);

// bipartition.cpp
// bipartitioning by rank-2 factorization and accessory statistical functions
std::vector<double> centroid(Rcpp::dgCMatrix& A, const std::vector<unsigned int>& samples);

std::vector<double> centroid(const Rcpp::NumericMatrix& A, const std::vector<unsigned int>& samples);

double rel_cosine(Rcpp::dgCMatrix& A, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
    const std::vector<double>& center1, const std::vector<double>& center2);

double rel_cosine(const Rcpp::NumericMatrix& A, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
    const std::vector<double>& center1, const std::vector<double>& center2);

bipartitionModel c_bipartition_sparse(Rcpp::dgCMatrix& A, const double tol, const bool nonneg,
    const std::vector<unsigned int> samples, bool calc_centers, bool calc_dist, const unsigned int maxit, const bool verbose,
    const bool diag, const std::vector<double>& random_values);

bipartitionModel c_bipartition_dense(const Rcpp::NumericMatrix& A, const double tol, const bool nonneg,
    const std::vector<unsigned int> samples, bool calc_centers, bool calc_dist, const unsigned int maxit, const bool verbose,
    const bool diag, const std::vector<double>& random_values);