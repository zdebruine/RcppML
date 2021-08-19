// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_common
#define RcppML_common

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#ifndef EIGEN_INITIALIZE_MATRICES_BY_ZERO
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#endif

#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

#ifndef RcppML_dgcmatrix
#include <RcppML/dgcmatrix.hpp>
#endif

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

#ifndef RcppML_bits
#include <RcppML/bits.hpp>
#endif

#endif