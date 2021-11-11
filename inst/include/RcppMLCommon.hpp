// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_common
#define RcppML_common

//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#ifndef TINY_NUM
#define TINY_NUM 1e-15 // epsilon for numerical stability
#endif

// parameters for coordinate descent
#ifndef CD_PARAMS
#define CD_PARAMS
#define CD_TOL 1e-8
#define CD_MAXIT 100
#endif

#ifndef EIGEN_INITIALIZE_MATRICES_BY_ZERO
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#endif

//[[Rcpp::depends(RcppSparse)]]
#ifndef RCPPSPARSE_H
#include <RcppSparse.h>
#endif

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

#ifndef RcppML_bits
#include <RcppML/bits.hpp>
#endif

#endif