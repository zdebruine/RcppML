// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

// this file provides proof-of-concept for several forms of
// vectorized sparse encodings, including CSC and Tabulated compression
//
// each class in this header contains a constructor from an R dgCMatrix (CSC) storage,
// an underdered column-major random access iterator

#ifndef VSE_H
#define VSE_H

#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// Vectorized Sparse Encoding:
//  - (CSC) Compressed Sparse Column format
//  - (P) Polymorphic, index/value pairs may have different types
//  - (NP) Non-polymorphic, index/value pairs must have same unsigned type
//  - (T) Tabulated, values must be discrete and redundant
//
// classes: CSC (non-vectorized), CSC_P, CSC_NP, TCSC_P, TCSC_NP
//

#include "VSE/CSC.h"
#include "VSE/CSC_NP.h"
#include "VSE/CSC_P.h"
#include "VSE/RTCSC_NP.h"
#include "VSE/TCSC_NP.h"
#include "VSE/TCSC_P.h"
#include "VSE/sparse_matrix.h"

#endif