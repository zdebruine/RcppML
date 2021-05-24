// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_H
#define RCPPML_H

#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#pragma GCC diagnostic ignored "-Wignored-attributes"

#ifdef(ENABLE_OPENMP)
#include <omp.h>
#else

#include <RcppEigen.h>

#endif // RCPPML_H