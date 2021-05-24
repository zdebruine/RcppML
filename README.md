# RcppML

High-performance parallelized machine learning toolkit for sparse non-negative matrices. Focus on exceptional speed and non-negative models.

## R package

High-level R functions:
* `solve`: Solve linear systems of equations with regularization and/or non-negativity constraints
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `factorize`: Regularized and/or non-negative orthogonal matrix factorization (e.g. NMF)

## C++ header library

Low-level C++ OOP header library with classes:
* `CoeffMatrix`: Solving linear systems
* `FactorModel`: Matrix factorization models for large sparse inputs
* `dgCMatrix`: Zero-copy access to an R dgCMatrix sparse matrix object using base Rcpp vectors

High-level R functions and low-level C++ OOP header library.
RcppEigen C++ header library and R package for solving regularized and constrained non-negative least squares, projecting linear models, matrix factorization, and divisive/agglomerative clustering.
* The R package contains high-level functions that seamlessly and adaptively call C++ subroutines
* The C++ header library contains low-level templated functional programming interfaces to subroutines written using Eigen

## The C++ header library
* Extends Eigen
* Functional programming interface
* A native Rcpp interface to the `Matrix::dgCMatrix` S4 sparse matrix class is provided for zero-copy pointer-based access to large R sparse matrices in memory.

```
src/ 
-- RcppML.h
-- solve.h          nnls, nnls2, L0, etc. all in one function
-- project.h        project linear models
-- mf.h
-- mf2.h
-- mf1.h
-- mf_rank_update.h
-- bipartition.h
-- cluster.h
R/
-- solve.R
-- project.R
-- factorize.R
-- cluster.R
```
Major focus:  Eigen library for machine learning (EigenML)
RcppML
 - functional interface (except for basic matrix factorization classes)
 - templated
 - functions overloaded for Eigen::SparseMatrix<T> or Rcpp::SparseMatrix<double>

Github bookdown documentation
- R functions
- C++ library

## Classes

An R package for high-performance linear model projection, non-negative least squares, L0-regularized least squares, and more.

## Active development
RcppML is under non-breaking active development. Functionality to be released in the next few months will build off the current library and includes:
* Unconstrained or non-negative diagonalized matrix factorization by alternating least squares with convex L1 regularization
* Efficient and naturally robust solutions to large matrix factorizations
* Extremely fast rank-1 factorization
* Extremely fast exact rank-2 matrix factorizations (faster than _irlba_ rank-2 SVD)
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
