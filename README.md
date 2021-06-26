# RcppML

High-performance machine learning methods for large sparse matrices. Focus on non-negative models and speed in parallelized computing systems.

## R package

High-level R functions in the RcppML package:

* `solve`: Solve linear systems of equations with regularization and/or non-negativity constraints
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `factorize`: Regularized and/or non-negative orthogonal matrix factorization (e.g. NMF)

## RcppEigen header library

Low-level C++ OOP header library written with Rcpp and Eigen. Contains classes:

* `CoeffMatrix`: Solving linear systems
* `MatrixFactorization`: Matrix factorization models for large sparse inputs, constructed from `dgCMatrix`.
* `CoeffMatrix2` and `MatrixFactorization2` specializations for 2-variable or rank-2 cases.

Classes, public methods, and basic Rcpp usage are documented in the RcppML bookdown.

## Vignettes and Documentation

All package documentation is provided in a bookdown with chapters on:
* Solving systems of equations
* Projecting linear models
* Matrix factorization
* Rank-2 matrix factorization

## Ongoing development
RcppML is under non-breaking active development. Functionality in development includes:
* Unconstrained or non-negative diagonalized matrix factorization by alternating least squares with convex L1 regularization
* Extremely fast rank-1 factorization
* Extremely fast exact rank-2 matrix factorizations (faster than _irlba_ rank-2 SVD)
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
* Efficient and naturally robust solutions to large matrix factorizations
