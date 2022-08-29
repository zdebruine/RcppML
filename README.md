# Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

`RcppML` is an R package for fast **non-negative matrix factorization** and **divisive clustering** using **large sparse matrices**. 

See `pkgdown` site here: [https://zdebruine.github.io/RcppML/](https://zdebruine.github.io/RcppML/)

RcppML NMF is:
 * The **fastest** NMF implementation in any language for sparse and dense matrices
 * More **interpretable** than other implementations due to diagonal scaling
 * Easy to **regularize** with an L1 penalty

## Installation

Install from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) or the development version from GitHub:

```
install.packages('RcppML')                       # install CRAN version
devtools::install_github("zdebruine/RcppML")     # compile dev version
```

NOTE: RcppML is being actively developed. Please check that your `packageVersion("RcppML")` is current before raising issues.

Check out the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf).

Once installed and loaded, RcppML C++ headers defining classes can be used in C++ files for any R package using `#include <RcppML.hpp>`. 

## Matrix Factorization
Sparse matrix factorization by alternating least squares:
* Non-negativity constraints
* L1 regularization
* Diagonal scaling
* Rank-1 and Rank-2 specializations (~2x faster than _irlba_ SVD equivalents)

Read (and cite) our [bioRXiv manuscript](https://www.biorxiv.org/content/10.1101/2021.09.01.458620v1) on NMF for single-cell experiments.

#### R functions
The `nmf` function runs matrix factorization by alternating least squares in the form `A = WDH`. The `project` function updates `w` or `h` given the other, while the `mse` function calculates mean squared error of the factor model.

```{R}
library(RcppML)
A <- Matrix::rsparsematrix(1000, 100, 0.1) # sparse Matrix::dgCMatrix
model <- RcppML::nmf(A, k = 10)
h0 <- predict(model, A)
evaluate(model, A) # calculate mean squared error
```

## Divisive Clustering
Divisive clustering by rank-2 spectral bipartitioning.
* 2nd SVD vector is linearly related to the difference between factors in rank-2 matrix factorization.
* Rank-2 matrix factorization (optional non-negativity constraints) for spectral bipartitioning **~2x faster** than _irlba_ SVD
* Sensitive distance-based stopping criteria similar to Newman-Girvan modularity, but orders of magnitude faster
* Stopping criteria based on minimum number of samples

#### R functions
The `dclust` function runs divisive clustering by recursive spectral bipartitioning, while the `bipartition` function exposes the rank-2 NMF specialization and returns statistics of the bipartition.

```{R}
library(RcppML)
A <- Matrix::rsparsematrix(1000, 1000, 0.1) # sparse Matrix::dgcMatrix
clusters <- dclust(A, min_dist = 0.001, min_samples = 5)
cluster0 <- bipartition(A)
```
