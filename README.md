# Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

`RcppML` is an R package for fast **non-negative matrix factorization** and **divisive clustering** using **large sparse matrices**. Optimized subroutines for non-negative least squares and spectral bipartitioning are also exposed.

RcppML NMF outperforms other implementations:
1. It is **more interpretable and robust**, due to diagonal scaling.
2. It is the **fastest** NMF implementation of which we are aware.

# Installation

RcppML is a [CRAN package](https://cran.r-project.org/web/packages/RcppML/index.html) so you can use `install.packages`.

```
install.packages('RcppML')
```

See the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf) for details.

The development version is newer than the CRAN version, and is better optimized and contains more features:

```
devtools::install_github("zdebruine/RcppML")
```

When the RcppML R library is loaded, the C++ classes can be directly included in any package using `#include <RcppML.hpp>`. This will also load RcppEigen, Rcpp, and OpenMP headers if needed.

# Features

### Matrix Factorization
Sparse matrix factorization by alternating least squares.
* Non-negativity constraints (optional)
* L1 regularization
* Diagonal scaling for interpretability and robustness
* Rank-1 and Rank-2 specializations (~2x faster than _irlba_ SVD equivalents)

#### R function
The `nmf` function runs matrix factorization by alternating least squares in the form `A = WDH`. The `project` function updates `w` or `h` given the other, while the `mse` function calculates mean squared error of the factor model.

```{R}
A <- Matrix::rsparsematrix(A, 1000, 1000, 0.1) # sparse Matrix::dgCMatrix
model <- RcppML::nmf(A, k = 10, nonneg = TRUE)
h0 <- RcppML::project(A, w = model$w)
RcppML::mse(A, m$w, m$d, m$h)
```

### Divisive Clustering

* `dclust`: Divisive clustering by recursive bipartitioning of a sample set
* `nnls`: Fast active-set/coordinate descent algorithms for solving non-negative least squares problems

See the [package vignette](https://cran.r-project.org/web/packages/RcppML/vignettes/RcppML.html) for a basic introduction to these functions.

All functions are written entirely in Rcpp and RcppEigen.

## Development News

To install the development version of the R package from GitHub, use `devtools`:

```
devtools::install_github('zdebruine/RcppML')
```

Current development version is v.0.3.5:

Note:  The CRAN version (v.0.1.0) only supports `nmf`, `project`, `nnls`, and `mse`, and all functions have been further optimized since initial release.
