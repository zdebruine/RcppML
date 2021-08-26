# RcppML: Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)

RcppML offers extremely fast **non-negative matrix factorization** and **divisive clustering**, thanks to highly optimized non-negative least squares solvers and a rank-2 NMF bipartitioning algorithm. 

Diagonal scaling in RcppML NMF improves robustness, interpretability, and enables convex L1 regularization.

## Main R functions:

* `nmf`: sparse matrix factorization by alternating least squares. Optional non-negativity constraints and L1 regularization. Special optimizations for symmetric, rank-1, and rank-2 decompositions
* `dclust`: Divisive clustering by recursive bipartitioning of a sample set
* `nnls`: Fast active-set/coordinate descent algorithms for solving non-negative least squares problems

See the [package vignette](https://cran.r-project.org/web/packages/RcppML/vignettes/RcppML.html) for a basic introduction to these functions.

All functions are written entirely in Rcpp and RcppEigen.

## C++ API:

An object-oriented API provides object-oriented access to NMF and clustering classes in C++:

## Installation

RcppML is a [CRAN package](https://cran.r-project.org/web/packages/RcppML/index.html) so you can use `install.packages`.

```
install.packages('RcppML')
```

See the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf) for details.

Once loaded, the RcppML C++ header library in the `inst/include` folder can be included in other packages.

## Development News

To install the development version of the R package from GitHub, use `devtools`:

```
devtools::install_github('zdebruine/RcppML')
```

Current development version is v.0.3.5:

Note:  The CRAN version (v.0.1.0) only supports `nmf`, `project`, `nnls`, and `mse`, and all functions have been further optimized since initial release.
