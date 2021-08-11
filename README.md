# RcppML: Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)

The RcppML package offers high-performance non-negative matrix factorization (NMF), linear model projection, and non-negative least squares (NNLS):
* `nmf`: Fast sparse matrix factorization by alternating least squares subject with support for non-negativity constraints and L1 regularization
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `nnls`: High-performance solver for non-negative least squares
* `mse`: Mean squared error of a linear factor model for a sparse matrix

See the [package vignette](https://cran.r-project.org/web/packages/RcppML/vignettes/RcppML.html) for a basic introduction to these functions.

## Getting Started

Install RcppML v.0.1.0 from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) ([manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf)) or the development version from GitHub:

```{R}
install.packages("RcppML")
devtools::install_github("zdebruine/RcppML")
```

## Development News

The CRAN version is current with the development version.

New functionality is under active development and will be released soon:
* v0.2.0: Specializations for dense, sparse-symmetric, and dense-symmetric factorization
* v0.3.0: Specializations for rank-1 and rank-2 factorizations (faster than _irlba_ SVD counterparts)

Additional functionality is planned for development:
* Robust solutions to large matrix factorizations
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
