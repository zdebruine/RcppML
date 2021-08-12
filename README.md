# RcppML: Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)

The RcppML package offers high-performance non-negative matrix factorization (NMF), linear model projection, and non-negative least squares (NNLS):
* `nmf`: Fast sparse matrix factorization by alternating least squares subject with support for non-negativity constraints and L1 regularization
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `nnls`: High-performance solver for non-negative least squares
* `mse`: Mean squared error of a linear factor model for a sparse matrix

See the [package vignette](https://cran.r-project.org/web/packages/RcppML/vignettes/RcppML.html) for a basic introduction to these functions.

## Installation

RcppML is a [CRAN package](https://cran.r-project.org/web/packages/RcppML/index.html) so you can use `install.packages`.

```
install.packages('RcppML')
```

See the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf) for details.

## Development News

To use the development version of the R package, use `devtools`:

```
devtools::install_github('zdebruine/RcppML')
```

Features in the development version (v.0.2.0) not in the CRAN version (v.0.1.0):
- Specialized NMF, projection, and MSE calculation for dense input matrices
- Specialized NMF for symmetric input matrices

Next planned release (late 2021):
* v0.3.0: Specializations for rank-2 factorization (faster than _irlba_ SVD counterparts)

Other planned development:
* Robust solutions to large matrix factorizations
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
