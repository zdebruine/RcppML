# RcppML: Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)

The RcppML package offers several high-performance tools that use matrix factorization. Of note:
* `nmf`: Matrix factorization by alternating least squares. Supports non-negativity constraints and L1 regularization. Specializations for dense/sparse asymmetric/symmetric inputs.
* `dclust`: Divisive clustering by recursive bipartitioning of a sample set. Very fast.
* `nnls`: Fastest-yet algorithm for non-negative least squares.

See the [package vignette](https://cran.r-project.org/web/packages/RcppML/vignettes/RcppML.html) for a basic introduction to these functions.

## Installation

RcppML is a [CRAN package](https://cran.r-project.org/web/packages/RcppML/index.html) so you can use `install.packages`.

```
install.packages('RcppML')
```

See the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf) for details.

Note that the CRAN version is not up-to-date with the development version.

## Development News

To use the development version of the R package, use `devtools`:

```
devtools::install_github('zdebruine/RcppML')
```

Current development version is v.0.3.2:
 - added `dclust`, `bipartition`, and performance updates to other functions.

The version on CRAN only supports `nmf`, `project`, `nnls`, and `mse`.
