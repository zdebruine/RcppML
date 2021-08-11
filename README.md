# RcppML

The Rcpp Machine Learning (RcppML) library R package offers high-performance non-negative matrix factorization (NMF), linear model projection, and non-negative least squares (NNLS).

Key functions:

* `nmf`: Fast sparse matrix factorization by alternating least squares subject to 0/1/2-sided non-negativity constraints with convex L1 regularization
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `nnls`: High-performance solver for non-negative least squares 
* `mse`: Mean squared error of a linear factor model for a sparse matrix

See the package vignette for a basic introduction to these functions.

## Getting Started

Install RcppML v.0.1.0 from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) ([manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf))

```{R}
install.packages("RcppML")
```

<img src = "https://cranlogs.r-pkg.org/badges/grand-total/mltools" />

For the development version, install this repository using `devtools::install_github()`:

```{R}
library(devtools)
install_github("zdebruine/RcppML")
```

## Development News

The CRAN version is current with the development version.

Anticipated functionality under active development includes:

* Dense matrix factorization
* Optimizations for symmetric factorization
* Specializations for rank-1 and rank-2 factorizations (faster than _irlba_ SVD counterparts)
* Efficient and robust solutions to large matrix factorizations
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
* Fast initializations for NMF from unconstrained solutions
