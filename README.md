# RcppML

The Rcpp Machine Learning (RcppML) library R package offers high-performance non-negative matrix factorization (NMF), linear model projection, and non-negative least squares (NNLS).

Key functions:

* `nmf`: Fast sparse matrix factorization by alternating least squares subject to 0/1/2-sided non-negativity constraints with convex L1 regularization
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `nnls`: High-performance solver for non-negative least squares 
* `mse`: Mean squared error of a linear factor model for a sparse matrix

## Getting Started

Install the development version of `RcppML`:

```{R}
library(devtools)
install_github("zdebruine/RcppML")
```

See the package vignette for an introduction to each function listed above.

## Ongoing development

RcppML is under non-breaking active development. Functionality in development includes:

* Specializations for factorization of dense matrices
* Optimizations for symmetric factorization
* Specializations for rank-1 and rank-2 factorizations (faster than _irlba_ SVD counterparts)
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
* Efficient and robust solutions to large matrix factorizations
