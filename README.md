# RcppML

Rcpp Machine Learning library for high-performance matrix factorization, linear model projection, non-negative least squares, and more.

## R package

R functions in the RcppML package:

* `nnls`: High-performance solver for non-negative least squares 
* `project`: Project linear models given sparse data inputs and one side of the orthogonal factor model
* `nmf`: Fast sparse matrix factorization by alternating least squares subject to 0/1/2-sided non-negativity constraints with convex L1 regularization
* `mse`: Mean squared error of a linear factor model for a sparse matrix

## Getting Started

See the package vignette in the `/vignettes` directory for a basic introduction to each function listed above.

Install the development version of `RcppML`:

```{R}
library(devtools)
install_github("zdebruine/RcppML")
```

## Ongoing development

RcppML is under non-breaking active development. Functionality in development includes:

* Extremely fast rank-1 factorization
* Extremely fast exact rank-2 matrix factorizations (faster than _irlba_ rank-2 SVD)
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
* Efficient and robust solutions to large matrix factorizations
* Tractable and near-exact L0-regularized non-negative least squares
