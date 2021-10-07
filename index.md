# About RcppML

The Rcpp Machine Learning (RcppML) library is an R/C++ package for fast non-negative matrix factorization, divisive clustering, and related subroutines.

RcppML NMF is:
1. **more interpretable and robust** due to diagonal scaling
2. The **fastest** NMF implementation available in R

## Installation

Install from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) or the development version from GitHub:

```
install.packages('RcppML')                   # install CRAN version
devtools::install_github("zdebruine/RcppML") # compile dev version
```

Because RcppML is being actively developed, please check that your `packageVersion("RcppML")` is current with the version on GitHub before raising issues.

Check out the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf).

## Matrix Factorization
Matrix factorization by alternating least squares:
* Non-negativity constraints
* L1 regularization
* Diagonal scaling
* Rank-1 and Rank-2 specializations (~2x faster than _irlba_ SVD equivalents)
* Dense and sparse backend specializations

Read (and cite) our [bioRXiv manuscript](https://www.biorxiv.org/content/10.1101/2021.09.01.458620v1) on NMF for single-cell experiments.

## Divisive Clustering
Divisive clustering by rank-2 spectral bipartitioning.
* 2nd SVD vector is linearly related to the difference between factors in rank-2 matrix factorization.
* Rank-2 matrix factorization (optional non-negativity constraints) for spectral bipartitioning **~2x faster** than _irlba_ SVD
* Sensitive distance-based stopping criteria similar to Newman-Girvan modularity, but orders of magnitude faster
* Stopping criteria based on minimum number of samples