# Bipartition a sample set

Spectral biparitioning by rank-2 matrix factorization

## Usage

``` r
bipartition(
  data,
  tol = 1e-05,
  nonneg = TRUE,
  threads = 0,
  verbose = FALSE,
  ...
)
```

## Arguments

- data:

  dense or sparse matrix of features in rows and samples in columns.
  Prefer `matrix` or `Matrix::dgCMatrix`, respectively. Also accepts a
  file path (character string) which will be auto-loaded based on
  extension.

- tol:

  tolerance of the fit (default 1e-4)

- nonneg:

  enforce non-negativity of the rank-2 factorization used for
  bipartitioning

- threads:

  number of threads for OpenMP parallelization (default 0 = all
  available)

- verbose:

  print progress information (default FALSE)

- ...:

  additional arguments (see Advanced Parameters section)

## Value

A list giving the bipartition and useful statistics:

- v : vector giving difference between sample loadings between factors
  in a rank-2 factorization

- dist : relative cosine distance of samples within a cluster to
  centroids of assigned vs. not-assigned cluster

- size1 : number of samples in first cluster (positive loadings in 'v')

- size2 : number of samples in second cluster (negative loadings in 'v')

- samples1: indices of samples in first cluster

- samples2: indices of samples in second cluster

- center1 : mean feature loadings across samples in first cluster

- center2 : mean feature loadings across samples in second cluster

## Details

Spectral bipartitioning is a popular subroutine in divisive clustering.
The sign of the difference between sample loadings in factors of a
rank-2 matrix factorization gives a bipartition that is nearly identical
to an SVD.

Rank-2 matrix factorization by alternating least squares is faster than
rank-2-truncated SVD (i.e. *irlba*).

This function is a specialization of rank-2
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) with
support for factorization of only a subset of samples, and with
additional calculations on the factorization model relevant to
bipartitioning. See
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) for details
regarding rank-2 factorization.

## Note

`bipartition()` uses scalar `nonneg` (not length-2) because rank-2
factorizations apply the same constraint to both factors. The default
`tol = 1e-5` (inherited from
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md)'s
internal rank-2 path) is tighter than
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md)'s global
`1e-4` because single rank-2 subproblems converge faster.

## Advanced Parameters

Several parameters may be specified in the `...` argument:

- `diag = TRUE`: scale factors in \\w\\ and \\h\\ to sum to 1 by
  introducing a diagonal, \\d\\. This should generally never be set to
  `FALSE`. Diagonalization enables symmetry of models in factorization
  of symmetric matrices, convex L1 regularization, and consistent factor
  scalings.

- `samples = 1:ncol(A)`: samples to include in bipartition, numbered
  from 1 to `ncol(A)`. Default is all samples.

- `calc_dist = TRUE`: calculate the relative cosine distance of samples
  within a cluster to either cluster centroid. If `TRUE`, centers for
  clusters will also be calculated.

- `seed = NULL`: random seed for model initialization, generally not
  needed for rank-2 factorizations because robust solutions are
  recovered when `diag = TRUE`

- `maxit = 100`: maximum number of alternating updates of \\w\\ and
  \\h\\. Generally, rank-2 factorizations converge quickly and this
  should not need to be adjusted.

## References

Kuang, D, Park, H. (2013). "Fast rank-2 nonnegative matrix factorization
for hierarchical document clustering." Proc. 19th ACM SIGKDD intl. conf.
on Knowledge discovery and data mining.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`dclust`](https://zdebruine.github.io/RcppML/reference/dclust.md)

## Author

Zach DeBruine

## Examples

``` r
# \donttest{
library(Matrix)
data(iris)
A <- as(as.matrix(iris[,1:4]), "dgCMatrix")
bipartition(A, calc_dist = TRUE)
#> $v
#> [1] -0.1231538 -0.1939326  0.2112760  0.1058103
#> 
#> $dist
#> [1] 0.01303372
#> 
#> $size1
#> [1] 2
#> 
#> $size2
#> [1] 2
#> 
#> $samples1
#> [1] 2 3
#> 
#> $samples2
#> [1] 0 1
#> 
#> $center1
#>   [1] 0.80 0.80 0.75 0.85 0.80 1.05 0.85 0.85 0.80 0.80 0.85 0.90 0.75 0.60 0.70
#>  [16] 0.95 0.85 0.85 1.00 0.90 0.95 0.95 0.60 1.10 1.05 0.90 1.00 0.85 0.80 0.90
#>  [31] 0.90 0.95 0.80 0.80 0.85 0.70 0.75 0.75 0.75 0.85 0.80 0.80 0.75 1.10 1.15
#>  [46] 0.85 0.90 0.80 0.85 0.80 3.05 3.00 3.20 2.65 3.05 2.90 3.15 2.15 2.95 2.65
#>  [61] 2.25 2.85 2.50 3.05 2.45 2.90 3.00 2.55 3.00 2.50 3.30 2.65 3.20 2.95 2.80
#>  [76] 2.90 3.10 3.35 3.00 2.25 2.45 2.35 2.55 3.35 3.00 3.05 3.10 2.85 2.70 2.65
#>  [91] 2.80 3.00 2.60 2.15 2.75 2.70 2.75 2.80 2.05 2.70 4.25 3.50 4.00 3.70 4.00
#> [106] 4.35 3.10 4.05 3.80 4.30 3.55 3.60 3.80 3.50 3.75 3.80 3.65 4.45 4.60 3.25
#> [121] 4.00 3.45 4.35 3.35 3.90 3.90 3.30 3.35 3.85 3.70 4.00 4.20 3.90 3.30 3.50
#> [136] 4.20 4.00 3.65 3.30 3.75 4.00 3.70 3.50 4.10 4.10 3.75 3.45 3.60 3.85 3.45
#> 
#> $center2
#>   [1] 4.30 3.95 3.95 3.85 4.30 4.65 4.00 4.20 3.65 4.00 4.55 4.10 3.90 3.65 4.90
#>  [16] 5.05 4.65 4.30 4.75 4.45 4.40 4.40 4.10 4.20 4.10 4.00 4.20 4.35 4.30 3.95
#>  [31] 3.95 4.40 4.65 4.85 4.00 4.10 4.50 4.25 3.70 4.25 4.25 3.40 3.80 4.25 4.45
#>  [46] 3.90 4.45 3.90 4.50 4.15 5.10 4.80 5.00 3.90 4.65 4.25 4.80 3.65 4.75 3.95
#>  [61] 3.50 4.45 4.10 4.50 4.25 4.90 4.30 4.25 4.20 4.05 4.55 4.45 4.40 4.45 4.65
#>  [76] 4.80 4.80 4.85 4.45 4.15 3.95 3.95 4.25 4.35 4.20 4.70 4.90 4.30 4.30 4.00
#>  [91] 4.05 4.55 4.20 3.65 4.15 4.35 4.30 4.55 3.80 4.25 4.80 4.25 5.05 4.60 4.75
#> [106] 5.30 3.70 5.10 4.60 5.40 4.85 4.55 4.90 4.10 4.30 4.80 4.75 5.75 5.15 4.10
#> [121] 5.05 4.20 5.25 4.50 5.00 5.20 4.50 4.55 4.60 5.10 5.10 5.85 4.60 4.55 4.35
#> [136] 5.35 4.85 4.75 4.50 5.00 4.90 5.00 4.25 5.00 5.00 4.85 4.40 4.75 4.80 4.45
#> 
# }
```
